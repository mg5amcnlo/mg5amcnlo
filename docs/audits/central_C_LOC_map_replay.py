#!/usr/bin/env python3
"""Replay logged C_LOC native-map failures without changing production sources.

Run from the repository root:
  python3 docs/audits/central_C_LOC_map_replay.py

The candidate getangles change exists only inside a temporary compilation.
This is a kinematic replay, not an integration or a cross-section estimate.
"""

from collections import Counter
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.unit_tests.fks.test_pythia_local_recoil import TestPythiaLocalRecoil


GETANGLES = """      subroutine getangles(pin,th,cth,sth,phi,cphi,sphi)
      implicit none
      double precision pin(0:3),th,cth,sth,phi,cphi,sphi,pt,norm
      pt=sqrt(sum(pin(1:2)**2))
      norm=sqrt(pt**2+pin(3)**2)
      th=0d0
      cth=1d0
      sth=0d0
      phi=0d0
      cphi=1d0
      sphi=0d0
      if(norm.gt.0d0)then
         cth=pin(3)/norm
         sth=pt/norm
         th=atan2(pt,pin(3))
      endif
      if(pt.gt.0d0)then
         phi=atan2(pin(2),pin(1))
         cphi=pin(1)/pt
         sphi=pin(2)/pt
      endif
      end subroutine getangles"""


def collect_points():
    records = []
    seen = set()
    occurrences = 0
    for path in sorted((ROOT / "central_runs_fixed/C_LOC/SubProcesses").glob("P*/GF*/log*.txt")):
        channel = path.parent.parent.name
        top_beam = 1 if channel in ("P0_bu_td", "P0_bdx_tux") else 2
        text = path.read_text(errors="replace")
        for block in text.split("Native MC reconstruction mismatch")[1:]:
            match = re.search(r"Native FKS legs and radiation coordinates:\s+(\d+)\s+(\d+)", block)
            rows = re.findall(r"^Native projection input\s+\d+\s+(.+)$", block, re.M)
            if match is None or len(rows) < 5:
                continue
            emitted, j = map(int, match.groups())
            if emitted != 5:
                raise ValueError("Replay assumes emitted leg 5")
            beam = j if j <= 2 else top_beam if j == 3 else 3 - top_beam
            partner = j if j > 2 else 3 if beam == top_beam else 4
            momenta = [[float(v) for v in row.split()] for row in rows[:5]]
            key = (beam, partner, j, tuple(v for row in momenta for v in row))
            occurrences += 1
            if key in seen:
                continue
            seen.add(key)
            records.append({"source": str(path.relative_to(ROOT)), "beam": beam,
                            "partner": partner, "j": j, "momenta": momenta,
                            "logged_relative_error": float(block.split()[1])})
    return records, occurrences


def category(row):
    return "ISR" if row["j"] <= 2 else "top_FSR" if row["j"] == 3 else "light_FSR"


def batch_driver():
    source = (ROOT / "tests/input_files/check_pythia_local_recoil.f90").read_text()
    source = source.replace("integer beam,partner,ifks", "integer ios,jtarget,beam,partner,ifks")
    source = source.replace("  call get_command_argument(1,mode)",
                            "42 continue\n  call get_command_argument(1,mode)")
    source = source.replace("  read(*,*)beam,partner,mass,rad",
                            "  read(*,*,iostat=ios)beam,partner,jtarget,mass,rad\n  if(ios.ne.0)stop")
    source = source.replace("  jfks=beam", "  jfks=jtarget")
    source = source.replace("  if(mode.eq.'native_point')then", """  if(mode.eq.'native_batch')then
     if(jtarget.gt.2)then
        initial_recoil_leg=beam
        resonance_recoil=.true.
     endif"""
                            )
    source = source.replace("     if(.not.pass)error stop 'native corner inverse'", "")
    source = source.replace("     if(maxval(abs(poutlab-plab)).gt.1d-4*maxval(abs(plab)))error stop 'native corner replay'", "")
    source = source.replace("     write(*,*)'PASS native point'", """     write(*,*)'AUDIT_RESULT',pass,maxval(abs(poutlab-plab))/maxval(abs(plab))
     goto 42"""
                            )
    return source


def main():
    records, occurrences = collect_points()
    audit_dir = ROOT / "docs/audits"
    fixtures_path = audit_dir / "central_C_LOC_map_points.json.gz"
    with gzip.open(fixtures_path, "wt") as stream:
        json.dump(records, stream)
    inputs = "".join(
        f"{r['beam']} {r['partner']} {r['j']} 173 .1 .2 .3\n" +
        "\n".join(" ".join(f"{v:.17g}" for v in p) for p in r["momenta"]) + "\n"
        for r in records)
    results = {"scope": "Logged reconstruction mismatches only, no integration",
               "occurrences": occurrences, "unique_points": len(records),
               "point_counts": dict(Counter(map(category, records))),
               "source_sha256": {name: hashlib.sha256((ROOT / "Template/NLO/SubProcesses" / name).read_bytes()).hexdigest()
                                 for name in ("genps_fks_helpers.f", "genps_fks.f", "initial_recoil.f", "genps_fks_radiation.f")}}
    with tempfile.TemporaryDirectory(prefix="C_LOC_map_audit_") as temp:
        work = Path(temp)
        driver = work / "driver.f90"
        driver.write_text(batch_driver())
        original = subprocess.run
        commands = []

        def substitute(command, *args, **kwargs):
            if isinstance(command, list) and any(str(v).endswith("check_pythia_local_recoil.f90") for v in command):
                command = [str(driver) if str(v).endswith("check_pythia_local_recoil.f90") else v
                           for v in command if not str(v).startswith("-ffpe-trap")]
                commands.append(command)
            return original(command, *args, **kwargs)

        subprocess.run = substitute
        try:
            TestPythiaLocalRecoil.setUpClass()
        finally:
            subprocess.run = original
        try:
            old_helper = ROOT / "Template/NLO/SubProcesses/genps_fks_helpers.f"
            source = old_helper.read_text()
            start = source.index("      subroutine getangles(")
            end = source.index("      end subroutine getangles", start) + len("      end subroutine getangles")
            candidate_source = source[:start] + GETANGLES + source[end:]
            candidate = work / "helpers.f"
            candidate.write_text(candidate_source)
            (audit_dir / "central_C_LOC_map_getangles.patch").write_text("".join(difflib.unified_diff(
                source.splitlines(True), candidate_source.splitlines(True),
                fromfile="a/Template/NLO/SubProcesses/genps_fks_helpers.f",
                tofile="b/Template/NLO/SubProcesses/genps_fks_helpers.f")))
            for label in ("current", "candidate_getangles"):
                executable = TestPythiaLocalRecoil.executable
                if label != "current":
                    executable = work / "candidate"
                    command = [str(candidate) if str(v) == str(old_helper) else v for v in commands[0]]
                    command[-1] = str(executable)
                    compilation = original(command, cwd=executable.parent, capture_output=True, text=True)
                    if compilation.returncode:
                        raise RuntimeError(compilation.stdout + compilation.stderr)
                run = original([str(executable), "native_batch"], input=inputs, capture_output=True, text=True)
                parsed = re.findall(r"AUDIT_RESULT\s+([TF])\s+(\S+)", run.stdout)
                if run.returncode or len(parsed) != len(records):
                    (audit_dir / "central_C_LOC_map_replay_failure.log").write_text(run.stdout + run.stderr)
                    raise RuntimeError(f"Incomplete replay: {len(parsed)} / {len(records)}; exit {run.returncode}")
                groups = {}
                for group in ("ISR", "top_FSR", "light_FSR"):
                    selected = [(state, float(delta)) for row, (state, delta) in zip(records, parsed) if category(row) == group]
                    groups[group] = {"points": len(selected), "passed": sum(state == "T" for state, _ in selected),
                                     "failed": sum(state == "F" for state, _ in selected),
                                     "max_relative_error_passed": max((delta for state, delta in selected if state == "T"), default=None)}
                results[label] = groups
                failures = [dict(row, replay_relative_error=float(delta)) for row, (state, delta) in zip(records, parsed) if state == "F"]
                with gzip.open(audit_dir / f"central_C_LOC_map_{label}_failures.json.gz", "wt") as stream:
                    json.dump(failures, stream)
                print(label, groups, flush=True)
        finally:
            TestPythiaLocalRecoil.doClassCleanups()
    (audit_dir / "central_C_LOC_map_replay.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
