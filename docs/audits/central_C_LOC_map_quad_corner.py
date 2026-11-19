#!/usr/bin/env python3
"""Distinguish inverse precision loss from already rounded input kinematics.

  python3 docs/audits/central_C_LOC_map_quad_corner.py

Uses the first light-FSR point saved by central_C_LOC_map_replay.py. Both
compilations are temporary. The on-shell repair is an investigative comparison,
not a proposed production change or relaxation of any validity checks.
"""

import gzip
import json
from pathlib import Path
import re
import subprocess
import tempfile

from central_C_LOC_map_replay import ROOT, GETANGLES, batch_driver, TestPythiaLocalRecoil


def main():
    audit_dir = ROOT / "docs/audits"
    records = json.load(gzip.open(audit_dir / "central_C_LOC_map_points.json.gz", "rt"))
    row = next(row for row in records if row["j"] == 4)
    inputs = (f"{row['beam']} {row['partner']} {row['j']} 173 .1 .2 .3\n" +
              "\n".join(" ".join(f"{v:.17g}" for v in p) for p in row["momenta"]) + "\n")
    result = {"scope": "One light-FSR corner, kinematic test only", "point": row}
    with tempfile.TemporaryDirectory(prefix="C_LOC_quad_corner_") as temp:
        work = Path(temp)
        source = (ROOT / "Template/NLO/SubProcesses/genps_fks_helpers.f").read_text()
        start = source.index("      subroutine getangles(")
        end = source.index("      end subroutine getangles", start) + len("      end subroutine getangles")
        helper = work / "helpers.f"
        helper.write_text(source[:start] + GETANGLES + source[end:])
        driver = work / "driver.f90"
        original_driver = batch_driver()
        driver.write_text(original_driver)
        commands = []
        original = subprocess.run

        def substitute(command, *args, **kwargs):
            if isinstance(command, list) and any(str(v).endswith("check_pythia_local_recoil.f90") for v in command):
                command = [str(driver) if str(v).endswith("check_pythia_local_recoil.f90") else
                           str(helper) if str(v).endswith("/genps_fks_helpers.f") else v
                           for v in command if not str(v).startswith("-ffpe-trap")]
                command.insert(1, "-freal-8-real-16")
                commands.append(command)
            return original(command, *args, **kwargs)

        subprocess.run = substitute
        try:
            TestPythiaLocalRecoil.setUpClass()
        finally:
            subprocess.run = original
        try:
            for repair in (False, True):
                executable = TestPythiaLocalRecoil.executable
                if repair:
                    driver.write_text(original_driver.replace(
                        "     call generate_native_momenta(plab,pout,poutlab,pcm,jagain,pass)",
                        """     pout=plab
     do i=3,5
        plab(0,i)=sqrt(sum(plab(1:3,i)**2)+pmass(i)**2)
     enddo
     plab(0,1)=(sum(plab(0,3:5))+sum(plab(3,3:5)))/2d0
     plab(3,1)=plab(0,1)
     plab(0,2)=(sum(plab(0,3:5))-sum(plab(3,3:5)))/2d0
     plab(3,2)=-plab(0,2)
     write(*,*)'AUDIT_INPUT_SHIFT',maxval(abs(plab-pout))
     call generate_native_momenta(plab,pout,poutlab,pcm,jagain,pass)"""))
                    executable = work / "onshell"
                    command = commands[0][:]
                    command[-1] = str(executable)
                    compile_result = original(command, cwd=work, capture_output=True, text=True)
                    if compile_result.returncode:
                        raise RuntimeError(compile_result.stdout + compile_result.stderr)
                run = original([str(executable), "native_batch"], input=inputs, capture_output=True, text=True)
                state, delta = re.search(r"AUDIT_RESULT\s+([TF])\s+(\S+)", run.stdout).groups()
                shift = re.search(r"AUDIT_INPUT_SHIFT\s+(\S+)", run.stdout)
                mismatch = re.search(r"relative/tolerance\):\s+(\S+)", run.stdout)
                result["quad_onshell_input" if repair else "quad_original_input"] = {
                    "passed": state == "T", "replay_output_relative_error": float(delta),
                    "reported_mismatch_before_rejection": float(mismatch.group(1)) if mismatch else None,
                    "max_input_component_change_GeV": float(shift.group(1)) if shift else 0.0}
        finally:
            TestPythiaLocalRecoil.doClassCleanups()
    (audit_dir / "central_C_LOC_map_quad_corner.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
