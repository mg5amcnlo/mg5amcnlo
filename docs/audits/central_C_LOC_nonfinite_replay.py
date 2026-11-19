#!/usr/bin/env python3
"""Replay C_LOC's first logged nonfinite soft counterterm at one fixed point.

This is a diagnostic: isolated copies of existing compiled process files,
no event generation and no integration result. It stops at native history 15.
Build the C_LOC process first; run from the repository root with:
  python3 docs/audits/central_C_LOC_nonfinite_replay.py
The two angle implementations are injected only into a fresh temporary copy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
POINT = [0.14521121978759766, 0.21933037042617798,
         0.0041975677013397217, 0.33865121006965637,
         0.66447198390960693, 0.55079606175422668,
         0.27437531948089600]
LEGACY_ANGLE = "      subroutine get_cms_costh_fks(p1,p2,ka,kb,E1,E2,xm1,xm2,\n     #                             beta1,beta2,costhfks)\n      use fks_phase_space_data, only: ybst_til_tocm\nc Given the momenta p1 and p2 in the \\tilde{k}_1+\\tilde{k}_2 c.m. frame, \nc returns the velocities beta1 and beta2, and the 3-angle between p1 and p2\nc in the k_1+k_2 c.m. frame. Here, ka=\\tilde{k}_1, ,kb=\\tilde{k}_2\n      implicit none\n      real*8 p1(0:3),p2(0:3),ka(0:3),kb(0:3)\n      real*8 E1,E2,xm1,xm2,beta1,beta2,costhfks\n      double precision tmp,costh_fks,get_cms_energy,dot\n      real*8 tiny\n      parameter (tiny=1.d-6)\n      external dot\nc\n      if(ybst_til_tocm.eq.0.d0)then\n        tmp=costh_fks(p1,p2)\n        beta1=sqrt(1-(xm1/p1(0))**2)\n        beta2=sqrt(1-(xm2/p2(0))**2)\n      else\n        if(E1.lt.0.d0)E1=get_cms_energy(p1,ka,kb)\n        beta1=sqrt(1-(xm1/E1)**2)\n        if(E2.lt.0.d0)E2=get_cms_energy(p2,ka,kb)\n        beta2=sqrt(1-(xm2/E2)**2)\n        tmp=(1-dot(p1,p2)/(E1*E2))/(beta1*beta2)\n        if((abs(tmp)-1.d0).gt.tiny)then\n          write(*,*)'Warning in get_cms_costh_fks',tmp\n          tmp=sign(1.d0,tmp)\n        elseif( (abs(tmp)-1.d0).le.tiny .and.\n     #          (abs(tmp)-1.d0).ge.0.d0 )then\n          tmp=sign(1.d0,tmp)\n        endif\n      endif\nc\n      costhfks=tmp\nc\n      return\n      end\n\n\n\n"

PROTOTYPE_ANGLE = '      subroutine get_cms_costh_fks(p1,p2,ka,kb,E1,E2,xm1,xm2,\n     #                             beta1,beta2,costhfks)\n      use fks_phase_space_data, only: ybst_til_tocm\n      implicit none\n      real*8 p1(0:3),p2(0:3),ka(0:3),kb(0:3)\n      real*8 E1,E2,xm1,xm2,beta1,beta2,costhfks\n      real*8 q(0:3),r1(0:3),r2(0:3),mass,bp1,bp2,n1,n2\n      q=ka+kb\n      r1=p1\n      r2=p2\n      if (ybst_til_tocm.ne.0d0) then\n        mass=sqrt((q(0)-q(3))*(q(0)+q(3))\n     $       -q(1)**2-q(2)**2)\n        bp1=sum(p1(1:3)*q(1:3))\n        bp2=sum(p2(1:3)*q(1:3))\n        r1(0)=(p1(0)*q(0)-bp1)/mass\n        r2(0)=(p2(0)*q(0)-bp2)/mass\n        r1(1:3)=p1(1:3)+q(1:3)*\n     $       (bp1/(mass*(q(0)+mass))-p1(0)/mass)\n        r2(1:3)=p2(1:3)+q(1:3)*\n     $       (bp2/(mass*(q(0)+mass))-p2(0)/mass)\n      endif\n      n1=sqrt(sum(r1(1:3)**2))\n      n2=sqrt(sum(r2(1:3)**2))\n      beta1=n1/r1(0)\n      beta2=n2/r2(0)\n      if (xm1.eq.0d0) beta1=1d0\n      if (xm2.eq.0d0) beta2=1d0\n      costhfks=1d0\n      if(n1.gt.0d0.and.n2.gt.0d0)then\n        costhfks=sum((r1(1:3)/n1)*(r2(1:3)/n2))\n        costhfks=max(-1d0,min(1d0,costhfks))\n      endif\n      return\n      end\n\n\n'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", type=Path,
                        default=ROOT / "central_runs_fixed/C_LOC")
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    process = args.process.resolve()
    source = process / "SubProcesses/P0_bdx_tux"
    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="C_LOC_nonfinite_"))
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / "work"
    work.mkdir(exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and (path.suffix in
                (".o", ".mod", ".inc", ".a", ".f", ".f90", ".h", ".dat")
                or path.name in ("FKS_params.dat", "randinit")):
            shutil.copyfile(path, scratch / path.name)
    shutil.copytree(source / "MadLoop5_resources",
                    scratch / "MadLoop5_resources", dirs_exist_ok=True)
    shutil.copytree(process / "lib/Pdfdata", work / "Pdfdata",
                    dirs_exist_ok=True)
    shutil.copyfile(process / "lib/libmc_born_support.so",
                    scratch / "libmc_born_support.so")
    for name in ("configs_and_props_info.dat", "FKS_params.dat",
                 "grid.MC_integer", "initial_states_map.dat",
                 "leshouche_info.dat", "param_card.dat", "randinit", "symfact.dat"):
        shutil.copyfile(source / "GF1.0" / name, work / name)
    grids = (source / "GF1.0/mint_grids").read_text()
    (work / "mint_grids").write_text("".join(
        line for line in grids.splitlines(True) if not line.startswith(" MAX")))
    (work / "input_app.txt").write_text(
        "8 1\n-1.0\n1 -0.1\n-1 -0.1\n1\n1\n1.0\n1\n1 1 1\nall\n")

    # Read link settings without running a make build in the production tree.
    keys = ("FC", "FFLAGS", "MINTMC", "NLOLIBS", "APPLLIBS", "LINKLIBS",
            "FO_EXTRAPATHS", "FO_EXTRALIBS", "FJLIBS", "LDFLAGS")
    makefile = scratch / "diagnostic.mk"
    makefile.write_text("include makefile\n" + "".join(
        "$(info " + key + "=$(" + key + "))\n" for key in keys) +
        "\n.PHONY: diagnostic_info\ndiagnostic_info:\n\t@true\n")
    settings = subprocess.check_output(
        ["make", "-s", "-f", str(makefile), "diagnostic_info"],
        cwd=source, text=True)
    variables = dict(line.split("=", 1) for line in settings.splitlines()
                     if "=" in line)
    compile_cmd = shlex.split(variables["FC"]) + shlex.split(variables["FFLAGS"])
    compile_cmd += ["-I" + str(source), "-I" + str(process / "lib")]

    driver = (source / "driver_mintMC.f").read_text()
    old = "         call update_vegas_x(xx,x_local)"
    assert driver.count(old) == 1
    point_fortran = ",\n     $       ".join(format(x, ".17g") + "d0" for x in POINT)
    driver = driver.replace(old, old + "\n         x_local(1:7)=(/" + point_fortran + "/)\n")
    (scratch / "driver_mintMC.f").write_text(driver)

    singular = (source / "fks_singular.f").read_text()
    name = "      subroutine compute_soft_counter_term(replace_MC_subt)\n"
    singular = singular.replace(name, name +
        "      use mc_native_context, only: active_history\n")
    marker = "          call add_wgt(8,orders,-wgt1*f_s_MC_H,0d0,0d0)"
    assert singular.count(marker) == 1
    singular = singular.replace(marker, """          if (active_history.eq.15) then
            write(*,*) 'DIAGNOSTIC_SOFT_HISTORY',active_history,
     $           i_fks,j_fks
            write(*,*) 'DIAGNOSTIC_FACTORS',amp_split(iamp),s_s,
     $           g22,replace_MC_subt,f_s_MC_S,f_s_MC_H
            write(*,*) 'DIAGNOSTIC_TYPE8',-wgt1*f_s_MC_H
            write(*,*) 'DIAGNOSTIC_KINEMATICS',soft_energy,
     $           y_ij_fks_ev,xi_i_fks_ev
            write(*,*) 'DIAGNOSTIC_SOFT_MOMENTA',p1_cnt(:,:,0)
            stop 901
          endif
""" + marker)
    (scratch / "fks_singular.f").write_text(singular)
    for name in ("driver_mintMC.f", "fks_singular.f", "handling_lhe_events.f"):
        subprocess.run(compile_cmd + ["-c", name], cwd=scratch, check=True)

    link = shlex.split(variables["FC"]) + ["-o", "mintMC_diagnostic"]
    link += shlex.split(variables["MINTMC"])
    for key in keys[3:]:
        link += shlex.split(variables[key])
    result = {"point": POINT, "scratch": str(scratch),
              "process": str(process), "source_sha256": {
                  name: digest(source / name) for name in
                  ("fks_Sij.f", "fks_singular.f", "driver_mintMC.f")}}
    original = (source / "fks_Sij.f").read_text()
    start = original.index("      subroutine get_cms_costh_fks(")
    stop = original.index("      double precision function costh_fks", start)
    for label, implementation in (("legacy", LEGACY_ANGLE),
                                  ("prototype", PROTOTYPE_ANGLE)):
        (scratch / "fks_Sij_diagnostic.f").write_text(
            original[:start] + implementation + original[stop:])
        subprocess.run(compile_cmd + ["-c", "fks_Sij_diagnostic.f", "-o", "fks_Sij.o"],
                       cwd=scratch, check=True)
        subprocess.run(link, cwd=scratch, check=True)
        log_path = work / (label + ".log")
        with log_path.open("w") as log, (work / "input_app.txt").open() as inp:
            run = subprocess.run(["../mintMC_diagnostic"], cwd=work,
                                 stdin=inp, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=60)
        text = log_path.read_text()
        factors = re.search(r"DIAGNOSTIC_FACTORS\s+([^\n]+)", text)
        if factors is None:
            raise RuntimeError("Target history was not reached; inspect " + str(log_path))
        numbers = [float(value.replace("D", "E")) for value in factors[1].split()]
        result[label] = {"returncode": run.returncode, "log": str(log_path),
                         "factors": [value if math.isfinite(value) else "NaN"
                                     for value in numbers]}
        if label == "legacy":
            assert not math.isfinite(numbers[1]), "Expected legacy fks_Sij NaN"
        else:
            assert all(math.isfinite(value) for value in numbers)
    (scratch / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
