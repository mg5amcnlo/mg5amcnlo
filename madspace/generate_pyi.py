#!/usr/bin/env python3
"""Generate madspace/_madspace_py.pyi from a freshly built _madspace_py module.

pybind11-stubgen has to import the module to introspect it, which in turn
requires the full "madspace" package (its __init__.py preloads libmadspace
and monkey-patches a few classes). Rather than requiring a full install, this
script assembles a throwaway copy of just the pieces needed for that import
in --stage-dir, then runs pybind11-stubgen against it.
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--lib", required=True, type=Path)
    parser.add_argument("--stage-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    stage_pkg = args.stage_dir / "madspace"
    if stage_pkg.exists():
        shutil.rmtree(stage_pkg)
    (stage_pkg / "lib").mkdir(parents=True)

    for name in ("__init__.py", "_madspace_py_loader.py", "python_function.py"):
        shutil.copy2(args.package_dir / name, stage_pkg / name)
    shutil.copy2(args.module, stage_pkg / args.module.name)
    shutil.copy2(args.lib, stage_pkg / "lib" / args.lib.name)
    # __init__.py imports this; it's normally written at install time (see
    # CMakeLists.txt) and irrelevant to the stubs, so stub it out here.
    (stage_pkg / "_source_hash.py").write_text('SOURCE_HASH = ""\n')

    env = os.environ.copy()
    pythonpath = [str(args.stage_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pybind11_stubgen",
            "madspace._madspace_py",
            "-o",
            str(args.output_dir),
            "--enum-class-locations=TChannelMode:PhaseSpaceMapping.TChannelMode",
            "--enum-class-locations=Activation:MLP.Activation",
            "--enum-class-locations=CutMode:Cuts.CutMode",
            "--enum-class-locations=LRSchedule:AdamOptimizer.LRSchedule",
            "--enum-class-locations=DataType:DataType",
            "--enum-class-locations=JetScaleScheme:MLMClustering.JetScaleScheme",
            "--enum-class-locations=ScaleScheme:MLMClustering.ScaleScheme",
            "--enum-class-locations=PartonLineScheme:MLMClustering.PartonLineScheme",
            "--enum-class-locations=AlphasScheme:MLMClustering.AlphasScheme",
            "--enum-class-locations=ClusteringMeasure:MLMClustering.ClusteringMeasure",
            "--enum-class-locations=ClusteringHistory:MLMClustering.ClusteringHistory",
            # Fail the build on unresolvable names/expressions (e.g. an enum
            # default value needing its own --enum-class-locations entry)
            # instead of silently emitting a stub with holes in it.
            "--exit-code",
        ],
        # cwd must not itself contain a "madspace" package: `-m` puts cwd
        # first on sys.path, which would shadow --stage-dir with the real,
        # not-yet-built source package.
        cwd=str(args.stage_dir),
        env=env,
        check=True,
    )

    # pybind11-stubgen's own error checks (above) don't guarantee the emitted
    # file is syntactically valid Python, e.g. a bad --print-safe-value-reprs
    # match can inline an expression verbatim; parse it so a broken stub
    # fails the build instead of shipping something downstream tools choke on.
    stub_path = args.output_dir / "madspace" / "_madspace_py.pyi"
    source = stub_path.read_text()
    try:
        ast.parse(source, filename=str(stub_path))
    except SyntaxError as e:
        raise SystemExit(f"Generated stub is not valid Python: {stub_path}: {e}")


if __name__ == "__main__":
    main()
