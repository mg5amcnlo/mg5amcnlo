#!/usr/bin/env python3
"""Check stock Pythia 8.318 local maps without modifying shower sources.

Link first-emission diagnostic UserHooks to an unmodified build. Check
routing and accepted q->qg maps for both beams and antiquarks, and compare
the analytic massless-incoming stock density with the raw Fortran helper.
The standard step veto stops each diagnostic event after its first branch;
it never changes that branch's kernel. Optionally shower matched LHE files
completely, using a steering file from the production launcher.
"""

import argparse
import gzip
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SHA256 = {
    "src/SimpleSpaceShower.cc": "79b9f2334af443005b7c4722dc629ca4facdce6663b73815c1fa0278aacfa789",
    "src/SimpleTimeShower.cc": "f20a26c9d6a52ea2330288b9558140cc70abb88cd5158c249c9221417120e49b",
    "share/Pythia8/xmldoc/SpacelikeShowers.xml": "0d4d3c6b54e7dc8f1dfb89088a96c9354a07ec447cb1b5a96b81baba00d09d2e",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pythia", type=Path, help="an unmodified Pythia 8.318 build")
    parser.add_argument("--lhe", type=Path, action="append", default=[],
                        help="matched stable-top LHE file (may be gzip compressed; repeatable)")
    parser.add_argument("--steering", type=Path, help="production Pythia8.cmd for complete showers")
    args = parser.parse_args()
    if bool(args.lhe) != bool(args.steering):
        parser.error("--lhe and --steering must be provided together")
    prefix = args.pythia.resolve()
    for name, expected in REFERENCE_SHA256.items():
        if hashlib.sha256((prefix / name).read_bytes()).hexdigest() != expected:
            parser.error(f"{name} is not the unmodified Pythia 8.318 reference")
    print("LOCAL_STOCK_SOURCE official_8.318 passed", flush=True)
    with tempfile.TemporaryDirectory(prefix="mg5_singletop_stock_") as temp:
        binary = Path(temp) / "check_shower"
        subprocess.run(["g++", "-std=c++11", "-O1", "-I"+str(prefix / "include"),
                        str(ROOT / "tests/input_files/check_pythia8_singletop_local.cc"),
                        "-L"+str(prefix / "lib"), "-Wl,-rpath,"+str(prefix / "lib"),
                        "-lpythia8", "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from tests.unit_tests.fks.test_pythia_local_matching import TestPythiaLocalMatching
        TestPythiaLocalMatching.setUpClass()
        check = TestPythiaLocalMatching()
        count = 0
        try:
            for swap in (0, 1):
                for anti in (0, 1):
                    result = subprocess.run([str(binary), str(prefix / "share/Pythia8/xmldoc"),
                                             str(swap), str(anti)], capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(result.stdout+result.stderr)
                    for line in result.stdout.splitlines():
                        if line.startswith("LOCAL_EMISSIONS"):
                            print(f"swap={swap} anti={anti} {line}", flush=True)
                        if not line.startswith("LOCAL_POINT "):
                            continue
                        point = np.fromstring(line[len("LOCAL_POINT "):], sep=" ")
                        beam, partner, mass, kind = point[:4]
                        assert kind == 1
                        p = point[4:24].reshape(5, 4)
                        z, q2, d = point[24:]
                        incoming = check.run_kernel(p, int(beam), int(partner), int(beam), mass)
                        outgoing = check.run_kernel(p, int(beam), int(partner), int(partner), mass)
                        np.testing.assert_allclose(incoming[[1, 2, 3]], [z, q2, d], rtol=1e-8)
                        # Stock q->qg numerator; no new veto or top soft factor.
                        expected = 2*(4/3)*(1+z*z)/((1-z)*z*q2)
                        np.testing.assert_allclose(incoming[0]+outgoing[0], expected, rtol=2e-8)
                        count += 1
        finally:
            TestPythiaLocalMatching.doClassCleanups()
        if count != 100:
            raise RuntimeError(f"Expected 100 accepted-point comparisons, found {count}")
        print(f"LOCAL_FORTRAN_RAW_DENSITY accepted_points={count} passed")
        if args.lhe:
            binary = Path(temp) / "check_lhe"
            subprocess.run(["g++", "-std=c++11", "-O1", "-I"+str(prefix / "include"),
                            str(ROOT / "tests/input_files/check_pythia8_singletop_lhe.cc"),
                            "-L"+str(prefix / "lib"), "-Wl,-rpath,"+str(prefix / "lib"),
                            "-lpythia8", "-o", str(binary)],
                           check=True, capture_output=True, text=True)
            for source in args.lhe:
                source = source.resolve()
                lhe = source
                if source.suffix == ".gz":
                    lhe = Path(temp) / "events.lhe"
                    with gzip.open(source, "rb") as stream, lhe.open("wb") as target:
                        shutil.copyfileobj(stream, target)
                result = subprocess.run([str(binary), str(prefix / "share/Pythia8/xmldoc"),
                                         str(args.steering.resolve()), str(lhe)],
                                        capture_output=True, text=True)
                print(f"LOCAL_LHE_FILE {source}", flush=True)
                print(result.stdout+result.stderr, flush=True)
                result.check_returncode()


if __name__ == "__main__":
    main()
