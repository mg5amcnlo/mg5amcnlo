#!/usr/bin/env python3
"""Check the private patch, its opt-out behaviour and accepted IF maps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stock', type=Path)
    parser.add_argument('private', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stock, private = args.stock.resolve(), args.private.resolve()
    record = json.loads((private / 'local_collinear_patch.json').read_text())
    for name, expected in record['private_sha256'].items():
        actual = hashlib.sha256((private / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Private build changed since installation: ' + name)

    def check_stock():
        for name, expected in record['stock_sha256'].items():
            actual = hashlib.sha256((stock / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Stock Pythia changed: ' + name)

    def points(stdout):
        return [line for line in stdout.splitlines()
                if line.startswith('LOCAL_POINT ')]

    check_stock()
    sys.path.insert(0, str(ROOT))
    from tests.unit_tests.fks.test_pythia_local_matching import TestPythiaLocalMatching
    TestPythiaLocalMatching.setUpClass()
    results = []
    try:
        check = TestPythiaLocalMatching()
        with tempfile.TemporaryDirectory(prefix='mg5_private_collinear_') as temp:
            # Compile the actual added veto block and check its channel scope.
            # A g->gg branch must not pass the equal-mother/daughter guard.
            source = (private / 'src/SimpleSpaceShower.cc').read_text()
            start = source.index('    // Optional quark IF numerator correction:')
            end = source.index('    // Optionally veto emissions not ordered', start)
            block = source[start:end]
            scope = Path(temp) / 'scope.cc'
            scope.write_text('''#include <cassert>
#include <cmath>
double pow2(double x) { return x*x; }
double weight(int idDaughter,int idMother,int idSister,int idColPartner,
              int iColPartner,bool enabled) {
  bool isGluon=idDaughter==21;
  double wt=1.,z=.7,Q2=.2,m2ColPair=1.,m2ColPartner=0.;
  auto flag=[&](const char*) { return enabled; };
''' + block + '''
  return wt;
}
int main() {
  double expected=(.7*.7+.86*.86)/(1.+.7*.7);
  assert(std::abs(weight(1,1,21,2,1,true)-expected)<1.e-14);
  assert(std::abs(weight(-1,-1,21,-2,1,true)-expected)<1.e-14);
  assert(weight(1,1,21,2,1,false)==1.);
  assert(weight(1,1,21,2,0,true)==1.);
  assert(weight(1,1,21,21,1,true)==1.);
  assert(weight(21,21,21,2,1,true)==1.);
  assert(weight(1,21,-1,2,1,true)==1.);
  assert(weight(1,1,-1,2,1,true)==1.);
}
''')
            scope_binary = Path(temp) / 'scope'
            subprocess.run(['g++', '-std=c++11', str(scope), '-o', str(scope_binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(scope_binary)], check=True)
            print('Kernel scope: 8 quark/antiquark and unaffected-channel checks passed.',
                  flush=True)
            binary = Path(temp) / 'check_shower'
            subprocess.run(['g++', '-std=c++11', '-O1', '-I'+str(private / 'include'),
                            str(ROOT / 'tests/input_files/check_pythia8_singletop_local.cc'),
                            '-L'+str(private / 'lib'), '-lpythia8', '-o', str(binary)],
                           check=True, capture_output=True, text=True)

            def shower(prefix, swap, anti, patch=False):
                xml = prefix / 'share/Pythia8/xmldoc'
                env = dict(os.environ, LD_LIBRARY_PATH=str(prefix / 'lib'),
                           PYTHIA8DATA=str(xml))
                command = [str(binary), str(xml), str(swap), str(anti)]
                if patch:
                    command.append('1')
                result = subprocess.run(command, env=env, capture_output=True,
                                        text=True, timeout=60)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                return result.stdout

            for swap in (0, 1):
                for anti in (0, 1):
                    reference = shower(stock, swap, anti)
                    default = shower(private, swap, anti)
                    patched = shower(private, swap, anti, patch=True)
                    if points(reference) != points(default):
                        raise RuntimeError('The disabled patch changes the stock shower')
                    if points(default) == points(patched):
                        raise RuntimeError('The enabled patch has no visible effect')
                    count = 0
                    for line in points(patched):
                        point = np.fromstring(line[len('LOCAL_POINT '):], sep=' ')
                        beam, partner, mass, kind = point[:4]
                        assert kind == 1
                        p = point[4:24].reshape(5, 4)
                        z, q2, d = point[24:]
                        incoming = check.run_kernel(p, int(beam), int(partner),
                                                    int(beam), mass, patched=True)
                        outgoing = check.run_kernel(p, int(beam), int(partner),
                                                    int(partner), mass, patched=True)
                        np.testing.assert_allclose(incoming[[1, 2, 3]],
                                                   [z, q2, d], rtol=1e-8)
                        z1 = 1-z*q2/d
                        expected = 2*(4/3)*(z*z+z1*z1)/((1-z)*z*q2)
                        np.testing.assert_allclose(incoming[0]+outgoing[0],
                                                   expected, rtol=2e-8)
                        count += 1
                    if count != 25:
                        raise RuntimeError('Expected 25 accepted-point comparisons')
                    summary = next(line for line in patched.splitlines()
                                   if line.startswith('LOCAL_EMISSIONS '))
                    results.append({'beam_swap': swap, 'antiquark': anti,
                                    'default_points_bitwise_equal': count,
                                    'patched_density_points': count,
                                    'patched_shower': summary})
                    print('swap=%d anti=%d: default identical; %d patched points; %s'
                          % (swap, anti, count, summary), flush=True)
    finally:
        TestPythiaLocalMatching.doClassCleanups()
        check_stock()
    report = {'stock_unchanged': True, 'private_patch_default': 'off',
              'kernel_scope_checks': 8,
              'stock': str(stock), 'private': str(private), 'results': results}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('Stock sources/libraries unchanged; private default reproduces stock.')


if __name__ == '__main__':
    main()
