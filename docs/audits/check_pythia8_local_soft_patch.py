#!/usr/bin/env python3
"""Check the two-patch IF kernel, opt-out behaviour and later shower history."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stock', type=Path)
    parser.add_argument('single_patch', type=Path)
    parser.add_argument('two_patches', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stock, single, private = (p.resolve() for p in
                             (args.stock, args.single_patch, args.two_patches))
    record = json.loads((private / 'local_collinear_patch.json').read_text())

    def check_hashes():
        for prefix, hashes in ((stock, record['stock_sha256']),
                               (private, record['private_sha256'])):
            for name, expected in hashes.items():
                if hashlib.sha256((prefix / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Source/library changed: ' + str(prefix / name))
        for patch in record['patches']:
            if hashlib.sha256(Path(patch['path']).read_bytes()).hexdigest() != patch['sha256']:
                raise RuntimeError('Recorded patch changed')
        name = 'src/SimpleTimeShower.cc'
        if (private / name).read_bytes() != (stock / name).read_bytes():
            raise RuntimeError('The final-state shower must remain unmodified')

    check_hashes()
    sys.path.insert(0, str(ROOT))
    from tests.unit_tests.fks.test_pythia_local_matching import TestPythiaLocalMatching
    TestPythiaLocalMatching.setUpClass()
    results = []
    try:
        check = TestPythiaLocalMatching()
        with tempfile.TemporaryDirectory(prefix='mg5_top_soft_') as temp:
            temp = Path(temp)
            source = (private / 'src/SimpleSpaceShower.cc').read_text()
            start = source.index('    // Optional massive-top soft correction,')
            end = source.index('    // Optionally veto emissions not ordered', start)
            block = source[start:end]
            scope = temp / 'scope.cc'
            scope.write_text('''#include <cassert>
#include <cmath>
double weight(int idDaughter,int idMother,int idSister,int idColPartner,
              int iColPartner,bool enabled,double delta=.2,double fraction=.5) {
  bool isGluon=idDaughter==21;
  double wt=1.,z=.7,Q2=delta/z,m2ColPair=2.,m2ColPartner=1.;
  double pT2=(1.-z)*Q2,pT2corr=fraction*pT2*(1.-delta);
  auto flag=[&](const char*) { return enabled; };
''' + block + '''
  return wt;
}
int main() {
  for(double delta : {.001,.1,.9})for(double fraction : {.00001,.2,.8,1.}) {
    assert(std::abs(weight(5,5,21,6,1,true,delta,fraction)-fraction)<1.e-13);
    assert(std::abs(weight(-5,-5,21,-6,1,true,delta,fraction)-fraction)<1.e-13);
  }
  assert(weight(5,5,21,6,1,false)==1.);
  assert(weight(5,5,21,6,0,true)==1.);
  assert(weight(5,5,21,1,1,true)==1.);
  assert(weight(5,5,21,21,1,true)==1.);
  assert(weight(21,21,21,6,1,true)==1.);
  assert(weight(5,21,-5,6,1,true)==1.);
}
''')
            # std::initializer_list is needed by the range-for tables above.
            scope.write_text('#include <initializer_list>\n' + scope.read_text())
            scope_binary = temp / 'scope'
            subprocess.run(['g++', '-std=c++11', str(scope), '-o', str(scope_binary)],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(scope_binary)], check=True)
            binaries = {}
            for name, source_name in (
                ('first', 'check_pythia8_singletop_local.cc'),
                ('history', 'check_pythia8_local_soft_history.cc')):
                binary = temp / name
                subprocess.run(['g++', '-std=c++11', '-O1', '-I'+str(private / 'include'),
                    str(ROOT / 'tests/input_files' / source_name), '-L'+str(private / 'lib'),
                    '-lpythia8', '-o', str(binary)], check=True, capture_output=True, text=True)
                binaries[name] = binary

            def run(prefix, swap, anti, mode=0, history=False):
                xml = prefix / 'share/Pythia8/xmldoc'
                command = [str(binaries['history' if history else 'first']),
                           str(xml), str(swap), str(anti)]
                if history or mode:
                    command.append(str(mode))
                result = subprocess.run(command, capture_output=True, text=True,
                    env=dict(os.environ, LD_LIBRARY_PATH=str(prefix / 'lib'),
                             PYTHIA8DATA=str(xml)), timeout=60)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                return result.stdout

            def points(stdout):
                return [line for line in stdout.splitlines() if line.startswith('LOCAL_POINT ')]

            for swap in (0, 1):
                for anti in (0, 1):
                    reference = points(run(stock, swap, anti))
                    disabled = points(run(private, swap, anti))
                    one = points(run(single, swap, anti, mode=1))
                    one_in_two = points(run(private, swap, anti, mode=1))
                    two = points(run(private, swap, anti, mode=2))
                    if reference != disabled or one != one_in_two:
                        raise RuntimeError('Disabling a patch changes its reference shower')
                    if two == one:
                        raise RuntimeError('The soft patch has no visible effect')
                    for line in two:
                        point = np.fromstring(line[len('LOCAL_POINT '):], sep=' ')
                        beam, partner, mass, _ = point[:4]
                        p = point[4:24].reshape(5, 4)
                        z, q2, d = point[24:]
                        z1 = 1-z*q2/d
                        ratio = 1-mass**2*z*z*q2/((1-z)*z1*d*d)
                        if ratio < -1e-9 or ratio > 1+1e-9:
                            raise RuntimeError('Unphysical massive eikonal ratio')
                        expected = 2*(4/3)*(z*z+z1*z1)/((1-z)*z*q2)*ratio
                        a = check.run_kernel(p, int(beam), int(partner), int(beam),
                                             mass, soft_patched=True)
                        b = check.run_kernel(p, int(beam), int(partner), int(partner),
                                             mass, soft_patched=True)
                        np.testing.assert_allclose(a[0]+b[0], expected, rtol=2e-8)
                        np.testing.assert_allclose(a[[1, 2, 3]], [z, q2, d], rtol=1e-8)
                    if len(two) != 25:
                        raise RuntimeError('Expected 25 accepted-point checks')
                    histories = {}
                    for label, prefix, mode in (('stock', stock, 0),
                            ('collinear', single, 1), ('two_patches', private, 2)):
                        history = run(prefix, swap, anti, mode=mode, history=True)
                        summary = next(line for line in history.splitlines()
                                       if line.startswith('FULL_LOCAL_SOFT '))
                        histories[label] = {k: int(v) for k, v in
                                            re.findall(r'(\w+)=(\d+)', summary)}
                    results.append({'beam_swap': swap, 'antiquark': anti,
                        'stock_default_points_bitwise_equal': len(reference),
                        'single_patch_points_bitwise_equal': len(one),
                        'two_patch_density_points': len(two), 'full_history': histories})
                    print('swap=%d anti=%d: %s' % (swap, anti, json.dumps(histories)),
                          flush=True)
    finally:
        TestPythiaLocalMatching.doClassCleanups()
        check_hashes()
    report = {'stock_unchanged': True, 'time_shower_unchanged': True,
              'patch_flags_default': 'off', 'bounded_scope_checks': 30,
              'results': results}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print('Two-patch density, scope and opt-out checks passed; full-event check '
          'rejections are recorded for stock and both patched showers.')


if __name__ == '__main__':
    main()
