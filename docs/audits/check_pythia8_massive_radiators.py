#!/usr/bin/env python3
"""Check massive-radiator kernels in an isolated PYTHIA 8.318 shower.

Usage: python3 docs/audits/check_pythia8_massive_radiators.py /path/to/pythia8318

Temporarily apply the colourless global-recoil patch and instrument actual
QCD/QED trial weights. Check W, squark and gluino radiation, including both
charge/colour ends and accepted emissions. This does not validate complete
MC@NLO matching. Installed sources, headers and libraries remain unchanged.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SHA256 = 'f20a26c9d6a52ea2330288b9558140cc70abb88cd5158c249c9221417120e49b'

PROBE = r'''
namespace {
int mg5Counts[4] = {}, mg5Signs[4] = {}, mg5Failures = 0;
void mg5MassiveProbe(int pdg, bool qcd, int end, double factor,
                     double z, double numerator) {
  const int id = abs(pdg);
  int slot = -1;
  if (id == 1000002) slot = qcd ? 0 : 2;
  if (id == 1000021 && qcd) slot = 1;
  if (id == 24 && !qcd) slot = 3;
  if (slot < 0) return;
  ++mg5Counts[slot];
  mg5Signs[slot] |= end > 0 ? 1 : 2;
  const double expectedFactor = qcd ? 4./3. : (id == 24 ? 1. : 4./9.);
  if (!(z > 0. && z < 1.) || !std::isfinite(numerator)
      || abs(numerator-(1.+z*z)) > 1.e-12
      || abs(factor-expectedFactor) > 1.e-12
      || (qcd && abs(end) != 1)) ++mg5Failures;
}
}
extern "C" int mg5_massive_probe_report(int pdg) {
  for (int slot = 0; slot < 4; ++slot) {
    if (!((pdg == 24 && slot == 3) || (pdg == 1000021 && slot == 1)
        || (pdg == 1000002 && (slot == 0 || slot == 2)))) continue;
    if (!mg5Counts[slot] || mg5Signs[slot] != 3) ++mg5Failures;
    cout << "MASSIVE_KERNEL_PROBE pdg=" << pdg << " channel="
         << (slot < 2 ? "QCD" : "QED") << " trials=" << mg5Counts[slot]
         << " both_ends=" << (mg5Signs[slot] == 3)
         << " failures=" << mg5Failures << endl;
  }
  return mg5Failures != 0;
}
'''


def insert_after(source, anchor, addition):
    if source.count(anchor) != 1:
        raise RuntimeError('Missing or ambiguous PYTHIA source anchor: '+anchor)
    return source.replace(anchor, anchor+addition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pythia', type=Path, help='an unmodified PYTHIA 8.318 build')
    prefix = parser.parse_args().pythia.resolve()
    original = prefix / 'src/SimpleTimeShower.cc'
    if hashlib.sha256(original.read_bytes()).hexdigest() != REFERENCE_SHA256:
        parser.error('SimpleTimeShower.cc is not the audited, unmodified PYTHIA 8.318 source')
    patch = ROOT / 'Template/NLO/MCatNLO/srcPythia8/patches/pythia8318-global-qed.patch'
    with tempfile.TemporaryDirectory(prefix='mg5_massive_radiators_') as temp:
        work = Path(temp)
        (work / 'src').mkdir()
        source = work / 'src/SimpleTimeShower.cc'
        source.write_bytes(original.read_bytes())
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=work, check=True, capture_output=True, text=True)
        instrumented = insert_after(source.read_text(), 'namespace Pythia8 {', PROBE)
        instrumented = insert_after(instrumented,
            '          wt = (1. + pow2(dip.z)) / wtPSglue;',
            '\n          mg5MassiveProbe(event[dip.iRadiator].id(), true, '
            'dip.colType, colFac, dip.z, wt*wtPSglue);')
        instrumented = insert_after(instrumented,
            '        wt = (1. + pow2(dip.z)) / wtPSgam;',
            '\n        mg5MassiveProbe(event[dip.iRadiator].id(), false, '
            'dip.chgType, pow2(dip.chgType/3.), dip.z, wt*wtPSgam);')
        source.write_text(instrumented)
        binary = work / 'check_massive'
        subprocess.run(['g++', '-std=c++11', '-O1', '-I'+str(prefix / 'include'),
                        str(ROOT / 'tests/input_files/check_pythia8_massive_radiators.cc'),
                        str(source), '-L'+str(prefix / 'lib'),
                        '-Wl,-rpath,'+str(prefix / 'lib'), '-lpythia8',
                        '-o', str(binary)], check=True, capture_output=True, text=True)
        for pdg in (24, 1000002, 1000021):
            result = subprocess.run([str(binary), str(prefix / 'share/Pythia8/xmldoc'),
                                     str(pdg)], capture_output=True, text=True)
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith('MASSIVE_')]
            if result.returncode or len(lines) != (3 if pdg == 1000002 else 2):
                raise RuntimeError('Massive-radiator check failed:\n'+result.stdout+result.stderr)
            print('\n'.join(lines))


if __name__ == '__main__':
    main()
