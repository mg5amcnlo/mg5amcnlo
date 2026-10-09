#!/usr/bin/env python3
"""Test the global QED recoil patch against an isolated PYTHIA 8.318 source.

Usage: python3 docs/audits/check_pythia8_qed_global.py /path/to/pythia8318

The existing library and headers are read, never changed. A temporary copy
of SimpleTimeShower.cc is patched and linked before the library so that its
definitions override the stock shared-library definitions. A stock control
must fail the S-event global-recoil assertion; the patched S and H samples
must pass first/later emission recoil and energy-momentum checks.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_SHA256 = 'f20a26c9d6a52ea2330288b9558140cc70abb88cd5158c249c9221417120e49b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pythia', type=Path, help='an unmodified PYTHIA 8.318 build')
    args = parser.parse_args()
    prefix = args.pythia.resolve()
    source = prefix / 'src/SimpleTimeShower.cc'
    if hashlib.sha256(source.read_bytes()).hexdigest() != REFERENCE_SHA256:
        parser.error('SimpleTimeShower.cc is not the audited, unmodified PYTHIA 8.318 source')
    patch = ROOT / 'Template/NLO/MCatNLO/srcPythia8/patches/pythia8318-global-qed.patch'
    fixture = ROOT / 'tests/input_files/check_pythia8_qed_global.cc'
    with tempfile.TemporaryDirectory(prefix='mg5_qed_global_shower_') as temp:
        work = Path(temp)
        (work / 'src').mkdir()
        patched = work / 'src/SimpleTimeShower.cc'
        shutil.copyfile(source, patched)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=work, check=True, capture_output=True, text=True)
        for mode in ('stock', 'patched'):
            binary = work / ('check_'+mode)
            command = ['g++', '-std=c++11', '-O1', '-I'+str(prefix / 'include'),
                       str(fixture)]
            if mode == 'patched':
                command.append(str(patched))
            command += ['-L'+str(prefix / 'lib'),
                        '-Wl,-rpath,'+str(prefix / 'lib'), '-lpythia8',
                        '-o', str(binary)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            for hard in ((0,) if mode == 'stock' else (0, 1)):
                result = subprocess.run([
                    str(binary), str(prefix / 'share/Pythia8/xmldoc'), str(hard)],
                    capture_output=True, text=True)
                lines = [line for line in result.stdout.splitlines()
                         if line.startswith('QED_GLOBAL_CHECK ')]
                if len(lines) != 1:
                    raise RuntimeError(result.stdout+result.stderr)
                print(mode+': '+lines[0])
                if mode == 'stock':
                    if result.returncode != 1:
                        raise RuntimeError('Stock control did not reproduce the recoil regression')
                elif result.returncode:
                    raise RuntimeError('Patched shower test failed:\n'+result.stdout+result.stderr)


if __name__ == '__main__':
    main()
