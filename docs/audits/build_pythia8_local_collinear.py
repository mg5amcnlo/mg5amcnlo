#!/usr/bin/env python3
"""Build a private opt-in IF-collinear patch, leaving stock Pythia untouched."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATCH = (ROOT / 'Template/NLO/MCatNLO/srcPythia8/patches'
         '/pythia8318-local-quark-collinear.patch')
SOFT_PATCH = PATCH.with_name('pythia8318-local-top-soft.patch')
STOCK_HASHES = {
    'src/SimpleSpaceShower.cc':
        '79b9f2334af443005b7c4722dc629ca4facdce6663b73815c1fa0278aacfa789',
    'src/SimpleTimeShower.cc':
        'f20a26c9d6a52ea2330288b9558140cc70abb88cd5158c249c9221417120e49b',
    'share/Pythia8/xmldoc/SpacelikeShowers.xml':
        '0d4d3c6b54e7dc8f1dfb89088a96c9354a07ec447cb1b5a96b81baba00d09d2e',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stock', type=Path, help='unmodified built Pythia 8.318')
    parser.add_argument('prefix', type=Path, help='new private build directory')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--top-soft', action='store_true',
                        help='also apply the massive-top IF soft patch')
    args = parser.parse_args()
    stock, prefix = args.stock.resolve(), args.prefix.resolve()
    if prefix.exists() or prefix == stock or stock in prefix.parents:
        parser.error('Use a new directory outside the stock source tree')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for name, expected in STOCK_HASHES.items():
        if sha256(stock / name) != expected:
            parser.error('Not the unmodified Pythia 8.318 reference: ' + name)
    checked = list(STOCK_HASHES) + ['lib/libpythia8.so', 'lib/libpythia8.a']
    before = {name: sha256(stock / name) for name in checked}
    patches = [PATCH] + ([SOFT_PATCH] if args.top_soft else [])
    commands = [
        ['patch', '--batch', '--forward', '-p1', '-i', str(p)] for p in patches]
    commands += [
        ['./configure', '--prefix=' + str(prefix),
         '--cxx-common=-O2 -std=c++11 -fPIC -pthread'],
        ['make', '-j' + str(args.jobs), 'lib/libpythia8.so', 'lib/libpythia8.a'],
    ]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Copy objects as an incremental build cache; never hard-link mutable files.
    shutil.copytree(stock, prefix, symlinks=False,
                    ignore=shutil.ignore_patterns('.git', '__pycache__'))
    try:
        for command in commands:
            subprocess.run(command, cwd=prefix, check=True)
        # pythia8-config expects this installed copy even for an in-tree build.
        examples = prefix / 'share/Pythia8/examples'
        examples.mkdir(exist_ok=True)
        shutil.copy2(prefix / 'examples/Makefile.inc', examples / 'Makefile.inc')
        record = {'stock': str(stock), 'private': str(prefix),
                  'patch': str(PATCH), 'patch_sha256': sha256(PATCH),
                  'patches': [{'path': str(p), 'sha256': sha256(p)} for p in patches],
                  'stock_sha256': before, 'commands': commands,
                  'private_sha256': {name: sha256(prefix / name) for name in checked},
                  'default_setting': 'SpaceShower:IFquarkCollinearCorrection=off',
                  'top_soft_patch': args.top_soft,
                  'top_soft_default': 'SpaceShower:IFtopSoftCorrection=off'}
        (prefix / 'local_collinear_patch.json').write_text(
            json.dumps(record, indent=2) + '\n')
    finally:
        after = {name: sha256(stock / name) for name in checked}
        if after != before:
            raise RuntimeError('Stock Pythia changed during the private build')
    print('Private patched build: ' + str(prefix))
    print('Stock Pythia sources and libraries unchanged.')


if __name__ == '__main__':
    main()
