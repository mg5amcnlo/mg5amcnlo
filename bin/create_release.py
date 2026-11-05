#! /usr/bin/env python3

################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Package a MadGraph7 release tarball from the current git checkout.

Non-interactive, meant to run from the release GitHub Actions workflow (or
locally for testing):

    python bin/create_release.py --version 0.2.0 --output dist/

It performs the following actions:
  1. Check that VERSION agrees with --version, that madspace still derives
     its version from it, and that pyamplicol's versions are synced to it. (UpdateNotes.txt still tracks the legacy MG5_aMC@NLO
     3.7.x series and is not part of this check.)
  2. git-archive the current HEAD into a clean MG7_v<version> directory
     (preserves the madgraph/VERSION symlink; ignores untracked/gitignored
     working-tree files).
  3. Prune bin/ to the release-facing scripts only.
  4. Materialize the default config/run-card files.
  5. Vendor offline copies of collier/ninja/SMWidth/the HEPToolsInstaller
     bundle into vendor/ (unless --skip-vendor; any failure aborts the release).
  6. Write input/authors.md (first-contribution date per author, used by the
     anniversary banner) and the input/.release marker read by
     madspace/install.py to decide whether the PyPI wheel may be offered --
     including, when --wheels-dir is given, the filenames of the wheels
     actually built, so install.py can check platform/Python availability
     locally instead of querying PyPI.
  7. tar everything up.
"""

import argparse
import glob
import os
import os.path as path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import unicodedata
import urllib.request
from datetime import date, datetime, timezone

pjoin = os.path.join
ROOT = path.dirname(path.dirname(path.realpath(__file__)))


def sanitize_author(name):
    # Remove email addresses, parenthesized asides and accents so that the
    # same person under slightly different git identities collapses to one key.
    name = re.sub(r'\S+@\S+', '', name)
    name = re.sub(r'\(.*?\)', '', name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '', name)
    return name.strip()


# Aliases for contributors who committed under more than one identity.
ALIAS = {'herquet': 'michelherquet',
         'mherquet': 'michelherquet',
         'janovak': 'jakobnovak',
         'davidepaganicluster': 'davidepagani',
         'davide': 'davidepagani',
         'pagani': 'davidepagani',
         'rikkert': 'rikkertfrederix',
         'frederix': 'rikkertfrederix',
         'riruiz': 'richardruiz',
         'richardphysics': 'richardruiz',
         'mguser': 'oliviermattelaer',
         'githubbot': 'oliviermattelaer',
         'shjeon': 'sihyunjeon',
         'paolotorriell': 'paolotorrielli',
         'sc': 'oliviermattelaer',
         'omatt': 'oliviermattelaer',
         'priscilaaquino': 'prisciladeaquino',
         'mattelaerolivier': 'oliviermattelaer',
         '': 'oliviermattelaer',
         'shaohuasheng': 'huashengshao',
         'ti5714vi': 'timstelzer',
         }


def get_first_contributions(repo_path):
    """Map each author (sanitized name) to the date of their first commit."""
    cmd = ["git", "-C", repo_path, "log", "--pretty=format:%an|%at"]
    output = subprocess.check_output(cmd, text=True)

    first_dates = {}
    for line in output.splitlines():
        try:
            author, timestamp = line.split("|")
        except ValueError:
            continue
        author = ALIAS.get(sanitize_author(author), sanitize_author(author))
        d = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
        if author not in first_dates or d < first_dates[author]:
            first_dates[author] = d
    return first_dates


def parse_info_file(filepath):
    """Parse a 'name = value' file, e.g. VERSION or input/.release."""
    info = {}
    for line in open(filepath):
        line = line.strip()
        if not line:
            continue
        name, _, value = line.partition('=')
        info[name.strip()] = value.strip()
    return info


def check_versions(version):
    """Fail loudly if VERSION disagrees with the version being released, if
    madspace/pyproject.toml has stopped deriving its version from VERSION, or
    if pyamplicol's literal (Cargo) versions are out of sync with VERSION."""
    errors = []

    mg_version = parse_info_file(pjoin(ROOT, 'VERSION')).get('version')
    if mg_version != version:
        errors.append(f"VERSION says '{mg_version}', expected '{version}'")

    with open(pjoin(ROOT, 'madspace', 'pyproject.toml'), 'rb') as f:
        pyproject = tomllib.load(f)
    if 'version' not in pyproject['project'].get('dynamic', []):
        errors.append(
            "madspace/pyproject.toml pins its own version instead of reading it "
            "from VERSION; madspace must be released in lockstep with MadGraph")

    sync = subprocess.run(
        [sys.executable, pjoin(ROOT, 'pyamplicol', 'tools', 'release', 'sync_version.py'),
         '--check', '--version', version],
        capture_output=True, text=True)
    if sync.returncode:
        errors.append(sync.stderr.strip() + "; run pyamplicol/tools/release/sync_version.py")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def archive_source(version, workdir):
    """git-archive HEAD into workdir/MG7_v<version_>/, preserving symlinks."""
    version_ = version.replace('.', '_')
    filepath = pjoin(workdir, f"MG7_v{version_}")
    os.makedirs(filepath)

    archive = subprocess.Popen(
        ['git', '-C', ROOT, 'archive', '--format=tar', 'HEAD'],
        stdout=subprocess.PIPE)
    extract = subprocess.Popen(['tar', '-x', '-C', filepath],
                                stdin=archive.stdout)
    archive.stdout.close()
    extract.communicate()
    if archive.wait() or extract.returncode:
        print("ERROR: git archive | tar extraction failed", file=sys.stderr)
        sys.exit(1)
    return filepath


def prune_bin(filepath):
    for name in ('create_release.py', 'create_aloha_release.py'):
        candidate = pjoin(filepath, 'bin', name)
        if path.exists(candidate):
            os.remove(candidate)
    compile_py = pjoin(filepath, 'bin', 'compile.py')
    if path.exists(compile_py):
        os.rename(compile_py, pjoin(filepath, 'bin', '.compile.py'))


def materialize_config(filepath):
    input_dir = pjoin(filepath, 'input')
    shutil.copy(pjoin(input_dir, '.mg7_configuration_default.txt'),
                pjoin(input_dir, 'mg7_configuration.txt'))
    for card in ('default_run_card_lo.dat', 'default_run_card_nlo.dat',
                 'default_run_card_mg7.toml', 'default_switch.txt'):
        src = pjoin(input_dir, f'.{card}')
        if path.exists(src):
            shutil.copy(src, pjoin(input_dir, card))
    shutil.copy(pjoin(input_dir, 'proc_card_default.dat'),
                pjoin(filepath, 'proc_card.dat'))


def vendor_offline_tools(filepath):
    """Bundle offline copies of collier/ninja/SMWidth and the
    HEPToolsInstaller scripts, so a release tarball can be used without
    network access to the HEPTools installer's usual sources. Required for a
    release: any failure here aborts it rather than shipping a tarball
    silently missing the offline installers."""
    vendor_dir = pjoin(filepath, 'vendor')
    os.makedirs(vendor_dir, exist_ok=True)

    clone_dir = pjoin(filepath, '..', 'HEPToolsInstallers')
    subprocess.run(
        ['git', 'clone', '--depth', '1',
         'https://github.com/mg5amcnlo/HEPToolsInstallers.git', clone_dir],
        check=True)
    shutil.rmtree(pjoin(clone_dir, '.git'))
    with tarfile.open(pjoin(vendor_dir, 'OfflineHEPToolsInstaller.tar.gz'), 'w:gz') as tf:
        tf.add(clone_dir, arcname='HEPToolsInstallers')

    sys.path.insert(0, path.dirname(clone_dir))
    from HEPToolsInstallers.HEPToolInstaller import _HepTools
    collier_link = _HepTools['collier']['tarball'][1] % _HepTools['collier']
    ninja_link = _HepTools['ninja']['tarball'][1] % _HepTools['ninja']
    urllib.request.urlretrieve(collier_link, pjoin(vendor_dir, 'collier.tar.gz'))
    urllib.request.urlretrieve(ninja_link, pjoin(vendor_dir, 'ninja.tar.gz'))
    urllib.request.urlretrieve(
        'http://madgraph.phys.ucl.ac.be/Downloads/SMWidth.tgz',
        pjoin(vendor_dir, 'SMWidth.tar.gz'))


def write_authors(filepath):
    first_contribs = get_first_contributions(ROOT)
    with open(pjoin(filepath, 'input', 'authors.md'), 'w') as f:
        for author in sorted(first_contribs):
            f.write(f'{author} {first_contribs[author]}\n')


def write_release_marker(filepath, version, wheels_dir=None):
    """Write input/.release, read back by madspace/install.py to decide
    whether the PyPI wheel may be offered. When wheels_dir is given, record
    the filenames of the wheels actually built for this release, so
    install.py can check platform/Python availability purely locally
    (no PyPI query) instead of guessing from the CI build matrix."""
    wheels = []
    if wheels_dir:
        wheels = sorted(path.basename(w) for w in glob.glob(pjoin(wheels_dir, '*.whl')))

    with open(pjoin(filepath, 'input', '.release'), 'w') as f:
        f.write(f'version = {version}\n')
        f.write(f'date = {date.today().isoformat()}\n')
        f.write(f'wheels = {",".join(wheels)}\n')


def make_tarball(workdir, filepath, output_dir, version):
    for pyc in glob.glob(pjoin(filepath, '**', '*.pyc'), recursive=True):
        os.remove(pyc)

    os.makedirs(output_dir, exist_ok=True)
    tarname = pjoin(output_dir, f"MG7_v{version}.tar.gz")
    subprocess.run(['tar', 'czf', tarname, '-C', workdir, path.basename(filepath)],
                    check=True)
    return tarname


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', required=True,
                         help="Version being released, e.g. 0.2.0. Must match "
                              "VERSION.")
    parser.add_argument('--output', default='dist',
                         help="Directory to write the tarball into (default: dist/).")
    parser.add_argument('--skip-vendor', action='store_true',
                         help="Skip bundling offline collier/ninja/SMWidth/HEPToolsInstaller.")
    parser.add_argument('--wheels-dir', default=None,
                         help="Directory holding the madspace wheels built for this "
                              "release (e.g. the downloaded cibuildwheel artifacts). "
                              "Their filenames are recorded in input/.release so "
                              "install.py can check platform/Python availability "
                              "without querying PyPI.")
    parser.add_argument('--check-only', action='store_true',
                         help="Only run the version consistency check, then exit.")
    args = parser.parse_args()

    check_versions(args.version)
    if args.check_only:
        return

    workdir = pjoin(args.output, '_work')
    filepath = archive_source(args.version, workdir)
    prune_bin(filepath)
    materialize_config(filepath)
    if not args.skip_vendor:
        vendor_offline_tools(filepath)
    write_authors(filepath)
    write_release_marker(filepath, args.version, wheels_dir=args.wheels_dir)
    tarname = make_tarball(workdir, filepath, args.output, args.version)
    shutil.rmtree(workdir)

    print(f"Created {tarname}")


if __name__ == '__main__':
    main()
