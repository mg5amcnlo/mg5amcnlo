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
"""Test pyamplicol/install.py, its version sync and its source hash.

Nothing here builds anything: main() runs with pip, rustup and the build
stubbed out.
"""

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MG_ROOT = Path(__file__).resolve().parents[3]
PYAMPLICOL = MG_ROOT / 'pyamplicol'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = load('pyamplicol_install', PYAMPLICOL / 'install.py')
source_hash = load('pyamplicol_source_hash', PYAMPLICOL / 'source_hash.py')


class TestReleaseWheel(unittest.TestCase):
    """--bin is only offered for a pyamplicol wheel built for this platform."""

    def available(self, wheels, system, machine):
        info = {'version': '0.2.1', 'wheels': ','.join(wheels)}
        with mock.patch.object(install, '_release_info', return_value=info), \
                mock.patch('platform.system', return_value=system), \
                mock.patch('platform.machine', return_value=machine):
            return install._release_wheel_available()

    WHEELS = [
        'madspace-0.2.1-cp312-cp312-manylinux_2_28_x86_64.whl',
        'pyamplicol-0.2.1-cp311-abi3-macosx_11_0_arm64.whl',
        'pyamplicol-0.2.1-cp311-abi3-macosx_11_0_x86_64.whl',
        'pyamplicol-0.2.1-cp311-abi3-manylinux_2_28_x86_64.whl',
    ]

    def test_abi3_wheels_match_any_supported_platform(self):
        self.assertTrue(self.available(self.WHEELS, 'Linux', 'x86_64'))
        self.assertTrue(self.available(self.WHEELS, 'Darwin', 'arm64'))
        self.assertTrue(self.available(self.WHEELS, 'Darwin', 'x86_64'))

    def test_no_wheel_for_linux_aarch64(self):
        self.assertFalse(self.available(self.WHEELS, 'Linux', 'aarch64'))

    def test_madspace_wheels_do_not_count(self):
        self.assertFalse(self.available(self.WHEELS[:1], 'Linux', 'x86_64'))


class TestMain(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        patches = {
            'INSTALL_DIR': self.tmp / 'install',
            'BUILD_DIR': self.tmp / 'build',
            'SETTINGS_FILE': self.tmp / 'install_settings.json',
        }
        for name, value in patches.items():
            patcher = mock.patch.object(install, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.commands = []
        for name, side_effect in (
            ('run', lambda cmd, **kw: self.commands.append([str(c) for c in cmd])),
            ('setup_rust', lambda env: env),
            ('install_build_deps', lambda env, system: env),
            ('fetch_prepared_models', lambda env: True),
            ('regenerate_prepared_models', lambda env: self.commands.append(['regenerate'])),
            ('check_license', lambda target: None),
        ):
            patcher = mock.patch.object(install, name, side_effect=side_effect)
            patcher.start()
            self.addCleanup(patcher.stop)

    def main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            install.main(list(argv))

    def test_source_build_targets_the_local_install_dir(self):
        self.main('--source', '-y', '-j', '3')
        build = self.commands[-1]
        self.assertIn('--no-build-isolation', build)
        # Linux hosts are newer than manylinux_2_28; the wheel is for this host only.
        self.assertIn('--config-settings=pyamplicol.host-wheel=true', build)
        self.assertIn(f'--target={self.tmp / "install"}', build)
        self.assertEqual(json.loads(install.SETTINGS_FILE.read_text()), {'mode': 'source'})

    def test_build_environment_is_local_and_incremental(self):
        env = install.build_environment(4)
        self.assertEqual(env['PYAMPLICOL_LOCAL_BUILD'], '1')
        self.assertEqual(env['PYAMPLICOL_BUILD_JOBS'], '4')
        self.assertEqual(env['PYAMPLICOL_BUILD_CACHE_ROOT'], str(install.CACHE_DIR))

    def test_build_models_regenerates_instead_of_downloading(self):
        self.main('--source', '-y', '--build-models')
        self.assertIn(['regenerate'], self.commands)
        install.fetch_prepared_models.assert_not_called()

    def test_stale_release_models_are_regenerated(self):
        install.fetch_prepared_models.side_effect = lambda env: False
        self.main('--source', '-y')
        self.assertIn(['regenerate'], self.commands)

    def test_bin_pins_the_madgraph_version(self):
        self.main('--bin')
        self.assertIn(f'pyamplicol=={install._pyamplicol_version()}', self.commands[-1])
        self.assertEqual(json.loads(install.SETTINGS_FILE.read_text()), {'mode': 'bin'})

    def test_license_is_checked_on_the_installed_package(self):
        self.main('--source', '-y')
        install.check_license.assert_called_once_with(self.tmp / 'install')

    def test_system_install_has_no_target(self):
        self.main('--source', '-y', '--system')
        self.assertFalse(any(a.startswith('--target') for a in self.commands[-1]))
        install.check_license.assert_called_once_with(None)

    def test_clean_wipes_build_and_install_but_keeps_the_settings(self):
        for directory in ('build', 'install'):
            (self.tmp / directory).mkdir()
        install.SETTINGS_FILE.write_text('{"mode": "source"}')
        self.main('--source', '-y', '--clean')
        self.assertFalse((self.tmp / 'build').exists())
        self.assertTrue(install.SETTINGS_FILE.exists())


class TestVersionSync(unittest.TestCase):

    def test_pyamplicol_version_follows_madgraph(self):
        result = subprocess.run(
            [sys.executable, str(PYAMPLICOL / 'tools' / 'release' / 'sync_version.py'), '--check'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestSourceHash(unittest.TestCase):

    def copy_sources(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        for name in ('src', 'rust'):
            (root / name / 'pyamplicol').mkdir(parents=True)
        (root / 'src' / 'pyamplicol' / '__init__.py').write_text('x = 1\n')
        for name in source_hash._SOURCE_FILES:
            (root / name).write_text(name)
        return root

    def test_source_change_changes_the_hash(self):
        root = self.copy_sources()
        before = source_hash.compute_source_hash(root)
        (root / 'src' / 'pyamplicol' / '__init__.py').write_text('x = 2\n')
        self.assertNotEqual(source_hash.compute_source_hash(root), before)

    def test_generated_files_are_ignored(self):
        root = self.copy_sources()
        before = source_hash.compute_source_hash(root)
        package = root / 'src' / 'pyamplicol'
        (package / '_rusticol.abi3.so').write_bytes(b'binary')
        (package / '_build_info.json').write_text('{}')
        (package / '_source_hash.py').write_text('SOURCE_HASH = ""\n')
        models = package / 'assets' / 'prepared_models'
        models.mkdir(parents=True)
        (models / 'built-in-sm-jit-o2-x86_64.pyamplicol-model').write_bytes(b'pack')
        (models / 'built-in-sm-jit-o2-x86_64.metadata.json').write_text('{}')
        self.assertEqual(source_hash.compute_source_hash(root), before)


if __name__ == '__main__':
    unittest.main()
