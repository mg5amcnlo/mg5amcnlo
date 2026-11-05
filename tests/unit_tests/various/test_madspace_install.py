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
"""Test how madspace/install.py resolves the source-build options.

Per option, a flag given on the command line wins; otherwise --yes reuses the
saved settings of the previous source build, else the platform default. Before
this was fixed, --yes was checked first, so a scripted
``install.py --source --yes --cuda --cuda-arch 80`` silently built for the CPU.
Nothing here builds anything: main() runs with pip/cmake stubbed out.
"""

from __future__ import absolute_import
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

INSTALL_PY = Path(__file__).resolve().parents[3] / 'madspace' / 'install.py'


def load_installer():
    spec = importlib.util.spec_from_file_location('madspace_install', INSTALL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = load_installer()

# settings a previous CPU-only source build would have saved
CPU_SAVED = {
    'mode': 'source',
    'cuda': False,
    'cuda_arch': '75',
    'hip': False,
    'hip_arch': 'gfx900',
    'openblas': False,
    'simd': True,
    'docs': False,
    'build_type': 'Debug',
}

# settings a previous GPU source build would have saved
GPU_SAVED = dict(CPU_SAVED, cuda=True, cuda_arch='86', hip=True,
                 hip_arch='gfx1100', simd=False, build_type='RelWithDebInfo')


def parse(*argv):
    return install.build_parser().parse_args(list(argv))


class TestResolveCompileOptions(unittest.TestCase):
    """resolve_compile_options: flag > saved (--yes only) > platform default."""

    def resolve(self, saved, *argv):
        return install.resolve_compile_options(parse(*argv), saved)

    def test_yes_flags_beat_saved_settings(self):
        options = self.resolve(CPU_SAVED, '--source', '--yes', '--cuda', '--hip',
                               '--openblas', '--no-simd', '--no-debug')
        self.assertEqual(options, {'cuda': True, 'hip': True, 'openblas': True,
                                   'simd': False, 'build_type': 'Release'})

    def test_yes_flags_beat_platform_defaults(self):
        options = self.resolve({}, '--source', '--yes', '--cuda', '--simd',
                               '--full-debug')
        self.assertEqual(options, dict(
            install._PLATFORM_SOURCE_DEFAULTS,
            cuda=True, simd=True, build_type='Debug'))

    def test_yes_negative_flags_beat_saved_settings(self):
        # an explicit False must win too, not only an explicit True
        options = self.resolve(GPU_SAVED, '--yes', '--no-cuda', '--no-hip')
        self.assertFalse(options['cuda'])
        self.assertFalse(options['hip'])

    def test_yes_unflagged_options_reuse_saved_settings(self):
        options = self.resolve(CPU_SAVED, '--source', '--yes', '--cuda')
        self.assertEqual(options, {'cuda': True, 'hip': False, 'openblas': False,
                                   'simd': True, 'build_type': 'Debug'})

    def test_yes_alone_reuses_saved_settings(self):
        options = self.resolve(GPU_SAVED, '--source', '--yes')
        self.assertEqual(options, {k: GPU_SAVED[k]
                                   for k in install.COMPILE_OPTIONS})

    def test_yes_alone_without_saved_settings_is_platform_default(self):
        for saved in ({}, {'mode': 'bin'}):
            options = self.resolve(saved, '--source', '--yes')
            self.assertEqual(options, {k: install._PLATFORM_SOURCE_DEFAULTS[k]
                                       for k in install.COMPILE_OPTIONS})

    def test_yes_legacy_debug_setting(self):
        legacy = {'mode': 'source', 'debug': True}
        self.assertEqual(self.resolve(legacy, '--yes')['build_type'],
                         'RelWithDebInfo')
        self.assertEqual(self.resolve(legacy, '--yes', '--no-debug')['build_type'],
                         'Release')

    def test_flags_without_yes_ignore_saved_settings(self):
        # without --yes the flags describe the whole build
        options = self.resolve(CPU_SAVED, '--source', '--cuda')
        self.assertEqual(options, dict(
            {k: install._PLATFORM_SOURCE_DEFAULTS[k]
             for k in install.COMPILE_OPTIONS}, cuda=True))

    def test_no_yes_no_flag_asks(self):
        self.assertIsNone(self.resolve(GPU_SAVED, '--source'))
        # --docs and the architectures are not menu entries: still ask
        self.assertIsNone(self.resolve(GPU_SAVED, '--source', '--docs',
                                       '--cuda-arch', '80'))


class TestResolveGpuArch(unittest.TestCase):
    """resolve_gpu_arch: --cuda-arch/--hip-arch > saved > default."""

    def test_flag_beats_saved_under_yes(self):
        args = parse('--source', '--yes', '--cuda', '--cuda-arch', '80',
                     '--hip', '--hip-arch', 'gfx942')
        self.assertEqual(install.resolve_gpu_arch(args, GPU_SAVED, 'cuda'), '80')
        self.assertEqual(install.resolve_gpu_arch(args, GPU_SAVED, 'hip'), 'gfx942')

    def test_flag_beats_saved_without_yes(self):
        args = parse('--source', '--cuda', '--cuda-arch', '75;80;86')
        self.assertEqual(install.resolve_gpu_arch(args, GPU_SAVED, 'cuda'),
                         '75;80;86')

    def test_saved_then_default(self):
        args = parse('--source', '--yes', '--cuda', '--hip')
        self.assertEqual(install.resolve_gpu_arch(args, GPU_SAVED, 'cuda'), '86')
        self.assertEqual(install.resolve_gpu_arch(args, GPU_SAVED, 'hip'), 'gfx1100')
        self.assertEqual(install.resolve_gpu_arch(args, {}, 'cuda'),
                         install.DEFAULT_CUDA_ARCH)
        self.assertEqual(install.resolve_gpu_arch(args, {}, 'hip'),
                         install.DEFAULT_HIP_ARCH)


class TestMainSourceBuildCommand(unittest.TestCase):
    """main() must pass the resolved options on to the pip/CMake command."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mg7_madspace_install_')
        self.install_dir = Path(self.tmpdir) / 'install'
        self.build_dir = Path(self.tmpdir) / 'build'
        (self.install_dir / 'madspace').mkdir(parents=True)  # "previous install"
        self.build_dir.mkdir()
        self.commands = []
        # the settings are read and written for real: whether they survive a
        # --clean is part of what is under test
        patches = [
            mock.patch.object(install, 'INSTALL_DIR', self.install_dir),
            mock.patch.object(install, 'BUILD_DIR', self.build_dir),
            mock.patch.object(install, 'SETTINGS_FILE',
                              Path(self.tmpdir) / 'install_settings.json'),
            mock.patch.object(install, '_LEGACY_SETTINGS_FILE',
                              self.build_dir / 'install_settings.json'),
            mock.patch.object(install, '_release_version', return_value=None),
            mock.patch.object(install, 'install_build_deps',
                              side_effect=lambda system=False: {}),
            mock.patch.object(install, 'set_build_parallelism',
                              side_effect=lambda env, jobs: env),
            mock.patch.object(install, 'run',
                              side_effect=lambda cmd, env=None:
                              self.commands.append(cmd)),
            mock.patch.object(install, 'ask_string',
                              side_effect=AssertionError('prompted')),
            mock.patch.object(install, 'ask_compile_options',
                              side_effect=AssertionError('prompted')),
            mock.patch.object(install, '_NONINTERACTIVE', False),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def main(self, saved, *argv):
        if saved:
            install.save_settings(saved)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            install.main(list(argv))
        self.assertEqual(len(self.commands), 1)
        return self.commands[0], install.load_settings(), out.getvalue()

    def test_yes_cuda_flags_reach_cmake(self):
        cmd, written, _ = self.main(CPU_SAVED, '--source', '--yes', '--cuda',
                                    '--cuda-arch', '80')
        self.assertIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=80', cmd)
        # the options not given on the command line reuse the saved ones
        self.assertIn('-Ccmake.define.ENABLE_SIMD=ON', cmd)
        self.assertIn('-Ccmake.build-type=Debug', cmd)
        self.assertEqual(written['cuda'], True)
        self.assertEqual(written['cuda_arch'], '80')

    def test_yes_hip_flags_reach_cmake(self):
        cmd, written, _ = self.main({}, '--source', '--yes', '--hip',
                                    '--hip-arch', 'gfx942')
        self.assertIn('-Ccmake.define.ENABLE_HIP=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_HIP_ARCHITECTURES=gfx942', cmd)
        self.assertNotIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertEqual(written['hip_arch'], 'gfx942')

    def test_yes_alone_reuses_saved_gpu_build(self):
        cmd, written, _ = self.main(GPU_SAVED, '--source', '--yes')
        self.assertIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86', cmd)
        self.assertIn('-Ccmake.define.ENABLE_HIP=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_HIP_ARCHITECTURES=gfx1100', cmd)
        self.assertIn('-Ccmake.build-type=RelWithDebInfo', cmd)
        self.assertEqual({k: written[k] for k in GPU_SAVED}, GPU_SAVED)

    def test_yes_openblas_and_debug_flags_reach_cmake(self):
        cmd, _, _ = self.main(CPU_SAVED, '--source', '--yes', '--openblas',
                              '--debug')
        self.assertIn('-Ccmake.define.ENABLE_OPENBLAS=ON', cmd)
        self.assertIn('-Ccmake.build-type=RelWithDebInfo', cmd)

    def test_clean_wipes_both_directories_but_keeps_the_settings(self):
        # a clean rebuild must not also forget how madspace is to be built:
        # the settings are read before the wipe and live outside both dirs
        cmd, written, out = self.main(GPU_SAVED, '--source', '--yes', '--clean')
        self.assertFalse(self.install_dir.exists())
        self.assertFalse(self.build_dir.exists())
        self.assertIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86', cmd)
        self.assertEqual({k: written[k] for k in GPU_SAVED}, GPU_SAVED)
        self.assertIn('Removed', out)

    def test_clean_still_takes_the_command_line_flags(self):
        cmd, written, _ = self.main(GPU_SAVED, '--source', '--yes', '--clean',
                                    '--no-cuda', '--cuda-arch', '80')
        self.assertNotIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertEqual(written['cuda_arch'], '80')

    def test_a_failed_clean_build_keeps_the_settings_for_the_retry(self):
        # --clean has removed install/, so the retry finds no previous
        # installation: the saved settings must be used all the same
        install.save_settings(GPU_SAVED)
        with mock.patch.object(install, 'run', side_effect=SystemExit(1)), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(SystemExit):
            install.main(['--source', '--yes', '--clean'])
        self.assertFalse(self.install_dir.exists())
        cmd, _, _ = self.main({}, '--source', '--yes')
        self.assertIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86', cmd)

    def test_system_build_reuses_the_saved_settings(self):
        # --system installs into site-packages, so install/ never exists
        shutil.rmtree(self.install_dir)
        cmd, _, _ = self.main(GPU_SAVED, '--source', '--system', '--yes')
        self.assertIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('-Ccmake.define.CMAKE_CUDA_ARCHITECTURES=86', cmd)
        self.assertNotIn('--target=%s' % self.install_dir, cmd)

    def test_a_plain_rebuild_keeps_the_build_tree(self):
        # the incremental build tree is what makes a rebuild fast: only an
        # explicit --clean may remove it
        self.main(CPU_SAVED, '--source', '--yes')
        self.assertTrue(self.build_dir.is_dir())
        self.assertTrue(self.install_dir.is_dir())

    def test_arch_without_backend_warns(self):
        cmd, _, out = self.main(CPU_SAVED, '--source', '--yes',
                                '--cuda-arch', '80')
        self.assertNotIn('-Ccmake.define.ENABLE_CUDA=ON', cmd)
        self.assertIn('--cuda-arch has no effect', out)


class TestSettingsFileLocation(unittest.TestCase):
    """Checked on the real constants, not the patched ones the other tests use."""

    def test_settings_live_outside_the_directories_clean_deletes(self):
        for directory in (install.BUILD_DIR, install.INSTALL_DIR):
            self.assertNotIn(directory, install.SETTINGS_FILE.parents)


class TestCleanAndSettingsFile(unittest.TestCase):
    """clean_install_dirs / load_settings on real files."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix='mg7_madspace_clean_'))
        self.install_dir = self.tmpdir / 'install'
        self.build_dir = self.tmpdir / 'build'
        (self.install_dir / 'madspace').mkdir(parents=True)
        self.build_dir.mkdir()
        (self.build_dir / 'CMakeFiles').mkdir()
        for patch in [
            mock.patch.object(install, 'INSTALL_DIR', self.install_dir),
            mock.patch.object(install, 'BUILD_DIR', self.build_dir),
            mock.patch.object(install, 'SETTINGS_FILE',
                              self.tmpdir / 'install_settings.json'),
            mock.patch.object(install, '_LEGACY_SETTINGS_FILE',
                              self.build_dir / 'install_settings.json'),
        ]:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def clean(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return install.clean_install_dirs()

    def test_removes_both_directories(self):
        removed = self.clean()
        self.assertEqual(set(removed), {self.build_dir, self.install_dir})
        self.assertFalse(self.build_dir.exists())
        self.assertFalse(self.install_dir.exists())

    def test_missing_directories_are_not_an_error(self):
        self.clean()
        self.assertEqual(self.clean(), [])

    def test_settings_survive_a_clean(self):
        install.save_settings(GPU_SAVED)
        self.clean()
        self.assertEqual(install.load_settings(), GPU_SAVED)

    def test_settings_in_the_old_location_are_still_read(self):
        # they used to live in build/, which --clean removes
        legacy = self.build_dir / 'install_settings.json'
        legacy.write_text(json.dumps(CPU_SAVED))
        self.assertEqual(install.load_settings(), CPU_SAVED)
        # ... and the next build writes them to the new one
        install.save_settings(CPU_SAVED)
        self.clean()
        self.assertEqual(install.load_settings(), CPU_SAVED)

    def test_no_settings_anywhere(self):
        self.assertEqual(install.load_settings(), {})


if __name__ == '__main__':
    unittest.main()
