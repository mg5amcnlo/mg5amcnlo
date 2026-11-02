################################################################################
#
# Copyright (c) 2012 The MadGraph7 Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import os
import gzip
import tempfile
import unittest
import subprocess
from unittest import mock
import madgraph.various.misc as misc
class TEST_misc(unittest.TestCase):

    def test_pool_chunksize(self):
        """same chunks as CPython's map without chunksize, from the pool size"""
        for nprocs in (1, 3, 8):
            pool = mock.Mock(_processes=nprocs, _pool=[])
            for nitems in (0, 1, 4 * nprocs - 1, 4 * nprocs, 4 * nprocs + 1, 100):
                chunksize, extra = divmod(nitems, 4 * nprocs)
                expected = max(1, chunksize + bool(extra))
                self.assertEqual(misc.pool_chunksize(pool, nitems), expected)


    def test_gzip_keep_preserves_plain_source(self):
        data = b"<event>\n1 2 3\n</event>\n" * 100
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'events.lhe')
            with open(path, 'wb') as out:
                out.write(data)
            misc.gzip(path, keep=True)
            with open(path, 'rb') as inp:
                self.assertEqual(inp.read(), data)
            with gzip.open(path + '.gz', 'rb') as inp:
                self.assertEqual(inp.read(), data)

    def test_gzip_failure_keeps_source_and_leaves_no_archive(self):
        data = b"<event>\n1 2 3\n</event>\n" * 100
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'events.lhe')
            with open(path, 'wb') as out:
                out.write(data)
            target = os.path.join(tmp, 'missing_dir', 'events.lhe.gz')
            misc.gzip(path, stdout=target, error=False)
            with open(path, 'rb') as inp:
                self.assertEqual(inp.read(), data)
            self.assertFalse(os.path.exists(target))
            with self.assertRaises(OSError):
                misc.gzip(path, stdout=target, error=True)
            self.assertTrue(os.path.exists(path))

    def test_interrupted_gzip_removes_partial_archive(self):
        data = b"<event>\n1 2 3\n</event>\n" * 100
        def truncated_copy(fsrc, fdst, length):
            fdst.write(b'\x1f\x8b')
            raise OSError('disk full')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'events.lhe')
            with open(path, 'wb') as out:
                out.write(data)
            with mock.patch.object(misc.shutil, 'copyfileobj',
                                   side_effect=truncated_copy):
                misc.gzip(path, error=False)
            self.assertFalse(os.path.exists(path + '.gz'))
            with open(path, 'rb') as inp:
                self.assertEqual(inp.read(), data)

    def test_external_gzip_failure_removes_partial_archive(self):
        original = (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
                    misc._gzip_tool_max_cores)
        def failing_tool(command, *args, **kwargs):
            with open(command[-1] + '.gz', 'wb') as out:
                out.write(b'\x1f\x8b')
            return 1
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'events.lhe')
                with open(path, 'wb') as out:
                    out.write(b'<event>\n</event>\n')
                with mock.patch.object(misc.open_file, 'find_valid',
                                       return_value='gzip'), \
                     mock.patch.object(misc, 'call', side_effect=failing_tool):
                    misc.configure_gzip({'use_pigz': False})
                    misc.gzip(path, forceexternal=True, error=False)
                    self.assertFalse(os.path.exists(path + '.gz'))
                    self.assertTrue(os.path.exists(path))
                    with self.assertRaises(RuntimeError):
                        misc.gzip(path, forceexternal=True, error=True)
                    self.assertFalse(os.path.exists(path + '.gz'))
                    self.assertTrue(os.path.exists(path))
        finally:
            (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
             misc._gzip_tool_max_cores) = original

    def test_external_gzip_keep_passes_thread_limit(self):
        original = (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
                    misc._gzip_tool_max_cores)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'events.lhe')
                with open(path, 'wb') as out:
                    out.write(b'<event>\n</event>\n')
                with mock.patch.object(misc.open_file, 'find_valid',
                                       return_value='pigz'), \
                     mock.patch.object(misc, 'call', return_value=0) as call:
                    misc.configure_gzip({'use_pigz': True, 'nb_core': 2})
                    misc.gzip(path, forceexternal=True, keep=True)
                    call.assert_called_once_with(
                        ['pigz', '-p', '2', '-k', '-f', path])
        finally:
            (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
             misc._gzip_tool_max_cores) = original

    def test_configure_gzip_resets_pigz_threads(self):
        original = (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
                    misc._gzip_tool_max_cores)
        try:
            with mock.patch.object(misc.open_file, 'find_valid',
                                   side_effect=lambda choices, _: choices[0]):
                misc.configure_gzip({'use_pigz': True, 'nb_core': 3})
                self.assertEqual(misc._gzip_tool, 'pigz')
                self.assertTrue(misc._gzip_tool_supports_multithreading)
                self.assertEqual(misc._gzip_tool_max_cores, 3)
                misc.configure_gzip({'use_pigz': False})
                self.assertEqual(misc._gzip_tool, 'gzip')
                self.assertFalse(misc._gzip_tool_supports_multithreading)
                self.assertIsNone(misc._gzip_tool_max_cores)
        finally:
            (misc._gzip_tool, misc._gzip_tool_supports_multithreading,
             misc._gzip_tool_max_cores) = original
    
    def test_equal(self):
        
        eq = misc.equal
        
        self.assertFalse(eq(1,2,1))
        self.assertTrue(eq(1,1.2,1))
        self.assertFalse(eq(1, 1.2, 2))
        
        self.assertFalse(eq(10,20,1))
        self.assertTrue(eq(10,12,1))
        self.assertFalse(eq(10, 12, 2))        
        
        self.assertTrue(eq(100,1e2))
        self.assertFalse(eq(100,1e2 + 1e-3))
        self.assertTrue(eq(100,1e2 + 1e-4))
        self.assertTrue(eq(100,1e2 + 1e-5))

        self.assertFalse(eq(0.1,0.2,1))
        self.assertTrue(eq(0.1,0.12,1))
        self.assertFalse(eq(0.10, 0.12, 2))         

        self.assertFalse(eq(-0.1,-0.2,1))
        self.assertTrue(eq(-0.1,-0.12,1))
        self.assertFalse(eq(-0.10, -0.12, 2))         
        
        self.assertFalse(eq(-10,-20,1))
        self.assertTrue(eq(-10,-12,1))
        self.assertFalse(eq(-10, -12, 2)) 

        self.assertTrue(eq(-100,-1e2))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-100,-1e2 + 1e-4))
        self.assertTrue(eq(-100,-1e2 + 1e-5))
        
        self.assertTrue(eq(1,1.0))
        self.assertTrue(eq(1,1.0, 1))
        self.assertTrue(eq(1,1.0, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 7))
        self.assertTrue(eq(1,1.0 - 2e-8, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 8))
        self.assertFalse(eq(1,1.0 + 2e-7, 8))
        self.assertTrue(eq(9,9.0 + 2e-8, 7))
        self.assertTrue(eq(9,9.0 - 2e-8, 7))
        self.assertTrue(eq(9,9.0 + 2e-8, 8))
        self.assertFalse(eq(9,9.0 + 2e-8, 9))
        self.assertFalse(eq(1,-1.0))
        self.assertTrue(eq(0 ,0e-5))
        

        self.assertTrue(eq(80.419, 80.419002))
        for i in range(1,4):
            self.assertTrue(eq(81.966005, 81.891469,i))
        for i in range(4,7):
            self.assertFalse(eq(81.966005, 81.891469,i))
        
        # Check negative number
        self.assertTrue(eq(-1,-1))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 8))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 8))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 8))
        self.assertFalse(eq(-1,-1.0 + 1e-8, 9))
        self.assertFalse(eq(-1,-1.0 + 2e-8, 9))
        self.assertFalse(eq(-1,-1.0 - 2e-8, 9))

        self.assertFalse(eq(-1,1.0))
        self.assertTrue(eq(-0 ,-0e-5))
        
        self.assertTrue(eq(-100,-1e2))
        self.assertTrue(eq(-100,-1e2 + 1e-6))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-80.419, -80.419002))
        
        # check with 0
        self.assertTrue(eq(0 ,1e-11))  
        self.assertTrue(eq(0 ,1e-8))
        self.assertTrue(eq(0 ,1e-7))
        self.assertFalse(eq(0 ,1e-6))          
        self.assertFalse(eq(0 ,1e-1))

        self.assertTrue(eq(1e-11, 0))  
        self.assertTrue(eq(1e-8, 0))
        self.assertTrue(eq(1e-7, 0))
        self.assertFalse(eq(1e-6, 0))          
        self.assertFalse(eq(1e-1, 0))
        
        self.assertFalse(eq(0 ,1e-11, zero_limit=False))  
        self.assertFalse(eq(0 ,1e-8, zero_limit=False))
        self.assertFalse(eq(0 ,1e-7, zero_limit=False))
        self.assertFalse(eq(0 ,1e-6, zero_limit=False))          
        self.assertFalse(eq(0 ,1e-1, zero_limit=False))
        
        self.assertFalse(eq(1e-11, 0, zero_limit=False))  
        self.assertFalse(eq(1e-8, 0, zero_limit=False))
        self.assertFalse(eq(1e-7, 0, zero_limit=False))
        self.assertFalse(eq(1e-6, 0, zero_limit=False))          
        self.assertFalse(eq(1e-1, 0, zero_limit=False))         

    def test_ordered_set(self):

        set = misc.OrderedSet

        a = set(['a'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 0)

        a = set(['a', 'b'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 1)

    def test_popen_with_closed_sys_stdout_and_stderr_stdout(self):
        class ClosedStdout(object):
            def fileno(self):
                raise ValueError('I/O operation on closed file')

        with mock.patch('sys.__stdout__', ClosedStdout()):
            proc = misc.Popen(['echo', 'ok'], stdout=None, stderr=subprocess.STDOUT)
            self.assertEqual(proc.wait(), 0)


_HEPTOOLS_LHAPDF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), 'HEPTools', 'lhapdf6_py3', 'bin', 'lhapdf-config')


@unittest.skipUnless(os.path.exists(_HEPTOOLS_LHAPDF), 'no HEPTools lhapdf6_py3')
class TEST_import_python_lhapdf(unittest.TestCase):
    """The HEPTools lhapdf python module has no rpath to libLHAPDF: setting
    LD_LIBRARY_PATH from inside the process is too late for the loader, so
    import_python_lhapdf loads the library itself first."""

    def test_import_without_ld_library_path(self):
        import sys
        root = os.path.dirname(os.path.dirname(os.path.dirname(_HEPTOOLS_LHAPDF)))
        env = dict(os.environ)
        env.pop('LD_LIBRARY_PATH', None)
        env.pop('PYTHONPATH', None)
        code = ("import sys; sys.path.insert(0, %r)\n"
                "import madgraph.various.misc as misc\n"
                "module = misc.import_python_lhapdf(%r)\n"
                "print(module.version())\n" % (root, _HEPTOOLS_LHAPDF))
        result = subprocess.run([sys.executable, '-c', code], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout.strip().splitlines()[-1], r'^\d+\.\d+')


class TEST_pythia8_main164(unittest.TestCase):
    """main164, the Pythia8 driver of the LO shower, is found or compiled with
    the Makefile of the Pythia8 examples directory (a stub here)."""

    # stands in for the Pythia8 examples Makefile: 'make main164' copies the
    # source to an executable, 'make mainMG' (the HEPToolsInstaller rule) too
    makefile = ("main164: main164.cc\n\tcp main164.cc main164; chmod +x main164\n"
                "%s")

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.py8 = os.path.join(self.tmpdir, 'pythia8')
        self.examples = os.path.join(self.py8, 'share', 'Pythia8', 'examples')
        os.makedirs(self.examples)

    def tearDown(self):
        import shutil
        os.chmod(self.examples, 0o755)
        shutil.rmtree(self.tmpdir)

    # the lines of Pythia8's main164.cc (8.316 to 8.318) that the
    # per-jet-sample DJR patch anchors on
    main164_djr_anchors = (
        '  pythia.settings.addWord("InternalAnalysis:output", "analysis.dat");\n'
        '  vector<Hist> d01Hists, d12Hists, d23Hists, d34Hists;\n'
        '      d34Hists.push_back(Hist("d34", 100., 0., 3.));\n'
        '        if (dij.size() > 3) d34Hists.back().fill(log10(dij[3]), w);\n'
        '      output << "<run id=\\"" << i << "\\"" << "\\">\\n";\n'
        '      d34Hists[i].table(output, false, false);\n'
        '      output << "</run>\\n";\n')

    def write_sources(self, mainMG=False, hepmc=2, source=None):
        with open(os.path.join(self.examples, 'main164.cc'), 'w') as fsock:
            fsock.write(source if source is not None else '#!/bin/sh\necho main164\n')
        rule = "mainMG: main164.cc\n\techo mainMG > main164; chmod +x main164\n" \
                                                               if mainMG else ''
        with open(os.path.join(self.examples, 'Makefile'), 'w') as fsock:
            fsock.write(self.makefile % rule)
        # the HepMC setup written by the Pythia8 configure script
        with open(os.path.join(self.examples, 'Makefile.inc'), 'w') as fsock:
            fsock.write('CXX_COMMON=-O2 -DHEPMC2HACK -DHEPMC2 -DGZIP\n'
                        'HEPMC2_USE=%s\nHEPMC2_INCLUDE=-I/old/include\n'
                        'HEPMC2_LIB=-L/old/lib -lHepMC\nHEPMC3_USE=%s\n'
                        'HEPMC3_INCLUDE=\nHEPMC3_LIB=\n' % (
                            'true' if hepmc == 2 else 'false',
                            'true' if hepmc == 3 else 'false'))

    def make_hepmc(self, version):
        """a fake HepMC<version> installation, as HEPToolsInstaller lays it out"""
        prefix = os.path.join(self.tmpdir, 'hepmc3' if version == 3 else 'hepmc')
        include = os.path.join(prefix, 'include', 'HepMC3' if version == 3 else 'HepMC')
        os.makedirs(include)
        os.makedirs(os.path.join(prefix, 'lib'))
        open(os.path.join(include, 'GenEvent.h'), 'w').close()
        open(os.path.join(prefix, 'lib', 'libHepMC%s.dylib' % ('3' if version == 3 else '')),
             'w').close()
        return prefix

    def test_find_main164(self):
        self.assertEqual(misc.find_pythia8_main164(self.py8), (None, None))
        self.write_sources()
        self.assertEqual(misc.find_pythia8_main164(self.py8), (None, self.examples))
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          os.path.join(self.tmpdir, 'nopythia8'))

    def test_compile_in_examples(self):
        self.write_sources()
        executable = misc.get_pythia8_main164(self.py8)
        self.assertEqual(executable, os.path.join(self.examples, 'main164'))
        self.assertEqual(misc.find_pythia8_main164(self.py8),
                         (executable, self.examples))

    def test_compile_prefers_mainMG_rule(self):
        self.write_sources(mainMG=True)
        executable = misc.get_pythia8_main164(self.py8)
        self.assertEqual(open(executable).read().strip(), 'mainMG')

    def test_compile_in_fallback_for_readonly_examples(self):
        self.write_sources()
        os.chmod(self.examples, 0o555)
        if os.access(self.examples, os.W_OK):
            self.skipTest('running as a user that can write read-only directories')
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164, self.py8)

        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback)
        self.assertEqual(executable, os.path.join(fallback, 'main164'))
        self.assertFalse(os.path.exists(os.path.join(self.examples, 'main164')))
        # reused as long as it was built against the same Pythia8
        mtime = os.path.getmtime(executable)
        self.assertEqual(misc.get_pythia8_main164(self.py8, fallback_dir=fallback),
                         executable)
        self.assertEqual(os.path.getmtime(executable), mtime)
        # rebuilt when made against another Pythia8, even if make would
        # consider the old executable up to date
        with open(os.path.join(fallback, 'BUILD_STAMP'), 'w') as fsock:
            fsock.write('/another/pythia8')
        with open(executable, 'w') as fsock:
            fsock.write('stale')
        misc.get_pythia8_main164(self.py8, fallback_dir=fallback)
        self.assertIn('echo main164', open(executable).read())
        self.assertEqual(open(os.path.join(fallback, 'BUILD_STAMP')).read().split('\n')[0],
                         os.path.realpath(self.py8))
        # rebuilt when Pythia8 is reinstalled at the same place
        with open(executable, 'w') as fsock:
            fsock.write('stale')
        os.chmod(self.examples, 0o755)
        source = os.path.join(self.examples, 'main164.cc')
        with open(source, 'a') as fsock:
            fsock.write('echo reinstalled\n')
        os.utime(source, (os.path.getmtime(source) + 10,) * 2)
        os.chmod(self.examples, 0o555)
        misc.get_pythia8_main164(self.py8, fallback_dir=fallback)
        self.assertIn('echo reinstalled', open(executable).read())

    def test_too_old_pythia8(self):
        """Pythia8 < 8.311 has no main164: say that a newer one is needed"""

        bindir = os.path.join(self.py8, 'bin')
        os.makedirs(bindir)
        with open(os.path.join(bindir, 'pythia8-config'), 'w') as fsock:
            fsock.write('#!/bin/sh\n[ "$1" = "--version" ] && echo 8.245\nexit 0\n')
        os.chmod(os.path.join(bindir, 'pythia8-config'), 0o755)
        self.assertEqual(misc.get_pythia8_version(self.py8), '8.245')
        with self.assertRaises(misc.MadGraph5Error) as error:
            misc.get_pythia8_main164(self.py8)
        self.assertIn('version 8.245', str(error.exception))
        self.assertIn('8.311', str(error.exception))

    def test_pythia8_version_from_xmldoc(self):
        xmldoc = os.path.join(self.py8, 'share', 'Pythia8', 'xmldoc')
        os.makedirs(xmldoc)
        with open(os.path.join(xmldoc, 'Version.xml'), 'w') as fsock:
            fsock.write('<parmfix name="Pythia:versionNumber" default="8.313">\n')
        self.assertEqual(misc.get_pythia8_version(self.py8), '8.313')
        self.assertEqual(misc.get_pythia8_version(os.path.join(self.tmpdir, 'none')), None)

    def test_other_hepmc_version_from_a_source_tree(self):
        """In a built source tree, the examples Makefile takes the library from
        ../lib; a copy of the examples must be told where it is"""

        source_tree = os.path.join(self.tmpdir, 'pythia8source')
        examples = os.path.join(source_tree, 'examples')
        os.makedirs(examples)
        os.makedirs(os.path.join(source_tree, 'lib'))
        open(os.path.join(source_tree, 'lib', 'libpythia8.a'), 'w').close()
        self.examples = examples
        self.write_sources(hepmc=2)
        with open(os.path.join(examples, 'Makefile.inc'), 'a') as fsock:
            fsock.write('PREFIX_LIB=/usr/local/lib\nPREFIX_INCLUDE=/usr/local/include\n')
        prefix = os.path.realpath(self.make_hepmc(3))
        executable = misc.get_pythia8_main164(source_tree, hepmc_version=3,
                                              hepmc_paths=[prefix])
        makefile_inc = open(os.path.join(os.path.dirname(executable), 'Makefile.inc')).read()
        self.assertIn('PREFIX_LIB=%s' % os.path.realpath(os.path.join(source_tree, 'lib')),
                      makefile_inc)
        self.assertIn('PREFIX_INCLUDE=%s' % os.path.realpath(os.path.join(source_tree, 'include')),
                      makefile_inc)
        self.assertNotIn('/usr/local', makefile_inc)

    def test_other_hepmc_version_needs_a_configured_pythia8(self):
        self.write_sources(hepmc=2)
        os.remove(os.path.join(self.examples, 'Makefile.inc'))
        prefix = self.make_hepmc(3)
        with self.assertRaises(misc.MadGraph5Error) as error:
            misc.get_pythia8_main164(self.py8, hepmc_version=3, hepmc_paths=[prefix])
        self.assertIn('Makefile.inc', str(error.exception))

    def test_hepmc_version_of_the_pythia8_build(self):
        self.write_sources(hepmc=2)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), 2)
        self.write_sources(hepmc=3)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), 3)
        self.write_sources(hepmc=None)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), None)

    def test_find_hepmc(self):
        self.assertEqual(misc.find_hepmc(3, [self.tmpdir]), None)
        prefix = self.make_hepmc(3)
        self.assertEqual(misc.find_hepmc(3, [None, self.tmpdir, prefix]),
                         (os.path.realpath(prefix),
                          os.path.realpath(os.path.join(prefix, 'lib'))))
        # a HepMC3 installation is not a HepMC2 one
        self.assertEqual(misc.find_hepmc(2, [prefix]), None)

    def test_main164_djr_patch(self):
        """the per-jet-sample split, the bin errors and the '<run id' fix are
        patched in; an unknown source is left alone"""
        patched = misc.patch_pythia8_main164_djrs(self.main164_djr_anchors)
        self.assertIsNotNone(patched)
        self.assertIn('pythia.settings.addFlag("InternalAnalysis:jetSamples", false);',
                      patched)
        self.assertIn('dynamic_pointer_cast<JetMatchingMadgraph>', patched)
        self.assertIn('jetMatchingPtr->nMEpartons().first', patched)
        self.assertIn('pythia.mode("Merging:nRecluster")', patched)
        self.assertIn('<jetsample njet=', patched)
        self.assertNotIn('<< "\\"" << "\\">', patched)
        self.assertTrue(misc.pythia8_main164_writes_jet_sample_djrs(patched))
        self.assertFalse(misc.pythia8_main164_writes_jet_sample_djrs(
                                                    self.main164_djr_anchors))
        self.assertIsNone(misc.patch_pythia8_main164_djrs('int main() {}\n'))

    def test_jet_sample_djrs_build_a_patched_main164(self):
        self.write_sources(source=self.main164_djr_anchors)
        executable = misc.get_pythia8_main164(self.py8, jet_sample_djrs=True)
        shared = os.path.join(self.examples, 'main164_djr')
        self.assertEqual(executable, os.path.join(shared, 'main164'))
        # the stub make copies the source: it is the patched one, which knows
        # InternalAnalysis:jetSamples (the stock one does not)
        self.assertIn('nMEpartons', open(executable).read())
        self.assertTrue(misc.pythia8_main164_knows_jet_samples(executable))
        self.assertFalse(misc.pythia8_main164_knows_jet_samples(
            misc.get_pythia8_main164(self.py8)))
        # Pythia8's own main164.cc is never modified
        self.assertEqual(open(os.path.join(self.examples, 'main164.cc')).read(),
                         self.main164_djr_anchors)
        # built once
        os.remove(os.path.join(shared, 'main164.cc'))
        self.assertEqual(misc.get_pythia8_main164(self.py8, jet_sample_djrs=True),
                         executable)
        # the default one is not affected
        self.assertEqual(misc.get_pythia8_main164(self.py8),
                         os.path.join(self.examples, 'main164'))

    def test_jet_sample_djrs_not_patched_when_not_needed_or_possible(self):
        # a main164 that splits per jet sample by itself
        self.write_sources(source=self.main164_djr_anchors +
            '  pythia.settings.addFlag("InternalAnalysis:jetSamples", false);\n')
        self.assertEqual(misc.get_pythia8_main164(self.py8, jet_sample_djrs=True),
                         os.path.join(self.examples, 'main164'))
        # a main164.cc the patch does not know: the stock one
        os.remove(os.path.join(self.examples, 'main164'))
        self.write_sources()
        self.assertEqual(misc.get_pythia8_main164(self.py8, jet_sample_djrs=True),
                         os.path.join(self.examples, 'main164'))
        self.assertFalse(os.path.exists(os.path.join(self.examples, 'main164_djr')))

    def test_jet_sample_djrs_with_another_hepmc_version(self):
        self.write_sources(hepmc=2, source=self.main164_djr_anchors)
        prefix = self.make_hepmc(3)
        executable = misc.get_pythia8_main164(self.py8, hepmc_version=3,
                                  hepmc_paths=[prefix], jet_sample_djrs=True)
        self.assertEqual(executable, os.path.join(self.examples,
                                                  'main164_hepmc3_djr', 'main164'))
        self.assertIn('nMEpartons', open(executable).read())

    def test_same_hepmc_version_uses_the_default_main164(self):
        self.write_sources(hepmc=2)
        executable = misc.get_pythia8_main164(self.py8, hepmc_version=2)
        self.assertEqual(executable, os.path.join(self.examples, 'main164'))

    def test_other_hepmc_version_gets_its_own_main164(self):
        self.write_sources(hepmc=2)
        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        # no HepMC3 installation: clear error
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          self.py8, fallback_dir=fallback, hepmc_version=3)
        # found next to Pythia8, as HEPToolsInstaller installs it
        prefix = os.path.realpath(self.make_hepmc(3))
        self.assertEqual(prefix, os.path.realpath(os.path.join(self.py8, os.pardir, 'hepmc3')))
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback,
                                              hepmc_version=3)
        # shared by all processes, next to the default main164
        shared = os.path.join(self.examples, 'main164_hepmc3')
        self.assertEqual(executable, os.path.join(shared, 'main164'))
        self.assertFalse(os.path.exists(os.path.join(self.examples, 'main164')))
        makefile_inc = open(os.path.join(shared, 'Makefile.inc')).read()
        self.assertIn('HEPMC2_USE=false', makefile_inc)
        self.assertIn('HEPMC3_USE=true', makefile_inc)
        self.assertIn('HEPMC3_INCLUDE=-I%s/include' % prefix, makefile_inc)
        self.assertIn('HEPMC3_LIB=-L%s/lib -Wl,-rpath,%s/lib -lHepMC3' % (prefix, prefix),
                      makefile_inc)
        self.assertIn('CXX_COMMON=-O2 -DGZIP', makefile_inc)
        self.assertNotIn('/old/', makefile_inc)
        # the default one is not affected
        self.assertEqual(misc.get_pythia8_main164(self.py8),
                         os.path.join(self.examples, 'main164'))

    def test_other_hepmc_version_in_readonly_pythia8(self):
        self.write_sources(hepmc=2)
        prefix = self.make_hepmc(3)
        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        os.chmod(self.examples, 0o555)
        if os.access(self.examples, os.W_OK):
            self.skipTest('running as a user that can write read-only directories')
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          self.py8, hepmc_version=3, hepmc_paths=[prefix])
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback,
                                              hepmc_version=3, hepmc_paths=[prefix])
        self.assertEqual(executable, fallback + '_hepmc3/main164')

    def test_pythia8_hepmc_flags(self):
        # a pythia8-config that only knows HepMC2 (Pythia8 configured with it)
        bindir = os.path.join(self.py8, 'bin')
        os.makedirs(bindir)
        with open(os.path.join(bindir, 'pythia8-config'), 'w') as fsock:
            fsock.write('#!/bin/sh\n[ "$1" = "--hepmc2" ] && echo "-I/h2/include -lHepMC"\n'
                        'exit 0\n')
        os.chmod(os.path.join(bindir, 'pythia8-config'), 0o755)
        self.assertEqual(misc.get_pythia8_hepmc_flags(self.py8, 2), '-I/h2/include -lHepMC')
        # HepMC3 is taken from its installation next to Pythia8
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_hepmc_flags, self.py8, 3)
        prefix = os.path.realpath(self.make_hepmc(3))
        self.assertEqual(misc.get_pythia8_hepmc_flags(self.py8, 3),
                         '-I%s/include -L%s/lib -Wl,-rpath,%s/lib -lHepMC3' % ((prefix,)*3))

    def test_hepmc_paths_use_the_option_of_each_version(self):
        """hepmc3_path is looked at first for HepMC3, hepmc_path for HepMC2"""

        import madgraph.interface.common_run_interface as common_run
        class Interface(object):
            options = {'hepmc_path': '/h2', 'hepmc3_path': '/h3',
                       'heptools_install_dir': '/heptools'}
        paths3 = common_run.CommonRunCmd.get_hepmc_paths(Interface(), 3)
        paths2 = common_run.CommonRunCmd.get_hepmc_paths(Interface(), 2)
        self.assertEqual(paths3[:2], ['/h3', '/heptools/hepmc3'])
        self.assertEqual(paths2[:2], ['/h2', '/heptools/hepmc'])
        self.assertNotIn('/h2', paths3)
        self.assertNotIn('/h3', paths2)

    def test_hepmc_file_version(self):
        import gzip
        hepmc2 = os.path.join(self.tmpdir, 'events.hepmc')
        with open(hepmc2, 'w') as fsock:
            fsock.write('\nHepMC::Version 2.06.09\n'
                        'HepMC::IO_GenEvent-START_EVENT_LISTING\nE 0 -1\n')
        hepmc3 = os.path.join(self.tmpdir, 'events3.hepmc.gz')
        with gzip.open(hepmc3, 'wt') as fsock:
            fsock.write('HepMC::Version 3.02.07\n'
                        'HepMC::Asciiv3-START_EVENT_LISTING\nW Weight\nE 0 3 5\n')
        self.assertEqual(misc.hepmc_file_version(hepmc2), 2)
        self.assertEqual(misc.hepmc_file_version(hepmc3), 3)
        self.assertEqual(misc.hepmc_file_version(os.path.join(self.tmpdir, 'none')), None)
