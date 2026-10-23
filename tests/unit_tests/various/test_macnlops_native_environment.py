"""Native LHAPDF ABI selection is independent of the hard-generator ABI."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from madgraph.various import macnlops as mc


class TestNativeLibraryEnvironment(unittest.TestCase):
    def fixture(self, base):
        prefix, backend, legacy = (base / x for x in ('pythia', 'configured', 'hard_generator'))
        for path in (prefix / 'lib', backend, legacy):
            path.mkdir(parents=True)
        (prefix / 'lib/libpythia8lhapdf6.so').write_text('compiled plugin fixture')
        for path in (backend, legacy):
            (path / 'libLHAPDF.so').write_text('distinct ABI fixture')
        (prefix / 'Makefile.inc').write_text('LHAPDF6_LIB=-L%s -Wl,-rpath,%s -lLHAPDF\n' % (backend, backend))
        return prefix, backend, legacy

    def test_inherited_legacy_abi_cannot_shadow_native_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix, backend, legacy = self.fixture(Path(tmp))
            with patch.dict(os.environ, {'LD_LIBRARY_PATH': str(legacy)}):
                environment = mc.native_library_environment(prefix)
                self.assertEqual(environment['LD_LIBRARY_PATH'].split(os.pathsep),
                                 [str(prefix / 'lib'), str(backend), str(legacy)])
                self.assertEqual(os.environ['LD_LIBRARY_PATH'], str(legacy))
                info = Path(tmp) / 'pdfs/set/set.info'
                shower = mc.shower_environment(prefix, info)
                self.assertEqual(shower['LD_LIBRARY_PATH'], environment['LD_LIBRARY_PATH'])
                self.assertEqual(shower['PYTHIA8DATA'], str(prefix / 'share/Pythia8/xmldoc'))
                self.assertEqual(shower['LHAPDF_DATA_PATH'].split(os.pathsep)[0], str(info.parent.parent))

    def test_ambiguous_configured_abi_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix, backend, legacy = self.fixture(Path(tmp))
            (prefix / 'Makefile.inc').write_text('LHAPDF6_LIB=-L%s -L%s -lLHAPDF\n' % (backend, legacy))
            with self.assertRaisesRegex(mc.MatchingError, 'unique configured'):
                mc.native_library_environment(prefix)
