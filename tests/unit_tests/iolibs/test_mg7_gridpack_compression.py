"""Compression policy of the standalone mg7 gridpack launcher."""

import gzip
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


_TEMPLATE = (Path(__file__).resolve().parents[3] / "madgraph" / "iolibs"
             / "template_files" / "mg7" / "gridpack.py")


def _gridpack_module():
    # The compression function does not need the optional compiled madspace
    # runtime. Stub only imports so this focused test runs on a CPU checkout.
    madspace = types.ModuleType("madspace")
    loader = types.ModuleType("custom_function_loader")
    loader.load_custom_function = lambda *args, **kwargs: None
    # bin/gridpack_setup.py, whose names gridpack.py imports
    setup = types.ModuleType("gridpack_setup")
    setup.__getattr__ = lambda name: None
    spec = importlib.util.spec_from_file_location("mg7_gridpack_compression", _TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"madspace": madspace,
                                       "custom_function_loader": loader,
                                       "gridpack_setup": setup}):
        spec.loader.exec_module(module)
    return module


class TestGridpackCompression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gridpack = _gridpack_module()

    def test_small_file_uses_python_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.lhe")
            data = b"<LesHouchesEvents>\n<event>\n1\n</event>\n" * 100
            with open(path, "wb") as out:
                out.write(data)
            with mock.patch.object(self.gridpack.subprocess, "run") as run:
                self.gridpack.compress_lhe(path, True, 2)
            run.assert_not_called()
            self.assertFalse(os.path.exists(path))
            with gzip.open(path + ".gz", "rb") as inp:
                self.assertEqual(inp.read(), data)

    def test_large_file_respects_pigz_preference_and_threads(self):
        path = "/tmp/events.lhe"
        with mock.patch.object(self.gridpack.os.path, "getsize",
                               return_value=257_000_000), \
             mock.patch.object(self.gridpack.shutil, "which",
                               side_effect=lambda name: "/usr/bin/" + name), \
             mock.patch.object(self.gridpack.subprocess, "run") as run:
            self.gridpack.compress_lhe(path, "auto", 3)
            run.assert_called_once_with(
                ["/usr/bin/pigz", "-p", "3", "-f", path], check=True)
            run.reset_mock()
            self.gridpack.compress_lhe(path, False, 3)
            run.assert_called_once_with(
                ["/usr/bin/gzip", "-f", path], check=True)

    @unittest.skipUnless(shutil.which("pigz"), "pigz unavailable")
    def test_pigz_output_is_gzip_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.lhe")
            data = b"<event> 1 2 3 4 5 </event>\n" * 1000
            with open(path, "wb") as out:
                out.write(data)
            with mock.patch.object(self.gridpack.os.path, "getsize",
                                   return_value=257_000_000):
                self.gridpack.compress_lhe(path, True, 2)
            self.assertFalse(os.path.exists(path))
            with gzip.open(path + ".gz", "rb") as inp:
                self.assertEqual(inp.read(), data)


if __name__ == "__main__":
    unittest.main()
