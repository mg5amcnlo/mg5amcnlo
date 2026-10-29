"""Check the shipped Delta table without requiring a shower installation."""

import ctypes
import gzip
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SUDGEN = ROOT / 'vendor/SudGen'
CERNLIB = ROOT / 'Template/NLO/Source/CERNLIB'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestDefaultSudakovTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory(prefix='mg5_sudakov_table_')
        cls.addClassCleanup(cls.tempdir.cleanup)
        work = Path(cls.tempdir.name)
        # Exports copy the selected table as a standalone Fortran source.
        # Compile that copy, rather than adding SudGen's directory to the
        # include path, so this also checks the exported source contract.
        selected = work / 'selected.f'
        shutil.copyfile(SUDGEN / 'sudakov.f', selected)
        archived = work / 'raw_pdfmode1.f'
        archived.write_bytes(gzip.decompress(
            (SUDGEN / 'sudakov_r515_PDFmode1.f.gz').read_bytes()))
        cls.libraries = []
        for name, source in [('selected', selected), ('archived', archived)]:
            library = work / (name + '.so')
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-std=legacy', '-shared',
                '-fPIC', str(source),
                *(str(CERNLIB / filename) for filename in
                  ['dfint.f', 'kerset.f', 'abend.f']),
                '-o', str(library)], cwd=work, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            loaded = ctypes.CDLL(str(library))
            cls.libraries.append(loaded)
            function = loaded.pysudakov_
            function.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_double
            setattr(cls, name, staticmethod(function))
        cls.masses = (ctypes.c_double * 21)(
            .33, .33, .5, 1.5, 4.8, 0, 0, 0, 0, 0,
            .000510998928, 0, .1056583715, 0, 1.77682, 0, 0, 0, 0, 0, 0)

    @classmethod
    def value(cls, function, scale, mass, pdg, topology):
        args = [ctypes.c_double(scale), ctypes.c_double(mass),
                ctypes.c_int(pdg), ctypes.c_int(topology)]
        result = function(*(ctypes.byref(arg) for arg in args), cls.masses)
        if not math.isfinite(result):
            raise AssertionError('Nonfinite Sudakov for %r' %
                                 ((scale, mass, pdg, topology),))
        return result

    def test_default_entries_represent_one_directed_colour_line(self):
        points = [(1., 1.), (3., 20.), (20., 91.1876),
                  (20., 500.), (125., 500.), (7000., 7000.)]
        for topology in [1, 2, 3, 4]:
            for pdg in [1, 2, 3, 4, 5, 6, 21]:
                # FF trials contain both final radiators. Each gluon
                # radiator in any topology has two colour connections.
                ends = (2 if topology == 2 else 1) * (2 if pdg == 21 else 1)
                for scale, mass in points:
                    for signed_pdg in [pdg, -pdg]:
                        with self.subTest(topology=topology, pdg=signed_pdg,
                                          scale=scale, mass=mass):
                            actual = self.value(self.selected, scale, mass,
                                                signed_pdg, topology)
                            raw = self.value(self.archived, scale, mass,
                                             signed_pdg, topology)
                            if ends == 1:
                                self.assertEqual(actual, raw)
                            else:
                                self.assertGreaterEqual(actual, 0.)
                                self.assertAlmostEqual(actual**ends, max(raw, 0.),
                                                       delta=2.e-15)

    def test_two_gluon_lines_reconstruct_single_raw_radiator(self):
        # II, IF and FI trials contain one active radiator. Its two colour
        # lines must reproduce the one full-gluon raw API probability.
        for topology in [1, 3, 4]:
            for scale, mass in [(1., 1.), (3., 20.), (20., 91.1876),
                                (20., 500.), (125., 500.), (7000., 7000.)]:
                for pdg in [21, -21]:
                    with self.subTest(topology=topology, pdg=pdg,
                                      scale=scale, mass=mass):
                        line = self.value(self.selected, scale, mass,
                                          pdg, topology)
                        full = self.value(self.archived, scale, mass,
                                          pdg, topology)
                        self.assertGreaterEqual(line, 0.)
                        self.assertAlmostEqual(line * line, max(full, 0.),
                                               delta=2.e-15)

    def test_colour_flows_reconstruct_raw_no_emission_ratios(self):
        # The raw FF API covers two quarks or two gluons. The consumer
        # reconstructs the same system using two or four directed lines.
        cases = [(2, 1, 1), (21, 1, 2), (21, 3, 2), (21, 4, 2),
                 (2, 2, 2), (21, 2, 4)]
        for pdg, topology, ends in cases:
            with self.subTest(pdg=pdg, topology=topology, ends=ends):
                line_stop = self.value(self.selected, 20., 500., pdg, topology)
                line_start = self.value(self.selected, 200., 500., pdg, topology)
                full_stop = self.value(self.archived, 20., 500., pdg, topology)
                full_start = self.value(self.archived, 200., 500., pdg, topology)
                self.assertGreater(full_stop, 0.)
                self.assertLess(full_stop, full_start)
                line_ratio = line_stop / line_start
                full_ratio = full_stop / full_start
                self.assertAlmostEqual(line_ratio**ends, full_ratio,
                                       delta=2.e-15)


if __name__ == '__main__':
    unittest.main()
