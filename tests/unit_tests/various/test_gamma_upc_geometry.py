################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tests of the nuclear geometry of gamma-UPC (OpticalGlauber_Geometry.f90),
compiled and run directly (needs gfortran)."""

from __future__ import absolute_import

import os
import shutil
import subprocess
import tempfile
import unittest

import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join


class TestGammaUPCGeometry(unittest.TestCase):
    """compile OpticalGlauber_Geometry.f90 (a few seconds) and test it directly"""

    gammaupc = pjoin(MG5DIR, 'Template', 'Common', 'Source', 'PDF', 'gammaUPC')
    # OpticalGlauber_Geometry and what it uses (tbessk: bessk0/bessk1), in
    # compilation order
    modules = ['tbessk', 'nielsen_generalized_polylog', 'nintlib', 'interpolation',
               'OpticalGlauber_Geometry']

    # Woods-Saxon T_B of beam B (args: nucleus A, nucleus B) from its grid
    # (IMETH=2) and from the direct integration (IMETH=0), after beam A made
    # its grid (IMETH=1); grids in memory, at s = 0, R_B/4, ..., 2 R_B
    tahat_driver = """
program check_tahat
  USE OpticalGlauber_Geometry
  implicit none
  character(len=7)::nameA,nameB
  real(kind(1d0))::A1,Z1,R1,aa1,w1,A2,Z2,R2,aa2,w2,t,tb,s
  integer::i
  call get_command_argument(1,nameA)
  call get_command_argument(2,nameB)
  call GetNuclearInfo(nameA,A1,Z1,R1,aa1,w1)
  call GetNuclearInfo(nameB,A2,Z2,R2,aa2,w2)
  t=TAhat_WoodsSaxon(0d0,0d0,R1,w1,aa1,A1,1,.FALSE.)
  do i=0,8
     s=0.25d0*i*R2
     tb=TAhat_WoodsSaxon(s,0d0,R2,w2,aa2,A2,2,.FALSE.)
     t=TAhat_WoodsSaxon(s,0d0,R2,w2,aa2,A2,0)
     write(*,'(a,3es24.15)') 'T', s, tb, t
  enddo
end program check_tahat
"""

    @classmethod
    def setUpClass(cls):
        cls.fc = misc.which('gfortran')
        if not cls.fc:
            raise unittest.SkipTest('gfortran not available')
        cls.tmpdir = tempfile.mkdtemp(prefix='upc_geometry')
        for mod in cls.modules:
            shutil.copy(pjoin(cls.gammaupc, '%s.f90' % mod), cls.tmpdir)
            cls.compile(['-c', '%s.f90' % mod])
        with open(pjoin(cls.tmpdir, 'check_tahat.f90'), 'w') as fsock:
            fsock.write(cls.tahat_driver)
        cls.compile(['-o', 'check_tahat', 'check_tahat.f90'] +
                    ['%s.o' % mod for mod in cls.modules])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'tmpdir', None):
            shutil.rmtree(cls.tmpdir)

    @classmethod
    def compile(cls, args):
        out = subprocess.run([cls.fc, '-O'] + args, cwd=cls.tmpdir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        if out.returncode:
            raise Exception('compilation failed: %s\n%s' % (' '.join(args), out.stdout))

    def test_woods_saxon_beam_b_grid(self):
        """TAhat_WoodsSaxon with IMETH=2 interpolates the grid of beam B, which
        can be another nucleus than beam A: it agrees with the direct
        integration (IMETH=0) up to the spline error. It used to interpolate
        the grid of beam A, which was only right for identical beams."""

        for beams in [('Pb208', 'O16'), ('O16', 'Pb208'), ('Pb208', 'Pb208')]:
            out = subprocess.run([pjoin(self.tmpdir, 'check_tahat')] + list(beams),
                                 cwd=self.tmpdir, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            self.assertEqual(out.returncode, 0, out.stdout)
            rows = [[float(x) for x in l.split()[1:]] for l in out.stdout.split('\n')
                    if l.startswith('T ')]
            self.assertEqual(len(rows), 9, out.stdout)
            for s, grid, direct in rows:
                self.assertGreater(direct, 0.)
                self.assertLess(abs(grid / direct - 1.), 1e-3, (beams, s, grid, direct))
