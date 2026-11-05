################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Tests of the s-hat map with a knee (pole=-25) of Template/LO/Source/transpole.f"""
from __future__ import absolute_import
import os
import shutil
import subprocess
import tempfile
import unittest

import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join

DRIVER = """
      program test_knee
      implicit none
      double precision shat_floor, knee_frac, knee_log
      common/to_shat_knee/shat_floor, knee_frac, knee_log
      double precision small_width_treatment
      common/narrow_width/small_width_treatment
      double precision xc, xb, x, y, jac, xr, jr, h, yp, ym, jd, y2, j2
      double precision err_inv, err_jac, err_fd, ymin
      integer i, n
      logical mono
      small_width_treatment = 1d-6
      xc = %(xc)r
      xb = %(xb)r
      shat_floor = xc
      knee_frac = 0.03d0
      knee_log = log(xb/xc)
      n = 20000
      err_inv = 0d0
      err_jac = 0d0
      err_fd = 0d0
      ymin = -1d0
      mono = .true.
      do i = 1, n-1
         x = dble(i)/dble(n)
         jac = 1d0
         call transpole(-25d0, xb, x, y, jac)
         if (y .le. ymin) mono = .false.
         ymin = y
         jr = 1d0
         call untranspole(-25d0, xb, xr, y, jr)
         err_inv = max(err_inv, abs(xr-x))
         err_jac = max(err_jac, abs(jr/jac-1d0))
         h = 1d-7*min(x, 1d0-x)
         jd = 1d0
         call transpole(-25d0, xb, x+h, yp, jd)
         jd = 1d0
         call transpole(-25d0, xb, x-h, ym, jd)
c        skip the two break points of the map
         if (abs(x-xc).gt.2*h .and. abs(x-xc-knee_frac).gt.2*h)
     $        err_fd = max(err_fd, abs((yp-ym)/(2*h)/jac-1d0))
      enddo
      write(*,*) 'INV', err_inv
      write(*,*) 'JAC', err_jac
      write(*,*) 'FD', err_fd
      write(*,*) 'MONO', mono
c     the floor and the knee are reached exactly at x=xc and x=xc+frac
      jac = 1d0
      call transpole(-25d0, xb, xc, y, jac)
      write(*,*) 'FLOOR', y/xc-1d0
      jac = 1d0
      call transpole(-25d0, xb, xc+knee_frac, y, jac)
      write(*,*) 'KNEE', y/xb-1d0
c     not set: same as pole=-2
      shat_floor = 0d0
      jac = 1d0
      call transpole(-25d0, xb, 0.5d0, y, jac)
      j2 = 1d0
      call transpole(-2d0, xb, 0.5d0, y2, j2)
      write(*,*) 'UNSET', abs(y/y2-1d0)+abs(jac/j2-1d0)
      end
"""


@unittest.skipIf(misc.which('gfortran') is None, 'gfortran not available')
class TestShatKneeMap(unittest.TestCase):
    """transpole/untranspole for pole=-25 (s-hat below non-forced BW)"""

    def run_driver(self, xc, xb):
        tmp = tempfile.mkdtemp()
        try:
            with open(pjoin(tmp, 'driver.f'), 'w') as f:
                f.write(DRIVER % {'xc': xc, 'xb': xb})
            subprocess.check_call(['gfortran', '-O0', '-o', pjoin(tmp, 'test'),
                                   pjoin(tmp, 'driver.f'),
                                   pjoin(MG5DIR, 'Template', 'LO', 'Source', 'transpole.f')],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = subprocess.check_output([pjoin(tmp, 'test')]).decode()
        finally:
            shutil.rmtree(tmp)
        res = {}
        for line in out.splitlines():
            key, val = line.split()
            res[key] = val
        return res

    def check(self, xc, xb):
        res = self.run_driver(xc, xb)
        self.assertLess(float(res['INV']), 1e-12)
        self.assertLess(float(res['JAC']), 1e-10)
        self.assertLess(float(res['FD']), 1e-5)
        self.assertEqual(res['MONO'], 'T')
        self.assertLess(abs(float(res['FLOOR'])), 1e-12)
        self.assertLess(abs(float(res['KNEE'])), 1e-12)
        self.assertLess(float(res['UNSET']), 1e-14)

    def test_w_knee_13tev(self):
        """W + j at 13 TeV: floor 30 GeV, knee M_W - 5 Gamma_W"""
        self.check((30. / 13000.) ** 2, (70.18 / 13000.) ** 2)

    def test_knee_far_above_floor(self):
        """no cut at all: floor 1/stot, knee from two nested tops"""
        self.check(1. / 13000. ** 2, (331. / 13000.) ** 2)
