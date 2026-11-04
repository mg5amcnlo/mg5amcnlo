################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
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
"""Tests of get_me_frame_boosts (Template/Common/Source/impose_frame_rest.f),
the me_frame rest frame shared by madevent, aMC@NLO and MadSpin."""
from __future__ import absolute_import
import os
import random
import re
import shutil
import subprocess
import tempfile
import unittest

import madgraph.various.misc as misc
from madgraph import MG5DIR

pjoin = os.path.join

NPART = 5

# Reads events (nincoming, mask, rapidity, momenta) from stdin; for each one
# prints how far the frame moves when the input is boosted along z, and the
# properties the frame has to have.
DRIVER = """
      program test_me_frame
      implicit none
      integer npart
      parameter (npart=%(npart)d)
      integer ids(npart), nincoming, i, j, isel, nsel
      double precision p(0:3,npart), pl(0:3,npart)
      double precision a(0:3,npart), b(0:3,npart)
      double precision c(0:3,npart), d(0:3,npart)
      double precision eta, ql(0:3), scale, n1(3), n2(3), r1, r2
 10   read(*,*,end=99) nincoming, (ids(i),i=1,npart), eta
      do i=1,npart
         read(*,*) (p(j,i),j=0,3)
      enddo
c     the same momenta, seen from a frame boosted along z by eta
      ql(0)=cosh(eta)
      ql(1)=0d0
      ql(2)=0d0
      ql(3)=sinh(eta)
      do i=1,npart
         call boostx(p(0,i), ql, pl(0,i))
      enddo
      call to_frame(p, ids, nincoming, a)
      call to_frame(pl, ids, nincoming, b)
      call old_frame(p, ids, c)
      call old_frame(pl, ids, d)
      scale=0d0
      do i=1,npart
         scale=max(scale, abs(a(0,i)))
      enddo
      write(*,'(a,es12.4)') 'DIFF ', maxval(abs(a-b))/scale
      write(*,'(a,es12.4)') 'OLDDIFF ', maxval(abs(c-d))/scale
      write(*,'(a,es12.4)') 'TOOLD ', maxval(abs(a-c))/scale
      nsel=0
      isel=0
      do i=1,npart
         if (ids(i).eq.1) then
            nsel=nsel+1
            isel=i
         endif
      enddo
      if (nsel.eq.1) then
         write(*,'(a,3es12.4)') 'REST ', (a(j,isel),j=1,3)
      endif
c     the bisector of beam 1 and of minus beam 2
      r1=sqrt(a(1,1)**2+a(2,1)**2+a(3,1)**2)
      r2=sqrt(a(1,2)**2+a(2,2)**2+a(3,2)**2)
      do j=1,3
         n1(j)=a(j,1)/r1
         n2(j)=-a(j,2)/r2
      enddo
      write(*,'(a,3es12.4)') 'BISECT ', n1(1)+n2(1), n1(2)+n2(2),
     &     n1(3)+n2(3)
      write(*,'(a)') 'END'
      goto 10
 99   end

      subroutine to_frame(p, ids, nincoming, pout)
      implicit none
      integer npart
      parameter (npart=%(npart)d)
      integer ids(npart), nincoming, i
      double precision p(0:3,npart), pout(0:3,npart)
      double precision qz(0:3), qt(0:3), tmp(0:3)
      call get_me_frame_boosts(ids, npart, nincoming, p, qz, qt)
      do i=1,npart
         call boostx(p(0,i), qz, tmp)
         call boostx(tmp, qt, pout(0,i))
      enddo
      call impose_frame_rest(ids, npart, pout)
      end

c     the single boost along the momentum of the selected legs, which the
c     LO, NLO and MadSpin boost_to_frame used to apply
      subroutine old_frame(p, ids, pout)
      implicit none
      integer npart
      parameter (npart=%(npart)d)
      integer ids(npart), i, j
      double precision p(0:3,npart), pout(0:3,npart), q(0:3)
      q(:)=0d0
      do i=1,npart
         if (ids(i).eq.1) then
            do j=0,3
               q(j)=q(j)+p(j,i)
            enddo
         endif
      enddo
      do j=1,3
         q(j)=-q(j)
      enddo
      do i=1,npart
         call boostx(p(0,i), q, pout(0,i))
      enddo
      call impose_frame_rest(ids, npart, pout)
      end
"""


def boostx_source():
    """HELAS boostx, as the processes get it, cut out of the ALOHA template so
    that the test does not need the ALOHA module that file opens with."""
    text = open(pjoin(MG5DIR, 'aloha', 'template_files',
                      'aloha_functions.f')).read()
    start = text.index('      subroutine boostx(')
    return text[start:text.index('\n      end\n', start) + len('\n      end\n')]


def collision(rng):
    """Two massless beams along z with unequal energies, a massive leg 3 with
    transverse momentum, and two more legs balancing the event."""
    e1, e2 = rng.uniform(100., 3000.), rng.uniform(100., 3000.)
    p = [(e1, 0., 0., e1), (e2, 0., 0., -e2)]
    px, py, pz = (rng.uniform(-300., 300.) for _ in range(3))
    p.append(((80.4**2 + px**2 + py**2 + pz**2) ** 0.5, px, py, pz))
    kx, ky, kz = (rng.uniform(-200., 200.) for _ in range(3))
    p.append(((kx**2 + ky**2 + kz**2) ** 0.5, kx, ky, kz))
    p.append(tuple(p[0][j] + p[1][j] - p[2][j] - p[3][j] for j in range(4)))
    return p


def decay(rng):
    """A decaying particle at rest and four decay products (only the
    kinematics of the boost is tested, so they need not conserve momentum)."""
    p = [(500., 0., 0., 0.)]
    for mass in (80.4, 0., 4.7, 0.):
        px, py, pz = (rng.uniform(-150., 150.) for _ in range(3))
        p.append(((mass**2 + px**2 + py**2 + pz**2) ** 0.5, px, py, pz))
    return p


@unittest.skipIf(misc.which('gfortran') is None, 'gfortran not available')
class TestMeFrameBoosts(unittest.TestCase):
    """The me_frame rest frame of a collision is reached by a boost along the
    beam, to zero rapidity of the selected legs, and then a transverse one, so
    that it does not depend on which frame along the beam the momenta are
    given in: the partonic c.m. in madevent, the lab in madspace and MadSpin,
    and in aMC@NLO the c.m. of the underlying Born of the FKS sector, which
    for an initial-state emission is a boost along z away from the real's own
    c.m.. The single boost along the momentum of the selected legs depended on
    it, by a rotation that turns the spin axis of a single leg held at rest."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        with open(pjoin(cls.tmp, 'driver.f'), 'w') as f:
            f.write(DRIVER % {'npart': NPART})
        with open(pjoin(cls.tmp, 'boostx.f'), 'w') as f:
            f.write(boostx_source())
        subprocess.check_call(
            ['gfortran', '-O0', '-ffixed-line-length-132', '-o',
             pjoin(cls.tmp, 'test'), pjoin(cls.tmp, 'driver.f'),
             pjoin(cls.tmp, 'boostx.f'),
             pjoin(MG5DIR, 'Template', 'Common', 'Source',
                   'impose_frame_rest.f')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_events(self, events):
        """events: (nincoming, legs, rapidity, momenta); legs counted from 1"""
        lines = []
        for nincoming, legs, eta, momenta in events:
            mask = ' '.join('1' if i + 1 in legs else '0' for i in range(NPART))
            lines.append('%d %s %r' % (nincoming, mask, eta))
            lines.extend('%r %r %r %r' % mom for mom in momenta)
        out = subprocess.run([pjoin(self.tmp, 'test')],
                             input='\n'.join(lines) + '\n',
                             stdout=subprocess.PIPE, universal_newlines=True,
                             check=True).stdout
        results = []
        for block in out.split('END')[:-1]:
            res = {}
            for line in block.strip().splitlines():
                key, vals = line.split(None, 1)
                vals = [float(v) for v in vals.split()]
                res[key] = vals[0] if len(vals) == 1 else vals
            results.append(res)
        self.assertEqual(len(results), len(events))
        return results

    def collisions(self, legs, n=40):
        rng = random.Random(1234)
        return [(2, legs, eta, collision(rng))
                for _ in range(n) for eta in (-1.3, 0.4, 2.1)]

    def test_single_leg_frame_does_not_depend_on_the_input_frame(self):
        results = self.run_events(self.collisions([3]))
        for res in results:
            self.assertLess(res['DIFF'], 1e-12)
            # exactly at rest: HELAS takes the frame z axis as spin axis only
            # at an exact zero
            self.assertEqual(res['REST'], [0., 0., 0.])
        # the single boost does depend on it, which is what this replaces
        self.assertGreater(max(res['OLDDIFF'] for res in results), 1e-2)

    def test_single_leg_frame_is_collins_soper(self):
        """The z axis bisects the directions of beam 1 and of minus beam 2."""
        for res in self.run_events(self.collisions([3])):
            x, y, z = res['BISECT']
            self.assertLess(abs(x), 1e-12 * z)
            self.assertLess(abs(y), 1e-12 * z)
            self.assertGreater(z, 0.)

    def test_multi_leg_frames(self):
        """Any selection of a collision: the frame is invariant too, and for
        two or more legs it differs from the single boost by a rotation only,
        which leaves |M|^2 alone (checked here through the energies)."""
        for legs in ([3, 4], [3, 4, 5], [1, 2]):
            for res in self.run_events(self.collisions(legs, 10)):
                self.assertLess(res['DIFF'], 1e-12, legs)
        # the partonic c.m. (whole final state, or the initial state) has no
        # transverse momentum: it is the plain boost along z, as before
        for legs in ([3, 4, 5], [1, 2]):
            for res in self.run_events(self.collisions(legs, 10)):
                self.assertLess(res['TOOLD'], 1e-12, legs)

    def test_decay_keeps_the_single_boost(self):
        """A decay has no beam axis: the frame is the single boost from the
        rest frame of the decaying particle, bit for bit as it always was."""
        rng = random.Random(4321)
        events = [(1, legs, 0., decay(rng))
                  for legs in ([2], [2, 3], [3]) for _ in range(10)]
        for res in self.run_events(events):
            self.assertEqual(res['TOOLD'], 0.)
