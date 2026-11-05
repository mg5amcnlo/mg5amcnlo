      subroutine impose_frame_rest(ids, npart, p)
c**************************************************************************
c     Impose the defining property of a single-leg me_frame: if the frame
c     is built from exactly one selected leg, that leg is analytically at
c     rest after the boost, so force its three-momentum to exactly zero.
c
c     Why this is not cosmetic. boostx only reaches rest up to the rounding
c     of the boost factor, leaving a residual of a few 1d-14 whose
c     direction is pure noise. vxxxxx (aloha_functions.f) branches on
c     pp.eq.rZero and, for a massive vector at exactly zero three-momentum,
c     takes the frame z axis as quantisation axis -- the longitudinal
c     polarisation vector (0,0,0,1). Off that branch it builds the
c     polarisation vectors from the momentum direction, i.e. from the
c     noise, giving an O(1) wrong polarisation state on precisely the
c     events that fail to round to zero. Measured at 1.8% (5 sigma) on the
c     shipped LO polarised cross-section when this was first found.
c
c     Only nsel==1 needs it: with two or more selected legs it is their
c     *sum* that is at rest, no individual leg sits on the branch point,
c     and the residual of the sum never reaches HELAS.
c
c     The energy is deliberately left untouched. Zeroing the three-momentum
c     shifts the invariant mass by O(1d-28) relative, and HELAS takes the
c     mass as a separate argument anyway.
c
c     This lives in Template/Common/Source because it had to be got right
c     in four places -- Template/LO/SubProcesses/genps.f,
c     Template/NLO/SubProcesses/boost_to_frame.f and both copies in
c     MadSpin/src/driver.f -- and was for a long time got right in only
c     one. Keep it here rather than inlining it again.
c
c     input:  ids(npart)   0/1 mask of the legs the frame is built from
c             npart        number of legs in ids and p
c     in/out: p(0:3,npart) momenta, already boosted into the frame
c**************************************************************************
      implicit none
      integer npart
      integer ids(npart)
      double precision p(0:3,npart)
      integer i, nsel, isel

      nsel = 0
      isel = 0
      do i = 1, npart
         if (ids(i).eq.1) then
            nsel = nsel + 1
            isel = i
         endif
      enddo

      if (nsel.eq.1) then
         p(1,isel) = 0d0
         p(2,isel) = 0d0
         p(3,isel) = 0d0
      endif

      return
      end


      subroutine get_me_frame_boosts(ids, npart, nincoming, p, qz, qt)
c**************************************************************************
c     The me_frame rest frame of the legs selected by ids, as two boosts
c     to apply in this order, each with boostx:
c         call boostx(p, qz, ptmp)     along the beam axis, to the frame
c                                      where the selected system has zero
c                                      rapidity
c         call boostx(ptmp, qt, pout)  transverse, to its rest frame
c
c     Why not the single boost along the momentum of the selected system.
c     That boost depends on the frame it starts from: from two frames
c     related by a boost along z it reaches two rest frames that differ by
c     a rotation (Wigner) about an axis transverse to the beam. |M|^2 does
c     not care, since every momentum turns with it, EXCEPT when a single
c     leg is selected: that leg is at rest, HELAS quantises its spin along
c     the frame z axis (see impose_frame_rest above), and that axis turns
c     with the starting frame. The starting frame is not unique:
c       - madevent hands the matrix element partonic-c.m. momenta, madspace
c         and MadSpin lab momenta;
c       - FKS hands the real emission its momenta in the c.m. frame of the
c         underlying Born. For an initial-state emission that is a boost
c         along z away from the real's own c.m., and a different one for
c         each FKS sector. The same real event then had a different spin
c         axis depending on the sector that generated it: in
c         p p > w+{0} j [QCD] the axis moved by 5 degrees on average (up
c         to 70), and the polarised NLO cross-section moved by 2.3% when
c         only the FKS partition (fks_b) was changed.
c     The real's own c.m. is no way out: at NLO it is not infrared safe,
c     since in the initial-state collinear limit it stays a finite boost
c     away from the c.m. of the Born the counterterm is evaluated in.
c
c     The composite depends only on the momentum of the selected system
c     and is invariant under any boost along z of the input, so all of the
c     above land in the same frame. For a single leg its z axis bisects the
c     directions of beam 1 and of minus beam 2: it is the Collins-Soper
c     frame, oriented along beam 1. A system without transverse momentum
c     (e.g. the whole final state) is only boosted along z, and for a
c     selection of two or more legs the composite differs from the single
c     boost by a rotation only, so |M|^2 is unchanged there.
c
c     A decay (nincoming=1) has no beam axis and no FKS initial-state
c     sector: its momenta are always given in the rest frame of the decaying
c     particle, and qz is the identity there, i.e. the frame is the single
c     boost from that rest frame, as it always was.
c
c     No include and no call, so that the LO and NLO libgeneric and the
c     MadSpin driver can all link it.
c
c     input:  ids(npart)   0/1 mask of the legs the frame is built from
c             npart        number of legs in ids and p
c             nincoming    number of incoming legs among them
c             p(0:3,npart) momenta, in any frame reached from the lab by a
c                          boost along the beam axis
c     output: qz(0:3)      longitudinal step, as a boostx argument
c             qt(0:3)      transverse step, as a boostx argument
c**************************************************************************
      implicit none
      integer npart, nincoming
      integer ids(npart)
      double precision p(0:3,npart), qz(0:3), qt(0:3)
      double precision ps(0:3)
      integer i, j

      do j = 0, 3
         ps(j) = 0d0
      enddo
      do i = 1, npart
         if (ids(i).eq.1) then
            do j = 0, 3
               ps(j) = ps(j) + p(j,i)
            enddo
         endif
      enddo

c     boostx(p, (E,-P)) expresses p in the rest frame of (E,P); a zero
c     spatial part is the identity, bit for bit
      if (nincoming.ne.2) then
         qz(0) = 1d0
         qz(1) = 0d0
         qz(2) = 0d0
         qz(3) = 0d0
         qt(0) = ps(0)
         qt(1) = -ps(1)
         qt(2) = -ps(2)
         qt(3) = -ps(3)
         return
      endif
      qz(0) = ps(0)
      qz(1) = 0d0
      qz(2) = 0d0
      qz(3) = -ps(3)
c     after it the system is (mT, px, py, 0), written out rather than
c     boosted so that its longitudinal momentum is exactly zero
      qt(0) = sqrt((ps(0)-ps(3))*(ps(0)+ps(3)))
      qt(1) = -ps(1)
      qt(2) = -ps(2)
      qt(3) = 0d0

      return
      end
