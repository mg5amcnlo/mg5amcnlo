c
c This file contains the default cuts (as defined in the run_card.dat)
c and can easily be extended by the user to include other.  This
c function should return true if event passes cuts
c (passcuts_user=.true.) and false otherwise (passcuts_user=.false.).
c
c NOTE THAT ONLY IRC-SAFE CUTS CAN BE APPLIED OTHERWISE THE INTEGRATION
c MIGHT NOT CONVERGE
c
      logical function passcuts_user(p,istatus,ipdg)
      implicit none
c This includes the 'nexternal' parameter that labels the number of
c particles in the (n+1)-body process
      include 'nexternal.inc'
c This include file contains common blocks filled with the cuts defined
c in the run_card.dat
      include 'cuts.inc'
c
c This is an array which is '-1' for initial state and '1' for final
c state particles
      integer istatus(nexternal)
c This is an array with (simplified) PDG codes for the particles. Note
c that channels that are combined (i.e. they have the same matrix
c elements) are given only 1 set of PDG codes. This means, e.g., that
c when using a 5-flavour scheme calculation (massless b quark), no
c b-tagging can be applied.
      integer iPDG(nexternal)
c The array of the momenta and masses of the initial and final state
c particles in the lab frame. The format is "E, px, py, pz, mass", while
c the second dimension loops over the particles in the process. Note
c that these are the (n+1)-body particles; for the n-body there is one
c momenta equal to all zero's (this is not necessarily the last particle
c in the list). If one uses IR-safe obserables only, there should be no
c difficulty in using this.
      double precision p(0:4,nexternal)
c
C     external functions that can be used. Some are defined in this
C     file, others are in ./Source/kin_functions.f
      REAL*8 R2_04,invm2_04,pt_04,eta_04,pt,eta
      external R2_04,invm2_04,pt_04,eta_04,pt,eta
C     recombination of photons
      double precision p_reco(0:4,nexternal), R_reco
      integer iPDG_reco(nexternal)
c local integers
      integer i,j
c temporary variable for caching locally computation
      double precision tmpvar
c jet cluster algorithm
      integer nQCD,NJET,JET(nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg,amcatnlo_fastjetdmerge
      integer njet_eta
      integer mm
c Photon isolation
      integer nph,nem,k,nin
      double precision ptg,chi_gamma_iso,iso_getdrv40
      double precision Etsum(0:nexternal)
      real drlist(nexternal)
      double precision pgamma(0:3,nexternal),pem(0:3,nexternal)
      logical alliso
c Sort array of results: ismode>0 for real, isway=0 for ascending order
      integer ismode,isway,izero,isorted(nexternal)
      parameter (ismode=1)
      parameter (isway=0)
      parameter (izero=0)
c The UNLOPS cut
      double precision p_unlops(0:3,nexternal)
      include "run.inc" ! includes the ickkw parameter
      logical passUNLOPScuts
c PDG specific cut
      double precision etmin(nincoming+1:nexternal-1)
      double precision etmax(nincoming+1:nexternal-1)
      double precision mxxmin(nincoming+1:nexternal-1,nincoming+1:nexternal-1)
      common /to_cuts/etmin,etmax,mxxmin
c logicals that define if particles are leptons, jets or photons. These
c are filled from the PDG codes (iPDG array) in this function.
      logical is_a_lp(nexternal),is_a_lm(nexternal),is_a_j(nexternal)
     $     ,is_a_ph(nexternal)
      integer ngamma
      double precision pz(0:3)
      double precision delta_phi
      double precision dot
      double precision ptgamma_hardest

      passcuts_user=.true. ! event is okay; otherwise it is changed

C***************************************************************
C***************************************************************
C Cuts from the run_card.dat
C***************************************************************
C***************************************************************
      !first recombine the photons and fermions
      call recombine_momenta(rphreco, etaphreco, lepphreco, quarkphreco,
     $                       p, iPDG, p_reco, iPDG_reco)

      ! reconstruct the Z
      pz(0:3) = 0d0
      do i = nincoming+1, nexternal
        if (abs(iPDG(i)).eq.12) then
            pz(0:3) = pz(0:3) + p(0:3,i)
        endif
      enddo

      ! find the photons
      ngamma = 0
      ptgamma_hardest = 0d0
      passcuts_user = .false.
      do i = nincoming+1, nexternal
        if (iPDG(i).eq.22) then
          if (pt(p(0,i)).gt.ptgamma_hardest.and.pt(p(0,i)).gt.ptgmin) then
              passcuts_user = abs(eta(p(0,i))).lt.etagamma.and.
     $       delta_phi(p(0,i), pz).gt.2.6d0
            ptgamma_hardest = pt(p(0,i))
          endif
        endif
      enddo

      passcuts_user = passcuts_user.and.pt(pz).gt.90d0
         
      return
      end





      subroutine recombine_momenta(R, etaph, reco_l, reco_q, p_in, pdg_in, p_out, pdg_out)
      implicit none
      ! recombine photons with the closest fermion if the distance is
      ! less than R and if the rapidity of photons is < etaph (etaph < 0
      ! means no cut). Output a new set of momenta and pdgs corresponding
      ! to the recombined particles. If recombination occurs the photon
      ! disappears from the output particles
      ! arguments
      include 'nexternal.inc'
      double precision R, etaph, p_in(0:4,nexternal), p_out(0:4,nexternal)
      logical reco_l, reco_q
      integer pdg_in(nexternal), pdg_out(nexternal)
      ! local variables
      integer nq, nl
      integer id_ph
      parameter (id_ph=22)
      integer n_ph, i_ph
      integer i,j
      integer ifreco
      double precision dreco, dthis
      integer skip
      logical is_light_charged_fermion
      double precision R2_04, eta_04
      ! 
      integer times_reco
      common/to_times_reco/ times_reco
      ! reset everything
      do j=1,nexternal
        pdg_out(j)=0
        do i=0,4
          p_out(i,j)=0d0
        enddo
      enddo

      ! check if we want to recombine with leptons
      if (reco_l) then
          nl = 3
      else 
          nl = 0
      endif

      ! check if we want to recombine with quarks
      if (reco_q) then
          nq = 5
      else 
          nq = 0
      endif

      ! count the photons
      n_ph=0
      do i=nincoming+1, nexternal
        if (pdg_in(i).eq.id_ph.and.
     $   (abs(eta_04(p_in(0,i))).lt.etaph.or.etaph.lt.0d0)) then
            n_ph=n_ph+1
            i_ph=i
        endif
      enddo
      if (n_ph.eq.0 .or. (nl.eq.0 .and. nq.eq.0)) then
        ! do nothing
        do j=1,nexternal
          pdg_out(j)=pdg_in(j)
          do i=0,4
            p_out(i,j)=p_in(i,j)
          enddo
        enddo
        return
      elseif (n_ph.eq.1) then
        ! do nothing for initial states
        do j=1,nincoming
          pdg_out(j)=pdg_in(j)
          do i=0,4
            p_out(i,j)=p_in(i,j)
          enddo
        enddo
        ! find the closest fermion to the photon
        ifreco=0
        dreco=R
        if (i_ph.gt.0) then
          do i = nincoming+1, nexternal
            if (is_light_charged_fermion(pdg_in(i),nq,nl)) then
              dthis=dsqrt(R2_04(p_in(0,i_ph),p_in(0,i)))
              if (dthis.le.dreco) then
                dreco=dthis
                ifreco=i
              endif
            endif
          enddo
        endif
        if (ifreco.eq.0) then
        ! do nothing also for final states
          do j=nincoming+1,nexternal
            pdg_out(j)=pdg_in(j)
            do i=0,4
              p_out(i,j)=p_in(i,j)
            enddo
          enddo
        else
          times_reco=times_reco+1
          skip=0
          do j=nincoming+1,nexternal
            if (j.ne.i_ph.and.j.ne.ifreco) then
              pdg_out(j-skip)=pdg_in(j)
              do i=0,4
                p_out(i,j-skip)=p_in(i,j)
              enddo
            elseif (j.eq.ifreco) then
              pdg_out(j-skip)=pdg_in(j)
              do i=0,3
                p_out(i,j-skip)=p_in(i,j)+p_in(i,i_ph)
              enddo
              p_out(4,j-skip)=p_in(4,j)
            elseif (j.eq.i_ph) then
              skip=skip+1
            endif
          enddo
        endif
      else
        write(*,*) 'ERROR, too many photons', n_ph
        stop 1
      endif

      return 
      end


      logical function is_light_charged_fermion(id, nf, nl)
      implicit none
      integer id, nf, nl
      if (abs(id).le.nf) then
          is_light_charged_fermion = .true.
      elseif ((abs(id).eq.11.and.nl.ge.1).or.
     $        (abs(id).eq.13.and.nl.ge.2).or.
     $        (abs(id).eq.15.and.nl.ge.3)) then
          is_light_charged_fermion = .true.
      else
          is_light_charged_fermion = .false.
      endif
      return
      end




C***************************************************************
C***************************************************************
C NO NEED TO CHANGE ANY OF THE FUNCTIONS BELOW
C***************************************************************
C***************************************************************
      logical function passcuts(p,rwgt)
      implicit none
      include "nexternal.inc"
      include 'run.inc'
      include 'genps.inc'
      include 'cuts.inc'
      include 'timing_variables.inc'
      REAL*8 P(0:3,nexternal),rwgt
      integer i,j,istatus(nexternal),iPDG(nexternal)
c For boosts
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision chybst,shybst,chybstmo
      double precision xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
c Momenta of the particles
      double precision plab(0:3, nexternal),pp(0:4, nexternal)
c Masses of external particles
      double precision pmass(nexternal)
      common/to_mass/pmass
c PDG codes of particles
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      logical passcuts_user
      external passcuts_user
      call cpu_time(tBefore)
c Make sure have reasonable 4-momenta
      if (p(0,1) .le. 0d0) then
         passcuts=.false.
         return
      endif
c Also make sure there's no INF or NAN
      do i=1,nexternal
         do j=0,3
            if(p(j,i).gt.1d32.or.p(j,i).ne.p(j,i))then
               passcuts=.false.
               return
            endif
         enddo
      enddo
      rwgt=1d0
c Boost the momenta p(0:3,nexternal) to the lab frame plab(0:3,nexternal)
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=1,nexternal
         call boostwdir2(chybst,shybst,chybstmo,xd,
     &        p(0,i),plab(0,i))
      enddo
c Fill the arrays (momenta, status and PDG):
      do i=1,nexternal
         if (i.le.nincoming) then
            istatus(i)=-1
         else
            istatus(i)=1
         endif
         do j=0,3
            pp(j,i)=plab(j,i)
         enddo
         pp(4,i)=pmass(i)
         ipdg(i)=idup(i,1)
         if (ipdg(i).eq.-21) ipdg(i)=21
      enddo
c Call the actual cuts function  
      passcuts = passcuts_user(pp,istatus,ipdg)
      call cpu_time(tAfter)
      t_cuts=t_cuts+(tAfter-tBefore)
      RETURN
      END


      function chi_gamma_iso(dr,R0,xn,epsgamma,pTgamma)
c Eq.(3.4) of Phys.Lett. B429 (1998) 369-374 [hep-ph/9801442]
      implicit none
      real*8 chi_gamma_iso,dr,R0,xn,epsgamma,pTgamma
      real*8 tmp,axn
c
      axn=abs(xn)
      tmp=epsgamma*pTgamma
      if(axn.ne.0.d0)then
        tmp=tmp*( (1-cos(dr))/(1-cos(R0)) )**axn
      endif
      chi_gamma_iso=tmp
      return
      end


*
* $Id: sortzv.F,v 1.1.1.1 1996/02/15 17:49:50 mclareni Exp $
*
* $Log: sortzv.F,v $
* Revision 1.1.1.1  1996/02/15 17:49:50  mclareni
* Kernlib
*
*
c$$$#include "kerngen/pilot.h"
      SUBROUTINE SORTZV (A,INDEX,N1,MODE,NWAY,NSORT)
C
C CERN PROGLIB# M101    SORTZV          .VERSION KERNFOR  3.15  820113
C ORIG. 02/10/75
C
      DIMENSION A(N1),INDEX(N1)
C
C
      N = N1
      IF (N.LE.0)            RETURN
      IF (NSORT.NE.0) GO TO 2
      DO 1 I=1,N
    1 INDEX(I)=I
C
    2 IF (N.EQ.1)            RETURN
      IF (MODE)    10,20,30
   10 CALL SORTTI (A,INDEX,N)
      GO TO 40
C
   20 CALL SORTTC(A,INDEX,N)
      GO TO 40
C
   30 CALL SORTTF (A,INDEX,N)
C
   40 IF (NWAY.EQ.0) GO TO 50
      N2 = N/2
      DO 41 I=1,N2
      ISWAP = INDEX(I)
      K = N+1-I
      INDEX(I) = INDEX(K)
   41 INDEX(K) = ISWAP
   50 RETURN
      END
*     ========================================
      SUBROUTINE SORTTF (A,INDEX,N1)
C
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTI (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTC (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF(ICMPCH(AI,A(I22)))3,3,21
   21 INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (ICMPCH(A(I22),A(I222))) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (ICMPCH(AI,A(I22))) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      FUNCTION ICMPCH(IC1,IC2)
C     FUNCTION TO COMPARE TWO 4 CHARACTER EBCDIC STRINGS - IC1,IC2
C     ICMPCH=-1 IF HEX VALUE OF IC1 IS LESS THAN IC2
C     ICMPCH=0  IF HEX VALUES OF IC1 AND IC2 ARE THE SAME
C     ICMPCH=+1 IF HEX VALUES OF IC1 IS GREATER THAN IC2
      I1=IC1
      I2=IC2
      IF(I1.GE.0.AND.I2.GE.0)GOTO 40
      IF(I1.GE.0)GOTO 60
      IF(I2.GE.0)GOTO 80
      I1=-I1
      I2=-I2
      IF(I1-I2)80,70,60
 40   IF(I1-I2)60,70,80
 60   ICMPCH=-1
      RETURN
 70   ICMPCH=0
      RETURN
 80   ICMPCH=1
      RETURN
      END


      function iso_getdrv40(p1,p2)
      implicit none
      real*8 iso_getdrv40,p1(0:3),p2(0:3)
      real*8 iso_getdr
c
      iso_getdrv40=iso_getdr(p1(0),p1(1),p1(2),p1(3),
     #                       p2(0),p2(1),p2(2),p2(3))
      return
      end


      function iso_getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 iso_getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # iso_getpseudorap,iso_getdelphi
c
      deta=iso_getpseudorap(en1,ptx1,pty1,pl1)-
     #     iso_getpseudorap(en2,ptx2,pty2,pl2)
      dphi=iso_getdelphi(ptx1,pty1,ptx2,pty2)
      iso_getdr=sqrt(dphi**2+deta**2)
      return
      end


      function iso_getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 iso_getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      iso_getpseudorap=eta
      return
      end


      function iso_getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 iso_getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      iso_getdelphi=tmp
      return
      end



      DOUBLE PRECISION FUNCTION R2_04(P1,P2)
c************************************************************************
c     Distance in eta,phi between two particles.
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:4),p2(0:4),p1a(0:3),p2a(0:3)
      integer i
c
c     External
c
      double precision eta,DELTA_PHI
      external eta,delta_phi
c-----
c  Begin Code
c-----
      do i=0,3
         p1a(i)=p1(i)
         p2a(i)=p2(i)
      enddo
      R2_04 = (DELTA_PHI(P1a,P2a))**2+(eta(p1a)-eta(p2a))**2
      RETURN
      END

      double precision function pt_04(p)
c************************************************************************
c     Returns transverse momentum of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:4)
c-----
c  Begin Code
c-----

      pt_04 = dsqrt(p(1)**2+p(2)**2)

      return
      end


      double precision function eta_04(p)
c************************************************************************
c     Returns pseudo rapidity of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:4),pa(0:3)
      integer i
c
c     external
c
      double precision theta,tp,pi
      parameter (pi=3.14159265358979323846264338327950d0)
      external theta
c-----
c  Begin Code
c-----
      do i=0,3
         pa(i)=p(i)
      enddo
      tp=theta(pa)
      if (abs(tp).lt.1d-5) then
         eta_04=25d0
      elseif (abs(tp-pi).lt.1d-5) then
         eta_04=-25d0
      else
         eta_04=-dlog(dtan(theta(pa)/2d0))
      endif

      return
      end



      DOUBLE PRECISION FUNCTION invm2_04(P1,P2,dsign)
c************************************************************************
c     Invarient mass of 2 particles
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:4),p2(0:4),dsign
c
c     Local
c      
      integer i
      double precision ptot(0:3)
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----

      do i=0,3
         ptot(i)=p1(i)+dsign*p2(i)
      enddo
      invm2_04 = dot(ptot,ptot)
      RETURN
      END


      subroutine get_ID_H(IDUP_tmp)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
c      include 'leshouche.inc'
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer IDUP_tmp(nexternal),i
c
      do i=1,nexternal
         IDUP_tmp(i)=IDUP(i,1)
      enddo
c
      return
      end

      subroutine get_ID_S(IDUP_tmp)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include 'born_leshouche.inc'
      integer IDUP_tmp(nexternal),i
c
      do i=1,nexternal-1
         IDUP_tmp(i)=IDUP(i,1)
      enddo
      IDUP_tmp(nexternal)=0
c
      return
      end


      subroutine bias_weight_function(p,ipdg,bias_wgt)
c This is a user-defined function to which to bias the event generation.
c A non-flat distribution will generate events with a certain weight
c inversely proportinal to the bias_wgt. This is particularly useful to
c generate more events (with smaller weight) in tails of distributions.
c It computes the bias_wgt factor from the momenta and multiplies the
c weight that goes into MINT (or vegas) with this factor.  Before
c writing out the events (or making the plots), this factor is again
c divided out. A value different from 1 makes that MINT (or vegas) does
c not list the correct cross section, but the cross section can still be
c computed from summing all the weights of the events (and dividing by
c the number of events). Since the weights of the events are no longer
c identical for all events, the statistical uncertainty on this total
c cross section can be much larger than without including the bias.
c
c The 'bias_wgt' should be a IR-safe function of the momenta.
c      
c For this to be used, the 'event_norm' option in the run_card should be
c set to
c      'bias' = event_norm      
c
      implicit none
      include 'nexternal.inc'
      double precision bias_wgt,p(0:3,nexternal),H_T
      integer ipdg(nexternal),i

      bias_wgt=1d0

c How to enhance the tails is very process dependent. For example for
c top quark production one could use:
c      do i=1,nexternal
c         if (ipdg(i).eq.6) then
c            bias_wgt=sqrt(p(1,i)**2+p(2,i)**2)**3
c         endif
c      enddo
c Or to use H_T^2 one does     
c      H_T=0d0
c      do i=3,nexternal
c         H_T=H_T+sqrt(max(0d0,(p(0,i)+p(3,i))*(p(0,i)-p(3,i))))
c      enddo
c      bias_wgt=H_T**2
      return
      end


