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
c local integers
      integer i,j
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
c logicals that define if particles are leptons, jets or photons. These
c are filled from the PDG codes (iPDG array) in this function.
      logical is_a_lp(nexternal),is_a_lm(nexternal),is_a_j(nexternal)
     $     ,is_a_ph(nexternal)

      passcuts_user=.true. ! event is okay; otherwise it is changed

C***************************************************************
C***************************************************************
C Cuts from the run_card.dat
C***************************************************************
C***************************************************************
c
c CHARGED LEPTON CUTS
c
c find the charged leptons (also used in the photon isolation cuts below)
      do i=1,nexternal
         if(istatus(i).eq.1 .and.
     &    (ipdg(i).eq.11 .or. ipdg(i).eq.13 .or. ipdg(i).eq.15)) then
            is_a_lm(i)=.true.
         else
            is_a_lm(i)=.false.
         endif
         if(istatus(i).eq.1 .and.
     &    (ipdg(i).eq.-11 .or. ipdg(i).eq.-13 .or. ipdg(i).eq.-15)) then
            is_a_lp(i)=.true.
         else
            is_a_lp(i)=.false.
         endif
      enddo
c apply the charged lepton cuts
      do i=nincoming+1,nexternal
         if (is_a_lp(i).or.is_a_lm(i)) then
c transverse momentum
            if (ptl.gt.0d0) then
               if (pt_04(p(0,i)).lt.ptl) then
                  passcuts_user=.false.
                  return
               endif
            endif
c pseudo-rapidity
            if (etal.gt.0d0) then
               if (abs(eta_04(p(0,i))).gt.etal) then
                  passcuts_user=.false.
                  return
               endif
            endif
c DeltaR and invariant mass cuts
            if (is_a_lp(i)) then
               do j=nincoming+1,nexternal
                  if (is_a_lm(j)) then
                     if (drll.gt.0d0) then
                        if (R2_04(p(0,i),p(0,j)).lt.drll**2) then
                           passcuts_user=.false.
                           return
                        endif
                     endif
                     if (mll.gt.0d0) then
                        if (invm2_04(p(0,i),p(0,j),1d0).lt.mll**2) then
                           passcuts_user=.false.
                           return
                        endif
                     endif
                     if (ipdg(i).eq.-ipdg(j)) then
                        if (drll_sf.gt.0d0) then
                           if (R2_04(p(0,i),p(0,j)).lt.drll_sf**2) then
                              passcuts_user=.false.
                              return
                           endif
                        endif
                        if (mll_sf.gt.0d0) then
                           if (invm2_04(p(0,i),p(0,j),1d0).lt.mll_sf**2)
     $                          then
                              passcuts_user=.false.
                              return
                           endif
                        endif
                     endif
                  endif
               enddo
            endif
         endif
      enddo
c
c JET CUTS
c
c find the jets
      do i=1,nexternal
         if (istatus(i).eq.1 .and.
     &        (abs(ipdg(i)).le.maxjetflavor .or. ipdg(i).eq.21)) then
            is_a_j(i)=.true.
         else
            is_a_j(i)=.false.
         endif
      enddo

c If we do not require a mimimum jet energy, there's no need to apply
c jet clustering and all that.
      if (ptj.ne.0d0.or.ptgmin.ne.0d0) then
c Put all (light) QCD partons in momentum array for jet clustering.
c From the run_card.dat, maxjetflavor defines if b quark should be
c considered here (via the logical variable 'is_a_jet').  nQCD becomes
c the number of (light) QCD partons at the real-emission level (i.e. one
c more than the Born).
         nQCD=0
         do j=nincoming+1,nexternal
            if (is_a_j(j)) then
               nQCD=nQCD+1
               do i=0,3
                  pQCD(i,nQCD)=p(i,j)
               enddo
            endif
         enddo
      endif

c THE UNLOPS CUT:
      if (ickkw.eq.4 .and. ptj.gt.0d0) then
c Use special pythia pt cut for minimal pT
         do i=1,nexternal
            do j=0,3
               p_unlops(j,i)=p(j,i)
            enddo
         enddo
         call pythia_UNLOPS(p_unlops,passUNLOPScuts)
         if (.not. passUNLOPScuts) then
            passcuts_user=.false.
            return
         endif
c Bypass normal jet cuts
         goto 122
c THE VETO XSEC CUT:
      elseif (ickkw.eq.-1 .and. ptj.gt.0d0) then
c Use veto'ed Xsec for analytic NNLL resummation
         if (nQCD.ne.1) then
            write (*,*) 'ERROR: more than one QCD parton in '/
     $           /'this event in cuts.f. There should only be one'
            stop
         endif
         if (pt(pQCD(0,1)) .gt. ptj) then
            passcuts_user=.false.
            return
         endif
      endif


      if (ptj.gt.0d0.and.nQCD.gt.1) then

c Cut some peculiar momentum configurations, i.e. two partons very soft.
c This is needed to get rid of numerical instabilities in the Real emission
c matrix elements when the Born has a massless final-state parton, but
c no possible divergence related to it (e.g. t-channel single top)
         mm=0
         do j=1,nQCD
            if(abs(pQCD(0,j)/p(0,1)).lt.1.d-8) mm=mm+1
         enddo
         if(mm.gt.1)then
            passcuts_user=.false.
            return
         endif


c Define jet clustering parameters (from cuts.inc via the run_card.dat)
         palg=JETALGO           ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
         rfj=JETRADIUS          ! the radius parameter
         sycut=PTJ              ! minimum transverse momentum

c******************************************************************************
c     call FASTJET to get all the jets
c
c     INPUT:
c     input momenta:               pQCD(0:3,nexternal), energy is 0th component
c     number of input momenta:     nQCD
c     radius parameter:            rfj
c     minumum jet pt:              sycut
c     jet algorithm:               palg, 1.0=kt, 0.0=C/A, -1.0 = anti-kt
c
c     OUTPUT:
c     jet momenta:                           pjet(0:3,nexternal), E is 0th cmpnt
c     the number of jets (with pt > SYCUT):  njet
c     the jet for a given particle 'i':      jet(i),   note that this is the
c                                            particle in pQCD, which doesn't
c                                            necessarily correspond to the particle
c                                            label in the process
c
         call amcatnlo_fastjetppgenkt_etamax_timed(
     $    pQCD,nQCD,rfj,sycut,etaj,palg,pjet,njet,jet)
c
c******************************************************************************

c Apply the jet cuts
         if (njet .ne. nQCD .and. njet .ne. nQCD-1) then
            passcuts_user=.false.
            return
         endif
      endif
 122  continue
c
c PHOTON (ISOLATION) CUTS
c
c find the photons
      do i=1,nexternal
         if (istatus(i).eq.1 .and. ipdg(i).eq.22) then
            is_a_ph(i)=.true.
         else
            is_a_ph(i)=.false.
         endif
      enddo
      if (ptgmin.ne.0d0) then
         nph=0
         do j=nincoming+1,nexternal
            if (is_a_ph(j)) then
               nph=nph+1
               do i=0,3
                  pgamma(i,nph)=p(i,j)
               enddo
            endif
         enddo
         if(nph.eq.0)goto 444
         
         if(isoEM)then
            nem=nph
            do k=1,nem
               do i=0,3
                  pem(i,k)=pgamma(i,k)
               enddo
            enddo
            do j=nincoming+1,nexternal
               if (is_a_lp(j).or.is_a_lm(j)) then
                  nem=nem+1
                  do i=0,3
                     pem(i,nem)=p(i,j)
                  enddo
               endif
            enddo
         endif
         
         alliso=.true.

         j=0
         do while(j.lt.nph.and.alliso)
c Loop over all photons
            j=j+1
            
            ptg=pt(pgamma(0,j))
            if(ptg.lt.ptgmin)then
               passcuts_user=.false.
               return
            endif
            if (etagamma.gt.0d0) then
               if (abs(eta(pgamma(0,j))).gt.etagamma) then
                  passcuts_user=.false.
                  return
               endif
            endif
         
c Isolate from hadronic energy
            do i=1,nQCD
               drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pQCD(0,i)))
            enddo
            call sortzv(drlist,isorted,nQCD,ismode,isway,izero)
            Etsum(0)=0.d0
            nin=0
            do i=1,nQCD
               if(dble(drlist(isorted(i))).le.R0gamma)then
                  nin=nin+1
                  Etsum(nin)=Etsum(nin-1)+pt(pQCD(0,isorted(i)))
               endif
            enddo
            do i=1,nin
               alliso=alliso .and.
     $              Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     $              R0gamma,xn,epsgamma,ptg)
            enddo
            
c Isolate from EM energy
            if(isoEM.and.nem.gt.1)then
               do i=1,nem
                  drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pem(0,i)))
               enddo
               call sortzv(drlist,isorted,nem,ismode,isway,izero)
c First of list must be the photon: check this, and drop it
               if(isorted(1).ne.j.or.drlist(isorted(1)).gt.1.e-4)then
                  write(*,*)'Error #1 in photon isolation'
                  write(*,*)j,isorted(1),drlist(isorted(1))
                  stop
               endif
               Etsum(0)=0.d0
               nin=0
               do i=2,nem
                  if(dble(drlist(isorted(i))).le.R0gamma)then
                     nin=nin+1
                     Etsum(nin)=Etsum(nin-1)+pt(pem(0,isorted(i)))
                  endif
               enddo
               do i=1,nin
                  alliso=alliso .and.
     $               Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     $               R0gamma,xn,epsgamma,ptg)
               enddo
            endif
c End of loop over photons
         enddo
         if(.not.alliso)then
            passcuts_user=.false.
            return
         endif
 444     continue
c End photon isolation
      endif

C***************************************************************
C***************************************************************
C PUT HERE YOUR USER-DEFINED CUTS
C***************************************************************
C***************************************************************
C
c$$$C EXAMPLE: cut on top quark pT
c$$$      do i=1,nexternal   ! loop over all external particles
c$$$         if (istatus(i).eq.1    ! final state particle
c$$$     &        .and. abs(ipdg(i)).eq.6) then    ! top quark
c$$$C apply the pT cut (pT should be large than 200 GeV for the event to
c$$$C pass cuts)
c$$$            if ( p(1,i)**2+p(2,i)**2 .lt. 200d0**2 ) then
c$$$C momenta do not pass cuts. Set passcuts_user to false and return
c$$$               passcuts_user=.false.
c$$$               return
c$$$            endif
c$$$         endif
c$$$      enddo
c
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
      integer maxflow
      parameter (maxflow=999)
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
      integer maxflow
      parameter (maxflow=999)
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
      integer    maxflow
      parameter (maxflow=999)
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


      subroutine unweight_function(p_born,unwgtfun)
c This is a user-defined function to which to unweight the events
c A non-flat distribution will generate events with a certain
c weight. This is particularly useful to generate more events
c (with smaller weight) in tails of distributions.
c It computes the unwgt factor from the momenta and multiplies
c the weight that goes into MINT (or vegas) with this factor.
c Before writing out the events (or making the plots), this factor
c is again divided out.
c This function should be called with the Born momenta to be sure
c that it stays the same for the events, counter-events, etc.
c A value different from 1 makes that MINT (or vegas) does not list
c the correct cross section.
      implicit none
      include 'nexternal.inc'
      double precision unwgtfun,p_born(0:3,nexternal-1),shat,sumdot
      external sumdot

      unwgtfun=1d0

c How to enhance the tails is very process dependent. But, it is
c probably easiest to enhance the tails using shat, e.g.:
c      shat=sumdot(p_born(0,1),p_born(0,2),1d0)
c      unwgtfun=max(100d0**2,shat)/100d0**2
c      unwgtfun=unwgtfun**2

      return
      end


