      logical function pass_point(p)
      implicit none
      double precision p
      logical passcuts
      external passcuts
      pass_point = .true.
c      pass_point = passcuts(p)
      end


      LOGICAL FUNCTION PASSCUTS(P,rwgt)
C**************************************************************************
C     INPUT:
C            P(0:3,1)           MOMENTUM OF INCOMING PARTON
C            P(0:3,2)           MOMENTUM OF INCOMING PARTON
C            P(0:3,3)           MOMENTUM OF d
C            P(0:3,4)           MOMENTUM OF b
C            P(0:3,5)           MOMENTUM OF bbar
C            P(0:3,6)           MOMENTUM OF e+
C            P(0:3,7)           MOMENTUM OF ve
C            COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C            TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
C
C In MadFKS, the momenta given in input to this function are in the
C reduced parton c.m. frame. If need be, boost them to the lab frame.
C The rapidity of this boost is
C
C       YBST_TIL_TOLAB
C
C given in the common block /PARTON_CMS_STUFF/
C
C This is the rapidity that enters in the arguments of the sinh() and
C cosh() of the boost, in such a way that
C       ylab = ycm - ybst_til_tolab
C where ylab is the rapidity in the lab frame and ycm the rapidity
C in the center-of-momentum frame.
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
c
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include "nexternal.inc"
C
C     ARGUMENTS
C
      REAL*8 P(0:3,nexternal),rwgt

C
C     LOCAL
C
      LOGICAL FIRSTTIME
      DATA FIRSTTIME/.TRUE./
      integer i,j
C
C     EXTERNAL
C
      REAL*8 R2,DOT,ET,RAP,DJ,SumDot,pt,rewgt,eta
      logical cut_bw
      external cut_bw,rewgt,eta,r2,dot,et,rap,dj,sumdot,pt
C
C     GLOBAL
C
      include 'run.inc'
      include 'cuts.inc'
c For boosts
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision pjetlab(0:3,nexternal)
      double precision chybst,shybst,chybstmo
      double precision xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
c Jets and charged leptons
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      include 'coupl.inc'
c jet cluster algorithm
      integer nQCD,NJET,JET(nexternal)
      double precision plab(0:3, nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg,amcatnlo_fastjetdmerge
      integer njet_eta
c Photon isolation
      integer nph,nem,k,nin
      double precision ptg,chi_gamma_iso,iso_getdrv40
      double precision Etsum(0:nexternal)
      real drlist(nexternal)
      double precision pgamma(0:3,nexternal),pem(0:3,nexternal)
     $     ,pgammalab(0:3)
      logical alliso
c Sort array of results: ismode>0 for real, isway=0 for ascending order
      integer ismode,isway,izero,isorted(nexternal)
      parameter (ismode=1)
      parameter (isway=0)
      parameter (izero=0)



      integer mm

C-----
C  BEGIN CODE
C-----
      PASSCUTS=.TRUE.             !EVENT IS OK UNLESS OTHERWISE CHANGED
      IF (FIRSTTIME) THEN
         FIRSTTIME=.FALSE.
         write (*,*) '================================================='
         write (*,*) 'From cuts.f'
         if (jetalgo.eq.1) then
            write (*,*) 'Jets are defined with the kT algorithm'
         elseif (jetalgo.eq.0) then
            write (*,*) 'Jets are defined with the C/A algorithm'
         elseif (jetalgo.eq.-1) then
            write (*,*) 'Jets are defined with the anti-kT algorithm'
         else
            write (*,*) 'Jet algorithm not defined in the run_card.dat,'
     &           //'or not correctly processed by the code.',jetalgo
         endif
         write (*,*) 'with a mimumal pT of ',ptj,'GeV'
         if (etaj.gt.0) then
            write (*,*) 'and maximal pseudo-rapidity of ',etaj,'.'
         else
            write (*,*) 'and no maximal pseudo-rapidity.'
         endif
         write (*,*) 'Charged leptons are required to have at least',ptl
     &        ,'GeV of transverse momentum and'
         if (etal.gt.0) then
            write (*,*) 'pseudo rapidity of maximum',etal,'.'
         else
            write (*,*) 'no maximum for the pseudo rapidity.'
         endif
         write (*,*) 'Opposite charged lepton pairs need to be'//
     &        ' separated by at least ',drll
         write (*,*) 'and have an invariant mass of',mll,' GeV'
         write (*,*) '================================================='
      ENDIF
c
c     Make sure have reasonable 4-momenta
c
      if (p(0,1) .le. 0d0) then
         passcuts=.false.
         return
      endif

c     Also make sure there's no INF or NAN
      do i=1,nexternal
         do j=0,3
            if(p(j,i).gt.1d32.or.p(j,i).ne.p(j,i))then
               passcuts=.false.
               return
            endif
         enddo
      enddo

      rwgt=1d0

c Uncomment for bypassing charged lepton cuts
c$$$      goto 124

c Boost the momenta p(0:3,nexternal) to the lab frame plab(0:3,nexternal)
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=1,nexternal
         call boostwdir2(chybst,shybst,chybstmo,xd,
     &        p(0,i),plab(0,i))
      enddo

c
c CHARGED LEPTON CUTS
c
      do i=nincoming+1,nexternal
         if (is_a_lp(i).or.is_a_lm(i)) then
c transverse momentum
            if (ptl.gt.0d0) then
               if (pt(p(0,i)).lt.ptl) then
                  passcuts=.false.
                  return
               endif
            endif
c pseudo-rapidity
            if (etal.gt.0d0) then
               if (abs(eta(plab(0,i))).gt.etal) then
                  passcuts=.false.
                  return
               endif
            endif
c DeltaR and invariant mass cuts
            if (is_a_lp(i)) then
               do j=nincoming+1,nexternal
                  if (is_a_lm(j)) then
                     if (drll.gt.0d0) then
                        if (R2(plab(0,i),plab(0,j)).lt.drll**2) then
                           passcuts=.false.
                           return
                        endif
                     endif
                     if (mll.gt.0d0) then
                        if (sumdot(p(0,i),p(0,j),1d0).lt.mll**2) then
                           passcuts=.false.
                           return
                        endif
                     endif
                  endif
               enddo
            endif
         endif
      enddo

 124  continue

c
c JET CUTS
c
c Uncomment for bypassing jet algo and cuts, and photon isolation
c$$$      goto 123

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
                  pQCD(i,nQCD)=p(i,j) ! Use C.o.M. frame momenta
               enddo
            endif
         enddo

      endif


c Uncomment for bypassing jet algo and cuts
c$$$      goto 122
      if (ptj.gt.0d0.or.nQCD.gt.1) then

c Cut some peculiar momentum configurations, i.e. two partons very soft.
c This is needed to get rid of numerical instabilities in the Real emission
c matrix elements when the Born has a massless final-state parton, but
c no possible divergence related to it (e.g. t-channel single top)
         mm=0
         do j=1,nQCD
            if(abs(pQCD(0,j)/p(0,1)).lt.1.d-8) mm=mm+1
         enddo
         if(mm.gt.1)then
            passcuts=.false.
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
c     jet momenta:                             pjet(0:3,nexternal), E is 0th cmpnt
c     the number of jets (with pt > SYCUT):    njet
c     the jet for a given particle 'i':        jet(i),   note that this is
c     the particle in pQCD, which doesn't necessarily correspond to the particle
c     label in the process
c
         call amcatnlo_fastjetppgenkt(pQCD,nQCD,rfj,sycut,palg,pjet,njet,jet)
c
c******************************************************************************

c Apply the maximal pseudo-rapidity cuts on the jets:      
         if (etaj.gt.0d0) then 
c Boost the jets to the lab frame for the pseudo-rapidity cut
            chybst=cosh(ybst_til_tolab)
            shybst=sinh(ybst_til_tolab)
            chybstmo=chybst-1.d0
            do i=1,njet
               call boostwdir2(chybst,shybst,chybstmo,xd,
     &              pjet(0,i),pjetlab(0,i))
            enddo
c Count the number of jets that pass the pseud-rapidity cut
            njet_eta=0
            do i=1,njet
               if (abs(eta(pjetlab(0,i))).lt.ETAJ) then
                  njet_eta=njet_eta+1
               endif
            enddo
            njet=njet_eta
         endif

c Apply the jet cuts
         if (njet .ne. nQCD .and. njet .ne. nQCD-1) then
            passcuts=.false.
            return
         endif
      endif

 122  continue

c Begin photon isolation
c NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE 
c   Use is made of parton cm frame momenta. If this must be
c   changed, pQCD used below must be redefined
c NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE 
      if (ptgmin.ne.0d0) then
        nph=0
        do j=nincoming+1,nexternal
          if (is_a_ph(j)) then
            nph=nph+1
            do i=0,3
              pgamma(i,nph)=p(i,j) ! Use C.o.M. frame momenta
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
                pem(i,nem)=p(i,j) ! Use C.o.M. frame momenta
              enddo
            endif
          enddo
        endif

        alliso=.true.

        j=0
        dowhile(j.lt.nph.and.alliso)
c Loop over all photons
          j=j+1

          ptg=pt(pgamma(0,j))
          if(ptg.lt.ptgmin)then
            passcuts=.false.
            return
          endif

          if (etagamma.gt.0d0) then
c     for rapidity cut, boost this one gamma to the lab frame
             call boostwdir2(chybst,shybst,chybstmo,xd,
     &            pgamma(0,j),pgammalab)
             if (abs(eta(pgammalab)).gt.etagamma) then
                passcuts=.false.
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
     #        Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     #                                  R0gamma,xn,epsgamma,ptg)
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
     #          Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     #                                    R0gamma,xn,epsgamma,ptg)
            enddo

          endif

c End of loop over photons
        enddo

        if(.not.alliso)then
          passcuts=.false.
          return
        endif

 444    continue
c End photon isolation
      endif


 123  continue

      RETURN
      END


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
      double precision unwgtfun,p_born(0:3,nexternal-1)

      unwgtfun=1d0

      return
      end


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
