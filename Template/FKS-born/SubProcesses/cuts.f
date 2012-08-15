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
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM
      include 'coupl.inc'
c jet cluster algorithm
      integer nQCD,NJET,JET(nexternal)
      double precision plab(0:3, nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal)
      double precision rfj,sycut,palg,fastjetdmerge
      integer njet_eta

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
         write (*,*) 'and maximal pseudo-rapidity of ',etaj,'.'
         write (*,*) 'Charged leptons are required to have at least',ptl
     &        ,'GeV of transverse momentum and'
         write (*,*) 'pseudo rapidity of maximum',etal,'.'
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
            if (etal.lt.100d0) then
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
c Uncomment for bypassing jet algo and cuts.
c$$$      goto 123

c If we do not require a mimimum jet energy, there's no need to apply
c jet clustering and all that.
      if (ptj.ne.0d0) then

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
         call fastjetppgenkt(pQCD,nQCD,rfj,sycut,palg,pjet,njet,jet)
c
c******************************************************************************

c Apply the maximal pseudo-rapidity cuts on the jets:      
         if (etaj.lt.100d0) then 
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
               if (abs(eta(pjet(0,i))).lt.ETAJ) then
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

