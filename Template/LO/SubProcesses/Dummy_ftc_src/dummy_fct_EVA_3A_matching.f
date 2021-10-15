C**************************************************************************
c     dummy_fct_EVA_3A_matching.f
c     2021 October (R. Ruiz)
c     Imposes phase space cuts on AAA final-state and on (mu-v_mu) momentum 
c     transfer (q) or transverse momentum of v_mu (pT) for matrix element 
c     matching of the processes:
c     1. e mu > v_e v_mu AAA 
c     2. e W > v_e AAA
c     pT/q cut are set to factorization scale (dsqrt_q2fact2) in run_card.dat. 
c     For further details, see Constantini et al [arXiv:2110.]
C**************************************************************************

      logical FUNCTION dummy_cuts(P)
C**************************************************************************
C     INPUT:
C     P(0:3,1)           MOMENTUM OF INCOMING PARTON
C     P(0:3,2)           MOMENTUM OF INCOMING PARTON
C     P(0:3,3)           MOMENTUM OF ...
C     ALL MOMENTA ARE IN THE REST FRAME!!
C     COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C     TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
      IMPLICIT NONE
c     
c     Constants
c     
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
      include 'maxamps.inc'
      integer i,j,k
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
C     
C     ARGUMENTS
C     
      REAL*8 P(0:3,nexternal)
C     
C     PARAMETERS
C     
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )

      integer ff,ii,kk,nrPho
      double precision evoThres2,evoScale2 ! neutrino
      double precision pAAA(0:3),minM2,minPtX,maxEta ! photon
      double precision rap,pt,r2,dot,SumDot
      external rap,pt,r2,dot,SumDot

      integer tmpPID1,tmpPID2
      integer ipdg(nexternal)
c     
c     particle identification
c     
      logical  doEVACuts,doEVAMatch
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL),IS_A_ONIUM(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      logical  do_cuts(nexternal)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY,
     .     IS_A_ONIUM, do_cuts


c     (re)set thresholds
      doEVACuts=.true.
      doEVAMatch=.true.         ! set to .true.(.false.) for emu (eW) scattering
      nrPho=0
      maxEta=3.d0      
      minPtX=80.d0
      minM2=(1d3)**2
      evoScale2 = 0d0
      evoThres2 = q2fact(2)


c     for each external:
      do ff=nincoming+1,nexternal

c     Cut on evolution variable:
c     1. Find v_mu (exists only if full ME)
c     2. Build evolution variable (0=q^2 or 1=pT^2)
c     3. Cut on evolution variable
         tmpPID1 = abs(idup(ff,1,1))
         if(tmpPID1.eq.14.and.doEVAMatch) then
            if(ievo_eva.eq.1) then
               evoScale2 = pT(p(0,ff))**2
            else
               do ii=1,nincoming
                  tmpPID2=abs(idup(ii,1,1))
                  if(tmpPID2.eq.13) then
                     evoScale2 = abs(SumDot(p(0,ff),p(0,ii),-1d0))
                     exit
                  endif
               enddo
            endif
            if(evoScale2.lt.evoThres2) then
               dummy_cuts=.false.
               return
            endif
         endif


c     Cut on photons:
c     1. Find photons and add momentum
c     2. Once three photons found: impose cut on invariant mass
c     3. pT/eta cuts are redundant/sanity check
         if(IS_A_A(ff).and.doEVACuts) then
            nrPho=nrPho+1
            if(nrPho.lt.2) then ! firstTime                       
               do kk=0,3
                  pAAA(kk) = p(kk,ff) ! p(4-momentum,list-ID)
               enddo
            else
               do kk=0,3
                  pAAA(kk) = pAAA(kk) + p(kk,ff) ! p(4-momentum,list-ID)
               enddo
            endif
c     check M(AAA)            
            if(nrPho.gt.2) then
               if(dot(pAAA,pAAA).lt.minM2) then
                  dummy_cuts=.false.
                  return
               endif
            endif
c     check eta of A        
            if(abs(rap(p(0,ff))).gt.maxEta) then
               dummy_cuts=.false.
               return
            endif
c     check pT of A
            if(pt(p(0,ff)).lt.minPtX) then
               dummy_cuts=.false.
               return
            endif            
         endif         
      enddo

      dummy_cuts=.true.
      return
      end

      subroutine get_dummy_x1(sjac, X1, R, pbeam1, pbeam2, stot, shat)
      implicit none
      include 'maxparticles.inc'
      include 'run.inc'
c     include 'genps.inc'
      double precision sjac     ! jacobian. should be updated not reinit
      double precision X1       ! bjorken X. output
      double precision R        ! random value after grid transfrormation. between 0 and 1
      double precision pbeam1(0:3) ! momentum of the first beam (input and/or output)
      double precision pbeam2(0:3) ! momentum of the second beam (input and/or output)
      double precision stot     ! total energy  (input and /or output)
      double precision shat     ! output

c     global variable to set (or not)
      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap
      
      set_cm_rap=.false.        ! then cm_rap will be set as .5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
!     ebeam(1) and ebeam(2) are defined here thanks to 'run.inc'
      shat = x1*ebeam(1)*ebeam(2)
      return 
      end

      subroutine get_dummy_x1_x2(sjac, X, R, pbeam1, pbeam2, stot,shat)
      implicit none
      include 'maxparticles.inc'
      include 'run.inc'
c     include 'genps.inc'
      double precision sjac     ! jacobian. should be updated not reinit
      double precision X(2)     ! bjorken X. output
      double precision R(2)     ! random value after grid transfrormation. between 0 and 1
      double precision pbeam1(0:3) ! momentum of the first beam
      double precision pbeam2(0:3) ! momentum of the second beam
      double precision stot     ! total energy
      double precision shat     ! output

c     global variable to set (or not)
      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap
      
      set_cm_rap=.false.        ! then cm_rap will be set as .5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
!     ebeam(1) and ebeam(2) are defined here thanks to 'run.inc'
      shat = x(1)*x(2)*ebeam(1)*ebeam(2)
      return 
      end


      logical  function dummy_boostframe()
      implicit none
c     
c     
      dummy_boostframe = .false.
      return
      end
      
