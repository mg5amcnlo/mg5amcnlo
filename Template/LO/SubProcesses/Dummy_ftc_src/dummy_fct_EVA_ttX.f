C**************************************************************************
c     dummy_fct_EVA_ttX.f
c     2021 October (R. Ruiz)
c     Imposes the following phase space cuts in the process VV > t tbar. 
c     cos(\theta_final) < 1 - mtop2 / sHat
c     
c     Used to reproduce results of Han, et al [arXiv:]
c     For further details, see Constantini, et al [arXiv:2110.]
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
      double precision mtop2,sHat,rRat,cosTh
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
      mtop2 = (173.0d0)**2

c     build s-hat
      sHat = sumdot(p(0,1), p(0,2), 1d0)
      rRat = 1.d0 - mtop2 / sHat

c     for each external:
      if(doEVACuts) then
         do ff=nincoming+1,nexternal
            cosTh = p(3,ff) / sqrt(p(1,ff)**2 + p(2,ff)**2 + p(3,ff)**2) ! pz / (px2 + py2 + pz2)
            if(cosTh.gt.rRat) then
               dummy_cuts=.false.
               return
            endif
         enddo
      endif

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
      
