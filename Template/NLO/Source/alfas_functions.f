C
C-----------------------------------------------------------------------------
C
      double precision function alfa(alfa0,qsq )
C
C-----------------------------------------------------------------------------
C
C	This function returns the 1-loop value of alpha.
C
C	INPUT: 
C		qsq   = Q^2
C
C-----------------------------------------------------------------------------
C
      implicit none
      double precision  qsq,alfa0
c
c constants
c
      double precision  One, Three, Pi,zmass
      parameter( One = 1.0d0, Three = 3.0d0 )
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( zmass = 91.188d0 )
cc
      alfa = alfa0 / ( 1.0d0 - alfa0*dlog( qsq/zmass**2 ) /Three /Pi )
ccc
      return
      end

C
C-----------------------------------------------------------------------------
C
      double precision function alfaw(alfaw0,qsq,nh )
C
C-----------------------------------------------------------------------------
C
C	This function returns the 1-loop value of alpha_w.
C
C	INPUT: 
C		qsq = Q^2
C               nh  = # of Higgs doublets
C
C-----------------------------------------------------------------------------
C
      implicit none
      double precision  qsq, alphaw, dum,alfaw0
      integer  nh, nq
c
c	  include
c
	  
c
c constants
c
      double precision  Two, Four, Pi, Twpi, zmass,tmass
      parameter( Two = 2.0d0, Four = 4.0d0 )
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Twpi = 3.0d0*Four*Pi )
      parameter( zmass = 91.188d0,tmass=174d0 )
cc
      if ( qsq.ge.tmass**2 ) then
         nq = 6
      else
         nq = 5
      end if
      dum = ( 22.0d0 - Four*nq - nh/Two ) / Twpi
      alfaw = alfaw0 / ( 1.0d0 + dum*alfaw0*dlog( qsq/zmass**2 ) )
ccc
      return
      end

      DOUBLE PRECISION FUNCTION ALPHAS(Q)

      DOUBLE PRECISION Q,ALPHAS_from_grids,alphas_not_timed
      external ALPHAS_from_grids,alphas_not_timed

c     timing statistics
      include "timing_variables.inc"

c     This function takes roughly 0.6 micro-seconds and the function
c     cpu_time takes 0.3 so it does not make much sense to always leave
c     the timing profile. But it can be uncommented for dedicated study.
c     call cpu_time(tbefore)
!     interpolate grids
      ALPHAS=ALPHAS_FROM_GRIDS(Q)
c$$$      ALPHAS=alphas_not_timed(Q)
c$$$      if (abs(ALPHAS/alphas_not_timed(Q)-1d0).gt.1d-4) then
c$$$         write (*,*) Q,ALPHAS,alphas_not_timed(Q),ALPHAS/alphas_not_timed(Q)-1d0
c$$$      endif
      
c     call cpu_time(tAfter)
      
c      tPDF = tPDF + (tAfter-tBefore)
      RETURN
      end

      real*8 function alphas_from_grids(scale)
      implicit none
      include 'alfas.inc'
      real*8 scale,CMASS,BMASS,Q,lscale
      integer i,ii,jj,gridsize_low,gridsize_mid,gridsize_high
      real*8 cut_off_low,cut_off_high
      parameter (gridsize_low=255)
      parameter (gridsize_mid=255)
      parameter (gridsize_high=255)
      parameter (cut_off_low=log(0.5d0))
      parameter (cut_off_high=log(100000d0)) ! 100 TeV
      real*8 grid_low(0:gridsize_low),grid_mid(0:gridsize_mid)
     $     ,grid_high(0:gridsize_high),grid_low_Q(0:gridsize_low)
     $     ,grid_mid_Q(0:gridsize_mid),grid_high_Q(0:gridsize_high)
     $     ,grid_low_sf(0:gridsize_low),grid_mid_sf(0:gridsize_mid)
     $     ,grid_high_sf(0:gridsize_high)
      logical firsttime
      data firsttime/.true./
      save CMASS,BMASS,grid_low,grid_mid,grid_high,grid_low_Q,grid_mid_Q
     $     ,grid_high_Q,firsttime,grid_low_sf,grid_mid_sf,grid_high_sf
      real*8 alphas_not_timed
      external alphas_not_timed
      if (firsttime) then
         write (*,*) 'Filling grid for alpha_S computation ... '
         CMASS=log(1.42D0)
         BMASS=log(4.7D0)
!     scales below the charm mass
         do i=0,gridsize_low
            Q=cut_off_low+(CMASS-cut_off_low)*dble(i)/dble(gridsize_low)
            grid_low_Q(i)=Q
            grid_low(i)=ALPHAS_not_timed(exp(Q))
            if (i.ne.0) grid_low_sf(i-1)=(grid_low(i)-grid_low(i-1))
     $           /(grid_low_Q(i)-grid_low_Q(i-1))
         enddo
!     scales between charm and bottom mass
         do i=0,gridsize_mid
            Q=CMASS+(BMASS-CMASS)*dble(i)/dble(gridsize_mid)
            grid_mid_Q(i)=Q
            grid_mid(i)=ALPHAS_not_timed(exp(Q))
            if (i.ne.0) grid_mid_sf(i-1)=(grid_mid(i)-grid_mid(i-1))
     $           /(grid_mid_Q(i)-grid_mid_Q(i-1))
         enddo
!     scales above the bottom mass
         do i=0,gridsize_high
            Q=BMASS+(cut_off_high-BMASS)*dble(i)/dble(gridsize_high)
            grid_high_Q(i)=Q
            grid_high(i)=ALPHAS_not_timed(exp(Q))
            if (i.ne.0) grid_high_sf(i-1)=(grid_high(i)-grid_high(i-1))
     $           /(grid_high_Q(i)-grid_high_Q(i-1))
         enddo
         firsttime=.false.
         write (*,*) 'done... '
      endif

      lscale=log(scale)
!     use linear interpolation to compute alpha_s
      if (lscale.lt.cut_off_low) then
         ! scale too low: cannot use grids
         alphas_from_grids=ALPHAS_not_timed(scale)
      elseif (lscale.lt.CMASS) then
         ii=int((lscale-cut_off_low)/(CMASS-cut_off_low)*gridsize_low)
         alphas_from_grids=grid_low(ii)+
     &        (lscale-grid_low_Q(ii))*grid_low_sf(ii)
      elseif (lscale.lt.BMASS) then
         ii=int((lscale-CMASS)/(BMASS-CMASS)*gridsize_mid)
         alphas_from_grids=grid_mid(ii)+
     &        (lscale-grid_mid_Q(ii))*grid_mid_sf(ii)
      elseif (lscale.lt.cut_off_high) then
         ii=int((lscale-BMASS)/(cut_off_high-BMASS)*gridsize_high)
         alphas_from_grids=grid_high(ii)+
     &        (lscale-grid_high_Q(ii))*grid_high_sf(ii)
      else
         ! scale too high: cannot use grids
         alphas_from_grids=ALPHAS_not_timed(scale)
      endif
      end
      
C-----------------------------------------------------------------------------
C
      DOUBLE PRECISION FUNCTION ALPHAS_not_timed(Q)
c
c     Evaluation of strong coupling constant alpha_S
c     Author: R.K. Ellis
c
c     q -- scale at which alpha_s is to be evaluated
c
c-- common block alfas.inc
c     asmz -- value of alpha_s at the mass of the Z-boson
c     nloop -- the number of loops (1,2, or 3) at which beta 
c
c     function is evaluated to determine running.
c     the values of the cmass and the bmass should be set
c     in common block qmass.
C-----------------------------------------------------------------------------

      IMPLICIT NONE
c
      include 'alfas.inc'
      DOUBLE PRECISION Q,T,AMZ0,AMB,AMC
      DOUBLE PRECISION AS_OUT
      INTEGER NLOOP0,NF3,NF4,NF5
      PARAMETER(NF5=5,NF4=4,NF3=3)
C
      REAL*8       CMASS,BMASS
      COMMON/QMASS/CMASS,BMASS
      DATA CMASS,BMASS/1.42D0,4.7D0/  ! HEAVY QUARK MASSES FOR THRESHOLDS
C
      REAL*8 ZMASS
      DATA ZMASS/91.188D0/
C
      SAVE AMZ0,NLOOP0,AMB,AMC
      DATA AMZ0,NLOOP0/0D0,0/
      IF (Q .LE. 0D0) THEN 
         WRITE(6,*) 'q .le. 0 in alphas'
         WRITE(6,*) 'q= ',Q
         STOP
      ENDIF
      IF (asmz .LE. 0D0) THEN 
         WRITE(6,*) 'asmz .le. 0 in alphas',asmz
c         WRITE(6,*) 'continue with asmz=0.1185'
         STOP
         asmz=0.1185D0
      ENDIF
      IF (CMASS .LE. 0.3D0) THEN 
         WRITE(6,*) 'cmass .le. 0.3GeV in alphas',CMASS
c         WRITE(6,*) 'continue with cmass=1.5GeV?'
         STOP
         CMASS=1.42D0
      ENDIF
      IF (BMASS .LE. 0D0) THEN 
         WRITE(6,*) 'bmass .le. 0 in alphas',BMASS
         WRITE(6,*) 'COMMON/QMASS/CMASS,BMASS'
         STOP
         BMASS=4.7D0
      ENDIF
c--- establish value of coupling at b- and c-mass and save
      IF ((asmz .NE. AMZ0) .OR. (NLOOP .NE. NLOOP0)) THEN
         AMZ0=asmz
         NLOOP0=NLOOP
         T=2D0*DLOG(BMASS/ZMASS)
         CALL NEWTON1(T,asmz,AMB,NLOOP,NF5)
         T=2D0*DLOG(CMASS/BMASS)
         CALL NEWTON1(T,AMB,AMC,NLOOP,NF4)
      ENDIF

c--- evaluate strong coupling at scale q
      IF (Q  .LT. BMASS) THEN
           IF (Q  .LT. CMASS) THEN
             T=2D0*DLOG(Q/CMASS)
             CALL NEWTON1(T,AMC,AS_OUT,NLOOP,NF3)
           ELSE
             T=2D0*DLOG(Q/BMASS)
             CALL NEWTON1(T,AMB,AS_OUT,NLOOP,NF4)
           ENDIF
      ELSE
      T=2D0*DLOG(Q/ZMASS)
      CALL NEWTON1(T,asmz,AS_OUT,NLOOP,NF5)
      ENDIF
      ALPHAS_not_timed=AS_OUT
      RETURN
      END


      SUBROUTINE NEWTON1(T,A_IN,A_OUT,NLOOP,NF)
C     Author: R.K. Ellis

c---  calculate a_out using nloop beta-function evolution 
c---  with nf flavours, given starting value as-in
c---  given as_in and logarithmic separation between 
c---  input scale and output scale t.
c---  Evolution is performed using Newton's method,
c---  with a precision given by tol.

      IMPLICIT NONE
      INTEGER NLOOP,NF
      REAL*8 T,A_IN,A_OUT,AS,TOL,F2,F3,F,FP,DELTA
      REAL*8 B0(3:5),C1(3:5),C2(3:5),DEL(3:5)
      PARAMETER(TOL=5.D-4)
C---     B0=(11.-2.*NF/3.)/4./PI
      DATA B0/0.716197243913527D0,0.66314559621623D0,0.61009394851893D0/
C---     C1=(102.D0-38.D0/3.D0*NF)/4.D0/PI/(11.D0-2.D0/3.D0*NF)
      DATA C1/.565884242104515D0,0.49019722472304D0,0.40134724779695D0/
C---     C2=(2857.D0/2.D0-5033*NF/18.D0+325*NF**2/54)
C---     /16.D0/PI**2/(11.D0-2.D0/3.D0*NF)
      DATA C2/0.453013579178645D0,0.30879037953664D0,0.14942733137107D0/
C---     DEL=SQRT(4*C2-C1**2)
      DATA DEL/1.22140465909230D0,0.99743079911360D0,0.66077962451190D0/
      F2(AS)=1D0/AS+C1(NF)*LOG((C1(NF)*AS)/(1D0+C1(NF)*AS))
      F3(AS)=1D0/AS+0.5D0*C1(NF)
     & *LOG((C2(NF)*AS**2)/(1D0+C1(NF)*AS+C2(NF)*AS**2))
     & -(C1(NF)**2-2D0*C2(NF))/DEL(NF)
     & *ATAN((2D0*C2(NF)*AS+C1(NF))/DEL(NF))

           
      A_OUT=A_IN/(1D0+A_IN*B0(NF)*T)
      IF (NLOOP .EQ. 1) RETURN
      A_OUT=A_IN/(1D0+B0(NF)*A_IN*T+C1(NF)*A_IN*LOG(1D0+A_IN*B0(NF)*T))
      IF (A_OUT .LT. 0D0) AS=0.3D0
 30   AS=A_OUT

      IF (NLOOP .EQ. 2) THEN
      F=B0(NF)*T+F2(A_IN)-F2(AS)
      FP=1D0/(AS**2*(1D0+C1(NF)*AS))
      ENDIF
      IF (NLOOP .EQ. 3) THEN
      F=B0(NF)*T+F3(A_IN)-F3(AS)
      FP=1D0/(AS**2*(1D0+C1(NF)*AS+C2(NF)*AS**2))
      ENDIF
      A_OUT=AS-F/FP
      DELTA=ABS(F/FP/AS)
      IF (DELTA .GT. TOL) GO TO 30
      RETURN
      END


C-----------------------------------------------------------------------------
C
      double precision function mfrun(mf,scale,asmz,nloop)
C
C-----------------------------------------------------------------------------
C
C	This function returns the 2-loop value of a MSbar fermion mass
C       at a given scale.
C
C	INPUT: mf    = MSbar mass of fermion at MSbar fermion mass scale 
C	       scale = scale at which the running mass is evaluated
C	       asmz  = AS(MZ) : this is passed to alphas(scale,asmz,nloop)
C              nloop = # of loops in the evolution
C       
C
C
C	EXTERNAL:      double precision alphas(scale,asmz,nloop)
C                      
C-----------------------------------------------------------------------------
C
      implicit none
C
C     ARGUMENTS
C
      double precision  mf,scale,asmz
      integer           nloop
C
C     LOCAL
C
      double precision  beta0, beta1,gamma0,gamma1
      double precision  A1,as,asmf,l2
      integer  nf
C
C     EXTERNAL
C
      double precision  alphas
      external          alphas
c
c     CONSTANTS
c
      double precision  One, Two, Three, Pi
      parameter( One = 1.0d0, Two = 2.0d0, Three = 3.0d0 )
      parameter( Pi = 3.14159265358979323846d0) 
      double precision tmass
      parameter(tmass=174d0)
cc
C
C
      if ( mf.gt.tmass ) then
         nf = 6
      else
         nf = 5
      end if

      beta0 = ( 11.0d0 - Two/Three *nf )/4d0
      beta1 = ( 102d0  - 38d0/Three*nf )/16d0
      gamma0= 1d0
      gamma1= ( 202d0/3d0  - 20d0/9d0*nf )/16d0
      A1    = -beta1*gamma0/beta0**2+gamma1/beta0
      as    = alphas(scale)
      asmf  = alphas(mf)
      l2    = (1+ A1*as/Pi)/(1+ A1*asmf/Pi)
      
      
      mfrun = mf * (as/asmf)**(gamma0/beta0)

      if(nloop.eq.2) mfrun =mfrun*l2
ccc
      return
      end

