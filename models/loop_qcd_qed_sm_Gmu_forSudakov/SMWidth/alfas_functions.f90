MODULE alfas_functions
  IMPLICIT NONE
CONTAINS
  FUNCTION alfa(alfa0,qsq )
!-----------------------------------------------------------------------------
!
!	This function returns the 1-loop value of alpha.
!
!	INPUT: 
!		qsq   = Q^2
!
!-----------------------------------------------------------------------------
    IMPLICIT NONE
    REAL(KIND(1d0))::alfa
    REAL(KIND(1d0)),INTENT(IN)::qsq,alfa0
    REAL(KIND(1d0)),PARAMETER::One=1d0, Three=3d0,&
         Pi=3.14159265358979323846d0,zmass=91.188d0
    alfa = alfa0 / ( 1.0d0 - alfa0*DLOG( qsq/zmass**2 ) /Three /Pi )
    RETURN
  END FUNCTION alfa

  FUNCTION alfaw(alfaw0,qsq,nh )
!-----------------------------------------------------------------------------
!
!	This function returns the 1-loop value of alpha_w.
!
!	INPUT: 
!		qsq = Q^2
!               nh  = # of Higgs doublets
!
!-----------------------------------------------------------------------------
    IMPLICIT NONE
    REAL(KIND(1d0))::alfaw
    REAL(KIND(1d0)),INTENT(IN)::alfaw0,qsq
    INTEGER,INTENT(IN)::nh
    REAL(KIND(1d0))::dum
    INTEGER::nq
    REAL(KIND(1d0)),PARAMETER::Two=2d0, Four=4d0, &
         Pi=3.14159265358979323846d0, &
         Twpi=37.69911184307752d0, &
         zmass=91.188d0,tmass=174d0
    IF(qsq.GE.tmass**2)THEN
       nq = 6
    ELSE
       nq = 5
    ENDIF
    dum = (22.0d0-Four*nq-nh/Two)/Twpi
    alfaw = alfaw0 / ( 1.0d0 + dum*alfaw0*DLOG( qsq/zmass**2 ) )
    RETURN
  END FUNCTION alfaw

  FUNCTION ALPHAS_MCFM(Q) RESULT(ALPHAS)
!
!     Evaluation of strong coupling constant alpha_S
!     Author: R.K. Ellis
!
!     q -- scale at which alpha_s is to be evaluated
!
!-- common block alfas.inc
!     alphaQCD2 -- value of alpha_s at the mass of the Z-boson
!     nloop -- the number of loops (1,2, or 3) at which beta 
!
!     function is evaluated to determine running.
!     the values of the cmass and the bmass should be set
!     in common block qmass.
!-----------------------------------------------------------------------------
    USE ParamModule
    IMPLICIT NONE
    REAL(KIND(1d0))::ALPHAS
    REAL(KIND(1d0)),INTENT(IN)::Q
    REAL(KIND(1d0))::T,AMZ0=0D0,AMB,AMC
    REAL(KIND(1d0))::AS_OUT
    INTEGER::NLOOP0=0
    INTEGER,PARAMETER::NF3=3,NF4=4,NF5=5
    REAL(KIND(1d0))::CMASS=1.42D0,BMASS=4.7D0 ! HEAVY QUARK MASSES FOR THRESHOLDS
    REAL(KIND(1d0))::ZMASS=91.188D0
    SAVE AMZ0,NLOOP0,AMB,AMC
    IF(Q.LE.0D0)THEN 
       WRITE(*,*) 'q .le. 0 in alphas'
       WRITE(*,*) 'q= ',Q
       STOP
    ENDIF
    IF(alphaQCD2.LE.0D0)THEN 
       WRITE(*,*) 'alphaQCD2.LE.0 in alphas',alphaQCD2
       STOP
    ENDIF
    IF(CMASS.LE.0.3D0)THEN 
       WRITE(*,*) 'cmass .le. 0.3GeV in alphas',CMASS
       STOP
       CMASS=1.42D0
    ENDIF
    IF(BMASS.LE.0D0)THEN 
       WRITE(*,*) 'bmass .le. 0 in alphas',BMASS
       STOP
    ENDIF
!--- establish value of coupling at b- and c-mass and save
    IF((alphaQCD2.NE.AMZ0).OR.(NLOOP.NE.NLOOP0))THEN
       AMZ0=alphaQCD2
       NLOOP0=NLOOP
       T=2D0*DLOG(BMASS/ZMASS)
       CALL NEWTON1(T,alphaQCD2,AMB,NLOOP,NF5)
       T=2D0*DLOG(CMASS/BMASS)
       CALL NEWTON1(T,AMB,AMC,NLOOP,NF4)
    ENDIF

!--- evaluate strong coupling at scale q
    IF(Q.LT.BMASS)THEN
       IF (Q.LT.CMASS)THEN
          T=2D0*DLOG(Q/CMASS)
          CALL NEWTON1(T,AMC,AS_OUT,NLOOP,NF3)
       ELSE
          T=2D0*DLOG(Q/BMASS)
          CALL NEWTON1(T,AMB,AS_OUT,NLOOP,NF4)
       ENDIF
    ELSE
       T=2D0*DLOG(Q/ZMASS)
       CALL NEWTON1(T,alphaQCD2,AS_OUT,NLOOP,NF5)
    ENDIF
    ALPHAS=AS_OUT
    RETURN
  END FUNCTION ALPHAS_MCFM


  SUBROUTINE NEWTON1(T,A_IN,A_OUT,NLOOP,NF)
!     Author: R.K. Ellis

!---  calculate a_out using nloop beta-function evolution 
!---  with nf flavours, given starting value as-in
!---  given as_in and logarithmic separation between 
!---  input scale and output scale t.
!---  Evolution is performed using Newton's method,
!---  with a precision given by tol.
    
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOP,NF
    REAL(KIND(1d0)),INTENT(IN)::T,A_IN
    REAL(KIND(1d0)),INTENT(OUT)::A_OUT
    REAL(KIND(1d0))::AS,F,FP,DELTA
    REAL(KIND(1d0)),DIMENSION(3:5)::B0,C1,C2,DEL
    REAL(KIND(1d0)),PARAMETER::TOL=5.D-4
! ---     B0=(11.-2.*NF/3.)/4./PI
    B0(3:5)=(/0.716197243913527D0,0.66314559621623D0,0.61009394851893D0/)
!---     C1=(102.D0-38.D0/3.D0*NF)/4.D0/PI/(11.D0-2.D0/3.D0*NF)
    C1(3:5)=(/.565884242104515D0,0.49019722472304D0,0.40134724779695D0/)
!---     C2=(2857.D0/2.D0-5033*NF/18.D0+325*NF**2/54)
!---     /16.D0/PI**2/(11.D0-2.D0/3.D0*NF)
    C2(3:5)=(/0.453013579178645D0,0.30879037953664D0,0.14942733137107D0/)
!---     DEL=SQRT(4*C2-C1**2)
    DEL(3:5)=(/1.22140465909230D0,0.99743079911360D0,0.66077962451190D0/)
!    F2(AS)=1D0/AS+C1(NF)*DLOG((C1(NF)*AS)/(1D0+C1(NF)*AS))
!    F3(AS)=1D0/AS+0.5D0*C1(NF)&
!         *DLOG((C2(NF)*AS**2)/(1D0+C1(NF)*AS+C2(NF)*AS**2))&
!         -(C1(NF)**2-2D0*C2(NF))/DEL(NF)&
!         *DATAN((2D0*C2(NF)*AS+C1(NF))/DEL(NF))

           
    A_OUT=A_IN/(1D0+A_IN*B0(NF)*T)
    IF(NLOOP.EQ.1)RETURN
    A_OUT=A_IN/(1D0+B0(NF)*A_IN*T+C1(NF)*A_IN*DLOG(1D0+A_IN*B0(NF)*T))
    IF (A_OUT.LT.0D0)AS=0.3D0
    DO
       AS=A_OUT ! 30

       IF(NLOOP.EQ.2)THEN
          F=B0(NF)*T+F2(A_IN)-F2(AS)
          FP=1D0/(AS**2*(1D0+C1(NF)*AS))
       ENDIF
       IF(NLOOP.EQ.3)THEN
          F=B0(NF)*T+F3(A_IN)-F3(AS)
          FP=1D0/(AS**2*(1D0+C1(NF)*AS+C2(NF)*AS**2))
       ENDIF
       A_OUT=AS-F/FP
       DELTA=ABS(F/FP/AS)
       IF(DELTA.LE.TOL)EXIT  ! GO TO 30
    ENDDO
    RETURN

    CONTAINS

      FUNCTION F2(AS)
        IMPLICIT NONE
        REAL(KIND(1d0)),INTENT(IN)::AS
        REAL(KIND(1d0))::F2
        F2=1D0/AS+C1(NF)*DLOG((C1(NF)*AS)/(1D0+C1(NF)*AS))
      END FUNCTION F2

      FUNCTION F3(AS)
        IMPLICIT NONE
        REAL(KIND(1d0)),INTENT(IN)::AS
        REAL(KIND(1d0))::F3
        F3=1D0/AS+0.5D0*C1(NF)&
             *DLOG((C2(NF)*AS**2)/(1D0+C1(NF)*AS+C2(NF)*AS**2))&
             -(C1(NF)**2-2D0*C2(NF))/DEL(NF)&
             *DATAN((2D0*C2(NF)*AS+C1(NF))/DEL(NF))
      END FUNCTION F3
  END SUBROUTINE NEWTON1


  FUNCTION mfrun(mf,scale,asmz,nloop)
!-----------------------------------------------------------------------------
!
!	This function returns the 2-loop value of a MSbar fermion mass
!       at a given scale.
!
!	INPUT: mf    = MSbar mass of fermion at MSbar fermion mass scale 
!	       scale = scale at which the running mass is evaluated
!	       asmz  = AS(MZ) : this is passed to alphas(scale,asmz,nloop)
!              nloop = # of loops in the evolution
!       
!
!
!                      
!-----------------------------------------------------------------------------
    IMPLICIT NONE
    REAL(KIND(1d0))::mfrun
    REAL(KIND(1d0)),INTENT(IN)::mf,scale,asmz
    INTEGER,INTENT(IN)::nloop
    REAL(KIND(1d0))::beta0, beta1,gamma0,gamma1
    REAL(KIND(1d0))::A1,as,asmf,l2
    INTEGER::nf
    REAL(KIND(1d0)),PARAMETER::One=1d0,Two=2d0,Three=3d0,&
         Pi=3.14159265358979323846d0,tmass=174d0
    IF(mf.GT.tmass)THEN
       nf = 6
    ELSE
       nf = 5
    ENDIF

    beta0 = ( 11.0d0 - Two/Three *nf )/4d0
    beta1 = ( 102d0  - 38d0/Three*nf )/16d0
    gamma0= 1d0
    gamma1= ( 202d0/3d0  - 20d0/9d0*nf )/16d0
    A1    = -beta1*gamma0/beta0**2+gamma1/beta0
    as    = alphas_MCFM(scale)
    asmf  = alphas_MCFM(mf)
    l2    = (1+ A1*as/Pi)/(1+ A1*asmf/Pi)
      
      
    mfrun = mf * (as/asmf)**(gamma0/beta0)

    IF(nloop.EQ.2) mfrun =mfrun*l2
    RETURN
  END FUNCTION mfrun
END MODULE alfas_functions

