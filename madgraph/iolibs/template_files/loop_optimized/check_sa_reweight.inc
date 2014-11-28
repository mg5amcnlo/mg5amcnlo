      PROGRAM DRIVER
C *************************************************************************
C     THIS IS THE DRIVER FOR CHECKING THE STANDALONE MATRIX ELEMENT.
C     IT USES A SIMPLE PHASE SPACE GENERATOR
C *************************************************************************
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

      LOGICAL READPS
      PARAMETER (READPS = .FALSE.)

      INTEGER NPSPOINTS
      PARAMETER (NPSPOINTS = 10)

      CHARACTER*200 param_card
      real*8 new_alphas
      real*8 pi

C integer nexternal C number particles (incoming+outgoing) in the me 
      INTEGER NEXTERNAL, NINCOMING
	  PARAMETER (NEXTERNAL=%(nexternal)d,NINCOMING=%(nincoming)d)

	  character(512) MadLoopResourcePath

C     
C     INCLUDE FILES
C     
C the include file with the values of the parameters and masses      
      INCLUDE "coupl.inc"
C particle masses
      REAL*8 PMASS(NEXTERNAL)      
C integer    n_max_cg
      INCLUDE "ngraphs.inc" 
	  include "nsquaredSO.inc"

C     
C     LOCAL
C     
      INTEGER I,J,K
C four momenta. Energy is the zeroth component.
      REAL*8 P(0:3,NEXTERNAL)
	  INTEGER MATELEM_ARRAY_DIM
	  REAL*8 , allocatable :: MATELEM(:,:)
      REAL*8 SQRTS,AO2PI
C sqrt(s)= center of mass energy 
      REAL*8 PIN(0:3), POUT(0:3)
      CHARACTER*120 BUFF(NEXTERNAL)
	  INTEGER RETURNCODE, UNITS, TENS, HUNDREDS
	  INTEGER NSQUAREDSO_LOOP
	  REAL*8 , allocatable :: PREC_FOUND(:)

	  LOGICAL INIT
      DATA INIT/.TRUE./
	  COMMON/INITCHECKSA/INIT

C
C     GLOBAL VARIABLES
C
C     This is from ML code for the list of split orders selected by
c     the process definition
c
      integer NLoopChosen
      character*20 chosen_loop_so_indices(NSQUAREDSO)
	  LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS

C     
C     EXTERNAL
C     
      REAL*8 DOT
      EXTERNAL DOT
      
C 
C     BEGIN CODE
C
      pi = 3.141592653589793d0
      IF (INIT) THEN
	    INIT=.FALSE.
	    CALL %(proc_prefix)sGET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
	  	ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
		CALL %(proc_prefix)sGET_NSQSO_LOOP(NSQUAREDSO_LOOP)
	  	ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))
C     
C     INITIALIZATION CALLS
C     
C     Call to initialize the values of the couplings, masses and widths 
C     used in the evaluation of the matrix element. The primary parameters of the
C     models are read from Cards/param_card.dat. The secondary parameters are calculated
C     in Source/MODEL/couplings.f. The values are stored in common blocks that are listed
C     in coupl.inc .
C first call to setup the paramaters
      read(*,*) param_card
      call setpara(param_card)
C set up masses
      include "pmass.inc"             
		
	  ENDIF

c     Start by initializing what is the squared split orders indices chosen
      NLoopChosen=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLoopChosen=NLoopChosen+1
c          WRITE(chosen_loop_so_indices(NLoopChosen),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO

      AO2PI=G**2/(8.D0*(3.14159265358979323846d0**2))

c      write(*,*) 'AO2PI=',AO2PI
C     Now use a simple multipurpose PS generator (RAMBO) just to get a 
C     RANDOM set of four momenta of given masses pmass(i) to be used to evaluate 
C     the madgraph matrix-element.       
C     Alternatevely, here the user can call or set the four momenta at his will, see below.
C           
      IF(nincoming.EQ.1) THEN
         SQRTS=PMASS(1)
      ELSE
C CMS energy in GEV
         SQRTS=1000d0
      ENDIF

C     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C     ++   Starting of the infinite loop (with a f77 goto 1!)               ++
C     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
1     continue

C     In standalone mode, always use sqrt_s as the renormalization scale.
      read(*,*) new_alphas
C     READ THE POINT
      do i=1,nexternal
         read (*,*) P(0,i),P(1,i),P(2,i),P(3,i)
      enddo
      G = 2* DSQRT(new_alphas*pi)
      do i=0,3
        PIN(i)=0.0d0
        do j=1,nincoming
          PIN(i)=PIN(i)+p(i,j)
        enddo
      enddo
      SQRTS=dsqrt(dabs(DOT(PIN(0),PIN(0))))
      MU_R=SQRTS
      AO2PI=G**2/(8.D0*(3.14159265358979323846d0**2))
C     Update the couplings with the new MU_R
      CALL UPDATE_AS_PARAM()

C
C     Now we can call the matrix element
C
      CALL %(proc_prefix)sSLOOPMATRIX_THRES(P,MATELEM,-1.0d0,PREC_FOUND,RETURNCODE)

      write(*,*) MATELEM(1,0)
      call flush()


      goto 1
C     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C     ++   END of the infinite loop (with a f77 goto 1!)                    ++
C     ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      end
      
        
        
        
         double precision function dot(p1,p2)
C *************************************************************
C     4-Vector Dot product
C *************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end


        SUBROUTINE GET_MOMENTA(ENERGY,PMASS,P)
C       auxiliary function to change convention between madgraph and rambo
c       four momenta.         
        IMPLICIT NONE
        INTEGER NEXTERNAL, NINCOMING
	    PARAMETER (NEXTERNAL=%(nexternal)d,NINCOMING=%(nincoming)d)
C        ARGUMENTS
        REAL*8 ENERGY,PMASS(NEXTERNAL),P(0:3,NEXTERNAL),PRAMBO(4,10),WGT
C         LOCAL
         INTEGER I
         REAL*8 etot2,mom,m1,m2,e1,e2

         ETOT2=energy**2
         m1=pmass(1)
         m2=pmass(2)
         mom=(Etot2**2 - 2*Etot2*m1**2 + m1**4 - 2*Etot2*m2**2 - 2*m1**2*m2**2 + m2**4)/(4.*Etot2)
         mom=dsqrt(mom)
         e1=DSQRT(mom**2+m1**2)
         e2=DSQRT(mom**2+m2**2)
c         write (*,*) e1+e2,mom

         if(nincoming.eq.2) then

            P(0,1)=e1
            P(1,1)=0d0
            P(2,1)=0d0
            P(3,1)=mom

            P(0,2)=e2
            P(1,2)=0d0
            P(2,2)=0d0
            P(3,2)=-mom
             
            call rambo(nexternal-2,energy,pmass(3),prambo,WGT)
            DO I=3, NEXTERNAL
               P(0,I)=PRAMBO(4,I-2)      
               P(1,I)=PRAMBO(1,I-2)
               P(2,I)=PRAMBO(2,I-2)
               P(3,I)=PRAMBO(3,I-2)      
            ENDDO
             
          elseif(nincoming.eq.1) then 
             
             P(0,1)=energy
             P(1,1)=0d0
             P(2,1)=0d0
             P(3,1)=0d0
             
             call rambo(nexternal-1,energy,pmass(2),prambo,WGT)
             DO I=2, NEXTERNAL
                P(0,I)=PRAMBO(4,I-1)      
                P(1,I)=PRAMBO(1,I-1)
                P(2,I)=PRAMBO(2,I-1)
                P(3,I)=PRAMBO(3,I-1)      
             ENDDO
          endif
          
        RETURN
        END
      

      SUBROUTINE RAMBO(N,ET,XM,P,WT)
C **********************************************************************
C                       RAMBO                                         *
C    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
C                                                                     *
C    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR                *
C    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING                 *
C    THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS                      *
C    -- ADJUSTED BY HANS KUIJF, WEIGHTS ARE LOGARITHMIC (20-08-90)    *
C                                                                     *
C    N  = NUMBER OF PARTICLES                                         *
C    ET = TOTAL CENTRE-OF-MASS ENERGY                                 *
C    XM = PARTICLE MASSES ( DIM=NEXTERNAL-nincoming )                 *
C    P  = PARTICLE MOMENTA ( DIM=(4,NEXTERNAL-nincoming) )            *
C    WT = WEIGHT OF THE EVENT                                         *
C **********************************************************************
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER NEXTERNAL, NINCOMING
	  PARAMETER (NEXTERNAL=%(nexternal)d,NINCOMING=%(nincoming)d)
      DIMENSION XM(NEXTERNAL-NINCOMING),P(4,NEXTERNAL-NINCOMING)
      DIMENSION Q(4,NEXTERNAL-NINCOMING),Z(NEXTERNAL-NINCOMING),R(4),B(3),P2(NEXTERNAL-NINCOMING),XM2(NEXTERNAL-NINCOMING),E(NEXTERNAL-NINCOMING),V(NEXTERNAL-NINCOMING),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
C
C INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,(NEXTERNAL-NINCOMING-1)
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,(NEXTERNAL-NINCOMING-1)
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
C
C CHECK ON THE NUMBER OF PARTICLES
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
C
C CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP
C
C THE PARAMETER VALUES ARE NOW ACCEPTED
C
C GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE
  201 DO 202 I=1,N
         r1=rn(1)
      C=2.*r1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RN(2)
      r1=rn(3)
      r2=rn(4)
      Q(4,I)=-LOG(r1*r2)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
  202 Q(1,I)=Q(4,I)*S*SIN(F)
C
C CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 203 I=1,4
  203 R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
  204 R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
  205 B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
C
C TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(4,I)=X*(G*Q(4,I)+BQ)
C
C CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
  208 IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
C
C RETURN FOR WEIGHTED MASSLESS MOMENTA
  209 IF(NM.NE.0) GOTO 210
C RETURN LOG OF WEIGHT
      WT=WT
      RETURN
C
C MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
  210 XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
  301 P2(I)=P(4,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
  302 F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
  303 G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      PRINT 1006,ITMAX
      GOTO 305
  304 X=X-F0/(X*G0)
      GOTO 302
  305 DO 307 I=1,N
      V(I)=X*P(4,I)
      DO 306 K=1,3
  306 P(K,I)=X*P(K,I)
  307 P(4,I)=E(I)
C
C CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
  308 WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
C
C RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
  309 IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
C RETURN LOG OF WEIGHT
  310 WT=WT
      RETURN
C
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',' SMALLER THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE',' DESIRED ACCURACY =',D15.6)
      END

      FUNCTION RN(IDUMMY)
      REAL*8 RN,RAN
      SAVE INIT
      DATA INIT /1/
      IF (INIT.EQ.1) THEN
        INIT=0
        CALL RMARIN(1802,9373)
      END IF
C
  10  CALL RANMAR(RAN)
      IF (RAN.LT.1D-16) GOTO 10
      RN=RAN
C
      END



      SUBROUTINE RANMAR(RVEC)
C     -----------------
C Universal random number generator proposed by Marsaglia and Zaman
C in report FSU-SCRI-87-50
C In this version RVEC is a double precision variable.
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
      UNI = RANU(IRANMR) - RANU(JRANMR)
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RANU(IRANMR) = UNI
      IRANMR = IRANMR - 1
      JRANMR = JRANMR - 1
      IF(IRANMR .EQ. 0) IRANMR = 97
      IF(JRANMR .EQ. 0) JRANMR = 97
      RANC = RANC - RANCD
      IF(RANC .LT. 0D0) RANC = RANC + RANCM
      UNI = UNI - RANC
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RVEC = UNI
      END
 
      SUBROUTINE RMARIN(IJ,KL)
C     -----------------
C Initializing routine for RANMAR, must be called before generating
C any pseudorandom numbers with RANMAR. The input values should be in
C the ranges 0<=ij<=31328 ; 0<=kl<=30081
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
C This shows correspondence between the simplified input seeds IJ, KL
C and the original Marsaglia-Zaman seeds I,J,K,L.
C To get the standard values in the Marsaglia-Zaman paper (i=12,j=34
C k=56,l=78) put ij=1802, kl=9373
      I = MOD( IJ/177 , 177 ) + 2
      J = MOD( IJ     , 177 ) + 2
      K = MOD( KL/169 , 178 ) + 1
      L = MOD( KL     , 169 )
      DO 300 II = 1 , 97
        S =  0D0
        T = .5D0
        DO 200 JJ = 1 , 24
          M = MOD( MOD(I*J,179)*K , 179 )
          I = J
          J = K
          K = M
          L = MOD( 53*L+1 , 169 )
          IF(MOD(L*M,64) .GE. 32) S = S + T
          T = .5D0*T
  200   CONTINUE
        RANU(II) = S
  300 CONTINUE
      RANC  =   362436D0 / 16777216D0
      RANCD =  7654321D0 / 16777216D0
      RANCM = 16777213D0 / 16777216D0
      IRANMR = 97
      JRANMR = 33
      END






