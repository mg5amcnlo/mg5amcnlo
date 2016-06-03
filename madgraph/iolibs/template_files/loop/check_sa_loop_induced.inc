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
      REAL*8 SQRTS,AO2PI,TOTMASS
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
      call setpara('param_card.dat')
C set up masses
      include "pmass.inc"             
		
	  ENDIF

c     Start by initializing what is the squared split orders indices chosen
      NLoopChosen=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLoopChosen=NLoopChosen+1
          WRITE(chosen_loop_so_indices(NLoopChosen),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO

      AO2PI=G**2/(8.D0*(3.14159265358979323846d0**2))

      write(*,*) 'AO2PI=',AO2PI
C     Now use a simple multipurpose PS generator (RAMBO) just to get a 
C     RANDOM set of four momenta of given masses pmass(i) to be used to evaluate 
C     the madgraph matrix-element.       
C     Alternatevely, here the user can call or set the four momenta at his will, see below.
C           
      IF(nincoming.EQ.1) THEN
         SQRTS=PMASS(1)
      ELSE
	     TOTMASS = 0.0d0
	     DO I=1,NEXTERNAL
		   TOTMASS = TOTMASS + PMASS(I)
		 ENDDO
C CMS energy in GEV
         SQRTS=max(1000d0,2.0d0*TOTMASS)
      ENDIF

      call printout()

      do K=1,NPSPOINTS

      if(READPS) then
        open(967, file="PS.input", err=976, status='OLD', action='READ')
        do i=1,NExternal
          read(967,*,end=978) P(0,i),P(1,i),P(2,i),P(3,i)
        enddo
       goto 978
  976  continue
         stop 'Could not read the PS.input phase-space point.'
  978  continue
       close(967)
      else
        if ((nincoming.eq.2).and.((nexternal - nincoming .eq.1))) then
          if (pmass(3).eq.0.0d0) then
              stop 'Cannot generate 2>1 kin. config. with m3=0.0d0'
          else
C deal with the case of only one particle in the final state
               p(0,1) = pmass(3)/2d0
               p(1,1) = 0d0
               p(2,1) = 0d0
               p(3,1) = pmass(3)/2d0
               if (pmass(1).GT.0d0) then
                 p(3,1) = dsqrt(pmass(3)**2/4d0 - pmass(1)**2)
               endif
               p(0,2) = pmass(3)/2d0
               p(1,2) = 0d0
               p(2,2) = 0d0
               p(3,2) = -pmass(3)/2d0
               if (pmass(2) > 0d0) then
                 p(3,2) = -dsqrt(pmass(3)**2/4d0 - pmass(1)**2)
               endif
               p(0,3) = pmass(3)
               p(1,3) = 0d0
               p(2,3) = 0d0
               p(3,3) = 0d0
          endif
        else
          CALL GET_MOMENTA(SQRTS,PMASS,P)
        endif
      endif

      do i=0,3
        PIN(i)=0.0d0
        do j=1,nincoming
          PIN(i)=PIN(i)+p(i,j)
        enddo
      enddo

C     In standalone mode, always use sqrt_s as the renormalization scale.
      SQRTS=dsqrt(dabs(DOT(PIN(0),PIN(0))))
      MU_R=SQRTS

C     Update the couplings with the new MU_R
      CALL UPDATE_AS_PARAM()

C     Optionally the user can set where to find the MadLoop5_resources folder.
C     Otherwise it will look for it automatically and find it if it has not
C     been moved
c     MadLoopResourcePath = '<MadLoop5_resources_path>'
c     CALL SETMADLOOPPATH(MadLoopResourcePath)
c     To force the stabiliy check to also be performed in the initialization phase
c     CALL %(proc_prefix)sFORCE_STABILITY_CHECK(.TRUE.)
c     To chose a particular tartget split order, SOTARGET is an integer labeling
c     the possible squared order couplings contributions (only in optimized mode)
c     CALL %(proc_prefix)sSET_COUPLINGORDERS_TARGET(SOTARGET)

C
C     Now we can call the matrix element
C
      CALL %(proc_prefix)sSLOOPMATRIX_THRES(P,MATELEM,-1.0d0,PREC_FOUND,RETURNCODE)
C
C        write the information on the four momenta 
C
      if (K.eq.NPSPOINTS) then
      write (*,*)
      write (*,*) " Phase space point:"
      write (*,*)
      write (*,*) "---------------------------------"
      write (*,*)  "n  E  px  py  pz  m"
      do i=1,nexternal
         write (*,'(i2,1x,5e15.7)') i, P(0,i),P(1,i),P(2,i),P(3,i),dsqrt(dabs(DOT(p(0,i),p(0,i))))
      enddo
      write (*,*) "---------------------------------"
      write (*,*) "Detailed result for each coupling orders combination."
	  %(print_so_born_results)s
	  %(print_so_loop_results)s
      UNITS=MOD(RETURNCODE,10)
      TENS=(MOD(RETURNCODE,100)-UNITS)/10
      HUNDREDS=(RETURNCODE-TENS*10-UNITS)/100
      if (HUNDREDS.eq.1) then
        if (TENS.eq.3.or.TENS.eq.4) then
          write(*,*) 'Unknown numerical stability because MadLoop is in the initialization stage.'
        else
          write(*,*) 'Unknown numerical stability, check CTModeRun value in MadLoopParams.dat.'
        endif
      elseif (HUNDREDS.eq.2) then
        write(*,*) 'Stable kinematic configuration (SPS).'
      elseif (HUNDREDS.eq.3) then
        write(*,*) 'Unstable kinematic configuration (UPS).'
        write(*,*) 'Quadruple precision rescue successful.'
      elseif (HUNDREDS.eq.4) then
        write(*,*) 'Exceptional kinematic configuration (EPS).'
        write(*,*) 'Both double an quadruple precision computations, are unstable.'
      endif
      if (TENS.eq.2.or.TENS.eq.4) then
        write(*,*) 'Quadruple precision computation used.'
      endif
      if (HUNDREDS.ne.1) then
        if (PREC_FOUND(0).gt.0.0d0) then
          write(*,'(1x,a23,1x,1e10.2)') 'Relative accuracy     =',PREC_FOUND(0)
        elseif (PREC_FOUND(0).eq.0.0d0) then
          write(*,'(1x,a23,1x,1e10.2,1x,a30)') 'Relative accuracy     =',PREC_FOUND(0),'(i.e. beyond double precision)'
        else
          write(*,*) 'Estimated accuracy could not be computed for an unknown reason.'
        endif
      endif
      write (*,*) "---------------------------------"
      IF (NLOOPCHOSEN.ne.NSQUAREDSO) THEN
        write (*,*) "Selected squared coupling orders combination for the loop summed result below:"
        write (*,*) (chosen_loop_so_indices(I),I=1,NLOOPCHOSEN)
      endif
      write (*,*) "---------------------------------"
      write (*,*) " This is a loop induced process, so only the "
	  write (*,*) " unnormalized finite part is output here. Be aware "
      write (*,*) " that all loops are expected to beUV-finite as no "
      write (*,*) " renormalization prescription is considered. "
      write (*,*) "---------------------------------"
      write (*,*) "Matrix element finite = ", MATELEM(1,0), " GeV^",-(2*nexternal-8)
      write (*,*) "---------------------------------"


      open(69, file="result.dat", err=976, action='WRITE')
      do i=1,nexternal      
          write (69,'(a2,1x,5ES30.15E3)') 'PS',P(0,i),P(1,i),P(2,i),P(3,i)
      enddo
      write (69,'(a3,1x,i3)') 'EXP',-(2*nexternal-8)
      write (69,'(a4,1x,1ES30.15E3)') 'BORN',0.0d0
      write (69,'(a3,1x,1ES30.15E3)') 'FIN',MATELEM(1,0)
      write (69,'(a4,1x,1ES30.15E3)') '1EPS',MATELEM(2,0)
      write (69,'(a4,1x,1ES30.15E3)') '2EPS',MATELEM(3,0)
      write (69,'(a6,1x,1ES30.15E3)') 'ASO2PI',AO2PI
      write (69,*) 'Export_Format LoopInduced'
	  write (69,'(a7,1x,i3)') 'RETCODE',RETURNCODE
	  write (69,'(a3,1x,1e10.4)') 'ACC',PREC_FOUND(0)
	  write (69,*) 'Born_kept F'
	  write (69,*) 'Loop_kept',(CHOSEN_LOOP_SO_CONFIGS(I),I=1,NSQUAREDSO)
	  %(write_so_born_results)s
	  %(write_so_loop_results)s	  
      close(69)
      else
          write (*,*) "PS Point #",K," done."
      endif
      enddo

C C
C C      Copy down here (or read in) the four momenta as a string. 
C C      
C C
C      buff(1)=" 1   0.5630480E+04  0.0000000E+00  0.0000000E+00  0.5630480E+04"
C      buff(2)=" 2   0.5630480E+04  0.0000000E+00  0.0000000E+00 -0.5630480E+04"
C      buff(3)=" 3   0.5466073E+04  0.4443190E+03  0.2446331E+04 -0.4864732E+04"
C      buff(4)=" 4   0.8785819E+03 -0.2533886E+03  0.2741971E+03  0.7759741E+03"
C      buff(5)=" 5   0.4916306E+04 -0.1909305E+03 -0.2720528E+04  0.4088757E+04"
C C
C C      Here the k,E,px,py,pz are read from the string into the momenta array.
C C      k=1,2          : incoming
C C      k=3,nexternal  : outgoing
C C
C      do i=1,nexternal
C         read (buff(i),*) k, P(0,i),P(1,i),P(2,i),P(3,i)
C      enddo
C
C C print the momenta out
C
C      do i=1,nexternal
C         write (*,'(i2,1x,5e15.7)') i, P(0,i),P(1,i),P(2,i),P(3,i), 
C     .dsqrt(dabs(DOT(p(0,i),p(0,i))))
C      enddo
C
C      CALL SLOOPMATRIX(P,MATELEM)
C
C      write (*,*) "-------------------------------------------------"
C      write (*,*) "Matrix element = ", MATELEM(1), " GeV^",
C      &-(2*nexternal-8)      
C      write (*,*) "-------------------------------------------------"

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
      DO 101 K=3,(NEXTERNAL-NINCOMING)
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,(NEXTERNAL-NINCOMING)
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






