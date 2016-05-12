      PROGRAM DRIVER
C     *****************************************************************
C     ********
C     THIS IS THE DRIVER FOR CHECKING THE STANDALONE MATRIX ELEMENT.
C     IT USES A SIMPLE PHASE SPACE GENERATOR
C     *****************************************************************
C     ********
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

      LOGICAL READPS
      PARAMETER (READPS = .FALSE.)

      INTEGER NPSPOINTS
      PARAMETER (NPSPOINTS = 4)

C     integer nexternal and number particles (incoming+outgoing) in
C      the me 
      INTEGER NEXTERNAL, NINCOMING
      PARAMETER (NEXTERNAL=4,NINCOMING=2)

      CHARACTER(512) MADLOOPRESOURCEPATH

C     
C     INCLUDE FILES
C     
C     the include file with the values of the parameters and masses   
C        
      INCLUDE 'coupl.inc'
C     particle masses
      REAL*8 PMASS(NEXTERNAL)
C     integer    n_max_cg
      INCLUDE 'ngraphs.inc'
      INCLUDE 'nsqso_born.inc'
      INCLUDE 'nsquaredSO.inc'

C     
C     LOCAL
C     
      INTEGER I,J,K
C     four momenta. Energy is the zeroth component.
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER MATELEM_ARRAY_DIM
      REAL*8 , ALLOCATABLE :: MATELEM(:,:)
      REAL*8 SQRTS,AO2PI,TOTMASS
C     sqrt(s)= center of mass energy 
      REAL*8 PIN(0:3), POUT(0:3)
      CHARACTER*120 BUFF(NEXTERNAL)
      INTEGER RETURNCODE, UNITS, TENS, HUNDREDS
      INTEGER NSQUAREDSO_LOOP
      REAL*8 , ALLOCATABLE :: PREC_FOUND(:)

C     
C     GLOBAL VARIABLES
C     
C     This is from ML code for the list of split orders selected by
C     the process definition
C     
      INTEGER NLOOPCHOSEN
      CHARACTER*20 CHOSEN_LOOP_SO_INDICES(NSQUAREDSO)
      LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/MG5_1_CHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS
      INTEGER NBORNCHOSEN
      CHARACTER*20 CHOSEN_BORN_SO_INDICES(NSQSO_BORN)
      LOGICAL CHOSEN_BORN_SO_CONFIGS(NSQSO_BORN)
      COMMON/MG5_1_CHOSEN_BORN_SQSO/CHOSEN_BORN_SO_CONFIGS

C     
C     SAVED VARIABLES
C     
      LOGICAL INIT
      DATA INIT/.TRUE./
      COMMON/INITCHECKSA/INIT
C     
C     EXTERNAL
C     
      REAL*8 DOT
      EXTERNAL DOT

C     
C     BEGIN CODE
C     
C     

      IF (INIT) THEN
        INIT=.FALSE.
        CALL MG5_1_GET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
        ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
        CALL MG5_1_GET_NSQSO_LOOP(NSQUAREDSO_LOOP)
        ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))

C       INITIALIZATION CALLS
C       
C       Call to initialize the values of the couplings, masses and
C        widths 
C       used in the evaluation of the matrix element. The primary
C        parameters of the
C       models are read from Cards/param_card.dat. The secondary
C        parameters are calculated
C       in Source/MODEL/couplings.f. The values are stored in common
C        blocks that are listed
C       in coupl.inc .
C       first call to setup the paramaters
        CALL SETPARA('param_card.dat')
C       set up masses
        INCLUDE 'pmass.inc'

      ENDIF


C     Start by initializing what is the squared split orders indices
C      chosen
      NLOOPCHOSEN=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLOOPCHOSEN=NLOOPCHOSEN+1
          WRITE(CHOSEN_LOOP_SO_INDICES(NLOOPCHOSEN),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO
      NBORNCHOSEN=0
      DO I=1,NSQSO_BORN
        IF (CHOSEN_BORN_SO_CONFIGS(I)) THEN
          NBORNCHOSEN=NBORNCHOSEN+1
          WRITE(CHOSEN_BORN_SO_INDICES(NBORNCHOSEN),'(I3,A2)') I,'B)'
        ENDIF
      ENDDO

      AO2PI=G**2/(8.D0*(3.14159265358979323846D0**2))

      WRITE(*,*) 'AO2PI=',AO2PI
C     Now use a simple multipurpose PS generator (RAMBO) just to get a 
C     RANDOM set of four momenta of given masses pmass(i) to be used
C      to evaluate 
C     the madgraph matrix-element.       
C     Alternatevely, here the user can call or set the four momenta at
C      his will, see below.
C     
      IF(NINCOMING.EQ.1) THEN
        SQRTS=PMASS(1)
      ELSE
        TOTMASS = 0.0D0
        DO I=1,NEXTERNAL
          TOTMASS = TOTMASS + PMASS(I)
        ENDDO
C       CMS energy in GEV
        SQRTS=MAX(1000D0,2.0D0*TOTMASS)
      ENDIF

      CALL PRINTOUT()



      DO K=1,NPSPOINTS

        IF(READPS) THEN
          OPEN(967, FILE='PS.input', ERR=976, STATUS='OLD', ACTION='RE'
     $     //'AD')
          DO I=1,NEXTERNAL
            READ(967,*,END=978) P(0,I),P(1,I),P(2,I),P(3,I)
          ENDDO
          GOTO 978
 976      CONTINUE
          STOP 'Could not read the PS.input phase-space point.'
 978      CONTINUE
          CLOSE(967)
        ELSE
          IF ((NINCOMING.EQ.2).AND.((NEXTERNAL - NINCOMING .EQ.1))
     $     ) THEN
            IF (PMASS(3).EQ.0.0D0) THEN
              STOP 'Cannot generate 2>1 kin. config. with m3=0.0d0'
            ELSE
C             deal with the case of only one particle in the final
C              state
              P(0,1) = PMASS(3)/2D0
              P(1,1) = 0D0
              P(2,1) = 0D0
              P(3,1) = PMASS(3)/2D0
              IF (PMASS(1).GT.0D0) THEN
                P(3,1) = DSQRT(PMASS(3)**2/4D0 - PMASS(1)**2)
              ENDIF
              P(0,2) = PMASS(3)/2D0
              P(1,2) = 0D0
              P(2,2) = 0D0
              P(3,2) = -PMASS(3)/2D0
              IF (PMASS(2) > 0D0) THEN
                P(3,2) = -DSQRT(PMASS(3)**2/4D0 - PMASS(1)**2)
              ENDIF
              P(0,3) = PMASS(3)
              P(1,3) = 0D0
              P(2,3) = 0D0
              P(3,3) = 0D0
            ENDIF
          ELSE
            CALL GET_MOMENTA(SQRTS,PMASS,P)
          ENDIF
        ENDIF

        DO I=0,3
          PIN(I)=0.0D0
          DO J=1,NINCOMING
            PIN(I)=PIN(I)+P(I,J)
          ENDDO
        ENDDO

C       In standalone mode, always use sqrt_s as the renormalization
C        scale.
        SQRTS=DSQRT(DABS(DOT(PIN(0),PIN(0))))
        MU_R=SQRTS

C       Update the couplings with the new MU_R
        CALL UPDATE_AS_PARAM()

C       Optionally the user can set where to find the MadLoop5_resource
C       s folder.
C       Otherwise it will look for it automatically and find it if it
C        has not
C       been moved
C       MadLoopResourcePath = '<MadLoop5_resources_path>'
C       CALL SETMADLOOPPATH(MadLoopResourcePath)
C       To force the stabiliy check to also be performed in the
C        initialization phase
C       CALL MG5_1_FORCE_STABILITY_CHECK(.TRUE.)
C       To chose a particular tartget split order, SOTARGET is an
C        integer labeling
C       the possible squared order couplings contributions (only in
C        optimized mode)
C       CALL MG5_1_SET_COUPLINGORDERS_TARGET(SOTARGET)


C       
C       Now we can call the matrix element
C       
        CALL MG5_1_SLOOPMATRIX_THRES(P,MATELEM,-1.0D0,PREC_FOUND
     $   ,RETURNCODE)

C       
C       write the information on the four momenta 
C       
        IF (K.EQ.NPSPOINTS) THEN
          WRITE (*,*)
          WRITE (*,*) ' Phase space point:'
          WRITE (*,*)
          WRITE (*,*) '---------------------------------'
          WRITE (*,*)  'n  E  px py pz m'
          DO I=1,NEXTERNAL
            WRITE (*,'(i2,1x,5e15.7)') I, P(0,I),P(1,I),P(2,I),P(3,I)
     $       ,DSQRT(DABS(DOT(P(0,I),P(0,I))))
          ENDDO
          WRITE (*,*) '---------------------------------'
          WRITE (*,*) 'Detailed result for each coupling order'
     $     //'s combination.'
          WRITE(*,*) 'All Born contributions are of split order'
     $     //'s (QCD=4)'
          WRITE (*,*) '---------------------------------'
          WRITE(*,*) 'All loop contributions are of split order'
     $     //'s (QCD=6)'
          WRITE (*,*) '---------------------------------'
          UNITS=MOD(RETURNCODE,10)
          TENS=(MOD(RETURNCODE,100)-UNITS)/10
          HUNDREDS=(RETURNCODE-TENS*10-UNITS)/100
          IF (HUNDREDS.EQ.1) THEN
            IF (TENS.EQ.3.OR.TENS.EQ.4) THEN
              WRITE(*,*) 'Unknown numerical stability because MadLoo'
     $         //'p is in the initialization stage.'
            ELSE
              WRITE(*,*) 'Unknown numerical stability, check CTModeRu'
     $         //'n value in MadLoopParams.dat.'
            ENDIF
          ELSEIF (HUNDREDS.EQ.2) THEN
            WRITE(*,*) 'Stable kinematic configuration (SPS).'
          ELSEIF (HUNDREDS.EQ.3) THEN
            WRITE(*,*) 'Unstable kinematic configuration (UPS).'
            WRITE(*,*) 'Quadruple precision rescue successful.'
          ELSEIF (HUNDREDS.EQ.4) THEN
            WRITE(*,*) 'Exceptional kinematic configuration (EPS).'
            WRITE(*,*) 'Both double an quadruple precision computation'
     $       //'s, are unstable.'
          ENDIF
          IF (TENS.EQ.2.OR.TENS.EQ.4) THEN
            WRITE(*,*) 'Quadruple precision computation used.'
          ENDIF
          IF (HUNDREDS.NE.1) THEN
            IF (PREC_FOUND(0).GT.0.0D0) THEN
              WRITE(*,'(1x,a23,1x,1e10.2)') 'Relative accuracy     ='
     $         ,PREC_FOUND(0)
            ELSEIF (PREC_FOUND(0).EQ.0.0D0) THEN
              WRITE(*,'(1x,a23,1x,1e10.2,1x,a30)') 'Relative accuracy'
     $         //'     =',PREC_FOUND(0),'(i.e. beyond double precisio'
     $         //'n)'
            ELSE
              WRITE(*,*) 'Estimated accuracy could not be computed fo'
     $         //'r an unknown reason.'
            ENDIF
          ENDIF
          WRITE (*,*) '---------------------------------'
          IF (NBORNCHOSEN.EQ.0) THEN
            WRITE (*,*) 'No Born contribution satisfied the square'
     $       //'d order constraints.'
          ELSE IF (NBORNCHOSEN.NE.NSQSO_BORN) THEN
            WRITE (*,*) 'Selected squared coupling orders combinatio'
     $       //'n for the Born summed result below:'
            WRITE (*,*) (CHOSEN_BORN_SO_INDICES(I),I=1,NBORNCHOSEN)
          ENDIF
          IF (NLOOPCHOSEN.NE.NSQUAREDSO) THEN
            WRITE (*,*) 'Selected squared coupling orders combinatio'
     $       //'n for the loop summed result below:'
            WRITE (*,*) (CHOSEN_LOOP_SO_INDICES(I),I=1,NLOOPCHOSEN)
          ENDIF
          WRITE (*,*) '---------------------------------'
          WRITE (*,*) 'Matrix element born   = ', MATELEM(0,0)
     $     , ' GeV^',-(2*NEXTERNAL-8)
          WRITE (*,*) 'Matrix element finite = ', MATELEM(1,0)
     $     , ' GeV^',-(2*NEXTERNAL-8)
          WRITE (*,*) 'Matrix element 1eps   = ', MATELEM(2,0)
     $     , ' GeV^',-(2*NEXTERNAL-8)
          WRITE (*,*) 'Matrix element 2eps   = ', MATELEM(3,0)
     $     , ' GeV^',-(2*NEXTERNAL-8)
          WRITE (*,*) '---------------------------------'
          IF (MATELEM(0,0).NE.0.0D0) THEN
            WRITE (*,*) 'finite / (born*ao2pi) = ', MATELEM(1,0)
     $       /MATELEM(0,0)/AO2PI
            WRITE (*,*) '1eps   / (born*ao2pi) = ', MATELEM(2,0)
     $       /MATELEM(0,0)/AO2PI
            WRITE (*,*) '2eps   / (born*ao2pi) = ', MATELEM(3,0)
     $       /MATELEM(0,0)/AO2PI
          ELSE
            WRITE (*,*) 'finite / ao2pi      = ', MATELEM(1,0)/AO2PI
            WRITE (*,*) '1eps   / ao2pi      = ', MATELEM(2,0)/AO2PI
            WRITE (*,*) '2eps   / ao2pi      = ', MATELEM(3,0)/AO2PI
          ENDIF
          WRITE (*,*) '---------------------------------'

          OPEN(69, FILE='result.dat', ERR=976, ACTION='WRITE')
          DO I=1,NEXTERNAL
            WRITE (69,'(a2,1x,5e25.15)') 'PS',P(0,I),P(1,I),P(2,I),P(3
     $       ,I)
          ENDDO
          WRITE (69,'(a3,1x,i2)') 'EXP',-(2*NEXTERNAL-8)
          WRITE (69,'(a4,1x,1e25.15)') 'BORN',MATELEM(0,0)
          IF (MATELEM(0,0).NE.0.0D0) THEN
            WRITE (69,'(a3,1x,1e25.15)') 'FIN',MATELEM(1,0)/MATELEM(0
     $       ,0)/AO2PI
            WRITE (69,'(a4,1x,1e25.15)') '1EPS',MATELEM(2,0)/MATELEM(0
     $       ,0)/AO2PI
            WRITE (69,'(a4,1x,1e25.15)') '2EPS',MATELEM(3,0)/MATELEM(0
     $       ,0)/AO2PI
          ELSE
            WRITE (69,'(a3,1x,1e25.15)') 'FIN',MATELEM(1,0)/AO2PI
            WRITE (69,'(a4,1x,1e25.15)') '1EPS',MATELEM(2,0)/AO2PI
            WRITE (69,'(a4,1x,1e25.15)') '2EPS',MATELEM(3,0)/AO2PI
          ENDIF
          WRITE (69,'(a6,1x,1e25.15)') 'ASO2PI',AO2PI
          WRITE (69,*) 'Export_Format Default'
          WRITE (69,'(a7,1x,i3)') 'RETCODE',RETURNCODE
          WRITE (69,'(a3,1x,1e10.4)') 'ACC',PREC_FOUND(0)
          WRITE (69,*) 'Born_kept',(CHOSEN_BORN_SO_CONFIGS(I),I=1
     $     ,NSQSO_BORN)
          WRITE (69,*) 'Loop_kept',(CHOSEN_LOOP_SO_CONFIGS(I),I=1
     $     ,NSQUAREDSO)
          WRITE (69,*) 'Born_SO_Results 4'
          WRITE (69,*) 'SO_Born BORN ',MATELEM(0,1)
          WRITE (69,*) 'Split_Orders_Names QCD'
          WRITE (69,*) 'Loop_SO_Results 6'
          WRITE (69,*) 'SO_Loop ACC  ',PREC_FOUND(1)
          WRITE (69,*) 'SO_Loop FIN  ',MATELEM(1,1)
          WRITE (69,*) 'SO_Loop 1EPS ',MATELEM(2,1)
          WRITE (69,*) 'SO_Loop 2EPS ',MATELEM(3,1)
          CLOSE(69)
        ELSE
          WRITE (*,*) 'PS Point #',K,' done.'
        ENDIF
      ENDDO

C     C
C     C      Copy down here (or read in) the four momenta as a string. 
C     C      
C     C
C     buff(1)=" 1   0.5630480E+04  0.0000000E+00  0.0000000E+00 
C      0.5630480E+04"
C     buff(2)=" 2   0.5630480E+04  0.0000000E+00  0.0000000E+00
C      -0.5630480E+04"
C     buff(3)=" 3   0.5466073E+04  0.4443190E+03  0.2446331E+04
C      -0.4864732E+04"
C     buff(4)=" 4   0.8785819E+03 -0.2533886E+03  0.2741971E+03 
C      0.7759741E+03"
C     buff(5)=" 5   0.4916306E+04 -0.1909305E+03 -0.2720528E+04 
C      0.4088757E+04"
C     C
C     C      Here the k,E,px,py,pz are read from the string into the
C      momenta array.
C     C      k=1,2          : incoming
C     C      k=3,nexternal  : outgoing
C     C
C     do i=1,nexternal
C     read (buff(i),*) k, P(0,i),P(1,i),P(2,i),P(3,i)
C     enddo
C     
C     C print the momenta out
C     
C     do i=1,nexternal
C     write (*,'(i2,1x,5e15.7)') i, P(0,i),P(1,i),P(2,i),P(3,i), 
C     &dsqrt(dabs(DOT(p(0,i),p(0,i))))
C     enddo
C     
C     CALL SLOOPMATRIX(P,MATELEM)
C     
C     write (*,*) "-------------------------------------------------"
C     write (*,*) "Matrix element = ", MATELEM(1), " GeV^",-(2*nexterna
C     l-8)      
C     write (*,*) "-------------------------------------------------"

      DEALLOCATE(MATELEM)
      DEALLOCATE(PREC_FOUND)

      END




      DOUBLE PRECISION FUNCTION DOT(P1,P2)
C     *************************************************************
C     4-Vector Dot product
C     *************************************************************
      IMPLICIT NONE
      DOUBLE PRECISION P1(0:3),P2(0:3)
      DOT=P1(0)*P2(0)-P1(1)*P2(1)-P1(2)*P2(2)-P1(3)*P2(3)
      END


      SUBROUTINE GET_MOMENTA(ENERGY,PMASS,P)
C     auxiliary function to change convention between madgraph and
C      rambo
C     four momenta.         
      IMPLICIT NONE
      INTEGER NEXTERNAL, NINCOMING
      PARAMETER (NEXTERNAL=4,NINCOMING=2)
C     ARGUMENTS
      REAL*8 ENERGY,PMASS(NEXTERNAL),P(0:3,NEXTERNAL),PRAMBO(4,10),WGT
C     LOCAL
      INTEGER I
      REAL*8 ETOT2,MOM,M1,M2,E1,E2

      ETOT2=ENERGY**2
      M1=PMASS(1)
      M2=PMASS(2)
      MOM=(ETOT2**2 - 2*ETOT2*M1**2 + M1**4 - 2*ETOT2*M2**2 - 2*M1**2
     $ *M2**2 + M2**4)/(4.*ETOT2)
      MOM=DSQRT(MOM)
      E1=DSQRT(MOM**2+M1**2)
      E2=DSQRT(MOM**2+M2**2)
C     write (*,*) e1+e2,mom

      IF(NINCOMING.EQ.2) THEN

        P(0,1)=E1
        P(1,1)=0D0
        P(2,1)=0D0
        P(3,1)=MOM

        P(0,2)=E2
        P(1,2)=0D0
        P(2,2)=0D0
        P(3,2)=-MOM

        CALL RAMBO(NEXTERNAL-2,ENERGY,PMASS(3),PRAMBO,WGT)
        DO I=3, NEXTERNAL
          P(0,I)=PRAMBO(4,I-2)
          P(1,I)=PRAMBO(1,I-2)
          P(2,I)=PRAMBO(2,I-2)
          P(3,I)=PRAMBO(3,I-2)
        ENDDO

      ELSEIF(NINCOMING.EQ.1) THEN

        P(0,1)=ENERGY
        P(1,1)=0D0
        P(2,1)=0D0
        P(3,1)=0D0

        CALL RAMBO(NEXTERNAL-1,ENERGY,PMASS(2),PRAMBO,WGT)
        DO I=2, NEXTERNAL
          P(0,I)=PRAMBO(4,I-1)
          P(1,I)=PRAMBO(1,I-1)
          P(2,I)=PRAMBO(2,I-1)
          P(3,I)=PRAMBO(3,I-1)
        ENDDO
      ENDIF

      RETURN
      END


      SUBROUTINE RAMBO(N,ET,XM,P,WT)
C     *****************************************************************
C     *****
C     RAMBO                                         *
C     RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                 
C      *
C     *
C     A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR               
C      *
C     AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING                
C      *
C     THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS                     
C      *
C     -- ADJUSTED BY HANS KUIJF, WEIGHTS ARE LOGARITHMIC (20-08-90)   
C      *
C     *
C     N  = NUMBER OF PARTICLES                                        
C      *
C     ET = TOTAL CENTRE-OF-MASS ENERGY                                
C      *
C     XM = PARTICLE MASSES ( DIM=NEXTERNAL-nincoming )                
C      *
C     P  = PARTICLE MOMENTA ( DIM=(4,NEXTERNAL-nincoming) )           
C      *
C     WT = WEIGHT OF THE EVENT                                        
C      *
C     *****************************************************************
C     *****
      IMPLICIT REAL*8(A-H,O-Z)
      INTEGER NEXTERNAL, NINCOMING
      PARAMETER (NEXTERNAL=4,NINCOMING=2)
      DIMENSION XM(NEXTERNAL-NINCOMING),P(4,NEXTERNAL-NINCOMING)
      DIMENSION Q(4,NEXTERNAL-NINCOMING),Z(NEXTERNAL-NINCOMING),R(4)
     $ ,B(3),P2(NEXTERNAL-NINCOMING),XM2(NEXTERNAL-NINCOMING)
     $ ,E(NEXTERNAL-NINCOMING),V(NEXTERNAL-NINCOMING),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
C     
C     INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,(NEXTERNAL-NINCOMING)
 101  Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,(NEXTERNAL-NINCOMING)
 102  Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
C     
C     CHECK ON THE NUMBER OF PARTICLES
 103  IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
C     
C     CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
 104  XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
 105  XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP
C     
C     THE PARAMETER VALUES ARE NOW ACCEPTED
C     
C     GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE
 201  DO 202 I=1,N
      R1=RN(1)
      C=2.*R1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RN(2)
      R1=RN(3)
      R2=RN(4)
      Q(4,I)=-LOG(R1*R2)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
 202  Q(1,I)=Q(4,I)*S*SIN(F)
C     
C     CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 203 I=1,4
 203  R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
 204  R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
 205  B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
C     
C     TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
 206  P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
 207  P(4,I)=X*(G*Q(4,I)+BQ)
C     
C     CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
 208  IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
C     
C     RETURN FOR WEIGHTED MASSLESS MOMENTA
 209  IF(NM.NE.0) GOTO 210
C     RETURN LOG OF WEIGHT
      WT=WT
      RETURN
C     
C     MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
 210  XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
 301  P2(I)=P(4,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
 302  F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
 303  G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      PRINT 1006,ITMAX
      GOTO 305
 304  X=X-F0/(X*G0)
      GOTO 302
 305  DO 307 I=1,N
      V(I)=X*P(4,I)
      DO 306 K=1,3
 306  P(K,I)=X*P(K,I)
 307  P(4,I)=E(I)
C     
C     CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
 308  WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
C     
C     RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
 309  IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
C     RETURN LOG OF WEIGHT
 310  WT=WT
      RETURN
C     
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',' SMALLE'
     $ //'R THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE'
     $ ,' DESIRED ACCURACY =',D15.6)
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
 10     CALL RANMAR(RAN)
        IF (RAN.LT.1D-16) GOTO 10
        RN=RAN
C       
        END



        SUBROUTINE RANMAR(RVEC)
C       -----------------
C       Universal random number generator proposed by Marsaglia and
C        Zaman
C       in report FSU-SCRI-87-50
C       In this version RVEC is a double precision variable.
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
C       -----------------
C       Initializing routine for RANMAR, must be called before
C        generating
C       any pseudorandom numbers with RANMAR. The input values should
C        be in
C       the ranges 0<=ij<=31328 ; 0<=kl<=30081
        IMPLICIT REAL*8(A-H,O-Z)
        COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
        COMMON/ RASET2 / IRANMR,JRANMR
        SAVE /RASET1/,/RASET2/
C       This shows correspondence between the simplified input seeds
C        IJ, KL
C       and the original Marsaglia-Zaman seeds I,J,K,L.
C       To get the standard values in the Marsaglia-Zaman paper
C        (i=12,j=34
C       k=56,l=78) put ij=1802, kl=9373
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
 200    CONTINUE
        RANU(II) = S
 300    CONTINUE
        RANC  =   362436D0 / 16777216D0
        RANCD =  7654321D0 / 16777216D0
        RANCM = 16777213D0 / 16777216D0
        IRANMR = 97
        JRANMR = 33
        END







