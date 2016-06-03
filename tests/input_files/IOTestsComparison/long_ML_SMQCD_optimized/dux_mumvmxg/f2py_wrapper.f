      SUBROUTINE INITIALISE(PATH)

      CHARACTER(128) PATH
CF2PY intent(in)::path

C     INCLUDE FILES
C     
C     the include file with the values of the parameters and masses   
C     
      INCLUDE 'coupl.inc'
      CALL SETPARA(PATH)
      RETURN
      END

      SUBROUTINE GET_ME(P, ALPHAS, SCALE2, NHEL , ANS,RETURNCODE)
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

C     integer nexternal C number particles (incoming+outgoing) in the
C     me 
      INCLUDE 'nexternal.inc'

C     CHARACTER(512) MADLOOPRESOURCEPATH
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
      INCLUDE 'nsquaredSO.inc'

C     LOCAL
C     
      INTEGER I
C     four momenta. Energy is the zeroth component.
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER MATELEM_ARRAY_DIM
      REAL*8 , ALLOCATABLE :: MATELEM(:,:)
      INTEGER RETURNCODE
      INTEGER NSQUAREDSO_LOOP
      REAL*8 , ALLOCATABLE :: PREC_FOUND(:)

      DOUBLE PRECISION ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS, SCALE2
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(OUT) :: RETURNCODE
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: SCALE2


C     
C     GLOBAL VARIABLES
C     
C     This is from ML code for the list of split orders selected by
C     the process definition
C     
      INTEGER NLOOPCHOSEN
      CHARACTER*20 CHOSEN_LOOP_SO_INDICES(NSQUAREDSO)
      LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/ML5_0_CHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS
C     
C     BEGIN CODE
C     
      CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)
      CALL ML5_0_GET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
      ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
      CALL ML5_0_GET_NSQSO_LOOP(NSQUAREDSO_LOOP)
      ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))
      INCLUDE 'pmass.inc'

C     Start by initializing what is the squared split orders indices
C     chosen
      NLOOPCHOSEN=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLOOPCHOSEN=NLOOPCHOSEN+1
          WRITE(CHOSEN_LOOP_SO_INDICES(NLOOPCHOSEN),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO

C     Update the couplings with the new ALPHAS
      CALL UPDATE_AS_PARAM2(SCALE2, ALPHAS)

C     
C     Now we can call the matrix element
C     
      IF (NHEL.EQ.0) THEN
        CALL ML5_0_SLOOPMATRIX_THRES(P,MATELEM,-1.0D0, PREC_FOUND,
     $    RETURNCODE)
      ELSE
        CALL ML5_0_SLOOPMATRIXHEL_THRES(P,NHEL, MATELEM,-1.0D0,
     $    PREC_FOUND, RETURNCODE)
      ENDIF

C     loop induce -> only finite part 
      ANS =  MATELEM(1,0)

      END








