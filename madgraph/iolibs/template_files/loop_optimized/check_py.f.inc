      Subroutine %(prefix_routine)sInitialise(path)

      CHARACTER(128) path
CF2PY intent(in)::path
 
C     INCLUDE FILES
C     
C     the include file with the values of the parameters and masses   
C        
      INCLUDE 'coupl.inc'
      CALL SETPARA(path)
      return 
      end

      SUBROUTINE %(prefix_routine)sGET_ME(P, ALPHAS, SCALE2, NHEL , ANS,RETURNCODE)
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

C     integer nexternal C number particles (incoming+outgoing) in the
C      me 
      include 'nexternal.inc'

C      CHARACTER(512) MADLOOPRESOURCEPATH
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
c
c     LOCAL BUFFERING OF ANSWER to avoid re-computing
c
	integer buffer_size
	parameter (buffer_size=5)
	double precision old_ALPHAS(buffer_size)
    double precision old_scale2(buffer_size)
    integer old_nhel(buffer_size)
    double precision old_p(0:3,nexternal,buffer_size)
      double precision old_me(buffer_size)
      integer old_return(buffer_size)
      integer buffer_position

	data buffer_position /1/
	save old_ALPHAS, old_scale2, old_nhel, old_p, buffer_position
	save old_me, old_return
C     
C     GLOBAL VARIABLES
C     
C     This is from ML code for the list of split orders selected by
C     the process definition
C     
      INTEGER NLOOPCHOSEN
      CHARACTER*20 CHOSEN_LOOP_SO_INDICES(NSQUAREDSO)
      LOGICAL CHOSEN_LOOP_SO_CONFIGS(NSQUAREDSO)
      COMMON/%(proc_prefix)sCHOSEN_LOOP_SQSO/CHOSEN_LOOP_SO_CONFIGS
C
C     CHECK BUFFERING
C
	do i=1, buffer_size
	  if(SCALE2.eq.OLD_SCALE2(i))then
	  if(ALPHAS.eq.OLD_ALPHAS(i).and.NHEL.eq.OLD_NHEL(i))then
	  	  if(ALL(P.eq.old_P(:,:,i)))then
		  	  ANS = old_me(i)
			  RETURNCODE = old_return(i)
			  return
		  endif
	  endif
	  endif
    enddo
C     
C     BEGIN CODE
C     
      CALL %(proc_prefix)sFORCE_STABILITY_CHECK(.True.)
      CALL %(proc_prefix)sGET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
      ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))
      CALL %(proc_prefix)sGET_NSQSO_LOOP(NSQUAREDSO_LOOP)
      ALLOCATE(PREC_FOUND(0:NSQUAREDSO_LOOP))
      INCLUDE 'pmass.inc'

C     Start by initializing what is the squared split orders indices
C      chosen
      NLOOPCHOSEN=0
      DO I=1,NSQUAREDSO
        IF (CHOSEN_LOOP_SO_CONFIGS(I)) THEN
          NLOOPCHOSEN=NLOOPCHOSEN+1
          WRITE(CHOSEN_LOOP_SO_INDICES(NLOOPCHOSEN),'(I3,A2)') I,'L)'
        ENDIF
      ENDDO

C       Update the couplings with the new ALPHAS
        CALL UPDATE_AS_PARAM2(SCALE2, ALPHAS)

C       
C       Now we can call the matrix element
C       
        if (NHEL.eq.0) then
          CALL %(proc_prefix)sSLOOPMATRIX_THRES(P,MATELEM,-1.0D0, PREC_FOUND, RETURNCODE)
       else
       CALL %(proc_prefix)sSLOOPMATRIXHEL_THRES(P,NHEL, MATELEM,-1.0D0, PREC_FOUND, RETURNCODE)
       endif
       
c      loop induce -> only finite part 
       ANS =  MATELEM(1,0)

c
c      Store result for the buffering
c
      old_me(buffer_position) = ans
	  old_return(buffer_position) = RETURNCODE
	  buffer_position = MOD(buffer_position, buffer_size) +1

      END







