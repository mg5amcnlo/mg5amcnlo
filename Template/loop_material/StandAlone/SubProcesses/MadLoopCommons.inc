      subroutine joinPath(str1,str2,path)

      character*(*) str1
      character*(*) str2
      character*(*) path

      integer i,j,k

      i =1
      do while (i.le.LEN(str1))
      if(str1(i:i).eq.' ') goto 800     
      path(i:i) = str1(i:i)
      i=i+1
      enddo
800   continue
      j=1
      do while (j.le.LEN(str2))
      if(str2(j:j).eq.' ') goto 801
      path(i-1+j:i-1+j) = str2(j:j)
      j=j+1
      enddo
801   continue
      k=i+j-1
      do while (k.le.LEN(path))
      path(k:k) = ' '
      k=k+1
      enddo

      return

      end

      SUBROUTINE SET_FORBID_HEL_DOUBLECHECK(ONOFF)
C
C Give the possibility to overwrite the value of MadLoopParams.dat
c for the helicity double checking.
C Make sure to call this subroutine before the first time you 
C call MadLoop.
C
      IMPLICIT NONE
C
C ARGUMENT
C
      LOGICAL ONOFF
C
C GLOBAL VARIABLES
C
      LOGICAL FORBID_HEL_DOUBLECHECK
	  DATA FORBID_HEL_DOUBLECHECK/.False./
	  COMMON/FORBID_HEL_DOUBLECHECK/FORBID_HEL_DOUBLECHECK
C ----------
C BEGIN CODE
C ----------
      FORBID_HEL_DOUBLECHECK = ONOFF
      END

      subroutine setMadLoopPath(path)

      character(512) path
      character(512) dummy      

      character(512) prefix,fpath
      character(17) nameToCheck
      parameter (nameToCheck='MadLoopParams.dat')

      LOGICAL ML_INIT
      DATA ML_INIT/.TRUE./
      common/ML_INIT/ML_INIT

      LOGICAL CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT
      DATA CTINIT,TIRINIT,GOLEMINIT,SAMURAIINIT,NINJAINIT/.TRUE.,.TRUE.,.TRUE.,.TRUE.,.TRUE./
      COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT, GOLEMINIT, SAMURAIINIT, NINJAINIT

      character(512) MLPath
      data MLPath/'[[NA]]'/      
      common/MLPATH/MLPath

      integer i

C     Just a dummy call for LD to pick up this function
C     when creating the BLHA2 dynamic library
      dummy = ' '
      CALL SETPARA2(dummy)

      if (LEN(path).ge.4 .and. path(1:4).eq.'auto') then
          if (MLPath(1:6).eq.'[[NA]]') then
C     Try to automatically find the path
          prefix='./'
          call joinPath(prefix,nameToCheck,fpath)
          OPEN(1, FILE=fpath, ERR=1, STATUS='OLD',ACTION='READ')
          MLPath=prefix
          goto 10
1         continue
          close(1)
          prefix='./MadLoop5_resources/'
          call joinPath(prefix,nameToCheck,fpath)
          OPEN(1, FILE=fpath, ERR=2, STATUS='OLD',ACTION='READ')
          MLPath=prefix
          goto 10
2         continue
          close(1)
          prefix='../MadLoop5_resources/'
          call joinPath(prefix,nameToCheck,fpath)
          OPEN(1, FILE=fpath, ERR=66, STATUS='OLD',ACTION='READ')
          MLPath=prefix
          goto 10
66        continue
          close(1)
c     We could not automatically find the auxiliary files
          write(*,*) '==='
          write(*,*) 'ERROR: MadLoop5 could not automatically find the file MadLoopParams.dat.'
          write(*,*) '==='
          write(*,*) '(Try using <CALL setMadLoopPath(/my/path)> (before your first call to MadLoop) in order to set the directory where this file is located as well as  other auxiliary files, such as <xxx>_ColorNumFactors.dat, <xxx>_ColorDenomFactors.dat, etc..)'
          stop
10        continue
          close(1)
          return
          endif
      else
C     Use the one specified by the user
C     Make sure there is a separator added
      i =1
      do while (i.le.LEN(path) .and. path(i:i).ne.' ')
      i=i+1
      enddo
      if (path(i-1:i-1).ne.'/') then
          path(i:i) = '/'
      endif
      MLpath=path          
      endif

C     Check that the FilePath set is correct
      call joinPath(MLPath,nameToCheck,fpath)
      OPEN(1, FILE=fpath, ERR=3, STATUS='OLD',ACTION='READ')
      goto 11
3     continue
      close(1)
      write(*,*) '==='
      write(*,*) 'ERROR: The MadLoop5 auxiliary files could not be found in ',MLPath
      write(*,*) '==='
      stop
11    continue
      close(1)

      end

      INTEGER FUNCTION SET_RET_CODE_U(MLRed,DOING_QP,STABLE)
C
C This functions returns the value of U
C
C
C     U == 0
C         Not stable.
C     U == 1
C         Stable with CutTools in double precision.
C     U == 2
C         Stable with PJFry++.
C     U == 3
C         Stable with IREGI.
C     U == 4
C         Stable with Golem95.
C     U == 5
C         Stable with Samurai.
C     U == 6
C         Stable with Ninja in double precision.
C     U == 8
C         Stable with Ninja in quadruple precision.
C     U == 9
C         Stable with CutTools in quadruple precision.
C
      IMPLICIT NONE
C     
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER MLRed
      LOGICAL DOING_QP,STABLE
C
C LOCAL VARIABLES
C
C
C FUNCTION
C
C
C BEGIN CODE
C
      IF(.NOT.STABLE)THEN
         SET_RET_CODE_U=0
         RETURN
      ENDIF
      IF(DOING_QP)THEN
         IF(MLRed.EQ.1)THEN
            SET_RET_CODE_U=9
            RETURN
         ELSEIF(MLRed.EQ.6)THEN
            SET_RET_CODE_U=8
            RETURN
		 ELSE
            STOP 'Only CutTools and Ninja can use quardruple precision'
         ENDIF
      ENDIF
      IF(MLRed.GE.1.AND.MLRed.LE.6)THEN
         SET_RET_CODE_U=MLRed
      ELSE
         STOP 'Only CutTools, PJFry++, IREGI, Golem95, Samurai and Ninja are available'
      ENDIF
      END

      SUBROUTINE DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,HAS_HEFT_VERTEX,MAX_SPIN_CONNECTED_TO_LOOP,LPASS)
C
C DETECT WHICH LOOP LIB PASSED
C
      IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER LIBNUM,NLOOPLINE,RANK,MAX_SPIN_CONNECTED_TO_LOOP
C     The argument HAS_HEFT_VERTEX is only to implement correctly CutTools limitation
      LOGICAL complex_mass,LPASS,HAS_HEFT_VERTEX
C     
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
      IF(LIBNUM.EQ.1)THEN
C CutTools
         CALL DETECT_CUTTOOLS(NLOOPLINE,RANK,complex_mass,HAS_HEFT_VERTEX,MAX_SPIN_CONNECTED_TO_LOOP,LPASS)
      ELSEIF(LIBNUM.EQ.2)THEN
C PJFry++
         CALL DETECT_PJFRY(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.3)THEN
C IREGI
         CALL DETECT_IREGI(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.4)THEN
C Golem95
         CALL DETECT_GOLEM(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.5)THEN
C Samurai
         CALL DETECT_SAMURAI(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.6)THEN
C Ninja 
         CALL DETECT_NINJA(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSE
         STOP 'ONLY CUTTOOLS,PJFry++,IREGI,Golem95 and Samurai are available'
      ENDIF
      RETURN
      END

      SUBROUTINE DETECT_CUTTOOLS(NLOOPLINE,RANK,complex_mass,HAS_HEFT_VERTEX,MAX_SPIN_CONNECTED_TO_LOOP,LPASS)
C
C DETECT whether CUTTOOLS CAN BE USED OR NOT
C
      IMPLICIT NONE

C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER NLOOPLINE,RANK
	  INTEGER MAX_SPIN_CONNECTED_TO_LOOP
      LOGICAL complex_mass,LPASS,HAS_HEFT_VERTEX
C
C LOCAL VARIABLES
C
      INTEGER MAX_RANK
C ----------
C BEGIN CODE
C ----------
      LPASS=.TRUE.
C     The limit of 10 loop lines is just a parameter hardcoded in CutTools sources.
C     It can easily be increased if necessary.
C     Also in the presence of spin2 particles, RANK=NLOOPLINE+1 is not supported,
C     or in general whenever the higher rank doesn't come from the Higgs effective vertex.

      IF (MAX_SPIN_CONNECTED_TO_LOOP.le.3.AND.HAS_HEFT_VERTEX) THEN
	    MAX_RANK = NLOOPLINE+1
	  ELSE
	    MAX_RANK = NLOOPLINE
	  ENDIF

      IF( (RANK.GT.MAX_RANK).OR.(NLOOPLINE.gt.10) ) THEN
	    LPASS=.FALSE.	  
	  endif

      RETURN
      END

      SUBROUTINE DETECT_SAMURAI(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT whether Samurai CAN BE USED OR NOT
C
      IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER NLOOPLINE,RANK
      LOGICAL complex_mass,LPASS
C
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
      LPASS=.TRUE.
C     The limit of 8 loop lines is just a parameter hardcoded in Samurai sources.
C     It can easily be increased if necessary.
      IF((NLOOPLINE+1.LT.RANK).OR.(NLOOPLINE.gt.8)) then
	    LPASS=.FALSE.
      endif
      RETURN
      END

      SUBROUTINE DETECT_NINJA(NLOOPLINE,RANK,complex_mass,LPASS)
C
C Detect whether Ninja can be used or not
C
      IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER NLOOPLINE,RANK
      LOGICAL complex_mass,LPASS
C
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
      LPASS=.TRUE.
C     The limit of rank 20 is just a parameter hardcoded in Ninja sources.
C     It can easily be increased if necessary.
      IF((NLOOPLINE+1.LT.RANK).OR.(RANK.GE.20)) then
	    LPASS=.FALSE.
      endif
      RETURN
      END

      SUBROUTINE DETECT_PJFRY(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT whether PJFRY++ CAN BE USED OR NOT
C
      IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER NLOOPLINE,RANK
      LOGICAL complex_mass,LPASS
C
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
      LPASS=.TRUE.
      IF(NLOOPLINE.LT.RANK.OR.RANK.GT.5.OR.NLOOPLINE.GT.5.OR.complex_mass.OR.NLOOPLINE.eq.1) THEN
        LPASS=.FALSE.
      ENDIF
      RETURN
      END

      SUBROUTINE DETECT_IREGI(NLOOPLINE,RANK,complex_mass,LPASS)
C     
C DETECT whether IREGI CAN BE USED OR NOT
C
      IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
      INTEGER NLOOPLINE,RANK
      LOGICAL complex_mass,LPASS
C     
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------
C     Stability studies show that IREGI is completely unstable at rank 7 and above.
      LPASS=.TRUE.
      IF(NLOOPLINE.GE.8.OR.RANK.GE.7)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE DETECT_GOLEM(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT whether Golem95 CAN BE USED OR NOT
C
            IMPLICIT NONE
C
C CONSTANTS
C
C
C ARGUMENTS
C
            INTEGER NLOOPLINE,RANK
            LOGICAL complex_mass,LPASS
C
C LOCAL VARIABLES
C
C
C GLOBAL VARIABLES
C
C ----------
C BEGIN CODE
C ----------

      LPASS=.TRUE.
      IF(NLOOPLINE.GE.7.OR.RANK.GE.7.OR.NLOOPLINE.LE.1)LPASS=.FALSE.
      IF(NLOOPLINE.LE.5.AND.RANK.GT.NLOOPLINE+1)LPASS=.FALSE.
      IF(NLOOPLINE.EQ.6.AND.RANK.GT.NLOOPLINE)LPASS=.FALSE.
      RETURN
      END

C     Now some sorting related routines. Only to be used for small 
C     arrays since these are not the most optimized sorting algorithms.

! --------------------------------------------------------------------
! INTEGER FUNCTION  FindMinimum():
!    This function returns the location of the minimum in the section
! between Start and End.
! --------------------------------------------------------------------

      INTEGER FUNCTION  FindMinimum(x, mStart, mEnd)
      IMPLICIT  NONE
      INTEGER MAXNREF_EVALS
      PARAMETER (MAXNREF_EVALS=100)
      INTEGER, DIMENSION(MAXNREF_EVALS), INTENT(IN) :: x
      INTEGER, INTENT(IN)							  :: mStart, mEnd
      INTEGER										  :: Minimum
      INTEGER										  :: Location
      INTEGER										  :: i

      Minimum  = x(mStart)           ! assume the first is the min
      Location = mStart              ! record its position
      DO i = mStart+1, mEnd          ! start with next elements
         IF (x(i) < Minimum) THEN    !   if x(i) less than the min?
            Minimum  = x(i)          !      Yes, a new minimum found
            Location = i             !      record its position
         END IF
      END DO
      FindMinimum = Location         ! return the position
      END FUNCTION  FindMinimum

! --------------------------------------------------------------------
! SUBROUTINE  Swap():
!    This subroutine swaps the values of its two formal arguments.
! --------------------------------------------------------------------

      SUBROUTINE  Swap(a, b)
      IMPLICIT  NONE
      REAL*8,  INTENT(INOUT) :: a, b
      REAL*8                 :: Temp

      Temp = a
      a    = b
      b    = Temp
      END SUBROUTINE  Swap

! --------------------------------------------------------------------
! SUBROUTINE  Sort():
!    This subroutine receives an array x() and sorts it into ascending
! order.
! --------------------------------------------------------------------

      SUBROUTINE  Sort(x, mSize)
      IMPLICIT  NONE
      INTEGER MAXNREF_EVALS
      PARAMETER (MAXNREF_EVALS=100)
      REAL*8, DIMENSION(MAXNREF_EVALS), INTENT(INOUT)  :: x
      INTEGER, INTENT(IN)							   :: mSize
      INTEGER										   :: i
      INTEGER										   :: Location
      INTEGER 										   :: FindMinimum
      DO i = 1, mSize-1                          ! except for the last
         Location = FindMinimum(x, i, mSize)     ! find min from this to last
         CALL  Swap(x(i), x(Location))           ! swap this and the minimum
      END DO
      END SUBROUTINE  Sort

! --------------------------------------------------------------------
! REAL*8 FUNCTION  Median() :
!    This function receives an array X of N entries, copies its value
! to a local array Temp(), sorts Temp() and computes the median.
!    The returned value is of REAL type.
! --------------------------------------------------------------------

      REAL*8 FUNCTION  Median(X, N)
      IMPLICIT  NONE
      INTEGER MAXNREF_EVALS
      PARAMETER (MAXNREF_EVALS=100)
      REAL*8, DIMENSION(MAXNREF_EVALS), INTENT(IN)  :: X
      INTEGER, INTENT(IN)                			  :: N
      REAL*8, DIMENSION(MAXNREF_EVALS)              :: Temp
      INTEGER                                         :: i

      DO i = 1, N                       ! make a copy
         Temp(i) = X(i)
      END DO
      CALL  Sort(Temp, N)               ! sort the copy
      IF (MOD(N,2) == 0) THEN           ! compute the median
         Median = (Temp(N/2) + Temp(N/2+1)) / 2.0d0
      ELSE
         Median = Temp(N/2+1)
      END IF
      END FUNCTION  Median


      SUBROUTINE PRINT_MADLOOP_BANNER()

      %(print_banner_commands)s

      END

