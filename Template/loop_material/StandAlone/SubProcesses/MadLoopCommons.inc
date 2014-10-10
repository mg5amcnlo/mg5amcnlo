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

      subroutine setMadLoopPath(path)

      character(512) path
      character(512) dummy      

      character(512) prefix,fpath
      character(17) nameToCheck
      parameter (nameToCheck='MadLoopParams.dat')

      LOGICAL ML_INIT
      DATA ML_INIT/.TRUE./
      common/ML_INIT/ML_INIT

      LOGICAL CTINIT,TIRINIT,GOLEMINIT
      DATA CTINIT,TIRINIT,GOLEMINIT/.TRUE.,.TRUE.,.TRUE./
      COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT, GOLEMINIT

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
C         Stable with Golem95
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
         ELSE
            STOP 'Only CutTools can use quardruple precision'
         ENDIF
      ENDIF
      IF(MLRed.GE.1.AND.MLRed.LE.4)THEN
         SET_RET_CODE_U=MLRed
      ELSE
         STOP 'Only CutTools,PJFry++,IREGI,Golem95 are available'
      ENDIF
      END

      SUBROUTINE DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,complex_mass,LPASS)
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
      INTEGER LIBNUM,NLOOPLINE,RANK
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
      IF(LIBNUM.EQ.1)THEN
C CutTools
         CALL DETECT_CUTTOOLS(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.2)THEN
C PJFry++
         CALL DETECT_PJFRY(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.3)THEN
C IREGI
         CALL DETECT_IREGI(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSEIF(LIBNUM.EQ.4)THEN
C Golem95
         CALL DETECT_GOLEM(NLOOPLINE,RANK,complex_mass,LPASS)
      ELSE
         STOP 'ONLY CUTTOOLS,PJFry++,IREGI,Golem95 are provided'
      ENDIF
      RETURN
      END

      SUBROUTINE DETECT_CUTTOOLS(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT THE CUTTOOLS CAN BE USED OR NOT
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
      IF(NLOOPLINE+1.LT.RANK)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE DETECT_PJFRY(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT THE PJFRY++ CAN BE USED OR NOT
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
C DETECT THE IREGI CAN BE USED OR NOT
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
      IF(NLOOPLINE.GE.7.OR.RANK.GE.7)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE DETECT_GOLEM(NLOOPLINE,RANK,complex_mass,LPASS)
C
C DETECT THE Golem95 CAN BE USED OR NOT
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

      SUBROUTINE PRINT_MADLOOP_BANNER()

      %(print_banner_commands)s

      END
