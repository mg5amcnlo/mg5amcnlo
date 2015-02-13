      SUBROUTINE JOINPATH(STR1,STR2,PATH)

      CHARACTER*(*) STR1
      CHARACTER*(*) STR2
      CHARACTER*(*) PATH

      INTEGER I,J,K

      I =1
      DO WHILE (I.LE.LEN(STR1))
        IF(STR1(I:I).EQ.' ') GOTO 800
        PATH(I:I) = STR1(I:I)
        I=I+1
      ENDDO
 800  CONTINUE
      J=1
      DO WHILE (J.LE.LEN(STR2))
        IF(STR2(J:J).EQ.' ') GOTO 801
        PATH(I-1+J:I-1+J) = STR2(J:J)
        J=J+1
      ENDDO
 801  CONTINUE
      K=I+J-1
      DO WHILE (K.LE.LEN(PATH))
        PATH(K:K) = ' '
        K=K+1
      ENDDO

      RETURN

      END

      SUBROUTINE SETMADLOOPPATH(PATH)

      CHARACTER(512) PATH
      CHARACTER(512) DUMMY

      CHARACTER(512) PREFIX,FPATH
      CHARACTER(17) NAMETOCHECK
      PARAMETER (NAMETOCHECK='MadLoopParams.dat')

      LOGICAL ML_INIT
      DATA ML_INIT/.TRUE./
      COMMON/ML_INIT/ML_INIT

      LOGICAL CTINIT,TIRINIT,GOLEMINIT
      DATA CTINIT,TIRINIT,GOLEMINIT/.TRUE.,.TRUE.,.TRUE./
      COMMON/REDUCTIONCODEINIT/CTINIT, TIRINIT, GOLEMINIT

      CHARACTER(512) MLPATH
      DATA MLPATH/'[[NA]]'/
      COMMON/MLPATH/MLPATH

      INTEGER I

C     Just a dummy call for LD to pick up this function
C     when creating the BLHA2 dynamic library
      DUMMY = ' '
      CALL SETPARA2(DUMMY)

      IF (LEN(PATH).GE.4 .AND. PATH(1:4).EQ.'auto') THEN
        IF (MLPATH(1:6).EQ.'[[NA]]') THEN
C         Try to automatically find the path
          PREFIX='./'
          CALL JOINPATH(PREFIX,NAMETOCHECK,FPATH)
          OPEN(1, FILE=FPATH, ERR=1, STATUS='OLD',ACTION='READ')
          MLPATH=PREFIX
          GOTO 10
 1        CONTINUE
          CLOSE(1)
          PREFIX='./MadLoop5_resources/'
          CALL JOINPATH(PREFIX,NAMETOCHECK,FPATH)
          OPEN(1, FILE=FPATH, ERR=2, STATUS='OLD',ACTION='READ')
          MLPATH=PREFIX
          GOTO 10
 2        CONTINUE
          CLOSE(1)
          PREFIX='../MadLoop5_resources/'
          CALL JOINPATH(PREFIX,NAMETOCHECK,FPATH)
          OPEN(1, FILE=FPATH, ERR=66, STATUS='OLD',ACTION='READ')
          MLPATH=PREFIX
          GOTO 10
 66       CONTINUE
          CLOSE(1)
C         We could not automatically find the auxiliary files
          WRITE(*,*) '==='
          WRITE(*,*) 'ERROR: MadLoop5 could not automatically find th'
     $     //'e file MadLoopParams.dat.'
          WRITE(*,*) '==='
          WRITE(*,*) '(Try using <CALL setMadLoopPath(/my/pat'
     $     //'h)> (before your first call to MadLoop) in order to se'
     $     //'t the directory where this file is located as well as'
     $     //'  other auxiliary files, such as <xxx>_ColorNumFactors.d'
     $     //'at, <xxx>_ColorDenomFactors.dat, etc..)'
          STOP
 10       CONTINUE
          CLOSE(1)
          RETURN
        ENDIF
      ELSE
C       Use the one specified by the user
C       Make sure there is a separator added
        I =1
        DO WHILE (I.LE.LEN(PATH) .AND. PATH(I:I).NE.' ')
          I=I+1
        ENDDO
        IF (PATH(I-1:I-1).NE.'/') THEN
          PATH(I:I) = '/'
        ENDIF
        MLPATH=PATH
      ENDIF

C     Check that the FilePath set is correct
      CALL JOINPATH(MLPATH,NAMETOCHECK,FPATH)
      OPEN(1, FILE=FPATH, ERR=3, STATUS='OLD',ACTION='READ')
      GOTO 11
 3    CONTINUE
      CLOSE(1)
      WRITE(*,*) '==='
      WRITE(*,*) 'ERROR: The MadLoop5 auxiliary files could not b'
     $ //'e found in ',MLPATH
      WRITE(*,*) '==='
      STOP
 11   CONTINUE
      CLOSE(1)

      END

      INTEGER FUNCTION SET_RET_CODE_U(MLRED,DOING_QP,STABLE)
C     
C     This functions returns the value of U
C     
C     
C     U == 0
C     Not stable.
C     U == 1
C     Stable with CutTools in double precision.
C     U == 2
C     Stable with PJFry++.
C     U == 3
C     Stable with IREGI.
C     U == 4
C     Stable with Golem95
C     U == 9
C     Stable with CutTools in quadruple precision.
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER MLRED
      LOGICAL DOING_QP,STABLE
C     
C     LOCAL VARIABLES
C     
C     
C     FUNCTION
C     
C     
C     BEGIN CODE
C     
      IF(.NOT.STABLE)THEN
        SET_RET_CODE_U=0
        RETURN
      ENDIF
      IF(DOING_QP)THEN
        IF(MLRED.EQ.1)THEN
          SET_RET_CODE_U=9
          RETURN
        ELSE
          STOP 'Only CutTools can use quardruple precision'
        ENDIF
      ENDIF
      IF(MLRED.GE.1.AND.MLRED.LE.4)THEN
        SET_RET_CODE_U=MLRED
      ELSE
        STOP 'Only CutTools,PJFry++,IREGI,Golem95 are available'
      ENDIF
      END

      SUBROUTINE DETECT_LOOPLIB(LIBNUM,NLOOPLINE,RANK,COMPLEX_MASS
     $ ,LPASS)
C     
C     DETECT WHICH LOOP LIB PASSED
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER LIBNUM,NLOOPLINE,RANK
      LOGICAL COMPLEX_MASS,LPASS
C     
C     LOCAL VARIABLES
C     
C     
C     GLOBAL VARIABLES
C     
C     ----------
C     BEGIN CODE
C     ----------
      IF(LIBNUM.EQ.1)THEN
C       CutTools
        CALL DETECT_CUTTOOLS(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
      ELSEIF(LIBNUM.EQ.2)THEN
C       PJFry++
        CALL DETECT_PJFRY(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
      ELSEIF(LIBNUM.EQ.3)THEN
C       IREGI
        CALL DETECT_IREGI(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
      ELSEIF(LIBNUM.EQ.4)THEN
C       Golem95
        CALL DETECT_GOLEM(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
      ELSE
        STOP 'ONLY CUTTOOLS,PJFry++,IREGI,Golem95 are provided'
      ENDIF
      RETURN
      END

      SUBROUTINE DETECT_CUTTOOLS(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
C     
C     DETECT THE CUTTOOLS CAN BE USED OR NOT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE,RANK
      LOGICAL COMPLEX_MASS,LPASS
C     
C     LOCAL VARIABLES
C     
C     
C     GLOBAL VARIABLES
C     
C     ----------
C     BEGIN CODE
C     ----------
      LPASS=.TRUE.
      IF(NLOOPLINE+1.LT.RANK)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE DETECT_PJFRY(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
C     
C     DETECT THE PJFRY++ CAN BE USED OR NOT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE,RANK
      LOGICAL COMPLEX_MASS,LPASS
C     
C     LOCAL VARIABLES
C     
C     
C     GLOBAL VARIABLES
C     
C     ----------
C     BEGIN CODE
C     ----------
      LPASS=.TRUE.
      IF(NLOOPLINE.LT.RANK.OR.RANK.GT.5.OR.NLOOPLINE.GT.5.OR.COMPLEX_MA
     $ SS.OR.NLOOPLINE.EQ.1) THEN
        LPASS=.FALSE.
      ENDIF
      RETURN
      END

      SUBROUTINE DETECT_IREGI(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
C     
C     DETECT THE IREGI CAN BE USED OR NOT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE,RANK
      LOGICAL COMPLEX_MASS,LPASS
C     
C     LOCAL VARIABLES
C     
C     
C     GLOBAL VARIABLES
C     
C     ----------
C     BEGIN CODE
C     ----------
      LPASS=.TRUE.
      IF(NLOOPLINE.GE.7.OR.RANK.GE.7)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE DETECT_GOLEM(NLOOPLINE,RANK,COMPLEX_MASS,LPASS)
C     
C     DETECT THE Golem95 CAN BE USED OR NOT
C     
      IMPLICIT NONE
C     
C     CONSTANTS
C     
C     
C     ARGUMENTS
C     
      INTEGER NLOOPLINE,RANK
      LOGICAL COMPLEX_MASS,LPASS
C     
C     LOCAL VARIABLES
C     
C     
C     GLOBAL VARIABLES
C     
C     ----------
C     BEGIN CODE
C     ----------

      LPASS=.TRUE.
      IF(NLOOPLINE.GE.7.OR.RANK.GE.7.OR.NLOOPLINE.LE.1)LPASS=.FALSE.
      IF(NLOOPLINE.LE.5.AND.RANK.GT.NLOOPLINE+1)LPASS=.FALSE.
      IF(NLOOPLINE.EQ.6.AND.RANK.GT.NLOOPLINE)LPASS=.FALSE.
      RETURN
      END

      SUBROUTINE PRINT_MADLOOP_BANNER()

      WRITE(*,*) ' ==================================================='
     $ //'======================================= '
      WRITE(*,*) '{                                                  '
     $ //'                                        }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'                     '
     $ //'                                                       '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'                     '
     $ //'          ,,                                           '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'`7MMM.     ,MMF'/
     $ /CHAR(39)//'             `7MM  `7MMF'//CHAR(39)//'            '
     $ //'                       '//CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'  MMMb    dPMM       '
     $ //'          MM    MM                                     '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'  M YM   ,M MM   ,6'/
     $ /CHAR(34)//'Yb.   ,M'//CHAR(34)//''//CHAR(34)//'bMM    MM     '
     $ //'    ,pW'//CHAR(34)//'Wq.   ,pW'//CHAR(34)//'Wq.`7MMpdMAo. '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'  M  Mb  M'//CHAR(39)/
     $ /' MM  8)   MM ,AP    MM    MM        6W'//CHAR(39)//'   `W'
     $ //'b 6W'//CHAR(39)//'   `Wb MM   `Wb '//CHAR(27)//'[0m'/
     $ /'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'  M  YM.P'//CHAR(39)/
     $ /'  MM   ,pm9MM 8MI    MM    MM      , 8M     M8 8M     M8 MM '
     $ //'   M8 '//CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'  M  `YM'//CHAR(39)/
     $ /'   MM  8M   MM `Mb    MM    MM     ,M YA.   ,A9 YA.  '
     $ //' ,A9 MM   ,AP '//CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'.JML. `'//CHAR(39)/
     $ /'  .JMML.`Moo9^Yo.`Wbmd'//CHAR(34)//'MML..JMMmmmmMMM  `Ybmd9'/
     $ /CHAR(39)//'   `Ybmd9'//CHAR(39)//'  MMbmmd'//CHAR(39)//'  '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'                     '
     $ //'                                              MM       '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'                     '
     $ //'                                            .JMML.     '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//CHAR(27)//'[0m'/
     $ /'v2.3.0 (2015-02-05), Ref: arXiv:1103.0621v2, arXiv:1405.0301'
     $ //CHAR(27)//'[32m'//'                '//CHAR(27)//'[0m'/
     $ /'       }'
      WRITE(*,*) '{       '//CHAR(27)//'[32m'//'                     '
     $ //'                                                       '/
     $ /CHAR(27)//'[0m'//'       }'
      WRITE(*,*) '{                                                  '
     $ //'                                        }'
      WRITE(*,*) ' ==================================================='
     $ //'======================================= '

      END

