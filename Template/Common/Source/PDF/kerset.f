          SUBROUTINE KERSET(ERCODE,LGFILE,LIMITM,LIMITR)
                    PARAMETER(KOUNTE  =  28)
          CHARACTER*6         ERCODE,   CODE(KOUNTE)
          LOGICAL             MFLAG,    RFLAG
          INTEGER             KNTM(KOUNTE),       KNTR(KOUNTE)
          DATA      LOGF      /  0  /
          DATA      CODE(1), KNTM(1), KNTR(1)  / 'C204.1', 100, 100 /
          DATA      CODE(2), KNTM(2), KNTR(2)  / 'C204.2', 100, 100 /
          DATA      CODE(3), KNTM(3), KNTR(3)  / 'C204.3', 100, 100 /
          DATA      CODE(4), KNTM(4), KNTR(4)  / 'C205.1', 100, 100 /
          DATA      CODE(5), KNTM(5), KNTR(5)  / 'C205.2', 100, 100 /
          DATA      CODE(6), KNTM(6), KNTR(6)  / 'C205.3', 100, 100 /
          DATA      CODE(7), KNTM(7), KNTR(7)  / 'C305.1', 100, 100 /
          DATA      CODE(8), KNTM(8), KNTR(8)  / 'C308.1', 100, 100 /
          DATA      CODE(9), KNTM(9), KNTR(9)  / 'C312.1', 100, 100 /
          DATA      CODE(10),KNTM(10),KNTR(10) / 'C313.1', 100, 100 /
          DATA      CODE(11),KNTM(11),KNTR(11) / 'C336.1', 100, 100 /
          DATA      CODE(12),KNTM(12),KNTR(12) / 'C337.1', 100, 100 /
          DATA      CODE(13),KNTM(13),KNTR(13) / 'C341.1', 100, 100 /
          DATA      CODE(14),KNTM(14),KNTR(14) / 'D103.1', 100, 100 /
          DATA      CODE(15),KNTM(15),KNTR(15) / 'D106.1', 100, 100 /
          DATA      CODE(16),KNTM(16),KNTR(16) / 'D209.1', 100, 100 /
          DATA      CODE(17),KNTM(17),KNTR(17) / 'D509.1', 100, 100 /
          DATA      CODE(18),KNTM(18),KNTR(18) / 'E100.1', 100, 100 /
          DATA      CODE(19),KNTM(19),KNTR(19) / 'E104.1', 100, 100 /
          DATA      CODE(20),KNTM(20),KNTR(20) / 'E105.1', 100, 100 /
          DATA      CODE(21),KNTM(21),KNTR(21) / 'E208.1', 100, 100 /
          DATA      CODE(22),KNTM(22),KNTR(22) / 'E208.2', 100, 100 /
          DATA      CODE(23),KNTM(23),KNTR(23) / 'F010.1', 100,   0 /
          DATA      CODE(24),KNTM(24),KNTR(24) / 'F011.1', 100,   0 /
          DATA      CODE(25),KNTM(25),KNTR(25) / 'F012.1', 100,   0 /
          DATA      CODE(26),KNTM(26),KNTR(26) / 'F406.1', 100,   0 /
          DATA      CODE(27),KNTM(27),KNTR(27) / 'G100.1', 100, 100 /
          DATA      CODE(28),KNTM(28),KNTR(28) / 'G100.2', 100, 100 /
          LOGF  =  LGFILE
          IF(ERCODE .EQ. ' ')  THEN
             L  =  0
          ELSE
             DO 10  L = 1, 6
                IF(ERCODE(1:L) .EQ. ERCODE)  GOTO 12
  10            CONTINUE
  12         CONTINUE
          ENDIF
          DO 14     I  =  1, KOUNTE
             IF(L .EQ. 0)  GOTO 13
             IF(CODE(I)(1:L) .NE. ERCODE(1:L))  GOTO 14
  13         KNTM(I)  =  LIMITM
             KNTR(I)  =  LIMITR
  14         CONTINUE
          RETURN
          ENTRY KERMTR(ERCODE,LOG,MFLAG,RFLAG)
          LOG  =  LOGF
          DO 20     I  =  1, KOUNTE
             IF(ERCODE .EQ. CODE(I))  GOTO 21
  20         CONTINUE
          WRITE(*,1000)  ERCODE
          CALL ABEND
          RETURN
  21      RFLAG  =  KNTR(I) .GE. 1
          IF(RFLAG  .AND.  (KNTR(I) .LT. 100))  KNTR(I)  =  KNTR(I) - 1
          MFLAG  =  KNTM(I) .GE. 1
          IF(MFLAG  .AND.  (KNTM(I) .LT. 100))  KNTM(I)  =  KNTM(I) - 1
          IF(.NOT. RFLAG)  THEN
             IF(LOGF .LT. 1)  THEN
                WRITE(*,1001)  CODE(I)
             ELSE
                WRITE(LOGF,1001)  CODE(I)
             ENDIF
          ENDIF
          IF(MFLAG .AND. RFLAG)  THEN
             IF(LOGF .LT. 1)  THEN
                WRITE(*,1002)  CODE(I)
             ELSE
                WRITE(LOGF,1002)  CODE(I)
             ENDIF
          ENDIF
          RETURN
1000      FORMAT(' KERNLIB LIBRARY ERROR. ' /
     +           ' ERROR CODE ',A6,' NOT RECOGNIZED BY KERMTR',
     +           ' ERROR MONITOR. RUN ABORTED.')
1001      FORMAT(/' ***** RUN TERMINATED BY CERN LIBRARY ERROR ',
     +           'CONDITION ',A6)
1002      FORMAT(/' ***** CERN LIBRARY ERROR CONDITION ',A6)
          END
