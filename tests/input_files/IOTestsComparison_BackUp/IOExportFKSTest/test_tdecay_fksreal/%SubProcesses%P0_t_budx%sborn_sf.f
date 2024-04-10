      SUBROUTINE SBORN_SF(P_BORN,M,N,WGT)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'coupl.inc'
      DOUBLE PRECISION P_BORN(0:3,NEXTERNAL-1), WGT
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO = 1)

C     return the color-linked borns if i_fks is a color octet, 
C     the charge-linked if it is a color singlet
      DOUBLE COMPLEX WGT_BORN(2,0:NSQAMPSO)
      DOUBLE PRECISION WGT_COL
      DOUBLE PRECISION CHARGEPROD
      INTEGER I,M,N
      INCLUDE 'orders.inc'
      COMPLEX*16 ANS_CNT(2, NSPLITORDERS)
      COMMON /C_BORN_CNT/ ANS_CNT
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      COMMON /C_KEEP_ORDER_CNT/ KEEP_ORDER_CNT
      DOUBLE PRECISION CHARGES_BORN(NEXTERNAL-1)
      COMMON /C_CHARGES_BORN/CHARGES_BORN
      LOGICAL NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      COMMON /C_NEED_LINKS/NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      DOUBLE PRECISION PMASS(NEXTERNAL), ZERO
      PARAMETER (ZERO=0D0)
      DOUBLE PRECISION AMP_SPLIT_SOFT(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_SOFT/AMP_SPLIT_SOFT

      CHARGEPROD = 0D0

      IF (NEED_COLOR_LINKS.AND.NEED_CHARGE_LINKS) THEN
        WRITE(*,*) 'ERROR IN SBORN_SF, both color and charged links'
     $   //' are needed'
        STOP
      ENDIF
C     check if need color or charge links, and include the gs/w**2
C      term here
      IF (NEED_COLOR_LINKS) THEN
C       link partons 1 and 1 
        IF (M.EQ.1 .AND. N.EQ.1) THEN
          CALL SB_SF_001(P_BORN,WGT_COL)
C         link partons 1 and 2 
        ELSEIF ((M.EQ.1 .AND. N.EQ.2).OR.(M.EQ.2 .AND. N.EQ.1)) THEN
          CALL SB_SF_002(P_BORN,WGT_COL)
C         link partons 1 and 3 
        ELSEIF ((M.EQ.1 .AND. N.EQ.3).OR.(M.EQ.3 .AND. N.EQ.1)) THEN
          CALL SB_SF_003(P_BORN,WGT_COL)
C         link partons 1 and 4 
        ELSEIF ((M.EQ.1 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.1)) THEN
          CALL SB_SF_004(P_BORN,WGT_COL)
C         link partons 2 and 2 
        ELSEIF (M.EQ.2 .AND. N.EQ.2) THEN
          CALL SB_SF_005(P_BORN,WGT_COL)
C         link partons 2 and 3 
        ELSEIF ((M.EQ.2 .AND. N.EQ.3).OR.(M.EQ.3 .AND. N.EQ.2)) THEN
          CALL SB_SF_006(P_BORN,WGT_COL)
C         link partons 2 and 4 
        ELSEIF ((M.EQ.2 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.2)) THEN
          CALL SB_SF_007(P_BORN,WGT_COL)
C         link partons 3 and 4 
        ELSEIF ((M.EQ.3 .AND. N.EQ.4).OR.(M.EQ.4 .AND. N.EQ.3)) THEN
          CALL SB_SF_008(P_BORN,WGT_COL)
        ENDIF

        WGT = WGT_COL * G**2
C       update the amp_split_soft, which is summed in sbornsoft
        AMP_SPLIT_SOFT(1:AMP_SPLIT_SIZE) =
     $    DBLE(AMP_SPLIT_CNT(1:AMP_SPLIT_SIZE,1,QCD_POS)) * G**2

      ELSE IF (NEED_CHARGE_LINKS) THEN
        CHARGEPROD = CHARGES_BORN(M) * CHARGES_BORN(N)
        IF ((M.LE.NINCOMING.AND.N.GT.NINCOMING) .OR.
     $    (N.LE.NINCOMING.AND.M.GT.NINCOMING)) CHARGEPROD = -
     $    CHARGEPROD
C       add a factor 1/2 for the self-eikonal soft link
        IF (M.EQ.N) CHARGEPROD = CHARGEPROD / 2D0
        WGT = DBLE(ANS_CNT(1, QED_POS)) * CHARGEPROD * DBLE(GAL(1))**2
C       update the amp_split_soft, which is summed in sbornsoft
        AMP_SPLIT_SOFT(1:AMP_SPLIT_SIZE) =
     $    DBLE(AMP_SPLIT_CNT(1:AMP_SPLIT_SIZE,1,QED_POS)) * CHARGEPROD
     $    * DBLE(GAL(1))**2
      ENDIF

      RETURN
      END

