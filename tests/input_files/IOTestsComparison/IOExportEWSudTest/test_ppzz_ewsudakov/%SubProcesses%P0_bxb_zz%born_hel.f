      subroutine sborn_hel(p,ans)
      use mc_born_types
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      double precision p(0:3,nexternal-1),ans
      type(BornRequest),save::request
      type(BornResult),save::result
      double precision helicities(36)
      common/c_born_hel/helicities
      double precision helicity_orders(1,36)
      common/c_born_hel_split/helicity_orders
      request%%helicities=.true.
      call mc_born_local(p,request,result)
      helicities=result%%helicities
      helicity_orders=result%%helicity_orders
      ans=sum(helicities)
      end
      SUBROUTINE PICKHELICITYMC(P,GOODHEL,HEL,IHEL_OUT,VOL)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'born_nhel.inc'
      DOUBLE PRECISION P(0:3, NEXTERNAL-1)
      INTEGER GOODHEL(MAX_BHEL),HEL(0:MAX_BHEL)
      INTEGER IHEL_OUT
      DOUBLE PRECISION VOL
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=1)
      DOUBLE PRECISION WGT_HEL(NSQAMPSO, MAX_BHEL)
      COMMON/C_BORN_HEL_SPLIT/WGT_HEL
      DOUBLE PRECISION SUM_HEL(NSQAMPSO)
      INTEGER I, IHEL
      INTEGER N_NONZERO_ORD
      DOUBLE PRECISION SUM_ALL
      DOUBLE PRECISION ACCUM, TARGET
      DOUBLE PRECISION BORN_WGT_RECOMP_DIRECT
      DOUBLE PRECISION RAN2
      CALL SBORN_HEL(P,BORN_WGT_RECOMP_DIRECT)
      N_NONZERO_ORD = 0
      SUM_ALL = 0D0
      DO I = 1, NSQAMPSO
        SUM_HEL(I) = 0D0
        DO IHEL = 1, HEL(0)
          IF (WGT_HEL(I, HEL(IHEL)).LT.0D0) THEN
            WRITE(*,*) 'Helicities from squared diagrams must be > 0  !'
            STOP 1
          ENDIF
          SUM_HEL(I)=SUM_HEL(I) + WGT_HEL(I, HEL(IHEL))      *DBLE(GOODHEL(IHEL))
        ENDDO
        IF (SUM_HEL(I).GT.0D0) THEN
          N_NONZERO_ORD = N_NONZERO_ORD + 1
          SUM_ALL = SUM_ALL + SUM_HEL(I)
        ENDIF
      ENDDO
      TARGET=RAN2()
      IHEL=1
      ACCUM=0D0
      DO I = 1, NSQAMPSO
        IF (SUM_HEL(I).EQ.0D0) CYCLE
        ACCUM=ACCUM+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)*DBLE(GOODHEL(IHEL))    /N_NONZERO_ORD
      ENDDO
      DO WHILE (ACCUM.LT.TARGET)
        IHEL=IHEL+1
        DO I = 1, NSQAMPSO
          IF (SUM_HEL(I).EQ.0D0) CYCLE
          ACCUM=ACCUM+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)      *DBLE(GOODHEL(IHEL))/N_NONZERO_ORD
        ENDDO
      ENDDO
      VOL=0D0
      DO I = 1, NSQAMPSO
        IF (SUM_HEL(I).EQ.0D0) CYCLE
        VOL=VOL+WGT_HEL(I,HEL(IHEL))/SUM_HEL(I)*DBLE(GOODHEL(IHEL))    /N_NONZERO_ORD
      ENDDO
      IHEL_OUT=IHEL
      RETURN
      END
