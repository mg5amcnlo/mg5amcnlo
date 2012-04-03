      subroutine rans(r)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c                                                                c
c     Random number generator                                    c
c                                                                c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      implicit none
      real*8 r,rangen
      r = rangen(1)
      end

*-- Author :    F. James, modified by Mike Seymour
C-----------------------------------------------------------------------
      FUNCTION RANGEN(I)
C-----------------------------------------------------------------------
C     MAIN RANDOM NUMBER GENERATOR
C     USES METHOD OF l'Ecuyer, (VIA F.JAMES, COMP PHYS COMM 60(1990)329)
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION RANGEN,RANSET,RANGET
      INTEGER I,ISEED(2),K,IZ,JSEED(2)
      SAVE ISEED
      DATA ISEED/12345,67890/
      K=ISEED(1)/53668
      ISEED(1)=40014*(ISEED(1)-K*53668)-K*12211
      IF (ISEED(1).LT.0) ISEED(1)=ISEED(1)+2147483563
      K=ISEED(2)/52774
      ISEED(2)=40692*(ISEED(2)-K*52774)-K*3791
      IF (ISEED(2).LT.0) ISEED(2)=ISEED(2)+2147483399
      IZ=ISEED(1)-ISEED(2)
      IF (IZ.LT.1) IZ=IZ+2147483562
      RANGEN=DBLE(IZ)*4.656613001013252D-10
C--->                (4.656613001013252D-10 = 1.D0/2147483589)
      RETURN
C-----------------------------------------------------------------------
      ENTRY RANSET(JSEED)
C-----------------------------------------------------------------------
      IF (JSEED(1).EQ.0.OR.JSEED(2).EQ.0) then
         write(6,*) 'Jseeds=0, wrong settings for RANSET'
         stop
      endif
      ISEED(1)=JSEED(1)
      ISEED(2)=JSEED(2)
 999  RETURN
C-----------------------------------------------------------------------
      ENTRY RANGET(JSEED)
C-----------------------------------------------------------------------
      JSEED(1)=ISEED(1)
      JSEED(2)=ISEED(2)
      RETURN
      END

