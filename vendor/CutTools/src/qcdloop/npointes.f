	program testff
	include 'ff.h'
*	call ffinit
*	lwrite = .TRUE.
	call form
	end
      subroutine FORM
      implicit DOUBLE PRECISION (A-H,O-Z)
      DOUBLE COMPLEX TADMU
      DOUBLE COMPLEX AA0,B0,B0PM,B1,B1PM,B2,C0,C1,C2,C3,D0,D1,D2,D3,D4
      COMMON PX(6),RM(4),DEL
      COMMON B0,B0PM,B1,B1PM,B2(2),C0,C1(2),C2(4),C3(6)
      COMMON D0,D1(3),D2(7),D3(13),D4(22)
      DEL=0.d0
      XM=105.6d0**2
      TADMU=AA0(XM,DEL)
      PRINT 1,TADMU
    1 FORMAT(1H ,4E20.13)
      PX(1)=-90000.d0**2
      RM(1)=105.6d0**2
      RM(2)=0.d0
      CALL NPOIN(2)
      PRINT 1,B0,B0PM,B1PM,B2(1)
      PX(1)=-105.6d0**2
      PX(2)=PX(1)
      PX(5)=-90000.d0**2
      RM(1)=80000.d0**2
      RM(2)=0.d0
      RM(3)=80000.d0**2
      CALL NPOIN(3)
      PRINT 1,C0,C1(1)
      PX(1)=-0.511d0**2
      PX(2)=PX(1)
      PX(3)=-105.6d0**2
      PX(4)=PX(3)
      PX(5)=-80000.d0**2
      PX(6)=20000.d0**2
      RM(1)=90000.d0**2
      RM(2)=0.511d0**2
      RM(3)=90000.d0**2
      RM(4)=105.6d0**2
      CALL NPOIN(4)
      PRINT 1,D0,D1(1)
      P12=-0.511d0**2
      P22=-105.6d0**2
      P1P2=0.5d0*(20000.d0**2-P12-P22)
      P10=40000.d0
      P20=-40000.d0
      DELE=2000.d0
      PM2=0.003d0**2
      BREM=ALIJ(P22,P12,P1P2,P20,P10,DELE,PM2)
      PRINT 1,BREM
      call ffexi
      STOP
C
C  THE FOLLOWING OUTPUT IS EXPECTED    *********************************
C 0.                   .9156199386460E+06
C-.3100623399361E+02 -.2054369006935E+03 -.5269961187649E-14 -.1218492318111E-08
C .7255214122160E-20  .6092365170853E-09 -.1033542556010E+02 -.6793071440282E+02
C-.2272160404013E-21  .1739461728623E-08  .3037109802926E-22 -.1286491875423E-08
C-.2462904245037E-31  .4952559280190E-18  .1165014267077E-31 -.4571732002934E-18
C-.8203023744960E-05
C
C  NUMBERS OF ORDER  E-20  ARE TO BE CONSIDERED ZERO.
C
C ******************************************************************************
C
      END
