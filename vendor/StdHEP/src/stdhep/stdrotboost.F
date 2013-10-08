
      subroutine STDROTBOOST(THE,PHI,BEX,BEY,BEZ)
*
* modified version of LUROBO from JETSET to work with STDHEP common block
* Doug Wright, March 1994
*
* replaced P(I,J) with PHEP(J,I) J=1-4
* replaced V(I,J) with VHEP(J,I) J=1-4
*
* JETSET feature that is not supported: MSTU(33)=1 is used to set 
*        the vertex to zero before boosting
*
C...Purpose: to perform rotations and boosts.
      IMPLICIT doUBLE PRECISION(D)
*DMW      COMMON/LUJETS/N,K(4000,5),P(4000,5),V(4000,5)
*DMW      COMMON/LUDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
*DMW      save /LUJETS/,/LUDAT1/
#include "stdhep.inc"
#include "stdlun.inc"

      DIMENSION ROT(3,3),PR(3),VR(3),DP(4),DV(4)

C...Find range of rotation/boost. Convert boost to double precision.
      IMIN=1
*DMW      if(MSTU(1).GT.0) IMIN=MSTU(1)
      IMAX=NHEP !DMW N
*DMW      if(MSTU(2).GT.0) IMAX=MSTU(2)
      DBX=BEX
      DBY=BEY
      DBZ=BEZ
      GOTO 110

C...Entry for specific range and double precision boost.
      ENTRY STDDBROTB(IMI,IMA,THE,PHI,DBEX,DBEY,DBEZ) !DMW
      IMIN=IMI
      if(IMIN.LE.0) IMIN=1
      IMAX=IMA
      if(IMAX.LE.0) IMAX=NHEP !DMW N
      DBX=DBEX
      DBY=DBEY
      DBZ=DBEZ

C...Optional resetting of V (when not set before.)
*DMW      if(MSTU(33).NE.0) then
*        do 100 I=MIN(IMIN,MSTU(4)),MIN(IMAX,MSTU(4))
*        do 100 J=1,5
*  100   V(I,J)=0.
*        MSTU(33)=0
*      endif

 110  continue !DMW
C...Check range of rotation/boost.
*DMW  110 if(IMIN.GT.MSTU(4).OR.IMAX.GT.MSTU(4)) then
*        call pyerrm(11,'(LUROBO:) range outside LUJETS memory')
*        return
*      endif

C...Rotate, typically from z axis to direction (theta,phi).
      if(THE**2+PHI**2.GT.1E-20) then
        ROT(1,1)=COS(THE)*COS(PHI)
        ROT(1,2)=-SIN(PHI)
        ROT(1,3)=SIN(THE)*COS(PHI)
        ROT(2,1)=COS(THE)*SIN(PHI)
        ROT(2,2)=COS(PHI)
        ROT(2,3)=SIN(THE)*SIN(PHI)
        ROT(3,1)=-SIN(THE)
        ROT(3,2)=0.
        ROT(3,3)=COS(THE)
        do 140 I=IMIN,IMAX
*DMW        if(K(I,1).LE.0) GOTO 140
        if(ISTHEP(I).LE.0) GOTO 140 !DMW
        do 120 J=1,3
        PR(J)=PHEP(J,I)
  120   VR(J)=VHEP(J,I)
        do 130 J=1,3
        PHEP(J,I)=ROT(J,1)*PR(1)+ROT(J,2)*PR(2)+ROT(J,3)*PR(3)
  130   VHEP(J,I)=ROT(J,1)*VR(1)+ROT(J,2)*VR(2)+ROT(J,3)*VR(3)
  140   CONTINUE
      endif

C...Boost, typically from rest to momentum/energy=beta.
      if(DBX**2+DBY**2+DBZ**2.GT.1E-20) then
        DB=SQRT(DBX**2+DBY**2+DBZ**2)
        if(DB.GT.0.99999999D0) then
C...Rescale boost vector if too close to unity.
*DMW          call pyerrm(3,'(LUROBO:) boost vector too large')
          write(lnhout,*) '(HEPROBO:) boost vector too large'
          DBX=DBX*(0.99999999D0/DB)
          DBY=DBY*(0.99999999D0/DB)
          DBZ=DBZ*(0.99999999D0/DB)
          DB=0.99999999D0
        endif
        DGA=1D0/SQRT(1D0-DB**2)
        do 160 I=IMIN,IMAX
*DMW        if(K(I,1).LE.0) GOTO 160
        if(ISTHEP(I).LE.0) GOTO 160 !DMW
        do 150 J=1,4
        DP(J)=PHEP(J,I)
  150   DV(J)=VHEP(J,I)
        DBP=DBX*DP(1)+DBY*DP(2)+DBZ*DP(3)
        DGABP=DGA*(DGA*DBP/(1D0+DGA)+DP(4))
        PHEP(1,I)=DP(1)+DGABP*DBX
        PHEP(2,I)=DP(2)+DGABP*DBY
        PHEP(3,I)=DP(3)+DGABP*DBZ
        PHEP(4,I)=DGA*(DP(4)+DBP)
        DBV=DBX*DV(1)+DBY*DV(2)+DBZ*DV(3)
        DGABV=DGA*(DGA*DBV/(1D0+DGA)+DV(4))
        VHEP(1,I)=DV(1)+DGABV*DBX
        VHEP(2,I)=DV(2)+DGABV*DBY
        VHEP(3,I)=DV(3)+DGABV*DBZ
        VHEP(4,I)=DGA*(DV(4)+DBV)
  160   CONTINUE
      endif

      return
      end
