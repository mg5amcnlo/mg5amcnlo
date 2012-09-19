c Derived from mcatnlo_hwanhtt2.scalar.f
c Relevant to bb(H->bb) production
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      REAL*4 pi
      parameter (pi=3.14160E0)
      integer k
      character*5 cc(5)
      data cc/'     ','cut1 ','cut2 ','cut3 ','cut4 '/
c
c b[1] -> not coming from Higgs decay
c b[2] -> coming from Higgs decay
c [x,1] -> hardest
c [x,2] -> next-to-hardest
      call inihist
      do i=1,2
         k=(i-1)*50
         call mbook(k+ 1,'total rate '//cc(i),1.e0,0.5e0,5.5e0)
         call mbook(k+ 2,'pt Higgs   '//cc(i),10.e0,0.e0,1000.e0)
         call mbook(k+ 3,'log[pt] bbH'//cc(i),0.05e0,0.0e0,5.e0)
         call mbook(k+ 4,'bb[1] pt   '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 5,'bb[1] dphi '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+ 6,'bb[1] m    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 7,'bb[1] DelR '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+ 8,'bb[2] pt   '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 9,'bb[2] dphi '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+10,'bb[2] m    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+11,'bb[2] DelR '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+12,'# of B[1]  '//cc(i),1.e0,-0.5e0,10.5e0)
         call mbook(k+13,'# of B[2]  '//cc(i),1.e0,-0.5e0,10.5e0)
         call mbook(k+14,'b[1,1] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+15,'b[1,1] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+16,'b[1,2] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+17,'b[1,2] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+18,'b[2,1] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+19,'b[2,1] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+20,'b[2,2] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+21,'b[2,2] eta '//cc(i),0.2e0,-8.e0,8.e0)
c
         k=(i-1)*50+25
         call mbook(k+ 3,'log[pt] jjH'//cc(i),0.05e0,0.0e0,5.e0)
         call mbook(k+ 4,'jj[1] pt   '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 5,'jj[1] dphi '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+ 6,'jj[1] m    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 7,'jj[1] DelR '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+ 8,'jj[2] pt   '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 9,'jj[2] dphi '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+10,'jj[2] m    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+11,'jj[2] DelR '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+12,'# of j[1]  '//cc(i),1.e0,-0.5e0,10.5e0)
         call mbook(k+13,'# of j[2]  '//cc(i),1.e0,-0.5e0,10.5e0)
         call mbook(k+14,'j[1,1] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+15,'j[1,1] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+16,'j[1,2] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+17,'j[1,2] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+18,'j[2,1] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+19,'j[2,1] eta '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+20,'j[2,2] pt  '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+21,'j[2,2] eta '//cc(i),0.2e0,-8.e0,8.e0)
      enddo
      END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,K
      OPEN(UNIT=99,NAME='HERHBB.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
c
      do i=1,2
        k=(i-1)*50
        call multitop(100+k+ 1,99,3,2,'total rate ',' ','LOG')
        call multitop(100+k+ 2,99,3,2,'pt Higgs   ',' ','LOG')
        call multitop(100+k+ 3,99,3,2,'log[pt] bbH',' ','LOG')
        call multitop(100+k+ 4,99,3,2,'bb[1] pt   ',' ','LOG')
        call multitop(100+k+ 5,99,3,2,'bb[1] dphi ',' ','LOG')
        call multitop(100+k+ 6,99,3,2,'bb[1] m    ',' ','LOG')
        call multitop(100+k+ 7,99,3,2,'bb[1] DelR ',' ','LOG')
        call multitop(100+k+ 8,99,3,2,'bb[2] pt   ',' ','LOG')
        call multitop(100+k+ 9,99,3,2,'bb[2] dphi ',' ','LOG')
        call multitop(100+k+10,99,3,2,'bb[2] m    ',' ','LOG')
        call multitop(100+k+11,99,3,2,'bb[2] DelR ',' ','LOG')
        call multitop(100+k+12,99,3,2,'# of B[1]  ',' ','LOG')
        call multitop(100+k+13,99,3,2,'# of B[2]  ',' ','LOG')
        call multitop(100+k+14,99,3,2,'b[1,1] pt  ',' ','LOG')
        call multitop(100+k+15,99,3,2,'b[1,1] eta ',' ','LOG')
        call multitop(100+k+16,99,3,2,'b[1,2] pt  ',' ','LOG')
        call multitop(100+k+17,99,3,2,'b[1,2] eta ',' ','LOG')
        call multitop(100+k+18,99,3,2,'b[2,1] pt  ',' ','LOG')
        call multitop(100+k+19,99,3,2,'b[2,1] eta ',' ','LOG')
        call multitop(100+k+20,99,3,2,'b[2,2] pt  ',' ','LOG')
        call multitop(100+k+21,99,3,2,'b[2,2] eta ',' ','LOG')
c
        k=(i-1)*50+25
        call multitop(100+k+ 3,99,3,2,'log[pt] jjH',' ','LOG')
        call multitop(100+k+ 4,99,3,2,'jj[1] pt   ',' ','LOG')
        call multitop(100+k+ 5,99,3,2,'jj[1] dphi ',' ','LOG')
        call multitop(100+k+ 6,99,3,2,'jj[1] m    ',' ','LOG')
        call multitop(100+k+ 7,99,3,2,'jj[1] DelR ',' ','LOG')
        call multitop(100+k+ 8,99,3,2,'jj[2] pt   ',' ','LOG')
        call multitop(100+k+ 9,99,3,2,'jj[2] dphi ',' ','LOG')
        call multitop(100+k+10,99,3,2,'jj[2] m    ',' ','LOG')
        call multitop(100+k+11,99,3,2,'jj[2] DelR ',' ','LOG')
        call multitop(100+k+12,99,3,2,'# of j[1]  ',' ','LOG')
        call multitop(100+k+13,99,3,2,'# of j[2]  ',' ','LOG')
        call multitop(100+k+14,99,3,2,'j[1,1] pt  ',' ','LOG')
        call multitop(100+k+15,99,3,2,'j[1,1] eta ',' ','LOG')
        call multitop(100+k+16,99,3,2,'j[1,2] pt  ',' ','LOG')
        call multitop(100+k+17,99,3,2,'j[1,2] eta ',' ','LOG')
        call multitop(100+k+18,99,3,2,'j[2,1] pt  ',' ','LOG')
        call multitop(100+k+19,99,3,2,'j[2,1] eta ',' ','LOG')
        call multitop(100+k+20,99,3,2,'j[2,2] pt  ',' ','LOG')
        call multitop(100+k+21,99,3,2,'j[2,2] eta ',' ','LOG')
      enddo
c
      CLOSE(99)
      END


C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL DIDSOF
      LOGICAL BHADRN,BMESON,BBARYON
      INTEGER IQH,IBHAD,IDHIGGS,IST,ID,IH1,IJ,I,J,IBFROMH,I1,
     # IMOTHER,KBMAX1,KBMAX2,IBNOTH,kk,JBHAD(30),KBFROMH(30),
     # JBFROMH(30),KBNOTH(30),JBNOTH(30),JBMAX(30)
      DOUBLE PRECISION PTB1MAX,PTB2MAX,PTB,PTCALC,pt3,pt4,ypt,ydphi,
     # getdelphi,ym,getinvm,ydelr,getdr,ypt11,ypt12,yeta11,yeta12,
     # getpseudorap,zpt,zdphi,zm,zdelr,zpt11,zpt12,zeta11,zeta12,
     # PBHAD(4,30),XPHQ(4),YPB1(4),YPB2(4),YPBB(4),ZPB1(4),ZPB2(4),
     # ZPBB(4),xpbbh(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
c
      IF (IERROR.NE.0) RETURN
c
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
      WWW0=EVWGT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
      IQH=0
      IBHAD=0
      IDHIGGS=25
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHEP(IHEP)
        CALL BHAD(ID,BHADRN,BMESON,BBARYON)
c Assume the relevant B-hadrons have been set stable in the driver
        IF(IST.EQ.195.AND.ID.EQ.IDHIGGS)THEN
C FOUND A HIGGS; KEEP ONLY THE FIRST ON RECORD
          IQH=IQH+1
          IF(IQH.EQ.1)IH1=IHEP
        ENDIF
        IF(BHADRN.AND.IST.EQ.1)THEN
          IBHAD=IBHAD+1
          JBHAD(IBHAD)=IHEP
          DO IJ=1,4
            PBHAD(IJ,IBHAD)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
  100 CONTINUE
      IF(IQH.NE.1)THEN
        CALL HWUEPR
        WRITE(*,*)IQH
        CALL HWWARN('HWANAL',500)
        GOTO 999
      ENDIF
      IF(IBHAD.LT.4)THEN
        CALL HWUEPR
        WRITE(*,*)IBHAD
        DO I=1,IBHAD
          WRITE(*,*)JBHAD(I)
        ENDDO
        CALL HWWARN('HWANAL',501)
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
C FILL THE FOUR-MOMENTA
      DO IJ=1,4
        XPHQ(IJ)=PHEP(IJ,IH1)
      ENDDO
c Find the B from Higgs
      IBFROMH=0
      DO I=1,IBHAD
        KBFROMH(I)=0
        IHEP=JMOHEP(1,JBHAD(I))
c find mother cluster first
        I1=0
        DOWHILE(IDHEP(IHEP).NE.91)
          IHEP=JMOHEP(1,IHEP)
          I1=I1+1
          IF(I1.GT.30)THEN
            CALL HWUEPR
            WRITE(*,*)'IHEP,I1=',IHEP,I1
            CALL HWWARN('HWANAL',514)
          ENDIF
        ENDDO
c find b quark in mother cluser
        IMOTHER=IHEP
        IHEP=JMOHEP(1,IMOTHER)
        IF(ABS(IDHEP(IHEP)).NE.5)IHEP=JMOHEP(2,IMOTHER)
        IF(ABS(IDHEP(IHEP)).NE.5)THEN
          CALL HWUEPR
          CALL HWWARN('HWANAL',522)
        ENDIF
        IHEP=JMOHEP(1,IHEP)
        I1=0
        DOWHILE(IDHEP(IHEP).NE.IDHIGGS.AND.IHEP.NE.6)
          IHEP=JMOHEP(1,IHEP)
          I1=I1+1
          IF(I1.GT.30)THEN
            CALL HWUEPR
            WRITE(*,*)'IHEP,I1=',IHEP,I1
            CALL HWWARN('HWANAL',515)
          ENDIF
        ENDDO
        IF(IDHEP(IHEP).EQ.IDHIGGS)THEN
          IF(IHEP.NE.IH1)THEN
            CALL HWUEPR
            WRITE(*,*)'IHEP,I1=',IHEP,I1
            CALL HWWARN('HWANAL',516)
          ENDIF
          IBFROMH=IBFROMH+1
          JBFROMH(IBFROMH)=I
          KBFROMH(I)=IBFROMH
        ENDIF
      ENDDO
      IF(IBFROMH.LT.2)THEN
        CALL HWUEPR
        WRITE(*,*)'FOUND:',IBFROMH
        DO I=1,IBFROMH
          WRITE(*,*)'In position:',JBHAD(JBFROMH(I))
        ENDDO
        CALL HWWARN('HWANAL',520)
      ENDIF
C If more than two found, keep the two hardest
      IF(IBFROMH.GT.2)THEN
        PTB1MAX=0.d0
        PTB2MAX=0.d0
        KBMAX1=0
        KBMAX2=0
        DO I=1,IBFROMH
          PTB=PTCALC(PBHAD(1,JBFROMH(I)))
          IF(PTB.GT.PTB1MAX)THEN
            KBMAX2=KBMAX1
            PTB2MAX=PTB1MAX
            KBMAX1=JBFROMH(I)
            PTB1MAX=PTB
          ELSEIF(PTB.GT.PTB2MAX)THEN
            KBMAX2=JBFROMH(I)
            PTB2MAX=PTB
          ENDIF
        ENDDO
        JBFROMH(1)=KBMAX1
        JBFROMH(2)=KBMAX2
      ENDIF
      IF(JBFROMH(1).EQ.0.OR.JBFROMH(2).EQ.0)THEN
        CALL HWUEPR
        DO I=1,IBFROMH
          WRITE(*,*)'Bs from Higgs:',JBFROMH(I),JBHAD(JBFROMH(I))
        ENDDO
        CALL HWWARN('HWANAL',530)
      ENDIF
c Tag the B not from Higgs
      IBNOTH=0
      DO I=1,IBHAD
        KBNOTH(I)=0
        IF(KBFROMH(I).EQ.0)THEN
          IBNOTH=IBNOTH+1
          JBNOTH(IBNOTH)=I
          KBNOTH(I)=IBNOTH
        ENDIF
      ENDDO
c Sanity checks
      IF(IBNOTH.LT.2)THEN
        CALL HWUEPR
        WRITE(*,*)'FOUND:',IBNOTH
        DO I=1,IBNOTH
          WRITE(*,*)'In position:',JBHAD(JBNOTH(I))
        ENDDO
        CALL HWWARN('HWANAL',560)
      ENDIF
      IF((IBNOTH+IBFROMH).NE.IBHAD)THEN
        CALL HWUEPR
        WRITE(*,*)IBHAD,IBNOTH,IBFROMH
        CALL HWWARN('HWANAL',502)
      ENDIF
      DO I=1,IBHAD
        IF( (KBFROMH(I).EQ.0.AND.KBNOTH(I).EQ.0) .OR.
     #      (KBFROMH(I).NE.0.AND.KBNOTH(I).NE.0) .OR.
     #      (KBFROMH(I).NE.0.AND.JBFROMH(KBFROMH(I)).NE.I) .OR.
     #      (KBNOTH(I).NE.0.AND.JBNOTH(KBNOTH(I)).NE.I) )THEN
          CALL HWUEPR
          WRITE(*,*)IBHAD,I,KBFROMH(I),KBNOTH(I)
          DO J=1,IBHAD
            WRITE(*,*)JBHAD(J)
          ENDDO
          CALL HWWARN('HWANAL',503)
        ENDIF
      ENDDO
c Find the two hardest B mesons not from the Higgs
      IF(IBNOTH.GT.2)THEN
        PTB1MAX=0.d0
        PTB2MAX=0.d0
        JBMAX(1)=0
        JBMAX(2)=0
        DO I=1,IBNOTH
          PTB=PTCALC(PBHAD(1,JBNOTH(I)))
          IF(PTB.GT.PTB1MAX)THEN
            JBMAX(2)=JBMAX(1)
            PTB2MAX=PTB1MAX
            JBMAX(1)=JBNOTH(I)
            PTB1MAX=PTB
          ELSEIF(PTB.GT.PTB2MAX)THEN
            JBMAX(2)=JBNOTH(I)
            PTB2MAX=PTB
          ENDIF
        ENDDO
        JBNOTH(1)=JBMAX(1)
        JBNOTH(2)=JBMAX(2)
      ENDIF
      IF(JBNOTH(1).EQ.0.OR.JBNOTH(2).EQ.0)THEN
        CALL HWUEPR
        DO I=1,IBNOTH
          WRITE(*,*)'Bs from Higgs:',JBNOTH(I),JBHAD(JBNOTH(I))
        ENDDO
        CALL HWWARN('HWANAL',561)
      ENDIF
c
c$$$      WRITE(*,*)'Hardest:',JBHAD(JBNOTH(1)),JBHAD(JBNOTH(2))
c$$$      WRITE(*,*)'From Higgs:',JBHAD(JBFROMH(1)),JBHAD(JBFROMH(2))
      DO IJ=1,4
        YPB1(IJ)=PBHAD(IJ,JBNOTH(1))
        YPB2(IJ)=PBHAD(IJ,JBNOTH(2))
        YPBB(IJ)=YPB1(IJ)+YPB2(IJ)
        ZPB1(IJ)=PBHAD(IJ,JBFROMH(1))
        ZPB2(IJ)=PBHAD(IJ,JBFROMH(2))
        ZPBB(IJ)=ZPB1(IJ)+ZPB2(IJ)
        xpbbh(ij)=YPB1(IJ)+YPB2(IJ)+XPHQ(IJ)
      ENDDO
c
      pt3=sqrt( xphq(1)**2+ xphq(2)**2)
      pt4=sqrt( xpbbh(1)**2+ xpbbh(2)**2)
c
      ypt=ptcalc(YPBB)
      ydphi=getdelphi(YPB1(1),YPB1(2),
     #                YPB2(1),YPB2(2))
      ym=getinvm(YPBB(4),YPBB(1),YPBB(2),YPBB(3))
      ydelr=getdr(YPB1(4),YPB1(1),YPB1(2),YPB1(3),
     #            YPB2(4),YPB2(1),YPB2(2),YPB2(3))
      ypt11=ptcalc(YPB1)
      ypt12=ptcalc(YPB2)
      yeta11=getpseudorap(YPB1(4),YPB1(1),YPB1(2),YPB1(3))
      yeta12=getpseudorap(YPB2(4),YPB2(1),YPB2(2),YPB2(3))
c
      zpt=ptcalc(ZPBB)
      zdphi=getdelphi(ZPB1(1),ZPB1(2),
     #                ZPB2(1),ZPB2(2))
      zm=getinvm(ZPBB(4),ZPBB(1),ZPBB(2),ZPBB(3))
      zdelr=getdr(ZPB1(4),ZPB1(1),ZPB1(2),ZPB1(3),
     #            ZPB2(4),ZPB2(1),ZPB2(2),ZPB2(3))
      zpt11=ptcalc(ZPB1)
      zpt12=ptcalc(ZPB2)
      zeta11=getpseudorap(ZPB1(4),ZPB1(1),ZPB1(2),ZPB1(3))
      zeta12=getpseudorap(ZPB2(4),ZPB2(1),ZPB2(2),ZPB2(3))
c
      if(zm.gt.(rmass(201)+10.d0))then
        CALL HWUEPR
        write(*,*)IBHAD,IBFROMH
        write(*,*)JBFROMH(1),JBFROMH(2)
        write(*,*)JBHAD(JBFROMH(1)),JBHAD(JBFROMH(2))
        DO IJ=1,4
          write(*,*),'MOM',ZPB1(IJ),ZPB2(IJ),ZPBB(IJ)
        ENDDO
        CALL HWWARN('HWANAL',125)
        GOTO 999
      endif
C
      do i=1,2
        kk=(i-1)*50
        if (i.eq.2 .and. pt3.lt.100d0) cycle
        call mfill(kk+ 1,sngl(1d0),sngl(WWW0))
        call mfill(kk+ 2,sngl(pt3),sngl(WWW0))
        if (pt4.gt.1d0) call mfill(kk+ 3,sngl(log10(pt4)),sngl(WWW0))
        call mfill(kk+ 4,sngl(ypt),sngl(WWW0))
        call mfill(kk+ 5,sngl(ydphi),sngl(WWW0))
        call mfill(kk+ 6,sngl(ym),sngl(WWW0))
        call mfill(kk+ 7,sngl(ydelr),sngl(WWW0))
        call mfill(kk+ 8,sngl(zpt),sngl(WWW0))
        call mfill(kk+ 9,sngl(zdphi),sngl(WWW0))
        call mfill(kk+10,sngl(zm),sngl(WWW0))
        call mfill(kk+11,sngl(zdelr),sngl(WWW0))
        call mfill(kk+12,float(ibnoth),sngl(WWW0))
        call mfill(kk+13,float(ibfromh),sngl(WWW0))
        call mfill(kk+14,sngl(ypt11),sngl(WWW0))
        call mfill(kk+15,sngl(yeta11),sngl(WWW0))
        call mfill(kk+16,sngl(ypt12),sngl(WWW0))
        call mfill(kk+17,sngl(yeta12),sngl(WWW0))
        call mfill(kk+18,sngl(zpt11),sngl(WWW0))
        call mfill(kk+19,sngl(zeta11),sngl(WWW0))
        call mfill(kk+20,sngl(zpt12),sngl(WWW0))
        call mfill(kk+21,sngl(zeta12),sngl(WWW0))
      enddo
C
 999  RETURN
      END


      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny )then
          y=0.5d0*log( xplus/xminus )
        else
          y=sign(1.d0,pl)*1.d8
        endif
      else
        y=sign(1.d0,pl)*1.d8
      endif
      getrapidity=y
      return
      end


      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end


      function getinvm(en,ptx,pty,pl)
      implicit none
      real*8 getinvm,en,ptx,pty,pl,tiny,tmp
      parameter (tiny=1.d-5)
c
      tmp=en**2-ptx**2-pty**2-pl**2
      if(tmp.gt.0.d0)then
        tmp=sqrt(tmp)
      elseif(tmp.gt.-tiny)then
        tmp=0.d0
      else
        write(*,*)'Attempt to compute a negative mass'
        stop
      endif
      getinvm=tmp
      return
      end


      function getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      getdelphi=tmp
      return
      end


      function getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getpseudorap,getdelphi
c
      deta=getpseudorap(en1,ptx1,pty1,pl1)-
     #     getpseudorap(en2,ptx2,pty2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdr=sqrt(dphi**2+deta**2)
      return
      end


C----------------------------------------------------------------------
      SUBROUTINE BHAD(IDPDG,BHADRN,BMESON,BBARYON)
C     TEST FOR A B-FLAVOURED HADRON
C----------------------------------------------------------------------
      INTEGER IDPDG,ID,IM,IB
      LOGICAL BHADRN,BMESON,BBARYON
C
      ID=ABS(IDPDG)
      IM=MOD(ID/100,100)
      IB=MOD(ID/1000,100)
      BMESON=IM.EQ.5
      BBARYON=IB.EQ.5
      IF(BMESON.AND.BBARYON)CALL HWWARN('BHAD  ',500)
      BHADRN=BMESON.OR.BBARYON
 999  END


      FUNCTION IHADR(ID)
c Returns the PDG code of the heavier quark in the hadron of PDG code ID
      IMPLICIT NONE
      INTEGER IHADR,ID,ID1
C
      IF(ID.NE.0)THEN
        ID1=ABS(ID)
        IF(ID1.GT.10000)ID1=ID1-1000*INT(ID1/1000)
        IHADR=ID1/(10**INT(LOG10(DFLOAT(ID1))))
      ELSE
        IHADR=0
      ENDIF
      RETURN
      END


      FUNCTION PTCALC(P)
      IMPLICIT NONE
      DOUBLE PRECISION PTCALC,P(4),PTSQ
      PTSQ=P(1)**2+P(2)**2
      IF (PTSQ.EQ.0D0) THEN
         PTCALC=0D0
      ELSE
         PTCALC=SQRT(PTSQ)
      ENDIF
      END


