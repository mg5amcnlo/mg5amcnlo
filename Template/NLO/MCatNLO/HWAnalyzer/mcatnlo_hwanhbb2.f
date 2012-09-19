c Derived from mcatnlo_hwanhbb.f
c Relevant to bb(H->tautau) production; assume the tau are stable
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
c x[1] -> hardest
c x[2] -> next-to-hardest
c l=tau
      call inihist
      do i=1,2
         k=(i-1)*50
         call mbook(k+ 1,'total rate '//cc(i),1.e0,0.5e0,5.5e0)
         call mbook(k+ 2,'pt Higgs   '//cc(i),10.e0,0.e0,1000.e0)
         call mbook(k+ 3,'log[pt] bbH'//cc(i),0.05e0,0.0e0,5.e0)
         call mbook(k+ 4,'bb pt      '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 5,'bb dphi    '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+ 6,'bb m       '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 7,'bb DelR    '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+ 8,'ll pt      '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+ 9,'ll dphi    '//cc(i),pi/20.e0,0.e0,pi)
         call mbook(k+10,'ll m       '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+11,'ll DelR    '//cc(i),pi/20.e0,0.e0,3*pi)
         call mbook(k+12,'# of B     '//cc(i),1.e0,-0.5e0,10.5e0)
         call mbook(k+13,'b[1] pt    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+14,'b[1] eta   '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+15,'b[2] pt    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+16,'b[2] eta   '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+17,'l[1] pt    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+18,'l[1] eta   '//cc(i),0.2e0,-8.e0,8.e0)
         call mbook(k+19,'l[2] pt    '//cc(i),2.e0,0.e0,200.e0)
         call mbook(k+20,'l[2] eta   '//cc(i),0.2e0,-8.e0,8.e0)
      enddo
      END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,K
      OPEN(UNIT=99,NAME='HERHBB2.TOP',STATUS='UNKNOWN')
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
        call multitop(100+k+ 4,99,3,2,'bb pt      ',' ','LOG')
        call multitop(100+k+ 5,99,3,2,'bb dphi    ',' ','LOG')
        call multitop(100+k+ 6,99,3,2,'bb m       ',' ','LOG')
        call multitop(100+k+ 7,99,3,2,'bb DelR    ',' ','LOG')
        call multitop(100+k+ 8,99,3,2,'ll pt      ',' ','LOG')
        call multitop(100+k+ 9,99,3,2,'ll dphi    ',' ','LOG')
        call multitop(100+k+10,99,3,2,'ll m       ',' ','LOG')
        call multitop(100+k+11,99,3,2,'ll DelR    ',' ','LOG')
        call multitop(100+k+12,99,3,2,'# of B     ',' ','LOG')
        call multitop(100+k+13,99,3,2,'b[1] pt    ',' ','LOG')
        call multitop(100+k+14,99,3,2,'b[1] eta   ',' ','LOG')
        call multitop(100+k+15,99,3,2,'b[2] pt    ',' ','LOG')
        call multitop(100+k+16,99,3,2,'b[2] eta   ',' ','LOG')
        call multitop(100+k+17,99,3,2,'l[1] pt    ',' ','LOG')
        call multitop(100+k+18,99,3,2,'l[1] eta   ',' ','LOG')
        call multitop(100+k+19,99,3,2,'l[2] pt    ',' ','LOG')
        call multitop(100+k+20,99,3,2,'l[2] eta   ',' ','LOG')
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
      INTEGER IQH,IBHAD,ILEP,IDHIGGS,IST,ID,IH1,IJ,I,
     # kk,JLEP(30),JLMX(30),JBHAD(30),JBMAX(30)
      DOUBLE PRECISION PTB1MAX,PTB2MAX,PTB,PTCALC,pt3,pt4,ypt,ydphi,
     # getdelphi,ym,getinvm,ydelr,getdr,ypt11,ypt12,yeta11,yeta12,
     # getpseudorap,zpt,zdphi,zm,zdelr,zpt11,zpt12,zeta11,zeta12,
     # PLEP(4,30),PBHAD(4,30),XPHQ(4),YPB1(4),YPB2(4),YPBB(4),
     # ZPL1(4),ZPL2(4),ZPLL(4),xpbbh(4)
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
      ILEP=0
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
        ELSEIF(IST.EQ.1.AND.ABS(ID).EQ.15)THEN
          ILEP=ILEP+1
          JLEP(ILEP)=IHEP
          DO IJ=1,4
            PLEP(IJ,ILEP)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
        IF(BHADRN.AND.IST.EQ.1)THEN
          IBHAD=IBHAD+1
          JBHAD(IBHAD)=IHEP
          DO IJ=1,4
            PBHAD(IJ,IBHAD)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
  100 CONTINUE
      IF(IQH.NE.1.OR.ILEP.LT.2)THEN
        CALL HWUEPR
        WRITE(*,*)IQH,ILEP
        CALL HWWARN('HWANAL',500)
      ENDIF
      IF(IBHAD.LT.2)THEN
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
c Find the two hardest B mesons
      JBMAX(1)=0
      JBMAX(2)=0
      PTB1MAX=0.d0
      PTB2MAX=0.d0
      DO I=1,IBHAD
        PTB=PTCALC(PBHAD(1,I))
        IF(PTB.GT.PTB1MAX)THEN
          JBMAX(2)=JBMAX(1)
          PTB2MAX=PTB1MAX
          JBMAX(1)=I
          PTB1MAX=PTB
        ELSEIF(PTB.GT.PTB2MAX)THEN
          JBMAX(2)=I
          PTB2MAX=PTB
        ENDIF
      ENDDO
      IF(JBMAX(1).EQ.0.OR.JBMAX(2).EQ.0)THEN
        CALL HWUEPR
        DO I=1,IBHAD
          WRITE(*,*)'B:',JBHAD(I),JBMAX(1),JBMAX(2)
        ENDDO
        CALL HWWARN('HWANAL',530)
      ENDIF
C Find the two hardest taus
      JLMX(1)=0
      JLMX(2)=0
      PTB1MAX=0.d0
      PTB2MAX=0.d0
      DO I=1,ILEP
        PTB=PTCALC(PLEP(1,I))
        IF(PTB.GT.PTB1MAX)THEN
          JLMX(2)=JLMX(1)
          PTB2MAX=PTB1MAX
          JLMX(1)=I
          PTB1MAX=PTB
        ELSEIF(PTB.GT.PTB2MAX)THEN
          JLMX(2)=I
          PTB2MAX=PTB
        ENDIF
      ENDDO
      IF(JLMX(1).EQ.0.OR.JLMX(2).EQ.0)THEN
        CALL HWUEPR
        DO I=1,ILEP
          WRITE(*,*)'B:',JLEP(I),JLMX(1),JLMX(2)
        ENDDO
        CALL HWWARN('HWANAL',531)
      ENDIF
C FILL THE FOUR-MOMENTA
      DO IJ=1,4
        XPHQ(IJ)=PHEP(IJ,IH1)
        YPB1(IJ)=PBHAD(IJ,JBMAX(1))
        YPB2(IJ)=PBHAD(IJ,JBMAX(2))
        YPBB(IJ)=YPB1(IJ)+YPB2(IJ)
        ZPL1(IJ)=PLEP(IJ,JLMX(1))
        ZPL2(IJ)=PLEP(IJ,JLMX(2))
        ZPLL(IJ)=ZPL1(IJ)+ZPL2(IJ)
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
      zpt=ptcalc(ZPLL)
      zdphi=getdelphi(ZPL1(1),ZPL1(2),
     #                ZPL2(1),ZPL2(2))
      zm=getinvm(ZPLL(4),ZPLL(1),ZPLL(2),ZPLL(3))
      zdelr=getdr(ZPL1(4),ZPL1(1),ZPL1(2),ZPL1(3),
     #            ZPL2(4),ZPL2(1),ZPL2(2),ZPL2(3))
      zpt11=ptcalc(ZPL1)
      zpt12=ptcalc(ZPL2)
      zeta11=getpseudorap(ZPL1(4),ZPL1(1),ZPL1(2),ZPL1(3))
      zeta12=getpseudorap(ZPL2(4),ZPL2(1),ZPL2(2),ZPL2(3))
c
      if(zm.gt.(rmass(201)+10.d0))then
c There may be other taus around, e.g. from g->ttbar and top decay.
c Hence, discard the event
        CALL HWUEPR
        write(*,*)ILEP
        write(*,*)JLMX(1),JLMX(2),JLEP(JLMX(1)),JLEP(JLMX(2))
        DO IJ=1,4
          write(*,*),'MOM',ZPL1(IJ),ZPL2(IJ),ZPLL(IJ)
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
        call mfill(kk+12,float(ibhad),sngl(WWW0))
        call mfill(kk+13,sngl(ypt11),sngl(WWW0))
        call mfill(kk+14,sngl(yeta11),sngl(WWW0))
        call mfill(kk+15,sngl(ypt12),sngl(WWW0))
        call mfill(kk+16,sngl(yeta12),sngl(WWW0))
        call mfill(kk+17,sngl(zpt11),sngl(WWW0))
        call mfill(kk+18,sngl(zeta11),sngl(WWW0))
        call mfill(kk+19,sngl(zpt12),sngl(WWW0))
        call mfill(kk+20,sngl(zeta12),sngl(WWW0))
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
        write(*,*)'Attempt to compute a negative mass',tmp
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


