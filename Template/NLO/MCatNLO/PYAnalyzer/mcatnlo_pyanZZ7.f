c Analysis for ZZ->eeee. Taken from mcatnlo_hwanZZ4.f. This file
c cannot be used for ZZ->mumumumu, but this case requires only trivial
c modifications
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      integer kchg
      double precision pmas,parf,vckm,zmass,gammax,gamz
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
c
      real * 8 xmlow,xmupp,xmbin
      real * 4 pi,xmls,xmus,xmbs
      parameter (pi=3.14160E0)
      integer i,j,k
      character*4 cc(2)
      data cc/'    ',' acc'/
      character*2 dd(3)
      data dd/' A',' B',' C'/
c
      ZMASS=PMAS(23,1)
      gamz=PMAS(23,2)
      gammax=30.d0
      xmlow=max(0.d0,zmass-gammax*gamz)
      xmupp=zmass+gammax*gamz
      xmbin=(xmupp-xmlow)/100.d0
      xmls=sngl(zmass-(49*xmbin+xmbin/2.d0))
      xmus=sngl(zmass+(50*xmbin+xmbin/2.d0))
      xmbs=sngl(xmbin)

c$$$      write(*,*)zmass,gammax*gamz,' zmass,gammax*gamz'
c$$$      write(*,*)xmbs,xmls,xmus,' xmbs,xmls,xmus'
c$$$      write(*,*)gammax,xmlow,xmupp,' gammax,xmlow,xmupp'
c$$$      write(*,*)xmbin,' xmbin'


c
      call inihist
c +- denotes unlike signs
c ++ denotes like signs
c ll+-[2] is a (e+,mu-) or (e-,mu+) pair in eemumu production. 
c   In eeee production, when no candidate Z's are found, ll+-[2]=ll+-;
c   otherwise, it is the unlike-sign pair which does not correspond
c   to a Z
      do j=1,2
c j=1 no cuts 
c j=2 lepton acceptance cuts
      do i=1,3
c i=1 no pair invariant mass contraint
c i=2 with pair invariant mass contraint
c i=3 flat decays
      k=(i-1)*32+(j-1)*100
      if(i.eq.1)
     #call mbook(k+32,'xsec         '//dd(i)//cc(j),1.e0,-0.5e0,9.5e0)
      call mbook(k+ 1,'l+ pt        '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 2,'l- pt        '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 3,'l+ eta       '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+ 4,'l- eta       '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+ 5,'ll+- pt      '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+ 6,'ll+- invM    '//dd(i)//cc(j),xmbs,xmls,xmus)
      call mbook(k+ 7,'ll+- eta     '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+ 8,'ll+- azimt   '//dd(i)//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+ 9,'ll+- Deta    '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+10,'ll+- DR      '//dd(i)//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+11,'ll+-[2] pt   '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+12,'ll+-[2] invM '//dd(i)//cc(j),5.e0,0.e0,500.e0)
      call mbook(k+13,'ll+-[2] eta  '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+14,'ll+-[2] azimt'//dd(i)//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+15,'ll+-[2] Deta '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+16,'ll+-[2] DR   '//dd(i)//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+17,'ll++ pt      '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+18,'ll++ invM    '//dd(i)//cc(j),5.e0,0.e0,500.e0)
      call mbook(k+19,'ll++ eta     '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+20,'ll++ azimt   '//dd(i)//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+21,'ll++ Deta    '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+22,'ll++ DR      '//dd(i)//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+23,'syst pt      '//dd(i)//cc(j),2.e0,0.e0,200.e0)
      call mbook(k+24,'syst lg[pt]  '//dd(i)//cc(j),0.05e0,-0.5e0,4.e0)
      call mbook(k+25,'syst invM    '//dd(i)//cc(j),10.e0,0.e0,1000.e0)
      call mbook(k+26,'syst eta     '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+27,'ZZ azimt     '//dd(i)//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+28,'ZZ Deta      '//dd(i)//cc(j),0.2e0,-6.e0,6.e0)
      call mbook(k+29,'ZZ DR        '//dd(i)//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+30,'theta1       '//dd(i)//cc(j),pi/20.e0,0.e0,pi)
      call mbook(k+31,'theta2       '//dd(i)//cc(j),pi/20.e0,0.e0,pi)
c
      enddo
      enddo
c
 999  END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      REAL*8 XNORM
      INTEGER I,J,K,ievt
      OPEN(UNIT=99,NAME='PYTVB.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB
      XNORM=1.D0/IEVT
      DO I=1,250              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+250)
        CALL MOPERA(I+250,'F',I+250,I+250,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+250)             
      ENDDO                          
C
      do j=1,2
      do i=1,3
      k=(i-1)*32+(j-1)*100
      if(i.eq.1)
     #call multitop(250+k+32,249,3,2,'xsec         ',' ','LOG')
      call multitop(250+k+ 1,249,3,2,'l+ pt        ',' ','LOG')
      call multitop(250+k+ 2,249,3,2,'l- pt        ',' ','LOG')
      call multitop(250+k+ 3,249,3,2,'l+ eta       ',' ','LOG')
      call multitop(250+k+ 4,249,3,2,'l- eta       ',' ','LOG')
      call multitop(250+k+ 5,249,3,2,'ll+- pt      ',' ','LOG')
      call multitop(250+k+ 6,249,3,2,'ll+- invM    ',' ','LOG')
      call multitop(250+k+ 7,249,3,2,'ll+- eta     ',' ','LOG')
      call multitop(250+k+ 8,249,3,2,'ll+- azimt   ',' ','LOG')
      call multitop(250+k+ 9,249,3,2,'ll+- Deta    ',' ','LOG')
      call multitop(250+k+10,249,3,2,'ll+- DR      ',' ','LOG')
      call multitop(250+k+11,249,3,2,'ll+-[2] pt   ',' ','LOG')
      call multitop(250+k+12,249,3,2,'ll+-[2] invM ',' ','LOG')
      call multitop(250+k+13,249,3,2,'ll+-[2] eta  ',' ','LOG')
      call multitop(250+k+14,249,3,2,'ll+-[2] azimt',' ','LOG')
      call multitop(250+k+15,249,3,2,'ll+-[2] Deta ',' ','LOG')
      call multitop(250+k+16,249,3,2,'ll+-[2] DR   ',' ','LOG')
      call multitop(250+k+17,249,3,2,'ll++ pt      ',' ','LOG')
      call multitop(250+k+18,249,3,2,'ll++ invM    ',' ','LOG')
      call multitop(250+k+19,249,3,2,'ll++ eta     ',' ','LOG')
      call multitop(250+k+20,249,3,2,'ll++ azimt   ',' ','LOG')
      call multitop(250+k+21,249,3,2,'ll++ Deta    ',' ','LOG')
      call multitop(250+k+22,249,3,2,'ll++ DR      ',' ','LOG')
      call multitop(250+k+23,249,3,2,'syst pt      ',' ','LOG')
      call multitop(250+k+24,249,3,2,'syst lg[pt]  ',' ','LOG')
      call multitop(250+k+25,249,3,2,'syst invM    ',' ','LOG')
      call multitop(250+k+26,249,3,2,'syst eta     ',' ','LOG')
      call multitop(250+k+27,249,3,2,'ZZ azimt     ',' ','LOG')
      call multitop(250+k+28,249,3,2,'ZZ Deta      ',' ','LOG')
      call multitop(250+k+29,249,3,2,'ZZ DR        ',' ','LOG')
      call multitop(250+k+30,249,3,2,'theta1       ',' ','LOG')
      call multitop(250+k+31,249,3,2,'theta2       ',' ','LOG')
      enddo
      enddo
c
      CLOSE(99)
 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      DOUBLE PRECISION PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP,IST,ID,ID1,I,J,KKK,L,L1,ITMP,jj,ibin,
     # IFOUNDPAIRS,ILEP(2,2),IWHERE(30,2,2),IORD(30,2,2)
      DOUBLE PRECISION PT1,GETPTV4,PT2,getinvmv4,TMP,
     # getpseudorapv4,getdelphiv4,ptsyst,xlptsyst,xmsyst,etasyst,
     # delphiZZ,deletaZZ,delrZZ,cth1,getcosv4,cth2,wt,cth1dc,cth2dc,
     # PLEP(4,4),PSYST(4),PSYST5(5),YP(4,4,4),ptlep(4),etalep(4),
     # ptyp(4,4),xmyp(4,4),etayp(4,4),delphiyp(4,4),deletayp(4,4),
     # delryp(4,4),xnormZ1(4),xnormZ2(4),pbst(4,4),z1(5),dec1z1(5),
     # dec2z1(5),z2(5),dec1z2(5),dec2z2(5),plepdc(4,4),ypdc(4,4,4),
     # ptlepdc(4),etalepdc(4),ptypdc(4,4),xmypdc(4,4),etaypdc(4,4),
     # delphiypdc(4,4),deletaypdc(4,4),delrypdc(4,4),xnormdcZ1(4),
     # xnormdcZ2(4),pbstdc(4,4),XM_LL(4),xm_lldc(4),
     & phep(5,4000)
      LOGICAL DIDSOF,FOUNDPAIRS,acceptance,dcacceptance,
     # gencuts,gencutsdc
      INTEGER KK
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
c Keep lepton pairs whose invariant masses are closer than PAIRWIDTH
c to the Z pole mass
      DOUBLE PRECISION PAIRWIDTH
      DATA PAIRWIDTH/10.D0/

      integer pychge
      external pydata

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)

      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
c
C--RETURN IF FAILURE
      IF(IFAIL.EQ.1) RETURN
c
      DO J=1,5
         PHEP(J,1)=P(1,J)
         PHEP(J,2)=P(2,J)
      ENDDO
c
      WWW0=EVWEIGHT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=PYCHGE(K(1,2))+PYCHGE(K(2,2))
c$$$      DIDSOF=.FALSE.
      ILEP(1,1)=0
      ILEP(1,2)=0
      ILEP(2,1)=0
      ILEP(2,2)=0
C
      DO 100 IHEP=1,N
c$$$        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
c$$$  what's the meaning of the line above?
        DO J=1,5
          PHEP(J,IHEP)=P(IHEP,J)
        ENDDO
        IF (K(IHEP,1).LE.10) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(K(IHEP,2))
        ENDIF
        IST=K(IHEP,1)
        ID1=K(IHEP,2)
C LOOK FOR FINAL-STATE ELECTRONS OR MUONS
C (I,J)==(TYPE,CHARGE)
C TYPE=1,2 -> ELECTRON, MUONS
C CHARGE=1,2 -> NEGATIVE, POSITIVE
        IF( IST.LE.10.AND.ID1.EQ.11 )THEN
          ILEP(1,1)=ILEP(1,1)+1
          IWHERE(ILEP(1,1),1,1)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.-11 )THEN
          ILEP(1,2)=ILEP(1,2)+1
          IWHERE(ILEP(1,2),1,2)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.13 )THEN
          ILEP(2,1)=ILEP(2,1)+1
          IWHERE(ILEP(2,1),2,1)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.-13 )THEN
          ILEP(2,2)=ILEP(2,2)+1
          IWHERE(ILEP(2,2),2,2)=IHEP
        ENDIF
  100 CONTINUE
C 
      IF( ILEP(1,1).LT.2.OR.ILEP(1,2).LT.2 )THEN
         write(*,*)'error 500 in pyanal',ILEP(1,1),ILEP(1,2)
         stop
      ENDIF
C ORDER LEPTONS BY HARDNESS. 
C IORD(K,I,J) IS THE K^th HARDEST OF TYPE (I,J)
      DO I=1,1
        DO J=1,2
          IF(ILEP(I,J).EQ.1)THEN
            IORD(1,I,J)=1
          ELSEIF(ILEP(I,J).EQ.2)THEN
            PT1=GETPTV4(PHEP(1,IWHERE(1,I,J)))
            PT2=GETPTV4(PHEP(1,IWHERE(2,I,J)))
            IF(PT1.GT.PT2)THEN
              IORD(1,I,J)=1
              IORD(2,I,J)=2
            ELSE
              IORD(1,I,J)=2
              IORD(2,I,J)=1
            ENDIF
          ELSE
            DO KKK=1,ILEP(I,J)
              IORD(KKK,I,J)=KKK
            ENDDO
            DO KKK=ILEP(I,J),2,-1
              DO L=1,KKK-1
                L1=L+1
                PT1=GETPTV4(PHEP(1,IWHERE(IORD(L,I,J),I,J)))
                PT2=GETPTV4(PHEP(1,IWHERE(IORD(L1,I,J),I,J)))
                IF(PT1.LT.PT2)THEN
                  ITMP=IORD(L,I,J)
                  IORD(L,I,J)=IORD(L1,I,J)
                  IORD(L1,I,J)=ITMP
                ENDIF
              ENDDO
            ENDDO
          ENDIF
        ENDDO
      ENDDO
C KEEP THE HARDEST. CONVENTIONS:
C  1 -> E- 1st hardest
C  2 -> E+ 1st hardest
C  3 -> E- 2nd hardest
C  4 -> E+ 2nd hardest
C NOTE: THESE WILL POSSIBLY BE CHANGED BELOW, IF CANDIDATE Z's ARE FOUND
      DO I=1,4
        PLEP(I,1)=PHEP(I,IWHERE(IORD(1,1,1),1,1))
        PLEP(I,2)=PHEP(I,IWHERE(IORD(1,1,2),1,2))
        PLEP(I,3)=PHEP(I,IWHERE(IORD(2,1,1),1,1))
        PLEP(I,4)=PHEP(I,IWHERE(IORD(2,1,2),1,2))
        PSYST(I)=PLEP(I,1)+PLEP(I,2)+PLEP(I,3)+PLEP(I,4)
        PSYST5(I)=PSYST(I)
        YP(I,1,2)=PLEP(I,1)+PLEP(I,2)
        YP(I,3,4)=PLEP(I,3)+PLEP(I,4)
        YP(I,1,4)=PLEP(I,1)+PLEP(I,4)
        YP(I,2,3)=PLEP(I,2)+PLEP(I,3)
      ENDDO
C FIND IF THERE ARE TWO PAIRS WHOSE INVARIANT MASS IS CLOSE TO THE Z POLE MASS
      XM_LL(1)=getinvmv4(yp(1,1,2))
      XM_LL(2)=getinvmv4(yp(1,3,4))
      XM_LL(3)=getinvmv4(yp(1,1,4))
      XM_LL(4)=getinvmv4(yp(1,2,3))
C GENERATION-LEVEL CUTS
      IF( XM_LL(1).LT.30.D0 .OR. XM_LL(2).LT.30.D0 .OR. 
     #    XM_LL(3).LT.30.D0 .OR. XM_LL(4).LT.30.D0 )THEN
        gencuts=.false.
      ELSE
        gencuts=.true.
      ENDIF
c
      IFOUNDPAIRS=0
      ZMASS=PMAS(23,1)
      IF( ABS(XM_LL(1)-ZMASS).LE.PAIRWIDTH .AND.
     #    ABS(XM_LL(2)-ZMASS).LE.PAIRWIDTH )
     #  IFOUNDPAIRS=1
      IF( ABS(XM_LL(3)-ZMASS).LE.PAIRWIDTH .AND.
     #    ABS(XM_LL(4)-ZMASS).LE.PAIRWIDTH )
     #  IFOUNDPAIRS=IFOUNDPAIRS+10
      IF(IFOUNDPAIRS.GT.0)THEN
        FOUNDPAIRS=.TRUE.
C IF THIS IS NOT THE CASE ALREADY, REDEFINE MOMENTA SO AS (1,2) AND (3,4)
C ARE THE PAIRS ASSOCIATED WITH Z BOSONS. WHEN BOTH (1,2)+(3,4) AND
C (1,4)+(2,3) ARE VALID CHOICES, LEAVE THINGS AS THEY ARE
        IF(IFOUNDPAIRS.EQ.10)THEN
          DO I=1,4
            TMP=PLEP(I,2)
            PLEP(I,2)=PLEP(I,4)
            PLEP(I,4)=TMP
          ENDDO
        ENDIF
      ELSE
        FOUNDPAIRS=.FALSE.
      ENDIF
      DO I=1,4
        DO KKK=1,3
          DO L=KKK+1,4
            YP(I,KKK,L)=PLEP(I,KKK)+PLEP(I,L)
          ENDDO
        ENDDO
      ENDDO
C
      acceptance=.true.
      do kkk=1,4
        ptlep(kkk)=getptv4(plep(1,kkk))
        etalep(kkk)=getpseudorapv4(plep(1,kkk))
        acceptance=acceptance .and.
     #             ptlep(kkk).gt.20.d0.and.abs(etalep(kkk)).lt.2.5d0
      enddo
      do kkk=1,4
        do l=kkk+1,4
          ptyp(kkk,l)=getptv4(yp(1,kkk,l))
          xmyp(kkk,l)=getinvmv4(yp(1,kkk,l))
          etayp(kkk,l)=getpseudorapv4(yp(1,kkk,l))
          delphiyp(kkk,l)=getdelphiv4(plep(1,kkk),plep(1,l))
          deletayp(kkk,l)=etalep(kkk)-etalep(l)
          delryp(kkk,l)=sqrt(delphiyp(kkk,l)**2+deletayp(kkk,l)**2)
        enddo
      enddo
      ptsyst=getptv4(psyst)
      if(ptsyst.gt.0.d0)then
        xlptsyst=log10(ptsyst)
      else
        xlptsyst=-1.d8
      endif
      xmsyst=getinvmv4(psyst)
      etasyst=getpseudorapv4(psyst)
      PSYST5(5)=xmsyst
c
      if(foundpairs)then
        delphiZZ=getdelphiv4(yp(1,1,2),yp(1,3,4))
        deletaZZ=getpseudorapv4(yp(1,1,2))-
     #           getpseudorapv4(yp(1,3,4))
        delrZZ=sqrt(delphiZZ**2+deletaZZ**2)
        call getperpenv4(plep(1,1),plep(1,2),xnormZ1)
        call getperpenv4(plep(1,3),plep(1,4),xnormZ2)
        cth1=getcosv4(xnormZ1,xnormZ2)
        cth1=acos(cth1)
        CALL HWULF4(PSYST5,plep(1,1),pbst(1,1))
        CALL HWULF4(PSYST5,plep(1,2),pbst(1,2))
        CALL HWULF4(PSYST5,plep(1,3),pbst(1,3))
        CALL HWULF4(PSYST5,plep(1,4),pbst(1,4))
        call getperpenv4(pbst(1,1),pbst(1,2),xnormZ1)
        call getperpenv4(pbst(1,3),pbst(1,4),xnormZ2)
        cth2=getcosv4(xnormZ1,xnormZ2)
        cth2=acos(cth2)
c Flat decays
        do i=1,4
          z1(i)=yp(i,1,2)
          z1(5)=xmyp(1,2)
          z2(i)=yp(i,3,4)
          z2(5)=xmyp(3,4)
        enddo
        dec1z1(5)=0.d0
        dec2z1(5)=0.d0
        call pdecay(z1,dec1z1,dec2z1,wt)
        dec1z2(5)=0.d0
        dec2z2(5)=0.d0
        call pdecay(z2,dec1z2,dec2z2,wt)
        do i=1,4
          plepdc(i,1)=dec1z1(i)
          plepdc(i,2)=dec2z1(i)
          plepdc(i,3)=dec1z2(i)
          plepdc(i,4)=dec2z2(i)
          do kkk=1,3
            do l=kkk+1,4
              ypdc(i,kkk,l)=plepdc(i,kkk)+plepdc(i,l)
            enddo
          enddo
        enddo
C Generation-level cuts
        xm_lldc(1)=getinvmv4(ypdc(1,1,2))
        xm_lldc(2)=getinvmv4(ypdc(1,3,4))
        xm_lldc(3)=getinvmv4(ypdc(1,1,4))
        xm_lldc(4)=getinvmv4(ypdc(1,2,3))
        if( xm_lldc(1).lt.30.d0 .or. xm_lldc(2).lt.30.d0 .or. 
     #      xm_lldc(3).lt.30.d0 .or. xm_lldc(4).lt.30.d0 )then
          gencutsdc=.false.
        else
          gencutsdc=.true.
        endif
c
        dcacceptance=.true.
        do kkk=1,4
          ptlepdc(kkk)=getptv4(plepdc(1,kkk))
          etalepdc(kkk)=getpseudorapv4(plepdc(1,kkk))
          dcacceptance=dcacceptance .and.
     #                 ptlepdc(kkk).gt.20.d0.and.
     #                 abs(etalepdc(kkk)).lt.2.5d0
        enddo
        do kkk=1,4
          do l=kkk+1,4
            ptypdc(kkk,l)=getptv4(ypdc(1,kkk,l))
            xmypdc(kkk,l)=getinvmv4(ypdc(1,kkk,l))
            etaypdc(kkk,l)=getpseudorapv4(ypdc(1,kkk,l))
            delphiypdc(kkk,l)=getdelphiv4(plepdc(1,kkk),plepdc(1,l))
            deletaypdc(kkk,l)=etalepdc(kkk)-etalepdc(l)
            delrypdc(kkk,l)=sqrt(delphiypdc(kkk,l)**2+
     &                           deletaypdc(kkk,l)**2)
          enddo
        enddo
        call getperpenv4(plepdc(1,1),plepdc(1,2),xnormdcZ1)
        call getperpenv4(plepdc(1,3),plepdc(1,4),xnormdcZ2)
        cth1dc=getcosv4(xnormdcZ1,xnormdcZ2)
        cth1dc=acos(cth1dc)
        CALL HWULF4(PSYST5,plepdc(1,1),pbstdc(1,1))
        CALL HWULF4(PSYST5,plepdc(1,2),pbstdc(1,2))
        CALL HWULF4(PSYST5,plepdc(1,3),pbstdc(1,3))
        CALL HWULF4(PSYST5,plepdc(1,4),pbstdc(1,4))
        call getperpenv4(pbstdc(1,1),pbstdc(1,2),xnormZ1)
        call getperpenv4(pbstdc(1,3),pbstdc(1,4),xnormZ2)
        cth2dc=getcosv4(xnormZ1,xnormZ2)
        cth2dc=acos(cth2dc)
      endif
c
      jj=0
      kk=0
 111  continue
      if(kk.ge.100.and.jj.lt.100)then
        write(6,*)'jj, kk=',jj,kk
        write(6,*)'foundpairs=',foundpairs
        write(6,*)'acceptances=',acceptance,dcacceptance
        write(*,*)'error 501 in pyanal'
        stop
      endif
      if(jj.eq.100.and.(.not.acceptance))goto 222
      if(.not.gencuts)goto 222
c
      if(kk.eq.0.or.kk.eq.100)then
        if( ILEP(1,1).eq.2 .and.
     #      ILEP(1,2).eq.2 )then
          ibin=0
        elseif( ILEP(1,1).gt.2 .and.
     #          ILEP(1,2).eq.2 )then
          ibin=1
        elseif( ILEP(1,1).eq.2 .and.
     #          ILEP(1,2).gt.2 )then
          ibin=2
        else
          ibin=3
        endif
        call mfill(kk+32,float(ibin),sngl(WWW0))
      endif
      call mfill(kk+ 1,sngl(ptlep(2)),sngl(WWW0))
      call mfill(kk+ 1,sngl(ptlep(4)),sngl(WWW0))
      call mfill(kk+ 2,sngl(ptlep(1)),sngl(WWW0))
      call mfill(kk+ 2,sngl(ptlep(3)),sngl(WWW0))
      call mfill(kk+ 3,sngl(etalep(2)),sngl(WWW0))
      call mfill(kk+ 3,sngl(etalep(4)),sngl(WWW0))
      call mfill(kk+ 4,sngl(etalep(1)),sngl(WWW0))
      call mfill(kk+ 4,sngl(etalep(3)),sngl(WWW0))
c
      call mfill(kk+ 5,sngl(ptyp(1,2)),sngl(WWW0))
      call mfill(kk+ 5,sngl(ptyp(3,4)),sngl(WWW0))
      call mfill(kk+ 6,sngl(xmyp(1,2)),sngl(WWW0))
      call mfill(kk+ 6,sngl(xmyp(3,4)),sngl(WWW0))
      call mfill(kk+ 7,sngl(etayp(1,2)),sngl(WWW0))
      call mfill(kk+ 7,sngl(etayp(3,4)),sngl(WWW0))
      call mfill(kk+ 8,sngl(delphiyp(1,2)),sngl(WWW0))
      call mfill(kk+ 8,sngl(delphiyp(3,4)),sngl(WWW0))
      call mfill(kk+ 9,sngl(deletayp(1,2)),sngl(WWW0))
      call mfill(kk+ 9,sngl(deletayp(3,4)),sngl(WWW0))
      call mfill(kk+10,sngl(delryp(1,2)),sngl(WWW0))
      call mfill(kk+10,sngl(delryp(3,4)),sngl(WWW0))
c
      call mfill(kk+11,sngl(ptyp(1,4)),sngl(WWW0))
      call mfill(kk+11,sngl(ptyp(2,3)),sngl(WWW0))
      call mfill(kk+12,sngl(xmyp(1,4)),sngl(WWW0))
      call mfill(kk+12,sngl(xmyp(2,3)),sngl(WWW0))
      call mfill(kk+13,sngl(etayp(1,4)),sngl(WWW0))
      call mfill(kk+13,sngl(etayp(2,3)),sngl(WWW0))
      call mfill(kk+14,sngl(delphiyp(1,4)),sngl(WWW0))
      call mfill(kk+14,sngl(delphiyp(2,3)),sngl(WWW0))
      call mfill(kk+15,sngl(deletayp(1,4)),sngl(WWW0))
      call mfill(kk+15,sngl(deletayp(2,3)),sngl(WWW0))
      call mfill(kk+16,sngl(delryp(1,4)),sngl(WWW0))
      call mfill(kk+16,sngl(delryp(2,3)),sngl(WWW0))
c
      if(kk.eq.0.or.kk.eq.100)then
        call mfill(kk+11,sngl(ptyp(1,2)),sngl(WWW0))
        call mfill(kk+11,sngl(ptyp(3,4)),sngl(WWW0))
        call mfill(kk+12,sngl(xmyp(1,2)),sngl(WWW0))
        call mfill(kk+12,sngl(xmyp(3,4)),sngl(WWW0))
        call mfill(kk+13,sngl(etayp(1,2)),sngl(WWW0))
        call mfill(kk+13,sngl(etayp(3,4)),sngl(WWW0))
        call mfill(kk+14,sngl(delphiyp(1,2)),sngl(WWW0))
        call mfill(kk+14,sngl(delphiyp(3,4)),sngl(WWW0))
        call mfill(kk+15,sngl(deletayp(1,2)),sngl(WWW0))
        call mfill(kk+15,sngl(deletayp(3,4)),sngl(WWW0))
        call mfill(kk+16,sngl(delryp(1,2)),sngl(WWW0))
        call mfill(kk+16,sngl(delryp(3,4)),sngl(WWW0))
c
        call mfill(kk+ 5,sngl(ptyp(1,4)),sngl(WWW0))
        call mfill(kk+ 5,sngl(ptyp(2,3)),sngl(WWW0))
        call mfill(kk+ 6,sngl(xmyp(1,4)),sngl(WWW0))
        call mfill(kk+ 6,sngl(xmyp(2,3)),sngl(WWW0))
        call mfill(kk+ 7,sngl(etayp(1,4)),sngl(WWW0))
        call mfill(kk+ 7,sngl(etayp(2,3)),sngl(WWW0))
        call mfill(kk+ 8,sngl(delphiyp(1,4)),sngl(WWW0))
        call mfill(kk+ 8,sngl(delphiyp(2,3)),sngl(WWW0))
        call mfill(kk+ 9,sngl(deletayp(1,4)),sngl(WWW0))
        call mfill(kk+ 9,sngl(deletayp(2,3)),sngl(WWW0))
        call mfill(kk+10,sngl(delryp(1,4)),sngl(WWW0))
        call mfill(kk+10,sngl(delryp(2,3)),sngl(WWW0))
      endif
c
      call mfill(kk+17,sngl(ptyp(1,3)),sngl(WWW0))
      call mfill(kk+17,sngl(ptyp(2,4)),sngl(WWW0))
      call mfill(kk+18,sngl(xmyp(1,3)),sngl(WWW0))
      call mfill(kk+18,sngl(xmyp(2,4)),sngl(WWW0))
      call mfill(kk+19,sngl(etayp(1,3)),sngl(WWW0))
      call mfill(kk+19,sngl(etayp(2,4)),sngl(WWW0))
      call mfill(kk+20,sngl(delphiyp(1,3)),sngl(WWW0))
      call mfill(kk+20,sngl(delphiyp(2,4)),sngl(WWW0))
      call mfill(kk+21,sngl(deletayp(1,3)),sngl(WWW0))
      call mfill(kk+21,sngl(deletayp(2,4)),sngl(WWW0))
      call mfill(kk+22,sngl(delryp(1,3)),sngl(WWW0))
      call mfill(kk+22,sngl(delryp(2,4)),sngl(WWW0))
c
      call mfill(kk+23,sngl(ptsyst),sngl(WWW0))
      call mfill(kk+24,sngl(xlptsyst),sngl(WWW0))
      call mfill(kk+25,sngl(xmsyst),sngl(WWW0))
      call mfill(kk+26,sngl(etasyst),sngl(WWW0))
c
      if(foundpairs)then
        call mfill(kk+27,sngl(delphiZZ),sngl(WWW0))
        call mfill(kk+28,sngl(deletaZZ),sngl(WWW0))
        call mfill(kk+29,sngl(delrZZ),sngl(WWW0))
        call mfill(kk+30,sngl(cth1),sngl(WWW0))
        call mfill(kk+31,sngl(cth2),sngl(WWW0))
      endif
c
 222  continue
c
      if((.not.foundpairs).and.jj.eq.0)then
        jj=100
        kk=100
        goto 111
      endif
c
      if(foundpairs.and.(kk.eq.0.or.kk.eq.100))then
        kk=32+jj
        goto 111
      endif
c
      if(foundpairs.and.(kk.eq.32.or.kk.eq.132))then
        if(jj.eq.100.and.(.not.dcacceptance))goto 444
        if(.not.gencutsdc)goto 444
        kk=64+jj
        call mfill(kk+ 1,sngl(ptlepdc(2)),sngl(WWW0))
        call mfill(kk+ 1,sngl(ptlepdc(4)),sngl(WWW0))
        call mfill(kk+ 2,sngl(ptlepdc(1)),sngl(WWW0))
        call mfill(kk+ 2,sngl(ptlepdc(3)),sngl(WWW0))
        call mfill(kk+ 3,sngl(etalepdc(2)),sngl(WWW0))
        call mfill(kk+ 3,sngl(etalepdc(4)),sngl(WWW0))
        call mfill(kk+ 4,sngl(etalepdc(1)),sngl(WWW0))
        call mfill(kk+ 4,sngl(etalepdc(3)),sngl(WWW0))
c
        call mfill(kk+ 5,sngl(ptypdc(1,2)),sngl(WWW0))
        call mfill(kk+ 5,sngl(ptypdc(3,4)),sngl(WWW0))
        call mfill(kk+ 6,sngl(xmypdc(1,2)),sngl(WWW0))
        call mfill(kk+ 6,sngl(xmypdc(3,4)),sngl(WWW0))
        call mfill(kk+ 7,sngl(etaypdc(1,2)),sngl(WWW0))
        call mfill(kk+ 7,sngl(etaypdc(3,4)),sngl(WWW0))
        call mfill(kk+ 8,sngl(delphiypdc(1,2)),sngl(WWW0))
        call mfill(kk+ 8,sngl(delphiypdc(3,4)),sngl(WWW0))
        call mfill(kk+ 9,sngl(deletaypdc(1,2)),sngl(WWW0))
        call mfill(kk+ 9,sngl(deletaypdc(3,4)),sngl(WWW0))
        call mfill(kk+10,sngl(delrypdc(1,2)),sngl(WWW0))
        call mfill(kk+10,sngl(delrypdc(3,4)),sngl(WWW0))
c
        call mfill(kk+11,sngl(ptypdc(1,4)),sngl(WWW0))
        call mfill(kk+11,sngl(ptypdc(2,3)),sngl(WWW0))
        call mfill(kk+12,sngl(xmypdc(1,4)),sngl(WWW0))
        call mfill(kk+12,sngl(xmypdc(2,3)),sngl(WWW0))
        call mfill(kk+13,sngl(etaypdc(1,4)),sngl(WWW0))
        call mfill(kk+13,sngl(etaypdc(2,3)),sngl(WWW0))
        call mfill(kk+14,sngl(delphiypdc(1,4)),sngl(WWW0))
        call mfill(kk+14,sngl(delphiypdc(2,3)),sngl(WWW0))
        call mfill(kk+15,sngl(deletaypdc(1,4)),sngl(WWW0))
        call mfill(kk+15,sngl(deletaypdc(2,3)),sngl(WWW0))
        call mfill(kk+16,sngl(delrypdc(1,4)),sngl(WWW0))
        call mfill(kk+16,sngl(delrypdc(2,3)),sngl(WWW0))
c
        call mfill(kk+17,sngl(ptypdc(1,3)),sngl(WWW0))
        call mfill(kk+17,sngl(ptypdc(2,4)),sngl(WWW0))
        call mfill(kk+18,sngl(xmypdc(1,3)),sngl(WWW0))
        call mfill(kk+18,sngl(xmypdc(2,4)),sngl(WWW0))
        call mfill(kk+19,sngl(etaypdc(1,3)),sngl(WWW0))
        call mfill(kk+19,sngl(etaypdc(2,4)),sngl(WWW0))
        call mfill(kk+20,sngl(delphiypdc(1,3)),sngl(WWW0))
        call mfill(kk+20,sngl(delphiypdc(2,4)),sngl(WWW0))
        call mfill(kk+21,sngl(deletaypdc(1,3)),sngl(WWW0))
        call mfill(kk+21,sngl(deletaypdc(2,4)),sngl(WWW0))
        call mfill(kk+22,sngl(delrypdc(1,3)),sngl(WWW0))
        call mfill(kk+22,sngl(delrypdc(2,4)),sngl(WWW0))
c
        call mfill(kk+23,sngl(ptsyst),sngl(WWW0))
        call mfill(kk+24,sngl(xlptsyst),sngl(WWW0))
        call mfill(kk+25,sngl(xmsyst),sngl(WWW0))
        call mfill(kk+26,sngl(etasyst),sngl(WWW0))
c
        call mfill(kk+27,sngl(delphiZZ),sngl(WWW0))
        call mfill(kk+28,sngl(deletaZZ),sngl(WWW0))
        call mfill(kk+29,sngl(delrZZ),sngl(WWW0))
        call mfill(kk+30,sngl(cth1dc),sngl(WWW0))
        call mfill(kk+31,sngl(cth2dc),sngl(WWW0))
c
 444    continue
      endif
c
      if(jj.eq.0)then
        jj=100
        kk=100
        goto 111
      endif
c
 999  END


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


      function getdry(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 getdry,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # getrapidity,getdelphi
c
      deta=getrapidity(en1,pl1)-
     #     getrapidity(en2,pl2)
      dphi=getdelphi(ptx1,pty1,ptx2,pty2)
      getdry=sqrt(dphi**2+deta**2)
      return
      end


      function getptv(p)
      implicit none
      real*8 getptv,p(5)
c
      getptv=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv(p)
      implicit none
      real*8 getpseudorapv,p(5)
      real*8 getpseudorap
c
      getpseudorapv=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv(p)
      implicit none
      real*8 getrapidityv,p(5)
      real*8 getrapidity
c
      getrapidityv=getrapidity(p(4),p(3))
      return
      end


      function getdrv(p1,p2)
      implicit none
      real*8 getdrv,p1(5),p2(5)
      real*8 getdr
c
      getdrv=getdr(p1(4),p1(1),p1(2),p1(3),
     #             p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv(p)
      implicit none
      real*8 getinvmv,p(5)
      real*8 getinvm
c
      getinvmv=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv(p1,p2)
      implicit none
      real*8 getdelphiv,p1(5),p2(5)
      real*8 getdelphi
c
      getdelphiv=getdelphi(p1(1),p1(2),
     #                     p2(1),p2(2))
      return
      end


      function getptv4(p)
      implicit none
      real*8 getptv4,p(4)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end


      function getpseudorapv4(p)
      implicit none
      real*8 getpseudorapv4,p(4)
      real*8 getpseudorap
c
      getpseudorapv4=getpseudorap(p(4),p(1),p(2),p(3))
      return
      end


      function getrapidityv4(p)
      implicit none
      real*8 getrapidityv4,p(4)
      real*8 getrapidity
c
      getrapidityv4=getrapidity(p(4),p(3))
      return
      end


      function getdrv4(p1,p2)
      implicit none
      real*8 getdrv4,p1(4),p2(4)
      real*8 getdr
c
      getdrv4=getdr(p1(4),p1(1),p1(2),p1(3),
     #              p2(4),p2(1),p2(2),p2(3))
      return
      end


      function getinvmv4(p)
      implicit none
      real*8 getinvmv4,p(4)
      real*8 getinvm
c
      getinvmv4=getinvm(p(4),p(1),p(2),p(3))
      return
      end


      function getdelphiv4(p1,p2)
      implicit none
      real*8 getdelphiv4,p1(4),p2(4)
      real*8 getdelphi
c
      getdelphiv4=getdelphi(p1(1),p1(2),
     #                      p2(1),p2(2))
      return
      end


      function getcosv4(q1,q2)
      implicit none
      real*8 getcosv4,q1(4),q2(4)
      real*8 xnorm1,xnorm2,tmp
c
      if(q1(4).lt.0.d0.or.q2(4).lt.0.d0)then
        getcosv4=-1.d10
        return
      endif
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        tmp=-1.d10
      else
        tmp=q1(1)*q2(1)+q1(2)*q2(2)+q1(3)*q2(3)
        tmp=tmp/(xnorm1*xnorm2)
        if(abs(tmp).gt.1.d0.and.abs(tmp).le.1.001d0)then
          tmp=sign(1.d0,tmp)
        elseif(abs(tmp).gt.1.001d0)then
          write(*,*)'Error in getcosv4',tmp
          stop
        endif
      endif
      getcosv4=tmp
      return
      end


      subroutine getperpenv4(q1,q2,qperp)
c Normal to the plane defined by \vec{q1},\vec{q2}
      implicit none
      real*8 q1(4),q2(4),qperp(4)
      real*8 xnorm1,xnorm2
      integer i
c
      xnorm1=sqrt(q1(1)**2+q1(2)**2+q1(3)**2)
      xnorm2=sqrt(q2(1)**2+q2(2)**2+q2(3)**2)
      if(xnorm1.lt.1.d-6.or.xnorm2.lt.1.d-6)then
        do i=1,4
          qperp(i)=-1.d10
        enddo
      else
        qperp(1)=q1(2)*q2(3)-q1(3)*q2(2)
        qperp(2)=q1(3)*q2(1)-q1(1)*q2(3)
        qperp(3)=q1(1)*q2(2)-q1(2)*q2(1)
        do i=1,3
          qperp(i)=qperp(i)/(xnorm1*xnorm2)
        enddo
        qperp(4)=1.d0
      endif
      return
      end


      SUBROUTINE PDECAY(P,Q1,Q2,WT)
C--- Decays a particle with momentum P into two particles with momenta
C--- Q(1) and Q(2). WT is the phase space density beta_cm
C--- The decay is spherically symmetric in the decay C_of_M frame
C--- Writte by MLM, modified by SF
      implicit none
      double precision pi,twopi
      PARAMETER(PI=3.14159,TWOPI=2.*PI)
      double precision q2e,qp,ctheta,stheta,phi,qplab,qplon,qptr,pmod,
     $     ptr,wt,bet,gam,randa
      double precision P(5),Q1(5),Q2(5),V(3),U(3),pm,q1m,q2m
      double precision x1,x2
      integer i,iseed
      data iseed/1/
c
      PM=P(5)
      Q1M=Q1(5)
      Q2M=Q2(5)
      X1=RANDA(ISEED)
      X2=RANDA(ISEED)
      Q2E=(PM**2-Q1M**2+Q2M**2)/(2.*PM)
      QP=SQRT(MAX(Q2E**2-Q2M**2,0.d0))
      CTHETA=2.*real(x1)-1.
      STHETA=SQRT(1.-CTHETA**2)
      PHI=TWOPI*real(x2)
      QPLON=QP*CTHETA
      QPTR=QP*STHETA
      PMOD=SQRT(P(1)**2+P(2)**2+P(3)**2)
      PTR=SQRT(P(2)**2+P(3)**2)                              

C--- if the decaying particle moves along the X axis:
      IF(PTR.LT.1.E-4) THEN
        V(1)=0.
        V(2)=1.
        V(3)=0.
        U(1)=0.
        U(2)=0.
        U(3)=1.
      ELSE
C--- 
        V(1)=0.
        V(2)=P(3)/PTR
        V(3)=-P(2)/PTR
        U(1)=PTR/PMOD
        U(2)=-P(1)*P(2)/PTR/PMOD
        U(3)=-P(1)*P(3)/PTR/PMOD
      ENDIF
      GAM=P(4)/PM
      BET=PMOD/P(4)
      QPLAB=GAM*(QPLON+BET*Q2E)
      DO I=1,3
      Q2(I)=QPLAB*P(I)/PMOD+QPTR*(V(I)*SIN(PHI)+U(I)*COS(PHI))
      Q1(I)=P(I)-Q2(I)
      END DO
      Q2(4)=GAM*(Q2E+BET*QPLON)
      Q1(4)=P(4)-Q2(4)
      WT=2.*QP/PM
      END            


      FUNCTION RANDA(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,RANDA
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      RANDA = SEED*MINV
      END
