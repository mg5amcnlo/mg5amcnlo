c
c Example analysis for "p p > l+ l- [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      real * 8 bin,xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,kk,l
      character*5 cc(2)
      data cc/'     ','cuts '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
      call inihist
      nwgt_analysis=nwgt
c
      xmi=50.d0
      xms=130.d0
      bin=0.8d0
      do kk=1,nwgt_analysis
      do j=1,2
      l=(kk-1)*42+(j-1)*21
      call mbook(l+ 1,'V pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call mbook(l+ 2,'V pt 1    '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call mbook(l+ 3,'V log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call mbook(l+ 4,'V y       '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call mbook(l+ 5,'V eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call mbook(l+ 6,'mV        '//cc(j)//weights_info(kk)
     &     ,bin,xmi,xms)
c
      call mbook(l+ 7,'lm pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call mbook(l+ 8,'lm pt 1    '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call mbook(l+ 9,'lm log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call mbook(l+10,'lm eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
      call mbook(l+11,'lp pt      '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call mbook(l+12,'lp pt 1    '//cc(j)//weights_info(kk)
     &     ,10.d0,0.d0,1000.d0)
      call mbook(l+13,'lp log[pt] '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.1d0,5.d0)
      call mbook(l+14,'lp eta     '//cc(j)//weights_info(kk)
     &     ,0.2d0,-9.d0,9.d0)
c
      call mbook(l+15,'lmlp delta eta     '//cc(j)//weights_info(kk)
     $     ,0.2d0,-9.d0,9.d0)
      call mbook(l+16,'lmlp azimt         '//cc(j)//weights_info(kk)
     $     ,pi/20.d0,0.d0,pi)
      call mbook(l+17,'lmlp log[pi-azimt] '//cc(j)//weights_info(kk)
     $     ,0.05d0,-4.d0,0.1d0)
      call mbook(l+18,'lmlp inv m         '//cc(j)//weights_info(kk)
     $     ,bin,xmi,xms)
      call mbook(l+19,'lmlp pt            '//cc(j)//weights_info(kk)
     $     ,2.d0,0.d0,200.d0)
      call mbook(l+20,'lmlp log[pt]       '//cc(j)//weights_info(kk)
     $     ,0.05d0,0.1d0,5.d0)
c
      call mbook(l+21,'total'//cc(j)//weights_info(kk),1.d0,-1.d0,1.d0)
      enddo
      enddo
 999  END

C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='HERLL.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,NPL
        CALL MFINAL3(I)
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
        CALL MFINAL3(I+NPL)
      ENDDO
C
      do kk=1,nwgt_analysis
      do i=1,2
      l=(kk-1)*42+(i-1)*21
C
      call multitop(NPL+l+ 1,NPL-1,3,2,'V pt',' ','LOG')
      call multitop(NPL+l+ 2,NPL-1,3,2,'V pt',' ','LOG')
      call multitop(NPL+l+ 3,NPL-1,3,2,'V log[pt]',' ','LOG')
      call multitop(NPL+l+ 4,NPL-1,3,2,'V y',' ','LOG')
      call multitop(NPL+l+ 5,NPL-1,3,2,'V eta',' ','LOG')
      call multitop(NPL+l+ 6,NPL-1,3,2,'mV',' ','LOG')
c
      call multitop(NPL+l+ 7,NPL-1,3,2,'lm pt',' ','LOG')
      call multitop(NPL+l+ 8,NPL-1,3,2,'lm pt',' ','LOG')
      call multitop(NPL+l+ 9,NPL-1,3,2,'lm log[pt]',' ','LOG')
      call multitop(NPL+l+10,NPL-1,3,2,'lm eta',' ','LOG')
      call multitop(NPL+l+11,NPL-1,3,2,'lm pt',' ','LOG')
      call multitop(NPL+l+12,NPL-1,3,2,'lm pt',' ','LOG')
      call multitop(NPL+l+13,NPL-1,3,2,'lm log[pt]',' ','LOG')
      call multitop(NPL+l+14,NPL-1,3,2,'lm eta',' ','LOG')
c
      call multitop(NPL+l+15,NPL-1,3,2,'lmlp deta',' ','LOG')
      call multitop(NPL+l+16,NPL-1,3,2,'lmlp azi',' ','LOG')
      call multitop(NPL+l+17,NPL-1,3,2,'lmlp azi',' ','LOG')
      call multitop(NPL+l+18,NPL-1,3,2,'lmlp inv m',' ','LOG')
      call multitop(NPL+l+19,NPL-1,3,2,'lmlp pt',' ','LOG')
      call multitop(NPL+l+20,NPL-1,3,2,'lmlp pt',' ','LOG')
c
      call multitop(NPL+l+21,NPL-1,3,2,'total',' ','LOG')
      enddo
      enddo
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),YCUT,XMV,PTV,YV,THV,ETAV,
     #  PPL(5),PPLB(5),PTL,YL,THL,ETAL,PLL,ENL,PTLB,YLB,THLB,ETALB,
     #  PLLB,ENLB,PTPAIR,DLL,CLL,AZI,AZINORM,XMLL,DETALLB
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD,ILL0,ILLB0
      LOGICAL DIDSOF,TEST1,TEST2,flag,ISLP,ISLM,FOUNDP,FOUNDM
      REAL*8 PI,wmass,wgamma,bwcutoff,getinvm,getdelphi,getrapidity,
     &getpseudorap
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK,i,l
      DATA TINY/.1D-5/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
      IF (IERROR.NE.0) RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
C CHOOSE IDENT = 11 FOR ELECTRON PAIRS
C        IDENT = 13 FOR MUON PAIRS
C        IDENT = 15 FOR TAU PAIRS
      IDENT=13
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,4)).EQ.SIGN(1.D0,PHEP(3,5)))THEN
        CALL HWWARN('HWANAL',111)
        GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
      FOUNDP=.FALSE.
      FOUNDM=.FALSE.
      DO 100 IHEP=1,NHEP
         IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
         IF (ISTHEP(IHEP).EQ.1) THEN
            CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
            ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
         ENDIF
         IST=ISTHEP(IHEP)      
         ID=IDHW(IHEP)
         ID1=IDHEP(IHEP)
         ISLP=ID1.EQ.IDENT
         ISLM=ID1.EQ.-IDENT
         IF(((IST.GE.120.AND.IST.LE.125).OR.IST.EQ.1.OR.IST.EQ.198)
     &       .AND.ISLM.AND..NOT.FOUNDM)THEN
            ILL=IHEP
            FOUNDM=.TRUE.
         ENDIF
         IF(((IST.GE.120.AND.IST.LE.125).OR.IST.EQ.1.OR.IST.EQ.198)
     &       .AND.ISLP.AND..NOT.FOUNDP)THEN
            ILLB=IHEP
            FOUNDP=.TRUE.
         ENDIF
 100  CONTINUE
      IF(.NOT.FOUNDP.OR..NOT.FOUNDM)THEN
         WRITE(*,*)'NO LEPTONS FOUND.'
         WRITE(*,*)'CURRENTLY THIS ANALYSIS LOOKS FOR'
         IF(IDENT.EQ.11)WRITE(*,*)'ELECTRON PAIRS'
         IF(IDENT.EQ.13)WRITE(*,*)'MUON PAIRS'
         IF(IDENT.EQ.15)WRITE(*,*)'TAU PAIRS'
         WRITE(*,*)'IF THIS IS NOT MEANT,'
         WRITE(*,*)'PLEASE CHANGE THE VALUE OF IDENT IN THIS FILE.'
         STOP
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
C FIND THE LEPTONS
      DO IJ=1,5
        PPL(IJ)=PHEP(IJ,ILL)
        PPLB(IJ)=PHEP(IJ,ILLB)
        PPV(IJ)=PPL(IJ)+PPLB(IJ)
      ENDDO
C FILL THE HISTOS
      YCUT=2.5D0
C Variables of the vector boson
      xmv=getinvm(ppv(4),ppv(1),ppv(2),ppv(3))
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C Variables of the leptons
      ptl=sqrt(ppl(1)**2+ppl(2)**2)
      yl=getrapidity(ppl(4),ppl(3))
      etal=getpseudorap(ppl(4),ppl(1),ppl(2),ppl(3))
c
      ptlb=sqrt(pplb(1)**2+pplb(2)**2)
      ylb=getrapidity(pplb(4),pplb(3))
      etalb=getpseudorap(pplb(4),pplb(1),pplb(2),pplb(3))
c
      ptpair=ptv
      azi=getdelphi(ppl(1),ppl(2),pplb(1),pplb(2))
      azinorm=(pi-azi)/pi
      xmll=xmv
      detallb=etal-etalb
c
      do kk=1,nwgt_analysis
      l=(kk-1)*42
      call mfill(l+1,(ptv),(WWW(kk)))
      call mfill(l+2,(ptv),(WWW(kk)))
      if(ptv.gt.0.d0)call mfill(l+3,(log10(ptv)),(WWW(kk)))
      call mfill(l+4,(yv),(WWW(kk)))
      call mfill(l+5,(etav),(WWW(kk)))
      call mfill(l+6,(xmv),(WWW(kk)))
c
      call mfill(l+7,(ptl),(WWW(kk)))
      call mfill(l+8,(ptl),(WWW(kk)))
      if(ptl.gt.0.d0)call mfill(l+9,(log10(ptl)),(WWW(kk)))
      call mfill(l+10,(etal),(WWW(kk)))
      call mfill(l+11,(ptlb),(WWW(kk)))
      call mfill(l+12,(ptlb),(WWW(kk)))
      if(ptlb.gt.0.d0)call mfill(l+13,(log10(ptlb)),(WWW(kk)))
      call mfill(l+14,(etalb),(WWW(kk)))
c
      call mfill(l+15,(detallb),(WWW(kk)))
      call mfill(l+16,(azi),(WWW(kk)))
      if(azinorm.gt.0.d0)
     #  call mfill(l+17,(log10(azinorm)),(WWW(kk)))
      call mfill(l+18,(xmll),(WWW(kk)))
      call mfill(l+19,(ptpair),(WWW(kk)))
      if(ptpair.gt.0)call mfill(l+20,(log10(ptpair)),(WWW(kk)))
      call mfill(l+21,(0d0),(WWW(kk)))
c
      l=l+21

      if(abs(etav).lt.ycut)then
        call mfill(l+1,(ptv),(WWW(kk)))
        call mfill(l+2,(ptv),(WWW(kk)))
        if(ptv.gt.0.d0)call mfill(l+3,(log10(ptv)),(WWW(kk)))
      endif
      if(ptv.gt.20.d0)then
        call mfill(l+4,(yv),(WWW(kk)))
        call mfill(l+5,(etav),(WWW(kk)))
      endif
      if(abs(etav).lt.ycut.and.ptv.gt.20.d0)then
         call mfill(l+6,(xmv),(WWW(kk)))
         call mfill(l+21,(0d0),(WWW(kk)))
      endif
c
      if(abs(etal).lt.ycut)then
        call mfill(l+7,(ptl),(WWW(kk)))
        call mfill(l+8,(ptl),(WWW(kk)))
        if(ptl.gt.0.d0)call mfill(l+9,(log10(ptl)),(WWW(kk)))
      endif
      if(ptl.gt.20.d0)call mfill(l+10,(etal),(WWW(kk)))
      if(abs(etalb).lt.ycut)then
        call mfill(l+11,(ptlb),(WWW(kk)))
        call mfill(l+12,(ptlb),(WWW(kk)))
        if(ptlb.gt.0.d0)call mfill(l+13,(log10(ptlb)),(WWW(kk)))
      endif
      if(ptlb.gt.20.d0)call mfill(l+14,(etalb),(WWW(kk)))
c
      if( abs(etal).lt.ycut.and.abs(etalb).lt.ycut .and.
     #    ptl.gt.20.d0.and.ptlb.gt.20.d0)then
        call mfill(l+15,(detallb),(WWW(kk)))
        call mfill(l+16,(azi),(WWW(kk)))
        if(azinorm.gt.0.d0)
     #    call mfill(l+17,(log10(azinorm)),(WWW(kk)))
        call mfill(l+18,(xmll),(WWW(kk)))
        call mfill(l+19,(ptpair),(WWW(kk)))
        if(ptpair.gt.0) 
     #    call mfill(l+20,(log10(ptpair)),(WWW(kk)))
      endif

      enddo

 999  END



      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y=0.5d0*log( xplus/xminus  )
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
