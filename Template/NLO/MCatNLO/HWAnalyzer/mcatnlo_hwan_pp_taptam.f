c
c Example analysis for "p p > ta+ ta- [QCD]" process.
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
      integer j,kk,l,i
      character*5 cc(2)
      data cc/'     ','Born '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
      call inihist
      nwgt_analysis=nwgt
      do i=1,1
      do kk=1,nwgt_analysis
        l=(kk-1)*8+(i-1)*4
        call mbook(l+1,'total rate  '//cc(i)//weights_info(kk),
     &       1.0d0,0.5d0,5.5d0)
        call mbook(l+2,'ta+ta- mass '//cc(i)//weights_info(kk),
     &       5d0,0d0,200d0)
        call mbook(l+3,'ta+ta- rap  '//cc(i)//weights_info(kk),
     &       0.25d0,-5d0,5d0)
        call mbook(l+4,'ta+ta- pt   '//cc(i)//weights_info(kk),
     &       20d0,0d0,400d0)
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
      do i=1,1
      do kk=1,nwgt_analysis
         l=(kk-1)*8+(i-1)*4
         call multitop(NPL+l+1,NPL-1,3,2,'total rate  ',' ','LIN')
         call multitop(NPL+l+2,NPL-1,3,2,'ta+ ta- mass',' ','LOG')
         call multitop(NPL+l+3,NPL-1,3,2,'ta+ ta- rap ',' ','LOG')
         call multitop(NPL+l+4,NPL-1,3,2,'ta+ ta- pt  ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PH(5),YCUT,XMH,PTH,YH,THV,ETAV,
     #  PPL(5),PPLB(5),PTL,YL,THL,ETAL,PLL,ENL,PTLB,YLB,THLB,ETALB,
     #  PLLB,ENLB,PTPAIR,DLL,CLL,AZI,AZINORM,XMLL,DETALLB,VAR
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
      IDENT=15
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
         WRITE(*,*)'NO TAUS FOUND.'
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
        PH(IJ)=PPL(IJ)+PPLB(IJ)
      ENDDO
      xmh = getinvm(ph(4),ph(1),ph(2),ph(3))
      yh  = getrapidity(ph(4),ph(3))
      pth = sqrt(max(ph(1)**2+ph(2)**2,0d0))
      var = 1.d0
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*8+(i-1)*4
            call mfill(l+1,var,www(kk))
            call mfill(l+2,xmh,www(kk))
            call mfill(l+3,yh,www(kk))
            call mfill(l+4,pth,www(kk))
         enddo
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
