c
c Example analysis for "p p > w+ [QCD]" process.
c Example analysis for "p p > w- [QCD]" process.
c Example analysis for "p p > z [QCD]" process.
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
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,kk,l,jpr
      character*5 cc(2)
      data cc/'     ','Born '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight)
      common/cwgtsinfo/weights_info
c
      call inihist
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      nwgt=1
      weights_info(nwgt)="central value  "
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      nwgt_analysis=nwgt
      xmi=40.d0
      xms=120.d0
      bin=1.0d0
      do j=1,1
      do kk=1,nwgt_analysis
      l=(kk-1)*10+(j-1)*5
      call mbook(l+ 1,'V pt     '//cc(j)//weights_info(kk)
     &     ,2.d0,0.d0,200.d0)
      call mbook(l+ 2,'V log pt '//cc(j)//weights_info(kk)
     &     ,0.05d0,0.d0,5.d0)
      call mbook(l+ 3,'V y      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-9.d0,9.d0)
      call mbook(l+ 4,'V eta    '//cc(j)//weights_info(kk)
     &     ,0.25d0,-9.d0,9.d0)
      call mbook(l+ 5,'mV       '//cc(j)//weights_info(kk)
     &     ,bin,xmi,xms)
      enddo
      enddo
 999  END

C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      OPEN(UNIT=99,FILE='HERSVB.TOP',STATUS='UNKNOWN')
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
      l=(kk-1)*10+(i-1)*5
      call multitop(NPL+l+ 1,NPL-1,3,2,'V pt     ',' ','LOG')
      call multitop(NPL+l+ 2,NPL-1,3,2,'V log pt ',' ','LOG')
      call multitop(NPL+l+ 3,NPL-1,3,2,'V y      ',' ','LOG')
      call multitop(NPL+l+ 4,NPL-1,3,2,'V eta    ',' ','LOG')
      call multitop(NPL+l+ 5,NPL-1,3,2,'mV       ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),XME,PPV(5),PPE(5),PPNU(5),
     # PPDCE(5),PPDCNU(5),WT,ETAEMIN(2),ETAEMAX(2),PTEMIN(2),
     # XMV,PTV,YV,GETRAPIDITY,PTE,THE,ETAE,PTNU,THNU,ETANU,
     # PTDCE,THDCE,ETADCE,PTDCNU,THDCNU,ETADCNU,ETAV,GETPSEUDORAP
      INTEGER ICHSUM,ICHINI,IHEP,JPR,IDENT,IFV,IST,ID,ID1,IHRD,IV,
     # IJ,IE,INU,J
      LOGICAL DIDSOF,FLAG
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 TINY
      INTEGER KK,i,l
      DATA TINY/.1D-5/
      DATA XME/5.11D-4/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c To be changed !!
      ww(1)=1d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
c
c CHOOSE IDENT=24 FOR W+, IDENT=-24 FOR W-, IDENT=23 FOR Z0
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
         WRITE(*,*)'WARNING 502 IN HWANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
      ICHSUM=0
      DIDSOF=.FALSE.
      IFV=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID1=IDHEP(IHEP)
        IF(ID1.EQ.IDENT)THEN
          IV=IHEP
          IFV=1
          DO IJ=1,5
             PPV(IJ)=PHEP(IJ,IHEP)
          ENDDO
        ENDIF
  100 CONTINUE
      IF(IFV.NE.1) THEN
         WRITE(*,*)'WARNING 503 IN HWANAL'
         GOTO 999
      ENDIF
C FILL THE HISTOS
C Variables of the vector boson
      xmv=ppv(5)
      ptv=sqrt(ppv(1)**2+ppv(2)**2)
      yv=getrapidity(ppv(4),ppv(3))
      etav=getpseudorap(ppv(4),ppv(1),ppv(2),ppv(3))
C
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*10+(i-1)*5
            call mfill(l+1,ptv,WWW(kk))
            if(ptv.gt.0) call mfill(l+2,log10(ptv),WWW(kk))
            call mfill(l+3,yv,WWW(kk))
            call mfill(l+4,etav,WWW(kk))
            call mfill(l+5,xmv,WWW(kk))
         enddo
      enddo
C
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
