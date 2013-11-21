c
c Example analysis for "p p > e+ ve [QCD]" process.
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      integer j,kk,l,i
      character*5 cc(2)
      data cc/'     ','     '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight)
      common/cwgtsinfo/weights_info
c
      call inihist
      nwgt_analysis=nwgt
      do i=1,1
      do kk=1,nwgt_analysis
        l=(kk-1)*16+(i-1)*8
        call mbook(l+1,'total rate  '//weights_info(kk)//cc(i),
     &       1.0d0,0.5d0,5.5d0)
        call mbook(l+2,'e rapidity  '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+3,'e pt        '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+4,'et miss     '//weights_info(kk)//cc(i),
     &       10d0,0d0,200d0)
        call mbook(l+5,'trans. mass '//weights_info(kk)//cc(i),
     &       5d0,0d0,200d0)
        call mbook(l+6,'w rapidity  '//weights_info(kk)//cc(i),
     &       0.5d0,-5d0,5d0)
        call mbook(l+7,'w pt        '//weights_info(kk)//cc(i),
     &       10d00,0d0,200d0)
        call mbook(l+8,'cphi[e,ve]  '//weights_info(kk)//cc(i),
     &       0.05d0,-1d0,1d0)
      enddo
      enddo
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
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
         l=(kk-1)*16+(i-1)*8
         call multitop(NPL+l+1,NPL-1,3,2,'total rate ',' ','LIN')
         call multitop(NPL+l+2,NPL-1,3,2,'e rapidity ',' ','LIN')
         call multitop(NPL+l+3,NPL-1,3,2,'e pt       ',' ','LOG')
         call multitop(NPL+l+4,NPL-1,3,2,'et miss    ',' ','LOG')
         call multitop(NPL+l+5,NPL-1,3,2,'trans. mass',' ','LOG')
         call multitop(NPL+l+6,NPL-1,3,2,'w rapidity ',' ','LIN')
         call multitop(NPL+l+7,NPL-1,3,2,'w pt       ',' ','LOG')
         call multitop(NPL+l+8,NPL-1,3,2,'cphi[e,ve] ',' ','LOG')
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4),PPV(5),PTW,YW,YE,PPL(5),PPLB(5),
     & PTE,PLL,PTLB,PLLB,var,mtr,etmiss,cphi
      INTEGER ICHSUM,ICHINI,IHEP,IV,IFV,IST,ID,IJ,ID1,JPR,IDENT,
     #  ILL,ILLB,IHRD
      LOGICAL DIDSOF,TEST1,TEST2,flag
      REAL*8 PI,getrapidity
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0,TINY
      INTEGER KK,I,L
      DATA TINY/.1D-5/
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
      IF (IERROR.NE.0) RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
      IDENT=24
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT'S A POWER-SUPPRESSED
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
      IFV=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHW(IHEP)
        ID1=IDHEP(IHEP)
C Herwig relabels the vector boson V in Drell-Yan; this doesn't happen with
C MC@NLO; in S events, V appears as HARD, in H as V, but with status 155
C rather than 195. We add here 195 for future compatibility
        IF(IPROC.LT.0)THEN
          TEST1=IST.EQ.155.OR.IST.EQ.195
          TEST2=ID1.EQ.IDENT
          IF(IST.EQ.120.AND.ID1.EQ.0)IHRD=IHEP
        ELSE
          TEST1=IST.EQ.120.OR.IST.EQ.195
          TEST2=ABS(ID1).EQ.IDENT
        ENDIF
        IF(TEST1.AND.TEST2)THEN
          IV=IHEP
          IFV=IFV+1
        ENDIF
  100 CONTINUE
      IF(IPROC.LT.0.AND.IFV.EQ.0)THEN
        IV=IHRD
        IFV=1
      ENDIF
      DO IJ=1,5
        PPV(IJ)=PHEP(IJ,IV)
      ENDDO
      IF(IFV.EQ.0.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',503)
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
      IF(IFV.GT.1.AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',55)
         GOTO 999
      ENDIF
C FIND THE LEPTONS
      IF( IDHEP(JDAHEP(1,JDAHEP(1,IV))).GT.0 .AND.
     #    IDHEP(JDAHEP(1,JDAHEP(2,IV))).LT.0 )THEN
        ILL=JDAHEP(1,JDAHEP(1,IV))
        ILLB=JDAHEP(1,JDAHEP(2,IV))
      ELSEIF( IDHEP(JDAHEP(1,JDAHEP(2,IV))).GT.0 .AND.
     #        IDHEP(JDAHEP(1,JDAHEP(1,IV))).LT.0 )THEN
        ILL=JDAHEP(1,JDAHEP(2,IV))
        ILLB=JDAHEP(1,JDAHEP(1,IV))
      ELSE
        goto 999
      ENDIF
      DO IJ=1,5
        PPL(IJ)=PHEP(IJ,ILL)
        PPLB(IJ)=PHEP(IJ,ILLB)
      ENDDO
C CHECK THAT THE LEPTONS ARE FINAL-STATE LEPTONS
      IF( ABS(IDHEP(ILL)).LT.11 .OR. ABS(IDHEP(ILL)).GT.16 .OR.
     #    ABS(IDHEP(ILLB)).LT.11 .OR. ABS(IDHEP(ILLB)).GT.16 )THEN
        goto 999
      ENDIF
      IF( (ISTHEP(ILL).NE.1.AND.ISTHEP(ILL).NE.195) .OR.
     #    (ISTHEP(ILLB).NE.1.AND.ISTHEP(ILLB).NE.195) )THEN
        goto 999
      ENDIF
      ye     = getrapidity(pplb(4), pplb(3))
      yw     = getrapidity(ppv(4), ppv(3))
      pte    = dsqrt(pplb(1)**2 + pplb(2)**2)
      ptw    = dsqrt(ppv(1)**2+ppv(2)**2)
      etmiss = dsqrt(ppl(1)**2 + ppl(2)**2)
      mtr    = dsqrt(2d0*pte*etmiss-2d0*ppl(1)*pplb(1)-2d0*ppl(2)*pplb(2))
      cphi   = (ppl(1)*pplb(1)+ppl(2)*pplb(2))/pte/etmiss
      var    = 1.d0
      do i=1,1
         do kk=1,nwgt_analysis
            l=(kk-1)*16+(i-1)*8
            call mfill(l+1,var,www(kk))
            call mfill(l+2,ye,www(kk))
            call mfill(l+3,pte,www(kk))
            call mfill(l+4,etmiss,www(kk))
            call mfill(l+5,mtr,www(kk))
            call mfill(l+6,yw,www(kk))
            call mfill(l+7,ptw,www(kk))
            call mfill(l+8,cphi,www(kk))
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
