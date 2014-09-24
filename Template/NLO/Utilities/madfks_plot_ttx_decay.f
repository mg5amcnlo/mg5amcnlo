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
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight)
      common/cwgtsinfo/weights_info
      REAL*8 pi
c      parameter (pi=3.14160E0)
      parameter (pi=3.141592653589793d0)
      integer ivlep1,ivlep2
      common/vvlin/ivlep1,ivlep2
      character*5 cc(3)
      data cc/'     ',' cuts',' CUTS'/
      integer maxset,maxplot,STEP,Nscales,Npdfpairs,
     # j2,k2,isc,jsc,kk1,i2,nps,nng,kk
      common/cnumplots/maxset,maxplot,STEP,Nscales,Npdfpairs

c different setups (no cuts, cuts, cuts+isolation)
c kinematic distributions we look at
      maxset=3
      maxplot=5+5+1
      Nscales=3
      Npdfpairs=20
      STEP=maxplot*(Nscales*Nscales+2*Npdfpairs)	  
c
      if((nwgt-1).ne.(Nscales*Nscales+2*Npdfpairs))then
        write(*,*)'Error #2',nwgt,Nscales,Npdfpairs
        CALL HWWARN('HWABEG',501)
      endif
      nwgt_analysis=nwgt
c
      call inihist

c loop over each setup	  
      do j2=1,maxset
      k2=(j2-1)*STEP
c loop over scales and PDFs; skip the first which is equal to (1,1)
       do kk=2,nwgt_analysis
         kk1=k2+maxplot*(kk-2)
         call mbook(kk1+ 1 ,'l+ pt      '//cc(j2)//weights_info(kk),
     #              4.d0,0.d0,400.d0)
         call mbook(kk1+ 2 ,'ll pt      '//cc(j2)//weights_info(kk),
     #              4.d0,0.d0,400.d0)
         call mbook(kk1+ 3 ,'ll m       '//cc(j2)//weights_info(kk),
     #              4.d0,0.d0,400.d0)
         call mbook(kk1+ 4 ,'ll EE      '//cc(j2)//weights_info(kk),
     #              12.d0,0.d0,1200.d0)
         call mbook(kk1+ 5 ,'pTl1+pTl2  '//cc(j2)//weights_info(kk),
     #              10.d0,0.d0,1000.d0)
c The centers of the bins correspond to the Mellin moments (eg, the bin
c centered at 1 is the first Mellin moment). The bin at zero should be
c equal to one (ie, zeroth Mellin moment), as a way of cross check 
         call mbook(kk1+ 6 ,'l+ pt      '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
         call mbook(kk1+ 7 ,'ll pt      '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
         call mbook(kk1+ 8 ,'ll m       '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
         call mbook(kk1+ 9 ,'ll EE      '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
         call mbook(kk1+ 10,'pTl1+pTl2  '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
         call mbook(kk1+ 11,'total      '//cc(j2)//weights_info(kk),
     #              1.d0,-0.5d0,4.5d0)
       enddo
      enddo
cccccccccccccccccccccccccccccccccccccccc
c Here is how scales are being scanned:
c       yfactR(1)=1.d0
c       yfactR(2)=rw_Rscale_down
c       yfactR(3)=rw_Rscale_up
c       yfactF(1)=1.d0
c       yfactF(2)=rw_Fscale_down
c       yfactF(3)=rw_Fscale_up
ccccccccccccccccccccccccccccccccccccccc
      return
      END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 SONE
      PARAMETER (SONE=1.d0)
      REAL*8 XNORM
      INTEGER I,IVLEP1,IVLEP2
      COMMON/VVLIN/IVLEP1,IVLEP2
      integer maxset,maxplot,STEP,Nscales,Npdfpairs,
     # j2,k2,isc,jsc,kk1,i2,nps,nng
      common/cnumplots/maxset,maxplot,STEP,Nscales,Npdfpairs
      INTEGER KK,nwgt_analysis
      common/c_analysis/nwgt_analysis

c$$$      OPEN(UNIT=99,NAME='HERQQ.TOP',STATUS='UNKNOWN')
      OPEN(UNIT=99,FILE='HERQQ.TOP',STATUS='UNKNOWN')

C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,5000              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+5000)
        CALL MOPERA(I+5000,'F',I+5000,I+5000,XNORM,0.d0)
      ENDDO                          
c Normalize Mellin moments (i.e. divide plots '6'-'10' by the total x-section plot '11')
c NOTE: below '5' == (maxplot-1)/2 ; '6'='5+1' !!!
      do j2=1,maxset
       k2=(j2-1)*STEP
c loop over scales and PDFs; skip the first which is equal to (1,1)
       do kk=2,nwgt_analysis
         kk1=k2+maxplot*(kk-2)+5
         do i=1,5
           CALL MOPERA(kk1+i+5000,'/',kk1+6+5000,kk1+i+5000,SONE,SONE)
         enddo
       enddo
      enddo
c
      DO I=1,5000              
 	CALL MFINAL3(I+5000)             
      ENDDO                          
c
c loop over each setup	  
      do j2=1,maxset
      k2=(j2-1)*STEP
c loop over scales and PDFs; skip the first which is equal to (1,1)
       do kk=2,nwgt_analysis
         kk1=k2+maxplot*(kk-2)
         call multitop(5000+kk1+ 1 ,4999,2,3,'l+ pt      ',' ','LOG')
         call multitop(5000+kk1+ 2 ,4999,2,3,'ll pt      ',' ','LOG')
         call multitop(5000+kk1+ 3 ,4999,2,3,'ll m       ',' ','LOG')
         call multitop(5000+kk1+ 4 ,4999,2,3,'ll EE      ',' ','LOG')
         call multitop(5000+kk1+ 5 ,4999,2,3,'pTl1+pTl2  ',' ','LOG')
c Mellin
         call multitop(5000+kk1+ 6 ,4999,2,3,'l+ pt      ',' ','LOG')
         call multitop(5000+kk1+ 7 ,4999,2,3,'ll pt      ',' ','LOG')
         call multitop(5000+kk1+ 8 ,4999,2,3,'ll m       ',' ','LOG')
         call multitop(5000+kk1+ 9 ,4999,2,3,'ll EE      ',' ','LOG')
         call multitop(5000+kk1+ 10,4999,2,3,'pTl1+pTl2  ',' ','LOG')
         call multitop(5000+kk1+ 11,4999,2,3,'total      ',' ','LOG')
       enddo
      enddo
c
      CLOSE(99)
      END


C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER ICHSUM,ICHINI,IHEP
      LOGICAL DIDSOF
      LOGICAL BHADRN,BMESON,BBARYON
      INTEGER NL,NN,IST,ID,I,J,IBHAD,mjet,mjet30,mbjet,mbjet30,
     # ILP,ILN,kk,ie,IL(30),jbjmax(30)
      DOUBLE PRECISION ptj1,ptj2,ptjb1,ptjb2,etaj1,etaj2,
     # etajb1,etajb2,dr1,dr2,ptpairlep,detalep,dphilep,dRlep,xmlep,
     # getdr,getptv4,getpseudorapv4,getdelphiv4,getdrv4,getinvmv4,
     # sumEE,sumPT,var,wgt0,PLEP(4,30),ptlep(30),etalep(30),
     # pairlep(4),obs(30)
      integer maxset,maxplot,STEP,Nscales,Npdfpairs,
     # j2,k2,isc,jsc,k1,iobs,ibin,kk1,i2,nps,nng
      common/cnumplots/maxset,maxplot,STEP,Nscales,Npdfpairs

      LOGICAL acceptance,isolation
c jet stuff
      INTEGER NMAX
      PARAMETER (NMAX=4000)
      INTEGER NERANGE
      PARAMETER (NERANGE=1)
      LOGICAL ISAB(NMAX),ISABJET(NMAX),KEEPJ(NMAX,NERANGE)
      INTEGER NJET,JET(NMAX),IPOS(NMAX),NUMBINJET(NMAX),
     # jjmax(30,NERANGE)
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX)
      REAL*8 WWW0
      INTEGER IVLEP1,IVLEP2
      COMMON/VVLIN/IVLEP1,IVLEP2
c pseudorapidity ranges defined as follows:
c   etarangeL(ie)<|eta|<etarangeU(ie),    ie=1,NERANGE
      double precision etarangeL(1),etarangeU(1)
      data etarangeL/0.0d0/
      data etarangeU/2.4d0/
c
      IF (IERROR.NE.0) RETURN
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO
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
      NL=0
      NN=0
      IBHAD=0
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
        IF((ABS(ID).EQ.11.OR.ABS(ID).EQ.13).AND.IST.EQ.1)THEN
          NL=NL+1
          IL(NL)=IHEP
          DO I=1,4
             PLEP(I,NL)=PHEP(I,IHEP)
          ENDDO
        ELSEIF(ABS(ID).GT.100.AND.IST.EQ.1) THEN
          NN=NN+1
          IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
          IPOS(NN)=IHEP
          DO I=1,4
             PP(I,NN)=PHEP(I,IHEP)
          ENDDO
        ENDIF
        IF(BHADRN.AND.IST.EQ.1)THEN
          IBHAD=IBHAD+1
          ISAB(IHEP)=.TRUE.
        ELSE
          ISAB(IHEP)=.FALSE.
        ENDIF
  100 CONTINUE
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.d-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
c
      IF( (NL.LT.2.OR.IBHAD.LT.2).AND.
     #    IERROR.EQ.0) THEN
         CALL HWUEPR
         WRITE(*,*)NL,IBHAD
         CALL HWWARN('HWANAL',503)
      ENDIF
C---CLUSTER THE EVENT
      palg=-1.d0
      rfj=0.5d0
      sycut=0.d0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
c Check order in pt, and keep only jet with |eta|<2.4
      mjet=0
      mjet30=0
      do i=1,njet
        ptjet(i)=getptv4(pjet(1,i))
        if(i.gt.1)then
          if (ptjet(i).gt.ptjet(i-1)) then
            write (*,*) "Error in fastjet: jets should be ordered in pt"
            CALL HWWARN('HWANAL',501)
          endif
        endif
        numbinjet(i)=0
        etajet(i)=getpseudorapv4(pjet(1,i))
        do ie=1,NERANGE
          keepj(i,ie)=.false.
          if( abs(etajet(i)).ge.etarangeL(ie) .and.
     #        abs(etajet(i)).lt.etarangeU(ie) )then
            keepj(i,ie)=.true.
            if(ie.eq.1)then
              mjet=mjet+1
              if(ptjet(i).ge.30.d0)mjet30=mjet30+1
            endif
          endif
        enddo
      enddo
c Check whether a jet is a light or a b jet.
c Do so only for jets within pseudorapidity acceptance
      do i=1,nn
        if(jet(i).ne.0)then
          if(keepj(jet(i),1))then
            ID=IDHEP(IPOS(i))
            CALL BHAD(ID,BHADRN,BMESON,BBARYON)
            if(BHADRN)then
              numbinjet(jet(i))=numbinjet(jet(i))+1
              if(.not.ISAB(IPOS(i)))then
                CALL HWWARN('HWANAL',502)
              endif
            endif
          endif
        elseif(jet(i).gt.njet)then
          CALL HWWARN('HWANAL',504)
        endif
      enddo
c
      mbjet=0
      mbjet30=0
c The two hardest b jets within acceptance. Keep in mind the jets are ordered
c in pt, so these are just the first two encountered
      jbjmax(1)=0
      jbjmax(2)=0
      do i=1,njet
        if(keepj(i,1))then
          if(numbinjet(i).eq.0)then
            ISABJET(i)=.FALSE.
          else
            ISABJET(i)=.TRUE.
            mbjet=mbjet+1
            if(jbjmax(2).eq.0.and.jbjmax(1).ne.0)jbjmax(2)=i
            if(jbjmax(1).eq.0)jbjmax(1)=i
            if(ptjet(i).ge.30.d0)mbjet30=mbjet30+1
          endif
        endif
      enddo
c Sanity checks
      if( (jbjmax(1).eq.jbjmax(2).and.jbjmax(1).ne.0) .or.
     #    (jbjmax(1).eq.0.and.jbjmax(2).ne.0) .or.
     #    (mbjet.ge.2.and.jbjmax(2).eq.0) .or.
     #    (mbjet.eq.1.and.jbjmax(2).ne.0) .or.
     #    (mbjet.eq.1.and.jbjmax(1).eq.0) )then
        write(*,*)njet,mjet,mbjet
        write(*,*)jbjmax(1),jbjmax(2)
        CALL HWWARN('HWANAL',504) 
      endif
c Tag the first two jets different from the hardest b jets -- these
c can be light jets, but also softer b jets. Also check isolation
c from leptons of all jets harder than 30 GeV
      isolation=.true.
      do ie=1,NERANGE
        jjmax(1,ie)=0
        jjmax(2,ie)=0
        do i=1,njet
          if(keepj(i,ie).and.jbjmax(1).ne.i.and.jbjmax(2).ne.i)then
            if(jjmax(2,ie).eq.0.and.jjmax(1,ie).ne.0)jjmax(2,ie)=i
            if(jjmax(1,ie).eq.0)jjmax(1,ie)=i
            if(ptjet(i).ge.30.d0)then
              if(ie.eq.1)then
                dr1=getdr(PLEP(4,1),PLEP(1,1),PLEP(2,1),PLEP(3,1),
     #                    pjet(4,i),pjet(1,i),pjet(2,i),pjet(3,i))
                dr2=getdr(PLEP(4,2),PLEP(1,2),PLEP(2,2),PLEP(3,2),
     #                    pjet(4,i),pjet(1,i),pjet(2,i),pjet(3,i))
                isolation=isolation .and.
     #                    (dr1.gt.0.4d0).and.(dr2.gt.0.4d0)
              endif
            endif
          endif
        enddo
      enddo
c
      if(jbjmax(1).ne.0)then
        ptjb1=ptjet(jbjmax(1))
        etajb1=etajet(jbjmax(1))
      else
        ptjb1=-1.d8
        etajb1=-1.d8
      endif
      if(jbjmax(2).ne.0)then
        ptjb2=ptjet(jbjmax(2))
        etajb2=etajet(jbjmax(2))
      else
        ptjb2=-1.d8
        etajb2=-1.d8
      endif
      if(jjmax(1,1).ne.0)then
        ptj1=ptjet(jjmax(1,1))
        etaj1=etajet(jjmax(1,1))
      else
        ptj1=-1.d8
        etaj1=-1.d8
      endif
      if(jjmax(2,1).ne.0)then
        ptj2=ptjet(jjmax(2,1))
        etaj2=etajet(jjmax(2,1))
      else
        ptj2=-1.d8
        etaj2=-1.d8
      endif
c Use the first two leptons in the event record
      sumEE=0.d0
      sumPT=0.d0
      do i=1,2
        ptlep(i)=getptv4(plep(1,i))
        etalep(i)=getpseudorapv4(plep(1,i))
        sumEE=sumEE+plep(4,i)
        sumPT=sumPT+ptlep(i)				
      enddo
	  
      if(IDHEP(IL(1)).LT.0)then
        ILP=1
        ILN=2
      else
        ILP=2
        ILN=1
      endif
      do j=1,4
        pairlep(j)=plep(j,1)+plep(j,2)
      enddo
      ptpairlep=getptv4(pairlep)
      dphilep=getdelphiv4(plep(1,1),plep(1,2))
      xmlep=getinvmv4(pairlep)
      detalep=etalep(ILP)-etalep(ILN)
      dRlep=getdrv4(plep(1,1),plep(1,2))
c
      acceptance=
     #  ptlep(1).ge.20.d0.and.abs(etalep(1)).le.2.4d0 .and.
     #  ptlep(2).ge.20.d0.and.abs(etalep(2)).le.2.4d0 .and.
     #  ptjb1.ge.30.d0.and.abs(etajb1).le.2.4d0 .and.
     #  ptjb2.ge.30.d0.and.abs(etajb2).le.2.4d0
c
      obs(1)=ptlep(ILP)
      obs(2)=ptpairlep
      obs(3)=xmlep
      obs(4)=sumEE
      obs(5)=sumPT
c loop over each setup	  
      do j2=1,maxset
      k2=(j2-1)*STEP
c loop over scales and PDFs; skip the first which is equal to (1,1)
       do kk=2,nwgt_analysis
        kk1=k2+maxplot*(kk-2)
c NOTE: '3' == maxset !
        if ((j2.eq.1).or.
     #     ((j2.eq.2).and.acceptance).or.
     #     ((j2.eq.3).and.acceptance.and.isolation)) then
         WWW0=WWW(kk)
c
         do ibin=0,4
          k1=kk1+11
          call mfill(k1,dfloat(ibin),WWW0)
          do iobs=1,5
            if(ibin.eq.0)then
              var=1.d0
            else
              var=obs(iobs)**ibin
            endif
            wgt0=WWW0*var
            k1=kk1+iobs+5
            call mfill(k1,dfloat(ibin),wgt0)
            if(ibin.eq.0)then
c Fill observables just once
              k1=kk1+iobs
              call mfill(k1,obs(iobs),WWW0)
            endif
          enddo
         enddo
c
        endif 
       enddo
      enddo
C
 999  RETURN
      END


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


      function getinvmv(p)
      implicit none
      real*8 getinvmv,p(5)
      real*8 getinvm
c
      getinvmv=getinvm(p(4),p(1),p(2),p(3))
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


