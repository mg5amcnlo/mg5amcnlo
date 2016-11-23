c
c Example analysis for "p p > t j [QCD]" process.
c
c It features the HwU format for histogram booking and output.
c The details of how to process/manipulate the resulting .HwU file,
c in particular how to plot it using gnuplot, I refer the reader to this
c FAQ:
c
c      https://answers.launchpad.net/mg5amcnlo/+faq/2671
c
c It mostly relies on using the following madgraph5 module in standalone
c
c  <MG5_aMC_install_dir>/madgraph/various/histograms.py
c
c You can learn about how to run it and what options are available with
c
c  python <MG5_aMC_install_dir>/madgraph/various/histograms.py --help
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
      REAL*8 pi
      integer j,kk,l,i
      PARAMETER (PI=3.14159265358979312D0)
c
c     The type suffix of the histogram title, with syntax 
c     |T@<type_name> is semantic in the HwU format. It allows for
c     various filtering when using the histogram.py module
c     (see comment at the beginning of this file).
c     It is in general a good idea to keep the same title for the
c     same observable (if they use the same range) and differentiate
c     them only using the type suffix.
c
      character*8 HwUtype(2)
      data HwUtype/'|T@NOCUT','|T@CUT  '/
      integer nwgt,max_weight,nwgt_analysis
      common/cnwgt/nwgt
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
c Initialize histograms
      call HwU_inithist(nwgt,weights_info)
c Set method for error estimation to '0', i.e., use Poisson statistics
c for the uncertainty estimate
      call set_error_estimation(0)
      nwgt_analysis=nwgt
      do j=1,1
      l=(j-1)*24
      call HwU_book(l+ 1,'t pt        '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+ 2,'t log pt    '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+ 3,'t y         '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+ 4,'t eta       '//HwUtype(j),48,-6d0,6d0)
c
      call HwU_book(l+ 5,'j1 pt       '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+ 6,'j1 log pt   '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+ 7,'j1 y        '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+ 8,'j1 eta      '//HwUtype(j),48,-6d0,6d0)
c
      call HwU_book(l+ 9,'j2 pt       '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+10,'j2 log pt   '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+11,'j2 y        '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+12,'j2 eta      '//HwUtype(j),48,-6d0,6d0)
c
      call HwU_book(l+13,'bj1 pt      '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+14,'bj1 log pt  '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+15,'bj1 y       '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+16,'bj1 eta     '//HwUtype(j),48,-6d0,6d0)
c
      call HwU_book(l+17,'bj2 pt      '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+18,'bj2 log pt  '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+19,'bj2 y       '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+20,'bj2 eta     '//HwUtype(j),48,-6d0,6d0)
c
      call HwU_book(l+21,'syst pt     '//HwUtype(j),40,0d0,200d0)
      call HwU_book(l+22,'syst log pt '//HwUtype(j),100,0d0,5d0)
      call HwU_book(l+23,'syst y      '//HwUtype(j),48,-6d0,6d0)
      call HwU_book(l+24,'syst eta    '//HwUtype(j),48,-6d0,6d0)
c
      enddo
      END

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
c Convert from nb to pb using xnorm. This assumes that event weights
c *average* to the total cross section, so no extra weight needed
      xnorm=1d3
c Collect accumulated results
      call finalize_histograms(nevhep)
c Write the histograms to disk. 
      open (unit=99,file='MADatNLO.HwU',status='unknown')
      call HwU_output(99,xnorm)
      close (99)
      return
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C     BASED ON AN ANALYSIS FILE WRITTEN BY E.RE
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION HWVDOT,PSUM(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER kk,mu,jpart,i,j,ihep,ichsum,nt,nb,nbjet,ist,id,
     &njet,id1,i1,i2,ibmatch,ichini,k,ihadr,count_j,count_bj
      integer maxtrack,maxjet,maxnum,l
      parameter (maxtrack=2048,maxjet=2048,maxnum=30)
      integer ntracks,jetvec(maxtrack),ib1
      double precision pttop,etatop,ytop,ptj1,etaj1,yj1,ptbj1,etabj1,
     &ybj1,ptbj2,etabj2,ybj2,jet_ktradius,jet_ktptmin,palg,pttemp_spec,
     &pttemp_bjet,pttemp,tmp,getrapidity,getpseudorap,getpt,
     &pjet(4,maxtrack),ptrack(4,maxtrack),p_top(4,maxnum),p_b(4,maxnum),
     &p_bjet(4,maxnum),psyst(4),ptsyst,ysyst,etasyst,ptj2,yj2,etaj2
      logical is_b_jet(maxnum)
      integer btrack(maxnum),ib(maxnum)
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      integer maxRWGT
      parameter (maxRWGT=100)
      double precision wgtxsecRWGT(maxRWGT)
      parameter (max_weight=maxscales*maxscales+maxpdfs+maxRWGT+1)
      double precision ww(max_weight),www(max_weight)
      common/cww/ww
c
      IF (IERROR.NE.0) RETURN
      IF (WW(1).EQ.0D0) THEN
         WRITE(*,*)'WW(1) = 0. Stopping'
         STOP
      ENDIF
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

C INITIALIZE
      NT=0
      NB=0
      NBJET=0
      NTRACKS=0
      NJET=0

      DO IHEP=1,NHEP
         IF (ISTHEP(IHEP).EQ.1) THEN
            CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
            ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
         ENDIF
         IST=ISTHEP(IHEP)      
         ID=IDHEP(IHEP)
         ID1=IHADR(ID) ! equal to the PDG of the massive quark in hadron
C TOP
        IF(IST.EQ.155.AND.ABS(ID).EQ.6) THEN
           DO MU=1,4
              P_TOP(MU,1)=PHEP(MU,IHEP)
           ENDDO
        ENDIF
c Define particles that go into jet. 
        IF (IST.EQ.1.AND.ABS(ID).GE.100)THEN
           NTRACKS=NTRACKS+1
           if (abs(id1).eq.5) THEN
c FOUND A stable B-FLAVOURED HADRON.
              NB=NB+1
              IB(NB)=IHEP
              DO MU=1,4
                 P_B(MU,NB)=PHEP(MU,IHEP)
              ENDDO
              BTRACK(NB)=NTRACKS
           endif
           DO MU=1,4
              PTRACK(MU,NTRACKS)=PHEP(MU,IHEP)
           ENDDO
           IF(NTRACKS.EQ.MAXTRACK) THEN
              WRITE(*,*)'HWANAL: TOO MANY PARTICLES, INCREASE MAXTRACK'
              STOP
           ENDIF
        ENDIF
      ENDDO
C END OF LOOP OVER IHEP

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

      IF (NTRACKS.EQ.0) THEN
         WRITE(*,*) 'NO TRACKS FOUND, DROP ANALYSIS OF THIS EVENT'
         GOTO 999
      ENDIF
         
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C KT ALGORITHM, FASTJET IMPLEMENTATION
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      NJET=0
      JET_KTRADIUS = 0.7D0          
      JET_KTPTMIN  = 5D0
      PALG=1D0
      CALL fastjetppgenkt(PTRACK,NTRACKS,JET_KTRADIUS,JET_KTPTMIN,PALG,
     $     PJET,NJET,JETVEC)

c Check that jets are ordered in pt
      do i=1,njet-1
         if (getpt(pjet(1,i)).lt.getpt(pjet(1,i+1)) ) then
            write (*,*) 'ERROR jets not ordered'
            stop
         endif
      enddo
         
C b-jet 
      do i=1,njet
         is_b_jet(i)=.false.
         do j=1,NB
            if (JETVEC(BTRACK(j)).eq.i) then
               is_b_jet(i)=.true.
               exit
            endif
         enddo
      enddo

      pttop = getpt(p_top(1,1))
      ytop  = getrapidity(p_top(1,1))
      etatop= getpseudorap(p_top(1,1))

      count_j=0
      count_bj=0
      do i=1,njet
         if(.not.is_b_jet(i))then
            count_j=count_j+1
            if(count_j.eq.1)then
               ptj1 = getpt(pjet(1,i))
               yj1  = getrapidity(pjet(1,i))
               etaj1= getpseudorap(pjet(1,i))
               do mu=1,4
                  psyst(mu)=p_top(mu,1)+pjet(mu,i)
               enddo
               ptsyst = getpt(psyst)
               ysyst  = getrapidity(psyst)
               etasyst= getpseudorap(psyst)
            elseif(count_j.eq.2)then
               ptj2 = getpt(pjet(1,i))
               yj2  = getrapidity(pjet(1,i))
               etaj2= getpseudorap(pjet(1,i))
            endif
         elseif(is_b_jet(i))then
            count_bj=count_bj+1
            if (count_bj.eq.1) then
               ptbj1 = getpt(pjet(1,i))
               ybj1  = getrapidity(pjet(1,i))
               etabj1= getpseudorap(pjet(1,i))
            elseif (count_bj.eq.2) then
               ptbj2 = getpt(pjet(1,i))
               ybj2  = getrapidity(pjet(1,i))
               etabj2= getpseudorap(pjet(1,i))
            endif
         endif
      enddo
      nbjet=count_bj
c fill the histograms
      do i=1,1
            l=(i-1)*24
            call HwU_fill(l+1,pttop,WWW)
            if(pttop.gt.0d0) call HwU_fill(l+2,log10(pttop),WWW)
            call HwU_fill(l+3,ytop,WWW)
            call HwU_fill(l+4,etatop,WWW)
            if(njet.ge.1)then
               call HwU_fill(l+5,ptj1,WWW)
               if (ptj1.gt.0d0) call HwU_fill(l+6,log10(ptj1),WWW)
               call HwU_fill(l+7,yj1,WWW)
               call HwU_fill(l+8,etaj1,WWW)
               call HwU_fill(l+21,ptsyst,WWW)
               if(ptsyst.gt.0d0) call HwU_fill(l+22,log10(ptsyst),WWW)
               call HwU_fill(l+23,ysyst,WWW)
               call HwU_fill(l+24,etasyst,WWW)
            endif
            if(njet.ge.2)then
               call HwU_fill(l+9,ptj2,WWW)
               if(ptj2.gt.0d0) call HwU_fill(l+10,log10(ptj2),WWW)
               call HwU_fill(l+11,yj2,WWW)
               call HwU_fill(l+12,etaj2,WWW)
            endif
            if(nbjet.ge.1)then
               call HwU_fill(l+13,ptbj1,WWW)
               if (ptbj1.gt.0d0) call HwU_fill(l+14,log10(ptbj1),WWW)
               call HwU_fill(l+15,ybj1,WWW)
               call HwU_fill(l+16,etabj1,WWW)
            endif
            if(nbjet.ge.2)then
               call HwU_fill(l+17,ptbj2,WWW)
               if (ptbj2.gt.0d0) call HwU_fill(l+18,log10(ptbj2),WWW)
               call HwU_fill(l+19,ybj2,WWW)
               call HwU_fill(l+20,etabj2,WWW)
            endif
         enddo
         call HwU_add_points      
 999  RETURN
      END


      function getrapidity(p)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y,p(4)
      parameter (tiny=1.d-5)
c
      en=p(4)
      pl=p(3)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny )then
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


      function getpseudorap(p)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th,p(4)
      parameter (tiny=1.d-5)
c
      en=p(4)
      ptx=p(1)
      pty=p(2)
      pl=p(3)
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

      function getpt(p)
      implicit none
      real*8 getpt,p(4)
      getpt=dsqrt(p(1)**2+p(2)**2)
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


