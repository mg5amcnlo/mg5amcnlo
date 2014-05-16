c
c Example analysis for "p p > t j [QCD]" process.
c Example analysis for "p p > t j $$ w+ w- [QCD]" process.
c Example analysis for "p p > w+ > t j  [QCD]" process.
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
      REAL*8 pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,kk,l,i,nnn
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
      do j=1,1
      do kk=1,nwgt_analysis
      l=(kk-1)*48+(j-1)*24
      call mbook(l+ 1,'t pt        '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+ 2,'t log pt    '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+ 3,'t y         '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+ 4,'t eta       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call mbook(l+ 5,'j1 pt       '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+ 6,'j1 log pt   '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+ 7,'j1 y        '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+ 8,'j1 eta      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call mbook(l+ 9,'j2 pt       '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+10,'j2 log pt   '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+11,'j2 y        '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+12,'j2 eta      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call mbook(l+13,'bj1 pt      '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+14,'bj1 log pt  '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+15,'bj1 y       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+16,'bj1 eta     '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call mbook(l+17,'bj2 pt      '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+18,'bj2 log pt  '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+19,'bj2 y       '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+20,'bj2 eta     '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
      call mbook(l+21,'syst pt     '//cc(j)//weights_info(kk)
     &     ,5d0,0d0,200d0)
      call mbook(l+22,'syst log pt '//cc(j)//weights_info(kk)
     &     ,0.05d0,0d0,5d0)
      call mbook(l+23,'syst y      '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
      call mbook(l+24,'syst eta    '//cc(j)//weights_info(kk)
     &     ,0.25d0,-6d0,6d0)
c
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
      OPEN(UNIT=99,FILE='HERST.top',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,NPL
        CALL MFINAL3(I)
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
        CALL MFINAL3(I+NPL)
      ENDDO
      do i=1,1
      do kk=1,nwgt_analysis
      l=(kk-1)*48+(i-1)*24
      call multitop(NPL+l+ 1,NPL-1,2,3,'t pt',' ','LOG')
      call multitop(NPL+l+ 2,NPL-1,2,3,'t log pt',' ','LOG')
      call multitop(NPL+l+ 3,NPL-1,2,3,'t y',' ','LOG')
      call multitop(NPL+l+ 4,NPL-1,2,3,'t eta',' ','LOG')
c
      call multitop(NPL+l+ 5,NPL-1,2,3,'j1 pt',' ','LOG')
      call multitop(NPL+l+ 6,NPL-1,2,3,'j1 log pt',' ','LOG')
      call multitop(NPL+l+ 7,NPL-1,2,3,'j1 y',' ','LOG')
      call multitop(NPL+l+ 8,NPL-1,2,3,'j1 eta',' ','LOG')
c
      call multitop(NPL+l+ 9,NPL-1,2,3,'j2 pt',' ','LOG')
      call multitop(NPL+l+10,NPL-1,2,3,'j2 log pt',' ','LOG')
      call multitop(NPL+l+11,NPL-1,2,3,'j2 y',' ','LOG')
      call multitop(NPL+l+12,NPL-1,2,3,'j2 eta',' ','LOG')
c
      call multitop(NPL+l+13,NPL-1,2,3,'bj1 pt',' ','LOG')
      call multitop(NPL+l+14,NPL-1,2,3,'bj1 log pt',' ','LOG')
      call multitop(NPL+l+15,NPL-1,2,3,'bj1 y',' ','LOG')
      call multitop(NPL+l+16,NPL-1,2,3,'bj1 eta',' ','LOG')
c
      call multitop(NPL+l+17,NPL-1,2,3,'bj2 pt',' ','LOG')
      call multitop(NPL+l+18,NPL-1,2,3,'bj2 log pt',' ','LOG')
      call multitop(NPL+l+19,NPL-1,2,3,'bj2 y',' ','LOG')
      call multitop(NPL+l+20,NPL-1,2,3,'bj2 eta',' ','LOG')
c
      call multitop(NPL+l+21,NPL-1,2,3,'syst pt',' ','LOG')
      call multitop(NPL+l+22,NPL-1,2,3,'syst log pt',' ','LOG')
      call multitop(NPL+l+23,NPL-1,2,3,'syst y',' ','LOG')
      call multitop(NPL+l+24,NPL-1,2,3,'syst eta',' ','LOG')
c
      enddo
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C     BASED ON AN ANALYSIS FILE WRITTEN BY E.RE
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
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
C INCOMING PARTONS MAY TRAVEL IN THE SAME DIRECTION: IT''S A POWER-SUPPRESSED
C EFFECT, SO THROW THE EVENT AWAY
      IF(SIGN(1.D0,PHEP(3,1)).EQ.SIGN(1.D0,PHEP(3,2)))THEN
         WRITE(*,*)'WARNING 111 IN HWANAL'
         GOTO 999
      ENDIF
      DO I=1,nwgt_analysis
         WWW(I)=EVWGT*ww(i)/ww(1)
      ENDDO

C INITIALIZE
      NT=0
      NB=0
      NBJET=0
      NTRACKS=0
      NJET=0

      DO IHEP=1,NHEP
         IST=ISTHEP(IHEP)      
         ID=IDHEP(IHEP)
         ID1=IHADR(ID) ! equal to the PDG of the massive quark in hadron
C TOP
        IF(ABS(ID).EQ.6) THEN
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
         do kk=1,nwgt_analysis
            l=(kk-1)*48+(i-1)*24
            call mfill(l+1,pttop,www(kk))
            if(pttop.gt.0d0) call mfill(l+2,log10(pttop),www(kk))
            call mfill(l+3,ytop,www(kk))
            call mfill(l+4,etatop,www(kk))
            if(njet.ge.1)then
               call mfill(l+5,ptj1,www(kk))
               if (ptj1.gt.0d0) call mfill(l+6,log10(ptj1),www(kk))
               call mfill(l+7,yj1,www(kk))
               call mfill(l+8,etaj1,www(kk))
               call mfill(l+21,ptsyst,www(kk))
               if(ptsyst.gt.0d0) call mfill(l+22,log10(ptsyst),www(kk))
               call mfill(l+23,ysyst,www(kk))
               call mfill(l+24,etasyst,www(kk))
            endif
            if(njet.ge.2)then
               call mfill(l+9,ptj2,www(kk))
               if(ptj2.gt.0d0) call mfill(l+10,log10(ptj2),www(kk))
               call mfill(l+11,yj2,www(kk))
               call mfill(l+12,etaj2,www(kk))
            endif
            if(nbjet.ge.1)then
               call mfill(l+13,ptbj1,www(kk))
               if (ptbj1.gt.0d0) call mfill(l+14,log10(ptbj1),www(kk))
               call mfill(l+15,ybj1,www(kk))
               call mfill(l+16,etabj1,www(kk))
            endif
            if(nbjet.ge.2)then
               call mfill(l+17,ptbj2,www(kk))
               if (ptbj2.gt.0d0) call mfill(l+18,log10(ptbj2),www(kk))
               call mfill(l+19,ybj2,www(kk))
               call mfill(l+20,etabj2,www(kk))
            endif
         enddo
      enddo
      
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




