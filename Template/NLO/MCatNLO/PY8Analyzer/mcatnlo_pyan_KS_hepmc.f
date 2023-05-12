c
c Example analysis for Kolmogorov Smirnov tests
c
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG(nnn,wwwi)
C     USER''S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      integer j,l,i,k,nnn,ipl,jpl
      integer nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*50 weights_info(max_weight_shower),wwwi(max_weight_shower)
      include 'process.inc'
c      
      call inihist
      weights_info(1)="central value  "
      do i=1,nnn+1
         weights_info(i+1)=wwwi(i)
      enddo
      nwgt=nnn+1
      nwgt_analysis=nwgt
c
      do k=1,nwgt_analysis
         l=(k-1)*2*(n_Born**2+n_Born+1)
         call mbook(l+1,'pT  system '//weights_info(k),5d0,5d0,500d0)
         call mbook(l+2,'eta system '//weights_info(k),0.24d0,-6d0,6d0)
         do ipl=1,n_Born
            i=l+2+(ipl-1)*2*(n_Born+1)
            call mbook(i+1,'pT  '//str_pdg(ipl)//' '//weights_info(k),5d0,5d0,500d0)
            call mbook(i+2,'eta '//str_pdg(ipl)//' '//weights_info(k),0.24d0,-6d0,6d0)
            do jpl=ipl+1,n_Born
               j=i+2+(jpl-1)*2
               call mbook(j+1,'m  '//str_pdg(ipl)//str_pdg(jpl)//' '//weights_info(k),5d0,5d0,500d0)
               call mbook(j+2,'DR '//str_pdg(ipl)//str_pdg(jpl)//' '//weights_info(k),0.1d0,0d0,10d0)
            enddo
         enddo
      enddo
c
 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVTTOT)
C     USER''S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      REAL*8 XNORM,IEVTTOT
      INTEGER I,J,KK,l,nwgt_analysis
      integer NPL
      parameter(NPL=15000)
      common/c_analysis/nwgt_analysis
      include 'process.inc'
      OPEN(UNIT=99,FILE='KS.top',STATUS='UNKNOWN')
c
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=IEVTTOT/DFLOAT(NEVHEP)
      DO I=1,NPL
        CALL MFINAL3(I)
        CALL MCOPY(I,I+NPL)
        CALL MOPERA(I+NPL,'F',I+NPL,I+NPL,(XNORM),0.D0)
        CALL MFINAL3(I+NPL)
      ENDDO
c
      do k=1,nwgt_analysis
         l=(k-1)*2*(n_Born**2+n_Born+1)
         call multitop(NPL+l+1,NPL-1,3,2,'pT  system',' ','LOG')
         call multitop(NPL+l+2,NPL-1,3,2,'eta system',' ','LOG')
         do ipl=1,n_Born
            i=l+2+(ipl-1)*2*(n_Born+1)
            call multitop(NPL+i+1,NPL-1,3,2,'pT  '//str_pdg(ipl),' ','LOG')
            call multitop(NPL+i+2,NPL-1,3,2,'eta '//str_pdg(ipl),' ','LOG')
            do jpl=ipl+1,n_Born
               j=i+2+(jpl-1)*2
               call multitop(NPL+j+1,NPL-1,3,2,'m  '//str_pdg(ipl)//str_pdg(jpl),' ','LOG')
               call multitop(NPL+j+2,NPL-1,3,2,'DR '//str_pdg(ipl)//str_pdg(jpl),' ','LOG')
            enddo
         enddo
      enddo
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL(nnn,xww)
C     USER''S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HEPMC.INC'
      include 'reweight0.inc'
      DOUBLE PRECISION PSUM(4),PJJ(4)
      INTEGER IHEP,IST,ID,IJ,J,NN,I,jj,l,mu
      double precision getpt,getpseudorap
c
c jet declarations
      INTEGER NMAX
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX)
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX),YJET(NMAX),RAPJET(NMAX),
     # P_PART(4,10),P_SYSTEM(4),PT(10),ETA(10),INV_M(10,10),
     # DELTAR(10,10),P_IJ(4)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,J1,J2,nh,ih,nj
      double precision getrapidityv4,getptv4,getinvmv4,getdelphiv4,
     &getdrv4,getpseudorapv4
      double precision njdble
      integer nwgt_analysis,max_weight
      common/c_analysis/nwgt_analysis
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision ww(max_weight),www(max_weight),xww(max_weight)
      common/cww/ww
      logical was_assigned(10),p_is_OK(10),syst_is_OK
      integer label(10)
      include 'process.inc'
c
c weight stuff
      ww(1)=xww(2)
      if(nnn.eq.0)ww(1)=1d0
      do i=2,nnn+1
         ww(i)=xww(i)
      enddo
      if (ww(1).eq.0d0) then
         write(*,*)'ww(1) = 0. Stopping'
         stop
      endif
      do i=1,nwgt_analysis
         www(i)=evwgt*ww(i)/ww(1)
      enddo
      pp=0d0
c
c loop over event particles
      NN=0
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)
        ID=IDHEP(IHEP)
c
c find QCD tracks if necessary
        if(n_jet_Born.gt.0)then
           IF(ABS(ID).GT.100.AND.IST.EQ.1) THEN
              NN=NN+1
              IF (NN.GT.NMAX) STOP 'Too many hadrons!'
              DO I=1,4
                 PP(I,NN)=PHEP(I,IHEP)
              ENDDO
           ENDIF
        endif
c
c find stable partons of the Born process
        was_assigned=.false.
        do ii=1,n_Born
           if(pdg(ii).eq.0)cycle
           if(was_assigned(ii))cycle
           if(id.eq.pdg(ii).and.ist.eq.1)then
              label(ii)=ihep
              was_assigned(ii)=.true.
           endif
        enddo
  100 CONTINUE
c
c cluster the event if necessary
      if(NN.gt.0)then
         palg =-1.d0
         rfj  =0.4d0
         sycut=20d0
c
         pjet=0d0
         ptjet=0d0
         yjet =0d0
         etajet=0d0
         jet=0
c
         call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
         do i=1,njet
            ptjet(i)=getptv4(pjet(1,i))
c
c check if pT ordering worked correctly
            if(i.gt.1)then
               if (ptjet(i).gt.ptjet(i-1)) then
                  write (*,*) "Error 1: jets should be ordered in pt"
                  WRITE(*,*)'ERROR 501 IN PYANAL'
                  STOP
               endif
            endif
         enddo
      endif
c
c load momenta
      i_jet=0
      p_part=0d0
      p_system=0d0
      p_is_OK=.true.
      syst_is_OK=.true.
      do ii=1,n_Born
         do mu=1,4
            if(pdg(ii).ne.0)then
               p_part(mu,ii)=phep(mu,label(ii))
            else
               i_jet=i_jet+1
               if(i_jet.le.njet)then
                  p_part(mu,ii)=pjet(mu,i_jet)
               else
                  p_is_OK(ii)=.false.
               endif
            endif
            if(p_is_OK(ii))p_system(mu)=p_system(mu)+p_part(mu,ii)
         enddo
         syst_is_OK=syst_is_OK.and.p_is_OK(ii)
      enddo
c
c define observables
      ptsyst=getptv4(p_system)
      etasyst=getpseudorapv4(p_system)
      do ii=1,n_Born
         pt(ii) =getptv4(p_part(1,ii))
         eta(ii)=getpseudorapv4(p_part(1,ii))
         do jj=1,n_Born
            do mu=1,4
               p_ij(mu)=p_part(mu,ii)+p_part(mu,jj)
            enddo
            inv_m(ii,jj)=getinvmv4(p_ij)
            DeltaR(ii,jj)=getdrv4(p_part(1,ii),p_part(1,jj))
         enddo 
      enddo
c
c fill histograms
      do k=1,nwgt_analysis
         l=(k-1)*2*(n_Born**2+n_Born+1)
         if(syst_is_OK)then
            call mfill(l+1,ptsyst,abs(www(k)))
            call mfill(l+2,etasyst,abs(www(k)))
         endif
         do ipl=1,n_Born
            i=l+2+(ipl-1)*2*(n_Born+1)
            if(p_is_OK(ipl))then
               call mfill(i+1,pt(ipl),abs(www(k)))
               call mfill(i+2,eta(ipl),abs(www(k)))
               do jpl=ipl+1,n_Born
                  j=i+2+(jpl-1)*2
                  if(p_is_OK(jpl))then
                     call mfill(j+1,inv_m(ipl,jpl),abs(www(k)))
                     call mfill(j+2,DeltaR(ipl,jpl),abs(www(k)))
                  endif
               enddo
            endif
         enddo
      enddo
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



      function getmod(p)
      implicit none
      double precision p(4),getmod

      getmod=sqrt(p(1)**2+p(2)**2+p(3)**2)

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





      subroutine boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
c chybstmo = chybst-1; if it can be computed analytically it improves
c the numerical accuracy
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xin(0:3),xout(0:3)
      real*8 tmp,en,pz
      integer i
c
      if(abs(xd(1)**2+xd(2)**2+xd(3)**2-1).gt.1.d-6)then
        write(*,*)'Error #1 in boostwdir2',xd
        stop
      endif
c
      en=xin(0)
      pz=xin(1)*xd(1)+xin(2)*xd(2)+xin(3)*xd(3)
      xout(0)=en*chybst-pz*shybst
      do i=1,3
        xout(i)=xin(i)+xd(i)*(pz*chybstmo-en*shybst)
      enddo
c
      return
      end




      subroutine boostwdir3(chybst,shybst,chybstmo,xd,xxin,xxout)
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xxin(4),xxout(4)
      real*8 xin(0:3),xout(0:3)
      integer i
c
      do i=1,4
         xin(mod(i,4))=xxin(i)
      enddo
      call boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
      do i=1,4
         xxout(i)=xout(mod(i,4))
      enddo
c
      return
      end




      subroutine getwedge(p1,p2,pout)
      implicit none
      real*8 p1(4),p2(4),pout(4)

      pout(1)=p1(2)*p2(3)-p1(3)*p2(2)
      pout(2)=p1(3)*p2(1)-p1(1)*p2(3)
      pout(3)=p1(1)*p2(2)-p1(2)*p2(1)
      pout(4)=0d0

      return
      end

