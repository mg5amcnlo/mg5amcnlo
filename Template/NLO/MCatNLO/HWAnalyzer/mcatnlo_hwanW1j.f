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
      real*4 pi,xmWlow,xmWupp,xmwbin
      real*8 wmass,wwidth
      parameter (pi=3.14160E0)
      integer j,k,jpr
      character*4 cc(10)
      data cc/' 2.5','   5','  10','  25','  50',
     &        'r2.5','r  5','r 10','r 25','r 50'/
c
      wmass=RMASS(198)
      wwidth=GAMW
      xmWlow=sngl(wmass-wwidth*30.5d0)
      xmWupp=sngl(wmass+wwidth*30.5d0)
      xmwbin=sngl(wwidth)
      call inihist
c
      call mbook(1,'xsec',1.e0,-0.5e0,12.5e0)
      
      do j=1,10
      k=(j-1)*17
c
      call mbook(k+ 2,'pt[e+]     '//cc(j),5.e0,0.e0,400.e0)
      call mbook(k+ 3,'eta[e+]    '//cc(j),0.1e0,-5.e0,5.e0)
      call mbook(k+ 4,'pt[ne]     '//cc(j),5.e0,0.e0,400.e0)
      call mbook(k+ 5,'eta[ne]    '//cc(j),0.1e0,-5.e0,5.e0)
      call mbook(k+ 6,'pt[j1]     '//cc(j),2.5e0,0.e0,250.e0)
      call mbook(k+ 7,'eta[j1]    '//cc(j),0.1e0,-5.e0,5.e0)
      call mbook(k+ 8,'pt[j2]     '//cc(j),2.5e0,0.e0,250.e0)
      call mbook(k+ 9,'eta[j2]    '//cc(j),0.1e0,-5.e0,5.e0)
      call mbook(k+10,'M[j]       '//cc(j),1.e0,0.e0,100.e0)
      call mbook(k+11,'M[en]      '//cc(j),xmwbin,xmWlow,xmWupp)
      call mbook(k+12,'pt[en]     '//cc(j),5.e0,0.e0,400.e0)
      call mbook(k+13,'DelR[en]   '//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+14,'dphi[en]   '//cc(j),pi/50e0,0e0,pi)
      call mbook(k+15,'M[j12]     '//cc(j),5.e0,0.e0,400.e0)
      call mbook(k+16,'pt[j12]    '//cc(j),5.e0,0.e0,400.e0)
      call mbook(k+17,'DelR[j12]  '//cc(j),pi/20.e0,0.e0,3*pi)
      call mbook(k+18,'dphi[j12]  '//cc(j),pi/50e0,0e0,pi)

      enddo
 999  END
C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I,J,K
      OPEN(UNIT=99,NAME='HERW.top',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,500              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+500)
        CALL MOPERA(I+500,'F',I+500,I+500,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+500)             
      ENDDO                          
C
      call multitop(500+1,499,3,2,'total rate',' ','LOG')

      do j=1,10
         k=(j-1)*17
         call multitop(500+k+ 2,499,3,2,'pt[e+]     ',' ','LOG')
         call multitop(500+k+ 3,499,3,2,'eta[e+]    ',' ','LOG')
         call multitop(500+k+ 4,499,3,2,'pt[ne]     ',' ','LOG')
         call multitop(500+k+ 5,499,3,2,'eta[ne]    ',' ','LOG')
         call multitop(500+k+ 6,499,3,2,'pt[j1]     ',' ','LOG')
         call multitop(500+k+ 7,499,3,2,'eta[j1]    ',' ','LOG')
         call multitop(500+k+ 8,499,3,2,'pt[j2]     ',' ','LOG')
         call multitop(500+k+ 9,499,3,2,'eta[j2]    ',' ','LOG')
         call multitop(500+k+10,499,3,2,'M[j]       ',' ','LOG')
         call multitop(500+k+11,499,3,2,'M[en]      ',' ','LOG')
         call multitop(500+k+12,499,3,2,'pt[en]     ',' ','LOG')
         call multitop(500+k+13,499,3,2,'DelR[en]   ',' ','LOG')
         call multitop(500+k+14,499,3,2,'dphi[en]   ',' ','LOG')
         call multitop(500+k+15,499,3,2,'M[j12]     ',' ','LOG')
         call multitop(500+k+16,499,3,2,'pt[j12]    ',' ','LOG')
         call multitop(500+k+17,499,3,2,'DelR[j12]  ',' ','LOG')
         call multitop(500+k+18,499,3,2,'dphi[j12]  ',' ','LOG')
      enddo
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4),PJJ(4)
      INTEGER ICHSUM,ICHINI,IHEP,IST,ID,IJ,IE(30),INU(30),J,NE,NNU,
     &     NN,I
      LOGICAL DIDSOF
      double precision PE(5),PNU(5),pw(5),getptv,getpseudorapv,ptenu,
     &     drenu,getdrv,dphienu,getdelphiv,getptv4,getpseudorapv4,
     &     getinvmv, getinvmv4,getdelphiv4,getdrv4,pte,etae,ptnu,etanu
c jet stuff
      INTEGER NMAX
      PARAMETER (NMAX=2000)
      INTEGER NJET,JET(NMAX),IPOS(NMAX)
      DOUBLE PRECISION PALG,RFJ,SYCUT,PP(4,NMAX),PJET(4,NMAX),
     # PTJET(NMAX),ETAJET(NMAX)
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
      INTEGER KK,J1,J2
      
c
      IF (IERROR.NE.0) RETURN
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
      ne=0
      nnu=0
      nn=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
        IST=ISTHEP(IHEP)      
        ID=IDHEP(IHEP)
        IF(ABS(ID).EQ.11.AND.IST.EQ.1)THEN
           NE=NE+1
           IE(NE)=IHEP
        ELSEIF(ABS(ID).EQ.12.AND.IST.EQ.1)THEN
           NNU=NNU+1
           INU(NNU)=IHEP
        ELSEIF(ABS(ID).GT.100.AND.IST.EQ.1) THEN
           NN=NN+1
           IF (NN.GT.NMAX) STOP 'Too many particles [hadrons]!'
           IPOS(NN)=IHEP
           DO I=1,4
              PP(I,NN)=PHEP(I,IHEP)
           ENDDO
        ENDIF
  100 CONTINUE
      IF( (NE.EQ.0.OR.NNU.EQ.0).AND.IERROR.EQ.0) THEN
         CALL HWUEPR
         WRITE(*,*)NE,NNU
         CALL HWWARN('HWANAL',503)
      ENDIF
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
c Take first leptons as they appear in the event record
      DO IJ=1,5
         PE(IJ)=PHEP(IJ,IE(1))
         PNU(IJ)=PHEP(IJ,INU(1))
         IF(IJ.LE.4)THEN
            PW(IJ)=PE(IJ)+PNU(IJ)
         ENDIF
      ENDDO
      PE(5)=0.d0
      PNU(5)=0.d0
      PW(5)=getinvmv(PW)
c Lepton pair variables and acceptances
      pte=getptv(pe)
      etae=getpseudorapv(pe)
      ptnu=getptv(pnu)
      etanu=getpseudorapv(pnu)
      ptenu=getptv(pw)
      drenu=getdrv(pe,pnu)
      dphienu=getdelphiv(pe,pnu)

C---CLUSTER THE EVENT
      palg=-1.d0
      rfj=0.4d0
      sycut=2.5d0
      call fastjetppgenkt(pp,nn,rfj,sycut,palg,pjet,njet,jet)
c Check order in pt
      j1=0
      j2=0
      do i=1,njet
         ptjet(i)=getptv4(pjet(1,i))
         if(i.gt.1)then
            if (ptjet(i).gt.ptjet(i-1)) then
               write (*,*) "Error in fastjet: "//
     &              "jets should be ordered in pt"
               CALL HWWARN('HWANAL',501)
            endif
         endif
         etajet(i)=getpseudorapv4(pjet(1,i))
         if (abs(etajet(i)).lt.3d0 .and. j1.eq.0) then
            j1=i
         elseif (abs(etajet(i)).lt.3d0 .and. j1.ne.0 .and. j2.eq.0) then
            j2=i
         endif
      enddo

      if (njet.le.0) return

      if (ptjet(1).gt.2.5d0) call mfill(1,float(1),sngl(www0))
      if (ptjet(1).gt.  5d0) call mfill(1,float(2),sngl(www0))
      if (ptjet(1).gt. 10d0) call mfill(1,float(3),sngl(www0))
      if (ptjet(1).gt. 25d0) call mfill(1,float(4),sngl(www0))
      if (ptjet(1).gt. 50d0) call mfill(1,float(5),sngl(www0))

      if (j1.ne.0) then
         if (ptjet(j1).gt.2.5d0) call mfill(1,float( 6),sngl(www0))
         if (ptjet(j1).gt.  5d0) call mfill(1,float( 7),sngl(www0))
         if (ptjet(j1).gt. 10d0) call mfill(1,float( 8),sngl(www0))
         if (ptjet(j1).gt. 25d0) call mfill(1,float( 9),sngl(www0))
         if (ptjet(j1).gt. 50d0) call mfill(1,float(10),sngl(www0))
      endif

      do j=1,5
         kk=17*(j-1)
         if (ptjet(1).lt.2.5d0 .and. j.eq.1) exit
         if (ptjet(1).lt.  5d0 .and. j.eq.2) exit
         if (ptjet(1).lt. 10d0 .and. j.eq.3) exit
         if (ptjet(1).lt. 25d0 .and. j.eq.4) exit
         if (ptjet(1).lt. 50d0 .and. j.eq.5) exit
         call mfill(kk+ 2,sngl(pte),sngl(www0))
         call mfill(kk+ 3,sngl(etae),sngl(www0))
         call mfill(kk+ 4,sngl(ptnu),sngl(www0))
         call mfill(kk+ 5,sngl(etanu),sngl(www0))
         call mfill(kk+ 6,sngl(ptjet(1)),sngl(www0))
         call mfill(kk+ 7,sngl(etajet(1)),sngl(www0))
         call mfill(kk+10,sngl(getinvmv4(pjet(1,1))),sngl(www0))
         call mfill(kk+11,sngl(pw(5)),sngl(www0))
         call mfill(kk+12,sngl(ptenu),sngl(www0))
         call mfill(kk+13,sngl(drenu),sngl(www0))
         call mfill(kk+14,sngl(dphienu),sngl(www0))
         if (njet.lt.2) cycle
         if (ptjet(2).lt. 2.5d0 .and. j.eq.1) cycle
         if (ptjet(2).lt.   5d0 .and. j.eq.2) cycle
         if (ptjet(2).lt.  10d0 .and. j.eq.3) cycle
         if (ptjet(2).lt.  25d0 .and. j.eq.4) cycle
         if (ptjet(2).lt.  50d0 .and. j.eq.5) cycle
         do i=1,4
            pjj(i)=pjet(i,1)+pjet(i,2)
         enddo
         call mfill(kk+ 8,sngl(ptjet(2)),sngl(www0))
         call mfill(kk+ 9,sngl(etajet(2)),sngl(www0))
         call mfill(kk+15,sngl(getinvmv4(pjj)),sngl(www0))
         call mfill(kk+16,sngl(getptv4(pjj)),sngl(www0))
         call mfill(kk+17,sngl(getdrv4(pjet(1,1),pjet(1,2))),
     &        sngl(www0))
         call mfill(kk+18,sngl(getdelphiv4(pjet(1,1),pjet(1,2))),
     &        sngl(www0))
      enddo


      if (j1.eq.0) return
      do j=1,5
         kk=17*(j-1+5)
         if (ptjet(j1).lt.2.5d0 .and. j.eq.1) exit
         if (ptjet(j1).lt.  5d0 .and. j.eq.2) exit
         if (ptjet(j1).lt. 10d0 .and. j.eq.3) exit
         if (ptjet(j1).lt. 25d0 .and. j.eq.4) exit
         if (ptjet(j1).lt. 50d0 .and. j.eq.5) exit
         call mfill(kk+ 2,sngl(pte),sngl(www0))
         call mfill(kk+ 3,sngl(etae),sngl(www0))
         call mfill(kk+ 4,sngl(ptnu),sngl(www0))
         call mfill(kk+ 5,sngl(etanu),sngl(www0))
         call mfill(kk+ 6,sngl(ptjet(j1)),sngl(www0))
         call mfill(kk+ 7,sngl(etajet(j1)),sngl(www0))
         call mfill(kk+10,sngl(getinvmv4(pjet(1,j1))),sngl(www0))
         call mfill(kk+11,sngl(pw(5)),sngl(www0))
         call mfill(kk+12,sngl(ptenu),sngl(www0))
         call mfill(kk+13,sngl(drenu),sngl(www0))
         call mfill(kk+14,sngl(dphienu),sngl(www0))
         if (j2.eq.0) cycle
         if (ptjet(j2).lt. 2.5d0 .and. j.eq.1) cycle
         if (ptjet(j2).lt.   5d0 .and. j.eq.2) cycle
         if (ptjet(j2).lt.  10d0 .and. j.eq.3) cycle
         if (ptjet(j2).lt.  25d0 .and. j.eq.4) cycle
         if (ptjet(j2).lt.  50d0 .and. j.eq.5) cycle
         do i=1,4
            pjj(i)=pjet(i,j1)+pjet(i,j2)
         enddo
         call mfill(kk+ 8,sngl(ptjet(j2)),sngl(www0))
         call mfill(kk+ 9,sngl(etajet(j2)),sngl(www0))
         call mfill(kk+15,sngl(getinvmv4(pjj)),sngl(www0))
         call mfill(kk+16,sngl(getptv4(pjj)),sngl(www0))
         call mfill(kk+17,sngl(getdrv4(pjet(1,j1),pjet(1,j2))),
     &        sngl(www0))
         call mfill(kk+18,sngl(getdelphiv4(pjet(1,j1),pjet(1,j2))),
     &        sngl(www0))
      enddo


 999  END


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
