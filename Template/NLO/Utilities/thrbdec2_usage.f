c From madfks_plot.ttlep5.f (used in 1407.2763)
      subroutine outfun(pp,ybst_til_tolab,www,itype)
.....
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
      real*8 rndec(10)
      common/crndec/rndec
      real*8 wgt1,wgt2,wwwd
      real*8 ptop(5),ptbr(5),xpbq(5),xplp(5),xpnu(5),xpbb(5),
     # xplm(5),xpnb(5)
      integer maxnum,imode,iunwgt,itqrkseed,itbarseed,ntr
      parameter (maxnum=10000000)
      parameter (imode=1)
c iunwgt=0 for weighted decays, iunwgt=1 for unweighted decays
      parameter (iunwgt=1)
      logical xext
      parameter (xext=.true.)
      double precision wmass

.....

c Stable tops
c Assume t is particle #3, tbar is particle #4; light parton is #5
      do i=1,4
        ptop(i)=pplab(mod(i,4),3)
        ptbr(i)=pplab(mod(i,4),4)
      enddo
      ptop(5)=pmass(3)
      ptbr(5)=pmass(4)
c
      wmass=MW
      itqrkseed=max(1,int(maxnum*rndec(1)))
      itbarseed=max(1,int(maxnum*rndec(2)))
      call thrbdec2(xext,imode,iunwgt,itqrkseed,ntr,
     #              rndec(1),rndec(2),rndec(3),rndec(4),rndec(5),
     #              wmass,ptop,xpbq,xplp,xpnu,wgt1)
      call thrbdec2(xext,imode,iunwgt,itbarseed,ntr,
     #              rndec(6),rndec(7),rndec(8),rndec(9),rndec(10),
     #              wmass,ptbr,xpbb,xplm,xpnb,wgt2)
      wwwd=wgt1*wgt2


c
c
c
c Package for top decay; the b and the leptons are taken to be massless
c This is an upgrade of the package written for Wt production, in that it
c gives one the possibility of using an external random number generator
c in the case of weighted events
c
      subroutine thrbdec2(xext,imode,iunwgt,iseed,ntr,rnd1,rnd2,rnd3,
     #                    rnd4,rnd5,xmw,xtq,xbq,xel,xnu,wgt)
c Inputs:
c   xext=.true. -> uses random numbers rnd1..rnd5, and ignores iseed
c   imode=0 -> phase space only
c   imode=1 -> top decay matrix element * phase space
c   iunwgt=0 -> weighted events
c   iunwgt=1 -> unweighted events
c   iseed: random number seed
c   xmw: W mass
c   xtq: top four momentum
c Outputs:
c   xbq,xel,xnu: b quark, charged lepton, neutrino four momenta
c   wgt: event weight; set to one if iunwgt=1
c   ntr: the number of trials in the unweighted-event mode
c The four momenta are arrays mom(i), with i=1,..,5, corresponding
c to (px,py,pz,E,m)
c
c The matrix element squared for top decay, averaged over top spin, is
c   |M|^2=2*g_W^4*|V_tb|^2*(top.el)*(b.nu)/( (Q2-mw^2)^2+(mw*wwidth)^2 )
c (where Q2=2*el.nu is the W invariant mass), and the three-body phase-space
c   dPhi_3=dQ2/(2*pi)*dPhi_2(t;W,b)*dPhi_2(W;el,nu)
c   dPhi_2(t;W,b)=1/(8*(2*pi)**2)*(mt**2-Q2)/mt**2
c   dPhi_2(W;el,nu)=1/(8*(2*pi)**2)
c with the two two-body phase spaces computed in the t and W rest frames
c respectively. In the weighted-event mode, the factor g_W^4*|V_tb|^2
c is NOT included in wgt.
c Since the phase-space variables are generated within this routine,
c and only a seed or random numbers are given in input, the weight wgt 
c includes the jacobian relevant to the definition of these variables
      implicit none
      logical xext
      integer imode,iunwgt,iseed,ntr
      real*8 rnd1,rnd2,rnd3,rnd4,rnd5,xmw,wgt,xtq(5),
     # xbq(5),xel(5),xnu(5)
      real*8 pi,pi2,wwidth,tiny,xmt,xmt2,xmw2,xmwlow2,xmwupp2,phspfact,
     # bwmdpl,bwmdmn,bwfmpl,bwfmmn,bwdelf,thbdrandom,cth1,phi1,cth2,
     # phi2,qw2,qw,drk2,thrbdec_dot,dpk1,decwgt,thrbdec_ibw,evwgt,
     # bound,rat,rrnd,xx(5),xtq0(5),xbq0(5),xel0(5),xnu0(5)
      integer i,ilocseed,maxtry
      parameter (pi=3.14159265358979312D0)
      parameter (pi2=pi*pi)
      parameter (wwidth=2.141d0)
      parameter (tiny=1.d-1)
      parameter (maxtry=10000000)
c
      if( (imode.ne.0.and.imode.ne.1) .or.
     #    (iunwgt.ne.0.and.iunwgt.ne.1) )then
        write(*,*)'Error in thrbdec2: unknown options',imode,iunwgt
        stop
      endif
      ilocseed=iseed
      xmt=xtq(5)
      xmt2=xmt**2
      xmw2=xmw**2
      if(xmt.le.tiny)then
        write(*,*)'Error in thrbdec2: mass too small',xmt
        stop
      endif
      xmwlow2=tiny**2
      xmwupp2=(xmt-tiny)**2
      phspfact=1/(2*pi)*(1/(32*pi2))**2
      if(imode.eq.1.and.iunwgt.eq.1)then
        bwmdpl=xmwupp2-xmw2
        bwmdmn=xmw2-xmwlow2
        bwfmpl=atan(bwmdpl/(xmw*wwidth))
        bwfmmn=atan(bwmdmn/(xmw*wwidth))
        bwdelf=(bwfmpl+bwfmmn)/pi
      endif
      ntr=0
 100  continue
      if(xext.and.iunwgt.eq.0)then
        xx(1)=rnd1
        xx(2)=rnd2
        xx(3)=rnd3
        xx(4)=rnd4
        xx(5)=rnd5
      else
        do i=1,5
          xx(i)=thbdrandom(ilocseed)
        enddo
      endif
c Prevents numerical inaccuracies inherited from rnd generation
c from affecting the rest of this routine
      do i=1,5
        if(xx(i).le.0.d0)xx(i)=0.d0
        if(xx(i).ge.1.d0)xx(i)=1.d0
      enddo
      cth1=-1+2*xx(1)
      phi1=2*pi*xx(2)
      cth2=-1+2*xx(3)
      phi2=2*pi*xx(4)
c
      if(iunwgt.eq.0)then
        qw2=xmwlow2+xx(5)*(xmwupp2-xmwlow2)
        qw=sqrt(qw2)
        call thrbdec_kin(xmt,qw,cth1,phi1,cth2,phi2,
     #                   xtq0,xbq0,xel0,xnu0)
        wgt=phspfact*(xmt2-qw2)/xmt2
        wgt=wgt*(xmwupp2-xmwlow2)*(2*2*pi)**2
        if(imode.eq.1)then
          drk2=thrbdec_dot(xbq0,xnu0)
          dpk1=thrbdec_dot(xtq0,xel0)
          decwgt=2*drk2*dpk1/((qw2-xmw2)**2+(xmw*wwidth)**2)
          wgt=wgt*decwgt
        endif
      else
        wgt=1
        ntr=ntr+1
        if(ntr.gt.maxtry)then
          wgt=0
          return
        endif
        if(imode.eq.0)then
c Inverse N*\int_xmwlow2^Q2 dqw2 (xmt2-qw2)/xmt2 with xmwlow2 < Q2 < xmwupp2,
c and N a normalization factor such that N*\int = 1 when Q2=xmwupp2
          qw2=xmt2-sqrt( xmt2**2-
     #                   2*xmt2*(xmwlow2+xx(5)*(xmwupp2-xmwlow2))+
     #                   xmwlow2**2*(1-xx(5))+xmwupp2**2*xx(5) )
          qw=sqrt(qw2)
          call thrbdec_kin(xmt,qw,cth1,phi1,cth2,phi2,
     #                     xtq0,xbq0,xel0,xnu0)
        elseif(imode.eq.1)then
          qw2=thrbdec_ibw(xx(5),xmw2,wwidth,bwdelf,bwfmmn)
          qw=sqrt(qw2)
          call thrbdec_kin(xmt,qw,cth1,phi1,cth2,phi2,
     #                     xtq0,xbq0,xel0,xnu0)
          drk2=thrbdec_dot(xbq0,xnu0)
          dpk1=thrbdec_dot(xtq0,xel0)
c The event weight (evwgt) and the upper bound (bound) should both contain
c a factor 1/((qw2-xmw2)**2+(xmw*wwidth)**2), and trivial phase-space 
c normalizations, which are not included since they cancel in the ratio.
c The Breit-Wigner must not be included since qw2 is generated with the
c inverse of its integral function
          evwgt=drk2*dpk1*(xmt2-qw2)/xmt2
          bound=xmt2**2/16.d0
          rat=evwgt/bound
          if(rat.lt.0.d0.or.rat.gt.1.d0)then
            write(*,*)'Error in thrbdec2: bound violation'
            write(*,*)evwgt,bound
            stop
          endif
          rrnd=thbdrandom(ilocseed)
          if(rat.lt.rrnd)goto 100
        endif
      endif
c
      if(xtq(4)-xtq(5).lt.1.d-8)then
        do i=1,5
          xbq(i)=xbq0(i)
          xel(i)=xel0(i)
          xnu(i)=xnu0(i)
        enddo
      else
        call thrb_hwulob(xtq,xbq0,xbq)
        call thrb_hwulob(xtq,xel0,xel)
        call thrb_hwulob(xtq,xnu0,xnu)
      endif
c
      return
      end


      function thrbdec_dot(q1,q2)
c Dot product
      implicit none
      real*8 thrbdec_dot,q1(5),q2(5)
c
      thrbdec_dot=q1(4)*q2(4)-q1(1)*q2(1)-q1(2)*q2(2)-q1(3)*q2(3)
      return
      end


      function thrbdec_ibw(t,xm02,ga,bwdelf,bwfmmn)
c Returns the mass squared, given 0<t<1, the pole mass squared (xm02),
c the width (ga), and the mass range (implicit in bwdelf and bwfmmn). 
c This function is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c This routine has been derived from xbwmass3 of MC@NLO
      implicit none
      real*8 thrbdec_ibw,t,xm02,ga,bwdelf,bwfmmn
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      thrbdec_ibw=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
      return
      end



      subroutine thrbdec_kin(xmt,xmw,cth1,phi1,cth2,phi2,
     #                       xtq,xbq,xel,xnu)
c Generates the four-momenta of the decay products of the top. These
c four-momenta are returned in the top rest frame (xbq, xel, xnu; the
c trivial top momentum is returned as well, xtq). 
c The inputs of the routine are cth1,phi1,cth2,phi2, which are cosines of
c polar angles and azimuthal angles, with
c   (cth1,phi1) --> direction of W in the top rest frame
c   (cth2,phi2) --> direction of l+ in the W rest frame
c This routine has been derived from gentopdmom of MC@NLO
      implicit none
      real*8 xmt,xmw,cth1,phi1,cth2,phi2,xtq(5),xbq(5),xel(5),xnu(5)
      real*8 xmt2,xmw2,sth1,sth2,ew,eb,pwx,pwy,pwz,pbx,pby,pbz,eel,
     # enu,pex,pey,pez,pnx,pny,pnz,thrb_sfromc,tmp(5),tmp1(5)
c
      xmt2=xmt**2
      xmw2=xmw**2
      sth1=thrb_sfromc(cth1)
      sth2=thrb_sfromc(cth2)
c
      xtq(1)=0.d0
      xtq(2)=0.d0
      xtq(3)=0.d0
      xtq(4)=xmt
      xtq(5)=xmt
c W and b momenta, top rest frame
      ew=(xmt2+xmw2)/(2*xmt)
      eb=(xmt2-xmw2)/(2*xmt)
      pwx=eb*sth1*cos(phi1)
      pwy=eb*sth1*sin(phi1)
      pwz=eb*cth1
      pbx=-pwx
      pby=-pwy
      pbz=-pwz
      xbq(1)=pbx
      xbq(2)=pby
      xbq(3)=pbz
      xbq(4)=eb
      xbq(5)=0.d0
c l+ and nu momenta, W rest frame
      eel=xmw/2.d0
      enu=eel
      pex=eel*sth2*cos(phi2)
      pey=eel*sth2*sin(phi2)
      pez=eel*cth2
      pnx=-pex
      pny=-pey
      pnz=-pez
c Boost lepton momenta to top rest frame
      tmp(1)=pwx
      tmp(2)=pwy
      tmp(3)=pwz
      tmp(4)=ew
      tmp(5)=xmw
c Boost l+
      tmp1(1)=pex
      tmp1(2)=pey
      tmp1(3)=pez
      tmp1(4)=eel
      tmp1(5)=0.d0
      call thrb_hwulob(tmp,tmp1,xel)
c Boost nu
      tmp1(1)=pnx
      tmp1(2)=pny
      tmp1(3)=pnz
      tmp1(4)=enu
      tmp1(5)=0.d0
      call thrb_hwulob(tmp,tmp1,xnu)
c
      return
      end


      function thrb_sfromc(c)
c Extract sine [with positive sign] from cosine 
c avoiding numerical inaccuracies
      implicit none
      real*8 thrb_sfromc,c,tiny,s
      parameter (tiny=1.d-5)
c
      if((abs(c)-1.d0).gt.tiny)then
        write(*,*)'Error in thrb_sfromc',c
        stop
      elseif((abs(c)-1.d0).ge.0.d0.and.(abs(c)-1.d0).le.tiny)then
        s=0.d0
      else
        s=sqrt(1-c**2)
      endif
      thrb_sfromc=s
      return
      end


      FUNCTION THBDRANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,THBDRANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      THBDRANDOM = SEED*MINV
      END
c
c The routines that follow have been taken from herwig6510.f; names have
c been changed to avoid multiple-definition problems when linking
c
C-----------------------------------------------------------------------
      SUBROUTINE THRB_HWULOB(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN REST FRAME OF PS) INTO PF (IN LAB)
C     N.B. P(1,2,3,4,5) = (PX,PY,PZ,E,M)
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION PS(5),PI(5),PF(5)
      CALL THRB_HWULB4(PS,PI,PF)
      PF(5)= PI(5)
      END


C-----------------------------------------------------------------------
      SUBROUTINE THRB_HWULOF(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN LAB) INTO PF (IN REST FRAME OF PS)
C     N.B. P(1,2,3,4,5) = (PX,PY,PZ,E,M)
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION PS(5),PI(5),PF(5)
      CALL THRB_HWULF4(PS,PI,PF)
      PF(5)= PI(5)
      END


C-----------------------------------------------------------------------
      SUBROUTINE THRB_HWULB4(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN REST FRAME OF PS) INTO PF (IN LAB)
C     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION PF4,FN,PS(5),PI(4),PF(4)
      IF (PS(4).EQ.PS(5)) THEN
        PF(1)= PI(1)
        PF(2)= PI(2)
        PF(3)= PI(3)
        PF(4)= PI(4)
      ELSE
        PF4  = (PI(1)*PS(1)+PI(2)*PS(2)
     &         +PI(3)*PS(3)+PI(4)*PS(4))/PS(5)
        FN   = (PF4+PI(4)) / (PS(4)+PS(5))
        PF(1)= PI(1) + FN*PS(1)
        PF(2)= PI(2) + FN*PS(2)
        PF(3)= PI(3) + FN*PS(3)
        PF(4)= PF4
      END IF
      END


C-----------------------------------------------------------------------
      SUBROUTINE THRB_HWULF4(PS,PI,PF)
C-----------------------------------------------------------------------
C     TRANSFORMS PI (GIVEN IN LAB) INTO PF (IN REST FRAME OF PS)
C     N.B. P(1,2,3,4) = (PX,PY,PZ,E); PS(5)=M
C-----------------------------------------------------------------------
      IMPLICIT NONE
      DOUBLE PRECISION PF4,FN,PS(5),PI(4),PF(4)
      IF (PS(4).EQ.PS(5)) THEN
        PF(1)= PI(1)
        PF(2)= PI(2)
        PF(3)= PI(3)
        PF(4)= PI(4)
      ELSE
        PF4  = (PI(4)*PS(4)-PI(3)*PS(3)
     &         -PI(2)*PS(2)-PI(1)*PS(1))/PS(5)
        FN   = (PF4+PI(4)) / (PS(4)+PS(5))
        PF(1)= PI(1) - FN*PS(1)
        PF(2)= PI(2) - FN*PS(2)
        PF(3)= PI(3) - FN*PS(3)
        PF(4)= PF4
      END IF
      END
