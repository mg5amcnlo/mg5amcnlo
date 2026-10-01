      module fks_phase_space_helpers
c Shared numerical kernels, independent of the phase-space driver and
c of kinematics_module, so recoil wrappers can import them directly.
      implicit none
      include 'nexternal.inc'
      private
      public get_massive_fsr_bounds,getangles,gentcms,lambda,yminmax,
     $     get_recoil
      contains

      subroutine get_massive_fsr_bounds(shat,sqrtshat,m_j_fks,
     $     xmrec2,y,xim,xibm,ximax,cffa2,cffb2,cffc2,cffdel2,
     $     xiimax,xinorm)
c Shared radiation bounds for the massive final-state map and its
c inverses, including a massless final or incoming recoil reservoir.
c Keep the exact massless-recoil expressions: near the two-body
c threshold the generic discriminant loses precision by cancellation.
      implicit none
      double precision shat,sqrtshat,m_j_fks,xmrec2,y,
     $     xim,xibm,ximax,cffa2,cffb2,cffc2,cffdel2,
     $     xiimax,xinorm
      double precision xmjhat,xmhat,zero_recoil_delta,xirplus,
     $     xirminus

      xmjhat=m_j_fks/sqrtshat
      xmhat=sqrt(xmrec2)/sqrtshat
      if(xmrec2.eq.0d0)then
         zero_recoil_delta=(shat-m_j_fks**2)/shat
         xim=(sqrtshat-m_j_fks)/sqrtshat
         cffa2=zero_recoil_delta+xmjhat**2*y**2
         cffb2=-2d0*zero_recoil_delta
         cffc2=zero_recoil_delta**2
         cffdel2=4d0*xmjhat**2*cffc2*(1d0-y**2)
         xibm=zero_recoil_delta/(1d0+xmjhat*
     $        sqrt(max(0d0,1d0-y**2)))
         ximax=zero_recoil_delta
      else
         xim=(1-xmhat**2-2*xmjhat+xmjhat**2)/(1-xmjhat)
         cffa2=1-xmjhat**2*(1-y**2)
         cffb2=-2*(1-xmhat**2-xmjhat**2)
         cffc2=(1-(xmhat-xmjhat)**2)*(1-(xmhat+xmjhat)**2)
         cffdel2=cffb2**2-4*cffa2*cffc2
         xibm=(-cffb2-sqrt(cffdel2))/(2*cffa2)
         ximax=1-(xmhat+xmjhat)**2
      endif
      if(y.ge.0d0)then
         xirplus=xim
         xirminus=0d0
      else
         xirplus=xibm
         xirminus=xibm-xim
      endif
      xiimax=xirplus
      xinorm=xirplus+xirminus
      end subroutine get_massive_fsr_bounds


      subroutine getangles(pin,th,cth,sth,phi,cphi,sphi)
      implicit none
      real*8 pin(0:3),th,cth,sth,phi,cphi,sphi,xlength
c
      xlength=pin(1)**2+pin(2)**2+pin(3)**2
      if(xlength.eq.0)then
        th=0.d0
        cth=1.d0
        sth=0.d0
        phi=0.d0
        cphi=1.d0
        sphi=0.d0
      else
        xlength=sqrt(xlength)
        cth=pin(3)/xlength
        th=acos(cth)
        if(cth.ne.1.d0)then
          sth=sqrt(1-cth**2)
          phi=atan2(pin(2),pin(1))
          cphi=cos(phi)
          sphi=sin(phi)
        else
          sth=0.d0
          phi=0.d0
          cphi=1.d0
          sphi=0.d0
        endif
      endif
      return
      end subroutine getangles


      subroutine gentcms(pa,pb,t,phi,m1,m2,p1,pr,jac)
c*************************************************************************
c     Generates 4 momentum for particle 1, and remainder pr
c     given the values t, and phi
c     Assuming incoming particles with momenta pa, pb
c     And outgoing particles with mass m1,m2
c     s = (pa+pb)^2  t=(pa-p1)^2
c*************************************************************************
      implicit none
c
c     Arguments
c
      double precision t,phi,m1,m2               !inputs
      double precision pa(0:3),pb(0:3),jac
      double precision p1(0:3),pr(0:3)           !outputs
c
c     local
c
      double precision ptot(0:3),E_acms,p_acms,pa_cms(0:3)
      double precision esum,ed,pp,md2,ma2,pt,ptotm(0:3)
      integer i
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----
      do i=0,3
         ptot(i)  = pa(i)+pb(i)
         if (i .gt. 0) then
            ptotm(i) = -ptot(i)
         else
            ptotm(i) = ptot(i)
         endif
      enddo
      ma2 = dot(pa,pa)
c
c     determine magnitude of p1 in cms frame (from dhelas routine mom2cx)
c
      ESUM = sqrt(max(0d0,dot(ptot,ptot)))
      if (esum .eq. 0d0) then
         jac=-8d0             !Failed esum must be > 0
         return
      endif
      MD2=(M1-M2)*(M1+M2)
      ED=MD2/ESUM
      IF (M1*M2.EQ.0.) THEN
         PP=(ESUM-ABS(ED))*0.5d0
      ELSE
         PP=(MD2/ESUM)**2-2.0d0*(M1**2+M2**2)+ESUM**2
         if (pp .gt. 0) then
            PP=SQRT(pp)*0.5d0
         else
            write(*,*) 'Warning #12 in genps_fks.f',pp
            jac=-1
            return
         endif
      ENDIF
c
c     Energy of pa in pa+pb cms system
c
      call boostx(pa,ptotm,pa_cms)
      E_acms = pa_cms(0)
      p_acms = dsqrt(pa_cms(1)**2+pa_cms(2)**2+pa_cms(3)**2)
c
      p1(0) = MAX((ESUM+ED)*0.5d0,0.d0)
      p1(3) = -(m1*m1+ma2-t-2d0*p1(0)*E_acms)/(2d0*p_acms)
      pt = dsqrt(max(pp*pp-p1(3)*p1(3),0d0))
      p1(1) = pt*cos(phi)
      p1(2) = pt*sin(phi)
c
      call rotxxx(p1,pa_cms,p1)          !Rotate back to pa_cms frame
      call boostx(p1,ptot,p1)            !boost back to lab fram
      do i=0,3
         pr(i)=pa(i)-p1(i)               !Return remainder of momentum
      enddo
      end subroutine gentcms


      DOUBLE PRECISION FUNCTION LAMBDA(S,MA2,MB2)
      IMPLICIT NONE
C****************************************************************************
C     THIS IS THE LAMBDA FUNCTION FROM VERNONS BOOK COLLIDER PHYSICS P 662
C     MA2 AND MB2 ARE THE MASS SQUARED OF THE FINAL STATE PARTICLES
C     2-D PHASE SPACE = .5*PI*SQRT(1.,MA2/S^2,MB2/S^2)*(D(OMEGA)/4PI)
C****************************************************************************
      DOUBLE PRECISION MA2,MB2,S,tiny,tmp,rat
      parameter (tiny=1.d-8)
c
c Keep the small recoil momentum when S approaches a massive threshold.
c The expanded polynomial loses (S-MA2)**2 for a soft massless recoil,
c collapsing the t-channel bounds used by native history inversions.
      if (MA2.eq.0d0.or.MB2.eq.0d0) then
         tmp=(S-MA2-MB2)**2
      elseif (MA2.gt.0d0.and.MB2.gt.0d0) then
         tmp=(S-(sqrt(MA2)+sqrt(MB2))**2)*
     $       (S-(sqrt(MA2)-sqrt(MB2))**2)
      else
         tmp=(S-MA2-MB2)**2-4d0*MA2*MB2
      endif
      if(tmp.le.0.d0)then
        if(ma2.lt.0.d0.or.mb2.lt.0.d0)then
          write(6,*)'Error #1 in function Lambda:',s,ma2,mb2
          stop
        endif
        rat=1-(sqrt(ma2)+sqrt(mb2))/s
        if(rat.gt.-tiny)then
          tmp=0.d0
        else
          write(6,*)'Error #2 in function Lambda:',s,ma2,mb2,rat
        endif
      endif
      LAMBDA=tmp
      RETURN
      end function LAMBDA


      SUBROUTINE YMINMAX(X,Y,Z,U,V,W,YMIN,YMAX)
C**************************************************************************
C     This is the G function from Particle Kinematics by
C     E. Byckling and K. Kajantie, Chapter 4 p. 91 eqs 5.28
C     It is used to determine physical limits for Y based on inputs
C**************************************************************************
      implicit none
c
c     Constant
c
      double precision tiny
      parameter       (tiny=1d-199)
c
c     Arguments
c
      Double precision x,y,z,u,v,w              !inputs  y is dummy
      Double precision ymin,ymax                !output
c
c     Local
c
      double precision y1,y2,yr,ysqr
c
c     External
c
c-----
c  Begin Code
c-----
      ysqr = lambda(x,u,v)*lambda(x,w,z)
      if (ysqr .ge. 0d0) then
         yr = dsqrt(ysqr)
      else
         print*,'Error in yminymax sqrt(-x)',lambda(x,u,v),lambda(x,w,z)
         yr=0d0
      endif
      y1 = u+w -.5d0* ((x+u-v)*(x+w-z) - yr)/(x+tiny)
      y2 = u+w -.5d0* ((x+u-v)*(x+w-z) + yr)/(x+tiny)
      ymin = min(y1,y2)
      ymax = max(y1,y2)
      end subroutine YMINMAX


      subroutine get_recoil(p_born,imother,shat_born,xmrec2,pass)
      implicit none
      double precision p_born(0:3,nexternal-1),xmrec2,shat_born
      logical pass
      integer imother,i
      double precision recoilbar(0:3),dot
      external dot
      pass=.true.
      do i=0,3
         if (nincoming.eq.2) then
            recoilbar(i)=p_born(i,1)+p_born(i,2)-p_born(i,imother)
         else
            recoilbar(i)=p_born(i,1)-p_born(i,imother)
         endif
      enddo
      xmrec2=dot(recoilbar,recoilbar)
      if(xmrec2.lt.0.d0)then
         if(abs(xmrec2).gt.(1.d-4*shat_born))then
            write(*,*)'Fatal error #14 in genps_fks.f',xmrec2,imother
            stop
         else
            write(*,*)'Error #15 in genps_fks.f',xmrec2,imother
            pass=.false.
            return
         endif
      endif
      if (xmrec2.ne.xmrec2) then
         write (*,*) 'Error #16 in setting up event in genps_fks.f,'//
     &        ' skipping event'
         pass=.false.
         return
      endif
      return
      end subroutine get_recoil
      end module fks_phase_space_helpers
