      module fks_phase_space_helpers
c Shared geometry and numerical kernels for phase-space maps, recoil
c adapters, shower scales and subtraction. Only generated dimensions
c are required; active point data and process initialization stay with
c callers. Soft coordinate recovery takes its endpoint direction explicitly.
      implicit none
      include 'nexternal.inc'
      private
      public get_massive_fsr_bounds,getangles,gentcms,lambda,yminmax,
     $     get_recoil,boost_isr_recoil
      public dot,rho,sumdot,pt,deltaR,delta_phi,delta_y,HTo2,HT,
     $     boost_n1_to_its_cms,boost_n1_to_lab,get_xi_from_p,
     $     get_yij_from_p,get_phi_from_p,flip_momenta,
     $     apply_momentum_permutation
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
c Keep the external legacy dot product and its near-zero clipping.
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
c Keep the external legacy dot product and its near-zero clipping.
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
c Boost roundoff grows with the event energy. The absolute threshold
c in dot alone cannot protect a massless recoil at large shat. Retain
c the rejection below for negative masses beyond this relative bound.
      if(xmrec2.lt.0d0.and.xmrec2.ge.-1d-12*shat_born)
     &     xmrec2=0d0
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
c Momentum utilities and FKS coordinate recovery.

      double precision function get_phi_from_p(i_fks,j_fks,p)
        implicit none
        double precision,parameter :: pi=3.1415926535897932d0
        integer :: i_fks,j_fks
        double precision,dimension(0:3,nexternal) :: p
        double precision,dimension(0:3) :: p_rot,p_mother
        double precision :: th_mother_fks,costh_mother_fks
     $     ,sinth_mother_fks, phi_mother_fks,cosphi_mother_fks
     $     ,sinphi_mother_fks
        if (j_fks.gt.nincoming) then
           p_mother(0:3)=p(0:3,i_fks)+p(0:3,j_fks)
c The forward FSR map preserves the mother's direction. Recover
c the rotation directly, without a recoil boost that can flip it.
           call getangles(p_mother,
     $     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     $     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
           call rotate_invar_inverse(p(0,i_fks),p_rot(0),
     $     costh_mother_fks,-sinth_mother_fks,
     $     cosphi_mother_fks,-sinphi_mother_fks)
c compute phi:
           get_phi_from_p=atan2(p_rot(2),p_rot(1))
           if (get_phi_from_p.lt.0d0) get_phi_from_p=get_phi_from_p+2d0*pi
        else
           get_phi_from_p=atan2(p(2,i_fks),p(1,i_fks))
           if (get_phi_from_p.lt.0d0) get_phi_from_p=get_phi_from_p+2d0*pi
        endif
      end function get_phi_from_p
      double precision function get_xi_from_p(i_fks,j_fks,p_cms)
        implicit none
        integer :: i_fks,j_fks
        double precision,dimension(0:3,nexternal) :: p_cms
        get_xi_from_p=sqrt(2d0)*p_cms(0,i_fks)/sqrt(dot(p_cms(0,1),p_cms(0,2)))
      end function get_xi_from_p
      subroutine boost_n1_to_its_cms(p,p_cm,y)
        implicit none
        double precision,dimension(0:3,nexternal),intent(in) :: p
        double precision,dimension(0:3,nexternal),intent(out) :: p_cm
        double precision,intent(out) :: y
        integer :: i
c Add each beam's light-cone components before adding the beams;
c subtracting their total E and pz loses the smaller beam at large y.
        y=log(((p(0,1)+p(3,1))+(p(0,2)+p(3,2)))/
     $     ((p(0,1)-p(3,1))+(p(0,2)-p(3,2))))/2d0
        do i=1,nexternal
           call boostz(p(0,i),y,p_cm(0,i))
        enddo
      end subroutine boost_n1_to_its_cms
      subroutine boost_n1_to_lab(p,p_lab,y)
        implicit none
        double precision,dimension(0:3,nexternal),intent(in) :: p
        double precision,dimension(0:3,nexternal),intent(out) :: p_lab
        double precision,intent(in) :: y
        integer :: i
        do i=1,nexternal
           call boostz(p(0,i),y,p_lab(0,i))
        enddo
      end subroutine boost_n1_to_lab
      double precision function get_yij_from_p(i_fks,j_fks,p_cms,soft_direction)
        implicit none
        integer :: i_fks,j_fks
        double precision,dimension(0:3,nexternal) :: p_cms
        double precision,dimension(0:3) :: pi,pj
        double precision,intent(in) :: soft_direction(0:3)
        double precision,dimension(3) :: ui,uj
c The supplied soft direction is defined in the "reduced frame" (where
c the Born is in its center-of-mass). Here, we only use it in the
c soft limit, where it coincides with the n+1-body cms frame.
c Finite real momenta retain their own directions, however small.
c A cached soft direction can belong to another FKS history during
c native inversion, and cannot represent a different soft sister.
        if (p_cms(0,i_fks).le.0d0) then ! Exactly soft: use momenta with energy divided out
           pi(0:3)=soft_direction
        else
           pi(0:3)=p_cms(0:3,i_fks)
        endif
        if (p_cms(0,j_fks).le.0d0) then ! Exactly soft: use momenta with energy divided out
           pj(0:3)=soft_direction
        else
           pj(0:3)=p_cms(0:3,j_fks)
        endif
        ui=pi(1:3)/rho(pi)
        uj=pj(1:3)/rho(pj)
c Preserve the small opening angle also for antiparallel daughters.
c A dot product of large momenta can lose several ulps near |y|=1.
        if (sum(ui*uj).ge.0d0) then
           get_yij_from_p=1d0-0.5d0*sum((ui-uj)**2)
        else
           get_yij_from_p=-1d0+0.5d0*sum((ui+uj)**2)
        endif
      end function get_yij_from_p
      double precision function dot3(p1,p2)
        implicit none
        double precision,dimension(0:3) :: p1,p2
        dot3=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      end function dot3
      double precision function rho(p1)
        implicit none
        double precision,dimension(0:3) :: p1
        rho=sqrt(dot3(p1,p1))
      end function rho
      subroutine boostz(p,yb,pb)
c boost in the z-direction with rapidity yb
        implicit none
        real(kind=8),dimension(0:3) :: p,pb
        real(kind=8) :: yb,pplus,pminus
c Scale light-cone components instead of subtracting boosted E/pz.
        pplus=(p(0)+p(3))*exp(-yb)
        pminus=(p(0)-p(3))*exp(yb)
        pb(0)=0.5d0*(pplus+pminus)
        pb(1:2)=p(1:2)
        pb(3)=0.5d0*(pplus-pminus)
      end subroutine boostz

      double precision function deltaR(p1,p2)
        implicit none
        double precision,dimension(0:3) :: p1,p2
        deltaR = sqrt((delta_phi(p1,p2))**2+(delta_y(p1,p2))**2)
      end function deltaR

      double precision function delta_phi(p1, p2)
        implicit none
        double precision,dimension(0:3) :: p1,p2
        double precision :: denom, temp
        double precision,parameter :: tiny=1d-8
        denom = sqrt(p1(1)**2 + p1(2)**2) * sqrt(p2(1)**2 + p2(2)**2)
        temp = max(-(1d0-tiny), (p1(1)*p2(1) + p1(2)*p2(2)) / denom)
        temp = min( (1d0-tiny), temp)
        delta_phi = acos(temp)
      end function delta_phi

      double precision  function delta_y(p1,p2)
        implicit none
        double precision,dimension(0:3) :: p1,p2
        delta_y =.5d0*dlog((p1(0)+p1(3))/(p1(0)-p1(3)))-
     $     .5d0*dlog((p2(0)+p2(3))/(p2(0)-p2(3)))
      end function delta_y

      double precision function pt(p)
        implicit none
        double precision,dimension(0:3) :: p
        pt = dsqrt(p(1)**2+p(2)**2)
      end function pt

      double precision function HTo2(n,p)
        implicit none
        integer :: n
        double precision,dimension(0:3,n) :: p
        HTo2=HT(n,p)/2d0
      end function HTo2

      double precision function HT(n,p)
        implicit none
        integer :: n,j
        double precision,dimension(0:3,n) :: p
        HT=0d0
        do j=3,n
           HT=HT+sqrt((p(0,j)+p(3,j))*(p(0,j)-p(3,j)))
        enddo
      end function HT

      double precision function sumdot(p1,p2,sign)
        implicit  none
        double precision,dimension(0:3) :: p1,p2
        double precision :: sign
        sumdot=dot(p1+sign*p2,p1+sign*p2)
      end function sumdot

      double precision function dot(p1,p2)
        implicit none
        double precision,dimension(0:3) :: p1,p2
        dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end function dot


      subroutine rotate_invar_inverse(pin,pout,cth,sth,cphi,sphi)
c Given the four momentum pin, returns the four momentum pout (in the
c same Lorentz frame) by performing a three-rotation of an angle phi
c (cos(phi)=cphi) along the z axis, followed by a three-rotation of an
c angle theta (cos(theta)=cth) around the y axis. The components of pin
c and pout are given along these axes This is the inverse of
c rotate_invar(), if the signs of the angles are flipped:
c     call rotate_invar(pin,pout,cth,sth,cphi,sphi)
c     call rotate_invar_inverse(pout,pin2,cth,-sth,cphi,-sphi)
c Then pin==pin2
        implicit none
        double precision :: cth,sth,cphi,sphi,pin(0:3),pout(0:3)
        double precision :: q1,q2,q3
        q1=pin(1)
        q2=pin(2)
        q3=pin(3)
        pout(1)=(q1*cphi-q2*sphi)*cth+q3*sth
        pout(2)=q1*sphi+q2*cphi
        pout(3)=-(q1*cphi-q2*sphi)*sth+q3*cth
        pout(0)=pin(0)
      end subroutine rotate_invar_inverse

      subroutine flip_momenta(i,ii,j,jj,p,p_flipped)
        implicit none
        integer :: i,ii,j,jj,k,pos,tmp,perm(nexternal)
        double precision :: p(0:3,nexternal),p_flipped(0:3,nexternal)
        if (min(i,ii,j,jj).lt.1.or.max(i,ii,j,jj).gt.nexternal.or.i.eq.j.or.ii.eq.jj) then
           write (*,*) 'Invalid FKS labels in flip_momenta',i,ii,j,jj
           stop 1
        endif
c Build a bijection, including overlapping swaps: original ii and jj
c must end up in the native FKS slots i and j, respectively.
        perm=[(k,k=1,nexternal)]
        tmp=perm(i)
        perm(i)=perm(ii)
        perm(ii)=tmp
        do pos=1,nexternal
           if (perm(pos).eq.jj) exit
        enddo
        tmp=perm(j)
        perm(j)=perm(pos)
        perm(pos)=tmp
        call apply_momentum_permutation(perm,p,p_flipped)
      end subroutine flip_momenta

      subroutine apply_momentum_permutation(perm,p,p_permuted,valid)
        use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
        implicit none
        integer,intent(in) :: perm(nexternal)
        double precision,intent(in) :: p(0:3,nexternal)
        double precision,intent(out) :: p_permuted(0:3,nexternal)
        logical,optional,intent(out) :: valid
        integer :: k
        double precision :: tolerance,mass2_delta
c Numerical failures can invalidate a whole integration point. Invalid
c integer maps remain fatal: they indicate a broken history table.
        if(present(valid))valid=.false.
c Check the integer map BEFORE using it as a vector subscript.
        if (any(perm.lt.1).or.any(perm.gt.nexternal)) then
           write (*,*) 'Out-of-range MC momentum permutation',perm
           stop 1
        endif
        do k=1,nexternal
           if (count(perm.eq.k).ne.1) then
              write (*,*) 'Non-bijective MC momentum permutation',perm
              stop 1
           endif
           if (k.le.nincoming.and.perm(k).ne.k) then
              write (*,*) 'MC momentum permutation exchanges an incoming leg',perm
              stop 1
           endif
        enddo
        p_permuted=p(:,perm)
        if (.not.all(ieee_is_finite(p))) then
           if(present(valid))return
           write (*,*) 'Nonfinite momenta in MC momentum permutation'
           stop 1
        endif
        tolerance=1d-12*max(1d0,sum(abs(p)))
        if (maxval(abs(sum(p_permuted(:,nincoming+1:),dim=2)
     $     -sum(p(:,nincoming+1:),dim=2))).gt.tolerance) then
           if(present(valid))return
           write (*,*) 'MC momentum permutation changes total four-momentum'
           stop 1
        endif
c The export-time identity check forbids exchanges of unlike species.
c Also check their on-shell invariants at the actual phase-space point.
        tolerance=1d-10*max(1d0,maxval(abs(p))**2)
        do k=nincoming+1,nexternal
           mass2_delta=dot(p_permuted(:,k),p_permuted(:,k))
     $          -dot(p(:,k),p(:,k))
           if (.not.ieee_is_finite(mass2_delta).or.abs(mass2_delta).gt.tolerance) then
              if(present(valid))return
              write (*,*) 'MC momentum permutation changes a leg mass',k,perm(k)
              stop 1
           endif
        enddo
        if(present(valid))valid=.true.
      end subroutine apply_momentum_permutation

      subroutine boost_isr_recoil(pin,pout,xi,y,phi,idir,inverse,
     $     mass2)
c A Lorentz transformation preserving the spectator beam's null ray.
c It maps the Born total momentum to the hard real total in the Born
c CM. The inverse uses the same radiation variables and frame. Both
c directions permit pin and pout to alias, as in the FSR boost helpers.
c An optional on-shell mass stabilizes the inverse at small 1-xi.
      implicit none
      double precision pin(0:3),pout(0:3),xi,y,phi
      integer idir
      logical inverse
      double precision,optional,intent(in) :: mass2
      double precision z,a,b(2),pplus,pminus,transverse(2)
      if(xi.eq.0d0.or.y.eq.1d0)then
         pout=pin
         return
      endif
      z=1d0-xi
      a=1d0+xi*(1d0-y)/(2d0*z)
      b=-xi*sqrt((1d0-y)*(1d0+y))/(2d0*sqrt(z))
     &     *[cos(phi),sin(phi)]
      pplus=pin(0)+idir*pin(3)
      pminus=pin(0)-idir*pin(3)
      if(inverse)then
c Recover a small incoming light-cone component without E-|pz|.
c The known external mass is needed: reconstructing it from the highly
c boosted input would retain the roundoff which the inverse amplifies.
         if(present(mass2))then
            if(pminus.gt.pplus)
     $           pplus=(mass2+sum(pin(1:2)**2))/pminus
         endif
         pplus=pplus/a
         transverse=pin(1:2)-b*pplus
         if(present(mass2))then
c The direct Lorentz formula subtracts O(1/(1-xi)) terms. Enforce
c p+ p- = m^2 + pt^2 instead. The exact spectator-beam null ray has
c p+=pt=m=0 and transforms by rescaling p-; do not divide by zero.
            if(pplus.gt.0d0)then
               pminus=(mass2+sum(transverse**2))/pplus
            else
               pminus=a*pminus
            endif
         else
            pminus=a*pminus-2d0*sum(b*transverse)-sum(b*b)*pplus
         endif
      else
         transverse=pin(1:2)+b*pplus
         pminus=(pminus+2d0*sum(b*pin(1:2))+sum(b*b)*pplus)/a
         pplus=a*pplus
      endif
      pout(0)=(pplus+pminus)/2d0
      pout(1:2)=transverse
      pout(3)=idir*(pplus-pminus)/2d0
      end subroutine boost_isr_recoil

      end module fks_phase_space_helpers
