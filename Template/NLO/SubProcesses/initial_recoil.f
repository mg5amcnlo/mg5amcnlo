      subroutine generate_momenta_initial_recoil(icountevts,
     $     isolsign,i_fks,j_fks,m_j_fks,irec,xbar,x,phi_i_fks,
     $     xp,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,
     $     xjac,xpswgt,q,mass2,beam_ratio,pass)
c Final-state radiation with a massless incoming spectator. Regard the
c unused beam momentum Rbar=(1/xbar-1)*pbar_a as a massless final-state
c recoil reservoir and apply the ordinary FSR map to K=pbar_j+Rbar.
c Its outgoing reservoir R=alpha*Rbar leaves p_a=P_a-R. Every physical
c spectator stays fixed, and x_a<=1 follows from R(0)>=0.
c
c The hadronic measure is NOT the ordinary FSR measure: the beam-fraction
c change supplies 1/alpha. Equivalently it is dPhi(k)*(P_a.pbar_j)/
c (P_a.p_j) before changing the Born-axis angle to the FKS opening angle.
c The flux, PDFs and full-process incoming invariant are caller-owned.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
      integer icountevts,isolsign,i_fks,j_fks,irec
      double precision m_j_fks,xbar,x(2),phi_i_fks,
     $     xp(0:3,nexternal),xiimax,xinorm,xi_i_fks,y_ij_fks,
     $     xi_i_hat,p_i_fks(0:3),xjac,xpswgt,q(0:3),mass2,
     $     beam_ratio,plocal(0:3,nexternal),qrest(0:3),
     $     mother(0:3),reservoir(0:3),phat(0:3),mass,
     $     uborn,u,ej,ei,delta,alpha,sintheta,dot,rho,pi
      parameter(pi=3.1415926535897932d0)
      double complex xij_aor
      common/cxij_aor/xij_aor
      logical pass
      external dot,rho

      pass=.false.
      beam_ratio=1d0
      q=0d0
      mass2=0d0
      if(nincoming.ne.2.or.irec.lt.1.or.irec.gt.nincoming)
     $     goto 900
      if(i_fks.le.nincoming.or.j_fks.le.nincoming.or.
     $     i_fks.gt.nexternal.or.j_fks.gt.nexternal.or.
     $     i_fks.eq.j_fks)goto 900
      if(.not.ieee_is_finite(xbar).or.xbar.le.0d0.or.
     $     xbar.ge.1d0.or.m_j_fks.lt.0d0)goto 900
      if(.not.all(ieee_is_finite(xp)))goto 900
      if(xp(0,irec).le.0d0.or.
     $     abs(dot(xp(:,irec),xp(:,irec))).gt.
     $     1d-10*xp(0,irec)**2)goto 900
      reservoir=((1d0-xbar)/xbar)*xp(:,irec)
      q=xp(:,j_fks)+reservoir
      mass2=m_j_fks**2+2d0*dot(xp(:,j_fks),reservoir)
      if(mass2.le.m_j_fks**2.or.q(0).le.0d0)goto 900
      mass=sqrt(mass2)
      qrest(0)=q(0)
      qrest(1:3)=-q(1:3)
      plocal=0d0
      plocal(:,1)=(/mass/2d0,0d0,0d0,mass/2d0/)
      plocal(:,2)=(/mass/2d0,0d0,0d0,-mass/2d0/)
      call boostm(xp(:,j_fks),qrest,mass,mother)
      plocal(:,j_fks)=mother
c No physical final slot is needed for the auxiliary reservoir: the
c forward FSR kernels compute its momentum from K minus the daughters.
      if(m_j_fks.eq.0d0)then
         isolsign=1
         call generate_momenta_massless_final(icountevts,i_fks,
     $        j_fks,mother,mass2,mass,x,0d0,plocal,phi_i_fks,
     $        xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,phat,
     $        xjac,xpswgt,pass)
      else
         call generate_momenta_massive_final(icountevts,isolsign,
     $        i_fks,j_fks,mother,mass2,mass,m_j_fks,x,0d0,
     $        plocal,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,
     $        xi_i_hat,phat,xjac,xpswgt,pass)
      endif
      if(.not.pass)return
      uborn=(mass2-m_j_fks**2)/(2d0*mass)
      u=rho(plocal(:,j_fks))
      ej=plocal(0,j_fks)
      ei=plocal(0,i_fks)
c Rationalize E_j-|p_j| in the massive case and retain the small
c borrowed beam momentum without subtracting nearly equal energies.
      delta=2d0*ei*(m_j_fks**2/(ej+u)+
     $     u*(1d0-y_ij_fks))/(mass2-m_j_fks**2)
      alpha=1d0-delta
      if(delta.gt.0.5d0)then
c Near an exhausted reservoir obtain its energy from the pair's
c spatial norm, keeping the native antiparallel angular coordinate.
         sintheta=sqrt(max(0d0,(1d0-y_ij_fks)*
     $        (1d0+y_ij_fks)))
         if(native_mapping.and.m_j_fks.gt.0d0)
     $        sintheta=sin(pi*min(x(2),1d0-x(2)))
         alpha=sqrt((ei+u*y_ij_fks)**2+
     $        (u*sintheta)**2)/uborn
      endif
      if(.not.ieee_is_finite(alpha).or.alpha.le.0d0.or.
     $     alpha.gt.1d0+1d-10.or.delta.lt.-1d-10.or.
     $     delta.gt.1d0+1d-10)goto 900
      beam_ratio=1d0+max(0d0,min(1d0,delta))*
     $     (1d0-xbar)/xbar
      if(m_j_fks.eq.0d0)
     $     call initial_recoil_collinear_phase(mother,q,mass,
     $     phi_i_fks,xij_aor)
      call boostm(plocal(:,i_fks),q,mass,xp(:,i_fks))
      call boostm(plocal(:,j_fks),q,mass,xp(:,j_fks))
      call boostm(phat,q,mass,p_i_fks)
      xp(:,irec)=beam_ratio*xp(:,irec)
      xpswgt=xpswgt/alpha
      pass=all(ieee_is_finite(xp)).and.ieee_is_finite(xpswgt)
      if(pass)return
 900  continue
      pass=.false.
      xjac=-148d0
      end


      subroutine invert_momenta_initial_recoil(p,irec,xreal,
     $     i_fks,j_fks,m_j_fks,pborn,x,xjac,xpswgt,q,mass2,
     $     xbar,pass)
c Invert at fixed physical beam momentum P_a=p_a/xreal. K=p_j+k+
c P_a-p_a is invariant under this map. The Born momentum follows from
c pbar_a=p_a*(1-p_j.k/(p_a.(p_j+k))) and pbar_j=p_j+k+pbar_a-p_a.
c Recover the ordinary FSR coordinates, then replay its forward kernel
c for the measure, including the massive second solution/native map.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
      integer irec,i_fks,j_fks,i,ib,isolsign
      double precision p(0:3,nexternal),xreal,m_j_fks,
     $     pborn(0:3,nexternal-1),x(3),xjac,xpswgt,q(0:3),
     $     mass2,xbar,qrest(0:3),pi4(0:3),pj4(0:3),
     $     mother(0:3),mrest(0:3),rot(0:3),pb(0:3,nexternal),
     $     unchanged(0:3),reservoir(0:3),born_ratio,
     $     work(0:3,nexternal),uv(3),vv(3),mass,denom,
     $     xi,y,phi,th,cth,sth,ph,cph,sph,
     $     xmjhat,xim,xibm,ximax,cffa2,cffb2,cffc2,cffdel2,
     $     rat,branch_sign,sstiny,cctiny,
     $     xiimax,xinorm,xihat,phat(0:3),beam_ratio,
     $     uborn,eborn,u,ej,native_delta,native_onepy,
     $     native_ratio,native_denom,native_fsr_angle,
     $     xinorm_ev,dot,rho,pi
      parameter(pi=3.1415926535897932d0)
      common/cxinormev/xinorm_ev
      logical pass,softtest,colltest
      common/sctests/softtest,colltest
      external dot,rho,native_fsr_angle

      pass=.false.
      pborn=0d0
      x=0d0
      xbar=0d0
      q=0d0
      mass2=0d0
      if(nincoming.ne.2.or.irec.lt.1.or.irec.gt.nincoming)
     $     goto 900
      if(i_fks.le.nincoming.or.j_fks.le.nincoming.or.
     $     i_fks.gt.nexternal.or.j_fks.gt.nexternal.or.
     $     i_fks.eq.j_fks)goto 900
      if(.not.ieee_is_finite(xreal).or.xreal.le.0d0.or.
     $     xreal.ge.1d0.or.m_j_fks.lt.0d0)goto 900
      if(.not.all(ieee_is_finite(p)))goto 900
      if(p(0,irec).le.0d0.or.
     $     abs(dot(p(:,irec),p(:,irec))).gt.
     $     1d-10*p(0,irec)**2)goto 900
c Use the unchanged spectators when reconstructing a small Born beam
c fraction. Subtracting the hard daughters from the enlarged incoming
c momentum loses all Born precision for a large borrowed energy.
      unchanged=p(:,3-irec)
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks)
     $        unchanged=unchanged-p(:,i)
      enddo
      denom=2d0*dot(p(:,irec),unchanged)
      if(denom.le.0d0)goto 900
      born_ratio=(m_j_fks**2-dot(unchanged,unchanged))/denom
      if(born_ratio.le.0d0.or.born_ratio.gt.1d0+1d-10)
     $     goto 900
      born_ratio=min(1d0,born_ratio)
      xbar=xreal*born_ratio
      mother=unchanged+born_ratio*p(:,irec)
      reservoir=(1d0-xbar)/xreal*p(:,irec)
      q=mother+reservoir
      mass2=m_j_fks**2+2d0*dot(mother,reservoir)
      if(mass2.le.m_j_fks**2.or.q(0).le.0d0)goto 900
      mass=sqrt(mass2)
      qrest(0)=q(0)
      qrest(1:3)=-q(1:3)
      call boostm(p(:,i_fks),qrest,mass,pi4)
      call boostm(p(:,j_fks),qrest,mass,pj4)
      u=rho(pj4)
      ej=pj4(0)
      if(pi4(0).le.0d0.or.u.le.0d0)goto 900
      uv=pi4(1:3)/rho(pi4)
      vv=pj4(1:3)/u
      if(sum(uv*vv).ge.0d0)then
         y=1d0-sum((uv-vv)**2)/2d0
      else
         y=-1d0+sum((uv+vv)**2)/2d0
      endif
      y=max(-1d0,min(1d0,y))
      pb=p
      pb(:,irec)=born_ratio*p(:,irec)
      pb(:,i_fks)=0d0
      pb(:,j_fks)=mother
      call boostm(mother,qrest,mass,mrest)
      call getangles(mrest,th,cth,sth,ph,cph,sph)
      call trp_rotate_invar(pi4,rot,cth,sth,cph,sph)
      phi=modulo(atan2(rot(2),rot(1)),2d0*pi)
      xi=2d0*pi4(0)/mass
      sstiny=1d-6
      cctiny=5d-7
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif
      if(native_mapping.and.m_j_fks.gt.0d0)then
         x(2)=native_fsr_angle(pi4,pj4)
      else
         x(2)=((1d0-y)/2d0-cctiny)/(1d0-cctiny)
         if(x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12)goto 900
         x(2)=sqrt(max(0d0,min(1d0,x(2))))
      endif
      if(m_j_fks.eq.0d0)then
         x(1)=(xi-sstiny)/(1d0-sstiny)
         if(x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12)goto 900
         x(1)=sqrt(max(0d0,min(1d0,x(1))))
      elseif(native_mapping)then
         uborn=(mass2-m_j_fks**2)/(2d0*mass)
         eborn=sqrt(uborn**2+m_j_fks**2)
         native_delta=uborn-u
         native_onepy=2d0*sin(pi*(1d0-x(2))/2d0)**2
         native_ratio=(uborn+u)/(eborn+ej)
         native_denom=mass*native_ratio-pi4(0)*(1d0+native_ratio)
         if(abs(native_delta).lt.1d-4*uborn.and.
     $        native_denom.gt.0d0)
     $        native_delta=pi4(0)*u*native_onepy/native_denom
         x(1)=native_delta/uborn
         if(x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12)goto 900
         x(1)=sqrt(max(0d0,min(1d0,x(1))))
      else
         xmjhat=m_j_fks/mass
         call get_massive_fsr_bounds(mass2,mass,m_j_fks,0d0,
     $        y,xim,xibm,ximax,cffa2,cffb2,cffc2,cffdel2,
     $        xiimax,xinorm)
         rat=xiimax/xinorm
         branch_sign=u/mass*(2d0-xi*(1d0-y))*
     $        (2d0-xi*(1d0+y))+xi*y*(1d0+xmjhat**2-xi)
         if(branch_sign.ge.0d0)then
            x(1)=(xi*rat/xinorm-sstiny)/(1d0-sstiny)
            if(x(1).lt.-1d-12.or.x(1).gt.rat**2+1d-12)goto 900
            x(1)=sqrt(max(0d0,min(rat**2,x(1))))
         else
            x(1)=((2d0*xiimax-xi)/xinorm-sstiny)/(1d0-sstiny)
            if(x(1).lt.rat-1d-12.or.x(1).gt.1d0+1d-12)goto 900
            x(1)=max(rat,min(1d0,x(1)))
         endif
      endif
      x(3)=phi/(2d0*pi)
      work=pb
      xjac=xjac*2d0*pi
      call generate_momenta_initial_recoil(-100,isolsign,i_fks,
     $     j_fks,m_j_fks,irec,xbar,x,phi,work,xiimax,xinorm,
     $     xi,y,xihat,phat,xjac,xpswgt,q,mass2,beam_ratio,pass)
      if(.not.pass)goto 900
      xinorm_ev=xinorm
      ib=0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         ib=ib+1
         pborn(:,ib)=pb(:,i)
      enddo
      pass=all(ieee_is_finite(pborn)).and.
     $     ieee_is_finite(xjac).and.xjac.gt.0d0.and.
     $     ieee_is_finite(xpswgt).and.xpswgt.gt.0d0
      if(pass)return
 900  continue
      pass=.false.
      xjac=-149d0
      end


      subroutine initial_recoil_collinear_phase(mother,q,mass,
     $     phi,phase)
c As for a resonance frame, transport the transverse helicity axis.
c Supply the invariant mass explicitly: K is very boosted for small x.
      implicit none
      double precision mother(0:3),q(0:3),mass,phi,r(0:3),
     $     rrot(0:3),rlab(0:3),mlab(0:3),th,cth,sth,ph,cph,
     $     sph,norm2
      double complex phase,z
      call getangles(mother,th,cth,sth,ph,cph,sph)
      r=(/0d0,cos(phi),sin(phi),0d0/)
      call rotate_invar(r,rrot,cth,sth,cph,sph)
      call boostm(rrot,q,mass,rlab)
      call boostm(mother,q,mass,mlab)
      call getangles(mlab,th,cth,sth,ph,cph,sph)
      call trp_rotate_invar(rlab,rrot,cth,sth,cph,sph)
      norm2=rrot(1)**2+rrot(2)**2
      z=dcmplx(cph,sph)*dcmplx(rrot(1),rrot(2))
      phase=-z*z/norm2
      end
