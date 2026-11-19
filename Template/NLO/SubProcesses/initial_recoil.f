      subroutine generate_momenta_initial_recoil(icountevts,
     $     isolsign,i_fks,j_fks,m_j_fks,irec,xbar,x,phi_i_fks,
     $     xp,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,
     $     xjac,xpswgt,q,mass2,beam_ratio,pass)
      use fks_phase_space_data, only: xij_aor
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
      use fks_radiation_maps, only: generate_momenta_massless_final,
     $     generate_momenta_massive_final
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
c A null final-state leg is best generated in the incoming spectator's
c light-cone basis. A boost to the reservoir rest frame cannot retain
c its small light-cone component when the laboratory jet is very hard.
      if(m_j_fks.eq.0d0)then
         isolsign=1
         call generate_massless_initial_recoil(icountevts,i_fks,
     $        j_fks,irec,xbar,x,phi_i_fks,xp,xiimax,xinorm,
     $        xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,
     $        q,mass2,beam_ratio,pass)
         return
      endif
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
      use fks_phase_space_data, only: xinorm_ev
c Invert at fixed physical beam momentum P_a=p_a/xreal. K=p_j+k+
c P_a-p_a is invariant under this map. The Born momentum follows from
c pbar_a=p_a*(1-p_j.k/(p_a.(p_j+k))) and pbar_j=p_j+k+pbar_a-p_a.
c Recover the ordinary FSR coordinates, then replay its forward kernel
c for the measure, including the massive second solution/native map.
      use fks_radiation_maps, only: native_fsr_angle
      use fks_phase_space_helpers, only: getangles,
     $     get_massive_fsr_bounds
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
      integer irec,i_fks,j_fks,i,ib,isolsign
      double precision p(0:3,nexternal),xreal,m_j_fks,pborn(0:3,nexternal-1),x(3),xjac,xpswgt,q(0:3),
     $     mass2,xbar,qrest(0:3),pi4(0:3),pj4(0:3),mother(0:3),mrest(0:3),rot(0:3),pb(0:3,nexternal),
     $     unchanged(0:3),reservoir(0:3),born_ratio,work(0:3,nexternal),uv(3),vv(3),mass,denom,xi,y,phi,th,
     $     cth,sth,ph,cph,sph,xmjhat,xim,xibm,ximax,cffa2,cffb2,cffc2,cffdel2,rat,branch_sign,sstiny,cctiny,
     $     xiimax,xinorm,xihat,phat(0:3),beam_ratio,uborn,eborn,u,ej,native_delta,native_onepy,native_ratio,
     $     native_denom,dot,rho,pi
      parameter(pi=3.1415926535897932d0)
      logical pass,softtest,colltest
      common/sctests/softtest,colltest
      external dot,rho

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
      if(m_j_fks.eq.0d0)then
         call invert_massless_initial_recoil(p,irec,xreal,i_fks,
     $        j_fks,pborn,x,xjac,xpswgt,q,mass2,xbar,pass)
         return
      endif
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


      subroutine null_recoil_components(p,idir,plus,minus,pass)
c Recover the small null component from the known model mass, rather
c than subtracting the energy and longitudinal momentum. Only roundoff
c deviations from the external mass shell are admitted here.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      double precision p(0:3),plus,minus,norm,pt2
      integer idir
      logical pass
      pt2=sum(p(1:2)**2)
      norm=sqrt(pt2+p(3)**2)
      pass=all(ieee_is_finite(p)).and.p(0).gt.0d0.and.
     $     abs(p(0)-norm).le.1d-10*max(1d0,p(0))
      plus=0d0
      minus=0d0
      if(.not.pass)return
      if(idir*p(3).ge.0d0)then
         plus=norm+idir*p(3)
         if(plus.gt.0d0)minus=pt2/plus
      else
         minus=norm-idir*p(3)
         if(minus.gt.0d0)plus=pt2/minus
      endif
      end


      subroutine null_recoil_axes(p,q,mass,u,v)
c Transport the ordinary FSR transverse axes. For an incoming null
c spectator their transverse projections are an orthonormal 2D basis.
      use fks_phase_space_helpers, only: getangles
      implicit none
      double precision p(0:3),q(0:3),mass,u(2),v(2),
     $     qr(0:3),pr(0:3),r(0:3),rr(0:3),rl(0:3),
     $     th,cth,sth,ph,cph,sph,orient
      qr(0)=q(0)
      qr(1:3)=-q(1:3)
      call boostm(p,qr,mass,pr)
      call getangles(pr,th,cth,sth,ph,cph,sph)
      r=(/0d0,1d0,0d0,0d0/)
      call rotate_invar(r,rr,cth,sth,cph,sph)
      call boostm(rr,q,mass,rl)
      u=rl(1:2)/sqrt(sum(rl(1:2)**2))
      r=(/0d0,0d0,1d0,0d0/)
      call rotate_invar(r,rr,cth,sth,cph,sph)
      call boostm(rr,q,mass,rl)
      orient=sign(1d0,u(1)*rl(2)-u(2)*rl(1))
      v=orient*(/-u(2),u(1)/)
      end


      subroutine generate_massless_initial_recoil(icountevts,
     $     i_fks,j_fks,irec,xbar,x,phi,xp,xiimax,xinorm,xi,y,
     $     xihat,phat,xjac,xpswgt,q,mass2,beam_ratio,pass)
      use fks_phase_space_data, only: xij_aor,veckn_ev,
     $     veckbarn_ev,xp0jfks
      use mc_native_context, only: native_mapping
      use FKSParams, only: pythia_local_matching
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      include 'nexternal.inc'
      integer icountevts,i_fks,j_fks,irec,idir
      double precision xbar,x(2),phi,xp(0:3,nexternal),
     $     xiimax,xinorm,xi,y,xihat,phat(0:3),xjac,xpswgt,
     $     q(0:3),mass2,beam_ratio,p(0:3),pr(0:3),reservoir(0:3),
     $     pp,pm,rp,mass,rad,c,v,d,n,a,b,a0,delta,
     $     u(2),vv(2),kt(2),kp,km,jp,jm,trans(2),
     $     aa,bb,st,ct,qr(0:3),mother(0:3),sstiny,cctiny
      double precision,parameter :: pi=3.1415926535897932d0
      double precision xi_fix,y_fix
      common/cxiyfix/xi_fix,y_fix
      logical pass,softtest,colltest
      common/sctests/softtest,colltest
      pass=.false.
      if(irec.lt.1.or.irec.gt.2.or.xbar.le.0d0.or.
     $     xbar.ge.1d0)goto 900
      idir=3-2*irec
      call null_recoil_components(xp(:,j_fks),idir,pp,pm,pass)
      if(.not.pass.or.pm.le.0d0)goto 900
      p=xp(:,j_fks)
      p(0)=(pp+pm)/2d0
      p(3)=idir*(pp-pm)/2d0
      reservoir=(1d0-xbar)/xbar*xp(:,irec)
      rp=2d0*reservoir(0)
      mass2=rp*pm
      if(mass2.le.0d0)goto 900
      mass=sqrt(mass2)
      q=p+reservoir
      call null_recoil_axes(p,q,mass,u,vv)
      sstiny=1d-6
      cctiny=5d-7
      if(native_mapping.or.pythia_local_matching().or.softtest.or.
     $     colltest)then
         sstiny=0d0
         cctiny=0d0
      endif
      if(abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
         y=dble(sign(1,icountevts))
         v=1d0+y
      elseif(colltest)then
         y=y_fix
         v=1d0+y
      elseif(native_mapping)then
         rad=min(x(2),1d0-x(2))
         if(x(2).le.0.5d0)then
            v=2d0*cos(pi*rad/2d0)**2
         else
            v=2d0*sin(pi*rad/2d0)**2
         endif
         y=cos(pi*x(2))
      else
         v=2d0*(1d0-cctiny)*(1d0-x(2))*(1d0+x(2))
         y=v-1d0
      endif
      if(.not.colltest)then
         if(native_mapping)then
            xjac=xjac*pi*sin(pi*min(x(2),1d0-x(2)))
         else
            xjac=xjac*4d0*x(2)*(1d0-cctiny)
         endif
      endif
      xiimax=1d0
      xinorm=1d0
      if(icountevts.eq.0.or.abs(icountevts).eq.2)then
         xi=0d0
         c=1d0
      elseif(softtest)then
         xi=xi_fix
         c=1d0-xi
      elseif(native_mapping)then
c Two energy hemispheres retain both the soft and hard complements.
         if(x(1).le.0.5d0)then
            xi=2d0*x(1)**2
            c=1d0-xi
         else
            c=2d0*(1d0-x(1))**2
            xi=1d0-c
         endif
      else
         c=(1d0-sstiny)*(1d0-x(1))*(1d0+x(1))
         xi=sstiny+(1d0-sstiny)*x(1)**2
      endif
      if(.not.softtest)then
         if(native_mapping)then
            xjac=xjac*4d0*min(x(1),1d0-x(1))
         else
            xjac=xjac*2d0*x(1)*(1d0-sstiny)
         endif
      endif
      if(xi.lt.0d0.or.c.le.0d0.or.v.lt.0d0.or.v.gt.2d0)
     $     goto 900
      d=2d0*c+xi*v
      n=2d0*c*c+xi*v*(1d0+c)
      if(d.le.0d0.or.n.le.0d0)goto 900
      aa=v/n
      bb=c*c*(2d0-v)/n
      a=xi*aa
      b=xi*bb
      a0=c*d/n
      delta=xi*c*(2d0-v)/d
      trans=cos(phi)*u+sin(phi)*vv
c The reduced emitted direction is also needed at the soft endpoint.
      kt=sqrt(max(0d0,aa*bb*mass2))*trans
      km=aa*pm
      phat(1:2)=aa*p(1:2)+kt
      if(km.gt.0d0)then
         kp=sum(phat(1:2)**2)/km
      else
         kp=bb*rp
      endif
      phat(0)=(kp+km)/2d0
      phat(3)=idir*(kp-km)/2d0
      xp(:,i_fks)=xi*phat
      kt=xi*kt
      jm=a0*pm
      xp(1:2,j_fks)=a0*p(1:2)-kt
      if(jm.le.0d0)goto 900
      jp=sum(xp(1:2,j_fks)**2)/jm
      xp(0,j_fks)=(jp+jm)/2d0
      xp(3,j_fks)=idir*(jp-jm)/2d0
      beam_ratio=1d0+delta*(1d0-xbar)/xbar
      xp(:,irec)=beam_ratio*xp(:,irec)
      xihat=xi
      xpswgt=xpswgt*4d0*mass2*c/
     $     (d*n*(4d0*pi)**3)
      qr(0)=q(0)
      qr(1:3)=-q(1:3)
      call boostm(p,qr,mass,mother)
      call initial_recoil_collinear_phase(mother,q,mass,phi,xij_aor)
      if(icountevts.eq.-100)then
         veckn_ev=mass*c/d
         veckbarn_ev=mass/2d0
         xp0jfks=veckn_ev
      endif
      pass=all(ieee_is_finite(xp)).and.
     $     ieee_is_finite(xpswgt).and.xpswgt.gt.0d0
      if(pass)return
 900  continue
      pass=.false.
      xjac=-148d0
      end


      subroutine invert_massless_initial_recoil(p,irec,xreal,
     $     i_fks,j_fks,pborn,x,xjac,xpswgt,q,mass2,xbar,pass)
      use fks_phase_space_data, only: xinorm_ev
      use fks_radiation_maps, only: native_fsr_angle
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
      integer irec,i_fks,j_fks,idir,i,ib
      double precision p(0:3,nexternal),xreal,pborn(0:3,nexternal-1),
     $     x(3),xjac,xpswgt,q(0:3),mass2,xbar,kp,km,jp,jm,
     $     pp,pm,mother(0:3),unchanged(0:3),reservoir(0:3),
     $     bornratio,alpha,a0,delta,c,xi,eta,beta,phi,angle,
     $     u(2),v(2),kt(2),krest(0:3),jrest(0:3),mass,
     $     work(0:3,nexternal),xiimax,xinorm,y,xihat,
     $     phat(0:3),beam_ratio,sstiny,cctiny
      double precision,parameter :: pi=3.1415926535897932d0
      logical pass,softtest,colltest
      common/sctests/softtest,colltest
      pass=.false.
      if(irec.lt.1.or.irec.gt.2.or.xreal.le.0d0.or.
     $     xreal.ge.1d0)goto 900
      idir=3-2*irec
      call null_recoil_components(p(:,i_fks),idir,kp,km,pass)
      if(.not.pass)goto 900
      call null_recoil_components(p(:,j_fks),idir,jp,jm,pass)
      if(.not.pass)goto 900
      pm=km+jm
      if(pm.le.0d0.or.jm.le.0d0.or.km.le.0d0)goto 900
      mother(1:2)=p(1:2,i_fks)+p(1:2,j_fks)
      pp=sum(mother(1:2)**2)/pm
      mother(0)=(pp+pm)/2d0
      mother(3)=idir*(pp-pm)/2d0
      unchanged=p(:,3-irec)
      do i=3,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks)
     $        unchanged=unchanged-p(:,i)
      enddo
      if(abs(pm-(unchanged(0)-idir*unchanged(3))).gt.
     $     1d-9*max(1d0,maxval(abs(p))))goto 900
      bornratio=(pp-unchanged(0)-idir*unchanged(3))/
     $     (2d0*p(0,irec))
      if(bornratio.le.0d0.or.bornratio.gt.1d0)goto 900
      xbar=xreal*bornratio
      reservoir=(1d0-xbar)/xreal*p(:,irec)
      q=mother+reservoir
      mass2=2d0*reservoir(0)*pm
      if(mass2.le.0d0)goto 900
      mass=sqrt(mass2)
      alpha=km/pm
      a0=jm/pm
      delta=(xreal-xbar)/(1d0-xbar)
      if(delta.lt.0d0.or.delta.ge.1d0)goto 900
      beta=delta*a0
      c=a0*(1d0-xreal)/(1d0-xbar)
      xi=alpha+beta
      eta=a0+delta*alpha
      krest=(/xi,2d0*sqrt(max(0d0,alpha*beta)),0d0,
     $     alpha-beta/)
      jrest=(/eta,-krest(1),0d0,a0-delta*alpha/)
      angle=native_fsr_angle(krest,jrest)
      call null_recoil_axes(mother,q,mass,u,v)
      kt=p(1:2,i_fks)-alpha*mother(1:2)
      phi=modulo(atan2(sum(kt*v),sum(kt*u)),2d0*pi)
      if(native_mapping)then
         if(xi.le.0.5d0)then
            x(1)=sqrt(xi/2d0)
         else
            x(1)=1d0-sqrt(c/2d0)
         endif
         x(2)=angle
      else
         sstiny=1d-6
         cctiny=5d-7
         if(softtest.or.colltest)then
            sstiny=0d0
            cctiny=0d0
         endif
         x(1)=sqrt(max(0d0,(xi-sstiny)/(1d0-sstiny)))
         y=cos(pi*angle)
         x(2)=sqrt(max(0d0,((1d0-y)/2d0-cctiny)/
     $        (1d0-cctiny)))
      endif
      x(3)=phi/(2d0*pi)
      work=p
      work(:,i_fks)=0d0
      work(:,j_fks)=mother
      work(:,irec)=bornratio*p(:,irec)
      ib=0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         ib=ib+1
         pborn(:,ib)=work(:,i)
      enddo
      xjac=xjac*2d0*pi
      call generate_massless_initial_recoil(-100,i_fks,j_fks,
     $     irec,xbar,x,phi,work,xiimax,xinorm,xi,y,xihat,
     $     phat,xjac,xpswgt,q,mass2,beam_ratio,pass)
      xinorm_ev=xinorm
      if(pass)return
 900  continue
      pass=.false.
      xjac=-149d0
      end


      subroutine initial_recoil_collinear_phase(mother,q,mass,
     $     phi,phase)
c As for a resonance frame, transport the transverse helicity axis.
c Supply the invariant mass explicitly: K is very boosted for small x.
      use fks_phase_space_helpers, only: getangles
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
