      subroutine prepare_resonance_frame(p,in_resonance,q,qrest,
     $     mass2,plocal,pass)
c Build the decay subsystem in its rest frame. in_resonance uses REAL
c external-leg labels, including the emitted leg (zero at Born level).
c The aunt can be an external particle or a complete decay subtree.
c Auxiliary incoming momenta are used only inside the radiation maps;
c the physical incoming momenta and all non-descendants remain untouched.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),q(0:3),qrest(0:3),mass2,
     $     plocal(0:3,nexternal),mass,dot
      logical in_resonance(nexternal),pass
      integer i
      external dot

      pass=.false.
      q=0d0
      qrest=0d0
      plocal=0d0
      mass2=0d0
      if(any(in_resonance(1:nincoming)))return
      if(.not.all(ieee_is_finite(p)))return
      do i=nincoming+1,nexternal
         if(in_resonance(i))q=q+p(:,i)
      enddo
      mass2=dot(q,q)
      if(q(0).le.0d0.or.mass2.le.0d0)return
      mass=sqrt(mass2)
      qrest(0)=q(0)
      qrest(1:3)=-q(1:3)
      do i=nincoming+1,nexternal
         if(in_resonance(i))
     $        call boostx(p(:,i),qrest,plocal(:,i))
      enddo
      if(nincoming.eq.2)then
         plocal(:,1)=(/mass/2d0,0d0,0d0,mass/2d0/)
         plocal(:,2)=(/mass/2d0,0d0,0d0,-mass/2d0/)
      elseif(nincoming.eq.1)then
         plocal(:,1)=(/mass,0d0,0d0,0d0/)
      else
         return
      endif
      pass=.true.
      end


      subroutine generate_momenta_resonance_final(icountevts,
     $     isolsign,i_fks,j_fks,m_j_fks,in_resonance,x,phi_i_fks,
     $     xp,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,
     $     xjac,xpswgt,q,mass2,pass)
      use fks_phase_space_data, only: xij_aor
c Resonance-preserving FKS map, as in sec. 3 of arXiv:1509.09071.
c Apply the ordinary FSR map to the resonance decay, with the aunt as
c its entire recoil. Boosting the aunt's descendants together also
c preserves every invariant mass within that subtree. No mass inversion
c or numerical derivative is needed. xi and y are in the resonance frame;
c xp and p_i_fks are returned in the original frame. The flux remains
c that of the full process and must be supplied by the caller.
      use fks_radiation_maps, only: generate_momenta_massless_final,
     $     generate_momenta_massive_final
      implicit none
      include 'nexternal.inc'
      integer icountevts,isolsign,i_fks,j_fks,i
      logical in_resonance(nexternal),pass
      double precision m_j_fks,x(2),phi_i_fks,xp(0:3,nexternal),
     $     xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks(0:3),
     $     xjac,xpswgt,q(0:3),mass2,plocal(0:3,nexternal),
     $     qrest(0:3),mother(0:3),recoil(0:3),phat(0:3),
     $     xmrec2,mass,dot
      external dot

      pass=.false.
      if(i_fks.le.nincoming.or.j_fks.le.nincoming.or.
     $     i_fks.gt.nexternal.or.j_fks.gt.nexternal.or.
     $     i_fks.eq.j_fks)goto 900
      if(.not.in_resonance(i_fks).or.
     $     .not.in_resonance(j_fks))goto 900
      call prepare_resonance_frame(xp,in_resonance,q,qrest,mass2,
     $     plocal,pass)
      if(.not.pass)goto 900
      mother=plocal(:,j_fks)
      recoil=0d0
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks)recoil=recoil+plocal(:,i)
      enddo
      xmrec2=dot(recoil,recoil)
      if(xmrec2.lt.-1d-12*mass2.or.recoil(0).le.0d0)goto 900
      xmrec2=max(0d0,xmrec2)
      mass=sqrt(mass2)
      if(m_j_fks.eq.0d0)then
         isolsign=1
         call generate_momenta_massless_final(icountevts,i_fks,
     $        j_fks,mother,mass2,mass,x,xmrec2,plocal,phi_i_fks,
     $        xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,phat,
     $        xjac,xpswgt,pass)
      else
         call generate_momenta_massive_final(icountevts,isolsign,
     $        i_fks,j_fks,mother,mass2,mass,m_j_fks,
     $        x,xmrec2,plocal,phi_i_fks,xiimax,xinorm,xi_i_fks,
     $        y_ij_fks,xi_i_hat,phat,xjac,xpswgt,pass)
      endif
      if(.not.pass)return
c The spin-correlated Born is evaluated in the original frame. Its
c helicity phase must follow the boosted transverse direction as well.
      if(m_j_fks.eq.0d0)
     $     call resonance_collinear_phase(mother,q,phi_i_fks,xij_aor)
      do i=nincoming+1,nexternal
         if(in_resonance(i))call boostx(plocal(:,i),q,xp(:,i))
      enddo
      call boostx(phat,q,p_i_fks)
      return
 900  continue
      pass=.false.
      xjac=-145d0
      end


      subroutine resonance_collinear_phase(mother,q,phi,phase)
c Transport a unit transverse vector instead of using the resonance-
c frame azimuth with a Born helicity amplitude in another frame.
      use fks_phase_space_helpers, only: getangles
      implicit none
      double precision mother(0:3),q(0:3),phi,r(0:3),rrot(0:3),
     $     rlab(0:3),mlab(0:3),th,cth,sth,ph,cph,sph,norm2
      double complex phase,z
      call getangles(mother,th,cth,sth,ph,cph,sph)
      r=(/0d0,cos(phi),sin(phi),0d0/)
      call rotate_invar(r,rrot,cth,sth,cph,sph)
      call boostx(rrot,q,rlab)
      call boostx(mother,q,mlab)
      call getangles(mlab,th,cth,sth,ph,cph,sph)
      call trp_rotate_invar(rlab,rrot,cth,sth,cph,sph)
      norm2=rrot(1)**2+rrot(2)**2
      z=dcmplx(cph,sph)*dcmplx(rrot(1),rrot(2))
      phase=-z*z/norm2
      end


      subroutine invert_momenta_resonance_final(p,in_resonance,
     $     i_fks,j_fks,m_j_fks,pborn,x,xjac,xpswgt,q,mass2,pass)
      use fks_phase_space_data, only: xij_aor
c Project a fixed real point onto the same local Born map. Remove the
c emitted leg before returning pborn. Internally put radiation last,
c as required by the ordinary inverse maps. The mask is supplied by
c the caller's resonance history, never inherited from a previous one.
      use fks_radiation_maps,
     $     only: generate_momenta_massless_final_inverse,
     $     generate_momenta_massive_final_inverse
      use fks_phase_space_helpers, only: getangles
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      double precision p(0:3,nexternal),m_j_fks,
     $     pborn(0:3,nexternal-1),x(3),xjac,xpswgt,q(0:3),mass2,
     $     plocal(0:3,nexternal),pcanon(0:3,nexternal),
     $     pb(0:3,-max_branch:nexternal-1),qrest(0:3),
     $     mass,xi,y,phi,mother(0:3),rot(0:3),th,cth,sth,
     $     ph,cph,sph,u(3),v(3),rho,pi
      parameter(pi=3.1415926535897932d0)
      logical in_resonance(nexternal),pass
      integer i_fks,j_fks,i,ib,jborn
      external rho

      pass=.false.
      pborn=0d0
      x=0d0
      if(i_fks.le.nincoming.or.j_fks.le.nincoming.or.
     $     i_fks.gt.nexternal.or.j_fks.gt.nexternal.or.
     $     i_fks.eq.j_fks)goto 900
      if(.not.in_resonance(i_fks).or.
     $     .not.in_resonance(j_fks))goto 900
      call prepare_resonance_frame(p,in_resonance,q,qrest,mass2,
     $     plocal,pass)
      if(.not.pass)goto 900
      mass=sqrt(mass2)
      if(plocal(0,i_fks).le.0d0.or.rho(plocal(:,j_fks)).le.0d0)
     $     goto 900
      pcanon(:,nexternal)=plocal(:,i_fks)
      ib=0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         ib=ib+1
         pcanon(:,ib)=plocal(:,i)
         if(i.eq.j_fks)jborn=ib
      enddo
      xi=2d0*pcanon(0,nexternal)/mass
      u=pcanon(1:3,nexternal)/rho(pcanon(:,nexternal))
      v=pcanon(1:3,jborn)/rho(pcanon(:,jborn))
      if(sum(u*v).ge.0d0)then
         y=1d0-sum((u-v)**2)/2d0
      else
         y=-1d0+sum((u+v)**2)/2d0
      endif
      mother=pcanon(:,nexternal)+pcanon(:,jborn)
      call getangles(mother,th,cth,sth,ph,cph,sph)
      call trp_rotate_invar(pcanon(:,nexternal),rot,cth,sth,cph,sph)
      phi=modulo(atan2(rot(2),rot(1)),2d0*pi)
      pb=0d0
      if(m_j_fks.eq.0d0)then
         call generate_momenta_massless_final_inverse(pcanon,xi,y,
     $        phi,pb,x,xjac,xpswgt,mass2,mass,nexternal,jborn)
      else
         call generate_momenta_massive_final_inverse(pcanon,xi,y,
     $        phi,pb,x,xjac,xpswgt,mass2,mass,nexternal,jborn,m_j_fks)
      endif
      if(.not.ieee_is_finite(xjac).or.xjac.le.0d0.or.
     $     .not.all(ieee_is_finite(x)))goto 900
      ib=0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         ib=ib+1
         if(in_resonance(i))then
            call boostx(pb(:,ib),q,pborn(:,ib))
         else
            pborn(:,ib)=p(:,i)
         endif
      enddo
      if(m_j_fks.eq.0d0)
     $     call resonance_collinear_phase(pb(:,jborn),q,phi,xij_aor)
      pass=all(ieee_is_finite(pborn)).and.
     $     ieee_is_finite(xpswgt).and.xpswgt.gt.0d0
      if(pass)return
 900  continue
      pass=.false.
      xjac=-146d0
      end


      subroutine resonance_subtraction_scales(total,q,pj,phat,
     $     soft_scale,coll_scale,angular_scale,mismatch_log,pass)
c Convert subtraction reference scales between the total-CM and decay
c frames. phat is any positive multiple of the null soft direction.
c a(k)=xi_K/xi_Q=s*(K.k)/(K^2*(Q.k)); a_j is its collinear limit.
c For a massless emitter, delta_K=delta_Q/D_j^2, D_j=E_j,K/E_j,Q.
c Integrating the soft-energy exponential difference (sec. 3 of
c arXiv:1509.09071) with the collinear reference a_j gives log(a_j/a).
c This routine supplies only kinematic factors, not a matrix-element or
c FKS-sector weight. The caller must include the finite mismatch once.
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      implicit none
      double precision total(0:3),q(0:3),pj(0:3),phat(0:3),
     $     soft_scale,coll_scale,angular_scale,mismatch_log,
     $     s,mass2,kq,kt,jq,jt,diff(0:3),delta,dot
      logical pass
      external dot
      pass=.false.
      soft_scale=1d0
      coll_scale=1d0
      angular_scale=1d0
      mismatch_log=0d0
      s=dot(total,total)
      mass2=dot(q,q)
      kq=dot(phat,q)
      kt=dot(phat,total)
      jq=dot(pj,q)
      jt=dot(pj,total)
      if(min(s,mass2,kq,kt,jq,jt).le.0d0)return
      soft_scale=(s/mass2)*(kq/kt)
      coll_scale=(s/mass2)*(jq/jt)
      angular_scale=(s/mass2)/coll_scale**2
c Evaluate the small logarithm using the difference of normalized
c directions. Do not subtract two potentially large logarithms near
c the collinear limit. A short series avoids rounding 1+delta to 1.
      diff=phat/kq-pj/jq
      delta=(jq/jt)*(total(0)*diff(0)-
     $     sum(total(1:3)*diff(1:3)))
      if(abs(delta).lt.1d-4)then
         mismatch_log=delta*(1d0-delta*(0.5d0-delta*(1d0/3d0-
     $        delta*(0.25d0-delta/5d0))))
      else
         mismatch_log=log(coll_scale/soft_scale)
      endif
      pass=ieee_is_finite(soft_scale).and.
     $     ieee_is_finite(coll_scale).and.
     $     ieee_is_finite(angular_scale).and.
     $     ieee_is_finite(mismatch_log)
      end


      subroutine project_global_fsr_partition(p,i_fks,j_fks,m_j_fks,
     $     pborn,pass)
      use fks_phase_space_data, only: xinorm_ev,xij_aor
c A common reference point for the real-channel partition of unity.
c This projection supplies channel weights only. The subtraction Born
c and the physical recoil are still those of the local resonance map.
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),pborn(0:3,nexternal-1),m_j_fks,x(3),jac,pswgt,q(0:3),mass2,xinorm_save
      double complex phase_save
      integer i_fks,j_fks
      logical pass,members(nexternal),native_save
      members=.true.
      members(1:nincoming)=.false.
      native_save=native_mapping
      native_mapping=.true.
      xinorm_save=xinorm_ev
      phase_save=xij_aor
      jac=1d0
      pswgt=1d0
      call invert_momenta_resonance_final(p,members,i_fks,j_fks,
     $     m_j_fks,pborn,x,jac,pswgt,q,mass2,pass)
      native_mapping=native_save
      xinorm_ev=xinorm_save
      xij_aor=phase_save
      end


      subroutine resonance_shower_frame(p,i_fks,j_fks,plocal,
     $     kn,knbar,kn0,mass2)
      use fks_phase_space_data,only: resonance_momentum,resonance_mass2,resonance_recoil,
     $     resonance_members,initial_recoil_leg,p_born
c The shower kernels use the same emission variables and recoil system
c as the FKS map. Born matrix elements, colour connections and PDFs keep
c their physical momenta outside this auxiliary decay frame.
      implicit none
      include 'nexternal.inc'
      integer i_fks,j_fks,jborn
      double precision p(0:3,nexternal),plocal(0:3,nexternal),
     $     kn,knbar,kn0,mass2,q(0:3),qrest(0:3),mother(0:3),dot,rho
      logical pass
      external dot,rho
      plocal=p
      q=sum(p(:,1:nincoming),dim=2)
      mass2=dot(q,q)
      if(.not.resonance_recoil)return
      call prepare_resonance_frame(p,resonance_members,q,qrest,
     $     mass2,plocal,pass)
      if(.not.pass.or.j_fks.le.nincoming)then
         write(*,*) 'Invalid local shower recoil frame'
         stop 1
      endif
      jborn=j_fks
      if(j_fks.gt.i_fks)jborn=jborn-1
      call boostx(p_born(:,jborn),qrest,mother)
      kn=rho(plocal(:,j_fks))
      knbar=rho(mother)
      kn0=plocal(0,j_fks)
      end
