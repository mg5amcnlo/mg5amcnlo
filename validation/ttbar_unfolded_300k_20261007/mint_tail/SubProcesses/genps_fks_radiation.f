      module fks_radiation_maps
c Forward and inverse FKS radiation maps, including the coupled ISR
c map without event projection. Recoil adapters use this module without
c depending on the higher-level phase-space orchestration.
      use fks_phase_space_helpers, only: getangles,get_recoil,
     $     get_massive_fsr_bounds,boost_isr_recoil
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      private
      public generate_momenta_massless_final,
     $     generate_momenta_massless_final_inverse,
     $     generate_momenta_massive_final,
     $     generate_momenta_massive_final_inverse,
     $     generate_momenta_initial,generate_momenta_initial_inverse,
     $     generate_momenta_initial_noevpr,native_fsr_angle,
     $     use_symmetric_isr_mapping

      contains

      logical function use_symmetric_isr_mapping()
c Run settings are not part of a saved phase-space point. Resolve them
c identically for generation, inverse projection and the lepton chart.
      use FKSParams, only: get_fks_isr_mapping
      implicit none
      logical fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
      character*10 shower_mc
      common /cMonteCarloType/shower_mc
      use_symmetric_isr_mapping=
     $     get_fks_isr_mapping(fixed_order,nlo_ps,shower_mc).eq.1
      end function use_symmetric_isr_mapping



      subroutine generate_momenta_massless_final(icountevts,i_fks,j_fks
     &     ,p_born_imother,shat,sqrtshat,x,xmrec2,xp,phi_i_fks,xiimax
     &     ,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt
     &     ,pass)
      use fks_phase_space_data, only: veckn_ev,veckbarn_ev,xp0jfks,xij_aor
      use mc_native_context, only: native_mapping
      implicit none
c arguments
      integer icountevts,i_fks,j_fks
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision E_i_fks,x3len_i_fks,x3len_j_fks,x3len_fks_mother
     &     ,costh_i_fks,sinth_i_fks,xpifksred(0:3),th_mother_fks
     &     ,costh_mother_fks,sinth_mother_fks, phi_mother_fks
     &     ,cosphi_mother_fks,sinphi_mother_fks,recoil(0:3),sumrec
     &     ,sumrec2,betabst,gammabst,shybst,chybst,chybstmo,xdir(3)
     &     ,veckn,veckbarn,xp_mother(0:3),cosphi_i_fks
     &     ,sinphi_i_fks,sin_ij_fks
      double complex resAoR0
      double precision native_delta,native_onepy,native_denom
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      double precision stiny,sstiny,qtiny,ctiny,cctiny
      double complex ximag
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (ctiny=5d-7)
      parameter (ximag=(0d0,1d0))
c
      pass=.true.
      if(softtest.or.colltest.or.native_mapping)then
        sstiny=0.d0
        cctiny=0.d0
      else
        sstiny=stiny
        cctiny=ctiny
      endif
c
c set-up y_ij_fks
c
      if(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=dble(sign(1,icountevts))
         sin_ij_fks=0d0
         if (.not. colltest) then
            if(native_mapping)then
               xjac=xjac*pi*sin(pi*min(x(2),1d0-x(2)))
            else
            xjac=xjac*2d0*x(2)*2d0
            endif
         else
            continue ! do not include jacobian for y in tests
         endif
      elseif (colltest) then
         y_ij_fks = y_ij_fks_fix
         sin_ij_fks=sqrt(max(0d0,(1d0-y_ij_fks)*(1d0+y_ij_fks)))
      elseif(native_mapping)then
c As in the massive native chart, retain the opening angle itself.
c The old sqrt((1-y)/2) coordinate loses the transverse component
c when two hard daughters are antiparallel and the recoil is soft.
         y_ij_fks=cos(pi*x(2))
         sin_ij_fks=sin(pi*min(x(2),1d0-x(2)))
         xjac=xjac*pi*sin_ij_fks
      else
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
         sin_ij_fks=sqrt(max(0d0,(1d0-y_ij_fks)*(1d0+y_ij_fks)))
         xjac=xjac*2d0*x(2)*2d0
      endif

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute maximum allowed xi_i_fks
      xiimax=1-xmrec2/shat
      xinorm=xiimax
c
c Define xi_i_fks
c
      if(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=0d0
         if (.not.softtest) then
            xjac=xjac*2d0*x(1)
         else
            continue ! no jacobian for xi in tests
         endif
      elseif (softtest) then
         xi_i_fks=xi_i_fks_fix*xiimax
      else
         xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         xi_i_fks=xi_i_hat*xiimax
         xjac=xjac*2d0*x(1)
      endif

c Check that xii is in the allowed range
      if( icountevts.eq.-100 .or. abs(icountevts).eq.1 )then
         if(xi_i_fks.gt.(1-xmrec2/shat))then
            xjac=-101
            pass=.false.
            return
         endif
      elseif(icountevts.eq.0 .or. abs(icountevts).eq.2)then
c May insert here a check on whether xii<xicut, rather than doing it
c in the cross sections
         continue
      endif
c
c Compute costh_i_fks from xi_i_fks et al.
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      if(native_mapping)then
c Both numerator and denominator vanish near xi=1,y=-1. Keep
c their small remainders from the radiation coordinates themselves.
         native_delta=xiimax-xi_i_fks
         if((icountevts.eq.-100.or.abs(icountevts).eq.1)
     $        .and..not.softtest)
     $        native_delta=xiimax*(1d0-x(1))*(1d0+x(1))
         native_onepy=1d0+y_ij_fks
         if(icountevts.eq.-100.or.icountevts.eq.0)then
            if(.not.colltest)
     $           native_onepy=2d0*sin(pi*(1d0-x(2))/2d0)**2
         endif
         native_denom=2d0*(xmrec2/shat+native_delta)
     $        +xi_i_fks*native_onepy
         x3len_j_fks=sqrtshat*native_delta/native_denom
      else
      x3len_j_fks=(shat-xmrec2-2*sqrtshat*x3len_i_fks)/
     &             (2*(sqrtshat-x3len_i_fks*(1-y_ij_fks)))
      endif
c Resolve the daughter parallel and transverse to the emitted momentum.
c This avoids subtracting squared momenta when the recoil is nearly at rest.
c Use this also in the outer map: if j is soft, computing sin(theta) from
c a rounded cos(theta)=1 makes the daughters spuriously collinear/off shell
c and the native inverse map cannot recover a positive Jacobian.
      costh_i_fks=x3len_i_fks+x3len_j_fks*y_ij_fks
      if(native_mapping)costh_i_fks=(x3len_i_fks-x3len_j_fks)
     $     +x3len_j_fks*native_onepy
      sinth_i_fks=x3len_j_fks*sin_ij_fks
      x3len_fks_mother=sqrt(costh_i_fks**2+sinth_i_fks**2)
      costh_i_fks=costh_i_fks/x3len_fks_mother
      sinth_i_fks=sinth_i_fks/x3len_fks_mother
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c The momentum if i_fks and j_fks
c
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
      sumrec=recoil(0)+rho(recoil)
      sumrec2=sumrec**2
      betabst=-(shat-sumrec2)/(shat+sumrec2)
      gammabst=1/sqrt(1-betabst**2)
      shybst=-(shat-sumrec2)/(2*sumrec*sqrtshat)
      chybst=(shat+sumrec2)/(2*sumrec*sqrtshat)
c cosh(y) is very often close to one, so define cosh(y)-1 as well
      chybstmo=(sqrtshat-sumrec)**2/(2*sumrec*sqrtshat)
c Use the Born mother direction also when the daughters nearly cancel.
      xdir(1)=sinth_mother_fks*cosphi_mother_fks
      xdir(2)=sinth_mother_fks*sinphi_mother_fks
      xdir(3)=costh_mother_fks
c     Perform the boost here
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Collinear limit of <ij>/[ij]. See innerp3.m.
      if( ( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) ) )then
         resAoR0=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c The term O(srt(1-y)) is formally correct but may be numerically large
c Set it to zero
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif
c
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)
      return
      end subroutine generate_momenta_massless_final


      subroutine generate_momenta_massless_final_inverse(xp,xi_i_fks
     $     ,y_ij_fks,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks
     $     ,j_fks)
      use fks_phase_space_data, only: xinorm_ev,xij_aor
      ! TODO: probably need xiimax as argument, to update the prefactors.
      use mc_native_context, only: native_mapping
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,th_mother_fks,costh_mother_fks,sinth_mother_fks
     $     ,phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks
      integer i_fks,j_fks
      double precision recoil(0:3),sumrec,sumrec2,betabst
     $     ,gammabst,shybst,chybst,chybstmo,xdir(1:3),veckn,veckbarn
     $     ,xiimax,xmrec2
      logical        softtest,colltest
      common/sctests/softtest,colltest
      logical pass
      integer i
      double precision rho,sstiny,cctiny
      external rho
      sstiny=1d-6
      cctiny=5d-7
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif

c Retain the recoil directly instead of subtracting the hard daughters.
      recoil=0d0
      do i=nincoming+1,nexternal
         if(i.eq.i_fks.or.i.eq.j_fks)cycle
         recoil=recoil+xp(:,i)
      enddo
      sumrec=recoil(0)+rho(recoil)
      sumrec2=sumrec**2
! shat_born=shat for final state j_fks.
      betabst=-(shat-sumrec2)/(shat+sumrec2)
      gammabst=1/sqrt(1-betabst**2)
      shybst=-(shat-sumrec2)/(2*sumrec*sqrtshat)
      chybst=(shat+sumrec2)/(2*sumrec*sqrtshat)
c     cosh(y) is very often close to one, so define cosh(y)-1 as well
      chybstmo=(sqrtshat-sumrec)**2/(2*sumrec*sqrtshat)
      xdir(1:3)=recoil(1:3)/rho(recoil)
c     Perform the boost here
      do i=nincoming+1,nexternal
!         if(i.eq.j_fks.or.shybst.eq.0.d0) cycle
         if(i.eq.j_fks) cycle
         if (i.lt.i_fks) then
            call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),
     &           p_born(0,i))
         elseif (i.gt.i_fks) then
            call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),
     &           p_born(0,i-1))
         endif
      enddo
      p_born(:,1:nincoming)=xp(:,1:nincoming)
      p_born(:,j_fks)=sum(xp(:,1:nincoming),dim=2)
      do i=nincoming+1,nexternal-1
         if (i.eq.j_fks) cycle
         p_born(0:3,j_fks)=p_born(0:3,j_fks)-p_born(0:3,i)
      enddo

c     Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born(0,j_fks))
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)

!     random number associated with xi_i_fks
      call get_recoil(p_born(0,1),j_fks,shat,xmrec2,pass)
      xiimax=1d0-xmrec2/shat
      xinorm_ev=xiimax
      x(1)=(xi_i_fks/xiimax-sstiny)/(1d0-sstiny)
      if (x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12) then
         xjac=-102d0
         return
      endif
      x(1)=sqrt(max(0d0,min(1d0,x(1))))
      xjac=xjac*2d0*x(1)

!     random number associated with y_ij_fks
      if(native_mapping)then
         x(2)=native_fsr_angle(xp(:,i_fks),xp(:,j_fks))
         xjac=xjac*pi*sin(pi*min(x(2),1d0-x(2)))
      else
      x(2)=((1d0-y_ij_fks)/2d0-cctiny)/(1d0-cctiny)
      if (x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12) then
         xjac=-33d0
         return
      endif
      x(2)=sqrt(max(0d0,min(1d0,x(2))))
      xjac=xjac*2d0*x(2)*2d0
      endif

!     random number associated with phi_i_fks
      x(3)=phi_i_fks/(2d0*pi)
      xjac=xjac*2d0*pi

c Collinear limit of <ij>/[ij]. See innerpin.m.
      call getangles(p_born(0:3,j_fks),
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
      xij_aor=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
      end subroutine generate_momenta_massless_final_inverse


      subroutine generate_momenta_massive_final(icountevts,isolsign
     &     ,i_fks,j_fks,p_born_imother,shat
     &     ,sqrtshat,m_j_fks,x,xmrec2,xp,phi_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,pass)
      use fks_phase_space_data, only: veckn_ev,veckbarn_ev,xp0jfks
      use mc_native_context, only: native_mapping
      implicit none
c arguments
      integer icountevts,i_fks,j_fks,isolsign
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),m_j_fks,phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision xmj,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2
     $     ,cffDEL2,xiBm,ximax,rat_xi
     $     ,E_i_fks,x3len_i_fks,b2m4ac,x3len_j_fks_num,x3len_j_fks_den
     $     ,x3len_j_fks,x3len_fks_mother,costh_i_fks,sinth_i_fks
     $     ,xpifksred(0:3),recoil(0:3),xp_mother(0:3),sumrec,expybst
     $     ,shybst,chybst,chybstmo,xdir(3),veckn,veckbarn ,cosphi_i_fks
     $     ,sinphi_i_fks,cosphi_mother_fks,costh_mother_fks
     $     ,phi_mother_fks,sinphi_mother_fks,th_mother_fks
     $     ,sinth_mother_fks,sin_ij_fks
      double precision native_u,native_uborn,native_eborn,native_ej,
     $     native_denom,native_radial,native_ps,native_sign,
     $     native_delta,native_onepy
      double precision zero_recoil_delta,root_conjugate
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision stiny,sstiny,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (ctiny=5d-7)
c
      if(colltest .or.
     &     abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
         write(*,*)'Error #5 in genps_fks.f:'
         write(*,*)
     &        'This parametrization cannot be used in FS coll limits'
         stop
      endif
c
      pass=.true.
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0.d0
         cctiny=0.d0
      else
         sstiny=stiny
         cctiny=ctiny
      endif
c
c set-up y_ij_fks
c

      if(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         write (*,*) 'Massive j_fks: should have no '/
     $        /'(soft-)collinear contribution'
         stop 1
      elseif (colltest) then
         y_ij_fks = y_ij_fks_fix
         write (*,*) 'Massive j_fks: should not do collinear tests'
         stop 1
         ! do not care about jacobian
      elseif (native_mapping) then
c An angle coordinate retains the small transverse component at y=-1.
c Encoding it only in 1-y loses precision for nearly stationary recoils.
         y_ij_fks=cos(pi*x(2))
         sin_ij_fks=sin(pi*min(x(2),1d0-x(2)))
         xjac=xjac*pi*sin_ij_fks
      else
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
c Here y=1-2*q. Factor 1-q=(1-cctiny)*(1-x)*(1+x)
c to retain the transverse component near the antiparallel endpoint.
         sin_ij_fks=2d0*sqrt(max(0d0,
     $        (cctiny+(1d0-cctiny)*x(2)**2)*(1d0-cctiny)*
     $        (1d0-x(2))*(1d0+x(2))))
         xjac=xjac*2d0*x(2)*2d0
      endif

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute the maximum allowed xi_i_fks
c
      xmj=m_j_fks
      xmj2=xmj**2
      xmjhat=xmj/sqrtshat
      xmhat=sqrt(xmrec2)/sqrtshat
      call get_massive_fsr_bounds(shat,sqrtshat,m_j_fks,xmrec2,
     $     y_ij_fks,xim,xiBm,ximax,cffA2,cffB2,cffC2,cffDEL2,
     $     xiimax,xinorm)
      if(xmrec2.eq.0d0)zero_recoil_delta=(shat-xmj2)/shat
      if(xiBm.lt.(xim-1.d-8).or.xim.lt.0.d0.or.xiBm.lt.0.d0.or.
     &     xiBm.gt.(ximax+1.d-8).or.ximax.gt.1.or.ximax.lt.0.d0)then
         write(*,*)'WARNING #4 in one_tree',xim,xiBm,ximax
         xjac=-104d0
         pass=.false.
         return
      endif
      rat_xi=xiimax/xinorm
c
c Generate xi_i_fks
c
c

      if(native_mapping)then
c Parameterize the radiator momentum, u=uBorn*(1-x**2), instead of
c solving the quadratic for u at fixed xi. This covers both solutions
c continuously and avoids loss of precision where they coalesce.
         native_uborn=sqrtshat*sqrt(cffC2)/2d0
         native_eborn=sqrt(native_uborn**2+m_j_fks**2)
         native_delta=native_uborn*x(1)**2
         native_u=native_uborn-native_delta
         native_ej=sqrt(native_u**2+m_j_fks**2)
c Rationalize W-E_j+u*y. Its terms nearly cancel for a soft
c massless recoil and antiparallel daughters. Keep 1+y through the
c half angle, and uBorn-u through the sampled radial coordinate.
         native_onepy=2d0*sin(pi*(1d0-x(2))/2d0)**2
         native_denom=xmrec2/(sqrtshat-native_eborn+native_uborn)
     $        +native_delta*(1d0+(native_uborn+native_u)/
     $        (native_eborn+native_ej))+native_u*native_onepy
         xi_i_fks=2d0*native_delta*
     $        (native_uborn+native_u)/
     $        ((native_eborn+native_ej)*native_denom)
         native_sign=(2d0-xi_i_fks)*native_u+
     $        xi_i_fks*native_ej*y_ij_fks
         native_radial=native_uborn*abs(native_sign)/
     $        (native_ej*native_denom)
         xinorm=1d0
         xjac=xjac*2d0*x(1)
         if(icountevts.eq.0)then
            xi_i_fks=0d0
            isolsign=1
            x3len_j_fks=native_uborn
            native_ps=shat/(4d0*pi)**3*native_radial
         else
            isolsign=int(sign(1d0,native_sign))
            x3len_j_fks=native_u
c Combine du/dxi with the real phase-space factor analytically: each
c separately becomes singular at the boundary between the two solutions.
            native_ps=2d0*shat/(4d0*pi)**3*native_u**2/
     $           (native_ej*native_denom)
         endif
         xi_i_hat=xi_i_fks
      elseif(icountevts.eq.0)then
         xi_i_fks=0d0
         isolsign=1
         if (x(1).le.rat_xi .and. (.not.softtest)) xjac=xjac*2d0*x(1)/rat_xi
      elseif (softtest) then
         if(xi_i_fks_fix.gt.xiimax)then
            xjac=-102
            pass=.false.
            return
         endif
         xi_i_fks=xi_i_fks_fix
         isolsign=1
      else
c Map regions (0,A) and (A,1) in x(1) onto (0,rat_xi) and (rat_xi,1)
c in xi_i_hat respectively. The parameter A is free, but it appears to be
c convenient to choose A=rat_xi
         if(x(1).le.rat_xi)then
            xi_i_hat=(sstiny+(1-sstiny)*x(1)**2)/rat_xi
            xi_i_fks=xinorm*xi_i_hat
            isolsign=1
            if (.not.softtest) xjac=xjac*2d0*x(1)/rat_xi
         else
            xi_i_hat=sstiny+(1-sstiny)*x(1)
            xi_i_fks=-xinorm*xi_i_hat+2*xiimax
            isolsign=-1
         endif
      endif

      if(isolsign.eq.0)then
         write(*,*)'Fatal error #11 in one_tree',isolsign
         stop
      endif
c
c Compute costh_i_fks
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      if(.not.native_mapping)then
      b2m4ac=xi_i_fks**2*cffA2 + xi_i_fks*cffB2 + cffC2
      if(xmrec2.eq.0d0)b2m4ac=(zero_recoil_delta-xi_i_fks)**2
     $     -xmjhat**2*xi_i_fks**2*(1d0-y_ij_fks**2)
      if(b2m4ac.le.0.d0)then
         if(abs(b2m4ac).lt.1.d-3)then
            b2m4ac=0.d0
         else
            write(*,*)'Fatal error #6 in one_tree'
            write(*,*)b2m4ac,xi_i_fks,cffA2,cffB2,cffC2
            write(*,*)y_ij_fks,xim,xiBm
            stop
         endif
      endif
      x3len_j_fks_num=-xi_i_fks*y_ij_fks*
     &                (1-xmhat**2+xmjhat**2-xi_i_fks) +
     &                (2-xi_i_fks)*sqrt(b2m4ac)*isolsign
      x3len_j_fks_den=(2-xi_i_fks*(1-y_ij_fks))*
     &                (2-xi_i_fks*(1+y_ij_fks))
      x3len_j_fks=sqrtshat*x3len_j_fks_num/x3len_j_fks_den
      if(xmrec2.eq.0d0)then
         root_conjugate=xi_i_fks*y_ij_fks*
     $        (1d0+xmjhat**2-xi_i_fks)+
     $        (2d0-xi_i_fks)*sqrt(b2m4ac)*isolsign
         if(abs(x3len_j_fks_num).lt.0.01d0*
     $        abs(root_conjugate))
     $        x3len_j_fks=sqrtshat*zero_recoil_delta*
     $        (xim-xi_i_fks)*(1d0+xmjhat-xi_i_fks)/
     $        root_conjugate
      endif
      if(x3len_j_fks.lt.0.d0)then
         write(*,*)'WARNING #7 in one_tree',
     &        x3len_j_fks_num,x3len_j_fks_den,xi_i_fks,y_ij_fks
         xjac=-107d0
         pass=.false.
         return
      endif
      endif
c Keep both angular components when the two daughters nearly cancel.
c Each map supplies the sine from its own sampling coordinate above.
      costh_i_fks=x3len_i_fks+x3len_j_fks*y_ij_fks
      sinth_i_fks=x3len_j_fks*sin_ij_fks
      x3len_fks_mother=sqrt(costh_i_fks**2+sinth_i_fks**2)
      costh_i_fks=costh_i_fks/x3len_fks_mother
      sinth_i_fks=sinth_i_fks/x3len_fks_mother
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c Generate momenta for j_fks and i_fks
c
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2+m_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
c
      sumrec=recoil(0)+rho(recoil)
c Rationalise the boost expression so a nearly massless recoil does
c not subtract two quantities of order shat and then divide by its mass.
      expybst=2d0*sqrtshat*sumrec/
     &     (shat+xmrec2-xmj2+shat*sqrt(cffC2))
      if(expybst.le.0.d0)then
         write(*,*)'Fatal error #10 in one_tree',expybst
         stop
      endif
      shybst=(expybst-1/expybst)/2.d0
      chybst=(expybst+1/expybst)/2.d0
      chybstmo=chybst-1.d0
c
c Use the original mother direction. Adding nearly opposite hard
c daughters corrupts its norm, which a large recoil boost amplifies.
      xdir(1)=sinth_mother_fks*cosphi_mother_fks
      xdir(2)=sinth_mother_fks*sinphi_mother_fks
      xdir(3)=costh_mother_fks

c Boost the momenta
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif
c
      if(native_mapping)then
         xpswgt=xpswgt*native_ps
      else
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      endif
      xpswgt=abs(xpswgt)
      return
      end subroutine generate_momenta_massive_final


      subroutine generate_momenta_massive_final_inverse(xp,xi_i_fks
     $           ,y_ij_fks,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat
     $           ,i_fks,j_fks,m_j_fks)
      use fks_phase_space_data, only: xinorm_ev
      use mc_native_context, only: native_mapping
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,m_j_fks
      integer i_fks,j_fks
      double precision recoil(0:3),sumrec,sumrec2,xmj
     $     ,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2,cffDEL2,xiBm,ximax
     $     ,xiimax,xinorm,rat_xi,expybst
     $     ,shybst,chybst,chybstmo,veckn,veckbarn,xdir(3),xmrec2
      integer i
      double precision native_uborn,native_u,native_denom,
     $     native_eborn,native_delta,native_onepy,native_ratio,
     $     native_recoil,native_inverse_denom
      double precision rho,dot,sstiny,cctiny,branch_sign
      external rho,dot
      logical        softtest,colltest
      common/sctests/softtest,colltest
      sstiny=1d-6
      cctiny=5d-7
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif
      ! y_ij_fks
      if(native_mapping)then
         x(2)=native_fsr_angle(xp(:,i_fks),xp(:,j_fks))
         xjac=xjac*pi*sin(pi*min(x(2),1d0-x(2)))
      else
      x(2)=((1d0-y_ij_fks)/2d0-cctiny)/(1d0-cctiny)
      if (x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12) then
         xjac=-33d0
         return
      endif
      x(2)=sqrt(max(0d0,min(1d0,x(2))))
      xjac=xjac*2d0*x(2)*2d0
      endif

      ! x_i_fks
c Summing the spectators retains a soft recoil and its invariant mass;
c subtracting the hard daughters from the beams loses that precision.
      recoil=0d0
      do i=nincoming+1,nexternal
         if(i.eq.i_fks.or.i.eq.j_fks)cycle
         recoil=recoil+xp(:,i)
      enddo
      sumrec=recoil(0)+rho(recoil)
      xmrec2=dot(recoil,recoil)
c Apply the same massless-recoil roundoff bound as get_recoil before
c taking a square root in the inverse map.
      if(xmrec2.lt.0d0.and.xmrec2.ge.-1d-12*shat)xmrec2=0d0
      xmj=m_j_fks
      xmj2=xmj**2
      xmjhat=xmj/sqrtshat
      xmhat=sqrt(xmrec2)/sqrtshat
      call get_massive_fsr_bounds(shat,sqrtshat,m_j_fks,xmrec2,
     $     y_ij_fks,xim,xiBm,ximax,cffA2,cffB2,cffC2,cffDEL2,
     $     xiimax,xinorm)
      rat_xi=xiimax/xinorm
      xinorm_ev=xinorm

c Recover the sign of the square root in the forward expression for
c |p_j|. At fixed xi and y both solutions can exist, but the supplied
c momentum selects one of them; choosing randomly changes the event.
      if(native_mapping)then
         native_uborn=sqrtshat*sqrt(cffC2)/2d0
         native_eborn=sqrt(native_uborn**2+xmj2)
         native_u=rho(xp(:,j_fks))
         native_delta=native_uborn-native_u
         native_onepy=2d0*sin(pi*(1d0-x(2))/2d0)**2
         native_ratio=(native_uborn+native_u)/
     $        (native_eborn+xp(0,j_fks))
         native_recoil=xmrec2/(sqrtshat-native_eborn+native_uborn)
c Close to uBorn, infer the small radial difference from the emitted
c energy and angle. Subtracting the two momenta amplifies their mass-shell
c roundoff when the recoil is soft, spoiling the forward reconstruction.
         native_inverse_denom=sqrtshat*native_ratio-
     $        xp(0,i_fks)*(1d0+native_ratio)
         if(abs(native_delta).lt.1d-4*native_uborn.and.
     $        native_inverse_denom.gt.0d0)then
            native_delta=xp(0,i_fks)*
     $           (native_recoil+native_u*native_onepy)/
     $           native_inverse_denom
         endif
         native_denom=native_recoil+native_delta*(1d0+native_ratio)
     $        +native_u*native_onepy
         x(1)=native_delta/native_uborn
         if(x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12)then
            xjac=-102d0
            return
         endif
         x(1)=sqrt(max(0d0,min(1d0,x(1))))
         xjac=xjac*2d0*x(1)
         xinorm_ev=1d0
      else
      branch_sign=rho(xp(0,j_fks))/sqrtshat*
     $     (2-xi_i_fks*(1-y_ij_fks))*(2-xi_i_fks*(1+y_ij_fks))
     $     +xi_i_fks*y_ij_fks*(1-xmhat**2+xmjhat**2-xi_i_fks)
      if (branch_sign.ge.0d0) then
         x(1)=(xi_i_fks*rat_xi/xinorm-sstiny)/(1d0-sstiny)
         if (x(1).lt.-1d-12.or.x(1).gt.rat_xi**2+1d-12) then
            xjac=-102d0
            return
         endif
         x(1)=sqrt(max(0d0,min(rat_xi**2,x(1))))
         xjac=xjac*2*x(1)/rat_xi
      else
         x(1)=((2*xiimax-xi_i_fks)/xinorm-sstiny)/(1d0-sstiny)
         if (x(1).lt.rat_xi-1d-12.or.x(1).gt.1d0+1d-12) then
            xjac=-102d0
            return
         endif
         x(1)=max(rat_xi,min(1d0,x(1)))
      endif

      endif

      ! phi_i_fks
      x(3)=phi_i_fks/(2d0*pi)
      xjac=xjac*2d0*pi

c This conjugate form is also regular at zero recoil mass.
      expybst=2d0*sqrtshat*sumrec/
     &     (shat+xmrec2-xmj2+shat*sqrt(cffC2))
      if(expybst.le.0.d0)then
         write(*,*)'Fatal error #10 in one_tree',expybst
         stop
      endif
      shybst=(expybst-1/expybst)/2.d0
      chybst=(expybst+1/expybst)/2.d0
      chybstmo=chybst-1.d0
      xdir(1:3)=recoil(1:3)/rho(recoil)
c Boost the momenta
      do i=nincoming+1,nexternal
         if(i.eq.j_fks) cycle
         if (i.lt.i_fks) then
            call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),
     &           p_born(0,i))
         elseif (i.gt.i_fks) then
            call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),
     &           p_born(0,i-1))
         endif
      enddo
      p_born(:,1:nincoming)=xp(:,1:nincoming)
      p_born(:,j_fks)=sum(xp(:,1:nincoming),dim=2)
      do i=nincoming+1,nexternal-1
         if (i.eq.j_fks) cycle
         p_born(0:3,j_fks)=p_born(0:3,j_fks)-p_born(0:3,i)
      enddo
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born(0,j_fks))
      if(native_mapping)then
         xpswgt=xpswgt*2d0*shat/(4d0*pi)**3*native_u**2/
     $        (xp(0,j_fks)*native_denom)
      else
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      endif
      xpswgt=abs(xpswgt)
      end subroutine generate_momenta_massive_final_inverse


      double precision function native_fsr_angle(p1,p2)
c Return theta/pi using half-angle vectors, retaining pi-theta for
c nearly antiparallel daughters without subtracting their cosines.
      implicit none
      double precision p1(0:3),p2(0:3),u(3),v(3),shalf,chalf,pi
      parameter(pi=3.1415926535897932d0)
      u=p1(1:3)/sqrt(sum(p1(1:3)**2))
      v=p2(1:3)/sqrt(sum(p2(1:3)**2))
      shalf=sqrt(sum((u-v)**2))
      chalf=sqrt(sum((u+v)**2))
      if(shalf.gt.chalf)then
         native_fsr_angle=1d0-2d0/pi*atan2(chalf,shalf)
      else
         native_fsr_angle=2d0/pi*atan2(shalf,chalf)
      endif
      end function native_fsr_angle


      subroutine generate_momenta_initial(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x
     &     ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,pass)
      implicit none
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat,xi_i_fks,shat,sqrtshat,tau,ycm
     &     ,xbjrk(2),p_i_fks(0:3)
      logical pass
c Use the same run-level choice for the event and every counterevent.
      if(use_symmetric_isr_mapping())then
         call generate_momenta_initial_symmetric(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x
     &     ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,pass)
      else
         call generate_momenta_initial_asymmetric(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x
     &     ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,pass)
      endif
      end subroutine generate_momenta_initial


      subroutine generate_momenta_initial_asymmetric(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x
     &     ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac,pass)
c Asymmetric initial-state event projection, retained for shower matching.
c Only the emitting beam changes fraction: x_j=bar{x}_j/(1-xi).
c All hard final momenta undergo the same light-cone Lorentz map. Its
c soft and emitter-collinear limits leave the underlying Born unchanged.
      use fks_phase_space_data, only: xij_aor
      use mc_native_context, only: native_mapping
      implicit none
      integer icountevts,i_fks,j_fks,idir,i
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat,xi_i_fks,shat,sqrtshat,tau,ycm
     &     ,xbjrk(2),p_i_fks(0:3),xiimin,z,sqrtz,sqrtborn
     &     ,pplus,pminus,sintheta,sstiny,cctiny
      logical pass,softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision stiny,ctiny,pi
      parameter (stiny=1d-6,ctiny=5d-7,pi=3.1415926535897932d0)

      pass=.true.
      if(j_fks.ne.1.and.j_fks.ne.2)then
         write(*,*) 'Invalid ISR emitter in generate_momenta_initial_asymmetric'
         stop 1
      endif
      idir=3-2*j_fks
      sstiny=stiny
      cctiny=ctiny
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif
c The angular domain is independent of the incoming fractions, even
c when the real invariant mass has a larger lower bound than the Born.
      if(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=dble(sign(1,icountevts))
      elseif(colltest)then
         y_ij_fks=y_ij_fks_fix
      else
         y_ij_fks=1d0-2d0*(cctiny+(1d0-cctiny)*x(2)**2)
      endif
      if(abs(y_ij_fks).gt.1d0)then
         xjac=-33d0
         pass=.false.
         return
      endif
      if(.not.colltest)xjac=xjac*4d0*x(2)*(1d0-cctiny)

      call get_isr_radiation_bounds(j_fks,xbjrk_born,tau_born,
     &     xiimin,xiimax)
      xinorm=xiimax-xiimin
      if(xinorm.le.0d0)then
         xjac=-342d0
         pass=.false.
         return
      endif
      if(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=0d0
      elseif(softtest)then
         xi_i_fks=xi_i_fks_fix
      else
         xi_i_hat=sstiny+(1d0-sstiny)*x(1)**2
         xi_i_fks=xiimin+xinorm*xi_i_hat
      endif
      if(.not.softtest)xjac=xjac*2d0*x(1)*(1d0-sstiny)
      if(xi_i_fks.lt.0d0.or.xi_i_fks.gt.xiimax.or.
     &     xi_i_fks.ge.1d0)then
         xjac=-102d0
         pass=.false.
         return
      endif

      z=1d0-xi_i_fks
      sqrtz=sqrt(z)
      sqrtborn=sqrt(shat_born)
      tau=tau_born/z
      ycm=ycm_born-idir*log(z)/2d0
      shat=shat_born/z
      sqrtshat=sqrtborn/sqrtz
      xbjrk=xbjrk_born
      xbjrk(j_fks)=xbjrk_born(j_fks)/z
c Work in the underlying Born CM (the common tilde frame), so that
c the usual single Born-to-lab boost also applies to the real event.
      do i=3,nexternal
         if(i.ne.i_fks)call boost_isr_recoil(xp(0,i),xp(0,i),
     &        xi_i_fks,y_ij_fks,phi_i_fks,idir,.false.)
      enddo
      xp(0,1:2)=sqrtborn/2d0
      xp(0,j_fks)=sqrtborn/(2d0*z)
      xp(1:2,1:2)=0d0
      xp(3,1)=xp(0,1)
      xp(3,2)=-xp(0,2)
c Keep the energy-divided radiation direction at the soft endpoint.
c The plus/minus components below are relative to the emitting beam.
      sintheta=sqrt((1d0-y_ij_fks)*(1d0+y_ij_fks))
      pplus=sqrtborn*(1d0+y_ij_fks)/(2d0*z)
      pminus=sqrtborn*(1d0-y_ij_fks)/2d0
      p_i_fks(0)=(pplus+pminus)/2d0
      p_i_fks(1)=sqrtshat*sintheta*cos(phi_i_fks)/2d0
      p_i_fks(2)=sqrtshat*sintheta*sin(phi_i_fks)/2d0
      p_i_fks(3)=idir*(pplus-pminus)/2d0
      xp(:,i_fks)=xi_i_fks*p_i_fks
      if(icountevts.eq.-100.or.
     &     (icountevts.eq.1.and.xij_aor.eq.0))then
         xij_aor=-exp(2*idir*ximag*phi_i_fks)
      endif
c Lorentz invariance preserves the hard phase-space measure. The 1/z
c is the Bjorken Jacobian; the FKS weight convention supplies xi later.
c xiimax/xinorm also feed the finite FKS endpoint logarithms.
      xpswgt=abs(xpswgt*shat/(4d0*pi)**3/z)
      end subroutine generate_momenta_initial_asymmetric


      subroutine generate_momenta_initial_symmetric(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt ,xjac ,pass)
      use fks_phase_space_data,only: tau_lower_bound,xij_aor
      use mc_native_context, only: native_mapping
      implicit none
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3)
      logical pass
c common blocks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j,idir
      double precision yijdir,costh_i_fks,omega,bstfact,
     $     shy_tbst,chy_tbst,chy_tbstmo
     $     ,xdir_t(3),cosphi_i_fks,sinphi_i_fks,shy_lbst,chy_lbst
     $     ,encmso2,E_i_fks,sinth_i_fks,xpifksred(0:3),xi_i_fks
     $     ,xiimin,yij_upp,yij_low,y_ij_fks_upp,y_ij_fks_low
      double complex resAoR0

      double precision ltau_born,e2ycm_born,em2ycm_born
c external
c
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      logical fks_as_is
      parameter (fks_as_is=.false.)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision stiny,sstiny,qtiny,zero,ctiny,cctiny,vtiny
      parameter (stiny=1d-6)
      parameter (vtiny=0d0)
      parameter (qtiny=1d-7)
      parameter (zero=0d0)
      parameter (ctiny=5d-7)
c
      pass=.true.
      if(j_fks.ne.1.and.j_fks.ne.2)then
         write(*,*) 'Invalid symmetric ISR emitter',j_fks
         stop 1
      endif
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0.d0
         cctiny=0.d0
      else
         sstiny=stiny
         cctiny=ctiny
      endif
c
c FKS for left or right incoming parton
c
      idir=0
      if(.not.fks_as_is)then
         if(j_fks.eq.1)then
            idir=1
         elseif(j_fks.eq.2)then
            idir=-1
         endif
      else
         idir=1
         write(*,*)'One_tree: option not checked'
         stop
      endif

      ! this is to overcome numerical instabilities in ee collisions
      if (1d0-tau_born.gt.stiny) then
        ltau_born = log(tau_born)
      else
        ltau_born = tau_born-1d0
      endif
      if (abs(ycm_born).gt.stiny) then
        e2ycm_born = exp(2*ycm_born)
        em2ycm_born = exp(-2*ycm_born)
      else
        e2ycm_born = 1d0 + 2*ycm_born + 2*ycm_born**2
        em2ycm_born = 1d0 - 2*ycm_born + 2*ycm_born**2
      endif
c
c set-up lower and upper bounds on y_ij_fks
c
      if( tau_born.le.tau_lower_bound) then
         if (ycm_born.gt. (0.5d0*ltau_born-log(tau_lower_bound)) )then
            yij_upp= (tau_lower_bound+tau_born)*
     &           ( 1-e2ycm_born*tau_lower_bound ) /
     &           ( (tau_lower_bound-tau_born)*
     &           (1+e2ycm_born*tau_lower_bound) )
         else
            yij_upp=1.d0
         endif
      else
         yij_upp=1.d0
      endif
      if( tau_born.le.tau_lower_bound) then
         if (ycm_born.lt. (-0.5d0*ltau_born+log(tau_lower_bound)) )then
            yij_low=-(tau_lower_bound+tau_born)*
     &           ( 1-em2ycm_born*tau_lower_bound ) /
     &           ( (tau_lower_bound-tau_born)*
     &           (1+em2ycm_born*tau_lower_bound) )
         else
            yij_low=-1.d0
         endif
      else
         yij_low=-1.d0
      endif
c
      if(idir.eq.1)then
         y_ij_fks_upp=yij_upp
         y_ij_fks_low=yij_low
      elseif(idir.eq.-1)then
         y_ij_fks_upp=-yij_low
         y_ij_fks_low=-yij_upp
      endif

      if(y_ij_fks_upp.le.y_ij_fks_low)then
         xjac=-33d0
         pass=.false.
         return
      endif
c
c set-up y_ij_fks
c

      if(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=dble(sign(1,icountevts))
         if (.not.colltest) then
            xjac=xjac*(y_ij_fks_upp-y_ij_fks_low)*x(2)*2d0
     $           *(1d0-cctiny)
         else
            continue ! do not include jacobian for y in tests
         endif
      elseif (colltest) then
         y_ij_fks = y_ij_fks_fix
      else
         y_ij_fks = y_ij_fks_upp -
     &        (y_ij_fks_upp-y_ij_fks_low)*(cctiny+(1-cctiny)*x(2)**2)
         xjac=xjac*(y_ij_fks_upp-y_ij_fks_low)*x(2)*2d0
     $        *(1d0-cctiny)
      endif
      if ( y_ij_fks.gt.y_ij_fks_upp .or.
     &     y_ij_fks.lt.y_ij_fks_low) then
         ! y_ij_fks is not in the allowed range, the counter-events do
         ! not need to be generated.
         xjac=-33d0
         pass=.false.
         return
      endif

c
c Compute costh_i_fks
c
      yijdir=idir*y_ij_fks
      costh_i_fks=yijdir
c
c Compute maximal xi_i_fks
c
      call get_symmetric_isr_radiation_bounds(j_fks,xbjrk_born,
     $     tau_born,y_ij_fks,xiimin,xiimax)
      if (xiimax.le.xiimin) then
         write (*,*) 'WARNING #10 in genps_fks.f',icountevts,xiimax
     $        ,xiimin,tau_born,tau_lower_bound
         xjac=-342d0
         pass=.false.
         return
      endif

      xinorm=xiimax-xiimin
      if( icountevts.ge.1 .and.
     &     ( (idir.eq.1.and.
     &     abs(xiimax-(1-xbjrk_born(1))).gt.1.d-5) .or.
     &     (idir.eq.-1.and.
     &     abs(xiimax-(1-xbjrk_born(2))).gt.1.d-5) ) )then
         write(*,*)'Fatal error #15 in one_tree'
         write(*,*)xiimax,xbjrk_born(1),xbjrk_born(2),idir
         stop
      endif
c
c Define xi_i_fks
c

      if(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=0d0
         if (.not.softtest) then
            xjac=xjac*2d0*x(1)*(1d0-sstiny)
         else
            continue ! no jacobian for xi in tests
         endif
      elseif (softtest) then
         xi_i_fks=xi_i_fks_fix
      else
         xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         xi_i_fks=xiimin+(xiimax-xiimin)*xi_i_hat
         xjac=xjac*2d0*x(1)*(1d0-sstiny)
      endif
      if(xi_i_fks.lt.0d0.or.xi_i_fks.ge.1d0.or.
     &     xi_i_fks.gt.xiimax)then
         ! xi_i_fks is not in the allowed range: no need to generate
         ! soft counter eevent kinematics.
         xjac=-102
         pass=.false.
         return
      endif
c
c Initial state variables are different for events and counterevents. Update them here.
c

      omega=sqrt( (2-xi_i_fks*(1+yijdir))/
     &     (2-xi_i_fks*(1-yijdir)) )
      if (icountevts.ne.0) then
         tau=tau_born/(1-xi_i_fks)
         ycm=ycm_born-log(omega)
         shat=tau*stot
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)/(sqrt(1-xi_i_fks)*omega)
         xbjrk(2)=xbjrk_born(2)*omega/sqrt(1-xi_i_fks)
      else
         tau=tau_born
         ycm=ycm_born
         shat=shat_born
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
      endif
c
c Define the boost factor here
c
      bstfact=sqrt( (2-xi_i_fks*(1-yijdir))*(2-xi_i_fks*(1+yijdir)) )
      shy_tbst=-xi_i_fks*sqrt(1-yijdir**2)/(2*sqrt(1-xi_i_fks))
      chy_tbst=bstfact/(2*sqrt(1-xi_i_fks))
      chy_tbstmo=chy_tbst-1.d0
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xdir_t(1)=-cosphi_i_fks
      xdir_t(2)=-sinphi_i_fks
      xdir_t(3)=zero
c
      shy_lbst=-xi_i_fks*yijdir/bstfact
      chy_lbst=(2-xi_i_fks)/bstfact
c Boost the momenta
      do i=3,nexternal
         if(i.ne.i_fks.and.shy_tbst.ne.0.d0)
     &        call boostwdir2(chy_tbst,shy_tbst,chy_tbstmo,xdir_t,
     &                        xp(0,i),xp(0,i))
      enddo
c
      encmso2=sqrtshat/2.d0
      p_i_fks(0)=encmso2
      E_i_fks=xi_i_fks*encmso2
      sinth_i_fks=sqrt(1-costh_i_fks**2)
c
      xp(0,1)=encmso2*(chy_lbst-shy_lbst)
      xp(1,1)=0.d0
      xp(2,1)=0.d0
      xp(3,1)=xp(0,1)
c
      xp(0,2)=encmso2*(chy_lbst+shy_lbst)
      xp(1,2)=0.d0
      xp(2,2)=0.d0
      xp(3,2)=-xp(0,2)
c
      xp(0,i_fks)=E_i_fks*(chy_lbst-shy_lbst*yijdir)
      p_i_fks(0)=p_i_fks(0)*(chy_lbst-shy_lbst*yijdir)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=chy_lbst*yijdir-shy_lbst
c
      do j=1,3
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         p_i_fks(j)=encmso2*xpifksred(j)
      enddo
c
c Collinear limit of <ij>/[ij]. See innerpin.m.
      if( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) )then
         resAoR0=-exp( 2*idir*ximag*phi_i_fks )
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii) * (tau,ycm)
      xpswgt=xpswgt*shat
      xpswgt=xpswgt/(4*pi)**3/(1-xi_i_fks)
      xpswgt=abs(xpswgt)
c
      return
      end subroutine generate_momenta_initial_symmetric


      subroutine generate_momenta_initial_inverse(xp,xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      implicit none
      integer i_fks,j_fks
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,stot,tau,ycm,xbjrk(2),tau_born,ycm_born,xbjrk_born(2)
     $     ,y_lab_to_cms
c Use the same run-level choice for the event and every counterevent.
      if(use_symmetric_isr_mapping())then
         call generate_momenta_initial_inverse_symmetric(xp,xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      else
         call generate_momenta_initial_inverse_asymmetric(xp,xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      endif
      end subroutine generate_momenta_initial_inverse


      subroutine generate_momenta_initial_inverse_asymmetric(xp,xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      use fks_phase_space_data, only: xinorm_ev,xij_aor
      use mc_native_context, only: native_mapping
      implicit none
      integer i_fks,j_fks,idir,i,iborn
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,stot,tau,ycm,xbjrk(2),tau_born,ycm_born,xbjrk_born(2)
     $     ,y_lab_to_cms,pred(0:3),z,sqrtborn,xiimax,xiimin,xinorm
     $     ,sstiny,cctiny,pplus,pminus,exp_y
      double precision pmass(nexternal),transverse_mass2
      common/to_mass/pmass
      double precision pi,stiny,ctiny
      parameter(pi=3.1415926535897932d0,stiny=1d-6,ctiny=5d-7)
      logical softtest,colltest
      common/sctests/softtest,colltest
      double complex ximag
      parameter(ximag=(0d0,1d0))
      if(j_fks.ne.1.and.j_fks.ne.2)then
         write(*,*) 'Invalid ISR emitter in generate_momenta_initial_inverse_asymmetric'
         stop 1
      endif
      idir=3-2*j_fks
      sstiny=stiny
      cctiny=ctiny
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif
      z=1d0-xi_i_fks
      if(z.le.0d0.or.z.gt.1d0)then
         xjac=-102d0
         return
      endif
      tau_born=tau*z
      ycm_born=ycm+idir*log(z)/2d0
      shat=tau*stot
      sqrtshat=sqrt(shat)
      sqrtborn=sqrt(shat*z)
      xbjrk_born=xbjrk
      xbjrk_born(j_fks)=z*xbjrk(j_fks)

      x(2)=((1d0-y_ij_fks)/2d0-cctiny)/(1d0-cctiny)
      if(x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12)then
         xjac=-33d0
         return
      endif
      x(2)=sqrt(max(0d0,min(1d0,x(2))))
      if(.not.colltest)xjac=xjac*4d0*x(2)*(1d0-cctiny)
      call get_isr_radiation_bounds(j_fks,xbjrk_born,tau_born,
     &     xiimin,xiimax)
      xinorm=xiimax-xiimin
      if(xinorm.le.0d0)then
         xjac=-342d0
         return
      endif
      xinorm_ev=xinorm
      x(1)=((xi_i_fks-xiimin)/xinorm-sstiny)/(1d0-sstiny)
      if(x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12)then
         xjac=-102d0
         return
      endif
      x(1)=sqrt(max(0d0,min(1d0,x(1))))
      if(.not.softtest)xjac=xjac*2d0*x(1)*(1d0-sstiny)
      x(3)=phi_i_fks/(2d0*pi)
      xjac=xjac*2d0*pi
c Undo the common laboratory boost, then the light-cone recoil. Work
c with light-cone components to retain the smaller beam at large y.
      exp_y=exp(ycm_born)
      do i=3,nexternal
         if(i.eq.i_fks)cycle
         iborn=i
         if(i.gt.i_fks)iborn=i-1
         pplus=xp(0,i)+xp(3,i)
         pminus=xp(0,i)-xp(3,i)
c Do not amplify cancellation in a nearly beam-collinear lab momentum
c when undoing a large Born rapidity. Its mass is supplied by the model.
         transverse_mass2=pmass(i)**2+sum(xp(1:2,i)**2)
         if(pplus.gt.pminus)then
            pminus=transverse_mass2/pplus
         elseif(pminus.gt.0d0)then
            pplus=transverse_mass2/pminus
         endif
         pplus=pplus/exp_y
         pminus=pminus*exp_y
         pred(0)=(pplus+pminus)/2d0
         pred(1:2)=xp(1:2,i)
         pred(3)=(pplus-pminus)/2d0
         call boost_isr_recoil(pred,p_born(0,iborn),xi_i_fks,
     &        y_ij_fks,phi_i_fks,idir,.true.,pmass(i)**2)
      enddo
      p_born(0,1:2)=sqrtborn/2d0
      p_born(1:2,1:2)=0d0
      p_born(3,1)=p_born(0,1)
      p_born(3,2)=-p_born(0,2)
      xpswgt=abs(xpswgt*shat/(4d0*pi)**3/z)
      xij_aor=-exp(2*idir*ximag*phi_i_fks)
      end subroutine generate_momenta_initial_inverse_asymmetric


      subroutine generate_momenta_initial_inverse_symmetric(xp,xi_i_fks ,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      use fks_phase_space_data,only: xinorm_ev,tau_lower_bound,xij_aor
      use mc_native_context, only: native_mapping
      implicit none
      double precision pi,stiny,qtiny,zero,ctiny
      parameter (pi=3.1415926535897932d0,stiny=1d-6,qtiny=1d-7,zero=0d0
     $     ,ctiny=5d-7)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      logical fks_as_is
      parameter (fks_as_is=.false.)
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,stot,tau,ycm,xbjrk(2),tau_born,ycm_born,xbjrk_born(2)
     $     ,x1,x2,y_lab_to_cms,xp_red(0:3,nexternal)
      integer i_fks,j_fks
      double precision yijdir,costh_i_fks,omega,ltau_born ,e2ycm_born
     $     ,em2ycm_born,yij_upp,yij_low ,y_ij_fks_upp ,y_ij_fks_low
     $     ,xiimax,xiimin,xinorm,bstfact ,shy_bst ,chy_bst,chy_bstmo
     $     ,cosphi_i_fks,sinphi_i_fks ,xdir_t(1:3),ybst,sstiny,cctiny
      integer idir,i
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
      logical softtest,colltest
      common/sctests/softtest,colltest
      if(j_fks.ne.1.and.j_fks.ne.2)then
         write(*,*) 'Invalid symmetric ISR emitter',j_fks
         stop 1
      endif
      if(xi_i_fks.lt.0d0.or.xi_i_fks.ge.1d0)then
         xjac=-102d0
         return
      endif
      sstiny=stiny
      cctiny=ctiny
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0d0
         cctiny=0d0
      endif
      idir=0
      if(.not.fks_as_is)then
         if(j_fks.eq.1)then
            idir=1
         elseif(j_fks.eq.2)then
            idir=-1
         endif
      else
         idir=1
         write(*,*)'One_tree: option not checked (inverse)'
         stop
      endif
      yijdir=idir*y_ij_fks
      costh_i_fks=yijdir
      omega=sqrt( (2-xi_i_fks*(1+yijdir))/
     &            (2-xi_i_fks*(1-yijdir)) )
      tau_born=tau*(1-xi_i_fks)
      ycm_born=ycm+log(omega)
      shat=tau*stot
      sqrtshat=sqrt(shat)
      xbjrk_born(1)=xbjrk(1)*(sqrt(1-xi_i_fks)*omega)
      xbjrk_born(2)=xbjrk(2)/omega*sqrt(1-xi_i_fks)

! this is to overcome numerical instabilities in ee collisions
      if (1d0-tau_born.gt.stiny) then
        ltau_born = log(tau_born)
      else
        ltau_born = tau_born-1d0
      endif
      if (abs(ycm_born).gt.stiny) then
        e2ycm_born = exp(2*ycm_born)
        em2ycm_born = exp(-2*ycm_born)
      else
        e2ycm_born = 1d0 + 2*ycm_born + 2*ycm_born**2
        em2ycm_born = 1d0 - 2*ycm_born + 2*ycm_born**2
      endif

      if( tau_born.le.tau_lower_bound) then
         if (ycm_born.gt. (0.5d0*ltau_born-log(tau_lower_bound)) )then
            yij_upp= (tau_lower_bound+tau_born)*
     &           ( 1-e2ycm_born*tau_lower_bound ) /
     &           ( (tau_lower_bound-tau_born)*
     &           (1+e2ycm_born*tau_lower_bound) )
         else
            yij_upp=1.d0
         endif
      else
         yij_upp=1.d0
      endif
      if( tau_born.le.tau_lower_bound) then
         if (ycm_born.lt. (-0.5d0*ltau_born+log(tau_lower_bound)) )then
            yij_low=-(tau_lower_bound+tau_born)*
     &           ( 1-em2ycm_born*tau_lower_bound ) /
     &           ( (tau_lower_bound-tau_born)*
     &           (1+em2ycm_born*tau_lower_bound) )
         else
            yij_low=-1.d0
         endif
      else
         yij_low=-1.d0
      endif
      if(idir.eq.1)then
         y_ij_fks_upp=yij_upp
         y_ij_fks_low=yij_low
      elseif(idir.eq.-1)then
         y_ij_fks_upp=-yij_low
         y_ij_fks_low=-yij_upp
      endif
      if (y_ij_fks_upp.le.y_ij_fks_low) then
         xjac=-33d0
         return
      endif
      x(2)=((y_ij_fks_upp-y_ij_fks)/
     $     (y_ij_fks_upp-y_ij_fks_low)-cctiny)/(1d0-cctiny)
      if (x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12) then
         xjac=-33d0
         return
      endif
      x(2)=sqrt(max(0d0,min(1d0,x(2))))

      if (colltest) then
         if ( y_ij_fks_fix.gt.y_ij_fks_upp .or.
     &        y_ij_fks_fix.lt.y_ij_fks_low) then
            xjac=-33d0
            return
         endif
      endif
      if (.not.colltest) xjac=xjac*
     $     (y_ij_fks_upp-y_ij_fks_low)*x(2)*2d0*(1d0-cctiny)

      call get_symmetric_isr_radiation_bounds(j_fks,xbjrk_born,
     $     tau_born,y_ij_fks,xiimin,xiimax)
      if (xiimax.le.xiimin) then
         write (*,*) 'WARNING #10 in genps_fks.f (inverse)'
     $        ,xiimax,xiimin
         xjac=-342d0
         return
      endif

      xinorm=xiimax-xiimin
      xinorm_ev=xinorm
      x(1)=((xi_i_fks-xiimin)/xinorm-sstiny)/(1d0-sstiny)
      if (x(1).lt.-1d-12.or.x(1).gt.1d0+1d-12) then
         xjac=-102d0
         return
      endif
      x(1)=sqrt(max(0d0,min(1d0,x(1))))
      if (softtest) then
         if(xi_i_fks/xiimax .gt. 1d0+stiny)then
            xjac=-102
            return
         endif
      endif
      if (.not.softtest) xjac=xjac*2d0*x(1)*(1d0-sstiny)

      x(3)=phi_i_fks/(2d0*pi)
      xjac=xjac*2d0*pi

c Boost the xp momenta from the lab frame to the reduced frame.
      ybst=log(omega)+ycm
      chy_bst=(exp(ybst)+exp(-ybst))/2d0
      shy_bst=(exp(ybst)-exp(-ybst))/2d0
      chy_bstmo=chy_bst-1d0
      do i=1,nexternal
         call boostwdir2(chy_bst,shy_bst,chy_bstmo,[0d0,0d0,1d0],
     &        xp(0,i),xp_red(0,i))
      enddo

c     Use xp in the reduced frame (a.k.a. tilde frame) to get the Born momenta.
      bstfact=sqrt( (2-xi_i_fks*(1-yijdir))*(2-xi_i_fks*(1+yijdir)) )
      shy_bst=-xi_i_fks*sqrt(1-yijdir**2)/(2*sqrt(1-xi_i_fks))
      chy_bst=bstfact/(2*sqrt(1-xi_i_fks))
      chy_bstmo=chy_bst-1.d0
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xdir_t(1)=cosphi_i_fks
      xdir_t(2)=sinphi_i_fks
      xdir_t(3)=zero
      do i=3,nexternal
         if (i.lt.i_fks) then
            call boostwdir2(chy_bst,shy_bst,chy_bstmo,xdir_t,
     $           xp_red(0,i),p_born(0,i))
         elseif (i.gt.i_fks) then
            call boostwdir2(chy_bst,shy_bst,chy_bstmo,xdir_t,
     $           xp_red(0,i),p_born(0,i-1))
         endif
      enddo

      p_born(1:2,1:2)=0d0
      p_born(0,1)=sum(p_born(0,3:nexternal-1))/2d0
      p_born(3,1)=p_born(0,1)
      p_born(0,2)=p_born(0,1)
      p_born(3,2)=-p_born(0,1)

      xpswgt=xpswgt*shat
      xpswgt=xpswgt/(4*pi)**3/(1-xi_i_fks)
      xpswgt=abs(xpswgt)

      xij_aor=-exp( 2*idir*ximag*phi_i_fks )
      end subroutine generate_momenta_initial_inverse_symmetric


      subroutine generate_momenta_initial_noevpr(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt ,xjac ,srec, pass)
      use fks_phase_space_data,only: tau_Born_lower_bound,tau_lower_bound_resonance,tau_lower_bound,
     $     veckn_ev,veckbarn_ev,xp0jfks,xij_aor
      use mc_native_context, only: native_mapping
      implicit none
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3),srec
      logical pass
c common blocks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j,idir
      double precision yijdir,costh_i_fks,x1bar2,x2bar2,yij_sol,xi1,xi2
     $     ,ximaxtmp,omega,bstfact,shy_tbst,chy_tbst,chy_tbstmo
     $     ,xdir_t(3),cosphi_i_fks,sinphi_i_fks,shy_lbst,chy_lbst
     $     ,encmso2,E_i_fks,sinth_i_fks,xpifksred(0:3),xi_i_fks
     $     ,xiimin,yij_upp,yij_low,y_ij_fks_upp,y_ij_fks_low
      double complex resAoR0
c external
c
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      logical fks_as_is
      parameter (fks_as_is=.false.)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision stiny,sstiny,qtiny,zero,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (zero=0d0)
      parameter (ctiny=5d-7)
c
      pass=.true.
      if(softtest.or.colltest.or.native_mapping)then
         sstiny=0.d0
         cctiny=0.d0
      else
         sstiny=stiny
         cctiny=ctiny
      endif

c
c FKS for left or right incoming parton
c
      idir=0
      if(.not.fks_as_is)then
         if(j_fks.eq.1)then
            idir=1
         elseif(j_fks.eq.2)then
            idir=-1
         endif
      else
         idir=1
         write(*,*)'One_tree: option not checked'
         stop
      endif

c
c set-up y_ij_fks
c
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or.
     &             (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks=y_ij_fks_fix
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
      else
         write(*,*)'Error #3 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(2)*2d0
c
c Compute costh_i_fks
c
      yijdir=idir*y_ij_fks
      costh_i_fks=yijdir
c
c Compute maximum allowed xi_i_fks
C      xiimax=1-xmrec2/shat
C MZ checked for single-top @NLO QCD
      xiimax=1-tau_born_lower_bound/xbjrk_born(1)/xbjrk_born(2)
      xinorm=xiimax
c
c Define xi_i_fks
c
      if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &     ((.not.colltest) .or.
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &     (.not.softtest)  )then
         if(icountevts.eq.-100)then
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->0
            xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         endif
c in the case of counter events, xi_i_hat is an input to this function
         xi_i_fks=xi_i_hat*xiimax
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
c This is to keep xi_i_hat, rather than xi_i, fixed in the tests.
c Changed in the context of granny stuff
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix*xiimax
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        softtest )then
         if(xi_i_fks_fix.lt.1d0)then
            xi_i_fks=xi_i_fks_fix*xiimax
         else
            xjac=-102
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=xi_i_fks_matrix(icountevts)
      else
         write(*,*)'Error #4 in genps_fks.f',icountevts
         stop
      endif
c remove the following if no importance sampling towards soft
c singularity is performed when integrating over xi_i_hat
      xjac=xjac*2d0*x(1)

c
c Update the variables here.
c
      tau=tau_born
      ycm=ycm_born
      shat=shat_born
      sqrtshat=sqrt(shat)
      xbjrk(1)=xbjrk_born(1)
      xbjrk(2)=xbjrk_born(2)

C build the momentum of i_fks in the partonic com frame

      encmso2=sqrtshat/2.d0
      p_i_fks(0)=encmso2
      E_i_fks=xi_i_fks*encmso2
      xp(0,i_fks)=E_i_fks
      sinth_i_fks=dsqrt(1-costh_i_fks**2)
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=yijdir
      do j=1,3
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         p_i_fks(j)=encmso2*xpifksred(j)
      enddo

C Now we need to generate the momenta for the born
C system, taking into account the radiation of i_fks
      srec = shat * (1-xi_i_fks)
      !write(*,*)'XI', xi_i_fks

c
c
c Collinear limit of <ij>/[ij]. See innerpin.m.
      if( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) )then
         resAoR0=-exp( 2*idir*ximag*phi_i_fks )
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii) * (tau,ycm)
      !write(*,*) 'SHAT END', xpswgt,shat
      xpswgt=xpswgt*shat
      !write(*,*) 'XPSWGT', xpswgt, xi_i_fks
      !xpswgt=xpswgt*srec
      xpswgt=xpswgt/(4*pi)**3!!/(1-xi_i_fks) MZ no need to include this
      !factor as it was related to the old (event-projection) mapping of x1x2
      xpswgt=abs(xpswgt)

      ! this is what happens in _massless_final
C      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
C     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
c
      return
      end subroutine generate_momenta_initial_noevpr

      subroutine get_symmetric_isr_radiation_bounds(j_fks,
     $     xbjrk_born,tau_born,y,xiimin,xiimax)
c Solve x_real <= 1 for each beam in the symmetric map. This is the
c old angle-dependent endpoint, written without subtracting nearly
c equal roots or selecting a beam using 1-tau_born in a denominator.
c For beam one, (1-xi)*(2-xi*(1+c))=xbar**2*(2-xi*(1-c)).
      use fks_phase_space_data, only: tau_lower_bound
      implicit none
      integer j_fks,beam
      double precision xbjrk_born(2),tau_born,y,xiimin,xiimax
      double precision omx_ee(2),c,omx,a,d,b,discriminant,h
      common /to_ee_omx1/omx_ee
      xiimax=1d0
      c=(3-2*j_fks)*y
      do beam=1,2
         omx=1d0-xbjrk_born(beam)
         if(omx.lt.5d-7.and.omx_ee(beam).gt.0d0)
     $        omx=omx_ee(beam)
         d=omx*(2d0-omx)
         a=xbjrk_born(beam)**2
         if(c.eq.-1d0)then
            h=1d0
         else
            b=2d0*(1d0+c)+d*(1d0-c)
            discriminant=4d0*a*(1d0+c)**2+d**2*(1d0-c)**2
            h=4d0*d/(b+sqrt(discriminant))
         endif
         xiimax=min(xiimax,h)
         c=-c
      enddo
      xiimin=0d0
      if(tau_born.lt.tau_lower_bound)
     $     xiimin=1d0-tau_born/tau_lower_bound
      end subroutine get_symmetric_isr_radiation_bounds


      subroutine get_isr_radiation_bounds(j_fks,xbjrk_born,tau_born,
     &     xiimin,xiimax)
      use fks_phase_space_data, only: tau_lower_bound
      implicit none
      integer j_fks
      double precision xbjrk_born(2),tau_born,xiimin,xiimax
      double precision omx_ee(2)
      common/to_ee_omx1/omx_ee
c Retain separately computed 1-x for dressed leptons close to x=1.
      xiimax=1d0-xbjrk_born(j_fks)
      if(xiimax.lt.5d-7.and.omx_ee(j_fks).gt.0d0)
     &     xiimax=omx_ee(j_fks)
      xiimin=0d0
      if(tau_born.lt.tau_lower_bound)
     &     xiimin=1d0-tau_born/tau_lower_bound
      end subroutine get_isr_radiation_bounds

      end module fks_radiation_maps
