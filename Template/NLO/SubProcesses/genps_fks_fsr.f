c Final-state radiation maps and their inverse coordinates.
c External routine interfaces and COMMON layouts are shared with
c genps_fks.f; keep coordinate and counterevent conventions consistent.

      subroutine generate_momenta_massless_final(icountevts,i_fks,j_fks
     &     ,p_born_imother,shat,sqrtshat,x,xmrec2,xp,phi_i_fks,xiimax
     &     ,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt
     &     ,pass)
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
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
     &     ,sinphi_i_fks
      double complex resAoR0
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
         if (.not. colltest) then
            xjac=xjac*2d0*x(2)*2d0
         else
            continue ! do not include jacobian for y in tests
         endif
      elseif (colltest) then
         y_ij_fks = y_ij_fks_fix
      else
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
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
      x3len_j_fks=(shat-xmrec2-2*sqrtshat*x3len_i_fks)/
     &             (2*(sqrtshat-x3len_i_fks*(1-y_ij_fks)))
c Resolve the daughter parallel and transverse to the emitted momentum.
c This avoids subtracting squared momenta when the recoil is nearly at rest.
c Use this also in the outer map: if j is soft, computing sin(theta) from
c a rounded cos(theta)=1 makes the daughters spuriously collinear/off shell
c and the native inverse map cannot recover a positive Jacobian.
      costh_i_fks=x3len_i_fks+x3len_j_fks*y_ij_fks
      sinth_i_fks=x3len_j_fks*sqrt(max(0d0,
     $     (1d0-y_ij_fks)*(1d0+y_ij_fks)))
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
      end


      subroutine generate_momenta_massless_final_inverse(xp,xi_i_fks
     $     ,y_ij_fks,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks
     $     ,j_fks)
      ! TODO: probably need xiimax as argument, to update the prefactors.
      use mc_native_context, only: native_mapping
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,th_mother_fks,costh_mother_fks,sinth_mother_fks
     $     ,phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks
      integer i_fks,j_fks
      double precision recoil(0:3),sumrec,sumrec2,betabst
     $     ,gammabst,shybst,chybst,chybstmo,xdir(1:3),veckn,veckbarn
     $     ,xiimax,xmrec2
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      logical        softtest,colltest
      common/sctests/softtest,colltest
      double complex xij_aor
      common/cxij_aor/xij_aor
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
      x(2)=((1d0-y_ij_fks)/2d0-cctiny)/(1d0-cctiny)
      if (x(2).lt.-1d-12.or.x(2).gt.1d0+1d-12) then
         xjac=-33d0
         return
      endif
      x(2)=sqrt(max(0d0,min(1d0,x(2))))
      xjac=xjac*2d0*x(2)*2d0

!     random number associated with phi_i_fks
      x(3)=phi_i_fks/(2d0*pi)
      xjac=xjac*2d0*pi

c Collinear limit of <ij>/[ij]. See innerpin.m.
      call getangles(p_born(0:3,j_fks),
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
      xij_aor=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
      end


      subroutine generate_momenta_massive_final(icountevts,isolsign
     &     ,i_fks,j_fks,p_born_imother,shat
     &     ,sqrtshat,m_j_fks,x,xmrec2,xp,phi_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,pass)
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks,isolsign
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),m_j_fks,phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
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
      if(xmrec2.lt.1.d-16*shat)then
         expybst=sqrtshat*sumrec/(shat-xmj2)*
     &           (1+xmj2*xmrec2/(shat-xmj2)**2)
      else
         expybst=sumrec/(2*sqrtshat*xmrec2)*
     &           (shat+xmrec2-xmj2-shat*sqrt(cffC2))
      endif
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
      end


      subroutine generate_momenta_massive_final_inverse(xp,xi_i_fks
     $           ,y_ij_fks,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat
     $           ,i_fks,j_fks,m_j_fks)
      use mc_native_context, only: native_mapping
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,m_j_fks
      integer i_fks,j_fks
      double precision recoil(0:3),sumrec,sumrec2,xmj
     $     ,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2,cffDEL2,xiBm,ximax
     $     ,xiimax,xinorm,rat_xi,expybst
     $     ,shybst,chybst,chybstmo,veckn,veckbarn,xdir(3),xmrec2
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      integer i
      double precision native_uborn,native_u,native_denom,
     $     native_eborn,native_delta,native_onepy,native_ratio,
     $     native_recoil,native_inverse_denom
      double precision rho,dot,sstiny,cctiny,branch_sign,
     $     native_fsr_angle
      external rho,dot,native_fsr_angle
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

      if(xmrec2.lt.1.d-16*shat)then
         expybst=sqrtshat*sumrec/(shat-xmj2)*
     &           (1+xmj2*xmrec2/(shat-xmj2)**2)
      else
         expybst=sumrec/(2*sqrtshat*xmrec2)*
     &           (shat+xmrec2-xmj2-shat*sqrt(cffC2))
      endif
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
      end


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
      end
