c Initial-state radiation maps with and without event projection.
c External routine interfaces and COMMON layouts are shared with
c genps_fks.f; keep coordinate and counterevent conventions consistent.

      subroutine generate_momenta_initial(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt ,xjac ,pass)
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3)
      logical pass
c common blocks
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
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

      double precision omx1_ee, omx2_ee
      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee
      double precision omx1bar2, omx2bar2
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
      omx1_ee = omx_ee(1)
      omx2_ee = omx_ee(2)
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
      ! these are to prevent numerical inaccuracies
      ! when x->1 (relevant for ee collisions)
      if (1d0-xbjrk_born(1).lt.ctiny.and.omx1_ee.gt.0d0) then
        x1bar2 = 1d0 - 2d0*omx1_ee + omx1_ee**2
        omx1bar2 = 2d0*omx1_ee - omx1_ee**2
      else
        x1bar2 = xbjrk_born(1)**2
        omx1bar2 = 1d0-x1bar2
      endif

      if (1d0-xbjrk_born(2).lt.ctiny.and.omx2_ee.gt.0d0) then
        x2bar2 = 1d0 - 2d0*omx2_ee + omx2_ee**2
        omx2bar2 = 2d0*omx2_ee - omx2_ee**2
      else
        x2bar2 = xbjrk_born(2)**2
        omx2bar2 = 1d0-x2bar2
      endif

      if(1-tau_born.gt.1.d-5.and.omx1_ee.eq.0d0.and.omx2_ee.eq.0d0)then
         yij_sol=-sinh(ycm_born)*(1+tau_born)/
     &            ( cosh(ycm_born)*(1-tau_born) )
      else if (omx1_ee.ne.0d0.and.omx2_ee.ne.0d0) then
         yij_sol = (omx1_ee - omx2_ee) * ( 1 + xbjrk_born(1)*xbjrk_born(2)) /
     $ (xbjrk_born(1)+xbjrk_born(2)) / (omx1_ee+omx2_ee-omx1_ee*omx2_ee)
      else
         yij_sol=-ycmhat
      endif
      if(abs(yij_sol).gt.1.d0)then
         if (abs(yij_sol).lt.1d0+qtiny) then
           yij_sol = sign(1d0, yij_sol)
         else
           write(*,*)'Error #9 in genps_fks.f',yij_sol,icountevts
           write(*,*)xbjrk_born(1),xbjrk_born(2),yijdir
         endif
      endif

      if(yijdir.eq.yij_sol)then
         ! this may be relevant only for ee collisions
         if (omx1_ee.ne.0d0.and.omx2_ee.ne.0d0) then
           ximaxtmp=omx1_ee+omx2_ee-omx1_ee*omx2_ee
         else
           ximaxtmp=1-xbjrk_born(1)*xbjrk_born(2)
         endif
      elseif(yijdir.ge.yij_sol)then
         !this is an expansion when both yij->-1 and x1->1
         ! in this case there may be precision loosses
         ! from the argument in the sqrt
         if (abs(yijdir+1d0).lt.ctiny.and.omx1bar2.lt.ctiny) then
           xi1=(4*x1bar2 + yijdir + 11*x1bar2*yijdir - 5*x1bar2**2*yijdir +
     &          x1bar2**3*yijdir+4*yijdir**2)/(2*(1 + yijdir)**2)
           ximaxtmp=1-xi1
         else if (omx1bar2.lt.ctiny) then
           ! compute directly ximaxtmp
           ximaxtmp = omx1bar2 / (1+yijdir)
         else
           xi1=2*(1+yijdir)*x1bar2/(
     &        sqrt( ((1+x1bar2)*(1-yijdir))**2+16*yijdir*x1bar2 ) +
     &        (1-yijdir)*(omx1bar2) )
           ximaxtmp=1-xi1
         endif
      elseif(yijdir.lt.yij_sol)then
         !this is an expansion when both yij->+1 and x1->1
         ! in this case there may be precision loosses
         ! from the argument in the sqrt
         if (abs(yijdir-1d0).lt.ctiny.and.omx2bar2.lt.ctiny) then
           xi2=(4*x2bar2 - yijdir - 11*x2bar2*yijdir + 5*x2bar2**2*yijdir -
     &          x2bar2**3*yijdir +4*yijdir**2)/(4*(-1 + yijdir)**2)
           ximaxtmp=1-xi2
         else if (omx2bar2.lt.ctiny) then
           ! compute directly ximaxtmp
           ximaxtmp = omx2bar2 / (1-yijdir)
         else
           xi2=2*(1-yijdir)*x2bar2/(
     &        sqrt( ((1+x2bar2)*(1+yijdir))**2-16*yijdir*x2bar2 ) +
     &        (1+yijdir)*(omx2bar2) )
           ximaxtmp=1-xi2
         endif
      else
         write(*,*)'Fatal error #14 in one_tree: unknown option'
         write(*,*)y_ij_fks,yij_sol,idir
         stop
      endif
      xiimax=ximaxtmp
c
c Lower bound on xi_i_fks
c
      if (tau_born.lt.tau_lower_bound) then
         xiimin=1d0-tau_born/tau_lower_bound
      else
         xiimin=0d0
      endif
      if (xiimax.lt.xiimin) then
         write (*,*) 'WARNING #10 in genps_fks.f',icountevts,xiimax
     $        ,xiimin,tau_born,tau_lower_bound
         xjac=-342d0
         pass=.false.
         return
      endif

      xinorm=xiimax-xiimin
      if( icountevts.ge.1 .and.
     &     ( (idir.eq.1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(1))).gt.1.d-5) .or.
     &     (idir.eq.-1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(2))).gt.1.d-5) ) )then
         write(*,*)'Fatal error #15 in one_tree'
         write(*,*)ximaxtmp,xbjrk_born(1),xbjrk_born(2),idir
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
      if(xi_i_fks.gt.xiimax)then
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
      end


      subroutine generate_momenta_initial_inverse(xp,xi_i_fks ,y_ij_fks
     $     ,phi_i_fks,p_born,x,xjac,xpswgt,shat,sqrtshat,i_fks,j_fks,stot
     $     ,tau,ycm,xbjrk,tau_born,ycm_born,xbjrk_born,y_lab_to_cms)
      use mc_native_context, only: native_mapping
      implicit none
      double precision pi,stiny,qtiny,zero,ctiny
      parameter (pi=3.1415926535897932d0,stiny=1d-6,qtiny=1d-7,zero=0d0
     $     ,ctiny=5d-7)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      logical fks_as_is
      parameter (fks_as_is=.false.)
      include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac,x(3),xp(0:3,nexternal),xi_i_fks,y_ij_fks
     $     ,phi_i_fks,p_born(0:3,-max_branch:nexternal-1),shat,sqrtshat
     $     ,xpswgt,stot,tau,ycm,xbjrk(2),tau_born,ycm_born,xbjrk_born(2)
     $     ,x1,x2,y_lab_to_cms,xp_red(0:3,nexternal)
      integer i_fks,j_fks
      double precision yijdir,costh_i_fks,omega,ltau_born ,e2ycm_born
     $     ,em2ycm_born,yij_upp,yij_low ,y_ij_fks_upp ,y_ij_fks_low
     $     ,x1bar2,omx1bar2,x2bar2,omx2bar2 ,yij_sol ,ximaxtmp,xi1,xi2
     $     ,xiimax,xiimin,xinorm,bstfact ,shy_bst ,chy_bst,chy_bstmo
     $     ,cosphi_i_fks,sinphi_i_fks ,xdir_t(1:3),ybst,sstiny,cctiny
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      integer idir,i
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
      logical softtest,colltest
      common/sctests/softtest,colltest
      double complex xij_aor
      common/cxij_aor/xij_aor
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

      x1bar2 = xbjrk_born(1)**2
      omx1bar2 = 1d0-x1bar2
      x2bar2 = xbjrk_born(2)**2
      omx2bar2 = 1d0-x2bar2
      if(1-tau_born.gt.1.d-8)then ! prevent numerical inaccuracies.
         yij_sol=-sinh(ycm_born)*(1+tau_born)/
     &        ( cosh(ycm_born)*(1-tau_born) )
      else
         yij_sol=yijdir
         return
      endif
      if(abs(yij_sol).gt.1.d0)then
         if (abs(yij_sol).lt.1d0+qtiny) then
           yij_sol = sign(1d0, yij_sol)
         else
            write(*,*)'Error #9 in genps_fks.f (inverse)',yij_sol
           write(*,*)xbjrk_born(1),xbjrk_born(2),yijdir
         endif
      endif

      if(yijdir.eq.yij_sol)then
         ximaxtmp=1-xbjrk_born(1)*xbjrk_born(2)
      elseif(yijdir.ge.yij_sol)then
         !this is an expansion when both yij->-1 and x1->1
         ! in this case there may be precision loosses
         ! from the argument in the sqrt
         if (abs(yijdir+1d0).lt.ctiny.and.omx1bar2.lt.ctiny) then
           xi1=(4*x1bar2 + yijdir + 11*x1bar2*yijdir - 5*x1bar2**2*yijdir +
     &          x1bar2**3*yijdir+4*yijdir**2)/(2*(1 + yijdir)**2)
           ximaxtmp=1-xi1
         else if (omx1bar2.lt.ctiny) then
           ! compute directly ximaxtmp
           ximaxtmp = omx1bar2 / (1+yijdir)
         else
           xi1=2*(1+yijdir)*x1bar2/(
     &        sqrt( ((1+x1bar2)*(1-yijdir))**2+16*yijdir*x1bar2 ) +
     &        (1-yijdir)*(omx1bar2) )
           ximaxtmp=1-xi1
         endif
      elseif(yijdir.lt.yij_sol)then
         !this is an expansion when both yij->+1 and x1->1
         ! in this case there may be precision loosses
         ! from the argument in the sqrt
         if (abs(yijdir-1d0).lt.ctiny.and.omx2bar2.lt.ctiny) then
           xi2=(4*x2bar2 - yijdir - 11*x2bar2*yijdir + 5*x2bar2**2*yijdir -
     &          x2bar2**3*yijdir +4*yijdir**2)/(4*(-1 + yijdir)**2)
           ximaxtmp=1-xi2
         else if (omx2bar2.lt.ctiny) then
           ! compute directly ximaxtmp
           ximaxtmp = omx2bar2 / (1-yijdir)
         else
           xi2=2*(1-yijdir)*x2bar2/(
     &        sqrt( ((1+x2bar2)*(1+yijdir))**2-16*yijdir*x2bar2 ) +
     &        (1+yijdir)*(omx2bar2) )
           ximaxtmp=1-xi2
         endif
      else
         write(*,*)'Fatal error #14 in one_tree:'/
     $        /' unknown option (inverse)'
         write(*,*)y_ij_fks,yij_sol,idir
         stop
      endif
      xiimax=ximaxtmp

c
c Lower bound on xi_i_fks
      if (tau_born.lt.tau_lower_bound) then
         xiimin=1d0-tau_born/tau_lower_bound
      else
         xiimin=0d0
      endif
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
      end


      subroutine generate_momenta_initial_noevpr(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt ,xjac ,srec, pass)
      use mc_native_context, only: native_mapping
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3),srec
      logical pass
c common blocks
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
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
      end
