c Incoming momentum fractions and rapidities for each beam scheme.
c External routine interfaces and COMMON layouts are shared with
c genps_fks.f; keep coordinate and counterevent conventions consistent.

      subroutine generate_tau_y_wrapper(
     $     qmass,qwidth,totmass,stot,rndx,tau_born,ycm_born,ycmhat,xjac)
      ! generates tau and y, calling the functions that correpsond to the
      ! case at hand
      implicit none

      include 'nexternal.inc'
      include 'resonance_recoil.inc'
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision totmass, stot
      double precision rndx(2)
      double precision tau_born, ycm_born, ycmhat, xjac
C
      include 'run.inc'
      integer ndim_dummy
      double precision fksmass
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      logical softtest,colltest
      common/sctests/softtest,colltest

      include 'genps.inc'
      integer itree_c(2,-max_branch:-1)
      integer ns_channel, nt_channel, ionebody, nbranch
      logical one_body
      common/born_trees/itree_c,ns_channel,nt_channel,ionebody,nbranch,one_body

      ndim_dummy=-1 ! this is actually not used anymore

      if (abs(lpp(1)).ne.abs(lpp(2))) then
          write(*,*) 'Different beams not implemented', lpp
          stop 1
      endif
      if (abs(lpp(1)).ge.1 .and. abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         if (abs(lpp(1)).ne.4.and.abs(lpp(1)).ne.3) then ! this is for pp collisions
c x(ndim-1) -> tau_cnt(0); x(ndim) -> ycm_cnt(0)
C rndx(1) -> tau; rndx(2) -> ycm
           if (one_body) then
c tau is fixed by the mass of the final state particle
              call compute_tau_one_body(totmass,stot,tau_born,xjac)
           else
               if(nt_channel.eq.0 .and. qwidth(-ns_channel-1).ne.0.d0 .and.
     $           cBW(-ns_channel-1).ne.2)then
c Generate tau according to a Breit-Wiger function
                 call generate_tau_BW(stot,ndim_dummy,rndx(1),qmass(
     $              -ns_channel-1),qwidth(-ns_channel-1),cBW(-ns_channel
     $              -1),cBW_mass(-1, -ns_channel-1),cBW_width(-1,
     $              -ns_channel-1),tau_born,xjac)
               else
c     not a Breit Wigner
                 call generate_tau(stot,ndim_dummy,rndx(1),tau_born,xjac)
               endif
           endif

c Generate the rapditity of the Born system
           call generate_y(tau_born,rndx(2),ycm_born,ycmhat,xjac)

        else                    ! this is for dressed ee collisions
           call generate_ee_tau_y(rndx(1), rndx(2), one_body, totmass,
     $        stot, nt_channel, qmass(-ns_channel-1),qwidth(-ns_channel-1),
     $        cBW(-ns_channel-1),cBW_mass(-1, -ns_channel-1),
     $        cBW_width(-1,-ns_channel-1),
     $        tau_born, ycm_born, ycmhat, xjac)
           ! for non-physical configurations, xjac=-1000
           if (xjac.eq.-1000d0) return
         endif
      elseif (abs(lpp(1)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x1 not implemented in one_tree'
         stop
      elseif (abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x2 not implemented in one_tree'
         stop
      else
c No PDFs (also use fixed energy when performing tests)
         call compute_tau_y_epem(j_fks,one_body,totmass,stot,
     &        tau_born,ycm_born,ycmhat)
c FI limit tests need nonzero beam momentum available for recoil. Keep
c their Born energy fixed, with xbar=1/2 instead of the usual xbar=1.
c A one-body Born fixes tau from its physical mass.
         if((softtest.or.colltest).and.initial_recoil_leg.gt.0)then
            if(one_body)then
               tau_born=totmass**2/stot
            else
               tau_born=0.25d0
            endif
         endif
         if (j_fks.le.nincoming .and. .not.(softtest.or.colltest)) then
            write (*,*) 'Process has incoming j_fks, but fixed shat: '/
     &           /'not allowed for processes generated at NLO.'
            stop 1
         endif
      endif

      return
      end


      subroutine compute_tau_one_body(totmass,stot,tau,jac)
      implicit none
      double precision totmass,stot,tau,jac,roH
      roH=totmass**2/stot
      tau=roH
c Jacobian due to delta() of tau_born
      jac=jac*2*totmass/stot
      return
      end


      subroutine generate_tau_BW(stot,idim,x,mass,width,cBW,BWmass
     $     ,BWwidth,tau,jac)
      implicit none
      integer cBW,idim
      double precision stot,x,tau,jac,mass,width,BWmass(-1:1),BWwidth(
     $     -1:1),s_mass,s
      double precision smax,smin
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      if (cBW.eq.1 .and. width.gt.0d0 .and. BWwidth(1).gt.0d0) then
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(5,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      else
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(3,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      endif
      return
      end


      subroutine generate_tau(stot,idim,x,tau,jac)
      use mc_native_context, only: native_mapping
      implicit none
      integer idim
      double precision x,tau,jac,smin,smax,s_mass,s,tiny,dum,dum3(-1:1)
     $     ,stot
      parameter (tiny=1d-8)
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound
! A flat auxiliary map is finite also when the physical threshold is zero.
      if(native_mapping)then
         tau=x
         return
      endif
      smin=tau_born_lower_bound*stot
      smax=stot
      s_mass=tau_lower_bound_resonance*stot
      if (s_mass.gt.smin*(1d0+tiny)) then
         call trans_x(2,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      elseif(abs(s_mass-smin).lt.tiny*smin) then
         call trans_x(7,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      else
         write (*,*) 'ERROR #39 in genps_fks.f',s_mass,smin,smax
         jac=-1d0
      endif
      tau=s/stot
      jac=jac/stot
      return
      end


      subroutine generate_y(tau,x,ycm,ycmhat,jac)
      implicit none
      double precision tau,x,ycm,jac
      double precision ylim,ycmhat
      ylim=-0.5d0*log(tau)
      ycmhat=2*x-1
      ycm=ylim*ycmhat
      jac=jac*ylim*2
      return
      end


      subroutine compute_tau_y_epem(j_fks,one_body,fksmass,
     &                              stot,tau,ycm,ycmhat)
      implicit none
      include 'nexternal.inc'
      integer j_fks
      logical one_body
      double precision fksmass,stot,tau,ycm,ycmhat
      if(j_fks.le.nincoming)then
c This should never happen in normal integration: when no PDFs, j_fks
c cannot be initial state (but needed for testing). If tau set to one,
c integration range in xi_i_fks will be zero, so lower it artificially
c when too large
         if(one_body)then
            tau=fksmass**2/stot
         else
            tau=max((0.85d0)**2,fksmass**2/stot)
         endif
         ycm=0.d0
      else
c For e+e- collisions, set tau to one and y to zero
         tau=1.d0
         ycm=0.d0
      endif
      ycmhat=0.d0
      return
      end


      subroutine get_tau_y_from_x12(x1, x2, omx1, omx2, tau, ycm, ycmhat, jac)
      implicit none
      double precision x1, x2, omx1, omx2, tau, ycm, ycmhat, jac
      double precision ylim
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound
      double precision tolerance
      parameter (tolerance=1e-3)
      double precision y_settozero
      parameter (y_settozero=1e-12)
      double precision lx1, lx2

      tau = x1*x2

      ! ycm=-log(tau)/2 ;  ylim = log(x1/x2)/2
      if (1d0-x1.gt.tolerance) then
        lx1 = dlog(x1)
      else
        lx1 = -omx1-omx1**2/2d0-omx1**3/3d0-omx1**4/4d0-omx1**5/5d0
      endif
      ylim = -0.5d0*lx1
      ycm = 0.5d0*lx1

      if (1d0-x2.gt.tolerance) then
        lx2 = dlog(x2)
      else
        lx2 = -omx2-omx2**2/2d0-omx2**3/3d0-omx2**4/4d0-omx2**5/5d0
      endif
      ylim = ylim-0.5d0*lx2
      ycm = ycm-0.5d0*lx2

      ycmhat = ycm / ylim

      ! this is to prevent numerical inaccuracies
      ! when botn x->1
      if (ylim.lt.y_settozero) then
        ylim = 0d0
        ycm = 0d0
        ycmhat = 1d0
      endif

      if (abs(ycmhat).gt.1d0) then
        if (abs(ycmhat).gt.1d0 + tolerance) then
          write(*,*) 'ERROR YCMHAT', ycmhat, x1, x2
          stop 1
        else
          ycmhat = sign(1d0, ycmhat)
        endif
      endif

      if (tau.lt.tau_born_lower_bound) then
        write(*,*) 'get_tau_y_from_x12: Warning, unphysical tau',
     $  tau, tau_born_lower_bound
        jac = -1000d0
      endif

      return
      end


      subroutine generate_x_ee(rnd, xmin, x, omx, jac)
      implicit none
      ! generates the momentum fraction with importance
      !  sampling suitable for ee collisions
      ! rnd is generated uniformly in [0,1],
      ! x is generated according to (1 -rnd)^-expo, starting
      ! from xmin
      ! jac is the corresponding jacobian
      ! omx is 1-x, stored to improve numerical accuracy
      double precision rnd, x, omx, jac, xmin
      double precision expo
      double precision get_ee_expo
      double precision tolerance
      parameter (tolerance=1.d-5)

      expo = get_ee_expo()

      x = 1d0 - rnd ** (1d0/(1d0-expo))
      omx = rnd ** (1d0/(1d0-expo))
      if (x.ge.1d0) then
        if (x.lt.1d0+tolerance) then
          x=1d0
        else
          write(*,*) 'ERROR in generate_x_ee', rnd, x
          stop 1
        endif
      endif
      jac = 1d0/(1d0-expo)
      ! then rescale it between xmin and 1
      x = x * (1d0 - xmin) + xmin
      omx = omx * (1d0 - xmin)
      jac = jac * (1d0 - xmin)**(1d0-expo)

      return
      end


      subroutine generate_ee_tau_y(rnd1_in, rnd2_in, one_body, totmass,
     $     stot, nt_channel, qmass, qwidth, cBW, cBW_mass, cBW_width,
     $     tau_born, ycm_born, ycmhat, xjac0)
      implicit none
      double precision rnd1_in, rnd2_in
      double precision rnd1, rnd2, totmass, stot, qmass, qwidth
      double precision cBW_mass(-1:1), cBW_width(-1:1)
      integer nt_channel, cBW
      logical one_body
      double precision tau_born, ycm_born, ycmhat, xjac0

      logical bw_exists, generate_with_bw
      common /to_ee_generatebw/ generate_with_bw
      double precision frac_bw
      parameter (frac_bw=0.5d0)
      integer idim_dum
C dressed lepton stuff
      double precision x1_ee, x2_ee, jac_ee

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee

      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound

      double precision get_ee_expo
      double precision tau_m, tau_w
      double precision omtau_born

      ! these common blocks are never used
      ! we leave them here for the moment
      ! as e.g. one may want to plot random numbers, etc.
      double precision r1, r2, x1bk, x2bk
      common /to_random_numbers/r1,r2, x1bk, x2bk
      logical use_evpr
      common /to_use_evpr/use_evpr

      ! copy the random numbers, as they may be rescaled
      ! (avoids side effects)
      rnd1=rnd1_in
      rnd2=rnd2_in

      ! these lines store the random numbers in the common
      ! block (may be removed)
      r1=rnd1
      r2=rnd2

      ! define the analogous of tau for mass and width
      tau_m = qmass**2/stot
      tau_w = qwidth**2/stot

      bw_exists = nt_channel.eq.0.and.qwidth.ne.0.d0.and.cBW.ne.2
     $ .and.tau_m.lt.1d0.and.totmass.lt.qmass

      generate_with_bw=.false.

      ! if there are BWs, decide whether to generate flat or
      ! to use the BW-specific generation (half and half, or
      ! as determined by frac_bw)
      if (bw_exists) then
        generate_with_bw = rnd1.lt.frac_bw
        if (generate_with_bw) then
            rnd1 = rnd1 / frac_bw
            xjac0 = xjac0 / frac_bw
        else
            rnd1 = (rnd1 - frac_bw) / (1d0 - frac_bw)
            xjac0 = xjac0 / (1d0 - frac_bw)
        endif
      endif

      if (one_body) then
        write(*,*) 'one body with ee collisions not implemented'
        stop 1
      endif

      ! if tau is generated accodring to a BW, then force the momentum
      ! mapping with event projection
      use_evpr = generate_with_bw

      if(generate_with_bw) then
        ! here we treat the case of resonances

        ! first generate tau with the dedicated function
        idim_dum = 1000 ! this is never used in practice
        call generate_tau_BW(stot,idim_dum,rnd1,qmass,qwidth,cBW,cBW_mass,
     $       cBW_width,tau_born,xjac0)
        ! multiply the jacobian by a multichannel factor
        xjac0 = xjac0 * (1d0/((tau_born-tau_m)**2 + tau_m*tau_w)) /
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (1d0-tau_born)**(1d0-2*get_ee_expo()))

        ! then pick either x1 or x2 and generate it the usual way;
        ! Note that:
        ! - setting xmin=sqrt(tau_born) ensures that the largest
        !    bjorken x is being generated.
        ! -  there is a jacobian for x1 x2 -> tau x1(2)
        ! -  we must include the factor 1/(1-x)^get_ee_expo,
        !    (x is the bjorken x which is not generated)
        !    because the compute_eepdf function assumes that
        !    this is the case in general
        if (rnd2.lt.0.5d0) then
          call generate_x_ee(rnd2*2d0, dsqrt(tau_born), x1_ee, omx_ee(1), jac_ee)
          x2_ee = tau_born / x1_ee
          omx_ee(2) = 1d0 - x2_ee
          xjac0 = xjac0 / x1_ee * 2d0 * jac_ee / (1d0-x2_ee)**get_ee_expo()
        else
          call generate_x_ee(1d0-2d0*(rnd2-0.5d0), dsqrt(tau_born), x2_ee, omx_ee(2), jac_ee)
          x1_ee = tau_born / x2_ee
          omx_ee(1) = 1d0 - x1_ee
          xjac0 = xjac0 / x2_ee * 2d0  * jac_ee / (1d0-x1_ee)**get_ee_expo()
        endif
      else
        ! standard (without resonances) generation:
        ! for dressed ee collisions the generation is different
        ! wrt the pp case. In the pp case, tau and y_cm are generated,
        ! while in the ee case x1 and x2 are generated first.

        call generate_x_ee(rnd1, tau_born_lower_bound,
     $      x1_ee, omx_ee(1), jac_ee)
        xjac0 = xjac0 * jac_ee
        call generate_x_ee(rnd2, tau_born_lower_bound/x1_ee,
     $      x2_ee, omx_ee(2), jac_ee)
        xjac0 = xjac0 * jac_ee

        tau_born = x1_ee * x2_ee
        ! better numerical accuracy
        omtau_born = omx_ee(1) + omx_ee(2) - omx_ee(1)*omx_ee(2)
        ! multiply the jacobian by a multichannel factor if the
        ! generation with resonances is also possible
        if (bw_exists) xjac0 = xjac0 * (omtau_born)**(1d0-2*get_ee_expo()) /
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (omtau_born)**(1d0-2*get_ee_expo()))
      endif

      ! Check here if the bjorken x's are physical (may not be so
      ! because of instabilities
      if (x1_ee.gt.1d0.or.x2_ee.gt.1d0) then
        write(*,*) 'generate_ee_tau_y: Warning, unphysical x:',
     $   x1_ee, x2_ee, generate_with_bw
        xjac0 = -1000d0
        return
      endif

      ! now we are done. We must call the following function
      ! in order to (re-)generate tau and ycm
      ! from x1 and x2. It also (re-)checks that tau_born
      ! is pysical, and otherwise sets xjac0=-1000
      call get_tau_y_from_x12(x1_ee, x2_ee, omx_ee(1), omx_ee(2), tau_born, ycm_born, ycmhat, xjac0)

      x1bk=x1_ee
      x2bk=x2_ee

      return
      end
