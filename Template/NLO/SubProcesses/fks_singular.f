      subroutine compute_born
c This subroutine computes the Born matrix elements and adds its value
c to the list of weights using the add_wgt subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp

      double precision wgt_c
      double precision wgt1
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision  xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision g22
      integer get_orders_tag
      call cpu_time(tBefore)
      if (f_b.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return
      call sborn(p_born,wgt_c)
      do iamp=1, amp_split_size
        if (amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=amp_split(iamp)*f_b/g**(qcd_power)
        call add_wgt(2,orders,wgt1,0d0,0d0)
      enddo

      call cpu_time(tAfter)
      tBorn=tBorn+(tAfter-tBefore)
      return
      end


      subroutine compute_6to5flav_cnt()
C This is the counterterm for the 6f->5f scheme change 
C of parton distributions (e.g. NNPDF2.3). 
C It is called in this function such that if is included
C in the LO cross section
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc' 
      include 'q_es.inc'
      include 'run.inc'
      include 'genps.inc'
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_6to5f(amp_split_size),
     &                 amp_split_6to5f_muf(amp_split_size),
     &                 amp_split_6to5f_mur(amp_split_size)
      common /to_amp_split_6to5f/ amp_split_6to5f, amp_split_6to5f_muf, 
     &                            amp_split_6to5f_mur
      integer orders_to_amp_split_pos
      integer niglu
      save niglu
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     $     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer i, j, k
      logical firsttime
      data firsttime /.true./
      double precision tf, pi
      parameter (tf=0.5d0)
      parameter (pi=3.1415926535897932385d0)
      integer alphasbpow
      double precision wgtborn, alphas
      ! switch on/off here
      logical include_6to5_cnt 
      data include_6to5_cnt /.false./ 

c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]

      ! set everything to 0
      amp_split_6to5f(1:amp_split_size) = 0d0
      amp_split_6to5f_muf(1:amp_split_size) = 0d0
      amp_split_6to5f_mur(1:amp_split_size) = 0d0

      ! skip if we don't want this piece or if the scale is
      ! below mt
      if (.not.include_6to5_cnt.or.scale.lt.mdl_mt) return

C the contribution is the following (if mu > mt):
C      Add a term -alphas n TF/3pi log (muR^2/mt^2) sigma(0) 
C      where n is the power of alphas for the Born xsec sigma(0)
C      Add a term −alphas TF/3pi log (mt^2/muF^2) sigma(0) for each
C      gluon in the initial state

      if (firsttime) then
          ! count the number of gluons
          do i = 1, nincoming
              if (idup(i, 1).eq.21) niglu = niglu + 1
          enddo
          write(*,*) 'compute_6to5flav_cnt found n initial gluons:', niglu
          firsttime=.false.
      endif

      ! compute the born
      call sborn(p_born,wgtborn)
      alphas = g**2/4d0/pi
      do iamp = 1, amp_split_size
        if (amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        alphasbpow = orders(qcd_pos)/2
        if (niglu.ne.0 .or. alphasbpow.ne.0) then
          ! this contribution will end up with one extra power
          ! of alpha_s. Check that we are including the corresponding
          ! order in the computation, otherwise skip the contribution
          orders(qcd_pos) = orders(qcd_pos) + 2
          if (orders(qcd_pos).gt.nlo_orders(qcd_pos)) cycle

          amp_split_6to5f_muf(orders_to_amp_split_pos(orders)) = 
     &     alphas / 3d0 / pi * TF * dble(niglu) * amp_split(iamp)  

          amp_split_6to5f_mur(orders_to_amp_split_pos(orders)) = 
     &    - alphas / 3d0 / pi * TF * dble(alphasbpow) * amp_split(iamp) 
        
          amp_split_6to5f(orders_to_amp_split_pos(orders)) = 
     &    dlog(qes2/mdl_mt**2) * 
     &     (alphas / 3d0 / pi * TF * dble(niglu)   
     &    - alphas / 3d0 / pi * TF * dble(alphasbpow)) * amp_split(iamp)
        endif
      enddo

      return
      end


      subroutine compute_ewsudakov
c This subroutine computes the NLO EW corrections in the Sudakov
c   approximation
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      include 'has_ewsudakov.inc'
      integer orders_qcd(nsplitorders), orders_ew(nsplitorders)
      integer iamp

      double precision wgt_c
      double precision wgt1
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision  xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision g22
      integer get_orders_tag

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double complex amp_split_ewsud_par(amp_split_size)
      common /to_amp_ewsud_par/amp_split_ewsud_par
      double complex amp_split_ewsud_qcd(amp_split_size)
      common /to_amp_ewsud_qcd/amp_split_ewsud_qcd
      double complex amp_split_ewsud_parqcd(amp_split_size)
      common /to_amp_ewsud_parqcd/amp_split_ewsud_parqcd
      ! sudakov mode
      integer sud_mod
      common /to_sud_mod/ sud_mod
      include 'ewsudakov_haslo.inc' 

      if (.not.has_ewsudakov) return

      call cpu_time(tBefore)
      if (f_b.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return

      if (cpower_pos.gt.0) then
          write(*,*)'Error, cannot compute EW sudakov with Cpower >0'
          stop 1
      endif

      ! sud_mod = 0
      do sud_mod = 0,1

       call sborn(p_born,wgt_c)
       call sudakov_wrapper(p_born)
       do iamp=1, amp_split_size
        if (amp_split_ewsud_lsc(iamp).eq.0d0.and.
     $      amp_split_ewsud_ssc(iamp).eq.0d0.and.
     $      amp_split_ewsud_xxc(iamp).eq.0d0.and.
     $      amp_split_ewsud_par(iamp).eq.0d0.and.
     $      AMP_SPLIT_EWSUD_QCD(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders_ew)
        orders_qcd(:) = orders_ew(:)
        ! we have two arrays of orders, one for the contributions
        ! of EW origin (from LO1) and one for those of QCD origin
        ! (from LO2)

        !!!! first the contribution of EW origin
        ! increase the EW-coupling of 2, since until here
        ! the EW sudakov amp_split has the same positions of 
        ! those for the Born
        if (has_lo1) then
          orders_ew(qed_pos)=orders_ew(qed_pos)+2
          QCD_power=orders_ew(qcd_pos)
          wgtcpower=0d0
          !!!!if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
          orders_tag=get_orders_tag(orders_ew)
          wgt1=(amp_split_ewsud_lsc(iamp)+
     $        amp_split_ewsud_ssc(iamp)+
     $        amp_split_ewsud_xxc(iamp)+
     $        amp_split_ewsud_par(iamp))
     $         *f_b/g**(qcd_power)
          wgt1=wgt1*2d0 ! missing factor in the sudakov correction
          ! the type will be 20+the value of the sudakov mode
          call add_wgt(20+sud_mod,orders_ew,wgt1,0d0,0d0)
        endif

        !!!! then the contribution of QCD origin
        ! increase the QCD-coupling of 2, since until here
        ! the EW sudakov amp_split has the same positions of 
        ! those for the Born, and for QCD this is LO2
        if (has_lo2) then
          orders_qcd(qcd_pos)=orders_qcd(qcd_pos)+2
          QCD_power=orders_qcd(qcd_pos)
          !!wgtcpower=0d0
          !!if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
          orders_tag=get_orders_tag(orders_qcd)
          wgt1=(amp_split_ewsud_qcd(iamp)+
     $        amp_split_ewsud_parqcd(iamp))
     $         *f_b/g**(qcd_power)
          wgt1=wgt1*2d0 ! missing factor in the sudakov correction
          ! the type will be 20+the value of the sudakov mode
          call add_wgt(20+sud_mod,orders_qcd,wgt1,0d0,0d0)
        endif
       enddo
      enddo
      call cpu_time(tAfter)
      t_ewsud=t_ewsud+(tAfter-tBefore)
      return
      end



      subroutine compute_alpha_cnt()
C This is the counterterm for the change of scheme
C in the UV renormalisation for alpha in (leptonic) PDFs
C wrt the hard matrix element. Relevant for lepton collisions. 
C It is called in this function such that if is included
C in the LO cross section
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc' 
      include 'q_es.inc'
      include 'run.inc'
      include 'genps.inc'
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_alpha(amp_split_size),
     &                 amp_split_alpha_muf(amp_split_size),
     &                 amp_split_alpha_mur(amp_split_size)
      common /to_amp_split_alpha/ amp_split_alpha, amp_split_alpha_muf, 
     &                            amp_split_alpha_mur
      integer orders_to_amp_split_pos
      integer i, j, k
      logical firsttime
      data firsttime /.true./
      double precision tf, nc, pi
      parameter (tf=0.5d0)
      parameter (nc=3d0)
      parameter (pi=3.1415926535897932385d0)
      integer alphabpow
      double precision wgtborn, alpha
      double precision ch_lep, ch_up, ch_dn
      parameter (ch_lep=1d0)
      parameter (ch_up=2d0/3d0)
      parameter (ch_dn=1d0/3d0)
      ! the number of families
      integer n_lep, n_up, n_dn
      double precision sumcharge, beta0, const
      double precision w_thresh
      integer jsign

c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]

      ! set everything to 0
      amp_split_alpha(1:amp_split_size) = 0d0
      amp_split_alpha_muf(1:amp_split_size) = 0d0
      amp_split_alpha_mur(1:amp_split_size) = 0d0

      if (firsttime) then
        write(*,*) 'Parameters related to the different treatment of'
     ^     // ' alpha in ME and PDFs'
        write(*,*) 'alphascheme', alphascheme
        write(*,*) 'nlep_run   ', nlep_run
        write(*,*) 'nupq_run   ', nupq_run
        write(*,*) 'ndnq_run   ', ndnq_run
        write(*,*) 'w_run      ', w_run
        firsttime = .false.
      endif

      ! skip if we don't want this piece 
      if (alphascheme.eq.0) then
         ! do nothing, assumes same UV scheme in alpha for
         ! PDF's and model
         return
      else if (abs(alphascheme).eq.1) then
         ! the sign of alphascheme controls the sign of the counterterm.
         ! negative sign used for testing (in the test process, the
         ! virtuals are computed in the MSbar scheme, unlike in the
         ! processes automatically generated by MG5aMC
         jsign = sign(1,alphascheme)
         ! compute the born
         call sborn(p_born,wgtborn)
         ! assumes alpha(MZ) for model, MSbar for PDFs
         ! the number of flavours depends on mur.
         ! here we will treat all leptons as massless, 
         ! blocking the code if mur < 5 gev
         if (nlep_run.eq.-1) then
            n_lep = 3
         else
            n_lep = nlep_run
         endif
         if (nupq_run.eq.-1) then
            n_up = 2
         else
            n_up = nupq_run
         endif
         if (ndnq_run.eq.-1) then
            n_dn = 3
         else
            n_dn = ndnq_run
         endif
         sumcharge = n_lep*ch_lep**2 + nc*(n_up*ch_up**2+n_dn*ch_dn**2)
         if (firsttime) then
            write(*,*) 'compute_alpha_cnt with nlep, nup, ndn, wrun', 
     #        n_lep, n_up, n_dn, w_run
            write(*,*) '  sum_{lep,up,dn} n*q^2*nc ', sumcharge 
            firsttime=.false.
          endif
         ! the factor has the form 
         ! alpha/3pi* (cons + beta0 log(mur^2/mz^2) * bpow
         ! where bpow is the power of alpha
         const = 5d0/3d0 * sumcharge - 
     #       w_run * (1d0/2d0 + 21d0/4d0 * dlog(mdl_mz**2/mdl_mw**2))
         ! in practice we have only the case mur > mw or mur<mw
         if (scale.gt.mdl_mw) then
            beta0 = sumcharge - 21d0/4d0 * w_run
            w_thresh = 0d0
         else if (scale.gt.5d0) then
            beta0 = sumcharge 
            w_thresh = - 21d0/4d0 * w_run
         else
            ! we hardcode 5d0 instead of MB as in the model the bottom
            ! is massless
            write(*,*) 'MUR too low, bottom threshold not implemented'
            stop 1
         endif
         alpha = dble(gal(1))**2/4d0/pi
         do iamp = 1, amp_split_size
           if (amp_split(iamp).eq.0d0) cycle
           call amp_split_pos_to_orders(iamp, orders)
           alphabpow = orders(qed_pos)/2
           if (alphabpow.ne.0) then
             ! this contribution will end up with one extra power
             ! of alpha
             orders(qed_pos) = orders(qed_pos) + 2
             amp_split_alpha_muf(orders_to_amp_split_pos(orders)) = 0d0 
             amp_split_alpha_mur(orders_to_amp_split_pos(orders)) = 
     &           - jsign * alpha / 3d0 / pi * alphabpow * amp_split(iamp) *
     &              beta0
             amp_split_alpha(orders_to_amp_split_pos(orders)) = 
     &           - jsign * alpha / 3d0 / pi * alphabpow * amp_split(iamp) * 
     &            (beta0 * dlog(qes2/mdl_mz**2) + 
     &             w_thresh * dlog(mdl_mw**2/mdl_mz**2) + const)
           endif
         enddo
      else 
         write(*,*) 'change of scheme factors for gmu not implemented'
         stop 1
      endif

      return
      end




      subroutine compute_nbody_noborn
c This subroutine computes the soft-virtual matrix elements and adds its
c value to the list of weights using the add_wgt subroutine
      use extra_weights
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp, i
      double precision amp_split_virt(amp_split_size),
     &     amp_split_born_for_virt(amp_split_size),
     &     amp_split_avv(amp_split_size)
      common /to_amp_split_virt/amp_split_virt,
     &                          amp_split_born_for_virt,
     &                          amp_split_avv
      double precision amp_split_wgtnstmp(amp_split_size),
     $                 amp_split_wgtwnstmpmuf(amp_split_size),
     $                 amp_split_wgtwnstmpmur(amp_split_size)
      common /to_amp_split_bsv/amp_split_wgtnstmp,
     $                         amp_split_wgtwnstmpmuf,
     $                         amp_split_wgtwnstmpmur

      ! stuff for the 6->5 flav scheme
      double precision amp_split_6to5f(amp_split_size),
     &                 amp_split_6to5f_muf(amp_split_size),
     &                 amp_split_6to5f_mur(amp_split_size)
      common /to_amp_split_6to5f/ amp_split_6to5f, amp_split_6to5f_muf, 
     &                            amp_split_6to5f_mur

      ! stuff for the alpha UV-scheme in lepton collisions
      double precision amp_split_alpha(amp_split_size),
     &                 amp_split_alpha_muf(amp_split_size),
     &                 amp_split_alpha_mur(amp_split_size)
      common /to_amp_split_alpha/ amp_split_alpha, amp_split_alpha_muf, 
     &                            amp_split_alpha_mur

      double precision wgt6f1,wgt6f2,wgt6f3
      double precision wgtal1,wgtal2,wgtal3

      double precision wgt1,wgt2,wgt3,bsv_wgt,virt_wgt,born_wgt,pi,g2
     &     ,g22,wgt4
      parameter (pi=3.1415926535897932385d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision  xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      integer get_orders_tag
      character*4      abrv
      common /to_abrv/ abrv
      integer iamp_test
      call cpu_time(tBefore)
      if (f_nb.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0) .gt. xiBSVcut_used) return
      call bornsoftvirtual(p1_cnt(0,1,0),bsv_wgt,virt_wgt,born_wgt)
      if (ickkw.eq.-1) then
         if (wgtbpower.ne.0) then
            write (*,*) 'ERROR in VETO XSec: bpower should'/
     $           /' be zero (no QCD partons at the'/
     $           /' Born allowed)', wgtbpower
         endif
         H1_factor_virt=virt_wgt/(g22/(4d0*pi))/born_wgt
         born_wgt_veto=born_wgt/g2
         call compute_veto_compensating_factor(H1_factor_virt
     $        ,born_wgt_veto,1d0,1d0,veto_compensating_factor)
C Since VETOXSEC must still be adapted in FKS_EW, I put a dummy
C order array here which I arbitrarily chose to be the (-1,-1)
C to make sure that it cannot be incorrectly understood.
         do i=1,nsplitorders
           orders(i)=-1
         enddo
         call add_wgt(7,orders,-veto_compensating_factor*f_nb,0d0,0d0)
        write(*,*) 'FIX VETOXSEC in FKS_EW'
        stop
      endif
      iamp_test=0
      do iamp=1, amp_split_size
        if (amp_split_wgtnstmp(iamp).eq.0d0.and.
     $      amp_split_wgtwnstmpmur(iamp).eq.0d0.and.
     $      amp_split_wgtwnstmpmuf(iamp).eq.0d0.and.
     $      amp_split_avv(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        wgt1=amp_split_wgtnstmp(iamp)*f_nb/g22
        wgt2=amp_split_wgtwnstmpmur(iamp)*f_nb/g22
        wgt3=amp_split_wgtwnstmpmuf(iamp)*f_nb/g22
        wgt4=amp_split_avv(iamp)*f_nb/g22
        if (ickkw.eq.3 .and. fxfx_exp_rewgt.ne.0d0
     &       .and. abrv.ne.'born') then
! This assumes a single Born order, which must always be the case for
! FxFx. Explicitly check this just to be sure.
           iamp_test=iamp_test+1
           if(iamp_test.ne.1) then
              write (*,*) "There should only be one possible"/
     $             /" Born order for FxFx"
              stop 1
           endif
           g2=g**(QCD_power-2)
           wgt1=wgt1 - fxfx_exp_rewgt*born_wgt*f_nb/g2/(4d0*pi)
        endif
        call add_wgt(3,orders,wgt1,wgt2,wgt3)
        call add_wgt(15,orders,wgt4,0d0,0d0)
      enddo
c Special for the soft-virtual needed for the virt-tricks. The
c *_wgt_mint variable should be directly passed to the mint-integrator
c and not be part of the plots nor computation of the cross section.
      virt_wgt_mint(0)=virt_wgt_mint(0)+virt_wgt*f_nb
      born_wgt_mint(0)=born_wgt_mint(0)+born_wgt*f_b
      do iamp=1, amp_split_size
        if (amp_split_virt(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=amp_split_virt(iamp)*f_nb
        virt_wgt_mint(iamp)=virt_wgt_mint(iamp)
     $       +wgt1
        born_wgt_mint(iamp)=born_wgt_mint(iamp)
     $       +amp_split_born_for_virt(iamp)*f_nb
        wgt1=wgt1/g**(QCD_power)
        call add_wgt(14,orders,wgt1,0d0,0d0)
      enddo

C This is the counterterm for the 6f->5f scheme change 
C of parton distributions (e.g. NNPDF2.3). 
      call compute_6to5flav_cnt()
      do iamp=1, amp_split_size
        if (amp_split_6to5f(iamp).eq.0d0.and.
     $      amp_split_6to5f_mur(iamp).eq.0d0.and.
     $      amp_split_6to5f_muf(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        g22=g**(QCD_power)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt6f1=amp_split_6to5f(iamp)*f_nb/g**(qcd_power)
        wgt6f2=amp_split_6to5f_mur(iamp)*f_nb/g**(qcd_power)
        wgt6f3=amp_split_6to5f_muf(iamp)*f_nb/g**(qcd_power)
        call add_wgt(3,orders,wgt6f1,wgt6f2,wgt6f3)
      enddo

C This is the counterterm for the change of scheme
C in the UV renormalisation for alpha in (leptonic) PDFs
C wrt the hard matrix element. Relevant for lepton collisions. 
      call compute_alpha_cnt()
      do iamp=1, amp_split_size
        if (amp_split_alpha(iamp).eq.0d0.and.
     $      amp_split_alpha_mur(iamp).eq.0d0.and.
     $      amp_split_alpha_muf(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        g22=g**(QCD_power)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgtal1=amp_split_alpha(iamp)*f_nb/g**(qcd_power)
        wgtal2=amp_split_alpha_mur(iamp)*f_nb/g**(qcd_power)
        wgtal3=amp_split_alpha_muf(iamp)*f_nb/g**(qcd_power)
        call add_wgt(3,orders,wgtal1,wgtal2,wgtal3)
      enddo

      call cpu_time(tAfter)
      tIS=tIS+(tAfter-tBefore)
      return
      end

      subroutine compute_real_emission(p,sudakov_damp)
c This subroutine computes the real-emission matrix elements and adds
c its value to the list of weights using the add_wgt subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision s_ev,fks_Sij,p(0:3,nexternal),wgt1,fx_ev
     $     ,sudakov_damp
      external fks_Sij
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      integer get_orders_tag
      call cpu_time(tBefore)
      if (f_r.eq.0d0) return
      s_ev = fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      if (s_ev.le.0.d0) return
      call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx_ev)
      do iamp=1, amp_split_size
        if (amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        wgt1=amp_split(iamp)*s_ev*f_r/g**(qcd_power)
        if (sudakov_damp.gt.0d0) then
          call add_wgt(1,orders,wgt1*sudakov_damp,0d0,0d0)
        endif
        if (sudakov_damp.lt.1d0) then
          call add_wgt(11,orders,wgt1*(1d0-sudakov_damp),0d0,0d0)
        endif
      enddo
      call cpu_time(tAfter)
      tReal=tReal+(tAfter-tBefore)
      return
      end

      subroutine compute_soft_counter_term(replace_MC_subt)
c This subroutine computes the soft counter term and adds its value to
c the list of weights using the add_wgt subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision wgt1,s_s,fks_Sij,fx_s,zero,replace_MC_subt,g22
      parameter (zero=0d0)
      external fks_Sij
      double precision     p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/ p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision  xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      integer get_orders_tag
      call cpu_time(tBefore)
      if (f_s.eq.0d0 .and. f_s_MC_S.eq.0d0 .and. f_s_MC_H.eq.0d0) return
      if (xi_i_hat_ev*xiimax_cnt(0).gt.xiScut_used .and. replace_MC_subt.eq.0d0)
     $     return
      s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
      if (s_s.le.0d0) return
      call sreal(p1_cnt(0,1,0),0d0,y_ij_fks_ev,fx_s)

      do iamp=1, amp_split_size
        if (amp_split(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=amp_split(iamp)*s_s/g22*replace_MC_subt
          call add_wgt(8,orders,-wgt1*f_s_MC_H,0d0,0d0)
          wgt1=wgt1*f_s_MC_S
        else
          wgt1=0d0
        endif
        if (xi_i_fks_ev.le.xiScut_used) then
          wgt1=wgt1-amp_split(iamp)*s_s*f_s/g22
        endif
        if (wgt1.ne.0d0) call add_wgt(4,orders,wgt1,0d0,0d0)
      enddo

      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_collinear_counter_term(replace_MC_subt)
c This subroutine computes the collinear counter term and adds its value
c to the list of weights using the add_wgt subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'fks_powers.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
      common /to_amp_split_deg/amp_split_wgtdegrem_xi,
     $                         amp_split_wgtdegrem_lxi,
     $                         amp_split_wgtdegrem_muF
      ! amp_split for the PDF scheme
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
      common /to_amp_split_dis/amp_split_wgtpsch_p,
     $                         amp_split_wgtpsch_l,
     $                         amp_split_wgtpsch_d
      double precision zero,one,s_c,fks_Sij,fx_c,deg_xi_c,deg_lxi_c,wgt1
     &     ,wgt3,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision pmass(nexternal)
      integer get_orders_tag
      call cpu_time(tBefore)
      include 'pmass.inc'
      if (f_c.eq.0d0 .and. f_dc.eq.0d0 .and. f_c_MC_S.eq.0d0 .and.
     $     f_c_MC_H.eq.0d0)return
      if ( (y_ij_fks_ev.le.1d0-deltaS .and. replace_MC_subt.eq.0d0) .or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(1),one)
      if (s_c.le.0d0) return
      ! sreal_deg should be called **BEFORE** sreal 
      ! in order not to overwrtie the amp_split array
      call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,deg_xi_c
     $     ,deg_lxi_c)
      call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx_c)

      do iamp=1, amp_split_size
        if (amp_split(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_xi(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_lxi(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_p(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_l(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_d(iamp).eq.0d0) cycle

        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=amp_split(iamp)*s_c/g22*replace_MC_subt
          call add_wgt(9,orders,-wgt1*f_c_MC_H,0d0,0d0)
          wgt1=wgt1*f_c_MC_S
        else
          wgt1=0d0
        endif
        if (y_ij_fks_ev.gt.1d0-deltaS) then
          wgt1=wgt1-amp_split(iamp)*s_c*f_c/g22
          wgt1=wgt1+
     $         (amp_split_wgtdegrem_xi(iamp)+amp_split_wgtpsch_p(iamp)+
     $         (amp_split_wgtdegrem_lxi(iamp)+amp_split_wgtpsch_l(iamp))
     $           *log(xi_i_fks_cnt(1)))*f_dc/g22
          wgt3=amp_split_wgtdegrem_muF(iamp)*f_dc/g22
        else
          wgt3=0d0
        endif
        if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt(5,orders,wgt1,0d0,wgt3)
      enddo

      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_soft_collinear_counter_term(replace_MC_subt)
c This subroutine computes the soft-collinear counter term and adds its
c value to the list of weights using the add_wgt subroutine
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'fks_powers.inc'
      include 'timing_variables.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer iamp
      double precision amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
      common /to_amp_split_deg/amp_split_wgtdegrem_xi,
     $                         amp_split_wgtdegrem_lxi,
     $                         amp_split_wgtdegrem_muF
      ! amp_split for the PDF scheme
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
      common /to_amp_split_dis/amp_split_wgtpsch_p,
     $                         amp_split_wgtpsch_l,
     $                         amp_split_wgtpsch_d
      double precision zero,one,s_sc,fks_Sij,fx_sc,wgt1,wgt3,deg_xi_sc
     $     ,deg_lxi_sc,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision  xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      ! PDF scheme prefactors
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      double precision pmass(nexternal)
      integer get_orders_tag
      include 'pmass.inc'
      call cpu_time(tBefore)
      if (f_sc.eq.0d0 .and. f_dsc(1).eq.0d0 .and. f_dsc(2).eq.0d0 .and.
     $     f_dsc(3).eq.0d0 .and. f_dsc(4).eq.0d0 .and. f_sc_MC_S.eq.0d0
     $     .and. f_sc_MC_H.eq.0d0) return
      if ( ((xi_i_hat_ev*xiimax_cnt(1).ge.xiScut_used .or. y_ij_fks_ev.le.1d0
     $     -deltaS) .and. replace_MC_subt.eq.0d0).or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
      if (s_sc.le.0d0) return
      ! sreal_deg should be called **BEFORE** sreal 
      ! in order not to overwrtie the amp_split array
      call sreal_deg(p1_cnt(0,1,2),zero,one, deg_xi_sc,deg_lxi_sc)
      call sreal(p1_cnt(0,1,2),zero,one,fx_sc)

      do iamp=1, amp_split_size
        if (amp_split(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_xi(iamp).eq.0d0.and.
     $      amp_split_wgtdegrem_lxi(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_p(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_l(iamp).eq.0d0.and.
     $      amp_split_wgtpsch_d(iamp).eq.0d0) cycle
        call amp_split_pos_to_orders(iamp, orders)
        QCD_power=orders(qcd_pos)
        wgtcpower=0d0
        if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
        orders_tag=get_orders_tag(orders)
        amp_pos=iamp
        g22=g**(QCD_power)
        if (replace_MC_subt.gt.0d0) then
          wgt1=-amp_split(iamp)*s_sc/g22*replace_MC_subt
          call add_wgt(10,orders,-wgt1*f_sc_MC_H,0d0,0d0)
          wgt1=wgt1*f_sc_MC_S
        else
          wgt1=0d0
        endif
        if (xi_i_fks_cnt(1).lt.xiScut_used .and. 
     $      y_ij_fks_ev.gt.1d0-deltaS) then
          wgt1=wgt1+amp_split(iamp)*s_sc*f_sc/g22
          wgt1=wgt1+
     $         (-(amp_split_wgtdegrem_xi(iamp)+amp_split_wgtpsch_p(iamp)+
     $           (amp_split_wgtdegrem_lxi(iamp)+amp_split_wgtpsch_l(iamp))
     $              *log(xi_i_fks_cnt(1)))*f_dsc(1)-
     $           (amp_split_wgtdegrem_xi(iamp)*f_dsc(2)+
     $            amp_split_wgtdegrem_lxi(iamp)*f_dsc(3))+
     $            amp_split_wgtpsch_d(iamp)*f_pdfsch_d+
     $            amp_split_wgtpsch_p(iamp)*f_pdfsch_p+
     $            amp_split_wgtpsch_l(iamp)*f_pdfsch_l)/g22
          wgt3=-amp_split_wgtdegrem_muF(iamp)*f_dsc(4)/g22
        else
          wgt3=0d0
        endif
        if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt(6,orders,wgt1,0d0,wgt3)
      enddo

      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_MC_subt_term(p,passcuts,gfactsf,gfactcl,probne)
      use extra_weights
      implicit none
c This subroutine computes the MonteCarlo subtraction terms and adds
c their values to the list of weights using the add_wgt subroutine. It
c returns the values for the gfactsf, gfactcl and probne to check if we
c need to include the FKS subtraction terms as replacements in the soft
c and collinear limits and the Sudakov damping for the real-emission,
c respectively.
      include 'nexternal.inc'
      include 'madfks_mcatnlo.inc'
      include 'timing_variables.inc'
      include 'coupl.inc'
      include 'orders.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      integer nofpartners,i
      double precision p(0:3,nexternal),gfactsf,gfactcl,probne,fks_Sij
     $     ,sevmc,zhw(nexternal),xmcxsec(nexternal),g22,wgt1
     $     ,xlum_mc_fact,fks_Hij
      external fks_Sij,fks_Hij
      logical lzone(nexternal),flagmc,passcuts
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer           fks_j_from_i(nexternal,0:nexternal)
     &                  ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      integer iamp
      integer orders(nsplitorders)
      double precision amp_split_xmcxsec(amp_split_size,nexternal)
      common /to_amp_split_xmcxsec/amp_split_xmcxsec
      integer get_orders_tag
      integer                     n_MC_subt_diverge
      common/counter_subt_diverge/n_MC_subt_diverge
      call cpu_time(tBefore)
      call compute_xmcsubt_complete(p,probne,gfactsf,gfactcl,flagmc
     $     ,lzone,zhw,nofpartners,xmcxsec)
      if (f_MC_S.eq.0d0 .and. f_MC_H.eq.0d0) return
      if(UseSfun)then
         sevmc = fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      else
         sevmc = fks_Hij(p,i_fks,j_fks)
      endif
      if (sevmc.eq.0d0) return
      if (passcuts .and. flagmc) then
         do i=1,nofpartners
            if(lzone(i))then
              call get_mc_lum(j_fks,zhw(i),xi_i_fks_ev,xlum_mc_fact)
              do iamp=1, amp_split_size
                if (amp_split_xmcxsec(iamp,i).eq.0d0) cycle
                call amp_split_pos_to_orders(iamp, orders)
                QCD_power=orders(qcd_pos)
                wgtcpower=0d0
                if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
                orders_tag=get_orders_tag(orders)
                amp_pos=iamp
                g22=g**(QCD_power)
                wgt1=sevmc*f_MC_S*xlum_mc_fact*
     &               amp_split_xmcxsec(iamp,i)/g22
                call add_wgt(12,orders,wgt1,0d0,0d0)
                wgt1=sevmc*f_MC_H*xlum_mc_fact*
     &               amp_split_xmcxsec(iamp,i)/g22
                call add_wgt(13,orders,-wgt1,0d0,0d0)
              enddo
            endif
         enddo
      endif
      if( (.not.flagmc) .and. gfactsf.eq.1.d0 .and.
     $     xi_i_fks_ev.lt.0.02d0 .and. particle_type(i_fks).eq.8 )then
         n_MC_subt_diverge=n_MC_subt_diverge+1
      endif
      call cpu_time(tAfter)
      t_MC_subt=t_MC_subt+(tAfter-tBefore)
      return
      end


      logical function pdg_equal(pdg1,pdg2)
c Returns .true. if the lists of PDG codes --'pdg1' and 'pdg2'-- are
c equal.
      implicit none
      include 'nexternal.inc'
      integer i,pdg1(nexternal),pdg2(nexternal)
      pdg_equal=.true.
      do i=1,nexternal
         if (pdg1(i).ne.pdg2(i)) then
            pdg_equal=.false.
            return
         endif
      enddo
      end

      logical function colour_con_equal(n,icol1,icol2)
c Returns .true. if the lists of colour-connection codes --'icol1' and
c 'icol2'-- are equal. It is not smart: if they are equal, but are
c labelled differently (e.g. replacing '501' with '502' and vice versa)
c this routine will NOT consider them equal. If the first element of
c icol1 or icol2 is equal to -1, it means that that array has not been
c set --> if they are both equal to -1, consider icol1 and icol2 equal
      implicit none
      include 'nexternal.inc'
      integer n,i,j,icol1(2,nexternal),icol2(2,nexternal)
      colour_con_equal=.true.
      if (icol1(1,1).eq.-1 .and. icol2(1,1).eq.-1) return
      do i=1,n
         do j=1,2
            if (icol1(j,i).ne.icol2(j,i)) then
               colour_con_equal=.false.
               return
            endif
         enddo
      enddo
      end
      
      logical function momenta_equal(p1,p2)
c Returns .true. only if the momenta p1 and p2 are equal. To save time,
c it only checks the 0th and 3rd components (energy and z-direction).
      implicit none
      include 'nexternal.inc'
      integer i,j
      double precision p1(0:3,nexternal),p2(0:3,nexternal),vtiny
      parameter (vtiny=1d-8)
      momenta_equal=.true.
      do i=1,nexternal
         do j=0,3,3
            if (p1(j,i).eq.0d0 .or. p2(j,i).eq.0d0) then
               if (abs(p1(j,i)-p2(j,i)).gt.vtiny) then
                  momenta_equal=.false.
                  return
               endif
            else
               if (abs((p1(j,i)-p2(j,i))/
     $                    max(abs(p1(j,i)),abs(p2(j,i)))).gt.vtiny) then
                  momenta_equal=.false.
                  return
               endif
            endif
         enddo
      enddo
      end
      
      logical function momenta_equal_uborn(p1,p2,jfks1,ifks1,jfks2
     $     ,ifks2)
c Returns .true. only if the momenta p1 and p2 are equal, but with the
c momenta of i_fks and j_fks summed. To save time, it only checks the
c 0th and 3rd components (energy and z-direction).
      implicit none
      include 'nexternal.inc'
      integer i,j,jfks1,ifks1,jfks2,ifks2
      double precision p1(0:3,nexternal),p2(0:3,nexternal),vtiny,pb1(0:3
     $     ,nexternal),pb2(0:3,nexternal)
      logical momenta_equal
      external momenta_equal
      parameter (vtiny=1d-8)
c Fill the underlying Born momenta pb1 and pb2
      do i=1,nexternal
         do j=0,3,3 ! skip x and y components, since they are not used in
                    ! the 'momenta_equal' function
            if (i.lt.ifks1) then
               pb1(j,i)=p1(j,i)
            elseif (i.eq.ifks1) then
c Sum the i_fks to the j_fks momenta (i_fks is always greater than
c j_fks, so this is fine: it will NOT be overwritten later in the
c do-loop)
               pb1(j,jfks1)=pb1(j,jfks1)+p1(j,i)
               pb1(j,nexternal)=0d0 ! fill the final one with zero's
            else
               pb1(j,i-1)=p1(j,i)   ! skip the i_fks momenta
            endif
            if (i.lt.ifks2) then
               pb2(j,i)=p2(j,i)
            elseif (i.eq.ifks2) then
               pb2(j,jfks2)=pb2(j,jfks2)+p2(j,i) ! sum i_fks to j_fks momenta
               pb2(j,nexternal)=0d0 ! fill the final one with zero's
            else
               pb2(j,i-1)=p2(j,i)   ! skip the i_fks momenta
            endif
         enddo
      enddo
c Check if they are equal
      momenta_equal_uborn=momenta_equal(pb1,pb2)
      end
      
      subroutine set_FxFx_scale(iterm,p)
c Sets the FxFx cluster scale and multiplies the f_* factors (computed
c by 'compute_prefactors_nbody' and 'compute_prefactors_n1body') by the
c Sudakov suppression. If called more than once with the same momenta
c and iterm, skip setting of the scales, and multiply the f_* factors by
c the cached Sudakovs.
c     iterm=  0 : reset the computation of the Sudakovs
c     iterm=  1 : Sudakov for n-body kinematics (f_b and f_nb)
c     iterm=  2 : Sudakov for n-body kinematics (all but f_b and f_nb)
c     iterm=  3 : Sudakov for n+1-body kinematics
c     iterm= -1 or -2 : only restore scales for n-body w/o recomputing
c     iterm= -3 : only restore scales for n+1-body w/o recomputing
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'nFKSconfigs.inc'
      integer iterm,iterm_last_izero,iterm_last_mohdr,i,j
     &     ,nfxfx_ren_scales_izero,nfxfx_ren_scales_mohdr
      double precision p(0:3,nexternal),p_last_izero(0:3,nexternal)
     &     ,p_last_mohdr(0:3,nexternal),rewgt,rewgt_izero,rewgt_mohdr
     &     ,rewgt_exp_izero,rewgt_exp_mohdr,pthardness
     &     ,fxfx_ren_scales_izero(0:nexternal),fxfx_fac_scale_izero(2)
     &     ,fxfx_ren_scales_mohdr(0:nexternal),fxfx_fac_scale_mohdr(2)
      logical setclscales,rewgt_izero_calculated,rewgt_mohdr_calculated
     &     ,momenta_equal,already_set
      external setclscales,rewgt,momenta_equal
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision         fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      integer                              nFxFx_ren_scales
      double precision     FxFx_ren_scales(0:nexternal),
     $                     FxFx_fac_scale(2)
      common/c_FxFx_scales/FxFx_ren_scales,nFxFx_ren_scales,
     $                     FxFx_fac_scale
      INTEGER              NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      save rewgt_mohdr_calculated,rewgt_izero_calculated,p_last_izero
     &     ,p_last_mohdr,iterm_last_izero,iterm_last_mohdr
     &     ,fxfx_ren_scales_izero ,fxfx_ren_scales_mohdr
     &     ,fxfx_fac_scale_izero ,fxfx_fac_scale_mohdr
     &     ,nfxfx_ren_scales_izero ,nfxfx_ren_scales_mohdr
      integer need_matching(nexternal),need_matching_izero(nexternal)
      integer need_matching_S(nexternal),need_matching_H(nexternal)
      common /c_need_matching/ need_matching_S,need_matching_H
      save need_matching_izero
      double precision shower_S_scale(fks_configs*2)
     &     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     &     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness
      call cpu_time(tBefore)
      ktscheme=1
      if (iterm.eq.0) then
         rewgt_mohdr_calculated=.false.
         rewgt_izero_calculated=.false.
         fxfx_exp_rewgt=0d0
         return
      endif
      already_set=.false.
      if (iterm.eq.1 .or. iterm.eq.2) then
c n-body momenta FxFx Sudakov factor (i.e. for S-events)
         if (rewgt_izero_calculated) then
            if (iterm_last_izero.ne.1 .and. iterm_last_izero.ne.2) then
               if (momenta_equal(p1_cnt(0,1,0),p_last_izero)) then
                  already_set=.true.
               endif
            endif
         endif
         if (.not.already_set) then
            call cluster_and_reweight(0,rewgt_izero,rewgt_exp_izero
     $           ,nFxFx_ren_scales,FxFx_ren_scales(0)
     $           ,fxfx_fac_scale(1),need_matching)
            fxfx_fac_scale(2)=fxfx_fac_scale(1)
            rewgt_izero=min(rewgt_izero,1d0)
            fxfx_exp_rewgt=min(rewgt_exp_izero,0d0)
            need_matching_S(1:nexternal)=need_matching(1:nexternal)
            need_matching_izero(1:nexternal)=need_matching_S(1:nexternal)
c Update shower starting scale to be the scale down to which the MINLO
c Sudakov factors are included.
            shower_S_scale(nFKSprocess*2-1)=
     $           minval(FxFx_ren_scales(0:nFxFx_ren_scales))
            shower_S_scale(nFKSprocess*2)=
     $           shower_S_scale(nFKSprocess*2-1)
         endif
         rewgt_izero_calculated=.true.
         iterm_last_izero=iterm
         do i=1,nexternal
            do j=0,3
               p_last_izero(j,i)=p1_cnt(j,i,0)
            enddo
         enddo
         if (iterm.eq.1) then
            f_b =f_b *rewgt_izero
            f_nb=f_nb*rewgt_izero
         elseif(iterm.eq.2) then
            f_s =f_s *rewgt_izero
            f_c =f_c *rewgt_izero
            f_dc=f_dc*rewgt_izero
            f_sc=f_sc*rewgt_izero
            do i=1,4
               f_dsc(i)=f_dsc(i)*rewgt_izero
            enddo
            f_MC_S =f_MC_S *rewgt_izero
            f_s_MC_S =f_s_MC_S *rewgt_izero
            f_c_MC_S =f_c_MC_S *rewgt_izero
            f_sc_MC_S=f_sc_MC_S*rewgt_izero
         endif
         nFxFx_ren_scales_izero=nFxFx_ren_scales
         do i=1,nexternal
            need_matching_izero(i)=need_matching(i)
         enddo
         do i=0,nexternal
            FxFx_ren_scales_izero(i)=FxFx_ren_scales(i)
         enddo
         do i=1,2
            FxFx_fac_scale_izero(i)=FxFx_fac_scale(i)
         enddo
      elseif (iterm.eq.3) then
c n+1-body momenta FxFx Sudakov factor (i.e. for H-events)
         if (rewgt_mohdr_calculated) then
            if (iterm.eq.iterm_last_mohdr) then
               if (momenta_equal(p,p_last_mohdr)) then
                  already_set=.true.
               endif
            endif
         endif
         if (.not. already_set) then
            call cluster_and_reweight(nFKSprocess,rewgt_mohdr
     $           ,rewgt_exp_mohdr,nFxFx_ren_scales,FxFx_ren_scales(0)
     $           ,fxfx_fac_scale(1),need_matching)
            fxfx_fac_scale(2)=fxfx_fac_scale(1)
            rewgt_mohdr=min(rewgt_mohdr,1d0)
            need_matching_H(1:nexternal)=need_matching(1:nexternal)
c Update shower starting scale
            pthardness=ref_H_scale(nFKSprocess*2)-
     $           shower_H_scale(nFKSprocess*2)
            shower_H_scale(nFKSprocess*2)=
     $           minval(FxFx_ren_scales(0:nFxFx_ren_scales))
            ref_H_scale(nFKSprocess*2)=shower_H_scale(nFKSprocess*2)
     $           +pthardness
            pthardness=ref_H_scale(nFKSprocess*2-1)-
     $           shower_H_scale(nFKSprocess*2-1)
            shower_H_scale(nFKSprocess*2-1)= 
     $           shower_H_scale(nFKSprocess*2)
            ref_H_scale(nFKSprocess*2-1)=shower_H_scale(nFKSprocess*2-1)
     $           +pthardness
         endif
         rewgt_mohdr_calculated=.true.
         iterm_last_mohdr=iterm
         do i=1,nexternal
            do j=0,3
               p_last_mohdr(j,i)=p(j,i)
            enddo
         enddo
         f_r=f_r*rewgt_mohdr
         f_MC_H =f_MC_H *rewgt_mohdr
         f_s_MC_H =f_s_MC_H *rewgt_izero
         f_c_MC_H =f_c_MC_H *rewgt_izero
         f_sc_MC_H=f_sc_MC_H*rewgt_izero
         nFxFx_ren_scales_mohdr=nFxFx_ren_scales
         do i=0,nexternal
            FxFx_ren_scales_mohdr(i)=FxFx_ren_scales(i)
         enddo
         do i=1,2
            FxFx_fac_scale_mohdr(i)=FxFx_fac_scale(i)
         enddo
         call cpu_time(tAfter)
         tFxFx=tFxFx+(tAfter-tBefore)
         return
      elseif (iterm.eq.-1 .or. iterm.eq.-2) then
c Restore scales for the n-body FxFx terms
         nFxFx_ren_scales=nFxFx_ren_scales_izero
         do i=0,nexternal
            FxFx_ren_scales(i)=FxFx_ren_scales_izero(i)
         enddo
         do i=1,2
            FxFx_fac_scale(i)=FxFx_fac_scale_izero(i)
         enddo
      elseif (iterm.eq.-3) then
c Restore scales for the n+1-body FxFx terms
         nFxFx_ren_scales=nFxFx_ren_scales_mohdr
         do i=0,nexternal
            FxFx_ren_scales(i)=FxFx_ren_scales_mohdr(i)
         enddo
         do i=1,2
            FxFx_fac_scale(i)=FxFx_fac_scale_mohdr(i)
         enddo
      else
         write (*,*) 'ERROR: unknown iterm in set_FxFx_scale',iterm
         stop 1
      endif
      call cpu_time(tAfter)
      tFxFx=tFxFx+(tAfter-tBefore)
      return
      end
      
      
      subroutine compute_prefactors_nbody(vegas_wgt)
c Compute all the relevant prefactors for the Born and the soft-virtual,
c i.e. all the nbody contributions. Also initialises the plots and
c bpower.
      use extra_weights
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'timing_variables.inc'
      double precision pi,vegas_wgt
      integer i,j
      logical firsttime
      data firsttime /.true./
      parameter (pi=3.1415926535897932385d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision  xinorm_ev
      common /cxinormev/xinorm_ev
      double precision  xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     $                         fkssymmetryfactorDeg
      integer                  ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                  fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      logical pineappl
      common /for_pineappl/ pineappl
      logical needrndec
      parameter (needrndec=.false.)
      real*8 ran2
      external ran2
      real*8 rndec(10)
      common/crndec/rndec
      include "pineappl_common.inc" 
      integer orders(nsplitorders)
C
      call cpu_time(tBefore)
c Random numbers to be used in the plotting routine: these numbers will
c not change between events, counter events and n-body contributions.
      if(needrndec)then
         do i=1,10
            rndec(i)=ran2()
         enddo
      endif
      if (firsttime) then
         if (pineappl) then 
         ! PineAPPL stuff
         appl_amp_split_size = amp_split_size
           do j=1,amp_split_size
             call amp_split_pos_to_orders(j, orders)
             appl_qcdpower(j) = orders(qcd_pos)
             appl_qedpower(j) = orders(qed_pos)
           enddo
         endif
c Initialize hiostograms for fixed order runs
         if (fixed_order) call initplot
         firsttime=.false.
      endif
      call set_cms_stuff(0)
c f_* multiplication factors for Born and nbody
      f_b=jac_cnt(0)*xinorm_ev/(min(xiimax_ev,xiBSVcut_used)*shat/(16
     $     *pi**2))*fkssymmetryfactorBorn*vegas_wgt
      f_nb=f_b
      call cpu_time(tAfter)
      tf_nb=tf_nb+(tAfter-tBefore)
      return
      end


      subroutine include_multichannel_enhance(imode)
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'timing_variables.inc'
      double precision xnoborn_cnt,xtot,wgt_c,enhance,enhance_real
     $     ,pas(0:3,nexternal)
      data xnoborn_cnt /0d0/
      integer inoborn_cnt,i,imode
      data inoborn_cnt /0/
      double precision p_born_used(0:3,nexternal-1)
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision p_born_ev(0:3,nexternal-1)
      common/pborn_ev/ p_born_ev
      double precision p_born_coll(0:3,nexternal-1)
      common/pborn_coll/p_born_coll
      double precision p_born_norad(0:3,nexternal-1)
      common/pborn_norad/p_born_norad
      double precision p_ev(0:3,nexternal)
      common/pev/      p_ev
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      integer            this_config
      common/to_mconfigs/this_config
      Double Precision amp2(ngraphs), jamp2(0:ncolor)
      common/to_amps/  amp2,          jamp2
      double precision   diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal),granny_is_res
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      logical use_evpr
      common /to_use_evpr/use_evpr

      call cpu_time(tBefore)

c Compute the multi-channel enhancement factor 'enhance'.
      enhance=1.d0
      if (p_born(0,1).gt.0d0) then
         call sborn(p_born,wgt_c)
      elseif(p_born(0,1).lt.0d0)then
         enhance=0d0
      endif
      if (enhance.eq.0d0)then
         xnoborn_cnt=xnoborn_cnt+1.d0
         if(log10(xnoborn_cnt).gt.inoborn_cnt)then
            write (*,*) 'WARNING: no Born momenta more than 10**',
     $           inoborn_cnt,'times'
            inoborn_cnt=inoborn_cnt+1
         endif
      else
         xtot=0d0
         if (mapconfig(0,0).eq.0) then
            write (*,*) 'Fatal error in compute_prefactor_nbody:'/
     &           /' no Born diagrams ',mapconfig,
     &           '. Check bornfromreal.inc'
            write (*,*) 'Is fks_singular compiled correctly?'
            stop 1
         endif
         do i=1, mapconfig(0,0)
            xtot=xtot+amp2(mapconfig(i,0))
         enddo
         if (xtot.ne.0d0) then
            enhance=amp2(mapconfig(this_config,0))/xtot
            enhance=enhance*diagramsymmetryfactor
         else
            enhance=0d0
         endif
      endif

c In the case there is the special phase-space mapping for resonances,
C or when not doing event projection
c use the Born computed with those as the mapping.
      enhance_real=1.d0
      if ((granny_is_res .or. .not.use_evpr).and. imode.eq.2) then
         if (granny_is_res) p_born_used(:,:) = p_born_ev(:,:) 
         if (.not.use_evpr) p_born_used(:,:) = p_born_norad(:,:) 
         if (p_born_ev(0,1).gt.0d0) then
            calculatedBorn=.false.
            pas(0:3,nexternal)=0d0
            pas(0:3,1:nexternal-1)=p_born_used(0:3,1:nexternal-1)
            call set_alphas(pas)
            call sborn(p_born_used,wgt_c)
            call set_alphas(p_ev)
            calculatedBorn=.false.
         elseif(p_born_used(0,1).lt.0d0)then
            if (enhance.ne.0d0) then 
               enhance_real=enhance
            else
               enhance_real=0d0
            endif
         endif
c Compute the multi-channel enhancement factor 'enhance_real'.
         if (enhance_real.eq.0d0)then
            xnoborn_cnt=xnoborn_cnt+1.d0
            if(log10(xnoborn_cnt).gt.inoborn_cnt)then
               write (*,*) 'WARNING: no Born momenta more than 10**',
     $              inoborn_cnt,'times'
               inoborn_cnt=inoborn_cnt+1
            endif
         else
            xtot=0d0
            if (mapconfig(0,0).eq.0) then
               write (*,*) 'Fatal error in compute_prefactor_n1body,'/
     &              /' no Born diagrams ',mapconfig
     &              ,'. Check bornfromreal.inc'
               write (*,*) 'Is fks_singular compiled correctly?'
               stop 1
            endif
            do i=1, mapconfig(0,0)
               xtot=xtot+amp2(mapconfig(i,0))
            enddo
            if (xtot.ne.0d0) then
               enhance_real=amp2(mapconfig(this_config,0))/xtot
               enhance_real=enhance_real*diagramsymmetryfactor
            else
               enhance_real=0d0
            endif
         endif
      else
         enhance_real=enhance
      endif

      if (imode.eq.1) then
         f_b=      f_b      *enhance
         f_nb=     f_nb     *enhance
      elseif(imode.eq.2) then
         f_r=      f_r      *enhance_real
      elseif(imode.eq.4) then
         f_MC_S=   f_MC_S   *enhance
         f_MC_H=   f_MC_H   *enhance
      elseif(imode.eq.3) then
         f_s=      f_s      *enhance
         f_s_MC_S= f_s_MC_S *enhance
         f_S_MC_H= f_S_MC_H *enhance
         f_c=      f_c      *enhance
         f_c_MC_S= f_c_MC_S *enhance
         f_c_MC_H= f_c_MC_H *enhance
         f_dc=     f_dc     *enhance
         f_sc=     f_sc     *enhance
         f_sc_MC_S=f_sc_MC_S*enhance
         f_sc_MC_H=f_sc_MC_H*enhance
         f_dsc(1)= f_dsc(1) *enhance
         f_dsc(2)= f_dsc(2) *enhance
         f_dsc(3)= f_dsc(3) *enhance
         f_dsc(4)= f_dsc(4) *enhance
         f_pdfsch_d=  f_pdfsch_d  *enhance
         f_pdfsch_p=  f_pdfsch_p  *enhance
         f_pdfsch_l=  f_pdfsch_l  *enhance
      endif
      call cpu_time(tAfter)
      tf_nb=tf_nb+(tAfter-tBefore)

      return
      end
      

      subroutine compute_prefactors_n1body(vegas_wgt,jac_ev)
c Compute all relevant prefactors for the real emission and counter
c terms.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'fks_powers.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      double precision vegas_wgt,prefact,prefact_cnt_ssc,prefact_deg
     $     ,prefact_c,prefact_coll,jac_ev,pi,prefact_cnt_ssc_c
     $     ,prefact_coll_c,prefact_deg_slxi,prefact_deg_sxi,zero
      integer i
      parameter (pi=3.1415926535897932385d0, ZERO=0d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision    xi_i_fks_ev,y_ij_fks_ev
      double precision    p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision  xinorm_ev
      common /cxinormev/xinorm_ev
      double precision  xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision   xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt
      double precision    delta_used
      common /cdelta_used/delta_used
      double precision    xicut_used
      common /cxicut_used/xicut_used
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                  fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      ! prefactors for the PDF scheme
      double precision prefact_pdfsch_d,prefact_pdfsch_p,prefact_pdfsch_l
      double precision f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      common/factor_pdfsch/f_pdfsch_d,f_pdfsch_p,f_pdfsch_l
      double precision pmass(nexternal)
      include 'pmass.inc'
      call cpu_time(tBefore)

c f_* multiplication factors for real-emission, soft counter, ... etc.       
      prefact=xinorm_ev/xi_i_fks_ev/(1-y_ij_fks_ev)
      f_r=prefact*jac_ev*fkssymmetryfactor*vegas_wgt
      f_MC_S=f_r
      f_MC_H=f_r
      if (.not.nocntevents) then
         prefact_cnt_ssc=xinorm_ev/min(xiimax_ev,xiScut_used)*
     $        log(xicut_used/min(xiimax_ev,xiScut_used))/(1
     $        -y_ij_fks_ev)
         f_s=(prefact+prefact_cnt_ssc)*jac_cnt(0)
     $        *fkssymmetryfactor*vegas_wgt
         f_s_MC_S=prefact*jac_cnt(0)
     $        *fkssymmetryfactor*vegas_wgt
         f_s_MC_H=f_s_MC_S

         if (pmass(j_fks).eq.0d0) then
c For the soft-collinear, these should be itwo. But they are always
c equal to ione, so no need to define separate factors.
            prefact_c=xinorm_cnt(1)/xi_i_fks_cnt(1)/(1-y_ij_fks_ev)
            prefact_coll=xinorm_cnt(1)/xi_i_fks_cnt(1)*log(delta_used
     $           /deltaS)/deltaS
            f_c=(prefact_c+prefact_coll)*jac_cnt(1)
     $           *fkssymmetryfactor*vegas_wgt
            f_c_MC_S=prefact_c*jac_cnt(1)
     $           *fkssymmetryfactor*vegas_wgt
            f_c_MC_H=f_c_MC_S

            call set_cms_stuff(1)
            prefact_deg=xinorm_cnt(1)/xi_i_fks_cnt(1)/deltaS
            prefact_cnt_ssc_c=xinorm_cnt(1)/min(xiimax_cnt(1)
     &           ,xiScut_used)*log(xicut_used/min(xiimax_cnt(1)
     &           ,xiScut_used))*1/(1-y_ij_fks_ev)
            prefact_coll_c=xinorm_cnt(1)/min(xiimax_cnt(1),xiScut_used)
     $           *log(xicut_used/min(xiimax_cnt(1),xiScut_used))
     $           *log(delta_used/deltaS)/deltaS
            f_dc=jac_cnt(1)*prefact_deg/(shat/(32*pi**2))
     $           *fkssymmetryfactorDeg*vegas_wgt
            f_sc=(prefact_c+prefact_coll+prefact_cnt_ssc_c
     &           +prefact_coll_c)*jac_cnt(2)
     &           *fkssymmetryfactorDeg*vegas_wgt
            f_sc_MC_S=prefact_c*jac_cnt(2)
     $           *fkssymmetryfactor*vegas_wgt
            f_sc_MC_H=f_sc_MC_S

            call set_cms_stuff(2)
            prefact_deg_sxi=xinorm_cnt(1)/min(xiimax_cnt(1),xiScut_used)
     &           *log(xicut_used/min(xiimax_cnt(1),xiScut_used))*1
     &           /deltaS
            prefact_deg_slxi=xinorm_cnt(1)/min(xiimax_cnt(1)
     &           ,xiScut_used)*( log(xicut_used)**2
     &           -log(min(xiimax_cnt(1),xiScut_used))**2 )*1/(2.d0
     &           *deltaS)
            f_dsc(1)=prefact_deg*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
            f_dsc(2)=prefact_deg_sxi*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
            f_dsc(3)=prefact_deg_slxi*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
            f_dsc(4)=( prefact_deg+prefact_deg_sxi )*jac_cnt(2)/(shat
     &           /(32*pi**2))*fkssymmetryfactorDeg
     &           *vegas_wgt
            ! prefactor for the PDF scheme
            prefact_pdfsch_d=xinorm_cnt(1)/xiScut_used/deltaS
            f_pdfsch_d=prefact_pdfsch_d*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
            prefact_pdfsch_p=xinorm_cnt(1)*dlog(xiScut_used)/xiScut_used/deltaS
            f_pdfsch_p=prefact_pdfsch_p*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
            prefact_pdfsch_l=xinorm_cnt(1)*dlog(xiScut_used)**2/2d0/xiScut_used/deltaS
            f_pdfsch_l=prefact_pdfsch_l*jac_cnt(2)/(shat/(32*pi**2))
     &           *fkssymmetryfactorDeg*vegas_wgt
         else
            f_c=0d0
            f_dc=0d0
            f_sc=0d0
            do i=1,4
               f_dsc(i)=0d0
            enddo
            f_c_MC_S=0d0
            f_c_MC_H=0d0
            f_sc_MC_S=0d0
            f_sc_MC_H=0d0
         endif
      else
         f_s=0d0
         f_c=0d0
         f_dc=0d0
         f_sc=0d0
         do i=1,4
            f_dsc(i)=0d0
         enddo
         f_s_MC_S=0d0
         f_s_MC_H=0d0
         f_c_MC_S=0d0
         f_c_MC_H=0d0
         f_sc_MC_S=0d0
         f_sc_MC_H=0d0
      endif
      call cpu_time(tAfter)
      tf_all=tf_all+(tAfter-tBefore)
      return
      end

      
      subroutine add_wgt(type,orders,wgt1,wgt2,wgt3)
c Adds a contribution to the list in weight_lines. 'type' sets the type
c of the contribution and wgt1..wgt3 are the coefficients multiplying
c the logs. The arguments are:
c     type=1 : real-emission
c     type=2 : Born
c     type=3 : integrated counter terms
c     type=4 : soft counter-term
c     type=5 : collinear counter-term
c     type=6 : soft-collinear counter-term
c     type=7 : O(alphaS) expansion of Sudakov factor for NNLL+NLO
c     type=8 : soft counter-term (with n+1-body kin.)
c     type=9 : collinear counter-term (with n+1-body kin.)
c     type=10: soft-collinear counter-term (with n+1-body kin.)
c     type=11: real-emission (with n-body kin.)
c     type=12: MC subtraction with n-body kin.
c     type=13: MC subtraction with n+1-body kin.
c     type=14: virtual corrections
c     type=15: virt-trick: average born contribution
c     type=20+x: EW sudakov, x=sud_mod
c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]
c
c
c The argument orders specifies what are the squared coupling orders
c factorizing the particular set of weights added here. The position of
c the QCD and QED orders there can be obtained via the parameter qcd_pos
c and qed_pos defined in orders.inc
c This is solely used for now in order to apply a potential user-defined filer.
c
c This subroutine increments the 'icontr' counter: each new call to this
c function makes sure that it's considered a new contribution. For each
c contribution, we save the
c     The type: itype(icontr)
c     The weights: wgt(1,icontr),wgt(2,icontr) and wgt(3,icontr) for
c         wgt1, wgt2 and wgt3, respectively.
c     The Bjorken x's: bjx(1,icontr), bjx(2,icontr)
c     The Ellis-Sexton scale squared used to compute the weight:
c        scales2(1,icontr)
c     The renormalisation scale squared used to compute the weight:
c        scales2(2,icontr)
c     The factorisation scale squared used to compute the weight:
c       scales2(3,icontr)
c     The value of the strong coupling: g_strong(icontr)
c     The FKS configuration: nFKS(icontr)
c     The boost to go from the momenta in the C.o.M. frame to the
c         laboratory frame: y_bst(icontr)      
c     The power of the strong coupling (g_strong) for the current
c       weight: QCDpower(icontr)
c     The momenta: momenta(j,i,icontr). For the Born contribution, the
c        counter-term momenta are used. This is okay for any IR-safe
c        observables.
c     The PDG codes: pdg(i,icontr). Always the ones with length
c        'nexternal' are used, because the momenta are also the 
c        'nexternal' ones. This is okay for IR-safe observables.
c     The PDG codes of the underlying Born process:
c        pdg_uborn(i,icontr). The PDGs of j_fks and i_fks are combined
c        to get the PDG code of the mother. The extra parton is given a
c        PDG=21 (gluon) code.
c     If the contribution belongs to an H-event or S-event:
c        H_event(icontr)
c     The weight of the born or real-emission matrix element
c        corresponding to this contribution: wgt_ME_tree. This weight does
c        include the 'ngluon' correction factor for the Born.
c
c Not set in this subroutine, but included in the weight_lines module
c are the
c     wgts(iwgt,icontr) : weights including scale/PDFs/logs. These are
c        normalised so that they can be used directly to compute cross
c        sections and fill plots. 'iwgt' goes from 1 to the maximum
c        number of weights obtained from scale and PDF reweighting, with
c        the iwgt=1 element being the central value.
c     plot_id(icontr) : =20 for Born, 11 for real-emission and 12 for
c        anything else.
c     plot_wgts(iwgt,icontr) : same as wgts(), but only non-zero for
c        unique contributions and non-unique are added to the unique
c        ones. 'Unique' here is defined that they would be identical in
c        an analysis routine (i.e. same momenta and PDG codes)
c     shower_scale(icontr) : The preferred shower starting scale for
c        this contribution
c     niproc(icontr) : number of combined subprocesses in parton_lum_*.f
c     parton_iproc(iproc,icontr) : value of the PDF for the iproc
c        contribution
c     parton_pdg(nexternal,iproc,icontr) : value of the PDG codes for
c     the iproc contribution
c     ipr(icontr): for separate_flavour_configs: the iproc of current
c        contribution
      use weight_lines
      use extra_weights
      use FKSParams
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'coupl.inc'
      include 'fks_info.inc'
      include 'q_es.inc'
      include 'orders.inc'
      integer type,i,j
      logical foundIt,foundOrders
      double precision wgt1,wgt2,wgt3
      integer orders(nsplitorders)
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision p_ev(0:3,nexternal)
      common/pev/      p_ev
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg
      integer                                      ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      integer need_matching_S(nexternal),need_matching_H(nexternal)
      common /c_need_matching/ need_matching_S,need_matching_H
      integer     fold,ifold_counter
      common /cfl/fold,ifold_counter
      integer ntagph
      double precision resc
      integer get_n_tagged_photons
      double precision get_rescale_alpha_factor
      external get_n_tagged_photons get_rescale_alpha_factor

      if (wgt1.eq.0d0 .and. wgt2.eq.0d0 .and. wgt3.eq.0d0) return
c Check for NaN's and INF's. Simply skip the contribution
      if (wgt1.ne.wgt1) return
      if (wgt2.ne.wgt2) return
      if (wgt3.ne.wgt3) return

C Apply user-defined (in FKS_params.dat) contribution type filters if necessary
      if (VetoedContributionTypes(0).gt.0) then
        do i=1,VetoedContributionTypes(0)
          if (type.eq.VetoedContributionTypes(i)) then
C This contribution was explicitely vetoed out by the user. Skip it.
            return
          endif
        enddo
      endif
      if (SelectedContributionTypes(0).gt.0) then
        foundIt = .False.
        do i=1,SelectedContributionTypes(0)
          if (type.eq.SelectedContributionTypes(i)) then
            foundIt = .True.
            exit
          endif
        enddo
        if (.not.foundIt) then
C This contribution was not part of the user selection. Skip it.
          return
        endif
      endif

C Apply the user-defined coupling-order filter if present
C First the simple QCD and QED filters 
      if (QCD_squared_selected.ne.-1.and.
     &   QCD_squared_selected.ne.orders(qcd_pos)) then
        return
      endif
      if (QED_squared_selected.ne.-1.and.
     &   QED_squared_selected.ne.orders(qed_pos)) then
        return
      endif
C Secondly, the more advanced filter
      if (SelectedCouplingOrders(1,0).gt.0) then
        foundIt = .False.
        do j=1,SelectedCouplingOrders(1,0)
          foundOrders = .True.
          do i=1,nsplitorders
            if (SelectedCouplingOrders(i,j).ne.orders(i)) then
              foundOrders = .False.
              exit
            endif
          enddo
          if (foundOrders) then
            foundIt = .True.
            exit
          endif
        enddo
        if (.not.foundIt) then
          return
        endif
      endif

      icontr=icontr+1
      call weight_lines_allocated(nexternal,icontr,max_wgt,max_iproc)
      itype(icontr)=type

C here we rescale the contributions by the ratio of alpha's in different
C schemes; it is needed when there are tagged photons around
      ntagph = get_n_tagged_photons()
      if (ntagph.eq.0) then
        wgt(1,icontr)=wgt1
        wgt(2,icontr)=wgt2
        wgt(3,icontr)=wgt3
      else if (ntagph.gt.0) then
          resc = get_rescale_alpha_factor(ntagph, orders(qed_pos)) 
          wgt(1,icontr) = wgt1 * resc
          wgt(2,icontr) = wgt2 * resc
          wgt(3,icontr) = wgt3 * resc
      endif

      bjx(1,icontr)=xbk(1)
      bjx(2,icontr)=xbk(2)
      scales2(1,icontr)=QES2
      scales2(2,icontr)=scale**2
      scales2(3,icontr)=q2fact(1)
      g_strong(icontr)=g
      nFKS(icontr)=nFKSprocess
      y_bst(icontr)=ybst_til_tolab
      shower_scale(icontr)=-99d9
      ifold_cnt(icontr)=ifold_counter
      icolour_con(1,1,icontr)=-1
      qcdpower(icontr)=QCD_power
      cpower(icontr)=wgtcpower
      orderstag(icontr)=orders_tag
      amppos(icontr)=amp_pos
      ipr(icontr)=0
      call set_pdg(icontr,nFKSprocess)

c Compensate for the fact that in the Born matrix elements, we use the
c identical particle symmetry factor of the corresponding real emission
c matrix elements
      wgt_ME_tree(1,icontr)=wgt_me_born
      wgt_ME_tree(2,icontr)=wgt_me_real
      do i=1,nexternal
         do j=0,3
            if (p1_cnt(0,1,0).gt.0d0.and.type.ne.5) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,0)
            elseif (p1_cnt(0,1,1).gt.0d0) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,1)
            elseif (p1_cnt(0,1,2).gt.0d0) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,2)
            else
               if (i.lt.fks_i_d(nFKSprocess)) then
                  momenta_m(j,i,1,icontr)=p_born(j,i)
               elseif(i.eq.fks_i_d(nFKSprocess)) then
                  momenta_m(j,i,1,icontr)=0d0
               else
                  momenta_m(j,i,1,icontr)=p_born(j,i-1)
               endif
            endif
            momenta_m(j,i,2,icontr)=p_ev(j,i)
         enddo
      enddo

      if(type.eq.1 .or. type.eq. 8 .or. type.eq.9 .or. type.eq.10 .or.
     &     type.eq.13) then
c real emission and n+1-body kin. contributions to counter terms and MC
c subtr term
         do i=1,nexternal
            do j=0,3
               momenta(j,i,icontr)=momenta_m(j,i,2,icontr)
            enddo
         enddo
         H_event(icontr)=.true.
         need_match(1:nexternal,icontr)=need_matching_H(1:nexternal)
      elseif(type.ge.2 .and. type.le.7 .or. type.eq.11 .or. type.eq.12
     $        .or. type.eq.14 .or. type.eq.15
     $        .or. (type.ge.20 .and. type.le.22)) then
c Born, counter term, soft-virtual, or n-body kin. contributions to real
c and MC subtraction terms.
         do i=1,nexternal
            do j=0,3
               momenta(j,i,icontr)=momenta_m(j,i,1,icontr)
            enddo
         enddo
         H_event(icontr)=.false.
         need_match(1:nexternal,icontr)=need_matching_S(1:nexternal)
      else
         write (*,*) 'ERROR: unknown type in add_wgt',type
         stop 1
      endif
      return
      end


      subroutine include_veto_multiplier
      use weight_lines
      use extra_weights
      implicit none
c Multiply all the weights by the NNLL-NLO jet veto Sudakov factors,
c i.e., the term on the 2nd line of Eq.(20) of arXiv:1412.8408.
      include 'nexternal.inc'
      integer i,j
      if (H1_factor_virt.ne.0d0) then
         call compute_veto_multiplier(H1_factor_virt,1d0,1d0
     &        ,veto_multiplier)
         do i=1,icontr
            do j=1,3
               wgt(j,i)=wgt(j,i)*veto_multiplier
            enddo
         enddo
      else
         veto_multiplier=1d0
      endif
      end
      
      subroutine include_PDF_and_alphas
c Multiply the saved wgt() info by the PDFs, alpha_S and the scale
c dependence and saves the weights in the wgts() array. The weights in
c this array are now correctly normalised to compute the cross section
c or to fill histograms.
      use weight_lines
      use extra_weights
      use mint_module
      use FKSParams
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'genps.inc'
      include 'orders.inc'
      integer orders(nsplitorders)
      integer i,j,k,iamp,icontr_orig
      logical virt_found
      double precision xlum,dlum,pi,mu2_r,mu2_f,mu2_q,rwgt_muR_dep_fac
     $     ,wgt_wo_pdf,conv
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      parameter (conv=389379660d0) ! conversion to picobarns
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      virt_found=.false.
c number of contributions before they are (possibly) increased through a
c call to separate_flavour_config().
      icontr_orig=icontr
      i=0
      do while (i.lt.icontr)
         i=i+1
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
         mu2_r=scales2(2,i)
         mu2_f=scales2(3,i)
         q2fact(1)=mu2_f
         q2fact(2)=mu2_f
c call the PDFs
         xlum = dlum()
c iwgt=1 is the central value (i.e. no scale/PDF reweighting).
         iwgt=1
         call weight_lines_allocated(nexternal,max_contr,iwgt,iproc)
c set_pdg_codes fills the niproc, parton_iproc, parton_pdg and
c parton_pdg_uborn [Do only for the contributions that were already
c available as part of the input -- NOT the ones that are created
c through the call to separate_flavour_config(), since that will
c overwrite the relevant information.]
         if (i.le.icontr_orig) call set_pdg_codes(iproc,pd,nFKSprocess,i)
         if (separate_flavour_configs .and. ipr(i).eq.0) then
            call separate_flavour_config(i) ! this increases icontr
         endif
         if (separate_flavour_configs .and. ipr(i).ne.0) then
            if (nincoming.eq.2) then
               xlum=pd(ipr(i))*conv
            else
               xlum=pd(ipr(i))
            endif
         endif
         wgt_wo_pdf=(wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q) + wgt(3,i)
     &        *log(mu2_f/mu2_q))*g_strong(i)**QCDpower(i)
     &        *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),cpower(i))
         wgts(iwgt,i)=xlum * wgt_wo_pdf
         do j=1,iproc
            parton_iproc(j,i)=parton_iproc(j,i) * wgt_wo_pdf
         enddo
         if (itype(i).eq.14 .and. .not.virt_found) then
            virt_found=.true.
c Special for the soft-virtual needed for the virt-tricks. The
c *_wgt_mint variable should be directly passed to the mint-integrator
c and not be part of the plots nor computation of the cross section.
            if (flavour_bias(2).ne.1) 
     $           call recompute_xlum_for_wgt_mint(i,xlum)
            virt_wgt_mint(0)=virt_wgt_mint(0)*xlum
     &           *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),cpower(i))
            born_wgt_mint(0)=born_wgt_mint(0)*xlum
     &           *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),cpower(i))
            do iamp=1,amp_split_size
               call amp_split_pos_to_orders(iamp, orders)
               QCD_power=orders(qcd_pos)
               wgtcpower=0d0
               if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
               virt_wgt_mint(iamp)=virt_wgt_mint(iamp)*xlum
     &              *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),wgtcpower)
               born_wgt_mint(iamp)=born_wgt_mint(iamp)*xlum
     &              *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),wgtcpower)
            enddo
         endif
      enddo
      call cpu_time(tAfter)
      t_as=t_as+(tAfter-tBefore)
      return
      end

      subroutine recompute_xlum_for_wgt_mint(i,xlum)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      double precision xlum,conv
      parameter (conv=389379660d0) ! conversion to picobarns
      integer i,j,iproc
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      xlum=0d0
      do j=1,iproc
         if (any(abs(parton_pdg_uborn(1:nexternal-1,j
     $        ,i)).eq.Flavour_Bias(1))) then
            if (nincoming.eq.2) then
               xlum=xlum+pd(j)*conv*dble(Flavour_Bias(2))
            else
               xlum=xlum+pd(j)*dble(Flavour_Bias(2))
            endif
         else
            if (nincoming.eq.2) then
               xlum=xlum+pd(j)*conv
            else
               xlum=xlum+pd(j)
            endif
         endif
      enddo
      end
      
      subroutine include_bias_wgt
c Include the weight from the bias_wgt_function to all the contributions
c in icontr. This only changes the weight of the central value (after
c inclusion of alphaS and parton luminosity). Both for 'wgts(1,icontr)'
c as well as the the 'parton_iproc(1:niproc(icontr),icontr)', since
c these are the ones used in MINT as well as for unweighting. Also the
c 'virt_wgt_mint' and 'born_wgt_mint' are updated. Furthermore, to
c include the weight also in the 'wgt' array that contain the
c coefficients for PDF and scale computations. 
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      include 'run.inc'
      integer orders(nsplitorders)
      integer i,j,iamp
      logical virt_found
      double precision bias
      character*7 event_norm
      common /event_normalisation/event_norm
c Set the bias_wgt to 1 in case we do not have to do any biassing
      if (event_norm(1:4).ne.'bias') then
         do i=1,icontr
            bias_wgt(i)=1d0
         enddo
         return
      endif
      virt_found=.false.
c loop over all contributions
      do i=1,icontr
         if (itype(i).eq.1) then
            ! use (n+1)-body momenta for the real emission. Pick the
            ! first IPROC for parton PDGs.
            call bias_weight_function(momenta_m(0,1,2,i),parton_pdg(1,1
     $           ,i),bias)
         else
            ! use n-body momenta for all the other contributions. Pick
            ! the first IPROC for parton PDGs.
            call bias_weight_function(momenta_m(0,1,1,i),parton_pdg(1,1
     $           ,i),bias)
         endif
         bias_wgt(i)=bias
c Update the weights:
         do j=1,niproc(i)
            parton_iproc(j,i)=parton_iproc(j,i)*bias_wgt(i)
            if (Flavour_bias(2).ne.1) then ! non-trivial flavour bias in the run_card.
               if (H_event(i)) then
                  if (any(abs(parton_pdg(1:nexternal,j
     $                 ,i)).eq.Flavour_Bias(1))) parton_iproc(j,i)
     $                 =parton_iproc(j,i)*dble(Flavour_Bias(2))
               else
                  if (any(abs(parton_pdg_uborn(1:nexternal-1,j
     $                 ,i)).eq.Flavour_Bias(1))) parton_iproc(j,i)
     $                 =parton_iproc(j,i)*dble(Flavour_Bias(2))
               endif
            endif
         enddo
         wgts(1,i)=sum(parton_iproc(1:niproc(i),i))
         do j=1,3
            wgt(j,i)=wgt(j,i)*bias_wgt(i)
            ! Do not update the wgt() with the Flavour_Bias here; only
            ! do it once the iproc_picked has been set (i.e., only for
            ! the events that are written out). In practice, we can do
            ! it in the include_inverse_bias_wgt() subroutine.
         enddo
         if (itype(i).eq.14 .and. .not.virt_found) then
            virt_found=.true.
            virt_wgt_mint(0)=virt_wgt_mint(0)*bias_wgt(i)
            born_wgt_mint(0)=born_wgt_mint(0)*bias_wgt(i)
            do iamp=1,amp_split_size
               call amp_split_pos_to_orders(iamp, orders)
               virt_wgt_mint(iamp)=virt_wgt_mint(iamp)*bias_wgt(i)
               born_wgt_mint(iamp)=born_wgt_mint(iamp)*bias_wgt(i)
            enddo
         endif
      enddo
      return
      end

      subroutine include_inverse_bias_wgt(inv_bias)
c Update the inverse of the bias in the event weight. All information in
c the rwgt_lines is NOT updated.
      use weight_lines
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'run.inc'
      integer i,ict,ipro,ii,flavour_bias_consistency
      double precision wgt_num,wgt_denom,inv_bias
      character*7 event_norm
      common /event_normalisation/event_norm
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      logical         Hevents
      common/SHevents/Hevents
      if (event_norm(1:4).ne.'bias') then
         inv_bias=1d0
         return
      endif
      wgt_num=0d0
      wgt_denom=0d0
      flavour_bias_consistency=0
      do i=1,icontr_sum(0,icontr_picked)
         ict=icontr_sum(i,icontr_picked)
         if (bias_wgt(ict).eq.0d0) then
            write (*,*) "ERROR in include_inverse_bias_wgt: "/
     $           /"bias_wgt is equal to zero",ict,bias_wgt
            stop 1
         endif
c for all the rwgt_lines, remove the bias-wgt contribution from the
c weights there. Note that the wgtref (also written in the event file)
c keeps its contribution from the bias_wgt.
         if (.not. Hevents) then
            ipro=eto(etoi(iproc_picked,nFKS(ict)),nFKS(ict))
            do ii=1,iproc_save(nFKS(ict))
               if (eto(ii,nFKS(ict)).ne.ipro) cycle
               wgt_denom=wgt_denom+parton_iproc(ii,ict)
               wgt_num=wgt_num+parton_iproc(ii,ict)/bias_wgt(ict)
               if (Flavour_Bias(2).ne.1) then ! non-trivial Flavour bias. Check consistency of flavour configuration
                  if (any(abs(parton_pdg_uborn(1:nexternal-1,ii
     $                 ,ict)).eq.Flavour_Bias(1))) then
                     if (flavour_bias_consistency .ge. 0) then
                        flavour_bias_consistency=1
                     else
                        write (*,*) 'Inconsistent Flavour Bias #1'
                        stop 1
                     endif
                  else
                     if (flavour_bias_consistency .le. 0) then
                        flavour_bias_consistency=-1
                     else
                        write (*,*) 'Inconsistent Flavour Bias #2'
                        stop 1
                     endif
                  endif
               endif
            enddo
         else
            ipro=iproc_picked
            wgt_denom=wgt_denom+parton_iproc(ipro,ict)
            wgt_num=wgt_num+parton_iproc(ipro,ict)/bias_wgt(ict)
            if (Flavour_Bias(2).ne.1) then ! non-trivial Flavour bias. Check consistency of flavour configuration
               if (any(abs(parton_pdg(1:nexternal,ipro,ict)) .eq.
     $              Flavour_Bias(1))) then
                  if (flavour_bias_consistency .ge. 0) then
                     flavour_bias_consistency=1
                  else
                     write (*,*) 'Inconsistent Flavour Bias #3'
                     stop 1
                  endif
               else
                  if (flavour_bias_consistency .le. 0) then
                     flavour_bias_consistency=-1
                  else
                     write (*,*) 'Inconsistent Flavour Bias #4'
                     stop 1
                  endif
               endif
            endif
         endif
      enddo
      wgtref=unwgt(iproc_picked,icontr_picked)
      if (abs((wgtref-wgt_denom)/(wgtref+wgt_denom)).gt.1d-10) then
         write (*,*) "ERROR in include_inverse_bias_wgt: "/
     $        /"reference weight not equal to recomputed weight",wgtref
     $        ,wgt_denom
         stop 1
      endif
c update the event weight to be written in the file
      inv_bias=wgt_num/wgt_denom
      if (flavour_bias_consistency.eq.1) then
         inv_bias=inv_bias/dble(Flavour_Bias(2))
         do i=1,icontr_sum(0,icontr_picked)
            ict=icontr_sum(i,icontr_picked)
            wgt(1:3,ict)=wgt(1:3,ict)*dble(Flavour_Bias(2))
            bias_wgt(ict)=bias_wgt(ict)*dble(Flavour_Bias(2))
         enddo
      endif
      return
      end
      

      subroutine separate_flavour_config(ict)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
      integer ict,i_add,i,j,k,ict_new,n
      if ((.not.fixed_order).or.nlo_ps .or. niproc(ict).eq.1) then
         return
      endif
      i_add=niproc(ict)-1
      call weight_lines_allocated(nexternal,icontr+i_add,max_wgt
     $     ,max_iproc)
      do i=1,niproc(ict)
         if (i.eq.1) then
            niproc(ict)=1
            ipr(ict)=1
            cycle
         endif
         ict_new=icontr+(i-1)
         ipr(ict_new)=i
         itype(ict_new)=itype(ict)
         do j=1,3
            wgt(j,ict_new)=wgt(j,ict)
            scales2(j,ict_new)=scales2(j,ict)
         enddo
         do j=1,2
            bjx(j,ict_new)=bjx(j,ict)
            wgt_ME_tree(j,ict_new)=wgt_ME_tree(j,ict)
         enddo
         g_strong(ict_new)=g_strong(ict)
         nFKS(ict_new)=nFKS(ict)
         y_bst(ict_new)=y_bst(ict)
         QCDpower(ict_new)=QCDpower(ict)
         cpower(ict_new)=cpower(ict)
         orderstag(ict_new)=orderstag(ict)
         H_event(ict_new)=H_event(ict)
         do k=1,nexternal
            do j=0,3
               momenta(j,k,ict_new)=momenta(j,k,ict)
               do n=1,2
                  momenta_m(j,k,n,ict_new)=momenta_m(j,k,n,ict)
               enddo
            enddo
            pdg(k,ict_new)=parton_pdg(k,i,ict)
            pdg_uborn(k,ict_new)=parton_pdg_uborn(k,i,ict)
            parton_pdg(k,1,ict_new)=parton_pdg(k,i,ict)
            parton_pdg_uborn(k,1,ict_new)=parton_pdg_uborn(k,i,ict)
         enddo
         niproc(ict_new)=1
         parton_iproc(1,ict_new)=parton_iproc(i,ict)
      enddo
      icontr=icontr+i_add
      return
      end

      
      subroutine set_pdg_codes(iproc,pd,iFKS,ict)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'fks_info.inc'
      integer j,k,iproc,ict,iFKS
      double precision  pd(0:maxproc),conv
      parameter (conv=389379660d0) ! conversion to picobarns
      include 'leshouche_decl.inc'
      common/c_leshouche_idup_d/ idup_d
      include 'orders.inc'
c save also the separate contributions to the PDFs and the corresponding
c PDG codes
      niproc(ict)=iproc
      do j=1,iproc
         if (nincoming.eq.2) then
            parton_iproc(j,ict)=pd(j)*conv
         else
c           Keep GeV's for decay processes (no conv. factor needed)
            parton_iproc(j,ict)=pd(j)
         endif
         do k=1,nexternal
            parton_pdg(k,j,ict)=idup_d(iFKS,k,j)
            if (k.lt.fks_j_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k,j)
            elseif(k.eq.fks_j_d(iFKS)) then
               if ( abs(idup_d(iFKS,fks_i_d(iFKS),j)) .eq.
     &              abs(idup_d(iFKS,fks_j_d(iFKS),j)) .and.
     &              abs(pdg(fks_i_d(iFKS),ict)).ne.21 .and.
     &              abs(pdg(fks_i_d(iFKS),ict)).ne.22 ) then
                 ! check if any extra cnt is needed
                 if (extra_cnt_d(iFKS).eq.0) then
                   ! if not, assign photon/gluon depending on split_type
                   if (split_type_d(iFKS,qcd_pos)) then
                     parton_pdg_uborn(k,j,ict)=21
                   elseif (split_type_d(iFKS,qed_pos)) then
                     parton_pdg_uborn(k,j,ict)=22
                   else
                     write (*,*) 'set_pdg_codes ',
     &                'ERROR#1 in PDG assigment for underlying Born'
                     stop 1
                   endif
                 else
                   ! if there are extra cnt's, assign the pdg of the
                   ! mother in the born (according to isplitorder_born_d)
                   if (isplitorder_born_d(iFKS).eq.qcd_pos) then
                     parton_pdg_uborn(k,j,ict)=21
                   else if (isplitorder_born_d(iFKS).eq.qcd_pos) then
                     parton_pdg_uborn(k,j,ict)=22
                   else
                     write (*,*) 'set_pdg_codes ',
     &                'ERROR#2 in PDG assigment for underlying Born'
                     stop 1
                   endif
                 endif
               elseif (abs(idup_d(iFKS,fks_i_d(iFKS),j)).eq.21.or.
     &                     idup_d(iFKS,fks_i_d(iFKS),j).eq.22) then
                 parton_pdg_uborn(k,j,ict)=idup_d(iFKS,fks_j_d(iFKS),j)
               elseif (idup_d(iFKS,fks_j_d(iFKS),j).eq.21.or.
     &                 idup_d(iFKS,fks_j_d(iFKS),j).eq.22) then
                 parton_pdg_uborn(k,j,ict)=-idup_d(iFKS,fks_i_d(iFKS),j)
               else
                 write (*,*) 'set_pdg_codes ',
     &                'ERROR#3 in PDG assigment for underlying Born'
                 stop 1
               endif
            elseif(k.lt.fks_i_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k,j)
            elseif(k.eq.nexternal) then
               if (split_type_d(iFKS,qcd_pos)) then
                  parton_pdg_uborn(k,j,ict)=21 ! give the extra particle a gluon PDG code
               elseif (split_type_d(iFKS,qed_pos)) then
                  parton_pdg_uborn(k,j,ict)=22 ! give the extra particle a photon PDG code
               endif
            elseif(k.ge.fks_i_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k+1,j)
            endif
         enddo
      enddo
      return
      end
      
      
      subroutine reweight_scale
c Use the saved weight_lines info to perform scale reweighting. Extends the
c wgts() array to include the weights.
      use weight_lines
      use extra_weights
      use FKSParams
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'genps.inc'
      integer i,kr,kf,iwgt_save,dd
      double precision xlum(maxscales),dlum,pi,mu2_r(maxscales),c_mu2_r
     $     ,c_mu2_f,mu2_f(maxscales),mu2_q,alphas,g(maxscales)
     $     ,rwgt_muR_dep_fac,conv
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      parameter (conv=389379660d0) ! conversion to picobarns
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c currently we have 'iwgt' weights in the wgts() array.
      iwgt_save=iwgt
c loop over all the contributions in the weight lines module
      do i=1,icontr
         iwgt=iwgt_save
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
c Loop over the dynamical_scale_choices
         do dd=1,dyn_scale(0)
c renormalisation scale variation (requires recomputation of the strong
c coupling)
            call set_mu_central(i,dd,c_mu2_r,c_mu2_f)
            do kr=1,nint(scalevarR(0))
               if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
               mu2_r(kr)=c_mu2_r*scalevarR(kr)**2
               g(kr)=sqrt(4d0*pi*alphas(sqrt(mu2_r(kr))))
            enddo
c factorisation scale variation (require recomputation of the PDFs)
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               mu2_f(kf)=c_mu2_f*scalevarF(kf)**2
               q2fact(1)=mu2_f(kf)
               q2fact(2)=mu2_f(kf)
               xlum(kf) = dlum()
               if (separate_flavour_configs .and. ipr(i).ne.0) then
                  if (nincoming.eq.2) then
                     xlum(kf)=pd(ipr(i))*conv
                  else
                     xlum(kf)=pd(ipr(i))
                  endif
               endif
            enddo
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               do kr=1,nint(scalevarR(0))
                  if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
                  iwgt=iwgt+1   ! increment the iwgt for the wgts() array
                  call weight_lines_allocated(nexternal,max_contr,iwgt
     $                 ,max_iproc)
c add the weights to the array
                  wgts(iwgt,i)=xlum(kf) * (wgt(1,i)+wgt(2,i)
     &                 *log(mu2_r(kr)/mu2_q)+wgt(3,i)*log(mu2_f(kf)
     &                 /mu2_q))*g(kr)**QCDpower(i)
                  wgts(iwgt,i)=wgts(iwgt,i)*rwgt_muR_dep_fac(
     &                 sqrt(mu2_r(kr)),sqrt(scales2(2,i)),cpower(i))
               enddo
            enddo
         enddo
      enddo
      call cpu_time(tAfter)
      tr_s=tr_s+(tAfter-tBefore)
      return
      end

      subroutine reweight_scale_NNLL
c Use the saved weight lines info to perform scale reweighting. Extends the
c wgts() array to include the weights. Special for the NNLL+NLO jet-veto
c computations (ickkw.eq.-1).
      use weight_lines
      use extra_weights
      use FKSParams
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'genps.inc'
      integer i,ks,kh,iwgt_save
      double precision xlum(maxscales),dlum,pi,mu2_r(maxscales)
     &     ,mu2_f(maxscales),mu2_q,alphas,g(maxscales),rwgt_muR_dep_fac
     &     ,veto_multiplier_new(maxscales,maxscales)
     &     ,veto_compensating_factor_new,conv
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      parameter (conv=389379660d0) ! conversion to picobarns
      call cpu_time(tBefore)
      write(*,*) 'FIX NLLL'
      stop 1
      if (icontr.eq.0) return
      if (dyn_scale(0).gt.1) then
         write (*,*) "When doing NNLL+NLO veto, "/
     $        /"can only do one dynamical_scale_choice",dyn_scale(0)
         stop
      endif

c currently we have 'iwgt' weights in the wgts() array.
      iwgt_save=iwgt
c compute the new veto multiplier factor      
      do ks=1,nint(scalevarR(0))
         if ((.not. lscalevar(1)) .and. ks.ne.1) exit
         do kh=1,nint(scalevarF(0))
            if ((.not. lscalevar(1)) .and. kh.ne.1) exit
            if (H1_factor_virt.ne.0d0) then
               call compute_veto_multiplier(H1_factor_virt,scalevarR(ks)
     $              ,scalevarF(kh),veto_multiplier_new(ks,kh))
               veto_multiplier_new(ks,kh)=veto_multiplier_new(ks,kh)
     &              /veto_multiplier
            else
               veto_multiplier_new(ks,kh)=1d0
            endif
         enddo
      enddo
c loop over all the contributions in the weight lines module
      do i=1,icontr
         iwgt=iwgt_save
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
c Hard scale variation
         do kh=1,nint(scalevarF(0))
            if ((.not. lscalevar(1)) .and. kh.ne.1) exit
c soft scale variation
            do ks=1,nint(scalevarR(0))
               if ((.not. lscalevar(1)) .and. ks.ne.1) exit
               mu2_r(ks)=scales2(2,i)*scalevarR(ks)**2
               g(ks)=sqrt(4d0*pi*alphas(sqrt(mu2_r(ks))))
               mu2_f(ks)=scales2(2,i)*scalevarR(ks)**2
               q2fact(1)=mu2_f(ks)
               q2fact(2)=mu2_f(ks)
               xlum(ks) = dlum()
               if (separate_flavour_configs .and. ipr(i).ne.0) then
                  if (nincoming.eq.2) then
                     xlum=pd(ipr(i))*conv
                  else
                     xlum=pd(ipr(i))
                  endif
               endif
               iwgt=iwgt+1      ! increment the iwgt for the wgts() array
               call weight_lines_allocated(nexternal,max_contr,iwgt
     $              ,max_iproc)
c add the weights to the array
               if (itype(i).ne.7) then
                  wgts(iwgt,i)=xlum(ks) * (wgt(1,i)+wgt(2,i)
     &                 *log(mu2_r(ks)/mu2_q)+wgt(3,i)*log(mu2_f(ks)
     &                 /mu2_q))*g(ks)**QCDpower(i)
               else
c special for the itype=7 (i.e, the veto-compensating factor)                  
                  call compute_veto_compensating_factor(H1_factor_virt
     &                 ,born_wgt_veto,scalevarR(ks),scalevarF(kh)
     &                 ,veto_compensating_factor_new)
                  wgts(iwgt,i)=xlum(ks) * wgt(1,i)*g(ks)**QCDpower(i)
     &                 /veto_compensating_factor
     &                 *veto_compensating_factor_new
               endif
               wgts(iwgt,i)=wgts(iwgt,i)*rwgt_muR_dep_fac(
     &              sqrt(mu2_r(ks)),sqrt(scales2(2,i)),cpower(i))
               wgts(iwgt,i)=wgts(iwgt,i)*veto_multiplier_new(ks,kh)
            enddo
         enddo
      enddo
      call cpu_time(tAfter)
      tr_s=tr_s+(tAfter-tBefore)
      return
      end

      subroutine reweight_pdf
c Use the saved weight_lines info to perform PDF reweighting. Extends the
c wgts() array to include the weights.
      use weight_lines
      use extra_weights
      use FKSParams
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      include 'genps.inc'
      integer n,izero,i,nn
      parameter (izero=0)
      double precision xlum,dlum,pi,mu2_r,mu2_f,mu2_q,rwgt_muR_dep_fac,g
     &     ,alphas,conv
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      parameter (conv=389379660d0) ! conversion to picobarns
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      do nn=1,lhaPDFid(0)
c Use as external loop the one over the PDF sets and as internal the one
c over the icontr. This reduces the number of calls to InitPDF and
c allows for better caching of the PDFs
         do n=0,nmemPDF(nn)
            iwgt=iwgt+1
            call weight_lines_allocated(nexternal,max_contr,iwgt
     $           ,max_iproc)
            call InitPDFm(nn,n)
            do i=1,icontr
               nFKSprocess=nFKS(i)
               xbk(1) = bjx(1,i)
               xbk(2) = bjx(2,i)
               mu2_q=scales2(1,i)
               mu2_r=scales2(2,i)
               mu2_f=scales2(3,i)
               q2fact(1)=mu2_f
               q2fact(2)=mu2_f
c Compute the luminosity
               xlum = dlum()
               if (separate_flavour_configs .and. ipr(i).ne.0) then
                  if (nincoming.eq.2) then
                     xlum=pd(ipr(i))*conv
                  else
                     xlum=pd(ipr(i))
                  endif
               endif
c Recompute the strong coupling: alpha_s in the PDF might change
               g=sqrt(4d0*pi*alphas(sqrt(mu2_r)))
c add the weights to the array
               wgts(iwgt,i)=xlum * (wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q)
     &              +wgt(3,i)*log(mu2_f/mu2_q))*g**QCDpower(i)
               wgts(iwgt,i)=wgts(iwgt,i)*
     &              rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r),cpower(i))
            enddo
         enddo
      enddo
      call InitPDFm(1,0)
      call cpu_time(tAfter)
      tr_pdf=tr_pdf+(tAfter-tBefore)
      return
      end

      subroutine fill_pineappl_weights(vegas_wgt)
c Fills the FineAPPL weights of pineappl_common.inc. This subroutine assumes
c that there is an unique PS configuration: at most one Born, one real
c and one set of counter terms. Among other things, this means that one
c must do MC over FKS directories.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'pineappl_common.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      integer orders(nsplitorders)
      integer i,j 
      double precision final_state_rescaling,vegas_wgt
      integer              flavour_map(fks_configs)
      common/c_flavour_map/flavour_map
      integer iproc_save(fks_configs),eto(maxproc,fks_configs),
     &     etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      integer lo_qcd_to_amp_pos, nlo_qcd_to_amp_pos
      integer pos
      do i=1,4
        do j=1,amp_split_size
          appl_w0(i,j)=0d0
          appl_wR(i,j)=0d0
          appl_wF(i,j)=0d0
          appl_wB(i,j)=0d0
        enddo
        appl_x1(i)=0d0
        appl_x2(i)=0d0
        appl_QES2(i)=0d0
        appl_muR2(i)=0d0
        appl_muF2(i)=0d0
        appl_flavmap(i)=0
      enddo
      appl_event_weight = 0d0
      appl_vegaswgt = vegas_wgt
      if (icontr.eq.0) return
      do i=1,icontr
         appl_event_weight=appl_event_weight+wgts(1,i)/vegas_wgt
         final_state_rescaling = dble(iproc_save(nFKS(i))) /
     &        dble(appl_nproc(flavour_map(nFKS(i))))
         if (itype(i).eq.2) then
            pos = lo_qcd_to_amp_pos(qcdpower(i))
         else
            pos = nlo_qcd_to_amp_pos(qcdpower(i))
         endif
         ! consistency check
         if (appl_qcdpower(pos).ne.qcdpower(i)) then
           write(*,*) 'ERROR in fill_pineappl_weights, QCDpower',
     %        appl_qcdpower(pos), qcdpower(i)  
           stop 1
         endif

         if (itype(i).eq.1) then
c     real
            appl_w0(1,pos)=appl_w0(1,pos)+wgt(1,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_x1(1)=bjx(1,i)
            appl_x2(1)=bjx(2,i)
            appl_flavmap(1) = flavour_map(nFKS(i))
            appl_QES2(1)=scales2(1,i)
            appl_muR2(1)=scales2(2,i)
            appl_muF2(1)=scales2(3,i)
         elseif (itype(i).eq.2) then
c     born
            appl_wB(2,pos)=appl_wB(2,pos)+wgt(1,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_x1(2)=bjx(1,i)
            appl_x2(2)=bjx(2,i)
            appl_flavmap(2) = flavour_map(nFKS(i))
            appl_QES2(2)=scales2(1,i)
            appl_muR2(2)=scales2(2,i)
            appl_muF2(2)=scales2(3,i)
         elseif (itype(i).eq.3 .or. itype(i).eq.4 .or. itype(i).eq.14
     &           .or. itype(i).eq.15)then
c     virtual, soft-virtual or soft-counter
            appl_w0(2,pos)=appl_w0(2,pos)+wgt(1,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_wR(2,pos)=appl_wR(2,pos)+wgt(2,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_wF(2,pos)=appl_wF(2,pos)+wgt(3,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_x1(2)=bjx(1,i)
            appl_x2(2)=bjx(2,i)
            appl_flavmap(2) = flavour_map(nFKS(i))
            appl_QES2(2)=scales2(1,i)
            appl_muR2(2)=scales2(2,i)
            appl_muF2(2)=scales2(3,i)
         elseif (itype(i).eq.5) then
c     collinear counter            
            appl_w0(3,pos)=appl_w0(3,pos)+wgt(1,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_wF(3,pos)=appl_wF(3,pos)+wgt(3,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_x1(3)=bjx(1,i)
            appl_x2(3)=bjx(2,i)
            appl_flavmap(3) = flavour_map(nFKS(i))
            appl_QES2(3)=scales2(1,i)
            appl_muR2(3)=scales2(2,i)
            appl_muF2(3)=scales2(3,i)
         elseif (itype(i).eq.6) then
c     soft-collinear counter            
            appl_w0(4,pos)=appl_w0(4,pos)+wgt(1,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_wF(4,pos)=appl_wF(4,pos)+wgt(3,i)/bias_wgt(i)*
     &                     final_state_rescaling
            appl_x1(4)=bjx(1,i)
            appl_x2(4)=bjx(2,i)
            appl_flavmap(4) = flavour_map(nFKS(i))
            appl_QES2(4)=scales2(1,i)
            appl_muR2(4)=scales2(2,i)
            appl_muF2(4)=scales2(3,i)
         else
            write(*,*) 'ERROR in fill_applgrid_weights', itype(i)
            stop 1
         endif
      enddo
      return
      end
      
      
      subroutine get_wgt_nbody(sig)
c Sums all the central weights that contribution to the nbody cross
c section
      use weight_lines
      implicit none
      include 'nexternal.inc'
      double precision sig
      integer i
      sig=0d0
      if (icontr.eq.0) return
      do i=1,icontr
         if (itype(i).eq.2 .or. itype(i).eq.3 .or. itype(i).eq.14 .or.
     &        itype(i).eq.7 .or. itype(i).eq.15) then
            sig=sig+wgts(1,i)
         endif
      enddo
      return
      end

      subroutine get_wgt_no_nbody(sig)
c Sums all the central weights that contribution to the cross section
c excluding the nbody contributions.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      double precision sig
      integer i
      sig=0d0
      if (icontr.eq.0) return
      do i=1,icontr
         if (itype(i).ne.2 .and. itype(i).ne.3 .and. itype(i).ne.14
     &        .and. itype(i).ne.7 .and. itype(i).ne.15.and.itype(i).lt.20) then
             ! MZ <20 is needed to exclude the ew sudakov 
            sig=sig+wgts(1,i)
         endif
      enddo
      return
      end

      subroutine fill_plots
c Calls the analysis routine (which fill plots) for all the
c contributions in the weight_lines module. Instead of really calling
c it for all, it first checks if weights can be summed (i.e. they have
c the same PDG codes and the same momenta) before calling the analysis
c to greatly reduce the calls to the analysis routines.
      use weight_lines
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'timing_variables.inc'
      include 'fks_info.inc'
      integer i,ii,j,max_weight
      logical momenta_equal,momenta_equal_uborn,pdg_equal
      external momenta_equal,momenta_equal_uborn,pdg_equal
      double precision,allocatable :: www(:)
      ! stuff for plotting the different splitorders
      integer orders_tag_plot
      common /corderstagplot/ orders_tag_plot
      integer amp_pos_plot
      common /campposplot/ amp_pos_plot
      save max_weight
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c fill the plots_wgts. Check if we can sum weights together before
c calling the analysis routines. This is the case if the PDG codes and
c the momenta are identical.
      do i=1,icontr
         do j=1,iwgt
            plot_wgts(j,i)=0d0
         enddo
         ! The following if lines have been changed with respect to the
         ! usual (with just 3 plot ids: 20, 11 and 12):
         !  The kinematics of soft and collinear counterterms may
         !  be different, for those processes without soft singularities
         !  from initial(final)-state configurations when the 
         !  final(initial) confs are integrated (e.g. a a > e+ e-)
         !  This gives no problem for normal histogramming (and in
         !  fact plot_id 11 13 and 14 are merged into ibody=2 in
         !  outfun, but it gives troubles e.g. with applgrid/pineappl. 
         !  Note that the separation between soft and soft-virtual
         !  may not be needed in reality
         if (itype(i).eq.2) then
            plot_id(i)=20 ! Born
         elseif(itype(i).eq.1) then
            plot_id(i)=11 ! real-emission
         elseif(itype(i).eq.5) then
            plot_id(i)=13 ! collinear counter term
         elseif(itype(i).eq.6) then
            plot_id(i)=14 ! soft collinear counter term
         elseif(itype(i).ge.20.and.itype(i).le.22) then
             plot_id(i)=100+itype(i)-20    ! EW sudakov (100-102)
         else
            plot_id(i)=12 ! soft-virtual and soft counter term
         endif
c Loop over all previous icontr. If the plot_id, PDGs and momenta are
c equal to a previous icountr, add the current weight to the plot_wgts
c of that contribution and exit the do-loop. This loop extends to 'i',
c so if the current weight cannot be summed to a previous one, the ii=i
c contribution makes sure that it is added as a new element.
         do ii=1,i
            if (orderstag(ii).ne.orderstag(i)) cycle
            if (plot_id(ii).ne.plot_id(i)) cycle
            if (plot_id(i).ne.11) then
               if (.not.pdg_equal(pdg_uborn(1,ii),pdg_uborn(1,i))) cycle
            else
               if (.not.pdg_equal(pdg(1,ii),pdg(1,i))) cycle
            endif
            if (plot_id(i).ne.11) then
               if (.not.momenta_equal_uborn(momenta(0,1,ii),momenta(0,1
     $              ,i),fks_j_d(nFKS(ii)),fks_i_d(nFKS(ii))
     $                 ,fks_j_d(nFKS(i)) ,fks_i_d(nFKS(i)))) cycle
            else
               if (.not.momenta_equal(momenta(0,1,ii),momenta(0,1,i)))
     $              cycle
            endif
            do j=1,iwgt
               plot_wgts(j,ii)=plot_wgts(j,ii)+wgts(j,i)
            enddo
            exit
         enddo
      enddo
      do i=1,icontr
         if (plot_wgts(1,i).ne.0d0) then
            if (.not.allocated(www)) then
               allocate(www(iwgt))
               max_weight=iwgt
            elseif(iwgt.ne.max_weight) then
               write (*,*) 'Error in fill_plots (fks_singular.f): '/
     $              /'number of weights should not vary between PS '/
     $              /'points',iwgt,max_weight
               stop
            endif
            do j=1,iwgt
               www(j)=plot_wgts(j,i)/bias_wgt(i)
            enddo
c call the analysis/histogramming routines
            orders_tag_plot=orderstag(i)
            amp_pos_plot=amppos(i)
            call outfun(momenta(0,1,i),y_bst(i),www,pdg(1,i),plot_id(i))
         endif
      enddo
      call cpu_time(tAfter)
      t_plot=t_plot+(tAfter-tBefore)
      return
      end

      subroutine fill_mint_function(f)
c Fills the function that is returned to the MINT integrator
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer i,iamp,ithree,isix
      double precision f(nintegrals),sigint
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      sigint=0d0
      do i=1,icontr
         sigint=sigint+wgts(1,i)
      enddo
      f(1)=abs(sigint)
      f(2)=sigint
      f(4)=virtual_over_born    ! not used for anything
      do iamp=0,amp_split_size
         if (iamp.eq.0) then
            f(3)=0d0
            f(6)=0d0
            f(5)=0d0
            do i=1,amp_split_size
               f(3)=f(3)+virt_wgt_mint(i)
               f(6)=f(6)+born_wgt_mint(i)
            enddo
            f(5)=abs(f(3))!v3.5.4, this fixes a wrong behaviour
         else
            ithree=2*iamp+5
            isix=2*iamp+6
            f(ithree)=virt_wgt_mint(iamp)
            f(isix)=born_wgt_mint(iamp)
         endif
      enddo
      return
      end
      
      subroutine set_colour_connections(iFKS,ifold_counter)
c If the 'complete_xmcsubt' subroutine has been called, update the
c icolour_con() information with the colour flow picked in that
c subroutine. Do this for all contributions that have the FKS
c configuration equal to iFKS and fold equal to ifold_counter, i.e., the
c FKS config and fold for which 'complete_xmcsubt' has been called. In
c case this subroutine was not called, simply set the (first element of)
c icolour_con() information for that contribution equal to -1 (i.e., some
c bogus value). We can check if complete_xmcsubt has been called by
c checking if the first element of icolup_s is positive.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      integer i,iFKS,ifold_counter,ii,jj
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common /colour_connections/ icolup_s,icolup_h
      do i=1,icontr
         if (ifold_cnt(i).ne.ifold_counter) cycle
         if (nFKS(i).ne.iFKS) cycle
         if (H_event(i)) then
            if (icolup_s(1,1).ge.0) then
               icolour_con(1:2,1:nexternal,i)=icolup_h(1:2,1:nexternal)
            else
               icolour_con(1,1,i)=-1
            endif
         else
            if (icolup_s(1,1).ge.0) then
               do ii=1,nexternal-1
                  do jj=1,2
                     icolour_con(jj,ii,i)=icolup_s(jj,ii)
                  enddo
               enddo
               do jj=1,2
                  icolour_con(jj,nexternal,i)=-1
               enddo
            else
               icolour_con(1,1,i)=-1
            endif
         endif
      enddo
      return
      end

      subroutine include_shape_in_shower_scale(p,iFKS,ifold_counter)
c Includes the shape function from the MC counter terms in the shower
c starting scale. This function needs to be called (at least) once per
c FKS configuration that is included in the current PS point.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'nFKSconfigs.inc'
      integer i,j,k,iFKS,izero,mohdr
      double precision p(0:3,nexternal)
      double precision     SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      double precision     SCALUP_a(fks_configs*2,nexternal,nexternal)
      common /cshowerscale_a/SCALUP_a
      parameter (izero=0,mohdr=-100)
      integer ifold_counter
c     Compute the shower starting scale including the shape function
      call set_cms_stuff(izero)
      call set_shower_scale(p,iFKS*2-1,.false.)
      call set_cms_stuff(mohdr)
      call set_shower_scale(p,iFKS*2,.true.)
c loop over all the contributions and update the relevant ones
c (i.e. when nFKS(i)=iFKS and ifold_cnt(i)=ifold_counter)
      do i=1,icontr
         if (ifold_cnt(i).ne.ifold_counter) cycle
         if (nFKS(i).ne.iFKS) cycle
         if (H_event(i)) then
c H-event contribution
            shower_scale(i)=SCALUP(iFKS*2)
            do j=1,nexternal
               do k=1,nexternal
                  shower_scale_a(i,j,k)=SCALUP_a(iFKS*2,j,k)
               enddo
            enddo
         else
c S-event contribution
            shower_scale(i)=SCALUP(iFKS*2-1)
            do j=1,nexternal
               do k=1,nexternal
                  shower_scale_a(i,j,k)=SCALUP_a(iFKS*2-1,j,k)
               enddo
            enddo
         endif
      enddo
      return
      end


      subroutine sum_identical_contributions
c Sum contributions that would lead to an identical event before taking
c the ABS value. In particular this means adding the real emission with
c the MC counter terms for the H-events FKS configuration by FKS
c configuration, while for the S-events also contributions from the
c various FKS configurations can be summed together.
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'timing_variables.inc'
      integer i,j,ii,jj,i_soft
      logical momenta_equal,pdg_equal,equal,found_S,colour_con_equal
      external momenta_equal,pdg_equal,colour_con_equal
      integer iproc_save(fks_configs),eto(maxproc,fks_configs),
     &     etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c Find the contribution to sum all the S-event ones. This should be one
c that has a soft singularity. We set it to 'i_soft'.
      i_soft=0
      found_S=.false.
      do i=1,icontr
         if (H_event(i)) then
            cycle
         else
            found_S=.true.
         endif
         if (abs(pdg_type_d(nFKS(i),fks_i_d(nFKS(i)))).eq.21
     &     .or.abs(pdg_type_d(nFKS(i),fks_i_d(nFKS(i)))).eq.22) then
            i_soft=i
            exit
         endif
      enddo
      if (found_S .and. i_soft.eq.0) then
         write (*,*) 'ERROR: S-event contribution found, '/
     $        /'but no FKS configuration with soft singularity'
         do j=1,icontr
            write (*,*) j,H_event(j),itype(j)
         enddo
         stop 1
      endif
c Main loop over contributions. For H-events we have to check explicitly
c to which contribution we can sum the current contribution (if any),
c while for the S-events we can sum it to the 'i_soft' one.
      do i=1,icontr
         do j=1,niproc(i)
            unwgt(j,i)=0d0
         enddo
      enddo
      icontr_sum(0,1:icontr)=0
      do i=1,icontr
         if (H_event(i)) then
            do ii=1,i
               if (.not.H_event(ii)) cycle
c H-event. If PDG codes, shower starting scale and momenta are equal, we
c can sum them before taking ABS value.
               if (niproc(ii).ne.niproc(i)) cycle
               if (shower_scale(ii).ne.shower_scale(i)) cycle
               equal=.true.
               do j=1,niproc(ii)
                  if (.not.pdg_equal(parton_pdg(1,j,ii),
     &                               parton_pdg(1,j,i))) then
                     equal=.false.
                     exit
                  endif
               enddo
               if (.not. equal) cycle
               if (.not. momenta_equal(momenta(0,1,ii),
     &                                 momenta(0,1,i))) cycle
c     Identical contributions found: sum the contribution "i" to "ii"
               icontr_sum(0,ii)=icontr_sum(0,ii)+1
               icontr_sum(icontr_sum(0,ii),ii)=i
               do j=1,niproc(ii)
                  unwgt(j,ii)=unwgt(j,ii)+parton_iproc(j,i)
               enddo
               if (.not. colour_con_equal(nexternal,icolour_con(1,1,ii)
     $              ,icolour_con(1,1,i))) then
                  write (*,*) 'ERROR in sum_identical_contributions: '/
     $                 /'colour connections in identical H-event '/
     $                 /'contributions should be equal'
                  write (*,*) 'ii: ',icolour_con(1,1:nexternal,ii)
                  write (*,*) '    ',icolour_con(2,1:nexternal,ii)
                  write (*,*) 'i:  ',icolour_con(1,1:nexternal,i)
                  write (*,*) '    ',icolour_con(2,1:nexternal,i)
                  stop 1
               endif
               exit
            enddo
         else
c S-event: we can sum everything to 'i_soft': all the contributions to
c the S-events can be summed together. Ignore the shower_scale: this
c will be updated later
            icontr_sum(0,i_soft)=icontr_sum(0,i_soft)+1
            icontr_sum(icontr_sum(0,i_soft),i_soft)=i
            do j=1,niproc(i_soft)
               do jj=1,iproc_save(nFKS(i))
                  if (eto(jj,nFKS(i)).eq.j) then
c When computing upper bounding envelope (imode.eq.1) do not include the
c virtual corrections. Exception: when computing only the virtual, do
c include it here!
                     if (itype(i).eq.14 .and. imode.eq.1 .and. .not.
     $                    only_virt) exit
                     unwgt(j,i_soft)=unwgt(j,i_soft)+parton_iproc(jj,i)
                  endif
               enddo
            enddo
         endif
      enddo
      call cpu_time(tAfter)
      t_isum=t_isum+(tAfter-tBefore)
      return
      end

      
      subroutine update_shower_scale_Sevents(ifold_counter,ifold_picked)
c When contributions from various FKS configrations are summed together
c for the S-events (see the sum_identical_contributions subroutine), we
c need to update the shower starting scale (because it is not
c necessarily the same for all of these summed FKS configurations and/or
c folds).
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      integer ifold_counter,i,j,k,ifold_picked,icolour(2,nexternal),jj
     $     ,ii
      double precision showerscale
      double precision showerscale_a(nexternal,nexternal)
      logical improved_scale_choice
      parameter (improved_scale_choice=.true.)
      if (icontr.eq.0) return
      if (.not. improved_scale_choice) then
         if (MCatNLO_delta) then
            write (*,*) 'Error in update_shower_scale_Sevents: Not '/
     $           /'correctly updated with icolour info'
            stop 1
         endif
         call update_shower_scale_Sevents_v1(ifold_counter,showerscale
     $        ,showerscale_a,ifold_picked)
      else
         call update_shower_scale_Sevents_v2(ifold_counter,showerscale
     $        ,showerscale_a,icolour,ifold_picked)
      endif
c Overwrite the shower scale for the S-events
      do i=1,icontr
         if (H_event(i)) cycle
         if (icontr_sum(0,i).ne.0)then
            shower_scale(i)= showerscale
            if (mcatnlo_delta) then
               do j=1,nexternal
                  do k=1,nexternal
                     shower_scale_a(i,j,k)= showerscale_a(j,k)
                  enddo
               enddo
               icolour_con(1:2,1:nexternal,i)=icolour(1:2,1:nexternal)
            endif
         endif
      enddo
      return
      end

      subroutine update_shower_scale_Sevents_v1(ifold_counter
     $     ,showerscale,showerscale_a,ifold_picked)
c Original way of assigning shower starting scales. This is for backward
c compatibility. It picks a fold randomly, based on the weight of the
c fold to the sum over all folds. Within a fold, take the weighted
c average of shower scales for the FKS configurations.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      integer i,j,k,l,ict,ifl,ifold_counter,iFKS,ifold_picked
      double precision tmp_wgt(fks_configs,ifold_counter),ran2,target
     $     ,tmp_scale(fks_configs,ifold_counter),showerscale,ifold_accum
     $     ,tmp_scale_a(fks_configs,ifold_counter,nexternal,nexternal)
     $     ,temp_wgt(ifold_counter),shsctemp(ifold_counter),temp_wgt_sum
     $     ,shsctemp_a(ifold_counter,nexternal,nexternal)
     $     ,showerscale_a(nexternal,nexternal)
      external ran2
c
      do ifl=1,ifold_counter
         do iFKS=1,fks_configs
            tmp_wgt(iFKS,ifl)=0d0
            tmp_scale(iFKS,ifl)=-1d0
         enddo
      enddo
c sum the weights that contribute to a single FKS configuration for each
c fold.
      do i=1,icontr
         if (H_event(i)) cycle
         if (icontr_sum(0,i).eq.0) cycle
         do j=1,icontr_sum(0,i)
            ict=icontr_sum(j,i)
            ifl=ifold_cnt(ict)
            tmp_wgt(nFKS(ict),ifl)=tmp_wgt(nFKS(ict),ifl)+
     $           wgts(1,(ifl-1)+i)
            if (tmp_scale(nFKS(ict),ifl).eq.-1d0) then
               tmp_scale(nFKS(ict),ifl)=shower_scale(ict)
c check that all the shower starting scales are identical for all the
c contribution to a given FKS configuration and fold.
            elseif(abs((tmp_scale(nFKS(ict),ifl)-shower_scale(ict))
     $              /(tmp_scale(nFKS(ict),ifl)+shower_scale(ict)))
     $              .gt. 1d-6 ) then
               write (*,*) 'ERROR in update_shower_scale_Sevents #1'
     $              ,tmp_scale(nFKS(ict),ifl),shower_scale(ict)
               stop 1
            endif
         enddo
      enddo
c Compute the weighted average of the shower scale for each fold. Weight
c is given by the ABS cross section to given FKS configuration.
      do ifl=1,ifold_counter
         shsctemp(ifl)=0d0
         temp_wgt(ifl)=0d0
      enddo
      temp_wgt_sum=0d0
      do ifl=1,ifold_counter
         do iFKS=1,fks_configs
            temp_wgt(ifl)=temp_wgt(ifl)+abs(tmp_wgt(iFKS,ifl))
            shsctemp(ifl)=shsctemp(ifl)+abs(tmp_wgt(iFKS,ifl))
     $           *tmp_scale(iFKS,ifl)
         enddo
         temp_wgt_sum=temp_wgt_sum+temp_wgt(ifl)
      enddo
      if (temp_wgt_sum.eq.0d0) return
c Randomly pick one of the folds
      target=temp_wgt_sum*ran2()
      ifold_accum=0d0
      do ifl=1,ifold_counter
         ifold_accum=ifold_accum+temp_wgt(ifl)
         if (ifold_accum.gt.target) exit
      enddo
      if (ifl.lt.1 .or. ifl.gt.ifold_counter) then
         write (*,*) 'ERROR in update_shower_starting scale #1',ifl
     $        ,ifold_counter,target,ifold_accum,temp_wgt_sum
         stop 1
      endif
c     Shower scale is weighted average within the fold
      showerscale=shsctemp(ifl)/temp_wgt(ifl)
      ifold_picked=ifl
      return
      end


      subroutine update_shower_scale_Sevents_v2(ifold_counter
     $     ,showerscale,showerscale_a,icolour,ifold_picked)
c Improved way of assigning shower starting scales. It picks a fold
c randomly, based on the weight of the fold to the sum over all
c folds. Within a fold, pick an FKS configuration randomly, weighted by
c its contribution without including the born (and nbody_noborn)
c contributions. (If there are only born (and nbody_noborn)
c contributions to the picked fold, use the weights of those instead).
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      integer i,j,k,l,ict,ifl,ifold_counter,iFKS,ifold_picked,icolour(2
     $     ,nexternal),ii,jj
      double precision wgt_fold_fks(fks_configs,ifold_counter),ran2
     $     ,target,tmp_scale(fks_configs,ifold_counter),showerscale
     $     ,tmp_scale_a(fks_configs,ifold_counter,nexternal,nexternal)
     $     ,wgt_fold_fks_born(fks_configs,ifold_counter)
     $     ,wgt_fold(ifold_counter),wgt_sum,wgt_accum
     $     ,showerscale_a(nexternal,nexternal)
      external ran2
c
      do ifl=1,ifold_counter
         do iFKS=1,fks_configs
            wgt_fold_fks(iFKS,ifl)=0d0
            wgt_fold_fks_born(iFKS,ifl)=0d0
            tmp_scale(iFKS,ifl)=-1d0
            do j=1,nexternal
               do k=1,nexternal
                  tmp_scale_a(iFKS,ifl,j,k)=-1d0
               enddo
            enddo
         enddo
         wgt_fold(ifl)=0d0
      enddo
c Collect the weights that contribute to a given Fold and FKS
c configuration.
      do i=1,icontr
         if (H_event(i)) cycle
         if (icontr_sum(0,i).eq.0) cycle
         do j=1,icontr_sum(0,i)
            ict=icontr_sum(j,i)
            ifl=ifold_cnt(ict)
            if ( itype(ict).ne.2 .and. itype(ict).ne.3 .and.
     $           itype(ict).ne.7 .and. itype(ict).ne.14 .and.
     $           itype(ict).ne.15) then
                  ! do not include the "born" or "nbody_noborn"
               wgt_fold_fks(nFKS(ict),ifl) = 
     $                 wgt_fold_fks(nFKS(ict),ifl)+wgts(1,ict)
            else
               wgt_fold_fks_born(nFKS(ict),ifl) = 
     $                 wgt_fold_fks_born(nFKS(ict),ifl)+wgts(1,ict)
            endif
            wgt_fold(ifl)=wgt_fold(ifl)+wgts(1,ict)
            if (tmp_scale(nFKS(ict),ifl).eq.-1d0) then
               tmp_scale(nFKS(ict),ifl)=shower_scale(ict)
c check that all the shower starting scales are identical for all the
c contribution to a given FKS configuration and fold.
            elseif(abs((tmp_scale(nFKS(ict),ifl)-shower_scale(ict))
     $              /(tmp_scale(nFKS(ict),ifl)+shower_scale(ict)))
     $              .gt. 1d-6 ) then
               write (*,*) 'ERROR in update_shower_scale_Sevents #2'
     $              ,tmp_scale(nFKS(ict),ifl),shower_scale(ict)
               stop 1
            endif
            do l=1,nexternal
               do k=1,nexternal
                  if (tmp_scale_a(nFKS(ict),ifl,l,k).eq.-1d0) then
                     tmp_scale_a(nFKS(ict),ifl,l,k)=shower_scale_a(ict,l,k)
c check that all the shower starting scales are identical for all the
c contribution to a given FKS configuration and fold.
                  elseif ( abs((tmp_scale_a(nFKS(ict),ifl,l,k)-shower_scale_a(ict,l,k))
     $               /(tmp_scale_a(nFKS(ict),ifl,l,k)+shower_scale_a(ict,l,k)))
     $               .gt. 1d-6 ) then
                     write (*,*) 'ERR 2 in update_shower_scale_Sevents2'
     $               ,tmp_scale_a(nFKS(ict),ifl,l,k),shower_scale_a(ict,l,k)
                     stop 1
                  endif
               enddo
            enddo
         enddo
      enddo
c pick the fold at random, weighted by their relative contributions
      wgt_sum=0d0
      do ifl=1,ifold_counter
         wgt_sum=wgt_sum+abs(wgt_fold(ifl))
      enddo
      if (wgt_sum.le.0d0) return
      target=wgt_sum*ran2()
      wgt_accum=0d0
      do ifl=1,ifold_counter
         wgt_accum=wgt_accum+abs(wgt_fold(ifl))
         if (wgt_accum.gt.target) exit
      enddo
      if (ifl.lt.1 .or. ifl.gt.ifold_counter) then
         write (*,*) 'ERROR in update_shower_starting scale #2',ifl
     $        ,ifold_counter,target,wgt_accum,wgt_sum
         stop 1
      endif
c Now that we have the fold, check within that fold to find the FKS
c configurations and the corresponding shower starting scale. Pick one
c randomly based on the weight for that FKS configuration (in the
c weight, the born and nbody_noborn should not be included since those
c are always assigned to the FKS configuration corresponding to a soft
c singularity. Therefore, including them would bias the chosen scale to
c that configuration.)
      wgt_sum=0d0
      do iFKS=1,fks_configs
         wgt_sum=wgt_sum+abs(wgt_fold_fks(iFKS,ifl))
      enddo
      if (wgt_sum.ne.0d0) then
         target=wgt_sum*ran2()
         wgt_accum=0d0
         do iFKS=1,fks_configs
            wgt_accum=wgt_accum+abs(wgt_fold_fks(iFKS,ifl))
            if (wgt_accum.gt.target) exit
         enddo
         if (iFKS.lt.1 .or. iFKS.gt.fks_configs) then
            write (*,*) 'ERROR in update_shower_starting scale #3',iFKS
     $           ,fks_configs,target,wgt_accum,wgt_sum
            stop 1
         endif
      else
c this fold has only born or nbody no-born contributions. Use those
c instead.
         wgt_sum=0d0
         do iFKS=1,fks_configs
            wgt_sum=wgt_sum+abs(wgt_fold_fks_born(iFKS,ifl))
         enddo
         if (wgt_sum.eq.0d0) return
         target=wgt_sum*ran2()
         wgt_accum=0d0
         do iFKS=1,fks_configs
            wgt_accum=wgt_accum+abs(wgt_fold_fks_born(iFKS,ifl))
            if (wgt_accum.gt.target) exit
         enddo
         if (iFKS.lt.1 .or. iFKS.gt.fks_configs) then
            write (*,*) 'ERROR in update_shower_starting scale #4',iFKS
     $           ,fks_configs,target,wgt_accum,wgt_sum
            stop 1
         endif
      endif
      showerscale=tmp_scale(iFKS,ifl)
      do i=1,icontr
         if (H_event(i)) cycle
         if (iFKS.ne.nFKS(i) .or. ifl.ne.ifold_cnt(i)) cycle
         icolour(1:2,1:nexternal)=icolour_con(1:2,1:nexternal,i)
         exit
      enddo
      do j=1,nexternal
         do k=1,nexternal
            showerscale_a(j,k)=tmp_scale_a(iFKS,ifl,j,k)
         enddo
      enddo
      ifold_picked=ifl
      return
      end


      subroutine fill_mint_function_NLOPS(f,n1body_wgt)
c Fills the function that is returned to the MINT integrator. Depending
c on the imode we should or should not include the virtual corrections.
      use weight_lines
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'orders.inc'
      integer i,j,ict,iamp,ithree,isix
      double precision f(nintegrals),sigint,sigint1,sigint_ABS
     $     ,n1body_wgt,tmp_wgt,max_weight
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      sigint=0d0
      sigint1=0d0
      sigint_ABS=0d0
      n1body_wgt=0d0
      max_weight=0d0
      if (icontr.eq.0) then
         sigint_ABS=0d0
         sigint=0d0
         sigint1=0d0
      else
         do i=1,icontr
            sigint=sigint+wgts(1,i)
            max_weight=max(max_weight,abs(wgts(1,i)))
            if (icontr_sum(0,i).eq.0) cycle
            do j=1,niproc(i)
               sigint_ABS=sigint_ABS+abs(unwgt(j,i))
               sigint1=sigint1+unwgt(j,i) ! for consistency check
               max_weight=max(max_weight,abs(unwgt(j,i)))
            enddo
         enddo
c check the consistency of the results up to machine precision (10^-10 here)
         if (imode.ne.1 .or. only_virt) then
            if (abs((sigint-sigint1)/max_weight).gt.1d-10) then
               write (*,*) 'ERROR: inconsistent integrals #0',sigint
     $              ,sigint1,max_weight,abs((sigint-sigint1)/max_weight)
               do i=1, icontr
                  write (*,*) i,icontr_sum(0,i),niproc(i),wgts(1,i)
     $                 ,H_event(i),itype(i),nFKS(i)
                  if (icontr_sum(0,i).eq.0) cycle
                  do j=1,niproc(i)
                     write (*,*) j,unwgt(j,i)
                  enddo
               enddo
               stop 1
            endif
         else
            sigint1=sigint1+virt_wgt_mint(0)
            if (abs((sigint-sigint1)/max_weight).gt.1d-10) then
               write (*,*) 'ERROR: inconsistent integrals #1',sigint
     $              ,sigint1,max_weight,abs((sigint-sigint1)/max_weight)
     $              ,virt_wgt_mint
               do i=1, icontr
                  write (*,*) i,icontr_sum(0,i),niproc(i),wgts(1,i)
                  if (icontr_sum(0,i).eq.0) cycle
                  do j=1,niproc(i)
                     write (*,*) j,unwgt(j,i)
                  enddo
               enddo
               stop 1
            endif
         endif
c n1body_wgt is used for the importance sampling over FKS directories
         do i=1,icontr
            if (icontr_sum(0,i).eq.0) cycle
            tmp_wgt=0d0
            do j=1,icontr_sum(0,i)
               ict=icontr_sum(j,i)
               if ( itype(ict).ne.2  .and. itype(ict).ne.3 .and.
     $              itype(ict).ne.14 .and. itype(ict).ne.15)
     $                              tmp_wgt=tmp_wgt+wgts(1,ict)
            enddo
            n1body_wgt=n1body_wgt+abs(tmp_wgt)
         enddo
      endif
      f(1)=sigint_ABS
      f(2)=sigint
      f(4)=virtual_over_born
      do iamp=0,amp_split_size
         if (iamp.eq.0) then
            f(3)=0d0
            f(6)=0d0
            f(5)=0d0
            do i=1,amp_split_size
               f(3)=f(3)+virt_wgt_mint(i)
               f(6)=f(6)+born_wgt_mint(i)
            enddo
            f(5)=abs(f(3))!v3.5.4, this fixes a wrong behaviour
         else
            ithree=2*iamp+5
            isix=2*iamp+6
            f(ithree)=virt_wgt_mint(iamp)
            f(isix)=born_wgt_mint(iamp)
         endif
      enddo
      return
      end


      subroutine pick_unweight_contr(iFKS_picked,ifold_picked)
c Randomly pick (weighted by the ABS values) the contribution to a given
c PS point that should be written in the event file.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'timing_variables.inc'
      integer i,j,k,l,iFKS_picked,ict,ifold_picked,jj,ii
      double precision tot_sum,rnd,ran2,current,target
      external ran2
      integer           i_process_addwrite
      common/c_addwrite/i_process_addwrite
      logical         Hevents
      common/SHevents/Hevents
      logical                 dummy
      double precision evtsgn
      common /c_unwgt/ evtsgn,dummy
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision     SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      double precision     SCALUP_a(fks_configs*2,nexternal,nexternal)
      common /cshowerscale_a/SCALUP_a
      integer colour_connections(2,nexternal)
      common /colour_connections_to_write/ colour_connections
      double precision tmp_wgt(fks_configs),sum_granny_wgt
      logical write_granny(fks_configs)
      integer which_is_granny(fks_configs)
      common/write_granny_resonance/which_is_granny,write_granny
      integer need_matching(nexternal)
      common /c_need_matching_to_write/ need_matching

      call cpu_time(tBefore)
      if (icontr.eq.0) return
      tot_sum=0d0
      do i=1,icontr
         do j=1,niproc(i)
            tot_sum=tot_sum+abs(unwgt(j,i))
         enddo
      enddo
      rnd=ran2()
      current=0d0
      target=rnd*tot_sum
      i=1
      j=0
      do while (current.lt.target)
         j=j+1
         if (mod(j,niproc(i)+1).eq.0) then
            j=1
            i=i+1
         endif
         current=current+abs(unwgt(j,i))
      enddo
c found the contribution that should be written:
      icontr_picked=i
      iproc_picked=j
      if (H_event(icontr_picked)) then
         Hevents=.true.
         i_process_addwrite=iproc_picked
         iFKS_picked=nFKS(icontr_picked)
         ifold_picked=ifold_cnt(icontr_picked)
         SCALUP(iFKS_picked*2)=shower_scale(icontr_picked)
         do k=1,nexternal
            do l=1,nexternal
               SCALUP_a(iFKS_picked*2,k,l)=shower_scale_a(icontr_picked
     $              ,k,l)
            enddo
         enddo
         colour_connections(1:2,1:nexternal)=icolour_con(1:2
     $        ,1:nexternal,icontr_picked)
      else
         Hevents=.false.
         i_process_addwrite=etoi(iproc_picked,nFKS(icontr_picked))
c For S-events, ifold_picked is already set in
c update_shower_scale_Sevents(). Note that for S-events, the fold chosen
c doesn't really matter.
c$$$         ifold_picked=ifold_cnt(icontr_picked)
         do k=1,icontr_sum(0,icontr_picked)
            ict=icontr_sum(k,icontr_picked)
            !MZif (particle_type_d(nFKS(ict),fks_i_d(nFKS(ict))).eq.8) then
            if (need_color_links_d(nFKS(ict)).or.need_charge_links_d(nFKS(ict))) then
               iFKS_picked=nFKS(ict)
               exit
            endif
            if (k.eq.icontr_sum(0,icontr_picked)) then
               write (*,*) 'ERROR: no configuration with i_fks a gluon'
               stop 1
            endif
         enddo
         SCALUP(iFKS_picked*2-1)=shower_scale(icontr_picked)
         do k=1,nexternal
            do l=1,nexternal
               SCALUP_a(iFKS_picked*2-1,k,l)
     $              =shower_scale_a(icontr_picked,k,l)
            enddo
         enddo
         colour_connections(1:2,1:nexternal)=icolour_con(1:2,1:nexternal
     $        ,icontr_picked)
c Determine if we need to write the granny (based only on the special
c mapping in genps_fks) randomly, weighted by the seperate contributions
c that are summed together in a single S-event.
         do i=1,fks_configs
            tmp_wgt(i)=0d0
         enddo
c fill tmp_wgt with the sum of weights per FKS configuration
         do k=1,icontr_sum(0,icontr_picked)
            ict=icontr_sum(k,icontr_picked)
            tmp_wgt(nFKS(ict))=tmp_wgt(nFKS(ict))+wgts(1,ict)
         enddo
c Randomly select an FKS configuration
         sum_granny_wgt=0d0
         do i=1,fks_configs
            sum_granny_wgt=sum_granny_wgt+abs(tmp_wgt(i))
         enddo
         target=ran2()*sum_granny_wgt
         current=0d0
         i=0
         do while (current.le.target)
            i=i+1
            current=current+abs(tmp_wgt(i))
         enddo
c Overwrite the granny information of the FKS configuration with the
c soft singularity with the FKS configuration randomly chosen.
         write_granny(iFKS_picked)=write_granny(i)
         which_is_granny(iFKS_picked)=which_is_granny(i)
      endif
      evtsgn=sign(1d0,unwgt(iproc_picked,icontr_picked))
      need_matching(1:nexternal)=need_match(1:nexternal,icontr_picked)
      call cpu_time(tAfter)
      t_p_unw=t_p_unw+(tAfter-tBefore)
      return
      end


      subroutine fill_rwgt_lines
c Fills the lines, n_ctr_str, to be written in an event file with the
c (internal) information to perform scale and/or PDF reweighting. All
c information is available in each line to do the reweighting, apart
c from the momenta: these are put in the momenta_str() array, and a
c label in each of the n_ctr_str refers to a corresponding set of
c momenta in the momenta_str() array.
      use weight_lines
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer k,i,ii,j,jj,ict,ipro,momenta_conf(2)
      logical momenta_equal,found
      double precision conv
      double precision,allocatable :: temp3(:,:,:)
      character(len=1024),allocatable :: ctemp(:)
      external momenta_equal
      character*512 procid,str_temp
      parameter (conv=389379660d0) ! conversion to picobarns
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      logical         Hevents
      common/SHevents/Hevents
      if (.not.allocated(momenta_str)) allocate(momenta_str(0:3
     $     ,max_mext,max_mom_str))
      wgtref=unwgt(iproc_picked,icontr_picked)
      n_ctr_found=0
      n_mom_conf=0
c Loop over all the contributions in the picked contribution (the latter
c is chosen in the pick_unweight_contr() subroutine)
      do i=1,icontr_sum(0,icontr_picked)
         ict=icontr_sum(i,icontr_picked)
c Check if the current set of momenta are already available in the
c momenta_str array. If not, add it.
         found=.false.
         do k=1,2
            do j=1,n_mom_conf
               if (momenta_m(0,1,k,ict).le.0d0) then
                  momenta_conf(k)=0
                  cycle
               endif
               if (momenta_equal(momenta_str(0,1,j),
     &                           momenta_m(0,1,k,ict))) then
                  momenta_conf(k)=j
                  found=.true.
                  exit
               endif
            enddo
            if (.not. found) then
               n_mom_conf=n_mom_conf+1
               if (n_mom_conf.gt.max_mom_str .or. nexternal.gt.max_mext)
     $              then
                  allocate(temp3(0:3,max(nexternal,max_mext)
     $                              ,max(n_mom_conf,max_mom_str)))
                  temp3(0:3,1:min(nexternal,max_mext)
     $                     ,1:min(max_mom_str,n_mom_conf))=momenta_str
                  call move_alloc(temp3,momenta_str)
                  max_mom_str=max(n_mom_conf,max_mom_str)
                  max_mext=max(nexternal,max_mext)
               endif
               do ii=1,nexternal
                  do jj=0,3
                     momenta_str(jj,ii,n_mom_conf)=
     &                                      momenta_m(jj,ii,k,ict)
                  enddo
               enddo
               momenta_conf(k)=n_mom_conf
            endif
         enddo
         if (.not. Hevents) then

             ! MZ write also orderstag!!
c For S-events, be careful to take all the IPROC that contribute to the
c iproc_picked:
            ipro=eto(etoi(iproc_picked,nFKS(ict)),nFKS(ict))
            do ii=1,iproc_save(nFKS(ict))
               if (eto(ii,nFKS(ict)).ne.ipro) cycle
               n_ctr_found=n_ctr_found+1

               if (.not.allocated(n_ctr_str))
     $              allocate(n_ctr_str(max_n_ctr))
               if (n_ctr_found.gt.max_n_ctr) then
                  allocate(ctemp(n_ctr_found))
                  ctemp(1:max_n_ctr)=n_ctr_str
                  call move_alloc(ctemp,n_ctr_str)
                  max_n_ctr=n_ctr_found
               endif
               
               if (nincoming.eq.2) then
                  write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &                 (wgt(j,ict)*conv,j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &                 nexternal
               else
                  write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &                 (wgt(j,ict),j=1,3),(wgt_me_tree(j,ict),j=1,2), 
     &                 nexternal
               endif

               procid=''
               do j=1,nexternal
                  write (str_temp,*) parton_pdg(j,ii,ict)
                  procid=trim(adjustl(procid))//' '
     &                 //trim(adjustl(str_temp))
               enddo
               n_ctr_str(n_ctr_found) =
     &              trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &              //trim(adjustl(procid))

               write (str_temp,30)
     &              orderstag(ict),
     &              QCDpower(ict),
     &              (bjx(j,ict),j=1,2),
     &              (scales2(j,ict),j=1,3),
     &              g_strong(ict),
     &              (momenta_conf(j),j=1,2),
     &              itype(ict),
     &              nFKS(ict),
     &              fks_i_d(nFKS(ict)),
     &              fks_j_d(nFKS(ict)),
     &              parton_pdg_uborn(fks_j_d(nFKS(ict)),ii,ict),
     &              parton_iproc(ii,ict),
     &              bias_wgt(ict)
               n_ctr_str(n_ctr_found) =
     &              trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &              //trim(adjustl(str_temp))
            enddo
         else
c H-event
            ipro=iproc_picked
            n_ctr_found=n_ctr_found+1

            if (.not.allocated(n_ctr_str))
     $           allocate(n_ctr_str(max_n_ctr))
            if (n_ctr_found.gt.max_n_ctr) then
               allocate(ctemp(n_ctr_found))
               ctemp(1:max_n_ctr)=n_ctr_str
               call move_alloc(ctemp,n_ctr_str)
               max_n_ctr=n_ctr_found
            endif

            if (nincoming.eq.2) then
               write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &              (wgt(j,ict)*conv,j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &              nexternal
            else
               write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &              (wgt(j,ict),j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &              nexternal
            endif

            procid=''
            do j=1,nexternal
               write (str_temp,*) parton_pdg(j,ipro,ict)
               procid=trim(adjustl(procid))//' '
     &              //trim(adjustl(str_temp))
            enddo
            n_ctr_str(n_ctr_found) =
     &           trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &           //trim(adjustl(procid))

            write (str_temp,30)
     &           orderstag(ict),
     &           QCDpower(ict),
     &           (bjx(j,ict),j=1,2),
     &           (scales2(j,ict),j=1,3),
     &           g_strong(ict),
     &           (momenta_conf(j),j=1,2),
     &           itype(ict),
     &           nFKS(ict),
     &           fks_i_d(nFKS(ict)),
     &           fks_j_d(nFKS(ict)),
     &           parton_pdg_uborn(fks_j_d(nFKS(ict)),ipro,ict),
     &           parton_iproc(ipro,ict),
     &           bias_wgt(ict)
            n_ctr_str(n_ctr_found) =
     &           trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &           //trim(adjustl(str_temp))


         endif
      enddo
      return
 30   format(i15,1x,i2,6(1x,d14.8),6(1x,i2),1x,i8,1x,d18.12,1x,d18.12)
      end
      
      
      subroutine rotate_invar(pin,pout,cth,sth,cphi,sphi)
c Given the four momentum pin, returns the four momentum pout (in the same
c Lorentz frame) by performing a three-rotation of an angle theta 
c (cos(theta)=cth) around the y axis, followed by a three-rotation of an
c angle phi (cos(phi)=cphi) along the z axis. The components of pin
c and pout are given along these axes
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth-q2*sphi+q3*cphi*sth
      pout(2)=q1*sphi*cth+q2*cphi+q3*sphi*sth
      pout(3)=-q1*sth+q3*cth 
      pout(0)=pin(0)
      return
      end


      subroutine trp_rotate_invar(pin,pout,cth,sth,cphi,sphi)
c This subroutine performs a rotation in the three-space using a rotation
c matrix that is the transpose of that used in rotate_invar(). Thus, if
c called with the *same* angles, trp_rotate_invar() acting on the output
c of rotate_invar() will return the input of rotate_invar()
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth+q2*sphi*cth-q3*sth
      pout(2)=-q1*sphi+q2*cphi 
      pout(3)=q1*cphi*sth+q2*sphi*sth+q3*cth
      pout(0)=pin(0)
      return
      end


      subroutine getaziangles(p,cphi,sphi)
      implicit none
      real*8 p(0:3),cphi,sphi
      real*8 xlength,cth,sth
      double precision rho
      external rho
c
      xlength=rho(p)
      if(xlength.ne.0.d0)then
        cth=p(3)/xlength
        sth=sqrt(1-cth**2)
        if(sth.ne.0.d0)then
          cphi=p(1)/(xlength*sth)
          sphi=p(2)/(xlength*sth)
        else
          cphi=1.d0
          sphi=0.d0
        endif
      else
        cphi=1.d0
        sphi=0.d0
      endif
      return
      end

      subroutine phspncheck_born(ecm,xmass,xmom,pass)
c Checks four-momentum conservation.
c WARNING: works only in the partonic c.m. frame
      implicit none
      include 'nexternal.inc'
      real*8 ecm,xmass(nexternal-1),xmom(0:3,nexternal-1)
      real*8 tiny,xm,xlen4,xsum(0:3),xsuma(0:3),xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      integer jflag,npart,i,j,jj
      logical pass
c
      pass=.true.
      jflag=0
      npart=nexternal-1
      do i=0,3
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=nincoming+1,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(i.eq.0)xsum(i)=xsum(i)-ecm
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        pass=.false.
        return
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(abs(xm-xmass(j))/ptmp(0).gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          pass=.false.
          return
        endif
      enddo
      return
      end


      subroutine phspncheck_nocms(npart,ecm,xmass,xmom,pass)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer npart,maxmom
      include "genps.inc"
      include "nexternal.inc"
      real*8 ecm,xmass(-max_branch:max_particles),
     # xmom(0:3,nexternal)
      real*8 tiny,vtiny,xm,xlen4,den,ecmtmp,xsum(0:3),xsuma(0:3),
     # xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-6)
      integer jflag,i,j,jj
      logical pass
      double precision dot
      external dot
c
      pass=.true.
      jflag=0
      do i=0,3
        if (nincoming.eq.2) then
          xsum(i)=-xmom(i,1)-xmom(i,2)
          xsuma(i)=abs(xmom(i,1))+abs(xmom(i,2))
        elseif(nincoming.eq.1) then
          xsum(i)=-xmom(i,1)
          xsuma(i)=abs(xmom(i,1))
        endif
        do j=nincoming+1,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved [nocms]'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(i2,1x,4(d14.8,1x))') j,(xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(a3,1x,4(d14.8,1x))') 'sum',(xsum(jj),jj=0,3)
        write(*,'(a3,1x,4(d14.8,1x))') 'rat',(xrat(jj),jj=0,3)
        pass=.false.
        return
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(ptmp(0).ge.1.d0)then
          den=ptmp(0)
        else
          den=1.d0
        endif
        if(abs(xm-xmass(j))/den.gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation [nocms]'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          pass=.false.
          return
        endif
      enddo
c
      if (nincoming.eq.2) then
         ecmtmp=sqrt(2d0*dot(xmom(0,1),xmom(0,2)))
      elseif (nincoming.eq.1) then
         ecmtmp=xmom(0,1)
      endif
      if(abs(ecm-ecmtmp).gt.vtiny)then
        write(*,*)'Inconsistent shat [nocms]'
        write(*,*)'ecm given=   ',ecm
        write(*,*)'ecm computed=',ecmtmp
        write(*,'(4(d14.8,1x))') (xmom(jj,1),jj=0,3)
        write(*,'(4(d14.8,1x))') (xmom(jj,2),jj=0,3)
        pass=.false.
        return
      endif

      return
      end


      function xlen4(v)
      implicit none
      real*8 xlen4,tmp,v(0:3)
c
      tmp=v(0)**2-v(1)**2-v(2)**2-v(3)**2
      xlen4=sign(1.d0,tmp)*sqrt(abs(tmp))
      return
      end


      subroutine set_shower_scale(p,iFKS,Hevents)
      implicit none
      include "nexternal.inc"
      include "madfks_mcatnlo.inc"
      include 'run.inc'
      integer iFKS,i,j
      logical Hevents,ldum
      double precision xi_i_fks_ev,y_ij_fks_ev,p(0:3,nexternal),ddum(6)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision emsca,scalemin,scalemax,emsca_bare
      logical emscasharp
      common/cemsca/emsca,emsca_bare,emscasharp,scalemin,scalemax
      double precision emsca_a(nexternal,nexternal)
     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
     $     ,nexternal)
      logical emscasharp_a(nexternal,nexternal)
      double precision scalemin_a(nexternal,nexternal)
     $     ,scalemax_a(nexternal,nexternal),emscwgt_a(nexternal
     $     ,nexternal)
      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2,emscasharp_a
     $     ,scalemin_a,scalemax_a,emscwgt_a
      character*4 abrv
      common/to_abrv/abrv
      include 'nFKSconfigs.inc'
      double precision SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      double precision SCALUP_a(fks_configs*2,nexternal,nexternal)
      common /cshowerscale_a/SCALUP_a
      double precision shower_S_scale(fks_configs*2)
     &     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     &     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness
      integer izero,mohdr
      parameter (izero=0,mohdr=-100)
      double precision xm12
      integer ileg
      common/cscaleminmax/xm12,ileg
      double precision SCALUP_tmp_S(nexternal,nexternal)
      double precision SCALUP_tmp_H(nexternal,nexternal)
      common/c_SCALUP_tmp/SCALUP_tmp_S,SCALUP_tmp_H
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
!     common block used to make the (scalar) reference scale partner
!     dependent in case of delta
      integer cur_part
      common /to_ref_scale/cur_part
c
      if (.not.dampMCsubt) then
         write (*,*) 'ERROR: dampMCsubt should be true'
         stop 1
      endif
! 1st bit (1) of MCcntcalled: call to set_shower_scale_noshape for S-event (or Born) done
! 2nd bit (2) of MCcntcalled: call to set_shower_scale_noshape for H-event done
! 3rd bit (4) of MCcntcalled: call to xmcsubt done (and is_pt_hard == false)
! 4th bit (8) of MCcntcalled: call to complete_xmcsubt done
! 5th bit (16) of MCcntcalled: is_pt_hard==True      

! initialize to zero
      SCALUP(iFKS)=0d0
      SCALUP_a(iFKS,1:nexternal,1:nexternal)=0d0
      
      if (MCcntcalled.eq.15) then
         ! both complete_xmcsubt and MC counter have been called. Set
         ! scales based on emsca and scalemax (for SCALUP), except for
         ! scale-array for H-events, which is based on what's returned
         ! by pythia.
         if (.not. Hevents) then
            SCALUP(iFKS)=min(emsca,scalemax,shower_S_scale(iFKS))
            do i=1,nexternal
               do j=1,nexternal
                  if(j.eq.i)cycle
                  SCALUP_a(iFKS,i,j)=SCALUP_tmp_S(i,j)
               enddo
            enddo
         else
            SCALUP(iFKS)=min(scalemax,max(shower_H_scale(iFKS),
     $                   ref_H_scale(iFKS)-min(emsca,scalemax)))
            do i=1,nexternal
               do j=1,nexternal
                  if(j.eq.i)cycle
                  SCALUP_a(iFKS,i,j)=SCALUP_tmp_H(i,j)
               enddo
            enddo
         endif
      elseif (MCcntcalled.eq.3 .or. MCcntcalled.eq.1) then
         ! Either we're doing Born, or MC counter terms have not been
         ! called.
         ! If Born: just use shower_S_scale (i.e., shower scale without
         ! shape)
         ! Else: set emsca and scaleminmax, and include them in the
         ! shower scale (if momenta are defined, else don't use shape)
         if (abrv.ne.'born' .and. ickkw.ne.4 .and. p(0,1).ne.-99d0) then
            if (mcatnlo_delta) cur_part=0 ! use s-hat as reference scale.
            call set_cms_stuff(mohdr)
            call assign_emsca(p,xi_i_fks_ev,y_ij_fks_ev)
            if (mcatnlo_delta)
     $           call assign_emsca_array(p,xi_i_fks_ev,y_ij_fks_ev)
            call kinematics_driver(xi_i_fks_ev,y_ij_fks_ev,shat_ev,p
     $           ,ileg,xm12,ddum(1),ddum(2),ddum(3),ddum(4),ddum(5)
     $           ,ddum(6),ldum)
            call assign_scaleminmax(shat_ev,xi_i_fks_ev,scalemin
     $           ,scalemax,ileg,xm12)
            if (mcatnlo_delta)
     $           call assign_scaleminmax_array(shat_ev,xi_i_fks_ev
     $           ,scalemin_a,scalemax_a,ileg,xm12)
            if (.not. Hevents) then
               SCALUP(iFKS)=min(emsca,scalemax,shower_S_scale(iFKS))
               if (mcatnlo_delta) then
                  do i=1,nexternal
                     do j=1,nexternal
                        if(j.eq.i)cycle
                        SCALUP_a(iFKS,i,j)=min(emsca_a(i,j),
     $                       scalemax_a(i,j))
                     enddo
                  enddo
               endif
            else
               SCALUP(iFKS)=min(scalemax,max(shower_H_scale(iFKS),
     $              ref_H_scale(iFKS)-min(emsca,scalemax)))
               if (mcatnlo_delta) then
                  do i=1,nexternal
                     do j=1,nexternal
                        if(j.eq.i)cycle
                        SCALUP_a(iFKS,i,j)=shower_H_scale(iFKS) ! we don't need the shape here
                     enddo
                  enddo
               endif
            endif
         else ! abrv.eq.Born .or. p(0,1).eq.-99
            SCALUP(iFKS)=shower_S_scale(iFKS)
            if (mcatnlo_delta) then
               do i=1,nexternal
                  do j=1,nexternal
                     if(j.eq.i)cycle
                     SCALUP_a(iFKS,i,j)=shower_S_scale(iFKS)
                  enddo
               enddo
            endif
         endif
      elseif (MCcntcalled.eq.19) then
         ! is_pt_hard is true. Use shower_s_scale and shower_h_scale for
         ! S and H events respectively. Hence, no shape function
         ! included.
         if (.not. Hevents) then
            SCALUP(iFKS)=shower_S_scale(iFKS)
            if (mcatnlo_delta) then
               do i=1,nexternal
                  do j=1,nexternal
                     if(j.eq.i)cycle
                     SCALUP_a(iFKS,i,j)=shower_S_scale(iFKS)
                  enddo
               enddo
            endif
         else
            SCALUP(iFKS)=shower_H_scale(iFKS)
            if (mcatnlo_delta) then
               do i=1,nexternal
                  do j=1,nexternal
                     if(j.eq.i)cycle
                     SCALUP_a(iFKS,i,j)=shower_H_scale(iFKS)
                  enddo
               enddo
            endif
         endif
      elseif (MCcntcalled.eq.7) then
         if (mcatnlo_delta) then
            write (*,*) "Incompatible 'MCcntcalled':"
            write (*,*) "When doing MCatNLO-delta, complete_xmcsubt "/
     $           /"should always be called if pt_hard is not true."
            stop 1
         endif
         if (.not. Hevents) then
            SCALUP(iFKS)=min(emsca,scalemax,shower_S_scale(iFKS))
         else
            SCALUP(iFKS)=min(scalemax,max(shower_H_scale(iFKS),
     $                   ref_H_scale(iFKS)-min(emsca,scalemax)))
         endif
         ! Do not need to fill the SCALUP_a() array
      else
         write (*,*) 'ERROR: MCcntcalled assigned wrongly.',MCcntcalled
         stop 1
      endif
c Safety measure
      SCALUP(iFKS)=max(SCALUP(iFKS),scaleMCcut)
      if (mcatnlo_delta) then
         do i=1,nexternal
            do j=1,nexternal
               if(j.eq.i)cycle
               if (SCALUP_a(iFKS,i,j).ne.-1d0) then
                  SCALUP_a(iFKS,i,j)=max(SCALUP_a(iFKS,i,j),scaleMCcut)
               endif
            enddo
         enddo
      endif
      return
      end


      subroutine set_shower_scale_noshape(pp,iFKS)
      implicit none
      integer iFKS,j,i,iSH,nmax
      include "nexternal.inc"
      include "madfks_mcatnlo.inc"
      include 'run.inc'
      include 'nFKSconfigs.inc'
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision shower_S_scale(fks_configs*2)
     &     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     &     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision ptparton,pt,pp(0:3,nexternal),ppp(0:3,nexternal)
      external pt
c jet cluster algorithm
      integer NN,NJET,JET(nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal),rfj,sycut
     $     ,palg,amcatnlo_fastjetdmergemax,di(nexternal)
      external amcatnlo_fastjetdmergemax

      if (btest(iFKS,0)) then
         MCcntcalled=MCcntcalled+1
      else
         MCcntcalled=MCcntcalled+2
      endif
c Initialise
      NN=0
      ppp=0d0
      pQCD=0d0
      pt_hardness=0d0
      do j=1,nexternal
         if (j.gt.nincoming.and.is_a_j(j)) then
            NN=NN+1
            ptparton=pt(pp(0,j))
         endif
      enddo

c Unphysical situation
      if(NN.le.0)then
         write(*,*)'Error in set_shower_scale_noshape:'
         write(*,*)'not enough QCD partons in process ',NN
         stop
c Processes without jets at the Born
      elseif(NN.eq.1)then
         shower_S_scale(iFKS)=sqrtshat_cnt(0)
         shower_H_scale(iFKS)=sqrtshat_ev-ptparton
c$$$         shower_H_scale(iFKS)=sqrtshat_cnt(0)
         ref_H_scale(iFKS)=0d0
c Processes with jets at the Born (iSH = 1 (2) means S (H) events)
      else
         do iSH=1,2
            if(iSH.eq.1)then
               nmax=nexternal-1
               do j=1,nmax
                  do i=0,3
                     ppp(i,j)=p_born(i,j)
                  enddo
               enddo
            elseif(iSH.eq.2)then
               nmax=nexternal
               do j=1,nmax
                  do i=0,3
                     ppp(i,j)=pp(i,j)
                  enddo
               enddo
            else
               write(*,*)'Wrong iSH inset_shower_scale_noshape: ',iSH
               stop
            endif
            if(ppp(0,1).gt.0d0)then
c Put all (light) QCD partons in momentum array for jet clustering.
               NN=0
               do j=nincoming+1,nmax
                  if (is_a_j(j))then
                     NN=NN+1
                     do i=0,3
                        pQCD(i,NN)=ppp(i,j)
                     enddo
                  endif
               enddo
c One MUST use kt, and no lower pt cut. The radius parameter can be changed
               palg=1d0         ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
               sycut=0d0        ! minimum jet pt
               rfj=1d0          ! the radius parameter
               call amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
     &                                            pjet,njet,jet)
               do i=1,NN
                  di(i)=sqrt(amcatnlo_fastjetdmergemax(i-1))
                  if (i.gt.1)then
                     if(di(i).gt.di(i-1))then
                        write(*,*)'Error in set_shower_scale_noshape'
                        write(*,*)NN,i,di(i),di(i-1)
                        stop
                     endif
                  endif
               enddo
               if(iSH.eq.1)shower_S_scale(iFKS)=di(NN)
               if(iSH.eq.2)then
                  ref_H_scale(iFKS)=di(NN-1)
                  pt_hardness=di(NN)
c$$$                  shower_H_scale(iFKS)=ref_H_scale(iFKS)-pt_hardness
                  shower_H_scale(iFKS)=ref_H_scale(iFKS)-pt_hardness/2d0
               endif
            else
               if(iSH.eq.1)shower_S_scale(iFKS)=sqrtshat_cnt(0)
               if(iSH.eq.2)then
                  ref_H_scale(iFKS)=shower_S_scale(iFKS)
                  shower_H_scale(iFKS)=ref_H_scale(iFKS)
               endif
            endif
         enddo
      endif

      return
      end



      subroutine sreal(pp,xi_i_fks,y_ij_fks,wgt)
c Wrapper for the n+1 contribution. Returns the n+1 matrix element
c squared reduced by the FKS damping factor xi**2*(1-y).
c Close to the soft or collinear limits it calls the corresponding
c Born and multiplies with the AP splitting function or eikonal factors.
      implicit none
      include "nexternal.inc"
      include "coupl.inc"

      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks

      double precision shattmp,dot
      integer i,j

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision zero,tiny
      parameter (zero=0d0)
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links

      double precision pmass(nexternal)
      include 'orders.inc'

      include "pmass.inc"

      if (softtest.or.colltest) then
         tiny=1d-12
      else
         tiny=1d-6
      endif

      if(pp(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
        wgt=0.d0
        return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(pp(0,1),pp(0,2))
      else
         shattmp=pp(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      if (1d0-y_ij_fks.lt.tiny)then
         if (pmass(j_fks).eq.zero.and.j_fks.le.nincoming)then
            call sborncol_isr(pp,xi_i_fks,y_ij_fks,wgt)
         elseif (pmass(j_fks).eq.zero.and.j_fks.ge.nincoming+1)then
            call sborncol_fsr(pp,xi_i_fks,y_ij_fks,wgt)
         else
            wgt=0d0
            amp_split(1:amp_split_size) = 0d0
         endif
      elseif (xi_i_fks.lt.tiny)then
         if (need_color_links.or.need_charge_links)then
c has soft singularities
            call sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
         else
            wgt=0d0
            amp_split(1:amp_split_size) = 0d0
         endif
      else
         call smatrix_real(pp,wgt)
         wgt=wgt*xi_i_fks**2*(1d0-y_ij_fks)
         amp_split(1:amp_split_size) = amp_split(1:amp_split_size)*xi_i_fks**2*(1d0-y_ij_fks)
      endif

      return
      end



      subroutine sborncol_fsr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      integer i,j,imother_fks,iord
C ap and Q contain the QCD(1) and QED(2) Altarelli-Parisi kernel
      double precision t,z,ap(2),E_j_fks,E_i_fks,Q(2),cphi_mother,
     # sphi_mother,pi(0:3),pj(0:3),wgt_born
      double complex W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      include 'orders.inc'
      double precision amp_split_local(amp_split_size)
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      common /c_born_cnt/ ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt

      double precision iden_comp
      common /c_iden_comp/iden_comp
C  
      amp_split_local(1:amp_split_size) = 0d0
      
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_fsr"
         wgt=0.d0
         return
      endif

      E_j_fks = p(0,j_fks)
      E_i_fks = p(0,i_fks)
      z = 1d0 - E_i_fks/(E_i_fks+E_j_fks)
      t = z * shat/4d0
      call sborn(p_born,wgt_born)
      if (iextra_cnt.gt.0)
     1    call extra_cnt(p_born, iextra_cnt, ans_extra_cnt)
      call AP_reduced(j_type,i_type,ch_j,ch_i,t,z,ap)
      call Qterms_reduced_timelike(j_type,i_type,ch_j,ch_i,t,z,Q)
      wgt=0d0
      do iord = 1, nsplitorders
         if (.not.split_type(iord) .or. (iord.ne.qed_pos .and.
     $        iord.ne.qcd_pos)) cycle
C check if any extra_cnt is needed
         if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
               call sborn(p_born,wgt_born)
               wgt1(1) = ans_cnt(1,iord)
               wgt1(2) = ans_cnt(2,iord)
            elseif (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt(p_born, iextra_cnt, ans_extra_cnt)
               wgt1(1) = ans_extra_cnt(1,iord)
               wgt1(2) = ans_extra_cnt(2,iord)
            else
               write(*,*) 'ERROR in sborncol_fsr', iord
               stop
            endif
         else
            call sborn(p_born,wgt_born)
            wgt1(1) = ans_cnt(1,iord)
            wgt1(2) = ans_cnt(2,iord)
         endif
         if ((abs(j_type).eq.3 .and.i_type.eq.8) .or.
     #       (dabs(ch_j).ne.0d0 .and.ch_i.eq.0d0)) then
            Q(1)=0d0
            Q(2)=0d0
            wgt1(2)=0d0
         elseif (m_type.eq.8.or.ch_m.eq.0d0) then
c Insert <ij>/[ij] which is not included by sborn()
            if (1d0-y_ij_fks.lt.vtiny)then
               azifact=xij_aor
            else
               do i=0,3
                  pi(i)=p_i_fks_ev(i)
                  pj(i)=p(i,j_fks)
               enddo
               CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
               CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
               CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
               CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
               Wij_angle=(0d0,0d0)
               Wij_recta=(0d0,0d0)
               do i=1,4
                  Wij_angle = Wij_angle + W1(i)*W2(i)
                  Wij_recta = Wij_recta + W3(i)*W4(i)
               enddo
               azifact=Wij_angle/Wij_recta
            endif
c Insert the extra factor due to Madgraph convention for polarization vectors
            imother_fks=min(i_fks,j_fks)
            call getaziangles(p_born(0,imother_fks),
     #                       cphi_mother,sphi_mother)
            wgt1(2) = -(cphi_mother-ximag*sphi_mother)**2 *
     #             wgt1(2) * azifact
            amp_split_cnt(1:amp_split_size,2,iord) = -(cphi_mother-ximag
     $           *sphi_mother)**2 *amp_split_cnt(1:amp_split_size,2
     $           ,iord) * azifact
         else
            write(*,*) 'FATAL ERROR in sborncol_fsr',i_type,j_type,i_fks
     $           ,j_fks
            stop 1
         endif
         if (iord.eq.qcd_pos) then
            wgt=wgt+dble(wgt1(1)*ap(1)+wgt1(2)*Q(1))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt(1:amp_split_size,1,iord)*AP(1)
     $           +amp_split_cnt(1:amp_split_size,2,iord)*Q(1))
         endif
         if (iord.eq.qed_pos) then
            wgt=wgt+dble(wgt1(1)*ap(2)+wgt1(2)*Q(2))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt(1:amp_split_size,1,iord)*AP(2)
     $           +amp_split_cnt(1:amp_split_size,2,iord)*Q(2))
         endif
      enddo
      wgt=wgt*iden_comp
      amp_split(1:amp_split_size) = amp_split_local(1:amp_split_size)
     $     *iden_comp
      return
      end



      subroutine sborncol_isr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born_coll(0:3,nexternal-1)
      common/pborn_coll/p_born_coll

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision p_born_used(0:3,nexternal-1)

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m

      integer i, j, iord
C ap and Q contain the QCD(1) and QED(2) Altarelli-Parisi kernel
      double precision t,z,ap(2),Q(2),cphi_mother,sphi_mother,
     $ pi(0:3),pj(0:3),wgt_born
      double complex W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      include 'orders.inc'
      double precision amp_split_local(amp_split_size)
      double complex amp_split_cnt_local(amp_split_size,2,nsplitorders)
      integer iamp
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      common /c_born_cnt/ ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt

      double precision iden_comp
      common /c_iden_comp/iden_comp

      logical use_evpr
      common /to_use_evpr/use_evpr
C  
      amp_split_local(1:amp_split_size) = 0d0

C in the case of the collinear CT, use p_born_coll
C  (when not doing event projection). 
C For the soft-collinear one, use p_born
      if (xi_i_fks.gt.0d0.and..not.use_evpr) then
          p_born_used(:,:) = p_born_coll(:,:)
      else ! if (xi_i_fks.eq.0d0) then
          p_born_used(:,:) = p_born(:,:)
      endif

      if(p_born_used(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_isr"
         wgt=0.d0
         return
      endif

      z = 1d0 - xi_i_fks
c sreal return {\cal M} of FKS except for the partonic flux 1/(2*s).
c Thus, an extra factor z (implicit in the flux of the reduced Born
c in FKS) has to be inserted here
      t = z*shat/4d0
      call AP_reduced(m_type,i_type,ch_m,ch_i,t,z,ap)
      call Qterms_reduced_spacelike(m_type,i_type,ch_m,ch_i,t,z,Q)
      wgt=0d0
      do iord = 1, nsplitorders
         if (.not.split_type(iord) .or. (iord.ne.qed_pos .and.
     $        iord.ne.qcd_pos)) cycle
C check if any extra_cnt is needed
         if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
            ! this is the contribution from the born ME
               call sborn(p_born_used,wgt_born)
               wgt1(1:2) = ans_cnt(1:2,iord)
            else if (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt(p_born_used, iextra_cnt, ans_extra_cnt)
               wgt1(1:2) = ans_extra_cnt(1:2,iord)
            else
               write(*,*) 'ERROR in sborncol_isr', iord
               stop
            endif
         else
            call sborn(p_born_used,wgt_born)
            wgt1(1:2) = ans_cnt(1:2,iord)
        endif
        amp_split_cnt_local(1:amp_split_size,1,iord)=
     $       amp_split_cnt(1:amp_split_size,1,iord)
        amp_split_cnt_local(1:amp_split_size,2,iord)=
     $       amp_split_cnt(1:amp_split_size,2,iord)
        if (abs(m_type).eq.3.or.ch_m.ne.0d0) then
           Q(1)=0d0
           Q(2)=0d0
           wgt1(2)=dcmplx(0d0,0d0)
           amp_split_cnt_local(1:amp_split_size,2,iord)=dcmplx(0d0,0d0)
        else
c Insert <ij>/[ij] which is not included by sborn()
           if (1d0-y_ij_fks.lt.vtiny)then
              azifact=xij_aor
           else
              do i=0,3
                 pi(i)=p_i_fks_ev(i)
                 pj(i)=p(i,j_fks)
              enddo
              if(j_fks.eq.2 .and. nincoming.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed
                 pi(1)=-pi(1)
                 pi(3)=-pi(3)
                 pj(1)=-pj(1)
                 pj(3)=-pj(3)
              endif
              CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
              CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
              CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
              CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
              Wij_angle=(0d0,0d0)
              Wij_recta=(0d0,0d0)
              do i=1,4
                 Wij_angle = Wij_angle + W1(i)*W2(i)
                 Wij_recta = Wij_recta + W3(i)*W4(i)
              enddo
              azifact=Wij_angle/Wij_recta
           endif
c Insert the extra factor due to Madgraph convention for polarization vectors
           cphi_mother=1.d0
           sphi_mother=0.d0
           wgt1(2) = -(cphi_mother+ximag*sphi_mother)**2 * wgt1(2) *
     $          dconjg(azifact)
           amp_split_cnt_local(1:amp_split_size,2,iord) = -(cphi_mother
     $          +ximag*sphi_mother)**2
     $          *amp_split_cnt_local(1:amp_split_size,2,iord) *
     $          dconjg(azifact)
        endif
        if (iord.eq.qcd_pos) then
            wgt=wgt+dble(wgt1(1)*ap(1)+wgt1(2)*Q(1))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)
     $           *AP(1)+amp_split_cnt_local(1:amp_split_size,2,iord)
     $           *Q(1))
         endif
         if (iord.eq.qed_pos) then
            wgt=wgt+dble(wgt1(1)*ap(2)+wgt1(2)*Q(2))
            amp_split_local(1:amp_split_size) =
     $           amp_split_local(1:amp_split_size)
     $           +dble(amp_split_cnt_local(1:amp_split_size,1,iord)
     $           *AP(2)+amp_split_cnt_local(1:amp_split_size,2,iord)
     $           *Q(2))
         endif
      enddo
      wgt=wgt*iden_comp
      amp_split(1:amp_split_size) = amp_split_local(1:amp_split_size)
     $     *iden_comp
      return
      end


      subroutine xkplus(PDFscheme, col1, col2, ch1, ch2, x, xkk)
c This function returns the quantity K^{(+)}_{ab}(x), relevant for
c the MS --> DIS (or any other) scheme change in the factorization scheme.
C It also includes regular terms, multiplied by (1-x).
c There's NO multiplicative (1-x) factor like in the previous functions.
C the first entry in xkk is for QCD splittings, the second QED
      implicit none
      integer PDFscheme, col1, col2
      double precision ch1, ch2
      double precision x, xkk(2)

      double precision pi, vcf, vtf, vca, xnc
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)

      include "coupl.inc"
c
      if (PDFscheme.eq.0) then
        ! MSbar, all terms are zero
        xkk(:) = 0d0
      else if (PDFscheme.eq.1) then
        ! DIS scheme
        if(col1.eq.8.and.col2.eq.8)then ! gg
          xkk(1)=-2*nf*vtf*(1-x)*(-(x**2+(1-x)**2)*log(x)+8*x*(1-x)-1)
          xkk(2)=0d0
        elseif((abs(col1).eq.3.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).gt.0d0))then ! qq
          xkk(1)=vtf*(1-x)*(-(x**2+(1-x)**2)*log(x)+8*x*(1-x)-1)
          xkk(2)=dble(abs(col1))*ch1**2*(1-x)*(-(x**2+(1-x)**2)*log(x)+8*x*(1-x)-1)
        elseif((col1.eq.8.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).eq.0d0.and.dabs(ch2).gt.0d0))then ! gq
          xkk(1)=-vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
          xkk(2)=-ch2**2*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
        elseif((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg
          xkk(1)=vcf*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
          xkk(2)=ch1**2*(-3.d0/2.d0-(1+x**2)*log(x)+(1-x)*(3+2*x))
        else
          write(6,*)'Error in xkplus: wrong values', col1, col2, ch1, ch2
          stop
        endif
      else if (PDFscheme.eq.2) then
        ! ETA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*(1-x)*(1+x)
          xkk(2)=-ch1**2*(1-x)*(1+x)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.3) then
        ! BETA scheme (lepton)
        xkk(:) = 0d0
      else if (PDFscheme.eq.4) then
        ! MIXED scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*(1-x)*(1+x)
          xkk(2)=-ch1**2*(1-x)*(1+x)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.5) then
        ! nobeta scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*(2-(1-x)*(1+x))
          xkk(2)=ch1**2*(2-(1-x)*(1+x))
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.6) then
        ! DELTA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*(1+x**2)
          xkk(2)=ch1**2*(1+x**2)
        elseif((abs(col1).eq.8.and.col2.eq.3) .or. 
     $         (dabs(ch1).eq.0d0.and.dabs(ch2).gt.0d0))then ! gq / game
          xkk(1)=vcf*(1-x)*(1+(1-x)**2)/x*(2*dlog(x)+1)
          xkk(2)=ch2**2*(1-x)*(1+(1-x)**2)/x*(2*dlog(x)+1)
        else
          xkk(:) = 0d0
        endif
      else
        write(6,*)'Error in xkplus: wrong PDF scheme', PDFscheme
        stop
      endif
      xkk(1) = xkk(1)*g**2
      xkk(2) = xkk(2)*dble(gal(1))**2
      return
      end


      subroutine xklog(PDFscheme, col1, col2, ch1, ch2, x, xkk)
c This function returns the quantity K^{(l)}_{ab}(x), relevant for
c the MS --> DIS (or any other) scheme change in the factorization scheme.
c There's NO multiplicative (1-x) factor like in the previous functions.
C the first entry in xkk is for QCD splittings, the second QED
      implicit none
      integer PDFscheme, col1, col2
      double precision ch1, ch2
      double precision x, xkk(2)

      double precision pi, vcf, vtf, vca, xnc
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)

      include "coupl.inc"
c
      if (PDFscheme.eq.0) then
        ! MSbar, all terms are zero
        xkk(:) = 0d0
      else if (PDFscheme.eq.1) then
        ! DIS scheme
        if(col1.eq.8.and.col2.eq.8)then ! gg
          xkk(1)=-2*nf*vtf*(1-x)*(x**2+(1-x)**2)
          xkk(2)=0d0
        elseif((abs(col1).eq.3.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).gt.0d0))then ! qq
          xkk(1)=vtf*(1-x)*(x**2+(1-x)**2)
          xkk(2)=dble(abs(col1))*ch1**2*(1-x)*(x**2+(1-x)**2)
        elseif((col1.eq.8.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).eq.0d0.and.dabs(ch2).gt.0d0))then ! gq
          xkk(1)=-vcf*(1+x**2)
          xkk(2)=-ch2**2*(1+x**2)
        elseif((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg
          xkk(1)=vcf*(1+x**2)
          xkk(2)=ch1**2*(1+x**2)
        else
          write(6,*)'Error in xklog: wrong values', col1, col2, ch1, ch2
          stop
        endif
      else if (PDFscheme.eq.2) then
        ! ETA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*2*(1+x**2)
          xkk(2)=ch1**2*2*(1+x**2)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.3) then
        ! BETA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*2*(1+x**2)
          xkk(2)=ch1**2*2*(1+x**2)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.4) then
        ! MIXED scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*2*(1+x**2)
          xkk(2)=ch1**2*2*(1+x**2)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.5) then
        ! nobeta scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*2*(1+x**2)
          xkk(2)=ch1**2*2*(1+x**2)
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.6) then
        ! DELTA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*2*(1+x**2)
          xkk(2)=ch1**2*2*(1+x**2)
        else
          xkk(:) = 0d0
        endif
      else
        write(6,*)'Error in xklog: wrong PDF scheme', PDFscheme
        stop
      endif
      xkk(1) = xkk(1)*g**2
      xkk(2) = xkk(2)*dble(gal(1))**2
      return
      end


      subroutine xkdelta(PDFscheme, col1, col2, ch1, ch2, xkk)
c This function returns the quantity K^{(d)}_{ab}, relevant for
c the MS --> DIS (or any other) scheme change in the factorization scheme.
C The first entry in xkk is for QCD splittings, the second QED
      implicit none
      integer PDFscheme, col1, col2
      double precision ch1, ch2
      double precision xkk(2)

      double precision pi, vcf, vtf, vca, xnc
      parameter (pi=3.14159265358979312D0)
      parameter (vcf=4.d0/3.d0)
      parameter (vtf=1.d0/2.d0)
      parameter (vca=3.d0)
      parameter (xnc=3.d0)

      include "coupl.inc"
c
      if (PDFscheme.eq.0) then
        ! MSbar, all terms are zero
        xkk(:) = 0d0
      else if (PDFscheme.eq.1) then
        ! DIS scheme
        if(col1.eq.8.and.col2.eq.8)then ! gg
          xkk(1)=0.d0
          xkk(2)=0.d0
        elseif((abs(col1).eq.3.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).gt.0d0))then ! qq
          xkk(1)=0.d0
          xkk(2)=0.d0
        elseif((col1.eq.8.and.abs(col2).eq.3) .or. 
     $         (dabs(ch1).eq.0d0.and.dabs(ch2).gt.0d0))then ! gq
          xkk(1)=vcf*(9.d0/2.d0+pi**2/3.d0)
          xkk(2)=ch2**2*(9.d0/2.d0+pi**2/3.d0)
        elseif((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg
          xkk(1)=-vcf*(9.d0/2.d0+pi**2/3.d0)
          xkk(2)=-ch1**2*(9.d0/2.d0+pi**2/3.d0)
        else
          write(6,*)'Error in xkdelta: wrong values', col1, col2, ch1, ch2
          stop
        endif
      else if (PDFscheme.eq.2) then
        ! ETA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*7d0/2d0
          xkk(2)=-ch1**2*7d0/2d0
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.3) then
        ! BETA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*7d0/2d0
          xkk(2)=-ch1**2*7d0/2d0
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.4) then
        ! MIXED scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*2d0
          xkk(2)=-ch1**2*2d0
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.5) then
        ! nobeta scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=-vcf*2d0
          xkk(2)=-ch1**2*2d0
        else
          xkk(:) = 0d0
        endif
      else if (PDFscheme.eq.6) then
        ! DELTA scheme (lepton)
        if((abs(col1).eq.3.and.col2.eq.8) .or. 
     $         (dabs(ch1).gt.0d0.and.dabs(ch2).eq.0d0))then ! qg / egam
          xkk(1)=vcf*(-2d0)
          xkk(2)=ch1**2*(-2d0)
        else
          xkk(:) = 0d0
        endif
      else
        write(6,*)'Error in xkdelta: wrong PDF scheme', PDFscheme
        stop
      endif
      xkk(1) = xkk(1)*g**2
      xkk(2) = xkk(2)*dble(gal(1))**2
      return
      end


      subroutine AP_reduced(col1, col2, ch1, ch2, t, z, ap)
c Returns Altarelli-Parisi splitting function summed/averaged over helicities
c times prefactors such that |M_n+1|^2 = ap * |M_n|^2. This means
c    AP_reduced = (1-z) P_{S(part1,part2)->part1+part2}(z) * g^2/t
C the first entry in AP is QCD, the second QED
c Therefore, the labeling conventions for particle IDs are not as in FKS:
c part1 and part2 are the two particles emerging from the branching.
c part1 and part2 can be either gluon (8) or (anti-)quark (+-3). z is the
c fraction of the energy of part1 and t is the invariant mass of the mother.
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,ap(2),t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

C sanity check
      if (col1.eq.8.and.dabs(ch1).ne.0d0 .or.
     1 col2.eq.8.and.dabs(ch2).ne.0d0) then
         write (*,*) 'Fatal Error #0 in AP_reduced',col1,col2,ch1,ch2
         stop
      endif

      if (col1.eq.8 .and. col2.eq.8)then
c g->gg splitting
         ap(1) = 2d0 * CA * ( (1d0-z)**2/z + z + z*(1d0-z)**2 )
         ap(2) = 0d0

      elseif ((abs(col1).eq.3 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).gt.0d0)) then
c g/a->qqbar splitting
         ap(1) = TR * ( z**2 + (1d0-z)**2 )*(1d0-z)
         ap(2) = dble(abs(col1)) * ch1**2 * ( z**2 + (1d0-z)**2 )*(1d0-z)

      elseif ((abs(col1).eq.3 .and. col2.eq.8) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).eq.0d0)) then
c q->q g/a splitting
         ap(1) = CF * (1d0+z**2)
         ap(2) = ch1**2 * (1d0+z**2) 

      elseif ((col1.eq.8 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).eq.0d0 .and. dabs(ch2).gt.0d0)) then
c q->gq splitting
         ap(1) = CF * (1d0+(1d0-z)**2)*(1d0-z)/z
         ap(2) = ch2**2 * (1d0+(1d0-z)**2)*(1d0-z)/z

      else
         write (*,*) 'Fatal Error #1 in AP_reduced',col1,col2,ch1,ch2
         stop
      endif

      ap(1) = ap(1)*g**2/t
      ap(2) = ap(2)*dble(gal(1))**2/t
      return
      end


      subroutine AP_reduced_prime(col1,col2,ch1,ch2,t,z,apprime)
c Returns (1-z)*P^\prime * gS^2/t, with the same conventions as AP_reduced
C the first entry in APprime is QCD, the second QED
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,apprime(2),t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (col1.eq.8 .and. col2.eq.8) then
c g->gg splitting
         apprime(1) = 0d0
         apprime(2) = 0d0

      elseif ((abs(col1).eq.3 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).gt.0d0)) then
c g/a->qqbar splitting
         apprime(1) = -2 * TR * z * (1d0-z)**2
         apprime(2) = -2 * dble(abs(col1)) * ch1**2 * z * (1d0-z)**2
         
      elseif ((abs(col1).eq.3 .and. col2.eq.8) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).eq.0d0)) then
c q->q g/a splitting
         apprime(1) = - CF * (1d0-z)**2
         apprime(2) = - ch1**2 * (1d0-z)**2

      elseif ((col1.eq.8 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).eq.0d0 .and. dabs(ch2).gt.0d0)) then
c q->g/a q splitting
         apprime(1) = - CF * z * (1d0-z)
         apprime(2) = - ch2**2 * z * (1d0-z)
      else
         write (*,*) 'Fatal error in AP_reduced_prime',col1,col2,ch1,ch2
         stop
      endif

      apprime(1) = apprime(1)*g**2/t
      apprime(2) = apprime(2)*dble(gal(1))**2/t
      return
      end


      subroutine Qterms_reduced_timelike(col1,col2,ch1,ch2,t,z,Qterms)
c Eq's B.31 to B.34 of FKS paper, times (1-z)*g^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced
C the first entry in Qterms is QCD, the second QED
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,Qterms(2),t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (col1.eq.8 .and. col2.eq.8) then
c g->gg splitting
         Qterms(1) = -4d0 * CA * z*(1d0-z)**2
         Qterms(2) = 0d0

      elseif ((abs(col1).eq.3 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).gt.0d0)) then
c g/a ->qqbar splitting
         Qterms(1) = 4d0 * TR * z*(1d0-z)**2
         Qterms(2) = 4d0 * dble(abs(col1)) * ch1**2 * z*(1d0-z)**2
         
      elseif ((abs(col1).eq.3 .and. col2.eq.8) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).eq.0d0)) then
c q->q g/a splitting
         Qterms(1) = 0d0
         Qterms(2) = 0d0

      elseif ((col1.eq.8 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).eq.0d0 .and. dabs(ch2).gt.0d0)) then
c q->g/a q splitting
         Qterms(1) = 0d0
         Qterms(2) = 0d0
      else
         write (*,*) 'Fatal error in Qterms_reduced_timelike',col1,col2,ch1,ch2
         stop
      endif

      Qterms(1) = Qterms(1)*g**2/t
      Qterms(2) = Qterms(2)*dble(gal(1))**2/t
      return
      end


      subroutine Qterms_reduced_spacelike(col1,col2,ch1,ch2,t,z,Qterms)
c Eq's B.42 to B.45 of FKS paper, times (1-z)*gS^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced.
C the first entry in Qterms is QCD, the second QED
c Thus, part1 has momentum fraction z, and it is the one off-shell
c (see (FKS.B.41))
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,Qterms(2),t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (col1.eq.8 .and. col2.eq.8)then
c g->gg splitting
         Qterms(1) = -4d0 * CA * (1d0-z)**2/z
         Qterms(2) = 0d0

      elseif ((abs(col1).eq.3 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).gt.0d0)) then
c g/a->qqbar splitting
         Qterms(1) = 0d0
         Qterms(2) = 0d0
         
      elseif ((abs(col1).eq.3 .and. col2.eq.8) .or.
     &       (dabs(ch1).gt.0d0 .and. dabs(ch2).eq.0d0)) then
c q->qg/a splitting
         Qterms(1) = 0d0
         Qterms(2) = 0d0

      elseif ((col1.eq.8 .and. abs(col2).eq.3) .or.
     &       (dabs(ch1).eq.0d0 .and. dabs(ch2).gt.0d0)) then
c q->g/a q splitting
         Qterms(1) = -4d0 * CF * (1d0-z)**2/z
         Qterms(2) = -4d0 * ch2**2 * (1d0-z)**2/z
      else
         write (*,*) 'Fatal error in Qterms_reduced_spacelike',col1,col2,ch1,ch2
         stop
      endif

      Qterms(1) = Qterms(1)*g**2/t
      Qterms(2) = Qterms(2)*dble(gal(1))**2/t
      return
      end


      subroutine AP_reduced_SUSY(col1,col2,ch1,ch2,t,z,ap)
c Same as AP_reduced, except for the fact that it only deals with
c   go -> go g
c   sq -> sq g
c splittings in SUSY. We assume this function to be called with 
c part2==colour(i_fks)
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,ap(2),t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"
      write(*,*) 'FIX AP REDUCED SUSY'

      if (col2.ne.8.and.ch2.ne.0d0)then
         write (*,*) 'Fatal error #0 in AP_reduced_SUSY',col1,col2,ch1,ch2
         stop
      endif

      if (col1.eq.8)then
c go->gog splitting
         ap(1) = CA * (1d0+z**2)
         ap(2) = 0d0

      elseif(abs(col1).eq.3.or.ch1.ne.0d0)then
c sq->sqg splitting
         ap(1) = 2d0 * CF * z
         ap(2) = 2d0 * ch1**2 * z

      else
         write (*,*) 'Fatal error in AP_reduced_SUSY',col1,col2,ch1,ch2
         stop
      endif

      ap(1) = ap(1)*g**2/t
      ap(2) = ap(2)*dble(gal(1))**2/t

      return
      end


      subroutine AP_reduced_massive(col1,col2,ch1,ch2,t,z,q2,m2,ap)
c Returns massive Altarelli-Parisi splitting function summed/averaged over helicities
c times prefactors such that |M_n+1|^2 = ap * |M_n|^2. This means
c    AP_reduced = (1-z) P_{S(part1,part2)->part1+part2}(z) * gS^2/t
c Therefore, the labeling conventions for particle IDs are not as in FKS:
c part1 and part2 are the two particles emerging from the branching.
c part1 and part2 can be either gluon (8) or (anti-)quark (+-3). z is the
c fraction of the energy of part1 and t is the invariant mass of the mother.
      implicit none

      integer col1, col2
      double precision ch1, ch2
      double precision z,ap(2),t,q2,m2

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"
      write(*,*) 'FIX AP REDUCED MASSIVE'

      if (col1.eq.8 .and. col2.eq.8)then
c g->gg splitting
         ap(1) = 2d0 * CA * ( (1d0-z)**2/z + z + z*(1d0-z)**2 )
         ap(2) = 0d0

      elseif((abs(col1).eq.3 .and. abs(col2).eq.3).or.
     &       (ch1.ne.0d0 .and. ch2.ne.0d0))then
c g->qqbar splitting
         ap(1) = TR * ( z**2 + (1d0-z)**2 )*(1d0-z) + TR * 2d0*m2/(z*q2)
         ap(1) = dble(abs(col1)) * ch1**2 * ( z**2 + (1d0-z)**2 )*(1d0-z) + TR * 2d0*m2/(z*q2)
      elseif((abs(col1).eq.3 .and. col2.eq.8).or.
     &      (ch1.ne.0d0.and.ch2.eq.0d0))then
c q->qg splitting
         ap(1) = CF * (1d0+z**2) - CF * 2d0*m2/(z*q2)
         ap(2) = ch1**2 * (1d0+z**2) - ch1**2 * 2d0*m2/(z*q2)

      elseif((col1.eq.8 .and. abs(col2).eq.3).or.
     &      (ch1.eq.0d0.and.ch2.ne.0d0))then
c q->gq splitting
         ap(1) = CF * (1d0+(1d0-z)**2)*(1d0-z)/z - CF * 2d0*m2/(z*q2)
         ap(2) = ch2**2 * (1d0+(1d0-z)**2)*(1d0-z)/z - ch2**2 * 2d0*m2/(z*q2)
      else
         write (*,*) 'Fatal error in AP_reduced',col1,col2,ch1,ch2
         stop
      endif

      ap(1) = ap(1)*g**2/t
      ap(2) = ap(2)*dble(gal(1))**2/t

      return
      end



      subroutine sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
      implicit none

      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "coupl.inc"

      integer m,n

      double precision softcontr,pp(0:3,nexternal),wgt,eik,xi_i_fks
     &     ,y_ij_fks
      double precision wgt1
      integer i,j,k 

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision zero,pmass(nexternal)
      parameter(zero=0d0)

      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      integer ipos_ord
      include 'orders.inc'
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft

      double precision iden_comp
      common /c_iden_comp/iden_comp

      include "pmass.inc"
c
c Call the Born to be sure that 'CalculatedBorn' is done correctly. This
c should always be done before calling the color-correlated Borns,
c because of the caching of the diagrams.
c
      call sborn(p_born(0,1),wgt1)
c
C Reset the amp_split array
      amp_split(1:amp_split_size) = 0d0

      softcontr=0d0
      do i=1,fks_j_from_i(i_fks,0)
         do j=1,i
            m=fks_j_from_i(i_fks,i)
            n=fks_j_from_i(i_fks,j)
            if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)) .and.
     &           n.ne.i_fks.and.m.ne.i_fks) then
C wgt includes the gs/w^2 
               call sborn_sf(p_born,m,n,wgt)
               if (wgt.ne.0d0) then
                  call eikonal_reduced(pp,m,n,i_fks,j_fks,
     #                                 xi_i_fks,y_ij_fks,eik)
                  softcontr=softcontr+wgt*eik*iden_comp
                  ! update the amp_split array
                  if (need_color_links) ipos_ord = qcd_pos
                  if (need_charge_links) ipos_ord = qed_pos
                  amp_split(1:amp_split_size) = amp_split(1:amp_split_size)
     $                - 2d0 * eik * amp_split_soft(1:amp_split_size)*iden_comp
               endif
            endif
         enddo
      enddo
      wgt=softcontr
c Add minus sign to compensate the minus in the color factor
c of the color-linked Borns (b_sf_0??.f)
c Factor two to fix the limits.
      wgt=-2d0*wgt
      return
      end


      subroutine eikonal_reduced(pp,m,n,i_fks,j_fks,xi_i_fks,y_ij_fks,eik)
c     Returns the eikonal factor
      implicit none

      include "nexternal.inc"
      double precision eik,pp(0:3,nexternal),xi_i_fks,y_ij_fks
      double precision dot,dotnm,dotni,dotmi,fact
      integer n,m,i_fks,j_fks,i
      integer softcol

      include "coupl.inc"

      external dot
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      real*8 phat_i_fks(0:3)

      double precision zero,pmass(nexternal),tiny
      parameter(zero=0d0)
      parameter(tiny=1d-6)
      include "pmass.inc"

c Define the reduced momentum for i_fks
      softcol=0
      if (1d0-y_ij_fks.lt.tiny)softcol=2
      if(p_i_fks_cnt(0,softcol).lt.0d0)then
        if(xi_i_fks.eq.0.d0)then
           write (*,*) 'Error #1 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
           stop
        endif
        if(pp(0,i_fks).ne.0.d0)then
          write(*,*)'WARNING in eikonal_reduced: no cnt momenta',
     #      softcol,xi_i_fks,y_ij_fks
          do i=0,3
            phat_i_fks(i)=pp(i,i_fks)/xi_i_fks
          enddo
        else
          write (*,*) 'Error #2 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
          stop
        endif
      else
        do i=0,3
          phat_i_fks(i)=p_i_fks_cnt(i,softcol)
        enddo
      endif
c Calculate the eikonal factor
      dotnm=dot(pp(0,n),pp(0,m))
      if ((m.ne.j_fks .and. n.ne.j_fks) .or. pmass(j_fks).ne.ZERO) then
         dotmi=dot(pp(0,m),phat_i_fks)
         dotni=dot(pp(0,n),phat_i_fks)
         fact= 1d0-y_ij_fks
      elseif (m.eq.j_fks .and. n.ne.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=dot(pp(0,n),phat_i_fks)
         dotmi=sqrtshat/2d0 * pp(0,j_fks)
         fact= 1d0
      elseif (m.ne.j_fks .and. n.eq.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=sqrtshat/2d0 * pp(0,j_fks)
         dotmi=dot(pp(0,m),phat_i_fks)
         fact= 1d0
      else
         write (*,*) 'Error #3 in eikonal_reduced'
         stop
      endif

      eik = dotnm/(dotni*dotmi)*fact
      return
      end


      subroutine sreal_deg(p,xi_i_fks,y_ij_fks,
     #                     collrem_xi,collrem_lxi)
      use extra_weights
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      include 'q_es.inc'
      include "run.inc"
      include "orders.inc"

      integer iord, iap
      double precision p(0:3,nexternal),collrem_xi,collrem_lxi
      double precision xi_i_fks,y_ij_fks
      double precision collrem_xi_tmp, collrem_lxi_tmp

      double precision p_born(0:3,nexternal-1), wgt_born
      common/pborn/p_born

      double precision p_born_coll(0:3,nexternal-1)
      common/pborn_coll/p_born_coll

      double precision p_born_used(0:3,nexternal-1)

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision delta_used
      common /cdelta_used/delta_used

      double precision rwgt,shattmp,dot,born_wgt,oo2pi,z,t,ap(2),
     # apprime(2),xkkernp(2),xkkernd(2),xkkernl(2),xnorm
      external dot

c Particle types (=color/charges) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i, ch_j, ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      common /c_born_cnt/ ans_cnt
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      
      double precision one,pi
      parameter (one=1.d0)
      parameter (pi=3.1415926535897932385d0)
      double precision iden_comp
      common /c_iden_comp/iden_comp

      double complex ans_extra_cnt(2,nsplitorders)
      integer iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

C keep track of each split orders
      integer iamp
      double precision amp_split_collrem_xi(amp_split_size), 
     $                 amp_split_collrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_xi(amp_split_size),
     $                 amp_split_wgtdegrem_lxi(amp_split_size),
     $                 amp_split_wgtdegrem_muF(amp_split_size)
      common /to_amp_split_deg/amp_split_wgtdegrem_xi,
     $                         amp_split_wgtdegrem_lxi,
     $                         amp_split_wgtdegrem_muF
      ! amp_split for the PDF scheme
      double precision amp_split_wgtpsch_p(amp_split_size),
     $                 amp_split_wgtpsch_l(amp_split_size),
     $                 amp_split_wgtpsch_d(amp_split_size)
      common /to_amp_split_dis/amp_split_wgtpsch_p,
     $                         amp_split_wgtpsch_l,
     $                         amp_split_wgtpsch_d
      double precision prefact_xi

      logical use_evpr
      common /to_use_evpr/use_evpr

      logical firsttime_pdf
      data firsttime_pdf /.true./

      ! This is needed for the PDFscheme variable 
      include "../../Source/PDF/pdf.inc"
      ! PDFscheme = : 
      ! 0 -> msbar
      ! 1 -> dis (hadronic)
      ! 2 -> eta (leptonic)
      ! 3 -> beta (leptonic)
      ! 4 -> mixed (leptonic)
      ! 5 -> nobeta (leptonic)
      ! 6 -> delta (leptonic)
      if(firsttime_pdf) then
        write(*,*) 'PDFscheme' , pdfscheme
        firsttime_pdf = .false.
      endif

      amp_split_collrem_xi(1:amp_split_size) = 0d0
      amp_split_collrem_lxi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_xi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_lxi(1:amp_split_size) = 0d0
      amp_split_wgtdegrem_muF(1:amp_split_size) = 0d0
      amp_split_wgtpsch_p(1:amp_split_size) = 0d0
      amp_split_wgtpsch_l(1:amp_split_size) = 0d0
      amp_split_wgtpsch_d(1:amp_split_size) = 0d0

C in the case of the collinear CT, use p_born_coll
C  (when not doing event projection). 
C For the soft-collinear one, use p_born
      if (xi_i_fks.gt.0d0.and..not.use_evpr) then
          p_born_used(:,:) = p_born_coll(:,:)
      else ! if (xi_i_fks.eq.0d0) then
          p_born_used(:,:) = p_born(:,:)
      endif

      if(j_fks.gt.nincoming)then
c Do not include this contribution for final-state branchings
         collrem_xi=0.d0
         collrem_lxi=0.d0
         if(doreweight)then
           wgtdegrem_xi=0.d0
           wgtdegrem_lxi=0.d0
           wgtdegrem_muF=0.d0
         endif
         return
      endif

      if(p_born_used(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sreal_deg"
         collrem_xi=0.d0
         collrem_lxi=0.d0
         if(doreweight)then
           wgtdegrem_xi=0.d0
           wgtdegrem_lxi=0.d0
           wgtdegrem_muF=0.d0
         endif
         return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

c A factor gS^2 is included in the Altarelli-Parisi kernels
      oo2pi=one/(8d0*PI**2)

      z = 1d0 - xi_i_fks
      t = one
      call AP_reduced(m_type,i_type,ch_m,ch_i,t,z,ap)
      call AP_reduced_prime(m_type,i_type,ch_m,ch_i,t,z,apprime)

      ! call the PDF-scheme kernels here 
      !   p-> [1/(1-z)]_+  
      !   l-> [log(1-z)/(1-z)]_+  
      !   d-> delta(1-z)
      call xkplus(PDFscheme,m_type,i_type,ch_m,ch_i,z,xkkernp)
      call xkdelta(PDFscheme,m_type,i_type,ch_m,ch_i,xkkernd)
      call xklog(PDFscheme,m_type,i_type,ch_m,ch_i,z,xkkernl)

      collrem_xi=0.d0
      collrem_lxi=0.d0
      calculatedborn=.false.
      do iord = 1, nsplitorders
        if (.not.split_type(iord).or.(iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle

C check if any extra_cnt is needed
        if (iextra_cnt.gt.0) then
            if (iord.eq.isplitorder_born) then
            ! this is the contribution from the born ME
               call sborn(p_born_used,wgt_born)
               wgt1(1) = ans_cnt(1,iord)
               wgt1(2) = ans_cnt(2,iord)
            else if (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               call extra_cnt(p_born_used,iextra_cnt,ans_extra_cnt)
               wgt1(1) = ans_extra_cnt(1,iord)
               wgt1(2) = ans_extra_cnt(2,iord)
            else
               write(*,*) 'ERROR in sreal_deg', iord
               stop
            endif
        else
           call sborn(p_born_used,wgt_born)
           wgt1(1) = ans_cnt(1,iord)
           wgt1(2) = ans_cnt(2,iord)
        endif
        
        if (iord.eq.qcd_pos) iap = 1
        if (iord.eq.qed_pos) iap = 2
        collrem_xi_tmp=ap(iap)*log(shat*delta_used/(2*q2fact(j_fks))) -
     #           apprime(iap) 
        collrem_lxi_tmp=2*ap(iap)

c The partonic flux 1/(2*s) is inserted in genps. Thus, an extra 
c factor z (implicit in the flux of the reduced Born in FKS) 
c has to be inserted here
        xnorm=1.d0/z *iden_comp

        collrem_xi=collrem_xi + oo2pi*dble(wgt1(1))*collrem_xi_tmp*
     &       xnorm
        collrem_lxi=collrem_lxi + oo2pi*dble(wgt1(1))*collrem_lxi_tmp*
     &       xnorm

        amp_split_collrem_xi(1:amp_split_size) = amp_split_collrem_xi(1:amp_split_size)+ 
     &   dble(amp_split_cnt(1:amp_split_size,1,iord))*oo2pi*collrem_xi_tmp*xnorm
        amp_split_collrem_lxi(1:amp_split_size) = amp_split_collrem_lxi(1:amp_split_size)+
     &   dble(amp_split_cnt(1:amp_split_size,1,iord))*oo2pi*collrem_lxi_tmp*xnorm

        prefact_xi=ap(iap)*log(shat*delta_used/(2*QES2)) -
     &               apprime(iap)
        amp_split_wgtdegrem_xi(1:amp_split_size) = amp_split_wgtdegrem_xi(1:amp_split_size)+
     &   oo2pi*dble(amp_split_cnt(1:amp_split_size,1,iord))*prefact_xi*xnorm
        amp_split_wgtdegrem_lxi(1:amp_split_size) = amp_split_collrem_lxi(1:amp_split_size)
        amp_split_wgtdegrem_muF(1:amp_split_size) = amp_split_wgtdegrem_muF(1:amp_split_size)-
     &   oo2pi*dble(amp_split_cnt(1:amp_split_size,1,iord))*ap(iap)*xnorm
        ! amp split for the PDF scheme
        if (PDFscheme.ne.0) then
          amp_split_wgtpsch_p(1:amp_split_size) = amp_split_wgtpsch_p(1:amp_split_size) - 
     $     dble(amp_split_cnt(1:amp_split_size,1,iord))*xkkernp(iap)*oo2pi*xnorm
          amp_split_wgtpsch_l(1:amp_split_size) = amp_split_wgtpsch_l(1:amp_split_size) - 
     $     dble(amp_split_cnt(1:amp_split_size,1,iord))*xkkernl(iap)*oo2pi*xnorm
          amp_split_wgtpsch_d(1:amp_split_size) = amp_split_wgtpsch_d(1:amp_split_size) - 
     $     dble(amp_split_cnt(1:amp_split_size,1,iord))*xkkernd(iap)*oo2pi*xnorm
        endif

      enddo
      calculatedborn=.false.

      return
      end


      subroutine set_cms_stuff(icountevts)
      implicit none
      include "run.inc"

      integer icountevts

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev

      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt

      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev

      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to lab frame --
c same for event and counterevents
c This is the rapidity that enters in the arguments of the sinh() and
c cosh() of the boost, in such a way that
c       y(k)_lab = y(k)_tilde - ybst_til_tolab
c where y(k)_lab and y(k)_tilde are the rapidities computed with a generic
c four-momentum k, in the lab frame and in the \tilde{k}_1+\tilde{k}_2 
c c.m. frame respectively
      ybst_til_tolab=-ycm_cnt(0)
      if(icountevts.eq.-100)then
c set Bjorken x's in run.inc for the computation of PDFs in auto_dsig
        xbk(1)=xbjrk_ev(1)
        xbk(2)=xbjrk_ev(2)
c shat=2*k1.k2 -- consistency of this assignment with momenta checked
c in phspncheck_nocms
        shat=shat_ev
        sqrtshat=sqrtshat_ev
c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to 
c k_1+k_2 c.m. frame
        ybst_til_tocm=ycm_ev-ycm_cnt(0)
      else
c do the same as above for the counterevents
        xbk(1)=xbjrk_cnt(1,icountevts)
        xbk(2)=xbjrk_cnt(2,icountevts)
        shat=shat_cnt(icountevts)
        sqrtshat=sqrtshat_cnt(icountevts)
        ybst_til_tocm=ycm_cnt(icountevts)-ycm_cnt(0)
      endif
      return
      end

      subroutine get_mc_lum(j_fks,zhw_used,xi_i_fks,xlum_mc_fact)
      implicit none
      include "run.inc"
      include "nexternal.inc"
      integer j_fks
      double precision zhw_used,xi_i_fks,xlum_mc_fact
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      if(zhw_used.lt.0.d0.or.zhw_used.gt.1.d0)then
        write(*,*)'Error #1 in get_mc_lum',zhw_used
        stop
      endif
      if(j_fks.gt.nincoming)then
        xbk(1)=xbjrk_cnt(1,0)
        xbk(2)=xbjrk_cnt(2,0)
        xlum_mc_fact=1.d0
      elseif(j_fks.eq.1)then
        xbk(1)=xbjrk_cnt(1,0)/zhw_used
        xbk(2)=xbjrk_cnt(2,0)
c Note that this is true for Pythia since, due to event projection and to
c the definition of the shower variable x = zhw_used, the Bjorken x's for
c the event (to be used in H events) are the ones for the counterevent
c multiplied by 1/x (by 1) for the emitting (non emitting) leg 
        if(xbk(1).gt.1.d0)then
          xlum_mc_fact = 0.d0
        else
          xlum_mc_fact = (1-xi_i_fks)/zhw_used
        endif
      elseif(j_fks.eq.2)then
        xbk(1)=xbjrk_cnt(1,0)
        xbk(2)=xbjrk_cnt(2,0)/zhw_used
        if(xbk(2).gt.1.d0)then
          xlum_mc_fact = 0.d0
        else
          xlum_mc_fact = (1-xi_i_fks)/zhw_used
        endif
      else
        write(*,*)'Error in get_mc_lum: unknown j_fks',j_fks
        stop
      endif
      if( xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #    ( (xbk(1).gt.1.d0.or.xbk(2).gt.1.d0).and.
     #      j_fks.gt.nincoming ) .or.
     #    (xbk(2).gt.1.d0.and.j_fks.eq.1) .or.
     #    (xbk(1).gt.1.d0.and.j_fks.eq.2) )then
      ! add an extra check on the bjorken x's (relevant for ee
      ! collisions)
         if (xbk(1).gt.1d0)then
            if(xbk(1)-1d0.lt.1d-12) then
               xbk(1) = 1d0
            else
               write(*,*)'Error in get_mc_lum: x_i',xbk(1),xbk(2)
               stop           
            endif
         endif
         if (xbk(2).gt.1d0)then
            if(xbk(2)-1d0.lt.1d-12) then
               xbk(2) = 1d0
            else
               write(*,*)'Error in get_mc_lum: x_i',xbk(1),xbk(2)
               stop           
            endif
         endif
      endif
      return
      end


      subroutine xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,pass)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer i_fks,j_fks
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision jac,jac_cnt(-2:2)
      integer izero,ione,itwo,iunit,isum
      logical verbose,pass,pass0
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)
      parameter (iunit=6)
      parameter (verbose=.false.)
      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax
c
      isum=0
      if(jac_cnt(0).gt.0.d0)isum=isum+1
      if(jac_cnt(1).gt.0.d0)isum=isum+2
      if(jac_cnt(2).gt.0.d0)isum=isum+4
      pass=.true.
c
      if(isum.eq.0.or.isum.eq.1.or.isum.eq.2.or.isum.eq.4)then
c Nothing to be done: 0 or 1 configurations computed
        if(verbose)write(iunit,*)'none'
      elseif(isum.eq.3.or.isum.eq.5.or.isum.eq.7)then
c Soft is taken as reference
        if(isum.eq.7)then
          if(verbose)then
            write(iunit,*)'all'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,pass0,ione,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
          if(verbose)then
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,pass0,itwo,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        elseif(isum.eq.3)then
          if(verbose)then
            write(iunit,*)'C+S'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,pass0,ione,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        elseif(isum.eq.5)then
          if(verbose)then
            write(iunit,*)'SC+S'
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,pass0,itwo,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        endif
      elseif(isum.eq.6)then
c Collinear is taken as reference
        if(verbose)then
          write(iunit,*)'SC+C'
          write(iunit,*)'    '
          write(iunit,*)'SC/C'
        endif
        call xmcompare(verbose,pass0,itwo,ione,i_fks,j_fks,p,p1_cnt)
        pass=pass.and.pass0
      else
        write(6,*)'Fatal error in xmom_compare',isum
        stop
      endif
      if(.not.pass)i_momcmp_count=i_momcmp_count +1
c
      return
      end


      subroutine xmcompare(verbose,pass0,inum,iden,i_fks,j_fks,p,p1_cnt)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      logical verbose,pass0
      integer inum,iden,i_fks,j_fks,iunit,ipart,i,j,k
      double precision tiny,vtiny,xnum,xden,xrat
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      parameter (iunit=6)
      parameter (tiny=1.d-4)
      parameter (vtiny=1.d-10)
      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax
      include "pmass.inc"
c
      pass0=.true.
      do ipart=1,nexternal
        do i=0,3
          xnum=p1_cnt(i,ipart,inum)
          xden=p1_cnt(i,ipart,iden)
          if(verbose)then
            if(i.eq.0)then
              write(iunit,*)' '
              write(iunit,*)'part=',ipart
            endif
            call xprintout(iunit,xnum,xden)
          else
            if(ipart.ne.i_fks.and.ipart.ne.j_fks)then
              if(xden.ne.0.d0)then
                xrat=abs(1-xnum/xden)
              else
                xrat=abs(xnum)
              endif
              if(abs(xnum).eq.0d0.and.abs(xden).le.vtiny)xrat=0d0
c The following line solves some problem as well, but before putting
c it as the standard, one should think a bit about it
              if(abs(xnum).le.vtiny.and.abs(xden).le.vtiny)xrat=0d0
              if(xrat.gt.tiny .and.
     &          (pmass(ipart).eq.0d0.or.xnum/pmass(ipart).gt.vtiny))then
                 write(*,*)'Kinematics of counterevents'
                 write(*,*)inum,iden
                 write(*,*)'is different. Particle:',ipart
                 write(*,*) xrat,xnum,xden
                 do j=1,nexternal
                    write(*,*) j,(p1_cnt(k,j,inum),k=0,3)
                 enddo
                 do j=1,nexternal
                    write(*,*) j,(p1_cnt(k,j,iden),k=0,3)
                 enddo
                 xratmax=max(xratmax,xrat)
                 pass0=.false.
              endif
            endif
          endif
        enddo
      enddo
      do i=0,3
        if(j_fks.gt.nincoming)then
          xnum=p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        else
          xnum=-p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=-p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        endif
        if(verbose)then
          if(i.eq.0)then
            write(iunit,*)' '
            write(iunit,*)'part=i+j'
          endif
          call xprintout(iunit,xnum,xden)
        else
          if(xden.ne.0.d0)then
            xrat=abs(1-xnum/xden)
          else
            xrat=abs(xnum)
          endif
          if(xrat.gt.tiny)then
            write(*,*)'Kinematics of counterevents'
            write(*,*)inum,iden
            write(*,*)'is different. Particle i+j'
            xratmax=max(xratmax,xrat)
            pass0=.false.
          endif
        endif
      enddo
      return
      end


      subroutine xprintout(iunit,xv,xlim)
      implicit real*8(a-h,o-z)
c
      if(abs(xlim).gt.1.d-30)then
        write(iunit,*)xv/xlim,xv,xlim
      else
        write(iunit,*)xv,xlim
      endif
      return
      end



c The following has been derived with minor modifications from the
c analogous routine written for VBF
      subroutine checkres(xsecvc,xseclvc,wgt,wgtl,xp,lxp,
     #                    iflag,imax,iev,nexternal,i_fks,j_fks,iret)
c Checks that the sequence xsecvc(i), i=1,imax, converges to xseclvc.
c Due to numerical inaccuracies, the test is deemed OK if there are
c at least ithrs+1 consecutive elements in the sequence xsecvc(i)
c which are closer to xseclvc than the preceding element of the sequence.
c The counting is started when an xsecvc(i0) is encountered, which is
c such that |xsecvc(i0)/xseclvc-1|<0.1 if xseclvc#0, or such that
c |xsecvc(i0)|<0.1 if xseclvc=0. In order for xsecvc(i+1 )to be defined 
c closer to xseclvc than xsecvc(i), the condition
c   |xsecvc(i)/xseclvc-1|/|xsecvc(i+1)/xseclvc-1| > rat
c if xseclvc#0, or 
c   |xsecvc(i)|/|xsecvc(i+1)| > rat
c if xseclvc=0 must be fulfilled; the value of rat is set equal to 4 and to 2
c for soft and collinear limits respectively, since the cross section is 
c expected to scale as xii**2 and sqrt(1-yi**2), and the values of xii and yi
c are chosen as powers of 10 (thus, if scaling would be exact, rat should
c be set equal to 10 and sqrt(10)).
c If the test is passed, icount=ithrs, else icount<ithrs; in the former
c case iret=0, in the latter iret=1.
c When the test is not passed, one may choose to stop the program dead here;
c in such a case, set istop=1 below. Each time the test is not passed,
c the results are written onto fort.77; set iwrite=0 to prevent the writing
      implicit none
      real*8 xsecvc(15),xseclvc,wgt(15),wgtl,lxp(0:3,21),xp(15,0:3,21)
      real*8 ckc(15),rckc(15),rat
      integer iflag,imax,iev,nexternal,i_fks,j_fks,iret,ithrs,istop,
     # iwrite,i,k,l,imin,icount
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checkres: imax is too large',imax
        stop
      endif
      do i=1,imax
        if(xseclvc.eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=4.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checkres: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      do while(ckc(i).gt.0.1d0 .and. xseclvc.ne.0d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).ne.0.d0)then
          rckc(i)=ckc(i)/ckc(i+1)
        else
          rckc(i)=1.d8
        endif
      enddo
      icount=0
      i=imin
      do while(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      iret=0
      if(icount.ne.ithrs)then
        iret=1
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
        if(iwrite.eq.1)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'ME*wgt:'
          do i=1,imax
             call xprintout(77,xsecvc(i),xseclvc)
          enddo
          write(77,*)'wgt:'
          do i=1,imax
             call xprintout(77,wgt(i),wgtl)
          enddo
c
          write(78,*)'    '
          if(iflag.eq.0)then
            write(78,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(78,*)'Collinear #',iev
          endif
          do k=1,nexternal
            write(78,*)''
            write(78,*)'part:',k
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,k),lxp(l,k))
              enddo
            enddo
          enddo
          if(iflag.eq.0)then
            write(78,*)''
            write(78,*)'part: i_fks reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,nexternal+1),
     #                            lxp(l,nexternal+1))
              enddo
            enddo
            write(78,*)''
            write(78,*)'part: i_fks full/reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks),
     #                            xp(i,l,nexternal+1))
              enddo
            enddo
          elseif(iflag.eq.1)then
            write(78,*)''
            write(78,*)'part: i_fks+j_fks'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks)+xp(i,l,j_fks),
     #                            lxp(l,i_fks)+lxp(l,j_fks))
              enddo
            enddo
          endif
        endif
      endif
      return
      end

c The following has been derived with minor modifications from the
c analogous routine written for VBF
      subroutine checkres2(xsecvc,xseclvc,wgt,wgtl,xp,lxp,
     #                    iflag,imax,iev,i_fks,j_fks,iret)
c     same as checkres, but also limits are arrays.
      implicit none
      include 'nexternal.inc'
      real*8 xsecvc(15),xseclvc(15),wgt(15),wgtl(15),lxp(0:3,nexternal+1)
     &     ,xp(15,0:3,nexternal+1)
      real*8 ckc(15),rckc(15),rat
      integer iflag,imax,iev,i_fks,j_fks,iret,ithrs,istop,
     # iwrite,i,k,l,imin,icount
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checkres: imax is too large',imax
        stop
      endif
      do i=1,imax
        if(xseclvc(i).eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc(i)-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=4.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checkres: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      do while(ckc(i).gt.0.1d0 .and. xseclvc(i).ne.0d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).ne.0.d0)then
          rckc(i)=ckc(i)/ckc(i+1)
        else
          rckc(i)=1.d8
        endif
      enddo
      icount=0
      i=imin
      do while(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      iret=0
      if(icount.ne.ithrs)then
        iret=1
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
        if(iwrite.eq.1)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'ME*wgt:'
          do i=1,imax
             call xprintout(77,xsecvc(i),xseclvc(i))
          enddo
          write(77,*)'wgt:'
          do i=1,imax
             call xprintout(77,wgt(i),wgtl(i))
          enddo
c
          write(78,*)'    '
          if(iflag.eq.0)then
            write(78,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(78,*)'Collinear #',iev
          endif
          do k=1,nexternal
            write(78,*)''
            write(78,*)'part:',k
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,k),lxp(l,k))
              enddo
            enddo
          enddo
          if(iflag.eq.0)then
            write(78,*)''
            write(78,*)'part: i_fks reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,nexternal+1),
     #                            lxp(l,nexternal+1))
              enddo
            enddo
            write(78,*)''
            write(78,*)'part: i_fks full/reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks),
     #                            xp(i,l,nexternal+1))
              enddo
            enddo
          elseif(iflag.eq.1)then
            write(78,*)''
            write(78,*)'part: i_fks+j_fks'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks)+xp(i,l,j_fks),
     #                            lxp(l,i_fks)+lxp(l,j_fks))
              enddo
            enddo
          endif
        endif
      endif
      return
      end
      


      subroutine bornsoftvirtual(p,bsv_wgt,virt_wgt,born_wgt)
      use extra_weights
      use mint_module
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      include 'q_es.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge
      include "run.inc"
      include "fks_powers.inc"
      double precision p(0:3,nexternal),bsv_wgt,born_wgt,avv_wgt
      double precision pp(0:3,nexternal)
      
      double precision wgt1
      double precision rwgt,Q,Ej,wgt,contr,eikIreg,m1l_W_finite_CDR
      double precision aso2pi, aeo2pi
      double precision shattmp,dot
      integer i,j,aj,m,n,k,iord,ipos_ord

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision xicut_used
      common /cxicut_used/xicut_used

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision pi
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      common/fks_colors/c,gamma,gammap,gamma_ph,gammap_ph
      double precision c_used, gamma_used, gammap_used
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision double,single,xmu2
      logical ComputePoles,fksprefact
      parameter (ComputePoles=.false.)
      parameter (fksprefact=.true.)

      double precision beta0,ren_group_coeff
      common/cbeta0/beta0,ren_group_coeff

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      double precision virt_fraction_inc
      data virt_fraction_inc /1d0/

      integer include_virt
      double precision vol3

c For tests of virtuals
      double precision xnormsv
      common/cxnormsv/xnormsv
      double precision vrat

      double precision virt_wgt,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

      double precision virtual_over_born
      common/c_vob/virtual_over_born

c timing statistics
      include "timing_variables.inc"

c For the MINT folding
      integer fold,ifold_counter
      common /cfl/fold,ifold_counter
      double precision virt_wgt_save
      save virt_wgt_save

      double precision pmass(nexternal),zero,tiny
      parameter (zero=0d0)
      parameter (tiny=1d-6)
      include 'orders.inc'
      logical firsttime
      data firsttime / .true. /
      logical need_color_links_used, need_charge_links_used
      data need_color_links_used / .false. /
      data need_charge_links_used / .false. /
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      complex*16 ans_cnt(2, nsplitorders)
      common /c_born_cnt/ ans_cnt
      double precision oneo8pi2
      parameter(oneo8pi2 = 1d0/(8d0*pi**2))
      include 'nFKSconfigs.inc'
      INTEGER nFKSprocess, nFKSprocess_save, nFKSprocess_col, nFKSprocess_chg
      COMMON/c_nFKSprocess/nFKSprocess
      data nFKSprocess_col / 0 /
      data nFKSprocess_chg / 0 /
      double precision bsv_wgt_mufoqes, bsv_wgt_mufomur
      double precision contr_mufoqes, contr_mufomur
C to keep track of the various split orders
      integer iamp
      integer orders(nsplitorders)
      double precision amp_split_born(amp_split_size)
      double precision amp_split_bsv(amp_split_size)
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft
      double precision amp_split_finite_ML(amp_split_size)
      common /to_amp_split_finite/amp_split_finite_ML
      double precision amp_split_virt_save(amp_split_size)
      save amp_split_virt_save
      double precision amp_split_virt(amp_split_size),
     &      amp_split_born_for_virt(amp_split_size),
     &      amp_split_avv(amp_split_size)
      common /to_amp_split_virt/amp_split_virt,
     &                          amp_split_born_for_virt,
     &                          amp_split_avv
      double precision amp_split_wgtnstmp(amp_split_size),
     $                 amp_split_wgtwnstmpmuf(amp_split_size),
     $                 amp_split_wgtwnstmpmur(amp_split_size)
      common /to_amp_split_bsv/amp_split_wgtnstmp,
     $                         amp_split_wgtwnstmpmuf,
     $                         amp_split_wgtwnstmpmur
      double precision coupl_wgtwnstmpmuf

      double precision amp_tot

      include "pmass.inc"
      
      if (firsttime) then
C check if any real emission need cahrge/color links
         nFKSprocess_save = nFKSprocess
         do nFKSprocess = 1, FKS_configs
            call fks_inc_chooser()
            need_color_links_used = need_color_links_used .or. need_color_links
            need_charge_links_used = need_charge_links_used .or. need_charge_links
C keep track of which FKS configuration actually needs color/charge
C links
            if (need_color_links.and.nFKSprocess_col.eq.0)
     1          nFKSprocess_col = nFKSprocess
            if (need_charge_links.and.nFKSprocess_chg.eq.0)
     1          nFKSprocess_chg = nFKSprocess
         enddo
         if (need_charge_links_used) then
             write(*,*) 'Charge-linked born are used'
         else
             write(*,*) 'Charge-linked born are not used'
         endif
         if (need_color_links_used) then
             write(*,*) 'Color-linked born are used'
         else
             write(*,*) 'Color-linked born are not used'
         endif
         firsttime = .false.
         nFKSprocess = nFKSprocess_save
         call fks_inc_chooser()
      endif
         

      aso2pi=g**2/(8*pi**2)
      aeo2pi=dble(gal(1))**2/(8*pi**2)

      amp_split_bsv(1:amp_split_size)=0d0
      amp_split_virt(1:amp_split_size)=0d0
      amp_split_avv(1:amp_split_size)=0d0

      if (.not.(need_color_links_used.or.need_charge_links_used)) then
C just return 0
         bsv_wgt=0d0
         virt_wgt=0d0
         born_wgt=0d0
         goto 999
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
         write(*,*)'Error in bornsoftvirtual: inconsistent shat'
         write(*,*)shattmp,shat
         stop
      endif

      call sborn(p_born,wgt1)

c Born contribution:
      bsv_wgt=wgt1
      born_wgt=wgt1
      virt_wgt=0d0
      avv_wgt=0d0 
      amp_split_born(1:amp_split_size)=amp_split(1:amp_split_size)
      amp_split_bsv(1:amp_split_size)=amp_split(1:amp_split_size)

      if (abrv.eq.'born') goto 549
      if (abrv.eq.'virt') goto 547

c Q contribution eq 5.5 and 5.6 of FKS
C loop over QCD/QED (iord=1,2 respectively)
      do iord= 1,2
         Q=0d0
C skip what we don't need
         if (iord.eq.1) ipos_ord = qcd_pos
         if (iord.eq.2) ipos_ord = qed_pos
         if (.not.split_type_used(ipos_ord)) cycle
         do i=1 ,nexternal
            if (i.ne.i_fks .and. pmass(i).eq.ZERO) then
c set the various color factors according to the 
c type of the leg
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               else
                  aj=-1
               endif
               Ej=p(0,i)
               
               if (ipos_ord.eq.qcd_pos) then
C     set colour factors
                  if (aj.eq.-1) cycle
                  c_used = c(aj)
                  gamma_used = gamma(aj)
                  gammap_used = gammap(aj)
               else if (ipos_ord.eq.qed_pos) then
C     skip particles which are not photons or charged
                  if (particle_charge(i).eq.0d0.and.pdg_type(i).ne.22)
     $                 cycle
C     set charge factors
                  if (pdg_type(i).eq.22.and..not.particle_tag(i)) then
                     c_used = 0d0
                     gamma_used = gamma_ph
                     gammap_used = gammap_ph
                  else
                     c_used = particle_charge(i)**2
                     gamma_used = 3d0/2d0 * particle_charge(i)**2
                     gammap_used = (13d0/2d0 - 2d0 * pi**2 / 3d0) *
     $                    particle_charge(i)**2
                  endif
               endif

               if (i.gt.nincoming) then 
C Q terms for final state partons
                  if(abrv.ne.'virt')then
c 1+2+3+4
                     Q = Q+gammap_used
     &                    -dlog(shat*deltaO/2d0/QES2)*( gamma_used-
     &                    2d0*c_used*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &                    +2d0*c_used*( dlog(2d0*Ej/sqrtshat)**2
     &                    -dlog(xicut_used)**2 )
     &                    -2d0*gamma_used*dlog(2d0*Ej/sqrtshat)
                  else
                     write(*,*)'Error in bornsoftvirtual'
                     write(*,*)'abrv in Q:',abrv
                     stop
                  endif

               else
C Q terms for initial state partons
                  if(abrv.ne.'virt')then
c 1+2+3+4
                     Q=Q-dlog(q2fact(i)/QES2)*(
     &                    gamma_used+2d0*c_used*dlog(xicut_used))
                  else
                     write(*,*)'Error in bornsoftvirtual'
                     write(*,*)'abrv in Q:',abrv
                     stop
                  endif
               endif
            endif
         enddo
C end of the external particle loop
         if (ipos_ord.eq.qcd_pos) then 
            bsv_wgt = bsv_wgt+aso2pi*Q*dble(ans_cnt(1,qcd_pos))
            amp_split_bsv(1:amp_split_size)=
     $           amp_split_bsv(1:amp_split_size)+aso2pi*Q
     $           *dble(amp_split_cnt(1:amp_split_size,1,qcd_pos))
         endif
         if (ipos_ord.eq.qed_pos) then
            bsv_wgt = bsv_wgt+aeo2pi*Q*dble(ans_cnt(1,qed_pos))
            amp_split_bsv(1:amp_split_size)=
     $           amp_split_bsv(1:amp_split_size)+aeo2pi*Q
     $           *dble(amp_split_cnt(1:amp_split_size,1,qed_pos))
         endif
      enddo

c     If doing MC over helicities, must sum over the two
c     helicity contributions for the Q-terms of collinear limit.
 547  continue
      if (abrv.eq.'virt') goto 548
c
c I(reg) terms, eq 5.5 of FKS
      nFKSprocess_save = nFKSprocess
      do iord = 1, nsplitorders
         if (iord.eq.qcd_pos) then
            if (.not. need_color_links_used) cycle
            need_color_links=need_color_links_used
            need_charge_links=.false.
            nFKSprocess=nFKSprocess_col
         else if (iord.eq.qed_pos) then
            if (.not. need_charge_links_used) cycle
            need_charge_links=need_charge_links_used
            need_color_links=.false.
            nFKSprocess=nFKSprocess_chg
         else
            cycle
         endif
C setup the fks i/j info
         call fks_inc_chooser()
C the following call to born is to setup the goodhel(nfksprocess)
         call sborn(p_born,wgt1)
         contr=0d0
         do i=1,fks_j_from_i(i_fks,0)
            do j=1,i
               m=fks_j_from_i(i_fks,i)
               n=fks_j_from_i(i_fks,j)
               if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)).and.
     &              n.ne.i_fks.and.m.ne.i_fks) then
c To be sure that color-correlated Borns work well, we need to have
c *always* a call to sborn(p_born,wgt) just before. This is okay,
c because there is a call above in this subroutine
C wgt includes the gs/w^2 
                  call sborn_sf(p_born,m,n,wgt)
                  if (wgt.ne.0d0) then
                     call eikonal_Ireg(p,m,n,xicut_used,eikIreg)
                     contr=contr+wgt*eikIreg
                     do k=1,amp_split_size
                        amp_split_bsv(k) = amp_split_bsv(k) - 2d0 *
     $                       eikIreg * oneo8pi2 * amp_split_soft(k)
                     enddo
                  endif
               endif
            enddo
         enddo

C WARNING: THE FACTOR -2 BELOW COMPENSATES FOR THE MISSING -2 IN THE
C COLOUR LINKED BORN -- SEE ALSO SBORNSOFT().
C If the colour-linked Borns were normalized as reported in the paper
c we should set
c   bsv_wgt=bsv_wgt+ao2pi*contr  <-- DO NOT USE THIS LINE
c
         bsv_wgt=bsv_wgt-2*oneo8pi2*contr
      enddo

C set back the fks i/j info as prior to enter this function
      nFKSprocess = nFKSprocess_save
      call fks_inc_chooser()

 548  continue
c Finite part of one-loop corrections
c convert to Binoth Les Houches Accord standards
      virt_wgt=0d0

      call sborn(p_born, wgt1)
      ! use the amp_split_cnt as the born to approximate the virtual
      ! check which one of the two (QCD, QED) is !=0
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     THIS IS DANGEROUS: if these are not always the same for all
c     events, the whole virt_trics doesn't work and gives wrong results!
c     CHECK THIS.
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      do iamp=1, amp_split_size
         amp_split_virt(iamp)=0d0
         amp_split_born_for_virt(iamp)=0d0
         if (dble(amp_split_cnt(iamp,1,qcd_pos)).ne.0d0) then
            amp_split_born_for_virt(iamp)=dble(amp_split_cnt(iamp,1
     $           ,qcd_pos))
         else if (dble(amp_split_cnt(iamp,1,qed_pos)).ne.0d0) then
            amp_split_born_for_virt(iamp)=dble(amp_split_cnt(iamp,1
     $           ,qed_pos))
         endif
      enddo

      if (fold.eq.0) then
         if ((ran2().le.virtual_fraction(ichan) .and.
     $        abrv(1:3).ne.'nov').or.abrv(1:4).eq.'virt') then
            call cpu_time(tBefore)
            Call BinothLHA(p_born,born_wgt,virt_wgt)
            do iamp=1,amp_split_size
               amp_split_virt(iamp)=amp_split_finite_ML(iamp)
            enddo
            virtual_over_born=virt_wgt/born_wgt
            if (ickkw.ne.-1) then
               virt_wgt = 0d0
               do iamp=1,amp_split_size
                  if (amp_split_virt(iamp).eq.0d0) cycle
                  if (use_poly_virtual) then
                     amp_split_virt(iamp)=amp_split_virt(iamp)-
     $                    polyfit(iamp)
     $                    *amp_split_born_for_virt(iamp)
                  else
                     amp_split_virt(iamp)=amp_split_virt(iamp)-
     $                    average_virtual(iamp,ichan)
     $                     *amp_split_born_for_virt(iamp)
                  endif
                  virt_wgt = virt_wgt + amp_split_virt(iamp)
               enddo
            endif
            if (abrv.ne.'virt') then
               virt_wgt=virt_wgt/virtual_fraction(ichan)
               do iamp=1,amp_split_size
                  amp_split_virt(iamp)=amp_split_virt(iamp)
     &                 /virtual_fraction(ichan)
               enddo
            endif
            call cpu_time(tAfter)
            tOLP=tOLP+(tAfter-tBefore)
         endif
         virt_wgt_save=virt_wgt
         amp_split_virt_save(1:amp_split_size)=
     $        amp_split_virt(1:amp_split_size)
      elseif(fold.eq.1) then
         virt_wgt=virt_wgt_save
         amp_split_virt(1:amp_split_size)=
     $        amp_split_virt_save(1:amp_split_size)
      endif
      if (abrv(1:4).ne.'virt' .and. ickkw.ne.-1) then
         if (use_poly_virtual) then
            avv_wgt=polyfit(0)*born_wgt
            do iamp=1, amp_split_size
               if (amp_split_born_for_virt(iamp).eq.0d0) cycle
               amp_split_avv(iamp)= polyfit(iamp)
     $              *amp_split_born_for_virt(iamp)
            enddo
         else
            avv_wgt=average_virtual(0,ichan)*born_wgt
            do iamp=1, amp_split_size
               if (amp_split_born_for_virt(iamp).eq.0d0) cycle
               amp_split_avv(iamp)= average_virtual(iamp,ichan)
     $              *amp_split_born_for_virt(iamp)
            enddo
         endif
      endif

c eq.(MadFKS.C.13)
      if(abrv.ne.'virt')then
         ! this is to update the amp_split array
         call sborn(p_born,wgt1)
         bsv_wgt_mufoqes=0d0
         do iamp=1,amp_split_size
            if (dble(amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
            call amp_split_pos_to_orders(iamp, orders)
            wgtcpower=0d0
            if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
            contr_mufoqes=2*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $           +ren_group_coeff*wgtcpower)*log(q2fact(1)/QES2)*aso2pi
     $           *dble(amp_split_cnt(iamp,1,qcd_pos))
            amp_split_bsv(iamp) = amp_split_bsv(iamp)+contr_mufoqes
            bsv_wgt_mufoqes=bsv_wgt_mufoqes+contr_mufoqes
         enddo
         bsv_wgt=bsv_wgt+bsv_wgt_mufoqes
      endif

c  eq.(MadFKS.C.14)
      if(abrv(1:2).ne.'vi')then
         bsv_wgt_mufomur=0d0
         do iamp=1,amp_split_size
            if (dble(amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
            call amp_split_pos_to_orders(iamp, orders)
            wgtcpower=0d0
            if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
            contr_mufomur=-2*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $           +ren_group_coeff*wgtcpower)*log(q2fact(1)/scale**2)
     $           *aso2pi*dble(amp_split_cnt(iamp,1,qcd_pos))
            amp_split_bsv(iamp) = amp_split_bsv(iamp)+contr_mufomur
            bsv_wgt_mufomur=bsv_wgt_mufomur+contr_mufomur
         enddo
         bsv_wgt=bsv_wgt+bsv_wgt_mufomur
      endif

 549  continue

      wgtwnstmpmuf=0.d0
      wgtnstmp=0d0
      wgtwnstmpmur=0.d0
      amp_split_wgtnstmp(1:amp_split_size)=0d0
      amp_split_wgtwnstmpmuf(1:amp_split_size)=0d0
      amp_split_wgtwnstmpmur(1:amp_split_size)=0d0

      if(abrv.ne.'born' .and. abrv.ne.'grid')then
         call sborn(p_born,wgt1)
         if(abrv(1:2).eq.'vi')then
            wgtwnstmpmur=0.d0
         else
C loop over QCD/QED (iord=1,2 respectively)
            do iord= 1,2
C skip what we don't need
               if (iord.eq.1) ipos_ord = qcd_pos
               if (iord.eq.2) ipos_ord = qed_pos
               if (.not.split_type_used(ipos_ord)) cycle
               do i=1,nincoming
                  if (particle_type(i).eq.8) then
                     aj=0
                  elseif(abs(particle_type(i)).eq.3) then
                     aj=1
                  else
                     aj=-1
                  endif
                  if (ipos_ord.eq.qcd_pos) then
C     set colour factors
                     if (aj.eq.-1) cycle
                     c_used = c(aj)
                     gamma_used = gamma(aj)
                     gammap_used = gammap(aj)
                  else if (ipos_ord.eq.qed_pos) then
C     skip particles which are not photons or charged
                     if (particle_charge(i).eq.0d0.and.pdg_type(i).ne.22)
     $                    cycle
C     set charge factors
                     if (pdg_type(i).eq.22.and..not.particle_tag(i)) then
                        c_used = 0d0
                        gamma_used = gamma_ph
                        gammap_used = gammap_ph
                     else
                        c_used = particle_charge(i)**2
                        gamma_used = 3d0/2d0 * particle_charge(i)**2
                        gammap_used = (13d0/2d0 - 2d0 * pi**2 / 3d0) *
     $                       particle_charge(i)**2
                     endif
                  endif
                  do iamp=1,amp_split_size
                     if (dble(amp_split_cnt(iamp,1,ipos_ord)).eq.0d0)
     $                    cycle
                     if (ipos_ord.eq.qcd_pos) then
                        coupl_wgtwnstmpmuf=aso2pi
                     else if (ipos_ord.eq.qed_pos) then
                        coupl_wgtwnstmpmuf=aeo2pi
                     endif
                     amp_split_wgtwnstmpmuf(iamp)
     $                    =amp_split_wgtwnstmpmuf(iamp)-(gamma_used+2d0
     $                    *c_used*dlog(xicut_used))
     $                    *dble(amp_split_cnt(iamp,1,ipos_ord))
     $                    *coupl_wgtwnstmpmuf
                  enddo
               enddo            !end loop i=1,nincoming
            enddo               !end loop iord=1,2
            do iamp=1,amp_split_size
               if (dble(amp_split_cnt(iamp,1,qcd_pos)).eq.0d0) cycle
               call amp_split_pos_to_orders(iamp, orders)
               wgtcpower=0d0
               if (cpower_pos.gt.0) wgtcpower=dble(orders(cpower_pos))
               amp_split_wgtwnstmpmur(iamp)=dble(amp_split_cnt(iamp,1
     $              ,qcd_pos))*2d0*pi*(beta0*dble(orders(qcd_pos)-2)/2d0
     $              +ren_group_coeff*wgtcpower)*aso2pi
            enddo
         endif
c bsv_wgt here always contains the Born; must subtract it, since 
c we need the pure NLO terms only
         amp_split_wgtnstmp(1:amp_split_size) =
     $        amp_split_bsv(1:amp_split_size)
     $        -amp_split_born(1:amp_split_size)-log(q2fact(1)/QES2)
     $        *amp_split_wgtwnstmpmuf(1:amp_split_size)-log(scale**2
     $        /QES2)*amp_split_wgtwnstmpmur(1:amp_split_size)
      endif

      amp_split(1:amp_split_size)=amp_split_bsv(1:amp_split_size)

      if (abrv(1:2).eq.'vi') then
         bsv_wgt=bsv_wgt-born_wgt
         born_wgt=0d0
      endif

      if (ComputePoles) then
         call sborn(p_born,wgt1)

         print*,"           "
         write(*,123)((p(i,j),i=0,3),j=1,nexternal)
         xmu2=q2fact(1)
         call getpoles(p,xmu2,double,single,fksprefact)
         print*,"BORN",born_wgt!/conv
         print*,"DOUBLE",double
         print*,"SINGLE",single
c         print*,"LOOP",virt_wgt!/born_wgt/ao2pi*2d0
c         print*,"LOOP2",(virtcor+born_wgt*4d0/3d0-double*pi**2/6d0)
c         stop
 123     format(4(1x,d22.16))
      endif

 999  continue
      return
      end



      subroutine eikonal_Ireg(p,m,n,xicut_used,eikIreg)
      implicit none
      double precision zero,pi,pi2
      parameter (zero=0.d0)
      parameter (pi=3.1415926535897932385d0)
      parameter (pi2=pi**2)
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal),xicut_used,eikIreg
      integer m,n

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      character*4 abrv
      common /to_abrv/ abrv

      double precision Ei,Ej,kikj,rij,tmp,xmj,betaj,betai,xmi2,xmj2,
     # vij,xi0,alij,tHVvl,tHVv,arg1,arg2,arg3,arg4,xi1a,xj1a,
     # dot,ddilog
      external dot,ddilog

      double precision pmass(nexternal)
      include "pmass.inc"

      tmp=0.d0
      if(pmass(m).eq.0.d0.and.pmass(n).eq.0.d0)then
        if(m.eq.n)then
          write(*,*)'Error #2 in eikonal_Ireg',m,n
          stop
        endif
        Ei=p(0,n)
        Ej=p(0,m)
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)
        if(abs(rij-1.d0).gt.1.d-6)then
          if(abrv.ne.'virt')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2+
     #          dlog(xicut_used**2*shat/QES2)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          else
             write(*,*)'Error #11 in eikonal_Ireg',abrv
             stop
          endif
        else
          if(abrv.ne.'virt')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2-pi2/6.d0
          else
             write(*,*)'Error #12 in eikonal_Ireg',abrv
             stop
          endif
        endif
      elseif( (pmass(m).ne.0.d0.and.pmass(n).eq.0.d0) .or.
     #        (pmass(m).eq.0.d0.and.pmass(n).ne.0.d0) )then
        if(m.eq.n)then
          write(*,*)'Error #3 in eikonal_Ireg',m,n
          stop
        endif
        if(pmass(m).ne.0.d0.and.pmass(n).eq.0.d0)then
          Ei=p(0,n)
          Ej=p(0,m)
          xmj=pmass(m)
          betaj=sqrt(1-xmj**2/Ej**2)
        else
          Ei=p(0,m)
          Ej=p(0,n)
          xmj=pmass(n)
          betaj=sqrt(1-xmj**2/Ej**2)
        endif
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)

        if(abrv.ne.'virt')then
c 1+2+3+4
          tmp=dlog(xicut_used)*( dlog(xicut_used*shat/QES2)+
     #                           2*dlog(kikj/(xmj*Ei)) )-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))-pi2/12.d0+
     #        1/4.d0*dlog(shat/QES2)**2-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        else
           write(*,*)'Error #13 in eikonal_Ireg',abrv
           stop
        endif
      elseif(pmass(m).ne.0.d0.and.pmass(n).ne.0.d0)then
        if(n.eq.m)then
          Ei=p(0,n)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          if(abrv.ne.'virt')then
c 1+2+3+4
            if (betai.gt.1d-6) then
               tmp=dlog(xicut_used**2*shat/QES2)-
     &              1/betai*dlog((1+betai)/(1-betai))
            else
               tmp=dlog(xicut_used**2*shat/QES2)-
     &              2d0*(1d0+betai**2/3d0+betai**4/5d0)
            endif
          else
             write(*,*)'Error #14 in eikonal_Ireg',abrv
             stop
          endif
        else
          Ei=p(0,n)
          Ej=p(0,m)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          betaj=sqrt(1-pmass(m)**2/Ej**2)
          xmi2=pmass(n)**2
          xmj2=pmass(m)**2
          kikj=dot(p(0,n),p(0,m))
          vij=sqrt(1-xmi2*xmj2/kikj**2)
          alij=kikj*(1+vij)/xmi2
          tHVvl=(alij**2*xmi2-xmj2)/2.d0
          tHVv=tHVvl/(alij*Ei-Ej)
          arg1=alij*Ei
          arg2=arg1*betai
          arg3=Ej
          arg4=arg3*betaj
          xi0=1/vij*log((1+vij)/(1-vij))
          xi1a=kikj**2*(1+vij)/xmi2*( xj1a(arg1,arg2,tHVv,tHVvl)-
     #                                xj1a(arg3,arg4,tHVv,tHVvl) )

          if(abrv.ne.'virt')then
c 1+2+3+4
            tmp=1/2.d0*xi0*dlog(xicut_used**2*shat/QES2)+1/2.d0*xi1a
          else
             write(*,*)'Error #15 in eikonal_Ireg',abrv
             stop
          endif
        endif
      else
        write(*,*)'Error #4 in eikonal_Ireg',m,n,pmass(m),pmass(n)
        stop
      endif
      eikIreg=tmp
      return
      end


      function xj1a(x,y,tHVv,tHVvl)
      implicit none
      real*8 xj1a,x,y,tHVv,tHVvl,ddilog
      external ddilog
c
      xj1a=1/(2*tHVvl)*( dlog((x-y)/(x+y))**2+4*ddilog(1-(x+y)/tHVv)+
     #                   4*ddilog(1-(x-y)/tHVv) )
      return
      end


      FUNCTION DDILOG(X)
*
* $Id: imp64.inc,v 1.1.1.1 1996/04/01 15:02:59 mclareni Exp $
*
* $Log: imp64.inc,v $
* Revision 1.1.1.1  1996/04/01 15:02:59  mclareni
* Mathlib gen
*
*
* imp64.inc
*
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(0:19)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (PI3 = PI**2/3, PI6 = PI**2/6, PI12 = PI**2/12)
      DATA C( 0) / 0.42996 69356 08136 97D0/
      DATA C( 1) / 0.40975 98753 30771 05D0/
      DATA C( 2) /-0.01858 84366 50145 92D0/
      DATA C( 3) / 0.00145 75108 40622 68D0/
      DATA C( 4) /-0.00014 30418 44423 40D0/
      DATA C( 5) / 0.00001 58841 55418 80D0/
      DATA C( 6) /-0.00000 19078 49593 87D0/
      DATA C( 7) / 0.00000 02419 51808 54D0/
      DATA C( 8) /-0.00000 00319 33412 74D0/
      DATA C( 9) / 0.00000 00043 45450 63D0/
      DATA C(10) /-0.00000 00006 05784 80D0/
      DATA C(11) / 0.00000 00000 86120 98D0/
      DATA C(12) /-0.00000 00000 12443 32D0/
      DATA C(13) / 0.00000 00000 01822 56D0/
      DATA C(14) /-0.00000 00000 00270 07D0/
      DATA C(15) / 0.00000 00000 00040 42D0/
      DATA C(16) /-0.00000 00000 00006 10D0/
      DATA C(17) / 0.00000 00000 00000 93D0/
      DATA C(18) /-0.00000 00000 00000 14D0/
      DATA C(19) /+0.00000 00000 00000 02D0/
      IF(X .EQ. 1) THEN
       H=PI6
      ELSEIF(X .EQ. -1) THEN
       H=-PI12
      ELSE
       T=-X
       IF(T .LE. -2) THEN
        Y=-1/(1+T)
        S=1
        A=-PI3+HF*(LOG(-T)**2-LOG(1+1/T)**2)
       ELSEIF(T .LT. -1) THEN
        Y=-1-T
        S=-1
        A=LOG(-T)
        A=-PI6+A*(A+LOG(1+1/T))
       ELSE IF(T .LE. -HF) THEN
        Y=-(1+T)/T
        S=1
        A=LOG(-T)
        A=-PI6+A*(-HF*A+LOG(1+T))
       ELSE IF(T .LT. 0) THEN
        Y=-T/(1+T)
        S=-1
        A=HF*LOG(1+T)**2
       ELSE IF(T .LE. 1) THEN
        Y=T
        S=1
        A=0
       ELSE
        Y=1/T
        S=-1
        A=PI6+HF*LOG(T)**2
       ENDIF
       H=Y+Y-1
       ALFA=H+H
       B1=0
       B2=0
       DO 1 I = 19,0,-1
       B0=C(I)+ALFA*B1-B2
       B2=B1
    1  B1=B0
       H=-(S*(B0-H*B2)+A)
      ENDIF
      DDILOG=H
      RETURN
      END


      subroutine getpoles(p,xmu2,double,single,fksprefact)
c Returns the residues of double and single poles according to 
c eq.(B.1) and eq.(B.2) if fksprefact=.true.. When fksprefact=.false.,
c the prefactor (mu2/Q2)^ep in eq.(B.1) is expanded, and giving an
c extra contribution to the single pole
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision particle_charge(nexternal), particle_charge_born(nexternal-1)
      common /c_charges/particle_charge
      common /c_charges_born/particle_charge_born
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal),xmu2,double,single
      logical fksprefact
      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      common/fks_colors/c,gamma,gammap,gamma_ph,gammap_ph
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision wgt1
      double precision born,wgt,kikj,dot,vij,aso2pi,aeo2pi
      double precision contr1, contr2
      integer aj,i,j,m,n,ilink,k
      double precision pmass(nexternal),zero,pi
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0d0)
      include 'orders.inc'
      double precision amp_split_poles_FKS(amp_split_size,2)
      common /to_amp_split_poles_FKS/amp_split_poles_FKS
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft
      complex*16 ans_cnt(2, nsplitorders)
      common /c_born_cnt/ ans_cnt
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      double precision oneo8pi2
      parameter(oneo8pi2 = 1d0/(8d0*pi**2))
      include "nFKSconfigs.inc"
      INTEGER nFKSprocess, nFKSprocess_save, nFKSprocess_col, nFKSprocess_chg
      COMMON/c_nFKSprocess/nFKSprocess
      logical need_color_links_used, need_charge_links_used
      double precision soft_fact

      include "pmass.inc"

      nFKSprocess_col = 0
      nFKSprocess_chg = 0

      need_color_links_used = .false.
      need_charge_links_used = .false.
      
C check if any real emission need cahrge/color links
      nFKSprocess_save = nFKSprocess
      do nFKSprocess = 1, FKS_configs
        call fks_inc_chooser()
        need_color_links_used = need_color_links_used .or. need_color_links
        need_charge_links_used = need_charge_links_used .or. need_charge_links
C keep track of which FKS configuration actually needs color/charge
C links
        if (need_color_links.and.nFKSprocess_col.eq.0)
     1          nFKSprocess_col = nFKSprocess
        if (need_charge_links.and.nFKSprocess_chg.eq.0)
     1          nFKSprocess_chg = nFKSprocess
      enddo
      nFKSprocess = nFKSprocess_save
      call fks_inc_chooser()

      double=0.d0
      single=0.d0
      ! reset the amp_split_poles_FKS
      do i=1,amp_split_size
        amp_split_poles_FKS(i,1)=0d0
        amp_split_poles_FKS(i,2)=0d0
      enddo
      aso2pi=g**2/(8d0*pi**2)
      aeo2pi=dble(gal(1))**2/(8d0*pi**2)
      call sborn(p_born,wgt1)
c QCD Born terms
      contr1 = 0d0
      contr2 = 0d0
      born=dble(ans_cnt(1,qcd_pos))
      do i=1,nexternal
        if(i.ne.i_fks .and. particle_type(i).ne.1)then
          if (particle_type(i).eq.8) then
             aj=0
          elseif(abs(particle_type(i)).eq.3) then
             aj=1
          endif
          if(pmass(i).eq.ZERO)then
            contr2=contr2-c(aj)
            contr1=contr1-gamma(aj)
          else
            contr1=contr1-c(aj)
          endif
        endif
      enddo

      double=double+contr2*born*aso2pi
      single=single+contr1*born*aso2pi

      do i=1,amp_split_size
        amp_split_poles_FKS(i,1) = amp_split_poles_FKS(i,1)+
     %      dble(amp_split_cnt(i,1,qcd_pos))*contr1*aso2pi
        amp_split_poles_FKS(i,2) = amp_split_poles_FKS(i,2)+
     %      dble(amp_split_cnt(i,1,qcd_pos))*contr2*aso2pi
      enddo

c QED Born terms
      contr1 = 0d0
      contr2 = 0d0
      born=dble(ans_cnt(1,qed_pos))
      do i=1,nexternal
        if(i.ne.i_fks.and.(particle_charge(i).ne.0d0.or.pdg_type(i).eq.22))then
          if(pmass(i).eq.ZERO)then
            if (pdg_type(i).ne.22) then
              contr2=contr2-particle_charge(i)**2
              contr1=contr1-3d0/2d0*particle_charge(i)**2
            elseif (.not.particle_tag(i)) then
              contr1=contr1-gamma_ph
            endif
          else
            contr1=contr1-particle_charge(i)**2
          endif
        endif
      enddo

      double=double+contr2*born*aeo2pi
      single=single+contr1*born*aeo2pi

      do i=1,amp_split_size
        amp_split_poles_FKS(i,1) = amp_split_poles_FKS(i,1)+
     %      dble(amp_split_cnt(i,1,qed_pos))*contr1*aeo2pi
        amp_split_poles_FKS(i,2) = amp_split_poles_FKS(i,2)+
     %      dble(amp_split_cnt(i,1,qed_pos))*contr2*aeo2pi
      enddo

c Colour and charge-linked Born terms
      nFKSprocess_save = nFKSprocess
      do ilink = 1, 2
        if (ilink.eq.1) then
          if (.not. need_color_links_used) cycle
          need_color_links = .true.
          need_charge_links = .false.
          nFKSprocess=nFKSprocess_col
        else
          if (.not. need_charge_links_used) cycle
          need_color_links = .false.
          need_charge_links = .true.
          nFKSprocess=nFKSprocess_chg
        endif

C setup the fks i/j info
        call fks_inc_chooser()
C the following call to born is to setup the goodhel(nfksprocess)
        call sborn(p_born,wgt1)

        contr1=0d0
        do i=1,fks_j_from_i(i_fks,0)
          do j=1,i
            m=fks_j_from_i(i_fks,i)
            n=fks_j_from_i(i_fks,j)
            if( m.ne.n .and. n.ne.i_fks .and. m.ne.i_fks )then
C wgt includes the gs/w^2 factor
              call sborn_sf(p_born,m,n,wgt)
c The factor -2 compensate for that missing in sborn_sf
              wgt=-2d0*wgt
              if(wgt.ne.0.d0)then
                if(pmass(m).eq.zero.and.pmass(n).eq.zero)then
                  kikj=dot(p(0,n),p(0,m))
                  soft_fact=dlog(2d0*kikj/QES2)
                elseif(pmass(m).ne.zero.and.pmass(n).eq.zero)then
                  kikj=dot(p(0,n),p(0,m))
                  soft_fact=-0.5d0*dlog(pmass(m)**2/QES2)+dlog(2d0*kikj/QES2)
                elseif(pmass(m).eq.zero.and.pmass(n).ne.zero)then
                  kikj=dot(p(0,n),p(0,m))
                  soft_fact=-0.5d0*dlog(pmass(n)**2/QES2)+dlog(2d0*kikj/QES2)
                elseif(pmass(m).ne.zero.and.pmass(n).ne.zero)then
                  kikj=dot(p(0,n),p(0,m))
                  vij=dsqrt(1d0-(pmass(n)*pmass(m)/kikj)**2)
                  if (vij .gt. 1d-6) then
                    soft_fact=0.5d0*1/vij*log((1+vij)/(1-vij))
                  else
                    soft_fact=(1d0+vij**2/3d0+vij**4/5d0)
                  endif
                else
                  write(*,*)'Error in getpoles',i,j,n,m,pmass(n),pmass(m)
                  stop
                endif
                contr1=contr1+soft_fact*wgt
                do k=1,amp_split_size
                  amp_split_poles_FKS(k,1)=amp_split_poles_FKS(k,1)+
     $             amp_split_soft(k)*(-2d0)*soft_fact*oneo8pi2
                enddo
              endif
            endif
          enddo
        enddo
        single=single+contr1*oneo8pi2
      enddo

C restore need_color/charge_links
      nFKSprocess = nFKSprocess_save
      call fks_inc_chooser()

      if(.not.fksprefact)single=single+double*dlog(xmu2/QES2)
c
      return
      end


      subroutine setfksfactor(match_to_shower)
      use weight_lines
      use extra_weights
      use mint_module
      implicit none
      
      double precision CA,CF, PI
      parameter (CA=3d0,CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1),gamma_ph,gammap_ph
      common/fks_colors/c,gamma,gammap,gamma_ph,gammap_ph

      double precision beta0,ren_group_coeff
      common/cbeta0/beta0,ren_group_coeff

      logical softtest,colltest
      common/sctests/softtest,colltest

      integer config_fks,i,j,fac1,fac2,kchan
      logical match_to_shower

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                  fkssymmetryfactorDeg,ngluons,nquarks,nphotons

      double precision iden_comp
      common /c_iden_comp/iden_comp
      
      include 'coupl.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'fks_powers.inc'
      include 'nFKSconfigs.inc'
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include 'run.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig

      logical firsttime,firsttime_nFKSprocess(fks_configs)
      data firsttime,firsttime_nFKSprocess/.true.,fks_configs*.true./

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision diagramsymmetryfactor_save(maxchannels)
      save diagramsymmetryfactor_save
      double precision diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      character*1 integrate
      integer i_fks,j_fks
      double precision dfac1
      common/fks_indices/i_fks,j_fks
      integer fac_i,fac_j,i_fks_pdg,j_fks_pdg,iden(nexternal)

      integer fac_i_FKS(fks_configs),fac_j_FKS(fks_configs)
     &     ,i_type_FKS(fks_configs),j_type_FKS(fks_configs)
     &     ,m_type_FKS(fks_configs),ngluons_FKS(fks_configs)
     &     ,nphotons_FKS(fks_configs),iden_real_FKS(fks_configs)
     &     ,iden_born_FKS(fks_configs)
      double precision ch_i_FKS(fks_configs),ch_j_FKS(fks_configs)
     &     ,ch_m_FKS(fks_configs)
      save fac_i_FKS,fac_j_FKS,i_type_FKS,j_type_FKS,m_type_FKS
     &     ,ngluons_FKS,ch_i_FKS,ch_j_FKS,ch_m_FKS,nphotons_FKS
     &     ,iden_real_FKS,iden_born_FKS

      character*13 filename

      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      double precision particle_charge(nexternal), particle_charge_born(nexternal-1)
      common /c_charges/particle_charge
      common /c_charges_born/particle_charge_born
      logical particle_tag(nexternal)
      common /c_particle_tag/particle_tag
      double precision zero
      parameter (zero=0d0)

      softtest=.false.
      colltest=.false.

      if (j_fks.gt.nincoming)then
         delta_used=deltaO
      else
         delta_used=deltaI
      endif
      
      xicut_used=xicut
      xiScut_used=xiScut
      if( nbody .or. (abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &     abrv(1:2).eq.'vi') )then
        xiBSVcut_used=1d0
      else
        xiBSVcut_used=xiBSVcut
      endif

      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=( 67d0/9d0 - 2d0*PI**2/3d0 )*CA - 23d0/18d0*Nf
      gammap(1)=( 13/2d0 - 2d0*PI**2/3d0 )*CF
C photon-related factors
      gamma_ph=0d0
      do i = 1, nint(nf)
        if (mod(i,2).eq.0) then
          ! u-type massless quarks
          gamma_ph=gamma_ph-4d0/9d0
        else
          ! d-type massless quarks
          gamma_ph=gamma_ph-1d0/9d0
        endif
      enddo
      ! then add the contribution from massless leptons
      gammap_ph=(gamma_ph*3d0-Nl)*23d0/9d0
      gamma_ph=(gamma_ph*3d0-Nl)*2d0/3d0
            
c Beta_0 defined according to (MadFKS.C.5)
      beta0=gamma(0)/(2*pi)
c ren_group_coeff defined accordingly
      ren_group_coeff=ren_group_coeff_in/(2*pi)

      if (firsttime_nFKSprocess(nFKSprocess)) then
         firsttime_nFKSprocess(nFKSprocess)=.false.
c---------------------------------------------------------------------
c              Symmetry Factors
c---------------------------------------------------------------------
c fkssymmetryfactor:
c Calculate the FKS symmetry factors to be able to reduce the number
c of directories to (maximum) 4 (neglecting quark flavors):
c     1. i_fks=gluon, j_fks=gluon 
c     2. i_fks=gluon, j_fks=quark
c     3. i_fks=gluon, j_fks=anti-quark
c     4. i_fks=quark, j_fks=anti-quark (or vice versa).
c This sets the fkssymmetryfactor (in which the quark flavors are taken
c into account) for the subtracted reals.
c
c fkssymmetryfactorBorn:
c Note that in the Born's included here, the final state identical
c particle factor is set equal to the identical particle factor
c for the real contribution to be able to get the correct limits for the
c subtraction terms and the approximated real contributions.
c However when we want to calculate the Born contributions only, we
c have to correct for this difference. Since we only include the Born
c related to a soft limit (this uniquely defines the Born for a given real)
c the difference is always n!/(n-1)!=n, where n is the number of final state
c gluons in the real contribution.
c
c Furthermore, because we are not integrating all the directories, we also
c have to include a fkssymmetryfactor for the Born contributions. However,
c this factor is not the same as the factor defined above, because in this
c case i_fks is fixed to the extra gluon (which goes soft and defines the
c Born contribution) and should therefore not be taken into account when
c calculating the symmetry factor. Together with the factor n above this
c sets the fkssymmetryfactorBorn equal to the fkssymmetryfactor for the
c subtracted reals.
c
c We set fkssymmetryfactorBorn to zero when i_fks not a gluon
c
         i_fks_pdg=pdg_type(i_fks)
         j_fks_pdg=pdg_type(j_fks)
      
         fac_i_FKS(nFKSprocess)=0
         fac_j_FKS(nFKSprocess)=0
         do i=nincoming+1,nexternal
            if (i_fks_pdg.eq.pdg_type(i)) fac_i_FKS(nFKSprocess) =
     $           fac_i_FKS(nFKSprocess) + 1
            if (j_fks_pdg.eq.pdg_type(i)) fac_j_FKS(nFKSprocess) =
     $           fac_j_FKS(nFKSprocess) + 1
         enddo
c Overwrite if initial state singularity
         if(j_fks.le.nincoming) fac_j_FKS(nFKSprocess)=1

c i_fks and j_fks of the same type? -> subtract 1 to avoid double counting
         if (j_fks.gt.nincoming .and. i_fks_pdg.eq.j_fks_pdg)
     $        fac_j_FKS(nFKSprocess)=fac_j_FKS(nFKSprocess)-1

c THESE TESTS WORK ONLY FOR FINAL STATE SINGULARITIES
C MZ the test may be removed sooner or later
         if (j_fks.gt.nincoming) then
            if ( i_fks_pdg.eq.j_fks_pdg .and. i_fks_pdg.ne.21) then
               write (*,*) 'ERROR, if PDG type of i_fks and j_fks '//
     &              'are equal, they MUST be gluons',
     &              i_fks,j_fks,i_fks_pdg,j_fks_pdg
               stop
            elseif(abs(particle_type(i_fks)).eq.3) then
               if ( particle_type(i_fks).ne.-particle_type(j_fks) .or.
     &              pdg_type(i_fks).ne.-pdg_type(j_fks)) then
                  write (*,*) 'ERROR, if i_fks is a color triplet,'//
     &                 ' j_fks must be its anti-particle,'//
     &                 ' or an initial state gluon.',
     &                 i_fks,j_fks,particle_type(i_fks),
     &                 particle_type(j_fks),pdg_type(i_fks),pdg_type(j_fks)
                  stop
               endif
            elseif(particle_type(i_fks).eq.1.and.abs(particle_charge(i_fks)).gt.0d0) then
               if ( particle_charge(i_fks).ne.-particle_charge(j_fks) .or.
     &              pdg_type(i_fks).ne.-pdg_type(j_fks)) then
                  write (*,*) 'ERROR, i_fks is a charged color singlet,'//
     &                 ' j_fks must be its anti-particle,'//
     &                 ' or an initial state gluon.',
     &                 i_fks,j_fks,particle_charge(i_fks),
     &                 particle_charge(j_fks),pdg_type(i_fks),pdg_type(j_fks)
                  stop
               endif
            elseif(abs(i_fks_pdg).ne.21.and.i_fks_pdg.ne.22) then ! if not already above, it MUST be a gluon or photon
               write (*,*) 'ERROR, i_fks is not a g/gamma and falls not'//
     $              ' in other categories',i_fks,j_fks,i_fks_pdg
     $              ,j_fks_pdg
               stop
            endif
         endif

         ngluons_FKS(nFKSprocess)=0
         nphotons_FKS(nFKSprocess)=0
         do i=nincoming+1,nexternal
            if (pdg_type(i).eq.21) ngluons_FKS(nFKSprocess)
     $           =ngluons_FKS(nFKSprocess)+1
            if (pdg_type(i).eq.22.and..not.particle_tag(i)) nphotons_FKS(nFKSprocess)
     $           =nphotons_FKS(nFKSprocess)+1
         enddo



c Set color types of i_fks, j_fks and fks_mother.
         i_type=particle_type(i_fks)
         j_type=particle_type(j_fks)
         ch_i=particle_charge(i_fks)
         ch_j=particle_charge(j_fks)
         call get_mother_col_charge(i_type,ch_i,j_type,ch_j,m_type,ch_m) 
         i_type_FKS(nFKSprocess)=i_type
         j_type_FKS(nFKSprocess)=j_type
         m_type_FKS(nFKSprocess)=m_type
         ch_i_FKS(nFKSprocess)=ch_i
         ch_j_FKS(nFKSprocess)=ch_j
         ch_m_FKS(nFKSprocess)=ch_m

c Compute the identical particle symmetry factor that is in the
c real-emission matrix elements.
         iden_real_FKS(nFKSprocess)=1
         do i=1,nexternal
            iden(i)=1
         enddo
         do i=nincoming+2,nexternal
            do j=nincoming+1,i-1
               if (pdg_type(j).eq.pdg_type(i)) then
                  iden(j)=iden(j)+1
                  iden_real_FKS(nFKSprocess)=
     &                 iden_real_FKS(nFKSprocess)*iden(j)
                  exit
               endif
            enddo
         enddo
c Compute the identical particle symmetry factor that is in the
c Born matrix elements.
         iden_born_FKS(nFKSprocess)=1
         call weight_lines_allocated(nexternal,max_contr,max_wgt
     $        ,max_iproc)
         call set_pdg(0,nFKSprocess)
         do i=1,nexternal
            iden(i)=1
         enddo
         do i=nincoming+2,nexternal-1
            do j=nincoming+1,i-1
               if (pdg_uborn(j,0).eq.pdg_uborn(i,0)) then
                  iden(j)=iden(j)+1
                  iden_born_FKS(nFKSprocess)=
     &                 iden_born_FKS(nFKSprocess)*iden(j)
                  exit
               endif
            enddo
         enddo
      endif

      i_type=i_type_FKS(nFKSprocess)
      j_type=j_type_FKS(nFKSprocess)
      m_type=m_type_FKS(nFKSprocess)
      ch_i=ch_i_FKS(nFKSprocess)
      ch_j=ch_j_FKS(nFKSprocess)
      ch_m=ch_m_FKS(nFKSprocess)

c Compensating factor needed in the soft & collinear counterterms for
c the fact that the identical particle symmetry factor in the Born
c matrix elements is not the one that should be used for those terms
c (should be the one in the real instead).
      iden_comp=dble(iden_born_FKS(nFKSprocess))/
     &          dble(iden_real_FKS(nFKSprocess))

      
c Set matrices used by MC counterterms
      if (match_to_shower) call set_mc_matrices

      fac_i=fac_i_FKS(nFKSprocess)
      fac_j=fac_j_FKS(nFKSprocess)
      ngluons=ngluons_FKS(nFKSprocess)
      nphotons=nphotons_FKS(nFKSprocess)
c Setup the FKS symmetry factors. 
      if (nbody.and.pdg_type(i_fks).eq.21) then
         fkssymmetryfactor=dble(ngluons)
         fkssymmetryfactorDeg=dble(ngluons)
         fkssymmetryfactorBorn=1d0
      elseif (nbody.and.pdg_type(i_fks).eq.22) then
         fkssymmetryfactor=dble(nphotons)
         fkssymmetryfactorDeg=dble(nphotons)
         fkssymmetryfactorBorn=1d0
      elseif(pdg_type(i_fks).eq.-21) then
         fkssymmetryfactor=1d0
         fkssymmetryfactorDeg=1d0
         fkssymmetryfactorBorn=1d0
      else
         fkssymmetryfactor=dble(fac_i*fac_j)
         fkssymmetryfactorDeg=dble(fac_i*fac_j)
         if (pdg_type(i_fks).eq.21) then
            fkssymmetryfactorBorn=dble(fac_i*fac_j)
         else
            fkssymmetryfactorBorn=0d0
         endif
         if (abrv.eq.'grid') then
            fkssymmetryfactorBorn=1d0
            fkssymmetryfactor=0d0
            fkssymmetryfactorDeg=0d0
         endif
      endif

      if (firsttime) then
c Check to see if this channel needs to be included in the multi-channeling
         do kchan=1,nchans
            diagramsymmetryfactor_save(kchan)=0d0
         enddo
         if (multi_channel) then
            open (unit=19,file="symfact.dat",status="old",err=14)
            i=0
            do
               i=i+1
               read (19,*,err=23,end=23) dfac1,fac2
               fac1=nint(dfac1)
               if (nint(dfac1*10)-fac1*10 .eq.2 ) then
                  i=i-1
                  cycle
               endif
               do kchan=1,nchans
                  if (i.eq.iconfigs(kchan)) then
                     if (mapconfig(iconfigs(kchan),0).ne.fac1) then
                        write (*,*) 'inconsistency in symfact.dat',i
     $                       ,kchan,iconfigs(kchan)
     $                       ,mapconfig(iconfigs(kchan),0),fac1
                        stop
                     endif
                     diagramsymmetryfactor_save(kchan)=dble(fac2)
                  endif
               enddo
            enddo
 23         continue
            close(19)
         else                   ! no multi_channel
            do kchan=1,nchans
               diagramsymmetryfactor_save(kchan)=1d0
            enddo
         endif
 12      continue
         firsttime=.false.
      endif
      diagramsymmetryfactor=diagramsymmetryfactor_save(ichan)

      return

 99   continue
      write (*,*) '"integrate.fks" or "nbodyonly.fks" not found.'
      write (*,*) 'make and run "genint_fks" first.'
      stop
 14   continue
      do kchan=1,nchans
         diagramsymmetryfactor_save(kchan)=1d0
      enddo
      goto 12
      end

      subroutine set_granny(nFKSprocess,iconf,mass_min)
c This determines of the grandmother of the FKS pair is a resonance. If
c so, set granny_is_res=.true. and also set to which internal propagator
c the grandmother corresponds (igranny) as well as the aunt (iaunt).
c This information can be used to improve the phase-space
c parametrisation.
      use mint_module
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
c arguments
      integer nFKSprocess,iconf
      double precision mass_min(-nexternal:nexternal)
c common block that is filled by this subroutine
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
c other common blocks
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c     local
      integer size
      parameter (size=fks_configs*maxchannels)
      logical firsttime_fks(fks_configs,maxchannels)
      data firsttime_fks/size*.true./
      integer i,imother
c save
      logical granny_is_res_fks(fks_configs,maxchannels)
      integer igranny_fks(fks_configs,maxchannels),iaunt_fks(fks_configs
     $     ,maxchannels)
      logical granny_chain_fks(-nexternal:nexternal,fks_configs
     $     ,maxchannels)
      save granny_is_res_fks,igranny_fks,iaunt_fks,granny_chain_fks
c itree info
      include 'born_conf.inc'
c propagator info
      double precision zero
      parameter (zero=0d0)
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      include 'coupl.inc'
      include 'born_props.inc'
c
c If it's the firsttime going into this subroutine for this nFKSprocess,
c save all the relevant information so that for later calls a simple
c copy will do.
      if (firsttime_fks(nFKSprocess,ichan)) then
         firsttime_fks(nFKSprocess,ichan)=.false.
c need to have at least 2->3 (or 1->3) process to have non-trivial
c grandmother
         if (nexternal-nincoming.lt.3) then
            igranny_fks(nFKSprocess,ichan)=0
            iaunt_fks(nFKSprocess,ichan)=0
            granny_is_res_fks(nFKSprocess,ichan)=.false.
            igranny=0
            iaunt=0
            granny_is_res=.false.
            return
c j_fks needs to be final state to have non-trivial grandmother
         elseif (j_fks.le.nincoming) then
            igranny_fks(nFKSprocess,ichan)=0
            iaunt_fks(nFKSprocess,ichan)=0
            granny_is_res_fks(nFKSprocess,ichan)=.false.
            igranny=0
            iaunt=0
            granny_is_res=.false.
            return
         endif
c determine if grandmother is an s-channel particle. If so, set igranny
c and iaunt.
         imother=min(i_fks,j_fks)
         do i=-1,-(nexternal-(2+nincoming)),-1
            if (iforest(1,i,iconf).eq.1 .or.
     &              iforest(1,i,iconf).eq.2) then
c no more s-channels, so exit the do-loop and set igranny=0
               igranny_fks(nFKSprocess,ichan)=0
               iaunt_fks(nFKSprocess,ichan)=0
               exit
            elseif (iforest(1,i,iconf).eq.imother) then
c Daughter 1 is the fks_mother.
               igranny_fks(nFKSprocess,ichan)=i
               iaunt_fks(nFKSprocess,ichan)=iforest(2,i,iconf)
               exit
            elseif (iforest(2,i,iconf).eq.imother) then
c Daughter 2 is the fks_mother.
               igranny_fks(nFKSprocess,ichan)=i
               iaunt_fks(nFKSprocess,ichan)=iforest(1,i,iconf)
               exit
            endif
         enddo
c If there is an s-channel grandmother, determine if it's a resonance by
c making sure that it's massive and has a non-zero width. In the special
c case that the grandmother is the s-hat propagator (which means that
c the process has no t-channels), set granny_is_res to false.
         if (igranny_fks(nFKSprocess,ichan).ne.0 .and.
     $        igranny_fks(nFKSprocess,ichan).ne.-(nexternal-(2+nincoming))) then
            if (pmass(igranny_fks(nFKSprocess,ichan),iconf).ne.0d0 .and.
     $           pwidth(igranny_fks(nFKSprocess,ichan),iconf).gt.0d0) then
               ! also check if the sum of all the masses of all final
               ! state particles originating from the granny is smaller
               ! than the mass of the granny. Otherwise it will never be
               ! on-shell, and we don't need the special mapping.
               if (pmass(igranny_fks(nFKSprocess,ichan),iconf) .gt.
     $              mass_min(igranny_fks(nFKSprocess,ichan))) then
                  granny_is_res_fks(nFKSprocess,ichan)=.true.
               else
                  granny_is_res_fks(nFKSprocess,ichan)=.false.
               endif
            else
               granny_is_res_fks(nFKSprocess,ichan)=.false.
            endif
         else
            granny_is_res_fks(nFKSprocess,ichan)=.false.
         endif
c Now we have igranny and granny_is_res_fks. We can now determine the
c chain of s-channels that originates from the grandmother
         do i=-nexternal,nexternal
            granny_chain_fks(i,nFKSprocess,ichan)=.false.
         enddo
         if (granny_is_res_fks(nFKSprocess,ichan)) then
c granny is part of the chain            
            granny_chain_fks(igranny_fks(nFKSprocess,ichan),nFKSprocess,ichan)
     &           =.true.
c loop from the granny to the external particles. If mother was part of
c the granny chain, so are the daugthers.
            do i=igranny_fks(nFKSprocess,ichan),-1
               if (granny_chain_fks(i,nFKSprocess,ichan)) then
                  granny_chain_fks(iforest(1,i,iconf),nFKSprocess,ichan) =
     $                 .true.
                  granny_chain_fks(iforest(2,i,iconf),nFKSprocess,ichan) =
     $                 .true.
               endif
            enddo
         endif
      endif
c Here is the simply copy for later calls to this subroutine: set
c igranny, iaunt and granny_is_res from the saved information
      if (granny_is_res_fks(nFKSprocess,ichan)) then
         igranny=igranny_fks(nFKSprocess,ichan)
         iaunt=iaunt_fks(nFKSprocess,ichan)
         granny_is_res=.true.
         do i=-nexternal,nexternal
            granny_chain(i)=granny_chain_fks(i,nFKSprocess,ichan)
            if (i.le.0) then
               granny_chain_real_final(i)=.false.
            elseif (i.lt.max(i_fks,j_fks)) then
               granny_chain_real_final(i)=granny_chain(i)
            elseif(i.eq.max(i_fks,j_fks)) then
               granny_chain_real_final(i)=.true.
            else
               granny_chain_real_final(i)=granny_chain(i-1)
            endif
         enddo
      else
         igranny=0
         iaunt=0
         granny_is_res=.false.
         do i=-nexternal,nexternal
            granny_chain(i)=.false.
            granny_chain_real_final(i)=.false.
         enddo
      endif
      return
      end


      subroutine set_mu_central(ic,dd,c_mu2_r,c_mu2_f)
      use weight_lines
      use extra_weights
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      integer ic,dd,i,j
      double precision c_mu2_r,c_mu2_f,muR,muF,pp(0:3,nexternal)
      if (dd.eq.1) then
         c_mu2_r=scales2(2,ic)
         c_mu2_f=scales2(3,ic)
      else
c need to recompute the scales using the momenta
         dynamical_scale_choice=dyn_scale(dd)
         do i=1,nexternal
            do j=0,3
               pp(j,i)=momenta(j,i,ic)
            enddo
         enddo
         call set_ren_scale(pp,muR)
         c_mu2_r=muR**2
         call set_fac_scale(pp,muF)
         c_mu2_f=muF**2
c     reset the default dynamical_scale_choice
         dynamical_scale_choice=dyn_scale(1)
      endif
      return
      end


      function ran2()
!     Wrapper for the random numbers; needed for the NLO stuff
      use mint_module
      implicit none
      double precision ran2,x,a,b
      integer ii,jconfig
      a=0d0                     ! min allowed value for x
      b=1d0                     ! max allowed value for x
      ii=0                      ! dummy argument of ntuple
      jconfig=iconfig           ! integration channel (for off-set)
      call ntuple(x,a,b,ii,jconfig)
      ran2=x
      return
      end

      

      subroutine fill_configurations_common
      implicit none
      include 'nexternal.inc'
      include 'maxparticles.inc'
      include 'maxconfigs.inc'
      include 'nFKSconfigs.inc'
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      INTEGER NFKSPROCESS,nFKSprocess_save
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      pmass(-nexternal:0,1:lmaxconfigs,0:fks_configs)=0d0
      pwidth(-nexternal:0,1:lmaxconfigs,0:fks_configs)=0
      iforest(1:2,-max_branch:-1,1:lmaxconfigs,0:fks_configs)=0
      sprop(-max_branch:-1,1:lmaxconfigs,0:fks_configs)=0
      tprid(-max_branch:-1,1:lmaxconfigs,0:fks_configs)=0
      mapconfig(0:lmaxconfigs,0:fks_configs)=0
      call fill_configurations_born(iforest(1,-max_branch,1,0),sprop(
     $     -max_branch,1,0),tprid(-max_branch,1,0),mapconfig(0,0),pmass(
     $     -nexternal,1,0),pwidth(-nexternal,1,0))
      nFKSprocess_save=nFKSprocess
      do nFKSprocess=1,fks_configs
         call configs_and_props_inc_chooser()
         call fill_configurations_real(iforest(1,-max_branch,1
     $        ,nFKSprocess),sprop(-max_branch,1,nFKSprocess),tprid(
     $        -max_branch,1,nFKSprocess),mapconfig(0,nFKSprocess),pmass(
     $        -nexternal,1,nFKSprocess),pwidth(-nexternal,1
     $        ,nFKSprocess))
      enddo
      nFKSprocess=nFKSprocess_save
      return
      end

      subroutine fill_configurations_born(iforest_in,sprop_in,tprid_in
     $     ,mapconfig_in,pmass_in,pwidth_in)
      include 'maxparticles.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include "coupl.inc"
      integer i,j,k
      double precision ZERO
      parameter (ZERO=0d0)
      integer iforest_in(2,-max_branch:-1,lmaxconfigs)
      integer sprop_in(-max_branch:-1,lmaxconfigs)
      integer tprid_in(-max_branch:-1,lmaxconfigs)
      integer mapconfig_in(0:lmaxconfigs)
      double precision pmass_in(-nexternal:0,lmaxconfigs)
      double precision pwidth_in(-nexternal:0,lmaxconfigs)
      include "born_conf.inc"
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      include "born_props.inc"
      do i=1,lmaxconfigsb_used
         do j=-max_branchb_used,-1
            do k=1,2
               iforest_in(k,j,i)=iforest(k,j,i)
            enddo
            sprop_in(j,i)=sprop(j,i)
            tprid_in(j,i)=tprid(j,i)
         enddo
         mapconfig_in(i)=mapconfig(i)
         do j=-max_branchb_used,-1
            pmass_in(j,i)=pmass(j,i)
            pwidth_in(j,i)=pwidth(j,i)
         enddo
      enddo
      mapconfig_in(0)=mapconfig(0)
      return
      end

      subroutine fill_configurations_real(iforest_in,sprop_in,tprid_in
     $     ,mapconfig_in,pmass_in,pwidth_in)
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer i,j,k
      double precision ZERO
      parameter (ZERO=0d0)
      integer iforest_in(2,-max_branch:-1,lmaxconfigs)
      integer sprop_in(-max_branch:-1,lmaxconfigs)
      integer tprid_in(-max_branch:-1,lmaxconfigs)
      integer mapconfig_in(0:lmaxconfigs)
      double precision pmass_in(-nexternal:0,lmaxconfigs)
      double precision pwidth_in(-nexternal:0,lmaxconfigs)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,lmaxconfigs)
      double precision prwidth(-max_branch:-1,lmaxconfigs)
      integer prow(-max_branch:-1,lmaxconfigs)
      common/c_props_inc/prmass,prwidth,prow
      do i=1,lmaxconfigs
         do j=-max_branch,-1
            do k=1,2
               iforest_in(k,j,i)=iforest(k,j,i)
            enddo
            sprop_in(j,i)=sprop(j,i)
            tprid_in(j,i)=tprid(j,i)
         enddo
         mapconfig_in(i)=mapconfig(i)
         do j=-max_branch,-1
            pmass_in(j,i)=prmass(j,i)
            pwidth_in(j,i)=prwidth(j,i)
         enddo
      enddo
      mapconfig_in(0)=mapconfig(0)
      return
      end
