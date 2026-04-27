
       subroutine get_lo2_orders(lo2_orders)
       implicit none
       include 'orders.inc'
       integer lo2_orders(nsplitorders)

       ! copy the born orders into the lo2 orders
       ! This assumes that there is only one contribution
       ! at the born that is integrated (checked in born.f)
       lo2_orders(:) = born_orders(:)

       ! now get the orders for LO2
       lo2_orders(qcd_pos) = lo2_orders(qcd_pos) - 2
       lo2_orders(qed_pos) = lo2_orders(qed_pos) + 2
       return
       end



      subroutine sdk_get_invariants(p, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      include "coupl.inc"
      double precision p(0:3, nexternal-1)
      integer iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer i,j
      double precision sumdot

      logical rij_ge_mw
      COMMON /rij_ge_mw/ rij_ge_mw


      do i = 1, nexternal-1
        do j = i, nexternal-1
          invariants(i,j) = sumdot(p(0,i),p(0,j),dble(iflist(i)*iflist(j)))
          if(rij_ge_mw.and.abs(invariants(i,j)).lt.mdl_mw**2) then
            invariants(i,j)=dsign(1d0,invariants(i,j))*mdl_mw**2
          endif
          invariants(j,i) = invariants(i,j)
        enddo
      enddo

      return 
      end

      subroutine ewsudakov_f77(p_born_in, gstr_in, results)
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
C arguments

      include 'nexternal.inc'
      double precision p_born_in(0:3,nexternal-1)
      double precision gstr_in, results(6)
      ! results contain (born, sud0, sud1)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
cc
      include 'coupl.inc'
      include 'orders.inc'

      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double precision amp_split_born(amp_split_size)
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_QCD/AMP_SPLIT_EWSUD_QCD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      logical sud_mc_hel
      COMMON /to_mc_hel/ sud_mc_hel

      double precision wgt_sud, wgt_born, born

      logical firsttime
      data firsttime/.true./

      integer i

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      COMMON /rij_ge_mw/ rij_ge_mw
C-----
C  BEGIN CODE
C-----

      nfksprocess=1

      ! let us explicitly sum over the helicities
      sud_mc_hel=.false.

      if (firsttime) then
       call setpara('param_card.dat')   !Sets up couplings and masses
       firsttime = .false.
      endif

      g = gstr_in
      call update_as_param()
      p_born(:,:) = p_born_in(:,:)

      s_to_rij = .true.
      rij_ge_mw = .true.
      do sud_mod = 0,1
        ! call the born
        call sborn(p_born, born)
        amp_split_born(:) = amp_split(:)
        wgt_born = amp_split_born(1)

        ! call the EWsudakov
        call sudakov_wrapper(p_born)
        wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
        results(1) = wgt_born
        results(2+sud_mod) = wgt_sud
      enddo
      !! MZ to be extended to LO_2 etc

      !! TV: add the various sudakov outputs
      sud_mod = 1
      s_to_rij = .false.
      rij_ge_mw = .true.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)

      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(4) = wgt_sud

      sud_mod = 1
      s_to_rij = .false.
      rij_ge_mw = .false.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)
      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(5) = wgt_sud

      sud_mod = 1
      s_to_rij = .true.
      rij_ge_mw = .false.
      ! call the born
      call sborn(p_born, born)
      amp_split_born(:) = amp_split(:)
      wgt_born = amp_split_born(1)
      ! call the EWsudakov
      call sudakov_wrapper(p_born)
      wgt_sud = 2d0*(amp_split_ewsud_lsc(1)+
     $        amp_split_ewsud_ssc(1)+
     $        amp_split_ewsud_xxc(1)+
     $        amp_split_ewsud_par(1))
      results(6) = wgt_sud
      return

      end
