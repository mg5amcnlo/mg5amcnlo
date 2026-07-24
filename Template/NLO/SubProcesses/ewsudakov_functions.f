! sud_mod:
! 0 = Dropping Lem and lem contributions (like Sherpa)
! 1 = totally excluding QED contribution
! 2 = including Lem and lem in DR ---> for MadLoop comparison
       BLOCK DATA Sudakov_mode 
       implicit none
       Integer sud_mod
       COMMON /to_sud_mod/ sud_mod
       DATA sud_mod/ 2 / 
       END    

C filter hel: if set to true, helicities which do not contribute
C at the Born are skipped 
       BLOCK DATA sud_filter_helicities
       implicit none
       logical sud_filter_hel
       COMMON /to_filter_hel/ sud_filter_hel
       DATA sud_filter_hel / .true. /
       END

       BLOCK DATA sud_mc_helicities
       implicit none
       logical sud_mc_hel
       COMMON /to_mc_hel/ sud_mc_hel
       DATA sud_mc_hel / .true. / 
       END    

       BLOCK DATA Model_usual_or_FAV4
       implicit none
       logical FAV4
       COMMON /to_FAV4/ FAV4
c it does not work if set to true, besides particular cases
       DATA FAV4 / .false. /
       END

       BLOCK DATA shift_s_to_rij
       implicit none
       logical s_to_rij
       COMMON /to_s_to_rij/ s_to_rij
       DATA s_to_rij / .true. /
       END

       BLOCK DATA are_we_using_check_sudakov
       implicit none
       logical cs_run
       COMMON /to_cs_run/ cs_run
       DATA cs_run / .false. /
       END

       BLOCK DATA are_we_preventing_rij_smaller_than_MW
       implicit none
       logical rij_ge_mw
       COMMON /rij_ge_mw/ rij_ge_mw
       DATA rij_ge_mw / .true. /
       END



       logical function has_lo2()
       implicit none
       include 'orders.inc'
       integer lo2_orders(nsplitorders)
       integer orders_check(nsplitorders)
       integer i, j
       logical firsttime
       data firsttime/.true./
       logical has_lo2_save
       data has_lo2_save/.true./

       if (.not.firsttime) then
           has_lo2 = has_lo2_save
           return
       endif

       ! What follows is done only once
       firsttime = .false.

       ! and check if it exists
       call get_lo2_orders(lo2_orders)

       do i = 1, amp_split_size_born
         has_lo2_save = .true.
         call amp_split_pos_to_orders(i, orders_check)
         do j = 1, nsplitorders
           has_lo2_save = has_lo2_save.and.orders_check(j).eq.lo2_orders(j)
         enddo
         ! if it is true, then we have found it
         if (has_lo2_save) exit
       enddo

       has_lo2 = has_lo2_save

       return
       end


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



      !! MZ declare all functions as double complex, since some (few)
      !  terms can be imaginary

      double complex function get_imlog(s)
      implicit none
      double precision s
      logical force_imlog_to_zero
      parameter (force_imlog_to_zero=.false.)
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      get_imlog = dcmplx(0d0,0d0)
      if (s.gt.0d0.and..not.force_imlog_to_zero) get_imlog = dcmplx(0d0,-1d0*pi) 

      return
      end

      
      double complex function get_lsc_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_diag, sdk_iz2_diag
      double complex sdk_chargesq, bigLem, fordeb
      external sdk_iz2_diag,  sdk_chargesq, bigLem
      integer i
      double precision get_mass_from_id,mass
      external get_mass_from_id

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      double precision lzfordebug

      get_lsc_diag = 0d0
      lzfordebug = 0d0

c exit and do nothing
c      return

      lzow = dlog(mdl_mz**2/mdl_mw**2)

      do i = 1, nexternal-1
        mass = get_mass_from_id(pdglist(i))
        get_lsc_diag = get_lsc_diag - 0.5d0 * 
     %   (sdk_cew_diag(pdglist(i),hels(i),iflist(i)) * bigL(invariants(1,2))
     %    - 2d0*lzow*sdk_iz2_diag(pdglist(i),hels(i),iflist(i))*smallL(invariants(1,2))+
     %    sdk_chargesq(pdglist(i),hels(i),iflist(i))*bigLem(invariants(1,2),mass**2)
     %)
        lzfordebug = lzfordebug - 0.5d0 * (
     %    - 2d0*lzow*sdk_iz2_diag(pdglist(i),hels(i),iflist(i))*smallL(invariants(1,2)))
      enddo

      if(printinewsdkf) print*, "L-->  ",(get_lsc_diag -lzfordebug )/bigL(invariants(1,2))
      if(printinewsdkf) print*, "lz--> ",(lzfordebug )/smallL(invariants(1,2))

      if(deb_settozero.ne.0) get_lsc_diag=0d0

      return
      end


      double complex function get_lsc_nondiag(pdglist, hels, iflist,
     $                              invariants, ileg, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg, pdg_old, pdg_new
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_nondiag

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      ! this function is non zero only for Z/gamma mixing)
      get_lsc_nondiag = 0d0

      ! check that the polarisation is transverse
      if (abs(hels(ileg)).ne.1) return 

c exit and do nothing
c      return

      if ((pdg_old.eq.23.and.pdg_new.eq.22).or.
     $    (pdg_old.eq.22.and.pdg_new.eq.23)) then
        lzow = dlog(mdl_mz**2/mdl_mw**2)
        get_lsc_nondiag = -0.5d0 * sdk_cew_nondiag() * bigL(invariants(1,2))

      if (printinewsdkf) WRITE (72,*) , hels, ileg, pdg_new, dble(sdk_cew_nondiag())

      endif

      if(deb_settozero.ne.0.and.deb_settozero.ne.10.and.deb_settozero.ne.111) get_lsc_nondiag=0d0

      return
      end


      double complex function get_ssc_c(ileg1, ileg2, pdglist, pdgp1, pdgp2, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer ileg1, ileg2, pdgp1, pdgp2
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_tpm

      double precision s, rij

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      double precision pi
      parameter (pi=3.14159265358979323846d0)
      double complex imlog

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij

      double complex smallL_a_over_b_sing, bigL_a_over_b_sing
      external smallL_a_over_b_sing, bigL_a_over_b_sing

      double complex smallL_rij_over_s, bigL_rij_over_s, get_imlog     

      get_ssc_c = 0d0
c exit and do nothing
c      return

      lzow = dlog(mdl_mz**2/mdl_mw**2)
      s = invariants(1,2)

      rij = invariants(ileg1,ileg2)

      imlog = get_imlog(rij)

      get_ssc_c = get_ssc_c + 2d0*smallL(s) * (dlog(dabs(rij)/s) + imlog) 
     $    * sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)
     $    * sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)

      if(s_to_rij) then
        smallL_rij_over_s=smallL_a_over_b_sing(dabs(rij),s)
        bigL_rij_over_s=bigL_a_over_b_sing(dabs(rij),s)

        get_ssc_c = get_ssc_c + 
     $      (bigL_rij_over_s + 2d0 * smallL_rij_over_s * imlog) 
     $    * sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)
     $    * sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)
      endif

      if (printinewsdkf) WRITE (72,*) , hels, ileg1, ileg2, pdgp1, pdgp2,
     $ dble(sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)*CMPLX(1d0,-1000d0)),
     $ dble(sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)*CMPLX(1d0,-1000d0)) 

      if(deb_settozero.ne.0.and.deb_settozero.ne.1.and.deb_settozero.ne.111) get_ssc_c = 0d0

      return
      end


      double complex function get_ssc_n_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_ia_diag, sdk_iz_diag, smallLem
      external smallLem
 
      integer i,j
      double precision s, rij

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      double precision pi
      parameter (pi=3.14159265358979323846d0)
      double complex imlog

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij

      double complex smallL_a_over_b_sing, bigL_a_over_b_sing
      external smallL_a_over_b_sing, bigL_a_over_b_sing

      double complex smallL_rij_over_s, bigL_rij_over_s, get_imlog

      get_ssc_n_diag = 0d0
c      return
c exit and do nothing

      lzow = dlog(mdl_mz**2/mdl_mw**2)
      s = invariants(1,2)

      do i = 1, nexternal-1
         do j = 1, i-1
          if(i.eq.j) cycle
          rij = invariants(i,j)
          ! photon, Lambda = MW
          imlog = get_imlog(rij)

c      2d0/3d0*smallLem(0d0) comes from l(MW2,0d0) in the formulas

          get_ssc_n_diag = get_ssc_n_diag + 2d0*(smallL(s)+2d0/3d0*smallLem(0d0)) 
     %      * (dlog(dabs(rij/s))+imlog) 
     %      * sdk_ia_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_ia_diag(pdglist(j),hels(j),iflist(j))

          ! Z
          get_ssc_n_diag = get_ssc_n_diag + 2d0*smallL(s)
     %      * (dlog(dabs(rij/s))+imlog) 
     %      * sdk_iz_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_iz_diag(pdglist(j),hels(j),iflist(j))

          if(s_to_rij) then
            smallL_rij_over_s=smallL_a_over_b_sing(dabs(rij),s)
            bigL_rij_over_s=bigL_a_over_b_sing(dabs(rij),s)

        !correct photons
            get_ssc_n_diag = get_ssc_n_diag +
     $      (bigL_rij_over_s + 2d0 * smallL_rij_over_s * imlog)
     %      * sdk_ia_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_ia_diag(pdglist(j),hels(j),iflist(j))
        !correct z
            get_ssc_n_diag = get_ssc_n_diag +
     $      (bigL_rij_over_s + 2d0 * smallL_rij_over_s * imlog)
     %      * sdk_iz_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_iz_diag(pdglist(j),hels(j),iflist(j))

            get_ssc_n_diag = get_ssc_n_diag +
     $      (2d0 * smallL_rij_over_s * dlog(mdl_mw**2/mdl_mz**2))
     %      * sdk_iz_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_iz_diag(pdglist(j),hels(j),iflist(j))
          endif
        enddo
      enddo

      if(printinewsdkf) print*, "log(t/u)ls  SSCN diag -->  ", 
     .       get_ssc_n_diag/(smallL(s)*dlog(dabs(invariants(1,3)/invariants(1,4))))

      if(deb_settozero.ne.0) get_ssc_n_diag = 0d0

      return
      end


      double complex function get_ssc_n_nondiag_1(pdglist, hels, iflist,
     $                              invariants, ileg, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg, pdg_old, pdg_new
      double complex bigL, smallL, sdk_iz_nondiag, sdk_iz_diag
      integer i
      double precision s
      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      include 'coupl.inc'

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      double precision pi
      parameter (pi=3.14159265358979323846d0)
      double complex imlog

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij

      double complex smallL_a_over_b_sing, bigL_a_over_b_sing
      external smallL_a_over_b_sing, bigL_a_over_b_sing

      double complex smallL_rij_over_s, bigL_rij_over_s, get_imlog

      ! this function corresponds to the case when *one* out of the two particles
      ! that enters the SSC contributions mixes as Chi <--> H (mediated
      ! by the Z).
      get_ssc_n_nondiag_1 = 0d0
c      return
c exit and do nothing

      s = invariants(1,2)

      if ((pdg_old.eq.25.and.pdg_new.eq.250).or.
     $    (pdg_old.eq.250.and.pdg_new.eq.25)) then
        do i = 1, nexternal-1
          if (i.eq.ileg) cycle

c not work for H or Chi0 in initial state
          if (iflist(ileg).eq.-1) then
            print*,"Error: sign imaginary part with longitudinally polarised 
     .          Z or H not implemented for the initial state"
            stop
          endif
          imlog = get_imlog(invariants(i,ileg))

! Multiplied by 1-1000*I only when printed in NonDiag_structure.dat

          if (printinewsdkf) WRITE (72,*) , hels, ileg, i, pdg_new,pdglist(i),
     $    dble(sdk_iz_nondiag(pdg_new,hels(ileg),iflist(ileg))*CMPLX(1d0,-1000d0)),
     $    dble(sdk_iz_diag(pdglist(i),hels(i),iflist(i))*CMPLX(1d0,-1000d0))

          get_ssc_n_nondiag_1 = get_ssc_n_nondiag_1 +
     $              sdk_iz_diag(pdglist(i),hels(i),iflist(i)) *
     $              sdk_iz_nondiag(pdg_new,hels(ileg),iflist(ileg))
     $              * 2d0 * smallL(s) * (dlog(abs(invariants(i,ileg))/s)+imlog)

          if(s_to_rij) then
            smallL_rij_over_s=smallL_a_over_b_sing(dabs(invariants(i,ileg)),s)
            bigL_rij_over_s=bigL_a_over_b_sing(dabs(invariants(i,ileg)),s)

        !correct z
            get_ssc_n_nondiag_1 = get_ssc_n_nondiag_1 +
     $      (bigL_rij_over_s + 2d0 * smallL_rij_over_s * imlog) *
     $      sdk_iz_diag(pdglist(i),hels(i),iflist(i)) *
     $      sdk_iz_nondiag(pdg_new,hels(ileg),iflist(ileg))

            get_ssc_n_nondiag_1 = get_ssc_n_nondiag_1 +
     $      (2d0 * smallL_rij_over_s * dlog(mdl_mw**2/mdl_mz**2)) *
     $      sdk_iz_diag(pdglist(i),hels(i),iflist(i)) *
     $      sdk_iz_nondiag(pdg_new,hels(ileg),iflist(ileg))
          endif
        enddo
      endif

      if(deb_settozero.ne.0.and.deb_settozero.ne.1.and.deb_settozero.ne.111) get_ssc_n_nondiag_1 = 0d0

      return
      end


      double complex function get_ssc_n_nondiag_2(pdglist, hels, iflist,
     $                          invariants, ileg1, pdg_old1,pdg_new1,
     $                                      ileg2, pdg_old2,pdg_new2)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg1, pdg_old1, pdg_new1, ileg2, pdg_old2, pdg_new2
      double complex bigL, smallL, sdk_iz_nondiag, sdk_iz_diag
      double precision s
      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      include 'coupl.inc'

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      double precision pi
      parameter (pi=3.14159265358979323846d0)
      double complex imlog

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij

      double complex smallL_a_over_b_sing, bigL_a_over_b_sing
      external smallL_a_over_b_sing, bigL_a_over_b_sing

      double complex smallL_rij_over_s, bigL_rij_over_s, get_imlog

      ! this function corresponds to the case when both the two particles
      ! that enters the SSC contributions mixes as Chi <--> H (mediated
      ! by the Z).
      get_ssc_n_nondiag_2 = 0d0

c      return
c exit and do nothing

      s = invariants(1,2)

c not work for H or Chi0 in initial state
      if (iflist(ileg1).eq.-1.or.iflist(ileg2).eq.-1) then
         print*,"Error: sign imaginary part with longitudinally polarised 
     .        Z or H not implemented for the initial state"
         stop
      endif

      imlog = get_imlog(invariants(ileg1,ileg2))

      if (((pdg_old1.eq.25.and.pdg_new1.eq.250).or.
     $     (pdg_old1.eq.250.and.pdg_new1.eq.25)).and.
     $    ((pdg_old2.eq.25.and.pdg_new2.eq.250).or.
     $     (pdg_old2.eq.250.and.pdg_new2.eq.25))) then

! Multiplied by 1-1000*I only when printed in NonDiag_structure.dat

        if (printinewsdkf) WRITE (72,*) , hels, ileg1, ileg2, pdg_new1, pdg_new2,
     $   dble(sdk_iz_nondiag(pdg_new1,hels(ileg1),iflist(ileg1))*CMPLX(1d0,-1000d0)),
     $   dble(sdk_iz_nondiag(pdg_new2,hels(ileg2),iflist(ileg2))*CMPLX(1d0,-1000d0))

        get_ssc_n_nondiag_2 = get_ssc_n_nondiag_2 +
     $              sdk_iz_nondiag(pdg_new1,hels(ileg1),iflist(ileg1)) * 
     $              sdk_iz_nondiag(pdg_new2,hels(ileg2),iflist(ileg2))
     $              * 2d0 * smallL(s) * (dlog(abs(invariants(ileg1,ileg2))/s)+imlog)

        if(s_to_rij) then
            smallL_rij_over_s=smallL_a_over_b_sing(dabs(invariants(ileg1,ileg2)),s)
            bigL_rij_over_s=bigL_a_over_b_sing(dabs(invariants(ileg1,ileg2)),s)

        !correct z
            get_ssc_n_nondiag_2 = get_ssc_n_nondiag_2 +
     $      (bigL_rij_over_s + 2d0 * smallL_rij_over_s * imlog) *
     $      sdk_iz_nondiag(pdg_new1,hels(ileg1),iflist(ileg1)) * 
     $      sdk_iz_nondiag(pdg_new2,hels(ileg2),iflist(ileg2))

            get_ssc_n_nondiag_2 = get_ssc_n_nondiag_2 +
     $      (2d0 * smallL_rij_over_s * dlog(mdl_mw**2/mdl_mz**2)) *
     $      sdk_iz_nondiag(pdg_new1,hels(ileg1),iflist(ileg1)) *        
     $      sdk_iz_nondiag(pdg_new2,hels(ileg2),iflist(ileg2))
          endif
      endif

      if(deb_settozero.ne.0.and.deb_settozero.ne.1.and.deb_settozero.ne.111) get_ssc_n_nondiag_2 = 0d0

      return
      end


      double complex function get_xxc_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_diag, sdk_betaew_diag
      external sdk_cew_diag, sdk_betaew_diag
      integer i
      double precision sw2, cw2
      double precision mass, isopart_mass, tmp

      double precision get_mass_from_id, get_isopart_mass_from_id
      external get_mass_from_id, get_isopart_mass_from_id

      double complex sdk_chargesq, smallLem,dZemAA_logs
      external sdk_chargesq, smallLem,dZemAA_logs

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      get_xxc_diag = 0d0
c      return
c exit and do nothing

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2
    
      tmp=0d0

      do i = 1, nexternal-1
        if (abs(pdglist(i)).le.6.or.
     %     (abs(pdglist(i)).ge.11.and.abs(pdglist(i)).le.16)) then
          ! fermions

          get_xxc_diag = get_xxc_diag + 
     %       1.5d0 * sdk_cew_diag(pdglist(i),hels(i),iflist(i)) * smallL(invariants(1,2))

          mass = get_mass_from_id(pdglist(i))
          isopart_mass = get_isopart_mass_from_id(pdglist(i))

          get_xxc_diag = get_xxc_diag - 
     %       1d0/8d0/sw2 * mass**2/mdl_mw**2 * smallL(invariants(1,2))
          if (pdglist(i)*hels(i).lt.0) then

            ! left-handed fermion
            get_xxc_diag = get_xxc_diag - 
     %         1d0/8d0/sw2 * isopart_mass**2/mdl_mw**2 * smallL(invariants(1,2))
          else

            ! right-handed fermion
            get_xxc_diag = get_xxc_diag - 
     %         1d0/8d0/sw2 * mass**2/mdl_mw**2 * smallL(invariants(1,2))
          endif

          get_xxc_diag = get_xxc_diag + sdk_chargesq(pdglist(i),hels(i),iflist(i))*smallLem(mass**2)

        elseif (abs(pdglist(i)).ge.22.and.abs(pdglist(i)).le.24.and.hels(i).ne.0) then

          ! transverse W/Z/photons bosons
          get_xxc_diag = get_xxc_diag + 
     %     sdk_betaew_diag(pdglist(i))/2d0 * smallL(invariants(1,2))

          if (abs(pdglist(i)).eq.24) get_xxc_diag = get_xxc_diag + sdk_chargesq(pdglist(i),hels(i),iflist(i))*smallLem(mdl_mw**2)

          if (abs(pdglist(i)).eq.22) get_xxc_diag = get_xxc_diag + 0.5d0*dZemAA_logs()

        elseif (abs(pdglist(i)).eq.250.or.abs(pdglist(i)).eq.251.or.pdglist(i).eq.25) then

          ! goldstones or Higgs
          get_xxc_diag = get_xxc_diag + 
     %     (2d0*sdk_cew_diag(pdglist(i),hels(i),iflist(i)) - 
     %      3d0/4d0/sw2*mdl_mt**2/mdl_mw**2) * smallL(invariants(1,2))

          if (abs(pdglist(i)).eq.251) get_xxc_diag = get_xxc_diag + smallLem(mdl_mw**2)
        endif

c uncomment if uncomment the full block down
c        tmp= get_xxc_diag
      enddo

      if(printinewsdkf) print*, "lC diag --> ",(get_xxc_diag)/smallL(invariants(1,2))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!! BEGIN: to be at some point removed by  the code !!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
c
c
c
c
c      if(printinewsdkf) then
c         if(abs(pdglist(3)).eq.5.or.
c     .      abs(pdglist(3)).eq.6.or.
c     .      abs(pdglist(3)).eq.13.or.
c     .          pdglist(3).eq.22.or.
c     .          pdglist(3).eq.23.or.
c     .      abs(pdglist(3)).eq.24  ) then
c
c!!! MANUAL IMPLEMENTATION OF PR logs for Denner and Pozzorini
c           
c             write(*,*) "sto mettendo PR logs a mano"
c             if (abs(pdglist(3)).eq.5.and.hels(1).eq.1.and.hels(3).eq.-1) then 
c                get_xxc_diag = get_xxc_diag -16.6d0 * smallL(invariants(1,2))  
c             elseif (abs(pdglist(3)).eq.6.and.hels(1).eq.1.and.hels(3).eq.-1) then 
c                 get_xxc_diag = get_xxc_diag -12.2d0 * smallL(invariants(1,2))  
c             elseif (abs(pdglist(3)).eq.13.and.hels(1).eq.1.and.hels(3).eq.-1) then 
c                 get_xxc_diag = get_xxc_diag -9.03d0 * smallL(invariants(1,2))  
c             elseif (abs(pdglist(3)).eq.22.and.hels(1).eq.-1.and.hels(2).eq.1) then
c                 get_xxc_diag = get_xxc_diag +(3.67d0) * smallL(invariants(1,2)) 
c             elseif (abs(pdglist(3)).eq.22.and.hels(1).eq.1.and.hels(2).eq.-1) then
c                 get_xxc_diag = get_xxc_diag +(3.67d0) * smallL(invariants(1,2)) 
c             elseif (abs(pdglist(3)).eq.23.and.hels(1).eq.-1 .and.
c     .               abs(pdglist(4)).eq.22.and.hels(2).eq.1) then
c                 get_xxc_diag = get_xxc_diag +(15.1d0) * smallL(invariants(1,2)) 
c             elseif (abs(pdglist(3)).eq.23.and.hels(1).eq.-1 .and.
c     .               abs(pdglist(4)).eq.23.and.hels(2).eq.1) then
c                 get_xxc_diag = get_xxc_diag +(26.6d0) * smallL(invariants(1,2))
c             elseif (abs(pdglist(3)).eq.23.and.hels(1).eq.1 .and.
c     .               abs(pdglist(4)).eq.22.and.hels(2).eq.-1) then
c                 get_xxc_diag = get_xxc_diag +(-17.1d0) * smallL(invariants(1,2)) 
c             elseif (abs(pdglist(3)).eq.23.and.hels(1).eq.1 .and.
c     .               abs(pdglist(4)).eq.23.and.hels(2).eq.-1) then
c                 get_xxc_diag = get_xxc_diag +(-37.9d0) * smallL(invariants(1,2)) 
c             elseif (abs(pdglist(3)).eq.24.and.hels(1).eq.1.and.
c     .               abs(hels(3)).eq.1.and.abs(hels(4)).eq.1) then
c                 get_xxc_diag = get_xxc_diag +(-14.2d0) * smallL(invariants(1,2))  
c             elseif (pdglist(3).ne.22.and.pdglist(3).ne.23.and.abs(pdglist(3)).ne.24) then
c                get_xxc_diag = get_xxc_diag +8.80d0 * smallL(invariants(1,2))
c             endif
c        print*, "PR log --> ",(get_xxc_diag-tmp)/smallL(invariants(1,2))
c        endif
c      endif
c
c
cc    PR LOG are then removed by get_xxc_diag
c      get_xxc_diag = tmp

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!! END: to be at some point removed by  the code !!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if(deb_settozero.ne.0) get_xxc_diag = 0d0

      return
      end


      double complex function get_xxc_nondiag(pdglist, hels, iflist,
     $                              invariants, ileg, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg, pdg_old, pdg_new
      double complex bigL, smallL, sdk_betaew_nondiag

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      integer   deb_settozero
      common /to_deb_settozero/deb_settozero

      ! this function is non zero only for Z/gamma mixing)
      get_xxc_nondiag = 0d0
c      return
c exit and do nothing

      ! check that the polarisation is transverse
      if (abs(hels(ileg)).ne.1) return 

      if ((pdg_old.eq.23.and.pdg_new.eq.22).or.
     $    (pdg_old.eq.22.and.pdg_new.eq.23)) then
        ! pdg_old -> N, pdg_new -> N' in Denner-Pozzorini notation
        !  pdg_old=23,  pdg_new=22, E_AZ = 1
        !  pdg_old=22,  pdg_new=23, E_ZA = -1
        ! Given DP eq 4.22, the only !=0 case is with E_AZ
        if (pdg_old.eq.23) then
          get_xxc_nondiag = get_xxc_nondiag + 
     %    sdk_betaew_nondiag() * smallL(invariants(1,2))
        endif
      endif

      if (printinewsdkf) WRITE (72,*) , hels, ileg, -666d0 , pdg_new, pdg_old,
     $    dble(get_xxc_nondiag), -666d0

      if(deb_settozero.ne.0.and.deb_settozero.ne.100.and.deb_settozero.ne.111) get_xxc_nondiag = 0d0

      return
      end


c      double complex function get_qcd_lo2(pdglist, hels, iflist, invariants)
c      implicit none
c      include 'nexternal.inc'
c      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
c      double precision invariants(nexternal-1, nexternal-1)
c
c      !MZ to be written
c      get_qcd_lo2 =  CMPLX(0d0,0d0)
c      return
c      end

 
      double complex function bigL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      bigL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)**2

      return
      end


      double complex function smallL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      smallL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)

      return
      end

      
      double complex function log_a_over_b_sing(a,b)
      implicit none
      double precision a,b,aok,bok
      include "q_es.inc"
      double precision pi

      if ((dabs(a).lt.1d0.and.dabs(a).ne.0d0).or.
     .     (dabs(b).lt.1d0.and.dabs(b).ne.0d0)) then
        print*,"Strange: in l_a_over_b_sin there is 
     .          a very small but not zero mass. a=",
     .  a," b=",b
        !stop
      endif

      aok=a
      bok=b
      if (a.eq.0d0) aok= QES2
      if (b.eq.0d0) bok= QES2

c      print*, "QES2=",QES2
      
      log_a_over_b_sing = dlog(aok/bok)

      return
      end

      double complex function smallL_a_over_b_sing(a,b)
      implicit none
      double precision a,b
      double complex log_a_over_b_sing
      external log_a_over_b_sing
      include 'coupl.inc'
      include "q_es.inc"
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      smallL_a_over_b_sing = dble(gal(1))**2 / (4d0*pi)**2 * log_a_over_b_sing(a,b)

      return
      end


      double complex function bigL_a_over_b_sing(a,b)
      implicit none
      double precision a,b
      double complex log_a_over_b_sing
      external log_a_over_b_sing
      include 'coupl.inc'
      include "q_es.inc"
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      bigL_a_over_b_sing = dble(gal(1))**2 / (4d0*pi)**2 * log_a_over_b_sing(a,b)**2

c      print*, "in bigL a over b", " a=", a, " b=", b

      return
      end


      double complex function bigLem(s,m2k)
      implicit none
      double precision m2k,s
      double complex log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL
      external log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL
      include 'coupl.inc'
      include "q_es.inc"

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      double precision pi
      parameter (pi=3.14159265358979323846d0)

      if (sud_mod.eq.0.or.sud_mod.eq.1) then
        bigLem = 0d0
        return
      elseif (sud_mod.eq.2) then
       bigLem = 2*smallL(s) * log_a_over_b_sing(mdl_mw**2,0d0)+
     .          bigL_a_over_b_sing(mdl_mw**2,0d0)-
     .          bigL_a_over_b_sing(m2k,0d0)

        return
      else
        print*,"sud_mod=",sud_mod,". It is not defined."
        stop
      endif

      return
      end


      double complex function smallLem(m2k)
      implicit none
      double precision m2k
      double complex log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL
      external log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL
      include 'coupl.inc'
      include "q_es.inc"

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      double precision pi
      parameter (pi=3.14159265358979323846d0)

      if (sud_mod.eq.0.or.sud_mod.eq.1) then
        smallLem = 0d0
        return
      elseif (sud_mod.eq.2) then
       smallLem = 0.5d0*smallL_a_over_b_sing(mdl_mw**2, m2k)+
     .          smallL_a_over_b_sing(mdl_mw**2, 0d0)
        return
      else
        print*,"sud_mod=",sud_mod,". It is not defined."
        stop
      endif

      end


      double complex function dZemAA_logs()
      implicit none
      double precision m2k
      double complex log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL,smallLem
      external log_a_over_b_sing, bigL_a_over_b_sing, smallL_a_over_b_sing,smallL,smallLem
      include 'coupl.inc'
      include "q_es.inc"

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      double precision pi
      parameter (pi=3.14159265358979323846d0)

      if (sud_mod.eq.0.or.sud_mod.eq.1) then
        dZemAA_logs= 0d0
        return
      elseif (sud_mod.eq.2) then
c up and down quarks    
       dZemAA_logs=3d0*(3d0*(1d0/3d0)**2+2d0*(2d0/3d0)**2) 
c charged lep
       dZemAA_logs=dZemAA_logs+1d0*3d0*1d0
c total factor

c 2d0/3d0*smallLem(0d0) comes from l(MW2,0d0) in the formulas

       dZemAA_logs=dZemAA_logs*(-4d0/3d0)*2d0/3d0*smallLem(0d0)

       return

      else
        print*,"sud_mod=",sud_mod,". It is not defined."
        stop
      endif

      return
      end



      double complex function sdk_chargesq(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      double complex sdk_charge

      logical FAV4
      COMMON /to_FAV4/ FAV4

      sdk_chargesq = sdk_charge(pdg, hel, ifsign)**2

      if(.not.FAV4.and.abs(pdg).eq.251) sdk_chargesq = sdk_chargesq *(-1d0) 

      return
      end


      double complex function sdk_charge(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      integer s_pdg

      logical FAV4
      COMMON /to_FAV4/ FAV4

      s_pdg = pdg*ifsign

      sdk_charge = 0d0

C lepton 
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_charge = -1d0
C antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_charge = 1d0

C up quark 
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_charge = 2d0/3d0
C anti up quark 
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_charge = -2d0/3d0

C down quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_charge = -1d0/3d0
C antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_charge = 1d0/3d0

C charged goldstones / W boson
      if (s_pdg.eq.251.or.s_pdg.eq.24) sdk_charge = 1d0
      if (s_pdg.eq.-251.or.s_pdg.eq.-24) sdk_charge = -1d0

      if (.not.FAV4) then
          sdk_charge = -1 * sdk_charge
          if (abs(s_pdg).eq.251)  sdk_charge = sdk_charge * CMPLX(0d0,1d0)
      endif

      return
      end


      double complex function sdk_tpm(pdg, hel, ifsign, pdgp)
      implicit none
      integer pdg, hel, ifsign, pdgp
      ! PDGP is the pdg of the born leg after its charge has been 
      !  changed by ±1. It is only necessary for vector bosons, where
      !  one can have either w+ > w+ gamma or w+ > w+ z
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4
C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity

C!!!!						ATTENTION						      !!!!
C!!!! following the notation of Denner and Pozzorini, the prime index pdgp is the first and pdg is the second !!!!
C!!!! It is not working for logitudinally polarised Z in the initial state (ifsign=-1), both as pdg or pdgp   !!!!                        

      if (ifsign.eq.-1.and.(pdg.eq.250.or.pdgp.eq.250)) then
        print*,"Error: Tpm invovling longitudinally polarised 
     .          Z not implemented for the initial state"
      endif

      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_tpm = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_tpm = sign(1d0,dble(ifsign*pdg)) 

C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_tpm = sign(1d0,dble(ifsign*pdg)) 

C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_tpm = sign(1d0,dble(ifsign*pdg))

C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_tpm = sign(1d0,dble(ifsign*pdg))

      ! if it has already been set, then add the correct normalisation
      ! and return
      if (sdk_tpm.ne.0d0) then

        sdk_tpm = sdk_tpm / dsqrt(2d0*sw2)

        if (.not.FAV4) sdk_tpm=  sdk_tpm *-1d0

        return
      endif

c goldstones, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251.and.pdgp.eq.25) sdk_tpm = sign(1d0,dble(s_pdg))
      if (abs(s_pdg).eq.251.and.pdgp.eq.250) sdk_tpm = CMPLX(0d0,-1d0)

      if (sdk_tpm.ne.0d0) then
         sdk_tpm = sdk_tpm / ( 2d0 * dsqrt(sw2))

         if (.not.FAV4) sdk_tpm=  sdk_tpm *1d0*CMPLX(0d0,1d0)
           if (pdgp.eq.25 .and. pdg*ifsign.eq.251) sdk_tpm =sdk_tpm * (-1d0)
           if (pdgp.eq.250 .and. pdg*ifsign.eq.-251) sdk_tpm =sdk_tpm * (-1d0)

         return
      endif

c following last .and. conditions are not strictly necessary
      if (abs(s_pdg).eq.250.and.abs(pdgp*ifsign).eq.251) sdk_tpm =  CMPLX(0d0,1d0)
      if (abs(s_pdg).eq.25.and.abs(pdgp*ifsign).eq.251) sdk_tpm = sign(1d0,dble((pdgp*ifsign)))

      if (sdk_tpm.ne.0d0) then
         sdk_tpm = sdk_tpm / ( 2d0 *dsqrt(sw2))

! opposite of what I see in HuaSheng modifications
         if (.not.FAV4) sdk_tpm=  sdk_tpm *1d0*CMPLX(0d0,1d0)
           if (abs(s_pdg).eq.25 .and. pdgp*ifsign.eq.-251) sdk_tpm =sdk_tpm * (-1d0)
           if (abs(s_pdg).eq.250 .and. pdgp*ifsign.eq.251) sdk_tpm =sdk_tpm * (-1d0)

         return
      endif
      
c vector bosons
      if (abs(pdg*ifsign).eq.24.and.pdgp.eq.22) sdk_tpm = sign(1d0,dble(pdg*ifsign))
      if (pdg.eq.22.and.abs(pdgp*ifsign).eq.24) sdk_tpm = sign(1d0,dble(pdgp*ifsign))
      if (abs(pdg*ifsign).eq.24.and.pdgp.eq.23) sdk_tpm = -sign(1d0,dble(pdg*ifsign)) * dsqrt(cw2/sw2)
      if (pdg.eq.23.and.abs(pdgp*ifsign).eq.24) sdk_tpm = -sign(1d0,dble(pdgp*ifsign)) * dsqrt(cw2/sw2)

      if ((.not.FAV4).and.(pdg.eq.23.or.pdgp.eq.23))    sdk_tpm=  sdk_tpm * (-1d0)

      return
      end



      double complex function sdk_t3_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4
C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity
      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_t3_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_t3_diag = sign(0.5d0,dble(ifsign*pdg)) 

C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_t3_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_t3_diag = sign(0.5d0,dble(ifsign*pdg))

C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_t3_diag = -sign(0.5d0,dble(ifsign*pdg))

C goldstones, they behave like left handed leptons (charged); neutrals
C mix
      if (abs(s_pdg).eq.251) sdk_t3_diag = sign(0.5d0,dble(s_pdg)) 

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_t3_diag = sign(1d0,dble(ifsign*pdg))

      if (.not.FAV4) then
         sdk_t3_diag =  sdk_t3_diag * -1
         if  (abs(s_pdg).eq.251) sdk_t3_diag =  sdk_t3_diag * CMPLX(0d0,1d0)
      endif

      return
      end



      double complex function sdk_yo2_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4
C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity
      if (hel.ne.0) then
          s_pdg = pdg*hel
      else
        s_pdg = pdg
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_yo2_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_yo2_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_yo2_diag = -sign(1d0,dble(ifsign*pdg))
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_yo2_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_yo2_diag = sign(2d0/3d0,dble(ifsign*pdg))
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_yo2_diag = sign(1d0/6d0,dble(ifsign*pdg))

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_yo2_diag = -sign(1d0/3d0,dble(ifsign*pdg))
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_yo2_diag = sign(1d0/6d0,dble(ifsign*pdg))

C goldstones, they behave like left handed leptons (charged); neutrals mix

      if (abs(s_pdg).eq.251) sdk_yo2_diag = sign(0.5d0,dble(s_pdg))

!      mio
      if (.not.FAV4) then
          print*,"it has never been checked"
          stop
         
          sdk_yo2_diag=sdk_yo2_diag*-1
          if  (abs(s_pdg).eq.251) sdk_yo2_diag =  sdk_yo2_diag * CMPLX(0d0,1d0)
      endif

      return
      end


      double complex function sdk_iz_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      include "coupl.inc"
      double precision sw2, cw2
      double complex sdk_t3_diag, sdk_charge

      logical FAV4
      COMMON /to_FAV4/ FAV4

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_iz_diag = sdk_t3_diag(pdg,hel,ifsign) - sw2*sdk_charge(pdg,hel,ifsign)
      sdk_iz_diag = sdk_iz_diag / sqrt(sw2*cw2)

      if (.not.FAV4) then
          sdk_iz_diag= sdk_iz_diag / (-1d0)
          
          if (abs(pdg).eq.251) sdk_iz_diag= sdk_iz_diag / CMPLX(0d0,1d0)

      endif

      return
      end


      double complex function sdk_iz_nondiag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      ! only works for the mixing chi/H
      sdk_iz_nondiag = 0d0
      if (pdg.eq.250.or.pdg.eq.25) then
          ! pdg (new) = 25
        if (pdg.eq.25) sdk_iz_nondiag = ifsign * dcmplx(0d0,-1d0) / 2d0
        if (pdg.eq.250) sdk_iz_nondiag = ifsign * dcmplx(0d0,1d0) / 2d0
        sdk_iz_nondiag = sdk_iz_nondiag / sqrt(sw2*cw2)
      endif

      if (.not.FAV4) then
          sdk_iz_nondiag = sdk_iz_nondiag * -1
      endif
            
      return
      end


      double complex function sdk_ia_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      double complex sdk_charge

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      if (sud_mod.eq.0.or.sud_mod.eq.2) then
        sdk_ia_diag = -sdk_charge(pdg,hel,ifsign)
        return
      elseif (sud_mod.eq.1) then
        sdk_ia_diag = 0
        return
      else
        print*,"sud_mod=",sud_mod,". It is not defined."
        stop
      endif

      return
      end



      double complex function sdk_iz2_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity
      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_iz2_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_iz2_diag =1d0 / (4*sw2*cw2) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_iz2_diag = sw2/cw2
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_iz2_diag = (cw2-sw2)**2 / (4*sw2*cw2) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_iz2_diag = 4*sw2/(9*cw2)
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_iz2_diag =(3*cw2-sw2)**2 / (36*sw2*cw2) 

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_iz2_diag = sw2/(9*cw2)
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_iz2_diag = (3*cw2+sw2)**2 / (36*sw2*cw2) 

C goldstones, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251) sdk_iz2_diag = (cw2-sw2)**2 / (4*sw2*cw2) 
      if (abs(s_pdg).eq.250) sdk_iz2_diag =1d0 / (4*sw2*cw2)
      if (abs(s_pdg).eq.25)  sdk_iz2_diag =1d0 / (4*sw2*cw2)  

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_iz2_diag = cw2 / sw2

      return
      end




      double complex function sdk_cew_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

      double complex sdk_chargesq
      external sdk_chargesq

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity

      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_cew_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_cew_diag = 1d0/cw2
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_cew_diag = 4d0/(9*cw2)
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_cew_diag = (sw2+27*cw2) / (36*sw2*cw2) 

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_cew_diag = 1d0/(9*cw2)
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_cew_diag = (sw2+27*cw2) / (36*sw2*cw2) 

C goldstones and Higgs, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)
      if (abs(s_pdg).eq.250) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)
      if (abs(s_pdg).eq.25) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_cew_diag = 2 / sw2

C transverse Z boson
      if (abs(s_pdg).eq.23) sdk_cew_diag = 2 * cw2 / sw2

C (transverse) photon
      if (abs(s_pdg).eq.22) sdk_cew_diag = 2d0 

      if (sud_mod.eq.0.or.sud_mod.eq.2) then
        return
      elseif (sud_mod.eq.1) then
        sdk_cew_diag =  sdk_cew_diag - sdk_chargesq(pdg, hel, ifsign)
        return
      else
        print*,"sud_mod=",sud_mod,". It is not defined."
        stop 
      endif

      return
      end


      double complex function sdk_cew_nondiag()
      implicit none
C returns the gamma/z mixing of sdk_cew
      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_cew_nondiag = - 2 / sw2 * dsqrt(cw2*sw2) 

      if (.not.FAV4) sdk_cew_nondiag = sdk_cew_nondiag * (-1d0)

      return
      end


      double complex function sdk_betaew_diag(pdg)
      implicit none
      integer pdg

      include "coupl.inc"
      double precision sw2, cw2

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_betaew_diag = 0d0

      if (abs(pdg).eq.24) then

        if (sud_mod.eq.0.or.sud_mod.eq.2) then
          sdk_betaew_diag = 19d0/6d0/sw2
          return
        elseif (sud_mod.eq.1) then
          sdk_betaew_diag = 19d0/6d0/sw2 - 11d0/3d0
          return
        else
          print*,"sud_mod=",sud_mod,". It is not defined."
          stop
        endif

      elseif (pdg.eq.23) then
        sdk_betaew_diag = (19d0 - 38d0*sw2 -22d0*sw2**2) / 6d0/sw2/cw2

      elseif (pdg.eq.22) then
        sdk_betaew_diag = -11d0/3d0

        if (sud_mod.eq.1) sdk_betaew_diag = sdk_betaew_diag +  80d0/9d0

      endif

      return
      end


      double complex function sdk_betaew_nondiag()
      implicit none

      include "coupl.inc"
      double precision sw2, cw2

      logical FAV4
      COMMON /to_FAV4/ FAV4

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_betaew_nondiag = -(19d0 + 22d0*sw2) / 6d0/dsqrt(sw2*cw2)

      if(.not.FAV4) sdk_betaew_nondiag= sdk_betaew_nondiag *(-1d0)     

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



      subroutine sdk_test_functions()
      implicit none
      ! performs consistency checks between the various functions
      integer npdgs
      parameter(npdgs=33)
      integer pdg_list(npdgs)
      data pdg_list /-251,-24,-16,-15,-14,-13,-12,-11,
     %               -6,-5,-4,-3,-2,-1,1,2,3,4,5,6,
     %               11,12,13,14,15,16,21,22,23,24,25,
     %               250,251/
      integer ihel, ifsign
      integer i

      include "coupl.inc"
      double precision sw2, cw2

      double complex sdk_charge, sdk_t3_diag, sdk_yo2_diag, sdk_iz2_diag
      double complex q, t3, yo2, iz2 

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      do ifsign = -1, 1, 2 
        do ihel = -1, 1, 2 
          do i = 1,npdgs
            ! t3-y-q relation
c            write(*,*) 'Q=t3+yo2_diag'
            q = sdk_charge(pdg_list(i), ihel, ifsign)
            t3 = sdk_t3_diag(pdg_list(i), ihel, ifsign) 
            yo2 = sdk_yo2_diag(pdg_list(i),ihel, ifsign)
            if (abs(q - (t3+yo2)).gt.1d-4) then
              write(*,*) 'Q=t3+yo2_diag'
              write(*,*) 'WRONG', pdg_list(i), ihel, ifsign,
     %                 q, t3, yo2
            stop
            endif
            
            ! t3-q-iz2 relation
c            write(*,*) 'IZ=t3-sw Q / sw cw'
            iz2 = sdk_iz2_diag(pdg_list(i),ihel, ifsign)

            if (abs(iz2 - (t3-sw2*q)**2/sw2/cw2).gt.1d-4) then
              write(*,*) 'IZ=t3-sw Q / sw cw'
              write(*,*) 'WRONG', pdg_list(i), ihel, ifsign,
     %                 q, t3, (t3-sw2*q)**2/sw2/cw2, iz2
            endif
          enddo
        enddo
      enddo

      return
      end


      double precision function get_isopart_mass_from_id(pdg)
      implicit none
      ! returns the mass of the isospin partner of particle pdg.
      ! works only for fermions
      integer pdg
      integer apdg, apdg_part

      double precision get_mass_from_id

      apdg = abs(pdg)

      if (apdg.le.6.or.(apdg.ge.11.and.apdg.le.16)) then
          ! if apdg is even, the partner is apdg-1
          ! if apdg is odd, the partner is apdg+1
          if (mod(apdg,2).eq.0) apdg_part = apdg - 1
          if (mod(apdg,2).eq.1) apdg_part = apdg + 1
          get_isopart_mass_from_id = get_mass_from_id(apdg_part)

      else
        write(*,*) 'ERROR: get_isopart_mass_from_id', pdg
      endif

      return
      end
      


      subroutine get_par_ren_alphamz(invariants)
      implicit none
      include 'nexternal.inc'
      double precision invariants(nexternal-1, nexternal-1)

      include 'orders.inc'
      integer imaxpara
      parameter (imaxpara=6)
      double complex amp_split_ewsud_der(amp_split_size,imaxpara)
      common /to_amp_split_ewsud_der/ amp_split_ewsud_der
      double complex amp_split_ewsud_der2(amp_split_size,imaxpara)
      common /to_amp_split_ewsud_der2/ amp_split_ewsud_der2
C     ipara = 1->AEWm1; 2->MZ; 3->MW; 4->MT/YMT; 5->MH

      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD

      double complex ls, ls_o_mt
      double complex dalpha, dcw, dmw2, dmz2, dmt, dmh2, dt, dheff_o_heff
      double complex dmt_QCD
      double complex smallL, sdk_betaew_diag, sdk_cew_diag
      external smallL, sdk_betaew_diag, sdk_cew_diag
      double precision pi, cw2, sw2, Qt
      parameter (pi=3.14159265358979323846d0)

      include 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      logical was_sud_mod_1

      include 'ewsudakov_haslo.inc'

      ! given the to_amp_split_ewsud_der (derivatives of the ME's wrt
      ! the various parameters, amp_split_ewsud are filled with
      ! the parameter-renormalisation contribution

      was_sud_mod_1=.false.

      if(sud_mod.eq.1) then
       was_sud_mod_1=.true.
       sud_mod=0
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      Qt=2d0/3d0

      amp_split_ewsud(:) = (0d0,0d0)

      ls = smallL(invariants(1,2))
      ls_o_mt = dble(gal(1))**2 / (4d0*pi)**2 * dlog(invariants(1,2)/mdl_mt**2) 

      ! the parameter renormalisation in Denner-Pozzorini reads:
      !  dM/de de + dM/dcw dcw + dM/dht dht + dM/dhh dhh

      ! 1) dM/de de = dM/dalpha dalpha, with dalpha=2 Z_e / 4Pi
      !    remember, we derive wrt alpha^-1
      dalpha = -sdk_betaew_diag(22) / aewm1 
      dalpha = dalpha * ls
c      amp_split_ewsud(:) = amp_split_ewsud_der(:,1) * ( - aewm1**2) * 
c     $       dAlpha 

      ! 2) dM/dcw = dM/dmw dmw + dM/dmz dmz
      dmw2 = - (sdk_betaew_diag(24) - 4d0 * sdk_cew_diag(250,0,1))
     $      - 3d0*mdl_mt**2/2d0/mdl_mw**2/sw2
      dmw2 =  dmw2 * mdl_mw**2 * ls

      dmz2 = - (sdk_betaew_diag(23) - 4d0 * sdk_cew_diag(250,0,1))
     $      - 3d0*mdl_mt**2/2d0/mdl_mw**2/sw2
      dmz2 =  dmz2 * mdl_mz**2 * ls

      dmt = 1d0/4d0/sw2 + 1d0/8d0/sw2/cw2 + 3d0/2d0/cw2*Qt - 3d0/cw2*Qt**2  
     $     + 3d0/8d0/sw2 * mdl_mt**2/mdl_mw**2

      dmt =  dmt * mdl_mt * ls

      dmh2 = 1d0/2d0/sw2 
     $ * (
     $ 9d0*mdl_mw**2/mdl_mh**2 * (1d0 + 1d0/2d0/cw2**2) 
     $ - 3d0/2d0 * (1d0 + 1d0/2d0/cw2) + 15d0/4d0 * mdl_mh**2/mdl_mw**2
     $) + 3d0/2d0/sw2 * mdl_mt**2/mdl_mw**2 * (1d0-6d0*mdl_mt**2/mdl_mh**2 )

      dmh2 = dmh2 * mdl_mh**2 * ls

      dt=1d0/(mdl_ee*dsqrt(sw2)*mdl_mw)*
     .   (-3d0/2d0 * mdl_mw**2 * ( mdl_mz**2/cw2+2d0* mdl_mw**2)
     .    -mdl_mh**2/4d0 *( 2d0* mdl_mw**2 + mdl_mz**2 +3d0* mdl_mh**2) + 6d0 * mdl_mt**4)
   
      dt = dt * ls 
      
      dheff_o_heff = mdl_ee/(2d0*dsqrt(sw2))*dt/(mdl_mw*mdl_mh**2)

      dmt_QCD =  - 3d0 * 4d0/3d0 

      dmt_QCD =  dmt_QCD * mdl_mt * ls_o_mt  * (G/gal(1))**2

      if(has_lo1) then

        amp_split_ewsud(:) = amp_split_ewsud_der(:,1) * ( - aewm1**2) *
     $       dAlpha

        amp_split_ewsud(:) = amp_split_ewsud(:) + 
     $      amp_split_ewsud_der(:,2)/(2d0*mdl_mz) * dmz2 + 
     $      amp_split_ewsud_der(:,3)/(2d0*mdl_mw) * dmw2 +
     $      amp_split_ewsud_der(:,4) * dmt +
     $      amp_split_ewsud_der(:,5)/(2d0*mdl_mh) * dmh2 +
     $      amp_split_ewsud_der(:,6) * dheff_o_heff 

      endif
    
      if(was_sud_mod_1) sud_mod=1

      if (has_lo2) then
        if (sud_mod.eq.2) then
         amp_split_ewsud(:) = amp_split_ewsud(:) +
     $      amp_split_ewsud_der2(:,4) * dmt_QCD 

c      print*,"from dmt_QCD", amp_split_ewsud_der2(:,4) * dmt_QCD
        endif
      endif       
 
      ! LEAVE EMPTY FOR THE MOMENT
!    correct by a factor 2 ok?
      amp_split_ewsud(:)=amp_split_ewsud(:)/2d0

      return
      end



      subroutine get_par_ren_gmu(invariants)
      implicit none
      include 'nexternal.inc'
      double precision invariants(nexternal-1, nexternal-1)

      include 'orders.inc'
      integer imaxpara
      parameter (imaxpara=6)
      double complex amp_split_ewsud_der(amp_split_size,imaxpara)
      common /to_amp_split_ewsud_der/ amp_split_ewsud_der
      double complex amp_split_ewsud_der2(amp_split_size,imaxpara)
      common /to_amp_split_ewsud_der2/ amp_split_ewsud_der2
C     ipara = 1->GF; 2->MZ; 3->MW; 4->MT/YMT; 5->MH

      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD

      double complex ls, ls_o_mt
      double complex dalpha, dcw, dmw2, dmz2, dmt, dmh2, dt, dheff_o_heff
      double complex dmt_QCD, dGmu, dGmudalpha, dGmudmz2, dGmudmw2
      double complex smallL, sdk_betaew_diag, sdk_cew_diag
      external smallL, sdk_betaew_diag, sdk_cew_diag
      double precision pi, cw2, sw2, Qt
      parameter (pi=3.14159265358979323846d0)

      include 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      logical was_sud_mod_1

      include 'ewsudakov_haslo.inc'

      ! given the to_amp_split_ewsud_der (derivatives of the ME's wrt
      ! the various parameters, amp_split_ewsud are filled with
      ! the parameter-renormalisation contribution

      was_sud_mod_1=.false.

      if(sud_mod.eq.1) then
       was_sud_mod_1=.true.
       sud_mod=0
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      Qt=2d0/3d0

      amp_split_ewsud(:) = (0d0,0d0)

      ls = smallL(invariants(1,2))
      ls_o_mt = dble(gal(1))**2 / (4d0*pi)**2 * dlog(invariants(1,2)/mdl_mt**2) 

CC DAVIDE CHANGE HERE      
      ! the parameter renormalisation in Denner-Pozzorini reads:
      !  dM/de de + dM/dcw dcw + dM/dht dht + dM/dhh dhh

      ! 1) dM/de de = dM/dalpha dalpha, with dalpha=2 Z_e / 4Pi
      !    remember, we derive wrt alpha^-1
      dalpha = -sdk_betaew_diag(22) / aewm1 
      dalpha = dalpha * ls
c      amp_split_ewsud(:) = amp_split_ewsud_der(:,1) * ( - aewm1**2) * 
c     $       dAlpha 

      ! 2) dM/dcw = dM/dmw dmw + dM/dmz dmz
      dmw2 = - (sdk_betaew_diag(24) - 4d0 * sdk_cew_diag(250,0,1))
     $      - 3d0*mdl_mt**2/2d0/mdl_mw**2/sw2
      dmw2 =  dmw2 * mdl_mw**2 * ls

      dmz2 = - (sdk_betaew_diag(23) - 4d0 * sdk_cew_diag(250,0,1))
     $      - 3d0*mdl_mt**2/2d0/mdl_mw**2/sw2
      dmz2 =  dmz2 * mdl_mz**2 * ls

      dGmudalpha=  MDL_GF*aewm1 
  
      dGmudmz2= MDL_GF*(-mdl_mw**2/mdl_mz**4/(1d0-mdl_mw**2/mdl_mz**2))

      dGmudmw2=MDL_GF*(-1d0/mdl_mw**2+1d0/mdl_mz**2/(1d0-mdl_mw**2/mdl_mz**2))

      dGmu=dGmudalpha*dalpha+dGmudmz2*dmz2+dGmudmw2*dmw2

      dmt = 1d0/4d0/sw2 + 1d0/8d0/sw2/cw2 + 3d0/2d0/cw2*Qt - 3d0/cw2*Qt**2  
     $     + 3d0/8d0/sw2 * mdl_mt**2/mdl_mw**2

      dmt =  dmt * mdl_mt * ls

      dmh2 = 1d0/2d0/sw2 
     $ * (
     $ 9d0*mdl_mw**2/mdl_mh**2 * (1d0 + 1d0/2d0/cw2**2) 
     $ - 3d0/2d0 * (1d0 + 1d0/2d0/cw2) + 15d0/4d0 * mdl_mh**2/mdl_mw**2
     $) + 3d0/2d0/sw2 * mdl_mt**2/mdl_mw**2 * (1d0-6d0*mdl_mt**2/mdl_mh**2 )

      dmh2 = dmh2 * mdl_mh**2 * ls

      dt=1d0/(mdl_ee*dsqrt(sw2)*mdl_mw)*
     .   (-3d0/2d0 * mdl_mw**2 * ( mdl_mz**2/cw2+2d0* mdl_mw**2)
     .    -mdl_mh**2/4d0 *( 2d0* mdl_mw**2 + mdl_mz**2 +3d0* mdl_mh**2) + 6d0 * mdl_mt**4)
   
      dt = dt * ls 
      
      dheff_o_heff = mdl_ee/(2d0*dsqrt(sw2))*dt/(mdl_mw*mdl_mh**2)

      dmt_QCD =  - 3d0 * 4d0/3d0 

      dmt_QCD =  dmt_QCD * mdl_mt * ls_o_mt * (G/gal(1))**2

      if(has_lo1) then
  
        amp_split_ewsud(:) = amp_split_ewsud_der(:,1) *dGmu

        amp_split_ewsud(:) = amp_split_ewsud(:) + 
     $      amp_split_ewsud_der(:,2)/(2d0*mdl_mz) * dmz2 + 
     $      amp_split_ewsud_der(:,3)/(2d0*mdl_mw) * dmw2 +
     $      amp_split_ewsud_der(:,4) * dmt +
     $      amp_split_ewsud_der(:,5)/(2d0*mdl_mh) * dmh2 +
     $      amp_split_ewsud_der(:,6) * dheff_o_heff 

      endif

      if(was_sud_mod_1) sud_mod=1

      if (has_lo2) then
        if (sud_mod.eq.2) then
         amp_split_ewsud(:) = amp_split_ewsud(:) +
     $      amp_split_ewsud_der2(:,4) * dmt_QCD 

c      print*,"from dmt_QCD", amp_split_ewsud_der2(:,4) * dmt_QCD

        endif       
      endif
      ! LEAVE EMPTY FOR THE MOMENT
!    correct by a factor 2 ok?
      amp_split_ewsud(:)=amp_split_ewsud(:)/2d0

      return
      end


      double complex function GET_QCD_LO2(pdglist, hels, iflist, invariants,iamp)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'

c      double complex bigL, smallL, smallLem
c      external smallLem

      integer i,j,iamp
      double precision s, rij

c      logical   printinewsdkf
c      common /to_printinewsdkf/printinewsdkf

c      integer   deb_settozero
c      common /to_deb_settozero/deb_settozero

      double precision pi
      parameter (pi=3.14159265358979323846d0)
c      double complex imlog

c      logical s_to_rij
c      COMMON /to_s_to_rij/ s_to_rij

c      double complex smallL_a_over_b_sing, bigL_a_over_b_sing
c      external smallL_a_over_b_sing, bigL_a_over_b_sing

c      double complex smallL_rij_over_s, bigL_rij_over_s

      include 'orders.inc'
      integer orders(nsplitorders)
      double precision QCDlogs,logfromLOip1

      include "q_es.inc"

      Integer sud_mod
      COMMON /to_sud_mod/ sud_mod

      include 'ewsudakov_haslo.inc'

      include 'run.inc'

      get_qcd_lo2 =  CMPLX(0d0,0d0)

      if(.not.has_lo2) return

      logfromLOip1=0d0

      if (sud_mod.eq.2) then

        QCDlogs=4d0/3d0 * (G**2/4d0/pi)/(2d0*pi) * (1d0) *
     .   ( (dlog(qes2/mdl_mt**2))**2 +  1d0 * dlog(qes2/mdl_mt**2)      )

        do i=1,nexternal-1
          if(abs(pdglist(i)).eq.6) logfromLOip1=
     .    logfromLOip1+QCDlogs/4d0
        enddo

        call amp_split_pos_to_orders(iamp, orders)

        logfromLOip1=logfromLOip1+1d0/3d0 /4d0* (G**2/4d0/pi)/(2d0*pi)
     .  *dble(orders(QCD_POS))  * dlog(qes2/mdl_mt**2)

        do i=1,nexternal-1
          if(abs(pdglist(i)).eq.21) then
            logfromLOip1=
     .      logfromLOip1-2d0*1d0/3d0 /4d0* (G**2/4d0/pi)/(2d0*pi)
     .      * dlog(qes2/mdl_mt**2)
           endif
        enddo

        get_qcd_lo2 = CMPLX(logfromLOip1,0d0)

      elseif (sud_mod.eq.1.or.sud_mod.eq.0) then

        s=invariants(1,2)

        call amp_split_pos_to_orders(iamp, orders)

        logfromLOip1=logfromLOip1+1d0/3d0 /4d0* (G**2/4d0/pi)/(2d0*pi)
     .  *dble(orders(QCD_POS))  * dlog(scale**2/mdl_mt**2)

        do i=1,nexternal-1
          if(abs(pdglist(i)).eq.21) then
            logfromLOip1=
     .      logfromLOip1-2d0*1d0/3d0 /4d0* (G**2/4d0/pi)/(2d0*pi)
     .      * dlog(s/mdl_mt**2)
           endif
        enddo

        get_qcd_lo2 = CMPLX(logfromLOip1,0d0)

c      elseif (sud_mod.eq.0) then  
c      TOBE modified 
c
c        print*,"sud_mod=0 and QCD not implemented.
c     .  Switch to sud_mod=1 or turn this error off."
c        stop
c      
      else 
        print*,"sud_mod=",sud_mod,"is not possible"
        stop
      endif

c      get_qcd_lo2 =0d0

      return
      end


