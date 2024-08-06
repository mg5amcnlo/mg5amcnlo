      subroutine ewsudakov_py(p_born_in, gstr_in, results) 
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
