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

      logical s_to_rij
      COMMON /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      COMMON /rij_ge_mw/ rij_ge_mw
C     deb_settozero must be 0 (nonzero values switch off Sudakov terms)
      integer deb_settozero
      common /to_deb_settozero/deb_settozero
C     sud_filter_hel (default .true.) is cleared on every call: the
C     helicity mask good_hel of ewsudakov_wrapper.f is set at the first
C     call and would otherwise be applied to all later events
      logical sud_filter_hel
      common /to_filter_hel/ sud_filter_hel
C-----
C  BEGIN CODE
C-----

      deb_settozero = 0
      sud_filter_hel = .false.

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


      subroutine density_sudakov_py(p_born_in, gstr_in, nres,
     $     res_indices, res_dims, density_born, density_delta,
     $     density_delta_var, born_diag, results)
c**************************************************************************
c     EW Sudakov corrections per resonance-helicity configuration for the
c     spin-density reweighting (see density_matrix_wrapper). Outputs, with
c     D = product of res_dims(1:nres):
c       density_delta(1:D)     - sum of ME*delta, s_to_rij=.true.
c       density_delta_var(1:D) - the same with s_to_rij=.false.
c       born_diag(1:D)         - Born, summed over spectator helicities
c       density_born           - zeros
c       results(1)             - Born amp_split(1); results(2:6) = 0
c     delta_h = density_delta(h)/born_diag(h) is amplitude-level. sud_mod=1,
c     rij_ge_mw=.true. (|r_ij| < MW^2 set to +-MW^2).
c**************************************************************************
      implicit none

      include 'nexternal.inc'
      include 'coupl.inc'
      include 'orders.inc'

C     Maximum density matrix dimension (for ttWW: 2*2*3*3 = 36)
      integer MAX_DIM
      parameter (MAX_DIM = 36)
C     Triangular storage size: MAX_DIM*(MAX_DIM+1)/2 = 666
      integer MAX_DIM_TRI
      parameter (MAX_DIM_TRI = 666)
      integer MAX_RES
      parameter (MAX_RES = 10)

C     Arguments
      double precision p_born_in(0:3,nexternal-1)
      double precision gstr_in
      integer nres
      integer res_indices(MAX_RES)
      integer res_dims(MAX_RES)
      double complex density_born(MAX_DIM_TRI)
      double complex density_delta(MAX_DIM)
      double complex density_delta_var(MAX_DIM)
      double precision born_diag(MAX_DIM)
      double precision results(6)

CF2PY intent(in) :: p_born_in, gstr_in, nres, res_indices, res_dims
CF2PY intent(out) :: density_born, density_delta, density_delta_var
CF2PY intent(out) :: born_diag, results

C     Local variables
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i, total_dim

      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double complex amp_split_ewsud_par(amp_split_size)
      common /to_amp_ewsud_par/amp_split_ewsud_par

      integer sud_mod
      common /to_sud_mod/ sud_mod
      integer nfksprocess
      common/c_nfksprocess/nfksprocess

      logical sud_mc_hel
      common /to_mc_hel/ sud_mc_hel

      double precision wgt_born, born

      logical firsttime
      data firsttime/.true./

      logical s_to_rij
      common /to_s_to_rij/ s_to_rij
      logical rij_ge_mw
      common /rij_ge_mw/ rij_ge_mw
C     deb_settozero must be 0 (nonzero values switch off Sudakov terms)
      integer deb_settozero
      common /to_deb_settozero/deb_settozero

      integer ewsud_helselect
      common/to_ewsud_helselect/ewsud_helselect
      logical sud_filter_hel
      COMMON /to_filter_hel/ sud_filter_hel

C-----
C  BEGIN CODE
C-----

C     Validate nres before any loop indexes the
C     fixed-size resonance arrays.
      if (nres.lt.0 .or. nres.gt.MAX_RES) then
        write (*,*) 'ERROR in density_sudakov_py: nres=', nres
        write (*,*) ' exceeds MAX_RES=', MAX_RES
        stop 1
      endif

      deb_settozero = 0

      nfksprocess = 1
      sud_mc_hel = .false.

      if (firsttime) then
        call setpara('param_card.dat')
        firsttime = .false.
      endif

      g = gstr_in
      call update_as_param()
      p_born(:,:) = p_born_in(:,:)

      total_dim = 1
      do i = 1, nres
        total_dim = total_dim * res_dims(i)
      enddo

C     density_delta/born_diag hold MAX_DIM entries and allow_hel in
C     density_matrix_wrapper holds 300 (indexed up to total_dim*nres);
C     both are passed as assumed-size dummies, so an oversize helicity
C     space would overrun them silently.
      if (total_dim.gt.MAX_DIM .or. total_dim*nres.gt.300) then
        write (*,*) 'ERROR in density_sudakov_py: helicity space too'
        write (*,*) ' large: total_dim=', total_dim, ' nres=', nres
        write (*,*) ' caps: MAX_DIM=', MAX_DIM, ' allow_hel=300'
        stop 1
      endif

      do i = 1, MAX_DIM * (MAX_DIM + 1) / 2
        density_born(i) = dcmplx(0d0, 0d0)
      enddo
      do i = 1, MAX_DIM
        density_delta(i) = dcmplx(0d0, 0d0)
        density_delta_var(i) = dcmplx(0d0, 0d0)
        born_diag(i) = 0d0
      enddo
      do i = 1, 6
        results(i) = 0d0
      enddo

      sud_mod = 1
      rij_ge_mw = .true.

      call sborn(p_born, born)
      wgt_born = amp_split(1)
      results(1) = wgt_born

C     No single-helicity selection, no helicity filtering
      ewsud_helselect = 0
      sud_filter_hel = .false.

C     Central: s_to_rij = .true. (SSC keeps the log^2(|r_ij|/s) and
C     imaginary-part terms of the s -> r_ij replacement)
      s_to_rij = .true.
      call density_matrix_wrapper(p_born, nres, res_indices, res_dims,
     $       density_born, density_delta, born_diag)
      results(2) = 0d0

C     Variation: s_to_rij = .false. (born_diag is recomputed unchanged)
      s_to_rij = .false.
      call density_matrix_wrapper(p_born, nres, res_indices, res_dims,
     $       density_born, density_delta_var, born_diag)
      results(3) = 0d0

      return
      end
