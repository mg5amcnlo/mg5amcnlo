C     EW Sudakov corrections per resonance-helicity configuration h for
C     the spin-density reweighting
C        B^EWSL[h,h'] = B[h,h'] * (1 + delta_h + delta*_h'),
C     with the amplitude-level delta_h averaged over spectator helicities
C     (exact for h = h' only). The density matrix B[h,h'] is computed
C     separately (GET_DENSITY of the standalone density-matrix output).

      subroutine density_matrix_wrapper(p, nres, res_indices, res_dims,
     $     density_born, density_delta, born_diag)
C --------------------------------------------------------------------------
C     Arguments:
C       p(0:3, nexternal-1)  - external momenta (restored on exit)
C       nres                 - number of resonances
C       res_indices(nres)    - resonance leg positions (1-based)
C       res_dims(nres)       - spin dimension per resonance (2=fermion, 3=vector)
C       density_born(*)      - output: zeros, D*(D+1)/2 entries
C       density_delta(*)     - output: sum of ME*delta, D entries
C       born_diag(*)         - output: Born, D entries
C     D = product of res_dims. density_delta and born_diag are summed over
C     spectator helicities and both carry 1/IDEN;
C     delta_h = density_delta(h)/born_diag(h). ME is the Born, or the
C     Goldstone-boson ME for configurations with longitudinal vector legs;
C     for those delta_h includes the ratio of the Goldstone ME to the Born.
C --------------------------------------------------------------------------
      implicit none
      include 'nexternal.inc'

C     Arguments
      double precision p(0:3, nexternal-1)
      integer nres
      integer res_indices(nres)
      integer res_dims(nres)
      double complex density_born(*)
      double complex density_delta(*)
      double precision born_diag(*)

C     Local variables
      integer i, k, n
      integer ih
      integer total_dim
      integer h_indices(10)        ! Individual resonance hel indices (0-based)
      integer hel_values(10)       ! Actual helicity values (-1, 0, +1)

      integer allow_hel(300)       ! D*nres entries, size checked by the caller

      double complex delta_total_this_hel
      double precision p_save(0:3, nexternal-1)
      integer ileg, imu
      logical saved_filter_hel

      double complex CZERO
      parameter (CZERO = (0d0, 0d0))

C     Results of sudakov_wrapper
      include 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_XXC(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      common /to_amp_ewsud_par/amp_split_ewsud_par
      DOUBLE PRECISION AMP_BORN_SUMHEL
      common /to_amp_born_sumhel/amp_born_sumhel

C     Resonance-helicity selection for sudakov_wrapper
      integer ewsud_nres
      integer ewsud_res_pos(10)
      integer ewsud_res_hel(10)
      common/to_ewsud_res_sel/ewsud_nres, ewsud_res_pos, ewsud_res_hel

      logical sud_filter_hel
      COMMON /to_filter_hel/ sud_filter_hel

      LOGICAL CALCULATEDBORN
      COMMON/CCALCULATEDBORN/CALCULATEDBORN


C --------------------------------------------------------------------------
C     BEGIN CODE
C --------------------------------------------------------------------------

      total_dim = 1
      do i = 1, nres
        total_dim = total_dim * res_dims(i)
      enddo

      do i = 1, total_dim * (total_dim + 1) / 2
        density_born(i) = CZERO
      enddo
      do i = 1, total_dim
        density_delta(i) = CZERO
        born_diag(i) = 0d0
      enddo

C     allow_hel((ih-1)*nres+n) = helicity of resonance n in configuration
C     ih = 1..D, with ih-1 = h_1 + res_dims(1)*(h_2 + res_dims(2)*(...))
C     for 0-based indices h_n (first resonance runs fastest).

      do ih = 1, total_dim
        k = ih - 1
        do n = 1, nres
          h_indices(n) = mod(k, res_dims(n))
          k = k / res_dims(n)
        enddo

        do n = 1, nres
          if (res_dims(n) .eq. 2) then
C           Fermion: 0 -> -1, 1 -> +1
            hel_values(n) = 2 * h_indices(n) - 1
          else if (res_dims(n) .eq. 3) then
C           Vector: 0 -> -1, 1 -> 0, 2 -> +1
            hel_values(n) = h_indices(n) - 1
          else
            hel_values(n) = h_indices(n)
          endif
        enddo

        do n = 1, nres
          allow_hel((ih-1)*nres + n) = hel_values(n)
        enddo
      enddo

C     For each configuration ih, sudakov_wrapper restricted to the
C     resonance helicities allow_hel(ih) sums over spectator helicities.

      saved_filter_hel = sud_filter_hel
      sud_filter_hel = .false.

C     Save momenta (PAR numerical derivatives may modify p)
      do ileg = 1, nexternal-1
        do imu = 0, 3
          p_save(imu, ileg) = p(imu, ileg)
        enddo
      enddo

      do ih = 1, total_dim
        do ileg = 1, nexternal-1
          do imu = 0, 3
            p(imu, ileg) = p_save(imu, ileg)
          enddo
        enddo

        ewsud_nres = nres
        do n = 1, nres
          ewsud_res_pos(n) = res_indices(n)
          ewsud_res_hel(n) = allow_hel((ih-1)*nres + n)
        enddo

        CALCULATEDBORN = .FALSE.

        call sudakov_wrapper(p)

        delta_total_this_hel = amp_split_ewsud_lsc(1)
     $       + amp_split_ewsud_ssc(1)
     $       + amp_split_ewsud_xxc(1)
     $       + amp_split_ewsud_par(1)

        density_delta(ih) = delta_total_this_hel
        born_diag(ih) = amp_born_sumhel
      enddo

      ewsud_nres = 0

      sud_filter_hel = saved_filter_hel
      CALCULATEDBORN = .FALSE.
      do ileg = 1, nexternal-1
        do imu = 0, 3
          p(imu, ileg) = p_save(imu, ileg)
        enddo
      enddo

      return
      end
