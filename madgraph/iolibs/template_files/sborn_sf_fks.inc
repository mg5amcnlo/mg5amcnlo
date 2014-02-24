subroutine sborn%(iborn)d_sf(p_born,m,n,wgt)
          implicit none
          include "nexternal.inc"
          double precision p_born(0:3,nexternal-1), wgt(0:%(nsqorders)d)

C return the color-linked borns if i_fks is a color octet, 
C the charge-linked if it is a color singlet
          double complex wgt_born(2,0:%(nsqorders)d)
          double precision wgt_col(0:%(nsqorders)d)
          double precision chargeprod
          integer i,m,n 
C FKS stuff:
      integer i_fks, j_fks
      common/fks_indices/i_fks, j_fks
      integer fks_j_from_i(nexternal,0:nexternal), particle_type(nexternal), pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i, particle_type, pdg_type
      double precision charges_born(nexternal-1)
      common /c_charges_born/charges_born
      include 'coupl.inc'
      double precision pmass(nexternal), zero
      parameter (zero=0d0)
      include 'pmass.inc'

      do i = 0, %(nsqorders)d
      wgt(i) = 0d0
      wgt_col(i) = 0d0
      wgt_born(1, i) = (0d0, 0d0)
      wgt_born(2, i) = (0d0, 0d0)
      enddo
      chargeprod = 0d0

C check if i_fks is singlet or octet
      if (particle_type(i_fks).eq.8) then
%(iflines_col)s
      do i = 0, %(nsqorders)d
      wgt(i) = wgt_col(i)
      enddo
      else if (particle_type(i_fks).eq.1) then
      if (pmass(j_fks).gt.0 .or. m.ne.n) then 
      call sborn%(iborn)d_splitorders(p_born, wgt_born)
      chargeprod = charges_born(m) * charges_born(n)
      endif
      do i = 0, %(nsqorders)d
      wgt(i) = dble(wgt_born(1,i)) * chargeprod
      enddo
      endif

      return 
      end
