subroutine sborn_sf(p_born,m,n,wgt)
          implicit none
          include "nexternal.inc"
          include 'coupl.inc'        
          double precision p_born(0:3,nexternal-1), wgt
          integer nsqampso
          parameter (nsqampso = %(nsqorders)d)

C return the color-linked borns if i_fks is a color octet, 
C the charge-linked if it is a color singlet
          double complex wgt_born(2,0:nsqampso)
          double precision wgt_col
          double precision chargeprod
          integer i,m,n 
      include 'orders.inc'
      COMPLEX*16 ANS_CNT(2, NSPLITORDERS)
      common /c_born_cnt/ ans_cnt
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt 
      double precision charges_born(nexternal-1)
      common /c_charges_born/charges_born
      logical need_color_links, need_charge_links
      common /c_need_links/need_color_links, need_charge_links
      double precision pmass(nexternal), zero
      parameter (zero=0d0)
      double precision amp_split_soft(amp_split_size)
      common /to_amp_split_soft/amp_split_soft

      chargeprod = 0d0

      if (need_color_links.and.need_charge_links) then
         write(*,*) 'ERROR IN SBORN_SF, both color and charged links are needed'
         stop
      endif
C check if need color or charge links, and include the gs/w**2 term here
      if (need_color_links) then
%(iflines_col)s
      wgt = wgt_col * g**2
C update the amp_split_soft, which is summed in sbornsoft
      amp_split_soft(1:amp_split_size) = dble(amp_split_cnt(1:amp_split_size,1,qcd_pos)) * g**2

      else if (need_charge_links) then
      chargeprod = charges_born(m) * charges_born(n)
      if ((m.le.nincoming.and.n.gt.nincoming) .or. (n.le.nincoming.and.m.gt.nincoming)) chargeprod = - chargeprod
C add a factor 1/2 for the self-eikonal soft link
      if (m.eq.n) chargeprod = chargeprod / 2d0
      wgt = dble(ans_cnt(1, qed_pos)) * chargeprod * dble(gal(1))**2
C update the amp_split_soft, which is summed in sbornsoft
      amp_split_soft(1:amp_split_size) = dble(amp_split_cnt(1:amp_split_size,1,qed_pos)) * chargeprod * dble(gal(1))**2
      endif

      return 
      end
