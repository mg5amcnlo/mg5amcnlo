subroutine sborn%(iborn)d_sf(p_born,m,n,wgt)
          implicit none
          include "nexternal.inc"
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
      include 'coupl.inc'
      double precision pmass(nexternal), zero
      parameter (zero=0d0)

      chargeprod = 0d0

      if (need_color_links.and.need_charge_links) then
         write(*,*) 'ERROR IN SBORN_SF, both color and charged links are needed'
         stop
      endif
C check if need color or charge links
      if (need_color_links) then
%(iflines_col)s
      wgt = wgt_col
      else if (need_charge_links) then
      chargeprod = charges_born(m) * charges_born(n)
      wgt = dble(ans_cnt(1, qed_pos)) * chargeprod
      endif

      return 
      end
