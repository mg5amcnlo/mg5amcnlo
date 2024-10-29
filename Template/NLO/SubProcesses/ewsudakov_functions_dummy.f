
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


