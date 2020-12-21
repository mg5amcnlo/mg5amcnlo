C some functions which deal with the splitorders
      subroutine fill_needed_splittings()
      ! loop over the FKS configurations and fill the split_type_used
      ! common blocks
      implicit none
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "fks_info.inc"
      include "orders.inc"
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used
      integer i, j
      do j = 1, nsplitorders
        split_type_used(j)=.false.
      enddo
      do i = 1, fks_configs
        do j = 1, nsplitorders
          split_type_used(j)=split_type_used(j).or.
     %      split_type_d(i,j)
        enddo
      enddo
      write(*,*) 'SPLIT TYPE USED:', split_type_used
      return
      end


      integer function get_orders_tag(ord)
C a function that assigns to a given order
C array an integer number
      implicit none
      include 'orders.inc'
      integer ord(nsplitorders)
      integer i,j
      integer base, step
      parameter(base=100)
      ! this is for the printout of the informations
      logical firsttime, firsttime_contr(amp_split_size)
      data firsttime/.true./
      data firsttime_contr/amp_split_size * .true./
      integer orders_to_amp_split_pos

      ! print out some extra informations
      if (firsttime) write(*,fmt='(a)',advance="NO") 
     $    "INFO: orders_tag_plot is computed as:"

      get_orders_tag=0
      step=1
      do i =1, nsplitorders
        if (firsttime) write(*,fmt='(3a,i8)',advance="NO") 
     $      "         + ", ordernames(i), " * ", step
        get_orders_tag=get_orders_tag+step*ord(i)
        step=step*100
      enddo
      if (firsttime) then
        write(*,*)
        firsttime=.false.
      endif

      if (firsttime_contr(orders_to_amp_split_pos(ord))) then
        write(*,*) 'orders_tag_plot= ', get_orders_tag, ' for ',
     #     (ordernames(i),",",i=1,nsplitorders), ' = ',
     #     (ord(i),",",i=1,nsplitorders)
        firsttime_contr(orders_to_amp_split_pos(ord)) = .false.
      endif

      return 
      end


      integer function get_orders_tag_from_amp_pos(iamp)
C     calls get_orders_tag for the orders corresponding to 
C     the iamp-th amp_split
      implicit none
      integer iamp
      include 'orders.inc'
      integer ord(nsplitorders)
      integer get_orders_tag

      call amp_split_pos_to_orders(iamp, ord)
      get_orders_tag_from_amp_pos = get_orders_tag(ord)

      return
      end
      
      
      integer function orders_to_amp_split_pos(ord)
C helper function to keep track of the different coupling order combinations
C given the squared orders ord, return the corresponding position into the amp_split array
      implicit none
      include 'orders.inc'
      integer ord(nsplitorders)
      integer i,j
      include 'amp_split_orders.inc'

      do i=1, amp_split_size
        do j=1, nsplitorders
          if (amp_split_orders(i,j).ne.ord(j)) goto 999 
        enddo
        orders_to_amp_split_pos = i
        return
 999    continue   
      enddo

      WRITE(*,*) 'ERROR:: Stopping function orders_to_amp_split_pos'
      WRITE(*,*) 'Could not find orders ',(ord(i),i=1
     $ ,nsplitorders)
      stop

      return
      end


      subroutine amp_split_pos_to_orders(pos, orders)
C helper function to keep track of the different coupling order combinations
C given the position pos, return the corresponding order powers orders
C it is the inverse of orders_to_amp_split_pos
      implicit none
      include 'orders.inc'
      integer pos, orders(nsplitorders)
      integer i
      include 'amp_split_orders.inc'

C sanity check
      if (pos.gt.amp_split_size.or.pos.lt.0) then
        write(*,*) 'ERROR in amp_split_pos_to_orders'
        write(*,*) 'Invalid pos', pos, amp_split_size
        stop 1
      endif

      do i = 1, nsplitorders
        orders(i) = amp_split_orders(pos,i)
      enddo
      return
      end


      integer function lo_qcd_to_amp_pos(qcdpower)
      implicit none
      integer qcdpower
      include 'orders.inc'
      integer pos, orders(nsplitorders)
      do pos = 1, amp_split_size_born
        call amp_split_pos_to_orders(pos, orders)
        if (orders(qcd_pos).eq.qcdpower) exit
      enddo
      lo_qcd_to_amp_pos = pos
      return
      end


      integer function nlo_qcd_to_amp_pos(qcdpower)
      implicit none
      integer qcdpower
      include 'orders.inc'
      integer pos, orders(nsplitorders)
      do pos = amp_split_size_born + 1, amp_split_size
        call amp_split_pos_to_orders(pos, orders)
        if (orders(qcd_pos).eq.qcdpower) exit
      enddo
      nlo_qcd_to_amp_pos = pos
      return
      end


      subroutine check_amp_split()
C check that amp_split_pos_to_orders and orders_to_amp_split_pos behave
C as expected (one the inverse of the other)
C Check also get_orders_tag vs get_orders_tag_from_amp_pos
C Stop the code if anything wrong is found
C Also, print on screen a summary of the orders in amp_split 
      implicit none
      include 'orders.inc'
      integer orders_to_amp_split_pos
      integer i, pos
      integer ord(nsplitorders)
      integer get_orders_tag, get_orders_tag_from_amp_pos

      do i = 1, amp_split_size
        call amp_split_pos_to_orders(i, ord)
        pos = orders_to_amp_split_pos(ord)

        if (pos.ne.i) then
          write(*,*) 'ERROR#1 in check amp_split', pos, i 
          write(*,*) 'ORD is ', ord
          stop 1
        endif

        if (get_orders_tag(ord).ne.get_orders_tag_from_amp_pos(i)) then
          write(*,*) 'ERROR#2 in check amp_split', get_orders_tag(ord), 
     $    get_orders_tag_from_amp_pos(i) 
          write(*,*) 'I, ORD ', i, ord
          stop 1
        endif

        write(*,*) 'AMP_SPLIT: ', i, 'correspond to S.O.', ord
      enddo

      return
      end



