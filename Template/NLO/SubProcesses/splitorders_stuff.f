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

      get_orders_tag=0
      step=1
      do i =1, nsplitorders
        get_orders_tag=get_orders_tag+step*ord(i)
        step=step*100
      enddo

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


      subroutine check_amp_split()
C check that amp_split_pos_to_orders and orders_to_amp_split_pos behave
C as expected (one the inverse of the other)
C Stop the code if anything wrong is found
C Also, print on screen a summary of the orders in amp_split 
      implicit none
      include 'orders.inc'
      integer orders_to_amp_split_pos
      integer i, pos
      integer ord(nsplitorders)

      do i = 1, amp_split_size
        call amp_split_pos_to_orders(i, ord)
        pos = orders_to_amp_split_pos(ord)
        if (pos.ne.i) then
          write(*,*) 'ERROR in check amp_split', pos, i 
          write(*,*) 'ORD is ', ord
          stop 1
        endif
        write(*,*) 'AMP_SPLIT: ', i, 'correspond to S.O.', ord
      enddo

      return
      end

      
