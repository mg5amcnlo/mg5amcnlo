      logical function testhel(n,nhel,ihel)
      implicit none
      integer n,nhel(n), ihel

      include "nexternal.inc"

      integer i

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      integer truehel(nexternal)
      data truehel  /1,-1,-1,1,1,-1/
      

      testhel=.true.

      if (n.eq.nexternal) then ! real matrix element
         do i=1,n
c$$$            if (.not.(i.eq.i_fks .or. i.eq.j_fks))then
c$$$            if (.not.(i.eq.i_fks))then ! if also fixing j_fks
               if (nhel(i).ne.truehel(i)) testhel=.false.
c$$$            endif
         enddo
      elseif(n.eq.nexternal-1) then ! Born-type configuration
         do i=1,n
            if (i.ge.max(i_fks,j_fks))then
               if (nhel(i).ne.truehel(i)-1) testhel=.false.
            elseif (i.ne.min(i_fks,j_fks)) then
               if (nhel(i).ne.truehel(i)) testhel=.false.               
            elseif (i.eq.min(i_fks,j_fks)) then  ! if also fixing j_fks
               if (nhel(i).ne.truehel(i)) testhel=.false.
            endif
         enddo
      else
         write (*,*) 'ERROR in testhel'
         stop
      endif

      return
      end
