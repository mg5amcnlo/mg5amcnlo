      subroutine block_stat(i,tag)
      implicit none
      integer i
      character*20 tag
      return
      end

      subroutine clear_block_stat(config,perm_pos)
      implicit none
      integer i,config,perm_pos
      return 
      end

c      integer good_points(max_blok,max_blok,max_perm),bad_points(max_blok,max_blok,max_perm)
c      common /to_stat/good_points, bad_points
c---
c Begin code
c---
c      if (jac.le.0) then
c        bad_points(i,config,perm_pos)=bad_points(i,config,perm_pos)+1
c      else
c        good_points(i,config,perm_pos)=good_points(i,config,perm_pos)+1
c      endif



