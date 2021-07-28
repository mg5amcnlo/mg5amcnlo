      subroutine get_orderstags_glob_infos()
      use extra_weights
      implicit none
      integer n_orderstags
      integer orderstags_glob(maxorders)
      common /c_orderstags_glob/n_orderstags, orderstags_glob
      integer j

      open(unit=78,file="orderstags_glob.dat",status="old",err=101)
      goto 99
101   open(unit=78,file="../orderstags_glob.dat",status="old")

99    read(78,*) n_orderstags
      write(*,*) 'get_orderstags_glob_infos: n_orderstags=', n_orderstags 
      read(78,*) (orderstags_glob(j), j=1, n_orderstags)
      write(*,*) 'get_orderstags_glob_infos: orderstags_glob',
     $             (orderstags_glob(j), j=1, n_orderstags)
      return
      end


      integer function get_orderstags_glob_pos(tag)
      use extra_weights
      implicit none
      integer tag
      integer n_orderstags
      integer orderstags_glob(maxorders)
      common /c_orderstags_glob/n_orderstags, orderstags_glob
      integer j

      do j = 1, n_orderstags
         if (orderstags_glob(j).eq.tag) then
             get_orderstags_glob_pos = j
             return
          endif
      enddo
      write(*,*) 'ERROR, get_orderstags_glob_pos, not found', tag
      stop 1
      return
      end
        
