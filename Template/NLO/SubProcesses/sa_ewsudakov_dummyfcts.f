      ! dummy sub
      subroutine fill_needed_splittings()
      implicit none
      return
      end


      function ran2()
!     Wrapper for the random numbers; needed for the NLO stuff
      implicit none
      double precision ran2,x,a,b
      integer ii,jconfig
      a=0d0                     ! min allowed value for x
      b=1d0                     ! max allowed value for x
      ii=0                      ! dummy argument of ntuple
      jconfig=1          ! integration channel (for off-set)
      call ntuple(x,a,b,ii,jconfig)
      ran2=x
      return
      end
