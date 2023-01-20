      program print_alpha
      implicit none
      integer id
      double precision energy, alpha, q

      read(*,*) id, q
      call initgridmela_lhaid(id)

      call alphaq2(q**2, alpha)

      write(*,*) 'ALPHAVALUE', alpha 


      return
      end


      subroutine fill_needed_splittings()
      implicit none
      ! dummy
      return
      end
