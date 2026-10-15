       double precision function photonpdfsquare(x1,x2)
       implicit none
       double precision x1,x2
       write(*,*) "ERROR: gamma-UPC dummy linked"
       photonpdfsquare = 1.0
       stop 'gamma-UPC 1'
       return
       end


       subroutine Get_nucleus_RA(nb_p,nb_n,RAI)
       implicit none
       integer nb_p,nb_n
       double precision RAI
       write(*,*) "ERROR: gamma-UPC dummy linked"
       RAI = 1d0
       stop 'gamma-UPC 2'
       return
       end
