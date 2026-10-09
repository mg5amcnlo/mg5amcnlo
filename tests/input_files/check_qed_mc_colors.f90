program check_qed_mc_colors
  implicit none
  include 'orders.inc'
  integer :: jpart(7,-2:7),ifks,jfks,fks_j_from_i(5,0:5),particle_type(5),pdg_type(5)
  integer :: mother_color,mother_anticolor
  logical :: split_type(nsplitorders)
  common /c_split_type/ split_type
  common /fks_indices/ ifks,jfks
  common /c_fks_inc/ fks_j_from_i,particle_type,pdg_type
  common /test_born_color/ mother_color,mother_anticolor
  ifks=5
  particle_type=1
  read(*,*) jfks,particle_type(ifks),particle_type(1),mother_color,mother_anticolor
  particle_type(jfks)=particle_type(1)
  split_type=.false.
  split_type(qed_pos)=.true.
  jpart=0
  call fill_icolor_H(1,jpart,.true.)
  write(*,*) jpart(4:5,ifks),jpart(4:5,jfks)
end program

subroutine fill_icolor_S(iflow,jpart,lc)
  implicit none
  integer :: iflow,jpart(7,-2:7),lc,ifks,jfks,mother_color,mother_anticolor
  common /fks_indices/ ifks,jfks
  common /test_born_color/ mother_color,mother_anticolor
  jpart=0
  jpart(4,jfks)=mother_color
  jpart(5,jfks)=mother_anticolor
  lc=503
end subroutine

double precision function ran2()
  implicit none
  ran2=0.3d0
end function
