!                                       !
! build up the necessary combinatorics  !
!                                       !
 module maxnumden
  implicit none
  private
!                                       !
! the maximum number of denominators    !
! (it can be changed by the user)       ! 
!                                       !
  integer, public, parameter :: maxden= 10
 end module maxnumden
! 
 module mbnvalues
  implicit none
  private
  integer, public, dimension(:,:,:), allocatable :: bn1
  integer, public, dimension(:,:,:), allocatable :: bn2
  integer, public, dimension(:,:,:), allocatable :: bn3
  integer, public, dimension(:,:,:), allocatable :: bn4
 end module mbnvalues
!
 module maxsolutions
  implicit none
  private 
  integer, public :: max_solutions= 18
 end module maxsolutions
!
 module combinatorics
  use mbnvalues
  use maxnumden
  implicit none
  private
  integer, public, dimension(1:maxden) :: nbn1= -1
  integer, public, dimension(2:maxden) :: nbn2= -1
  integer, public, dimension(3:maxden) :: nbn3= -1
  integer, public, dimension(4:maxden) :: nbn4= -1
  public load_combinatorics
  contains
!
  subroutine load_combinatorics
  integer :: k,kden,ierr
  integer, parameter :: n1= 1,n2= 2,n3= 3,n4= 4
  integer :: k1,k2,k3,k4,kref,kcount,kc,kvalue
  if (maxden.ge.1) then
    do k= 1,maxden
     nbn1(k)= comb(k,n1)
    enddo
    ierr= -1
    allocate  (bn1(1:maxden,maxden,nbn1(maxden)), stat=ierr)
    if (ierr.ne.0) STOP "Not enough memory to allocate bn1"
    bn1= -1
    do kden= 1,maxden
     kcount= 0
     do k1= 1,kden
       kcount= kcount+1
       bn1(kden,1,kcount)= k1
       kref  = 0; kc    = 1
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1)) then
          kc= kc+1
          kref= k
          bn1(kden,kc,kcount)= k
        endif
       enddo
     enddo
    enddo
  endif
  if (maxden.ge.2) then
    do k= 2,maxden
     nbn2(k)= comb(k,n2)
    enddo
    ierr= -1
    allocate  (bn2(2:maxden,maxden,nbn2(maxden)), stat=ierr)
    if (ierr.ne.0) STOP "Not enough memory to allocate bn2"
    bn2= -1
    do kden= 2,maxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden
       kcount= kcount+1
       bn2(kden,1,kcount)= k1
       bn2(kden,2,kcount)= k2
       kref  = 0; kc    = 2
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2)) then
          kc= kc+1
          kref= k
          bn2(kden,kc,kcount)= k
        endif
       enddo
     enddo; enddo
    enddo
  endif
  if (maxden.ge.3) then
    do k= 3,maxden
     nbn3(k)= comb(k,n3)
    enddo
    ierr= -1
    allocate  (bn3(3:maxden,maxden,nbn3(maxden)), stat=ierr)
    if (ierr.ne.0) STOP "Not enough memory to allocate bn3"
    bn3= -1
    do kden= 3,maxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden; do k3= k2+1,kden
       kcount= kcount+1
       bn3(kden,1,kcount)= k1
       bn3(kden,2,kcount)= k2
       bn3(kden,3,kcount)= k3
       kref  = 0; kc    = 3
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2).and.(k.ne.k3)) then
          kc= kc+1
          kref= k
          bn3(kden,kc,kcount)= k
        endif
       enddo
     enddo; enddo; enddo
    enddo
  endif
  if (maxden.ge.4) then
    do k= 4,maxden
     nbn4(k)= comb(k,n4)
    enddo
    ierr= -1
    allocate  (bn4(4:maxden,maxden,nbn4(maxden)), stat=ierr)
    if (ierr.ne.0) STOP "Not enough memory to allocate bn4"
    bn4= -1
    do kden= 4,maxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden; do k3= k2+1,kden; do k4= k3+1,kden
       kcount= kcount+1
       bn4(kden,1,kcount)= k1
       bn4(kden,2,kcount)= k2
       bn4(kden,3,kcount)= k3
       bn4(kden,4,kcount)= k4
       kref  = 0; kc    = 4
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2) &
           .and.(k.ne.k3).and.(k.ne.k4)) then
          kc= kc+1
          kref= k
          bn4(kden,kc,kcount)= k
        endif
       enddo
     enddo; enddo; enddo; enddo
    enddo
  endif
  end subroutine load_combinatorics
!
  integer function comb(no,ko)
  integer, intent(in) :: no,ko
  integer, dimension(1:4), parameter :: ifact= (/1,2,6,24/) 
  integer :: k,kf
  if ((ko.le.0).or.(ko.gt.4)) then
   stop 'error in function comb'
  endif
  kf= no
  do k= 1,(ko-1)
   kf= kf*(no-k)
  enddo
  kf= kf/ifact(ko)
  comb= kf
  end function comb
!
  subroutine testcomb(np)
  integer, intent(in) :: np
  integer :: ib,j
  call load_combinatorics
!
! D sector:
!
  print*,'          '
  print*,' D-sector:'   
  print*,'          '
  do ib= 1,nbn4(np)
    print*,'         ib=',ib
    do j= 1,np
      print*,'bn4(np,',j,',ib)=',bn4(np,j,ib)
    enddo
  enddo
!
! C sector:
!
  print*,'          '
  print*,' C-sector:'   
  print*,'          '
  do ib= 1,nbn3(np)
    print*,'         ib=',ib
    do j= 1,np
      print*,'bn3(np,',j,',ib)=',bn3(np,j,ib)
    enddo
  enddo
!
! B sector:
!
  print*,'          '
  print*,' B-sector:'   
  print*,'          '
  do ib= 1,nbn2(np)
    print*,'         ib=',ib
    do j= 1,np
      print*,'bn2(np,',j,',ib)=',bn2(np,j,ib)
    enddo
  enddo
!
! A sector:
!
  print*,'          '
  print*,' A-sector:'   
  print*,'          '
  do ib= 1,nbn1(np)
    print*,'         ib=',ib
    do j= 1,np
      print*,'bn1(np,',j,',ib)=',bn1(np,j,ib)
    enddo
  enddo
  end subroutine testcomb 
 end module combinatorics
!
 module dimensions 
  use combinatorics
  use mbnvalues
  implicit none                    
  private
  integer, public :: dmns,dmns_d,dmns_c,dmns_b,dmns_a
  integer, public :: dmns_4,dmns_3,dmns_2,dmns_1
  integer, public, dimension(:,:), allocatable :: bbn1
  integer, public, dimension(:,:), allocatable :: bbn2
  integer, public, dimension(:,:), allocatable :: bbn3
  integer, public, dimension(:,:), allocatable :: bbn4
  public load_dimensions
  contains
! 
  subroutine load_dimensions
  use maxnumden
  dmns  = maxden
  dmns_a= nbn1(dmns)
  dmns_b= nbn2(dmns)
  dmns_c= nbn3(dmns)
  dmns_d= nbn4(dmns)
  end subroutine
 end module dimensions 
