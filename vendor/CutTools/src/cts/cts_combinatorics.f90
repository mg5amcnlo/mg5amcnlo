!                                       !
! build up the necessary combinatorics  !
!                                       !
 module maxnumden
  implicit none
!                                       !
! the maximum number of denominators    !
! (it can be changed by the user)       ! 
!                                       !
  integer, public, parameter :: maxden= 6
 end module maxnumden
!
 module mbnvalues
  implicit none
  integer, public, dimension(:,:), allocatable :: mbn1
  integer, public, dimension(:,:), allocatable :: mbn2
  integer, public, dimension(:,:), allocatable :: mbn3
  integer, public, dimension(:,:), allocatable :: mbn4
  integer, public, dimension(:,:,:), allocatable :: bn1
  integer, public, dimension(:,:,:), allocatable :: bn2
  integer, public, dimension(:,:,:), allocatable :: bn3
  integer, public, dimension(:,:,:), allocatable :: bn4
 end module mbnvalues
!
 module combinations
  use maxnumden
  implicit none
  integer, dimension(1:maxden), public :: string
  integer, public :: max_solutions= 18
 end module combinations
!
 module combinatorics
  use combinations
  use mbnvalues
  implicit none
  integer, public, dimension(1:maxden) :: nbn1= -1
  integer, public, dimension(2:maxden) :: nbn2= -1
  integer, public, dimension(3:maxden) :: nbn3= -1
  integer, public, dimension(4:maxden) :: nbn4= -1
  contains
!
  subroutine load_combinatorics
  implicit none
  integer :: k,kden,ierr
  integer, parameter :: n1= 1,n2= 2,n3= 3,n4= 4
  integer :: k1,k2,k3,k4,kref,kcount,kc,kvalue,mxden
  mxden= maxden
  do k= 1,mxden
   string(k) = 2**(k-1)
  enddo
  if (mxden.ge.1) then
    do k= 1,mxden
     nbn1(k)= comb(k,n1)
    enddo
    allocate  (mbn1(1:mxden,nbn1(mxden)), stat=ierr)
    allocate  (bn1(1:mxden,mxden,nbn1(mxden)), stat=ierr)
    mbn1= -1
    bn1= -1
    do kden= 1,mxden
     kcount= 0
     do k1= 1,kden
       kcount= kcount+1
       kvalue= string(k1)
       mbn1(kden,kcount)= kvalue
       bn1(kden,1,kcount)= string(k1)
       kref  = 0; kc    = 1
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1)) then
          kc= kc+1
          kref= k
          bn1(kden,kc,kcount)= string(k)
        endif
       enddo
     enddo
    enddo
  endif
  if (mxden.ge.2) then
    do k= 2,mxden
     nbn2(k)= comb(k,n2)
    enddo
    allocate  (mbn2(2:mxden,nbn2(mxden)), stat=ierr)
    allocate  (bn2(2:mxden,mxden,nbn2(mxden)), stat=ierr)
    mbn2= -1
    bn2= -1
    do kden= 2,mxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden
       kcount= kcount+1
       kvalue= string(k1)+string(k2)
       mbn2(kden,kcount)= kvalue
       bn2(kden,1,kcount)= string(k1)
       bn2(kden,2,kcount)= string(k2)
       kref  = 0; kc    = 2
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2)) then
          kc= kc+1
          kref= k
          bn2(kden,kc,kcount)= string(k)
        endif
       enddo
     enddo; enddo
    enddo
  endif
  if (mxden.ge.3) then
    do k= 3,mxden
     nbn3(k)= comb(k,n3)
    enddo
    allocate  (mbn3(3:mxden,nbn3(mxden)), stat=ierr)
    allocate  (bn3(3:mxden,mxden,nbn3(mxden)), stat=ierr)
    mbn3= -1
    bn3= -1
    do kden= 3,mxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden; do k3= k2+1,kden
       kcount= kcount+1
       kvalue= string(k1)+string(k2)+string(k3)
       mbn3(kden,kcount)= kvalue
       bn3(kden,1,kcount)= string(k1)
       bn3(kden,2,kcount)= string(k2)
       bn3(kden,3,kcount)= string(k3)
       kref  = 0; kc    = 3
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2).and.(k.ne.k3)) then
          kc= kc+1
          kref= k
          bn3(kden,kc,kcount)= string(k)
        endif
       enddo
     enddo; enddo; enddo
    enddo
  endif
  if (mxden.ge.4) then
    do k= 4,mxden
     nbn4(k)= comb(k,n4)
    enddo
    allocate  (mbn4(4:mxden,nbn4(mxden)), stat=ierr)
    allocate  (bn4(4:mxden,mxden,nbn4(mxden)), stat=ierr)
    mbn4= -1
    bn4= -1
    do kden= 4,mxden
     kcount= 0
     do k1= 1,kden; do k2= k1+1,kden; do k3= k2+1,kden; do k4= k3+1,kden
       kcount= kcount+1
       kvalue= string(k1)+string(k2)+string(k3)+string(k4)
       mbn4(kden,kcount)= kvalue
       bn4(kden,1,kcount)= string(k1)
       bn4(kden,2,kcount)= string(k2)
       bn4(kden,3,kcount)= string(k3)
       bn4(kden,4,kcount)= string(k4)
       kref  = 0; kc    = 4
       do k= 1,kden
        if ((k.gt.kref).and.(k.ne.k1).and.(k.ne.k2) &
           .and.(k.ne.k3).and.(k.ne.k4)) then
          kc= kc+1
          kref= k
          bn4(kden,kc,kcount)= string(k)
        endif
       enddo
     enddo; enddo; enddo; enddo
    enddo
  endif
  end subroutine load_combinatorics
!
  integer function comb(no,ko)
  implicit none
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
 end module combinatorics
