      ! gfortran -O3 -o sum_event_wgts sum_event_wgts.f90


program sum_weights
  implicit none
  integer ifile,nPSpoints(2)
  logical :: done
  character*140 filename
  character*50 weights_info(10)
  double precision dummy
  real(kind=8),dimension(0:3,100) :: p
  real(kind=8) :: evt_wgt,abs_wgt,wgt,upper_bound,frac
  real(kind=8),dimension(2) :: sum_evt_wgt,sum_abs_wgt,sum_wgt
  real(kind=8) :: absx,absv,xsec
  integer :: next
  write (*,*) 'Give LHE file name'
  read (*,'(a)') filename
  ifile=11
  open(unit=ifile,file=filename,status='OLD')
  
  sum_abs_wgt=0d0
  sum_evt_wgt=0d0
  sum_wgt=0d0
  nPSpoints(1:2)=0
  xsec=0d0
  do
     call read_event(ifile,done)
     if (done) exit
!!$     if (frac.eq.1d0) then
!!$        if (evt_wgt.lt.0d0) then
!!$           xsec=xsec-(absv+absx)*abs_wgt/upper_bound
!!$        else
!!$           xsec=xsec+(absv+absx)*abs_wgt/upper_bound
!!$        endif
!!$     else
!!$        if (evt_wgt.lt.0d0) then
!!$           xsec=xsec-(absv+absx)*abs_wgt/upper_bound!/frac
!!$        else
!!$           xsec=xsec+(absv+absx)*abs_wgt/upper_bound!/frac
!!$        endif
!!$     endif

     xsec=xsec+evt_wgt

     
     if (frac.eq.1d0) then
        nPSpoints(1)=nPSpoints(1)+1
        sum_abs_wgt(1)=sum_abs_wgt(1)+abs_wgt
        sum_evt_wgt(1)=sum_evt_wgt(1)+evt_wgt
        sum_wgt(1)=sum_wgt(1)+wgt
     else
        nPSpoints(2)=nPSpoints(2)+1
        sum_abs_wgt(2)=sum_abs_wgt(2)+abs_wgt/frac
        sum_evt_wgt(2)=sum_evt_wgt(2)+evt_wgt/frac
        sum_wgt(2)=sum_wgt(2)+wgt/frac
     endif
  enddo

  write (*,*) 'sum of weights is (1)',sum_abs_wgt(1)/nPSpoints(1),sum_evt_wgt(1)/nPSpoints(1),sum_wgt(1)/nPSpoints(1)
  write (*,*) 'sum of weights is (2)',sum_abs_wgt(2)/nPSpoints(2),sum_evt_wgt(2)/nPSpoints(2),sum_wgt(2)/nPSpoints(2)
  write (*,*) xsec/(nPSpoints(1)+nPSpoints(2)),nPSpoints(1),nPSpoints(2)
  write (*,*) xsec/dble(9192)
  write (*,*) xsec


  close (ifile)

contains
  subroutine read_event(iunit,done)
    implicit none
    integer :: i,iunit,idum
    logical :: done
    character(len=100) :: dummy
    character(len=15) :: dummy2
    real(kind=8) :: dum
    done=.false.
    do
       read (iunit,*,err=99,end=99) dummy
       if (index(dummy,'</LesHouchesEvents>').ne.0) then
          done=.true.
          return
       elseif(index(dummy,'<event>').ne.0) then
          read (iunit,*,err=99,end=99) next,idum,evt_wgt,dum,dum,dum
          do i=1,next
             read (iunit,*,err=99,end=99) dum,p(1:3,i),p(0,i)
          enddo
          read (iunit,*,err=99,end=99) dummy2,abs_wgt,wgt,upper_bound,frac,absx,absv
          return
       endif
    enddo
    return
99  done=.true.
  end subroutine read_event
end program sum_weights
