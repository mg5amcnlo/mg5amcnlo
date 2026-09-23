module pdf_test_state
  implicit none
  integer :: calls=0,set_calls=0,active_history=0,pdf_epoch=1
contains
  subroutine reference_pdf(nfks,history,x,q2,pd,nproc)
    integer,intent(in) :: nfks,history
    double precision,intent(in) :: x(2),q2(2)
    double precision,intent(out) :: pd(0:3)
    integer,intent(out) :: nproc
    integer k
    pd=0d0
    nproc=1+mod(nfks+history,3)
    do k=1,nproc
       pd(k)=pdf_epoch*(10*nfks+3*history+k)*(x(1)+2d0*x(2))*(q2(1)+q2(2))
    enddo
    pd(0)=-sum(pd(1:nproc))
  end subroutine
end module

program check_pdf_luminosity_cache
  use weight_lines
  use FKSParams
  use pdf_test_state
  implicit none
  include 'run.inc'
  double precision QES2
  common/COUPL_ES/QES2
  integer :: i,mode,expected_calls
  logical :: fixed_order,nlo_ps
  common/c_fnlo_nlops/fixed_order,nlo_ps
  fixed_order=.true.
  nlo_ps=.false.
  flavour_bias=[0,1]
  QES2=49d0

  ! A,B,A exercises full PD/IPROC restoration. Further pairs isolate
  ! each key coordinate, native history, FKS identity and PDF scale.
  do mode=1,4
     call prepare(14)
     lpp=1
     if(mode.eq.2)lpp=2
     if(mode.eq.3)lpp=3
     if(mode.eq.4)pdf_epoch=2 ! no previous-call values may survive
     nFKS(2)=2
     nFKS(4)=2
     native_ids(3,5:6)=2
     bjx(1,7:8)=0.125d0
     bjx(2,9:10)=0.375d0
     scales2(3,11:12)=25d0
     scales2(2,13:14)=36d0 ! muR alone must not invalidate PDFs
     calls=0
     set_calls=0
     call include_PDF_and_alphas()
     expected_calls=6
     if(mode.eq.2.or.mode.eq.3)expected_calls=14
     if(calls.ne.expected_calls.or.set_calls.ne.14)then
        write(*,*)'PDF calls',mode,calls,expected_calls,set_calls
        error stop 'incorrect central PDF reuse'
     endif
     call check_weights()
  enddo

  ! Fill and wrap the bounded ring. An evicted key must be recalculated,
  ! while its immediate repeat and a remaining recent key are reused.
  call prepare(44)
  lpp=1
  do i=1,20
     bjx(1,2*i-1:2*i)=dble(i)/64d0
  enddo
  bjx(1,41:42)=1d0/64d0
  bjx(1,43:44)=20d0/64d0
  calls=0
  call include_PDF_and_alphas()
  if(calls.ne.21)error stop 'PDF ring wrap lost repeated keys'
  call check_weights()

  ! Actual fixed-order flavour splitting appends contributions while the
  ! central loop is active, including an allocation growth from one row.
  call prepare(1)
  separate_flavour_configs=.true.
  calls=0
  set_calls=0
  call include_PDF_and_alphas()
  if(icontr.ne.2.or.calls.ne.1.or.set_calls.ne.1) &
       error stop 'appended flavour records were not reused safely'
  call check_weights()
  call deallocate_weight_lines()
  write(*,*)'PASS PDF luminosity cache'
contains
  subroutine prepare(n)
    integer,intent(in) :: n
    integer j
    call deallocate_weight_lines()
    call weight_lines_allocated(5,n,1,1)
    icontr=n
    separate_flavour_configs=.false.
    nFKS=1
    native_ids=0
    event_nFKS=1
    ipr=0
    bjx(1,:)=0.25d0
    bjx(2,:)=0.5d0
    scales2=16d0
    wgt=0d0
    do j=1,n
       wgt(1,j)=j
    enddo
    wgt_ME_tree=1d0
    g_strong=2d0
    QCDpower=2
    cpower=0d0
    itype=1
    H_event=.true.
    y_bst=0d0
    orderstag=1
    momenta=1d0
    momenta_m=1d0
  end subroutine
  subroutine check_weights()
    double precision :: densities(0:3),scales(2),expected
    double precision,parameter :: conv=389379660d0
    integer j,k,np
    do j=1,icontr
       scales=scales2(3,j)
       if(all(abs(lpp).eq.2))scales=QES2
       call reference_pdf(nFKS(j),native_ids(3,j),bjx(:,j),scales,densities,np)
       if(separate_flavour_configs)then
          expected=densities(ipr(j))*conv*wgt(1,j)*4d0
          if(niproc(j).ne.1)error stop 'flavour splitting changed multiplicity'
          if(parton_iproc(1,j).ne.expected)error stop 'split flavour weight changed'
       else
          expected=sum(densities(1:np))*conv*wgt(1,j)*4d0
          if(niproc(j).ne.np)error stop 'cached IPROC changed'
          do k=1,np
             if(parton_iproc(k,j).ne.densities(k)*conv*wgt(1,j)*4d0) &
                  error stop 'cached PD changed flavour weight'
          enddo
       endif
       if(wgts(1,j).ne.expected)error stop 'central luminosity weight changed'
    enddo
  end subroutine
end program

double precision function dlum()
  use pdf_test_state
  implicit none
  include 'run.inc'
  integer nfks,iproc
  double precision pd(0:3)
  common/c_nFKSprocess/nfks
  common/SUBPROC/pd,iproc
  calls=calls+1
  call reference_pdf(nfks,active_history,xbk,q2fact,pd,iproc)
  dlum=sum(pd(1:iproc))*389379660d0
end function

subroutine mc_set_history(history)
  use pdf_test_state
  implicit none
  integer history,iproc
  double precision pd(0:3)
  common/SUBPROC/pd,iproc
  active_history=history
  ! Force hits to restore the complete active result, not only its sum.
  pd=-1000d0
  iproc=3
end subroutine

subroutine set_pdg_codes(iproc,pd,fks,ict)
  use weight_lines
  use pdf_test_state
  implicit none
  include 'run.inc'
  integer iproc,fks,ict,k,np
  double precision pd(0:3),expected(0:3)
  set_calls=set_calls+1
  call reference_pdf(fks,active_history,xbk,q2fact,expected,np)
  if(iproc.ne.np.or.any(pd(0:np).ne.expected(0:np)))error stop 'luminosity COMMON not restored'
  niproc(ict)=iproc
  do k=1,iproc
     parton_iproc(k,ict)=pd(k)*389379660d0
     parton_pdg(:,k,ict)=k
     parton_pdg_uborn(:,k,ict)=k
  enddo
end subroutine

double precision function rwgt_muR_dep_fac(mu1,mu2,power)
  implicit none
  double precision mu1,mu2,power
  rwgt_muR_dep_fac=1d0
end function

subroutine recompute_xlum_for_wgt_mint(i,xlum)
  integer i
  double precision xlum
  error stop 'unexpected flavour bias in PDF test'
end subroutine

subroutine amp_split_pos_to_orders(i,orders)
  integer i,orders(1)
  orders=2
end subroutine
