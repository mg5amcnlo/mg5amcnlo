module analytic_rng
  implicit none
  integer(kind=8) :: rng_state=1234567_8
end module analytic_rng

double precision function ran2()
  use analytic_rng
  implicit none
  rng_state=mod(48271_8*rng_state,2147483647_8)
  ran2=dble(rng_state)/2147483647d0
end function ran2

program analytic_history
  use simple_integrator_mod, only: staged_integrator
  use analytic_rng
  implicit none
  type(staged_integrator) :: sampler
  real(kind=8),external :: ran2
  real(kind=8) :: x(2),xf(2),u(2),jac,jfold,absfold(2),signedfold(2),ff(2),base
  real(kind=8) :: rates(2),errors(2),moments(5),fold_reference(2,17),actual_sign,rx,tail_width
  integer :: quota,reserve,unit,observations,j,k,nstored
  logical :: stored,batch_done,completed,unused
  character(len=128) :: argument
  call get_command_argument(1,argument)
  read(argument,*) rng_state
  call get_command_argument(2,argument)
  read(argument,*) quota
  call get_command_argument(3,argument)
  read(argument,*) tail_width
  reserve=(11*quota+9)/10
  call sampler%init(2,2)
  ! Exact integrals seed a uniform surveyed proposal. This isolates changes
  ! to generation; it is not a replacement survey-accuracy experiment.
  base=0.1d0+(1d0-exp(-30d0))/30d0+0.01d0
  do j=1,17
     do k=1,2
        call sampler%map_fold([0.37d0,dble(j)/18d0],[1,k],[1,2],xf,jfold)
        fold_reference(k,j)=xf(2)
     enddo
  enddo
  call sampler%start_native_production(reserve,quota,[1,2],0.58d0*base,3.5d0,20000000_8)
  open(newunit=observations,file='observables.dat',status='replace')
  nstored=0
  do
     call sampler%sample(x,jac,u)
     do k=1,2
        call sampler%map_fold(u,[1,k],[1,2],xf,jfold)
        rx=0.1d0+exp(-30d0*xf(1))
        if (xf(1).ge.0.72d0.and.xf(1).lt.0.72d0+tail_width) rx=rx+0.01d0/tail_width
        absfold(k)=0d0
        signedfold(k)=0d0
        if ((xf(2).ge.0.1d0.and.xf(2).lt.0.3d0).or. &
             (xf(2).ge.0.6d0.and.xf(2).lt.0.8d0)) then
           absfold(k)=rx*(1d0+xf(2))*jfold
           actual_sign=-1d0
           if ((xf(2).ge.0.225d0.and.xf(2).lt.0.3d0).or. &
                (xf(2).ge.0.6d0.and.xf(2).lt.0.725d0)) actual_sign=1d0
           signedfold(k)=actual_sign*absfold(k)
        endif
     enddo
     ff=[sum(absfold),sum(signedfold)]
     call sampler%native_consider(ff,x,stored,batch_done)
     if (stored) then
        nstored=nstored+1
        actual_sign=sign(1d0,signedfold(2))
        if (ran2().lt.absfold(1)/ff(1)) actual_sign=sign(1d0,signedfold(1))
        write(observations,'(3(es24.16,1x))') &
             merge(1d0,0d0,x(1).lt.0.1d0), &
             merge(1d0,0d0,x(1).ge.0.72d0.and.x(1).lt.0.72d0+tail_width),actual_sign
        call sampler%record_candidate_factor(1d0,unused)
     endif
     completed=.false.
     if (sampler%native_completion_due) call sampler%check_native_completion(completed)
     if (completed) exit
     if (batch_done) then
        call sampler%finish_native_iteration(completed)
        do j=1,17
           do k=1,2
              call sampler%map_fold([0.37d0,dble(j)/18d0],[1,k],[1,2],xf,jfold)
              if (xf(2).ne.fold_reference(k,j)) error stop 'folded map changed'
           enddo
        enddo
     endif
     if (completed) exit
  enddo
  close(observations)
  call sampler%native_rates(rates,errors,moments)
  if (.not.sampler%quota_complete.or.sampler%overweight.ge.0.01d0) error stop 'tail bound failed'
  open(newunit=unit,file='ampli_pool.dat',status='replace')
  call sampler%write_native_pool(unit)
  close(unit)
  write(*,*) 'ANALYTIC_RESULT',sampler%ntrials,nstored,sampler%native_iteration,sampler%adaptation_updates,rates,errors
end program analytic_history
