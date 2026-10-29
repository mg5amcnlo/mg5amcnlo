program check_fxfx_delta_lhe
  use extra_weights
  implicit none
  integer, parameter :: maxnup=500
  integer :: nup,idprup,idup(maxnup),istup(maxnup),mothup(2,maxnup),icolup(2,maxnup)
  double precision :: xwgtup,scalup,aqedup,aqcdup,pup(5,maxnup),vtimup(maxnup),spinup(maxnup)
  double precision :: scalup_a(maxnup,maxnup)
  integer :: ickkw,nattr,npNLO,npLO,mg_rwgt_count,n_orderstags,orderstags_glob(maxorders)
  logical :: mcatnlo_delta,do_rwgt_scale,do_rwgt_pdf,store_rwgt_info
  common/test_run/ickkw,mcatnlo_delta,do_rwgt_scale,do_rwgt_pdf,store_rwgt_info
  common/event_attributes/nattr,npNLO,npLO
  common/rwgt_count/mg_rwgt_count
  common/c_orderstags_glob/n_orderstags,orderstags_glob
  character(1000) :: buff,ptclusstring,saved_tags
  common/c_ptclusstring/ptclusstring
  character(40) :: mode,arg,attr
  integer :: i,j,reader,comments,reweight
  call get_command_argument(1,mode)
  call get_command_argument(2,arg)
  read(arg,*) comments
  call get_command_argument(3,arg)
  read(arg,*) reweight
  ickkw=0
  if(mode=='fxfx'.or.mode=='combined') ickkw=3
  mcatnlo_delta=mode/='fxfx'
  do_rwgt_scale=reweight==1
  do_rwgt_pdf=.false.
  store_rwgt_info=.false.
  mg_rwgt_count=0
  n_orderstags=0
  dyn_scale=1
  lscalevar=.true.
  scalevarR=1d0
  scalevarF=1d0
  wgtxsecmu=2.5d0
  nup=12
  idprup=1
  xwgtup=1d0
  scalup=30d0
  aqedup=.01d0
  aqcdup=.1d0
  idup=1
  istup=1
  istup(1:2)=-1
  mothup=0
  icolup=0
  pup=0d0
  pup(4,:)=10d0
  vtimup=0d0
  spinup=9d0
  ptclusstring='<scales'
  do i=1,nup
    icolup(:,i)=[500+i,499+i]
    do j=1,nup
      scalup_a(i,j)=10d0+i+j/100d0
    enddo
    if(i>2) then
      write(attr,'(a,i0,a,f8.5,a)') ' pt_clust_',i,'="',merge(29.999d0,30d0,i==12),'"'
      ptclusstring=trim(ptclusstring)//trim(attr)
    endif
  enddo
  ptclusstring=trim(ptclusstring)//'></scales>'
  saved_tags=ptclusstring
  nattr=2
  npNLO=1
  npLO=-1
  buff=''
  if(comments==1) then
    jwgtinfo=0
    if(reweight==1) jwgtinfo=9
    write(buff,*) '#aMCatNLO 1 0 0 0 0 0 0 ',jwgtinfo,' 0 0 0 0 0 0 0'
    buff=adjustl(buff)
  endif
  open(20,file='event.lhe',status='replace')
  call write_lhef_event(20,nup,idprup,xwgtup,scalup,aqedup,aqcdup, &
      idup,istup,mothup,icolup,pup,vtimup,spinup,buff,scalup_a)
  close(20)
  do reader=1,2
    open(20,file='event.lhe',status='old')
    scalup_a=-99d0
    ptclusstring='stale'
    wgtxsecmu=0d0
    if(reader==1) then
      call read_lhef_event(20,nup,idprup,xwgtup,scalup,aqedup,aqcdup, &
          idup,istup,mothup,icolup,pup,vtimup,spinup,buff,scalup_a)
    else
      call read_lhef_event_catch(20,nup,idprup,xwgtup,scalup,aqedup,aqcdup, &
          idup,istup,mothup,icolup,pup,vtimup,spinup,buff,scalup_a)
    endif
    close(20)
    if(nup/=12.or.nattr/=2.or.npNLO/=1.or.scalup/=30d0) stop 1
    if(reweight==1.and.abs(wgtxsecmu(0,1,1,1)-2.5d0)>1d-12) stop 2
    if(ickkw==3.and.ptclusstring/=saved_tags) stop 3
    if(mcatnlo_delta.and.ickkw/=3) then
      do i=1,nup
        do j=1,nup
          if(abs(i-j)==1) then
            if(abs(scalup_a(i,j)-(10d0+i+j/100d0))>1d-10) stop 4
          else
            if(scalup_a(i,j)/=-1d0) stop 5
          endif
        enddo
      enddo
    endif
    open(21,file='roundtrip.lhe',status='replace')
    call write_lhef_event(21,nup,idprup,xwgtup,scalup,aqedup,aqcdup, &
        idup,istup,mothup,icolup,pup,vtimup,spinup,buff,scalup_a)
    close(21)
  enddo
  print *, 'PASS LHE scale roundtrip'
end program
