! Minimal native metadata for exercising the production colour-pair selector.
module mc_native_context
  implicit none
  type metadata
    integer :: born_ids(4,2)
    integer,allocatable :: born_colours(:,:,:)
  end type
  type(metadata),target :: storage
  type(metadata),pointer :: native_metadata=>storage
contains
  subroutine ensure_native_context()
  end subroutine
end module mc_native_context

program check_pythia_local_selection
  use mc_native_context
  implicit none
  integer ifks,jfks,pair(2),i,j,realj,top,nflow
  common/fks_indices/ifks,jfks
  double precision pmass(5),ebeam(2),xbk(2)
  common/to_mass/pmass
  integer lpp(2)
  common/test_run/ebeam,xbk,lpp
  logical split_type(2)
  common/c_split_type/split_type
  character(32) mode

  call get_command_argument(1,mode)
  nflow=1
  if(mode.eq.'flows')nflow=2
  allocate(storage%born_colours(2,4,nflow))
  storage%born_ids(:,1)=[5,2,6,1]
  storage%born_ids(:,2)=[5,4,6,3]
  if(mode.eq.'gluon')storage%born_ids(1,:)=21
  storage%born_colours=0
  storage%born_colours(1,:,1)=[501,502,501,502]
  if(mode.eq.'antiquark')then
    storage%born_ids(2,:)=-storage%born_ids(2,:)
    storage%born_ids(4,:)=-storage%born_ids(4,:)
    storage%born_colours(1,2,1)=0
    storage%born_colours(1,4,1)=0
    storage%born_colours(2,2,1)=502
    storage%born_colours(2,4,1)=502
  endif
  if(mode.eq.'no_partner')storage%born_colours(1,3:4,1)=[503,504]
  lpp=1
  if(mode.eq.'beam')lpp(1)=3
  split_type=[.true.,.false.]
  if(mode.eq.'qed')split_type(2)=.true.
  do ifks=3,5
    pmass=0d0
    top=3
    if(top.ge.ifks)top=top+1
    pmass(top)=173d0
    if(mode.eq.'top_mass')pmass(top)=0d0
    do j=1,4
      realj=j
      if(realj.ge.ifks)realj=realj+1
      jfks=realj
      call mc_select_pythia_pair(pair)
      i=j
      if(i.gt.2)i=i-2
      realj=i+2
      if(realj.ge.ifks)realj=realj+1
      if(any(pair.ne.[i,realj]))error stop 'incorrect native colour pair'
    enddo
  enddo
  write(*,*)'PASS selector'
end program check_pythia_local_selection
