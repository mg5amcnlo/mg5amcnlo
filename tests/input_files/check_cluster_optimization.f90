program check_cluster_optimization
  use fks_phase_space_data
  use mc_native_context
  implicit none
  include 'maxconfigs.inc'
  integer :: lpp(2),builds(2),sudakov_calls,this_config,nFKSprocess
  double precision :: bwcutoff,d,bmass,gmass,hmass
  double precision :: pmass(-6:0,lmaxconfigs,0:2),pwidth(-6:0,lmaxconfigs,0:2)
  integer :: iforest(2,-4:-1,lmaxconfigs,0:2),sprop(-4:-1,lmaxconfigs,0:2)
  integer :: tprid(-4:-1,lmaxconfigs,0:2),mapconfig(0:lmaxconfigs,0:2)
  common /test_run/ bwcutoff,lpp
  common /to_dj/ d
  common /test_model/ bmass,gmass,hmass
  common /test_counts/ builds,sudakov_calls
  common /to_mconfigs/ this_config
  common /c_nFKSprocess/ nFKSprocess
  common /c_configurations/ pmass,pwidth,iforest,sprop,tprid,mapconfig
  character(len=32) :: mode
  integer :: i,context,process,old_calls
  double precision :: ren(0:6),fac,other_ren(0:6),other_fac
  integer :: nren,other_nren,matching(6),other_matching(6)
  call get_command_argument(1,mode)
  lpp=1
  bwcutoff=15d0
  d=1d0
  bmass=4.7d0
  gmass=0d0
  hmass=125d0
  builds=0
  sudakov_calls=0
  this_config=1
  nFKSprocess=1
  pmass=0d0
  pwidth=0d0
  iforest=0
  sprop=0
  tprid=0
  mapconfig=0
  mapconfig(0,:)=2
  call real_tables(1)
  call real_tables(2)
  select case (mode)
  case ('reuse')
     do i=1,16
        context=1+mod(i,2)
        call born_tables(context)
        this_config=1+mod(i/2,2)
        call momenta(context,1d0+i/20d0)
        ! Born slots are per context, real slots per FKS process. Exercise
        ! each real process in both contexts as well as both Born call forms.
        call compare(0,mod(i,3).eq.0)
        call compare(-context,.true.)
        do process=1,2
           call momenta(process,1d0+i/20d0)
           call compare(process,mod(i,3).eq.0)
        enddo
     enddo
     if (builds(1).ne.8.or.builds(2).ne.128) error stop 'unexpected topology rebuilds'
  case ('mass')
     do i=1,6
        if (i.eq.2) bmass=0d0
        if (i.eq.3) gmass=10d0
        if (i.eq.4) bmass=8d0
        if (i.eq.5) hmass=150d0
        if (i.eq.6) gmass=0d0
        do context=1,2
           call born_tables(context)
           call momenta(context,1d0)
           call compare(0,.false.)
           call compare(context,.false.)
        enddo
     enddo
     if (builds(1).ne.8) error stop 'model changes rebuilt immutable topologies'
  case ('resonance')
     do context=1,2
        call born_tables(context)
        call momenta(context,1d0)
        do i=1,4
           pwidth(-1,1,0)=1d0
           pmass(-1,1,0)=sqrt((p_born(0,3)+p_born(0,4))**2 &
                -sum((p_born(1:3,3)+p_born(1:3,4))**2))
           if (i.eq.2) pmass(-1,1,0)=pmass(-1,1,0)+100d0
           if (i.eq.3) bwcutoff=0.01d0
           if (i.eq.4) then
              lpp=0
              d=0.4d0
              pwidth=0d0
           endif
           this_config=2
           call compare(0,.false.)
        enddo
        lpp=1
        bwcutoff=15d0
     enddo
     if (builds(1).ne.4) error stop 'kinematic inputs rebuilt topology tables'
  case ('scales')
     do context=1,2
        call born_tables(context)
        call momenta(context,1d0)
        old_calls=sudakov_calls
        call evaluate(0,.true.,ren,fac,nren,matching)
        if (sudakov_calls.ne.old_calls) error stop 'shower scales evaluated Sudakovs'
        call evaluate(0,.false.,other_ren,other_fac,other_nren,other_matching)
        call same(ren,fac,nren,matching,other_ren,other_fac,other_nren,other_matching)
        call compare(context,.true.)
        call compare(context,.false.)
     enddo
     call check_scale_bookkeeping()
  case ('candidates')
     call check_candidates(4)
  case ('words')
     call check_candidates(70)
     call check_many_configurations()
  case ('vertices')
     call check_vertices()
  case default
     error stop 'unknown clustering test mode'
  end select
  print *, 'PASS cluster optimization '//trim(mode)
contains
  subroutine born_tables(ctx)
    integer, intent(in) :: ctx
    active_context=ctx
    native_epoch=native_epoch+1
    nFKSprocess=ctx
    iforest(:,-1,1,0)=[3,4]
    iforest(:,-2,1,0)=[-1,5]
    iforest(:,-3,1,0)=[1,2]
    iforest(:,-1,2,0)=[4,5]
    iforest(:,-2,2,0)=[3,-1]
    iforest(:,-3,2,0)=[1,2]
    sprop(:,:,0)=21
    if (ctx.eq.2) sprop(-1,2,0)=-5
  end subroutine

  subroutine real_tables(proc)
    integer, intent(in) :: proc
    iforest(:,-1,1,proc)=[3,4]
    iforest(:,-2,1,proc)=[-1,5]
    iforest(:,-3,1,proc)=[-2,6]
    iforest(:,-4,1,proc)=[1,2]
    iforest(:,-1,2,proc)=[4,5]
    iforest(:,-2,2,proc)=[3,-1]
    iforest(:,-3,2,proc)=[-2,6]
    iforest(:,-4,2,proc)=[1,2]
    sprop(:,:,proc)=21
    if (proc.eq.2) sprop(-1,2,proc)=-5
  end subroutine

  subroutine momenta(proc,factor)
    integer, intent(in) :: proc
    double precision, intent(in) :: factor
    double precision :: m
    integer :: k
    m=gmass
    if (proc.eq.2) m=bmass
    p_ev=0d0
    p_ev(1:3,3)=[30d0,0d0,40d0]*factor
    p_ev(1:3,4)=[-20d0,40d0,-50d0]*factor
    do k=3,4
       p_ev(0,k)=sqrt(sum(p_ev(1:3,k)**2)+m**2)
    enddo
    p_ev(1:3,6)=[0d0,-10d0,20d0]*factor
    p_ev(0,6)=sqrt(sum(p_ev(1:3,6)**2)+gmass**2)
    p_born=p_ev(:,1:5)
    p_born(1:3,5)=-sum(p_born(1:3,3:4),dim=2)
    p_born(0,5)=sqrt(sum(p_born(1:3,5)**2)+hmass**2)
    p_ev(1:3,5)=-sum(p_ev(1:3,3:4),dim=2)-p_ev(1:3,6)
    p_ev(0,5)=sqrt(sum(p_ev(1:3,5)**2)+hmass**2)
    p_born(0,1:2)=sum(p_born(0,3:5))/2d0
    p_ev(0,1:2)=sum(p_ev(0,3:6))/2d0
    p_born(3,1)=p_born(0,1)
    p_born(3,2)=-p_born(0,2)
    p_ev(3,1)=p_ev(0,1)
    p_ev(3,2)=-p_ev(0,2)
  end subroutine

  subroutine compare(proc,scale_only)
    integer, intent(in) :: proc
    logical, intent(in) :: scale_only
    integer :: n1,n2,match1(6),match2(6)
    double precision :: sud1,sud2,exp1,exp2,r1(0:6),r2(0:6),f1,f2
    r1=-777d0
    r2=-777d0
    call cluster_and_reweight(proc,sud1,exp1,n1,r1,f1,match1,scale_only)
    call cluster_and_reweight_fresh(proc,sud2,exp2,n2,r2,f2,match2,scale_only)
    call same(r1,f1,n1,match1,r2,f2,n2,match2)
    if (sud1.ne.sud2.or.exp1.ne.exp2) error stop 'cached Sudakov changed'
  end subroutine

  subroutine evaluate(proc,scale_only,ren,fac,nren,matching)
    integer, intent(in) :: proc
    logical, intent(in) :: scale_only
    double precision, intent(out) :: ren(0:6),fac
    integer, intent(out) :: nren,matching(6)
    double precision :: sud,expanded
    call cluster_and_reweight(proc,sud,expanded,nren,ren,fac,matching,scale_only)
    if (scale_only.and.(sud.ne.1d0.or.expanded.ne.0d0)) error stop 'shower-scale weight'
  end subroutine

  subroutine same(r1,f1,n1,m1,r2,f2,n2,m2)
    integer, intent(in) :: n1,n2,m1(6),m2(6)
    double precision, intent(in) :: r1(0:6),r2(0:6),f1,f2
    if (n1.ne.n2.or.any(m1.ne.m2)) error stop 'clustering labels changed'
    if (any(r1(0:n1).ne.r2(0:n2)).or.f1.ne.f2) error stop 'clustering scales changed'
  end subroutine

  subroutine check_scale_bookkeeping()
    integer :: pdgs(4),cpdg(0:2,0:2),ij(1),iord(0:1),calls,qcd_count(0:1)
    double precision :: scales(0:1),sud,expanded
    pdgs=21
    cpdg=21
    ij=12
    iord=[0,1]
    scales=[100d0,20d0]
    qcd_count=3
    matching=-99
    other_matching=-99
    calls=sudakov_calls
    call reweighting(4,p_born,1,.false.,ij,pdgs,cpdg,scales,iord,qcd_count,sud,expanded, &
         nren,ren,fac,matching,.true.)
    if (sudakov_calls.ne.calls.or.sud.ne.1d0.or.expanded.ne.0d0) error stop 'scale-only Sudakov work'
    call reweighting(4,p_born,1,.false.,ij,pdgs,cpdg,scales,iord,qcd_count,sud,expanded, &
         other_nren,other_ren,other_fac,other_matching,.false.)
    call same(ren,fac,nren,matching,other_ren,other_fac,other_nren,other_matching)
    if (sudakov_calls.le.calls.or.sud.ge.1d0.or.expanded.ge.0d0) error stop 'FxFx omitted Sudakovs'
    if (nren.ne.1.or.ren(0).ne.100d0.or.ren(1).ne.20d0.or.fac.ne.20d0) error stop 'QCD scales'
  end subroutine

  subroutine check_vertices()
    integer :: mask,ids(0:2)
    integer, external :: qcd_vertex_mask
    do mask=0,7
       ids=25
       if (btest(mask,0)) ids(0)=5
       if (btest(mask,1)) ids(1)=-5
       if (btest(mask,2)) ids(2)=21
       if (qcd_vertex_mask(ids).ne.mask) error stop 'wrong coloured legs in vertex'
    enddo
  end subroutine

  subroutine check_many_configurations()
    integer :: lst(6,70),cpdg(0:2,0:6,70),qcd_mask(0:6,70),tree(2,-4:-1)
    integer :: ids(6),ctype(63),sp(-3:-1),conf,ij(3),iord(0:3),qcd_count(0:3),k
    integer(kind=8) :: masks(2,63)
    double precision :: p(0:3,6),mass(-3:-1),width(-3:-1),scales(0:3)
    lst(:,1)=[12,28,60,51,35,3]
    lst(:,2)=[48,56,60,15,7,3]
    lst(:,3)=[20,28,60,43,35,3]
    lst(:,4)=[5,13,29,58,50,34]
    do k=5,70
       lst(:,k)=lst(:,1+mod(k-1,4))
    enddo
    tree(:,-1)=[1,3]
    tree(:,-2)=[-1,4]
    tree(:,-3)=[-2,5]
    tree(:,-4)=[2,6]
    p(:,1)=[145d0,0d0,0d0,145d0]
    p(:,2)=[145d0,0d0,0d0,-145d0]
    p(:,3)=[45d0,27d0,0d0,36d0]
    p(:,4)=[45d0,-27d0,0d0,-36d0]
    p(:,5)=[100d0,100d0,0d0,0d0]
    p(:,6)=[100d0,-100d0,0d0,0d0]
    ids=21
    cpdg=21
    sp=21
    ctype=1
    qcd_mask=7
    mass=0d0
    width=0d0
    call make_cluster_config_masks(6,3,70,lst,masks)
    call cluster(6,p,70,3,lst,masks,cpdg,qcd_mask,tree,ids,mass,width,68,sp, &
         conf,scales,ij,iord,ctype,qcd_count)
    if (conf.ne.68.or.any(qcd_count.ne.3)) error stop 'lost configuration beyond first word'
  end subroutine

  subroutine check_candidates(nconf)
    integer, intent(in) :: nconf
    integer :: lst(6,nconf),mask,step,nleft,imap(6),bw(2,0:3),ctype(63),ptypes(6)
    integer :: i,j,conf,id,wi,wj,win,ei,ej,ewin,scenario,word
    integer(kind=8) :: config_masks((nconf+63)/64,63),valid_words((nconf+63)/64)
    double precision :: p(0:4,6),scale,expected_scale,test_scale,inter(0:4,0:2),cluster_scale
    logical :: valid(nconf),is_bw,expected_bw,candidate_bw,present
    external cluster_scale
    lst(:,1)=[12,28,60,51,35,3]
    lst(:,2)=[48,56,60,15,7,3]
    lst(:,3)=[20,28,60,43,35,3]
    lst(:,4)=[5,13,29,58,50,34]
    do conf=5,nconf
       lst(:,conf)=lst(:,1+mod(conf-1,4))
    enddo
    call make_cluster_config_masks(6,3,nconf,lst,config_masks)
    do scenario=1,2
       do mask=1,15
          p(:,1)=[145d0,0d0,0d0,145d0,0d0]
          p(:,2)=[145d0,0d0,0d0,-145d0,0d0]
          p(:,3)=[45d0,27d0,0d0,36d0,0d0]
          p(:,4)=[45d0,-27d0,0d0,-36d0,0d0]
          p(:,5)=[100d0,100d0,0d0,0d0,0d0]
          p(:,6)=[100d0,-100d0,0d0,0d0,0d0]
          do i=1,6
             imap(i)=ishft(1,i-1)
          enddo
          valid_words=0_8
          do conf=1,nconf
             valid(conf)=btest(mask,mod(conf-1,4))
             if (nconf.gt.64.and.scenario.eq.2) valid(conf)=valid(conf).and.conf.ge.64
             if (valid(conf)) then
                word=(conf-1)/64+1
                valid_words(word)=ibset(valid_words(word),mod(conf-1,64))
             endif
          enddo
          ctype=1
          ptypes=1
          bw=0
          if (scenario.eq.2) then
             bw(1,0)=1
             bw(:,1)=[12,21]
          endif
          nleft=6
          do step=1,3
             call cluster_one_step(nleft,p,imap,3,nconf,valid_words,config_masks,bw,wi,wj,win,scale,is_bw,ctype,ptypes)
             expected_scale=99d99
             ei=-1
             do i=3,nleft
                do j=1,i-1
                   id=imap(i)+imap(j)
                   present=.false.
                   do conf=1,nconf
                      if (valid(conf).and.any(lst(:,conf).eq.id)) present=.true.
                   enddo
                   if (.not.present) cycle
                   test_scale=cluster_scale(bw,3,j,id,p(:,i),p(:,j),[ctype(id),ptypes(i),ptypes(j)],candidate_bw)
                   if (test_scale.lt.expected_scale) then
                      ei=i
                      ej=j
                      ewin=id
                      expected_scale=test_scale
                      expected_bw=candidate_bw
                   endif
                enddo
             enddo
             if (ei.ne.wi.or.ej.ne.wj.or.ewin.ne.win) error stop 'candidate order or membership changed'
             if (scale.ne.expected_scale.or.(is_bw.neqv.expected_bw)) error stop 'candidate scale or resonance'
             call update_valid_confs(win,nconf,valid_words,config_masks)
             do conf=1,nconf
                valid(conf)=valid(conf).and.any(lst(:,conf).eq.win)
                if (btest(valid_words((conf-1)/64+1),mod(conf-1,64)).neqv.valid(conf)) &
                     error stop 'configuration filtering changed'
             enddo
             call update_momenta(nleft,p,wi,wj,inter,is_bw,ptypes,ctype(win))
             call update_imap(nleft,imap,wi,wj,win)
             nleft=nleft-1
          enddo
       enddo
    enddo
  end subroutine
end program

subroutine count_cluster_builds(tag)
  implicit none
  integer :: tag,builds(2),sudakov_calls
  common /test_counts/ builds,sudakov_calls
  builds(tag)=builds(tag)+1
end subroutine

subroutine set_pdg(dummy,proc)
  use mc_native_context, only: active_context
  use weight_lines
  implicit none
  integer :: dummy,proc
  if (dummy.ne.0.or.proc.lt.1.or.proc.gt.2) error stop 'unexpected process lookup'
  pdg(:,0)=[21,21,21,21,25,21]
  if (proc.eq.2) pdg(:,0)=[2,-2,5,-5,25,21]
  pdg_uborn(:,0)=[21,21,21,21,25,21]
  if (active_context.eq.2) pdg_uborn(:,0)=[2,-2,5,-5,25,21]
end subroutine

integer function get_color(id)
  implicit none
  integer :: id
  get_color=1
  if (abs(id).le.6.and.id.ne.0) get_color=3
  if (id.eq.21) get_color=8
end function

integer function get_spin(id)
  implicit none
  integer :: id
  get_spin=2
  if (id.eq.21) get_spin=3
  if (id.eq.25) get_spin=1
end function

double precision function get_mass_from_id(id)
  implicit none
  integer :: id
  double precision :: bmass,gmass,hmass
  common /test_model/ bmass,gmass,hmass
  get_mass_from_id=0d0
  if (abs(id).eq.5) get_mass_from_id=bmass
  if (id.eq.21) get_mass_from_id=gmass
  if (id.eq.25) get_mass_from_id=hmass
end function

! Controlled exponents keep the production line-by-line Sudakov bookkeeping
! and its reuse, while avoiding a numerical integral in this clustering test.
double precision function sudakov_exp(q0,q1,itype,imass)
  implicit none
  integer :: itype,builds(2),sudakov_calls
  double precision :: q0,q1,imass
  common /test_counts/ builds,sudakov_calls
  sudakov_calls=sudakov_calls+1
  sudakov_exp=-1d-4*itype*(1d0+imass)*log(q1/q0)**2
end function

double precision function expanded_sudakov_exp(q0,q1,itype,imass)
  implicit none
  integer :: itype
  double precision :: q0,q1,imass
  expanded_sudakov_exp=-1d-3*itype*(1d0+imass)*log(q1/q0)**2
end function
