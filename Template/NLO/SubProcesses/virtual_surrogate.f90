module virt_surrogate
! Per-channel fits of the finite virtual, frozen within each iteration.
! See docs/nlo_virt_trick_surrogate.md for the basis and fallback criteria.
  implicit none
  private
  integer, parameter :: vs_maxfeat=94, vs_maxchan=20, vs_minpts=50
  public :: vs_begin_point,vs_clear_links,vs_add_link,vs_features,vs_select
  public :: vs_reset,vs_value,vs_accumulate,vs_solve,vs_write_grids,vs_read_grids
  public :: vs_maxfeat

! Current-point data shared by the matrix-element and weighted-integrand stages.
  logical, public :: vs_vcomp=.false.,vs_featok=.false.,vs_ratio_set=.false.
  double precision, public :: vs_virt=0d0,vs_born=0d0,vs_fvirt=1d0,vs_ratio=0d0
! A separate virtual event omits the approximation weight. One nonzero
! residual order supplies its normalization after PDFs, bias and folds.
  integer, public :: vs_residual_order=0
  double precision, public :: vs_residual_reference=0d0
  integer, public :: vs_nfeat=4
  double precision, public :: vs_feat(vs_maxfeat)=0d0
  logical :: have_links=.false.
  double precision :: dipole_features(3)=0d0

! Normal equations and frozen coefficients for each integration channel.
  double precision :: vs_xtx(vs_maxfeat,vs_maxfeat,vs_maxchan)=0d0, &
       vs_xty(vs_maxfeat,vs_maxchan)=0d0, vs_yty(vs_maxchan)=0d0
  double precision :: vs_theta(vs_maxfeat,vs_maxchan)=0d0
  logical :: vs_fitted(vs_maxchan)=.false., vs_on(vs_maxchan)=.false.
  integer :: vs_npt(vs_maxchan)=0, vs_it_npt(vs_maxchan)=0
! The first fit uses its training residual. Later choices compare the two
! frozen approximations on points observed after fitting.
  double precision :: vs_it_s2grid(vs_maxchan)=0d0
  integer :: vs_cum_npt(vs_maxchan)=0
  double precision :: vs_cum_s2grid(vs_maxchan)=0d0, vs_cum_s2fit(vs_maxchan)=0d0, &
       vs_cum_neggrid(vs_maxchan)=0d0, vs_cum_negfit(vs_maxchan)=0d0

contains

  subroutine vs_begin_point
    vs_vcomp=.false.
    vs_featok=.false.
    vs_ratio_set=.false.
    vs_virt=0d0
    vs_born=0d0
    vs_fvirt=1d0
    vs_ratio=0d0
    vs_residual_order=0
    vs_residual_reference=0d0
    call vs_clear_links
  end subroutine vs_begin_point

  subroutine vs_clear_links
    have_links=.false.
    dipole_features=0d0
  end subroutine vs_clear_links

  subroutine vs_add_link(p,qes2,unresolved,m,n,bkl)
! Link indices belong to the real process; p contains the native Born legs.
    double precision, intent(in) :: p(0:,:),qes2,bkl
    integer, intent(in) :: unresolved,m,n
    integer :: k,l
    double precision :: logarithm
    if (bkl.eq.0d0.or.qes2.le.0d0) return
    k=m
    l=n
    if (k.gt.unresolved) k=k-1
    if (l.gt.unresolved) l=l-1
    have_links=.true.
    if (k.eq.l) then
       dipole_features(3)=dipole_features(3)+bkl
    else
       logarithm=log_invariant(p(:,k),p(:,l),qes2)
       dipole_features(1)=dipole_features(1)+bkl*logarithm
       dipole_features(2)=dipole_features(2)+bkl*logarithm**2
    endif
  end subroutine vs_add_link

  subroutine vs_features(p,born,qes2,mode)
    double precision, intent(in) :: p(0:,:),born,qes2
    integer, intent(in) :: mode
    integer :: nlegs,k,l,m
    double precision :: logarithm
    logical, save :: warned=.false.
    nlegs=size(p,2)
    vs_nfeat=4
    if (mode.eq.2) then
       if (4+nlegs*(nlegs-1).le.vs_maxfeat) then
          vs_nfeat=4+nlegs*(nlegs-1)
       else
          if (.not.warned) write(*,*) 'Too many Born particles for '// &
               'VirtTrickSurrogate=2: using the dipole features only'
          warned=.true.
       endif
    endif
    vs_feat(1)=born
    vs_feat(2:vs_nfeat)=0d0
    vs_featok=have_links.and.qes2.gt.0d0
    if (.not.vs_featok) return
    vs_feat(2:4)=dipole_features
    if (vs_nfeat.eq.4) return
    m=4
    do k=1,nlegs-1
       do l=k+1,nlegs
          logarithm=log_invariant(p(:,k),p(:,l),qes2)
          vs_feat(m+1)=born*logarithm
          vs_feat(m+2)=born*logarithm**2
          m=m+2
       enddo
    enddo
  end subroutine vs_features

  double precision function log_invariant(p,q,qes2) result(logarithm)
    double precision, intent(in) :: p(0:3),q(0:3),qes2
    double precision :: invariant
    invariant=abs(2d0*(p(0)*q(0)-p(1)*q(1)-p(2)*q(2)-p(3)*q(3)))
    logarithm=0d0
    if (invariant.gt.0d0) logarithm=log(invariant/qes2)
  end function log_invariant

  subroutine vs_select(kchan,born)
    integer, intent(in) :: kchan
    double precision, intent(in) :: born
    vs_ratio_set=.false.
    if (.not.vs_featok.or.born.eq.0d0) return
    if (kchan.lt.1.or.kchan.gt.vs_maxchan) return
    if (.not.vs_on(kchan)) return
    vs_ratio=vs_value(kchan)/born
    vs_ratio_set=.true.
  end subroutine vs_select

  subroutine vs_reset
! A fresh grid setup must not retain a fit from a previous integration.
    call vs_begin_point
    vs_nfeat=4
    vs_feat=0d0
    vs_xtx=0d0
    vs_xty=0d0
    vs_yty=0d0
    vs_theta=0d0
    vs_fitted=.false.
    vs_on=.false.
    vs_npt=0
    vs_it_npt=0
    vs_it_s2grid=0d0
    vs_cum_npt=0
    vs_cum_s2grid=0d0
    vs_cum_s2fit=0d0
    vs_cum_neggrid=0d0
    vs_cum_negfit=0d0
  end subroutine vs_reset

  double precision function vs_value(kchan)
! the fitted surrogate at the current point (matrix-element level)
    integer, intent(in) :: kchan
    vs_value=dot_product(vs_theta(1:vs_nfeat,kchan),vs_feat(1:vs_nfeat))
  end function vs_value

  subroutine vs_accumulate(kchan,virt,surr_grid,w)
! add the current point (virtual 'virt' and grid surrogate 'surr_grid', at
! matrix-element level, weight w) to the fit of channel kchan
    integer, intent(in) :: kchan
    double precision, intent(in) :: virt,surr_grid,w
    integer :: i
    double precision :: afit
    if (kchan.lt.1 .or. kchan.gt.vs_maxchan) return
    do i=1,vs_nfeat
       vs_xtx(1:vs_nfeat,i,kchan)=vs_xtx(1:vs_nfeat,i,kchan)+w*vs_feat(1:vs_nfeat)*vs_feat(i)
    enddo
    vs_xty(1:vs_nfeat,kchan)=vs_xty(1:vs_nfeat,kchan)+w*vs_feat(1:vs_nfeat)*virt
    vs_yty(kchan)=vs_yty(kchan)+w*virt**2
    vs_npt(kchan)=vs_npt(kchan)+1
    vs_it_npt(kchan)=vs_it_npt(kchan)+1
    vs_it_s2grid(kchan)=vs_it_s2grid(kchan)+w*(virt-surr_grid)**2
    if (vs_fitted(kchan)) then
       afit=vs_value(kchan)
       vs_cum_npt(kchan)=vs_cum_npt(kchan)+1
       vs_cum_s2fit(kchan)=vs_cum_s2fit(kchan)+w*(virt-afit)**2
       vs_cum_s2grid(kchan)=vs_cum_s2grid(kchan)+w*(virt-surr_grid)**2
       vs_cum_negfit(kchan)=vs_cum_negfit(kchan)+sqrt(w)*max(0d0,afit-virt)
       vs_cum_neggrid(kchan)=vs_cum_neggrid(kchan)+sqrt(w)*max(0d0,surr_grid-virt)
    endif
  end subroutine vs_accumulate

  subroutine vs_solve(kchan,signaware)
! end of an iteration: refit channel kchan on all its points so far and
! choose between the fit and the grid for the next iteration (signaware:
! compare the weights of the negative residuals, else the variances)
    integer, intent(in) :: kchan
    logical, intent(in) :: signaware
    integer :: i,j,k,nf
    double precision :: a(vs_maxfeat,vs_maxfeat),b(vs_maxfeat),ridge,s2fit,piv,fac
    if (kchan.lt.1 .or. kchan.gt.vs_maxchan) return
    nf=vs_nfeat
    if (vs_npt(kchan).ge.max(vs_minpts,3*nf)) then
       a(1:nf,1:nf)=vs_xtx(1:nf,1:nf,kchan)
       b(1:nf)=vs_xty(1:nf,kchan)
       ! small ridge for (nearly) degenerate features
       ridge=0d0
       do i=1,nf
          ridge=ridge+a(i,i)
       enddo
       ridge=1d-9*ridge/nf
       do i=1,nf
          a(i,i)=a(i,i)+ridge
       enddo
       ! Gaussian elimination with partial pivoting
       do k=1,nf
          j=k
          do i=k+1,nf
             if (abs(a(i,k)).gt.abs(a(j,k))) j=i
          enddo
          if (j.ne.k) then
             do i=1,nf
                piv=a(k,i)
                a(k,i)=a(j,i)
                a(j,i)=piv
             enddo
             piv=b(k)
             b(k)=b(j)
             b(j)=piv
          endif
          if (a(k,k).eq.0d0) return
          do i=k+1,nf
             fac=a(i,k)/a(k,k)
             a(i,k:nf)=a(i,k:nf)-fac*a(k,k:nf)
             b(i)=b(i)-fac*b(k)
          enddo
       enddo
       do k=nf,1,-1
          b(k)=(b(k)-dot_product(a(k,k+1:nf),b(k+1:nf)))/a(k,k)
       enddo
       if (vs_cum_npt(kchan).gt.0) then
          if (signaware) then
             vs_on(kchan)=vs_cum_negfit(kchan).lt.vs_cum_neggrid(kchan)
          else
             vs_on(kchan)=vs_cum_s2fit(kchan).lt.vs_cum_s2grid(kchan)
          endif
       elseif (vs_it_npt(kchan).gt.0) then
          ! first fit: in-sample residual of the fit against the grid's
          s2fit=vs_yty(kchan)-2d0*dot_product(b(1:nf),vs_xty(1:nf,kchan)) &
               +dot_product(b(1:nf),matmul(vs_xtx(1:nf,1:nf,kchan),b(1:nf)))
          vs_on(kchan)=s2fit/vs_npt(kchan).lt.vs_it_s2grid(kchan)/vs_it_npt(kchan)
       endif
       vs_theta(1:nf,kchan)=b(1:nf)
       vs_fitted(kchan)=.true.
       write (*,'(a,i3,a,i7,a,l2)') ' fitted surrogate of the virtual, channel',kchan, &
            ': points',vs_npt(kchan),', used',vs_on(kchan)
    endif
    vs_it_npt(kchan)=0
    vs_it_s2grid(kchan)=0d0
  end subroutine vs_solve

  subroutine vs_write_grids(lun,nchans)
! VSU records follow the grid/polynomial fallback in mint_grids.
    integer, intent(in) :: lun,nchans
    integer :: kchan
    do kchan=1,nchans
       write(lun,*) 'VSU',merge(1,0,vs_on(kchan)),merge(1,0,vs_fitted(kchan)),vs_nfeat, &
            vs_theta(1:vs_nfeat,kchan)
    enddo
  end subroutine vs_write_grids

  subroutine vs_read_grids(lun,nchans,enabled)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: lun,nchans
    logical, intent(in) :: enabled
    integer :: kchan,ios,ion,ifit,nfeat,i
    character(len=3) :: tag
    character(len=4096) :: line
    do kchan=1,nchans
       read(lun,'(a)',iostat=ios) line
       if (ios.ne.0) call grid_error('Missing virtual surrogate in mint_grids.')
       read(line,*,iostat=ios) tag,ion,ifit,nfeat
       if (ios.ne.0) call grid_error('Malformed virtual surrogate in mint_grids.')
       if (tag.ne.'VSU'.or.nfeat.lt.4.or.nfeat.gt.vs_maxfeat.or. &
            (ion.ne.0.and.ion.ne.1).or.(ifit.ne.0.and.ifit.ne.1)) &
            call grid_error('Invalid virtual surrogate in mint_grids.')
       read(line,*,iostat=ios) tag,ion,ifit,nfeat,(vs_theta(i,kchan),i=1,nfeat)
       if (ios.ne.0) call grid_error('Incomplete virtual surrogate in mint_grids.')
       if (.not.all(ieee_is_finite(vs_theta(1:nfeat,kchan)))) &
            call grid_error('Nonfinite virtual surrogate in mint_grids.')
       vs_nfeat=nfeat
       vs_on(kchan)=enabled.and.ion.eq.1
       vs_fitted(kchan)=ifit.eq.1
    enddo
  end subroutine vs_read_grids

  subroutine grid_error(message)
    character(len=*), intent(in) :: message
    write(*,*) 'ERROR: ',message
    stop 1
  end subroutine grid_error

end module virt_surrogate
