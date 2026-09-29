module recorded_histograms
  implicit none
  real(8) :: xvalue(14), weight(14), lower(14), upper(14)
end module

program check_resonance_analysis
  use recorded_histograms
  implicit none
  include 'nexternal.inc'
  include 'cuts.inc'
  real(8) :: p(0:4,nexternal), original(0:4,nexternal), reference(14)
  real(8) :: wgts(1)
  integer :: status(nexternal), ids(nexternal)
  character(7) :: info(1)
  logical, external :: dummy_cuts
  if (nexternal /= 7 .or. nincoming /= 2) stop 1
  jetradius=0.4d0
  jetalgo=-1d0
  ptj=20d0
  etaj=-1d0
  status=1
  status(1:2)=-1
  ids=[2,5,5,12,1,-11,21]
  p=0d0
  p(:,1)=[500d0,0d0,0d0,500d0,0d0]
  p(:,2)=[500d0,0d0,0d0,-500d0,0d0]
  p(:,3)=[50d0,30d0,0d0,40d0,0d0]
  p(:,4)=[40d0,0d0,-40d0,0d0,0d0]
  p(:,5)=[60d0,-60d0,0d0,0d0,0d0]
  p(:,6)=[40d0,0d0,40d0,0d0,0d0]
  original=p
  info='central'
  call analysis_begin(1,info)
  if (.not. dummy_cuts(p,status,ids)) stop 2
  wgts=1d0
  call analysis_fill(p,status,ids,wgts,3)
  reference=xvalue
  if (any(abs(weight-1d0)>1d-12)) stop 3
  if (abs(xvalue(3)-120d0)>1d-12) stop 4
  if (abs(xvalue(5)-80d0)>1d-12) stop 5
  if (abs(xvalue(6)-30d0)>1d-12) stop 6
  if (abs(xvalue(8)-60d0)>1d-12) stop 7
  if (abs(xvalue(14)-2d0)>1d-12) stop 8
  if (any(xvalue<lower).or.any(xvalue>=upper)) stop 9

  ! Splitting the bottom into a collinear bottom and gluon leaves all
  ! reconstructed observables unchanged, including the tagged jet.
  p(:,3)=0.7d0*original(:,3)
  p(:,7)=0.3d0*original(:,3)
  weight=0d0
  call analysis_fill(p,status,ids,wgts,1)
  if (any(abs(xvalue-reference)>1d-11)) stop 10
  if (weight(2)/=0d0) stop 11

  ! A soft, unclustered gluon cannot produce a resolved third jet.
  p=original
  p(:,7)=[1d-8,0d0,1d-8,0d0,0d0]
  call analysis_fill(p,status,ids,wgts,1)
  if (any(abs(xvalue-reference)>1d-11)) stop 12

  ! Signed counterevents and Born contributions fill identical spectra.
  p=original
  weight=0d0
  wgts=2d0
  call analysis_fill(p,status,ids,wgts,1)
  wgts=-1d0
  call analysis_fill(p,status,ids,wgts,2)
  wgts=0.5d0
  call analysis_fill(p,status,ids,wgts,3)
  if (abs(weight(1)-1.5d0)>1d-12) stop 13
  if (abs(weight(2)-0.5d0)>1d-12) stop 14
  if (any(abs(weight(3:)-1.5d0)>1d-12)) stop 15

  ! A harder gluon must not replace the tagged down-flavour recoil jet.
  p(:,7)=[70d0,0d0,-70d0,0d0,0d0]
  call analysis_fill(p,status,ids,wgts,1)
  if (abs(xvalue(8)-60d0)>1d-12) stop 16
  if (abs(xvalue(14)-3d0)>1d-12) stop 17

  ! A collinear bottom-antibottom pair has zero net bottom flavour.
  p=original
  p(:,7)=p(:,3)
  ids(7)=-5
  if (dummy_cuts(p,status,ids)) stop 18

  ! These limits have different underlying Born flavours from u b and
  ! are unsubtracted in the restricted generated process. Both must fail.
  p=original
  p(:,3)=[50d0,0d0,0d0,-50d0,0d0]
  p(:,7)=[70d0,0d0,-70d0,0d0,0d0]
  ids(7)=-5
  if (dummy_cuts(p,status,ids)) stop 19
  p=original
  p(:,5)=[60d0,0d0,0d0,60d0,0d0]
  p(:,7)=[70d0,0d0,-70d0,0d0,0d0]
  ids(7)=-2
  if (dummy_cuts(p,status,ids)) stop 20

  ! Separate flavour jets are required, even in a three-parton event.
  p=original
  p(:,5)=p(:,3)
  p(:,7)=[70d0,0d0,-70d0,0d0,0d0]
  ids(7)=21
  if (dummy_cuts(p,status,ids)) stop 21
  print *, 'PASS recoil analysis: observables, IR limits, weights, jets'
end program

subroutine HwU_inithist(nwgt,info)
  use recorded_histograms
  implicit none
  integer :: nwgt
  character(*) :: info(*)
  xvalue=0d0
  weight=0d0
end subroutine

subroutine HwU_book(label,title,nbin,xmin,xmax)
  use recorded_histograms
  implicit none
  integer :: label,nbin
  character(*) :: title
  real(8) :: xmin,xmax
  lower(label)=xmin
  upper(label)=xmax
end subroutine

subroutine HwU_fill(label,x,wgts)
  use recorded_histograms
  implicit none
  integer :: label
  real(8) :: x,wgts(*)
  xvalue(label)=x
  weight(label)=weight(label)+wgts(1)
end subroutine

subroutine HwU_write_file
end subroutine
