*     -*-fortran-*-

      integer maxchannels
      parameter (maxchannels=20) ! same as in amcatnlo_run_interface
      integer nchans,iconfigs(maxchannels),kchan,iconfig,ichan
      double precision wgt_mult
      common /channels/ wgt_mult,nchans,iconfigs,iconfig,ichan

c Note that the number of intervals in the integration
c grids, 'nintervals', should be equal to
c     nintervals = min_inter * 2^n,
c where 'n' is an integer smaller than or equal
c to min(min_it0,min_it1).
c
c The number of intergrals should be equal to
c     nintegrals=6+2*n_ave_virt
c

      integer nintervals,ndimmax,n_ave_virt,nintegrals,nintervals_virt,
     &        min_inter,min_it0,min_it1,max_points
      parameter (
     &		nintervals=32,    ! max number of intervals in the integration grids
     &		ndimmax=60,       ! max number of dimensions of the integral
     &		n_ave_virt=10,    ! max number of grids to set up to approx virtual
     &		nintegrals=26,    ! number of integrals to keep track of
     &		nintervals_virt=8,! max number of intervals in the grids for the approx virtual
     &          min_inter=4,      ! minimal number of iterations
     &		min_it0=4,        ! minimal number of iterations in the mint step 0 phase
     &		min_it1=5,        ! minimal number of iterations in the mint step 1 phase
     &		max_points=100000 ! maximum number of points to trow per iteration if not enough non-zero points can be found.
     &		)

      double precision accuracy
      common /to_accuracy/ accuracy

      character*13 title(nintegrals)
      data title( 1)/'ABS integral '/
      data title( 2)/'Integral     '/
      data title( 3)/'Virtual      '/
      data title( 4)/'Virtual ratio'/
      data title( 5)/'ABS virtual  '/
      data title( 6)/'Born         '/
      data title( 7)/'V  1         '/
      data title( 8)/'B  1         '/
      data title( 9)/'V  2         '/
      data title(10)/'B  2         '/
      data title(11)/'V  3         '/
      data title(12)/'B  3         '/
      data title(13)/'V  4         '/
      data title(14)/'B  4         '/
      data title(15)/'V  5         '/
      data title(16)/'B  5         '/
      data title(17)/'V  6         '/
      data title(18)/'B  6         '/
      data title(19)/'V  7         '/
      data title(20)/'B  7         '/
      data title(21)/'V  8         '/
      data title(22)/'B  8         '/
      data title(23)/'V  9         '/
      data title(24)/'B  9         '/
      data title(25)/'V 10         '/
      data title(26)/'B 10         '/

      integer nvirt(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels)
     $     ,nvirt_acc(nintervals_virt,ndimmax,0:n_ave_virt,maxchannels)
      double precision ave_virt(nintervals_virt,ndimmax,0:n_ave_virt
     $     ,maxchannels),ave_virt_acc(nintervals_virt,ndimmax
     $     ,0:n_ave_virt,maxchannels),ave_born_acc(nintervals_virt
     $     ,ndimmax,0:n_ave_virt,maxchannels)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
