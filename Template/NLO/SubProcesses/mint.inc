*     -*-fortran-*-

      integer maxchannels
      parameter (maxchannels=20) ! same as in amcatnlo_run_interface
      integer nchans,iconfigs(maxchannels),kchan,iconfig,ichan
      double precision wgt_mult
      common /channels/ wgt_mult,nchans,iconfigs,iconfig,ichan

c Note that the number of intervals in the integration
c grids, 'nintervals', should be equal to
c "nintervals = min_inter * 2^n",
c where 'n' is smaller or equal to 'min_it'.

      integer nintervals,ndimmax,nintegrals,nintervals_virt,max_fold
      parameter (nintervals=64,    ! 8*2*2*2: allow for 3 times doubling of intervals
     &           ndimmax=60,       ! maximum number of integration dimensions
     &           nintegrals=6,     ! keeping track of 6 separate integrals
     &           nintervals_virt=8,! virtuals just require a coarse grid
     &           max_fold=512)     ! 8*8*8 is max folding for the three variables

      integer min_inter,min_it0,min_it1,max_it,max_points
      data min_inter /8/ ! start the grids with 8 intervals
      data min_it0   /4/ ! at least 4 iterations for the grids setup phase
      data min_it1   /5/ ! at least 5 iterations for the upper bound envelope
      data max_points /100000/ ! if after this many points the integral is still zero, an error is raised
      
      double precision accuracy
      common /to_accuracy/ accuracy

      integer nvirt(nintervals_virt,ndimmax,maxchannels)
     $     ,nvirt_acc(nintervals_virt,ndimmax,maxchannels)
      double precision ave_virt(nintervals_virt,ndimmax,maxchannels)
     $     ,ave_virt_acc(nintervals_virt,ndimmax,maxchannels)
     $     ,ave_born_acc(nintervals_virt ,ndimmax,maxchannels)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
