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

      integer nintervals,ndimmax,nintegrals,nintervals_virt
      parameter (nintervals=64,ndimmax=60,nintegrals=6,
     &		nintervals_virt=8)

      integer min_inter,min_it0,min_it1,max_it,max_points
      data min_inter /8/
      data min_it0 /4/
      data min_it1 /5/
      data max_it /100/
      data max_points /100000/

      double precision accuracy
      common /to_accuracy/ accuracy

      integer nvirt(nintervals_virt,ndimmax,maxchannels)
     $     ,nvirt_acc(nintervals_virt,ndimmax,maxchannels)
      double precision ave_virt(nintervals_virt,ndimmax,maxchannels)
     $     ,ave_virt_acc(nintervals_virt,ndimmax,maxchannels)
     $     ,ave_born_acc(nintervals_virt ,ndimmax,maxchannels)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
