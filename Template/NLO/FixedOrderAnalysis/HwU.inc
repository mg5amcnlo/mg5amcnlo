*     -*-fortran-*-

      integer max_plots,max_bins,max_wgts,max_points
      parameter (max_plots=200)
      parameter (max_bins=100)
      parameter (max_wgts=1024)
      parameter (max_points=max_plots*40)

      logical booked(max_plots)
      integer nbin(max_plots),nwgts,np,p_bin(max_points)
     &     ,p_label(max_points),histi(max_plots,max_bins)
      character*50 title(max_plots)
      character*50 wgts_info(max_wgts)
      double precision histy(max_wgts,max_plots,max_bins)
     $     ,histy_acc(max_wgts,max_plots,max_bins),histy2(max_plots
     $     ,max_bins),histy_err(max_plots,max_bins),histxl(max_plots
     $     ,max_bins),histxm(max_plots,max_bins),step(max_plots)
     $     ,p_wgts(max_wgts,max_points)

      common/HwU_common/histy,histy_acc,histy2,histy_err,histxl,histxm
     &     ,p_wgts,step,histi,nbin,p_bin,p_label,np,nwgts
     &     ,booked,title,wgts_info
