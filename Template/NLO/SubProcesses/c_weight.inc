*     -*-fortran-*-

      integer max_contr,max_wgt
      parameter (max_contr=128,max_wgt=256)
      integer nFKS(max_contr),itype(max_contr),QCDpower(max_contr)
     $     ,pdg(nexternal,max_contr),pdg_uborn(nexternal,max_contr)
     $     ,icontr,iwgt,plot_id(max_contr)
      double precision momenta(0:3,nexternal,max_contr),wgt(3,max_contr)
     &     ,bjx(2,max_contr),scales2(3,max_contr),g_strong(max_contr)
     &     ,wgts(max_wgt,max_contr),y_bst(max_contr),plot_wgts(max_wgt
     &     ,max_contr)
      common /c_weights/momenta,wgt,wgts,plot_wgts,bjx,scales2,g_strong
     &     ,y_bst,nFKS,itype,QCDpower,pdg,pdg_uborn,plot_id,icontr,iwgt
