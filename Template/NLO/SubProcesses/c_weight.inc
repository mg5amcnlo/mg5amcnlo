*     -*-fortran-*-

      integer max_contr,max_wgt,max_iproc
      parameter (max_contr=128,max_wgt=256,max_iproc=100)
      logical H_event(max_contr)
      integer nFKS(max_contr),itype(max_contr),QCDpower(max_contr)
     $     ,pdg(nexternal,max_contr),parton_pdg_uborn(nexternal
     $     ,max_iproc,max_contr),parton_pdg(nexternal,max_iproc
     $     ,max_contr),icontr,iwgt,plot_id(max_contr),niproc(max_contr)
     $     ,parton_pdf(nexternal,max_iproc,max_contr)
      double precision momenta(0:3,nexternal,max_contr),wgt(3,max_contr)
     &     ,bjx(2,max_contr),scales2(3,max_contr),g_strong(max_contr)
     &     ,wgts(max_wgt,max_contr),parton_iproc(max_iproc,max_contr)
     &     ,y_bst(max_contr),plot_wgts(max_wgt,max_contr)
     &     ,shower_scale(max_contr),unwgt(max_iproc,max_contr)
     &     ,unwgt_sum(max_contr)
      common /c_weights/momenta,wgt,wgts,plot_wgts,parton_iproc,bjx
     &     ,scales2,g_strong,y_bst,shower_scale,unwgt,unwgt_sum
     &     ,parton_pdg,nFKS,itype,QCDpower,pdg,parton_pdg_uborn,plot_id
     &     ,niproc,icontr,iwgt,H_event
