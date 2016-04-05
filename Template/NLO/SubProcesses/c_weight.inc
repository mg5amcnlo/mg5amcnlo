*     -*-fortran-*-

      integer max_contr,max_wgt,max_iproc
      parameter (max_contr=128,max_wgt=1024,max_iproc=100)
      logical H_event(max_contr)
      integer nFKS(max_contr),itype(max_contr),QCDpower(max_contr)
     $     ,pdg(nexternal,0:max_contr),pdg_uborn(nexternal,0:max_contr)
     $     ,parton_pdg_uborn(nexternal,max_iproc,max_contr)
     $     ,parton_pdg(nexternal,max_iproc,max_contr),icontr,iwgt
     $     ,plot_id(max_contr),niproc(max_contr),parton_pdf(nexternal
     $     ,max_iproc,max_contr),icontr_sum(0:max_iproc,max_contr)
     $     ,icontr_picked,iproc_picked
      double precision momenta(0:3,nexternal,max_contr),momenta_m(0:3
     $     ,nexternal,2,max_contr),wgt(3,max_contr),wgt_ME_tree(2
     $     ,max_contr),bjx(2,max_contr),scales2(3,max_contr)
     $     ,g_strong(max_contr),wgts(max_wgt,max_contr)
     $     ,parton_iproc(max_iproc,max_contr),y_bst(max_contr)
     $     ,plot_wgts(max_wgt,max_contr),shower_scale(max_contr)
     $     ,unwgt(max_iproc,max_contr)
      common /c_weights/momenta,momenta_m,wgt,wgt_ME_tree,wgts,plot_wgts
     $     ,parton_iproc,bjx,scales2,g_strong,y_bst,shower_scale,unwgt
     $     ,parton_pdg,nFKS,itype,QCDpower,pdg,pdg_uborn
     $     ,parton_pdg_uborn,plot_id,niproc,icontr,iwgt,icontr_sum
     $     ,icontr_picked,iproc_picked,H_event
