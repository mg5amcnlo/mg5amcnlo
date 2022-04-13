c*************************************************************************
c	Parameters used by cluster
c*************************************************************************
      include 'ncombs.inc'
      include 'ngraphs.inc'
      include 'vector.inc'
      include 'maxconfigs.inc'
c     parameters for clustering:
c     id_cl gives diagrams for propagators     
      integer id_cl(maxsproc,n_max_cl,0:n_max_cg)
      logical heavyrad(n_max_cg)
      common/cl_map/id_cl,heavyrad
c     vectorised version
      integer v_id_cl(maxsproc,n_max_cl,0:n_max_cg, nb_page)
      logical v_heavyrad(n_max_cg,nb_page)
      common/v_cl_map/v_id_cl,v_heavyrad
c     resmap gives potential resonances for diagrams
      logical resmap(n_max_cl,n_max_cg)
      common/res_map/resmap
c     vectorised version
      logical v_resmap(n_max_cl,n_max_cg,nb_page)
      common/v_res_map/v_resmap
c     cl_val      
      double precision pt2ijcl(nexternal),zcl(nexternal),mt2ij(nexternal),mt2last
      double precision pcl(0:4,n_max_cl) ! 4 is mass**2
      integer imocl(nexternal), idacl(nexternal,2), igraphs(0:n_max_cg)
      integer ipdgcl(n_max_cl,n_max_cg,maxsproc)
      logical clustered
      common/cl_val/pcl,pt2ijcl,zcl,mt2ij,mt2last,imocl,idacl,igraphs,ipdgcl,
     &    clustered
c     vectorised_version
      double precision v_pt2ijcl(nexternal,nb_page),v_zcl(nexternal,nb_page),v_mt2ij(nexternal,nb_page)
      double precision v_mt2last(nb_page)
      double precision v_pcl(0:4,n_max_cl,nb_page) ! 4 is mass**2
      integer v_imocl(nexternal,nb_page), v_idacl(nexternal,2,nb_page), v_igraphs(0:n_max_cg,nb_page)
      integer v_ipdgcl(n_max_cl,n_max_cg,maxsproc,nb_page)
      logical v_clustered(nb_page)
      common/v_cl_val/v_pcl,v_pt2ijcl,v_zcl,v_mt2ij,v_mt2last,v_imocl,v_idacl,v_igraphs,v_ipdgcl,
     &   v_clustered
c     variable for keeping track of BWs
      integer nbw,ibwlist(2,nexternal)
      logical isbw(n_max_cl)
      common/cl_isbw/nbw,isbw,ibwlist
c     vectorized version
      integer v_nbw(nb_page),v_ibwlist(2,nexternal,nb_page)
      logical v_isbw(n_max_cl,nb_page)
      common/v_cl_isbw/v_nbw,v_isbw,v_ibwlist
c     variable for LH output of clustering info (R,E,S,I)
      integer icluster(4,nexternal)
      common/cl_iclus/icluster
c     vectorised version
      integer v_icluster(4,nexternal,nb_page)
      common/v_cl_iclus/v_icluster
c     parameters for qcd jets
      double precision ptclus(nexternal)
      common/cl_jets/ptclus
c     vectorised version      
      double precision v_ptclus(nexternal,nb_page)
      common/v_cl_jets/v_ptclus      
c     parameters for sudakovs
      double precision m_colfac(-6:21),m_dlog(-6:21),m_slog(-6:21),
     &     m_power(-6:21,3,2),m_qmass(21)
      double precision m_as_factor,m_kfac,m_lastas
      integer m_pca(-6:21,2)
      common/cl_sud/m_colfac,m_dlog,m_slog,m_power,m_qmass,
     &     m_as_factor,m_kfac,m_lastas,m_pca

c     parameters for sudakovs vectorised version
c      double precision v_m_colfac(-6:21,nb_page),v_m_dlog(-6:21,nb_page),v_m_slog(-6:21,nb_page),
c     &     v_m_power(-6:21,3,2,nb_page),v_m_qmass(21,nb_page)
c      double precision v_m_as_factor(nb_page),v_m_kfac(nb_page),v_m_lastas(nb_page)
c      integer v_m_pca(-6:21,2,nb_page)
c      common/v_cl_sud/v_m_colfac,v_m_dlog,v_m_slog,v_m_power,v_m_qmass,
c     &    v_m_as_factor,v_m_kfac,v_m_lastas,v_m_pca

      double precision Q1
      integer iipdg,iimode
      common/gamma_args/Q1,iipdg,iimode
c     vectorised version      
c      double precision v_Q1(nb_page)
c      integer v_iipdg(nb_page),v_iimode(nb_page)
c      common/v_gamma_args/v_Q1,v_iipdg,v_iimode      


c     Common block for reweighting info
c     q2bck holds the central q2fact scales
      integer v_jlast(2,nb_page)
      integer v_njetstore(lmaxconfigs,nb_page),v_iqjetstore(nexternal-2,lmaxconfigs,nb_page)
      real*8 v_q2bck(2,nb_page)
      integer v_njets(nb_page),v_iqjets(nexternal,nb_page)
      common /v_to_rw/v_jlast,v_njetstore,v_iqjetstore,v_njets,v_iqjets,v_q2bck