c*************************************************************************
c	Parameters used by cluster
c*************************************************************************
      include 'ncombs.inc'
      include 'ngraphs.inc'
c     parameters for clustering:
c     id_cl gives diagrams for propagators     
      integer id_cl(maxsproc,n_max_cl,0:n_max_cg)
      logical heavyrad(n_max_cg)
      common/cl_map/id_cl,heavyrad
c     resmap gives potential resonances for diagrams
      logical resmap(n_max_cl,n_max_cg)
      common/res_map/resmap
      double precision pt2ijcl(nexternal),zcl(nexternal),mt2ij(nexternal),mt2last
      double precision pcl(0:4,n_max_cl) ! 4 is mass**2
      integer imocl(nexternal), idacl(nexternal,2), igraphs(0:n_max_cg)
      integer ipdgcl(n_max_cl,n_max_cg,maxsproc)
      logical clustered
      common/cl_val/pcl,pt2ijcl,zcl,mt2ij,mt2last,imocl,idacl,igraphs,ipdgcl,
     &    clustered
c     variable for keeping track of BWs
      integer nbw,ibwlist(2,nexternal)
      logical isbw(n_max_cl)
      common/cl_isbw/nbw,isbw,ibwlist
c     variable for LH output of clustering info (R,E,S,I)
      integer icluster(4,nexternal)
      common/cl_iclus/icluster
c     parameters for qcd jets
      double precision ptclus(nexternal)
      common/cl_jets/ptclus
c     parameters for sudakovs
      double precision m_colfac(-6:21),m_dlog(-6:21),m_slog(-6:21),
     &     m_power(-6:21,3,2),m_qmass(21)
      double precision m_as_factor,m_kfac,m_lastas
      integer m_pca(-6:21,2)
      common/cl_sud/m_colfac,m_dlog,m_slog,m_power,m_qmass,
     &     m_as_factor,m_kfac,m_lastas,m_pca
      double precision Q1
      integer iipdg,iimode
      common/gamma_args/Q1,iipdg,iimode
