c*************************************************************************
c	Parameters used by cluster
c*************************************************************************
      include 'ncombs.inc'
      include 'ngraphs.inc'
c     parameters for clustering
      integer id_cl(n_max_cl,0:n_max_cg)
      common/cl_map/id_cl
      logical resmap(n_max_cl,n_max_cg)
      common/res_map/resmap
      double precision pt2ijcl(nexternal),mtlast
      double precision pcl(0:3,n_max_cl)
      integer imocl(nexternal), idacl(nexternal,2), igscl(0:n_max_cg) 
      common/cl_val/pcl,pt2ijcl,mtlast,imocl,idacl,igscl
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
