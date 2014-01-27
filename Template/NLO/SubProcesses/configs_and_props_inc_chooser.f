      subroutine configs_and_props_inc_chooser()
c For a given nFKSprocess, it fills the c_configs_inc common block with
c the configs.inc information (i.e. IFOREST(), SPROP(), TPRID() and
c MAPCONFIG())
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision ZERO
      parameter (ZERO=0d0)
      include 'maxparticles.inc'
      include 'ngraphs.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig(0:n_max_cg)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,n_max_cg)
      double precision prwidth(-max_branch:-1,n_max_cg)
      integer prow(-max_branch:-1,n_max_cg)
      common/c_props_inc/prmass,prwidth,prow
      double precision pmass(nexternal)
      include 'configs_and_props_info.inc'
      include "pmass.inc"
c     
      if (max_branch_used.gt.max_branch) then
         write (*,*) 'ERROR in configs_and_props_inc_chooser:'/
     $        /' increase max_branch',max_branch,max_branch_used
         stop
      endif
      if (lmaxconfigs_used.gt.n_max_cg) then
         write (*,*) 'ERROR in configs_and_propsinc_chooser:'/
     $        /' increase n_max_cg' ,n_max_cg,lmaxconfigs_used
         stop
      endif
c
c Fill the arrays of the c_configs_inc and c_props_inc common
c blocks. Some of the information might not be available in the
c configs_and_props_info.inc include file, but there is no easy way of skipping
c it. This will simply fill the common block with some bogus
c information.
      do i=0,MAPCONFIG_D(nFKSprocess,0)
         mapconfig(i)=MAPCONFIG_D(nFKSprocess,i)
         if (i.ne.0) then
            do j=-max_branch_used,-1
               do k=1,2
                  iforest(k,j,i)=IFOREST_D(nFKSprocess,k,j,i)
               enddo
               sprop(j,i)=SPROP_D(nFKSprocess,j,i)
               tprid(j,i)=TPRID_D(nFKSprocess,j,i)
               prmass(j,i)=PMASS_D(nFKSprocess,j,i)
               prwidth(j,i)=PWIDTH_D(nFKSprocess,j,i)
               prow(j,i)=POW_D(nFKSprocess,j,i)
            enddo
c for the mass, also fill for the external masses
            prmass(0,i)=0d0
            do j=1,nexternal
               prmass(j,i)=pmass(j)
            enddo
         endif
      enddo
c
      return
      end

