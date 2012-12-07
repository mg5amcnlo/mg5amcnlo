      subroutine configs_and_props_inc_chooser()
c For a given nFKSprocess, it fills the c_configs_inc common block with
c the configs.inc information (i.e. IFOREST(), SPROP(), TPRID() and
c MAPCONFIG())
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      double precision ZERO
      parameter (ZERO=0d0)
      include 'configs_and_props_info.inc'
      include 'maxparticles.inc'
      integer max_branch
      parameter (max_branch=max_particles-1)
      include 'maxconfigs.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iforest(2,-max_branch:-1,lmaxconfigs),
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      common/c_configs_inc/iforest,sprop,tprid,mapconfig

      double precision ppmass(-max_branch:nexternal,lmaxconfigs)
      double precision ppwidth(-max_branch:-1,lmaxconfigs)
      integer ppow(-max_branch:-1,lmaxconfigs)
      common/c_props_inc/ppmass,ppwidth,ppow
      include "pmass.inc"
c     
      if (max_branch_used.gt.max_branch) then
         write (*,*) 'ERROR in configs_and_props_inc_chooser:'/
     $        /' increase max_branch',max_branch,max_branch_used
         stop
      endif
      if (lmaxconfigs_used.gt.lmaxconfigs) then
         write (*,*) 'ERROR in configs_and_propsinc_chooser:'/
     $        /' increase lmaxconfigs' ,lmaxconfigs,lmaxconfigs_used
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
            do j=-maxbranch_used,-1
               do k=1,2
                  iforest(k,j,i)=IFOREST_D(nFKSprocess,k,j,i)
               enddo
               sprop(j,i)=SPROP_D(nFKSprocess,j,i)
               tprid(j,i)=TPRID_D(nFKSprocess,j,i)
               ppmass(j,i)=PMASS_D(nFKSprocess,j,i)
               ppwidth(j,i)=PWIDTH_D(nFKSprocess,j,i)
               ppow(j,i)=POW_D(nFKSprocess,j,i)
            enddo
c for the mass, also fill for the external masses
            ppmass(0,i)=0d0
            do j=1,nexternal
               ppmass(j,i)=pmass(j)
            enddo
         endif
      enddo
c
      return
      end

