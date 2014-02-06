      subroutine born_configs_and_props_inc_chooser()
c For a given NBORN, it fills the c_configs_inc common block with
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
      INTEGER NBORN
      COMMON/C_NBORN/NBORN
      integer iforest(2,-max_branch:-1,n_max_cg)
      integer sprop(-max_branch:-1,n_max_cg)
      integer tprid(-max_branch:-1,n_max_cg)
      integer mapconfig(0:n_max_cg)
      common/c_born_configs_inc/iforest,sprop,tprid,mapconfig
      double precision prmass(-max_branch:nexternal,n_max_cg)
      double precision prwidth(-max_branch:-1,n_max_cg)
      integer prow(-max_branch:-1,n_max_cg)
      common/c_born_props_inc/prmass,prwidth,prow
      logical gforcebw(-max_branch:-1,n_max_cg)
      common/c_born_gforcebw/gforcebw
      double precision pmass(nexternal)
      include 'born_configs_and_props_info.inc'
      include "pmass.inc"
c     
      if (max_branchib_used.gt.max_branch) then
         write (*,*) 'ERROR in born_configs_and_props_inc_chooser:'/
     $        /' increase max_branch',max_branch,max_branchb_used
         stop
      endif
      if (lmaxconfigsb_used.gt.n_max_cg) then
         write (*,*) 'ERROR in born_configs_and_propsinc_chooser:'/
     $        /' increase n_max_cg' ,n_max_cg,lmaxconfigsb_used
         stop
      endif
c
c Fill the arrays of the c_configs_inc and c_props_inc common
c blocks. Some of the information might not be available in the
c configs_and_props_info.inc include file, but there is no easy way of skipping
c it. This will simply fill the common block with some bogus
c information.
      do i=0,MAPCONFIG_B(NBORN,0)
         mapconfig(i)=MAPCONFIG_B(NBORN,i)
         if (i.ne.0) then
            do j=-max_branchb_used,-1
               do k=1,2
                  iforest(k,j,i)=IFOREST_D(NBORN,k,j,i)
               enddo
               sprop(j,i)=SPROP_B(NBORN,j,i)
               tprid(j,i)=TPRID_B(NBORN,j,i)
               prmass(j,i)=PMASS_B(NBORN,j,i)
               prwidth(j,i)=PWIDTH_B(NBORN,j,i)
               prow(j,i)=POW_B(NBORN,j,i)
               gforcebw(j,i)=gforcebw_B(NBORN,j,i)
            enddo
c for the mass, also fill for the external masses
            prmass(0,i)=0d0
            do j=1,nexternal - 1 
               prmass(j,i)=pmass(j)
            enddo
         endif
      enddo
c
      return
      end

