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
      include 'born_ngraphs.inc'
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
      if (max_branchb_used.gt.max_branch) then
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
                  iforest(k,j,i)=IFOREST_B(NBORN,k,j,i)
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

      subroutine born_leshouche_inc_chooser()
c For a given nborn, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'born_leshouche_info.inc'
      integer i,j,k
      INTEGER NBORN
      COMMON/C_NBORN/NBORN
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_born_leshouche_inc/idup,mothup,icolup
c
      if (maxprocb_used.gt.maxproc) then
         write (*,*) 
     &    'ERROR in born_leshouche_inc_chooser: increase maxproc',
     &        maxproc,maxprocb_used
         stop
      endif
      if (maxflowb_used.gt.maxflow) then
         write (*,*) 
     &    'ERROR in born_leshouche_inc_chooser: increase maxflow',
     &        maxflow,maxflowb_used
         stop
      endif
      do j=1,maxprocb_used
         do i=1,nexternal-1
            IDUP(i,j)=IDUP_B(nborn,i,j)
            MOTHUP(1,i,j)=MOTHUP_B(nborn,1,i,j)
            MOTHUP(2,i,j)=MOTHUP_B(nborn,2,i,j)
         enddo
      enddo
c
      do j=1,maxflowb_used
         do i=1,nexternal-1
            ICOLUP(1,i,j)=ICOLUP_B(nborn,1,i,j)
            ICOLUP(2,i,j)=ICOLUP_B(nborn,2,i,j)
         enddo
      enddo
c
      return
      end


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


      subroutine fks_inc_chooser()
c For a given nFKSprocess, it fills the c_fks_inc common block with the
c fks.inc information
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      integer i,j
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision particle_charge(nexternal), particle_charge_born(nexternal-1)
      common /c_charges/particle_charge
      common /c_charges_born/particle_charge_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      integer particle_type_born(nexternal-1)
      common /c_particle_type_born/particle_type_born
      include 'orders.inc'
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
c
      i_fks=fks_i_D(nFKSprocess)
      j_fks=fks_j_D(nFKSprocess)
      do i=1,nexternal
         if (fks_j_from_i_D(nFKSprocess,i,0).ge.0 .and.
     &        fks_j_from_i_D(nFKSprocess,i,0).le.nexternal) then
            do j=0,fks_j_from_i_D(nFKSprocess,i,0)
               fks_j_from_i(i,j)=fks_j_from_i_D(nFKSprocess,i,j)
            enddo
         else
            write (*,*) 'ERROR in fks_inc_chooser'
            stop
         endif
         particle_type(i)=particle_type_D(nFKSprocess,i)
         particle_charge(i)=particle_charge_D(nFKSprocess,i)
         pdg_type(i)=pdg_type_D(nFKSprocess,i)
      enddo
      do i=1,nexternal

         if (i.lt.min(i_fks,j_fks)) then
            particle_type_born(i)=particle_type(i)
            particle_charge_born(i)=particle_charge(i)
         elseif (i.gt.max(i_fks,j_fks)) then
            particle_type_born(i-1)=particle_type(i)
            particle_charge_born(i-1)=particle_charge(i)
         elseif (i.eq.min(i_fks,j_fks)) then
            i_type=particle_type(i_fks)
            j_type=particle_type(j_fks)
            ch_i=particle_charge(i_fks)
            ch_j=particle_charge(j_fks)
            if (abs(i_type).eq.abs(j_type)) then
               m_type=8
               ch_m = 0d0
               if ( (j_fks.le.nincoming .and.
     &              abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &              (j_fks.gt.nincoming .and.
     &              abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
                  write(*,*)'Flavour mismatch #1 in fks_inc_chooser',
     &                 i_fks,j_fks,i_type,j_type
                  stop
               endif
            elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
               if(j_fks.le.nincoming)then
                  m_type=-i_type
                  ch_m = -ch_i
               else
                  write (*,*) 'Error in fks_inc_chooser: (i,j)=(q,g)'
                  stop
               endif
            elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
               m_type=j_type
               ch_m= ch_j
            else
               write(*,*)'Flavour mismatch #2 in fks_inc_chooser',
     &              i_type,j_type,m_type
               stop
            endif
            particle_type_born(i)=m_type
            particle_charge_born(i)=ch_m
         elseif (i.ne.max(i_fks,j_fks)) then
            particle_type_born(i)=particle_type(i)
            particle_charge_born(i)=particle_charge(i)
         endif
      enddo
      
      do i = 1, nsplitorders
         split_type(i) = split_type_d(nFKSprocess,i)
      enddo
      return
      end


      subroutine leshouche_inc_chooser()
c For a given nFKSprocess, it fills the c_leshouche_inc common block with the
c leshouche.inc information
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'leshouche_info.inc'
      integer i,j,k
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
c
      if (maxproc_used.gt.maxproc) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxproc',
     &        maxproc,maxproc_used
         stop
      endif
      if (maxflow_used.gt.maxflow) then
         write (*,*) 'ERROR in leshouche_inc_chooser: increase maxflow',
     &        maxflow,maxflow_used
         stop
      endif
      do j=1,maxproc_used
         do i=1,nexternal
            IDUP(i,j)=IDUP_D(nFKSprocess,i,j)
            MOTHUP(1,i,j)=MOTHUP_D(nFKSprocess,1,i,j)
            MOTHUP(2,i,j)=MOTHUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      do j=1,maxflow_used
         do i=1,nexternal
            ICOLUP(1,i,j)=ICOLUP_D(nFKSprocess,1,i,j)
            ICOLUP(2,i,j)=ICOLUP_D(nFKSprocess,2,i,j)
         enddo
      enddo
c
      return
      end

