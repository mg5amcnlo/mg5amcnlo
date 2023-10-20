      subroutine cluster_and_reweight(iproc,sudakov,expanded_sudakov
     $     ,nqcdrenscale,qcd_ren_scale,qcd_fac_scale,need_matching)
C main wrapper routine for the FxFx clustering, Sudakov inclusion and
C renormalisation scale setting (to be used to somewhere else to
C reweight alphaS). Should be called with iproc=0 for n-body
C contributions and iproc=nFKSprocess for the real-emissions. These
C routines assume that the c_configuration common block has already been
C filled with the iforest etc. information, the momenta are given in the
C pborn and pev common blocks and the current integration channel is
C given by the to_mconfigs common block.
      use weight_lines
      implicit none
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'maxparticles.inc'
      include 'nFKSconfigs.inc'
      include 'real_from_born_configs.inc'
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_ev(0:3,nexternal)
      common/pev/      p_ev
      integer            this_config
      common/to_mconfigs/this_config
      integer nfks1,iproc
      parameter (nfks1=fks_configs+1)
      integer i,j,il_list,il_pdg,next,nbr,ipdg(nexternal,0:fks_configs)
     $     ,cluster_list(2*max_branch*lmaxconfigs*(fks_configs+1))
     $     ,cluster_pdg(3*3*max_branch*lmaxconfigs*(fks_configs+1))
     $     ,cluster_conf,iconf,cluster_ij(max_branch),iord(0:max_branch)
     $     ,nqcdrenscale,iconfig,need_matching(nexternal)
     $     ,cluster_type(maskr(nexternal),0:fks_configs)
      double precision cluster_scales(0:max_branch),qcd_fac_scale
     $     ,qcd_ren_scale(0:nexternal),sudakov,expanded_sudakov,pcl(0:3
     $     ,nexternal)
      logical firsttime(0:fks_configs),skip_first
      data (firsttime(i),i=0,fks_configs) /nfks1*.true./
      save ipdg,cluster_list,cluster_pdg,cluster_type,firsttime
      if (iproc.eq.0) then      ! n-body contribution
         next=nexternal-1
         do i=1,next
            do j=0,3
               pcl(j,i)=p_born(j,i)
            enddo
         enddo
      else                      ! n+1-body contribution
         next=nexternal
         do i=1,next
            do j=0,3
               pcl(j,i)=p_ev(j,i)
            enddo
         enddo
      endif
      nbr=next-3 ! number of clusterings to get to a 2->1 process
      if (firsttime(iproc)) then
         cluster_type(1:maskr(nexternal),iproc)=0 ! set to zero
         call set_pdg(0,max(1,iproc)) ! use max() here to get something for iproc=0
         if (iproc.eq.0) then
            do i=1,next
               ipdg(i,iproc)=pdg_uborn(i,0)
            enddo
         else
            do i=1,next
               ipdg(i,iproc)=pdg(i,0)
            enddo
         endif
c Links the configurations (given in iforest) to the allowed clusterings
c (in cluster_list) with their corresponding PDG codes (in cluster_pdg)
         do iconf=1,mapconfig(0,iproc)
            call set_array_indices(iproc,iconf,il_list,il_pdg)
            call iforest_to_list(next,nincoming,nbr,iforest(1,-(nbr+1)
     $           ,iconf,iproc),sprop(-nbr,iconf,iproc),tprid(-nbr,iconf
     $           ,iproc),pwidth(-nbr,iconf,iproc),ipdg(1,iproc)
     $           ,cluster_list(il_list),cluster_pdg(il_pdg)
     $           ,cluster_type(1,iproc))
         enddo
         firsttime(iproc)=.false.
      endif
c Cluster the event. This gives the most-likely clustering (in
c cluster_ij) with scales (in cluster_scales)
      call set_array_indices(iproc,1,il_list,il_pdg)
      if (iproc.eq.0) then
         iconfig=this_config
      else
         iconfig=real_from_born_conf(this_config,iproc)
      endif
      call cluster(next,pcl,mapconfig(0,iproc),nbr
     $     ,cluster_list(il_list),cluster_pdg(il_pdg),iforest(1,-(nbr+1)
     $     ,iconfig,iproc),ipdg(1,iproc),pmass(-nbr,iconfig,iproc)
     $     ,pwidth(-nbr,iconfig,iproc),iconfig,sprop(-nbr,iconfig,iproc)
     $     ,cluster_conf,cluster_scales,cluster_ij,iord,cluster_type(1
     $     ,iproc))
c Given the most-likely clustering, it returns the corresponding Sudakov
c form factor and renormalisation and factorisation scales.
      if (iproc.eq.0) then
         skip_first=.false.
      else
         skip_first=.true. ! for real-emission: skip the first clustering
      endif
      call set_array_indices(iproc,cluster_conf,il_list,il_pdg)
      need_matching(1:nexternal)=-99
      call reweighting(next,pcl,nbr,skip_first,cluster_ij,ipdg(1,iproc)
     $     ,cluster_pdg(il_pdg),cluster_scales,iord,sudakov
     $     ,expanded_sudakov,nqcdrenscale,qcd_ren_scale,qcd_fac_scale
     $     ,need_matching(1))
      return
      end
      
      subroutine set_array_indices(iproc,iconf,il_list,il_pdg)
c Given 'iproc' and 'iconf', returns the corresponding location in the
c cluster_list and cluster_pdg arrays ('il_list' and 'il_pdg',
c respectively)
      implicit none
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'maxparticles.inc'
      include 'nFKSconfigs.inc'
      integer i,iproc,iconf,il_list,il_pdg
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      il_list=1
      il_pdg=1
      do i=0,iproc-1
         if (i.eq.0) then
            il_list=il_list+2*(nexternal-4)*mapconfig(0,i)
            il_pdg=il_pdg+3*(1+2*(nexternal-4))*mapconfig(0,i)
         else
            il_list=il_list+2*(nexternal-3)*mapconfig(0,i)
            il_pdg=il_pdg+3*(1+2*(nexternal-3))*mapconfig(0,i)
         endif
      enddo
      if (iproc.eq.0) then
         il_list=il_list+2*(nexternal-4)*(iconf-1)
         il_pdg=il_pdg+3*(1+2*(nexternal-4))*(iconf-1)
      else
         il_list=il_list+2*(nexternal-3)*(iconf-1)
         il_pdg=il_pdg+3*(1+2*(nexternal-3))*(iconf-1)
      endif
      return
      end

      
CCCCCCCCCCCCCCC-- INITIALISATION -- CCCCCCCCCCCCCCCC

      subroutine iforest_to_list(next,nincoming,nbr,iforest,sprop
     $     ,tprid,prwidth,ipdg,cluster_list,cluster_pdg,cluster_type)
c Given iforest it returns cluster_list. This is a binary list where the
c intermediate particle numbers are the sums of the external ones (which
c go from 1,2,4,8... If 4 and 8 are connected, the label of intermediate
c particle is 4+8=12. Since t-channels can be clustered top-down or
c bottom-up, intermediate particles have two different labels depending
c on two possible clustering orders). It also fills cluster_pdg, the PDG
c codes for particles in the cluster_list as well as their daughters.
      implicit none
      integer ibr,i,j,next,nbr,iforest(2,-(nbr+1):-1),cluster_list(2
     $     *nbr),cluster_pdg(0:2,0:2*nbr),cluster_tmp(-nbr:next)
     $     ,ipdg(next),nincoming,sprop(-nbr:-1),tprid(-nbr:-1),n_tchan
     $     ,cluster_type(maskr(next)),iord(0:nbr),iclus,ico,imo
     $     ,get_color
      double precision prwidth(-nbr:-1),mass,get_mass_from_id
      logical s_chan
      external get_color,get_mass_from_id
      if (nincoming.ne.2) then
         write (*,*) 'clustering only works for 2->X process;'/
     $        /' not for decays',nincoming
      endif
      if (nbr.ne.next-3) then
         ! 'nbr' is the number of clusterings to end up with a trivial
         ! 2->1 process.
         write (*,*) 'Wrong number of clusterings',nbr,next
         stop 1
      endif
      do j=1,next
         cluster_tmp(j)=ishft(1,j-1) ! 1, 2, 4, 8, 16, 32, ...
      enddo
      n_tchan=0
      s_chan=.true.
      do j=-1,-nbr,-1
         cluster_tmp(j)=cluster_tmp(iforest(1,j))
     $                 +cluster_tmp(iforest(2,j))
         if ( btest(cluster_tmp(j),0) .or. 
     $        btest(cluster_tmp(j),1) ) s_chan=.false.
         if (s_chan) then
            cluster_pdg(0,-j)=sprop(j)
         else
            cluster_pdg(0,-j)=tprid(j)
            n_tchan=n_tchan+1
         endif
         do i=1,2               ! daughters
            if (iforest(i,j).gt.0) then ! daughters are final state
               cluster_pdg(i,-j)=ipdg(iforest(i,j))
            else                ! daughters are also internal particles
               cluster_pdg(i,-j)=cluster_pdg(0,-iforest(i,j))
            endif
         enddo
         if (s_chan .and. (btest(cluster_tmp(j),0) .or.
     $                     btest(cluster_tmp(j),1))) then
            ! for s-channels should never have combined with initial
            ! state particles
            write (*,*) 's-channel/t-channel not consistent',s_chan
     $           ,cluster_tmp(j)
            stop 1
         endif
      enddo
c Can also go the reverse way through the t-channels. Hence, we need to
c add that information. Simply duplicate everything (also the
c s-channels); just need to be careful to update the PDG codes for the
c daughters correctly.
      do ibr=1,nbr
         cluster_list(ibr)=cluster_tmp(-ibr)
         cluster_list(ibr+nbr)=maskr(next)-cluster_tmp(-ibr)
         cluster_pdg(0,ibr+nbr)=cluster_pdg(0,ibr)
         if (ibr.le.nbr-n_tchan) then ! s-channel
            ! This is not relevant, since the reverse ordering is
            ! never a valid clustering. Fill it anyway.
            do i=1,2 
               cluster_pdg(i,ibr+nbr)=cluster_pdg(i,ibr)
            enddo
         else
            ! Here we need to be careful and rearrange the daughters
            ! correctly.  The t-channels in iforest are such that the
            ! 2nd daughter is an s-channel particle (or a final state
            ! particle). Hence, we can use the one of the next
            ! t-channel (in the normal ordering) as daughter for the
            ! current one (the reverse ordering).
            if (iforest(2,-(ibr+1)).gt.2) then ! final state particle
               cluster_pdg(2,ibr+nbr)=ipdg(iforest(2,-(ibr+1)))
            elseif (iforest(2,-(ibr+1)).lt.0 .and.
     $              iforest(2,-(ibr+1)).ge.-(nbr-n_tchan)) then ! s-channel
              cluster_pdg(2,ibr+nbr)=cluster_pdg(0,-iforest(2,-(ibr+1)))
            else
               write (*,*) 'In iforest, 2nd daughter should be '/
     $              /'final state particle, or s-channel',
     $              iforest(2,-(ibr+1)),ibr,n_tchan,nbr
               stop 1
            endif
            ! First daughter is the next t-channel (or 2nd incoming
            ! particle)
            if (ibr.eq.nbr) then
               ! this is the final t-channel; hence, it's attached to
               ! the 2nd incoming particle.
               cluster_pdg(1,ibr+nbr)=ipdg(2)
            else
               cluster_pdg(1,ibr+nbr)=cluster_pdg(0,ibr+1)
            endif
         endif
      enddo

c Set the type of the clustered particle. Use binary coding, since some
c clustered particles can be multiple types (e.g. gluon/photon splitting
c to a quark anti-quark pair) and through summing the binary labels, we
c can keep track of that information. We use this to determine the
c clustering scale in 'cluster_one_step'.
      do ibr=1,nbr*2
         iclus=cluster_list(ibr)
         imo=cluster_pdg(0,ibr)
         ico=get_color(imo)
         mass=get_mass_from_id(imo)
         call set_particle_type(cluster_type(iclus),ico,mass,.false.) 
      enddo
      return
      end

      subroutine set_particle_type(itype,ico,mass,ext)
c Based on the colour (ico) and mass of the particle, use a binary
c labeling for the particle-type. Colour singlets are always treated as
c massive, except when they are external particles (ie., ext=.true.).
      implicit none
      integer ico,itype
      double precision mass
      logical ext
      if (ico.eq.8 .and. mass.eq.0d0) then
         if (.not.btest(itype,0)) 
     $        itype = itype+1
      elseif (abs(ico).eq.3 .and. mass.eq.0d0 ) then
         if (.not.btest(itype,1)) 
     $        itype = itype+2
      elseif (abs(ico).eq.3 .and. mass.ne.0d0 ) then
         if (.not.btest(itype,2)) 
     $        itype = itype+4
      elseif (abs(ico).eq.1 .and. mass.eq.0d0 .and. ext) then
         if (.not.btest(itype,3)) 
     $        itype = itype+8
      elseif (abs(ico).eq.1) then
!        for intermediate colour singlets, we do not care if it is
!        massive or not (they should always be far enough
!        off-shell). Treat them always as massive to avoid issues with
!        Z/gamma^* interference.
         if (.not.btest(itype,4)) 
     $        itype = itype+16
      else
c Stop if a particle type not implemented
         write (*,*) 'Unknown particle in update_type, mass='
     $        ,mass,', color=',ico
         stop 1
      endif
      end
      

CCCCCCCCCCCCCCC-- MAIN CLUSTER ROUTINE -- CCCCCCCCCCCCCCCC

      subroutine cluster(next,p,nconf,nbr,cluster_list,cluster_pdg,itree
     $     ,ipdg,prmass,prwidth,iconfig,sprop,cluster_conf
     $     ,cluster_scales,cluster_ij,iord,cluster_type)
c Takes a set of momenta and clusters them according to possible diagram
c configurations (given by cluster_list).  The idea is to perform the
c clusterings until we have a 2->1 process. Clusterings are only done if
c there is a diagram that has the same topology as given by the
c clusterings.  Returns the diagram corresponding to the clustering
c (cluster_conf), the order of the clusterings (cluster_ij) and the
c clustering scales (cluster_scales).
      implicit none
      integer next,i,j,nleft,imap(next),iwin,jwin,win_id,nconf,nvalid
     $     ,nbr,cluster_list(2*nbr,nconf),cluster_conf,iclus,ipdg(next)
     $     ,cluster_ij(nbr),cluster_pdg(0:2,0:2*nbr,nconf),iord(0:nbr)
     $     ,iBWlist(2,0:nbr),itree(2,-(nbr+1):-1),iconf,iconfig,sprop(
     $     -nbr:-1),cluster_type(maskr(next)),get_color,ico
     $     ,particle_type(next)
      double precision p(0:3,next),pcl(0:4,next),cluster_scales(0:nbr)
     $     ,scale,p_inter(0:4,0:2,0:nbr),prmass(-nbr:-1),prwidth(-nbr:-1)
     $     ,djb_clus,get_mass_from_id,mt_2to2,mass
      logical valid_conf(nconf),is_bw,cluster_according_to_iconfig
      parameter (cluster_according_to_iconfig=.false.)
      external djb_clus,get_mass_from_id,get_color
      do i=1,next
         do j=0,3
            pcl(j,i)=p(j,i)
         enddo
         pcl(4,i)=abs(get_mass_from_id(ipdg(i)))
c imap links the current particle labels with the binary coding.
         imap(i)=ishft(1,i-1)
c fill the type for the external particles
         particle_type(i)=0
         ico=get_color(ipdg(i))
         mass=get_mass_from_id(ipdg(i))
         call set_particle_type(particle_type(i),ico,mass,.true.)
      enddo
c Set all diagrams (according to which we cluster) as valid
      call reset_valid_confs(nconf,nvalid,valid_conf)
      if (cluster_according_to_iconfig) 
     $     call limit_cluster_iconfig(nconf,iconfig,nvalid,valid_conf)
c Remove diagrams (according to which we cluster) that are not
c compatible with the resonance structure of the current phase-space
c point. First check which s-channel particles are a Breit-Wigner
c (according to the integration channel)
      call IsBreitWigner(next,nbr,p,itree,prwidth,prmass,ipdg,sprop
     $     ,iBWlist)
      call remove_confs_BW(nconf,nbr,nvalid,valid_conf,iBWlist
     $     ,cluster_list,cluster_pdg)
      nleft=next
c Loop over the clusterings until we have a 2->1 process. Hence there
c should be 'nbr' (number of branchings) clusterings.
      do iclus=1,nbr
c Do one clustering (returning iwin, jwin and win_id and the scale)
         call cluster_one_step(nleft,pcl(0,1),imap(1),nbr,nconf
     $        ,valid_conf,cluster_list,iBWlist,iwin,jwin,win_id
     $        ,scale,is_bw,cluster_type,particle_type)
c Remove diagrams that do not have the win_id among its clusterings
         call update_valid_confs(win_id,nconf,nbr,nvalid,valid_conf
     $        ,cluster_list)
         if (nleft.eq.4 .and. (btest(win_id,0).or.btest(win_id,1))) then
            ! save the value of mT^2 of the particle not clustered,
            ! since it will set the final cluster_scale below
            mT_2to2=sqrt(djb_clus(pcl(0,7-iwin)))
         endif
c Combine the momenta (and update particle types)
         call update_momenta(nleft,pcl(0,1),iwin,jwin,p_inter(0,0
     $        ,iclus),is_bw,particle_type,cluster_type(win_id))
c Update imap (the map that links current particle labels with the
c binary labeling). Since we combine particles, we need to update the
c corresponding imap label with the combined particle label.
         call update_imap(nleft,imap(1),iwin,jwin,win_id)
c Save the cluster scales and cluster IDs.
         cluster_scales(iclus)=scale
         cluster_ij(iclus)=win_id
         nleft=nleft-1    ! 'nleft' particles are left after the cluster
      enddo
c Set the cluster_conf to (one of) the diagram(s) compatible with the
c clustering found just above
      call set_cluster_conf(nconf,nvalid,valid_conf,iconfig
     $     ,cluster_conf)
c Set the cluster_scale of the final 2->1 process.      
      if (.not.(btest(win_id,0).or. btest(win_id,1))) then
         ! s-channel 2->2 process. Use m_T^2 of final state particle in
         ! 2->1 process (which is equal to its invariant mass, since
         ! pT=0)
         cluster_scales(0)=sqrt(djb_clus(pcl(0,3)))
      else
         ! Set the final scale to the m_T^2 of the final state particle
         ! that was not clustered when clustering from the 2->2 to the
         ! 2->1 process, using the momenta from the 2->2 process.
         cluster_scales(0)=mT_2to2
      endif
c Set the daughter momenta of the 2->1 process (do not need the mother)
      p_inter(:,1,0)=pcl(:,1)
      p_inter(:,2,0)=pcl(:,2)
c Link the cluster_ij values to the ordering used in cluster_pdg (which
c is similar to the one in iforest)
      call link_clustering_to_iforest(nbr,cluster_ij,cluster_list(1
     $     ,cluster_conf),iord)
c Fill the cluster_pdg(0:2,0) with the information of the PDG codes for
c the final, completely clustered 2->1 system
      call set_cluster_pdg_2_1_process(next,nbr,ipdg,cluster_pdg(0,0
     $     ,cluster_conf),cluster_ij,iord)
c Now that we have the diagram configuration corresponding to the
c clustering, we can update some of the clustering scales depending on
c the PDG codes of the particles. In particular, the correct HARD scale
c in the Sudakov Form Factors (given by vertices with only 2 QCD
c partons, e.g. Z->q+qbar), is NOT the clustering scale, but rather
c simply 2*p1.p2, where p1 and p2 are the momenta of the QCD particles
c (this is correct for both massless and massive particles).  Also, if
c final 2->2 process is s-channel QCD process (e.g. qqbar->ttbar),
c update the central hard scale and the final cluster scale to the
c geometric mean of the transverse masses in the 2->2 system.
      call update_cluster_scales(nbr,p_inter,cluster_scales
     $     ,cluster_pdg(0,0,cluster_conf),cluster_ij,iord)
      return
      end


CCCCCCCCCCCCCCC -- MAIN REWEIGHTING ROUTINE -- CCCCCCCCCCCCCCCC
c     loop through the clusterings. Determine
c     1. The lowest QCD cluster scale
c     2. If cluster is a QCD one, apply Sudakov to all lines and
c        determine renormalisation scale
c     3. If QCD line changes, apply Sudakov to all lines
c     Note: Since computation of the Sudakov is somewhat slow (it
c     includes a numerical 1D integral), more efficiently would be, for
c     each line, to compute where it starts and where it ends and only
c     after compute the Sudakov for that line. We might consider this if
c     need be.
c     - The renormalisation scale of the 'central process' will be set
c     to the geometric mean of ALL cluster scales that involve
c     clusterings with exactly 2 QCD partons with scales harder than the
c     hardest clusterings that involve 3 QCD partons. In case the
c     hardest clustering scale is one with 3 QCD partons, that scale is
c     the central hard scale (except if the 2nd to hardest is also one
c     with 3 QCD partons: in that case the hardest clustering scale is
c     reduced to the 2nd to hardest). The value will be returned in
c     qcd_ren_scale(0). Furthermore, in the
c     qcd_ren_scale(1:nqcdrenscale) the renormalisation scales relevant
c     to reweighting are given for nqcdrenscale alpha_S values.
c     - The factorisation scale will be set to the smallest
c     cluster_scale (irrespective if that is an initial scale
c     splitting). In case there is no valid QCD-vertex, its set equal to
c     the renormalisation scale.
      subroutine Reweighting(next,p,nbr,skip_first,cluster_ij,ipdg
     $     ,cluster_pdg,cluster_scales,iord,sudakov,expanded_sudakov
     $     ,nqcdrenscale,qcd_ren_scale,qcd_fac_scale,need_matching)
      implicit none
      integer next,i,j,nbr,cluster_ij(nbr),iord(0:nbr),cluster_pdg(0:2
     $     ,0:2*nbr),type(0:next),nqcdrenscale,nqcdrenscalecentral
     $     ,ipdg(next),first,need_matching(next)
      double precision prev_qcd_scale,hard_qcd_scale,sudakov
     $     ,expanded_sudakov,mass(next),exponent_sukodav,QCDsukakov_exp
     $     ,cluster_scales(0:nbr),qcd_ren_scale(0:nbr),lowest_qcd_scale
     $     ,p(0:3,next),qcd_fac_scale,expanded_exponent_sudakov
     $     ,exponent_sudakov
      logical QCDvertex,QCDchangeline,skip_first,startQCDvertex
      external QCDvertex,QCDchangeline,startQCDvertex

c Determine which particles need clustering and which do not
      call matching_particles(next,nbr,ipdg,cluster_pdg,cluster_ij,iord
     $     ,need_matching)

      nqcdrenscale=0       ! number of alpha-s needing reweighting
      nqcdrenscalecentral=0
      hard_qcd_scale=0d0   ! hardest scale in all clusterings so far.
      prev_qcd_scale=-1d0  ! hardest scale for which Sudakov has been included
      exponent_sudakov=0d0
      expanded_exponent_sudakov=0d0
      qcd_fac_scale=0d0    ! factorisation scale
      qcd_ren_scale(0)=1d0 ! renormalisation scale for 'central process'
      if (skip_first) then ! If .true., skip the first 'startQCDvertex':
         first=0           ! useful for FxFx/MINLO real emission.
      else
         first=-1
      endif
      call fill_type(next,ipdg,type,mass)
      do i=1,nbr                ! cluster all the way to 2->1 process
         if (QCDvertex(i,nbr,cluster_pdg,iord))then
c The vertex is such that it is a QCD clustering, which means a clustering with 3 QCD partons
            if (cluster_scales(i).gt.hard_qcd_scale) then
               hard_qcd_scale=cluster_scales(i)
               ! reset the renormalisation scale for the 'central process'
               nqcdrenscalecentral=-1
               qcd_ren_scale(0)=hard_qcd_scale
            endif
            if (nqcdrenscale.eq.0 .and. first.ne.0) then
c This is the first QCD cluster. Hence, it determines the lowest QCD
c scale that enters all Sudakovs. No Sudakov reweighting required so
c far. Just make sure that we have a valid QCD starting vertex.
               if (startQCDvertex(i,first,cluster_ij(i),nbr,cluster_pdg
     $              ,iord,next,need_matching)) then
                  lowest_qcd_scale=cluster_scales(i)
                  nqcdrenscale=nqcdrenscale+1
                  qcd_ren_scale(nqcdrenscale)=cluster_scales(i)
                  prev_qcd_scale=cluster_scales(i)
                  qcd_fac_scale=cluster_scales(i)
               endif
            elseif (nqcdrenscale.eq.0 .and. first.eq.0) then
c Special case for real-emission FxFx: need to skip the first clustering.
               if (startQCDvertex(i,first,cluster_ij(i),nbr,cluster_pdg
     $              ,iord,next,need_matching)) then
                  first=cluster_ij(i)
               endif
            else
c New QCD vertex found. All lines require Sudakov veto from this scale
c down to the scale of the previous QCD vertex.
               nqcdrenscale=nqcdrenscale+1
               qcd_ren_scale(nqcdrenscale)=hard_qcd_scale
               if (hard_qcd_scale.gt.prev_qcd_scale) then
                  call QCDsudakov(lowest_qcd_scale,hard_qcd_scale
     $                 ,prev_qcd_scale,next,type,mass,exponent_sudakov
     $                 ,expanded_exponent_sudakov)
                  prev_qcd_scale=hard_qcd_scale
               endif
            endif
         elseif(QCDchangeline(i,nbr,cluster_pdg,iord)) then
c The vertex is not a QCD clustering, but it changes a QCD line. E.g.,
c q->Zq, t->Wb, or H->gg. Hence, in case there was already a qcd-cluster
c with a scale below the current scale, apply Sudakov to all QCD
c particles. Do not add this vertex to the qcd_ren_scale, but do update
c prev_qcd_scale, since all Sudakovs up to that scale have been applied.
            if (cluster_scales(i).gt.hard_qcd_scale) then
               hard_qcd_scale=cluster_scales(i)
               ! update the renormalisation scale for the 'central process'
               if (nqcdrenscalecentral.ge.0) then
                  qcd_ren_scale(0)=qcd_ren_scale(0)*hard_qcd_scale
                  nqcdrenscalecentral=nqcdrenscalecentral+1
               elseif (nqcdrenscalecentral.eq.-1) then
                  qcd_ren_scale(0)=hard_qcd_scale
                  nqcdrenscalecentral=1
               endif
            endif
            if (nqcdrenscale.gt.0 .and.
     &           hard_qcd_scale.gt.prev_qcd_scale) then
               call QCDsudakov(lowest_qcd_scale,hard_qcd_scale
     $              ,prev_qcd_scale,next,type,mass,exponent_sudakov
     $              ,expanded_exponent_sudakov)
               prev_qcd_scale=hard_qcd_scale
            endif
         endif
c Remove the daughters and add the mother to the list of QCD lines.
         call update_type(next,i,nbr,cluster_pdg,iord,type
     $        ,mass)
      enddo
c Now we have included all up to the final 2->1 process. In case this is
c a QCD cluster, include the final clustering scale here [hard scale is
c given by cluster_scales(0)]. It will never need alpha_s-reweighting,
c since its scale is equal to the hardest scale by construction.
      if (QCDvertex(0,nbr,cluster_pdg,iord)) then
         if (cluster_scales(0).gt.hard_qcd_scale) then
            hard_qcd_scale=cluster_scales(0)
            ! update the renormalisation scale for the 'central process'
            qcd_ren_scale(0)=hard_qcd_scale
            nqcdrenscalecentral=1
         endif
         if (nqcdrenscale.gt.0) then
            if (hard_qcd_scale.gt.prev_qcd_scale) then
               call QCDsudakov(lowest_qcd_scale,hard_qcd_scale
     $              ,prev_qcd_scale,next,type,mass,exponent_sudakov
     $              ,expanded_exponent_sudakov)
            endif
         endif
      elseif(QCDchangeline(0,nbr,cluster_pdg,iord)) then
         if (cluster_scales(0).gt.hard_qcd_scale) then
            hard_qcd_scale=cluster_scales(0)
            ! update the renormalisation scale for the 'central process'
            if (nqcdrenscalecentral.ge.0) then
               qcd_ren_scale(0)=qcd_ren_scale(0)*hard_qcd_scale
               nqcdrenscalecentral=nqcdrenscalecentral+1
            elseif (nqcdrenscalecentral.eq.-1) then
               qcd_ren_scale(0)=hard_qcd_scale
               nqcdrenscalecentral=1
            endif
         endif
         if (nqcdrenscale.gt.0) then
            if (hard_qcd_scale.gt.prev_qcd_scale) then
               call QCDsudakov(lowest_qcd_scale,hard_qcd_scale
     $              ,prev_qcd_scale,next,type,mass,exponent_sudakov
     $              ,expanded_exponent_sudakov)
            endif
         endif
      endif
c Compute the Sudakov by exponentiation
      sudakov=exp(exponent_sudakov)
      expanded_sudakov=expanded_exponent_sudakov
c Set the renormalisation scale for the central process (and update the
c factorisation scale if need be)
      if (nqcdrenscalecentral.gt.1) then
         qcd_ren_scale(0)=qcd_ren_scale(0)**
     &        (1d0/dble(nqcdrenscalecentral))
      elseif(nqcdrenscalecentral.eq.0) then ! no (relevant) QCD clusterings
         qcd_ren_scale(0)=max(hard_qcd_scale,cluster_scales(0))
      endif
      if (qcd_fac_scale.eq.0) then
         qcd_fac_scale=qcd_ren_scale(0)
      endif

      return
      end


      
CCCCCCCCCCCCCCC -- INTERNAL SUBROUTINES AND FUNCTIONS -- CCCCCCCCCCCCCCC
      subroutine reset_valid_confs(nconf,nvalid,valid_conf)
c Sets all configurations as valid configurations.
      implicit none
      integer iconf,nvalid,nconf
      logical valid_conf(nconf)
      do iconf=1,nconf
         valid_conf(iconf)=.true.
      enddo
      nvalid=nconf
      return
      end

      subroutine limit_cluster_iconfig(nconf,iconfig,nvalid,valid_conf)
c Sets all configurations as invalid, except the one corresponding to
c iconfig
      implicit none
      integer nconf,iconf,iconfig,nvalid
      logical valid_conf(nconf)
      nvalid=0
      do iconf=1,nconf
         if (iconf.eq.iconfig) then
            if (.not. valid_conf(iconf)) then
               write (*,*) 'iconfig is not a valid configuration',iconf
     $              ,nvalid
               stop 1
            endif
            nvalid=1
         else
            valid_conf(iconf)=.false.
         endif
      enddo
      if (nvalid.ne.1) then
         write (*,*) 'iconfig not found in list',nvalid,iconfig,nconf
         stop 1
      endif
      end
      
      subroutine IsBreitWigner(next,nbr,p,itree,prwidth,prmass,ipdg
     $     ,sprop,iBWlist)
c Loop over all the s-channel propagators of the current configuration
c (typically associated with the integration channel) and checks which
c resonances are close to their mass shell. If there are, must only
c cluster according to diagrams/topologies that have that s-channel
c particle as well.
      implicit none
      include 'run.inc'         ! contains 'bwcutoff'
      integer i,j,next,nbr,itree(2,-(nbr+1):-1),idenpart,iBWlist(2
     $     ,0:nbr),ipdg(next),sprop(-nbr:-1),icl(-nbr:next),nbw,ida(2)
      double precision p(0:3,next),prwidth(-nbr:-1),prmass(-nbr:-1)
     $     ,xp(0:3,-nbr:next),mass(-nbr:-1),dot
      logical onBW(nbr),onshell
      external dot
      do i=1,next
         do j=0,3
            xp(j,i)=p(j,i)
         enddo
      enddo
      do i=-1,-nbr,-1
         onbw(-i)=.false.
      enddo
c Loop over all propagators in configuration
      do i=-1,-nbr,-1
c Skip the t-channels
         if (itree(1,i).eq.1 .or. itree(2,i).eq.1 .or.
     &       itree(1,i).eq.2 .or. itree(2,i).eq.2 ) exit
c Momenta of s-channel:
         do j=0,3
            xp(j,i) = xp(j,itree(1,i))+xp(j,itree(2,i))
         enddo
c Possible resonance
         if (prwidth(i) .gt. 0d0) then
            mass(i) = sqrt(dot(xp(0,i),xp(0,i)))
            onshell = abs(mass(i)-prmass(i)).lt.bwcutoff*prwidth(i)
c Close to the on-shell mass.
            if(onshell)then
               OnBW(-i) = .true.
c If mother and daughter have the same ID, remove one of them
               idenpart=0
               do j=1,2
                  ida(j)=itree(j,i)
                  if(ida(j).lt.0) then
                     if (sprop(i).eq.sprop(ida(j))) then
                        idenpart=ida(j) ! mother and daugher have same ID
                     endif
                  elseif (ida(j).gt.0) then
                     if (sprop(i).eq.ipdg(ida(j))) then
                        idenpart=ida(j) ! mother and daugher have same ID
                     endif
                  endif
               enddo
c     Always remove if daughter is final-state (and identical)
               if(idenpart.gt.0) then
                  OnBW(-i)=.false.
c     Else remove either this resonance or daughter, whichever is closer
c     to mass shell
               elseif(idenpart.lt.0 .and. abs(mass(i)-prmass(i)).gt.
     $                 abs(mass(idenpart)-prmass(i))) then
                  OnBW(-i)=.false.         ! mother off-shell
               elseif(idenpart.lt.0) then
                  OnBW(-idenpart)=.false.  ! daughter off-shell
               endif
            endif
         endif
      enddo
c Found all the on-shell BWs. Now fill the ibwlist with the relevant
c information.
      nbw=0
      do i=1,next
        icl(i)=ishft(1,i-1)
      enddo
      do i=-1,-nbr,-1
         icl(i)=icl(itree(1,i))+icl(itree(2,i))
         if(OnBW(-i))then
            nbw=nbw+1
            ibwlist(1,nbw)=icl(i)   ! binary coding
            ibwlist(2,nbw)=sprop(i) ! pdg-code
         endif
      enddo
      ibwlist(1,0)=nbw          ! number of on-shell BWs
      return
      end

      
      subroutine remove_confs_BW(nconf,nbr,nvalid,valid_conf,iBWlist
     $     ,cluster_list,cluster_pdg)
c Given the required s-channels in iBWlist for this phase-space point,
c it loops over all configurations in the cluster_list and sets all
c configurations to invalid that do not have the required s-channel
c propagators.
      implicit none
      integer iconf,ibw,nbw,nconf,nbr,nvalid,ibwlist(2,0:nbr),ipdg,ibr
     $     ,cluster_list(2*nbr,nconf),cluster_pdg(0:2,0:2*nbr,nconf),icl
      logical valid_conf(nconf),in_conf
      nbw=ibwlist(1,0) ! number of BWs that are on-shell
      if (nbw.eq.0) then
         ! no BWs to be considered. All configurations remain valid
         return
      endif
      do ibw=1,nbw
         icl=ibwlist(1,ibw)     ! binary label corresponding to s-channel
         ipdg=ibwlist(2,ibw)    ! PDG-code of the s-channel
         do iconf=1,nconf
            if (.not.valid_conf(iconf)) cycle
            in_conf=.false.
            ! Loop over all branchings in the cluster_list and check if
            ! one corresponds to the s-channel we are looking for. No
            ! need to let the loop go to 2*nbr, since the t-channel
            ! reversing plays no role.
            do ibr=1,nbr
               if (cluster_list(ibr,iconf) .eq.icl .and.
     &             cluster_pdg(0,ibr,iconf).eq.ipdg) then
                  in_conf=.true.
                  exit
               endif
            enddo
            if (.not.in_conf) then
c s-channel is not in iconf. Set it as a invalid configuration
               valid_conf(iconf)=.false.
               nvalid=nvalid-1
            endif
         enddo
         if (nvalid.le.0) then
            write (*,*) 'No valid configurations with BW',nvalid,ibw,nbw
            stop 1
         endif
      enddo
      return
      end
      
      subroutine cluster_one_step(next,p,imap,nbr,nconf,valid_conf
     $     ,cluster_list,iBWlist,iwin,jwin,win_id,min_scale,is_bw
     $     ,cluster_type,particle_type)
c Finds the pair of particles with the smallest clustering scale and
c returns that pair (and the scale) in 'iwin', 'jwin', 'win_id' and
c 'min_scale'. Any clustering that is possible among the diagram
c topologies is considered (irrespective of it being a QCD or EW (or
c anything else) cluster).
      implicit none
      integer next,imap(next),iwin,jwin,id_ij,win_id,nbr,nconf
     $     ,cluster_list(2*nbr,nconf),i,j,iBWlist(2,0:nbr)
     $     ,cluster_type(maskr(next)),particle_type(next),cl(0:2)
      double precision p(0:4,next),cluster_scale,min_scale,scale
      logical in_list,valid_conf(nconf),is_bw
      external in_list,cluster_scale
      iwin=-1 ! set iwin to -1, so that we can track if a single valid
              ! clustering is found
      min_scale=99d99
      do i=3,next ! one of the particles is always a final-state particle
         do j=1,i-1 ! can be either initial state or final state.
            id_ij=imap(i)+imap(j)
            ! Check if current clustering is in the list of still-valid
            ! diagram topologies. If not, simply go to the next possible
            ! clustering
            if (.not.in_list(id_ij,nbr,nconf,valid_conf,cluster_list))
     $           cycle
            cl(0)=cluster_type(id_ij)
            cl(1)=particle_type(i)
            cl(2)=particle_type(j)
            scale=cluster_scale(iBWlist,nbr,j,id_ij,p(0,i),p(0,j),cl
     $           ,is_bw)
            if (scale.lt.min_scale) then
               min_scale=scale
               iwin=i
               jwin=j
               win_id=id_ij
            endif
         enddo
      enddo
      if (iwin.eq.-1) then
         write (*,*) 'No valid clustering found',iwin,next
         do i=1,nconf
            if (valid_conf(i)) write (*,*) i,valid_conf(i)
         enddo
      endif
      return
      end


      subroutine update_valid_confs(id_ij,nconf,nbr,nvalid,valid_conf
     $     ,cluster_list)
c Updates the list of valid configurations based on if id_ij is part of
c that list.
      implicit none
      integer id_ij,iconf,ibr,nconf,nbr,nvalid,cluster_list(2*nbr,nconf)
      logical in_conf,valid_conf(nconf)
c Loop over all diagram topologies
      do iconf=1,nconf
c if current topology was already invalid, go to the next         
         if (.not. valid_conf(iconf)) cycle
c Check if the id_ij is in the cluster_list of current configuration
         in_conf=.false.
         do ibr=1,nbr*2
            if (cluster_list(ibr,iconf).eq.id_ij) then
               in_conf=.true.
               exit
            endif
         enddo
c If current branching is not in the list, remove that configuration as
c a valid cluster configuration/topology.
         if (.not. in_conf) then
            valid_conf(iconf)=.false.
            nvalid=nvalid-1
         endif
      enddo
      if (nvalid.le.0) then
         write (*,*) 'No valid configurations',nvalid
         stop 1
      endif
      return
      end

      subroutine update_momenta(nleft,pcl,iwin,jwin,p_inter,is_bw
     $     ,particle_type,cl_type)
c Updates the 'pcl' momenta list by combining particles iwin and
c jwin. Also updates the particle_type of the combined particle.
      implicit none
      integer nleft,iwin,jwin,i,j,k,particle_type(nleft),cl_type
      double precision pcl(0:4,nleft),p(0:3),nr(0:3),nn2,ct,st
     $     ,pcmsp(0:3),p_inter(0:4,0:2),pi(0:3),dot,pz(0:3)
      logical is_bw
      data (pz(i),i=0,3)/1d0,0d0,0d0,1d0/
      external dot
      do i=0,4
         p_inter(i,1)=pcl(i,iwin)
         p_inter(i,2)=pcl(i,jwin)
      enddo
      if (jwin.le.2) then
c initial state clustering
         do i=0,3
            pcl(i,jwin)=pcl(i,jwin)-pcl(i,iwin)
            pcmsp(i)=-pcl(i,jwin)-pcl(i,3-jwin)
         enddo
         pcmsp(0)=-pcmsp(0)
         if( (pcl(4,jwin).gt.0.or.pcl(4,iwin).gt.0) .and. 
     $       .not.(pcl(4,jwin).gt.0.and.pcl(4,iwin).gt.0)) then
            pcl(4,jwin)=max(pcl(4,jwin),pcl(4,iwin))
         else
            pcl(4,jwin)=0
         endif
         ! mother momenta
         do i=0,4
            p_inter(i,0)=pcl(i,jwin)
         enddo
         if (pcmsp(0)**2-pcmsp(1)**2-pcmsp(2)**2-pcmsp(3)**2.gt.100d0)
     $        then ! prevent too extreme boost
            call boostx(pcl(0,jwin),pcmsp,p)
            call constr(p,pz,nr,nn2,ct,st)
            do j=1,nleft
               if (j.eq.iwin) cycle
               call boostx(pcl(0,j),pcmsp,p)
               call rotate(p,pi,nr,nn2,ct,st,1)
               if (j.lt.iwin) then
                  do k=0,3
                     pcl(k,j)=pi(k)
                  enddo
               else
                  do k=0,3
                     pcl(k,j-1)=pi(k)
                  enddo
                  pcl(4,j-1)=pcl(4,j)
               endif
            enddo
         else
            do i=1,nleft-1
               if (i.ge.iwin) then
                  do j=0,4
                     pcl(j,i)=pcl(j,i+1)
                  enddo
               endif
            enddo
         endif
      else
c final state clustering
         do i=1,nleft-1
            if (i.eq.jwin) then
               do j=0,3
                  pcl(j,jwin)=pcl(j,iwin)+pcl(j,jwin)
               enddo
               if (is_bw) then
                  ! mass is invariant mass for resonance decay
                  pcl(4,jwin)=sqrt(dot(pcl(0,jwin),pcl(0,jwin)))
               else
                  ! the heaviest of the two daughter masses
                  pcl(4,jwin)=max(pcl(4,jwin),pcl(4,iwin))
               endif
            elseif (i.ge.iwin) then
               do j=0,4
                  pcl(j,i)=pcl(j,i+1)
               enddo
            endif
         enddo
         ! mother momenta
         do i=0,4
            p_inter(i,0)=pcl(i,jwin)
         enddo
      endif
      do i=jwin,nleft-1
         if (i.eq.jwin) then
            particle_type(i)=cl_type
         elseif(i.ge.iwin) then
            particle_type(i)=particle_type(i+1)
         endif
      enddo
      end
      
      subroutine update_imap(nleft,imap,iwin,jwin,win_id)
c Updates the mapping list between the external particle labels to the
c binary labeling when particles iwin and jwin are combined to win_id;
c the latter being treated as a new external particle.
      implicit none
      integer i,nleft,imap(nleft),iwin,jwin,win_id
      do i=1,nleft-1
         if (i.eq.jwin) then
            imap(i)=win_id
         elseif (i.ge.iwin) then
            imap(i)=imap(i+1)
         endif
      enddo
      end

      subroutine link_clustering_to_iforest(nbr,cluster_ij,cluster_list
     $     ,iord)
c Links the clustering id (i.e., the binary number) to the propagator
c label used in the iforest information (the latter as used in
c cluster_pdg). This allows for easy access to PDG numbers etc.
      implicit none
      integer i,ibr,nbr,cluster_ij(nbr),cluster_list(2*nbr),iord(0:nbr)
      do i=1,nbr
         do ibr=1,2*nbr
            if (cluster_ij(i).eq.cluster_list(ibr)) then
               iord(i)=ibr
               exit
            endif
         enddo
      enddo
      iord(0)=0
      end

      subroutine update_cluster_scales(nbr,p_inter,cluster_scales
     $     ,cluster_pdg,cluster_ij,iord)
c Updates the clustering scale to be abs(2*p1.p2) for clusterings that
c involve exaclty 2 QCD partons. This will be the correct hard scale in
c the Sudakov Form Factors (for both massless and massive quarks and
c gluons).
      implicit none
      integer i,j,k,cij,nbr,cluster_pdg(0:2,0:2*nbr),cluster_ij(nbr)
     $     ,iord(0:nbr),next,get_color,iqcd(0:2)
      double precision cluster_scales(0:nbr),dot,p_inter(0:4,0:2,0:nbr)
     $     ,djb_clus
      logical QCDchangeline,QCDvertex
      external dot,get_color,djb_clus,QCDchangeline,QCDvertex
      do i=0,nbr
         if (QCDchangeline(i,nbr,cluster_pdg,iord)) then
            iqcd(0)=0
            do j=0,2
               if (abs(get_color(cluster_pdg(j,iord(i)))).gt.1) then
                  iqcd(0)=iqcd(0)+1
                  iqcd(iqcd(0))=j
               endif
            enddo
            cluster_scales(i)=sqrt(2d0*abs(dot(p_inter(0,iqcd(1),i)
     $                                        ,p_inter(0,iqcd(2),i))))
         endif
      enddo
c Treat here the special case where the final 2->2 process is a pure
c s-channel QCD process. In that case set the last and next-to-last
c clustering scale to the average transverse masses of the (combined)
c particles.
      cij=cluster_ij(nbr)
      if (QCDvertex(nbr,nbr,cluster_pdg,iord) .and.
     $     (.not.(btest(cij,0) .or. btest(cij,1)))) then
         ! Clustered 2->2 process is s-channel with 2 final state QCD
         ! particles. Check if the two initial state particles are also
         ! QCD. If so, update the cluster_scales to be the average m_T^2
         ! of the two QCD final state particles
         if (QCDvertex(0,nbr,cluster_pdg,iord)) then
            cluster_scales(nbr)=(djb_clus(p_inter(0,1,nbr))*
     &                           djb_clus(p_inter(0,2,nbr)))**0.25d0
            cluster_scales(0)=cluster_scales(nbr)
         endif
      endif
      return
      end

      subroutine set_cluster_pdg_2_1_process(next,nbr,ipdg,cluster_pdg
     $     ,cluster_ij,iord)
c This set the cluster_pdg(0:2,0) to the PDGs of the final 2->1
c process. The two daughters [cluster_pdg(1:2,0)] are the two incoming
c particles, and the mother [cluster_pdg(0,0)] is the outgoing one.
      implicit none
      integer nbr,next,ipdg(next),cluster_pdg(0:2,0:2*nbr),iord(0:nbr)
     $     ,cluster_ij(nbr),cij,i
      ! In case there is no clustering with incoming lines 1 or 2, start
      ! by setting the cluster_pdg to the PDGs of the incoming particles
      cluster_pdg(1,0)=ipdg(1)
      cluster_pdg(2,0)=ipdg(2)
      cluster_pdg(0,0)=0 ! bogus value for mother. Will be updated below
      if (nbr.eq.0) then
         ! in case we have a 2->1 process, the assignment is trivial
         cluster_pdg(0,0)=ipdg(3)
         return
      endif
      ! Loop over all the clusterings. Everytime there is a cluster with
      ! any of the two incoming lines, update the cluster_pdg(1:2,0)
      do i=1,nbr
         cij=cluster_ij(i)
         if (btest(cij,0)) then ! initial state clustering with incoming line 1
            cluster_pdg(1,0)=cluster_pdg(0,iord(i))
         endif
         if (btest(cij,1)) then ! initial state clustering with incoming line 2
            cluster_pdg(2,0)=cluster_pdg(0,iord(i))
         endif
      enddo
      cij=cluster_ij(nbr)
      if (btest(cij,0) .or. btest(cij,1)) then
         ! last clustering is initial state cluster. Hence, mother is
         ! given by 2nd daughter in reversed t-channel
         if (iord(nbr).gt.nbr) then
            cluster_pdg(0,0)=cluster_pdg(2,iord(nbr)-nbr)
         else
            cluster_pdg(0,0)=cluster_pdg(2,nbr+iord(nbr))
         endif
      else
         ! last clustering is final state clustering. Hence mother is
         ! identical to mother in last clustering
         cluster_pdg(0,0)=cluster_pdg(0,iord(nbr))
      endif
      if (cluster_pdg(0,0).eq.0) then
         write (*,*) 'PDG code for mother in final 2->1 process is'/
     $        /' not set',cluster_pdg(:,0)
         stop 1
      endif
      end
      
      
      logical function in_list(id_ij,nbr,nconf,valid_conf,cluster_list)
c Checks if the id_ij particle is somewhere in the list of valid
c configurations.
      implicit none
      integer id_ij,iconf,ibr,nconf,nbr,cluster_list(2*nbr,nconf)
      logical valid_conf(nconf)
      in_list=.false.
      do iconf=1,nconf
         if (.not.valid_conf(iconf)) cycle
         do ibr=1,nbr*2
            if (cluster_list(ibr,iconf).eq.id_ij) then
               in_list=.true.
               return
            endif
         enddo
      enddo
      return
      end

      subroutine set_cluster_conf(nconf,nvalid,valid_conf,iconfig
     $     ,cluster_conf)
c Given the final valid_conf list, returns the "best" cluster_conf
      implicit none
      integer nconf,nvalid,iconfig,cluster_conf,ic,ifind,iconf_counter
     $     ,iconf
      logical valid_conf(nconf)
      data iconf_counter/149/ ! some random value.
      save iconf_counter
      if (nvalid.eq.1) then
c     There is only one possible valid diagram. Hence, find that diagram
c     and set cluster_conf equal to it.
         do iconf=1,nconf
            if (valid_conf(iconf)) then
               cluster_conf=iconf
               return
            endif
         enddo
      elseif (nvalid.le.0) then
         write (*,*)"no valid diagrams",nvalid
         stop 1
      else    ! more than one valid configuration
         if (valid_conf(iconfig)) then
c     integration channel among valid configurations. Choose that one
            cluster_conf=iconfig
            return
         else
c     pick one at "random"
            iconf_counter=iconf_counter+1
            ifind=mod(iconf_counter,nvalid)+1
            ic=0
            do iconf=1,nconf
               if (valid_conf(iconf)) then
                  ic=ic+1
                  if (ic.eq.ifind) then
                     cluster_conf=iconf
                     return
                  endif
               endif
            enddo
         endif
      endif
      end
      
      
      double precision function cluster_scale(iBWlist,nbr,j,id_ij,pi,pj
     $     ,cl,is_bw)
c Determines the cluster scale for the clustering of momenta pi and pj
      implicit none
      integer nbr,i,j,id_ij,iBWlist(2,0:nbr),cl(0:2),itype
      double precision pi(0:4),pj(0:4),sumdot,dj_clus,one_plus_tiny
     $     ,djb_clus,dot
      parameter (one_plus_tiny=1.000001d0)
      logical is_bw
      external sumdot,dj_clus,djb_clus,dot
      if (j.le.2) then
c     initial state clustering
         cluster_scale=sqrt(djb_clus(pi))
         ! prefer clustering when outgoing is in the direction of incoming
         if(sign(1d0,pi(3)).ne.sign(1d0,pj(3)))
     $        cluster_scale=cluster_scale*one_plus_tiny
      else
c     final state clustering
         is_bw=.false.
         do i=1,iBWlist(1,0)
            if (iBWlist(1,i).eq.id_ij) then
               is_bw=.true.
               exit
            endif
         enddo
         if (is_bw) then
            ! for decaying resonance, the scale is the mass of the resonance
            cluster_scale=sqrt(max(sumdot(pi,pj,1d0),0d0))
         else
            ! Check that it is unique (might need fixing)
            do i=0,2
               if (popcnt(cl(i)).gt.1)  then
                  write (*,*) 'more than one possibility for clustering'
     $                 ,i,cl(i)
                  stop 1
               endif
            enddo
            call get_clustering_type(cl,itype)
c Different scale depending on itype:
            if (itype.eq.1 .or. itype.eq.6 .or.itype.eq.2 .or.
     $          itype.eq.3 .or. itype.eq.7) then
               cluster_scale=sqrt(dj_clus(pi,pj))
            elseif (itype.eq.4) then
               cluster_scale=sqrt(abs(dot(pj,(pi+pj))))/2d0
            elseif (itype.eq.5) then
               cluster_scale=sqrt(abs(dot(pi,(pi+pj))))/2d0
            elseif (itype.eq.8) then
               cluster_scale=sqrt(max(sumdot(pi,pj,1d0),0d0))
            endif
         endif
      endif
      end

      subroutine get_clustering_type(cl,itype)
      ! This sets the type of clustering based on the masses of the
      ! particles involved. If need be, this can be extended to include
      ! colour information: the latter is already available in the cl()
      ! array. Note that internal colour singlets are always treated as
      ! massive particles in the cl() array (set by the
      ! set_particle_type() subroutine).
      implicit none
      integer cl(0:2),itype
      if ( (btest(cl(0),0).or.btest(cl(0),1).or.btest(cl(0),3)) .and.
     $     (btest(cl(1),0).or.btest(cl(1),1).or.btest(cl(1),3)) .and.
     $     (btest(cl(2),0).or.btest(cl(2),1).or.btest(cl(2),3))) then 
         ! three massless particles
         itype=1
      elseif ((btest(cl(0),2).or.btest(cl(0),4)) .and.
     $        (btest(cl(1),0).or.btest(cl(1),1).or.btest(cl(1),3)) .and.
     $        (btest(cl(2),2).or.btest(cl(2),4))) then
         ! massive emitting a massless particle 1
         itype=2
      elseif ((btest(cl(0),2).or.btest(cl(0),4)) .and.
     $        (btest(cl(1),2).or.btest(cl(1),4)) .and.
     $        (btest(cl(2),0).or.btest(cl(2),1).or.btest(cl(2),3))) then
         ! massive emitting a massless particle 2
         itype=3
      elseif ((btest(cl(0),0).or.btest(cl(0),1).or.btest(cl(0),3)) .and.
     $        (btest(cl(1),2).or.btest(cl(1),4)) .and.
     $        (btest(cl(2),0).or.btest(cl(2),1).or.btest(cl(2),3))) then
         ! massless emitting a massive particle 1
         itype=4
      elseif ((btest(cl(0),0).or.btest(cl(0),1).or.btest(cl(0),3)) .and.
     $        (btest(cl(1),0).or.btest(cl(1),1).or.btest(cl(1),3)) .and.
     $        (btest(cl(2),2).or.btest(cl(2),4))) then
         ! massless emitting a massive particle 2
         itype=5
      elseif ((btest(cl(0),0).or.btest(cl(0),1).or.btest(cl(0),3)) .and.
     $        (btest(cl(1),2).or.btest(cl(1),4)) .and.
     $        (btest(cl(2),2).or.btest(cl(2),4))) then
         ! massless to two massive particles
         itype=6
      elseif ((btest(cl(0),2).or.btest(cl(0),4)) .and.
     $        (btest(cl(1),2).or.btest(cl(1),4)) .and.
     $        (btest(cl(2),2).or.btest(cl(2),4))) then
         ! Three massive particles
         itype=7
      elseif ((btest(cl(0),2).or.btest(cl(0),4)) .and.
     $        (btest(cl(1),0).or.btest(cl(1),1).or.btest(cl(1),3)) .and.
     $        (btest(cl(2),0).or.btest(cl(2),1).or.btest(cl(2),3))) then
         ! massive decaying into two massless
         itype=8
      else
         write (*,*) 'Unknown clustering type',cl
         stop
      endif
      end
      
      subroutine fill_type(next,ipdg,type,mass)
c Loops over all external particles and sets up 'type' using the
c get_type() subroutine. 'type' will just be a list of what's there as
c external particles; there is no particular order. As particles get
c clustered, the update_type() subroutine will remove the clustered
c particles and add the mother back. 'type' together with 'mass'
c contains all the information needed to determine which Sudakov FF
c should be used.
      implicit none
      integer i,type(0:next),itype,next,ipdg(next)
      double precision mas,mass(next)
      type(0)=0
      do i=1,next
         call get_type(ipdg(i),itype,mas)
         if (itype.gt.0) then ! do not include colour singlets in 'type' list
            type(0)=type(0)+1
            type(type(0))=itype
            mass(type(0))=mas
         endif
      enddo
      end

      
      subroutine get_type(ipdg,itype,imass)
c Defines the type of a particle 'type':
c     0 : colour singlet
c     1 : massless fermion, colour triplet
c     2 : massive fermion, colour triplet
c     3 : massless vector, colour octet
c     4 : massive particle, colour triplet (treated in inf. mass limit)
c     5 : massive particle, colour octet   (treated in inf. mass limit)
c     else : unknown --> give ERROR
c Types 4 and 5 are treated in the infinite mass limit, and are
c therefore independent from the particle spin (only soft
c singularities).
      implicit none
      integer ipdg,itype,get_color,get_spin,icol,ispin
      double precision imass,get_mass_from_id
      external get_color,get_spin,get_mass_from_id
      icol=abs(get_color(ipdg))
      ispin=abs(get_spin(ipdg))
      imass=abs(get_mass_from_id(ipdg))
      if (icol.eq.1) then
         itype=0
      elseif (ispin.eq.2 .and. icol.eq.3 .and. imass.eq.0d0) then
         itype=1
      elseif (ispin.eq.2 .and. icol.eq.3 .and. imass.gt.0d0) then
         itype=2
      elseif (ispin.eq.3 .and. icol.eq.8 .and. imass.eq.0d0) then
         itype=3
      elseif (icol.eq.3 .and. imass.gt.0d0) then
         itype=4
      elseif (icol.eq.8 .and. imass.gt.0d0) then
         itype=5
      else
         write (*,*) 'Cannot compute Sudakovs for '/
     $        /'QCD charged particle',ipdg,':',ispin,icol,imass
         stop 1
      endif
      return
      end


      subroutine update_type(next,iclus,nbr,cluster_pdg,iord,type,mass)
c Updates 'type' (which lists all current QCD-charged particles in
c process in no particular order). It removes the two entries which are
c consistent with the two daughters of the clustering and adds the
c mother.
      implicit none
      integer i,j,k,next,nbr,iclus,ipdg(0:2),itype,iord(0:nbr)
     $     ,cluster_pdg(0:2,0:2*nbr),type(0:next)
      double precision mass(next),imass
      logical found
      do i=0,2 ! mother and two daughters
         ipdg(i)=cluster_pdg(i,iord(iclus))
      enddo
      ! first remove the two daughters
      do i=1,2
         call get_type(ipdg(i),itype,imass)
         if (itype.eq.0) cycle ! not a QCD particle: nothing to remove
         found=.false.
         do j=1,type(0)
            if (itype.eq.type(j).and.imass.eq.mass(j)) then
               found=.true.
               ! found daughter 'i'. Remove it from the list
               type(0)=type(0)-1
               do k=j,type(0)
                  type(k)=type(k+1)
                  mass(k)=mass(k+1)
               enddo
               exit
            endif
         enddo
         if (.not.found) then
            write (*,*) 'Daughter type not found in type list',ipdg(i)
     $           ,itype,imass,i
            write (*,*) (type(j),j=1,type(0))
            write (*,*) (mass(j),j=1,type(0))
            stop 1
         endif
      enddo
      ! Add the mother
      call get_type(ipdg(0),itype,imass)
      if (itype.ne.0) then ! QCD particle: add it to the list
         type(0)=type(0)+1
         type(type(0))=itype
         mass(type(0))=imass
      endif
      end

      integer function numberQCDcharged(iclus,nbr,cluster_pdg,iord)
c Returns the number of QCD charged particles in the clustering vertex
c 'cij'.
      implicit none
      integer i,ipart,iclus,get_color,nbr,cluster_pdg(0:2,0:2*nbr)
     $     ,iord(0:nbr)
      external get_color
      numberQCDcharged=0
      do i=0,2
         ipart=cluster_pdg(i,iord(iclus))
         if (abs(get_color(ipart)).gt.1) then
            numberQCDcharged=numberQCDcharged+1
         endif
      enddo
      return
      end

      
      logical function QCDchangeline(iclus,nbr,cluster_pdg,iord)
c Checks if a vertex changes a QCD line. This means that 2 out of the
c three particles are QCD charged.
      implicit none
      integer numberQCDcharged,iclus,nbr,cluster_pdg(0:2,0:2*nbr)
     $     ,iord(0:nbr)
      external numberQCDcharged
      if (numberQCDcharged(iclus,nbr,cluster_pdg,iord).eq.2)
     $     then
         QCDchangeline=.true.
      else
         QCDchangeline=.false.
      endif
      end

      
      logical function QCDvertex(iclus,nbr,cluster_pdg,iord)
c Checks if all three particles involved are QCD particles.
      implicit none
      integer numberQCDcharged,iclus,nbr,cluster_pdg(0:2,0:2*nbr)
     $     ,iord(0:nbr)
      external numberQCDcharged
      if (numberQCDcharged(iclus,nbr,cluster_pdg,iord).eq.3)
     $     then
         QCDvertex=.true.
      else
         QCDvertex=.false.
      endif
      end

      
      logical function startQCDvertex(iclus,first,cij,nbr,cluster_pdg
     $     ,iord,next,need_matching)
c Checks if cluster is associated with a valid starting vertex. For
c this, the cluster must be associated with an IR divergence (if there
c would be no cuts on the clustered partons). Hence, this cluster must
c have 2 external particles with
c     1. At least one final state gluon, and/or,
c     2. A final state massless quark with an initial state gluon or
c        quark (with same flavour), and/or,
c     3. A final state q-qbar pair of massless quarks of the same
c        flavour
c***  We might consider including the case where these two particles are
c***  NOT external particles. E.g., a relatively soft W-boson or photon
c***  emission of a quark line might not render the latter too hard to
c***  be considered as a parton coming from the starting vertex. We do
c***  not consider this currently
      implicit none
      integer iclus,cij,imo,da1,da2,nbr,cluster_pdg(0:2,0:2*nbr)
     $     ,iord(0:nbr),first,pc,next,need_matching(next),nn,i
      logical final_state,IR_cluster
      external IR_cluster
      pc=popcnt(cij) ! number of non-zero bits in cij
      if (pc.gt.3) then
c The number of non-zero bits in cij corresponds to the total number of
c external particles clustered into the cij cluster. We need to have
c exactly 2 since we need to have the two daughters to be external
c particles for a valid starting QCD vertex.
         startQCDvertex=.false.
         return
      elseif (pc.eq.3) then
c Special case for real-emission FxFx, where we skipped the first
c clustering that was a startQCDvertex. Hence, we can have 3 particles
c clustered. Explicitly check that the first cluster is contained in the
c current cluster
         if (iand(cij,first).ne.first) then
            startQCDvertex=.false.
            return
         endif
      elseif (pc.lt.2) then
         write (*,*) 'ERROR less than two external particles'/
     $        /' involved in the cluster',cij,popcnt(cij)
         stop 1
      endif
c If pc=2, then at least one of the final state particles need to have
c the 'need_matching' tag equal to 1. If pc=3, there need to be at least
c two of them. Otherwise this cluster cannot be a startQCDvertex.
      nn=0
      do i=3,next
         if (btest(cij,i-1) .and. need_matching(i).eq.1) then
            nn=nn+1
         endif
      enddo
      if (nn.lt.pc-1) then
         startQCDvertex=.false.
         return
      endif
c Finally, the cluster should be a cluster that could generate an IR
c singularity.
      imo=cluster_pdg(0,iord(iclus))
      da1=cluster_pdg(1,iord(iclus))
      da2=cluster_pdg(2,iord(iclus))
      if (.not.(btest(cij,0) .or. btest(cij,1))) then
         final_state=.true.
      else
         final_state=.false.
      endif
      startQCDvertex=IR_cluster(imo,da1,da2,final_state)
      return
      end

      logical function IR_cluster(imo,da1,da2,final_state)
c Checks if the branching imo->da1+da2 might contain an IR singularity:
c     1. At least one final state gluon, and/or,
c     2. A final state massless quark with an initial state gluon or
c        quark (with same flavour), and/or,
c     3. A final state q-qbar pair of massless quarks of the same
c        flavour
      implicit none
      include 'cuts.inc'        ! includes maxjetflavor
      integer imo,da1,da2
      logical final_state
      double precision get_mass_from_id
      external get_mass_from_id
      IR_cluster=.false.
      if (final_state) then ! final state clustering
         if ( (da1.eq.21 .and. da2.eq.imo) .or.
     &        (da2.eq.21 .and. da1.eq.imo)) then ! X->X+g
            IR_cluster=.true.
            return
         elseif (abs(da1).le.maxjetflavor .and. da1+da2.eq.0 .and.
     &           get_mass_from_id(imo).eq.0d0) then ! X->qqbar (with X massless)
            IR_cluster=.true.
            return
         endif
      else ! initial state clustering. 'da1' is always the initial
           ! state; 'da2' is the final state:  da1 -> imo + da2
         if (da2.eq.21 .and. abs(da1).eq.abs(imo)) then ! X->X+g
            IR_cluster=.true.
            return
         elseif(abs(da2).le.maxjetflavor .and.
     &      ((da1.eq.21  .and. abs(imo).eq.abs(da2)) .or. ! g->q+qbar
     &       (abs(da1).eq.abs(da2) .and. get_mass_from_id(imo).eq.0d0))) then ! q->X+q (with X massless)
            IR_cluster=.true.
            return
         endif
      endif
      end
      

      subroutine matching_particles(next,nbr,ipdg,cluster_pdg
     $     ,cluster_ij,iord,need_matching)
      ! determines which particles need matching (i.e. a correspondence
      ! at the level of shower jet with particle jets a la MLM) and
      ! which need 'anti-matching' (this particle cannot be part of a
      ! jet that is matched to another parton)
      !
      ! The code fills the array:
      !
      ! need_matching= 0 : particle does not require matching (it is
      !                    massive or not colour-charged)
      ! need_matching= 1 : particle requires matching
      ! need_matching=-1 : particle is massless colour-charged, but does
      !                     not require matching (i.e., it is
      !                    'anti-matched')
      !
      ! The algorithm:
      !
      ! 0. All non-QCD particles, and all massive particles do not
      !    require matching.
      ! 1. All final state gluons require matching.
      ! 2. All particles that are part of an s-channel clustering, and
      !    there is a particle among them that does not require
      !    matching: mark all not-yet-labeled particles as NOT requiring
      !    matching.
      ! 3. All particles that are part of an s-channel clustering, and
      !    there is NOT a particle among them that does not require
      !    matching, and the mother is a gluon: mark all not-yet-labeled
      !    particles as require matching.
      ! 4. If the clustering includes an initial-state particle, and
      !    there is NOT a particle among them that does not require
      !    matching and t-channel and initial state particle are
      !    massless: mark all not-yet-labeled particles as requiring
      !    matching.
      !    If, instead, the t-channel or the initial state particle are
      !    massive: mark the not-yet-labeled particles as NOT requiring
      !    matching.
      !
      ! NOTE: this function only works correctly for MAXIMALLY QCD-LIKE
      ! PROCESSES.
      !
      implicit none
      integer next,nbr,cluster_pdg(0:2,0:2*nbr),cluster_ij(nbr),iord(0:nbr)
     $     ,need_matching(next),ipdg(next),i,j,k,cij,imo,id1,id2,ii,l
     $     ,get_color,matching_sum
      logical s_chan,IR_cluster
      double precision get_mass_from_id
      external IR_cluster,get_mass_from_id,get_color
c First loop to assign the matching condition
      do i=3,next
c Color is 1,3,8 for singlet(no QCD),triplet(quark),octet(gluon)
         if (get_color(ipdg(i)).eq.1) then ! non-coloured
            ! no matching needed for non-coloured particles
            need_matching(i)=0
            cycle
         elseif (abs(get_color(ipdg(i))).eq.3) then ! quark/antiquark
            ! massive (triplet coloured) particles do not need matching
            if (get_mass_from_id(ipdg(i)).ne.0d0) then
               need_matching(i)=-1
            else
               need_matching(i)=-99 ! massless quark, undecided yet
            endif
            cycle
         elseif (get_color(ipdg(i)).eq.8) then ! gluon or BSM massive octet
             ! massive (octet coloured) particles do not need matching
            if (get_mass_from_id(ipdg(i)).ne.0d0) then
               need_matching(i)=-1
            else
               need_matching(i)=1 ! ! gluons always need to be matched
            endif 
            cycle
         else
c Stop if a particle passed all the previous statements
            write (*,*) 'Unknown particle, pdgid=',ipdg(i),'color='
     $           ,get_color(ipdg(i))
            stop
         endif
      enddo
c Check and return if everything is already assigned a need_matching -1,0,1
      matching_sum=0
      do l = 3,next
         matching_sum=matching_sum+need_matching(l)
      enddo
      if (matching_sum.gt.-80) then
         return
      endif
c Second loop to assign the matching condition
      do ii=1,nbr+1
         i=mod(ii,nbr+1) ! i=1,2,3,..,nbr,0
         if (i.ne.0) then
            cij=cluster_ij(i)
         else
            cij=maskr(next+1) ! include everything in final 2->1 process
         endif
         s_chan=.not.(btest(cij,0).or.btest(cij,1))
         imo=cluster_pdg(0,iord(i))
         id1=cluster_pdg(1,iord(i))
         id2=cluster_pdg(2,iord(i))
         if (s_chan) then
            ! s-channel. Check points 2 and 3 of the Algorithm
            do j=3,next
               if (.not.btest(cij,j-1)) cycle
               if ( need_matching(j).ne.0 .and.
     &              need_matching(j).ne.-1) cycle
               ! found a particle that does not require matching in the
               ! clustering. Hence, set all that have not yet been set
               ! as requiring anti-matching
               do k=3,next
                  if (.not.btest(cij,k-1)) cycle
                  if (need_matching(k).eq.-99) need_matching(k)=-1
               enddo
c Check and return if everything is already assigned a need_matching -1,0,1
               matching_sum=0
               do l = 3,next
                  matching_sum=matching_sum+need_matching(l)
               enddo
               if (matching_sum.gt.-80) then
                  return
               endif
            enddo
            if (imo.eq.21 .and. IR_cluster(imo,id1,id2,.true.)) then
               ! massless g->qqbar splitting. Hence, set all that have
               ! not yet been set as requiring matching
               do k=3,next
                  if (.not.btest(cij,k-1)) cycle
                  if (need_matching(k).eq.-99) need_matching(k)=1
               enddo
c Check and return if everything is already assigned a need_matching -1,0,1
               matching_sum=0
               do l = 3,next
                  matching_sum=matching_sum+need_matching(l)
               enddo
               if (matching_sum.gt.-80) then
                  return
               endif
            endif
         else ! i.e., a clustering with initial state t-channel. Check
              ! points 4 of the algorithm
            if ( get_mass_from_id(imo).eq.0d0 .and.
     &           get_mass_from_id(id1).eq.0d0) then
               do j=3,next
                  if (.not.btest(cij,j-1)) cycle
                  if ( need_matching(j).ne.0 .and.
     &                 need_matching(j).ne.-1) cycle
                  ! found a particle that does not require matching in
                  ! the clustering. Hence, set all that have not yet
                  ! been set as requiring anti-matching
                  do k=3,next
                     if (.not.btest(cij,k-1)) cycle
                     if (need_matching(k).eq.-99) need_matching(k)=-1
                  enddo
c Check and return if everything is already assigned a need_matching -1,0,1
                  matching_sum=0
                  do l = 3,next
                     matching_sum=matching_sum+need_matching(l)
                  enddo
                  if (matching_sum.gt.-80) then
                     return
                  endif
               enddo
               ! if particles were not set, there was no particle that
               ! did not require matching. Hence, all non-set particles
               !  here require matching.
               do k=3,next
                  if (.not.btest(cij,k-1)) cycle
                  if (need_matching(k).eq.-99) need_matching(k)=1
               enddo
c Check and return if everything is already assigned a need_matching -1,0,1
               matching_sum=0
               do l = 3,next
                  matching_sum=matching_sum+need_matching(l)
               enddo
               if (matching_sum.gt.-80) then
                  return
               endif
            else
               ! mother, or initial state is not massless. Hence, set
               ! all that have not yet been set as not requiring
               ! matching
               do k=3,next
                  if (.not.btest(cij,k-1)) cycle
                  if (need_matching(k).eq.-99) need_matching(k)=-1
               enddo
c Check and return if everything is already assigned a need_matching -1,0,1
               matching_sum=0
               do l = 3,next
                  matching_sum=matching_sum+need_matching(l)
               enddo
               if (matching_sum.gt.-80) then
                  return
               endif
            endif
         endif
      enddo
c Uncomment to print all the particle info to check
c      write (*,*) 'pdgid=',ipdg(1),'color=',get_color(ipdg(1))
c      write (*,*) 'pdgid=',ipdg(2),'color=',get_color(ipdg(2))
c      write (*,*) '---final states'
c      do i = 3,next
c        write (*,*) 'pdgid=',ipdg(i),'need matching=',need_matching(i)
c      enddo
      end

CCCCCCCCCCCCCCC -- SUDAKOV FUNCTIONS -- CCCCCCCCCCCCCCCC

      subroutine QCDsudakov(q0,q2,q1,next,type,mass,QCDsudakov_exp
     $     ,expanded_QCDsudakov_exp)
c Wrapper function for computing the sudakov for the particles listed in
c itype(). It checks for identical type and mass, and makes sure to use
c cached ones if already computed. It adds the results to the
c QCDsudakov_exp and expanded_QCDsudakov_exp (the latter containing the
c strict O(alpha_s) expansion of the former, needed to subtract double
c counting with NLO corrections to the main process).
      implicit none
      integer i,j,next,type(0:next)
      double precision q0,q2,q1,mass(next),tmp1(next),tmp2(next)
     $     ,expanded_QCDsudakov_exp,QCDsudakov_exp,expanded_sudakov_exp
     $     ,sudakov_exp,q1tmp
      logical found
      external sudakov_exp,expanded_sudakov_exp
      do i=1,type(0)
         found=.false.
         ! use cached one if already computed:
         do j=1,i-1
            if (type(j).eq.type(i) .and. mass(j).eq.mass(i)) then
               tmp1(i)=tmp1(j)
               tmp2(i)=tmp2(j)
               found=.true.
               exit
            endif
         enddo
c$$$         q1tmp=q1
         q1tmp=max(q1,mass(i))
         if (.not. found) then
            ! not yet computed. Do it now:
            if (q2.gt.q1tmp .and. q1tmp.gt.q0) then
               tmp1(i)=sudakov_exp(q0,q2,type(i),mass(i))
     $                -sudakov_exp(q0,q1tmp,type(i),mass(i))
               tmp2(i)=expanded_sudakov_exp(q0,q2,type(i),mass(i))
     $                -expanded_sudakov_exp(q0,q1tmp,type(i),mass(i))
            elseif(q2.gt.q1tmp .and. q1tmp.eq.q0) then
               tmp1(i)=sudakov_exp(q0,q2,type(i),mass(i))
               tmp2(i)=expanded_sudakov_exp(q0,q2,type(i),mass(i))
            else
               tmp1(i)=0d0
               tmp2(i)=0d0
            endif
         endif
         ! Add this sudakov to the result:
         QCDsudakov_exp=QCDsudakov_exp+tmp1(i)
         expanded_QCDsudakov_exp=expanded_QCDsudakov_exp+tmp2(i)
      enddo
      return
      end


      double precision function sudakov_exp(q0,Q11,itype,imass)
c Wrapper code for computation of the exponent of the Sudakov Form
c Factor numerically using Gaussian integration. It also saves the last
c 'nc' computed values, which is helpful when dealing, e.g., with many
c real-emission processes for which many Sudakovs might be identical.
      implicit none
c Arguments
      integer itype
      double precision q0,Q11,imass
c Gauss
      double precision DGAUSS,eps
      parameter (eps=1d-7)
      external DGAUSS
c Function to integrate
      double precision Gamma
      external Gamma
      integer type,mode
      double precision Q1,mass
      common /to_sud_exp/Q1,mass,type,mode
c Cache (adapted from the pdg2pdf() functions)
      integer nc
      parameter (nc=64) ! number of calls to put in cache
      integer i,ireuse,ii,i_replace,itypelast(nc)
      double precision sudlast(nc),imasslast(nc),q0last(nc),Q11last(nc)
      save sudlast,i_replace,itypelast,imasslast,q0last,Q11last
      data sudlast/nc*-99d9/
      data itypelast/nc*-99/
      data imasslast/nc*-99d9/
      data q0last/nc*-99d9/
      data Q11last/nc*-99d9/
      data i_replace/nc/
c Check if we can re-use any of the last 'nc' calls
      ireuse=0
      ii=i_replace
      do i=1,nc
         if (itype.eq.itypelast(ii)) then
            if (Q11.eq.Q11last(ii)) then
               if (imass.eq.imasslast(ii)) then
                  if (q0.eq.q0last(ii)) then
                     ireuse=ii
                     exit
                  endif
               endif
            endif
         endif
         ii=ii-1
         if (ii.eq.0) ii=ii+nc
      enddo
c Re-use previous result
      if (ireuse.gt.0) then
         sudakov_exp=sudlast(ireuse)
         return 
      endif
c Cannot reuse anything. Compute a new value
      Q1=Q11
      type=itype
      mass=imass
      mode=2
      sudakov_exp=DGAUSS(Gamma,q0,Q1,eps)
c Calculated a new value: replace the value in cache computed longest
c ago with the newly computed one
      i_replace=mod(i_replace,20)+1
      itypelast(i_replace)=itype
      imasslast(i_replace)=imass
      q0last(i_replace)=q0
      Q11last(i_replace)=Q11
      sudlast(i_replace)=sudakov_exp
      return
      end

      
      double precision function expanded_sudakov_exp(q0,Q11,itype,imass)
c Wrapper code for computation of the expanded exponent of the Sudakov
c Form Factor numerically using Gaussian integration. It also saves the
c last 'nc' computed values, which is helpful when dealing, e.g., with
c many real-emission processes for which many Sudakovs might be
c identical.
      implicit none
c Arguments
      integer itype
      double precision q0,Q11,imass
c Gauss
      double precision DGAUSS,eps
      parameter (eps=1d-7)
      external DGAUSS
c Function to integrate
      double precision Gamma
      external Gamma
      integer type,mode
      double precision Q1,mass
      common /to_sud_exp/Q1,mass,type,mode
c Cache (adapted from the pdg2pdf() functions)
      integer nc
      parameter (nc=64)         ! number of calls to put in cache
      integer i,ireuse,ii,i_replace,itypelast(nc)
      double precision sudlast(nc),imasslast(nc),q0last(nc),Q11last(nc)
      save sudlast,i_replace,itypelast,imasslast,q0last,Q11last
      data sudlast/nc*-99d9/
      data itypelast/nc*-99/
      data imasslast/nc*-99d9/
      data q0last/nc*-99d9/
      data Q11last/nc*-99d9/
      data i_replace/nc/
c Check if we can re-use any of the last 'nc' calls
      ireuse=0
      ii=i_replace
      do i=1,nc
         if (itype.eq.itypelast(ii)) then
            if (Q11.eq.Q11last(ii)) then
               if (imass.eq.imasslast(ii)) then
                  if (q0.eq.q0last(ii)) then
                     ireuse=ii
                     exit
                  endif
               endif
            endif
         endif
         ii=ii-1
         if (ii.eq.0) ii=ii+nc
      enddo
c Re-use previous result
      if (ireuse.gt.0) then
         expanded_sudakov_exp=sudlast(ireuse)
         return 
      endif
c Cannot reuse anything. Compute a new value
      Q1=Q11
      type=itype
      mass=imass
      mode=1
      expanded_sudakov_exp=DGAUSS(Gamma,q0,Q1,eps)
c Calculated a new value: replace the value in cache computed longest
c ago with the newly computed one
      i_replace=mod(i_replace,20)+1
      itypelast(i_replace)=itype
      imasslast(i_replace)=imass
      q0last(i_replace)=q0
      Q11last(i_replace)=Q11
      sudlast(i_replace)=expanded_sudakov_exp
      return
      end

      
      double precision function Gamma(q0)
c Calculates the argument of the integral of the exponent of the Sudakov
c Form Factor. (Type, mass, Q^2, etc, given in the to_sud_exp common
c block)
c   o. For mode=2: take terms logarithmic in log(Q/q0) and constants
c      into account (up to NLL).
c   o. For mode=1: take the strict O(alphaS) expansion, with the alphaS
c      stripped off.
c Types are
c   1 : massless fermion, colour triplet
c   2 : massive fermion, colour triplet
c   3 : massless vector, colour octet
c   4 : massive particle, colour triplet (treated in inf. mass limit)
c   5 : massive particle, colour octet   (treated in inf. mass limit)
      implicit none
      include 'cuts.inc'        ! includes maxjetflavor
      integer i
      double precision q0,alphasq0,mu,qom,moq2,alphas,qmass(6)
     $     ,get_mass_from_id,colfac
      external alphas,get_mass_from_id
      logical firsttime
      save qmass,firsttime
      data firsttime/.true./
c Constants
      double precision ZERO,PI,CA,CF,kappa
      parameter (ZERO=0d0)
      parameter (PI = 3.14159265358979323846d0)
      parameter (CA = 3d0)
      parameter (CF = 4d0/3d0)
c Sudakov type and related
      integer type,mode
      double precision Q1,mass
      common /to_sud_exp/Q1,mass,type,mode
      if (firsttime) then
c Quark masses in g->qqbar splittings (assumes that all other X in g->XX are
c infinitely heavy).
         do i=1,6  ! assume 6 quarks flavours
            qmass(i)=get_mass_from_id(i)
         enddo
         firsttime=.false.
      endif
      Gamma=0.0d0
      kappa = CA*(67d0/18d0-pi**2/6d0)-maxjetflavor*5d0/9d0
      if (q0.ge.Q1) then
         return
      endif
c Compute alphaS at the scale of the branching; with a freeze-out at 0.5
c GeV
      if (mode.eq.2) then
         alphasq0=alphas(max(q0,0.5d0))
      elseif(mode.eq.1) then
         alphasq0=1d0
      else
         write (*,*) 'Unknown mode for Sudakov',mode
         stop
      endif

      if(type.eq.1 .or. type.eq.2) then
c Quark Sudakov
c     q->q+g
         Gamma=CF*alphasq0*(log(Q1/q0)-3d0/4d0)/pi ! A1+B1
         if (type.eq.2) then ! include mass effects
            qom=q0/mass
            Gamma=Gamma+CF*alphasq0/pi/2d0*( 0.5d0 - qom*atan(1d0/qom) -
     $           (1d0-0.5d0*qom**2)*log(1d0+1d0/qom**2) )
         endif
         if (mode.eq.2) then
c$$$            Gamma=Gamma*(1d0+alphasq0*kappa/(2d0*pi)) ! A2
            Gamma=Gamma+CF*alphasq0*log(Q1/q0)/pi*alphasq0*kappa/(2d0*pi) ! A2
         endif
      elseif (type.eq.3) then !  massless vector, colour octet
c Gluon sudakov         
c     g->g+g contribution
         Gamma=CA*alphasq0*(log(Q1/q0)-11d0/12d0)/pi ! A1+B1
c     g->q+qbar contribution
         do i=1,6 ! assume 6 quark flavours
            if (i.le.maxjetflavor) then   ! massless splitting
               Gamma=Gamma+alphasq0/(6d0*pi)   ! B1
            else                          ! massive splitting
               moq2=(qmass(i)/q0)**2
               Gamma=Gamma+alphasq0/(4d0*pi)/(1d0+moq2)*
     &              (1d0 - 1d0/(3d0*(1+moq2)))  ! B1
            endif
         enddo
         if (mode.eq.2) then
c$$$            Gamma=Gamma*(1d0+alphasq0*kappa/(2d0*pi)) ! A2
            Gamma=Gamma+CA*alphasq0*log(Q1/q0)/pi*alphasq0*kappa/(2d0*pi) ! A2
         endif
      elseif (type.eq.4 .or. type.eq.5) then
c Sudakov for massive particle (in the large-mass limit, hence no spin
c information enters the expressions, nor is there a g->XX contribution)
c     X->X+g
         if (type.eq.4) then
            colfac=CF ! X is colour triplet
         elseif(type.eq.5) then
            colfac=CA ! X is colour octet
         endif
         Gamma=colfac*alphasq0*(log(Q1/mass)-1d0/2d0)/pi
c$$$         if (mode.eq.2) then
c$$$            Gamma=Gamma*(1d0+alphasq0*kappa/(2d0*pi)) ! A2
c$$$            Gamma=Gamma*(1d0+alphasq0*kappa/(2d0*pi)) ! A2
c$$$         endif
      else
         write (*,*) 'ERROR in reweight.f: do not know'/
     $        /' which Sudakov to compute',type
         write (*,*) 'FxFx is not supported for models'/
     $        /' with some BSM particles'
         stop 1
      endif
c Integration is over dq^2/q^2 = 2*dq/q, so factor 2. Also, include
c already the minus sign here
      Gamma=-Gamma*2d0/q0
      return
      end


CCCCCCCCCCCCCCC -- KINEMATIC FUNCTIONS -- CCCCCCCCCCCCCCCC

      subroutine crossp(p1,p2,p)
c**************************************************************************
c     input:
c            p1, p2    vectors to cross
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), p(0:3)

      p(0)=0d0
      p(1)=p1(2)*p2(3)-p1(3)*p2(2)
      p(2)=p1(3)*p2(1)-p1(1)*p2(3)
      p(3)=p1(1)*p2(2)-p1(2)*p2(1)

      return 
      end


      subroutine rotate(p1,p2,n,nn2,ct,st,d)
c**************************************************************************
c     input:
c            p1        vector to be rotated
c            n         vector perpendicular to plane of rotation
c            nn2       squared norm of n to improve numerics
c            ct, st    cos/sin theta of rotation in plane 
c            d         direction: 1 there / -1 back
c     output:
c            p2        p1 rotated using defined rotation
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), n(0:3), at(0:3), ap(0:3), cr(0:3)
      double precision nn2, ct, st, na, nn
      integer d, i

      if (nn2.eq.0d0) then
         do i=0,3
            p2(i)=p1(i)
         enddo   
         return
      endif
      nn=dsqrt(nn2)
      na=(n(1)*p1(1)+n(2)*p1(2)+n(3)*p1(3))/nn2
      do i=1,3
         at(i)=n(i)*na
         ap(i)=p1(i)-at(i)
      enddo
      p2(0)=p1(0)
      call crossp(n,ap,cr)
      do i=1,3
         if (d.ge.0) then
            p2(i)=at(i)+ct*ap(i)+st/nn*cr(i)
         else 
            p2(i)=at(i)+ct*ap(i)-st/nn*cr(i)
         endif
      enddo
      
      return 
      end


      subroutine constr(p1,p2,n,nn2,ct,st)
c**************************************************************************
c     input:
c            p1, p2    p1 rotated onto p2 defines plane of rotation
c     output:
c            n         vector perpendicular to plane of rotation
c            nn2       squared norm of n to improve numerics
c            ct, st    cos/sin theta of rotation in plane 
c**************************************************************************
      implicit none
      real*8 p1(0:3), p2(0:3), n(0:3), tr(0:3)
      double precision nn2, ct, st, mct

      ct=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      ct=ct/dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      ct=ct/dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
      mct=ct
c     catch bad numerics
      if (mct-1d0>0d0) mct=0d0
      st=dsqrt(1d0-mct*mct)
      call crossp(p1,p2,n)
      nn2=n(1)**2+n(2)**2+n(3)**2
c     don't rotate if nothing to rotate
      if (nn2.le.1d-34) then
         nn2=0d0
         return
      endif
      return 
      end
      



      double precision function dj_clus(p1,p2)
c***************************************************************************
c     Uses Durham algorythm to calculate the y value for two partons
c     If collision type is hh, hadronic jet measure is used
c       y_{ij} = 2min[p_{i,\perp}^2,p_{j,\perp}^2]/S
c                  (cosh(\eta_i-\eta_j)-cos(\phi_1-\phi_2))
c***************************************************************************
      implicit none
      include 'run.inc'         ! includes 'lpp'
      include 'cuts.inc'        ! includes maxjetflavor
      double precision D
      common/to_dj/D            ! for FxFx: D=1 (set in setrun)
      double precision pt1,pt2,ptm1,ptm2,eta1,eta2,phi1,phi2,p1a,p2a
     $     ,costh,p1(0:4),p2(0:4),m1,m2,djb_clus
      integer j
      external djb_clus
      m1=p1(4)
      m2=p2(4)
      if ((lpp(1).eq.0).and.(lpp(2).eq.0)) then
         p1a = dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
         p2a = dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
         if (p1a*p2a .ne. 0d0) then
            costh = (p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3))/(p1a*p2a)
            dj_clus = 2d0*min(p1(0)**2,p2(0)**2)*max((1d0-costh),0d0)
         else
            dj_clus = 0d0
         endif
      else
        if ( m1.lt.1d0.and.(m2.ge.3d0.and.maxjetflavor.gt.4.or.
     $       m2.ge.1d0.and.maxjetflavor.gt.3))then
           ! particle 1 massless, particle 2 massive
           dj_clus = DJB_clus(p1)*(1d0+1d-6)
        elseif (m2.lt.1d0.and.(m1.ge.3d0.and.maxjetflavor.gt.4.or.
     $          m1.ge.1d0.and.maxjetflavor.gt.3))then
           ! particle 2 massless, particle 1 massive
           dj_clus = DJB_clus(p2)*(1d0+1d-6)
        else
           ! two massless or two massive particles
           pt1 = p1(1)**2+p1(2)**2
           pt2 = p2(1)**2+p2(2)**2
           if (pt1.eq.0d0 .or. pt2.eq.0d0) then
              dj_clus=0d0
              return
           endif
           p1a = dsqrt(pt1+p1(3)**2)
           p2a = dsqrt(pt2+p2(3)**2)
           eta1 = 0.5d0*log((p1a+p1(3))/(p1a-p1(3)))
           eta2 = 0.5d0*log((p2a+p2(3))/(p2a-p2(3)))
           dj_clus = max(m1,m2)**2+min(pt1,pt2)*2d0*(cosh(eta1-eta2)-
     &          (p1(1)*p2(1)+p1(2)*p2(2))/dsqrt(pt1*pt2))/D**2
           if (dj_clus.lt.0d0 .or. dj_clus.ne.dj_clus)
     &                 dj_clus=0d0     ! prevent numerical inaccuracies
        endif
      endif
      end
      
      
      double precision function DJB_clus(p1)
c***************************************************************************
c     Uses kt algorythm to calculate the y value for one parton
c       y_i    = p_{i,\perp}^2/S
c***************************************************************************
      implicit none
      double precision p1(0:3)
      include 'run.inc'
      if ((lpp(1).eq.0).and.(lpp(2).eq.0)) then
         djb_clus=max(p1(0),0d0)**2
      else
         djb_clus=(p1(0)-p1(3))*(p1(0)+p1(3)) ! = M^2+pT^2
      endif
      if (djb_clus.lt.0d0) djb_clus=0d0 ! prevent numerical inaccuracies
      end
