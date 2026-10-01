      subroutine select_native_recoil(iconfig_native)
c Select the chart's own decay tree. Evaluate its resonance criterion
c with the ordinary generation cuts; native radiation projection itself
c still covers the full physical real phase space.
      use mc_native_context, only: native_mapping
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_conf.inc'
      include 'resonance_recoil.inc'
      integer iconfig_native,itree(2,-max_branch:-1),iconf,this_config
      common /to_itree/itree,iconf
      common /to_mconfigs/this_config
      integer igranny,iaunt
      logical granny_is_res,granny_chain(-nexternal:nexternal),
     $     granny_chain_real_final(-nexternal:nexternal),mapping_save,
     $     default_members(nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain,
     $     granny_chain_real_final
      this_config=iconfig_native
      iconf=iconfig_native
      itree=iforest(:,:,iconfig_native)
      mapping_save=native_mapping
      native_mapping=.false.
      call set_tau_min()
      native_mapping=mapping_save
      default_members=.false.
      if(granny_is_res)
     $     default_members=granny_chain_real_final(1:nexternal)
      call select_fks_recoil(default_members,.true.)
      end


      subroutine native_recoil_groups(ngroups,group_config,group_of)
c Diagrams with the same recoil mask and beam share a native projection.
c Born-channel weights are added, so no extra chart sampling or Jacobian
c enters the native H density. With no resonances this is one group.
      use mint_module, only: iconfig
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_conf.inc'
      include 'resonance_recoil.inc'
      integer ngroups,group_config(lmaxconfigs),group_of(lmaxconfigs),
     $     i,g,recoil_beams(lmaxconfigs)
      logical masks(nexternal,lmaxconfigs)
      ngroups=0
      group_of=0
      do i=1,mapconfig(0)
         iconfig=i
         call select_native_recoil(i)
         do g=1,ngroups
            if(all(masks(:,g).eqv.resonance_members).and.
     $           recoil_beams(g).eq.initial_recoil_leg)exit
         enddo
         if(g.gt.ngroups)then
            ngroups=g
            masks(:,g)=resonance_members
            recoil_beams(g)=initial_recoil_leg
            group_config(g)=i
         endif
         group_of(i)=g
      enddo
      end


      double precision function native_recoil_weight(p,group,group_of)
c Sum the diagram partition at this projection. Native providers may
c have more diagrams than the local Born routine's to_amps COMMON.
      use mc_native_context, only: active_context,local_context,
     $     native_result
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'born_conf.inc'
      integer group,group_of(lmaxconfigs),i,d
      double precision p(0:3,nexternal-1),ans,total,numerator,diagram,
     $     amp2(ngraphs),jamp2(0:ncolor),pas(0:3,nexternal),
     $     p_ev(0:3,nexternal)
      common /to_amps/amp2,jamp2
      common /pev/p_ev
      logical calculatedBorn
      common /ccalculatedBorn/calculatedBorn
      pas=0d0
      pas(:,1:nexternal-1)=p
      call set_alphas(pas)
      calculatedBorn=.false.
      call sborn_native(p,ans)
      total=0d0
      numerator=0d0
      do i=1,mapconfig(0)
         d=mapconfig(i)
         if(active_context.eq.local_context)then
            diagram=amp2(d)
         else
            diagram=native_result%diagrams(d)
         endif
         total=total+diagram
         if(group_of(i).eq.group)numerator=numerator+diagram
      enddo
      native_recoil_weight=0d0
      if(total.gt.0d0)native_recoil_weight=numerator/total
      call set_alphas(p_ev)
      calculatedBorn=.false.
      end
