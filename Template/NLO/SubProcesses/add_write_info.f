      subroutine add_write_info(p_born,pp,ybst_til_tolab,iconfig,Hevents
     &     ,putonshell,ndim,x,jpart,npart,pb,shower_scale)
c Computes all the info needed to write out the events including the
c intermediate resonances. It also boosts the events to the lab frame
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "born_nhel.inc"
      include "coloramps.inc"
      include "reweight0.inc"
      include "nFKSconfigs.inc"
      include "leshouche_decl.inc"
      include "run.inc"

c Arguments
      double precision p_born(0:3,nexternal-1),pp(0:3,nexternal)
      double precision ybst_til_tolab,shower_scale
      integer iconfig
      logical Hevents,putonshell
      integer ndim,jpart(7,-nexternal+3:2*nexternal-3),npart
      double precision pb(0:4,-nexternal+3:2*nexternal-3)

c Local
      integer np,i,j,iBornGraph,ida(2),size,nexpart,idum,ires,nres,ns
      integer icolalt(2,-nexternal+3:2*nexternal-3),ip,iflow
      integer ito(-nexternal+3:nexternal)
      double precision xtarget,sumborn,jampsum
      double complex wgt1(2)
      logical firsttime,firsttime2
      data firsttime/.true./
      data firsttime2/.true./

c The process chosen to write
      integer i_process_addwrite
      common/c_addwrite/i_process_addwrite
      
c Random numbers
      double precision ran2
      external ran2

c Jamp amplitudes of the Born (to be filled with a call the sborn())
      double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

C iforest and other configuration info. Read once and saved.
      integer itree_S_t(2,-max_branch:-1),sprop_tree_S_t(-max_branch:-1)
     $     ,itree_H_t(2,-max_branch:-1),sprop_tree_H_t(-max_branch:-1)
      integer itree_S(2,-max_branch:-1,fks_configs),sprop_tree_S(
     $     -max_branch:-1,fks_configs),itree_H(2,-max_branch:-1
     $     ,fks_configs),sprop_tree_H(-max_branch:-1,fks_configs)
      save itree_S,sprop_tree_S,itree_H,sprop_tree_H

c Masses and widths of the (internal) propagators. Read once and saved.
      double precision pmass_tree_S_t(-nexternal:0) ,pwidth_tree_S_t(
     $     -nexternal:0),pmass_tree_H_t(-nexternal:0),pwidth_tree_H_t(
     $     -nexternal:0)
      double precision pmass_tree_S(-nexternal:0,fks_configs)
     $     ,pwidth_tree_S(-nexternal:0,fks_configs),pmass_tree_H(
     $     -nexternal:0,fks_configs),pwidth_tree_H(-nexternal:0
     $     ,fks_configs)
      save pmass_tree_S,pwidth_tree_S,pmass_tree_H,pwidth_tree_H

c tree, s-channel props, masses and widths, copied from the save values
c above
      integer itree(2,-max_branch:-1),sprop_tree(-max_branch:-1)
      double precision pmass_tree(-nexternal:0)
      double precision pwidth_tree(-nexternal:0)

c On Breit-Wigner
      logical OnBW(-nexternal:0)

c LesHouches info
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
c      include "leshouche.inc"
      common /c_leshouche_inc/idup,mothup,icolup,niprocs

c Common block to check if we are doing MC over helicities.
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

c Masses of external (n+1)-body particles
      real*8         emass(nexternal)
      common/to_mass/emass

c The FKS partons
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

c For the boost to the lab frame
      double precision chy,shy,chymo,p1(0:3,nexternal),xdir(3)
      data (xdir(i),i=1,3) /0,0,1/

c For (n+1)-body this is the configuration mapping
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

c For shifting QCD partons from zero to their mass-shell
      double precision x(99),p(0:3,99)
      integer mfail
      double precision xmi,xmj,xm1,xm2,emsum,tmpecm,dot,wgt
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision xmcmass(nexternal)
      common/cxmcmass/xmcmass
      integer mohdr,izero
      parameter (mohdr=-100)
      parameter (izero=0)
c cFKSprocess
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer save_nFKSprocess
      double precision SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      integer iSorH_lhe,ifks_lhe(fks_configs) ,jfks_lhe(fks_configs)
     &     ,fksfather_lhe(fks_configs) ,ipartner_lhe(fks_configs)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
c
c Set the leshouche info and fks info
c
      call fks_inc_chooser()
      call leshouche_inc_chooser()

c
c Set the number of external particles and overwrite the iSorH_lhe value
c
      if (Hevents) then
         nexpart=nexternal
         iSorH_lhe=2
      else
         nexpart=nexternal-1
         iSorH_lhe=1
      endif
c
c Determine which Born graph was used for multi-channeling
c
      iBornGraph=iconfig

c
c Fill the itree, sprop, pmass and pwidth of this configuration. Needed
c to determine possible intermediate s-channel resonances. Note that the
c set_itree subroutine does not properly set the t-channel info.
c
      if (firsttime) then
         save_nFKSprocess=nFKSprocess
         do nFKSprocess=1,FKS_configs
            call fks_inc_chooser()
c For the S-events
            call set_itree(iconfig,.false.,itree_S_t,sprop_tree_S_t
     $           ,pmass_tree_S_t,pwidth_tree_S_t)
c For the H-events
            call set_itree(iconfig,.true.,itree_H_t,sprop_tree_H_t
     $           ,pmass_tree_H_t,pwidth_tree_H_t)
            do j=-max_branch,-1
               itree_H(1,j,nFKSprocess)=itree_H_t(1,j)
               itree_H(2,j,nFKSprocess)=itree_H_t(2,j)
               sprop_tree_H(j,nFKSprocess)=sprop_tree_H_t(j)
               pmass_tree_H(j,nFKSprocess)=pmass_tree_H_t(j)
               pwidth_tree_H(j,nFKSprocess)=pwidth_tree_H_t(j)
               itree_S(1,j,nFKSprocess)=itree_S_T(1,j)
               itree_S(2,j,nFKSprocess)=itree_S_t(2,j)
               sprop_tree_S(j,nFKSprocess)=sprop_tree_S_t(j)
               pmass_tree_S(j,nFKSprocess)=pmass_tree_S_t(j)
               pwidth_tree_S(j,nFKSprocess)=pwidth_tree_S_t(j)
            enddo
         enddo
         firsttime=.false.
         nFKSprocess=save_nFKSprocess
         call fks_inc_chooser()
      endif
c Copy the saved information to the arrays actually used
      if (Hevents) then
         do j=-(nexternal-3),-1
            do i=1,2
               itree(i,j)=itree_H(i,j,nFKSprocess)
            enddo
            sprop_tree(j)=sprop_tree_H(j,nFKSprocess)
            pmass_tree(j)=pmass_tree_H(j,nFKSprocess)
            pwidth_tree(j)=pwidth_tree_H(j,nFKSprocess)
         enddo
      else
         do j=-(nexternal-4),-1
            do i=1,2
               itree(i,j)=itree_S(i,j,nFKSprocess)
            enddo
            sprop_tree(j)=sprop_tree_S(j,nFKSprocess)
            pmass_tree(j)=pmass_tree_S(j,nFKSprocess)
            pwidth_tree(j)=pwidth_tree_S(j,nFKSprocess)
         enddo
      endif
c Set the shower scale
      if (Hevents) then
         shower_scale=SCALUP(nFKSprocess*2)
      else
         shower_scale=SCALUP(nFKSprocess*2-1)
      endif

c This is an (n+1)-body process (see update_unwgt_table in
c driver_mintMC.f). For S events it corresponds to the underlying Born
c process chosen
      ip=i_process_addwrite
      if (ip.lt.1 .or. ip.gt.maxproc_used) then
         write (*,*)'ERROR #12 in add_write_info,'/
     &        /' not a well-defined process',ip,Hevents
         stop
      endif

c
c Fill jpart particle info for the final state particles of
c the (n+1)-body events. Color is done below.
c
      do i=1,nexternal
         jpart(1,i) = idup(i,ip)
         jpart(2,i) = mothup(1,i,ip)
         jpart(3,i) = mothup(2,i,ip)
         if (i.le.nincoming) then
            jpart(6,i) = -1
         else
            jpart(6,i) = 1
         endif
      enddo
c Assume helicity summed
      do i=1,nexternal
         jpart(7,i)=9
      enddo
      if (firsttime2 .and. isum_hel.ne.0) then
         write (*,*) 'WARNING: for writing the events, no helicity '//
     &        'info is used even though some info could be available.'
         firsttime2=.false.
      endif
c Can be filled when doing MC over helicities...
c$$$   read(hel_buf,'(15i5)') (jpart(7,i),i=1,nexternal)

c
c Get color flow that is consistent with iconfig from Born 
c
      call sborn(p_born,wgt1)
      sumborn=0.d0
      do i=1,max_bcol
         if (icolamp(i,iBornGraph,1)) then
            sumborn=sumborn+jamp2(i)
         endif
      enddo
      if (sumborn.eq.0d0) then
         write (*,*) 'Error #1 in add_write_info:'
         write (*,*) 'in MadFKS, sumborn should always be larger'//
     $        ' than zero, because always QCD partons around',sumborn
     $        ,max_bcol
         do i=1,max_bcol
            write (*,*) i,iBornGraph,icolamp(i,iBornGraph,1),jamp2(i)
         enddo
         stop
      endif
      xtarget=ran2()*sumborn

      iflow=1
      if (icolamp(1,iBornGraph,1)) then
         jampsum=jamp2(1)
      else
         jampsum=0d0
      endif
      do while (jampsum .lt. xtarget)
         iflow=iflow+1
         if (icolamp(iflow,iBornGraph,1)) then
            jampsum=jampsum+jamp2(iflow)
         endif
      enddo
      if (iflow.gt.max_bcol) then
         write (*,*) 'ERROR #2 in add_write_info',iflow,max_bcol
         stop
      endif

c
c Shift particle momenta to put them on the mass shell as given in the
c subroutine fill_MC_mshell().
c
      if(putonshell)then
         mfail=-1
c fills the common block with the MC masses (special treatment of i_fks
c and j_fks, because phase-space generation won't work in all cases).
         call put_on_MC_mshell(Hevents,jpart,xmi,xmj,xm1,xm2,emsum)
c Prevents the code from crashing in the extremely rare case in which
c the cm energy is smaller than sum of masses - keep massless partons
         tmpecm=min(dot(pp(0,1),pp(0,2)),
     #              dot(p_born(0,1),p_born(0,2)))
         tmpecm=sqrt(2d0*tmpecm)
         if(tmpecm.lt.0.99*emsum)then
           write (*,*) 'Momenta generation for put_on_MC_mshell failed'
           mfail=1
           goto 888
         endif
         wgt=1d0
c generate a phase-space point with the MC masses
         call generate_momenta(ndim,iconfig,wgt,x,p)
         if(Hevents)then
            call set_cms_stuff(mohdr)
c special treament here for i_fks and j_fks masses
            call put_on_MC_mshell_Hev(p,xmi,xmj,xm1,xm2,mfail)
c include initial state masses
            if(j_fks.gt.nincoming.and.mfail.eq.0)
     &           call put_on_MC_mshell_in(p,xm1,xm2,mfail)
         else
c include initial state masses
            call set_cms_stuff(izero)
            call put_on_MC_mshell_in(p1_cnt(0,1,0),xm1,xm2,mfail)
         endif
 888     continue
c restore the common block for the masses to the original MG masses
         call put_on_MG_mshell()
         if (mfail.eq.0) then 
c all went fine and we can copy the new momenta onto the old ones.
            do i=1,nexternal
               do j=0,3
                  if(Hevents) then
                     pp(j,i)=p(j,i)
                  elseif(.not.Hevents .and. i.lt.max(i_fks,j_fks)) then
                     p_born(j,i)=p1_cnt(j,i,0)
                  elseif(.not.Hevents .and. i.gt.max(i_fks,j_fks)) then
                     p_born(j,i-1)=p1_cnt(j,i,0)
                  endif
               enddo
            enddo
         elseif(mfail.eq.1)then
c Probably not needed, but just to make sure: fill the momenta common
c blocks again by call generate momenta again.
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            if(Hevents)then
              call set_cms_stuff(mohdr)
            else
              call set_cms_stuff(izero)
            endif
c Also, set the masses that need to written in the event file to the MG
c masses
            call write_masses_lhe_MG()
         elseif(mfail.eq.-1)then
            write(*,*)'Error in driver: mfail not set',mfail
            stop
         endif
      else
c Use the MadGraph masses in the event file
         call write_masses_lhe_MG()
      endif

c
c Derive the n-body from the (n+1)-body if we are doing S-events
c
      if (.not.Hevents) then
         do i=1,nexternal-1
            if(i.lt.min(i_fks,j_fks)) then
               jpart(1,i) = jpart(1,i)
               jpart(2,i) = jpart(2,i)
               jpart(3,i) = jpart(3,i)
            elseif(i.eq.min(i_fks,j_fks)) then
               if(jpart(1,i_fks).eq.-jpart(1,j_fks)
     &              .and.j_fks.gt.nincoming) then
                  jpart(1,i) = 21
               elseif(jpart(1,i_fks).eq.jpart(1,j_fks)
     &                 .and.j_fks.le.nincoming) then
                  jpart(1,i)=21
               elseif(abs(jpart(1,i_fks)).eq.21) then
                  jpart(1,i)=jpart(1,j_fks)
               elseif(jpart(1,j_fks).eq.21.and.j_fks.le.nincoming) then
                  jpart(1,i)=-jpart(1,i_fks)
               else
                  write (*,*) 'ERROR #5 in add_write_info()',
     &                 i_fks,j_fks,jpart(1,i_fks),jpart(1,j_fks)
                  stop
               endif
               jpart(2,i) = jpart(2,i)
               jpart(3,i) = jpart(3,i)
            elseif(i.lt.max(i_fks,j_fks)) then
               jpart(1,i) = jpart(1,i)
               jpart(2,i) = jpart(2,i)
               jpart(3,i) = jpart(3,i)
            else
               jpart(1,i) = jpart(1,i+1)
               jpart(2,i) = jpart(2,i+1)
               jpart(3,i) = jpart(3,i+1)
            endif
         enddo
      endif

c
c Fill color of external particles
c
      if (Hevents) then
         call fill_icolor_H(iflow,jpart)
      else
         call fill_icolor_S(iflow,jpart,idum)
      endif
      do i=1,nexpart
         icolalt(1,i)=jpart(4,i)
         icolalt(2,i)=jpart(5,i)
      enddo

c
c Set-up the external momenta that should be written in event file
c Also boost them to the lab frame.
c
      if (abs(ybst_til_tolab).le.1d-7) then
         do i=1,nexpart
            do j=0,3
               if (Hevents) then
                  pb(j,i)=pp(j,i)
               else
                  pb(j,i)=p_born(j,i)
               endif
            enddo
            if (Hevents .or. i.lt.i_fks) then
               pb(4,i)=xmcmass(i)
            else
               pb(4,i)=xmcmass(i+1)
            endif
         enddo
      else
         chy=cosh(ybst_til_tolab)
         shy=sinh(ybst_til_tolab)
         chymo=chy-1d0
         do i=1,nexpart
            if (Hevents) then
               call boostwdir2(chy,shy,chymo,xdir,
     &              pp(0,i),p1(0,i))
            else
               call boostwdir2(chy,shy,chymo,xdir,
     &              p_born(0,i),p1(0,i))
            endif
            do j=0,3
               pb(j,i)=p1(j,i)
            enddo
            if (Hevents .or. i.lt.i_fks) then
               pb(4,i)=xmcmass(i)
            else
               pb(4,i)=xmcmass(i+1)
            endif
         enddo
c In some rare cases (due to numerical inaccuracies in the boost) the
c energy of the incoming partons can be larger than the beam energy This
c is, of course, non-physical, so we use the non-boosted events
         if (pb(0,1).gt.ebeam(1).or.pb(0,2).gt.ebeam(2)) then
            write (*,*) 'WARNING: boost from center-of-momentum to '/
     &           /'laboratory frame too extreme. Use the center-of-mo'/
     &           /'mentum momenta instead.',pb(0,1),pb(0,2),ebeam(1)
     &           ,ebeam(2)
            do i=1,nexpart
               do j=0,3
                  if (Hevents) then
                     pb(j,i)=pp(j,i)
                  else
                     pb(j,i)=p_born(j,i)
                  endif
               enddo
            enddo
         endif
      endif

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C ALL EXTERNAL PARTICLE INFO SET-UP. CHECK WHICH RESONANCES SHOULD BE WRITTEN C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c
c Fill the OnBW array to determine which resonances should be written
c
      call OnBreitWigner(pp,p_born,Hevents,itree,sprop_tree,pmass_tree,
     &     pwidth_tree,OnBW)

c     First check number of resonant s-channel propagators
      ns=0
      nres=0

c     Loop over propagators to find mother-daughter information
      do i=-1,-nexpart+3,-1
c     Daughters
         ida(1)=itree(1,i)
         ida(2)=itree(2,i)
c Skip the t-channels
         if ( itree(1,i) .eq. 1 .or. itree(2,i) .eq. 1 .or.
     &        itree(1,i) .eq. 2 .or. itree(2,i) .eq. 2 ) exit
         jpart(1,i)=sprop_tree(i)
         ns=ns+1
c     Set status codes for propagator
         if(OnBW(i)) then 
c     Resonance whose mass should be preserved
            jpart(6,i)=2
            nres=nres+1
         else
c     Tag the s-channel internal propagators that we'll remove later
            jpart(6,i)=3
         endif
c     Calculate momentum (p1+p2 for s-channel)
         do j=0,3
            pb(j,i)=pb(j,ida(1))+pb(j,ida(2))
         enddo
         pb(4,i)=
     &        sqrt(max(0d0,pb(0,i)**2-pb(1,i)**2-pb(2,i)**2-pb(3,i)**2))

c     Set color info for all s-channels
c     Fist set "safe" color info
         if(icolalt(1,ida(1))+icolalt(1,ida(2))-
     $        icolalt(2,ida(1))-icolalt(2,ida(2)).eq.0) then ! color singlet
            icolalt(1,i) = 0
            icolalt(2,i) = 0            
         elseif(icolalt(1,ida(1))-icolalt(2,ida(2)).eq.0) then
            icolalt(1,i) = icolalt(1,ida(2))
            icolalt(2,i) = icolalt(2,ida(1))
         else if(icolalt(1,ida(2))-icolalt(2,ida(1)).eq.0) then
            icolalt(1,i) = icolalt(1,ida(1))
            icolalt(2,i) = icolalt(2,ida(2))
         else if(jpart(6,i).ge.3) then ! Don't need to match
            icolalt(1,i) = icolalt(1,ida(1))+icolalt(1,ida(2))
            icolalt(2,i) = icolalt(2,ida(1))+icolalt(2,ida(2))
         else
c     Erraneous color assignment for propagator
            write(*,*) 'ERROR: Safe color assignment wrong!'/
     &           /' This should never happen !',i,icolalt(1,ida(1))
     &           ,icolalt(1,ida(2)),icolalt(2,ida(1)),icolalt(2,ida(2))
     &           ,ida(1),ida(2)
            stop
         endif
c     Set initial state as tentative mothers
         jpart(2,i) = 1
         jpart(3,i) = 2
c     Set mother info for daughters
         do j=1,2
            jpart(2,ida(j)) = i
            jpart(3,ida(j)) = i
         enddo
c     Just zero helicity info for intermediate states
         jpart(7,i) = 0
      enddo                     ! do i (loop over internal propagators)
      
c     Remove non-resonant mothers, set position of particles
      ires=0
      do i=-ns,nexpart
         jpart(4,i)=icolalt(1,i)
         jpart(5,i)=icolalt(2,i)
         if(i.eq.1.or.i.eq.2) then 
            ito(i)=i            ! initial state particle
         elseif(i.ge.3) then 
            ito(i)=i+nres       ! final state particle
         elseif(i.le.-1.and.jpart(6,i).eq.2) then
            ires=ires+1
            ito(i)=2+ires       ! s-channel resonances
         else 
            ito(i)=0
            if(i.eq.0) cycle
         endif
         if(jpart(2,i).lt.0.and.jpart(6,jpart(2,i)).ne.2) then
            jpart(2,i)=jpart(2,jpart(2,i))
            jpart(3,i)=jpart(3,jpart(3,i))
         endif
      enddo
        
c
c Shift particles to right place
c     
      do i=nexpart,-ns,-1
         if(ito(i).le.0) cycle
         do j=1,7
            jpart(j,ito(i))=jpart(j,i)
         enddo
         if(jpart(2,ito(i)).lt.0) then
            jpart(2,ito(i))=ito(jpart(2,ito(i)))
            jpart(3,ito(i))=ito(jpart(3,ito(i)))
         endif
         do j=0,4
            pb(j,ito(i))=pb(j,i)
         enddo
      enddo
c
c     Set the number of particles that needs to be written in event file
c
      npart = nexpart+nres
         
      return
      end


      subroutine set_itree(iconfig,Hevents,itree,sprop_tree,pmass_tree
     &     ,pwidth_tree)
      implicit none
      integer iconfig
      logical Hevents
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      double precision ZERO
      parameter (ZERO=0d0)
      integer itree(2,-max_branch:-1)
      integer sprop_tree(-max_branch:-1)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs),mapconfig(0:lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include "born_conf.inc"
      integer i,j,jj
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      double precision pmass_tree(-nexternal:0)
      double precision pwidth_tree(-nexternal:0)
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      include "born_props.inc"
c
c Do not really care about t-channels: loop should just go to
c nexternal-4, even though there is one more when there are t-channels
c around
      do j=-(nexternal-4),-1
         do i=1,2
            itree(i,j)=iforest(i,j,iconfig)
         enddo
         sprop_tree(j)=sprop(j,iconfig)
         pmass_tree(j)=pmass(j,iconfig)
         pwidth_tree(j)=pwidth(j,iconfig)
      enddo
c
c When we are doing H-events, we need to add --when j_fks is final
c state-- the s-channel branching fks_mother -> j_fks + i_fks.  When
c j_fks is initial state, we need to add a 'bogus' t-channel splitting
c to make sure that all the loops stop properly
c
      if (Hevents) then
c must re-label the external particles to get the correct daughters
         if (i_fks.le.j_fks) then
            write (*,*) 'ERROR: i_fks should be greater than j_fks'
            stop
         endif
         do j=-(nexternal-4),-1
            do i=1,2
               if ( itree(i,j).ge.i_fks ) then
                  itree(i,j)=itree(i,j)+1
               endif
            enddo
         enddo
c
         if (j_fks.gt.nincoming) then
c we must add an extra s-channel. Easiest is to add it all the way at
c the beginning. Therefore, relabel everything else first. Use the
c original ones to make sure we are doing it correctly (and not
c accidentally overwriting something).
            do j=-(nexternal-4),-1
               sprop_tree(j-1)=sprop(j,iconfig)
               pmass_tree(j-1)=pmass(j,iconfig)
               pwidth_tree(j-1)=pwidth(j,iconfig)
               do i=1,2
                  itree(i,j-1)=iforest(i,j,iconfig)
c Also update the internal references
                  if ( itree(i,j-1).lt. 0 ) then
                     itree(i,j-1)=itree(i,j-1)-1
                  endif
               enddo
            enddo
c
c Add the new s-channel
c
            itree(1,-1)=i_fks
            itree(2,-1)=j_fks
c This will never be an on-shell s-channel. Hence, give it some bogus values:
            sprop_tree(-1)=0
            pmass_tree(-1)=0d0
            pwidth_tree(-1)=0d0
c
c We have to make sure that the fks_mother (which is equal to the j_fks
c label of the Born) is replaced by the new s-channel
c
            do j=-(nexternal-3),-2 ! do not include the new s-channel
               do i=1,2
                  if ( itree(i,j).eq. j_fks ) then
                     itree(i,j)=-1 ! reference to the new s-channel
                  endif
               enddo
            enddo

         else
c j_fks is initial state
            jj=-(nexternal-3)     ! Just add it at the end
c setting itree to 1 (or 2) makes sure that the loops over s-channel
c propagators will exit
            itree(1,jj)=1
            itree(2,jj)=1
            sprop_tree(jj)=0
            pmass_tree(jj)=0d0
            pwidth_tree(jj)=0d0
         endif
      endif

      return
      end




      subroutine OnBreitWigner(p,p_born,Hevents,itree,sprop_tree,
     &     pmass,pwidth,OnBW)
c*****************************************************************************
c Decides if internal s-channel propagator is on-shell
c*****************************************************************************
      implicit none
c
c     Constants
c     
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
c
c     Arguments
c
      double precision p(0:3,nexternal),p_born(0:3,nexternal-1)
      logical OnBW(-nexternal:0),Hevents
      integer itree(2,-max_branch:-1),sprop_tree(-max_branch:-1)
      double precision pmass(-nexternal:0)
      double precision pwidth(-nexternal:0)
c
c     Local
c
      double precision xp(0:3,-nexternal:nexternal)
      integer i,j,iloop
      logical onshell
      double precision xmass
      integer ida(2),idenpart
      integer IDUP(nexternal)
c
c     External
c
      double precision dot
c-----
c  Begin Code
c-----      
      if (Hevents) then
         call get_ID_H(IDUP)
         iloop=nexternal-3
      else
         call get_ID_S(IDUP)
         iloop=nexternal-4
      endif
      do i=1,nexternal
         do j=0,3
            if (Hevents) then
               xp(j,i)=p(j,i)
            else
               if (i.ne.nexternal) then
                  xp(j,i)=p_born(j,i)
               elseif (i.eq.nexternal) then
                  xp(j,i)=0d0
               endif
            endif
         enddo
      enddo
c
      do i=-1,-iloop,-1                      !Loop over propagators
         onbw(i) = .false.
c Skip the t-channels
         if ( itree(1,i) .eq. 1 .or. itree(2,i) .eq. 1 .or.
     &        itree(1,i) .eq. 2 .or. itree(2,i) .eq. 2 ) exit
         do j=0,3
            xp(j,i) = xp(j,itree(1,i))+xp(j,itree(2,i))
         enddo
         if (pwidth(i) .gt. 0d0) then !This is B.W.
c
c     If the invariant mass is close to pole mass, set OnBW to true
c
            xmass = sqrt(dot(xp(0,i),xp(0,i)))
            onshell = ( abs(xmass-pmass(i)) .lt. bwcutoff*pwidth(i) )
            if(onshell)then
               OnBW(i) = .true.
c     If mother and daughter have the same ID, remove one of them
               idenpart=0
               do j=1,2
                  ida(j)=itree(j,i)
                  if(ida(j).lt.0) then
                     if (sprop_tree(i).eq.sprop_tree(ida(j))) then
                        idenpart=ida(j) ! mother and daugher have same ID
                     endif
                  elseif (ida(j).gt.0) then
                     if (sprop_tree(i).eq.IDUP(ida(j))) then
                        idenpart=ida(j) ! mother and daugher have same ID
                     endif
                  endif
               enddo
c     Always remove if daughter final-state (and identical)
               if(idenpart.gt.0) then
                  OnBW(i)=.false.
c     Else remove either this resonance or daughter,
c                  whichever is closer to mass shell
               elseif(idenpart.lt.0.and.abs(xmass-pmass(i)).gt.
     $                 abs(sqrt(dot(xp(0,idenpart),xp(0,idenpart)))-
     $                 pmass(i))) then
                  OnBW(i)=.false.         ! mother off-shell
               elseif(idenpart.lt.0) then
                  OnBW(idenpart)=.false.  ! daughter off-shell
               endif
            endif
         endif
      enddo
      end

      subroutine fill_icolor_H(iflow,jpart)
      implicit none
      include "nexternal.inc"
c      include 'fks.inc'
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision xtarget,ran2
      external ran2
      integer jpart(7,-nexternal+3:2*nexternal-3),iflow
      integer i_part,j_part,imother,lc
c
      call fill_icolor_S(iflow,jpart,lc)
c
      j_part = particle_type(j_fks)
      i_part = particle_type(i_fks)
c
      do i=nexternal,1,-1
         if (i.gt.max(i_fks,j_fks)) then
            jpart(4,i)=jpart(4,i-1)
            jpart(5,i)=jpart(5,i-1)
         endif
      enddo
c
      imother=min(i_fks,j_fks)
c The following works only if i_fks is always greater than j_fks.      
      if (j_fks.gt.nincoming) then
         if (j_part.eq.3.and.i_part.eq.-3) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #3 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=0
            jpart(5,i_fks)=jpart(5,imother)
            jpart(4,j_fks)=jpart(4,imother)
            jpart(5,j_fks)=0
         elseif (j_part.eq.-3.and.i_part.eq.3) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #4 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=jpart(4,imother)
            jpart(5,i_fks)=0
            jpart(4,j_fks)=0
            jpart(5,j_fks)=jpart(5,imother) 
         elseif (j_part.eq.3.and.i_part.eq.8) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).ne.0)then
               write (*,*) 'Error #5 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=jpart(4,imother)
            jpart(5,i_fks)=lc+1
            jpart(4,j_fks)=lc+1
            jpart(5,j_fks)=0
         elseif (j_part.eq.-3.and.i_part.eq.8) then
            if(jpart(4,imother).ne.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #6 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=lc+1
            jpart(5,i_fks)=jpart(5,imother)
            jpart(4,j_fks)=0
            jpart(5,j_fks)=lc+1
         elseif (j_part.eq.8.and.i_part.eq.8) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #7 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            if (ran2().gt.0.5d0) then 
               jpart(4,i_fks)=lc+1
               jpart(5,i_fks)=jpart(5,imother)
               jpart(4,j_fks)=jpart(4,imother)
               jpart(5,j_fks)=lc+1
            else
               jpart(4,i_fks)=jpart(4,imother)
               jpart(5,i_fks)=lc+1
               jpart(4,j_fks)=lc+1
               jpart(5,j_fks)=jpart(5,imother)
            endif
         else
            write (*,*) 'Error #1 in fill_icolor_H',i_part,j_part
            stop
         endif
      else
         if (j_part.eq.-3.and.i_part.eq.-3) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #8 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=0
            jpart(5,i_fks)=jpart(4,imother)
            jpart(4,j_fks)=0
            jpart(5,j_fks)=jpart(5,imother)
         elseif (j_part.eq.3.and.i_part.eq.3) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #9 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother),'HERE'
               stop
            endif
            jpart(4,i_fks)=jpart(5,imother)
            jpart(5,i_fks)=0
            jpart(4,j_fks)=jpart(4,imother)
            jpart(5,j_fks)=0
         elseif (j_part.eq.3.and.i_part.eq.8) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).ne.0)then
               write (*,*) 'Error #10 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=lc+1
            jpart(5,i_fks)=jpart(4,imother)
            jpart(4,j_fks)=lc+1
            jpart(5,j_fks)=0
         elseif (j_part.eq.-3.and.i_part.eq.8) then
            if(jpart(4,imother).ne.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #11 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=jpart(5,imother)
            jpart(5,i_fks)=lc+1
            jpart(4,j_fks)=0
            jpart(5,j_fks)=lc+1
         elseif (j_part.eq.8.and.i_part.eq.3) then
            if(jpart(4,imother).ne.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #12 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=lc+1
            jpart(5,i_fks)=0
            jpart(4,j_fks)=lc+1
            jpart(5,j_fks)=jpart(5,imother)
         elseif (j_part.eq.8.and.i_part.eq.-3) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).ne.0)then
               write (*,*) 'Error #13 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            jpart(4,i_fks)=0
            jpart(5,i_fks)=lc+1
            jpart(4,j_fks)=jpart(4,imother)
            jpart(5,j_fks)=lc+1
         elseif (j_part.eq.8.and.i_part.eq.8) then
            if(jpart(4,imother).eq.0 .or. jpart(5,imother).eq.0)then
               write (*,*) 'Error #14 in fill_icolor_H',
     &              jpart(4,imother),jpart(5,imother)
               stop
            endif
            if (ran2().gt.0.5d0) then 
               jpart(4,i_fks)=lc+1
               jpart(5,i_fks)=jpart(4,imother)
               jpart(4,j_fks)=lc+1
               jpart(5,j_fks)=jpart(5,imother)
            else
               jpart(4,i_fks)=jpart(5,imother)
               jpart(5,i_fks)=lc+1
               jpart(4,j_fks)=jpart(4,imother)
               jpart(5,j_fks)=lc+1
            endif
         else
            write (*,*) 'Error #2 in fill_icolor_H',i_part,j_part
            stop
         endif
      endif
c
      return
      end


      subroutine fill_icolor_S(iflow,jpart,lc)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer    maxflow
      parameter (maxflow=999)
      integer i
      integer idup(nexternal,maxproc)
      integer mothup(2,nexternal,maxproc)
      integer icolup(2,nexternal,maxflow)
      include "born_leshouche.inc"
      integer jpart(7,-nexternal+3:2*nexternal-3),lc,iflow
c
      lc=0
      do i=1,nexternal-1
         jpart(4,i)=ICOLUP(1,i,iflow)
         lc=max(lc,jpart(4,i))
         jpart(5,i)=ICOLUP(2,i,iflow)
         lc=max(lc,jpart(5,i))
      enddo
c
      return
      end



      subroutine put_on_MC_mshell(Hevents,jpart,xmi,xmj,xm1,xm2,emsum)
c Sets the common block /to_mass/emass, to be passed to one_tree
c to generate a massive kinematics with efficiency 1.
c
c ip is the number of the partonic process chosen at random in the case
c of multiple possibilities.
c This routines assumes that the mother of (i_fks,j_fks) has
c label min(i_fks,j_fks)
      implicit none
      integer ip
      double precision xmi,xmj,xm1,xm2,emsum
      logical Hevents

      integer i,j,idpart,idparti,idpartj
      double precision zero,tmpmass
      parameter (zero=0.d0)
c jpart contains ID of particles
      include 'nexternal.inc'
      integer jpart(7,-nexternal+3:2*nexternal-3)

c Masses to be given to genps_fks.f
      double precision emass(nexternal)
      common/to_mass/  emass

c Monte Carlo masses: use PDG conventions
      double precision mcmass(-16:21)
      common/cmcmass/mcmass

c Masses used by write_events_lhe
      double precision xmcmass(nexternal)
      common/cxmcmass/xmcmass

c cuts.inc contains maxjetflavor
      include 'cuts.inc'

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

c Masses of the real process, as set by MadGraph
      include 'coupl.inc'
      double precision pmass(nexternal)
      include 'pmass.inc'
c
      if(j_fks.ne.min(i_fks,j_fks))then
        write(*,*)'j_fks#min(i_fks,j_fks) in put_on_MC_mshell'
        stop
      endif
c
      xmi=-1.d0
      xmj=-1.d0
      xm1=-1.d0
      xm2=-1.d0
c WARNING: what follows will need to be reconsidered the case of 
c QED corrections, for what is relevant to i_fks and j_fks
      do i=1,nexternal
        if(i.eq.i_fks)then
          if(pmass(i).ne.0.d0)then
            write(*,*)'Fatal error in put_on_MC_mshell',i_fks,i,pmass(i)
            stop
          endif
          emass(i)=0.d0
        elseif(i.eq.j_fks)then
          idpartj=jpart(1,j_fks)
          idparti=jpart(1,i_fks)
          if(.not.Hevents)then
c S events
            xmi=0.d0
            if(idparti.eq.-idpartj.and.j_fks.gt.nincoming) then
               idpart=21
            elseif(idparti.eq.idpartj.and.j_fks.le.nincoming) then
               idpart=21
            elseif(abs(idparti).eq.21) then
               idpart=idpartj
            elseif(idpartj.eq.21.and.j_fks.le.nincoming) then
               idpart=-idparti
            else
               write(*,*)'Error #2 in put_on_MC_mshell',
     &              i_fks,j_fks,idparti,idpartj
               stop
            endif
            if( (abs(idpart).ge.1.and.abs(idpart).le.3) .or.
     #          idpart.eq.21 .or.
     #          ( (abs(idpart).eq.4.or.abs(idpart).eq.5) .and.
     #            pmass(j_fks).eq.0.d0 ) )then
              xmj=mcmass(idpart)
            else
c j_fks is an heavy particle
              if(abs(idparti).ne.21)then
                write(*,*)'Error #3 in put_on_MC_mshell',
     &            i_fks,j_fks,i,pmass(j_fks)
                stop
              endif
c If MadFKS has a non-zero mass for a c or b quark, one probably wants
c to use that in the shower phase as well (ie bottom or charm production)
              xmj=pmass(j_fks)
            endif
            if(j_fks.gt.nincoming)then
              emass(j_fks)=xmj
            else
c one_tree assumes massless incoming QCD particles
              emass(j_fks)=0.d0
              if(j_fks.eq.1)then
                xm1=xmj
              else
                xm2=xmj
              endif
            endif
            xmcmass(i)=xmj
          else
c H events
            xmi=mcmass(idparti)
            if( (abs(idpartj).ge.1.and.abs(idpartj).le.3) .or.
     #          idpartj.eq.21 .or.
     #          ( (abs(idpartj).eq.4.or.abs(idpartj).eq.5) .and.
     #            pmass(j_fks).eq.0.d0 ) )then
              xmj=mcmass(idpartj)
            else
c j_fks is an heavy particle
              if(idparti.ne.21)then
                write(*,*)'Error #3 in put_on_MC_mshell',
     &            i_fks,j_fks,i,pmass(j_fks),(jpart(1,j),j=1,nexternal)
                stop
              endif
c If MadFKS has a non-zero mass for a c or b quark, one probably wants
c to use that in the shower phase as well (ie bottom or charm production)
              xmj=pmass(j_fks)
            endif
            if(j_fks.gt.nincoming)then
              emass(j_fks)=xmi+xmj
            else
c one_tree assumes massless incoming QCD particles
              emass(j_fks)=0.d0
              if(j_fks.eq.1)then
                xm1=xmj
              else
                xm2=xmj
              endif
            endif
            xmcmass(i_fks)=xmi
            xmcmass(j_fks)=xmj
          endif
        else
          idpart=jpart(1,i)
          if( idpart.eq.21 .or.
     #        (abs(idpart).ge.1.and.abs(idpart).le.5) .or.
     #        (abs(idpart).ge.11.and.abs(idpart).le.16) )then
            if(pmass(i).eq.0.d0)then
              tmpmass=mcmass(idpart)
            else
c If MadFKS has a non-zero mass for a "light" quark or lepton, one probably 
c wants to use that in the shower phase as well (ie bottom or charm production).
c One may use equivalently a condition on maxjetflavor
              tmpmass=pmass(i)
            endif
          else
            tmpmass=pmass(i)
          endif
          if(i.gt.nincoming)then
            emass(i)=tmpmass
          elseif(i.eq.1)then
            emass(i)=0.d0
            xm1=tmpmass
          elseif(i.eq.2)then
            emass(i)=0.d0
            xm2=tmpmass
          endif
          xmcmass(i)=tmpmass
        endif
      enddo
c
      emsum=0.d0
      do i=nincoming+1,nexternal
        emsum=emsum+emass(i)
      enddo
c
      if( xmi.eq.-1.d0.or.xmj.eq.-1.d0 .or.
     #    xm1.eq.-1.d0.or.xm2.eq.-1.d0 )then
        write(*,*)'Error #4 in put_on_MC_mshell',i_fks,j_fks
        write(*,*)xmi,xmj,xm1,xm2
        stop
      endif
c
      return
      end


      subroutine put_on_MC_mshell_Hev(p,xmi,xmj,xm1,xm2,mfail)
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,99),xmi,xmj,xm1,xm2
      integer mfail
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c
      if (p(0,1).lt.0d0) then
         mfail=1
         write (*,*) 'Momenta generation for put_on_MC_mshell failed'
         return
      endif

      if(j_fks.le.nincoming)then
        call put_on_MC_mshell_Hevin(p,xmi,xm1,xm2,mfail)
      else
        call put_on_MC_mshell_Hevout(p,xmi,xmj,mfail)
      endif
c
      return
      end


      subroutine put_on_MC_mshell_in(p,xm1,xm2,mfail)
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal),xm1,xm2
      integer mfail
      double precision xm1_r,xm2_r

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision xmcmass(nexternal)
      common/cxmcmass/xmcmass
c
      if (p(0,1).lt.0d0) then
         mfail=1
         write (*,*) 'Momenta generation for put_on_MC_mshell failed'
         return
      endif

      if(abs(p(3,1)+p(3,2)).gt.1.d-10)then
        write(*,*)'Error #1 in put_on_MC_mshell_in',p(3,1),p(3,2)
        stop
      endif
      if(shat.le.(xm1+xm2)**2)then
        mfail=1
        return
      endif
      xm1_r=xm1
      xm2_r=xm2
c$$$CHECK AGAIN USE OF ybst_til_tolab IN getxmss.
c$$$MUST BE THE SAME BOOST AS WHEN WRITING EVENTS
      call getxmss_madfks(shat,ybst_til_tolab,
     #                    p(3,1),xm1_r,p(3,2),xm2_r,
     #                    p(3,1),p(3,2),mfail)
      if(mfail.eq.0)then
        p(0,1)=sqrt(xm1_r**2+p(3,1)**2)
        p(0,2)=sqrt(xm2_r**2+p(3,2)**2)
        xmcmass(1)=xm1_r
        xmcmass(2)=xm2_r
      endif
c
      return
      end


      subroutine getxmss_madfks(shat,ycm,p13cm,xm1,p23cm,xm2,
     #                          p13,p23,mfail)
c This routine is taken from the MC@NLO package. It is identical
c to the routine getxmss() there, except for the fact that here,
c by setting donotforce=.true., the partons are left massless if
c putting them on the MC mass shell implies that they travel in
c the same direction.
c NOTE: the convention on the sign of the boost is opposite to that
c  used in MC@NLO, hence the different definition of ytmp here
c
c Original comment in getxmss():
c After putting the momenta on shell, the two incoming partons may
c travel in the same direction. This routine prevents this to happen,
c redefining Monte Carlo masses if necessary
      implicit none
      real*8 shat,ycm,p13cm,xm1,p23cm,xm2,p13,p23
      integer mfail
      real*8 tiny,fact,sqs,xm1s,xm2s,xkp2prime_norm2,xkp2prime_norm,
     #  ytmp,e1,e2,p13p,p23p,s1p,s2p,xif,sol
      integer iflag,idone,ileg
      parameter (fact=0.98d0)
      parameter (tiny=1.d-6)
      logical donotforce
      parameter (donotforce=.false.)
c
      sqs=sqrt(shat)
      xm1s=xm1
      xm2s=xm2
c NOTE: was ytmp=-ycm in MC@NLO owing to different conventions
      ytmp=ycm
      idone=0
 100  continue
      xkp2prime_norm2=( shat-2*(xm1**2+xm2**2)+
     #                  (xm1**2-xm2**2)**2/shat )/4.d0
      xkp2prime_norm=sqrt(xkp2prime_norm2)
      if(sign(1.d0,p13cm).ne.1.d0.or.sign(1.d0,p23cm).ne.-1.d0)then
        write(*,*)'Error # 0 in getxmss_madfks'
        stop
      endif
      p13=xkp2prime_norm
      p23=-xkp2prime_norm
      e1=sqrt(p13**2+xm1**2)
      e2=sqrt(p23**2+xm2**2)
      p13p=p13*cosh(ytmp)-e1*sinh(ytmp)
      p23p=p23*cosh(ytmp)-e2*sinh(ytmp)
      s1p=sign(1.d0,p13p)
      s2p=sign(1.d0,p23p)
      iflag=0
      if(s1p.eq.1.d0 .and. s2p.eq.-1.d0)then
        iflag=1
      elseif(s1p.eq.-1.d0 .and. s2p.eq.-1.d0)then
        if(ytmp.lt.0.d0)then
          write(*,*)'getxmss_madfks: wrong y sign, # 1'
          stop
        endif
        ileg=1
        xif=xm2**2/shat
      elseif(s1p.eq.1.d0 .and. s2p.eq.1.d0)then
        if(ytmp.gt.0.d0)then
          write(*,*)'getxmss_madfks: wrong y sign, # 2'
          stop
        endif
        ileg=2
        xif=xm1**2/shat
      else
        write(*,*)'Error # 1 in getxmss_madfks',s1p,s2p
        write(*,*)shat,xm1,xm2
        write(*,*)p13,e1,p23,e2,ytmp
        stop
      endif
      if(iflag.eq.0.and.(.not.donotforce))then
        sol=xif+cosh(2*ytmp)-
     #      sqrt(2.d0)*cosh(ytmp)*sqrt(cosh(2*ytmp)-1+2*xif)
        if(sol.le.0.d0.or.idone.eq.1)then
c The procedure failed; pass the massless event to the Monte Carlo, 
c and let the Monte Carlo deal with it
          xm1=0.d0
          xm2=0.d0
          p13=sqs/2.d0
          p23=-sqs/2.d0
          mfail=1
          return
        endif
        if(ileg.eq.1)then
          xm1=fact*sqrt(sol*shat)
          if(xm1.gt.xm1s)then
            write(*,*)'Mass # 1 too large in getxmss_madfks'
            stop
          endif
        elseif(ileg.eq.2)then
          xm2=fact*sqrt(sol*shat)
          if(xm2.gt.xm2s)then
            write(*,*)'Mass # 2 too large in getxmss_madfks'
            stop
          endif
        else
          write(*,*)'Error # 2 in getxmss_madfks'
          stop
        endif
        idone=1
        goto 100
      endif
      mfail=1-iflag
      return
      end


      subroutine put_on_MC_mshell_Hevout(p,xmi,xmj,mfail)
      implicit none
      double precision p(0:3,99),xmi,xmj
      integer mfail

      double precision dirbst(1:3),dirpj(1:3)
      double precision pipj(0:3),pibst(0:3),pjbst(0:3)

      double precision p1(0:3),p1R(0:3),pipjR(0:3)

      double precision E1o(-1:1),p1o(-1:1),E2o(-1:1),p2o(-1:1)
      integer ifail(-1:1)

      integer i,i1vec,i2vec,isol
      double precision dot,rho,threedot,Q,Q2,Q0,Qv,xm1,xm2,
     # cosphi_pipj,cosphi_p1R,costh_pipj,costh_p1R,projj,proji,
     # phi_p1R,phi_pipj,sinth_p1R,sinphi_pipj,sinphi_p1R,sinth_pipj,
     # th_p1R,th_pipj
      external dot,rho,threedot
      include 'nexternal.inc'
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c
      if(j_fks.le.nincoming)then
        write(*,*)'put_on_MC_mshell_Hevout must not be called for ISR'
        stop
      endif
c
      do i=0,3
        pipj(i)=p(i,i_fks)+p(i,j_fks)
      enddo
      Q2=dot(pipj,pipj)
      Q=sqrt(Q2)
      if(Q2.lt.(xmi+xmj)**2)then
        write(*,*)'Error in put_on_MC_mshell_Hevout',pipj(0),xmi,xmj
        stop
      endif
      Q0=pipj(0)
      Qv=rho(pipj)
      proji=threedot(pipj,p(0,i_fks))
      projj=threedot(pipj,p(0,j_fks))
      if(proji.gt.projj)then
        i1vec=i_fks
        i2vec=j_fks
        xm1=xmi
        xm2=xmj
      else
        i1vec=j_fks
        i2vec=i_fks
        xm1=xmj
        xm2=xmi
      endif
      do i=0,3
        p1(i)=p(i,i1vec)
      enddo
      call getangles(pipj,th_pipj,costh_pipj,sinth_pipj,
     #                    phi_pipj,cosphi_pipj,sinphi_pipj)
      call trp_rotate_invar(pipj,pipjR,
     #                      costh_pipj,sinth_pipj,
     #                      cosphi_pipj,sinphi_pipj)
      if( abs(pipjR(0)-Q0).gt.max(Q0,1.d0)*1.d-8 .or.
     #    abs(pipjR(3)-Qv).gt.max(Qv,1.d0)*1.d-8 )then
        write(*,*)'Error #1 in put_on_MC_mshell_Hevout',pipj,Q0,Qv
        stop
      endif
      call trp_rotate_invar(p1,p1R,
     #                      costh_pipj,sinth_pipj,
     #                      cosphi_pipj,sinphi_pipj)
      call getangles(p1R,th_p1R,costh_p1R,sinth_p1R,
     #                   phi_p1R,cosphi_p1R,sinphi_p1R)
      call xkin_2body(Q0,Qv,xm1,xm2,costh_p1R,E1o,p1o,E2o,p2o,ifail)
      if(ifail(1).eq.1.and.ifail(-1).eq.1)then
        mfail=1
        return
      elseif(ifail(1).eq.1.and.ifail(-1).eq.0)then
        isol=-1
      elseif(ifail(1).eq.0.and.ifail(-1).eq.1)then
        isol=1
      else
        if(abs(p1R(0)-E1o(1)).lt.abs(p1R(0)-E1o(-1)))then
          isol=1
        else
          isol=-1
        endif
      endif
      p1R(0)=E1o(isol)
      p1R(1)=0.d0
      p1R(2)=0.d0
      p1R(3)=p1o(isol)
c
      call rotate_invar(p1R,p1,
     #                  costh_p1R,sinth_p1R,
     #                  cosphi_p1R,sinphi_p1R)
      call rotate_invar(p1,p1,
     #                  costh_pipj,sinth_pipj,
     #                  cosphi_pipj,sinphi_pipj)
      do i=0,3
        p(i,i1vec)=p1(i)
        p(i,i2vec)=pipj(i)-p1(i)
      enddo
      mfail=0
c
      return
      end


      subroutine xkin_2body(Q0,Qv,xm1,xm2,cth1,E1,p1,E2,p2,ifail)
c Returns energies and moduli of 3-momenta of the two final-state particles
c in a two-body phase space, in a frame where the incoming momentum has
c energy equal to Q0 and modulus of 3-momentum equal to Qv.
c cth1 is the cosine of the angle between \vec{Q} and \vec{p1}
c
c There are two possible solutions, returned in arrays E*(#) and p*(#)
c with #=-1,1. If a given solution is acceptable, ifail(#)=0, 
c and ifail(#)=1 otherwise
      implicit none
      double precision Q0,Qv,xm1,xm2,cth1
      double precision E1(-1:1),p1(-1:1),E2(-1:1),p2(-1:1)
      integer ifail(-1:1)
      double precision Q02,Qv2,delta,den,arg,xA,xB
      integer i
c
      ifail( 1)=0
      ifail(-1)=0
      Q02=Q0**2
      Qv2=Qv**2
      delta=(Q02-Qv2)+xm1**2-xm2**2
      den=Q02-Qv2*cth1**2
      arg=delta**2-4*xm1**2*den
      if(arg.lt.0.d0)then
        ifail( 1)=1
        ifail(-1)=1
        return
      endif
      xA=Q0*delta/(2*den)
      xB=Qv*abs(cth1)*sqrt(arg)/(2*den)
      E1( 1)=xA+xB
      E1(-1)=xA-xB
      if(cth1.gt.0.d0)then
        if( (delta-2*Q0*E1( 1)).gt.0.d0 )ifail( 1)=1
        if( (delta-2*Q0*E1(-1)).gt.0.d0 )ifail(-1)=1
      elseif(cth1.gt.0.d0)then
        if( (delta-2*Q0*E1( 1)).lt.0.d0 )ifail( 1)=1
        if( (delta-2*Q0*E1(-1)).lt.0.d0 )ifail(-1)=1
      endif
      if(ifail(1).eq.1.and.ifail(-1).eq.1)return
c
      do i=-1,1
        if(i.eq.0)goto 111
        if(E1(i).lt.xm1)then
          ifail(i)=1
          goto 111
        endif
        p1(i)=E1(i)**2-xm1**2
        if(p1(i).lt.0.d0)then
          ifail(i)=1
          goto 111
        endif
        p1(i)=sqrt(p1(i))
c
        E2(i)=Q0-E1(i)
        if(E2(i).lt.xm2)then
          ifail(i)=1
          goto 111
        endif
        p2(i)=E2(i)**2-xm2**2
        if(p2(i).lt.0.d0)then
          ifail(i)=1
          goto 111
        endif
        p2(i)=sqrt(p2(i))
 111    continue
      enddo
c
      return
      end


      subroutine put_on_MC_mshell_Hevin(p,xmi,xm1,xm2,mfail)
      implicit none
      double precision p(0:3,99),xmi,xm1,xm2
      integer mfail
      include 'run.inc'
      include 'nexternal.inc'
      double precision chy,shy,chymo,shatp,q0,stot,xm1_r,xm2_r,
     # ybst_cm_tolab
      double precision p1(0:3),p2(0:3),xk(0:3)
      integer i 

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xmcmass(nexternal)
      common/cxmcmass/xmcmass

      double precision zaxis(1:3)
      data zaxis/0,0,1/

      double precision rho
      external rho
c
      if(j_fks.gt.nincoming)then
        write(*,*)'put_on_MC_mshell_Hevin must not be called for FSR'
        stop
      endif
c
      stot=4d0*ebeam(1)*ebeam(2)
      chy=cosh(ybst_til_tocm)
      shy=sinh(ybst_til_tocm)
      chymo=chy-1.d0
      do i=0,3
        p1(i)=p(i,1)
        p2(i)=p(i,2)
        xk(i)=p(i,i_fks)
      enddo
c        write(*,*) 'p1:',(p1(i),i=0,3)
c        write(*,*) 'p2:',(p2(i),i=0,3)
c        write(*,*) 'xk:',(xk(i),i=0,3)
      call boostwdir2(chy,shy,chymo,zaxis,p1,p1)
      call boostwdir2(chy,shy,chymo,zaxis,p2,p2)
      call boostwdir2(chy,shy,chymo,zaxis,xk,xk)
      if(abs(p1(3)+p2(3)).gt.1.d-6)then
        write(*,*)'Error #1 in put_on_MC_mshell_Hevin',p1(3),p2(3)
        write(*,*) 'p1:',(p1(i),i=0,3)
        write(*,*) 'p2:',(p2(i),i=0,3)
        write(*,*) 'xk:',(xk(i),i=0,3)
        stop
      endif
      q0=p1(0)+p2(0)-xk(0)
      xk(0)=sqrt(xmi**2+rho(xk)**2)
      shatp=(q0+xk(0))**2
      if(shatp.lt.shat*0.9999d0)then
        write(*,*)'Error #2 in put_on_MC_mshell_Hevin',shat,shatp
        stop
      elseif(shatp.le.shat)then
        shatp=shat
      endif
      if(shatp.ge.stot)then
        mfail=1
        return
      endif
      ybst_cm_tolab=ybst_til_tolab-ybst_til_tocm
      xm1_r=xm1
      xm2_r=xm2
      call getxmss_madfks(shatp,ybst_cm_tolab,
     #                    p1(3),xm1_r,p2(3),xm2_r,
     #                    p1(3),p2(3),mfail)
      if(mfail.eq.1)return
      p1(0)=sqrt(xm1_r**2+p1(3)**2)
      p2(0)=sqrt(xm2_r**2+p2(3)**2)
      xmcmass(1)=xm1_r
      xmcmass(2)=xm2_r
      call boostwdir2(chy,-shy,chymo,zaxis,p1,p1)
      call boostwdir2(chy,-shy,chymo,zaxis,p2,p2)
      call boostwdir2(chy,-shy,chymo,zaxis,xk,xk)
      do i=0,3
        p(i,1)=p1(i)
        p(i,2)=p2(i)
        p(i,i_fks)=xk(i)
      enddo
c
      return
      end


      subroutine put_on_MG_mshell()
c Puts particles back on the MadGraph mass shell
      implicit none
      include 'nexternal.inc'
      integer i
      double precision zero
      parameter (zero=0.d0)
c Masses to be given to genps_fks.f
      double precision emass(nexternal)
      common/to_mass/  emass
c Masses of the real process, as set by MadGraph
      include 'coupl.inc'
      double precision pmass(nexternal)
      include 'pmass.inc'
c
      do i=1,nexternal
        emass(i)=pmass(i)
      enddo
c
      return
      end


      subroutine write_masses_lhe_MG()
c Set masses used by MC equal to MG ones
      implicit none
      include 'nexternal.inc'
      double precision xmcmass(nexternal)
      common/cxmcmass/xmcmass
      integer i

      double precision zero
      parameter       (ZERO = 0d0)
      include 'coupl.inc'
      double precision pmass(nexternal)
      include 'pmass.inc'
c
      do i=1,nexternal
        xmcmass(i)=pmass(i)
      enddo
c
      return
      end


      subroutine get3space(pin,xlength,xdir)
      implicit none
      real*8 pin(0:3),xlength,xdir(1:3)
      integer i
c
      xlength=pin(1)**2+pin(2)**2+pin(3)**2
      if(xlength.eq.0)then
        xdir(1)=0.d0
        xdir(2)=0.d0
        xdir(3)=1.d0
      else
        xlength=sqrt(xlength)
        do i=1,3
          xdir(i)=pin(i)/xlength
        enddo
      endif
      return
      end


      subroutine put_on_MC_mshell_Hevout_old(p,xmi,xmj)
      implicit none
      double precision p(0:3,99),xmi,xmj
      double precision dirbst(1:3),dirpj(1:3)
      double precision pipj(0:3),pibst(0:3),pjbst(0:3)

      integer i
      double precision dot,Q2,Q,expy,chybst,shybst,chybstmo,tmp,
     # Emo,xpmo,pij,xmi2,xmj2
      external dot

      include 'nexternal.inc'

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c
      do i=0,3
        pipj(i)=p(i,i_fks)+p(i,j_fks)
      enddo
      Q2=dot(pipj,pipj)
      Q=sqrt(Q2)
      if(Q2.lt.(xmi+xmj)**2)then
        write(*,*)'Error in put_on_MC_mshell_Hevout',Emo,xmi,xmj
        stop
      endif
      Emo=pipj(0)
      call get3space(pipj,xpmo,dirbst)
      expy=sqrt((Emo+xpmo)/(Emo-xpmo))
      chybst=0.5d0*(expy+1.d0/expy)
      shybst=0.5d0*(expy-1.d0/expy)
      chybstmo=chybst-1.d0
      call boostwdir2(chybst,shybst,chybstmo,dirbst,p(0,j_fks),pjbst)
      call boostwdir2(chybst,shybst,chybstmo,dirbst,p(0,i_fks),pibst)
      call get3space(pjbst,tmp,dirpj)
      if(tmp.eq.0.d0)then
        write(*,*)'Parton j soft in put_on_MC_mshell_Hevout'
        stop
      endif
c
      xmj2=xmj**2
      xmi2=xmi**2
      pjbst(0)=Q/2.d0*(1+(xmj2-xmi2)/Q2)            
      pibst(0)=Q/2.d0*(1-(xmj2-xmi2)/Q2)            
      pij=Q2**2-2*Q2*(xmi2+xmj2)+(xmi2-xmj2)**2
      pij=1/2.d0*sqrt( pij/Q2 )
      do i=1,3
        pjbst(i)= dirpj(i) * pij
        pibst(i)=-dirpj(i) * pij
      enddo
c
      call boostwdir2(chybst,-shybst,chybstmo,dirbst,pibst,p(0,i_fks))
      call boostwdir2(chybst,-shybst,chybstmo,dirbst,pjbst,p(0,j_fks))
c
      if(j_fks.le.nincoming)then
        write(*,*)'ISR not yet implemented in put_on_MC_mshell_Hevout'
        stop
      endif
c
      return
      end
