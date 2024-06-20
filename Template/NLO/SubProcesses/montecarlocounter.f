      subroutine set_QCD_flows
      ! Fills ipartners, colorflow and isspecial (at the Born level)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "born_nhel.inc"
      include 'nFKSconfigs.inc'
c Nexternal is the number of legs (initial and final) al NLO, while max_bcol
c is the number of color flows at Born level
      integer i,j,k,l,k0,mothercol(2),i1(2)
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fksfather
      logical notagluon,found
      integer nglu,nsngl
      logical isspecial(max_bcol)
      common/cisspecial/isspecial
      logical spec_case

      include 'orders.inc'
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge

c
      logical is_leading_cflow(max_bcol)
      integer num_leading_cflows
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      include 'born_coloramps.inc'
c
      do i=0,nexternal-1
         ipartners(i)=0
      enddo
      do i=1,nexternal-1
         do j=0,max_bcol
            colorflow(i,j)=0
         enddo
      enddo

C What follows is true for QCD-type splittings.
C For QED-type splittings, ipartner is simply all the charged particles
C in the event except for FKSfather. In this case, all the born color
C flows are allowed

c ipartners(0): number of particles that can be colour or anticolour partner 
c   of the father, the Born-level particle to which i_fks and j_fks are 
c   attached. If one given particle is the colour/anticolour partner of
c   the father in more than one colour flow, it is counted only once
c   in ipartners(0)
c ipartners(i), 1<=i<=nexternal-1: the label (according to Born-level
c   labelling) of the i^th colour partner of the father
c
c colorflow(i,0), 1<=i<=nexternal-1: number of colour flows in which
c   the particle ipartners(i) is a colour partner of the father
c colorflow(i,j): the actual label (according to born_leshouche.inc)
c   of the j^th colour flow in which the father and ipartners(i) are
c   colour partners
c
c Example: in the process q(1) qbar(2) -> g(3) g(4), the two color flows are
c
c j=1    i    icolup(1)    icolup(2)       j=2    i    icolup(1)    icolup(2)
c        1      500           0                   1      500           0
c        2       0           501                  2       0           501
c        3      500          502                  3      502          501
c        4      502          501                  4      500          502
c
c and if one fixes for example fksfather=3, then the situation is the following.
c
c fksfather = 3
c  
c ipartners(0) = 3
c ipartners(1,2,3) = 1, 4, 2
c  
c colorflow(1,0) = 1 = number of flows where ipartners(1) = 1 is connected to 3
c colorflow(2,0) = 2 = number of flows where ipartners(2) = 4 is connected to 3
c colorflow(3,0) = 1 = number of flows where ipartners(3) = 2 is connected to 3
c colorflow(1,1) = 1 = flow where ipartners(1) = 1 is connected to 3
c colorflow(1,2) = 0 -> no other flow connecting 1 and 3
c colorflow(2,1) = 1 = first flow where ipartners(2) = 4 is connected to 3
c colorflow(2,2) = 2 = second flow where ipartners(2) = 4 is connected to 3
c colorflow(3,1) = 2 = flow where ipartners(3) = 2 is connected to 3
c colorflow(3,2) = 0 -> no other flow connecting 2 and 3
c colorflow(4,1) = 0 -> there is no fourth partner of 3
c colorflow(4,2) = 0 -> there is no fourth partner of 3
c  
c Thus
c
c ipartners(0..3) = 3, 1, 4, 2
c  
c colorflow(1,0..2) = 1, 1, 0
c colorflow(2,0..2) = 2, 1, 2
c colorflow(3,0..2) = 1, 2, 0
c colorflow(4,0..2) = 0, 0, 0

      fksfather=min(i_fks,j_fks)

c isspecial will be set equal to .true. colour flow by colour flow only
c if the father is a gluon, and another gluon will be found which is
c connected to it by both colour and anticolour
      isspecial=.false.
c
      if (split_type(qcd_pos)) then
        ! identify the color partners 
c consider only leading colour flows
        num_leading_cflows=0
        do i=1,max_bcol
          is_leading_cflow(i)=.false.
          do j=1,mapconfig(0,0)
            if(icolamp(i,j,1))then
               is_leading_cflow(i)=.true.
               num_leading_cflows=num_leading_cflows+1
               exit
            endif
          enddo
        enddo
c
        do i=1,max_bcol
          if(.not.is_leading_cflow(i))cycle
c Loop over Born-level colour flows
c nglu and nsngl are the number of gluons (except for the father) and of 
c colour singlets in the Born process, according to the information 
c stored in ICOLUP
          nglu=0
          nsngl=0
          mothercol(1)=ICOLUP(1,fksfather,i)
          mothercol(2)=ICOLUP(2,fksfather,i)
          notagluon=(mothercol(1).eq.0 .or. mothercol(2).eq.0)
c
          do j=1,nexternal-1
c Loop over Born-level particles; j is the possible colour partner of father,
c and whether this is the case is determined inside this loop
            if (j.ne.fksfather) then
c Skip father (it cannot be its own colour partner)
               if(ICOLUP(1,j,i).eq.0.and.ICOLUP(2,j,i).eq.0)
     #           nsngl=nsngl+1
               if(ICOLUP(1,j,i).ne.0.and.ICOLUP(2,j,i).ne.0)
     #           nglu=nglu+1
               if ( (j.le.nincoming.and.fksfather.gt.nincoming) .or.
     #              (j.gt.nincoming.and.fksfather.le.nincoming) ) then
c father and j not both in the initial or in the final state -- connect
c colour (1) with colour (i1(1)), and anticolour (2) with anticolour (i1(2))
                  i1(1)=1
                  i1(2)=2
               else
c father and j both in the initial or in the final state -- connect
c colour (1) with anticolour (i1(2)), and anticolour (2) with colour (i1(1))
                  i1(1)=2
                  i1(2)=1
               endif
               do l=1,2
c Loop over colour and anticolour of father
                  found=.false.
                  if( ICOLUP(i1(l),j,i).eq.mothercol(l) .and.
     &                ICOLUP(i1(l),j,i).ne.0 ) then
c When ICOLUP(i1(l),j,i) = mothercol(l), the colour (if i1(l)=1) or
c the anticolour (if i1(l)=2) of particle j is connected to the
c colour (if l=1) or the anticolour (if l=2) of the father
                     k0=-1
                     do k=1,ipartners(0)
c Loop over previously-found colour/anticolour partners of father
                        if(ipartners(k).eq.j)then
                           if(found)then
c Safety measure: if this condition is met, it means that there exist
c k1 and k2 such that ipartners(k1)=ipartners(k2). This is thus a bug,
c since ipartners() is the list of possible partners of father, where each
c Born-level particle must appears at most once
                              write(*,*)'Error #1 in set_matrices'
                              write(*,*)i,j,l,k
                              stop
                           endif
                           found=.true.
                           k0=k
                        endif
                     enddo
                     if (.not.found) then
                        ipartners(0)=ipartners(0)+1
                        ipartners(ipartners(0))=j
                        k0=ipartners(0)
                     endif
c At this point, k0 is the k0^th colour/anticolour partner of father.
c Therefore, ipartners(k0)=j
                     if(k0.le.0.or.ipartners(k0).ne.j)then
                        write(*,*)'Error #2 in set_matrices'
                        write(*,*)i,j,l,k0,ipartners(k0)
                        stop
                     endif
                     spec_case=l.eq.2 .and. colorflow(k0,0).ge.1 .and.
     &                    colorflow(k0,colorflow(k0,0)).eq.i 
                     if (.not.spec_case)then
c Increase by one the number of colour flows in which the father is
c (anti)colour-connected with its k0^th partner (according to the
c list defined by ipartners)
                        colorflow(k0,0)=colorflow(k0,0)+1
c Store the label of the colour flow thus found
                        colorflow(k0,colorflow(k0,0))=i
                     elseif (spec_case)then
c Special case: father and ipartners(k0) are both gluons, connected
c by colour AND anticolour: the number of colour flows was overcounted
c by one unit, so decrease it
                         if( notagluon .or.
     &                       ICOLUP(i1(1),j,i).eq.0 .or.
     &                       ICOLUP(i1(2),j,i).eq.0 )then
                            write(*,*)'Error #3 in set_matrices'
                            write(*,*)i,j,l,k0,i1(1),i1(2)
                            stop
                         endif
                         colorflow(k0,colorflow(k0,0))=i
                         isspecial(i)=.true.
                     endif
                  endif
               enddo
            endif
         enddo
         if( ((nglu+nsngl).gt.(nexternal-2)) .or.
     #       (isspecial(i).and.(nglu+nsngl).ne.(nexternal-2)) )then
           write(*,*)'Error #4 in set_matrices'
           write(*,*)isspecial(i),nglu,nsngl
           stop
          endif
        enddo

      else if (split_type(qed_pos)) then
        ! do nothing, the partner will be assigned at run-time 
        ! (it is kinematics-dependent)
        continue
      endif
      call check_QCD_flows(notagluon)
      return
      end



      subroutine check_QCD_flows(notagluon)
      implicit none
      include "nexternal.inc"
      include "born_nhel.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      integer i,j,ipart,iflow,ntot,ithere(1000)
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fksfather
      logical notagluon
      logical isspecial(max_bcol)
      common/cisspecial/isspecial

      include 'orders.inc'
      logical split_type(nsplitorders) 
      common /c_split_type/split_type

      logical is_leading_cflow(max_bcol)
      integer num_leading_cflows
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows
c
      fksfather=min(i_fks,j_fks)
      if(ipartners(0).gt.nexternal-1)then
        write(*,*)'Error #1 in check_QCD_flows',ipartners(0)
        stop
      endif
c
      if (split_type(QCD_pos)) then
      ! these tests only apply for QCD-type splittings
        do i=1,ipartners(0)
          ipart=ipartners(i)
          if( ipart.eq.fksfather .or.
     #        ipart.le.0 .or. ipart.gt.nexternal-1 .or.
     #        ( abs(particle_type(ipart)).ne.3 .and.
     #          particle_type(ipart).ne.8 ) )then
            write(*,*)'Error #2 in check_QCD_flows',i,ipart,
     #  particle_type(ipart)
            stop
          endif
        enddo
c
        do i=1,nexternal-1
          ithere(i)=1
        enddo
        do i=1,ipartners(0)
          ipart=ipartners(i)
          ithere(ipart)=ithere(ipart)-1
          if(ithere(ipart).lt.0)then
            write(*,*)'Error #3 in check_QCD_flows',i,ipart
            stop
          endif
        enddo
c
c ntot is the total number of colour plus anticolour partners of father
        ntot=0
        do i=1,ipartners(0)
          ntot=ntot+colorflow(i,0)
c
          if( colorflow(i,0).le.0 .or.
     #        colorflow(i,0).gt.max_bcol )then
            write(*,*)'Error #4 in check_QCD_flows',i,colorflow(i,0)
            stop
          endif
c
          do j=1,max_bcol
            ithere(j)=1
          enddo
          do j=1,colorflow(i,0)
            iflow=colorflow(i,j)
            ithere(iflow)=ithere(iflow)-1
            if(ithere(iflow).lt.0)then
              write(*,*)'Error #5 in check_QCD_flows',i,j,iflow
              stop
            endif
          enddo
c
        enddo
c
        if( (notagluon.and.ntot.ne.num_leading_cflows) .or.
     #    ( (.not.notagluon).and.
     #      ( (.not.isspecial(1)).and.ntot.ne.(2*num_leading_cflows) .or.
     #        (isspecial(1).and.ntot.ne.num_leading_cflows) ) ) )then
         write(*,*)'Error #6 in check_QCD_flows',
     #     notagluon,ntot,num_leading_cflows,max_bcol
          stop
        endif
c
        if(num_leading_cflows.gt.max_bcol)then
          write(*,*)'Error #7 in check_QCD_flows',
     #     num_leading_cflows,max_bcol
          stop
        endif

      else if (split_type(QED_pos)) then
        ! write here possible checks for QED-type splittings
        continue
      endif
      return
      end


      subroutine set_QED_flows(pp)
      use process_module
      use kinematics_module
      implicit none
      include 'nexternal.inc'
      double precision pp(0:3, nexternal)

      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision particle_charge(nexternal)
      common /c_charges/particle_charge

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision pmass(nexternal)
      double precision zero
      parameter (zero=0d0)

      include 'genps.inc'
      include "born_nhel.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      include 'coupl.inc'
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
c
c     Shower MonteCarlo
c     

      logical found
      logical same_state
      double precision ppmin, ppnow
      integer partner
      integer i,j
      double precision chargeprod

      include 'pmass.inc'
      
      found=.false.
      ppmin=1d99

      if (shower_mc_mod(1:7).eq.'PYTHIA8') then
        ! this should follow what is done in TimeShower::setupQEDdip
        ! first, look for the lowest-mass same- (opposite-)flavour pair of
        ! particles in the opposite (same) state of the system
        do j=1,nexternal
          if (j.ne.fksfather.and.j.ne.i_fks) then
            same_state = (j.gt.nincoming.and.fksfather.gt.nincoming).or.
     $                   (j.le.nincoming.and.fksfather.le.nincoming)

            if ((pdg_type(j).eq.pdg_type(fksfather).and..not.same_state).or. 
     $          (pdg_type(j).eq.-pdg_type(fksfather).and.same_state)) then

              ppnow=dot(pp(0,fksfather),pp(0,j)) - pmass(fksfather)*pmass(j)
              if (ppnow.lt.ppmin) then
                found=.true.
                partner=j
              endif
            endif
          endif
        enddo
        
        ! if no partner has been found, then look for the
        ! lowest-mass/chargeprod pair
        if (.not.found) then
          do j=1,nexternal
            if (j.ne.fksfather.and.j.ne.i_fks) then
              if (particle_charge(fksfather).ne.0d0.and.particle_charge(j).ne.0d0) then
                ppnow=dot(pp(0,fksfather),pp(0,j)) - pmass(fksfather)*pmass(j) / 
     $            (particle_charge(fksfather) * particle_charge(j))
                if (ppnow.lt.ppmin) then
                  found=.true.
                  partner=j
                endif
              endif
            endif
          enddo
        endif

        ! if no partner has been found, then look for the
        ! lowest-mass pair
        if (.not.found) then
          do j=1,nexternal
            if (j.ne.fksfather.and.j.ne.i_fks) then
              ppnow=dot(pp(0,fksfather),pp(0,j)) - pmass(fksfather)*pmass(j) 
              if (ppnow.lt.ppmin) then
                found=.true.
                partner=j
              endif
            endif
          enddo
        endif

      else
        ! other showers need to be implemented
        write(*,*) 'ERROR in set_QED_flows, not implemented', shower_mc_mod
        stop 1
      endif

      if (.not.found) then
        write(*,*) 'ERROR in set_QED_flows, no parthern found'
        stop 1
      endif

      ! now, set ipartners
      ipartners(0) = 1
      ipartners(ipartners(0)) = partner
      ! all color flows have to be included here
      colorflow(ipartners(0),0)= max_bcol
      do i = 1, max_bcol
        colorflow(ipartners(0),i)=i
      enddo
      return
      end

      subroutine compute_xmcsubt_for_checks(pp,xi_i_fks,y_ij_fks,wgt)
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include "nexternal.inc"
c$$$      include 'madfks_mcatnlo.inc'
      include 'run.inc'
      include 'born_nhel.inc'
      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
      double precision xmc,xrealme,gfactsf,gfactcl,probne,sumMCsec
      double precision z(nexternal),ddum,dummy
      integer nofpartners,idum,ione,iord
      logical lzone(nexternal),flagmc

      ! amp split stuff
      include 'orders.inc'
      integer iamp
      double precision amp_split_mc(amp_split_size)
      common /to_amp_split_mc/amp_split_mc
      double precision amp_split_gfunc(amp_split_size)
      common /to_amp_split_gfunc/amp_split_gfunc
      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
     $                 amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
      common /to_amp_split_bornbars/amp_split_bornbars,
     $                              amp_split_bornbarstilde
      logical split_type(nsplitorders) 
      common /c_split_type/split_type

      integer npartner,cflows
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      logical first_MCcnt_call,is_pt_hard
      common/cMCcall/first_MCcnt_call,is_pt_hard

      double precision xkern(2),xkernazi(2),factor,N_p
      double precision bornbars(max_bcol,nsplitorders),
     $     bornbarstilde(max_bcol,nsplitorders)
      double precision emsca_a(nexternal,nexternal)
     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
     $     ,nexternal) ,scalemin_a(nexternal,nexternal)
     $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
     $     ,nexternal)
      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2
     $     ,scalemin_a,scalemax_a,emscwgt_a
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision evnt_wgt
      integer i, j,iord_val
      double precision mu_r
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      double precision p_read(0:4,2*nexternal-3), wgt_read
      integer npart
      double precision MCsec(nexternal,max_bcol)
      logical isspecial(max_bcol)
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      common/cisspecial/isspecial
!     common block used to make the (scalar) reference scale partner
!     dependent in case of delta
      integer cur_part
      common /to_ref_scale/cur_part
      double precision smin,smax,ptresc,emscafun,qMC
      first_MCcnt_call=.true.
      is_pt_hard=.false.
      MCsec(1:nexternal,1:max_bcol)=0d0
      sumMCsec=0d0
      amp_split_mc(1:amp_split_size)=0d0
      if (mcatnlo_delta) then
c Call assign_emsca_array uniquely to fill emscwgt_a, to be used to
c define 'factor'.  This damping 'factor' is used only here, and not in
c the following.  A subsequent call to assign_emsca_array, in
c complete_xmcsubt, will set emsca_a and related quantities.  This means
c that, event by event, MC damping factors D(mu_ij) corresponding to the
c emscwgt_a determined now, are not computed with the actual mu_ij
c scales used as starting scales (which are determined in the subsequent
c call to assign_emsca_array), which however is fine statistically
c$$$         call assign_emsca_array(pp,xi_i_fks,y_ij_fks)
      endif         
      do npartner=1,ipartners(0)
         cur_part=ipartners(npartner)
         call xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne
     $        ,nofpartners,lzone,flagmc,z,xkern,xkernazi
     $        ,bornbars,bornbarstilde,npartner)
         if(is_pt_hard)exit
         if (.not.mcatnlo_delta) then
            smin=shower_scale_nbody_min(cur_part,fksfather)
            smax=shower_scale_nbody_max(cur_part,fksfather)
            qMC=get_qMC(xi_i_fks,y_ij_fks)
            ptresc=(qMC-smin)/(smax-smin)
            factor=1d0-emscafun(ptresc,1d0)
         else
c min(i_fks,j_fks) is the mother of the FKS pair
            factor=emscwgt_a(min(i_fks,j_fks),cur_part)
         endif
         do cflows=1,max_bcol
            if (colorflow(npartner,cflows).eq.0) cycle
            if(isspecial(cflows)) then
               N_p=2d0
            else
               N_p=1d0
            endif
            ione=0
            do iord = 1, nsplitorders
               if (.not.split_type(iord) .or.
     $              (iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
               if (iord.eq.qcd_pos) then
                  iord_val=1
               elseif(iord.eq.qed_pos) then
                  iord_val=2
               endif
               ione=ione+1
               MCsec(npartner,colorflow(npartner,cflows))=factor
     $              *(xkern(iord_val)*N_p*bornbars(colorflow(npartner
     $              ,cflows),iord)+xkernazi(iord_val)*N_p
     $              *bornbarstilde(colorflow(npartner,cflows),iord))
               amp_split_mc(1:amp_split_size) =
     $              amp_split_mc(1:amp_split_size)+factor
     $              *(xkern(iord_val)*N_p
     $              *amp_split_bornbars(1:amp_split_size
     $              ,colorflow(npartner,cflows),iord)+xkernazi(iord_val)
     $              *N_p *amp_split_bornbarstilde(1:amp_split_size
     $              ,colorflow(npartner,cflows),iord))
            enddo
            if (ione.ne.1) then
               write (*,*) 'Error: incompatible split orders in '/
     $              /'compute_xmcsubt_for_checks',ione
               stop 1
            endif
            sumMCsec=sumMCsec+MCsec(npartner,colorflow(npartner
     $           ,cflows))
         enddo
      enddo
      call xmcsubtME(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,xrealme)
      wgt=sumMCsec+xrealme
      do iamp=1, amp_split_size
        amp_split_mc(iamp) = amp_split_mc(iamp) + amp_split_gfunc(iamp)
      enddo
      return
      end

      
      subroutine compute_xmcsubt_complete(p,probne,gfactsf,gfactcl
     $     ,flagmc,lzone,z_shower,nofpartners,xmcxsec)
      use scale_module
      implicit none
      include 'nexternal.inc'
c$$$      include 'madfks_mcatnlo.inc'
      include 'born_nhel.inc'
      include 'run.inc'
      include 'orders.inc'
      integer npartner,nofpartners,cflows,idum,ione,iord,iord_val
      logical lzone(nexternal),flagmc
      double precision bornbars(max_bcol,nsplitorders),
     $     bornbarstilde(max_bcol,nsplitorders)
      double precision p(0:3,nexternal),probne,z_shower(nexternal)
     $     ,xmcxsec(nexternal),xkern(2),xkernazi(2),damping,N_p
     $     ,MCsec(nexternal,max_bcol),sumMCsec
     $     ,xmcxsec2(max_bcol),gfactsf,gfactcl,ddum
      double precision emsca_a(nexternal,nexternal)
     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
     $     ,nexternal) ,scalemin_a(nexternal,nexternal)
     $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
     $     ,nexternal)
      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2
     $     ,scalemin_a,scalemax_a,emscwgt_a
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      logical isspecial(max_bcol)
      common/cisspecial/isspecial
      logical first_MCcnt_call,is_pt_hard
      common/cMCcall/first_MCcnt_call,is_pt_hard
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
     $                 amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
      common /to_amp_split_bornbars/amp_split_bornbars,
     $                              amp_split_bornbarstilde
      double precision amp_split_xmcxsec(amp_split_size,nexternal)
      common /to_amp_split_xmcxsec/amp_split_xmcxsec
      double precision amp_split_mc(amp_split_size)
      common /to_amp_split_mc/amp_split_mc
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
!     common block used to make the (scalar) reference scale partner
!     dependent in case of delta
      integer cur_part
      common /to_ref_scale/cur_part
      double precision smin,smax,ptresc,compute_damping_weight,qMC
c -- call to MC counterterm functions
      first_MCcnt_call=.true.
      is_pt_hard=.false.
      xmcxsec(1:nexternal)=0d0
      xmcxsec2(1:max_bcol)=0d0
      MCsec(1:nexternal,1:max_bcol)=0d0
      sumMCsec=0d0
      amp_split_xmcxsec(1:amp_split_size,1:nexternal)=0d0
      do npartner=1,ipartners(0)
         cur_part=ipartners(npartner)
         call xmcsubt(p,xi_i_fks_ev,y_ij_fks_ev,gfactsf,gfactcl,probne
     $        ,nofpartners,lzone,flagmc,z_shower,xkern,xkernazi
     $        ,bornbars,bornbarstilde,npartner)
         if(is_pt_hard)exit
         damping=compute_damping_weight(cur_part,fks_father,xi_i_fks
     $        ,y_ij_fks)
         do cflows=1,max_bcol
            if (colorflow(npartner,cflows).eq.0) cycle
            if(isspecial(cflows)) then
               N_p=2d0
            else
               N_p=1d0
            endif
            ione=0
            do iord = 1, nsplitorders
               if (.not.split_type(iord) .or.
     $              (iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
               if (iord.eq.qcd_pos) then
                  iord_val=1
               elseif(iord.eq.qed_pos) then
                  iord_val=2
               endif
               ione=ione+1
               MCsec(npartner,colorflow(npartner,cflows))=damping
     $              *(xkern(iord_val)*N_p*bornbars(colorflow(npartner
     $              ,cflows),iord)+xkernazi(iord_val)*N_p
     $              *bornbarstilde(colorflow(npartner,cflows),iord))
               amp_split_mc(1:amp_split_size) =
     $              amp_split_mc(1:amp_split_size)+damping
     $              *(xkern(iord_val)*N_p
     $              *amp_split_bornbars(1:amp_split_size
     $              ,colorflow(npartner,cflows),iord)+xkernazi(iord_val)
     $              *N_p *amp_split_bornbarstilde(1:amp_split_size
     $              ,colorflow(npartner,cflows),iord))
            enddo
            if (ione.ne.1) then
               write (*,*) 'Error: incompatible split orders in '/
     $              /'compute_xmcsubt_complete',ione
               stop 1
            endif
            xmcxsec(npartner)=xmcxsec(npartner)+MCsec(npartner
     $           ,colorflow(npartner,cflows))
            xmcxsec2(colorflow(npartner,cflows))=
     $           xmcxsec2(colorflow(npartner,cflows))+MCsec(npartner
     $           ,colorflow(npartner,cflows))
            sumMCsec=sumMCsec+MCsec(npartner,colorflow(npartner
     $           ,cflows))
         enddo
      enddo
! check the MC cross sections are positive:
      call check_positivity_MCxsec(sumMCsec,xmcxsec,xmcxsec2)
      if (mcatnlo_delta) then
! compute and include the Delta Sudakov:
         if(.not.is_pt_hard) call complete_xmcsubt(p,lzone,xmcxsec
     $        ,xmcxsec2,MCsec,probne)
      else
! assign emsca on statistical basis (don't need flow here): 
         if(.not.is_pt_hard) call assign_emsca_and_flow_statistical(
     $        xmcxsec,xmcxsec2,MCsec,lzone,idum,ddum)
! include the bogus no-emission probability:
         xmcxsec(1:ipartners(0))=xmcxsec(1:ipartners(0))*probne
         amp_split_xmcxsec(1:amp_split_size,1:ipartners(0))=
     $        amp_split_xmcxsec(1:amp_split_size,1:ipartners(0))*probne
      endif
      if (btest(Mccntcalled,4)) then
         write (*,*) 'Fifth bit of MCcntcalled should not '/
     $        /'have been set yet',MCcntcalled
         stop 1
      endif
      if (is_pt_hard) MCcntcalled=MCcntcalled+16
      return
      end

      double precision function compute_damping_weight(cur_part
     $     ,fks_father,xi_i_fks,y_ij_fks)
      use scale_module
      implicit none
      integer :: cur_part,fks_father
      double precision :: xi_i_fks,y_ij_fks,emscafun,smin,smax,qMC
     $     ,ptresc
      smin=shower_scale_nbody_min(cur_part,fks_father)
      smax=shower_scale_nbody_max(cur_part,fks_father)
      qMC=get_qMC(xi_i_fks,y_ij_fks)
      ptresc=(qMC-smin)/(smax-smin)
      compute_damping_weight=1d0-emscafun(ptresc,1d0)
      end

      subroutine check_positivity_MCxsec(sumMCsec,xmcxsec,xmcxsec2)
      implicit none
      include 'nexternal.inc'
      include "born_nhel.inc"
      double precision tiny
      parameter (tiny=1d-7)
      integer cflows,npartner
      double precision sumMCsec,xmcxsec2(max_bcol),xmcxsec(nexternal)
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
c     positivity check
      if(sumMCsec.lt.0d0)then
         write(*,*)'Negative sumMCsec',sumMCsec
         stop 1
      elseif(sumMCsec.gt.0d0) then
         do cflows=1,max_bcol
            do npartner=1,ipartners(0)
               if(xmcxsec(npartner)/sumMCsec.le.-tiny)then
                  write(*,*)'Negative xmcxsec',npartner
     $                 ,xmcxsec(npartner)
                  stop 1
               elseif(xmcxsec(npartner).le.0d0)then
                  xmcxsec(npartner)=0d0
               endif
               if(xmcxsec2(cflows)/sumMCsec.le.-tiny)then
                  write(*,*)'Negative xmcxsec2',cflows,xmcxsec2(cflows)
                  stop 1
               elseif(xmcxsec2(cflows).le.0d0)then
                  xmcxsec2(cflows)=0d0
               endif
            enddo
         enddo
      endif
      end
      

      subroutine xmcsubtME(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,wgt)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      double precision pp(0:3,nexternal),gfactsf,gfactcl,wgt,wgts,wgtc,wgtsc
      double precision xi_i_fks,y_ij_fks

      double precision zero,one
      parameter (zero=0d0)
      parameter (one=1d0)

      integer izero,ione,itwo
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

c Particle types (=colour) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m

      logical is_aorg(nexternal)
      common /c_is_aorg/is_aorg
      ! amp split stuff
      include 'orders.inc'
      integer iamp
      double precision amp_split_gfunc(amp_split_size)
      common /to_amp_split_gfunc/amp_split_gfunc
      double precision amp_split_s(amp_split_size), 
     $                 amp_split_c(amp_split_size), 
     $                 amp_split_sc(amp_split_size)
      double precision pmass(nexternal)
      include "pmass.inc"
c
      wgt=0d0
      do iamp=1, amp_split_size
        amp_split_gfunc(iamp) = 0d0
      enddo
      ! this contribution is needed only for i_fks being a gluon/photon
      ! (soft limit)
      if (is_aorg(i_fks))then
c i_fks is gluon/photon
         call set_cms_stuff(izero)
         call sreal(p1_cnt(0,1,0),zero,y_ij_fks,wgts)
         do iamp=1, amp_split_size
           amp_split_s(iamp) = amp_split(iamp)
         enddo
         call set_cms_stuff(ione)
         call sreal(p1_cnt(0,1,1),xi_i_fks,one,wgtc)
         do iamp=1, amp_split_size
           amp_split_c(iamp) = amp_split(iamp)
         enddo
         call set_cms_stuff(itwo)
         call sreal(p1_cnt(0,1,2),zero,one,wgtsc)
         do iamp=1, amp_split_size
           amp_split_sc(iamp) = amp_split(iamp)
         enddo
         wgt=wgts+(1-gfactcl)*(wgtc-wgtsc)
         wgt=wgt*(1-gfactsf)
         do iamp = 1, amp_split_size
           amp_split_gfunc(iamp) = amp_split_s(iamp)+(1-gfactcl)*(amp_split_c(iamp)-amp_split_sc(iamp))
           amp_split_gfunc(iamp) = amp_split_gfunc(iamp)*(1-gfactsf)
         enddo
      elseif (abs(i_type).ne.3.and.ch_i.eq.0d0)then
         ! we should never get here
         write(*,*) 'FATAL ERROR #1 in xmcsubtME',i_type,i_fks
         stop
      endif
c
      return
      end




!!!! WE ARE HERE !!!!

      

c Main routine for MC counterterms. Now to be called inside a loop
c over colour partners
      subroutine xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
     &     nofpartners,lzone,flagmc,z,xkern,xkernazi,
     &     bornbars,bornbarstilde,npartner)
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'orders.inc'
      include 'fks_powers.inc'
      include 'coupl.inc'
! arguments:
      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,gfactsf,gfactcl
     $     ,probne,z(nexternal),xkern(2),xkernazi(2),bornbars(max_bcol
     $     ,nsplitorders),bornbarstilde(max_bcol,nsplitorders)
      integer nofpartners,npartner
      logical lzone(nexternal),flagmc

! local
      double precision ztmp,xitmp,xjactmp,gfactazi,qMC,delta,E0sq
     $     ,PY6PTweight,pmass(nexternal),xi,xjac
! external
      double precision bogus_probne_fun,gfunction,zHW6,xiHW6
     $     ,xjacHW6
      external bogus_probne_fun,gfunction,zHW6,xiHW6,xjacHW6
! parameters      
      double precision ymin,zero
      parameter (ymin=0.9d0)
      parameter(zero=0d0)
! common
      logical first_MCcnt_call,is_pt_hard
      common/cMCcall/first_MCcnt_call,is_pt_hard
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      save

      include "pmass.inc"

c Initialise if first time
      if(.not.first_MCcnt_call)goto 222
      if (split_type(QED_pos)) then
         ! QED partners are dynamically found
         call set_QED_flows(pp)
      endif
      flagmc   = .false.
      ztmp     = 0d0
      xitmp    = 0d0
      xjactmp  = 0d0
      gfactazi = 0d0
      nofpartners = ipartners(0)

      qMC=get_qMC(xi_i_fks,y_ij_fks)

c New or standard MC@NLO formulation
      probne=bogus_probne_fun(qMC)

c Call barred Born and assign shower scale
      call get_mbar(pp,y_ij_fks,ileg,bornbars,bornbarstilde)

c Distinguish ISR and FSR
      if(ileg.le.2)then
         delta=min(1d0,deltaI)
      elseif(ileg.ge.3)then
         delta=min(1d0,deltaO)
      endif

c G-function parameters 
      gfactsf=gfunction(x,alsf,besf,2d0)
      if(abs(i_type).eq.3)gfactsf=1d0 ! if fks parton is quark, soft limit is finite
      gfactcl=gfunction(y_ij_fks,alsf,-(1d0-ymin),1d0)
      if(alazi.lt.0d0)gfactazi=1-gfunction(y_ij_fks,-alazi,beazi,delta)

      if (btest(MCcntcalled,2)) then
         write (*,*) 'Third bit of MCcntcalled should not be set yet'
     $        ,MCcntcalled
         stop 1
      endif

      MCcntcalled=MCcntcalled+4
      
c Shower variables (all except HW6, since that one depends on the
c partner)
      call get_shower_variables(ztmp,xitmp,xjactmp)
      
      first_MCcnt_call=.false.
 222  continue
c Main loop over colour partners used to begin here
      E0sq=dot(p_born(0,fksfather),
     $                   p_born(0,ipartners(npartner)))
      if(E0sq.lt.0d0)then
         write(*,*)'Error in xmcsubt: negative E0sq'
         write(*,*)E0sq,ileg,npartner
         stop
      endif
      if(shower_mc_mod(1:7).eq.'HERWIG6')then
         z(npartner)=zHW6(E0sq)
         xi=xiHW6(E0sq,z(npartner))
         xjac=xjacHW6(E0sq,xi,z(npartner))
      else
         z(npartner)=ztmp
         xi=xitmp
         xjac=xjactmp
      endif
c Compute dead zones
      call get_dead_zone(z(npartner),xi,qMC
     $     ,ipartners(npartner),fksfather,lzone(npartner),PY6PTweight)

c Compute MC subtraction terms
      if(lzone(npartner))then
         if(.not.flagmc)flagmc=.true.
         call limits(xi_i_fks,y_ij_fks)
         call compute_spitting_kernels(xkern,xkernazi,z(npartner)
     $        ,xi,xjac)
      else
        xkern(1:2)=0d0
        xkernazi(1:2)=0d0
      endif
c
      xkern(1:2)=xkern(1:2)*gfactsf
      xkernazi(1:2)=xkernazi(1:2)*gfactazi*gfactsf
      if (shower_mc_mod(1:9).eq.'PYTHIA6PT') then
         xkern(1:2)=xkern(1:2)*PY6PTweight
         xkernazi(1:2)=xkernazi(1:2)*PY6PTweight
      endif

c Main loop over colour partners used to end here
      return
      end




      subroutine compute_spitting_kernels(xkern,xkernazi,z,xi,xjac)
      implicit none
      double precision xkern(1:2),xkernazi(1:2),z,xi,xjac
      double precision tiny
      parameter (tiny=1d-6)
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      xkern(1:2)    = 0d0
      xkernazi(1:2) = 0d0
      if( (ileg.ge.3 .and.
     $     (m_type.eq.8.or.(m_type.eq.1.and.dabs(ch_m).lt.tiny))) .or.
     $    (ileg.le.2 .and.
     $     (j_type.eq.8.or.(j_type.eq.1.and.dabs(ch_j).lt.tiny))) )then
         if(i_type.eq.8)then
c g->gg, go->gog (icode=1)
            call compute_splitting_kernel_icode1(xkern,xkernazi,z,xi)
         elseif(abs(i_type).eq.3.or.(i_type.eq.1.and.dabs(ch_i).gt.tiny))then
c g->qq, a->qq, a->ee (icode=2)
            call compute_splitting_kernel_icode2(xkern,xkernazi,z,xi)
         else
            write(*,*)'Error 1 in xmcsubt: unknown particle type'
            write(*,*)i_type
            stop
         endif
      elseif( (ileg.ge.3 .and.
     $        (abs(m_type).eq.3.or.(m_type.eq.1.and.dabs(ch_m).gt.tiny))) .or.
     $        (ileg.le.2 .and.
     $        (abs(j_type).eq.3.or.(j_type.eq.1.and.dabs(ch_j).gt.tiny))) )
     $        then
         if(abs(i_type).eq.3.or.(i_type.eq.1.and.dabs(ch_i).gt.tiny))then
c q->gq, q->aq, e->ae (icode=3)
            call compute_splitting_kernel_icode3(xkern,xkernazi,z,xi)
         elseif(i_type.eq.8.or.(i_type.eq.1.and.dabs(ch_i).lt.tiny))then
c q->qg, q->qa, sq->sqg, sq->sqa, e->ea (icode=4)
            call compute_splitting_kernel_icode4(xkern,xkernazi,z,xi)
         else
            write(*,*)'Error 2 in xmcsubt: unknown particle type'
            write(*,*)i_type
            stop
         endif
      else
         write(*,*)'Error 3 in xmcsubt: unknown particle type'
         write(*,*)j_type,i_type
         stop
      endif
      if (non_limit) then
         ! If limit, the jacobian is already included in the kernel
         ! (through the subroutines 'compute_splitting_kernel_icode)
         xkern(1:2)    = xkern(1:2)*xjac
         xkernazi(1:2) = xkernazi(1:2)*xjac
      endif
      return
      end

      subroutine limits(xi_i_fks,y_ij_fks)
      implicit none
      double precision tiny
      logical softtest,colltest
      common/sctests/softtest,colltest
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
c Logical variables to control the IR limits:
c one can remove any reference to xi_i_fks
      tiny = 1d-6
      if (softtest.or.colltest)tiny = 1d-12
      limit = 1-y_ij_fks.lt.tiny .and. xi_i_fks.ge.tiny ! collinear (and not soft)
      non_limit = xi_i_fks.ge.tiny  ! not collinear (and not soft)
      ! (Note, if soft, we should use the G-functions and not the MC subtraction terms)
      end
      
      double precision function xfact_ileg12(N_p)
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg12=(1d0-yi)*(1d0-x)/x * 4d0/(s*N_p)
      end

      double precision function xfact_ileg3(N_p)
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg3=(2d0-(1d0-x)*(1d0-(kn0/kn)*yj))/
     &     kn*knbar*(1d0-x)*(1d0-yj) * 2d0/(s*N_p)
      end

      double precision function xfact_ileg4(N_p)
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg4=(2d0-(1d0-x)*(1d0-yj))/
     &     xij*(1d0-xm12/s)*(1d0-x)*(1d0-yj) * 2d0/(s*N_p)
      end

      subroutine compute_splitting_kernel_icode1(xkern,xkernazi,z,xi)
      use kinematics_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      double precision xfact_ileg12,xfact_ileg3,xfact_ileg4
      external xfact_ileg12,xfact_ileg3,xfact_ileg4
      integer N_P
      double precision vca,one
      parameter (vca=3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      s=shat_n1
c g->gg, go->gog (icode=1)
      if(ileg.le.2)then
         N_p=2
         if(limit)then
            xkern(1)=(g**2/N_p)*8*vca*(1-x*(1-x))**2/(s*x**2)
            xkernazi(1)=-(g**2/N_p)*16*vca*(1-x)**2/(s*x**2)
            xkern(2)=0d0
            xkernazi(2)=0d0
         elseif(non_limit)then
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
            call Qterms_reduced_spacelike(m_type,i_type,ch_m,ch_i,one,z
     $           ,Q)
            xkernazi(1:2)=xfact*Q(1:2)/(xi*(1-z))
            if (xkern(2).ne.0d0 .or.xkernazi(2).ne.0d0) then
               write(*,*) 'ERROR#1, g->gg splitting QED' /
     $              /'contributions should be 0', xkern,
     $              xkernazi
               stop
            endif
         else
! We are soft. The G-function will take care of this.
            continue
         endif
c     
      elseif(ileg.eq.3)then
         N_p=2
         if(non_limit)then
            xfact=xfact_ileg3(N_p)
            call AP_reduced_SUSY(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.4)then
         N_p=2
         if(limit)then
            xkern(1)=(g**2/N_p)*( 8*vca*
     &           (s**2*(1-(1-x)*x)-s*(1+x)*xm12+xm12**2)**2 )/
     &           ( s*(s-xm12)**2*(s*x-xm12)**2 )
            xkernazi(1)=-(g**2/N_p)*(16*vca*s*(1-x)**2)/((s-xm12)**2)
            xkern(2)=0d0
            xkernazi(2)=0d0
         elseif(non_limit)then
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
            call Qterms_reduced_timelike(j_type,i_type,ch_m,ch_i,one,z
     $           ,Q)
            xkernazi(1:2)=xfact*Q(1:2)/(xi*(1-z))
            if (xkern(2).ne.0d0 .or.xkernazi(2).ne.0d0) then
               write(*,*) 'ERROR#1, g->gg splitting QED' /
     $              /'contributions should be 0', xkern,
     $              xkernazi
               stop
            endif
         else
! We are soft. The G-function will take care of this.
            continue
         endif
      endif
      end
      
      subroutine compute_splitting_kernel_icode2(xkern,xkernazi,z,xi)
      use kinematics_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      double precision xfact_ileg12,xfact_ileg4
      external xfact_ileg12,xfact_ileg4
      integer N_p
      double precision vtf,one
      parameter (vtf=1d0/2d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      s=shat_n1
c g->qq, a->qq, a->ee (icode=2)
      if(ileg.le.2)then
         N_p=1
         if(limit)then
            xkern(1)=(g**2/N_p)*4*vtf*(1-x)*((1-x)**2+x**2)/(s*x)
            xkern(2)=xkern(1) * dble(gal(1))**2 / g**2 * 
     &           ch_i**2 * abs(i_type) / vtf
         elseif(non_limit)then
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.4)then
         N_p=2
         if(limit)then
            xkern(1)=(g**2/N_p)*( 4*vtf*(1-x)*
     &           (s**2*(1-2*(1-x)*x)-2*s*x*xm12+xm12**2) )/
     &           ( (s-xm12)**2*(s*x-xm12) )
            xkern(2)=xkern(1) * dble(gal(1))**2 / g**2 *
     &           ch_i**2 * abs(i_type) / vtf
            xkernazi(1)=(g**2/N_p)*(16*vtf*s*(1-x)**2)/((s-xm12)**2)
            xkernazi(2)=xkernazi(1) * dble(gal(1))**2 / g**2 *
     &           ch_i**2 * abs(i_type) / vtf
         elseif(non_limit)then
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
            call Qterms_reduced_timelike(j_type,i_type,ch_m,ch_i,one,z
     $           ,Q)
            xkernazi(1:2)=xfact*Q(1:2)/(xi*(1-z))
         endif
      endif
      end
      
      subroutine compute_splitting_kernel_icode3(xkern,xkernazi,z,xi)
      use kinematics_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      double precision xfact_ileg12,xfact_ileg3,xfact_ileg4
      external xfact_ileg12,xfact_ileg3,xfact_ileg4
      integer N_P
      double precision vcf,one
      parameter (vcf=4d0/3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      s=shat_n1
c q->gq, q->aq, e->ae (icode=3)
      if(ileg.le.2)then
         N_p=2
         if(limit)then
            xkern(1)=(g**2/N_p)*4*vcf*(1-x)*((1-x)**2+1)/(s*x**2)
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) * 
     &           (ch_i**2 / vcf)
            xkernazi(1)=-(g**2/N_p)*16*vcf*(1-x)**2/(s*x**2)
            xkernazi(2)=xkernazi(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_i**2 / vcf)
         elseif(non_limit)then
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
            call Qterms_reduced_spacelike(m_type,i_type,ch_m,ch_i,one,z
     $           ,Q)
            xkernazi(1:2)=xfact*Q(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.3)then
         N_p=1
         if(non_limit)then
            xfact=xfact_ileg3(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.4)then
         N_p=1
         if(limit)then
            xkern(1)=(g**2/N_p)*
     &           ( 4*vcf*(1-x)*(s**2*(1-x)**2+(s-xm12)**2) )/
     &           ( (s-xm12)*(s*x-xm12)**2 )
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) * 
     &           (ch_i**2 / vcf)
         elseif(non_limit)then
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
      endif
      end

      subroutine compute_splitting_kernel_icode4(xkern,xkernazi,z,xi)
      use process_module
      use kinematics_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      double precision xfact_ileg12,xfact_ileg3,xfact_ileg4
      external xfact_ileg12,xfact_ileg3,xfact_ileg4
      integer N_P
      double precision vcf,one
      parameter (vcf=4d0/3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      s=shat_n1
c q->qg, q->qa, sq->sqg, sq->sqa, e->ea (icode=4)
      if(ileg.le.2)then
         N_p=1
         if(limit)then
            xkern(1)=(g**2/N_p)*4*vcf*(1+x**2)/(s*x)
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) * 
     &           (ch_m**2 / vcf)
         elseif(non_limit)then
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.3)then
         N_p=1
         if(non_limit)then
            xfact=xfact_ileg3(N_p)
            if(abs(PDG_type(j_fks)).le.6)then
               if(shower_mc_mod(1:8).ne.'HERWIGPP')
     &              call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
               if(shower_mc_mod(1:8).eq.'HERWIGPP')
     &              call AP_reduced_massive(j_type,i_type,ch_m,ch_i,one,
     &              z,xi,xm12,ap)
            else
               call AP_reduced_SUSY(j_type,i_type,ch_m,ch_i,one,z,ap)
            endif
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c     
      elseif(ileg.eq.4)then
         N_p=1
         if(limit)then
            xkern(1)=(g**2/N_p)*4*vcf*
     &           ( s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12) )/
     &           ( s*(s-xm12)*(s*x-xm12) )
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) * 
     &           (ch_j**2 / vcf)
         elseif(non_limit)then
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
      endif
      end



      
      subroutine get_shower_variables(ztmp,xitmp,xjactmp)
      use process_module
      use kinematics_module
      implicit none
      double precision ztmp,xitmp,xjactmp
      double precision zHWPP,xiHWPP,xjacHWPP,zPY6Q,xiPY6Q,xjacPY6Q
     $     ,zPY6PT,xiPY6PT,xjacPY6PT,zPY8,xiPY8,xjacPY8
      external zHWPP,xiHWPP,xjacHWPP,zPY6Q,xiPY6Q,xjacPY6Q,zPY6PT
     $     ,xiPY6PT,xjacPY6PT,zPY8,xiPY8,xjacPY8
      if(shower_mc_mod(1:8).eq.'HERWIGPP')then
         ztmp=zHWPP()
         xitmp=xiHWPP()
         xjactmp=xjacHWPP()
      elseif(shower_mc_mod(1:8).eq.'PYTHIA6Q')then
         ztmp=zPY6Q()
         xitmp=xiPY6Q()
         xjactmp=xjacPY6Q()
      elseif(shower_mc_mod(1:9).eq.'PYTHIA6PT')then
         ztmp=zPY6PT()
         xitmp=xiPY6PT()
         xjactmp=xjacPY6PT()
      elseif(shower_mc_mod(1:7).eq.'PYTHIA8')then
         ztmp=zPY8()
         xitmp=xiPY8()
         xjactmp=xjacPY8()
      endif
      end

c Finalises the MC counterterm computations performed in xmcsubt(),
c fills arrays relevant to shower scales, and computes Delta
      subroutine complete_xmcsubt(p,lzone,xmcxsec,xmcxsec2,MCsec
     $     ,probne)
      implicit none
      include "born_nhel.inc"
      include 'nFKSconfigs.inc'
      include 'nexternal.inc'
c$$$      include 'madfks_mcatnlo.inc'
      include 'run.inc'
      include 'orders.inc'

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision emsca_bare,ptresc,ref_scale,
     & scalemin,scalemax,emscainv
      double precision emscav_a(nexternal,nexternal)
      double precision emscav_a2(nexternal,nexternal)
      integer cflows,jflow
      common/c_colour_flow/jflow

      double precision emsca
      common/cemsca/emsca,emsca_bare,scalemin,scalemax
      double precision emsca_a(nexternal,nexternal)
     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
     $     ,nexternal) ,scalemin_a(nexternal,nexternal)
     $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
     $     ,nexternal)
      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2
     $     ,scalemin_a,scalemax_a,emscwgt_a
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

c Controls assignments of scales in H events in LHE file.
c Set iHscale=0 for scale=target_scale
c     iHscale=1 for scale=dipole_mass
      integer iHscale,jbar,ifksscl(2)
      parameter (iHscale=0)
      double precision dipole_mass,fksscales(3)
      external dipole_mass

c Maps real labels onto Born labels, by excluding i_fks
c  1<=iRtoB(k)<=nexternal-1,  1<=k<=nexternal
      integer iRtoB(nexternal)
c Maps Born labels onto real labels, by excluding i_fks
c  1<=iBtoR(k)<=nexternal,  1<=k<=nexternal-1
      integer iBtoR(nexternal-1)

      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      double precision emscav_tmp_a(nexternal,nexternal)
      double precision emscav_tmp_a2(nexternal,nexternal)
      common/cemscav_tmp_a/emscav_tmp_a,emscav_tmp_a2

      double precision xmcxsec(nexternal),xmcxsec2(max_bcol),probne,wgt
      logical lzone(nexternal)

      integer i,j,k,i1,i2

      double precision p(0:3,nexternal)
c For the boost to the lab frame
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision chy,shy,chymo,xdir(3),p_lab(0:3,nexternal)
      data (xdir(i),i=1,3) /0d0,0d0,1d0/

      double precision xkern(2),xkernazi(2),factor
      double precision MCsec(nexternal,max_bcol)
      include "genps.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      integer idup_s(nexternal-1)
      integer mothup_s(2,nexternal-1)
      integer icolup_s(2,nexternal-1)
      integer idup_h(nexternal)
      integer mothup_h(2,nexternal)
      integer icolup_h(2,nexternal)
      integer spinup_local(nexternal)
      integer istup_local(nexternal)
      double precision wgt_sudakov
      double precision scales(0:99)
      common /colour_connections/ icolup_s,icolup_h

c To access Pythia8 control variables
      include 'pythia8_control.inc'
      include "born_leshouche.inc"
      integer jpart(7,-nexternal+3:2*nexternal-3),lc,iflow
      logical firsttime1
      data firsttime1 /.true./
      include 'leshouche_decl.inc'
      save idup_d, mothup_d, icolup_d, niprocs_d

C To allow retrieval of S-event from Pythia
      include 'hep_event_streams.inc'

      logical         Hevents
      common/SHevents/Hevents
      integer nexternal_now
c SCALUP_tmp_S = m_ij scales that determine S-event scales written onto LHE
      double precision SCALUP_tmp_S(nexternal,nexternal)
c SCALUP_tmp_S2 = m_ij starting scales for Delta
      double precision SCALUP_tmp_S2(nexternal,nexternal)
c SCALUP_tmp_H = t_ij scales that determine H-event scales written onto LHE
      double precision SCALUP_tmp_H(nexternal,nexternal)
c SCALUP_tmp_H2 = t_ij target scales for Delta
      double precision SCALUP_tmp_H2(nexternal,nexternal)
      common/c_SCALUP_tmp/SCALUP_tmp_S,SCALUP_tmp_H
      double precision SCALUP_tmp_H3(nexternal,nexternal)

c Lower and upper limits of fitted st and xm ranges.
c Require one prior call to pysudakov() to be set,
c here done in the firsttime1 clause
      real*8 cstlow,cstupp,cxmlow,cxmupp
      common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp

c Set Delta(pt,..)=0 for pt<smallptlow, and interpolate
c between 0 and Delta(smallptupp,..) for smallptlow<pt<smallptupp
c For things to work properly, one must have:
c               cstlow <= smallptupp
      real*8 smallptlow,smallptupp,get_to_zero
      parameter (smallptlow=0.5d0)
      parameter (smallptupp=1.01d0)

      integer iii,jjj,LP
      double precision xscales(0:99,0:99)
      double precision xmasses(0:99,0:99)
      double precision xscales2(0:99,0:99)
      double precision xmasses2(0:99,0:99)
      logical*1 dzones(0:99,0:99)
      logical*1 dzones2(0:99,0:99)

      integer id,type,icount,jcount,kcount,jindex(2)
      integer iflip(2)
      data iflip/2,1/

      double precision emscav_a2_tmp,emscav_tmp_a2_tmp,ptresc_a_tmp
      double precision sref,acll1,acll2,acllfct(2),dot,sumdot
      external dot,sumdot
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

cSF ARE noemProb AND mDipole USEFUL?
      double precision startingScale0,stoppingScale0
      double precision noemProb, startingScale(2), stoppingScale(2), mDipole
      double precision mcmass(21)
      double precision pysudakov,deltanum(2,2),deltaden(2),deltarat(2,2)
      double precision gltmp,xtmp(2),glfact(2),glrat(2)
      integer nG_S,nQ_S,i_dipole_counter,isudtype(2)
      integer i_dipole_dead_counter
c
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

      double precision pdg2pdf,pdffnum(2),pdffden(2)
      external pdg2pdf
c
      double precision amp_split_xmcxsec(amp_split_size,nexternal)
      common /to_amp_split_xmcxsec/amp_split_xmcxsec
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      integer idIn1, idIn2
      integer idOut(0:9)
      double precision tBefore,tAfter
      double precision masses_to_MC(0:25)
      double precision pi
      parameter(pi=3.1415926535897932384626433d0)
      logical are_col_conn_S(nexternal-1,nexternal-1)
      logical are_col_conn_H(nexternal,nexternal)
      double precision get_mass_from_id
      external get_mass_from_id
      logical isspecial(max_bcol)
      common/cisspecial/isspecial
      double precision qMC_a2(nexternal-1,nexternal-1)
      common /to_complete/qMC_a2
      double precision scales_for_HEPEUP(nexternal,nexternal)
      logical force_II_connection
      parameter(force_II_connection=.true.)
c
      mcmass=0d0
      masses_to_MC=0d0
      include 'MCmasses_PYTHIA8.inc'
c
      do i=1,2
        istup_local(i) = -1
      enddo
      do i=3,nexternal
        istup_local(i) = 1
      enddo
      do i=1,nexternal
        spinup_local(i) = -9
      enddo
      pythia_cmd_file=''
      
c Given xmcxec,etc., returns jflow, wgt and fills emsca in common block:
      call assign_emsca_and_flow_statistical(xmcxsec,xmcxsec2,MCsec
     $     ,lzone,jflow,wgt)
      
c S-event information:
c id's and mothers read from born_leshouche.inc;
c colour configuration read from born_leshouche.inc and jflow 
      do i=1,nexternal-1
        IDUP_S(i)=IDUP(i,1)
        MOTHUP_S(1,i)=MOTHUP(1,i,1)
        MOTHUP_S(2,i)=MOTHUP(2,i,1)
        ICOLUP_S(1,i)=ICOLUP(1,i,jflow)
        ICOLUP_S(2,i)=ICOLUP(2,i,jflow)
      enddo
      are_col_conn_S=.false.
      do i=1,nexternal-1
         do j=1,nexternal-1
            if(i.ne.j)
     &        are_col_conn_S(i,j)=
     &      (ICOLUP_S(1,i).ne.0.and.ICOLUP_S(1,i).eq.ICOLUP_S(1,j)).or.
     &      (ICOLUP_S(1,i).ne.0.and.ICOLUP_S(1,i).eq.ICOLUP_S(2,j)).or.
     &      (ICOLUP_S(2,i).ne.0.and.ICOLUP_S(2,i).eq.ICOLUP_S(1,j)).or.
     &      (ICOLUP_S(2,i).ne.0.and.ICOLUP_S(2,i).eq.ICOLUP_S(2,j))
         enddo
      enddo
c SCALUP_tmp_S* are the m_ij scales, ie the starting scales (as determined
c by the D(mu) function) for extra radiation; they are copies of the
c emscav_tmp_a* arrays, originally filled by xmcsubt(). Only the (i,j) 
c entries associated with a colour line that belongs to jflow have
c meaningful values; the others are set equal to -1.
c SCALUP_tmp_S and SCALUP_tmp_S2 are chosen in exactly the same way, except
c for the random numbers that enter their definitions. The former will
c help determine the S-event shower scales written onto the LHE file, 
c the latter is employed in the computation of Delta
      SCALUP_tmp_S=-1d0
c$$$      SCALUP_tmp_S2=-1d0
      do i=1,nexternal-2
         do j=i+1,nexternal-1
            if(are_col_conn_S(i,j))then
c$$$               SCALUP_tmp_S(i,j)=emscav_tmp_a(i,j)
c$$$               SCALUP_tmp_S(j,i)=emscav_tmp_a(j,i)
c$$$               SCALUP_tmp_S2(i,j)=emscav_tmp_a2(i,j)
c$$$               SCALUP_tmp_S2(j,i)=emscav_tmp_a2(j,i)
               SCALUP_tmp_S(i,j)=shower_scale_nbody(i,j)
               SCALUP_tmp_S(j,i)=shower_scale_nbody(j,i)
c$$$               SCALUP_tmp_S2(i,j)=shower_scale_nbody(i,j)
c$$$               SCALUP_tmp_S2(j,i)=shower_scale_nbody(j,i)
            endif
         enddo
      enddo
c
c force IF colour connection to have II scale
c if a sensible II scale exists
      if(force_II_connection)then
         do i=1,2
            do j=3,nexternal-1
               if(are_col_conn_S(i,j))then
                  if(are_col_conn_S(i,3-i))then
                     SCALUP_tmp_S(i,j) =SCALUP_tmp_S(i,3-i)
c$$$                     SCALUP_tmp_S2(i,j)=SCALUP_tmp_S2(i,3-i)
                  else
                     continue
c if no other available colour connection, we keep the IF scale
c rather than calculating some new kinematic variable e.g. pT
                  endif
               endif
            enddo
         enddo
      endif
c
c H-event information.
c First write ids, mothers and all colours.
cSF NOTE: reconsider how much H-event information is actually needed
cSF by Pythia. For example, the colour flow is used only to reconstruct
cSF the underlying S-event flow, which is already available here, and
cSF thus can be directly passed rather than reconstructed
      if (firsttime1)then
        firsttime1=.false.
        call read_leshouche_info(idup_d,mothup_d,icolup_d,niprocs_d)
c Fake call for initialisation
        deltanum(1,1)=pysudakov(1.d2,2.d2,1,1,mcmass)
        if(cstlow.gt.smallptupp)then
          write(*,*)'Error in xmcsubt: cstlow,smallptupp',
     &               cstlow,smallptupp
          stop
        endif
      endif
      do i=1,nexternal
         IDUP_H(i)=IDUP_D(nFKSprocess,i,1)
         MOTHUP_H(1,i)=MOTHUP_D(nFKSprocess,1,i,1)
         MOTHUP_H(2,i)=MOTHUP_D(nFKSprocess,2,i,1)
      enddo
c Fill selected color configuration into jpart array. 
      call fill_icolor_H(jflow,jpart)
      do i=1,nexternal
        ICOLUP_H(1,i)=jpart(4,i)
        ICOLUP_H(2,i)=jpart(5,i)
      enddo
      are_col_conn_H=.false.
      do i=1,nexternal
         do j=1,nexternal
            if(i.ne.j)
     &        are_col_conn_H(i,j)=
     &      (ICOLUP_H(1,i).ne.0.and.ICOLUP_H(1,i).eq.ICOLUP_H(1,j)).or.
     &      (ICOLUP_H(1,i).ne.0.and.ICOLUP_H(1,i).eq.ICOLUP_H(2,j)).or.
     &      (ICOLUP_H(2,i).ne.0.and.ICOLUP_H(2,i).eq.ICOLUP_H(1,j)).or.
     &      (ICOLUP_H(2,i).ne.0.and.ICOLUP_H(2,i).eq.ICOLUP_H(2,j))
         enddo
      enddo
      do i=1,nexternal
        if(i.lt.i_fks)then
          iRtoB(i)=i
          iBtoR(i)=i
        elseif(i.eq.i_fks)then
          iRtoB(i)=-1
          if(i.lt.nexternal)iBtoR(i)=i+1
        elseif(i.gt.i_fks)then
          iRtoB(i)=i-1
          if(i.lt.nexternal)iBtoR(i)=i+1
        endif
      enddo
c
      nexternal_now=nexternal
      call clear_HEPEUP_event()
      
c Boost H-event momenta to lab frame before passing to pythia
      chy=cosh(ybst_til_tolab)
      shy=sinh(ybst_til_tolab)
      chymo=chy-1d0
      do i=1,nexternal
         call boostwdir2(chy,shy,chymo,xdir,p(0,i),p_lab(0,i))
      enddo

c Call fill_HEPEUP_event with S scale information.
c If it is an H event, this information should not be
c used by Pythia, hence a dummy -1d0 value is passed
      if(Hevents)then
         scales_for_HEPEUP = -1d0
      else
         scales_for_HEPEUP = SCALUP_tmp_S
      endif
c
      call fill_HEPEUP_event(p_lab, wgt, nexternal_now, idup_h,
     &       istup_local, mothup_h, icolup_h, spinup_local,
     &       emsca, scales_for_HEPEUP)
      xscales=-1d0
      xmasses=-1d0
      dzones=.true.
      xscales2=-1d0
      xmasses2=-1d0
      dzones2=.true.
      if (is_pythia_active.eq.0) then
c Fill masses
         do i=7,20
            if(i.le.10.or.i.ge.17)masses_to_MC(i)=-1d0
         enddo
         masses_to_MC(5) =get_mass_from_id(5)
         masses_to_MC(6) =get_mass_from_id(6)
         masses_to_MC(15)=get_mass_from_id(15)
         masses_to_MC(23)=get_mass_from_id(23)
         masses_to_MC(24)=get_mass_from_id(24)
         masses_to_MC(25)=get_mass_from_id(25)
c
         idOut=0
         do i=3,nexternal-1
            idOut(i-3) = IDUP_S(i)
            if ( is_a_j(i) ) idOut(i-3)=2212
         enddo
         idIn1 = idup_s(1)
         idIn2 = idup_s(2)
         if ( abs(idIn1) < 10 .or. idIn1 .eq. 21) idIn1=2212
         if ( abs(idIn2) < 10 .or. idIn2 .eq. 21) idIn2=2212
         call pythia_init_default(idIn1, idIn2, idOut, masses_to_MC)
      endif
      call pythia_setevent()
      call pythia_next()
      call pythia_get_stopping_info(xscales,xmasses)
      call pythia_get_dead_zones(dzones)
      call pythia_clear()

c     Check if the S-event state (as created from the H-event by Pythia)
c     is consistent with the MG_aMC S-event state.
      if (NUP_in .ne. nexternal-1) then
         write (*,*) 'montecarlocounter.f: States not compatible #1'
     $        ,nup_in,nexternal-1
         stop 1
      endif
      do i=1,nup_in
         do j=1,nexternal-1
            if (i.le.nincoming) then
c incoming momenta should always be particle 1 and 2.
               if (j.ne.i) cycle
            elseif (j.le.nincoming) then
               cycle
            endif
            if (idup_in(i).eq.idup_s(j)) then
c found the same particle ID. Check that colour is okay. 
               if (all(icolup_in(1:2,i).eq.icolup_s(1:2,j))) then
                  exit ! Agreement found.
               endif
            endif
         enddo
         if (j.gt.nexternal-1) then
c went all the way through the 2nd do-loop without finding the corresponding particle.
            write (*,*) 'montecarlocounter.f: States not compatible #2'
            write (*,*) 'returned by Pythia:'
            write (*,*) idup_in(1:nup_in)
            write (*,*) icolup_in(1,1:nup_in)
            write (*,*) icolup_in(2,1:nup_in)
            write (*,*) 'available in MG5_aMC:'
            write (*,*) idup_s(1:nup_in)
            write (*,*) icolup_s(1,1:nup_in)
            write (*,*) icolup_s(2,1:nup_in)
            stop 1
         endif
      enddo

c After the calls above, we have
c   xscales(i,j)=t_ij
c with t_ij == scale(Pythia)_{emitter,recoiler}, and the particle being
c emitted equal to the FKS parton. Although both emitter and recoiler
c are Born-level quantities, their labellings follow the real-process
c conventions. Thus, in the matrix xscales(i,j) one has 1<=i,j<=nexternal, 
c with xscales(i_fks,*)=xscales(*,i_fks)=-1.
c The same labelling conventions apply to xmasses(i,j) (which is the
c dipole mass associated with the colour line that connects i and j)
c and dzones(i,j) (which is the dead zone relevant to the emission
c from parton i colour-connected with recoiler j).
c
c Since any the pair of indices (i,j) associated with sensible
c entries in the arrays returned by Pythia is in one-to-one correspondence
c with Born-level quantities, it is convenient to define relabelled
c copies of such arrays (which we call xscales2, xmasses2, and dzones2),
c for which 1<=i,j<=nexternal-1
c
c By construction, t_ij are the target scales. For notational consistency
c with the case of SCALUP_tmp_S2, a copy of xscales2 is created and called
c SCALUP_tmp_H2, meant to be used in the computation of Delta. 

! Since pythia simply does a one-branch cluster, it does not check if
! the stopping scale (in xscales) is smaller than the starting scale (as
! determined by MG5_aMC in SCALUP_tmp_S). If this is the case, put the
! event in the dead-zone.
      do i=1,nexternal-1
         do j=1,nexternal-1
            if (i.eq.j) cycle
            if (.not. dzones(iBtoR(i),iBtoR(j))) then
               if (xscales(iBtoR(i),iBtoR(j)).gt.SCALUP_tmp_S(i,j)) then
                  dzones(iBtoR(i),iBtoR(j))=.true.
               endif
            endif
         enddo
      enddo
      
      SCALUP_tmp_H2=-1d0
      do i=1,nexternal
         if(i.eq.i_fks)cycle
         do j=1,nexternal
            if(j.eq.i_fks)cycle
            xscales2(iRtoB(i),iRtoB(j))=xscales(i,j)
c In pythia the dipole masses can be arbitary large since the clustering
c does not know exactly all the phase-space boundaries. Use min() to put
c a cap on this (i.e., equal to the largest allowed value in pysudakov()
c tables).
            xmasses2(iRtoB(i),iRtoB(j))=min(xmasses(i,j),cxmupp)
            dzones2(iRtoB(i),iRtoB(j))=dzones(i,j)
            scalup_tmp_H2(iRtoB(i),iRtoB(j))=
     &           xscales2(iRtoB(i),iRtoB(j))
         enddo
      enddo
c Checks
      do i=1,nexternal-1
         do j=1,nexternal-1
            if((xscales2(i,j).ne.-1d0.and.xmasses2(i,j).eq.-1d0).or.
     &         (xscales2(i,j).eq.-1d0.and.xmasses2(i,j).ne.-1d0))then
               write(*,*)'Error in xmcsubt: xscales, xmasses',
     &                   i,j,xscales2(i,j),xmasses2(i,j)
               stop
            endif
         enddo
      enddo

      call set_SCALUP_tmp_H(are_col_conn_H,are_col_conn_S,iBtoR,iRtoB
     $     ,xscales2,dzones2,p,SCALUP_tmp_H)
c
c force IF colour connection to have II scale
c if a sensible II scale exists
      if(force_II_connection)then
         do i=1,2
            do j=3,nexternal
               if(are_col_conn_H(i,j))then
                  if(are_col_conn_H(i,3-i))then
                     SCALUP_tmp_H(i,j) =SCALUP_tmp_H(i,3-i)
                  else
                     continue
c if no other available colour connection, we keep the IF scale
c rather than calculating some new kinematic variable e.g. pT
                  endif
               endif
               if (iRtoB(j).eq.-1) cycle ! prevent out-of-bounds
               if(are_col_conn_S(iRtoB(i),iRtoB(j)))then
                  if(are_col_conn_S(iRtoB(i),iRtoB(3-i)))then
                     SCALUP_tmp_H2(iRtoB(i),iRtoB(j))=SCALUP_tmp_H2(iRtoB(i),iRtoB(3-i))
                  else
                     continue
c if no other available colour connection, we keep the IF scale
c rather than calculating some new kinematic variable e.g. pT
                  endif
               endif
            enddo
         enddo
      endif
ccccccccccccccccccccc
c
c     *** WARNING ***
c
c     Pythia resets the scale for FI and FF to the min between the
c     scale t_ij we give it and p_i.p_j/2.
c     Should we implement this minimisation here as well?
c     For S events no implementation is needed since we take
c     scales directly from Pythia. For H event this implementation
c     shuld be needed only for i_fks and j_fks, as only in that case
c     we (over)write their scales ourselves, but could be applied to
c     all FI and FF connections.
c
ccccccccccccccccccccc
c
c Computation of Delta = wgt_sudakov as the product of Sudakovs between
c starting scales (SCALUP_tmp_S2) and target scales (SCALUP_tmp_H2).
c For initial-state legs, Delta contains a PDF ratio with S-event Bjorken
c fraction and SCALUP_tmp_S2, SCALUP_tmp_H2 scales, see also formula (5.62)
c in Ellis-Stirling-Webber
      wgt_sudakov=1d0
      i_dipole_counter=0
      i_dipole_dead_counter=0
      nG_S=0
      nQ_S=0
c
      do i=1,nexternal-1
         if(idup_s(i).eq.21)nG_S=nG_S+1
         if(abs(idup_s(i)).le.6)nQ_S=nQ_S+1
         icount=0
         jindex(1)=-1
         jindex(2)=-1
         do j=1,nexternal-1
c At fixed i, loop over j to find the colour lines that begin at i 
c (at most [because of dead zones] one for quarks, two for gluons).
c For each of these colour lines, find the starting and stopping
c scales and store them in startingScale(*) and stoppingScale(*).
c Store the corresponding Sudakov type in isudtype(*)
            if(j.eq.i)cycle
            if(xscales2(i,j).eq.-1d0)cycle
            if(dzones2(i,j))then
               i_dipole_dead_counter=i_dipole_dead_counter+1
               if (isspecial(jflow) .and. idup_s(i).eq.21) then
c double colour connection, count twice
                  i_dipole_dead_counter=i_dipole_dead_counter+1
               endif
               cycle
            endif
            icount=icount+1
            if( (abs(idup_s(i)).le.6.and.icount.gt.1) .or.
     #          (idup_s(i).eq.21 .and.
     #            (icount.gt.2.and.(.not.isspecial(jflow))) .or.
     #            (icount.gt.1.and.isspecial(jflow)) ) )then
              write(*,*)'Error #6 in complete_xmcsubt'
              write(*,*)i,idup_s(i),icount,isspecial(jflow)
              stop
            endif
c The following definition of startingScale is unprotected:
c cstupp must be sufficiently large
c$$$            startingScale0 = min(SCALUP_tmp_S2(i,j),cstupp)
            startingScale0 = min(SCALUP_tmp_S(i,j),cstupp)
c Same comment on cstupp as above. Inserted here as a safety
c measure, since Pythia might give very large scales. In those
c case, the computed Sudakovs are actually discarded later
            stoppingScale0 = min(SCALUP_tmp_H2(i,j),cstupp)
            acllfct(icount)=1.d0
c Passing the following if clause must be exceedingly rare
            if(startingScale0.le.smallptupp)then
              write(*,*)'Warning in xmcsubt: startingScale0, smallptupp'
              write(*,*)startingScale0,smallptupp
c$$$              stop
              startingScale0=smallptupp
            endif
            stoppingScale(icount)=stoppingScale0
            startingScale(icount)=startingScale0
            jindex(icount)=j
            if(i.le.2.and.j.le.2)then
               isudtype(icount)=1
            elseif(i.gt.2.and.j.gt.2)then
               isudtype(icount)=2
            elseif(i.le.2.and.j.gt.2)then
c For Pythia: IF is identical to II
               isudtype(icount)=1
            elseif(i.gt.2.and.j.le.2)then
               isudtype(icount)=4
            endif
            if(stoppingScale(icount).gt.startingScale(icount))then
               i_dipole_dead_counter=i_dipole_dead_counter+1
            else
               i_dipole_counter=i_dipole_counter+1
            endif
            if (isspecial(jflow) .and. idup_s(i).eq.21) then
c double colour connection, count twice
               if (idup_s(j).ne.21) then
                  write (*,*) 'SPECIAL and a gluon is not connected '/
     $                 /'to another gluon',i,j,idup_s(i),idup_s(j)
                  stop 1
               endif
               if(stoppingScale(icount).gt.startingScale(icount))then
                  i_dipole_dead_counter=i_dipole_dead_counter+1
               else
                  i_dipole_counter=i_dipole_counter+1
               endif
            endif
c
c Conventions:
c   deltanum(i,j) <--> stoppingScale(i),isudtype(j)
c   deltaden(j)   <--> startingScale(j)
c Given our definitions, whenever a quantity is computed for
c which both the stopping scale and the Sudakov type are relevant,
c the arguments of stoppingScale(*) and isudtype(*) must be equal.
c The ratio:
c    deltarat(i,j) = deltanum(i,j)/deltaden(j)
c is the Sudakov of type isudtype(j) (with CF for quarks, and CA/2 for gluons) 
c between scales [stoppingScale(i),startingScale(j)]
c
            if(stoppingScale(icount).le.smallptlow)then
c Still inside the j loop, but the information is sufficient to 
c compute deltaden(*) and deltanum(*,*)
               deltanum(icount,icount)=0.d0
            elseif( stoppingScale(icount).gt.smallptlow .and.
     $              stoppingScale(icount).le.smallptupp )then
               deltanum(icount,icount)= pysudakov(smallptupp,xmasses2(i
     $              ,j), idup_s(i),isudtype(icount),mcmass)
               deltanum(icount,icount)=deltanum(icount,icount)*
     $              get_to_zero(stoppingScale(icount),smallptlow
     $              ,smallptupp)
            else
               deltanum(icount,icount)= pysudakov(stoppingScale(icount)
     $              ,xmasses2(i,j), idup_s(i),isudtype(icount),mcmass)
            endif
            deltaden(icount)=pysudakov(startingScale(icount),xmasses2(i
     $           ,j),idup_s(i),isudtype(icount),mcmass)
c End of primary loop over j
         enddo

c No live colour connection starting from/ending in i has been found:
c go to the next i
         if(icount.eq.0)goto 111
c
         if(i.le.nincoming)then
           LP=SIGN(1,LPP(i))
           if (idup_s(i).le.6) then ! (anti-)quark 
              id=LP*idup_s(i)
           elseif (idup_s(i).eq.21) then ! gluon
              id=0
           elseif (idup_s(i).eq.22) then ! photon
              id=7
           endif
           do jcount=1,icount
             pdffnum(jcount)=pdg2pdf(abs(lpp(i)),id,1,xbjrk_cnt(i,0),
     #                               stoppingScale(jcount))
             pdffden(jcount)=pdg2pdf(abs(lpp(i)),id,1,xbjrk_cnt(i,0),
     #                               startingScale(jcount))
           enddo
         else
           pdffnum(1:2)=1.d0
           pdffden(1:2)=1.d0
         endif
c
         if(icount.eq.1)then
c This is either a quark, or a gluon with either only one colour line 
c corresponding to a live zone or a gluon with a single colour line but
c a double colour connection (eg in gg->H). The condition that the starting
c scale is larger than the stopping scale has not been enforced
c so far, so do it here
           if(stoppingScale(1).lt.startingScale(1))then
             if (isspecial(jflow).and.idup_s(i).eq.21) then
               deltanum(1,1)=deltanum(1,1)**2*pdffnum(1)
               deltaden(1)=deltaden(1)**2*pdffden(1)
             else
               deltanum(1,1)=deltanum(1,1)*pdffnum(1)
               deltaden(1)=deltaden(1)*pdffden(1)
             endif
             if(deltaden(1).eq.0.d0)then
               deltarat(1,1)=1.d0
             else
               deltarat(1,1)=deltanum(1,1)/deltaden(1) * acllfct(1)
             endif
             if(deltarat(1,1).ge.1.d0)deltarat(1,1)=1.d0
             if(deltarat(1,1).le.0.d0)deltarat(1,1)=0.d0
             wgt_sudakov=wgt_sudakov*deltarat(1,1)
           endif
         else
c A gluon with two partners corresponding to a live zone
           if(jindex(1).eq.-1.or.jindex(2).eq.-1)then
             write(*,*)'Error #10 in complete_xmcsubt:',
     #                 jindex(1),jindex(2),i,icount
             stop
           endif
           do jcount=1,icount
c Start by computing deltanum(1,2) and deltanum(2,1)
             if(stoppingScale(jcount).le.smallptlow)then
               deltanum(jcount,iflip(jcount))=0.d0
            elseif( stoppingScale(jcount).gt.smallptlow .and.
     $              stoppingScale(jcount).le.smallptupp )then
                deltanum(jcount,iflip(jcount))= pysudakov(smallptupp
     $               ,xmasses2(i,jindex(iflip(jcount))), idup_s(i)
     $               ,isudtype(iflip(jcount)),mcmass)
               deltanum(jcount,iflip(jcount))=deltanum(jcount
     $              ,iflip(jcount))*get_to_zero(stoppingScale(jcount)
     $              ,smallptlow,smallptupp)
             else
                deltanum(jcount,iflip(jcount))=
     $               pysudakov(stoppingScale(jcount),xmasses2(i
     $               ,jindex(iflip(jcount))), idup_s(i)
     $               ,isudtype(iflip(jcount)),mcmass)
             endif
           enddo
c Here, deltaden(*) and deltanum(*,*) must be filled with sensible values.
c Proceed to compute the corresponding Sudakov; the effective colour factor 
c is CA, with a single stopping scale and two possibly different starting scales
c (each of the latter is responsible for CA/2)
           do jcount=1,icount
             if( deltaden(jcount).eq.0.d0 )then
               deltarat(1,jcount)=1.d0
               deltarat(2,jcount)=1.d0
             else
               do kcount=1,icount
                 if(stoppingScale(kcount).lt.startingScale(jcount))then
                    deltarat(kcount,jcount)=deltanum(kcount,jcount)/
     $                   deltaden(jcount)
                 else
                   deltarat(kcount,jcount)=1.d0
                 endif
               enddo
             endif
c glfact(*) define the relative weights of the two no-emission probabilities.
c In their computations, use the CA/2 Sudakov with starting and stopping
c scales relevant to the same colour line
             gltmp=deltarat(jcount,jcount)*acllfct(jcount)
             if(gltmp.le.0.d0)then
               glfact(jcount)=1.d8
             elseif(gltmp.ge.1.d0)then
               glfact(jcount)=0.d0
             else
               glfact(jcount)=-2*log(gltmp)
             endif
           enddo
           glrat(1)=glfact(1)/(max(glfact(1)+glfact(2),1.d-8))
           glrat(2)=glfact(2)/(max(glfact(1)+glfact(2),1.d-8))
           if( glrat(1).lt.0.d0.or.glrat(1).gt.1.d0 .or.
     #         glrat(2).lt.0.d0.or.glrat(2).gt.1.d0 )then
             write(*,*)'Error #7 in complete_xmcsubt'
             write(*,*)glrat(1),glrat(2)
             stop
          elseif (glrat(1).eq.0d0 .and. glrat(2).eq.0d0) then
             glrat(1)=0.5d0
             glrat(2)=0.5d0
           endif
c
           do jcount=1,icount
             xtmp(jcount)=1.d0
             if( pdffden(jcount).gt.0.d0 )then
               if(stoppingScale(jcount).lt.startingScale(1))
     #           xtmp(jcount)=xtmp(jcount)*deltarat(jcount,1)
               if(stoppingScale(jcount).lt.startingScale(2))
     #           xtmp(jcount)=xtmp(jcount)*deltarat(jcount,2)
               if(xtmp(jcount).lt.1.d0)
     #           xtmp(jcount)=xtmp(jcount)*acllfct(jcount)
               if(stoppingScale(jcount).lt.startingScale(jcount))
     #           xtmp(jcount)=xtmp(jcount)*pdffnum(jcount)/pdffden(jcount)
             endif
             if(xtmp(jcount).ge.1.d0)xtmp(jcount)=1.d0
             if(xtmp(jcount).le.0.d0)xtmp(jcount)=0.d0
           enddo
           wgt_sudakov=wgt_sudakov*(glrat(1)*xtmp(1)+glrat(2)*xtmp(2))
         endif
 111     continue
c End of primary loop over i
      enddo
      if(i_dipole_counter+i_dipole_dead_counter.ne.nQ_S+2*nG_S)then
         write(*,*)'Mismatch in number of dipole ends and Delta factors'
         write(*,*)i_dipole_counter+i_dipole_dead_counter,nQ_S,nG_S
         stop
      endif


      if (btest(MCcntcalled,3)) then
         write (*,*) 'Fourth bit of MCcntcalled should not '/
     $        /'have been set yet',MCcntcalled
         stop 1
      endif
      MCcntcalled=MCcntcalled+8

      probne = wgt_sudakov
      if(probne.lt.0.d0)then
         write(*,*)'Error in MC@NLO-Delta: Sudakov smaller than 0',probne
         probne=0.d0
         stop 1
      endif
      if(probne.gt.1.d0)then
         write(*,*)'Error in MC@NLO-Delta: Sudakov larger than 1',probne
         probne=1.d0
         stop 1
      endif
c
      do i=1,nexternal
         if(i.le.ipartners(0)) then
            xmcxsec(i)=xmcxsec(i)*probne
            amp_split_xmcxsec(1:amp_split_size,i)
     $           =amp_split_xmcxsec(1:amp_split_size,i)*probne
         endif
         if(i.gt.ipartners(0)) then
            xmcxsec(i)=0d0
            amp_split_xmcxsec(1:amp_split_size,i)=0d0
         endif
      enddo

      return
      end

      subroutine set_SCALUP_tmp_H(are_col_conn_H,are_col_conn_S,iBtoR
     $     ,iRtoB,xscales2,dzones2,p,SCALUP_tmp_H)
! Fills the SCALUP_tmp_H based on the S-event stopping scales. In the
! MC-picture, all scales for which i_fks and j_fks are emitter are set
! to a common scale 'pT'. In the ME-picture, a rather more strict
! relation between the dipoles is followed, and each dipole for which
! i_fks and j_fks are the emitter can get different values, based on the
! colour connections of the mother.
! In case we are in the deadzone, use a scale based on the dipole mass
! (using H-event kinematics) instead.
! WARNING: this subroutine does NOT enforce the scales for the IF
! dipoles to be overwritten by the II dipoles.
      implicit none
      include 'nexternal.inc'
      logical are_col_conn_H(nexternal,nexternal)
     $     ,are_col_conn_S(nexternal-1,nexternal-1)
      logical*1 dzones2(0:99,0:99)
      double precision xscales2(0:99,0:99),SCALUP_tmp_H(nexternal
     $     ,nexternal),p(0:3,nexternal)
      integer iRtoB(nexternal),iBtoR(nexternal-1)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i1,i2,ip,imother,i1bar,i2bar
      double precision t(nexternal,nexternal),pT,pTparton
      integer ipbar,ipbar2
      double precision compute_pTparton
      external ipbar,ipbar2,comput_pTparton
      logical MCpicture,ptparton_computed
      parameter (MCpicture=.true.) ! Switch between MC- and ME-pictures.

      ptparton_computed=.false.
      t(1:nexternal,1:nexternal)=-1d0
      imother=iRtoB(j_fks)
      if (MCpicture) then
         ! Let pT to be the minimum of the stopping scales related to
         ! the mother.
         pT=99d99
         do i2bar=1,nexternal-1
            if (are_col_conn_S(imother,i2bar)) then
               if (.not.dzones2(imother,i2bar))
     &              pT=min(pT,xscales2(imother,i2bar))
            endif
         enddo
         if (pT.eq.99d99) then
            if (.not.ptparton_computed) then
               ptparton=compute_pTparton(p)
               ptparton_computed=.true.
            endif
            pt=ptparton
         endif
      endif
      do i1=1,nexternal
         do i2=1,nexternal
            if (.not.are_col_conn_H(i1,i2)) cycle
            ! Find the (i1bar,i2bar) S-event dipole corresponding to
            ! the (i1,i2) H-event dipole.
            if (i1.eq.i_fks .and. i2.eq.j_fks) then
               if (MCpicture) then
                  i1bar=-99
               else
                  i1bar=ipbar(are_col_conn_H,imother,iRtoB)
                  i2bar=imother
               endif
            elseif (i1.eq.j_fks .and. i2.eq.i_fks) then
               if (MCpicture) then
                  i1bar=-99
               else
                  i1bar=imother
                  i2bar=ipbar(are_col_conn_H,imother,iRtoB)
               endif
            elseif (i1.eq.i_fks .or. i1.eq.j_fks) then
               if (MCpicture) then
                  i1bar=-99
               else
                  i1bar=imother
                  i2bar=iRtoB(i2)
               endif
            elseif (i2.eq.i_fks .or. i2.eq.j_fks) then
               i1bar=iRtoB(i1)
               i2bar=imother
            else ! both i1 and i2 are not equal to i_fks and/or j_fks
               i1bar=iRtoB(i1)
               i2bar=iRtoB(i2)
            endif
            ! (i1bar,i2bar) dipole found. Set the (i1,i2) dipole
            ! starting scale based on the (i1bar,i2bar) stopping scale
            ! (or inv. mass in case of dead zone).
            if (i1bar.eq.-99) then
               if (.not. MCpicture) then
                  write (*,*) 'This should only happen in the MCpicture'
                  stop 1
               endif
               t(i1,i2)=pT
            else
               if (.not.are_col_conn_S(i1bar,i2bar)) then
                  write (*,*) 'Lines not color connected #2',
     $                 i1,i2,i1bar,i2bar
                  stop 1
               endif
               if (.not. dzones2(i1bar,i2bar)) then
                  t(i1,i2)=xscales2(i1bar,i2bar)
               else
                  if (.not.ptparton_computed) then
                     ptparton=compute_pTparton(p)
                     ptparton_computed=.true.
                  endif
                  t(i1,i2)=pTparton
               endif
            endif
         enddo
      enddo
      ! check that all have been set
      do i1=1,nexternal
         do i2=1,nexternal
            if (.not.are_col_conn_H(i1,i2)) cycle
            if (t(i1,i2).eq.-1d0) then
               write (*,*) 'ERROR, scale still equal to -1',i1,i2
               stop 1
            endif
         enddo
      enddo
      SCALUP_tmp_H(1:nexternal,1:nexternal)=t(1:nexternal,1:nexternal)
      end

      double precision function compute_pTparton(p)
      implicit none
      include 'nexternal.inc'
      double precision p(0:3,nexternal)
      double precision pQCD(0:3,nexternal-1),palg,sycut,rfj,pjet(0:3
     $     ,nexternal-1)
      integer i,j,NN,njet,jet(nexternal-1)
      double precision pt,amcatnlo_fastjetdmergemax
      external pt,amcatnlo_fastjetdmergemax
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      NN=0
      do j=nincoming+1,nexternal
         if (is_a_j(j))then
            NN=NN+1
            do i=0,3
               pQCD(i,NN)=p(i,j)
            enddo
         endif
      enddo
! reduce by kT-cluster scale of massless QCD partons
      if (NN.eq.1) then
         compute_pTparton=pt(pQCD(0,1))
      elseif (NN.ge.2) then
         palg=1d0
         sycut=0d0
         rfj=1d0
         call amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
     &        pjet,njet,jet)
         compute_pTparton=sqrt(amcatnlo_fastjetdmergemax(NN-1))
      else
         write (*,*) 'Error in compute_pTparton(): '/
     $        /'Must have at least one QCD parton at the NLO level'
         stop 1
      endif
      end

      
      integer function ipbar(are_col_conn_H,imother,iRtoB)
      ! ipbar is the colour connection of i_fks (if it exists and is not
      ! equal to the mother). Otherwise it is the colour connection of
      ! j_fks. The latter only happens when i_fks is a quark and j_fks
      ! is an (incoming gluon).
      implicit none
      include 'nexternal.inc'
      logical are_col_conn_H(nexternal,nexternal)
      integer imother,iRtoB(nexternal)
      integer ip
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      ipbar=0
      do ip=1,nexternal
         if(ip.eq.i_fks)cycle
         if (are_col_conn_H(ip,i_fks) .and. iRtoB(ip).ne.imother) then
            if (ipbar.ne.0) then
               write (*,*) 'Too many colour connections #1'
               stop 1
            endif
            ipbar=iRtoB(ip)
         endif
      enddo
      if (ipbar.eq.0) then
         do ip=1,nexternal
            if(ip.eq.i_fks)cycle
            if (are_col_conn_H(ip,j_fks) .and. iRtoB(ip).ne.imother)
     $           then
               if (ipbar.ne.0) then
                  write (*,*) 'Too many colour connections #2'
                  stop 1
               endif
               ipbar=iRtoB(ip)
            endif
         enddo
      endif
      end
      



      
      subroutine assign_emsca_and_flow_statistical(xmcxsec,xmcxsec2
     $     ,MCsec,lzone,jflow,wgt)
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include "born_nhel.inc"
c$$$      include 'madfks_mcatnlo.inc'
      include "genps.inc"
      include 'nFKSconfigs.inc'
      double precision tiny
      parameter       (tiny=1d-7)
      integer npartner,cflows,i,jflow,jpartner,mpartner
      double precision xmcxsec(nexternal),xmcxsec2(max_bcol),wgt,wgt2,
     $     sumMCsec(max_bcol),MCsec(nexternal,max_bcol),rrnd,wgt1
     $     ,dummy
      logical lzone(nexternal)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer          ipartners(0:nexternal-1)
     &                          ,colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
c Jamp amplitudes of the Born (to be filled with a call the sborn())
      double Precision amp2(ngraphs),jamp2(0:ncolor)
      common/to_amps/  amp2         ,jamp2
c Stuff to be written onto the LHE file
      integer iSorH_lhe,ifks_lhe(fks_configs) ,jfks_lhe(fks_configs)
     &     ,fksfather_lhe(fks_configs) ,ipartner_lhe(fks_configs)
      double precision scale1_lhe(fks_configs),scale2_lhe(fks_configs)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     &                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe
      double precision emsca,emsca_bare,           scalemin,scalemax
      common /cemsca/  emsca,emsca_bare,scalemin,scalemax
      double precision qMC
      common /cqMC/    qMC
      INTEGER              NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      double precision ran2
      external ran2
      if (mcatnlo_delta) then
c Input check
         do npartner=1,ipartners(0)
            if(xmcxsec(npartner).lt.0d0)then
               write(*,*)'Fatal error 1 in complete_xmcsubt'
               write(*,*)npartner,xmcxsec(npartner)
               stop
            endif
         enddo
         do cflows=1,max_bcol
            if(xmcxsec2(cflows).lt.0d0)then
               write(*,*)'Fatal error 2 in complete_xmcsubt'
               write(*,*)cflows,xmcxsec2(cflows)
               stop
            endif
         enddo

c Compute MC cross section
         wgt=0d0
         wgt2=0d0
         do i=1,max_bcol
            sumMCsec(i)=0d0
         enddo
         do npartner=1,ipartners(0)
            wgt=wgt+xmcxsec(npartner)
         enddo
         do cflows=1,max_bcol
            wgt2=wgt2+xmcxsec2(cflows)
         enddo
         do cflows=1,max_bcol
            do npartner=1,ipartners(0)
               sumMCsec(cflows)=sumMCsec(cflows)+MCsec(npartner,cflows)
            enddo
         enddo
c     
         if((abs(wgt).gt.1.d-10 .and.abs(wgt-wgt2)/abs(wgt).gt.tiny).or.
     &        (abs(wgt).le.1.d-10 .and.abs(wgt-wgt2).gt.tiny) )then
            write(*,*)'Fatal error 3 in complete_xmcsubt'
            write(*,*)wgt,wgt2
            stop
         endif
         do cflows=1,max_bcol
            if( (abs(sumMCsec(cflows)).gt.1.d-10 .and.
     &            abs(sumMCsec(cflows)-xmcxsec2(cflows))/
     &            abs(sumMCsec(cflows)).gt.tiny) .or.
     &           (abs(sumMCsec(cflows)).le.1.d-10 .and.
     &            abs(sumMCsec(cflows)-xmcxsec2(cflows)).gt.tiny) )then
               write(*,*)'Fatal error 3 in complete_xmcsubt'
               write(*,*)sumMCsec(cflows),xmcxsec2(cflows)
               stop
            endif
         enddo

c Assign flow on statistical basis
         if (wgt2.gt.0d0) then
            ! use born-bars times kernels
            rrnd=ran2()
            wgt1=0d0
            jflow=0
            cflows=0
            do while(jflow.eq.0.and.cflows.lt.max_bcol)
               cflows=cflows+1
               wgt1=wgt1+xmcxsec2(cflows)
               if(wgt1.ge.rrnd*wgt2)jflow=cflows
            enddo
            if(jflow.eq.0)then
               write(*,*)'Error in xmcsubt: flow unweighting failed'
               stop
            endif
         else
             ! use the born-bars
            call sborn(p_born,dummy)
            wgt1=0.d0
            do i=1,max_bcol
               wgt1=wgt1+jamp2(i)
            enddo
            wgt2=ran2()*wgt1
            jflow=0
            wgt1=0d0
            do while (wgt1 .lt. wgt2)
               jflow=jflow+1
               wgt1=wgt1+jamp2(jflow)
            enddo
         endif
c Assign emsca (scalar) on statistical basis -- ensure backward compatibility
         if(wgt.gt.1d-30)then
            rrnd=ran2()
            wgt1=0d0
            jpartner=0
            do npartner=1,ipartners(0)
               if(lzone(npartner).and.jpartner.eq.0)then
                  wgt1=wgt1+MCsec(npartner,jflow)
                  if(wgt1.ge.rrnd*xmcxsec2(jflow))then
                     jpartner=ipartners(npartner)
                     mpartner=npartner
                  endif
               endif
            enddo
            if(jpartner.eq.0)then
               write(*,*)'Error in xmcsubt: emsca unweighting failed'
               stop
            else
               emsca=shower_scale_nbody(mpartner,fks_father)
            endif
         else
            emsca=scalemax
         endif

      else                      ! mcatnlo-delta = .false.
c Compute MC cross section
         wgt=0d0
         do npartner=1,ipartners(0)
            wgt=wgt+xmcxsec(npartner)
         enddo
c Assign emsca on statistical basis
         if(wgt.gt.1d-30)then
            rrnd=ran2()
            wgt1=0d0
            jpartner=0
            do npartner=1,ipartners(0)
               if(lzone(npartner).and.jpartner.eq.0)then
                  wgt1=wgt1+xmcxsec(npartner)
                  if(wgt1.ge.rrnd*wgt)then
                     jpartner=ipartners(npartner)
                     mpartner=npartner
                  endif
               endif
            enddo
            if(jpartner.eq.0)then
               write(*,*)'Error in xmcsubt: emsca unweighting failed'
               stop
            else
               emsca=shower_scale_nbody(mpartner,fks_father)
            endif
         else
            emsca=scalemax
         endif
      endif


c Additional information for LHE
      ifks_lhe(nFKSprocess)=i_fks
      jfks_lhe(nFKSprocess)=j_fks
      fksfather_lhe(nFKSprocess)=min(i_fks,j_fks)
      if(jpartner.ne.0)then
         ipartner_lhe(nFKSprocess)=jpartner
      else
c min() avoids troubles if ran2()=1
         ipartner_lhe(nFKSprocess)=min(int(ran2()*ipartners(0))+1,
     $        ipartners(0) )
         ipartner_lhe(nFKSprocess)=
     $        ipartners(ipartner_lhe(nFKSprocess))
      endif
      scale1_lhe(nFKSprocess)=qMC
      if(emsca.lt.scalemin)then
         write(*,*)'Error in xmcsubt: emsca too small'
         write(*,*)emsca,jpartner,lzone
         stop
      endif
      return
      end



      function get_to_zero(sc,xlow,xupp)
      implicit none
      double precision get_to_zero,xlow,xupp,sc
      double precision x,emscafun
      x=(xupp-sc)/(xupp-xlow)
      get_to_zero=1-emscafun(x,2d0)
      return
      end


      function dipole_mass(p,i,j)
      implicit none
      include 'nexternal.inc'
      double precision dipole_mass,sign,tmp
      double precision p(0:3,nexternal)
      integer i,j,k
c
      sign=1.d0
      if(i.le.2)sign=-sign
      if(j.le.2)sign=-sign
      tmp=(p(0,i)+sign*p(0,j))**2
      do k=1,3
        tmp=tmp-(p(k,i)+sign*p(k,j))**2
      enddo
      tmp=sqrt(max(0.d0,tmp))
      dipole_mass=tmp
      return
      end


      subroutine assign_ifks_Hscale(ipdg,ifksscl,fksscales)
      implicit none
      double precision fksscales(3)
      integer ipdg,ifksscl(2),i,icount
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical wrong
c Set itype=0 to set scale according to the colour line to which i_fks belongs
c     itype=1 to take the minimum of the two scales in the case of mother=gluon
      integer itype
      parameter (itype=0)
c
      fksscales(3)=1.d10
      if(itype.eq.0)then
        wrong=.not.((ifksscl(1).eq.1.and.ifksscl(2).eq.0).or.
     &              (ifksscl(1).eq.0.and.ifksscl(2).eq.1))
        if(wrong)then
          write(*,*)'Something wrong in assign_ifks_Hscale (0):'
          write(*,*)ipdg,icount,i_fks,j_fks
          write(*,*)ifksscl(1),ifksscl(2),fksscales(1),fksscales(2)
          stop
        endif
        if(ifksscl(1).ne.0)fksscales(3)=fksscales(1)
        if(ifksscl(2).ne.0)fksscales(3)=fksscales(2)
      elseif(itype.eq.1)then
        do i=1,2
          if(fksscales(i).gt.0d0)
     #      fksscales(3)=min(fksscales(3),fksscales(i))
        enddo
      endif
      if(fksscales(3).eq.1.d10)then
        write(*,*)'Could not assign scale in assign_ifks_Hscale:'
        write(*,*)ipdg,ifksscl(1),ifksscl(2),fksscales(1),fksscales(2)
        stop
      endif
      return
      end


      subroutine get_mbar(p,y_ij_fks,ileg,bornbars,bornbarstilde)
c Computes barred amplitudes (bornbars) squared according
c to Odagiri's prescription (hep-ph/9806531).
c Computes barred azimuthal amplitudes (bornbarstilde) with
c the same method 
      implicit none

      include "genps.inc"
      include "nexternal.inc"
      include "born_nhel.inc"
      include "orders.inc"

      double precision p(0:3,nexternal)
      double precision y_ij_fks,bornbars(max_bcol,nsplitorders),
     &                          bornbarstilde(max_bcol,nsplitorders)

      double precision zero
      parameter (zero=0.d0)
      double complex czero
      parameter (czero=dcmplx(0d0,0d0))
      double precision p_born_rot(0:3,nexternal-1)

      integer imother_fks,ileg

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double Precision amp2(ngraphs), jamp2(0:ncolor)
      common/to_amps/  amp2,       jamp2

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision wgt_born
      double complex W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

      double complex xij_aor
      common/cxij_aor/xij_aor

      double precision sumborn
      integer i

      double precision vtiny,pi(0:3),pj(0:3),cphi_mother,sphi_mother
      parameter (vtiny=1d-12)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))

      double precision xi_i_fks_ev,y_ij_fks_ev,t
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      double precision iden_comp
      common /c_iden_comp/iden_comp

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      double precision ch_i,ch_j,ch_m
      common/cparticle_types/i_type,j_type,m_type,ch_i,ch_j,ch_m

      double precision born(nsplitorders)
      double complex borntilde(nsplitorders)
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      complex*16 ans_cnt(2, nsplitorders), wgt1(2)
      common /c_born_cnt/ ans_cnt
      double complex ans_extra_cnt(2,nsplitorders)
      integer iord, iextra_cnt, isplitorder_born, isplitorder_cnt
      common /c_extra_cnt/iextra_cnt, isplitorder_born, isplitorder_cnt

      integer iamp
      double precision amp_split_born(amp_split_size,nsplitorders) 
      double complex amp_split_borntilde(amp_split_size,nsplitorders)
      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
     $                 amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
      common /to_amp_split_bornbars/amp_split_bornbars,
     $                              amp_split_bornbarstilde
c
      logical is_leading_cflow(max_bcol)
      integer num_leading_cflows
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows
      
c
c BORN/BORNTILDE
C check if momenta have to be rotated
      if ((ileg.eq.1.or.ileg.eq.2) .and.
     &    (j_fks.eq.2 .and. nexternal-1.ne.3)) then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed.
c Exclude 2->1 (at the Born level) processes: matrix elements are
c independent of the PS point, but non-zero helicity configurations
c might flip when rotating the momenta.
         do i=1,nexternal-1
            p_born_rot(0,i)=p_born(0,i)
            p_born_rot(1,i)=-p_born(1,i)
            p_born_rot(2,i)=p_born(2,i)
            p_born_rot(3,i)=-p_born(3,i)
         enddo
         calculatedBorn=.false.
         call sborn(p_born_rot,wgt_born)
         if (iextra_cnt.gt.0) call extra_cnt(p_born_rot, iextra_cnt, ans_extra_cnt)
         calculatedBorn=.false.
      else
         call sborn(p_born,wgt_born)
         if (iextra_cnt.gt.0) call extra_cnt(p_born, iextra_cnt, ans_extra_cnt)
      endif

      do iord = 1, nsplitorders
        if (.not.split_type(iord).or.(iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
C check if any extra_cnt is needed
        if (iextra_cnt.gt.0) then
            write(*,*) 'FIXEXTRACNTMC'
            stop
            if (iord.eq.isplitorder_born) then
            ! this is the contribution from the born ME
               wgt1(1) = ans_cnt(1,iord)
               wgt1(2) = ans_cnt(2,iord)
            else if (iord.eq.isplitorder_cnt) then
            ! this is the contribution from the extra cnt
               wgt1(1) = ans_extra_cnt(1,iord)
               wgt1(2) = ans_extra_cnt(2,iord)
            else
               write(*,*) 'ERROR in sborncol_isr', iord
               stop
            endif
        else
           wgt1(1) = ans_cnt(1,iord)
           wgt1(2) = ans_cnt(2,iord)
        endif
        if (abs(m_type).eq.3.or.dabs(ch_m).gt.0d0) wgt1(2) = czero
        born(iord) = dble(wgt1(1))
        borntilde(iord) = wgt1(2)
        do iamp=1, amp_split_size
          amp_split_born(iamp,iord) = dble(amp_split_cnt(iamp,1,iord))
          if (abs(m_type).eq.3.or.dabs(ch_m).gt.0d0) then
            amp_split_borntilde(iamp,iord) = czero
          else
            amp_split_borntilde(iamp,iord) = amp_split_cnt(iamp,2,iord)
          endif
        enddo
      enddo
      
c BORN TILDE
      if(ileg.eq.1.or.ileg.eq.2)then
c Insert <ij>/[ij] which is not included by sborn()
         if (1d0-y_ij_fks.lt.vtiny)then
            azifact=xij_aor
         else
            do i=0,3
               pi(i)=p_i_fks_ev(i)
               pj(i)=p(i,j_fks)
            enddo
            if(j_fks.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed
               pi(1)=-pi(1)
               pi(3)=-pi(3)
               pj(1)=-pj(1)
               pj(3)=-pj(3)
            endif
            CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
            CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
            CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
            CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
            Wij_angle=(0d0,0d0)
            Wij_recta=(0d0,0d0)
            do i=1,4
               Wij_angle = Wij_angle + W1(i)*W2(i)
               Wij_recta = Wij_recta + W3(i)*W4(i)
            enddo
            azifact=Wij_angle/Wij_recta
         endif
c Insert the extra factor due to Madgraph convention for polarization vectors
         if(j_fks.eq.2)then
            cphi_mother=-1.d0
            sphi_mother=0.d0
         else
            cphi_mother=1.d0
            sphi_mother=0.d0
         endif
         do iord=1, nsplitorders
           borntilde(iord) = -(cphi_mother+ximag*sphi_mother)**2 *
     #                borntilde(iord) * dconjg(azifact)
           do iamp=1, amp_split_size
             amp_split_borntilde(iamp,iord) = -(cphi_mother+ximag*sphi_mother)**2 *
     #                amp_split_borntilde(iamp,iord) * dconjg(azifact)
            enddo
         enddo
      elseif(ileg.eq.3.or.ileg.eq.4)then
         if((abs(j_type).eq.3.or.ch_j.ne.0d0).and.
     &     (i_type.eq.8.or.i_type.eq.1).and.
     &     ch_i.eq.0d0)then
            do iord=1, nsplitorders
               borntilde(iord)=czero
               do iamp=1, amp_split_size
                 amp_split_borntilde(iamp,iord) = czero
               enddo
            enddo
         elseif((m_type.eq.8.or.m_type.eq.1).and.ch_m.eq.0d0)then
c Insert <ij>/[ij] which is not included by sborn()
            if(1.d0-y_ij_fks.lt.vtiny)then
               azifact=xij_aor
            else
               do i=0,3
                  pi(i)=p_i_fks_ev(i)
                  pj(i)=p(i,j_fks)
               enddo
               CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
               CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
               CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
               CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
               Wij_angle=(0d0,0d0)
               Wij_recta=(0d0,0d0)
               do i=1,4
                  Wij_angle = Wij_angle + W1(i)*W2(i)
                  Wij_recta = Wij_recta + W3(i)*W4(i)
               enddo
               azifact=Wij_angle/Wij_recta
            endif
c Insert the extra factor due to Madgraph convention for polarization vectors
            imother_fks=min(i_fks,j_fks)
            call getaziangles(p_born(0,imother_fks),
     #                        cphi_mother,sphi_mother)
            do iord=1, nsplitorders
               borntilde(iord) = -(cphi_mother-ximag*sphi_mother)**2 *
     #                  borntilde(iord) * azifact
               do iamp=1, amp_split_size
                 amp_split_borntilde(iamp,iord) = -(cphi_mother-ximag*sphi_mother)**2 *
     #                amp_split_borntilde(iamp,iord) * azifact
               enddo
            enddo
         else
            write(*,*)'FATAL ERROR in get_mbar',
     #           i_type,j_type,i_fks,j_fks
            stop
         endif
      else
         write(*,*)'unknown ileg in get_mbar'
         stop
      endif

CMZ! this has to be all changed according to the correct jamps

c born is the total born amplitude squared
      sumborn=0.d0
      do i=1,max_bcol
         if(is_leading_cflow(i))sumborn=sumborn+jamp2(i)
c sumborn is the sum of the leading-color amplitudes squared
      enddo


c BARRED AMPLITUDES
      do i=1,max_bcol
        do iord=1,nsplitorders
          if (sumborn.ne.0d0.and.is_leading_cflow(i)) then
            bornbars(i,iord)=jamp2(i)/sumborn * born(iord) *iden_comp
            do iamp=1,amp_split_size
              amp_split_bornbars(iamp,i,iord)=jamp2(i)/sumborn * 
     &                              amp_split_born(iamp,iord) *iden_comp
            enddo
          elseif (born(iord).eq.0d0 .or. jamp2(i).eq.0d0
     &           .or..not.is_leading_cflow(i)) then
            bornbars(i,iord)=0d0
            do iamp=1,amp_split_size
              amp_split_bornbars(iamp,i,iord)=0d0
            enddo
          else
            write (*,*) 'ERROR #1, dividing by zero'
            stop
          endif
          if (sumborn.ne.0d0.and.is_leading_cflow(i)) then
            bornbarstilde(i,iord)=jamp2(i)/sumborn * dble(borntilde(iord)) *iden_comp
            do iamp=1,amp_split_size
              amp_split_bornbarstilde(iamp,i,iord)=jamp2(i)/sumborn * 
     &                      dble(amp_split_borntilde(iamp,iord)) *iden_comp
            enddo
          elseif (borntilde(iord).eq.0d0 .or. jamp2(i).eq.0d0
     &           .or..not.is_leading_cflow(i)) then
            bornbarstilde(i,iord)=0d0
            do iamp=1,amp_split_size
              amp_split_bornbarstilde(iamp,i,iord)=0d0 
            enddo
          else
            write (*,*) 'ERROR #2, dividing by zero'
            stop
          endif      
c bornbars(i) is the i-th leading-color amplitude squared re-weighted
c in such a way that the sum of bornbars(i) is born rather than sumborn.
c the same holds for bornbarstilde(i).
        enddo
      enddo

      return
      end

      subroutine get_born_flow(flow_picked)
      ! This assumes that the Born matrix elements are called. This is
      ! always the case if either the compute_born or the virtual
      ! (through bornsoftvirtual) are evaluated.
      implicit none
      integer flow_picked
c sumborn is the sum of the leading colour flow contributions to the Born.
      sumborn=0.d0
      do i=1,max_bcol
         if(is_leading_cflow(i)) sumborn=sumborn+jamp2(i)
      enddo
      target=ran2()*sumborn
      sum=0d0
      do i=1,max_bcol
         if (.not.is_leading_cflow(i)) cycle
         sum=sum+jamp2(i)
         if(sum.gt.target) then
            flow_picked=i
            return
         endif
      enddo
      write (*,*) 'Error #1 in get_born_flow',sum,target,i
      stop 1
      end

      function gfunction(w,alpha,beta,delta)
c Gets smoothly to 0 as w goes to 1.
c Call with
c   alpha > 1, or alpha < 0; if alpha < 0, gfunction = 1;
c   0 < |beta| <= 1;
c   0 < delta <= 2.
      implicit none
      double precision tiny
      parameter (tiny=1.d-5)
      double precision gfunction,alpha,beta,delta,w,wmin,wg,tt,tmp
      logical firsttime
      save firsttime
      data firsttime /.true./
      double precision cutoff,cutoff2
      parameter(cutoff=1d0)
      parameter(cutoff2=0.99d0)
c
c     set cutoff < 1 and cutoff2 = cutoff in the final version
c
      if(firsttime)then
        firsttime=.false.
        if(alpha.ge.0d0.and.alpha.lt.1d0)then
          write(*,*)'Incorrect alpha in gfunction',alpha
          stop
        endif
        if(abs(beta).gt.1d0)then
          write(*,*)'Incorrect beta in gfunction',beta
          stop
        endif
        if(delta.gt.2d0.or.delta.le.0d0)then
          write(*,*)'Incorrect delta in gfunction',delta
          stop
        endif
      endif
c
      tmp=1d0
      if(alpha.gt.0d0)then
        if(beta.lt.0d0)then
          wmin=0d0
        else
          wmin=max(0d0,1d0-delta)
        endif
        wg=min(1d0-(1-wmin)*abs(beta),cutoff-tiny)
        if(abs(w).gt.wg.and.abs(w).lt.cutoff2)then
          tt=(abs(w)-wg)/(cutoff-wg)
          if(tt.gt.1d0)then
            write(*,*)'Fatal error in gfunction',tt
            stop
          endif
          tmp=(1-tt)**(2*alpha)/(tt**(2*alpha)+(1-tt)**(2*alpha))
        elseif(abs(w).ge.cutoff2)then
          tmp=0d0
        endif
      endif
      gfunction=tmp
      return
      end



c$$$      subroutine kinematics_driver(xi_i_fks,y_ij_fks,sh,pp,ileg,xm12
c$$$     $     ,xm22,xtk,xuk,xq1q,xq2q,qMC)
c$$$c Determines Mandelstam invariants and assigns ileg and shower-damping
c$$$c variable qMC
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      include "coupl.inc"
c$$$      include "run.inc"
c$$$      double precision pp(0:3,nexternal),pp_rec(0:3)
c$$$      double precision xi_i_fks,y_ij_fks,xij
c$$$
c$$$      integer ileg,j,i,nfinal
c$$$      double precision xp1(0:3),xp2(0:3),xk1(0:3),xk2(0:3),xk3(0:3)
c$$$      common/cpkmomenta/xp1,xp2,xk1,xk2,xk3
c$$$      double precision sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12,xm22
c$$$      double precision qMC,zPY8,zeta1,zeta2,get_zeta,z,qMCarg,dot
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$      integer fksfather
c$$$      integer i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      double precision tiny
c$$$      parameter(tiny=1d-5)
c$$$      double precision zero
c$$$      parameter(zero=0d0)
c$$$
c$$$      integer isqrtneg
c$$$      save isqrtneg
c$$$
c$$$      double precision pmass(nexternal)
c$$$      include "pmass.inc"
c$$$
c$$$c Initialise
c$$$      do i=0,3
c$$$         pp_rec(i)=0d0
c$$$         xp1(i)=0d0
c$$$         xp2(i)=0d0
c$$$         xk1(i)=0d0
c$$$         xk2(i)=0d0
c$$$         xk3(i)=0d0
c$$$      enddo
c$$$      nfinal=nexternal-2
c$$$      xm12=0d0
c$$$      xm22=0d0
c$$$      xq1q=0d0
c$$$      xq2q=0d0
c$$$      qMC=-1d0
c$$$
c$$$c Discard if unphysical FKS variables
c$$$      if(xi_i_fks.lt.0d0.or.xi_i_fks.gt.1d0.or.
c$$$     &   abs(y_ij_fks).gt.1d0)then
c$$$         write(*,*)'Error 0 in kinematics_driver: fks variables'
c$$$         write(*,*)xi_i_fks,y_ij_fks
c$$$         stop
c$$$      endif
c$$$
c$$$c Determine ileg
c$$$c ileg = 1 ==> emission from left     incoming parton
c$$$c ileg = 2 ==> emission from right    incoming parton
c$$$c ileg = 3 ==> emission from massive  outgoing parton
c$$$c ileg = 4 ==> emission from massless outgoing parton
c$$$c Instead of pmass(j_fks), one should use pmass(fksfather), but the
c$$$c kernels where pmass(fksfather) != pmass(j_fks) are non-singular
c$$$      fksfather=min(i_fks,j_fks)
c$$$      if(fksfather.le.2)then
c$$$        ileg=fksfather
c$$$      elseif(pmass(j_fks).ne.0d0)then
c$$$        ileg=3
c$$$      elseif(pmass(j_fks).eq.0d0)then
c$$$        ileg=4
c$$$      else
c$$$        write(*,*)'Error 1 in kinematics_driver: unknown ileg'
c$$$        write(*,*)ileg,fksfather,pmass(j_fks)
c$$$        stop
c$$$      endif
c$$$
c$$$c Determine and assign momenta:
c$$$c xp1 = incoming left parton  (emitter (recoiler) if ileg = 1 (2))
c$$$c xp2 = incoming right parton (emitter (recoiler) if ileg = 2 (1))
c$$$c xk1 = outgoing parton       (emitter (recoiler) if ileg = 3 (4))
c$$$c xk2 = outgoing parton       (emitter (recoiler) if ileg = 4 (3))
c$$$c xk3 = extra parton          (FKS parton)
c$$$      do j=0,3
c$$$c xk1 and xk2 are never used for ISR
c$$$         xp1(j)=pp(j,1)
c$$$         xp2(j)=pp(j,2)
c$$$         xk3(j)=pp(j,i_fks)
c$$$         if(ileg.gt.2)pp_rec(j)=pp(j,1)+pp(j,2)-pp(j,i_fks)-pp(j,j_fks)
c$$$         if(ileg.eq.3)then
c$$$            xk1(j)=pp(j,j_fks)
c$$$            xk2(j)=pp_rec(j)
c$$$         elseif(ileg.eq.4)then
c$$$            xk1(j)=pp_rec(j)
c$$$            xk2(j)=pp(j,j_fks)
c$$$         endif
c$$$      enddo
c$$$
c$$$c Determine the Mandelstam invariants needed in the MC functions in terms
c$$$c of FKS variables: the argument of MC functions are (p+k)^2, NOT 2 p.k
c$$$c
c$$$c Definitions of invariants in terms of momenta
c$$$c
c$$$c xm12 =     xk1 . xk1
c$$$c xm22 =     xk2 . xk2
c$$$c xtk  = - 2 xp1 . xk3
c$$$c xuk  = - 2 xp2 . xk3
c$$$c xq1q = - 2 xp1 . xk1 + xm12
c$$$c xq2q = - 2 xp2 . xk2 + xm22
c$$$c w1   = + 2 xk1 . xk3        = - xq1q + xq2q - xtk
c$$$c w2   = + 2 xk2 . xk3        = - xq2q + xq1q - xuk
c$$$c xq1c = - 2 xp1 . xk2        = - s - xtk - xq1q + xm12
c$$$c xq2c = - 2 xp2 . xk1        = - s - xuk - xq2q + xm22
c$$$c
c$$$c Parametrisation of invariants in terms of FKS variables
c$$$c
c$$$c ileg = 1
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xk1  =  irrelevant
c$$$c xk2  =  irrelevant
c$$$c yi = y_ij_fks
c$$$c x = 1 - xi_i_fks
c$$$c B = sqrt(s)/2*(1-x)
c$$$c
c$$$c ileg = 2
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , -yi )
c$$$c xk1  =  irrelevant
c$$$c xk2  =  irrelevant
c$$$c yi = y_ij_fks
c$$$c x = 1 - xi_i_fks
c$$$c B = sqrt(s)/2*(1-x)
c$$$c
c$$$c ileg = 3
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c$$$c xk1  =  ( sqrt(veckn_ev**2+xm12) , 0 , 0 , veckn_ev )
c$$$c xk2  =  xp1 + xp2 - xk1 - xk3
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
c$$$c yj = y_ij_fks
c$$$c yi = irrelevant
c$$$c x = 1 - xi_i_fks
c$$$c veckn_ev is such that xk2**2 = xm22
c$$$c B = sqrt(s)/2*(1-x)
c$$$c azimuth = irrelevant (hence set = 0)
c$$$c
c$$$c ileg = 4
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c$$$c xk1  =  xp1 + xp2 - xk2 - xk3
c$$$c xk2  =  A * ( 1 , 0 , 0 , 1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
c$$$c yj = y_ij_fks
c$$$c yi = irrelevant
c$$$c x = 1 - xi_i_fks
c$$$c A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-yj)))
c$$$c B = sqrt(s)/2*(1-x)
c$$$c azimuth = irrelevant (hence set = 0)
c$$$
c$$$      if(ileg.eq.1)then
c$$$         xtk=-sh*xi_i_fks*(1-y_ij_fks)/2
c$$$         xuk=-sh*xi_i_fks*(1+y_ij_fks)/2
c$$$         if(shower_mc_mod.eq.'HERWIG6'  .or.
c$$$     &        shower_mc_mod.eq.'HERWIGPP' )qMC=xi_i_fks/2*sqrt(sh*(1-y_ij_fks**2))
c$$$         if(shower_mc_mod.eq.'PYTHIA6Q' )qMC=sqrt(-xtk)
c$$$         if(shower_mc_mod.eq.'PYTHIA6PT'.or.
c$$$     &        shower_mc_mod.eq.'PYTHIA8'  )qMC=sqrt(-xtk*xi_i_fks)
c$$$      elseif(ileg.eq.2)then
c$$$         xtk=-sh*xi_i_fks*(1+y_ij_fks)/2
c$$$         xuk=-sh*xi_i_fks*(1-y_ij_fks)/2
c$$$         if(shower_mc_mod.eq.'HERWIG6'  .or.
c$$$     &        shower_mc_mod.eq.'HERWIGPP' )qMC=xi_i_fks/2*sqrt(sh*(1-y_ij_fks**2))
c$$$         if(shower_mc_mod.eq.'PYTHIA6Q' )qMC=sqrt(-xuk)
c$$$         if(shower_mc_mod.eq.'PYTHIA6PT'.or.
c$$$     &        shower_mc_mod.eq.'PYTHIA8'  )qMC=sqrt(-xuk*xi_i_fks)
c$$$      elseif(ileg.eq.3)then
c$$$         xm12=pmass(j_fks)**2
c$$$         xm22=dot(pp_rec,pp_rec)
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         xq1q=-2*dot(xp1,xk1)+xm12
c$$$         xq2q=-2*dot(xp2,xk2)+xm22
c$$$         w1=-xq1q+xq2q-xtk
c$$$         w2=-xq2q+xq1q-xuk
c$$$         if(shower_mc_mod.eq.'HERWIG6'.or.
c$$$     &        shower_mc_mod.eq.'HERWIGPP')then
c$$$            zeta1=get_zeta(sh,w1,w2,xm12,xm22)
c$$$            qMCarg=zeta1*((1-zeta1)*w1-zeta1*xm12)
c$$$            if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny)qMCarg=0d0
c$$$            if(qMCarg.lt.-tiny) then
c$$$               isqrtneg=isqrtneg+1
c$$$               write(*,*)'Error 2 in kinematics_driver: negtive sqrt'
c$$$               write(*,*)qMCarg,isqrtneg
c$$$               if(isqrtneg.ge.100)stop
c$$$            endif
c$$$            qMC=sqrt(qMCarg)
c$$$         elseif(shower_mc_mod.eq.'PYTHIA6Q')then
c$$$            qMC=sqrt(w1+xm12)
c$$$         elseif(shower_mc_mod.eq.'PYTHIA6PT')then
c$$$            write(*,*)'PYTHIA6PT not available for FSR'
c$$$            stop
c$$$         elseif(shower_mc_mod.eq.'PYTHIA8')then
c$$$            z=zPY8(ileg,xm12,xm22,sh,1d0-xi_i_fks,0d0,y_ij_fks,xtk
c$$$     $              ,xuk,xq1q,xq2q)
c$$$            qMC=sqrt(z*(1-z)*w1)
c$$$         endif
c$$$      elseif(ileg.eq.4)then
c$$$         xm12=dot(pp_rec,pp_rec)
c$$$         xm22=0d0
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         xij=2*(1-xm12/sh-xi_i_fks)/(2-xi_i_fks*(1-y_ij_fks))
c$$$         w2=sh*xi_i_fks*xij*(1-y_ij_fks)/2d0
c$$$         xq2q=-sh*xij*(2-dot(xp1,xk2)*4d0/(sh*xij))/2d0
c$$$         xq1q=xuk+xq2q+w2
c$$$         w1=-xq1q+xq2q-xtk
c$$$         if(shower_mc_mod.eq.'HERWIG6'.or.
c$$$     &        shower_mc_mod.eq.'HERWIGPP')then
c$$$            zeta2=get_zeta(sh,w2,w1,xm22,xm12)
c$$$            qMCarg=zeta2*(1-zeta2)*w2
c$$$            if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny)qMCarg=0d0
c$$$            if(qMCarg.lt.-tiny)then
c$$$               isqrtneg=isqrtneg+1
c$$$               write(*,*)'Error 3 in kinematics_driver: negtive sqrt'
c$$$               write(*,*)qMCarg,isqrtneg
c$$$               if(isqrtneg.ge.100)stop
c$$$            endif
c$$$            qMC=sqrt(qMCarg)
c$$$         elseif(shower_mc_mod.eq.'PYTHIA6Q')then
c$$$            qMC=sqrt(w2)
c$$$         elseif(shower_mc_mod.eq.'PYTHIA6PT')then
c$$$            write(*,*)'PYTHIA6PT not available for FSR'
c$$$            stop
c$$$         elseif(shower_mc_mod.eq.'PYTHIA8')then
c$$$            z=zPY8(ileg,xm12,xm22,sh,1d0-xi_i_fks,0d0,y_ij_fks,xtk
c$$$     $           ,xuk,xq1q,xq2q)
c$$$            qMC=sqrt(z*(1-z)*w2)
c$$$         endif
c$$$      else
c$$$         write(*,*)'Error 4 in kinematics_driver: assigned wrong ileg'
c$$$         stop
c$$$      endif
c$$$
c$$$c Checks on invariants
c$$$      call check_invariants(ileg,sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12,xm22)
c$$$
c$$$      return
c$$$      end


c$$$      block data check_invariants_block
c$$$      integer imprecision(7),max_imprecision
c$$$      common /c_check_invariants/ max_imprecision,imprecision
c$$$      data imprecision /7*0/
c$$$      data max_imprecision /10/
c$$$      end
c$$$
c$$$      subroutine check_invariants(ileg,sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12
c$$$     $     ,xm22)
c$$$      implicit none
c$$$      integer ileg
c$$$      double precision sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12,xm22
c$$$      double precision tiny,dot
c$$$      parameter(tiny=1d-5)
c$$$      double precision xp1(0:3),xp2(0:3),xk1(0:3),xk2(0:3),xk3(0:3)
c$$$      common/cpkmomenta/xp1,xp2,xk1,xk2,xk3
c$$$      integer imprecision(7),max_imprecision
c$$$      common /c_check_invariants/ max_imprecision,imprecision
c$$$
c$$$      if(ileg.le.2)then
c$$$         if((abs(xtk+2*dot(xp1,xk3))/sh.ge.tiny).or.
c$$$     &      (abs(xuk+2*dot(xp2,xk3))/sh.ge.tiny))then
c$$$            write(*,*)'Warning: imprecision 1 in check_invariants'
c$$$            write(*,*)abs(xtk+2*dot(xp1,xk3))/sh,
c$$$     &                abs(xuk+2*dot(xp2,xk3))/sh
c$$$            imprecision(1)=imprecision(1)+1
c$$$            if (imprecision(1).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' imprecisions. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$c
c$$$      elseif(ileg.eq.3)then
c$$$         if(sqrt(w1+xm12).ge.sqrt(sh)-sqrt(xm22))then
c$$$            write(*,*)'Warning: imprecision 2 in check_invariants'
c$$$            write(*,*)sqrt(w1),sqrt(sh),xm22
c$$$            imprecision(2)=imprecision(2)+1
c$$$            if (imprecision(2).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' imprecisions. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$         if(((abs(w1-2*dot(xk1,xk3))/sh.ge.tiny)).or.
c$$$     &      ((abs(w2-2*dot(xk2,xk3))/sh.ge.tiny)))then
c$$$            write(*,*)'Warning: imprecision 3 in check_invariants'
c$$$            write(*,*)abs(w1-2*dot(xk1,xk3))/sh,
c$$$     &                abs(w2-2*dot(xk2,xk3))/sh
c$$$            imprecision(3)=imprecision(3)+1
c$$$            if (imprecision(3).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' imprecisions. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$         if(xm12.eq.0d0)then
c$$$            write(*,*)'Warning 4 in check_invariants'
c$$$            imprecision(4)=imprecision(4)+1
c$$$            if (imprecision(4).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' warnings. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$c
c$$$      elseif(ileg.eq.4)then
c$$$         if(sqrt(w2).ge.sqrt(sh)-sqrt(xm12))then
c$$$            write(*,*)'Warning: imprecision 5 in check_invariants'
c$$$            write(*,*)sqrt(w2),sqrt(sh),xm12
c$$$            imprecision(5)=imprecision(5)+1
c$$$            if (imprecision(5).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' imprecisions. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$         if(((abs(w2-2*dot(xk2,xk3))/sh.ge.tiny)).or.
c$$$     &      ((abs(xq2q+2*dot(xp2,xk2))/sh.ge.tiny)).or.
c$$$     &      ((abs(xq1q+2*dot(xp1,xk1)-xm12)/sh.ge.tiny)))then
c$$$            write(*,*)'Warning: imprecision 6 in check_invariants'
c$$$            write(*,*)abs(w2-2*dot(xk2,xk3))/sh,
c$$$     &                abs(xq2q+2*dot(xp2,xk2))/sh,
c$$$     &                abs(xq1q+2*dot(xp1,xk1)-xm12)/sh
c$$$            imprecision(6)=imprecision(6)+1
c$$$            if (imprecision(6).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' imprecisions. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$         if(xm22.ne.0d0)then
c$$$            write(*,*)'Warning 7 in check_invariants'
c$$$            imprecision(7)=imprecision(7)+1
c$$$            if (imprecision(7).ge.max_imprecision) then
c$$$               write (*,*) 'Error: ',max_imprecision
c$$$     $              ,' warnings. Stopping...'
c$$$               stop
c$$$            endif
c$$$         endif
c$$$      endif
c$$$      return
c$$$      end



c Monte Carlo functions
c
c The invariants given in input to these routines follow FNR conventions
c (i.e., are defined as (p+k)^2, NOT 2 p.k). 
c The invariants used inside these routines follow MNR conventions
c (i.e., are defined as -2p.k, NOT (p+k)^2)

c Herwig6

      double precision function zHW6(e0sq)!(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c     Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,e0sq,ss,betae0,beta,zeta,tbeta,get_zeta
c$$$      integer ileg
c$$$      double precision zHW6,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q
c$$$     $     ,tiny,ss,w1,w2,tbeta1,zeta1,tbeta2,zeta2,get_zeta,beta,betae0
c$$$     $     ,betad,betas
      parameter (tiny=1d-5)

c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            zHW6=1-(1-x)*(shat_n1*(1-yi)+4*e0sq*(1+yi))/(8*e0sq)
         elseif(1-yi.lt.tiny)then
            zHW6=x-(1-yi)*(1-x)*(shat_n1*x**2-4*e0sq)/(8*e0sq)
         else
            ss=1-(1+xuk/shat_n1)/(e0sq/xtk)
            if(ss.lt.0d0)goto 999
            zHW6=2*(e0sq/xtk)*(1-sqrt(ss))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            zHW6=1-(1-x)*(shat_n1*(1-yi)+4*e0sq*(1+yi))/(8*e0sq)
         elseif(1-yi.lt.tiny)then
            zHW6=x-(1-yi)*(1-x)*(shat_n1*x**2-4*e0sq)/(8*e0sq)
         else
            ss=1-(1+xtk/shat_n1)/(e0sq/xuk)
            if(ss.lt.0d0)goto 999
            zHW6=2*(e0sq/xuk)*(1-sqrt(ss))
         endif
c
      elseif(ileg.eq.3)then
         if(e0sq.le.(w1+xm12))goto 999
         if(1-x.lt.tiny)then
            beta=1-xm12/shat_n1
            betae0=sqrt(1-xm12/e0sq)
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            zHW6=1+(1-x)*( shat_n1*(yj*betad-betas)/(4*e0sq*(1+betae0))-
     $           betae0*(xm12-xm22+shat_n1*(1+(1+yj)*betad-betas))/
     $           (betad*(xm12-xm22+shat_n1*(1+betad))) )
         else
            tbeta=sqrt(1-(w1+xm12)/e0sq)
            zeta=get_zeta(shat_n1,w1,w2,xm12,xm22)
            zHW6=1-tbeta*zeta-w1/(2*(1+tbeta)*e0sq)
         endif
c
      elseif(ileg.eq.4)then
         if(e0sq.le.w2)goto 999
         if(1-x.lt.tiny)then
            zHW6=1-(1-x)*( (shat_n1-xm12)*(1-yj)/(8*e0sq)+
     &                     shat_n1*(1+yj)/(2*(shat_n1-xm12)) )
         elseif(1-yj.lt.tiny)then
            zHW6=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yj)*(1-x)*(shat_n1*x
     $           -xm12)*( (shat_n1-xm12)**2*(shat_n1*(1-2*x)+xm12)+4
     $           *e0sq*shat_n1*(shat_n1*x-xm12*(2-x)) )/( 8*e0sq
     $           *(shat_n1-xm12)**3 )
         else
            tbeta=sqrt(1-w2/e0sq)
            zeta=get_zeta(shat_n1,w2,w1,xm22,xm12)
            zHW6=1-tbeta*zeta-w2/(2*(1+tbeta)*e0sq)
         endif
c
      else
         write(*,*)'zHW6: unknown ileg'
         stop
      endif

      if(zHW6.lt.0d0.or.zHW6.gt.1d0)goto 999

      return
 999  continue
      zHW6=-1d0
      return
      end



      double precision function xiHW6(e0sq,z)!(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,e0sq,betae0,beta,z
c$$$      integer ileg
c$$$      double precision xiHW6,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q
c$$$     $     ,tiny,z,zHW6,w1,w2,beta,betae0,betad,betas
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            xiHW6=2*shat_n1*(1-yi)/(shat_n1*(1-yi)+4*e0sq*(1+yi))
         elseif(1-yi.lt.tiny)then
            xiHW6=(1-yi)*shat_n1*x**2/(4*e0sq)
         else
            xiHW6=2*(1+xuk/(shat_n1*(1-z)))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            xiHW6=2*shat_n1*(1-yi)/(shat_n1*(1-yi)+4*e0sq*(1+yi))
         elseif(1-yi.lt.tiny)then
            xiHW6=(1-yi)*shat_n1*x**2/(4*e0sq)
         else
            xiHW6=2*(1+xtk/(shat_n1*(1-z)))
         endif
c
      elseif(ileg.eq.3)then
         if(e0sq.le.(w1+xm12))goto 999
         if(1-x.lt.tiny)then
            beta=1-xm12/shat_n1
            betae0=sqrt(1-xm12/e0sq)
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            xiHW6=( shat_n1*(1+betae0)*betad*(xm12-xm22+shat_n1*(1
     $           +betad))*(yj*betad-betas) )/( -4*e0sq*betae0*(1+betae0)
     $           *(xm12-xm22+shat_n1*(1+(1+yj)*betad-betas))+(shat_n1
     $           *betad*(xm12-xm22+shat_n1*(1+betad))*(yj*betad-betas))
     $           )
         else
            xiHW6=w1/(2*z*(1-z)*e0sq)
         endif
c
      elseif(ileg.eq.4)then
         if(e0sq.le.w2)goto 999
         if(1-x.lt.tiny)then
            xiHW6=2*(shat_n1-xm12)**2*(1-yj)/( (shat_n1-xm12)**2*(1-yj)
     $           +4*e0sq*shat_n1*(1+yj) )
         elseif(1-yj.lt.tiny)then
            xiHW6=(shat_n1-xm12)**2*(1-yj)/(4*e0sq*shat_n1)
         else
            xiHW6=w2/(2*z*(1-z)*e0sq)
         endif
c
      else
         write(*,*)'xiHW6: unknown ileg'
         stop
      endif

      if(xiHW6.lt.0d0)goto 999

      return
 999  continue
      xiHW6=-1d0
      return
      end



      double precision function xjacHW6(e0sq,xi,z)
!     $     ,xq1q,xq2q)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,xi,tmp,e0sq,beta,betae0,zmo
     $     ,tbeta,eps,dw1dx,dw2dx,dw1dy,dw2dy
c$$$      integer ileg
c$$$      double precision xjacHW6,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk
c$$$     $     ,xq1q,xq2q,tiny,tmp,z,zHW6,xi,xiHW6,w1,w2,tbeta1,zeta1,dw1dx
c$$$     $     ,dw2dx,dw1dy,dw2dy,tbeta2,get_zeta,beta,betae0,betad,betas
c$$$     $     ,eps1,eps2,beta1,beta2,zmo
      parameter (tiny=1d-5)

c$$$      z=zHW6(e0sq)!(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c$$$      xi=xiHW6(e0sq)!(ileg,e0sq,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0.or.xi.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            tmp=-2*shat_n1/(shat_n1*(1-yi)+4*(1+yi)*e0sq)
         elseif(1-yi.lt.tiny)then
            tmp=-shat_n1*x**2/(4*e0sq)
         else
            tmp=-shat_n1*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            tmp=-2*shat_n1/(shat_n1*(1-yi)+4*(1+yi)*e0sq)
         elseif(1-yi.lt.tiny)then
            tmp=-shat_n1*x**2/(4*e0sq)
         else
            tmp=-shat_n1*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))
         endif
c
      elseif(ileg.eq.3)then
         if(e0sq.le.(w1+xm12))goto 999
         if(1-x.lt.tiny)then
            beta=1-xm12/shat_n1
            betae0=sqrt(1-xm12/e0sq)
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            tmp=( shat_n1*betae0*(1+betae0)*betad*(xm12-xm22+shat_n1*(1
     $           +betad)) )/( (-4*e0sq*(1+betae0)*(xm12-xm22+shat_n1*(1
     $           +betad*(1+yj)-betas)))+(xm12-xm22+shat_n1*(1+betad))
     $           *(xm12*(4+yj*betad-betas)-(xm22-shat_n1)*(yj*betad
     $           -betas)) )
         else
            eps=1-(xm12-xm22)/(shat_n1-w1)
            beta=sqrt(eps**2-4*shat_n1*xm22/(shat_n1-w1)**2)
            tbeta=sqrt(1-(w1+xm12)/e0sq)
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=-(dw1dy*dw2dx-dw1dx*dw2dy)*tbeta/(2*e0sq*z*(1-z)
     $           *(shat_n1-w1)*beta)
         endif
c
      elseif(ileg.eq.4)then
         if(e0sq.le.w2)goto 999
         if(1-x.lt.tiny)then
            zmo=(shat_n1-xm12)*(1-yj)/(8*e0sq)+shat_n1*(1+yj)/(2
     $           *(shat_n1-xm12))
            tmp=-shat_n1/(4*e0sq*zmo)
         elseif(1-yj.lt.tiny)then
            tmp=-(shat_n1-xm12)/(4*e0sq)
         else
            eps=1+xm12/(shat_n1-w2)
            beta=sqrt(eps**2-4*shat_n1*xm12/(shat_n1-w2)**2)
            tbeta=sqrt(1-w2/e0sq)
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=-(dw1dy*dw2dx-dw1dx*dw2dy)*tbeta/(2*e0sq*z*(1-z)
     $           *(shat_n1-w2)*beta)
         endif
c
      else
         write(*,*)'xjacHW6: unknown ileg'
         stop
      endif
      xjacHW6=abs(tmp)

      return
 999  continue
      xjacHW6=0d0
      return
      end



c Hewrig++

      double precision function zHWPP() !(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c     Shower energy variable
      use process_module
      use kinematics_module
      implicit none
!      integer ileg
!      double precision zHWPP,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,
!     &w1,w2,zeta1,zeta2,get_zeta,betad,betas
      double precision tiny,get_zeta,zeta1,zeta2
      parameter (tiny=1d-5)

c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         zHWPP=1-(1-x)*(1+yi)/2d0
c
      elseif(ileg.eq.2)then
         zHWPP=1-(1-x)*(1+yi)/2d0
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/shat_n1)**2-(4*xm22/shat_n1))
c$$$            betas=1+(xm12-xm22)/shat_n1
            zHWPP=1-(1-x)*(1+yj)/(betad+betas)
         else
            zeta1=get_zeta(shat_n1,w1,w2,xm12,xm22)
            zHWPP=1-zeta1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zHWPP=1-(1-x)*(1+yj)*shat_n1/(2*(shat_n1-xm12))
         elseif(1-yj.lt.tiny)then
            zHWPP=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yj)*(1-x)*shat_n1
     $           *(shat_n1*x+xm12*(x-2))*(shat_n1*x-xm12)/(2*(shat_n1
     $           -xm12)**3)
         else
            zeta2=get_zeta(shat_n1,w2,w1,xm22,xm12)
            zHWPP=1-zeta2 
         endif
c
      else
         write(*,*)'zHWPP: unknown ileg'
         stop
      endif

      if(zHWPP.lt.0d0.or.zHWPP.gt.1d0)goto 999

      return
 999  continue
      zHWPP=-1d0
      return
      end



      double precision function xiHWPP()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c     Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
c$$$      integer ileg
c$$$      real*8 xiHWPP,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,w1,w2,
c$$$  &betad,betas,z,zHWPP
      double precision z,zHWPP,tiny
      parameter (tiny=1d-5)

      z=zHWPP()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c 
      if(ileg.eq.1)then
         xiHWPP=shat_n1*(1-yi)/(1+yi)
c
      elseif(ileg.eq.2)then
         xiHWPP=shat_n1*(1-yi)/(1+yi)
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/shat_n1)**2-(4*xm22/shat_n1))
c$$$            betas=1+(xm12-xm22)/shat_n1
            xiHWPP=-shat_n1*(betad+betas)*(yj*betad-betas)/(2*(1+yj))
         else
            xiHWPP=w1/(z*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiHWPP=(1-yj)*(shat_n1-xm12)**2/(shat_n1*(1+yj))
         elseif(1-yj.lt.tiny)then
            xiHWPP=(1-yj)*(shat_n1-xm12)**2/(2*shat_n1)
         else
            xiHWPP=w2/(z*(1-z))
         endif
c
      else
         write(*,*)'xiHWPP: unknown ileg'
         stop
      endif

      if(xiHWPP.lt.0d0)goto 999

      return
 999  continue
      xiHWPP=-1d0
      return
      end



      double precision function xjacHWPP()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q
                                                  !$ ,xq2q)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision z,zHWPP,tmp,eps,beta,dw1dx,dw2dx,dw1dy,dw2dy,tiny
c$$$      integer ileg
c$$$      double precision xjacHWPP,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,
c$$$     &xq2q,tiny,tmp,z,zHWPP,w1,w2,zeta1,dw1dx,dw2dx,dw1dy,dw2dy,get_zeta,
c$$$     &betad,betas,eps1,eps2,beta1,beta2
      parameter (tiny=1d-5)

      tmp=0d0
      z=zHWPP()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk 
c
      if(ileg.eq.1)then
         tmp=-shat_n1/(1+yi)
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1/(1+yi)
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            tmp=-shat_n1*(betad+betas)/(2*(1+yj))
         else
            eps=1-(xm12-xm22)/(shat_n1-w1)
            beta=sqrt(eps**2-4*shat_n1*xm22/(shat_n1-w1)**2)
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=-(dw1dy*dw2dx-dw1dx*dw2dy)/(z*(1-z))/((shat_n1-w1)*beta)
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=-(shat_n1-xm12)/(1+yj)
         elseif(1-yj.lt.tiny)then
            tmp=-(shat_n1-xm12)/2
         else
            eps=1+xm12/(shat_n1-w2)
            beta=sqrt(eps**2-4*shat_n1*xm12/(shat_n1-w2)**2)
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=-(dw1dy*dw2dx-dw1dx*dw2dy)/(z*(1-z))/((shat_n1-w2)*beta)
         endif
c
      else
         write(*,*)'xjacHWPP: unknown ileg'
         stop
      endif
      xjacHWPP=abs(tmp)

      return
 999  continue
      xjacHWPP=0d0
      return
      end



c Pythia6Q

      double precision function zPY6Q()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
c$$$      integer ileg
c$$$      double precision zPY6Q,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,
c$$$     &w1,w2,betad,betas
      parameter(tiny=1d-5)

c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         zPY6Q=x
c
      elseif(ileg.eq.2)then
         zPY6Q=x
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            zPY6Q=1-(2*xm12)/(shat_n1*betas*(betas-betad*yj))
         else
            zPY6Q=1-shat_n1*(1-x)*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)
c This is equation (3.10) of hep-ph/1102.3795. In the partonic
c CM frame it is equal to (xk1(0)+xk3(0)*f)/(xk1(0)+xk3(0)),
c where f = xm12/( s+xm12-xm22-2*sqrt(s)*(xk1(0)+xk3(0)) )
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zPY6Q=1-shat_n1*(1-x)/(shat_n1-xm12)
         elseif(1-yj.lt.tiny)then
            zPY6Q=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yj)*(1-x)**2
     $           *shat_n1*(shat_n1*x-xm12)/( 2*(shat_n1-xm12)**2 )
         else
            zPY6Q=1-shat_n1*(1-x)/(shat_n1+w2-xm12)
         endif
c
      else
         write(*,*)'zPY6Q: unknown ileg'
         stop
      endif

      if(zPY6Q.lt.0d0.or.zPY6Q.gt.1d0)goto 999

      return
 999  continue
      zPY6Q=-1d0
      return
      end



      double precision function xiPY6Q()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c     Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
c$$$  integer ileg
c$$$      double precision xiPY6Q,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,z,
c$$$     &zPY6Q,w1,w2,betad,betas
      parameter(tiny=1d-5)

c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         xiPY6Q=shat_n1*(1-x)*(1-yi)/2
c
      elseif(ileg.eq.2)then
         xiPY6Q=shat_n1*(1-x)*(1-yi)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            xiPY6Q=shat_n1*(1-x)*(betas-betad*yj)/2
         else
            xiPY6Q=w1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiPY6Q=(1-yj)*(1-x)*(shat_n1-xm12)/2
         elseif(1-yj.lt.tiny)then
            xiPY6Q=(1-yj)*(1-x)*(shat_n1*x-xm12)/2
         else
            xiPY6Q=w2
         endif
c
      else
        write(*,*)'xiPY6Q: unknown ileg'
        stop
      endif

      if(xiPY6Q.lt.0d0)goto 999

      return
 999  continue
      xiPY6Q=-1d0
      return
      end



      double precision function xjacPY6Q()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q
c$$$     $     ,xq2q)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c     variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,zPY6Q,z,tmp,dw1dx,dw1dy,dw2dx,dw2dy
c$$$  integer ileg
c$$$      double precision xjacPY6Q,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,
c$$$     &xq2q,tiny,tmp,z,zPY6Q,w1,w2,dw1dx,dw1dy,dw2dx,dw2dy,betad,betas
      parameter (tiny=1d-5)

      z=zPY6Q()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         tmp=-shat_n1*(1-x)/2
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1*(1-x)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            tmp=xm12*betad/betas/(betas-betad*yj)
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=shat_n1*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)*dw1dy
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=shat_n1*(1-x)/2
         elseif(1-yj.lt.tiny)then
            tmp=-shat_n1*(1-x)*(shat_n1*x-xm12)/( 2*(shat_n1-xm12) )
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy) 
            tmp=shat_n1/(shat_n1+w2-xm12)*dw2dy
         endif
c
      else
         write(*,*)'xjacPY6Q: unknown ileg'
         stop
      endif
      xjacPY6Q=abs(tmp)

      return
 999  continue
      xjacPY6Q=0d0
      return
      end



c Pythia6PT

      double precision function zPY6PT()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower energy variable
      use kinematics_module
      implicit none
c$$$      double precision tiny
c$$$      integer ileg
c$$$      double precision zPY6PT,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny
c$$$      parameter(tiny=1d-5)

      if(ileg.eq.1)then
         zPY6PT=x
c
      elseif(ileg.eq.2)then
         zPY6PT=x
c
      elseif(ileg.eq.3)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      elseif(ileg.eq.4)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      else
         write(*,*)'zPY6PT: unknown ileg'
         stop
      endif

      if(zPY6PT.lt.0d0.or.zPY6PT.gt.1d0)goto 999

      return
 999  continue
      zPY6PT=-1d0
      return
      end



      double precision function xiPY6PT()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
c$$$      integer ileg
c$$$      double precision xiPY6PT,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,z
c$$$      parameter(tiny=1d-5)

      if(ileg.eq.1)then
         xiPY6PT=shat_n1*(1-x)**2*(1-yi)/2
c
      elseif(ileg.eq.2)then
         xiPY6PT=shat_n1*(1-x)**2*(1-yi)/2
c
      elseif(ileg.eq.3)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      elseif(ileg.eq.4)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      else
         write(*,*)'xiPY6PT: unknown ileg'
         stop
      endif

      if(xiPY6PT.lt.0d0)goto 999

      return
 999  continue
      xiPY6PT=-1d0
      return
      end



      double precision function xjacPY6PT()
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c     variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tmp
      if(ileg.eq.1)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.3)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      elseif(ileg.eq.4)then
         write(*,*)'PYTHIA6PT not available for FSR'
         stop
c
      else
         write(*,*)'xjacPY6PT: unknown ileg'
         stop
      endif
      xjacPY6PT=abs(tmp)

      return
 999  continue
      xjacPY6PT=0d0
      return
      end



c Pythia8

      double precision function zPY8()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
c$$$  integer ileg
c$$$      double precision zPY8,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,
c$$$     &w1,w2,betad,betas
      parameter(tiny=1d-5)

c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         zPY8=x
c
      elseif(ileg.eq.2)then
         zPY8=x
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            zPY8=1-(2*xm12)/(shat_n1*betas*(betas-betad*yj))
         else
            zPY8=1-shat_n1*(1-x)*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)
c This is equation (3.10) of hep-ph/1102.3795. In the partonic
c CM frame it is equal to (xk1(0)+xk3(0)*f)/(xk1(0)+xk3(0)),
c where f = xm12/( s+xm12-xm22-2*sqrt(s)*(xk1(0)+xk3(0)) )
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zPY8=1-shat_n1*(1-x)/(shat_n1-xm12)
         elseif(1-yj.lt.tiny)then
            zPY8=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yj)*(1-x)**2*shat_n1
     $           *(shat_n1*x-xm12)/( 2*(shat_n1-xm12)**2 )
         else
            zPY8=1-shat_n1*(1-x)/(shat_n1+w2-xm12)
         endif
c
      else
         write(*,*)'zPY8: unknown ileg'
         stop
      endif

      if(zPY8.lt.0d0.or.zPY8.gt.1d0)goto 999

      return
 999  continue
      zPY8=-1d0
      return
      end



      double precision function xiPY8()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,zPY8,z0
c$$$      integer ileg
c$$$      double precision xiPY8,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q,tiny,z,
c$$$     &zPY8,w1,w2,betas,betad,z0
      parameter(tiny=1d-5)

      z=zPY8()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         xiPY8=shat_n1*(1-x)**2*(1-yi)/2
c
      elseif(ileg.eq.2)then
         xiPY8=shat_n1*(1-x)**2*(1-yi)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            z0=1-(2*xm12)/(shat_n1*betas*(betas-betad*yj))
            xiPY8=shat_n1*(1-x)*(betas-betad*yj)*z0*(1-z0)/2
         else
            xiPY8=z*(1-z)*w1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiPY8=shat_n1*(1-x)**2*(1-yj)/2
         elseif(1-yj.lt.tiny)then
            xiPY8=shat_n1*(1-x)**2*(1-yj)*(shat_n1*x-xm12)**2/(2
     $           *(shat_n1-xm12)**2)
         else
            xiPY8=z*(1-z)*w2
         endif
c
      else
        write(*,*)'xiPY8: unknown ileg'
        stop
      endif

      if(xiPY8.lt.0d0)goto 999

      return
 999  continue
      xiPY8=-1d0
      return
      end



      double precision function xjacPY8()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,z0,zPY8,dw1dx,dw1dy,dw2dx,dw2dy,tmp
c$$$      integer ileg
c$$$      double precision xjacPY8,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,
c$$$     &xq2q,tiny,tmp,z,zPY8,w1,w2,dw1dx,dw1dy,dw2dx,dw2dy,betad,betas,z0
c$$$      parameter (tiny=1d-5)

      z=zPY8()!(ileg,xm12,xm22,s,x,yi,yj,xtk,xuk,xq1q,xq2q)
      if(z.lt.0d0)goto 999
c$$$      w1=-xq1q+xq2q-xtk
c$$$      w2=-xq2q+xq1q-xuk
c
      if(ileg.eq.1)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
c$$$            betad=sqrt((1-(xm12-xm22)/s)**2-(4*xm22/s))
c$$$            betas=1+(xm12-xm22)/s
            z0=1-(2*xm12)/(shat_n1*betas*(betas-betad*yj))
            tmp=xm12*betad/betas/(betas-betad*yj)*z0*(1-z0)
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=shat_n1*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)*dw1dy*z*(1-z)
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=shat_n1**2*(1-x)**2/( 2*(shat_n1-xm12) )
         elseif(1-yj.le.tiny)then
            tmp=4*shat_n1**2*(1-x)**2*(shat_n1*x-xm12)**2/( 2*(shat_n1
     $           -xm12) )**3
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=shat_n1/(shat_n1+w2-xm12)*dw2dy*z*(1-z)
         endif
c
      else
         write(*,*)'xjacPY8: unknown ileg'
         stop
      endif
      xjacPY8=abs(tmp)

      return
 999  continue
      xjacPY8=0d0
      return
      end

c End of Monte Carlo functions



      function get_zeta(xs,xw1,xw2,xxm12,xxm22)
      implicit none
      double precision get_zeta,xs,xw1,xw2,xxm12,xxm22
      double precision eps,beta
c
      eps=1-(xxm12-xxm22)/(xs-xw1)
      beta=sqrt(eps**2-4*xs*xxm22/(xs-xw1)**2)
      get_zeta=( (2*xs-(xs-xw1)*eps)*xw2+(xs-xw1)*((xw1+xw2)*beta-eps*xw1) )/
     &         ( (xs-xw1)*beta*(2*xs-(xs-xw1)*eps+(xs-xw1)*beta) )
c
      return
      end



      function emscafun(x,alpha)
      implicit none
      double precision emscafun,x,alpha
      if(x.le.0d0) then
         emscafun=0d0
      elseif(x.ge.1d0) then
         emscafun=1d0
      else
         emscafun=x**(2*alpha)/(x**(2*alpha)+(1-x)**(2*alpha))
      endif
      return
      end



      function emscainv(r,alpha)
c Inverse of emscafun, implemented only for alpha=1 for the moment
      implicit none
      double precision emscainv,r,alpha
c
      if(r.lt.0d0.or.r.gt.1d0.or.alpha.ne.1d0)then
         write(*,*)'Fatal error in emscafun'
         stop
      endif
      emscainv=sqrt(r)/(sqrt(r)+sqrt(1d0-r))
      return
      end



      function bogus_probne_fun(qMC)
      implicit none
      double precision bogus_probne_fun,qMC
      double precision x,tmp,emscafun
      integer itype
      data itype/2/
c
      if(itype.eq.1)then
c Theta function
         tmp=1d0
         if(qMC.le.2d0)tmp=0d0
      elseif(itype.eq.2)then
c Smooth function
         x=(1d1-qMC)/(1d1-0.5d0)
         tmp=1-emscafun(x,2d0)
      elseif(itype.eq.3) then
c No (bogus) sudakov factor
         tmp=1d0
      else
        write(*,*)'Error in bogus_probne_fun: unknown option',itype
        stop
      endif
      bogus_probne_fun=tmp
      return
      end



      function get_angle(p1,p2)
      implicit none
      double precision get_angle,p1(0:3),p2(0:3)
      double precision tiny,mod1,mod2,cosine
      parameter (tiny=1d-5)
c
      mod1=sqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      mod2=sqrt(p2(1)**2+p2(2)**2+p2(3)**2)

      if(mod1.eq.0d0.or.mod2.eq.0d0)then
         write(*,*)'Undefined angle in get_angle',mod1,mod2
         stop
      endif
c
      cosine=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)
      cosine=cosine/(mod1*mod2)
c
      if(abs(cosine).gt.1d0+tiny)then
         write(*,*)'cosine larger than 1 in get_angle',cosine,p1,p2
         stop
      elseif(abs(cosine).ge.1d0)then
         cosine=sign(1d0,cosine)
      endif
c
      get_angle=acos(cosine)

      return
      end



c Shower scale

c$$$      subroutine assign_emsca(pp,xi_i_fks,y_ij_fks)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$c$$$      include "madfks_mcatnlo.inc"
c$$$      include "run.inc"
c$$$
c$$$      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks
c$$$      double precision shattmp,dot,emsca_bare,ref_scale,scalemin,
c$$$     &scalemax,rrnd,ran2,emscainv,dum(5),xm12,qMC,ptresc
c$$$      integer ileg
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$
c$$$      logical emscasharp
c$$$      double precision emsca
c$$$      common/cemsca/emsca,emsca_bare,emscasharp,scalemin,scalemax
c$$$
c$$$      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
c$$$      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
c$$$
c$$$c Consistency check
c$$$      shattmp=2d0*dot(pp(0,1),pp(0,2))
c$$$      if(abs(shattmp/shat-1d0).gt.1d-5)then
c$$$         write(*,*)'Error in assign_emsca: inconsistent shat'
c$$$         write(*,*)shattmp,shat
c$$$         stop
c$$$      endif
c$$$
c$$$      call kinematics_driver(xi_i_fks,y_ij_fks,shat,pp,ileg,xm12,dum(1)
c$$$     $     ,dum(2),dum(3),dum(4),dum(5),qMC)
c$$$
c$$$      emsca=2d0*sqrt(ebeam(1)*ebeam(2))
c$$$      call assign_scaleminmax(shat,xi_i_fks,scalemin,scalemax,ileg
c$$$     $        ,xm12)
c$$$      emscasharp=(scalemax-scalemin).lt.(1d-3*scalemax)
c$$$      if(emscasharp)then
c$$$         emsca_bare=scalemax
c$$$         emsca=emsca_bare
c$$$      else
c$$$         rrnd=ran2()
c$$$         rrnd=emscainv(rrnd,1d0)
c$$$         emsca_bare=scalemin+rrnd*(scalemax-scalemin)
c$$$         ptresc=(qMC-scalemin)/(scalemax-scalemin)
c$$$         if(ptresc.lt.1d0)emsca=emsca_bare
c$$$         if(ptresc.ge.1d0)emsca=scalemax
c$$$      endif
c$$$
c$$$      return
c$$$      end


c$$$      subroutine assign_emsca_array(pp,xi_i_fks,y_ij_fks)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$c$$$      include "madfks_mcatnlo.inc"
c$$$      include "run.inc"
c$$$      include "born_nhel.inc"
c$$$      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,shattmp,dot
c$$$      double precision rrnd,ran2,emscainv, dum(5),xm12,qMC
c$$$     $     ,ptresc_a(nexternal,nexternal),ref_scale_a(nexternal
c$$$     $     ,nexternal)
c$$$      integer ileg,npartner,i,j
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
c$$$      common /MC_info/ ipartners,colorflow
c$$$
c$$$      logical emscasharp_a(nexternal,nexternal)
c$$$      double precision emsca_a(nexternal,nexternal)
c$$$     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
c$$$     $     ,nexternal) ,scalemin_a(nexternal,nexternal)
c$$$     $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
c$$$     $     ,nexternal)
c$$$      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2, emscasharp_a
c$$$     $     ,scalemin_a,scalemax_a,emscwgt_a
c$$$
c$$$      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
c$$$      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
c$$$
c$$$      double precision Eem,qMC_a2(nexternal-1,nexternal-1),emscafun
c$$$      integer iBtoR(nexternal-1)
c$$$      integer i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      integer fks_j_from_i(nexternal,0:nexternal)
c$$$     &     ,particle_type(nexternal),pdg_type(nexternal)
c$$$      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
c$$$
c$$$c Consistency check
c$$$      shattmp=2d0*dot(pp(0,1),pp(0,2))
c$$$      if(abs(shattmp/shat-1d0).gt.1d-5)then
c$$$         write(*,*)'Error in assign_emsca_array: inconsistent shat'
c$$$         write(*,*)shattmp,shat
c$$$         stop
c$$$      endif
c$$$
c$$$      call kinematics_driver(xi_i_fks,y_ij_fks,shat,pp,ileg,xm12,dum(1)
c$$$     $     ,dum(2),dum(3),dum(4),dum(5),qMC)
c$$$      call assign_scaleminmax_array(shat,xi_i_fks,scalemin_a,scalemax_a
c$$$     $     ,ileg,xm12)
c$$$      emsca_a=-1d0
c$$$      emscwgt_a=0d0
c$$$c
c$$$      do i=1,nexternal
c$$$        if(i.lt.i_fks)then
c$$$          iBtoR(i)=i
c$$$        elseif(i.eq.i_fks)then
c$$$          if(i.lt.nexternal)iBtoR(i)=i+1
c$$$        elseif(i.gt.i_fks)then
c$$$          if(i.lt.nexternal)iBtoR(i)=i+1
c$$$        endif
c$$$      enddo
c$$$c      
c$$$      call assign_qMC_array(xi_i_fks,y_ij_fks,shat,pp,qMC,qMC_a2)
c$$$      do i=1,nexternal-1
c$$$c     skip if not QCD dipole (safety)
c$$$         if(.not.(pdg_type(iBtoR(i)).eq.21 .or.
c$$$     $            abs(pdg_type(iBtoR(i))).le.6))cycle
c$$$         do j=1,nexternal-1
c$$$            if(j.eq.i)cycle
c$$$c     skip if not QCD dipole (safety)
c$$$            if(.not.(pdg_type(iBtoR(j)).eq.21 .or.
c$$$     $               abs(pdg_type(iBtoR(j))).le.6))cycle
c$$$            emscasharp_a(i,j)=(scalemax_a(i,j)-scalemin_a(i,j)).lt.
c$$$     #                           (1d-3*scalemax_a(i,j))
c$$$            if(emscasharp_a(i,j))then
c$$$               if(qMC_a2(i,j).le.scalemax_a(i,j))emscwgt_a(i,j)=1d0
c$$$               emsca_bare_a(i,j)=scalemax_a(i,j)
c$$$               emsca_bare_a2(i,j)=scalemax_a(i,j)
c$$$               emsca_a(i,j)=emsca_bare_a(i,j)
c$$$            else
c$$$               rrnd=ran2()
c$$$               rrnd=emscainv(rrnd,1d0)
c$$$               emsca_bare_a(i,j)=scalemin_a(i,j)+
c$$$     #                               rrnd*(scalemax_a(i,j)-scalemin_a(i,j))
c$$$               rrnd=ran2()
c$$$               rrnd=emscainv(rrnd,1d0)
c$$$               emsca_bare_a2(i,j)=scalemin_a(i,j)+
c$$$     #                               rrnd*(scalemax_a(i,j)-scalemin_a(i,j))
c$$$               ptresc_a(i,j)=(qMC_a2(i,j)-scalemin_a(i,j))/
c$$$     #                          (scalemax_a(i,j)-scalemin_a(i,j))
c$$$               if(ptresc_a(i,j).le.0d0)then
c$$$                  emscwgt_a(i,j)=1d0
c$$$                  emsca_a(i,j)=emsca_bare_a(i,j)
c$$$               elseif(ptresc_a(i,j).lt.1d0)then
c$$$                  emscwgt_a(i,j)=1-emscafun(ptresc_a(i,j),1d0)
c$$$                  emsca_a(i,j)=emsca_bare_a(i,j)
c$$$               else
c$$$                  emscwgt_a(i,j)=0d0
c$$$                  emsca_a(i,j)=scalemax_a(i,j)
c$$$               endif
c$$$            endif
c$$$         enddo
c$$$      enddo
c$$$
c$$$      return
c$$$      end



c$$$      subroutine assign_scaleminmax(shat,xi,xscalemin,xscalemax,ileg
c$$$     $     ,xm12)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      include "run.inc"
c$$$c$$$      include "madfks_mcatnlo.inc"
c$$$      integer i,ileg
c$$$      double precision shat,xi,ref_scale,xscalemax,xscalemin,xm12
c$$$      character*4 abrv
c$$$      common/to_abrv/abrv
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$
c$$$      call assign_ref_scale(p_born,xi,shat,ref_scale)
c$$$      xscalemin=max(shower_scale_factor*frac_low*ref_scale,scaleMClow)
c$$$      xscalemax=max(shower_scale_factor*frac_upp*ref_scale,
c$$$     &              xscalemin+scaleMCdelta)
c$$$      xscalemax=min(xscalemax,2d0*sqrt(ebeam(1)*ebeam(2)))
c$$$      xscalemin=min(xscalemin,xscalemax)
c$$$c
c$$$      if(abrv.ne.'born'.and.shower_mc(1:7).eq.'PYTHIA6' .and.
c$$$     $     ileg.eq.3)then
c$$$         xscalemin=max(xscalemin,sqrt(xm12))
c$$$         xscalemax=max(xscalemin,xscalemax)
c$$$      endif
c$$$
c$$$      return
c$$$      end
c$$$
c$$$
c$$$      subroutine assign_scaleminmax_array(shat,xi,xscalemin_a
c$$$     $     ,xscalemax_a,ileg,xm12)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      include "run.inc"
c$$$c$$$      include "madfks_mcatnlo.inc"
c$$$      integer i,j,ileg
c$$$      double precision shat,xi,ref_scale_a(nexternal,nexternal),xm12
c$$$      double precision xscalemax_a(nexternal,nexternal)
c$$$     $     ,xscalemin_a(nexternal,nexternal)
c$$$      character*4 abrv
c$$$      common/to_abrv/abrv
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$
c$$$      xscalemax_a=-1d0
c$$$      xscalemin_a=-1d0
c$$$      call assign_ref_scale_array(p_born,ref_scale_a)
c$$$      do i=1,nexternal-2
c$$$         do j=i+1,nexternal-1
c$$$            xscalemin_a(i,j)=max(shower_scale_factor*frac_low
c$$$     $           *ref_scale_a(i,j),scaleMClow)
c$$$            xscalemax_a(i,j)=max(shower_scale_factor*frac_upp
c$$$     $           *ref_scale_a(i,j),xscalemin_a(i,j)+scaleMCdelta)
c$$$            xscalemax_a(i,j)=min(xscalemax_a(i,j),2d0
c$$$     $           *sqrt(ebeam(1)*ebeam(2)))
c$$$            xscalemin_a(i,j)=min(xscalemin_a(i,j),xscalemax_a(i,j))
c$$$c
c$$$            if(abrv.ne.'born'.and.shower_mc(1:7).eq.'PYTHIA6' .and.
c$$$     $           ileg.eq.3)then
c$$$               xscalemin_a(i,j)=max(xscalemin_a(i,j),sqrt(xm12))
c$$$               xscalemax_a(i,j)=max(xscalemin_a(i,j),xscalemax_a(i,j))
c$$$            endif
c$$$c
c$$$            xscalemin_a(j,i)=xscalemin_a(i,j)
c$$$            xscalemax_a(j,i)=xscalemax_a(i,j)
c$$$         enddo
c$$$      enddo
c$$$c
c$$$      return
c$$$      end
c$$$
c$$$      block data reference_scale
c$$$!     common block used to make the (scalar) reference scale partner
c$$$!     dependent in case of delta. [Set it to -1 by default: in case of
c$$$!     the non-delta running, it never gets updated so that it remains
c$$$!     equal to -1, and the normal code will be used].
c$$$      integer cur_part
c$$$      common /to_ref_scale/cur_part
c$$$      data cur_part/-1/
c$$$      end

      
c$$$      subroutine assign_ref_scale(p,xii,sh,ref_sc)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$c$$$      include "madfks_mcatnlo.inc"
c$$$      double precision p(0:3,nexternal-1),xii,sh,ref_sc
c$$$      integer i_scale,i,fks_father
c$$$      parameter(i_scale=1)
c$$$      double precision ref_sc_a(nexternal,nexternal)
c$$$      double precision sumdot
c$$$      external sumdot
c$$$!     common block used to make the (scalar) reference scale partner
c$$$!     dependent in case of delta
c$$$      integer cur_part
c$$$      common /to_ref_scale/cur_part
c$$$      integer            i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      ref_sc=0d0
c$$$      if (cur_part.eq.-1) then ! this is non-delta (or no MC subtr. needed)
c$$$         if(i_scale.eq.0)then
c$$$c Born-level CM energy squared
c$$$            ref_sc=dsqrt(max(0d0,(1-xii)*sh))
c$$$         elseif(i_scale.eq.1)then
c$$$c Sum of final-state transverse masses
c$$$            do i=3,nexternal-1
c$$$               ref_sc=ref_sc+dsqrt(max(0d0,(p(0,i)+p(3,i))*(p(0,i)-p(3,i))))
c$$$            enddo
c$$$            ref_sc=ref_sc/2d0
c$$$         else
c$$$            write(*,*)'Wrong i_scale in assign_ref_scale',i_scale
c$$$            stop
c$$$         endif
c$$$c Safety threshold for the reference scale
c$$$         ref_sc=max(ref_sc,scaleMClow+scaleMCdelta)
c$$$      elseif (cur_part.eq.0) then
c$$$         call get_global_ref_sc(p,ref_sc)
c$$$      else
c$$$! in the case of mc@nlo-delta, make the scalar reference scale equal to
c$$$! the corresponding element of the ref scale array, i.e., the fks-father
c$$$! and the partner. (The cur_part is set by the loop over the colour
c$$$! partners in the compute_xmcsubt_complete subroutine).
c$$$         call assign_ref_scale_array(p,ref_sc_a)
c$$$         fks_father=min(i_fks,j_fks)
c$$$         ref_sc=ref_sc_a(fks_father,cur_part)
c$$$      endif
c$$$      return
c$$$      end
c$$$
c$$$      
c$$$      subroutine assign_ref_scale_array(p,ref_sc_a)
c$$$c--------------------------------------------------------------------------
c$$$c     The setting of the reference scales, formerly equal to the dipole
c$$$c     masses, is achieved by taking the minimum between the relevant
c$$$c     dipole mass and a global quantity X, defined as follows:
c$$$c     1. processes without coloured final-state particles: X=sqrt{shat};
c$$$c     2. processes with massless coloured final-state particles: X=first
c$$$c     kt-clustering scale as returned by fastjet;
c$$$c     3. processes with massive coloured final-state particles:
c$$$c     X=minimum of the transverse energies of such particles;
c$$$c     4. processes with both massive and massless coloured final-state
c$$$c     particles: X=minimum of the transverse energies of the massive
c$$$c     particles, and of first kt-clustering scale as returned by fastjet
c$$$c     run over the massless particles.
c$$$c
c$$$c     Possible variants:
c$$$c     - run fastjet over both massless and massive particles, and use
c$$$c     the first kt-clustering scale as returned by fastjet as X in case
c$$$c     4. To be done: check how fastjet deals with masses;
c$$$c     - go more local: on top of defining X as above, for each particle
c$$$c     redefine it as the minimum of itself and of the particle
c$$$c     transverse energy. Drawbacks: unclear why this should be relevant
c$$$c     to final-final dipoles, and possible IR sensitivity in the case
c$$$c     of massless collinear particles.
c$$$c--------------------------------------------------------------------------
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      double precision p(0:3,nexternal-1)
c$$$      double precision ref_sc_a(nexternal,nexternal)
c$$$      double precision ref_sc
c$$$      integer i,j
c$$$      double precision sumdot
c$$$      external sumdot
c$$$c
c$$$      call get_global_ref_sc(p,ref_sc)
c$$$c
c$$$      do i=1,nexternal-2
c$$$         do j=i+1,nexternal-1
c$$$            ref_sc_a(i,j)=min(sqrt(max(0d0,sumdot(p(0,i),p(0,j),1d0)))
c$$$     $           ,ref_sc)
c$$$            ref_sc_a(j,i)=ref_sc_a(i,j)
c$$$         enddo
c$$$      enddo
c$$$c
c$$$      return
c$$$      end
c$$$
c$$$      subroutine get_global_ref_sc(p,ref_sc)
c$$$      implicit none
c$$$      include 'nexternal.inc'
c$$$      double precision p(0:3,nexternal-1),ref_sc,pQCD(0:3,nexternal-1)
c$$$     $     ,palg,sycut,rfj,pjet(0:3,nexternal-1)
c$$$      integer i,j,NN,Nmass,njet,jet(nexternal-1)
c$$$      double precision sumdot,pt,get_mass_from_id
c$$$     $     ,amcatnlo_fastjetdmergemax
c$$$      integer get_color
c$$$      external sumdot,pt,get_color,get_mass_from_id
c$$$     $     ,amcatnlo_fastjetdmergemax
c$$$      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
c$$$      LOGICAL  IS_A_PH(NEXTERNAL)
c$$$      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
c$$$      integer fks_j_from_i(nexternal,0:nexternal)
c$$$     &     ,particle_type(nexternal),pdg_type(nexternal)
c$$$      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
c$$$ ! start from s-hat      
c$$$      ref_sc=sqrt(sumdot(p(0,1),p(0,2),1d0))
c$$$      NN=0
c$$$      Nmass=0
c$$$      do j=nincoming+1,nexternal-1
c$$$         if (is_a_j(j))then
c$$$            NN=NN+1
c$$$            do i=0,3
c$$$               pQCD(i,NN)=p(i,j)
c$$$            enddo
c$$$         elseif (abs(get_color(pdg_type(j))).ne.1 .and.
c$$$     $           abs(get_mass_from_id(pdg_type(j))).ne.0d0) then
c$$$!     reduce by ET of massive QCD particles
c$$$            
c$$$            ref_sc=min(ref_sc,sqrt((p(0,j)+p(3,j))*(p(0,j)-p(3,j))))
c$$$         elseif (abs(get_color(pdg_type(j))).ne.1 .and.
c$$$     $           abs(get_mass_from_id(pdg_type(j))).eq.0d0) then
c$$$            write (*,*) 'Error in assign_ref_scale(): colored'/
c$$$     $           /' massless particle that does not enter jets'
c$$$            stop 1
c$$$         endif
c$$$      enddo
c$$$! reduce by kT-cluster scale of massless QCD partons
c$$$      if (NN.eq.1) then
c$$$         ref_sc=min(ref_sc,pt(pQCD(0,1)))
c$$$      elseif (NN.ge.2) then
c$$$         palg=1d0
c$$$         sycut=0d0
c$$$         rfj=1d0
c$$$         call amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
c$$$     &        pjet,njet,jet)
c$$$         ref_sc=min(ref_sc,sqrt(amcatnlo_fastjetdmergemax(NN-1)))
c$$$      endif
c$$$      end

c$$$      subroutine dinvariants_dFKS(ileg,s,x,yi,yj,xm12,xm22,dw1dx,dw1dy,dw2dx,dw2dy)
      subroutine dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
c Returns derivatives of Mandelstam invariants with respect to FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision s,dw1dx,dw2dx,dw1dy,dw2dy
      double precision afun,bfun,cfun,mom_fks_sister_p,mom_fks_sister_m,
     &diff_p,diff_m,signfac,dadx,dady,dbdx,dbdy,dcdx,dcdy,mom_fks_sister,
     &dmomfkssisdx,dmomfkssisdy,en_fks,en_fks_sister,dq1cdx,dq2qdx,dq1cdy,
     &dq2qdy
      double precision veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double precision tiny
      parameter(tiny=1d-5)

      s=shat_n1
      if(ileg.eq.1)then
         write(*,*)'dinvariants_dFKS should not be called for ileg = 1'
         stop
c
      elseif(ileg.eq.2)then
         write(*,*)'dinvariants_dFKS should not be called for ileg = 2'
         stop
c
      elseif(ileg.eq.3)then
c For ileg = 3, the mother 3-momentum is [afun +- sqrt(bfun) ] / cfun
         afun=sqrt(s)*(1-x)*(xm12-xm22+s*x)*yj
         bfun=s*( (1+x)**2*(xm12**2+(xm22-s*x)**2-
     &        xm12*(2*xm22+s*(1+x**2)))+xm12*s*(1-x**2)**2*yj**2 )
         cfun=s*(-(1+x)**2+(1-x)**2*yj**2)
         dadx=sqrt(s)*yj*(xm22-xm12+s*(1-2*x))
         dady=sqrt(s)*(1-x)*(xm12-xm22+s*x)
         dbdx=2*s*(1+x)*( xm12**2+(xm22-s*x)*(xm22-s*(1+2*x))
     &        -xm12*(2*xm22+s*(1+x+2*(x**2)+2*(1-x)*x*(yj**2))) )
         dbdy=2*xm12*(s**2)*((1-x**2)**2)*yj
         dcdx=-2*s*(1+x+(yj**2)*(1-x))
         dcdy=2*s*((1-x)**2)*yj
c Determine correct sign
         mom_fks_sister_p=(afun+sqrt(bfun))/cfun
         mom_fks_sister_m=(afun-sqrt(bfun))/cfun
         diff_p=abs(mom_fks_sister_p-veckn_ev)
         diff_m=abs(mom_fks_sister_m-veckn_ev)
         if(min(diff_p,diff_m)/max(abs(veckn_ev),1d0).ge.1d-3)then
            write(*,*)'Fatal error 1 in dinvariants_dFKS'
            write(*,*)mom_fks_sister_p,mom_fks_sister_m,veckn_ev
            stop
         elseif(min(diff_p,diff_m)/max(abs(veckn_ev),1d0).ge.tiny)then
            write(*,*)'Numerical imprecision 1 in dinvariants_dFKS'
         endif
         signfac=1d0
         if(diff_p.ge.diff_m)signfac=-1d0
         mom_fks_sister=veckn_ev
         en_fks=sqrt(s)*(1-x)/2
         en_fks_sister=sqrt(mom_fks_sister**2+xm12)
         dmomfkssisdx=(dadx+signfac*dbdx/(2*sqrt(bfun))-dcdx*mom_fks_sister)/cfun
         dmomfkssisdy=(dady+signfac*dbdy/(2*sqrt(bfun))-dcdy*mom_fks_sister)/cfun
         dw1dx=sqrt(s)*( yj*mom_fks_sister-en_fks_sister+(1-x)*
     &                   (mom_fks_sister/en_fks_sister-yj)*dmomfkssisdx )
         dw1dy=-sqrt(s)*(1-x)*( mom_fks_sister+
     &                   (yj-mom_fks_sister/en_fks_sister)*dmomfkssisdy )
         dw2dx=-dw1dx-s
         dw2dy=-dw1dy
c
      elseif(ileg.eq.4)then
         dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
         dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
         dw2dx=(1-yj)*(s*(1+yj-x*(2*(1+yj)+x*(1-yj)))+2*xm12)/(1+yj+x*(1-yj))**2
         dw1dx=dq1cdx+dq2qdx
         dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
         dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
         dw2dy=-2*(1-x)*(s*x-xm12)/(1+yj+x*(1-yj))**2
         dw1dy=dq1cdy+dq2qdy
c
      else
         write(*,*)'Error in dinvariants_dFKS: unknown ileg',ileg
         stop
      endif

      return
      end



      subroutine get_dead_zone(z,xi,qMC,ipartner,ifather,lzone
     $     ,PY6PTweight)
      ! TODO: check that we can use fksfather instead of ifather
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      integer ipartner,ifather,i
      double precision z,xi,qMC,PY6PTweight
      logical lzone

      double precision upscale2,xmp2,xmm2,xmr2,ww,Q2,lambda,sumdot,dot
     $     ,e0sq,beta,ycc,mdip,mdip_g,zp1,zm1,zp2,zm2,zp3,zm3,get_angle
     $     ,theta2p
      external sumdot,dot,get_angle

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision pip(0:3),pifather(0:3)

      ! PYTHIA6 variables
      integer mstj50,mstp67
      double precision parp67
      parameter (mstj50=2,mstp67=2,parp67=1d0)

c Skip if unphysical shower variables
      if(z.lt.0d0.or.xi.lt.0d0) then
         lzone=.false.
         return
      endif

c Definition and initialisation of variables
      lzone=.true.
      PY6PTweight=-1d0
      do i=0,3
         pifather(i)=p_born(i,ifather) ! father momentum (Born level)
         pip(i)  =p_born(i,ipartner) ! partner momentum (Born level)
      enddo
      if (shower_mc_mod(1:6).eq.'HERWIG') e0sq=dot(pip,pifather)
      theta2p=get_angle(pip,pifather)**2
      if(ileg.eq.3 .or. ileg.eq.4) then
         if (ileg.eq.3) then
            xmm2=xm12           ! emitter mass squared
            ww=w1               ! FKS parent/sister dot product
            xmr2=xm22           ! global-recoiler mass squared
         elseif (ileg.eq.4) then
            xmm2=0d0
            ww=w2               ! FKS parent/sister dot product
            xmr2=xm12           ! global-recoiler mass squared
         endif
         Q2=sumdot(pifather,pip,1d0) ! parent dipole mass squared (Born level)
         xmp2=dot(pip,pip)      ! mass squared of the partner
         if (shower_mc_mod(1:8).eq.'HERWIGPP')
     &        lambda=sqrt((Q2+xmm2-xmp2)**2-4*Q2*xmm2)
         if (shower_mc_mod(1:8).eq.'PYTHIA6Q') then
            beta=sqrt(1-4*s*(xmm2+ww)/(s-xmr2+xmm2+ww)**2)
            zp1=(1+(xmm2+beta*ww)/(xmm2+ww))/2
            zm1=(1+(xmm2-beta*ww)/(xmm2+ww))/2
         endif
         if (shower_mc_mod(1:7).eq.'PYTHIA8') then
            beta=sqrt(1-4*s*(xmm2+ww)/(s-xmr2+xmm2+ww)**2)
            mdip  =sqrt((sqrt(xmp2+xmm2+2*e0sq)-sqrt(xmp2))**2-xmm2)
            ! mdip corresponds to sqrt(dip.m2DipCorr)
            ! (around line 2305 in Pythia TimeShower.cc)
            mdip_g=sqrt((sqrt(s) -sqrt(xmr2))**2-xmm2)
            ! Global-recoil adaption of the above
            zp2=(1+beta)/2      ! These are the solutions of equation q2 s == z(1-z)(s+q2-xmr2)^2
            zm2=(1-beta)/2      ! where q2 = (p_i_FKS + p_j_FKS)^2
            ! Note that this is the global-recoil analogue of eq. (24) in 0408302
            zp3=(1+sqrt(1-4*xi/mdip_g**2))/2 ! These are the analogous of eq. (23) in 0408302
            zm3=(1-sqrt(1-4*xi/mdip_g**2))/2 ! for the global recoil
         endif
      endif
      
c Dead zones
c IMPLEMENT QED DZ's!
      if(shower_mc_mod(1:7).eq.'HERWIG6')then
         lzone=.false.
         if(ileg.le.2.and.z**2.ge.xi)lzone=.true.
         if(ileg.gt.2.and.e0sq*xi*z**2.ge.xmm2
     &               .and.xi.le.1d0)lzone=.true.
         if(e0sq.eq.0d0)lzone=.false.
c
      elseif(shower_mc_mod(1:8).eq.'HERWIGPP')then
         lzone=.false.
         if(ileg.le.2)upscale2=2*e0sq
         if(ileg.gt.2)then
            upscale2=2*e0sq+xmm2
            if(ipartner.gt.2)upscale2=(Q2+xmm2-xmp2+lambda)/2
         endif
         if(xi.lt.upscale2)lzone=.true.
c
      elseif(shower_mc_mod(1:8).eq.'PYTHIA6Q')then
         if(ileg.le.2)then
            if(mstp67.eq.2.and.ipartner.gt.2.and.
     &         4*xi/s/(1-z).ge.theta2p)lzone=.false.
         elseif(ileg.gt.2)then
            if(mstj50.eq.2.and.ipartner.le.2.and.
c around line 71636 of pythia6428: V(IEP(1),5)=virtuality, P(IM,4)=sqrt(s)
     &           max(z/(1-z),(1-z)/z)*4*(xi+xmm2)/s.ge.theta2p)
     &           lzone=.false.
            if(z.gt.zp1.or.z.lt.zm1)lzone=.false.
         endif
c
      elseif(shower_mc_mod(1:9).eq.'PYTHIA6PT')then
         ycc=1-parp67*x/(1-x)**2/2
         if(mstp67.eq.1.and.yi.lt.ycc)lzone=.false.
         if(mstp67.eq.2) PY6PTweight=min(1d0,(1-ycc)/(1-yi))
c
      elseif(shower_mc_mod(1:7).eq.'PYTHIA8')then
         if(ileg.le.2.and.z.gt.1-sqrt(xi/z/s)*
     &      (sqrt(1+xi/4/z/s)-sqrt(xi/4/z/s)))lzone=.false.
         if(ileg.gt.2)then
            max_scale=min(min(scalemax,mdip/2),mdip_g/2) ! Pythia as well
   ! in the global recoil scheme, constrains radiation to be softer than local
   ! dipole mass divided by two 
            if(z.gt.min(zp2,zp3).or.z.lt.max(zm2,zm3))lzone=.false.
         endif

      endif

! If the relative pT of the splitting is larger then the maximum shower
! scale, we are in the deadzone
      if (qMC.gt.shower_scale_nbody_max(ipartner,ifather))
     &     lzone=.false.

      return
      end



      function charge(ipdg)
c computes the electric charge given the pdg code
      implicit none
      integer ipdg
      double precision charge,tmp,dipdg

      dipdg=dble(ipdg)
c quarks
      if(abs(dipdg).eq.1) tmp=-1d0/3d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.2) tmp= 2d0/3d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.3) tmp=-1d0/3d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.4) tmp= 2d0/3d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.5) tmp=-1d0/3d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.6) tmp= 2d0/3d0*sign(1d0,dipdg)
c leptons
      if(abs(dipdg).eq.11)tmp=-1d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.12)tmp= 0d0
      if(abs(dipdg).eq.13)tmp=-1d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.14)tmp= 0d0
      if(abs(dipdg).eq.15)tmp=-1d0*sign(1d0,dipdg)
      if(abs(dipdg).eq.16)tmp= 0d0
c bosons
      if(dipdg.eq.21)     tmp= 0d0
      if(dipdg.eq.22)     tmp= 0d0
      if(dipdg.eq.23)     tmp= 0d0
      if(abs(dipdg).eq.24)tmp= 1d0*sign(1d0,dipdg)
      if(dipdg.eq.25)     tmp= 0d0
c
      charge=tmp

      return
      end


c$$$      subroutine assign_qMC_array(xi_i_fks,y_ij_fks,sh,pp,qMC,qMC_a2)
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$      include "coupl.inc"
c$$$      include "run.inc"
c$$$      double precision pp(0:3,nexternal),pp_rec(0:3)
c$$$      double precision xi_i_fks,y_ij_fks,xij
c$$$
c$$$      integer ileg,j,i,nfinal,ipart
c$$$      double precision xp1(0:3),xp2(0:3),xk1(0:3),xk2(0:3),xk3(0:3)
c$$$c      common/cpkmomenta/xp1,xp2,xk1,xk2,xk3
c$$$      integer fks_j_from_i(nexternal,0:nexternal)
c$$$     &     ,particle_type(nexternal),pdg_type(nexternal)
c$$$      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
c$$$      double precision sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12,xm22
c$$$      double precision qMC,qMC_a(nexternal),qMC_a2(nexternal-1,nexternal-1)
c$$$      double precision zPY8,zeta1,zeta2,get_zeta,z,qMCarg,dot
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$      integer i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      double precision tiny
c$$$      parameter(tiny=1d-5)
c$$$      double precision zero
c$$$      parameter(zero=0d0)
c$$$      integer iRtoB(nexternal)
c$$$
c$$$      integer isqrtneg
c$$$      save isqrtneg
c$$$
c$$$      double precision pmass(nexternal)
c$$$      include "pmass.inc"
c$$$
c$$$c Stop if not PYTHIA8
c$$$      if(shower_mc.ne.'PYTHIA8')then
c$$$         write(*,*)'assign_qMC_array should be called only for PY8'
c$$$         stop
c$$$      endif
c$$$
c$$$c Initialise
c$$$      do i=0,3
c$$$         pp_rec(i)=0d0
c$$$         xp1(i)=0d0
c$$$         xp2(i)=0d0
c$$$         xk1(i)=0d0
c$$$         xk2(i)=0d0
c$$$         xk3(i)=0d0
c$$$      enddo
c$$$      nfinal=nexternal-2
c$$$      xm12=0d0
c$$$      xm22=0d0
c$$$      xq1q=0d0
c$$$      xq2q=0d0
c$$$      qMC_a=-1d0
c$$$      qMC_a2=-1d0
c$$$
c$$$c Discard if unphysical FKS variables
c$$$      if(xi_i_fks.lt.0d0.or.xi_i_fks.gt.1d0.or.
c$$$     &   abs(y_ij_fks).gt.1d0)then
c$$$         write(*,*)'Error 0 in assign_qMC_array: fks variables'
c$$$         write(*,*)xi_i_fks,y_ij_fks
c$$$         stop
c$$$      endif
c$$$
c$$$      do ipart=1,nexternal
c$$$      if(ipart.eq.i_fks.or..not.
c$$$     &   (pdg_type(ipart).eq.21.or.abs(pdg_type(ipart)).le.6))cycle
c$$$      if(ipart.eq.j_fks)then
c$$$         qMC_a(ipart)=qMC
c$$$         cycle
c$$$      endif
c$$$c Determine ileg
c$$$c ileg = 1 ==> emission from left     incoming parton
c$$$c ileg = 2 ==> emission from right    incoming parton
c$$$c ileg = 3 ==> emission from massive  outgoing parton
c$$$c ileg = 4 ==> emission from massless outgoing parton
c$$$      if(ipart.le.2)then
c$$$         ileg=ipart
c$$$      elseif(pmass(ipart).ne.0d0)then
c$$$         ileg=3
c$$$      elseif(pmass(ipart).eq.0d0)then
c$$$         ileg=4
c$$$      else
c$$$         write(*,*)'Error 1 in assign_qMC_array: unknown ileg'
c$$$         write(*,*)ileg,ipart,pmass(ipart)
c$$$         stop
c$$$      endif
c$$$
c$$$c Determine and assign momenta:
c$$$c xp1 = incoming left parton  (emitter (recoiler) if ileg = 1 (2))
c$$$c xp2 = incoming right parton (emitter (recoiler) if ileg = 2 (1))
c$$$c xk1 = outgoing parton       (emitter (recoiler) if ileg = 3 (4))
c$$$c xk2 = outgoing parton       (emitter (recoiler) if ileg = 4 (3))
c$$$c xk3 = extra parton          (FKS parton)
c$$$      do j=0,3
c$$$c xk1 and xk2 are never used for ISR
c$$$         xp1(j)=pp(j,1)
c$$$         xp2(j)=pp(j,2)
c$$$         xk3(j)=pp(j,i_fks)
c$$$         if(ileg.gt.2)pp_rec(j)=pp(j,1)+pp(j,2)-pp(j,i_fks)-pp(j,ipart)
c$$$         if(ileg.eq.3)then
c$$$            xk1(j)=pp(j,ipart)
c$$$            xk2(j)=pp_rec(j)
c$$$         elseif(ileg.eq.4)then
c$$$            xk1(j)=pp_rec(j)
c$$$            xk2(j)=pp(j,ipart)
c$$$         endif
c$$$      enddo
c$$$
c$$$c Determine the Mandelstam invariants needed in the MC functions in terms
c$$$c of FKS variables: the argument of MC functions are (p+k)^2, NOT 2 p.k
c$$$c
c$$$c Definitions of invariants in terms of momenta
c$$$c
c$$$c xm12 =     xk1 . xk1
c$$$c xm22 =     xk2 . xk2
c$$$c xtk  = - 2 xp1 . xk3
c$$$c xuk  = - 2 xp2 . xk3
c$$$c xq1q = - 2 xp1 . xk1 + xm12
c$$$c xq2q = - 2 xp2 . xk2 + xm22
c$$$c w1   = + 2 xk1 . xk3        = - xq1q + xq2q - xtk
c$$$c w2   = + 2 xk2 . xk3        = - xq2q + xq1q - xuk
c$$$c xq1c = - 2 xp1 . xk2        = - s - xtk - xq1q + xm12
c$$$c xq2c = - 2 xp2 . xk1        = - s - xuk - xq2q + xm22
c$$$c
c$$$c Parametrisation of invariants in terms of FKS variables
c$$$c
c$$$c ileg = 1
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xk1  =  irrelevant
c$$$c xk2  =  irrelevant
c$$$c yi = y_ij_fks
c$$$c x = 1 - xi_i_fks
c$$$c B = sqrt(s)/2*(1-x)
c$$$c
c$$$c ileg = 2
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yi**2) , -yi )
c$$$c xk1  =  irrelevant
c$$$c xk2  =  irrelevant
c$$$c yi = y_ij_fks
c$$$c x = 1 - xi_i_fks
c$$$c B = sqrt(s)/2*(1-x)
c$$$c
c$$$c ileg = 3
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c$$$c xk1  =  ( sqrt(veckn_ev**2+xm12) , 0 , 0 , veckn_ev )
c$$$c xk2  =  xp1 + xp2 - xk1 - xk3
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
c$$$c yj = y_ij_fks
c$$$c yi = irrelevant
c$$$c x = 1 - xi_i_fks
c$$$c veckn_ev is such that xk2**2 = xm22
c$$$c B = sqrt(s)/2*(1-x)
c$$$c azimuth = irrelevant (hence set = 0)
c$$$c
c$$$c ileg = 4
c$$$c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c$$$c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c$$$c xk1  =  xp1 + xp2 - xk2 - xk3
c$$$c xk2  =  A * ( 1 , 0 , 0 , 1 )
c$$$c xk3  =  B * ( 1 , 0 , sqrt(1-yj**2) , yj )
c$$$c yj = y_ij_fks
c$$$c yi = irrelevant
c$$$c x = 1 - xi_i_fks
c$$$c A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-yj)))
c$$$c B = sqrt(s)/2*(1-x)
c$$$c azimuth = irrelevant (hence set = 0)
c$$$
c$$$      if(ileg.eq.1)then
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         if(shower_mc.eq.'HERWIG6'  .or.
c$$$     &        shower_mc.eq.'HERWIGPP' )qMC_a(ipart)=xi_i_fks/2*sqrt(sh*(1-y_ij_fks**2))
c$$$         if(shower_mc.eq.'PYTHIA6Q' )qMC_a(ipart)=sqrt(-xtk)
c$$$         if(shower_mc.eq.'PYTHIA6PT'.or.
c$$$     &        shower_mc.eq.'PYTHIA8'  )qMC_a(ipart)=sqrt(-xtk*xi_i_fks)
c$$$      elseif(ileg.eq.2)then
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         if(shower_mc.eq.'HERWIG6'  .or.
c$$$     &        shower_mc.eq.'HERWIGPP' )qMC_a(ipart)=xi_i_fks/2*sqrt(sh*(1-y_ij_fks**2))
c$$$         if(shower_mc.eq.'PYTHIA6Q' )qMC_a(ipart)=sqrt(-xuk)
c$$$         if(shower_mc.eq.'PYTHIA6PT'.or.
c$$$     &        shower_mc.eq.'PYTHIA8'  )qMC_a(ipart)=sqrt(-xuk*xi_i_fks)
c$$$      elseif(ileg.eq.3)then
c$$$         xm12=pmass(ipart)**2
c$$$         xm22=dot(pp_rec,pp_rec)
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         xq1q=-2*dot(xp1,xk1)+xm12
c$$$         xq2q=-2*dot(xp2,xk2)+xm22
c$$$         w1=-xq1q+xq2q-xtk
c$$$         w2=-xq2q+xq1q-xuk
c$$$         if(shower_mc.eq.'HERWIG6'.or.
c$$$     &        shower_mc.eq.'HERWIGPP')then
c$$$            zeta1=get_zeta(sh,w1,w2,xm12,xm22)
c$$$            qMCarg=zeta1*((1-zeta1)*w1-zeta1*xm12)
c$$$            if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny)qMCarg=0d0
c$$$            if(qMCarg.lt.-tiny) then
c$$$               isqrtneg=isqrtneg+1
c$$$               write(*,*)'Error 2 in assign_qMC_array: negtive sqrt'
c$$$               write(*,*)qMCarg,isqrtneg
c$$$               if(isqrtneg.ge.100)stop
c$$$            endif
c$$$            qMC_a(ipart)=sqrt(qMCarg)
c$$$         elseif(shower_mc.eq.'PYTHIA6Q')then
c$$$            qMC_a(ipart)=sqrt(w1+xm12)
c$$$         elseif(shower_mc.eq.'PYTHIA6PT')then
c$$$            write(*,*)'PYTHIA6PT not available for FSR'
c$$$            stop
c$$$         elseif(shower_mc.eq.'PYTHIA8')then
c$$$            z=zPY8(ileg,xm12,xm22,sh,1d0-xi_i_fks,0d0,y_ij_fks,xtk
c$$$     $           ,xuk,xq1q,xq2q)
c$$$            qMC_a(ipart)=sqrt(z*(1-z)*w1)
c$$$         endif
c$$$      elseif(ileg.eq.4)then
c$$$         xm12=dot(pp_rec,pp_rec)
c$$$         xm22=0d0
c$$$         xtk=-2*dot(xp1,xk3)
c$$$         xuk=-2*dot(xp2,xk3)
c$$$         xij=2*(1-xm12/sh-xi_i_fks)/(2-xi_i_fks*(1-y_ij_fks))
c$$$         w2=2*dot(xk2,xk3)
c$$$         xq2q=-2*dot(xp2,xk2)+xm22
c$$$         xq1q=xuk+xq2q+w2
c$$$         w1=-xq1q+xq2q-xtk
c$$$         if(shower_mc.eq.'HERWIG6'.or.
c$$$     &        shower_mc.eq.'HERWIGPP')then
c$$$            zeta2=get_zeta(sh,w2,w1,xm22,xm12)
c$$$            qMCarg=zeta2*(1-zeta2)*w2
c$$$            if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny)qMCarg=0d0
c$$$            if(qMCarg.lt.-tiny)then
c$$$               isqrtneg=isqrtneg+1
c$$$               write(*,*)'Error 3 in assign_qMC_array: negtive sqrt'
c$$$               write(*,*)qMCarg,isqrtneg
c$$$               if(isqrtneg.ge.100)stop
c$$$            endif
c$$$            qMC_a(ipart)=sqrt(qMCarg)
c$$$         elseif(shower_mc.eq.'PYTHIA6Q')then
c$$$            qMC_a(ipart)=sqrt(w2)
c$$$         elseif(shower_mc.eq.'PYTHIA6PT')then
c$$$            write(*,*)'PYTHIA6PT not available for FSR'
c$$$            stop
c$$$         elseif(shower_mc.eq.'PYTHIA8')then
c$$$            z=zPY8(ileg,xm12,xm22,sh,1d0-xi_i_fks,0d0,y_ij_fks,xtk
c$$$     $           ,xuk,xq1q,xq2q)
c$$$            qMC_a(ipart)=sqrt(z*(1-z)*w2)
c$$$         endif
c$$$      else
c$$$         write(*,*)'Error 4 in assign_qMC_array: assigned wrong ileg'
c$$$         stop
c$$$      endif
c$$$      enddo
c$$$c
c$$$c qMC_a2 is generated from qMC_a through two operations (here, n is the
c$$$c number of particles at the Born level):
c$$$c - a relabelling from n+1 entries, where i_fks is skipped, to n entries, 
c$$$c   all filled (thus, the conventions are the same as those relevant e.g. 
c$$$c   to xscales and xscales2);
c$$$c - a conversion from an array to a matrix, where the first index represents 
c$$$c   the emitter of i_fks, and the second one is the recoiler (connected with
c$$$c   a colour line to the emitter). Since in PY8 the dependence on the shower 
c$$$c   variable is immaterial (or negligible), all columns are filled with
c$$$c   the same value. In more realistic cases, only one (two) column(s) per
c$$$c   row must be non-zero in the case of quarks (gluons)
c$$$      do i=1,nexternal
c$$$        if(i.lt.i_fks)then
c$$$          iRtoB(i)=i
c$$$        elseif(i.eq.i_fks)then
c$$$          iRtoB(i)=-1
c$$$        elseif(i.gt.i_fks)then
c$$$          iRtoB(i)=i-1
c$$$        endif
c$$$      enddo
c$$$      do i=1,nexternal
c$$$         if(i.eq.i_fks)cycle
c$$$         do j=1,nexternal
c$$$            if(j.eq.i_fks)cycle
c$$$            if(j.eq.i)cycle
c$$$            if(.not.(pdg_type(j).eq.21.or.abs(pdg_type(j)).le.6))cycle
c$$$            qMC_a2(iRtoB(i),iRtoB(j))=qMC_a(i)
c$$$         enddo
c$$$      enddo
c$$$
c$$$c Checks on invariants
c$$$c      call check_invariants(ileg,sh,xtk,xuk,w1,w2,xq1q,xq2q,xm12,xm22)
c$$$
c$$$      return
c$$$      end
