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

c$$$      subroutine compute_xmcsubt_for_checks(pp,xi_i_fks,y_ij_fks,wgt)
c$$$      use process_module
c$$$      use kinematics_module
c$$$      use scale_module
c$$$      implicit none
c$$$      include "nexternal.inc"
c$$$c$$$      include 'madfks_mcatnlo.inc'
c$$$      include 'run.inc'
c$$$      include 'born_nhel.inc'
c$$$      double precision pp(0:3,nexternal),wgt
c$$$      double precision xi_i_fks,y_ij_fks
c$$$      double precision xmc,xrealme,probne,sumMCsec
c$$$      double precision z(nexternal),ddum,dummy
c$$$      integer nofpartners,idum,ione,iord
c$$$      logical lzone(nexternal),flagmc
c$$$
c$$$      ! amp split stuff
c$$$      include 'orders.inc'
c$$$      integer iamp
c$$$      double precision amp_split_mc(amp_split_size)
c$$$      common /to_amp_split_mc/amp_split_mc
c$$$      double precision amp_split_gfunc(amp_split_size)
c$$$      common /to_amp_split_gfunc/amp_split_gfunc
c$$$      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
c$$$     $                 amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
c$$$      common /to_amp_split_bornbars/amp_split_bornbars,
c$$$     $                              amp_split_bornbarstilde
c$$$      logical split_type(nsplitorders) 
c$$$      common /c_split_type/split_type
c$$$
c$$$      integer npartner,cflows
c$$$      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
c$$$      common /MC_info/ ipartners,colorflow
c$$$      logical first_MCcnt_call
c$$$      common/cMCcall/first_MCcnt_call
c$$$
c$$$      double precision xkern(2),xkernazi(2),factor,N_p
c$$$      double precision bornbars(max_bcol,nsplitorders),
c$$$     $     bornbarstilde(max_bcol,nsplitorders)
c$$$c$$$      double precision emsca_a(nexternal,nexternal)
c$$$c$$$     $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
c$$$c$$$     $     ,nexternal) ,scalemin_a(nexternal,nexternal)
c$$$c$$$     $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
c$$$c$$$     $     ,nexternal)
c$$$c$$$      common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2
c$$$c$$$     $     ,scalemin_a,scalemax_a,emscwgt_a
c$$$      integer i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      double precision evnt_wgt
c$$$      integer i, j,iord_val
c$$$      double precision mu_r
c$$$      double precision pb(0:4,-nexternal+3:2*nexternal-3)
c$$$      double precision p_read(0:4,2*nexternal-3), wgt_read
c$$$      integer npart
c$$$      double precision MCsec(nexternal,max_bcol)
c$$$      logical isspecial(max_bcol)
c$$$      integer              MCcntcalled
c$$$      common/c_MCcntcalled/MCcntcalled
c$$$      common/cisspecial/isspecial
c$$$!     common block used to make the (scalar) reference scale partner
c$$$!     dependent in case of delta
c$$$      integer cur_part
c$$$      common /to_ref_scale/cur_part
c$$$      double precision smin,smax,ptresc,emscafun,qMC,damping
c$$$     $     ,compute_damping_weight
c$$$      first_MCcnt_call=.true.
c$$$      MCsec(1:nexternal,1:max_bcol)=0d0
c$$$      sumMCsec=0d0
c$$$      amp_split_mc(1:amp_split_size)=0d0
c$$$      do npartner=1,ipartners(0)
c$$$         cur_part=ipartners(npartner)
c$$$         call xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne
c$$$     $        ,nofpartners,lzone,flagmc,z,xkern,xkernazi
c$$$     $        ,bornbars,bornbarstilde,npartner)
c$$$         if(.not.lzone(npartner)) cycle
c$$$         damping=compute_damping_weight(cur_part,xi_i_fks
c$$$     $        ,y_ij_fks)
c$$$         do cflows=1,max_bcol
c$$$            if (colorflow(npartner,cflows).eq.0) cycle
c$$$            if (isspecial(cflows)) then
c$$$               N_p=2d0
c$$$            else
c$$$               N_p=1d0
c$$$            endif
c$$$            ione=0
c$$$            do iord = 1, nsplitorders
c$$$               if (.not.split_type(iord) .or.
c$$$     $              (iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
c$$$               if (iord.eq.qcd_pos) then
c$$$                  iord_val=1
c$$$               elseif(iord.eq.qed_pos) then
c$$$                  iord_val=2
c$$$               endif
c$$$               ione=ione+1
c$$$               MCsec(npartner,colorflow(npartner,cflows))=damping
c$$$     $              *(xkern(iord_val)*N_p*bornbars(colorflow(npartner
c$$$     $              ,cflows),iord)+xkernazi(iord_val)*N_p
c$$$     $              *bornbarstilde(colorflow(npartner,cflows),iord))
c$$$               amp_split_mc(1:amp_split_size) =
c$$$     $              amp_split_mc(1:amp_split_size)+damping
c$$$     $              *(xkern(iord_val)*N_p
c$$$     $              *amp_split_bornbars(1:amp_split_size
c$$$     $              ,colorflow(npartner,cflows),iord)+xkernazi(iord_val)
c$$$     $              *N_p *amp_split_bornbarstilde(1:amp_split_size
c$$$     $              ,colorflow(npartner,cflows),iord))
c$$$            enddo
c$$$            if (ione.ne.1) then
c$$$               write (*,*) 'Error: incompatible split orders in '/
c$$$     $              /'compute_xmcsubt_complete',ione
c$$$               stop 1
c$$$            endif
c$$$            sumMCsec=sumMCsec+MCsec(npartner,colorflow(npartner
c$$$     $           ,cflows))
c$$$         enddo
c$$$      enddo
c$$$      call xmcsubtME(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,xrealme)
c$$$      wgt=sumMCsec+xrealme
c$$$      do iamp=1, amp_split_size
c$$$        amp_split_mc(iamp) = amp_split_mc(iamp) + amp_split_gfunc(iamp)
c$$$      enddo
c$$$      return
c$$$      end
c$$$

! New structure:
!
!     0. Given Born flow (with a MC sum over flows):
!      
!     1. Outside loop over FKS configurations
!      
!     2. For each configuration, compute relevant kinematic variables
!     (xi_fks, yij_fks, etc.)
!      
!     3. Compute value of MC subtraction, given those kinematic
!     variables
!      
!     4. For H-event: take sum of all of them and use that to subtract
!     from Real emission, so that it can be multiplied by an overall
!     S-function relevant to the original i_fks and j_fks configuration
!     (i.e., the same that multiplies the real emission).
!     --> \sum_ij S_ij ( R - \sum_kl MC_kl )
!      
!     5. For S-event: Take only the one relevant for the original i_fks
!     and j_fks configuration. (Same as original code).

      
      subroutine compute_MCsubtraction_kl(k_fks,l_fks,xi,y,p,p_cm,p_born
     $     ,include_gfun,z,n_connect,amp_split_xmcxsec)
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'fks_info.inc'
      include 'orders.inc'
      integer k_fks,l_fks,i
      logical lzone(2)
      double precision p(0:3,nexternal),p_born(0:3,nexternal-1),xi,y
     $     ,mass,z(2),amp_split_xmcxsec(1:amp_split_size,2)
     $     ,p_cm(0:3,nexternal)
      double precision pmass(nexternal)
      common /to_mass/pmass
      double precision :: veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      integer n_connect,i_connect(2),iconnect
      logical include_gfun
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
      mass=pmass(l_fks)
      veckn_ev=rho(p_cm(0,l_fks))
      veckbarn_ev=rho(p_born(0,min(k_fks,l_fks)))
      xp0jfks=p_cm(0,l_fks)

      call fill_kinematics_module(p_cm,k_fks,l_fks,xi,y,mass
     $     ,include_gfun)
!     compute MC subtraction term for the 'kl' configuration
      
!     find to which particle(s) fksfather connects in the colour flow
      call find_color_connectors(born_flow_picked,fksfather,n_connect
     $     ,i_connect)

!     given the flow, loop over the (up to two) partners of the
!     fks-father.
      do iconnect=1,n_connect
         call xmcsubt_connection(p,xi,y,p_born,i_connect(iconnect)
     $        ,lzone(iconnect),z(iconnect),amp_split_xmcxsec(1
     $        ,iconnect))
      enddo
      if (.not.any(lzone(1:n_connect)) .and. include_gfun) then
! include_gfun is only .true. if kl==ij. If we are in the
! deadzone, we do not want to include the MC counter terms (and
! therefore also not the gfun contributions).
! Exception: we are in a soft-wide-angle configuration, there the shower
! might be 'incorrect', and the g-function should be included.
         if (xi.gt.abs(besf)/2d0) then ! besf=0.1 by default
            include_gfun=.false.
            gfactsf=1d0
            gfactcl=1d0
            gfactazi=0d0
         endif
      endif

!     TODO: "check_positivity_MCxsec" at some point?
      if (any(lzone(1:n_connect))) then
         amp_split_xmcxsec(1:amp_split_size,1:2)=amp_split_xmcxsec(
     $        1:amp_split_size,1:2)
     $        /(xi**2*(1d0-y)) ! re-instate 1/xi^2 and 1/(1-y); they
                               ! should not depend on 'kl', but rather
                               ! on 'ij'
      else
         amp_split_xmcxsec(1:amp_split_size,1:2)=0d0
      endif
      end
      
      subroutine find_color_connectors(iflow,iparticle,n_connect
     $     ,i_connect)
      use process_module
      implicit none
      include 'nexternal.inc'
      include "genps.inc"
      include "born_nhel.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include "born_leshouche.inc"
      integer iflow,iparticle,n_connect,i_connect(2),i
      logical isspecial(max_bcol)
      common/cisspecial/isspecial
      n_connect=0
      do i=1,next_n
         if (valid_dipole_n(i,iparticle,iflow)) then
            n_connect=n_connect+1
            if (n_connect.gt.2) then
               write (*,*) 'ERROR: too many connections.'
               write (*,*) iflow,iparticle
               write (*,*) valid_dipole_n(1:next_n,iparticle,iflow)
               stop 1
            endif
            i_connect(n_connect)=i
         endif
      enddo
      if (n_connect.eq.1 .and. idup(iparticle,1).eq.21) then
         if (isspecial(iflow)) then
!     This is the ISSPECIAL case. Add one more (identical) connection.
            ! TODO: this can be optimised, since now we compute twice
            ! the same subtraction terms.
            n_connect=n_connect+1
            i_connect(n_connect)=i_connect(n_connect-1)
         endif
      endif
      if (n_connect.eq.0) then
         write (*,*) 'ERROR: no connections found.'
         write (*,*) iflow,iparticle
         write (*,*) valid_dipole_n(1:next_n,iparticle,iflow)
         stop 1
      endif
      end
      
      
c$$$      subroutine compute_xmcsubt_complete(p,probne,gfactsf,gfactcl
c$$$     $     ,flagmc,lzone,z_shower,nofpartners,xmcxsec)
c$$$      use kinematics_module
c$$$      use scale_module
c$$$      implicit none
c$$$      include 'nexternal.inc'
c$$$c$$$  include 'madfks_mcatnlo.inc'
c$$$      include 'born_nhel.inc'
c$$$      include 'run.inc'
c$$$      include 'orders.inc'
c$$$      integer npartner,nofpartners,cflows,idum,ione,iord,iord_val
c$$$      logical lzone(nexternal),flagmc
c$$$      double precision bornbars(max_bcol,nsplitorders),
c$$$     $     bornbarstilde(max_bcol,nsplitorders)
c$$$      double precision p(0:3,nexternal),probne,z_shower(nexternal)
c$$$     $     ,xmcxsec(nexternal),xkern(2),xkernazi(2),damping,N_p
c$$$     $     ,MCsec(nexternal,max_bcol),sumMCsec
c$$$     $     ,xmcxsec2(max_bcol),gfactsf,gfactcl,ddum
c$$$      integer i_fks,j_fks
c$$$      common/fks_indices/i_fks,j_fks
c$$$      integer              MCcntcalled
c$$$      common/c_MCcntcalled/MCcntcalled
c$$$      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
c$$$      common /MC_info/ ipartners,colorflow
c$$$      logical isspecial(max_bcol)
c$$$      common/cisspecial/isspecial
c$$$      logical first_MCcnt_call
c$$$      common/cMCcall/first_MCcnt_call
c$$$      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
c$$$     $     ,p_i_fks_cnt(0:3,-2:2)
c$$$      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
c$$$      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
c$$$     $     amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
c$$$      common /to_amp_split_bornbars/amp_split_bornbars,
c$$$     $     amp_split_bornbarstilde
c$$$      double precision amp_split_xmcxsec(amp_split_size,nexternal)
c$$$      common /to_amp_split_xmcxsec/amp_split_xmcxsec
c$$$      double precision amp_split_mc(amp_split_size)
c$$$      common /to_amp_split_mc/amp_split_mc
c$$$      logical split_type(nsplitorders) 
c$$$      common /c_split_type/split_type
c$$$!     common block used to make the (scalar) reference scale partner
c$$$!     dependent in case of delta
c$$$      integer cur_part
c$$$      common /to_ref_scale/cur_part
c$$$      double precision smin,smax,ptresc,compute_damping_weight,qMC
c$$$c     -- call to MC counterterm functions
c$$$      first_MCcnt_call=.true.
c$$$      xmcxsec(1:nexternal)=0d0
c$$$      xmcxsec2(1:max_bcol)=0d0
c$$$      MCsec(1:nexternal,1:max_bcol)=0d0
c$$$      sumMCsec=0d0
c$$$      amp_split_xmcxsec(1:amp_split_size,1:nexternal)=0d0
c$$$      do npartner=1,ipartners(0)
c$$$         cur_part=ipartners(npartner)
c$$$         call xmcsubt(p,xi_i_fks_ev,y_ij_fks_ev,gfactsf,gfactcl,probne
c$$$     $        ,nofpartners,lzone,flagmc,z_shower,xkern,xkernazi
c$$$     $        ,bornbars,bornbarstilde,npartner)
c$$$         if(.not. lzone(npartner)) cycle
c$$$         damping=compute_damping_weight(cur_part,xi_i_fks_ev
c$$$     $        ,y_ij_fks_ev)
c$$$         do cflows=1,max_bcol
c$$$            if (colorflow(npartner,cflows).eq.0) cycle
c$$$            if (isspecial(cflows)) then
c$$$               N_p=2d0
c$$$            else
c$$$               N_p=1d0
c$$$            endif
c$$$            ione=0
c$$$            do iord = 1, nsplitorders
c$$$               if (.not.split_type(iord) .or.
c$$$     $              (iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
c$$$               if (iord.eq.qcd_pos) then
c$$$                  iord_val=1
c$$$               elseif(iord.eq.qed_pos) then
c$$$                  iord_val=2
c$$$               endif
c$$$               ione=ione+1
c$$$               MCsec(npartner,colorflow(npartner,cflows))=damping
c$$$     $              *(xkern(iord_val)*N_p*bornbars(colorflow(npartner
c$$$     $              ,cflows),iord)+xkernazi(iord_val)*N_p
c$$$     $              *bornbarstilde(colorflow(npartner,cflows),iord))
c$$$               amp_split_xmcxsec(1:amp_split_size,npartner) =
c$$$     $              amp_split_xmcxsec(1:amp_split_size,npartner) +
c$$$     $              damping *(xkern(iord_val)*N_p
c$$$     $              *amp_split_bornbars(1:amp_split_size
c$$$     $              ,colorflow(npartner,cflows),iord)+xkernazi(iord_val)
c$$$     $              *N_p*amp_split_bornbarstilde(1:amp_split_size
c$$$     $              ,colorflow(npartner,cflows),iord))
c$$$            enddo
c$$$            if (ione.ne.1) then
c$$$               write (*,*) 'Error: incompatible split orders in '/
c$$$     $              /'compute_xmcsubt_complete',ione
c$$$               stop 1
c$$$            endif
c$$$            xmcxsec(npartner)=xmcxsec(npartner)+MCsec(npartner
c$$$     $           ,colorflow(npartner,cflows))
c$$$            xmcxsec2(colorflow(npartner,cflows))=
c$$$     $           xmcxsec2(colorflow(npartner,cflows))+MCsec(npartner
c$$$     $           ,colorflow(npartner,cflows))
c$$$            sumMCsec=sumMCsec+MCsec(npartner,colorflow(npartner
c$$$     $           ,cflows))
c$$$         enddo
c$$$      enddo
c$$$
c$$$!     check the MC cross sections are positive:
c$$$      call check_positivity_MCxsec(sumMCsec,xmcxsec,xmcxsec2)
c$$$      if (mcatnlo_delta) then
c$$$!     compute and include the Delta Sudakov:
c$$$         if(any(lzone(1:ipartners(0)))) call compute_delta(p
c$$$     $        ,probne)
c$$$      endif
c$$$      xmcxsec(1:ipartners(0))=xmcxsec(1:ipartners(0))*probne
c$$$      amp_split_xmcxsec(1:amp_split_size,1:ipartners(0))=
c$$$     $     amp_split_xmcxsec(1:amp_split_size,1:ipartners(0))*probne
c$$$      if (btest(Mccntcalled,4)) then
c$$$         write (*,*) 'Fifth bit of MCcntcalled should not '/
c$$$     $        /'have been set yet',MCcntcalled
c$$$         stop 1
c$$$      endif
c$$$      if(any(lzone(1:ipartners(0)))) MCcntcalled=MCcntcalled+16
c$$$      return
c$$$      end

      double precision function compute_damping_weight(cur_part
     $     ,xi_i_fks,y_ij_fks)
      use kinematics_module
      use scale_module
      implicit none
      integer :: cur_part
      double precision :: xi_i_fks,y_ij_fks,emscafun,smin,smax,qMC
     $     ,ptresc
      smin=shower_scale_nbody_min(cur_part,fksfather)
      smax=shower_scale_nbody_max(cur_part,fksfather)
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
      

c Main routine for MC counterterms. Now to be called inside a loop
c over colour partners
      subroutine xmcsubt_connection(pp,xi_i_fks,y_ij_fks,p_born
     $     ,i_connect,lzone,z,amp_split_xmcxsec)
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include 'born_nhel.inc'
      include 'orders.inc'
      include 'fks_powers.inc'
      include 'coupl.inc'
!     arguments:
      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,p_born(0:3
     $     ,nexternal-1) ,probne ,z,xkern(2),xkernazi(2)
     $     ,bornbars(max_bcol ,nsplitorders),bornbarstilde(max_bcol
     $     ,nsplitorders),amp_split_xmcxsec(1:amp_split_size)
      integer i_connect,ione,iord,iord_val
      logical lzone
!     local
      double precision ztmp,xitmp,xjactmp,qMC,delta,E0sq
     $     ,PY6PTweight,pmass(nexternal),xi,xjac
!     external
      double precision bogus_probne_fun,gfunction,zHW6,xiHW6
     $     ,xjacHW6,compute_damping_weight
      external bogus_probne_fun,gfunction,zHW6,xiHW6,xjacHW6
     $     ,compute_damping_weight
!     parameters      
      double precision ymin,zero
      parameter (ymin=0.9d0)
      parameter(zero=0d0)
!     common
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      logical split_type(nsplitorders) 
      common /c_split_type/split_type
      double precision amp_split_bornbars(amp_split_size,max_bcol,nsplitorders),
     $                 amp_split_bornbarstilde(amp_split_size,max_bcol,nsplitorders)
      common /to_amp_split_bornbars/amp_split_bornbars,
     $                              amp_split_bornbarstilde
      include "pmass.inc"

c     Initialise if first time
      if (split_type(QED_pos)) then
!     TODO set QED flows correctly (but not here, rather in
!     compute_MCsubtraction_kl)
         write (*,*) 'TODO set QED flows correctly'
         stop 1
         call set_QED_flows(pp)
      endif
      ztmp     = 0d0
      xitmp    = 0d0
      xjactmp  = 0d0

      qMC=get_qMC(xi_i_fks,y_ij_fks)

c     New or standard MC@NLO formulation
      probne=bogus_probne_fun(qMC)

c     Call barred Born and assign shower scale
      call get_mbar(pp,xi_i_fks,y_ij_fks,p_born,ileg,bornbars
     $     ,bornbarstilde)
      
c$$$  c     Distinguish ISR and FSR
c$$$  if(ileg.le.2)then
c$$$  delta=min(1d0,deltaI)
c$$$  elseif(ileg.ge.3)then
c$$$  delta=min(1d0,deltaO)
c$$$  endif
c$$$  
c$$$  c     G-function parameters 
c$$$  gfactsf=gfunction(x,alsf,besf,2d0)
c$$$  if(abs(i_type).eq.3)gfactsf=1d0 ! if fks parton is quark, soft limit is finite
c$$$  gfactcl=gfunction(y_ij_fks,alsf,-(1d0-ymin),1d0)
c$$$  if(alazi.lt.0d0)gfactazi=1-gfunction(y_ij_fks,-alazi,beazi,delta)

c$$$      if (btest(MCcntcalled,2)) then
c$$$         write (*,*) 'Third bit of MCcntcalled should not be set yet'
c$$$     $        ,MCcntcalled
c$$$         stop 1
c$$$      endif
c$$$
c$$$      MCcntcalled=MCcntcalled+4
      
c     Shower variables
      E0sq=dot(p_born(0,fksfather),p_born(0,i_connect))
      call get_shower_variables(E0sq,z,xi,xjac)
      
c     Compute dead zones
      call get_dead_zone(z,xi,p_born,qMC,i_connect,lzone,PY6PTweight)
      
c     Compute MC subtraction terms
      if (lzone) then
         call limits(xi_i_fks,y_ij_fks)
         call compute_splitting_kernels(xkern,xkernazi,z,xi,xjac)
      else
         xkern(1:2)=0d0
         xkernazi(1:2)=0d0
      endif
c     
      if (shower_mc_mod(1:9).eq.'PYTHIA6PT') then
         xkern(1:2)=xkern(1:2)*PY6PTweight
         xkernazi(1:2)=xkernazi(1:2)*PY6PTweight
      endif

      ! For ij-fks, we include gfactazi to remove power corrections (due
      ! to gluon-correlations) away from the limit---for another sectors
      ! we can do whatever, since kinematic configurations for which
      ! that is relevant are damped by the S-function.
      xkern(1:2)=xkern(1:2)*gfactsf
      xkernazi(1:2)=xkernazi(1:2)*gfactazi*gfactsf
         
      ione=0
      amp_split_xmcxsec(1:amp_split_size)=0d0
      do iord = 1, nsplitorders
         if (.not.split_type(iord) .or.
     $        (iord.ne.qed_pos.and.iord.ne.qcd_pos)) cycle
         if (iord.eq.qcd_pos) then
            iord_val=1
         elseif(iord.eq.qed_pos) then
            iord_val=2
         endif
         ione=ione+1
         amp_split_xmcxsec(1:amp_split_size)=(xkern(iord_val)*
     $        amp_split_bornbars(1:amp_split_size,born_flow_picked,iord)
     $        +xkernazi(iord_val)*
     $        amp_split_bornbarstilde(1:amp_split_size,born_flow_picked,iord))
     $        *compute_damping_weight(i_connect,xi_i_fks,y_ij_fks)
      enddo
      if (ione.ne.1) then
         write (*,*) 'Error: incompatible split orders in '/
     $        /'xmcsubt_connection: there should be exactly'/
     $        /' one in MC@NLO. You can either do QCD *or* '/
     $        /'QED corrections',ione
         stop 1
      endif
      return
      end

      
c$$$c Main routine for MC counterterms. Now to be called inside a loop
c$$$c over colour partners
c$$$      subroutine xmcsubt(pp,xi_i_fks,y_ij_fks,gfactsf,gfactcl,probne,
c$$$     &     nofpartners,lzone,flagmc,z,xkern,xkernazi,
c$$$     &     bornbars,bornbarstilde,npartner)
c$$$      ! TODO cleanup 'flagmc'
c$$$      use process_module
c$$$      use kinematics_module
c$$$      use scale_module
c$$$      implicit none
c$$$      include 'nexternal.inc'
c$$$      include 'born_nhel.inc'
c$$$      include 'orders.inc'
c$$$      include 'fks_powers.inc'
c$$$      include 'coupl.inc'
c$$$! arguments:
c$$$      double precision pp(0:3,nexternal),xi_i_fks,y_ij_fks,gfactsf,gfactcl
c$$$     $     ,probne,z(nexternal),xkern(2),xkernazi(2),bornbars(max_bcol
c$$$     $     ,nsplitorders),bornbarstilde(max_bcol,nsplitorders)
c$$$      integer nofpartners,npartner
c$$$      logical lzone(nexternal),flagmc
c$$$
c$$$! local
c$$$      double precision ztmp,xitmp,xjactmp,gfactazi,qMC,delta,E0sq
c$$$     $     ,PY6PTweight,pmass(nexternal),xi,xjac
c$$$! external
c$$$      double precision bogus_probne_fun,gfunction,zHW6,xiHW6
c$$$     $     ,xjacHW6
c$$$      external bogus_probne_fun,gfunction,zHW6,xiHW6,xjacHW6
c$$$! parameters      
c$$$      double precision ymin,zero
c$$$      parameter (ymin=0.9d0)
c$$$      parameter(zero=0d0)
c$$$! common
c$$$      logical first_MCcnt_call
c$$$      common/cMCcall/first_MCcnt_call
c$$$      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
c$$$      common /MC_info/ ipartners,colorflow
c$$$      double precision alsf,besf
c$$$      common/cgfunsfp/alsf,besf
c$$$      double precision alazi,beazi
c$$$      common/cgfunazi/alazi,beazi
c$$$      integer              MCcntcalled
c$$$      common/c_MCcntcalled/MCcntcalled
c$$$      double precision       ch_i,ch_j,ch_m
c$$$      integer                i_type,j_type,m_type
c$$$      common/cparticle_types/ch_i,ch_j,ch_m,
c$$$     &                       i_type,j_type,m_type
c$$$      logical split_type(nsplitorders) 
c$$$      common /c_split_type/split_type
c$$$      double precision p_born(0:3,nexternal-1)
c$$$      common/pborn/p_born
c$$$      save
c$$$
c$$$      include "pmass.inc"
c$$$
c$$$c Initialise if first time
c$$$      if(.not.first_MCcnt_call)goto 222
c$$$      if (split_type(QED_pos)) then
c$$$         ! QED partners are dynamically found
c$$$         call set_QED_flows(pp)
c$$$      endif
c$$$      flagmc   = .false.
c$$$      ztmp     = 0d0
c$$$      xitmp    = 0d0
c$$$      xjactmp  = 0d0
c$$$      gfactazi = 0d0
c$$$      nofpartners = ipartners(0)
c$$$
c$$$      qMC=get_qMC(xi_i_fks,y_ij_fks)
c$$$
c$$$c     New or standard MC@NLO formulation
c$$$      probne=bogus_probne_fun(qMC)
c$$$
c$$$c Call barred Born and assign shower scale
c$$$      call get_mbar(pp,y_ij_fks,ileg,bornbars,bornbarstilde)
c$$$
c$$$c Distinguish ISR and FSR
c$$$      if(ileg.le.2)then
c$$$         delta=min(1d0,deltaI)
c$$$      elseif(ileg.ge.3)then
c$$$         delta=min(1d0,deltaO)
c$$$      endif
c$$$c G-function parameters 
c$$$      gfactsf=gfunction(x,alsf,besf,2d0)
c$$$      if(abs(i_type).eq.3)gfactsf=1d0 ! if fks parton is quark, soft limit is finite
c$$$      gfactcl=gfunction(y_ij_fks,alsf,-(1d0-ymin),1d0)
c$$$      if(alazi.lt.0d0)gfactazi=1-gfunction(y_ij_fks,-alazi,beazi,delta)
c$$$
c$$$      if (btest(MCcntcalled,2)) then
c$$$         write (*,*) 'Third bit of MCcntcalled should not be set yet'
c$$$     $        ,MCcntcalled
c$$$         stop 1
c$$$      endif
c$$$
c$$$      MCcntcalled=MCcntcalled+4
c$$$      
c$$$c Shower variables (all except HW6, since that one depends on the
c$$$c partner)
c$$$      call get_shower_variables(E0sq,ztmp,xitmp,xjactmp)
c$$$      
c$$$      first_MCcnt_call=.false.
c$$$ 222  continue
c$$$c Main loop over colour partners used to begin here
c$$$      E0sq=dot(p_born(0,fksfather),
c$$$     $                   p_born(0,ipartners(npartner)))
c$$$      if(E0sq.lt.0d0)then
c$$$         write(*,*)'Error in xmcsubt: negative E0sq'
c$$$         write(*,*)E0sq,ileg,npartner
c$$$         stop
c$$$      endif
c$$$      if(shower_mc_mod(1:7).eq.'HERWIG6')then
c$$$         z(npartner)=zHW6(E0sq)
c$$$         xi=xiHW6(E0sq,z(npartner))
c$$$         xjac=xjacHW6(E0sq,xi,z(npartner))
c$$$      else
c$$$         z(npartner)=ztmp
c$$$         xi=xitmp
c$$$         xjac=xjactmp
c$$$      endif
c$$$c Compute dead zones
c$$$      call get_dead_zone(z(npartner),xi,qMC
c$$$     $     ,ipartners(npartner),lzone(npartner),PY6PTweight)
c$$$
c$$$c Compute MC subtraction terms
c$$$      if(lzone(npartner))then
c$$$         if(.not.flagmc)flagmc=.true.
c$$$         call limits(xi_i_fks,y_ij_fks)
c$$$         call compute_spitting_kernels(xkern,xkernazi,z(npartner)
c$$$     $        ,xi,xjac)
c$$$      else
c$$$        xkern(1:2)=0d0
c$$$        xkernazi(1:2)=0d0
c$$$      endif
c$$$c
c$$$      xkern(1:2)=xkern(1:2)*gfactsf
c$$$      xkernazi(1:2)=xkernazi(1:2)*gfactazi*gfactsf
c$$$      if (shower_mc_mod(1:9).eq.'PYTHIA6PT') then
c$$$         xkern(1:2)=xkern(1:2)*PY6PTweight
c$$$         xkernazi(1:2)=xkernazi(1:2)*PY6PTweight
c$$$      endif
c$$$
c$$$c Main loop over colour partners used to end here
c$$$      return
c$$$      end




      subroutine compute_splitting_kernels(xkern,xkernazi,z,xi,xjac)
      use kinematics_module
      implicit none
      double precision xkern(1:2),xkernazi(1:2),z,xi,xjac
      double precision tiny
      parameter (tiny=1d-6)
      logical limit,non_limit
      common /MCcnt_limit/limit,non_limit
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      xkern(1:2)    = 0d0
      xkernazi(1:2) = 0d0

      ! TODO: check m_type, j_type, etc. when looping over k_fks and l_fks
      
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
      double precision tiny,xi_i_fks,y_ij_fks
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
      use process_module
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg12=(1d0-yij)*(1d0-x)/x * 4d0/(shat_n1*N_p)
      end

      double precision function xfact_ileg3(N_p)
      use process_module
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg3=(2d0-(1d0-x)*(1d0-(kn0/kn)*yij))/
     &     kn*knbar*(1d0-x)*(1d0-yij) * 2d0/(shat_n1*N_p)
      end

      double precision function xfact_ileg4(N_p)
      use process_module
      use kinematics_module
      implicit none
      integer N_p
      xfact_ileg4=(2d0-(1d0-x)*(1d0-yij))/
     &     xij*(1d0-xm12/shat_n1)*(1d0-x)*(1d0-yij) * 2d0/(shat_n1*N_p)
      end

      subroutine compute_splitting_kernel_icode1(xkern,xkernazi,z,xi)
      use process_module
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
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
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
      use process_module
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
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
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
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
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
      include "nexternal.inc"
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
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
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
            if(abs(j_pdg).le.6)then
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



      
      subroutine get_shower_variables(E0sq,z,xi,xjac)
      use process_module
      use kinematics_module
      implicit none
      double precision E0sq,z,xi,xjac
      double precision zHW6,xiHW6,xjacHW6,zHWPP,xiHWPP,xjacHWPP,zPY6Q
     $     ,xiPY6Q,xjacPY6Q,zPY6PT,xiPY6PT,xjacPY6PT,zPY8,xiPY8,xjacPY8
      external zHW6,xiHW6,xjacHW6,zHWPP,xiHWPP,xjacHWPP,zPY6Q,xiPY6Q
     $     ,xjacPY6Q,zPY6PT,xiPY6PT,xjacPY6PT,zPY8,xiPY8,xjacPY8
      if(shower_mc_mod(1:7).eq.'HERWIG6')then
         z=zHW6(E0sq)
         xi=xiHW6(E0sq,z)
         xjac=xjacHW6(E0sq,xi,z)
      elseif(shower_mc_mod(1:8).eq.'HERWIGPP')then
         z=zHWPP()
         xi=xiHWPP(z)
         xjac=xjacHWPP(z)
      elseif(shower_mc_mod(1:8).eq.'PYTHIA6Q')then
         z=zPY6Q()
         xi=xiPY6Q()
         xjac=xjacPY6Q(z)
      elseif(shower_mc_mod(1:9).eq.'PYTHIA6PT')then
         z=zPY6PT()
         xi=xiPY6PT()
         xjac=xjacPY6PT()
      elseif(shower_mc_mod(1:7).eq.'PYTHIA8')then
         z=zPY8()
         xi=xiPY8(z)
         xjac=xjacPY8(z)
      endif
      end

c Finalises the MC counterterm computations performed in xmcsubt(),
c fills arrays relevant to shower scales, and computes Delta
      subroutine compute_delta(p,probne)
      use process_module
      use scale_module
      implicit none
      include "born_nhel.inc"
      include 'nFKSconfigs.inc'
      include 'nexternal.inc'
c$$$  include 'madfks_mcatnlo.inc'
      include 'run.inc'
      include 'orders.inc'

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ptresc,ref_scale,emscainv
c$$$  double precision emscav_a(nexternal,nexternal)
c$$$  double precision emscav_a2(nexternal,nexternal)
      integer cflows,jflow
      common/c_colour_flow/jflow

c$$$  double precision emsca_a(nexternal,nexternal)
c$$$  $     ,emsca_bare_a(nexternal,nexternal),emsca_bare_a2(nexternal
c$$$  $     ,nexternal) ,scalemin_a(nexternal,nexternal)
c$$$  $     ,scalemax_a(nexternal ,nexternal),emscwgt_a(nexternal
c$$$  $     ,nexternal),emsca
c$$$  common/cemsca_a/emsca_a,emsca_bare_a,emsca_bare_a2
c$$$  $     ,scalemin_a,scalemax_a,emscwgt_a
      integer              MCcntcalled
      common/c_MCcntcalled/MCcntcalled

      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      common /MC_info/ ipartners,colorflow

      integer ip

c     Controls assignments of scales in H events in LHE file.
c     Set iHscale=0 for scale=target_scale
c     iHscale=1 for scale=dipole_mass
      integer iHscale,jbar,ifksscl(2)
      parameter (iHscale=0)
      double precision dipole_mass,fksscales(3)
      external dipole_mass


      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

c$$$  double precision emscav_tmp_a(nexternal,nexternal)
c$$$  double precision emscav_tmp_a2(nexternal,nexternal)
c$$$  common/cemscav_tmp_a/emscav_tmp_a,emscav_tmp_a2

      double precision probne
     $     ,dummy_wgt

      integer i,j,k,i1,i2

      double precision p(0:3,nexternal)
c     For the boost to the lab frame
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #sqrtshat,shat
      double precision chy,shy,chymo,xdir(3),p_lab(0:3,nexternal)
      data (xdir(i),i=1,3) /0d0,0d0,1d0/

      double precision xkern(2),xkernazi(2),factor
      include "genps.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      integer idup_s(nexternal-1)
      integer icolup_s(2,nexternal-1)
      integer idup_h(nexternal)
      integer mothup_h(2,nexternal)
      integer icolup_h(2,nexternal)
      integer spinup_local(nexternal)
      integer istup_local(nexternal)
      double precision wgt_sudakov
      double precision scales(0:99)
      common /colour_connections/ icolup_s,icolup_h

c     To access Pythia8 control variables
      include 'pythia8_control.inc'
      include "born_leshouche.inc"
      integer jpart(7,-nexternal+3:2*nexternal-3),lc,iflow
      logical firsttime1
      data firsttime1 /.true./
      include 'leshouche_decl.inc'
      save idup_d, mothup_d, icolup_d, niprocs_d

C     To allow retrieval of S-event from Pythia
      include 'hep_event_streams.inc'

      logical         Hevents
      common/SHevents/Hevents
c     Sevent_starting_scales = m_ij scales that determine S-event scales written onto LHE
      double precision Sevent_starting_scales(nexternal-1,nexternal-1)
c     Hevent_starting_scales = t_ij scales that determine H-event scales written onto LHE
      double precision Hevent_starting_scales(nexternal,nexternal)

c     Lower and upper limits of fitted st and xm ranges.
c     Require one prior call to pysudakov() to be set,
c     here done in the firsttime1 clause
      real*8 cstlow,cstupp,cxmlow,cxmupp
      common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp

c     Set Delta(pt,..)=0 for pt<smallptlow, and interpolate
c     between 0 and Delta(smallptupp,..) for smallptlow<pt<smallptupp
c     For things to work properly, one must have:
c     cstlow <= smallptupp
      real*8 smallptlow,smallptupp,get_to_zero
      parameter (smallptlow=0.5d0)
      parameter (smallptupp=1.01d0)

      integer iii,jjj,LP
      double precision xscales_PY(0:99,0:99),xmasses_PY(0:99,0:99)
      logical*1 dzones_PY(0:99,0:99)
      double precision Sevent_stopping_scales(1:nexternal-1,1:nexternal-1)
     $     ,xmasses_nbody(1:nexternal-1,1:nexternal-1)
      logical*1 dzones_nbody(1:nexternal-1,1:nexternal-1)

      integer id,type,icount,jcount,kcount,jindex(2)
      integer iflip(2)
      data iflip/2,1/

      double precision emscav_a2_tmp,emscav_tmp_a2_tmp,ptresc_a_tmp
      double precision sref,acll1,acll2,dot,sumdot
      external dot,sumdot
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

c     SF ARE noemProb AND mDipole USEFUL?
      double precision startingScale0,stoppingScale0
      double precision noemProb, startingScale(2), stoppingScale(2), mDipole
      double precision mcmass(21)
      double precision pysudakov,deltanum,deltaden,delta(2,2)
      double precision gltmp,xtmp(2),glfact(2),glrat(2)
      integer nG_S,nQ_S,i_dipole_counter,isudtype
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

      double precision gl(2),pdfnum,pdfden,PIk,Fk(2)
      double precision pysudakov_safe,gl_safe
      double precision mu_ij(2,nexternal-1),t_ij(2,nexternal-1)
      integer in_con,out_con,n_connect(nexternal-1)
      integer i_connect(2,nexternal-1)
      integer get_parton_id,setSudType
      integer     fold,ifold_counter
      common /cfl/fold,ifold_counter
      double precision tiny
      parameter       (tiny=1d-10)
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
      
      if (born_flow_picked.le.0) then
         write (*,*) 'born_flow_picked <= 0 in compute_delta'
     $        ,born_flow_picked
         stop 1
      endif
      
c     S-event information:
c     id's and mothers read from born_leshouche.inc;
c     colour configuration read from born_leshouche.inc and born_flow_picked 
      do i=1,nexternal-1
         IDUP_S(i)=IDUP(i,1)
         ICOLUP_S(1,i)=ICOLUP(1,i,born_flow_picked)
         ICOLUP_S(2,i)=ICOLUP(2,i,born_flow_picked)
      enddo

c     Sevent_starting_scales* are the m_ij scales, ie the starting scales (as determined
c     by the D(mu) function) for extra radiation; they are copies of the
c     emscav_tmp_a* arrays, originally filled by xmcsubt(). Only the (i,j) 
c     entries associated with a colour line that belongs to born_flow_picked have
c     meaningful values; the others are set equal to -1.
      Sevent_starting_scales(1:nexternal-1,1:nexternal-1)=
     &     shower_scale_nbody(1:nexternal-1,1:nexternal-1)

c     H-event information.
c     First write ids, mothers and all colours.
      if (firsttime1)then
         firsttime1=.false.
         call read_leshouche_info(idup_d,mothup_d,icolup_d,niprocs_d)
c     Fake call for initialisation
         deltanum=pysudakov_safe(1.d2,2.d2,1,1,mcmass)
         if(cstlow.gt.smallptupp)then
            write(*,*)'Error in xmcsubt: cstlow,smallptupp',
     &           cstlow,smallptupp
            stop
         endif
      endif
      do i=1,nexternal
         IDUP_H(i)=IDUP_D(nFKSprocess,i,1)
         MOTHUP_H(1,i)=MOTHUP_D(nFKSprocess,1,i,1)
         MOTHUP_H(2,i)=MOTHUP_D(nFKSprocess,2,i,1)
      enddo
c     Fill selected color configuration into jpart array. 
      call fill_icolor_H(born_flow_picked,jpart,.false.)
      do i=1,nexternal
         ICOLUP_H(1,i)=jpart(4,i)
         ICOLUP_H(2,i)=jpart(5,i)
      enddo
c     
      call clear_HEPEUP_event()
      
c     Boost H-event momenta to lab frame before passing to pythia
      chy=cosh(ybst_til_tolab)
      shy=sinh(ybst_til_tolab)
      chymo=chy-1d0
      do i=1,nexternal
         call boostwdir2(chy,shy,chymo,xdir,p(0,i),p_lab(0,i))
      enddo
c     
      dummy_wgt=1d0
      call fill_HEPEUP_event(p_lab, dummy_wgt, nexternal, idup_h,
     &     istup_local, mothup_h, icolup_h, spinup_local)
      xscales_PY=-1d0
      xmasses_PY=-1d0
      dzones_PY=.true.
      if (is_pythia_active.eq.0) then
c     Fill masses
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
         if ( abs(idIn1) .lt. 10 .or. idIn1 .eq. 21) idIn1=2212
         if ( abs(idIn2) .lt. 10 .or. idIn2 .eq. 21) idIn2=2212
         call pythia_init_default(idIn1, idIn2, idOut, masses_to_MC)
      endif
      call pythia_setevent()
      call pythia_next()
      call pythia_get_stopping_info(xscales_PY,xmasses_PY)
      call pythia_get_dead_zones(dzones_PY)
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
c     incoming momenta should always be particle 1 and 2.
               if (j.ne.i) cycle
            elseif (j.le.nincoming) then
               cycle
            endif
            if (idup_in(i).eq.idup_s(j)) then
c     found the same particle ID. Check that colour is okay. 
               if (all(icolup_in(1:2,i).eq.icolup_s(1:2,j))) then
                  exit          ! Agreement found.
               endif
            endif
         enddo
         if (j.gt.nexternal-1) then
c     went all the way through the 2nd do-loop without finding the corresponding particle.
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

c     After the calls above, we have
c     xscales_PY(i,j)=t_ij
c     with t_ij == scale(Pythia)_{emitter,recoiler}, and the particle being
c     emitted equal to the FKS parton. Although both emitter and recoiler
c     are Born-level quantities, their labellings follow the real-process
c     conventions. Thus, in the matrix xscales_PY(i,j) one has 1<=i,j<=nexternal, 
c     with xscales_PY(i_fks,*)=xscales_PY(*,i_fks)=-1.
c     The same labelling conventions apply to xmasses_PY(i,j) (which is the
c     dipole mass associated with the colour line that connects i and j)
c     and dzones_PY(i,j) (which is the dead zone relevant to the emission from
c     parton i colour-connected with recoiler j).
c     
c     Since any the pair of indices (i,j) associated with sensible entries
c     in the arrays returned by Pythia is in one-to-one correspondence with
c     Born-level quantities, it is convenient to define relabelled copies of
c     such arrays (which we call Sevent_stopping_scales, xmasses_nbody, and
c     dzones_nbody), for which 1<=i,j<=nexternal-1
c     

      do i=1,nexternal
         if(i.eq.i_fks)cycle
         do j=1,nexternal
            if(j.eq.i_fks)cycle
            Sevent_stopping_scales(iRtoB(i),iRtoB(j))=xscales_PY(i,j)
c     In pythia the dipole masses can be arbitary large since the clustering
c     does not know exactly all the phase-space boundaries. Use min() to put
c     a cap on this (i.e., equal to the largest allowed value in pysudakov()
c     tables).
            xmasses_nbody(iRtoB(i),iRtoB(j))=min(xmasses_PY(i,j),cxmupp)
            dzones_nbody(iRtoB(i),iRtoB(j))=dzones_PY(i,j)
         enddo
      enddo
c     Checks
      if(any(Sevent_stopping_scales(1:nexternal-1,1:nexternal-1)*
     &     xmasses_nbody(1:nexternal-1,1:nexternal-1).lt.0d0)) then
         do i=1,nexternal-1
            do j=1,nexternal-1
               write(*,*)'Error in xmcsubt: xscales, xmasses',
     &              i,j,Sevent_stopping_scales(i,j),xmasses_nbody(i,j)
            enddo
         enddo
         stop
      endif

!     Since pythia simply does a one-branch cluster, it does not check if
!     the stopping scale (in Sevent_stopping_scales) is smaller than the
!     starting scale (as determined by MG5_aMC in Sevent_starting_scales). If this
!     is the case, put the event in the dead-zone.
      do i=1,nexternal-1
         do j=1,nexternal-1
            if (i.eq.j) cycle
            if (.not. dzones_nbody(i,j)) then
               if ( Sevent_stopping_scales(i,j).gt.
     $              Sevent_starting_scales(i,j)) then
                  dzones_nbody(i,j)=.true.
               endif
            endif
         enddo
      enddo
      call get_Hevent_starting_scales(Sevent_stopping_scales
     $     ,dzones_nbody,p,Hevent_starting_scales)
      
c     
c     force IF colour connection to have II scale
c     if a sensible II scale exists
      if(force_II_connection)then
         do i=1,2
            do j=3,nexternal
               if(valid_dipole_n1(i,j) .and. valid_dipole_n1(i,3-i))then
                  Hevent_starting_scales(i,j) =
     $                 Hevent_starting_scales(i,3-i)
               else
                  continue
c     if no other available colour connection, we keep the IF scale
c     rather than calculating some new kinematic variable e.g. pT
               endif
            enddo
         enddo
      endif

      
ccccccccccccccccccccc
c     
c     *** WARNING ***
c     
c     Pythia resets the scale for FI and FF to the min between the scale
c     t_ij we give it and p_i.p_j/2.  Should we implement this
c     minimisation here as well? (We do not do this at thee moment. For
c     H events this implementation should be needed only for i_fks and
c     j_fks, as only in that case we (over)write their scales ourselves
c     (in the set_Hevent_starting_scales() above), but could be applied to all
c     FI and FF connections.
c     
ccccccccccccccccccccc
c     

!     overwrite the emsca_H() (for this iFKS and ifold_counter) with the
!     actual stopping-scales defined here.
      emsca_H(nFKSprocess,ifold_counter,1:ndelH,1:ndelH)=
     &     Hevent_starting_scales(1:nexternal,1:nexternal)

      
c     Computation of Delta = wgt_sudakov as the product of Sudakovs between
c     Sevent_starting_scales and Sevent_stopping_scales.  For initial-state legs, Delta
c     contains a PDF ratio with S-event Bjorken fraction and
c     Sevent_starting_scales, Sevent_stopping_scales scales, see also formula (5.62) in
c     Ellis-Stirling-Webber


!     we are here
!     
!     1. loop over dipoles and find the (up to) 2 contributing. This sets
!     also the types and everything. Including scales.
!     2. loop over the (up to) 2 contributing, and compute the sudakovs.
!     
!     Paper: eq.3.14 (times the PDF factor in 3.32) defines what to compute
!     for each QCD particle in the n-body process. This is updated to
!     3.31 for quarks, and 3.34 for gluons. (Check the curly brackets in
!     3.14 & 3.34).  First term in 3.34 is equal to
!     glfact(1)*Deltarat(1,1)*Deltarat(1,2)*pdffactor(1) in the notation
!     of the code below (and equivalently for the 2nd term). 3.37 is
!     wgt_sudakov.
!     

!     loop over particles ('a' in eq.3.14)
      do i=1,nexternal-1
!     for each particle, find the connections (beta in eq.3.14)
         n_connect(i)=0
         do j=1,nexternal-1
            if (.not.valid_dipole_n(i,j,born_flow_picked)) cycle
            if (dzones_nbody(i,j)) cycle
!     found a connection; determine the starting and stopping scales of
!     this connection (eq.3.31 & 3.34). Check that it is in the
!     livezone, and overwrite it such that it gives a value that is not
!     too large or too small (eq.3.33)
            if (Sevent_starting_scales(i,j).lt.Sevent_stopping_scales(i
     $           ,j)) cycle
            n_connect(i)=n_connect(i)+1
            i_connect(n_connect(i),i)=j
            mu_ij(n_connect(i),i)=max(min(Sevent_starting_scales(i,j)
     $           ,cstupp),smallptupp)
            t_ij(n_connect(i),i)=min(Sevent_stopping_scales(i,j)
     $           ,cstupp)
         enddo
         if (n_connect(i).eq.1 .and. idup_s(i).eq.21) then
            if (isspecial(born_flow_picked)) then
!     This is the ISSPECIAL case. Add one more (identical) connection.
               n_connect(i)=n_connect(i)+1
               i_connect(n_connect(i),i)=i_connect(n_connect(i)-1,i)
               mu_ij(n_connect(i),i)=mu_ij(n_connect(i)-1,i)
               t_ij(n_connect(i),i)=t_ij(n_connect(i)-1,i)
            else
!     just continue, since one of the two gluon connections could be in the dead-zone.
!     TODO : FIXTHIS --> what to do if this happens and isspecial is true???
!     Also think about the check below in the next do-loop...
               continue
c$$$  write (*,*) 'A gluon with only one connection, but '/
c$$$  $              /'the born_flow_picked is not special.',i
c$$$  stop 1
            endif
         endif
      enddo
      
      wgt_sudakov=1d0
! loop over 'k' in eq.3.31 and 3.34
      do i=1,nexternal-1
         if (n_connect(i).eq.0) cycle ! no colour connection for particle 'i'
c$$$  if(  (n_connect(i).ne.1 .and. abs(idup_s(i)).le.6) .or.
c$$$  &        (n_connect(i).ne.2 .and. idup_s(i).eq.21)) then
c$$$  write (*,*) 'n_connect should be 1 for quarks and 2 '/
c$$$  $           /'for gluons',n_connect(i),i,idup_s(i)
c$$$  stop 1
c$$$  endif

         do out_con=1,n_connect(i) ! loop over two lines of 3.34
!     compute g1 and g2 (FIX FOR SAFETY MEASURES)
            if (n_connect(i).eq.1) then
               gl(out_con)=1d0
            else
               isudtype=setSudType(i,i_connect(out_con,i))
               deltanum=pysudakov_safe(t_ij(out_con,i),xmasses_nbody(i
     $              ,i_connect(out_con,i)),idup_s(i),isudtype,mcmass)
               deltaden=pysudakov_safe(mu_ij(out_con,i),xmasses_nbody(i
     $              ,i_connect(out_con,i)),idup_s(i),isudtype,mcmass)
               gl(out_con)=gl_safe(deltanum,deltaden)
            endif
!     compute F_k
            if(i.le.nincoming)then
!     The correct thing to do here (if we follow the paper) would be to have
!     a separate wgt_sudakov for each flavour configuration, since the PDF
!     ratio would be different for each of them. This is very tricky in the
!     current code setup, since at this point all flavour configurations are
!     always summed together. Therefore, we use an approximation, where we
!     take a (weighted) average of PDF ratios. We take as weights the PDF
!     used in the Born, which is (roughly) equal to the PDF computed at the
!     starting scales, which is pdfden. We can write this weighted average
!     as
!     Fk(out_con) = weighted_average(ratio_1, ..., ratio_n)
!     = (ratio_1*pdfden_1+...+ratio_n*pdfden_n)/sum(pdfden_1, ..., pdf_den_n)
!     = sum(pdfnum_1, ..., pdfnum_n)/sum(pdfden_1, ..., pdfden_n)
!     
               LP=SIGN(1,LPP(i))
               pdfnum=0d0
               pdfden=0d0
               do ip=1,iproc_born
                  id=get_parton_id(idup(i,ip),lp)
                  pdfnum=pdfnum+pdg2pdf(abs(lpp(i)),id,LP,xbjrk_cnt(i,0)
     $                 ,t_ij(out_con,i))
                  pdfden=pdfden+pdg2pdf(abs(lpp(i)),id,LP,xbjrk_cnt(i,0)
     $                 ,mu_ij(out_con,i))
               enddo
            else
               pdfnum=1d0
               pdfden=1d0
            endif
            if (pdfden.eq.0d0) then
! this should be extremely rare, but can happen if the
! scale is just right
               pdfden=1d-99
            endif
            Fk(out_con)=pdfnum/pdfden
!     compute delta
            do in_con=1,n_connect(i) ! loop over gamma in 3.34
               isudtype=setSudType(i,i_connect(in_con,i))
               deltanum=pysudakov_safe(t_ij(out_con,i)
     $              ,xmasses_nbody(i,i_connect(in_con,i)),idup_s(i)
     $              ,isudtype,mcmass)
               deltaden=pysudakov_safe(mu_ij(in_con,i)
     $              ,xmasses_nbody(i,i_connect(in_con,i)),idup_s(i)
     $              ,isudtype,mcmass)
               if (deltaden.eq.0d0) then
                  if (deltanum.ne.0d0) then
                     write (*,*) 'Denominator is zero in Sudakov'
                     write (*,*) deltanum,deltaden
                     write (*,*) t_ij(out_con,i),mu_ij(in_con,i),in_con
     $                    ,i,i_connect(in_con,i),xmasses_nbody(i
     $                    ,i_connect(in_con,i)),idup_s(i),isudtype
                     stop 1
                  endif
                  delta(out_con,in_con)=0d0
               else
                  delta(out_con,in_con)=deltanum/deltaden
               endif
            enddo
         enddo
!     multiply to get 3.34
         PIk=0d0
         do out_con=1,n_connect(i) ! loop over two lines of 3.34
            PIk=PIk + gl(out_con)/sum(gl(1:n_connect(i)))*Fk(out_con)
     $           *product(delta(out_con,1:n_connect(i)))
         enddo
! take min(max()) since it can be set between zero and one at
! the accuracy we are working, and interpret it as a
! probability.
         wgt_sudakov=wgt_sudakov * min(max(PIk,0d0),1d0)
      enddo

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

      return
      end

      double precision function gl_safe(num,den)
      implicit none
      double precision ratio,num,den
      if (den.eq.0d0) then
         gl_safe=0.d0
         return
      else
         ratio=num/den
      endif
      if(ratio.le.1d-20)then
         gl_safe=1.d8
      elseif(ratio.ge.1.d0)then
         gl_safe=0.d0
      else
         gl_safe=-2d0*log(ratio)
      endif
      end
      
      double precision function pysudakov_safe(scale,mass,id,type
     $     ,mcmass)
      implicit none
      double precision scale,mass,pysudakov
      double precision mcmass(21)
      integer id,type
      real*8 smallptlow,smallptupp,get_to_zero
      parameter (smallptlow=0.5d0)
      parameter (smallptupp=1.01d0)
      if(scale.lt.0d0)then
         write (*,*) 'scale smaller than 0 in pysudakov_safe',scale
         stop 1
      elseif(scale.le.smallptlow)then
         pysudakov_safe=0.d0
      elseif( scale.gt.smallptlow .and.
     $        scale.le.smallptupp )then
         pysudakov_safe = pysudakov(smallptupp,mass,id,type,mcmass)
     $        *get_to_zero(scale,smallptlow,smallptupp)
      else
         pysudakov_safe=pysudakov(scale,mass,id,type,mcmass)
      endif
      end
      

      integer function get_parton_id(ipdg,lp)
      implicit none
      integer ipdg,id,lp
      if (ipdg.le.6) then       ! (anti-)quark 
         id=lp*ipdg
      elseif (ipdg.eq.21) then  ! gluon
         id=0
      elseif (ipdg.eq.22) then  ! photon
         id=7
      else
         write (*,*) 'unknown PDG for PDF',ipdg
         stop 1
      endif
      end

      integer function setSudType(i,j)
      implicit none
      integer i,j
      if(i.le.2.and.j.le.2)then
         setsudtype=1
      elseif(i.gt.2.and.j.gt.2)then
         setsudtype=2
      elseif(i.le.2.and.j.gt.2)then
c     For Pythia: IF is identical to II
         setsudtype=1
      elseif(i.gt.2.and.j.le.2)then
         setsudtype=4
      endif
      end

      subroutine get_Hevent_starting_scales(Sevent_stopping_scales
     $     ,dzones_nbody,p,Hevent_starting_scales)
! Fills the Hevent_starting_scales based on the S-event stopping scales. In the
! MC-picture, all scales for which i_fks and j_fks are emitter are set
! to a common scale 'pT'. In the ME-picture, a rather more strict
! relation between the dipoles is followed, and each dipole for which
! i_fks and j_fks are the emitter can get different values, based on the
! colour connections of the mother.
! In case we are in the deadzone, use a scale based on the dipole mass
! (using H-event kinematics) instead.
! WARNING: this subroutine does NOT enforce the scales for the IF
! dipoles to be overwritten by the II dipoles.
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      logical*1 dzones_nbody(nexternal-1,nexternal-1)
      double precision Sevent_stopping_scales(nexternal-1,nexternal-1)
     $     ,Hevent_starting_scales(nexternal,nexternal),p(0:3,nexternal)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i1,i2,ip,imother,i1bar,i2bar,i
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
            if (valid_dipole_n(imother,i2bar,born_flow_picked)) then
               if (.not.dzones_nbody(imother,i2bar))
     &              pT=min(pT,Sevent_stopping_scales(imother,i2bar))
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
            if (.not.valid_dipole_n1(i1,i2)) cycle
            ! Find the (i1bar,i2bar) S-event dipole corresponding to
            ! the (i1,i2) H-event dipole.
            if (i1.eq.i_fks .and. i2.eq.j_fks) then
               if (MCpicture) then
                  i1bar=-99
               else
                  i1bar=ipbar(imother)
                  i2bar=imother
               endif
            elseif (i1.eq.j_fks .and. i2.eq.i_fks) then
               if (MCpicture) then
                  i1bar=-99
               else
                  i1bar=imother
                  i2bar=ipbar(imother)
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
               if (.not.valid_dipole_n(i1bar,i2bar,born_flow_picked))
     $              then
                  write (*,*) 'Lines not color connected #2',
     $                 i1,i2,i1bar,i2bar
                  stop 1
               endif
               if (.not. dzones_nbody(i1bar,i2bar)) then
                  t(i1,i2)=Sevent_stopping_scales(i1bar,i2bar)
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
            if (.not.valid_dipole_n1(i1,i2)) cycle
            if (t(i1,i2).eq.-1d0) then
               write (*,*) 'ERROR, scale still equal to -1',i1,i2
     $              ,pTparton,i1bar,i2bar,pT
               do i=1,nexternal-1
                  write (*,*) Sevent_stopping_scales(i,1:nexternal-1)
               enddo
               stop 1
            endif
         enddo
      enddo
      Hevent_starting_scales(1:nexternal,1:nexternal)=t(1:nexternal
     $     ,1:nexternal)
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

      
      integer function ipbar(imother)
      ! ipbar is the colour connection of i_fks (if it exists and is not
      ! equal to the mother). Otherwise it is the colour connection of
      ! j_fks. The latter only happens when i_fks is a quark and j_fks
      ! is an (incoming gluon).
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      integer imother
      integer ip
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      ipbar=0
      do ip=1,nexternal
         if(ip.eq.i_fks)cycle
         if (valid_dipole_n1(ip,i_fks) .and. iRtoB(ip).ne.imother) then
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
            if (valid_dipole_n1(ip,j_fks) .and. iRtoB(ip).ne.imother)
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


      subroutine get_mbar(p,xi_i_fks,y_ij_fks,p_born,ileg,bornbars
     $     ,bornbarstilde)
c Computes barred amplitudes (bornbars) squared according
c to Odagiri's prescription (hep-ph/9806531).
c Computes barred azimuthal amplitudes (bornbarstilde) with
c the same method 
      implicit none

      include "genps.inc"
      include "nexternal.inc"
      include "born_nhel.inc"
      include "orders.inc"
      include "nFKSconfigs.inc"
      
      double precision p(0:3,nexternal),p_born(0:3,nexternal-1)
      double precision xi_i_fks,y_ij_fks,bornbars(max_bcol,nsplitorders)
     $     ,bornbarstilde(max_bcol,nsplitorders)

      double precision zero
      parameter (zero=0.d0)
      double complex czero
      parameter (czero=dcmplx(0d0,0d0))
      double precision p_born_rot(0:3,nexternal-1)

      integer imother_fks,ileg

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
      double precision iden_comp,iden_comp_FKS(fks_configs)
      common /c_iden_comp/iden_comp,iden_comp_FKS

c Particle types (=color) of i_fks, j_fks and fks_mother
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
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
               write(*,*) 'ERROR in get_mbar', iord
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
               if (xi_i_fks.lt.1d-8) then
                  pi(i)=p_i_fks_ev(i)
               else
                  pi(i)=p(i,i_fks)
               endif
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
                  if (xi_i_fks.lt.1d-8) then
                     pi(i)=p_i_fks_ev(i)
                  else
                     pi(i)=p(i,i_fks)
                  endif
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
         write(*,*)'unknown ileg in get_mbar',ileg
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




c Monte Carlo functions
c
c The invariants given in input to these routines follow FNR conventions
c (i.e., are defined as (p+k)^2, NOT 2 p.k). 
c The invariants used inside these routines follow MNR conventions
c (i.e., are defined as -2p.k, NOT (p+k)^2)

c Herwig6

      double precision function zHW6(e0sq)
c     Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,e0sq,ss,betae0,beta,zeta,tbeta,get_zeta
      parameter (tiny=1d-5)
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            zHW6=1-(1-x)*(shat_n1*(1-yij)+4*e0sq*(1+yij))/(8*e0sq)
         elseif(1-yij.lt.tiny)then
            zHW6=x-(1-yij)*(1-x)*(shat_n1*x**2-4*e0sq)/(8*e0sq)
         else
            ss=1-(1+xuk/shat_n1)/(e0sq/xtk)
            if(ss.lt.0d0)goto 999
            zHW6=2*(e0sq/xtk)*(1-sqrt(ss))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            zHW6=1-(1-x)*(shat_n1*(1-yij)+4*e0sq*(1+yij))/(8*e0sq)
         elseif(1-yij.lt.tiny)then
            zHW6=x-(1-yij)*(1-x)*(shat_n1*x**2-4*e0sq)/(8*e0sq)
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
            zHW6=1+(1-x)*( shat_n1*(yij*betad-betas)/(4*e0sq*(1+betae0))-
     $           betae0*(xm12-xm22+shat_n1*(1+(1+yij)*betad-betas))/
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
            zHW6=1-(1-x)*( (shat_n1-xm12)*(1-yij)/(8*e0sq)+
     &                     shat_n1*(1+yij)/(2*(shat_n1-xm12)) )
         elseif(1-yij.lt.tiny)then
            zHW6=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yij)*(1-x)*(shat_n1*x
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



      double precision function xiHW6(e0sq,z)
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,e0sq,betae0,beta,z
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            xiHW6=2*shat_n1*(1-yij)/(shat_n1*(1-yij)+4*e0sq*(1+yij))
         elseif(1-yij.lt.tiny)then
            xiHW6=(1-yij)*shat_n1*x**2/(4*e0sq)
         else
            xiHW6=2*(1+xuk/(shat_n1*(1-z)))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            xiHW6=2*shat_n1*(1-yij)/(shat_n1*(1-yij)+4*e0sq*(1+yij))
         elseif(1-yij.lt.tiny)then
            xiHW6=(1-yij)*shat_n1*x**2/(4*e0sq)
         else
            xiHW6=2*(1+xtk/(shat_n1*(1-z)))
         endif
c
      elseif(ileg.eq.3)then
         if(e0sq.le.(w1+xm12))goto 999
         if(1-x.lt.tiny)then
            beta=1-xm12/shat_n1
            betae0=sqrt(1-xm12/e0sq)
            xiHW6=( shat_n1*(1+betae0)*betad*(xm12-xm22+shat_n1*(1
     $           +betad))*(yij*betad-betas) )/( -4*e0sq*betae0*(1+betae0)
     $           *(xm12-xm22+shat_n1*(1+(1+yij)*betad-betas))+(shat_n1
     $           *betad*(xm12-xm22+shat_n1*(1+betad))*(yij*betad-betas))
     $           )
         else
            xiHW6=w1/(2*z*(1-z)*e0sq)
         endif
c
      elseif(ileg.eq.4)then
         if(e0sq.le.w2)goto 999
         if(1-x.lt.tiny)then
            xiHW6=2*(shat_n1-xm12)**2*(1-yij)/( (shat_n1-xm12)**2*(1-yij)
     $           +4*e0sq*shat_n1*(1+yij) )
         elseif(1-yij.lt.tiny)then
            xiHW6=(shat_n1-xm12)**2*(1-yij)/(4*e0sq*shat_n1)
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
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,xi,tmp,e0sq,beta,betae0,zmo
     $     ,tbeta,eps,dw1dx,dw2dx,dw1dy,dw2dy
      parameter (tiny=1d-5)

      if(z.lt.0d0.or.xi.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         if(1-x.lt.tiny)then
            tmp=-2*shat_n1/(shat_n1*(1-yij)+4*(1+yij)*e0sq)
         elseif(1-yij.lt.tiny)then
            tmp=-shat_n1*x**2/(4*e0sq)
         else
            tmp=-shat_n1*(1-x)*z**3/(4*e0sq*(1-z)*(xi*(1-z)+z))
         endif
c
      elseif(ileg.eq.2)then
         if(1-x.lt.tiny)then
            tmp=-2*shat_n1/(shat_n1*(1-yij)+4*(1+yij)*e0sq)
         elseif(1-yij.lt.tiny)then
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
            tmp=( shat_n1*betae0*(1+betae0)*betad*(xm12-xm22+shat_n1*(1
     $           +betad)) )/( (-4*e0sq*(1+betae0)*(xm12-xm22+shat_n1*(1
     $           +betad*(1+yij)-betas)))+(xm12-xm22+shat_n1*(1+betad))
     $           *(xm12*(4+yij*betad-betas)-(xm22-shat_n1)*(yij*betad
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
            zmo=(shat_n1-xm12)*(1-yij)/(8*e0sq)+shat_n1*(1+yij)/(2
     $           *(shat_n1-xm12))
            tmp=-shat_n1/(4*e0sq*zmo)
         elseif(1-yij.lt.tiny)then
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

      double precision function zHWPP()
c     Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,get_zeta,zeta1,zeta2
      parameter (tiny=1d-5)
c
      if(ileg.eq.1)then
         zHWPP=1-(1-x)*(1+yij)/2d0
c
      elseif(ileg.eq.2)then
         zHWPP=1-(1-x)*(1+yij)/2d0
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            zHWPP=1-(1-x)*(1+yij)/(betad+betas)
         else
            zeta1=get_zeta(shat_n1,w1,w2,xm12,xm22)
            zHWPP=1-zeta1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zHWPP=1-(1-x)*(1+yij)*shat_n1/(2*(shat_n1-xm12))
         elseif(1-yij.lt.tiny)then
            zHWPP=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yij)*(1-x)*shat_n1
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



      double precision function xiHWPP(z)
c     Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision z,zHWPP,tiny
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c 
      if(ileg.eq.1)then
         xiHWPP=shat_n1*(1-yij)/(1+yij)
c
      elseif(ileg.eq.2)then
         xiHWPP=shat_n1*(1-yij)/(1+yij)
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            xiHWPP=-shat_n1*(betad+betas)*(yij*betad-betas)/(2*(1+yij))
         else
            xiHWPP=w1/(z*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiHWPP=(1-yij)*(shat_n1-xm12)**2/(shat_n1*(1+yij))
         elseif(1-yij.lt.tiny)then
            xiHWPP=(1-yij)*(shat_n1-xm12)**2/(2*shat_n1)
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



      double precision function xjacHWPP(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision z,zHWPP,tmp,eps,beta,dw1dx,dw2dx,dw1dy,dw2dy,tiny
      parameter (tiny=1d-5)

      tmp=0d0
      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         tmp=-shat_n1/(1+yij)
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1/(1+yij)
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            tmp=-shat_n1*(betad+betas)/(2*(1+yij))
         else
            eps=1-(xm12-xm22)/(shat_n1-w1)
            beta=sqrt(eps**2-4*shat_n1*xm22/(shat_n1-w1)**2)
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=-(dw1dy*dw2dx-dw1dx*dw2dy)/(z*(1-z))/((shat_n1-w1)*beta)
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=-(shat_n1-xm12)/(1+yij)
         elseif(1-yij.lt.tiny)then
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

      double precision function zPY6Q()
c Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
      parameter(tiny=1d-5)
c
      if(ileg.eq.1)then
         zPY6Q=x
c
      elseif(ileg.eq.2)then
         zPY6Q=x
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            zPY6Q=1-(2*xm12)/(shat_n1*betas*(betas-betad*yij))
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
         elseif(1-yij.lt.tiny)then
            zPY6Q=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yij)*(1-x)**2
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



      double precision function xiPY6Q()
c     Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
      parameter(tiny=1d-5)
c
      if(ileg.eq.1)then
         xiPY6Q=shat_n1*(1-x)*(1-yij)/2
c
      elseif(ileg.eq.2)then
         xiPY6Q=shat_n1*(1-x)*(1-yij)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            xiPY6Q=shat_n1*(1-x)*(betas-betad*yij)/2
         else
            xiPY6Q=w1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiPY6Q=(1-yij)*(1-x)*(shat_n1-xm12)/2
         elseif(1-yij.lt.tiny)then
            xiPY6Q=(1-yij)*(1-x)*(shat_n1*x-xm12)/2
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



      double precision function xjacPY6Q(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c     variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,zPY6Q,z,tmp,dw1dx,dw1dy,dw2dx,dw2dy
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         tmp=-shat_n1*(1-x)/2
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1*(1-x)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            tmp=xm12*betad/betas/(betas-betad*yij)
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=shat_n1*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)*dw1dy
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=shat_n1*(1-x)/2
         elseif(1-yij.lt.tiny)then
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

      double precision function zPY6PT()
c Shower energy variable
      use kinematics_module
      implicit none
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



      double precision function xiPY6PT()
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none

      if(ileg.eq.1)then
         xiPY6PT=shat_n1*(1-x)**2*(1-yij)/2
c
      elseif(ileg.eq.2)then
         xiPY6PT=shat_n1*(1-x)**2*(1-yij)/2
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

      double precision function zPY8()
c Shower energy variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny
      parameter(tiny=1d-5)
c
      if(ileg.eq.1)then
         zPY8=x
c
      elseif(ileg.eq.2)then
         zPY8=x
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            zPY8=1-(2*xm12)/(shat_n1*betas*(betas-betad*yij))
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
         elseif(1-yij.lt.tiny)then
            zPY8=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yij)*(1-x)**2*shat_n1
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



      double precision function xiPY8(z)
c Shower evolution variable
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,zPY8,z0
      parameter(tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         xiPY8=shat_n1*(1-x)**2*(1-yij)/2
c
      elseif(ileg.eq.2)then
         xiPY8=shat_n1*(1-x)**2*(1-yij)/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            z0=1-(2*xm12)/(shat_n1*betas*(betas-betad*yij))
            xiPY8=shat_n1*(1-x)*(betas-betad*yij)*z0*(1-z0)/2
         else
            xiPY8=z*(1-z)*w1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiPY8=shat_n1*(1-x)**2*(1-yij)/2
         elseif(1-yij.lt.tiny)then
            xiPY8=shat_n1*(1-x)**2*(1-yij)*(shat_n1*x-xm12)**2/(2
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



      double precision function xjacPY8(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower 
c variables, and x and y are FKS variables
      use process_module
      use kinematics_module
      implicit none
      double precision tiny,z,z0,zPY8,dw1dx,dw1dy,dw2dx,dw2dy,tmp

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.2)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            z0=1-(2*xm12)/(shat_n1*betas*(betas-betad*yij))
            tmp=xm12*betad/betas/(betas-betad*yij)*z0*(1-z0)
         else
            call dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
            tmp=shat_n1*(xm12+w1)/w1/(shat_n1+w1+xm12-xm22)*dw1dy*z*(1-z)
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            tmp=shat_n1**2*(1-x)**2/( 2*(shat_n1-xm12) )
         elseif(1-yij.le.tiny)then
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
         afun=sqrt(s)*(1-x)*(xm12-xm22+s*x)*yij
         bfun=s*( (1+x)**2*(xm12**2+(xm22-s*x)**2-
     &        xm12*(2*xm22+s*(1+x**2)))+xm12*s*(1-x**2)**2*yij**2 )
         cfun=s*(-(1+x)**2+(1-x)**2*yij**2)
         dadx=sqrt(s)*yij*(xm22-xm12+s*(1-2*x))
         dady=sqrt(s)*(1-x)*(xm12-xm22+s*x)
         dbdx=2*s*(1+x)*( xm12**2+(xm22-s*x)*(xm22-s*(1+2*x))
     &        -xm12*(2*xm22+s*(1+x+2*(x**2)+2*(1-x)*x*(yij**2))) )
         dbdy=2*xm12*(s**2)*((1-x**2)**2)*yij
         dcdx=-2*s*(1+x+(yij**2)*(1-x))
         dcdy=2*s*((1-x)**2)*yij
c Determine correct sign
         mom_fks_sister_p=(afun+sqrt(bfun))/cfun
         mom_fks_sister_m=(afun-sqrt(bfun))/cfun
         diff_p=abs(mom_fks_sister_p-veckn_ev)
         diff_m=abs(mom_fks_sister_m-veckn_ev)
         if(min(diff_p,diff_m)/max(abs(veckn_ev),1d0).ge.1d-3)then
            write(*,*)'Fatal error 1 in dinvariants_dFKS'
            write(*,*)mom_fks_sister_p,mom_fks_sister_m,veckn_ev
            write (*,*) 1d0-x,yij,sqrt(xm12),sqrt(xm22)
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
         dw1dx=sqrt(s)*( yij*mom_fks_sister-en_fks_sister+(1-x)*
     &                   (mom_fks_sister/en_fks_sister-yij)*dmomfkssisdx )
         dw1dy=-sqrt(s)*(1-x)*( mom_fks_sister+
     &                   (yij-mom_fks_sister/en_fks_sister)*dmomfkssisdy )
         dw2dx=-dw1dx-s
         dw2dy=-dw1dy
c
      elseif(ileg.eq.4)then
c$$$         dq1cdx=-(1-yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
c$$$         dq2qdx=-(1+yi)*(s*(1+yj)+xm12*(1-yj))/(1+yj+x*(1-yj))**2
c$$$         dw1dx=dq1cdx+dq2qdx
         dw1dx=-2*(s*(1+yij)+xm12*(1-yij))/(1+yij+x*(1-yij))**2
         dw2dx=(1-yij)*(s*(1+yij-x*(2*(1+yij)+x*(1-yij)))+2*xm12)/(1+yij+x*(1-yij))**2
c$$$         dq1cdy=(1-x)*(1-yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
c$$$         dq2qdy=(1-x)*(1+yi)*(s*x-xm12)/(1+yj+x*(1-yj))**2
c$$$         dw1dy=dq1cdy+dq2qdy
         dw1dy=(1-x)*2*(s*x-xm12)/(1+yij+x*(1-yij))**2
         dw2dy=-2*(1-x)*(s*x-xm12)/(1+yij+x*(1-yij))**2
c
      else
         write(*,*)'Error in dinvariants_dFKS: unknown ileg',ileg
         stop
      endif

      return
      end



      subroutine get_dead_zone(z,xi,p_born,qMC,ipartner,lzone,PY6PTweight)
      use process_module
      use kinematics_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      integer ipartner,i
      double precision z,xi,qMC,PY6PTweight
      logical lzone

      double precision p_born(0:3,nexternal-1)
      double precision upscale2,xmp2,xmm2,xmr2,ww,Q2,lambda
     $     ,e0sq,beta,ycc,mdip,mdip_g,zp1,zm1,zp2,zm2,zp3,zm3,get_angle
     $     ,theta2p,max_scale
      external get_angle

      double precision ppartner(0:3),pfather(0:3)

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
      max_scale=shower_scale_nbody_max(ipartner,fksfather)
      ! TODO: fix max_scale for tests. maybe ipartner?
      do i=0,3
         pfather(i)=p_born(i,fksfather) ! father momentum (Born level)
         ppartner(i)=p_born(i,ipartner) ! partner momentum (Born level)
      enddo
      e0sq=dot(ppartner,pfather)
      theta2p=get_angle(ppartner,pfather)**2
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
         Q2=sumdot(pfather,ppartner,1d0) ! parent dipole mass squared (Born level)
         xmp2=dot(ppartner,ppartner)     ! mass squared of the partner
         if (shower_mc_mod(1:8).eq.'HERWIGPP')
     &        lambda=sqrt((Q2+xmm2-xmp2)**2-4*Q2*xmm2)
         if (shower_mc_mod(1:8).eq.'PYTHIA6Q') then
            beta=sqrt(1-4*shat_n1*(xmm2+ww)/(shat_n1-xmr2+xmm2+ww)**2)
            zp1=(1+(xmm2+beta*ww)/(xmm2+ww))/2
            zm1=(1+(xmm2-beta*ww)/(xmm2+ww))/2
         endif
         if (shower_mc_mod(1:7).eq.'PYTHIA8') then
            beta=sqrt(1-4*shat_n1*(xmm2+ww)/(shat_n1-xmr2+xmm2+ww)**2)
            mdip  =sqrt((sqrt(xmp2+xmm2+2*e0sq)-sqrt(xmp2))**2-xmm2)
            ! mdip corresponds to sqrt(dip.m2DipCorr)
            ! (around line 2305 in Pythia TimeShower.cc)
            mdip_g=sqrt((sqrt(shat_n1) -sqrt(xmr2))**2-xmm2)
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
     &         4*xi/shat_n1/(1-z).ge.theta2p)lzone=.false.
         elseif(ileg.gt.2)then
            if(mstj50.eq.2.and.ipartner.le.2.and.
c around line 71636 of pythia6428: V(IEP(1),5)=virtuality, P(IM,4)=sqrt(s)
     &           max(z/(1-z),(1-z)/z)*4*(xi+xmm2)/shat_n1.ge.theta2p)
     &           lzone=.false.
            if(z.gt.zp1.or.z.lt.zm1)lzone=.false.
         endif
c
      elseif(shower_mc_mod(1:9).eq.'PYTHIA6PT')then
         ycc=1-parp67*x/(1-x)**2/2
         if(mstp67.eq.1.and.yij.lt.ycc)lzone=.false.
         if(mstp67.eq.2) PY6PTweight=min(1d0,(1-ycc)/(1-yij))
c
      elseif(shower_mc_mod(1:7).eq.'PYTHIA8')then
         if(ileg.le.2.and.z.gt.1-sqrt(xi/z/shat_n1)*
     &      (sqrt(1+xi/4/z/shat_n1)-sqrt(xi/4/z/shat_n1)))lzone=.false.
         if(ileg.gt.2)then
   ! Pythia as well in the global recoil scheme, constrains radiation to be
   ! softer than local dipole mass divided by two
            max_scale=min(max_scale,mdip/2,mdip_g/2)
            if(z.gt.min(zp2,zp3).or.z.lt.max(zm2,zm3))lzone=.false.
         endif

      endif

! If the relative pT of the splitting is larger then the maximum shower
! scale, we are in the deadzone
      if (qMC.gt.max_scale) lzone=.false.

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

