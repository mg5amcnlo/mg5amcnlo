      module mc_counterterms
c Explicit interfaces for MC subtraction and Delta matching. Only the
c history driver and shared history controls are public; numerical helpers
c and shower invariants stay local.
      use process_module, only: next_n1,shower_mc_mod
      use fks_phase_space_helpers, only: dot,rho,sumdot
      implicit none
      private
      public :: set_QCD_flows,compute_MCsubtraction_kl,compute_delta,
     $     bogus_probne_fun
      public :: prepare_mc_kinematics,fill_father_and_ileg,get_qMC,
     $     mc_shower_scale_mass,fksfather,gfactsf,gfactcl,gfactazi

c Shower state for the active FKS history. General momentum utilities
c remain in fks_phase_space_helpers, independent of subtraction and matching.
      integer :: ileg,fksfather
      double precision :: xm12,xm22,xtk,xuk,xq1q,xq2q,w1,w2,yij,x,
     $     xij,betad,betas,kn,knbar,kn0,shat_n1,gfactsf,gfactcl,gfactazi
      double precision :: xp1(0:3),xp2(0:3),xk1(0:3),xk2(0:3),
     $     xk3(0:3),pp_rec(0:3),jmass
      double precision,parameter :: tiny=1d-5

      include 'nexternal.inc'
      include 'born_nhel.inc'
c Flow setup also retains double-gluon flags between evaluations.
c COMMON blocks retained in the procedures are shared with the surrounding
c FKS/native-Born machinery or the external Sudakov tables.
      logical :: isspecial(max_bcol)=.false.

      type mc_kernel_limits
         logical :: collinear=.false.
         logical :: nonsoft=.false.
      end type mc_kernel_limits

c     Complete live colour-connection data for one Born leg in Delta.
      type :: delta_connections
         integer :: count=0,partner(2)=0,sudakov_type(2)=0
         double precision :: start(2)=0d0,stop(2)=0d0,mass(2)=0d0
      end type delta_connections

      contains

c MC counterterms and matching support.
c   Flow setup: set_QCD_flows and find_color_connectors.
c   History evaluation: compute_MCsubtraction_kl prepares Born data,
c     then xmcsubt_connection evaluates each distinct colour connection.
c   Delta weights: compute_delta prepares stopping scales and live
c     connections before evaluating the Sudakov and PDF factors.
c   Born amplitudes: get_mbar uses mc_born_azimuth_phase for spin terms.
c   Shower maps and analytic kernels are private module procedures.

      subroutine set_QCD_flows
c Refresh leading Born flows and double-gluon flags. Partner lists are
c scratch data used only to validate these colour connections.
      implicit none
      include "genps.inc"
      include 'nFKSconfigs.inc'
c Nexternal is the number of legs (initial and final) at NLO, while max_bcol
c is the number of color flows at Born level
      integer i,j,k,l,k0,mothercol(2),i1(2)
      integer ipartners(0:nexternal-1),colorflow(nexternal-1,0:max_bcol)
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fksfather
      logical notagluon,found
      integer nglu,nsngl
      logical spec_case

      include 'orders.inc'
      logical split_type(nsplitorders)
      common /c_split_type/split_type

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
      ipartners=0
      colorflow=0
      notagluon=.true.

c This prepares QCD partners and the allowed Born colour flows.
c QED connection selection is not supported by the matching driver.

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
c Born colour sampling is also needed for LO-only and QED sectors.
c Refresh the allowed flows even when no QCD splitting is active.
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

      if (split_type(qcd_pos)) then
        ! identify the color partners
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
                              write(*,*)'Error #1 in set_QCD_flows'
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
                     if(k0.le.0.or.k0.gt.nexternal-1)then
                        write(*,*)'Error #2 in set_QCD_flows'
                        write(*,*)i,j,l,k0
                        stop
                     endif
                     if(ipartners(k0).ne.j)then
                        write(*,*)'Error #2 in set_QCD_flows'
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
                     else
c Special case: father and ipartners(k0) are both gluons, connected
c by colour AND anticolour. Keep its single flow entry and mark it
c so that the two connections can be restored when evaluating kernels.
                         if( notagluon .or.
     &                       ICOLUP(i1(1),j,i).eq.0 .or.
     &                       ICOLUP(i1(2),j,i).eq.0 )then
                            write(*,*)'Error #3 in set_QCD_flows'
                            write(*,*)i,j,l,k0,i1(1),i1(2)
                            stop
                         endif
                         isspecial(i)=.true.
                     endif
                  endif
               enddo
            endif
         enddo
         if( ((nglu+nsngl).gt.(nexternal-2)) .or.
     #       (isspecial(i).and.(nglu+nsngl).ne.(nexternal-2)) )then
           write(*,*)'Error #4 in set_QCD_flows'
           write(*,*)isspecial(i),nglu,nsngl
           stop
          endif
        enddo

      else if (split_type(qed_pos)) then
        ! do nothing, the partner will be assigned at run-time
        ! (it is kinematics-dependent)
        continue
      endif
      call check_QCD_flows(notagluon,ipartners,colorflow)
      return
      end subroutine set_QCD_flows


      subroutine check_QCD_flows(notagluon,ipartners,colorflow)
      implicit none
      integer, intent(in) :: ipartners(0:nexternal-1)
      integer, intent(in) :: colorflow(nexternal-1,0:max_bcol)
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i,j,ipart,iflow,ntot,nexpected
      integer ithere(max(nexternal-1,max_bcol))
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fksfather
      logical, intent(in) :: notagluon

      include 'orders.inc'
      logical split_type(nsplitorders)
      common /c_split_type/split_type

      logical is_leading_cflow(max_bcol)
      integer num_leading_cflows
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows
c
      fksfather=min(i_fks,j_fks)
      if(ipartners(0).lt.0.or.ipartners(0).gt.nexternal-1)then
        write(*,*)'Error #1 in check_QCD_flows',ipartners(0)
        stop
      endif
c
      if (split_type(QCD_pos)) then
      ! these tests only apply for QCD-type splittings
        do i=1,ipartners(0)
          ipart=ipartners(i)
          if(ipart.le.0.or.ipart.gt.nexternal-1)then
            write(*,*)'Invalid partner in check_QCD_flows',i,ipart
            stop 1
          endif
          if( ipart.eq.fksfather .or.
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
            if(iflow.le.0.or.iflow.gt.max_bcol)then
              write(*,*)'Invalid flow in check_QCD_flows',i,j,iflow
              stop 1
            endif
            ithere(iflow)=ithere(iflow)-1
            if(ithere(iflow).lt.0)then
              write(*,*)'Error #5 in check_QCD_flows',i,j,iflow
              stop
            endif
          enddo
c
        enddo
c
c Special double gluon connections are stored once in each flow.
c Count them per allowed flow; the first flow need not be leading.
        nexpected=0
        do iflow=1,max_bcol
          if (.not.is_leading_cflow(iflow)) cycle
          nexpected=nexpected+1
          if (.not.notagluon.and..not.isspecial(iflow))
     #         nexpected=nexpected+1
        enddo
        if(ntot.ne.nexpected)then
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
      end subroutine check_QCD_flows




c Evaluate one FKS history for the selected Born colour flow. For
c S events the caller uses the original history. For H events,
c repartition_MC_H sums complete native contributions, including the
c G replacement: Hhat_ij = S_ij sum_kl P_kl (S_kl R - M_kl).


      subroutine compute_MCsubtraction_kl(k_fks,l_fks,xi,y,p,p_cm,p_born
     $     ,include_gfun,z,n_connect,amp_split_xmcxsec)
      use fks_phase_space_data, only: veckn_ev,veckbarn_ev,xp0jfks
      use scale_module, only: born_flow_picked
      implicit none
      include 'orders.inc'
      integer k_fks,l_fks,kernel_index
      type(mc_kernel_limits) kernel_limits
      logical lzone(2)
      double precision p(0:3,nexternal),p_born(0:3,nexternal-1),xi,y
     $     ,mass,z(2),amp_split_xmcxsec(1:amp_split_size,2)
     $     ,p_cm(0:3,nexternal)
      double precision pmass(nexternal)
      common /to_mass/pmass
      integer n_connect,i_connect(2),iconnect
      logical include_gfun
      double precision g_damping,qMC,connection_damping(2)
      double precision born_weights(amp_split_size)
      double precision born_spin_weights(amp_split_size)
      intent(in) :: k_fks,l_fks,xi,y,p,p_cm,p_born,include_gfun
      intent(out) :: z,n_connect,amp_split_xmcxsec
      amp_split_xmcxsec=0d0
      z=0d0
      lzone=.false.
      connection_damping=0d0
      mass=pmass(l_fks)
      veckn_ev=rho(p_cm(0,l_fks))
      veckbarn_ev=rho(p_born(0,min(k_fks,l_fks)))
      xp0jfks=p_cm(0,l_fks)

      call prepare_mc_kinematics(p_cm,k_fks,l_fks,xi,y,mass
     $     ,include_gfun)
!     compute MC subtraction term for the 'kl' configuration

!     find to which particle(s) fksfather connects in the colour flow
      call find_color_connectors(born_flow_picked,fksfather,n_connect
     $     ,i_connect)

c Born amplitudes and the splitting order belong to the history and do
c not depend on its colour connection. Prepare them once per history.
      call prepare_MCsubtraction_born(p,xi,y,p_born,kernel_index,
     $     born_weights,born_spin_weights)
      kernel_limits=classify_mc_kernel_limits(xi,y)
      qMC=get_qMC(xi,y)

c Evaluate each distinct connection; keep both entries for a gluon
c connected twice to the same partner, including their multiplicity.
      do iconnect=1,n_connect
         if (iconnect.eq.2) then
            if (i_connect(2).eq.i_connect(1)) then
               lzone(2)=lzone(1)
               z(2)=z(1)
               connection_damping(2)=connection_damping(1)
               amp_split_xmcxsec(:,2)=amp_split_xmcxsec(:,1)
               cycle
            endif
         endif
         connection_damping(iconnect)=compute_damping_weight(
     $        i_connect(iconnect),xi,y)
         call xmcsubt_connection(p_born,i_connect(iconnect),qMC,
     $        connection_damping(iconnect),include_gfun,kernel_index,
     $        kernel_limits,born_weights,born_spin_weights,
     $        lzone(iconnect),z(iconnect),amp_split_xmcxsec(:,iconnect))
      enddo
      if (include_gfun) then
! The G replacement must vanish smoothly at the shower-scale boundary.
! Average over the equally probable colour connections, including those
! outside angular support: G must still supply the soft wide-angle limit.
! Apply this only after the raw kernels have used the original G factors.
! The returned (1-gfactsf) multiplies the same replacement in S and H.
         g_damping=0d0
         do iconnect=1,n_connect
            g_damping=g_damping+connection_damping(iconnect)
         enddo
         g_damping=g_damping/dble(n_connect)
         gfactsf=1d0-(1d0-gfactsf)*g_damping
      endif

      if (any(lzone(1:n_connect))) then
         amp_split_xmcxsec(1:amp_split_size,1:2)=amp_split_xmcxsec(
     $        1:amp_split_size,1:2)
     $        /(xi**2*(1d0-y)) ! re-instate 1/xi^2 and 1/(1-y); they
                               ! should not depend on 'kl', but rather
                               ! on 'ij'
      else
         amp_split_xmcxsec(1:amp_split_size,1:2)=0d0
      endif
      end subroutine compute_MCsubtraction_kl

      subroutine find_color_connectors(iflow,iparticle,n_connect
     $     ,i_connect)
      use process_module, only: next_n,valid_dipole_n
      implicit none
      include "genps.inc"
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include "born_leshouche.inc"
      integer iflow,iparticle,n_connect,i_connect(2),i
      intent(in) :: iflow,iparticle
      intent(out) :: n_connect,i_connect
      n_connect=0
      i_connect=0
      if (iflow.lt.1.or.iflow.gt.max_bcol.or.
     $    iparticle.lt.1.or.iparticle.gt.nexternal-1) then
         write(*,*) 'Invalid flow or particle in find_color_connectors',
     $        iflow,iparticle
         stop 1
      endif
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
      end subroutine find_color_connectors


      double precision function compute_damping_weight(cur_part
     $     ,xi_i_fks,y_ij_fks)
      use scale_module, only: shower_scale_nbody_min,
     $     shower_scale_nbody_max
      implicit none
      integer :: cur_part
      double precision :: xi_i_fks,y_ij_fks,smin,smax,qMC,ptresc
      smin=shower_scale_nbody_min(fksfather,cur_part)
      smax=shower_scale_nbody_max(fksfather,cur_part)
      qMC=get_qMC(xi_i_fks,y_ij_fks)
      ptresc=(qMC-smin)/(smax-smin)
      compute_damping_weight=1d0-emscafun(ptresc,1d0)
      end function compute_damping_weight



c Prepare the connection-independent Born amplitudes and select the
c single correction order supported by MC@NLO.
      subroutine prepare_MCsubtraction_born(p,xi,y,p_born,
     $     kernel_index,born_weights,born_spin_weights)
      use scale_module, only: born_flow_picked
      implicit none
      include 'orders.inc'
      double precision, intent(in) :: p(0:3,nexternal),xi,y
      double precision, intent(in) :: p_born(0:3,nexternal-1)
      integer, intent(out) :: kernel_index
      double precision, intent(out) :: born_weights(amp_split_size)
      double precision, intent(out) :: born_spin_weights(amp_split_size)
      logical split_type(nsplitorders)
      common /c_split_type/split_type
      integer order,norders,iord

      norders=0
      iord=0
      do order=1,nsplitorders
         if (.not.split_type(order)) cycle
         if (order.ne.qcd_pos.and.order.ne.qed_pos) cycle
         norders=norders+1
         iord=order
      enddo
      if (norders.ne.1) then
         write (*,*) 'Error: MC@NLO requires exactly one QCD or QED ',
     $        'correction order',norders
         stop 1
      endif
      if (iord.eq.qed_pos) then
         write (*,*) 'QED colour connections are not implemented ',
     $        'in compute_MCsubtraction_kl'
         stop 1
      endif
      kernel_index=1
      call get_mbar(p,xi,y,p_born,ileg,born_flow_picked,iord,
     $     born_weights,born_spin_weights)
      end subroutine prepare_MCsubtraction_born

c Evaluate one connection with this history's Born amplitudes and limit
c classification. Damping is also retained by the caller for G replacement.
      subroutine xmcsubt_connection(p_born,i_connect,qMC,damping,
     $     include_gfun,kernel_index,kernel_limits,born_weights,
     $     born_spin_weights,lzone,z,amp_split_xmcxsec)
      use process_module, only: shower_mc_mod
      implicit none
      include 'orders.inc'
      double precision, intent(in) :: p_born(0:3,nexternal-1)
      double precision, intent(in) :: qMC,damping
      integer, intent(in) :: i_connect,kernel_index
      type(mc_kernel_limits), intent(in) :: kernel_limits
      double precision, intent(in) :: born_weights(amp_split_size)
      double precision, intent(in) :: born_spin_weights(amp_split_size)
      logical, intent(in) :: include_gfun
      logical, intent(out) :: lzone
      double precision, intent(out) :: z
      double precision, intent(out) :: amp_split_xmcxsec(amp_split_size)
      double precision xkern(2),xkernazi(2),E0sq
      double precision PY6PTweight,xi,xjac

      E0sq=dot(p_born(0,fksfather),p_born(0,i_connect))
      call get_shower_variables(E0sq,z,xi,xjac)
      call get_dead_zone(z,xi,p_born,qMC,i_connect,lzone,PY6PTweight)

      xkern=0d0
      xkernazi=0d0
      if (lzone) then
         call compute_splitting_kernels(xkern,xkernazi,z,xi,xjac,
     $        kernel_limits)
      endif
      if (shower_mc_mod(1:9).eq.'PYTHIA6PT') then
         xkern=xkern*PY6PTweight
         xkernazi=xkernazi*PY6PTweight
      endif

c Apply this history's G-functions before the outer sector performs
c the G replacement and repartitions complete native H weights.
      if (include_gfun) then
         xkern=xkern*gfactsf
         xkernazi=xkernazi*gfactazi*gfactsf
      endif
      amp_split_xmcxsec=(xkern(kernel_index)*born_weights
     $     +xkernazi(kernel_index)*born_spin_weights)*damping
      end subroutine xmcsubt_connection


      subroutine compute_splitting_kernels(xkern,xkernazi,z,xi,
     $     xjac,kernel_limits)
      use process_module
      implicit none
      double precision xkern(1:2),xkernazi(1:2),z,xi,xjac
      double precision tiny
      parameter (tiny=1d-6)
      logical needs_shower_jacobian
      type(mc_kernel_limits), intent(in) :: kernel_limits
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
            call compute_splitting_kernel_icode1(xkern,xkernazi,z,xi
     $           ,needs_shower_jacobian,kernel_limits)
         elseif(abs(i_type).eq.3.or.(i_type.eq.1.and.dabs(ch_i).gt.tiny))then
c g->qq, a->qq, a->ee (icode=2)
            call compute_splitting_kernel_icode2(xkern,xkernazi,z,xi
     $           ,needs_shower_jacobian,kernel_limits)
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
            call compute_splitting_kernel_icode3(xkern,xkernazi,z,xi
     $           ,needs_shower_jacobian,kernel_limits)
         elseif(i_type.eq.8.or.(i_type.eq.1.and.dabs(ch_i).lt.tiny))then
c q->qg, q->qa, sq->sqg, sq->sqa, e->ea (icode=4)
            call compute_splitting_kernel_icode4(xkern,xkernazi,z,xi
     $           ,needs_shower_jacobian,kernel_limits)
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
      if (needs_shower_jacobian) then
!     Analytic limits include xjac; generic massive kernels still need it.
         xkern(1:2)    = xkern(1:2)*xjac
         xkernazi(1:2) = xkernazi(1:2)*xjac
      endif
!     The configured PYTHIA8 first hard FSR uses global recoil and
!     recoilDeadCone=on, without a hard-system MEC. Only its scalar
!     g->gg kernel receives this factor; retain the helicity/G terms.
!     Keep the exact analytic collinear coefficient: PYTHIA's guarded
!     numerator has a numerical floor, not a perturbative mass term.
      if (shower_mc_mod.eq.'PYTHIA8'.and.nincoming_mod.eq.2.and.
     &    ileg.eq.4.and..not.kernel_limits%collinear.and.
     &    m_type.eq.8.and.i_type.eq.8.and.j_type.eq.8) then
         xkern(1)=xkern(1)*
     &        py8_gluon_recoil_weight(z,shat_n1,xm12,w2)
      endif
      return
      end subroutine compute_splitting_kernels

      double precision function py8_gluon_recoil_weight(z,s,mrec2,
     &     mpair2)
      implicit none
      double precision z,s,mrec2,mpair2,r,v,d1,d2,xmargin
      parameter (xmargin=1d-12)
!     SimpleTimeShower::pT2nextQCD (PYTHIA 8.318). The recoiler is
!     the sum of all hard spectators, including massless ones.
      py8_gluon_recoil_weight=1d0
      if (mrec2.le.0d0) return
      r=mrec2/s
      v=mpair2/s
!     x1+x2-1-r and 1-r-x1, written without subtracting unit terms.
!     z <-> 1-z exchanges d1,d2: D_rec is symmetric even with guards.
!     Thus the existing symmetric AP kernel and complementary fks_Hij
!     partitions give P_end(z)*D_rec(z)+P_end(1-z)*D_rec(1-z).
      d1=(1d0-r)*z-v*(1d0-z)
      d2=(1d0-r)*(1d0-z)-v*z
!     PYTHIA rejects a negative trial weight. Reproduce that zero
!     probability, retaining its XMARGIN guards on each denominator.
      py8_gluon_recoil_weight=max(0d0,1d0-
     &     r*max(xmargin,v)/(max(xmargin,d1)*max(xmargin,d2)))
      return
      end function py8_gluon_recoil_weight

      function classify_mc_kernel_limits(xi_i_fks,y_ij_fks)
     $     result(region)
      implicit none
      double precision, intent(in) :: xi_i_fks,y_ij_fks
      double precision tiny
      type(mc_kernel_limits) :: region
      logical softtest,colltest
      common/sctests/softtest,colltest
c Classify once per history; all its colour connections share the same
c collinear approximation and soft boundary. G-functions cover soft points.
      tiny = 1d-6
      if (softtest.or.colltest)tiny = 1d-12
      region%collinear=1-y_ij_fks.lt.tiny .and. xi_i_fks.ge.tiny
      region%nonsoft=xi_i_fks.ge.tiny
      end function classify_mc_kernel_limits

      double precision function xfact_ileg12(N_p)
      use process_module
      implicit none
      integer N_p
      xfact_ileg12=(1d0-yij)*(1d0-x)/x * 4d0/(shat_n1*N_p)
      end function xfact_ileg12

      double precision function xfact_ileg3(N_p)
      use process_module
      implicit none
      integer N_p
      double precision geometry
!     Both massive FKS solutions have positive phase-space measures.
!     Every shower xjac takes an absolute radiation determinant, so
!     the shared FKS radial factor must also use its magnitude.
!     Form the geometry before dividing by a small momentum.
      geometry=(1d0+x)*kn+(1d0-x)*yij*kn0
      xfact_ileg3=abs(geometry)/kn**2*knbar*(1d0-x)*
     &     (1d0-yij)*2d0/(shat_n1*N_p)
      end function xfact_ileg3

      double precision function xfact_ileg4(N_p)
      use process_module
      implicit none
      integer N_p
      xfact_ileg4=(2d0-(1d0-x)*(1d0-yij))/
     &     xij*(1d0-xm12/shat_n1)*(1d0-x)*(1d0-yij) * 2d0/(shat_n1*N_p)
      end function xfact_ileg4

      subroutine compute_splitting_kernel_icode1(xkern,xkernazi,z,xi
     $     ,needs_shower_jacobian,kernel_limits)
      use process_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      integer N_P
      double precision vca,one
      parameter (vca=3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      logical needs_shower_jacobian
      type(mc_kernel_limits), intent(in) :: kernel_limits
      needs_shower_jacobian=.false.
      s=shat_n1
c g->gg, go->gog (icode=1)
      if(ileg.le.2)then
         N_p=2
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*8*vca*(1-x*(1-x))**2/(s*x**2)
            xkernazi(1)=-(g**2/N_p)*16*vca*(1-x)**2/(s*x**2)
            xkern(2)=0d0
            xkernazi(2)=0d0
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
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
         if(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg3(N_p)
            call AP_reduced_SUSY(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         N_p=2
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*( 8*vca*
     &           (s**2*(1-(1-x)*x)-s*(1+x)*xm12+xm12**2)**2 )/
     &           ( s*(s-xm12)**2*(s*x-xm12)**2 )
            xkernazi(1)=-(g**2/N_p)*(16*vca*s*(1-x)**2)/((s-xm12)**2)
            xkern(2)=0d0
            xkernazi(2)=0d0
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
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
      end subroutine compute_splitting_kernel_icode1

      subroutine compute_splitting_kernel_icode2(xkern,xkernazi,z,xi
     $     ,needs_shower_jacobian,kernel_limits)
      use process_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      integer N_p
      double precision vtf,one
      parameter (vtf=1d0/2d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      logical needs_shower_jacobian
      type(mc_kernel_limits), intent(in) :: kernel_limits
      needs_shower_jacobian=.false.
      s=shat_n1
c g->qq, a->qq, a->ee (icode=2)
      if(ileg.le.2)then
         N_p=1
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*4*vtf*(1-x)*((1-x)**2+x**2)/(s*x)
            xkern(2)=xkern(1) * dble(gal(1))**2 / g**2 *
     &           ch_i**2 * abs(i_type) / vtf
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         N_p=2
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*( 4*vtf*(1-x)*
     &           (s**2*(1-2*(1-x)*x)-2*s*x*xm12+xm12**2) )/
     &           ( (s-xm12)**2*(s*x-xm12) )
            xkern(2)=xkern(1) * dble(gal(1))**2 / g**2 *
     &           ch_i**2 * abs(i_type) / vtf
            xkernazi(1)=(g**2/N_p)*(16*vtf*s*(1-x)**2)/((s-xm12)**2)
            xkernazi(2)=xkernazi(1) * dble(gal(1))**2 / g**2 *
     &           ch_i**2 * abs(i_type) / vtf
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
            call Qterms_reduced_timelike(j_type,i_type,ch_m,ch_i,one,z
     $           ,Q)
            xkernazi(1:2)=xfact*Q(1:2)/(xi*(1-z))
         endif
      endif
      end subroutine compute_splitting_kernel_icode2

      subroutine compute_splitting_kernel_icode3(xkern,xkernazi,z,xi
     $     ,needs_shower_jacobian,kernel_limits)
      use process_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2),Q(1:2)
      integer N_P
      double precision vcf,one
      parameter (vcf=4d0/3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      logical needs_shower_jacobian
      type(mc_kernel_limits), intent(in) :: kernel_limits
      needs_shower_jacobian=.false.
      s=shat_n1
c q->gq, q->aq, e->ae (icode=3)
      if(ileg.le.2)then
         N_p=2
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*4*vcf*(1-x)*((1-x)**2+1)/(s*x**2)
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_i**2 / vcf)
            xkernazi(1)=-(g**2/N_p)*16*vcf*(1-x)**2/(s*x**2)
            xkernazi(2)=xkernazi(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_i**2 / vcf)
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
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
         if(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg3(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         N_p=1
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*
     &           ( 4*vcf*(1-x)*(s**2*(1-x)**2+(s-xm12)**2) )/
     &           ( (s-xm12)*(s*x-xm12)**2 )
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_i**2 / vcf)
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
      endif
      end subroutine compute_splitting_kernel_icode3

      subroutine compute_splitting_kernel_icode4(xkern,xkernazi,z,xi
     $     ,needs_shower_jacobian,kernel_limits)
      use process_module
      implicit none
      include "coupl.inc"
      double precision xkern(1:2),xkernazi(1:2),s,z,xi,xfact
     $     ,ap(1:2)
      integer N_P
      double precision vcf,one
      parameter (vcf=4d0/3d0)
      parameter (one=1d0)
c Particle types (=color) of i_fks, j_fks and fks_mother
      double precision       ch_i,ch_j,ch_m
      integer                i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &                       i_type,j_type,m_type,j_pdg
      logical needs_shower_jacobian
      type(mc_kernel_limits), intent(in) :: kernel_limits
      needs_shower_jacobian=.false.
      s=shat_n1
c q->qg, q->qa, sq->sqg, sq->sqa, e->ea (icode=4)
      if(ileg.le.2)then
         N_p=1
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*4*vcf*(1+x**2)/(s*x)
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_m**2 / vcf)
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg12(N_p)
            call AP_reduced(m_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
c
      elseif(ileg.eq.3)then
         N_p=1
         if(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg3(N_p)
            if(abs(j_pdg).le.6)then
               if(shower_mc_mod(1:7).ne.'HERWIG7')
     &              call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
               if(shower_mc_mod(1:7).eq.'HERWIG7')
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
         if(kernel_limits%collinear)then
            xkern(1)=(g**2/N_p)*4*vcf*
     &           ( s**2*(1+x**2)-2*xm12*(s*(1+x)-xm12) )/
     &           ( s*(s-xm12)*(s*x-xm12) )
            xkern(2)=xkern(1) * (dble(gal(1))**2 / g**2) *
     &           (ch_j**2 / vcf)
         elseif(kernel_limits%nonsoft)then
            needs_shower_jacobian=.true.
            xfact=xfact_ileg4(N_p)
            call AP_reduced(j_type,i_type,ch_m,ch_i,one,z,ap)
            xkern(1:2)=xfact*ap(1:2)/(xi*(1-z))
         endif
      endif
      end subroutine compute_splitting_kernel_icode4


      subroutine get_shower_variables(E0sq,z,xi,xjac)
      use process_module, only: shower_mc_mod
      implicit none
      double precision E0sq,z,xi,xjac
      if(shower_mc_mod(1:7).eq.'HERWIG6')then
         z=zHW6(E0sq)
         xi=xiHW6(E0sq,z)
         xjac=xjacHW6(E0sq,xi,z)
      elseif(shower_mc_mod(1:7).eq.'HERWIG7')then
         z=zHW7()
         xi=xiHW7(z)
         xjac=xjacHW7(z)
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
      else
         write(*,*) 'Unknown shower in get_shower_variables: ',
     $        shower_mc_mod
         stop 1
      endif
      end subroutine get_shower_variables

c Delta matching: no-emission probability and H-event scales.
      subroutine compute_delta(p,probne)
c     Assemble the no-emission probability and the H-event start scales.
      use fks_phase_space_data, only: xbjrk_cnt
      use process_module, only: valid_dipole_n1,ndelH
      use scale_module, only: born_flow_picked,shower_scale_nbody,
     $     force_II_connection,emsca_H
      implicit none
      include 'nFKSconfigs.inc'
      include 'run.inc'
      include 'genps.inc'

      double precision p(0:3,nexternal),probne,wgt_sudakov
      integer MCcntcalled,nFKSprocess,fold,ifold_counter
      common/c_MCcntcalled/MCcntcalled
      common/c_NFKSPROCESS/nFKSprocess
      common/cfl/fold,ifold_counter
      integer idup(nexternal-1,maxproc)
      integer mothup(2,nexternal-1,maxproc)
      integer icolup(2,nexternal-1,max_bcol)
      include 'born_leshouche.inc'
      integer idup_s(nexternal-1),idup_h(nexternal)
      integer icolup_s(2,nexternal-1),icolup_h(2,nexternal)
      common/colour_connections/icolup_s,icolup_h
      integer jpart(7,-nexternal+3:2*nexternal-3)
      include 'leshouche_decl.inc'
      save idup_d,mothup_d,icolup_d,niprocs_d
      logical firsttime1
      data firsttime1/.true./

c     The first Sudakov call initializes these fitted table bounds.
      double precision cstlow,cstupp,cxmlow,cxmupp
      common/cstxmbds/cstlow,cstupp,cxmlow,cxmupp
      double precision smallptupp
      parameter (smallptupp=1.01d0)
      double precision Sevent_starting_scales(nexternal-1,nexternal-1)
      double precision Sevent_stopping_scales(nexternal-1,nexternal-1)
      double precision xmasses_nbody(nexternal-1,nexternal-1)
      double precision Hevent_starting_scales(nexternal,nexternal)
      logical*1 dzones_nbody(nexternal-1,nexternal-1)
      type(delta_connections) leg_connections(nexternal-1)
      integer i,j
      double precision mcmass(21),deltanum,leg_probability

      mcmass=0d0
      include 'MCmasses_PYTHIA8.inc'
c
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
c     by the D(mu) function) for extra radiation, copied from the
c     shower_scale_nbody array. Only entries associated with a colour
c     line in born_flow_picked have meaningful values; others are -1.
      Sevent_starting_scales(1:nexternal-1,1:nexternal-1)=
     &     shower_scale_nbody(1:nexternal-1,1:nexternal-1)

c     H-event information: real-state IDs and the selected colours.
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
      enddo
c     Fill selected color configuration into jpart array.
      call fill_icolor_H(born_flow_picked,jpart,.false.)
      do i=1,nexternal
         ICOLUP_H(1,i)=jpart(4,i)
         ICOLUP_H(2,i)=jpart(5,i)
      enddo
      call get_delta_stopping_scales(p,idup_h,icolup_s,
     $     Sevent_starting_scales,cxmupp,Sevent_stopping_scales,
     $     xmasses_nbody,dzones_nbody)
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


!     Paper: eq.3.14 (times the PDF factor in 3.32) defines what to compute
!     for each QCD particle in the n-body process. This is updated to
!     3.31 for quarks, and 3.34 for gluons. (Check the curly brackets in
!     3.14 & 3.34).  First term in 3.34 is equal to
!     gl(1)/sum(gl)*delta(1,1)*delta(1,2)*Fk(1) in the notation
!     of the code below (and equivalently for the 2nd term). 3.37 is
!     wgt_sudakov.
!

      call get_delta_connections(Sevent_starting_scales,
     $     Sevent_stopping_scales,dzones_nbody,xmasses_nbody,idup_s,
     $     born_flow_picked,isspecial(born_flow_picked),smallptupp,
     $     cstupp,leg_connections)

      wgt_sudakov=1d0
c     Each Born leg contributes one probability, including its live
c     colour connections. Only incoming legs have beam PDF inputs.
      do i=1,nincoming
         if (leg_connections(i)%count.eq.0) cycle
         leg_probability=delta_leg_probability(i,idup_s(i),
     $        idup(i,1:iproc_born),lpp(i),xbjrk_cnt(i,0),
     $        leg_connections(i),mcmass)
         wgt_sudakov=wgt_sudakov*leg_probability
      enddo
      do i=nincoming+1,nexternal-1
         if (leg_connections(i)%count.eq.0) cycle
         leg_probability=delta_leg_probability(i,idup_s(i),
     $        idup(i,1:iproc_born),0,0d0,leg_connections(i),mcmass)
         wgt_sudakov=wgt_sudakov*leg_probability
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
      end subroutine compute_delta

      subroutine get_delta_stopping_scales(p,idup_h,icolup_s,
     $     Sevent_starting_scales,mass_upper,Sevent_stopping_scales,
     $     xmasses_nbody,dzones_nbody)
c     Reconstruct real dipoles, relabel to Born legs and apply the veto.
      use process_module, only: iRtoB
      use mcatnlo_delta_scales, only: delta_scale_matrices,delta_ok
      implicit none
      double precision p(0:3,nexternal),mass_upper
      integer idup_h(nexternal),icolup_s(2,nexternal-1)
      double precision Sevent_starting_scales(nexternal-1,nexternal-1)
      double precision Sevent_stopping_scales(nexternal-1,nexternal-1)
      double precision xmasses_nbody(nexternal-1,nexternal-1)
      logical*1 dzones_nbody(nexternal-1,nexternal-1)
      double precision xscales(0:nexternal,0:nexternal)
      double precision xmasses(0:nexternal,0:nexternal)
      logical dzones(0:nexternal,0:nexternal)
      integer i,j,delta_status,i_fks,j_fks,nFKSprocess
      common/fks_indices/i_fks,j_fks
      common/c_NFKSPROCESS/nFKSprocess
      double precision get_mass_from_id
      external get_mass_from_id
      intent(in) :: p,idup_h,icolup_s,Sevent_starting_scales,mass_upper
      intent(out) :: Sevent_stopping_scales,xmasses_nbody,dzones_nbody

c     Reconstruct the PYTHIA stopping-scale prescription from the real
c     momenta and the selected Born colour flow. These expressions are
c     Lorentz invariant, so no boost to the lab frame is needed. Use model
c     charm and bottom masses; get_mass_from_id returns zero for flavours
c     made massless by the model restriction. The separate Sudakov-table
c     mass input is supplied by compute_delta.
      if (nincoming.ne.2) then
         write(*,*) 'MC@NLO-Delta scales require two incoming legs'
         stop 1
      endif
      call delta_scale_matrices(nexternal,i_fks,p,idup_h,icolup_s,
     $     get_mass_from_id(4),get_mass_from_id(5),xscales,xmasses,
     $     dzones,delta_status)
      if (delta_status.ne.delta_ok) then
         write(*,*) 'MC@NLO-Delta scale reconstruction failed: ',
     $        delta_status,'; FKS configuration ',nFKSprocess,
     $        '; emitted/radiator ',i_fks,j_fks
         stop 1
      endif

c     After the reconstruction above, we have
c     xscales(i,j)=t_ij
c     with t_ij == scale(Pythia)_{emitter,recoiler}, and the particle being
c     emitted equal to the FKS parton. Although both emitter and recoiler
c     are Born-level quantities, their labellings follow the real-process
c     conventions. Thus, in the matrix xscales(i,j) one has 1<=i,j<=nexternal,
c     with xscales(i_fks,*)=xscales(*,i_fks)=-1.
c     The same labelling conventions apply to xmasses(i,j) (which is the
c     dipole mass associated with the colour line that connects i and j)
c     and dzones(i,j) (which is the dead zone relevant to the emission from
c     parton i colour-connected with recoiler j).
c
c     Since any the pair of indices (i,j) associated with sensible entries
c     in the reconstructed arrays is in one-to-one correspondence with
c     Born-level quantities, it is convenient to define relabelled copies of
c     such arrays (which we call Sevent_stopping_scales, xmasses_nbody, and
c     dzones_nbody), for which 1<=i,j<=nexternal-1
c

      do i=1,nexternal
         if(i.eq.i_fks)cycle
         do j=1,nexternal
            if(j.eq.i_fks)cycle
            Sevent_stopping_scales(iRtoB(i),iRtoB(j))=xscales(i,j)
c     The reconstructed dipole masses can exceed the Sudakov table range.
c     Cap them at the largest allowed value in the pysudakov() tables.
            xmasses_nbody(iRtoB(i),iRtoB(j))=min(xmasses(i,j),mass_upper)
            dzones_nbody(iRtoB(i),iRtoB(j))=dzones(i,j)
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

!     Apply the MG5 starting-scale veto after the reconstruction: a dipole
!     is in the dead zone when its stopping scale exceeds its starting scale.
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
      end subroutine get_delta_stopping_scales

      subroutine get_delta_connections(Sevent_starting_scales,
     $     Sevent_stopping_scales,dzones_nbody,xmasses_nbody,idup_s,
     $     iflow,special_flow,scale_lower,scale_upper,connections)
c     Select live connections and bound their Sudakov table scales.
      use process_module, only: valid_dipole_n
      implicit none
      double precision, intent(in) ::
     $     Sevent_starting_scales(nexternal-1,nexternal-1),
     $     Sevent_stopping_scales(nexternal-1,nexternal-1),
     $     xmasses_nbody(nexternal-1,nexternal-1)
      logical*1, intent(in) :: dzones_nbody(nexternal-1,nexternal-1)
      integer, intent(in) :: idup_s(nexternal-1),iflow
      logical, intent(in) :: special_flow
      double precision, intent(in) :: scale_lower,scale_upper
      type(delta_connections), intent(out) :: connections(nexternal-1)
      integer i,j,slot

      connections=delta_connections()
c     Loop over particles and their connections (a and beta in eq.3.14).
      do i=1,nexternal-1
         do j=1,nexternal-1
            if (.not.valid_dipole_n(i,j,iflow)) cycle
            if (dzones_nbody(i,j)) cycle
c     Keep live connections and bound their scales as in eq.3.33.
            if (Sevent_starting_scales(i,j).lt.
     $           Sevent_stopping_scales(i,j)) cycle
            if (connections(i)%count.eq.2) then
               write(*,*) 'Too many Delta colour connections',i,iflow
               stop 1
            endif
            slot=connections(i)%count+1
            connections(i)%count=slot
            connections(i)%partner(slot)=j
            connections(i)%start(slot)=max(min(
     $           Sevent_starting_scales(i,j),scale_upper),scale_lower)
            connections(i)%stop(slot)=min(
     $           Sevent_stopping_scales(i,j),scale_upper)
            connections(i)%mass(slot)=xmasses_nbody(i,j)
            connections(i)%sudakov_type(slot)=setSudType(i,j)
         enddo
         if (connections(i)%count.eq.1.and.idup_s(i).eq.21) then
            if (special_flow) then
c     A special flow carries two colour lines to the same partner.
               connections(i)%count=2
               connections(i)%partner(2)=connections(i)%partner(1)
               connections(i)%start(2)=connections(i)%start(1)
               connections(i)%stop(2)=connections(i)%stop(1)
               connections(i)%mass(2)=connections(i)%mass(1)
               connections(i)%sudakov_type(2)=
     $              connections(i)%sudakov_type(1)
            endif
c     A non-special gluon can have only one live connection when the
c     other connection lies in a dead zone.
         endif
      enddo
      end subroutine get_delta_connections

      double precision function delta_leg_probability(ipart,pdg,
     $     flavours,beam_type,bjorken,connections,mcmass)
c     Evaluate eqs.3.31/3.34 for one Born leg from its live connections.
c     Keep the PDF flavour average and the order of Sudakov-table calls.
      implicit none
      integer, intent(in) :: ipart,pdg,flavours(:),beam_type
      double precision, intent(in) :: bjorken,mcmass(21)
      type(delta_connections), intent(in) :: connections
      integer in_con,out_con,isudtype,ip,lp,id
      double precision deltanum,deltaden,delta(2,2),gl(2),Fk(2)
      double precision pdfnum,pdfden,PIk
      double precision pdg2pdf
      external pdg2pdf

      delta_leg_probability=1d0
      if (connections%count.eq.0) return
      do out_con=1,connections%count
c     Compute g1 and g2 for the two lines of eq.3.34.
         if (connections%count.eq.1) then
            gl(out_con)=1d0
         else
            isudtype=connections%sudakov_type(out_con)
            deltanum=pysudakov_safe(connections%stop(out_con),
     $           connections%mass(out_con),pdg,isudtype,mcmass)
            deltaden=pysudakov_safe(connections%start(out_con),
     $           connections%mass(out_con),pdg,isudtype,mcmass)
            gl(out_con)=gl_safe(deltanum,deltaden)
         endif
c     Flavour configurations are summed in the Born matrix element.
c     Approximate their separate PDF ratios by the average weighted by
c     the starting-scale PDFs: sum(pdfnum)/sum(pdfden).
         if (ipart.le.nincoming) then
            lp=sign(1,beam_type)
            pdfnum=0d0
            pdfden=0d0
            do ip=1,size(flavours)
               id=get_parton_id(flavours(ip),lp)
               pdfnum=pdfnum+pdg2pdf(abs(beam_type),id,lp,bjorken,
     $              connections%stop(out_con))
               pdfden=pdfden+pdg2pdf(abs(beam_type),id,lp,bjorken,
     $              connections%start(out_con))
            enddo
         else
            pdfnum=1d0
            pdfden=1d0
         endif
         if (pdfden.eq.0d0) then
c     This can occur at an isolated PDF scale.
            pdfden=1d-99
         endif
         Fk(out_con)=pdfnum/pdfden
c     Loop over gamma in eq.3.34.
         do in_con=1,connections%count
            isudtype=connections%sudakov_type(in_con)
            deltanum=pysudakov_safe(connections%stop(out_con),
     $           connections%mass(in_con),pdg,isudtype,mcmass)
            deltaden=pysudakov_safe(connections%start(in_con),
     $           connections%mass(in_con),pdg,isudtype,mcmass)
            if (deltaden.eq.0d0) then
               if (deltanum.ne.0d0) then
                  write (*,*) 'Denominator is zero in Sudakov'
                  write (*,*) deltanum,deltaden
                  write (*,*) connections%stop(out_con),
     $                 connections%start(in_con),in_con,ipart,
     $                 connections%partner(in_con),
     $                 connections%mass(in_con),pdg,isudtype
                  stop 1
               endif
               delta(out_con,in_con)=0d0
            else
               delta(out_con,in_con)=deltanum/deltaden
            endif
         enddo
      enddo
c     Sum the colour-line contributions in eq.3.34 and bound the result
c     to interpret it as a probability at the accuracy used here.
      PIk=0d0
      do out_con=1,connections%count
         PIk=PIk+gl(out_con)/sum(gl(1:connections%count))*Fk(out_con)
     $        *product(delta(out_con,1:connections%count))
      enddo
      delta_leg_probability=min(max(PIk,0d0),1d0)
      end function delta_leg_probability

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
      end function gl_safe

      double precision function pysudakov_safe(scale,mass,id,type
     $     ,mcmass)
      implicit none
      double precision scale,mass,pysudakov
      double precision mcmass(21)
      integer id,type
      real*8 smallptlow,smallptupp
      parameter (smallptlow=0.5d0)
      parameter (smallptupp=1.01d0)
      if(scale.lt.0d0)then
         write (*,*) 'scale smaller than 0 in pysudakov_safe',scale
         stop 1
      elseif(scale.le.smallptlow)then
         pysudakov_safe=0.d0
      elseif(scale.le.smallptupp)then
         pysudakov_safe = pysudakov(smallptupp,mass,id,type,mcmass)
     $        *get_to_zero(scale,smallptlow,smallptupp)
      else
         pysudakov_safe=pysudakov(scale,mass,id,type,mcmass)
      endif
      end function pysudakov_safe


      integer function get_parton_id(ipdg,lp)
      implicit none
      integer ipdg,lp
      if (abs(ipdg).ge.1.and.abs(ipdg).le.6) then
         get_parton_id=lp*ipdg
      elseif (ipdg.eq.21) then  ! gluon
         get_parton_id=0
      elseif (ipdg.eq.22) then  ! photon
         get_parton_id=7
      else
         write (*,*) 'unknown PDG for PDF',ipdg
         stop 1
      endif
      end function get_parton_id

      integer function setSudType(i,j)
      implicit none
      integer i,j
      if(i.le.2)then
c     For Pythia: IF is identical to II.
         setsudtype=1
      elseif(j.gt.2)then
         setsudtype=2
      else
         setsudtype=4
      endif
      end function setSudType

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
      use process_module, only: iRtoB,valid_dipole_n,valid_dipole_n1
      use scale_module, only: born_flow_picked
      implicit none
      logical*1 dzones_nbody(nexternal-1,nexternal-1)
      double precision Sevent_stopping_scales(nexternal-1,nexternal-1)
     $     ,Hevent_starting_scales(nexternal,nexternal),p(0:3,nexternal)
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i1,i2,imother,i1bar,i2bar,i
      double precision t(nexternal,nexternal),pT,pTparton
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
      end subroutine get_Hevent_starting_scales

      double precision function compute_pTparton(p)
      implicit none
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
      end function compute_pTparton


      integer function ipbar(imother)
      ! ipbar is the colour connection of i_fks (if it exists and is not
      ! equal to the mother). Otherwise it is the colour connection of
      ! j_fks. The latter only happens when i_fks is a quark and j_fks
      ! is an (incoming gluon).
      use process_module, only: iRtoB,valid_dipole_n1
      implicit none
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
      end function ipbar


      function get_to_zero(sc,xlow,xupp)
      implicit none
      double precision get_to_zero,xlow,xupp,sc
      double precision x
      x=(xupp-sc)/(xupp-xlow)
      get_to_zero=1-emscafun(x,2d0)
      return
      end function get_to_zero






      subroutine get_mbar(p,xi_i_fks,y_ij_fks,p_born,ileg,iflow,iord,
     $     born_weights,born_spin_weights)
c Return the scalar and azimuthal split amplitudes for one selected
c Born flow and correction order, using Odagiri's prescription.
      implicit none
      include 'orders.inc'
      include 'nFKSconfigs.inc'
      double precision, intent(in) :: p(0:3,nexternal)
      double precision, intent(in) :: p_born(0:3,nexternal-1)
      double precision, intent(in) :: xi_i_fks,y_ij_fks
      integer, intent(in) :: ileg,iflow,iord
      double precision, intent(out) :: born_weights(amp_split_size)
      double precision, intent(out) :: born_spin_weights(amp_split_size)
      double precision, external :: mc_born_flow_weight
      double complex czero,ximag
      parameter (czero=(0d0,0d0),ximag=(0d0,1d0))
      double precision p_born_rot(0:3,nexternal-1),wgt_born
      double precision born,amp_split_born(amp_split_size)
      double complex borntilde,amp_split_borntilde(amp_split_size)
      double complex azifact
      double precision sumborn,flow_weight,flow_fraction
      double precision cphi_mother,sphi_mother
      integer i,iamp,imother_fks
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      double precision iden_comp,iden_comp_FKS(fks_configs)
      common /c_iden_comp/iden_comp,iden_comp_FKS
      double precision ch_i,ch_j,ch_m
      integer i_type,j_type,m_type,j_pdg
      common/cparticle_types/ch_i,ch_j,ch_m,
     &     i_type,j_type,m_type,j_pdg
      double complex ans_cnt(2,nsplitorders)
      common /c_born_cnt/ans_cnt
      integer iextra_cnt,isplitorder_born,isplitorder_cnt
      common /c_extra_cnt/iextra_cnt,isplitorder_born,isplitorder_cnt
      logical is_leading_cflow(max_bcol)
      integer num_leading_cflows
      common/c_leading_cflows/is_leading_cflow,num_leading_cflows

      if (iflow.lt.1.or.iflow.gt.max_bcol.or.
     $    iord.lt.1.or.iord.gt.nsplitorders) then
         write(*,*) 'Invalid Born flow or correction order in get_mbar',
     $        iflow,iord
         stop 1
      endif
      if (iextra_cnt.gt.0) then
         write(*,*) 'Extra-counterterm matching is not implemented'
         stop 1
      endif

c Rotate the second incoming leg into the matrix-element convention.
c Exclude 2->1 Born processes, whose nonzero helicity configurations
c can flip even though their matrix elements have no momentum dependence.
      if ((ileg.eq.1.or.ileg.eq.2).and.
     $    (j_fks.eq.2.and.nexternal-1.ne.3)) then
         do i=1,nexternal-1
            p_born_rot(0,i)=p_born(0,i)
            p_born_rot(1,i)=-p_born(1,i)
            p_born_rot(2,i)=p_born(2,i)
            p_born_rot(3,i)=-p_born(3,i)
         enddo
         calculatedBorn=.false.
         call sborn_native(p_born_rot,wgt_born)
         calculatedBorn=.false.
      else
         call sborn_native(p_born,wgt_born)
      endif

      born=dble(ans_cnt(1,iord))
      borntilde=ans_cnt(2,iord)
      amp_split_born=dble(amp_split_cnt(:,1,iord))
      amp_split_borntilde=amp_split_cnt(:,2,iord)
      if (abs(m_type).eq.3.or.dabs(ch_m).gt.0d0) then
         borntilde=czero
         amp_split_borntilde=czero
      endif

c Complete the spin-correlated amplitudes with <ij>/[ij] and the
c polarization-vector phase. Keep the original multiplication order.
      if (ileg.eq.1.or.ileg.eq.2) then
         azifact=mc_born_azimuth_phase(p,xi_i_fks,y_ij_fks,
     $        j_fks.eq.2)
         if (j_fks.eq.2) then
            cphi_mother=-1d0
         else
            cphi_mother=1d0
         endif
         sphi_mother=0d0
         borntilde=-(cphi_mother+ximag*sphi_mother)**2*
     $        borntilde*dconjg(azifact)
         do iamp=1,amp_split_size
            amp_split_borntilde(iamp)=
     $           -(cphi_mother+ximag*sphi_mother)**2*
     $           amp_split_borntilde(iamp)*dconjg(azifact)
         enddo
      elseif (ileg.eq.3.or.ileg.eq.4) then
         if ((abs(j_type).eq.3.or.ch_j.ne.0d0).and.
     $       (i_type.eq.8.or.i_type.eq.1).and.ch_i.eq.0d0) then
            borntilde=czero
            amp_split_borntilde=czero
         elseif ((m_type.eq.8.or.m_type.eq.1).and.ch_m.eq.0d0) then
            azifact=mc_born_azimuth_phase(p,xi_i_fks,y_ij_fks,.false.)
            imother_fks=min(i_fks,j_fks)
            call getaziangles(p_born(0,imother_fks),
     $           cphi_mother,sphi_mother)
            borntilde=-(cphi_mother-ximag*sphi_mother)**2*
     $           borntilde*azifact
            do iamp=1,amp_split_size
               amp_split_borntilde(iamp)=
     $              -(cphi_mother-ximag*sphi_mother)**2*
     $              amp_split_borntilde(iamp)*azifact
            enddo
         else
            write(*,*) 'FATAL ERROR in get_mbar',
     $           i_type,j_type,i_fks,j_fks
            stop
         endif
      else
         write(*,*) 'Unknown ileg in get_mbar',ileg
         stop
      endif

c Normalize with all leading flows, summed in their original order.
      sumborn=0d0
      do i=1,max_bcol
         if (is_leading_cflow(i))
     $        sumborn=sumborn+mc_born_flow_weight(i)
      enddo
      born_weights=0d0
      born_spin_weights=0d0
      if (sumborn.eq.0d0) then
c Preserve the zero-sum diagnostics even for a different selected flow.
         do i=1,max_bcol
            if (.not.is_leading_cflow(i)) cycle
            flow_weight=mc_born_flow_weight(i)
            if (flow_weight.eq.0d0) cycle
            if (born.ne.0d0) then
               write(*,*) 'ERROR #1, dividing by zero'
               stop
            endif
            if (borntilde.ne.czero) then
               write(*,*) 'ERROR #2, dividing by zero'
               stop
            endif
         enddo
         return
      endif
      if (.not.is_leading_cflow(iflow)) return
      flow_fraction=mc_born_flow_weight(iflow)/sumborn
      do iamp=1,amp_split_size
         born_weights(iamp)=flow_fraction*amp_split_born(iamp)*iden_comp
         born_spin_weights(iamp)=flow_fraction*
     $        dble(amp_split_borntilde(iamp))*iden_comp
      enddo
      end subroutine get_mbar


      double complex function mc_born_azimuth_phase(p,xi_i_fks,
     &     y_ij_fks,rotate_beam)
c The spinor ratio <ij>/[ij] common to ISR and FSR barred amplitudes.
c At the collinear endpoint use the phase retained by the FKS map;
c in the soft region use its rescaled emitted momentum.
      use fks_phase_space_data, only: xij_aor,p_i_fks_ev
      implicit none
      double precision p(0:3,nexternal),xi_i_fks,y_ij_fks
      logical rotate_beam
      intent(in) :: p,xi_i_fks,y_ij_fks,rotate_beam
      integer i,i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision pi(0:3),pj(0:3),zero,vtiny
      parameter (zero=0d0,vtiny=1d-12)
      double complex w1(6),w2(6),w3(6),w4(6),wij_angle,wij_recta

      if (1d0-y_ij_fks.lt.vtiny) then
         mc_born_azimuth_phase=xij_aor
         return
      endif
      do i=0,3
         if (xi_i_fks.lt.1d-8) then
            pi(i)=p_i_fks_ev(i)
         else
            pi(i)=p(i,i_fks)
         endif
         pj(i)=p(i,j_fks)
      enddo
      if (rotate_beam) then
c Rotation according to innerpin.m for the second incoming leg.
         pi(1)=-pi(1)
         pi(3)=-pi(3)
         pj(1)=-pj(1)
         pj(3)=-pj(3)
      endif
      call IXXXSO(pi,zero,+1,+1,w1)
      call OXXXSO(pj,zero,-1,+1,w2)
      call IXXXSO(pi,zero,-1,+1,w3)
      call OXXXSO(pj,zero,+1,+1,w4)
      wij_angle=(0d0,0d0)
      wij_recta=(0d0,0d0)
      do i=1,4
         wij_angle=wij_angle+w1(i)*w2(i)
         wij_recta=wij_recta+w3(i)*w4(i)
      enddo
      mc_born_azimuth_phase=wij_angle/wij_recta
      end function mc_born_azimuth_phase

c Monte Carlo functions
c
c The invariants given in input to these routines follow FNR conventions
c (i.e., are defined as (p+k)^2, NOT 2 p.k).
c The invariants used inside these routines follow MNR conventions
c (i.e., are defined as -2p.k, NOT (p+k)^2)

c Herwig6

      double precision function zHW6(e0sq)
c     Shower energy variable
      implicit none
      double precision tiny,e0sq,ss,betae0,beta,zeta,tbeta
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
      end function zHW6


      double precision function xiHW6(e0sq,z)
c Shower evolution variable
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
      end function xiHW6


      double precision function xjacHW6(e0sq,xi,z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower
c variables, and x and y are FKS variables
      implicit none
      double precision tiny,z,xi,tmp,e0sq,beta,betae0,zmo
     $     ,tbeta,eps,dw1dx,dw2dx,dw1dy,dw2dy
      parameter (tiny=1d-5)

      if(z.lt.0d0.or.xi.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
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
      end function xjacHW6


c Herwig7

      double precision function zHW7()
c     Shower energy variable
      implicit none
      double precision tiny,zeta1,zeta2
      parameter (tiny=1d-5)
c
      if(ileg.eq.1.or.ileg.eq.2)then
         zHW7=1-(1-x)*(1+yij)/2d0
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            zHW7=1-(1-x)*(1+yij)/(betad+betas)
         else
            zeta1=get_zeta(shat_n1,w1,w2,xm12,xm22)
            zHW7=1-zeta1
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            zHW7=1-(1-x)*(1+yij)*shat_n1/(2*(shat_n1-xm12))
         elseif(1-yij.lt.tiny)then
            zHW7=(shat_n1*x-xm12)/(shat_n1-xm12)+(1-yij)*(1-x)*shat_n1
     $           *(shat_n1*x+xm12*(x-2))*(shat_n1*x-xm12)/(2*(shat_n1
     $           -xm12)**3)
         else
            zeta2=get_zeta(shat_n1,w2,w1,xm22,xm12)
            zHW7=1-zeta2
         endif
c
      else
         write(*,*)'zHW7: unknown ileg'
         stop
      endif

      if(zHW7.lt.0d0.or.zHW7.gt.1d0)goto 999

      return
 999  continue
      zHW7=-1d0
      return
      end function zHW7


      double precision function xiHW7(z)
c     Shower evolution variable
      implicit none
      double precision z,tiny
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
         xiHW7=shat_n1*(1-yij)/(1+yij)
c
      elseif(ileg.eq.3)then
         if(1-x.lt.tiny)then
            xiHW7=-shat_n1*(betad+betas)*(yij*betad-betas)/(2*(1+yij))
         else
            xiHW7=w1/(z*(1-z))
         endif
c
      elseif(ileg.eq.4)then
         if(1-x.lt.tiny)then
            xiHW7=(1-yij)*(shat_n1-xm12)**2/(shat_n1*(1+yij))
         elseif(1-yij.lt.tiny)then
            xiHW7=(1-yij)*(shat_n1-xm12)**2/(2*shat_n1)
         else
            xiHW7=w2/(z*(1-z))
         endif
c
      else
         write(*,*)'xiHW7: unknown ileg'
         stop
      endif

      if(xiHW7.lt.0d0)goto 999

      return
 999  continue
      xiHW7=-1d0
      return
      end function xiHW7


      double precision function xjacHW7(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower
c variables, and x and y are FKS variables
      implicit none
      double precision z,tmp,eps,beta,dw1dx,dw2dx,dw1dy,dw2dy,tiny
      parameter (tiny=1d-5)

      tmp=0d0
      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
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
         write(*,*)'xjacHW7: unknown ileg'
         stop
      endif
      xjacHW7=abs(tmp)

      return
 999  continue
      xjacHW7=0d0
      return
      end function xjacHW7


c Pythia6Q

      double precision function zPY6Q()
c Shower energy variable
      implicit none
      double precision tiny
      parameter(tiny=1d-5)
c
      if(ileg.eq.1.or.ileg.eq.2)then
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
      end function zPY6Q


      double precision function xiPY6Q()
c     Shower evolution variable
      implicit none
      double precision tiny
      parameter(tiny=1d-5)
c
      if(ileg.eq.1.or.ileg.eq.2)then
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
      end function xiPY6Q


      double precision function xjacPY6Q(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower
c     variables, and x and y are FKS variables
      implicit none
      double precision tiny,z,tmp,dw1dx,dw1dy,dw2dx,dw2dy
      parameter (tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
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
      end function xjacPY6Q


c Pythia6PT

      double precision function zPY6PT()
c Shower energy variable
      implicit none
      if(ileg.eq.1.or.ileg.eq.2)then
         zPY6PT=x
c
      elseif(ileg.eq.3.or.ileg.eq.4)then
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
      end function zPY6PT


      double precision function xiPY6PT()
c Shower evolution variable
      implicit none

      if(ileg.eq.1.or.ileg.eq.2)then
         xiPY6PT=shat_n1*(1-x)**2*(1-yij)/2
c
      elseif(ileg.eq.3.or.ileg.eq.4)then
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
      end function xiPY6PT


      double precision function xjacPY6PT()
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower
c     variables, and x and y are FKS variables
      implicit none
      double precision tmp
      if(ileg.eq.1.or.ileg.eq.2)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.3.or.ileg.eq.4)then
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
      end function xjacPY6PT


c Pythia8

      double precision function zPY8()
c Shower energy variable
      implicit none
      double precision tiny,omz
      parameter(tiny=1d-5)
c
      if(ileg.eq.1.or.ileg.eq.2)then
         zPY8=x
c
      elseif(ileg.eq.3)then
         call py8_massive_fsr_fractions(1d0-x,yij,zPY8,omz)
c This is equation (3.10) of hep-ph/1102.3795. In the partonic
c CM frame it is equal to (xk1(0)+xk3(0)*f)/(xk1(0)+xk3(0)),
c where f = xm12/( s+xm12-xm22-2*sqrt(s)*(xk1(0)+xk3(0)) )
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
      end function zPY8


      double precision function xiPY8(z)
c Shower evolution variable
      implicit none
      double precision tiny,z,z0,omz,gap
      parameter(tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
         xiPY8=shat_n1*(1-x)**2*(1-yij)/2
c
      elseif(ileg.eq.3)then
         call py8_massive_fsr_fractions(1d0-x,yij,z0,omz)
         gap=xm12/(kn0+kn)+(1d0-yij)*kn
         xiPY8=z*omz*sqrt(shat_n1)*(1d0-x)*gap
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
      end function xiPY8


      double precision function xjacPY8(z)
c Returns the jacobian d(z,xi)/d(x,y), where z and xi are the shower
c variables, and x and y are FKS variables
      implicit none
      double precision tiny,z,z0,dw1dx,dw1dy,dw2dx,dw2dy,tmp
     &     ,omz,geometry
!     Use the same endpoint expansion threshold as zPY8 and xiPY8.
      parameter(tiny=1d-5)

      if(z.lt.0d0)goto 999
c
      if(ileg.eq.1.or.ileg.eq.2)then
         tmp=-shat_n1*(1-x)**2/2
c
      elseif(ileg.eq.3)then
         call py8_massive_fsr_fractions(1d0-x,yij,z0,omz)
!     Differentiate (2-xi)*E+xi*y*k at fixed recoil mass.
!     dw1/dy=-2*sqrt(shat)*xi*k^2/geometry on either FKS branch.
!     Cancel xi analytically and retain the absolute determinant below.
         geometry=(1d0+x)*kn+(1d0-x)*yij*kn0
         tmp=2d0*sqrt(shat_n1)*kn**2/geometry*z*omz**2
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
      end function xjacPY8

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
      end function get_zeta


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
      end function emscafun




      function bogus_probne_fun(qMC)
      implicit none
      double precision bogus_probne_fun,qMC
      double precision x,tmp
      integer itype
! Artificial no-emission factor for the non-Delta debugging path.
! Mode 2 is smooth: P=0 below 0.5 GeV and P=1 above 10 GeV.
! Mode 3 disables it. The native-history evaluator applies P once to
! the real, MC and G-replacement terms, retaining (1-P)*R in S.
! Physical Delta uses compute_delta instead, not a product with this P.
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
      end function bogus_probne_fun


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
      end function get_angle


      subroutine dinvariants_dFKS(dw1dx,dw1dy,dw2dx,dw2dy)
      use fks_phase_space_data, only: veckn_ev
c Returns derivatives of Mandelstam invariants with respect to FKS variables
      use process_module
      implicit none
      double precision s,dw1dx,dw2dx,dw1dy,dw2dy
      double precision afun,bfun,cfun,mom_fks_sister_p,mom_fks_sister_m,
     &diff_p,diff_m,signfac,dadx,dady,dbdx,dbdy,dcdx,dcdy,mom_fks_sister,
     &dmomfkssisdx,dmomfkssisdy,en_fks,en_fks_sister
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
         dw1dx=-2*(s*(1+yij)+xm12*(1-yij))/(1+yij+x*(1-yij))**2
         dw2dx=(1-yij)*(s*(1+yij-x*(2*(1+yij)+x*(1-yij)))+2*xm12)/(1+yij+x*(1-yij))**2
         dw1dy=(1-x)*2*(s*x-xm12)/(1+yij+x*(1-yij))**2
         dw2dy=-2*(1-x)*(s*x-xm12)/(1+yij+x*(1-yij))**2
c
      else
         write(*,*)'Error in dinvariants_dFKS: unknown ileg',ileg
         stop
      endif

      return
      end subroutine dinvariants_dFKS


      subroutine get_dead_zone(z,xi,p_born,qMC,ipartner,lzone,PY6PTweight)
      use process_module
      use scale_module
      implicit none
      integer ipartner,i
      double precision z,xi,qMC,PY6PTweight
      logical lzone

      double precision p_born(0:3,nexternal-1)
      double precision upscale2,xmp2,xmm2,xmr2,ww,Q2,lambda,e0sq,beta,ycc,mdip,mdip_g,zp1,zm1,zp2,zm2,zp3,zm3,theta2p
     $     ,max_scale

      double precision ppartner(0:3),pfather(0:3)

      ! PYTHIA6 variables
      integer mstj50,mstp67
      double precision parp67
      parameter (mstj50=2,mstp67=2,parp67=1d0)

c Define the auxiliary weight even for an invalid shower point.
      PY6PTweight=1d0
c Skip if unphysical shower variables
      if(z.lt.0d0.or.xi.lt.0d0) then
         lzone=.false.
         return
      endif

c Definition and initialisation of variables
      lzone=.true.
      max_scale=shower_scale_nbody_max(fksfather,ipartner)
      do i=0,3
         pfather(i)=p_born(i,fksfather) ! father momentum (Born level)
         ppartner(i)=p_born(i,ipartner) ! partner momentum (Born level)
      enddo
      e0sq=dot(ppartner,pfather)
      if (shower_mc_mod(1:8).eq.'PYTHIA6Q')
     &     theta2p=get_angle(ppartner,pfather)**2
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
! Use the on-shell mass: reconstructing a massless partner's invariant
! can give a small negative value and silently lose the PYTHIA8 bound.
         xmp2=mass_n(ipartner)**2        ! mass squared of the partner
         if (shower_mc_mod(1:7).eq.'HERWIG7')
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
      elseif(shower_mc_mod(1:7).eq.'HERWIG7')then
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
      end subroutine get_dead_zone



c Shower-history preparation, invariants, damping scale and G functions.
c These helpers share the same active history as the analytic kernels.

      double precision function get_qMC(xi_i_fks,y_ij_fks)
c This is the (relative) pT of the splitting. For some showers this is
c equal to the shower variable, but not for all. This is what is used for
c the damping.
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        if(ileg.eq.1)then
           get_qMC=qMC_ileg1(xi_i_fks,y_ij_fks)
        elseif(ileg.eq.2)then
           get_qMC=qMC_ileg2(xi_i_fks,y_ij_fks)
        elseif(ileg.eq.3)then
           get_qMC=qMC_ileg3(xi_i_fks,y_ij_fks)
        elseif(ileg.eq.4)then
           get_qMC=qMC_ileg4(xi_i_fks,y_ij_fks)
        endif
        if(get_qMC.lt.0d0)then
           write(*,*) 'Error in get_qMC: qMC=',get_qMC
           stop 1
        endif
      end function get_qMC


      double precision function qMC_ileg1(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        if(shower_mc_mod.eq.'HERWIG6'  .or.
     $     shower_mc_mod.eq.'HERWIG7') qMC_ileg1=xi_i_fks/2d0*sqrt(shat_n1*(1-y_ij_fks**2))
        if(shower_mc_mod.eq.'PYTHIA6Q') qMC_ileg1=sqrt(-xtk)
        if(shower_mc_mod.eq.'PYTHIA6PT'.or.
     $     shower_mc_mod.eq.'PYTHIA8') qMC_ileg1=sqrt(-xtk*xi_i_fks)
      end function qMC_ileg1

      double precision function qMC_ileg2(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        if(shower_mc_mod.eq.'HERWIG6'  .or.
     $     shower_mc_mod.eq.'HERWIG7') qMC_ileg2=xi_i_fks/2d0*sqrt(shat_n1*(1-y_ij_fks**2))
        if(shower_mc_mod.eq.'PYTHIA6Q') qMC_ileg2=sqrt(-xuk)
        if(shower_mc_mod.eq.'PYTHIA6PT'.or.
     $     shower_mc_mod.eq.'PYTHIA8') qMC_ileg2=sqrt(-xuk*xi_i_fks)
      end function qMC_ileg2

      double precision function qMC_ileg3(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks,zeta1,qMCarg,z,omz,gap
        if(shower_mc_mod.eq.'HERWIG6'.or.
     $     shower_mc_mod.eq.'HERWIG7')then
           zeta1=get_zeta(shat_n1,w1,w2,xm12,xm22)
           qMCarg=zeta1*((1-zeta1)*w1-zeta1*xm12)
           if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
           if(qMCarg.lt.-tiny) then
              write(*,*) 'Error 1 in qMC_ileg3: negtive sqrt'
              write(*,*) qMCarg
              stop 1
           endif
           qMC_ileg3=sqrt(qMCarg)
        elseif(shower_mc_mod.eq.'PYTHIA6Q')then
           qMC_ileg3=sqrt(w1+xm12)
        elseif(shower_mc_mod.eq.'PYTHIA6PT')then
           write(*,*)'PYTHIA6PT not available for FSR'
           stop
        elseif(shower_mc_mod.eq.'PYTHIA8')then
           call py8_massive_fsr_fractions(xi_i_fks,y_ij_fks,z,omz)
           gap=xm12/(kn0+kn)+(1d0-y_ij_fks)*kn
           qMC_ileg3=sqrt(z*omz*sqrt(shat_n1)*xi_i_fks*gap)
        endif
      end function qMC_ileg3

      subroutine py8_massive_fsr_fractions(xi_i_fks,y_ij_fks,z,omz)
        implicit none
        double precision :: xi_i_fks,y_ij_fks,z,omz,eminus,gap,emitted
c Undo PYTHIA's massive daughter rescaling using the real radiator.
c E-k=m^2/(E+k) avoids cancellation for a small radiator mass.
c Keep the exact finite-xi terms even when xi is comparable to m^2/shat.
c Share the fractions with zPY8, xiPY8 and xjacPY8 so the damping and
c support scale stays consistent with the radiation variable at endpoints.
        eminus=xm12/(kn0+kn)
        gap=eminus+(1d0-y_ij_fks)*kn
        emitted=sqrt(shat_n1)*xi_i_fks/2d0
        z=(eminus**2+2d0*kn0*kn*(1d0-y_ij_fks))/(2d0*gap*(kn0+emitted))
        omz=(xm12/(2d0*gap)+emitted)/(kn0+emitted)
      end subroutine py8_massive_fsr_fractions

      double precision function qMC_ileg4(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks,zeta2,qMCarg,z,omz
        if(shower_mc_mod.eq.'HERWIG6'.or.shower_mc_mod.eq.'HERWIG7')then
           zeta2=get_zeta(shat_n1,w2,w1,xm22,xm12)
           qMCarg=zeta2*(1d0-zeta2)*w2
           if(qMCarg.lt.0d0.and.qMCarg.ge.-tiny) qMCarg=0d0
           if(qMCarg.lt.-tiny)then
              write(*,*)'Error 1 in qMC_ileg4: negtive sqrt'
              write(*,*)qMCarg
              stop 1
           endif
           qMC_ileg4=sqrt(qMCarg)
        elseif(shower_mc_mod.eq.'PYTHIA6Q')then
           qMC_ileg4=sqrt(w2)
        elseif(shower_mc_mod.eq.'PYTHIA6PT')then
           write(*,*)'PYTHIA6PT not available for FSR'
           stop
        elseif(shower_mc_mod.eq.'PYTHIA8')then
           omz=shat_n1*xi_i_fks/(shat_n1+w2-xm12)
           z=1d0-omz
           qMC_ileg4=sqrt(z*omz*w2)
        endif
      end function qMC_ileg4

      subroutine fill_father_and_ileg(i_fks,j_fks,mass)
        implicit none
        double precision :: mass
        integer :: i_fks,j_fks
        fksfather=min(i_fks,j_fks)
        jmass=mass ! this is the mass of j_fks
c Determine ileg
        call fill_ileg()
      end subroutine fill_father_and_ileg


      subroutine prepare_mc_kinematics(pp,i_fks,j_fks,xi_i_fks,y_ij_fks,mass,include_gfun)
        use fks_phase_space_data, only: veckn_ev,veckbarn_ev,xp0jfks
c takes an n+1-body phase-space point, and fills invariants relevant for
c computation of shower subtraction terms
        implicit none
        double precision,dimension(0:3,next_n1) :: pp
        double precision :: xi_i_fks,y_ij_fks,mass
        logical :: include_gfun
        integer :: i_fks,j_fks
        double precision :: pshower(0:3,next_n1)

        call fill_father_and_ileg(i_fks,j_fks,mass)

        xm12=0d0
        xm22=0d0
        xq1q=0d0
        xq2q=0d0
        kn=veckn_ev
        knbar=veckbarn_ev
        kn0=xp0jfks
        call resonance_shower_frame(pp,i_fks,j_fks,pshower,
     $     kn,knbar,kn0,shat_n1)


c fill the momenta for the recoilers and emitters and emitted.
        call get_momenta_emitter_recoiler(pshower,i_fks,j_fks)

c Determine the Mandelstam invariants needed in the MC functions in terms
c of FKS variables: the argument of MC functions are (p+k)^2, NOT 2 p.k
c
c Definitions of invariants in terms of momenta
c
c xm12 =     xk1 . xk1
c xm22 =     xk2 . xk2
c xtk  = - 2 xp1 . xk3
c xuk  = - 2 xp2 . xk3
c xq1q = - 2 xp1 . xk1 + xm12
c xq2q = - 2 xp2 . xk2 + xm22
c w1   = + 2 xk1 . xk3        = - xq1q + xq2q - xtk
c w2   = + 2 xk2 . xk3        = - xq2q + xq1q - xuk
c xq1c = - 2 xp1 . xk2        = - s - xtk - xq1q + xm12
c xq2c = - 2 xp2 . xk1        = - s - xuk - xq2q + xm22
c
c Parametrisation of invariants in terms of FKS variables
c
c ileg = 1
c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c xk3  =  B * ( 1 , 0 , sqrt(1-yij**2) , yij )
c xk1  =  irrelevant
c xk2  =  irrelevant
c yij = y_ij_fks
c x = 1 - xi_i_fks
c B = sqrt(s)/2*(1-x)
c
c ileg = 2
c xp1  =  sqrt(s)/2 * ( 1 , 0 , 0 , 1 )
c xp2  =  sqrt(s)/2 * ( 1 , 0 , 0 , -1 )
c xk3  =  B * ( 1 , 0 , sqrt(1-yij**2) , -yij )
c xk1  =  irrelevant
c xk2  =  irrelevant
c yij = y_ij_fks
c x = 1 - xi_i_fks
c B = sqrt(s)/2*(1-x)
c
c ileg = 3
c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c xk1  =  ( sqrt(veckn_ev**2+xm12) , 0 , 0 , veckn_ev )
c xk2  =  xp1 + xp2 - xk1 - xk3
c xk3  =  B * ( 1 , 0 , sqrt(1-yij**2) , yij )
c yij = y_ij_fks
c yi = irrelevant
c x = 1 - xi_i_fks
c veckn_ev is such that xk2**2 = xm22
c B = sqrt(s)/2*(1-x)
c azimuth = irrelevant (hence set = 0)
c
c ileg = 4
c xp1  =  sqrt(s)/2 * ( 1 , 0 , sqrt(1-yi**2) , yi )
c xp2  =  sqrt(s)/2 * ( 1 , 0 , -sqrt(1-yi**2) , -yi )
c xk1  =  xp1 + xp2 - xk2 - xk3
c xk2  =  A * ( 1 , 0 , 0 , 1 )
c xk3  =  B * ( 1 , 0 , sqrt(1-yij**2) , yij )
c yij = y_ij_fks
c yi = irrelevant
c x = 1 - xi_i_fks
c A = (s*x-xm12)/(sqrt(s)*(2-(1-x)*(1-yij)))
c B = sqrt(s)/2*(1-x)
c azimuth = irrelevant (hence set = 0)

        if(ileg.eq.1)then
           call fill_invariants_ileg1(xi_i_fks,y_ij_fks)
           call check_invariants_ileg12
        elseif(ileg.eq.2)then
           call fill_invariants_ileg2(xi_i_fks,y_ij_fks)
           call check_invariants_ileg12
        elseif(ileg.eq.3)then
           call fill_invariants_ileg3(xi_i_fks,y_ij_fks)
           call check_invariants_ileg3
        elseif(ileg.eq.4)then
           call fill_invariants_ileg4(xi_i_fks,y_ij_fks)
           call check_invariants_ileg4
        else
           write(*,*)'Error 4 in prepare_mc_kinematics: assigned wrong ileg'
           stop
        endif
        x=1d0-xi_i_fks
        yij=y_ij_fks
        betad=sqrt((1d0-(xm12-xm22)/shat_n1)**2-(4d0*xm22/shat_n1))
        betas=1d0+(xm12-xm22)/shat_n1
        if (include_gfun) call compute_gfun()
      end subroutine prepare_mc_kinematics


      subroutine fill_invariants_ileg1(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        xtk=-shat_n1*xi_i_fks*(1-y_ij_fks)/2d0
        xuk=-shat_n1*xi_i_fks*(1+y_ij_fks)/2d0
      end subroutine fill_invariants_ileg1

      subroutine fill_invariants_ileg2(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        xtk=-shat_n1*xi_i_fks*(1+y_ij_fks)/2d0
        xuk=-shat_n1*xi_i_fks*(1-y_ij_fks)/2d0
      end subroutine fill_invariants_ileg2

      subroutine fill_invariants_ileg3(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        xm12=jmass**2
        xm22=dot(pp_rec,pp_rec)
        xtk=-2d0*dot(xp1,xk3)
        xuk=-2d0*dot(xp2,xk3)
        xq1q=-2d0*dot(xp1,xk1)+xm12
        xq2q=-2d0*dot(xp2,xk2)+xm22
        w1=-xq1q+xq2q-xtk
        w2=-xq2q+xq1q-xuk
      end subroutine fill_invariants_ileg3

      subroutine fill_invariants_ileg4(xi_i_fks,y_ij_fks)
        implicit none
        double precision :: xi_i_fks,y_ij_fks
        xm12=dot(pp_rec,pp_rec)
        xm22=0d0
        xtk=-2d0*dot(xp1,xk3)
        xuk=-2d0*dot(xp2,xk3)
        xij=2d0*(1d0-xm12/shat_n1-xi_i_fks)/(2d0-xi_i_fks*(1d0-y_ij_fks))
        w2=shat_n1*xi_i_fks*xij*(1d0-y_ij_fks)/2d0
        xq2q=-shat_n1*xij*(2d0-dot(xp1,xk2)*4d0/(shat_n1*xij))/2d0
        xq1q=xuk+xq2q+w2
        w1=-xq1q+xq2q-xtk
      end subroutine fill_invariants_ileg4


      subroutine fill_ileg()
        implicit none
c ileg = 1 ==> emission from left     incoming parton
c ileg = 2 ==> emission from right    incoming parton
c ileg = 3 ==> emission from massive  outgoing parton
c ileg = 4 ==> emission from massless outgoing parton
c Instead of jmass, one should use pmass(fksfather), but the
c kernels where pmass(fksfather) != jmass are non-singular
        if(fksfather.le.2 .and. fksfather.gt.0)then
           ileg=fksfather
        elseif(jmass.ne.0d0)then
           ileg=3
        elseif(jmass.eq.0d0)then
           ileg=4
        else
           write(*,*)'Error 1 in get_ileg: unknown ileg'
           write(*,*)ileg,fksfather,jmass
           stop
        endif
        if(ileg.gt.2 .and. shower_mc_mod.eq.'PYTHIA6PT')then
           write (*,*) 'FSR not allowed when matching PY6PT'
           stop 1
        endif
      end subroutine fill_ileg



      subroutine get_momenta_emitter_recoiler(pp,i_fks,j_fks)
        implicit none
        double precision,dimension(0:3,next_n1) :: pp
        integer :: i_fks,j_fks
c Determine and assign momenta:
c xp1 = incoming left parton  (emitter (recoiler) if ileg = 1 (2))
c xp2 = incoming right parton (emitter (recoiler) if ileg = 2 (1))
c xk1 = outgoing parton       (emitter (recoiler) if ileg = 3 (4))
c xk2 = outgoing parton       (emitter (recoiler) if ileg = 4 (3))
c xk3 = extra parton          (FKS parton)
c (xk1 and xk2 are never used for ISR)
        xp1(0:3)=pp(0:3,1)
        xp2(0:3)=pp(0:3,2)
        xk3(0:3)=pp(0:3,i_fks)
        if(ileg.gt.2)pp_rec(0:3)=pp(0:3,1)+pp(0:3,2)-pp(0:3,i_fks)-pp(0:3,j_fks)
        if(ileg.eq.3)then
           xk1(0:3)=pp(0:3,j_fks)
           xk2(0:3)=pp_rec(0:3)
        elseif(ileg.eq.4)then
           xk1(0:3)=pp_rec(0:3)
           xk2(0:3)=pp(0:3,j_fks)
        endif
      end subroutine get_momenta_emitter_recoiler



      subroutine check_invariants_ileg12
        implicit none
        integer,parameter :: max_imprecision=10
        integer,save,dimension(7) :: imprecision=0
        if((abs(xtk+2*dot(xp1,xk3))/shat_n1.ge.tiny).or.
     $     (abs(xuk+2*dot(xp2,xk3))/shat_n1.ge.tiny))then
           write(*,*)'Warning: imprecision 1 in check_invariants_ileg12'
           write(*,*)abs(xtk+2*dot(xp1,xk3))/shat_n1,
     $     abs(xuk+2*dot(xp2,xk3))/shat_n1
           imprecision(1)=imprecision(1)+1
           if (imprecision(1).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' imprecisions. Stopping...'
              stop
           endif
        endif
      end subroutine check_invariants_ileg12

      subroutine check_invariants_ileg3
        implicit none
        integer,parameter :: max_imprecision=10
        integer,save,dimension(7) :: imprecision=0
        if(sqrt(w1+xm12).ge.sqrt(shat_n1)-sqrt(xm22))then
           write(*,*)'Warning: imprecision 2 in check_invariants_ileg3'
           write(*,*)sqrt(w1),sqrt(shat_n1),xm22
           imprecision(2)=imprecision(2)+1
           if (imprecision(2).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' imprecisions. Stopping...'
              stop
           endif
        endif
        if(((abs(w1-2*dot(xk1,xk3))/shat_n1.ge.tiny)).or.
     $     ((abs(w2-2*dot(xk2,xk3))/shat_n1.ge.tiny)))then
           write(*,*)'Warning: imprecision 3 in check_invariants_ileg3'
           write(*,*)abs(w1-2*dot(xk1,xk3))/shat_n1,
     $     abs(w2-2*dot(xk2,xk3))/shat_n1
           imprecision(3)=imprecision(3)+1
           if (imprecision(3).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' imprecisions. Stopping...'
              stop
           endif
        endif
        if(xm12.eq.0d0)then
           write(*,*)'Warning 4 in check_invariants_ileg3'
           imprecision(4)=imprecision(4)+1
           if (imprecision(4).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' warnings. Stopping...'
              stop
           endif
        endif
      end subroutine check_invariants_ileg3

      subroutine check_invariants_ileg4
        implicit none
        integer,parameter :: max_imprecision=10
        integer,save,dimension(7) :: imprecision=0
        if(sqrt(w2).ge.sqrt(shat_n1)-sqrt(xm12))then
           write(*,*)'Warning: imprecision 5 in check_invariants_ileg4'
           write(*,*)sqrt(w2),sqrt(shat_n1),xm12
           imprecision(5)=imprecision(5)+1
           if (imprecision(5).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' imprecisions. Stopping...'
              stop
           endif
        endif
        if(((abs(w2-2*dot(xk2,xk3))/shat_n1.ge.tiny)).or.
     $     ((abs(xq2q+2*dot(xp2,xk2))/shat_n1.ge.tiny)).or.
     $     ((abs(xq1q+2*dot(xp1,xk1)-xm12)/shat_n1.ge.tiny)))then
           write(*,*)'Warning: imprecision 6 in check_invariants_ileg4'
           write(*,*)abs(w2-2*dot(xk2,xk3))/shat_n1,
     $     abs(xq2q+2*dot(xp2,xk2))/shat_n1,
     $     abs(xq1q+2*dot(xp1,xk1)-xm12)/shat_n1
           imprecision(6)=imprecision(6)+1
           if (imprecision(6).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' imprecisions. Stopping...'
              stop
           endif
        endif
        if(xm22.ne.0d0)then
           write(*,*)'Warning 7 in check_invariants_ileg4'
           imprecision(7)=imprecision(7)+1
           if (imprecision(7).ge.max_imprecision) then
              write (*,*) 'Error: ',max_imprecision
     $     ,' warnings. Stopping...'
              stop
           endif
        endif
      end subroutine check_invariants_ileg4

      subroutine compute_gfun()
        implicit none
        include 'fks_powers.inc'
        double precision :: delta
        double precision,parameter :: ymin=0.9d0
        double precision alsf,besf
        common/cgfunsfp/alsf,besf
        double precision alazi,beazi
        common/cgfunazi/alazi,beazi
        if(ileg.le.2)then
           delta=min(1d0,deltaI)
        elseif(ileg.ge.3)then
           delta=min(1d0,deltaO)
        endif
c See for details on how the limits work out e.g. Paolo's PhD thesis
        gfactsf=gfunction(x,alsf,besf,2d0) ! x=1-xi_i_fks, so gfactsf is zero in the soft limit
        gfactcl=gfunction(yij,alsf,-(1d0-ymin),1d0) ! yij=y_ij_fks, so gfactcl is zero in the collinear limit
        gfactazi=0d0
        if(alazi.lt.0d0)gfactazi=1-gfunction(yij,-alazi,beazi,delta)
      end subroutine compute_gfun


      double precision function gfunction(w,alpha,beta,delta)
c Gets smoothly to 0 as w goes to 1.
c Call with
c   alpha > 1, or alpha < 0; if alpha < 0, gfunction = 1;
c   0 < |beta| <= 1;
c   0 < delta <= 2.
        implicit none
        double precision,parameter :: tiny=1d-5,cutoff=1d0,cutoff2=0.99d0
        double precision :: alpha,beta,delta,w,wmin,wg,tt,tmp
        gfunction=1d0
        if(alpha.gt.0d0)then
           if(beta.lt.0d0)then
              wmin=0d0
           else
              wmin=max(0d0,1d0-delta)
           endif
           wg=min(1d0-(1d0-wmin)*abs(beta),cutoff-tiny)
           if(abs(w).gt.wg.and.abs(w).lt.cutoff2)then
              tt=(abs(w)-wg)/(cutoff-wg)
              if(tt.gt.1d0)then
                 write(*,*)'Fatal error in gfunction',tt
                 stop
              endif
              gfunction=(1d0-tt)**(2*alpha)/(tt**(2*alpha)+(1d0-tt)**(2*alpha))
           elseif(abs(w).ge.cutoff2)then
              gfunction=0d0
           endif
        endif
      end function gfunction

      double precision function mc_shower_scale_mass()
c Preserve the legacy massive-FSR bound from the prepared shower state.
c Scale generation receives this value explicitly, avoiding a dependency
c on the counterterm module. Other showers and Born-only paths do not
c read the massive invariant, just as in the original scale prescription.
      use process_module, only: abrv_mod
      implicit none
      mc_shower_scale_mass=0d0
      if (abrv_mod.ne.'born'.and.
     $    shower_mc_mod(1:7).eq.'PYTHIA6') then
         if (ileg.eq.3) mc_shower_scale_mass=sqrt(xm12)
      endif
      end function mc_shower_scale_mass

      end module mc_counterterms
