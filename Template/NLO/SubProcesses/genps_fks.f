      subroutine generate_momenta(ndim,iconfig,wgt,x,p)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
c     Timing profile statistics
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,x(99),p(0:3,nexternal)
      integer iforest(2,-max_branch:-1,lmaxconfigs)
c      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      include 'born_conf.inc'
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),jac
      integer i,j
      double precision zero
      parameter (zero=0d0)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer iconfig0
      common/ciconfig0/iconfig0
      include 'coupl.inc'
      include 'born_props.inc'
c     
      call cpu_time(tBefore)
c
      this_config=iconfig
      iconf=iconfig
      iconfig0=iconfig
      do i=-max_branch,-1
         do j=1,2
            itree(j,i)=iforest(j,i,iconfig)
         enddo
      enddo
      do i=-nexternal,0
         qmass(i)=pmass(i,iconfig)
         qwidth(i)=pwidth(i,iconfig)
      enddo
c
      call generate_momenta_conf(ndim,jac,x,itree,qmass,qwidth,p)
c If the input weight 'wgt' to this subroutine was not equal to one,
c make sure we update all the (counter-event) jacobians and return also
c the updated wgt (i.e. the jacobian for the event)
      do i=-2,2
         jac_cnt(i)=jac_cnt(i)*wgt
      enddo
      wgt=wgt*jac
c
      call cpu_time(tAfter)      
      tGenPS = tGenPS + (tAfter-tBefore)
      return
      end 
      
      subroutine generate_momenta_conf(ndim,jac,x,itree,qmass,qwidth,p)
c
c x(1)...x(ndim-5) --> invariant mass & angles for the Born
c x(ndim-4) --> tau_born
c x(ndim-3) --> y_born
c x(ndim-2) --> xi_i_fks
c x(ndim-1) --> y_ij_fks
c x(ndim) --> phi_i
c
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
c arguments
      integer ndim
      double precision jac,x(99),p(0:3,nexternal)
      integer itree(2,-max_branch:-1)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
c common
c     Arguments have the following meanings:
c     -2 soft-collinear, incoming leg, - direction as in FKS paper
c     -1 collinear, incoming leg, - direction as in FKS paper
c     0 soft
c     1 collinear
c     2 soft-collinear
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_ev(0:3,nexternal)
      common/pev/p_ev
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double complex xij_aor
      common/cxij_aor/xij_aor
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     &                        sqrtshat,shat
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt
      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev
      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      logical softtest,colltest
      common/sctests/softtest,colltest
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      logical nbody
      common/cnbody/nbody
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt
      integer iconfig0,iconfigsave
      common/ciconfig0/iconfig0
      save iconfigsave
c Masses of particles. Should be filled in setcuts.f
      double precision pmass(nexternal)
      common /to_mass/pmass
c     For MINT:
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij
c local
      integer i,j,nbranch,ns_channel,nt_channel,ionebody
     &     ,fksconfiguration,icountevts,imother,ixEi,ixyij,ixpi,isolsign
      double precision M(-max_branch:max_particles),totmassin,totmass
     &     ,stot,xp(0:3,nexternal),pb(0:3,-max_branch:nexternal-1),xjac0
     &     ,tau_born,ycm_born,ycmhat,fksmass,xbjrk_born(2),shat_born
     &     ,sqrtshat_born,S(-max_branch:max_particles),xpswgt0
     &     ,m_born(nexternal-1),m_j_fks,xmrec2,xjac,xpswgt,phi_i_fks
     &     ,tau,ycm,xbjrk(2),xiimax,xinorm,xi_i_fks ,y_ij_fks,flux
     &     ,p_i_fks(0:3),pwgt,p_born_CHECK(0:3,nexternal-1)
      logical one_body,pass,check_cnt
c external
      double precision lambda
      external lambda
c parameters
      logical fks_as_is
      parameter (fks_as_is=.false.)
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      logical firsttime
      data firsttime/.true./
      double precision zero
      parameter (zero=0d0)
c saves
      save m,stot,totmassin,totmass,ns_channel,nt_channel,one_body
     &     ,ionebody,fksmass,nbranch
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-nexternal:-1,-1:1),
     &     cBW_width(-nexternal:-1,-1:1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level

      pass=.true.
      do i=1,nexternal-1
         if (i.lt.i_fks) then
            m(i)=pmass(i)
         else
            m(i)=pmass(i+1)
         endif
      enddo
      if( firsttime .or. iconfig0.ne.iconfigsave ) then
         if (nincoming.eq.2) then
            stot = 4d0*ebeam(1)*ebeam(2)
         else
            stot=pmass(1)**2
         endif
c Make sure have enough mass for external particles
         totmassin=0d0
         do i=1,nincoming
            totmassin=totmassin+m(i)
         enddo
         totmass=0d0
         nbranch = nexternal-3 ! nexternal is for n+1-body, while itree uses n-body
         do i=nincoming+1,nexternal-1
            totmass=totmass+m(i)
         enddo
         fksmass=totmass
         if (stot .lt. max(totmass,totmassin)**2) then
            write (*,*) 'Fatal error #0 in one_tree:'/
     &           /'insufficient collider energy'
            stop
         endif
c Determine number of s- and t-channel branches, at this point it
c includes the s-channel p1+p2
         ns_channel=1
         do while(itree(1,-ns_channel).ne.1 .and.
     &        itree(1,-ns_channel).ne.2 .and. ns_channel.lt.nbranch)
            m(-ns_channel)=0d0                 
            ns_channel=ns_channel+1         
         enddo
         ns_channel=ns_channel - 1
         nt_channel=nbranch-ns_channel-1
c If no t-channles, ns_channels is one less, because we want to exclude
c the s-channel p1+p2
         if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
            ns_channel=ns_channel-1
         endif
c Set one_body to true if it's a 2->1 process at the Born (i.e. 2->2 for the n+1-body)
         if((nexternal-nincoming).eq.2)then
            one_body=.true.
            ionebody=nexternal-1
            ns_channel=0
            nt_channel=0
         elseif((nexternal-nincoming).gt.2)then
            one_body=.false.
         else
            write(*,*)'Error #1 in genps_fks.f',nexternal,nincoming
            stop
         endif
         firsttime=.false.
         iconfigsave=iconfig0
      endif                     ! firsttime
c Set the minimal tau = x1*x2.
      call set_tau_min()
c
      xjac0=1d0
      xpswgt0=1d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Generate Bjorken x's if need be and update jacobian c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      if (abs(lpp(1)).ge.1 .and. abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
c x(ndim-1) -> tau_cnt(0); x(ndim) -> ycm_cnt(0)
         if (one_body) then
c tau is fixed by the mass of the final state particle
            call compute_tau_one_body(totmass,stot,tau_born,xjac0)
         else
            if(nt_channel.eq.0 .and. qwidth(-ns_channel-1).ne.0.d0 .and.
     $           cBW(-ns_channel-1).ne.2)then
c Generate tau according to a Breit-Wiger function
               call generate_tau_BW(stot,x(ndim-4),qmass(-ns_channel-1)
     $              ,qwidth(-ns_channel-1),cBW(-ns_channel-1),cBW_mass(
     $              -ns_channel-1,1),cBW_width(-ns_channel-1,1),tau_born
     $              ,xjac0)
            else
c not a Breit Wigner
               call generate_tau(x(ndim-4),tau_born,xjac0)
            endif
         endif
c Generate the rapditity of the Born system
         call generate_y(tau_born,x(ndim-3),ycm_born,ycmhat,xjac0)
      elseif (abs(lpp(1)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x1 not implemented in one_tree'
         stop
      elseif (abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x2 not implemented in one_tree'
         stop
      else
c No PDFs (also use fixed energy when performing tests)
         call compute_tau_y_epem(j_fks,one_body,fksmass,stot,
     &        tau_born,ycm_born,ycmhat)
         if (j_fks.le.nincoming .and. .not.(softtest.or.colltest)) then
            write (*,*) 'Process has incoming j_fks, but fixed shat: '/
     &           /'not allowed for processes generated at NLO.'
            stop 1
         endif
      endif
c Compute Bjorken x's from tau and y
      xbjrk_born(1)=sqrt(tau_born)*exp(ycm_born)
      xbjrk_born(2)=sqrt(tau_born)*exp(-ycm_born)
c Compute shat and sqrt(shat)
      if(.not.one_body)then
        shat_born=tau_born*stot
        sqrtshat_born=sqrt(shat_born)
      else
c Trivial, but prevents loss of accuracy
        shat_born=fksmass**2
        sqrtshat_born=fksmass
      endif
c Generate the momenta for the initial state of the Born system
      if(nincoming.eq.2) then
        call mom2cx(sqrtshat_born,m(1),m(2),1d0,0d0,pb(0,1),pb(0,2))
      else
         pb(0,1)=sqrtshat_born
         do i=1,2
            pb(i,1)=0d0
         enddo
         p(3,1)=1e-14           ! For HELAS routine ixxxxx for neg. mass
      endif
      s(-nbranch)  = shat_born
      m(-nbranch)  = sqrtshat_born
      pb(0,-nbranch)= m(-nbranch)
      pb(1,-nbranch)= 0d0
      pb(2,-nbranch)= 0d0
      pb(3,-nbranch)= 0d0
c
c Generate Born-level momenta
c
c Start by generating all the invariant masses of the s-channels
      call generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     &     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      if (.not.pass) goto 222
c If only s-channels, also set the p1+p2 s-channel
      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         s(-nbranch+1)=s(-nbranch) 
         m(-nbranch+1)=m(-nbranch)       !Basic s-channel has s_hat 
         pb(0,-nbranch+1) = m(-nbranch+1)!and 0 momentum
         pb(1,-nbranch+1) = 0d0
         pb(2,-nbranch+1) = 0d0
         pb(3,-nbranch+1) = 0d0
      endif
c
c     Next do the T-channel branchings
c
      if (nt_channel.ne.0) then
         call generate_t_channel_branchings(ns_channel,nbranch,itree,m,s
     &        ,x,pb,xjac0,xpswgt0,pass)
         if (.not.pass) goto 222
      endif
c
c     Now generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
      call fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      if (.not.pass) goto 222
c
c  Now I have the Born momenta
c
      do i=1,nexternal-1
         do j=0,3
            p_born(j,i)=pb(j,i)
            p_born_CHECK(j,i)=pb(j,i)
         enddo
         m_born(i)=m(i)
      enddo
      call phspncheck_born(sqrtshat_born,m_born,p_born_CHECK,pass)
      if (.not.pass) then
         xjac0=-142
         goto 222
      endif

c
c
c Here we start with the FKS Stuff
c
c
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
c if event/counterevents will not be generated, the following
c energy components will stay negative. Also set the upper limits of
c the xi ranges to negative values to force crash if something
c goes wrong. The jacobian of the counterevents are set negative
c to prevent using those skipped because e.g. m(j_fks)#0
      p_i_fks_ev(0)=-1.d0
      xiimax_ev=-1.d0
      do i=-2,2
         p_i_fks_cnt(0,i)=-1.d0
         xiimax_cnt(i)=-1.d0
         jac_cnt(i)=-1.d0
      enddo
c set cm stuff to values to make the program crash if not set elsewhere
      ybst_til_tolab=1.d14
      ybst_til_tocm=1.d14
      sqrtshat=0.d0
      shat=0.d0
c if collinear counterevent will not be generated, the following
c quantity will stay zero
      xij_aor=(0.d0,0.d0)
c
c These will correspond to the vegas x's for the FKS variables xi_i,
c y_ij and phi_i
      ixEi=ndim-2
      ixyij=ndim-1
      ixpi=ndim
c Set up the MINT folding:
      ifold_energy=ixEi
      ifold_phi=ixpi
      ifold_yij=ixyij
c
      imother=min(j_fks,i_fks)
      m_j_fks=pmass(j_fks)
c
c For final state j_fks, compute the recoil invariant mass
      if (j_fks.gt.nincoming) then
         call get_recoil(p_born,imother,shat_born,xmrec2,pass)
         if (.not.pass) then
            xjac0=-44
            goto 222
         endif
      endif

c Here is the beginning of the loop over the momenta for the event and
c counter-events. This will fill the xp momenta with the event and
c counter-event momenta.
 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0
c
c Put the Born momenta in the xp momenta, making sure that the mapping
c is correct; put i_fks momenta equal to zero.
      do i=1,nexternal
         if(i.lt.i_fks) then
            do j=0,3
               xp(j,i)=p_born(j,i)
            enddo
            m(i)=m_born(i)
         elseif(i.eq.i_fks) then
            do j=0,3
               xp(j,i)=0d0
            enddo
            m(i)=0d0
         elseif(i.ge.i_fks) then
            do j=0,3
               xp(j,i)=p_born(j,i-1)
            enddo
            m(i)=m_born(i-1)
         endif
      enddo
c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi
c To keep track of the special phase-space region with massive j_fks
      isolsign=0
c
c consider the three cases:
c case 1: j_fks is massless final state
c case 2: j_fks is massive final state
c case 3: j_fks is initial state
      if (j_fks.gt.nincoming) then
         shat=shat_born
         sqrtshat=sqrtshat_born
         tau=tau_born
         ycm=ycm_born
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
         if (m_j_fks.eq.0d0) then
            isolsign=1
            call generate_momenta_massless_final(icountevts,i_fks,j_fks
     &           ,p_born(0,imother),shat,sqrtshat ,x(ixEi),xmrec2,xp
     &           ,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,p_i_fks
     &           ,xjac,xpswgt,pass)
            if (.not.pass) goto 112
         elseif(m_j_fks.gt.0d0) then
            call generate_momenta_massive_final(icountevts,isolsign
     &           ,i_fks,j_fks,p_born(0,imother),shat,sqrtshat,m_j_fks
     &           ,x(ixEi),xmrec2,xp,phi_i_fks,xiimax,xinorm
     &           ,xi_i_fks,y_ij_fks,p_i_fks,xjac,xpswgt,pass)
            if (.not.pass) goto 112
         endif
      elseif(j_fks.le.nincoming) then
         isolsign=1
         call generate_momenta_initial(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp ,x(ixEi)
     &        ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &        ,xi_i_fks,y_ij_fks ,xpswgt,xjac,pass)
         if (.not.pass) goto 112
      else
         write (*,*) 'Error #2 in genps_fks.f',j_fks
         stop
      endif
c At this point, the phase space lacks a factor xi_i_fks, which need be 
c excluded in an NLO computation according to FKS, being taken into 
c account elsewhere
c$$$      xpswgt=xpswgt*xi_i_fks
c
c All done, so check four-momentum conservation
      if(xjac.gt.0.d0)then
         call phspncheck_nocms(nexternal,sqrtshat,m,xp,pass)
         if (.not.pass) then
            xjac=-199
            goto 112
         endif
      endif
c      
      if(nincoming.eq.2)then
         flux  = 1d0 /(2.D0*SQRT(LAMBDA(shat,m(1)**2,m(2)**2)))
      else                      ! Decays
         flux = 1d0/(2d0*sqrtshat)
      endif
c The pi-dependent factor inserted below is due to the fact that the
c weight computed above is relevant to R_n, as defined in Kajantie's
c book, eq.(III.3.1), while we need the full n-body phase space
      flux  = flux / (2d0*pi)**(3 * (nexternal-nincoming) - 4)
c This extra pi-dependent factor is due to the fact that the phase-space
c part relevant to i_fks and j_fks does contain all the pi's needed for 
c the correct normalization of the phase space
      flux  = flux * (2d0*pi)**3
      pwgt=max(xjac*xpswgt,1d-99)
      xjac = pwgt*flux
c
 112  continue
c Catch the points for which there is no viable phase-space generation
c (still fill the common blocks with some information that is needed
c (e.g. ycm_cnt)).
      if (xjac .le. 0d0 ) then
         xp(0,1)=-99d0
      endif
c
c Fill common blocks
      if (icountevts.eq.-100) then
         tau_ev=tau
         ycm_ev=ycm
         shat_ev=shat
         sqrtshat_ev=sqrtshat
         xbjrk_ev(1)=xbjrk(1)
         xbjrk_ev(2)=xbjrk(2)
         xiimax_ev=xiimax
         xinorm_ev=xinorm
         xi_i_fks_ev=xi_i_fks
         do i=0,3
            p_i_fks_ev(i)=p_i_fks(i)
         enddo
         y_ij_fks_ev=y_ij_fks
         do i=1,nexternal
            do j=0,3
               p(j,i)=xp(j,i)
               p_ev(j,i)=xp(j,i)
            enddo
         enddo
         jac=xjac
      else
         tau_cnt(icountevts)=tau
c Special fix in the case the soft counter-events are not generated but
c the Born and real are. (This can happen if ptj>0 in the
c run_card). This fix is needed for set_cms_stuff to work properly.
         if (icountevts.eq.0) then
            ycm=ycm_born
         endif
         ycm_cnt(icountevts)=ycm
         shat_cnt(icountevts)=shat
         sqrtshat_cnt(icountevts)=sqrtshat
         xbjrk_cnt(1,icountevts)=xbjrk(1)
         xbjrk_cnt(2,icountevts)=xbjrk(2)
         xiimax_cnt(icountevts)=xiimax
         xinorm_cnt(icountevts)=xinorm
         xi_i_fks_cnt(icountevts)=xi_i_fks
         do i=0,3
            p_i_fks_cnt(i,icountevts)=p_i_fks(i)
         enddo
         do i=1,nexternal
            do j=0,3
               p1_cnt(j,i,icountevts)=xp(j,i)
            enddo
         enddo
         jac_cnt(icountevts)=xjac
c the following two are obsolete, but still part of some common block:
c so give some non-physical values
         wgt_cnt(icountevts)=-1d99
         pswgt_cnt(icountevts)=-1d99
      endif
c
      if(icountevts.eq.-100)then
         if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
            icountevts=-2
         else
            icountevts=0
         endif
c skips counterevents when integrating over second fold for massive
c j_fks
        if( isolsign.eq.-1 )icountevts=5 
      else
         icountevts=icountevts+1
      endif

      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody) .or.
     &    (icountevts.eq.0.and.m_j_fks.ne.0.d0) )then
         goto 111
      elseif(icountevts.eq.5) then
c icountevts=5 only when integrating over the second fold with j_fks
c massive. The counterevents have been skipped, so make sure their
c momenta are unphysical. Born are physical if event was generated, and
c must stay so for the computation of enhancement factors.
         do i=0,2
            jac_cnt(i)=-299
            p1_cnt(0,1,i)=-99
         enddo
      endif
      nocntevents=(jac_cnt(0).le.0.d0) .and.
     &            (jac_cnt(1).le.0.d0) .and.
     &            (jac_cnt(2).le.0.d0)
      call xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,
     &                  p_i_fks_ev,p_i_fks_cnt,
     &                  xi_i_fks_ev,y_ij_fks_ev,check_cnt)
c check_cnt=.false. is an exceedingly rare situation -- just dump the event
      if(.not.check_cnt)goto 222
c
c If all went well, we are done and can exit
c
      return
c
 222  continue
c
c Born momenta have not been generated. Neither events nor counterevents exist.
c Set all to negative values and exit
      jac=-222
      jac_cnt(0)=-222
      jac_cnt(1)=-222
      jac_cnt(2)=-222
      p(0,1)=-99
      do i=-2,2
        p1_cnt(0,1,i)=-99
      enddo
      p_born(0,1)=-99
      nocntevents=.true.
      return
      end
      

      subroutine generate_momenta_massless_final(icountevts,i_fks,j_fks
     &     ,p_born_imother,shat,sqrtshat,x,xmrec2,xp,phi_i_fks,
     &     xiimax,xinorm,xi_i_fks,y_ij_fks,p_i_fks,xjac,xpswgt,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),phi_i_fks
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision E_i_fks,x3len_i_fks,x3len_j_fks,x3len_fks_mother
     &     ,costh_i_fks,sinth_i_fks,xpifksred(0:3),th_mother_fks
     &     ,costh_mother_fks,sinth_mother_fks, phi_mother_fks
     &     ,cosphi_mother_fks,sinphi_mother_fks,recoil(0:3),sumrec
     &     ,sumrec2,betabst,gammabst,shybst,chybst,chybstmo,xdir(3)
     &     ,veckn,veckbarn,xp_mother(0:3),cosphi_i_fks
     &     ,sinphi_i_fks,xi_i_hat
      double complex resAoR0
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      double precision stiny,sstiny,qtiny,ctiny,cctiny
      double complex ximag
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (ctiny=5d-7)
      parameter (ximag=(0d0,1d0))
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
c
c set-up y_ij_fks
c
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &             (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks=y_ij_fks_fix
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
      else
         write(*,*)'Error #3 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(2)*2d0

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute maximum allowed xi_i_fks
      xiimax=1-xmrec2/shat
      xinorm=xiimax
c
c Define xi_i_fks
c
      if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &     (.not.softtest)  )then
         if(icountevts.eq.-100)then
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->0
            xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         endif
         xi_i_fks=xi_i_hat*xiimax
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        softtest )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xjac=-102
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=xi_i_fks_matrix(icountevts)
      else
         write(*,*)'Error #4 in genps_fks.f',icountevts
         stop
      endif
c remove the following if no importance sampling towards soft
c singularity is performed when integrating over xi_i_hat
      xjac=xjac*2d0*x(1)

c Check that xii is in the allowed range
      if( icountevts.eq.-100 .or. abs(icountevts).eq.1 )then
         if(xi_i_fks.gt.(1-xmrec2/shat))then
            xjac=-101
            pass=.false.
            return
         endif
      elseif(icountevts.eq.0 .or. abs(icountevts).eq.2)then
c May insert here a check on whether xii<xicut, rather than doing it 
c in the cross sections
         continue
      endif
c
c Compute costh_i_fks from xi_i_fks et al.
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      x3len_j_fks=(shat-xmrec2-2*sqrtshat*x3len_i_fks)/
     &             (2*(sqrtshat-x3len_i_fks*(1-y_ij_fks)))
      x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     &                       2*x3len_i_fks*x3len_j_fks*y_ij_fks )
      if(xi_i_fks.lt.qtiny)then
         costh_i_fks=y_ij_fks+shat*(1-y_ij_fks**2)*xi_i_fks/
     &                                          (shat-xmrec2)
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
      elseif(1-y_ij_fks.lt.qtiny)then
         costh_i_fks=1-(shat*(1-xi_i_fks)-xmrec2)**2*(1-y_ij_fks)/
     &                                          (shat-xmrec2)**2
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=1.d0
      else
         costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)
     &               /(2*x3len_fks_mother*x3len_i_fks)
         if(abs(costh_i_fks).gt.1.d0)then
            if(abs(costh_i_fks).le.(1.d0+1.d-5))then
               costh_i_fks=sign(1.d0,costh_i_fks)
            else
               write(*,*)'Fatal error #5 in one_tree',
     &              costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
               stop
            endif
         endif
      endif
      sinth_i_fks=sqrt(1-costh_i_fks**2)
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c The momentum if i_fks and j_fks
c
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c  
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be 
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
      sumrec=recoil(0)+rho(recoil)
      sumrec2=sumrec**2
      betabst=-(shat-sumrec2)/(shat+sumrec2)
      gammabst=1/sqrt(1-betabst**2)
      shybst=-(shat-sumrec2)/(2*sumrec*sqrtshat)
      chybst=(shat+sumrec2)/(2*sumrec*sqrtshat)
c cosh(y) is very often close to one, so define cosh(y)-1 as well
      chybstmo=(sqrtshat-sumrec)**2/(2*sumrec*sqrtshat)
      do j=1,3
         xdir(j)=xp_mother(j)/x3len_fks_mother
      enddo
c Perform the boost here
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Collinear limit of <ij>/[ij]. See innerp3.m. 
      if( ( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) ) )then
         resAoR0=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c The term O(srt(1-y)) is formally correct but may be numerically large
c Set it to zero
c$$$          resAoR5=-ximag*sqrt(2.d0)*
c$$$       &          sinphi_i_fks*tan(th_mother_fks/2.d0)*
c$$$       &          exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c$$$          xij_aor=resAoR0+resAoR5*sqrt(1-y_ij_fks)
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif 
c
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)
      return
      end

      subroutine generate_momenta_massive_final(icountevts,isolsign
     &     ,i_fks,j_fks,p_born_imother,shat,sqrtshat,m_j_fks,x,xmrec2,xp
     &     ,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,p_i_fks,xjac
     &     ,xpswgt,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks,isolsign
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),m_j_fks,phi_i_fks
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision xmj,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2
     $     ,cffDEL2,xiBm,ximax,xirplus,xirminus,rat_xi,xjactmp,xitmp1
     $     ,E_i_fks,x3len_i_fks,b2m4ac,x3len_j_fks_num,x3len_j_fks_den
     $     ,x3len_j_fks,x3len_fks_mother,costh_i_fks,sinth_i_fks
     $     ,xpifksred(0:3),recoil(0:3),xp_mother(0:3),sumrec,expybst
     $     ,shybst,chybst,chybstmo,xdir(3),veckn,veckbarn ,cosphi_i_fks
     $     ,sinphi_i_fks,cosphi_mother_fks,costh_mother_fks
     $     ,phi_mother_fks,sinphi_mother_fks,th_mother_fks,xitmp2
     $     ,sinth_mother_fks,xi_i_hat
      save xjactmp
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      double precision stiny,sstiny,qtiny,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (ctiny=5d-7)
c
      if(colltest .or.
     &     abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
         write(*,*)'Error #5 in genps_fks.f:'
         write(*,*)
     &        'This parametrization cannot be used in FS coll limits'
         stop
      endif
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
c
c set-up y_ij_fks
c
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &             (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks=y_ij_fks_fix
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
      else
         write(*,*)'Error #6 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(2)*2d0

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute the maximum allowed xi_i_fks
c
      xmj=m_j_fks
      xmj2=xmj**2
      xmjhat=xmj/sqrtshat
      xmhat=sqrt(xmrec2)/sqrtshat
      xim=(1-xmhat**2-2*xmjhat+xmjhat**2)/(1-xmjhat)
      cffA2=1-xmjhat**2*(1-y_ij_fks**2)
      cffB2=-2*(1-xmhat**2-xmjhat**2)
      cffC2=(1-(xmhat-xmjhat)**2)*(1-(xmhat+xmjhat)**2)
      cffDEL2=cffB2**2-4*cffA2*cffC2
      xiBm=(-cffB2-sqrt(cffDEL2))/(2*cffA2)
      ximax=1-(xmhat+xmjhat)**2
      if(xiBm.lt.(xim-1.d-8).or.xim.lt.0.d0.or.xiBm.lt.0.d0.or.
     &     xiBm.gt.(ximax+1.d-8).or.ximax.gt.1.or.ximax.lt.0.d0)then
         write(*,*)'Fatal error #4 in one_tree',xim,xiBm,ximax
         xjac=-1d0
         pass=.false.
         return
      endif
      if(y_ij_fks.ge.0.d0)then
         xirplus=xim
         xirminus=0.d0
      else
         xirplus=xiBm
         xirminus=xiBm-xim
      endif
      xiimax=xirplus
      xinorm=xirplus+xirminus
      rat_xi=xiimax/xinorm
c
c Generate xi_i_fks
c
      if( icountevts.eq.-100 .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &      (.not.softtest)  )then
         xjactmp=1.d0
         xitmp1=x(1)
c Map regions (0,A) and (A,1) in xitmp1 onto regions (0,rat_xi) and (rat_xi,1)
c in xi_i_hat respectively. The parameter A is free, but it appears to be 
c convenient to choose A=rat_xi
         if(xitmp1.le.rat_xi)then
            xitmp1=xitmp1/rat_xi
            xjactmp=xjactmp/rat_xi
c importance sampling towards soft singularity
c insert here further importance samplings
            xitmp2=sstiny+(1-sstiny)*xitmp1**2
            xjactmp=xjactmp*2*xitmp1
            xi_i_hat=xitmp2*rat_xi
            xjactmp=xjactmp*rat_xi
            xi_i_fks=xinorm*xi_i_hat
            isolsign=1
         else
c insert here further importance samplings
            xi_i_hat=xitmp1
            xi_i_fks=-xinorm*xi_i_hat+2*xiimax
            isolsign=-1
         endif
      elseif( icountevts.eq.-100 .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
         xjactmp=1.d0
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
         isolsign=1
      elseif( (icountevts.eq.-100) .and.
     &        softtest )then
         xjactmp=1.d0
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xjac=-102
            pass=.false.
            return
         endif
         isolsign=1
      elseif(icountevts.eq.0)then
c Don't set xjactmp here, because we should use the same as what was
c used for the (real-emission) event
         xi_i_fks=xi_i_fks_matrix(icountevts)
         isolsign=1
      else
         write(*,*)'Error #7 in genps_fks.f',icountevts
         stop
      endif
      xjac=xjac*xjactmp
c
      if(isolsign.eq.0)then
         write(*,*)'Fatal error #11 in one_tree',isolsign
         stop
      endif
c
c Compute costh_i_fks
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      b2m4ac=xi_i_fks**2*cffA2 + xi_i_fks*cffB2 + cffC2
      if(b2m4ac.le.0.d0)then
         if(abs(b2m4ac).lt.1.d-8)then
            b2m4ac=0.d0
         else
            write(*,*)'Fatal error #6 in one_tree'
            write(*,*)b2m4ac,xi_i_fks,cffA2,cffB2,cffC2
            write(*,*)y_ij_fks,xim,xiBm
            stop
         endif
      endif
      x3len_j_fks_num=-xi_i_fks*y_ij_fks*
     &                (1-xmhat**2+xmjhat**2-xi_i_fks) +
     &                (2-xi_i_fks)*sqrt(b2m4ac)*isolsign
      x3len_j_fks_den=(2-xi_i_fks*(1-y_ij_fks))*
     &                (2-xi_i_fks*(1+y_ij_fks))
      x3len_j_fks=sqrtshat*x3len_j_fks_num/x3len_j_fks_den
      if(x3len_j_fks.lt.0.d0)then
         write(*,*)'Fatal error #7 in one_tree',
     &        x3len_j_fks_num,x3len_j_fks_den,xi_i_fks,y_ij_fks
         stop
      endif
      x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     &                       2*x3len_i_fks*x3len_j_fks*y_ij_fks )
      if(xi_i_fks.lt.qtiny)then
         costh_i_fks=y_ij_fks+(1-y_ij_fks**2)*xi_i_fks/sqrt(cffC2)
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
      else
         costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)/
     &               (2*x3len_fks_mother*x3len_i_fks)
         if(abs(costh_i_fks).gt.1.d0+qtiny)then
            write(*,*)'Fatal error #8 in one_tree',
     &           costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
            stop
         elseif(abs(costh_i_fks).gt.1.d0)then
            costh_i_fks = sign(1d0,costh_i_fks)
         endif
      endif
      sinth_i_fks=sqrt(1-costh_i_fks**2)
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c Generate momenta for j_fks and i_fks
c     
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2+m_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c  
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be 
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
c
      sumrec=recoil(0)+rho(recoil)
      if(xmrec2.lt.1.d-16*shat)then
         expybst=sqrtshat*sumrec/(shat-xmj2)*
     &           (1+xmj2*xmrec2/(shat-xmj2)**2)
      else
         expybst=sumrec/(2*sqrtshat*xmrec2)*
     &           (shat+xmrec2-xmj2-shat*sqrt(cffC2))
      endif
      if(expybst.le.0.d0)then
         write(*,*)'Fatal error #10 in one_tree',expybst
         stop
      endif
      shybst=(expybst-1/expybst)/2.d0
      chybst=(expybst+1/expybst)/2.d0
      chybstmo=chybst-1.d0
c
      do j=1,3
         xdir(j)=xp_mother(j)/x3len_fks_mother
      enddo
c Boost the momenta
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif 
c
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)
      return
      end


      subroutine generate_momenta_initial(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xpswgt ,xjac ,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2) ,y_ij_fks
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3)
      logical pass
c common blocks
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j,idir
      double precision yijdir,costh_i_fks,x1bar2,x2bar2,yij_sol,xi1,xi2
     $     ,ximaxtmp,omega,bstfact,shy_tbst,chy_tbst,chy_tbstmo
     $     ,xdir_t(3),cosphi_i_fks,sinphi_i_fks,shy_lbst,chy_lbst
     $     ,encmso2,E_i_fks,sinth_i_fks,xpifksred(0:3),xi_i_fks
     $     ,xi_i_hat,xiimin,yij_upp,yij_low,y_ij_fks_upp,y_ij_fks_low
      save xi_i_hat
      double complex resAoR0
c external
c
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      logical fks_as_is
      parameter (fks_as_is=.false.)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision stiny,sstiny,qtiny,zero,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (zero=0d0)
      parameter (ctiny=5d-7)
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
c
c FKS for left or right incoming parton
c
      idir=0
      if(.not.fks_as_is)then
         if(j_fks.eq.1)then
            idir=1
         elseif(j_fks.eq.2)then
            idir=-1
         endif
      else
         idir=1
         write(*,*)'One_tree: option not checked'
         stop
      endif
c
c set-up lower and upper bounds on y_ij_fks
c
      if( tau_born.le.tau_lower_bound .and.ycm_born.gt.
     &         (0.5d0*log(tau_born)-log(tau_lower_bound)) )then
         yij_upp= (tau_lower_bound+tau_born)*
     &        ( 1-exp(2*ycm_born)*tau_lower_bound ) /
     &                  ( (tau_lower_bound-tau_born)*
     &                    (1+exp(2*ycm_born)*tau_lower_bound) )
      else
         yij_upp=1.d0
      endif
      if( tau_born.le.tau_lower_bound .and. ycm_born.lt.
     &        (-0.5d0*log(tau_born)+log(tau_lower_bound)) )then
         yij_low=-(tau_lower_bound+tau_born)*
     &        ( 1-exp(-2*ycm_born)*tau_lower_bound ) / 
     &                   ( (tau_lower_bound-tau_born)*
     &                     (1+exp(-2*ycm_born)*tau_lower_bound) )
      else
         yij_low=-1.d0
      endif
c
      if(idir.eq.1)then
         y_ij_fks_upp=yij_upp
         y_ij_fks_low=yij_low
      elseif(idir.eq.-1)then
         y_ij_fks_upp=-yij_low
         y_ij_fks_low=-yij_upp
      endif
      
c
c set-up y_ij_fks
c
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &            (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = y_ij_fks_upp -
     &        (y_ij_fks_upp-y_ij_fks_low)*(cctiny+(1-cctiny)*x(2)**2)
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks = y_ij_fks_fix
         if ( y_ij_fks.gt.y_ij_fks_upp+1d-12 .or.
     &        y_ij_fks.lt.y_ij_fks_low-1d-12) then
            xjac=-33d0
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
c Check that y_ij_fks is in the allowed range. If not, counter events
c cannot be generated
         if ( y_ij_fks.gt.y_ij_fks_upp+1d-12 .or.
     &        y_ij_fks.lt.y_ij_fks_low-1d-12) then
            xjac=-33d0
            pass=.false.
            return
         endif
      else
         write(*,*)'Error #8 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*(y_ij_fks_upp-y_ij_fks_low)*x(2)*2d0
c
c Compute costh_i_fks
c
      yijdir=idir*y_ij_fks
      costh_i_fks=yijdir
c
c Compute maximal xi_i_fks
c
      x1bar2=xbjrk_born(1)**2
      x2bar2=xbjrk_born(2)**2
      if(1-tau_born.gt.1.d-5)then
         yij_sol=-sinh(ycm_born)*(1+tau_born)/
     &            ( cosh(ycm_born)*(1-tau_born) )
      else
         yij_sol=-ycmhat
      endif
      if(abs(yij_sol).gt.1.d0)then
         write(*,*)'Error #9 in genps_fks.f',yij_sol,icountevts
         write(*,*)xbjrk_born(1),xbjrk_born(2),yijdir
      endif
      if(yijdir.ge.yij_sol)then
         xi1=2*(1+yijdir)*x1bar2/(
     &        sqrt( ((1+x1bar2)*(1-yijdir))**2+16*yijdir*x1bar2 ) +
     &        (1-yijdir)*(1-x1bar2) )
         ximaxtmp=1-xi1
      elseif(yijdir.lt.yij_sol)then
         xi2=2*(1-yijdir)*x2bar2/(
     &        sqrt( ((1+x2bar2)*(1+yijdir))**2-16*yijdir*x2bar2 ) +
     &        (1+yijdir)*(1-x2bar2) )
         ximaxtmp=1-xi2
      else
         write(*,*)'Fatal error #14 in one_tree: unknown option'
         write(*,*)y_ij_fks,yij_sol,idir
         stop
      endif
      xiimax=ximaxtmp
c
c Lower bound on xi_i_fks
c
      if (tau_born.lt.tau_lower_bound) then
         xiimin=1d0-tau_born/tau_lower_bound
      else
         xiimin=0d0
      endif
      if (xiimax.lt.xiimin) then
         write (*,*) 'WARNING #10 in genps_fks.f',icountevts,xiimax
     $        ,xiimin
         xjac=-342d0
         pass=.false.
         return
      endif

      xinorm=xiimax-xiimin
      if( icountevts.ge.1 .and.
     &     ( (idir.eq.1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(1))).gt.1.d-5) .or.
     &     (idir.eq.-1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(2))).gt.1.d-5) ) )then 
         write(*,*)'Fatal error #15 in one_tree'
         write(*,*)ximaxtmp,xbjrk_born(1),xbjrk_born(2),idir
         stop
      endif
c
c Define xi_i_fks
c
      if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &     (.not.softtest)  )then
         if(icountevts.eq.-100)then
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->0
            xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         endif
c$$$         xi_i_fks=xi_i_hat*xiimax
         xi_i_fks=xiimin+(xiimax-xiimin)*xi_i_hat
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        softtest )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xjac=-102
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=xi_i_fks_matrix(icountevts)
c Check that xi_i_fks is in the allowed range. If not, counter events
c cannot be generated
         if ( xi_i_fks.gt.xiimax+1d-12 .or.
     &        xi_i_fks.lt.xiimin-1d-12 ) then
            xjac=-34d0
            pass=.false.
            return
         endif
      else
         write(*,*)'Error #11 in genps_fks.f',icountevts
         stop
      endif
c remove the following if no importance sampling towards soft
c singularity is performed when integrating over xi_i_hat
      xjac=xjac*2d0*x(1)
c
c Initial state variables are different for events and counterevents. Update them here.
c
      omega=sqrt( (2-xi_i_fks*(1+yijdir))/
     &     (2-xi_i_fks*(1-yijdir)) )
      if (icountevts.ne.0) then
         tau=tau_born/(1-xi_i_fks)
         ycm=ycm_born-log(omega)
         shat=tau*stot
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)/(sqrt(1-xi_i_fks)*omega)
         xbjrk(2)=xbjrk_born(2)*omega/sqrt(1-xi_i_fks)
      else
         tau=tau_born
         ycm=ycm_born
         shat=shat_born
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
      endif
c
c Define the boost factor here
c
      bstfact=sqrt( (2-xi_i_fks*(1-yijdir))*(2-xi_i_fks*(1+yijdir)) )
      shy_tbst=-xi_i_fks*sqrt(1-yijdir**2)/(2*sqrt(1-xi_i_fks))
      chy_tbst=bstfact/(2*sqrt(1-xi_i_fks))
      chy_tbstmo=chy_tbst-1.d0
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xdir_t(1)=-cosphi_i_fks
      xdir_t(2)=-sinphi_i_fks
      xdir_t(3)=zero
c
      shy_lbst=-xi_i_fks*yijdir/bstfact
      chy_lbst=(2-xi_i_fks)/bstfact
c Boost the momenta
      do i=3,nexternal
         if(i.ne.i_fks.and.shy_tbst.ne.0.d0)
     &        call boostwdir2(chy_tbst,shy_tbst,chy_tbstmo,xdir_t,
     &                        xp(0,i),xp(0,i))
      enddo
c
      encmso2=sqrtshat/2.d0
      p_i_fks(0)=encmso2
      E_i_fks=xi_i_fks*encmso2
      sinth_i_fks=sqrt(1-costh_i_fks**2)
c
      xp(0,1)=encmso2*(chy_lbst-shy_lbst)
      xp(1,1)=0.d0
      xp(2,1)=0.d0
      xp(3,1)=xp(0,1)
c
      xp(0,2)=encmso2*(chy_lbst+shy_lbst)
      xp(1,2)=0.d0
      xp(2,2)=0.d0
      xp(3,2)=-xp(0,2)
c
      xp(0,i_fks)=E_i_fks*(chy_lbst-shy_lbst*yijdir)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks    
      xpifksred(2)=sinth_i_fks*sinphi_i_fks    
      xpifksred(3)=chy_lbst*yijdir-shy_lbst
c
      do j=1,3
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         p_i_fks(j)=encmso2*xpifksred(j)
      enddo
c
c Collinear limit of <ij>/[ij]. See innerpin.m. 
      if( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) )then
         resAoR0=-exp( 2*idir*ximag*phi_i_fks )
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii) * (tau,ycm)
      xpswgt=xpswgt*shat
      xpswgt=xpswgt/(4*pi)**3/(1-xi_i_fks)
      xpswgt=abs(xpswgt)
c
      return
      end

         
      subroutine getangles(pin,th,cth,sth,phi,cphi,sphi)
      implicit none
      real*8 pin(0:3),th,cth,sth,phi,cphi,sphi,xlength
c
      xlength=pin(1)**2+pin(2)**2+pin(3)**2
      if(xlength.eq.0)then
        th=0.d0
        cth=1.d0
        sth=0.d0
        phi=0.d0
        cphi=1.d0
        sphi=0.d0
      else
        xlength=sqrt(xlength)
        cth=pin(3)/xlength
        th=acos(cth)
        if(cth.ne.1.d0)then
          sth=sqrt(1-cth**2)
          phi=atan2(pin(2),pin(1))
          cphi=cos(phi)
          sphi=sin(phi)
        else
          sth=0.d0
          phi=0.d0
          cphi=1.d0
          sphi=0.d0
        endif
      endif
      return
      end


      function bwfunc(s,xm02,gah)
c Returns the Breit Wigner function, normalized in such a way that
c its integral in the range (-inf,inf) is one
      implicit none
      real*8 bwfunc,s,xm02,gah
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      bwfunc=xm0*gah/(pi*((s-xm02)**2+xm02*gah**2))
      return
      end


      function xbwmass3(t,xm02,ga,bwdelf,bwfmmn)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmass3,t,xm02,ga,bwdelf,bwfmmn
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      xbwmass3=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
      return
      end


      subroutine gentcms(pa,pb,t,phi,m1,m2,p1,pr,jac)
c*************************************************************************
c     Generates 4 momentum for particle 1, and remainder pr
c     given the values t, and phi
c     Assuming incoming particles with momenta pa, pb
c     And outgoing particles with mass m1,m2
c     s = (pa+pb)^2  t=(pa-p1)^2
c*************************************************************************
      implicit none
c
c     Arguments
c
      double precision t,phi,m1,m2               !inputs
      double precision pa(0:3),pb(0:3),jac
      double precision p1(0:3),pr(0:3)           !outputs
c
c     local
c
      double precision ptot(0:3),E_acms,p_acms,pa_cms(0:3)
      double precision esum,ed,pp,md2,ma2,pt,ptotm(0:3)
      integer i
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----
      do i=0,3
         ptot(i)  = pa(i)+pb(i)
         if (i .gt. 0) then
            ptotm(i) = -ptot(i)
         else
            ptotm(i) = ptot(i)
         endif
      enddo
      ma2 = dot(pa,pa)
c
c     determine magnitude of p1 in cms frame (from dhelas routine mom2cx)
c
      ESUM = sqrt(max(0d0,dot(ptot,ptot)))
      if (esum .eq. 0d0) then
         jac=-8d0             !Failed esum must be > 0
         return
      endif
      MD2=(M1-M2)*(M1+M2)
      ED=MD2/ESUM
      IF (M1*M2.EQ.0.) THEN
         PP=(ESUM-ABS(ED))*0.5d0
      ELSE
         PP=(MD2/ESUM)**2-2.0d0*(M1**2+M2**2)+ESUM**2
         if (pp .gt. 0) then
            PP=SQRT(pp)*0.5d0
         else
            write(*,*) 'Warning #12 in genps_fks.f',pp
            jac=-1
            return
         endif
      ENDIF
c
c     Energy of pa in pa+pb cms system
c
      call boostx(pa,ptotm,pa_cms)
      E_acms = pa_cms(0)
      p_acms = dsqrt(pa_cms(1)**2+pa_cms(2)**2+pa_cms(3)**2)
c
      p1(0) = MAX((ESUM+ED)*0.5d0,0.d0)
      p1(3) = -(m1*m1+ma2-t-2d0*p1(0)*E_acms)/(2d0*p_acms)
      pt = dsqrt(max(pp*pp-p1(3)*p1(3),0d0))
      p1(1) = pt*cos(phi)
      p1(2) = pt*sin(phi)
c
      call rotxxx(p1,pa_cms,p1)          !Rotate back to pa_cms frame
      call boostx(p1,ptot,p1)            !boost back to lab fram
      do i=0,3
         pr(i)=pa(i)-p1(i)               !Return remainder of momentum
      enddo
      end


      DOUBLE PRECISION FUNCTION LAMBDA(S,MA2,MB2)
      IMPLICIT NONE
C****************************************************************************
C     THIS IS THE LAMBDA FUNCTION FROM VERNONS BOOK COLLIDER PHYSICS P 662
C     MA2 AND MB2 ARE THE MASS SQUARED OF THE FINAL STATE PARTICLES
C     2-D PHASE SPACE = .5*PI*SQRT(1.,MA2/S^2,MB2/S^2)*(D(OMEGA)/4PI)
C****************************************************************************
      DOUBLE PRECISION MA2,MB2,S,tiny,tmp,rat
      parameter (tiny=1.d-8)
c
      tmp=S**2+MA2**2+MB2**2-2d0*S*MA2-2d0*MA2*MB2-2d0*S*MB2
      if(tmp.le.0.d0)then
        if(ma2.lt.0.d0.or.mb2.lt.0.d0)then
          write(6,*)'Error #1 in function Lambda:',s,ma2,mb2
          stop
        endif
        rat=1-(sqrt(ma2)+sqrt(mb2))/s
        if(rat.gt.-tiny)then
          tmp=0.d0
        else
          write(6,*)'Error #2 in function Lambda:',s,ma2,mb2,rat
        endif
      endif
      LAMBDA=tmp
      RETURN
      END


      SUBROUTINE YMINMAX(X,Y,Z,U,V,W,YMIN,YMAX)
C**************************************************************************
C     This is the G function from Particle Kinematics by
C     E. Byckling and K. Kajantie, Chapter 4 p. 91 eqs 5.28
C     It is used to determine physical limits for Y based on inputs
C**************************************************************************
      implicit none
c
c     Constant
c
      double precision tiny
      parameter       (tiny=1d-199)
c
c     Arguments
c
      Double precision x,y,z,u,v,w              !inputs  y is dummy
      Double precision ymin,ymax                !output
c
c     Local
c
      double precision y1,y2,yr,ysqr
c     
c     External
c
      double precision lambda
c-----
c  Begin Code
c-----
      ysqr = lambda(x,u,v)*lambda(x,w,z)
      if (ysqr .ge. 0d0) then
         yr = dsqrt(ysqr)
      else
         print*,'Error in yminymax sqrt(-x)',lambda(x,u,v),lambda(x,w,z)
         yr=0d0
      endif
      y1 = u+w -.5d0* ((x+u-v)*(x+w-z) - yr)/(x+tiny)
      y2 = u+w -.5d0* ((x+u-v)*(x+w-z) + yr)/(x+tiny)
      ymin = min(y1,y2)
      ymax = max(y1,y2)
      end


      subroutine compute_tau_one_body(totmass,stot,tau,jac)
      implicit none
      double precision totmass,stot,tau,jac,roH
      roH=totmass**2/stot
      tau=roH
c Jacobian due to delta() of tau_born
      jac=jac*2*totmass/stot
      return
      end


      subroutine generate_tau_BW(stot,x,mass,width,cBW,BWmass,BWwidth
     $     ,tau,jac)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      integer nsamp
      parameter (nsamp=1)
      include 'run.inc'
      include 'genps.inc'
      integer cBW,icount
      data icount /0/
      double precision stot,x,tau,jac,mass,width,BWmass,BWwidth,m,w,a,b
      double precision smax,smin,xm02,stemp,bwmdpl,bwmdmn,bwfmpl,bwfmmn
      double precision bwdelf,xbwmass3,bwfunc,x0
      external xmwmass3,bwfunc
      double precision tau_Born_lower_bound_save
     $     ,tau_lower_bound_resonance_save,tau_lower_bound_save
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      if (cBW.eq.1 .and. width.gt.0d0 .and. BWwidth.gt.0d0) then
c conflicting Breit-Wigner
c     use the ratio of the widths to determine which is the most-likely
c     one to go on-shell
         a=1d0-width/(width+BWwidth)
c     use the mass separation to determine which region of phase-space
c     should be generated by which BW
         if (BWmass.le.mass) then
            write (*,*) 'Error in generate_tau_BW #1',mass,BWmass
            stop
         endif
         b=(BWmass-mass)/(width+BWwidth)
         b=mass+b*width
         b=b**2
         if (x.lt.0.5d0) then
c the resonace BW
            m=mass
            w=width
            x0=x*2d0
            jac=jac*2d0
            smax=b
            smin=tau_Born_lower_bound*stot
         else
c the alternative "BW". It's a rather flat distribution peaked above
c BWmass: we can use the normal generate_tau with the
c 'tau_lower_bound_resonance' equal to the BW mass, and
c 'tau_lower_bound' equal to the parameter 'b' (with the correct
c normalization)
            x0=2d0*x-1d0
            jac=jac*2d0
c     save default tau's
            tau_lower_bound_resonance_save=tau_lower_bound_resonance
            tau_Born_lower_bound_save=tau_Born_lower_bound
            tau_lower_bound_save=tau_lower_bound
c     overwrite the tau's
            tau_Born_lower_bound=b/stot
            tau_lower_bound=tau_Born_lower_bound
            tau_lower_bound_resonance=BWmass**2/stot
c     Generate tau
            call generate_tau(x0,tau,jac)
c     restore default tau's
            tau_lower_bound_resonance=tau_lower_bound_resonance_save
            tau_Born_lower_bound=tau_Born_lower_bound_save
            tau_lower_bound=tau_lower_bound_save
            return      ! tau and jac generate: exit this function
         endif
      elseif (cBW.eq.0) then
c Normal Breit-Wigner
         m=mass
         w=width
         x0=x
         smax=stot
         smin=tau_Born_lower_bound*stot
      else
         write (*,*) 'Error in generate_tau_BW #2',cBW
         stop
      endif
      xm02=m**2
      bwmdpl=smax-xm02
      bwmdmn=xm02-smin
      bwfmpl=atan(bwmdpl/(m*w))
      bwfmmn=atan(bwmdmn/(m*w))
      bwdelf=(bwfmpl+bwfmmn)/pi
      stemp=xbwmass3(x0,xm02,w,bwdelf,bwfmmn)
      jac=jac*bwdelf/bwfunc(stemp,xm02,w)
      tau=stemp/stot
      jac=jac/stot
      return
      end


      subroutine generate_tau(x,tau,jac)
      double precision x,tau,jac
      double precision roH,roHs,fract,ximax0,ximin0,tmp,fract1,fract2
     &     ,roHj
      integer nsamp
      parameter (nsamp=1)
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      character*4 abrv
      common /to_abrv/ abrv
c The only possible order for the lower bounds on tau is as follows:
c tau_Born_lower_bound <= tau_lower_bound <= tau_lower_bound_resonance
c If the order is different, we should quit.
      if ( tau_Born_lower_bound.ge.tau_lower_bound+1d8 .or.
     &     tau_Born_lower_bound.ge.tau_lower_bound_resonance+1d8 .or.
     &     tau_lower_bound.ge.tau_lower_bound_resonance+1d8 ) then
         write (*,*) 'Errors on the tau_Born_lower_bound'
     &        ,tau_Born_lower_bound,tau_lower_bound
     &        ,tau_lower_bound_resonance
         stop
      endif
      roH=tau_Born_lower_bound
      roHj=tau_lower_bound
      roHs=tau_lower_bound_resonance
c User x below 'fract1' for phase-space region below jet cut-off
      if (tau_lower_bound.le.tau_Born_lower_bound +1d-8) then
         fract1=0d0
      else
         fract1=0.05d0
      endif
c User x between 'fract1' and 'fract2' for phase-space region between
c jet and soft cut-off
      if(tau_lower_bound_resonance.le.tau_lower_bound + 1d-8)then
         fract2=0.0d0
      else
         fract2=0.05d0
      endif
      if (x.lt.fract1) then
c Use x^2 importance sampling below jet cut-off
         if (fract1.lt.1d-5) then
            write (*,*) 'fract1 is too small'
            stop
         endif
         tau=roHj-(roHj-roH)*(1d0-x/fract1)**2
         if (tau.le.1d-8) tau=1d-8 ! avoid numerical pathologies
         jac=jac*(roHj-roH)*2d0*(1d0-x/fract1)/fract1
         if (abrv.eq.'grid') then
c If we are setting-up the integration grids, using 'grid' (i.e. only
c the Born), we have to make sure that we have a viable PS point also
c when tau_born < tau_lower_bound. Simply set tau equal to the lower
c bound. Don't worry about putting a correct Jacobian, because the
c 'grid' cross section is non-physical anyway. We can use the Jacobian
c to enhance a bit the region close to the cut (and compensate for the
c Jacobian above, which goes in the other direction).
            tau=roHj+1d-8
            jac=jac*(x/fract1)**2
         endif
      elseif (x.lt.fract1+fract2) then
c Flat grid below soft cut-off
         if (fract2.lt.1d-5) then
            write (*,*) 'fract2 is too small'
            stop
         endif
         tau=roHj-(roHs-roHj)*(fract1-x)/fract2
         jac=jac*(roHs-roHj)/fract2
      else
c Use 1/x^(nsamp) importance sampling above soft cut-off
         fract=fract1+fract2
         ximax0 = roHs**(-nsamp)
         ximin0 = 1.d0
         tmp  = ximin0 +(1d0-(x-fract)/(1d0-fract))*(ximax0-ximin0)
         tau = tmp**(-1/dble(nsamp))
         jac= jac/nsamp*tau**(nsamp+1)*
     &        (ximax0-ximin0)/(1d0-fract)
      endif
      return
      end


      subroutine generate_y(tau,x,ycm,ycmhat,jac)
      implicit none
      double precision tau,x,ycm,jac
      double precision ylim,ycmhat
      ylim=-0.5d0*log(tau)
      ycmhat=2*x-1
      ycm=ylim*ycmhat
      jac=jac*ylim*2
      return
      end

      
      subroutine compute_tau_y_epem(j_fks,one_body,fksmass,
     &                              stot,tau,ycm,ycmhat)
      implicit none
      include 'nexternal.inc'
      integer j_fks
      logical one_body
      double precision fksmass,stot,tau,ycm,ycmhat
      if(j_fks.le.nincoming)then
c This should never happen in normal integration: when no PDFs, j_fks
c cannot be initial state (but needed for testing). If tau set to one,
c integration range in xi_i_fks will be zero, so lower it artificially
c when too large
         if(one_body)then
            tau=fksmass**2/stot
         else
            tau=max((0.85d0)**2,fksmass**2/stot)
         endif
         ycm=0.d0
      else
c For e+e- collisions, set tau to one and y to zero
         tau=1.d0
         ycm=0.d0
      endif
      ycmhat=0.d0
      return
      end

      
      subroutine generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     &     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      integer ns_channel
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision sqrtshat_born,totmass,xjac0
      integer itree(2,-max_branch:-1)
      integer i,j,nsamp
      parameter (nsamp=1)
      double precision smin,smax,xm02,bwmdpl,bwmdmn,bwfmpl,bwfmmn,bwdelf
     &     ,totalmass,tmp,ximin0,ximax0
      double precision xbwmass3,bwfunc
      external xbwmass3,bwfunc
      logical pass
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-nexternal:-1,-1:1),
     &     cBW_width(-nexternal:-1,-1:1)
      double precision b(-1:1),x0
      double precision s_mass(-nexternal:-1),xi,fract
      parameter (fract=0.1d0)
      common/to_phase_space_s_channel/s_mass
      pass=.true.
      totalmass=totmass
      do i = -1,-ns_channel,-1
c Generate invariant masses for all s-channel branchings of the Born
         smin = (m(itree(1,i))+m(itree(2,i)))**2
         smax = (sqrtshat_born-totalmass+sqrt(smin))**2
         if(smax.lt.smin.or.smax.lt.0.d0.or.smin.lt.0.d0)then
            write(*,*)'Error #13 in genps_fks.f'
            write(*,*)smin,smax,i
            stop
         endif
c Choose the appropriate s given our constraints smin,smax
         if(qwidth(i).ne.0.d0 .and. cBW(i).ne.2)then
c Breit Wigner
            if (cBW(i).eq.1 .and.
     &          cBW_width(i,1).gt.0d0 .and. cBW_width(i,-1).gt.0d0) then
c     conflicting BW on both sides
               do j=-1,1,2
                  b(j)=(cBW_mass(i,j)-qmass(i))/
     &                 (qwidth(i)+cBW_width(i,j))
                  b(j)=qmass(i)+b(j)*qwidth(i)
                  b(j)=b(j)**2
               enddo
               if (x(-i).lt.1d0/3d0) then
                  x0=3d0*x(-i)
                  s(i)=(b(-1)-smin)*x0+smin
                  xjac0=3d0*xjac0*(b(-1)-smin)
               elseif (x(-i).gt.1d0/3d0 .and. x(-i).lt.2d0/3d0) then
                  x0=3d0*x(-i)-1d0
                  xm02=qmass(i)**2
                  bwmdpl=b(1)-xm02
                  bwmdmn=xm02-b(-1)
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=3d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               else
                  x0=3d0*x(-i)-2d0
                  s(i)=(smax-b(1))*x0+b(1)
                  xjac0=3d0*xjac0*(smax-b(1))
               endif
            elseif (cBW(i).eq.1.and.cBW_width(i,1).gt.0d0) then
c     conflicting BW with alternative mass larger
               b(1)=(cBW_mass(i,1)-qmass(i))/
     &              (qwidth(i)+cBW_width(i,1))
               b(1)=qmass(i)+b(1)*qwidth(i)
               b(1)=b(1)**2
               if (x(-i).lt.0.5d0) then
                  x0=2d0*x(-i)
                  xm02=qmass(i)**2
                  bwmdpl=b(1)-xm02
                  bwmdmn=xm02-smin
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=2d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               else
                  x0=2d0*x(-i)-1d0
                  s(i)=(smax-b(1))*x0+b(1)
                  xjac0=2d0*xjac0*(smax-b(1))
               endif
            elseif (cBW(i).eq.1.and.cBW_width(i,-1).gt.0d0) then
c     conflicting BW with alternative mass smaller
               b(-1)=(cBW_mass(i,-1)-qmass(i))/
     &              (qwidth(i)+cBW_width(i,-1)) ! b(-1) is negative here
               b(-1)=qmass(i)+b(-1)*qwidth(i)
               b(-1)=b(-1)**2

               if (b(-1).gt.smax) then
                   s(i)=(smax-smin)*x(-i)+smin
                   xjac0=xjac0*(smax-smin)
               elseif (x(-i).lt.0.5d0) then
                  x0=2d0*x(-i)
                  s(i)=(b(-1)-smin)*x0+smin
                  xjac0=2d0*xjac0*(b(-1)-smin)
               else
                  x0=2d0*x(-i)-1d0
                  xm02=qmass(i)**2
                  bwmdpl=smax-xm02
                  bwmdmn=xm02-b(-1)
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=2d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               endif
            else
c     normal BW
               xm02=qmass(i)**2
               bwmdpl=smax-xm02
               bwmdmn=xm02-smin
               bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
               bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
               bwdelf=(bwfmpl+bwfmmn)/pi
               s(i)=xbwmass3(x(-i),xm02,qwidth(i),bwdelf
     &              ,bwfmmn)
               xjac0=xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
            endif
         else
c not a Breit Wigner
            if (smin.eq.0d0 .and. s_mass(i).eq.0d0) then
c     no lower limit on invariant mass from cuts or final state masses:
c     use flat distribution
               s(i) = (smax-smin)*x(-i)+smin
               xjac0 = xjac0*(smax-smin)
            elseif (smin.ge.s_mass(i) .and. smin.gt.0d0) then
c     A lower limit on smin, which is larger than lower limit from cuts
c     or masses. Use 1/x^nsamp importance sampling
               ximax0 = smin**(-nsamp)
               ximin0 = smax**(-nsamp)
               tmp  = ximin0 +(1d0-x(-i))*(ximax0-ximin0)
               s(i) = tmp**(-1/dble(nsamp))
               xjac0= xjac0/nsamp*s(i)**(nsamp+1)*(ximax0-ximin0)
            elseif (smin.lt.s_mass(i) .and. s_mass(i).gt.0d0) then
c     Use flat grid between smin and s_mass(i), and 1/x^nsamp above
c     s_mass(i)
               if (x(-i).lt.fract) then
                  xi=x(-i)/fract ! between 0 and 1
                  xjac0=xjac0/fract
                  s(i) = (s_mass(i)-smin)*xi+smin
                  xjac0 = xjac0*(s_mass(i)-smin)
               else
                  xi=(x(-i)-fract)/(1d0-fract) ! between 0 and 1
                  xjac0=xjac0/(1d0-fract)
                  ximax0 = s_mass(i)**(-nsamp)
                  ximin0 = smax**(-nsamp)
                  tmp  = ximin0 +(1d0-xi)*(ximax0-ximin0)
                  s(i) = tmp**(-1/dble(nsamp))
                  xjac0= xjac0/nsamp*s(i)**(nsamp+1)*(ximax0-ximin0)
               endif
            else
               write (*,*) "ERROR in genps_fks.f:"/
     $              /" cannot set s-channel without BW"
               stop 1
            endif
         endif
c If numerical inaccuracy, quit loop
         if (xjac0 .lt. 0d0) then
            xjac0 = -6
            pass=.false.
            return
         endif
         if (s(i) .lt. smin) then
            xjac0=-5
            pass=.false.
            return
         endif
c
c     fill masses, update totalmass
c
         m(i) = sqrt(s(i))
         totalmass=totalmass+m(i)-
     &        m(itree(1,i))-m(itree(2,i))
         if ( totalmass.gt.sqrtshat_born )then
            xjac0 = -4
            pass=.false.
            return
         endif
      enddo
      return
      end


      subroutine generate_t_channel_branchings(ns_channel,nbranch,itree
     &     ,m,s,x,pb,xjac0,xpswgt0,pass)
c First we need to determine the energy of the remaining particles this
c is essentially in place of the cos(theta) degree of freedom we have
c with the s channel decay sequence
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac0,xpswgt0
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      integer ns_channel,nbranch
      logical pass
c
      double precision totalmass,smin,smax,s1,ma2,mbq,m12,mnq,tmin,tmax
     &     ,t,tmax_temp,phi
      integer i,ibranch
      double precision lambda,dot
      external lambda,dot
c 
      pass=.true.
      totalmass=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totalmass=totalmass+m(itree(2,ibranch))
      enddo
      m(-ns_channel-1) = dsqrt(S(-nbranch))
c     
c Choose invariant masses of the pseudoparticles obtained by taking together
c all final-state particles or pseudoparticles found from the current 
c t-channel propagator down to the initial-state particle found at the end
c of the t-channel line.
      do ibranch = -ns_channel-1,-nbranch+2,-1
         totalmass=totalmass-m(itree(2,ibranch))  
         smin = totalmass**2                    
         smax = (m(ibranch) - m(itree(2,ibranch)))**2
         if (smin .gt. smax) then
            xjac0=-3d0
            pass=.false.
            return
         endif
         m(ibranch-1)=dsqrt((smax-smin)*
     &        x(nbranch-1+(-ibranch)*2)+smin)
         xjac0 = xjac0*(smax-smin)
         if (m(ibranch-1)**2.lt.smin.or.m(ibranch-1)**2.gt.smax
     &        .or.m(ibranch-1).ne.m(ibranch-1)) then
            xjac0=-1d0
            pass=.false.
            return
         endif
      enddo
c     
c Set m(-nbranch) equal to the mass of the particle or pseudoparticle P
c attached to the vertex (P,t,p2), with t being the last t-channel propagator
c in the t-channel line, and p2 the incoming particle opposite to that from
c which the t-channel line starts
      m(-nbranch) = m(itree(2,-nbranch))
c
c     Now perform the t-channel decay sequence. Most of this comes from: 
c     Particle Kinematics Chapter 6 section 3 page 166
c
c     From here, on we can just pretend this is a 2->2 scattering with
c     Pa                    + Pb     -> P1          + P2
c     p(0,itree(ibranch,1)) + p(0,2) -> p(0,ibranch)+ p(0,itree(ibranch,2))
c     M(ibranch) is the total mass available (Pa+Pb)^2
c     M(ibranch-1) is the mass of P2  (all the remaining particles)
c
      do ibranch=-ns_channel-1,-nbranch+1,-1
         s1  = m(ibranch)**2    !Total mass available
         ma2 = m(2)**2
         mbq = dot(pb(0,itree(1,ibranch)),pb(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mnq = m(ibranch-1)**2
         call yminmax(s1,t,m12,ma2,mbq,mnq,tmin,tmax)
         tmax_temp = tmax
         t = (tmax_temp-tmin)*x(-ibranch)+tmin
         xjac0=xjac0*(tmax_temp-tmin)
         if (t .lt. tmin .or. t .gt. tmax) then
            xjac0=-3d0
            pass=.false.
            return
         endif
         phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
         xjac0 = xjac0*2d0*pi
c Finally generate the momentum. The call is of the form
c pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
         call gentcms(pb(0,itree(1,ibranch)),pb(0,2),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),pb(0,itree(2,ibranch)),
     &        pb(0,ibranch),xjac0)
c
         if (xjac0 .lt. 0d0) then
            write(*,*) 'Failed gentcms',ibranch,xjac0
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0/(4d0*dsqrt(lambda(s1,ma2,mbq)))
      enddo
c We need to get the momentum of the last external particle.  This
c should just be the sum of p(0,2) and the remaining momentum from our
c last t channel 2->2
      do i=0,3
         pb(i,itree(2,-nbranch)) = pb(i,-nbranch+1)+pb(i,2)
      enddo
      return
      end


      subroutine fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      integer nbranch,nt_channel,ionebody
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      double precision xjac0,xpswgt0
      logical pass,one_body
c
      double precision one
      parameter (one=1d0)
      double precision costh,phi,xa2,xb2
      integer i,ix
      double precision lambda,dot
      external lambda,dot
      double precision vtiny
      parameter (vtiny=1d-12)
c
      pass=.true.
      do i = -nbranch+nt_channel+(nincoming-1),-1
         ix = nbranch+(-i-1)*2+(2-nincoming)
         if (nt_channel .eq. 0) ix=ix-1
         costh= 2d0*x(ix)-1d0
         phi  = 2d0*pi*x(ix+1)
         xjac0 = xjac0 * 4d0*pi
         xa2 = m(itree(1,i))*m(itree(1,i))/s(i)
         xb2 = m(itree(2,i))*m(itree(2,i))/s(i)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            xjac0=-8
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI)
         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        pb(0,itree(1,i)),pb(0,itree(2,i)))
c If there is an extremely large boost needed here, skip the phase-space point
c because of numerical stabilities.
         if (dsqrt(abs(dot(pb(0,i),pb(0,i))))/pb(0,i) 
     &        .lt.vtiny) then
            xjac0=-81
            pass=.false.
            return
         else
            call boostm(pb(0,itree(1,i)),pb(0,i),m(i),pb(0,itree(1,i)))
            call boostm(pb(0,itree(2,i)),pb(0,i),m(i),pb(0,itree(2,i)))
         endif
      enddo
c
c
c Special phase-space fix for the one_body
      if (one_body) then
c Factor due to the delta function in dphi_1
         xpswgt0=pi/m(ionebody)
c Kajantie's normalization of phase space (compensated below in flux)
         xpswgt0=xpswgt0/(2*pi)
         do i=0,3
            pb(i,3) = pb(i,1)+pb(i,2)
         enddo
      endif
      return
      end


      subroutine get_recoil(p_born,imother,shat_born,xmrec2,pass)
      implicit none
      include 'nexternal.inc'
      double precision p_born(0:3,nexternal-1),xmrec2,shat_born
      logical pass
      integer imother
      integer i
      double precision recoilbar(0:3),dot,tmp
      external dot
      pass=.true.
      do i=0,3
         if (nincoming.eq.2) then
            recoilbar(i)=p_born(i,1)+p_born(i,2)-p_born(i,imother)
         else
            recoilbar(i)=p_born(i,1)-p_born(i,imother)
         endif
      enddo
      xmrec2=dot(recoilbar,recoilbar)
      if(xmrec2.lt.0.d0)then
         if(abs(xmrec2).gt.(1.d-4*shat_born))then
            write(*,*)'Fatal error #14 in genps_fks.f',xmrec2,imother
            stop
         else
            write(*,*)'Error #15 in genps_fks.f',xmrec2,imother
            pass=.false.
            return
         endif
      endif
      if (xmrec2.ne.xmrec2) then
         write (*,*) 'Error #16 in setting up event in genps_fks.f,'//
     &        ' skipping event'
         pass=.false.
         return
      endif
      return
      end
