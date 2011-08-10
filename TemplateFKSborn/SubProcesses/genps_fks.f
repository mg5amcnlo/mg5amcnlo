      integer function f_get_nargs(ndim)
c**************************************************************************
c     Returns number of arguments which come from x_to_f_arg
c**************************************************************************
      implicit none
      include 'genps.inc'
      include "nexternal.inc"
      integer ndim
      f_get_nargs=4*nexternal+2      !All 4-momentum and x1,x2
      end

      subroutine x_to_f_arg(ndim,iconfig,mincfig,maxcfig,invar,wgt,x,p)
c**************************************************************************
c     This is a routine called from sample to transform the integration
c     variables into the arguments of the function. Often these will be
c     4 momentum, but it could also be a trivial 1->1 mapping.
c
c     INPUTS:  ndim     == number of dimensions
c              iconfig  == configuration working on
c              mincfig  == First configuration to include
c              maxcfig  == Last configuration to include
c              invar    == Number of invarients we are mapping (ndim*maxcfig)
c              wgt      == wgt for choosing point thus far. 1/npnts*iter
c     OUTPUTS: wgt      == updated weight after choosing points
c              x        == points choosen from sample grid
c              p        == transformed points call is f(p(x))
c
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc' 
      include "nexternal.inc"
c
c     Arguments
c
      integer ndim                             !Number of dimensions(input)
      integer iconfig                          !Configuration (input)
      integer mincfig,maxcfig                  !Range of configurations
      integer invar
      double precision wgt                     !(input and output)
      double precision x(99),p(0:3,99)         !x,p (output) [p(0:3,nexternal)]
c
c     Local
c
c
c     External
c     
c
c     Global
c
c-----
c  Begin Code
c-----
      call gen_mom(iconfig,mincfig,maxcfig,invar,wgt,x,p)
      end

      subroutine gen_mom(iconfig,mincfig,maxcfig,invar,wgt,x,p1)
c**************************************************************************
c
c     Routine to generate 4 momentum based on tree-level decomposition
c     using generalized s,t,u variables as integration variables. Need to
c     describe different configurations using variable iforest.
c     
c     INPUTS:    iconfig   == Current configuration working on
c                mincfig   == First configuration to include
c                maxcfig   == Last configuration to include
c                wgt       == wgt for choosing x's so far.
c     OUTPUTS:   wgt       == updated wgt including all configs
c                x()       == Integration variabls for all configs
c                p1(0:3,n) == 4 momentum of external particles
c                p1_cnt(0:3,n,-2:2) == 4 momentum for counterevents
c                wgt_cnt(-2:2) == wgts for counterevents
c                pst_cnt(-2:2) == phase-space wgts for counterevents
c                jac_cnt(-2:2) == jacobians for counterevents
c
c     REQUIRES: IFOREST() set in data statement (see configs.inc)
c               NEXTERNAL set in data statement (see genps.inc)
c
c     Note regarding integration variables mapping to invarients
c     the first nbranch variables go for the masses of branches -1,-2,..
c     For each t-channel invarient x(ndim-1), x(ndim-3), .... are used
c     in place of the cos(theta) variable used in s-channel.
c     x(ndim), x(ndim-2),.... are the phi angles.
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include "nexternal.inc"
      double precision pi
      parameter       (pi=3.1415926d0)
c
c     Arguments
c
      integer iconfig,mincfig,maxcfig,invar
      double precision p1(0:3,nexternal+1)
      double precision x(maxinvar)
      double precision wgt
c
c     Local
c
      integer nbranch,ndim
      integer i,j,jconfig,n,ipole
      double precision P(0:3,-max_branch:max_particles),xx(maxinvar)
      double precision M(-max_branch:max_particles)
      double precision s(-max_branch:0), pole_type
      integer nparticles,nfinal
      double precision jac,sjac,pswgt,pwgt(maxconfigs),flux
      double precision tprb, mtot
      double precision stot
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
      integer          lwgt(0:maxconfigs,maxinvar)
      logical firsttime

      double precision xprop(3,nexternal),tprop(3,nexternal)
      double precision maxwgt
      integer imatch
      save maxwgt

      integer ninvar, nconfigs

      integer iconfig_save
      
c
c     External
c
      double precision lambda,dot
      logical passcuts
c
c     Global
c
      double precision pmass(nexternal)
      common/to_mass/  pmass

      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      double precision   prb(maxconfigs,maxpoints,maxplace)
      double precision   fprb(maxinvar,maxpoints,maxplace)
      integer                      jpnt,jplace
      common/to_mconfig1/prb ,fprb,jpnt,jplace
      double precision   psect(maxconfigs),alpha(maxconfigs)
      common/to_mconfig2/psect            ,alpha

      include 'run.inc'


      integer iforest(2,-max_branch:-1,lmaxconfigs)
      common/to_forest/ iforest

      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac

      save lwgt
      save ndim,nfinal,nbranch,nparticles

      integer jfig,k
      
      save iconfig_save

c
c     data
c
      include 'configs.inc'
      data firsttime/.true./
      integer isym(0:100)
c      data isym /2,1,5,27,42,47,0,0,0,0,0/
      data jfig/1/
c-----
c  Begin Code
c----
      this_config = iconfig             !Pass iconfig to amplitude routine
c      write(*,*) 'using iconfig',iconfig
      do i=1,nexternal
         m(i)=pmass(i)
      enddo
      if (firsttime.or.(iconfig.ne.iconfig_save)) then
         firsttime=.false.
         do i=1,mapconfig(0)
            if (mapconfig(i) .eq. iconfig) this_config=i
         enddo
         write(*,*) 'Mapping Graph',iconfig,' to config',this_config
         iconfig = this_config
         iconfig_save=iconfig
         nconfigs = 1
         mincfig=iconfig
         maxcfig=iconfig
         call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming)
         maxwgt=0d0
c         write(*,'(a,12i4)') 'Summing configs',(isym(i),i=1,isym(0))
         nparticles   = nexternal
         nfinal       = nparticles-nincoming
         nbranch      = nparticles-2
         ndim         = 3*nfinal-4
         if (abs(lpp(1)) .ge. 1) ndim=ndim+1
         if (abs(lpp(2)) .ge. 1) ndim=ndim+1
c$$$
c May insert the following, according to one_tree
c$$$         if( nparticles-nincoming.eq.2 .and.
c$$$     #       ((abs(lpp(1)).ge.1).or.(abs(lpp(2)).ge.1)) )ndim=ndim-1
c$$$
c$$$         call set_peaks
         if (.false. ) then
            call find_matches(iconfig,isym(0))
            write(*,'(a,12i4)') 'Summing configs',(isym(i),i=1,isym(0))
         endif
         if (.false.) then
            i=1
            do while (mapconfig(i) .ne. iconfig
     $          .and. i .lt. mapconfig(0))
               i=i+1
            enddo
         endif

         write(*,'(a,12e10.3)') ' Masses:',(m(i),i=1,nparticles)
         do j=1,invar
            lwgt(0,j)=0
         enddo
c
c     Here we set up which diagrams contribute to each variable
c     in principle more than 1 diagram can contribute to a variable
c     if we believe they will have identical structure.
c
c         do i=1,mapconfig(0)
         do i=mincfig,maxcfig
c         do k=1,isym(0)
c            i = isym(k)
            write(*,'(15i4)') i,(minvar(j,i),j=1,ndim)
            do j=1,ndim
               ipole = minvar(j,i)
               if (ipole .ne. 0) then
                  n = lwgt(0,ipole)+1
                  lwgt(n,ipole)=mapconfig(i)  
                  lwgt(0,ipole)=n
               endif
            enddo
         enddo

      else
         do i=1,11
c            swidth(i)=-5d0         !tells us to use the same point over again
         enddo
c         swidth(10)=0d0
      endif                          !First_time

      if (.false.) then
         iconfig = isym(jfig)
         jfig = jfig+1
         if (jfig .gt. isym(0)) jfig=1      
      endif
      this_config = iconfig             !Pass iconfig to amplitude routine
c
c Bjorken x's are now computed in one_tree; sjac used to be the jacobian
c associated with the Bjorken x's mappings
c
      sjac = 1d0
      pswgt = 1d0
      jac   = sjac*wgt ! 1d0
c$$$TOUCHES IFOREST TO AVOID ITREE READ FROM SOMEWHERE ELSE
c$$$FIX THIS
      iforest(1,-1,iconfig)=iforest(1,-1,iconfig)
      iforest(2,-1,iconfig)=iforest(2,-1,iconfig)
      call one_tree(iforest(1,-max_branch,iconfig),mincfig,
     &     nbranch,ndim,P,M,S,X,jac,pswgt)
c$$$TOUCHES IFOREST TO AVOID ITREE READ FROM SOMEWHERE ELSE
c$$$FIX THIS
c
c     Add what I think are the essentials
c
         if (jac .gt. 0d0 ) then
            wgt=jac
            do i=1,nparticles
               do j=0,3
                  p1(j,i) = p(j,i)
               enddo
            enddo
c$$$THIS IS NOW AMBIGUOUS DUE TO EVENT PROJECTION
c$$$            p1(0,nparticles+1)=xbk(1)
c$$$            p1(1,nparticles+1)=xbk(2)
c$$$May set momenta equal to xbjrk_ev(*), since p1 correspond to event 
c$$$kinematics, but is preferable to cause the program to crash if
c$$$these entries are used
            p1(0,nparticles+1)=-1.d0
            p1(1,nparticles+1)=-1.d0

         else
            p1(0,1)=-99
         endif
      end


      subroutine one_tree(itree,iconfig,nbranch,ndim,P,M,S,X,jac,pswgt)
c************************************************************************
c     Calculates the momentum for everything below in the tree until
c     it reaches the end.
c     Note that the tree structure must have at least one t channel
c     part to it, and that the t-channel propagators must always appear
c     as the first element, that is itree(1,i)
c************************************************************************
      implicit none
c
c     Constants
c      
      include 'genps.inc'
      include "nexternal.inc"
      double precision pi            , one
      parameter       (pi=3.1415926d0, one=1d0)
      double precision zero
      parameter (zero=0.d0)
c
c     Arguments
c
      integer itree(2,-max_branch:-1) !Structure of configuration
      integer iconfig                 !Which configuration working on
      double precision P(0:3,-max_branch:max_particles)
      double precision M(-max_branch:max_particles)
      double precision S(-max_branch:0)
c      double precision spole(-max_branch:0),swidth(-max_branch:0)
      double precision jac,pswgt
      integer nbranch,ndim
      double precision x(21)
c
c     Local
c
      logical pass,one_body,fixsch,goodx
      integer ibranch,i,ns_channel,nt_channel,ix,ixEi,ixpi,ixyij,ionebody
      double precision smin,smax,totmass,totmassin,xa2,xb2,xwgt
      double precision costh,phi,tmin,tmax,t, stot,tmax_temp
      double precision ma2,mbq,m12,mnq,s1
      double precision dummy

c Argument has the following meaning:
c  -2 soft-collinear, incoming leg, - direction as in FKS paper
c  -1 collinear, incoming leg, - direction as in FKS paper
c   0 soft
c   1 collinear
c   2 soft-collinear
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/

c Set bst_to_pcms=.true. to return momenta in the incoming parton
c cm frame rather than in the reduced frame. The two coincide if
c j_fks is in the final state.
      logical bst_to_pcms
      data bst_to_pcms/.false./

c Set fks_as_is=.true. if the two initial state collinear singularities
c are subtracted as done in the FKS paper, ie within a unique S function
c contribution
      logical fks_as_is
      data fks_as_is/.false./
c Set int_nlo=.false. when testing this code with tree-level matrix elements
c only (ie no subtraction)
      logical int_nlo
      data int_nlo/.true./

c For PDF importance sampling
      integer nsamp
      parameter (nsamp=1)
      double precision ximax0,ximin0,tmp

      logical samplevar
      integer icountevts,icnt
      double precision xp(0:3,-max_branch:max_particles)
      double precision xp0(0:3,-max_branch:max_particles)
      double precision xjac0,xpswgt0,xjac,xpswgt,wgt

      real*8 ranu_save(97),ranc_save,rancd_save,rancm_save
      integer iranmr_save,jranmr_save

      integer fksmother,fksgrandmother,fksaunt,j,imother,iconfig_save
      integer fksconfiguration,compare

      logical firsttime
      data firsttime /.true./

c Set xexternal to false if x() are to be generated here by means of
c calls to sample_get_x
      logical xexternal
      common /toxexternal/ xexternal

      logical is_granny_cms
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown
      double precision fksmass
      double precision xi_i_fks, E_i_fks, phi_i_fks, y_ij_fks,phi_mother_fks
      double precision th_mother_fks,costh_mother_fks,sinth_mother_fks 
      double precision cosphi_i_fks,sinphi_i_fks
      double precision cosphi_mother_fks,sinphi_mother_fks
      double precision costh_i_fks,sinth_i_fks,xi_mother_fks
      double precision savejac,pwgt,flux
      double precision xmrec2,x3len_i_fks,x3len_j_fks,x3len_fks_mother
      double precision betabst,gammabst,shybst,chybst,chybstmo,sumrec,sumrec2

      double precision recoilbar(0:3),xpifksred(0:3),recoil(0:3),xdir(1:3)

      double complex ximag
      double complex resAoR0,resAoR5
      parameter (ximag=(0.d0,1.d0))

      double precision tiny,qtiny,stiny,sstiny,ctiny,cctiny
      parameter (tiny=1d-5)
      parameter (qtiny=1d-7)
      parameter (stiny=1d-6)
      parameter (ctiny=5d-7)

      double precision xmj,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2,
     # cffDEL2,xiBm,ximax,xitmp1,xitmp2,xirplus,xirminus,rat_xi,b2m4ac,
     # x3len_j_fks_num,x3len_j_fks_den,expybst,veckn,veckbarn,
     # xjactmp,veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks

      integer isolsign

      double precision sjac,roH,ylim,yijdir,x1bar2,x2bar2,yij_sol,ycmhat,
     # xi1,xi2,ximaxtmp,omega,bstfact,shy_tbst,chy_tbst,chy_tbstmo,
     # shy_lbst,chy_lbst,chy_lbstmo,encmso2,xdir_t(1:3),xdir_l(1:3)
      integer idir
      double precision roHs,fract
c
c     External
c
      double precision lambda,dot,rho
      external lambda,dot,rho
c
c     Global
c
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision m_born(nexternal-1)

      double precision xi_i_hat
      common /cxiihat/xi_i_hat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      
      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

c Parton c.m. stuff - to be set by call to set_cms_stuff() elsewhere
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

c Parton c.m. energy for event
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
c Parton c.m. energy for counterevents
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt

      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev

      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

      double precision tau_lower_bound,tau_lower_bound_soft
      common/ctau_lower_bound/tau_lower_bound,tau_lower_bound_soft

      logical softtest,colltest
      common/sctests/softtest,colltest

      logical nocntevents
      common/cnocntevents/nocntevents

      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used

      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt

      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt

      logical nbodyonly
      common/cnbodyonly/nbodyonly

c For MINT:
      include "mint.inc"
      integer ifold(ndimmax) 
      common /cifold/ifold
      integer ifold_energy,ifold_phi,ifold_yij
      common /cifoldnumbers/ifold_energy,ifold_phi,ifold_yij

c Seeds for ranmar
      real*8 ranu,ranc,rancd,rancm
      common/ raset1 / ranu(97),ranc,rancd,rancm
      integer iranmr,jranmr
      common/ raset2 / iranmr,jranmr

      include 'run.inc'
      include 'fks.inc'
      include 'fks_powers.inc'

      save compare,ns_channel,nt_channel
      save fksmother,fksgrandmother,fksaunt,is_granny_cms,iconfig_save
      save xjactmp

c For Breit-Wigner importance sampling
      integer ilast_schannel
      double precision xm02,bwmdpl,bwmdmn,bwfmpl,bwfmmn,bwdelf,
     #  xbwmass3,bwfunc,stemp
      double precision pmass(-nexternal:0,lmaxconfigs)
      double precision pwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      include 'coupl.inc'
      include 'props.inc'

c-----
c  Begin Code
c-----
      savejac = jac

c if samplevar=T, calls sample_get_x, else keep x() previously generated
      if( (xi_i_fks_ev.eq.0.1d0.and.softtest) .or.
     #    (y_ij_fks_ev.eq.0.9d0.and.colltest) .or.
     #           .not. (softtest.or.colltest) )then
        samplevar = .true.         
      else
        samplevar = .false.         
      endif

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
c the following quantity will remain equal to zero in all cases except
c when the event is generated in the course of a normal (ie not testing) run
      xi_i_hat=0.d0
c if collinear counterevent will not be generated, the following
c quantity will stay zero
      xij_aor=(0.d0,0.d0)
c if sample_get_x was not called before, seeds for ranmar may not be
c initialized. Place a dummy call here. Do it only when integrating
c subtracted matrix elements
      if(int_nlo.and.(.not.xexternal))then
        ix=1
        if(firsttime)call sample_get_x(dummy,dummy,ix,iconfig,0d0,1d0) 
        do i=1,97
          ranu_save(i)=ranu(i)
        enddo
        ranc_save=ranc
        rancd_save=rancd
        rancm_save=rancm
        iranmr_save=iranmr
        jranmr_save=jranmr
      endif

      xjac0   = 1d0
      xpswgt0 = 1d0
      if(samplevar.or.xexternal)xwgt = 1d0
      pass = .true.
      stot = 4d0*ebeam(1)*ebeam(2)
c
c     Make sure have enough mass for external particls
c
      totmassin=0d0
      do ibranch=3-nincoming,2
         totmassin=totmassin+m(ibranch)
      enddo
      totmass=0d0
      do ibranch=3,nbranch+2
         totmass=totmass+m(ibranch)
      enddo
      fksmass=totmass
      if (sqrt(stot) .lt. max(totmass,totmassin)) then
         write(*,*)
     #     'Fatal error #0 in one_tree: insufficient collider energy'
         stop
      endif
      if (firsttime.or.(iconfig.ne.iconfig_save))then
         iconfig_save=iconfig
         searchforgranny=.false.
c Read FKS configuration from file
         open (unit=61,file='config.fks',status='old')
         read(61,'(I2)',err=99,end=99) fksconfiguration
 99      close(61)
c Use the fks.inc include file to set i_fks and j_fks
         i_fks=fks_i(fksconfiguration)
         j_fks=fks_j(fksconfiguration)
         write (*,*) 'FKS configuration number is ',fksconfiguration
         write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks
c
c     Determine number of s channel branches, this doesn't count
c     the s channel p1+p2
c
         if (j_fks.eq.2)then
c then t-channels in configs.inc are inverted.
            compare=2
         else
c t-channels are ordered normally 
            compare=1
         endif
         ns_channel=1
         do while(itree(1,-ns_channel).ne.compare .and.
     &                            ns_channel.lt.nbranch)
            m(-ns_channel)=0d0                 
            ns_channel=ns_channel+1         
         enddo
         ns_channel=ns_channel - 1
         nt_channel=nbranch-ns_channel-1
c Find mother, grandmother, and aunt of the two fks parton. The latter
c must be given at this point
         call grandmother_fks(iconfig,nbranch,ns_channel,nt_channel,
     &                        i_fks,j_fks,searchforgranny,
     &                        fksmother,fksgrandmother,fksaunt,
     &                        is_beta_cms,is_granny_sch,topdown)
         if ( (j_fks.gt.nincoming .and.
c$$$     &           (fksgrandmother.ge.0.or.fksmother.eq.0)) .or.
     &           (fksmother.eq.0)) .or.
     &        (j_fks.le.nincoming .and. ! Then no fksgrandmother and fksmother=j_fks
c$$$     &           (fksgrandmother.ne.0.or.fksmother.ne.j_fks)))then
     &           (fksmother.ne.j_fks)))then
            write(*,*)'Error #1a in one_tree',
     #                 fksgrandmother,fksmother,iconfig
            stop
         endif
         is_granny_cms=is_beta_cms

         if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
            ns_channel=ns_channel-1
         endif

         if((nexternal-nincoming).eq.2)then
            one_body=.true.
            ionebody=nexternal
            if(ionebody.eq.i_fks)ionebody=nexternal-1
         elseif((nexternal-nincoming).gt.2)then
            one_body=.false.
         else
            write(*,*)'Error #500 in one_tree',nexternal,nincoming
            stop
         endif

         if( one_body .and. 
     #       (j_fks.gt.nincoming.or.nt_channel.eq.0) )then
            write(*,*)'Error #502 in one_tree',
     #                j_fks,nincoming,nt_channel
            stop
         endif

         fixsch=.false.
         if(nt_channel.eq.1.and.j_fks.le.nincoming)fixsch=.true.

         ilast_schannel = -ns_channel
         if(j_fks.gt.nincoming)ilast_schannel=-ns_channel-1

c For initial state singularity fksmother=j_fks from grandmother_fks
c Convert it here to the correct number
         if(fksmother.gt.0) fksmother=-ns_channel-1

c Grid pre-setting. Used to be in genmom
c
         call set_tau_min()
         call set_peaks_MadFKS(i_fks,j_fks,fksmother,compare)

         firsttime=.false.
      endif

c
c Generate Bjorken x's if need be
c
      sjac = 1d0
      if(.not.xexternal)then
c xbk(i) is stored in common/to_collider/ in run.inc. Here, it is used
c solely as a local variable. Their actual values will be set elsewhere
c$$$NOTE: the following calls to sample_get_x had mincfig instead
c$$$of iconfig
        if (abs(lpp(1)) .ge. 1 .and. abs(lpp(2)) .ge. 1) then
           if(samplevar)then
             call sample_get_x(sjac,x(ndim-1),ndim-1,iconfig,0d0,1d0)
             call sample_get_x(sjac,x(ndim),ndim,iconfig,0d0,1d0)
           endif
           CALL GENCMS(STOT,Xbk(1),Xbk(2),X(ndim-1),0d0,SJAC)
           ycmhat=2*x(ndim)-1
           if(one_body)then
             write(*,*)'One_tree: option fix_mass not tested'
             stop
           endif
        elseif (abs(lpp(1)) .ge. 1) then
           if(samplevar)
     #       call sample_get_x(sjac,x(ndim),ndim,iconfig,0d0,1d0)
           xbk(1) = x(ndim)
        elseif (abs(lpp(2)) .ge. 1) then
           if(samplevar)
     #       call sample_get_x(sjac,x(ndim),ndim,iconfig,0d0,1d0)
           xbk(2) = x(ndim)
        else
           if(j_fks.le.nincoming)then
c If set to one, integration range in xi_i_fks will be zero
             if(one_body)then
               xbk(1) = fksmass/sqrt(stot)
               xbk(2) = fksmass/sqrt(stot)
             else
               xbk(1) = max(0.85d0,fksmass/sqrt(stot))
               xbk(2) = max(0.85d0,fksmass/sqrt(stot))
             endif
           else
             xbk(1) = 1.d0
             xbk(2) = 1.d0
           endif
           ycmhat=0.d0
        endif
        tau_cnt(0)=xbk(1)*xbk(2)
        ycm_cnt(0)=0.5d0*log(xbk(1)/xbk(2))
c Skip events which are extremely boosted        
        if (abs(ycm_cnt(0)).gt.15d0) goto 222
      else
        if (abs(lpp(1)) .ge. 1 .and. abs(lpp(2)) .ge. 1) then
c x(ndim-1) -> tau_cnt(0); x(ndim) -> ycm_cnt(0)
           if(one_body)then
             roH=totmass**2/stot
             tau_cnt(0)=roH
c Jacobian due to delta() of taubar
             sjac=sjac*2*totmass/stot
             if(totmass.ne.m(ionebody))then
               write(*,*)'Error #511 in one_tree',
     #                   totmass,m(ionebody),ionebody
               stop
             endif
           else
             if( (fixsch .or. nt_channel.eq.0) .and.
     #           pwidth(ilast_schannel,iconfig).ne.0.d0 )then
c Breit Wigner
               smax=stot
               smin=tau_lower_bound*stot
               xm02=pmass(ilast_schannel,iconfig)**2
               bwmdpl=smax-xm02
               bwmdmn=xm02-smin
               bwfmpl=atan(bwmdpl/
     #  (pmass(ilast_schannel,iconfig)*pwidth(ilast_schannel,iconfig)))
               bwfmmn=atan(bwmdmn/
     #  (pmass(ilast_schannel,iconfig)*pwidth(ilast_schannel,iconfig)))
               bwdelf=(bwfmpl+bwfmmn)/pi
               stemp=xbwmass3(x(ndim-1),xm02,
     #                        pwidth(ilast_schannel,iconfig),bwdelf,bwfmmn)
               sjac=sjac*bwdelf/bwfunc(stemp,xm02,
     #                                 pwidth(ilast_schannel,iconfig))
               tau_cnt(0)=stemp/stot
               sjac=sjac/stot
             else
c not a Breit Wigner
                roH=tau_lower_bound
                roHs=tau_lower_bound_soft
c User x below 'fract' for phase-space region below soft cut-off
                if(tau_lower_bound_soft.le.tau_lower_bound + 1d-8)then
                   fract=0.0d0
                else
                   fract=0.1d0
                endif
                if (x(ndim-1).lt.fract) then
c Flat grid below soft cut-off
                   tau_cnt(0)=roH+(roHs-roH)*x(ndim-1)/fract
                   sjac=sjac*(roHs-roH)/fract
                else
c Use 1/x importance sampling above soft cut-off
                   ximax0 = roHs**(-nsamp)
                   ximin0 = 1.d0
                   tmp  = ximin0 +(1d0-(x(ndim-1)-fract)/(1d0-fract))*
     &                  (ximax0-ximin0)
                   tau_cnt(0) = tmp**(-1/dfloat(nsamp))
                   sjac= sjac/nsamp*tau_cnt(0)**(nsamp+1)*
     &                  (ximax0-ximin0)/(1d0-fract)
                endif
             endif
           endif
           ylim=-0.5d0*log(tau_cnt(0))
           ycmhat=2*x(ndim)-1
           ycm_cnt(0)=ylim*ycmhat
           sjac=sjac*ylim*2
c Skip events which are extremely boosted        
           if (abs(ycm_cnt(0)).gt.15d0) goto 222
        elseif (abs(lpp(1)) .ge. 1) then
           write(*,*)'Option x1 not implemented in one_tree'
           stop
        elseif (abs(lpp(2)) .ge. 1) then
           write(*,*)'Option x2 not implemented in one_tree'
           stop
        else
           if(j_fks.le.nincoming)then
c If tau set to one, integration range in xi_i_fks will be zero
             if(one_body)then
               tau_cnt(0)=fksmass**2/stot
             else
               tau_cnt(0)=max((0.85d0)**2,fksmass**2/stot)
             endif
             ycm_cnt(0)=0.d0
           else
             tau_cnt(0)=1.d0
             ycm_cnt(0)=0.d0
           endif
           ycmhat=0.d0
        endif
      endif

      if(softtest.or.colltest)then
c Perform tests at fixed energy
        sjac=1.d0
        if(j_fks.le.nincoming )then
          if(one_body)then
            tau_cnt(0)=fksmass**2/stot
          else
            tau_cnt(0)=max((0.85d0)**2,fksmass**2/stot)
          endif
          ycm_cnt(0)=0.d0
        else
          tau_cnt(0)=1.d0
          ycm_cnt(0)=0.d0
        endif
        ycmhat=0.d0
      endif
      xjac0=xjac0*sjac
      xbjrk_cnt(1,0)=sqrt(tau_cnt(0))*exp(ycm_cnt(0))
      xbjrk_cnt(2,0)=sqrt(tau_cnt(0))*exp(-ycm_cnt(0))
      if(.not.one_body)then
        shat_cnt(0)=tau_cnt(0)*stot
        sqrtshat_cnt(0)=sqrt(shat_cnt(0))
      else
c Trivial, but prevents loss of accuracy
        sqrtshat_cnt(0)=fksmass
        shat_cnt(0)=fksmass**2
      endif
c
      if(j_fks.gt.nincoming)then
        do i=1,2
          tau_cnt(i)=tau_cnt(0)
          ycm_cnt(i)=ycm_cnt(0)
          shat_cnt(i)=shat_cnt(0)
          sqrtshat_cnt(i)=sqrtshat_cnt(0)
          xbjrk_ev(i)=xbjrk_cnt(i,0)
          do j=1,2
            xbjrk_cnt(i,j)=xbjrk_cnt(i,0)
          enddo
        enddo
        tau_ev=tau_cnt(0)
        ycm_ev=ycm_cnt(0)
        shat_ev=shat_cnt(0)
        sqrtshat_ev=sqrtshat_cnt(0)
        s(-nbranch)  = shat_ev
        m(-nbranch)  = sqrtshat_ev
        p(0,-nbranch)= m(-nbranch)
        p(1,-nbranch)= 0d0
        p(2,-nbranch)= 0d0
        p(3,-nbranch)= 0d0
c generate momenta for initial state particles
        if(nincoming.eq.2) then
          call mom2cx(m(-nbranch),m(1),m(2),1d0,0d0,p(0,1),p(0,2))
        else
          do i=0,3
            p(i,1)=p(i,-nbranch)
          enddo
          p(3,1)=1e-14 ! For HELAS routine ixxxxx for neg. mass
        endif
        do i=1,nincoming
          do j=0,3
             xp0(j,i)=p(j,i)
          enddo
        enddo
      else
c Set the following to zero in order to make the program crash if used
        s(-nbranch)  = 0.d0
        m(-nbranch)  = 0.d0
        if(fixsch)then
c Energy of the reduced process (merge i_fks and j_fks) if t-channel
c structure is entirely dealt with by FKS
          if(fksmother.ne.-(nbranch-1))then
             write(*,*)'Error #508 in one_tree',fksmother,nbranch
             stop
          endif
          s(fksmother)  = shat_cnt(0)
          m(fksmother)  = sqrtshat_cnt(0)
        endif
c Most likely the following will give troubles if m(1) or m(2) # 0.
        call mom2cx(sqrtshat_cnt(0),m(1),m(2),1d0,0d0,xp0(0,1),xp0(0,2))
      endif
c
      do j=0,3
         xp0(j,i_fks)=0.d0
         if (j_fks.gt.nincoming) xp0(j,j_fks)=0.d0
      enddo

      if(j_fks.gt.nincoming)then
        xp0(0,fksmother)=-99
      else
        do j=0,3
          xp0(j,fksmother)=xp0(j,j_fks)
        enddo
        if(one_body)xp0(0,ionebody)=-99
      endif

c
c     Determine masses for all intermediate states.  Starting
c     from outer most (real particle) states
c
      if (sqrtshat_cnt(0) .lt. max(totmass,totmassin)) then
         pass=.false.
         xjac0 = -5d0
         goto 222
      endif

      if(m(i_fks).ne.0.d0)then
        write(*,*)'Parton i_fks needs be massless',i_fks
        stop
      endif
c
c Generate Born-level momenta
c

      do ibranch = -1,-ns_channel,-1
c Generate invariant masses for all s-channel branchings except that of 
c FKS mother, whose mass is set to zero
        if( .not.( (ibranch.eq.fksmother.and.(.not.fixsch)) .or.
     #             ibranch.eq.-ns_channel.and.fixsch ) )then

           smin = (m(itree(1,ibranch))+m(itree(2,ibranch)))**2
c$$$           smax = (dsqrt(s(-nbranch))-totmass+sqrt(smin))**2
           smax = (sqrtshat_cnt(0)-totmass+sqrt(smin))**2
c
c        Choose the appropriate s given our constraints smin,smax
c     
           if(samplevar.and.(.not.xexternal))
     #        call sample_get_x(xwgt,x(-ibranch),-ibranch,iconfig,
     #                          smin/stot,smax/stot)
           if(.not.xexternal)then
             s(ibranch) = x(-ibranch)*stot
             xjac0 = xjac0*stot
           else
             if(pwidth(ibranch,iconfig).ne.0.d0)then
c Breit Wigner
               if(smax.lt.smin.or.smax.lt.0.d0.or.smin.lt.0.d0)then
                 write(*,*)'Error #1 in one_tree'
                 write(*,*)smin,smax,ibranch
                 stop
               endif
               xm02=pmass(ibranch,iconfig)**2
               bwmdpl=smax-xm02
               bwmdmn=xm02-smin
               bwfmpl=atan(bwmdpl/
     #  (pmass(ibranch,iconfig)*pwidth(ibranch,iconfig)))
               bwfmmn=atan(bwmdmn/
     #  (pmass(ibranch,iconfig)*pwidth(ibranch,iconfig)))
               bwdelf=(bwfmpl+bwfmmn)/pi
               s(ibranch)=xbwmass3(x(-ibranch),xm02,
     #                       pwidth(ibranch,iconfig),bwdelf,bwfmmn)
               xjac0=xjac0*bwdelf/bwfunc(s(ibranch),xm02,
     #                                   pwidth(ibranch,iconfig))
             else
c not a Breit Wigner
               s(ibranch) = (smax-smin)*x(-ibranch)+smin
               xjac0 = xjac0*(smax-smin)
             endif
           endif

           if (xjac0 .lt. 0d0 .or. .not. pass) then
              xjac0 = -6
              goto 222
           endif
           if (s(ibranch) .lt. smin) then
              xjac0=-5
              goto 222
           endif
        else
           if(.not.fixsch)then
              if(j_fks.le.nincoming)then
                write(*,*)'Error #506 in one_tree'
                stop
              endif
c Set mother squared mass equal to the mass squared of j_fks
              s(ibranch)=m(j_fks)**2
           else
              if(j_fks.gt.nincoming)then
                write(*,*)'Error #507 in one_tree'
                stop
              endif
c Fix shat for innermost (most massive) s channel. Done here rather than
c below as in the case nt_channel=0 since it needs to update totmass
              s(fksmother+1)  = s(fksmother)
              m(fksmother+1)  = m(fksmother)
c Innermost s-channel is Borm cm frame
              xp0(0,fksmother+1) = m(fksmother+1)
              xp0(1,fksmother+1) = 0d0
              xp0(2,fksmother+1) = 0d0
              xp0(3,fksmother+1) = 0d0
           endif
        endif
c
c     Check that s is ok, and fill masses, update totmass
c
        m(ibranch) = sqrt(s(ibranch))
        totmass=totmass+m(ibranch)-
     &       m(itree(1,ibranch))-m(itree(2,ibranch))
c$$$        if (totmass .gt. M(-nbranch)) then
c If the reduced process is initiated by an s-channel, here totmass must
c coincide with shat up to rounding error -- do not kill event 
c if simply inaccurate
        if ( ((.not.fixsch) .and. totmass.gt.sqrtshat_cnt(0)) .or.
     #       (fixsch.and. totmass.gt.1.00001*sqrtshat_cnt(0)) )then
           xjac0 = -4
           goto 222
        endif
        if (.not. pass) then
           xjac0=-9
           goto 222
        endif
      enddo

      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         if(j_fks.le.nincoming)then
           write(*,*)'Fatal error #12 in one_tree',j_fks
           stop
         endif
         s(-nbranch+1)=s(-nbranch) 
         m(-nbranch+1)=m(-nbranch)      !Basic s-channel has s_hat 
         xp0(0,-nbranch+1) = m(-nbranch+1)!and 0 momentum
         xp0(1,-nbranch+1) = 0d0
         xp0(2,-nbranch+1) = 0d0
         xp0(3,-nbranch+1) = 0d0
      endif
c
c     Next do the T-channel branchings
c
c
c     First we need to determine the energy of the remaining particles
c     this is essentially in place of the cos(theta) degree of freedom
c     we have with the s channel decay sequence
c


      if ( (nt_channel .gt. 0) .and. (.not.fixsch) ) then   
c
c Non trivial t-channel stuff
c

      totmass=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totmass=totmass+m(itree(2,ibranch))
      enddo
      if(j_fks.gt.nincoming)then
        m(-ns_channel-1) = dsqrt(S(-nbranch))
      else
c -ns_channel-1=fksmother here. Its mass must be unused in what follows
        m(-ns_channel-1) = 0.d0
      endif
c Choose invariant masses of the pseudoparticles obtained by taking together
c all final-state particles or pseudoparticles found from the current 
c t-channel propagator down to the initial-state particle found at the end
c of the t-channel line. Skip fksmother, whose associated invariant mass
c is set equal to the reduced c.m. energy
      do ibranch = -ns_channel-1,-nbranch+2,-1
         if (ibranch.ne.fksmother)then
           if(fixsch)then
              write(*,*)'Error #503 in one_tree'
              stop
           endif
           totmass=totmass-m(itree(2,ibranch))  
           smin = totmass**2                    
           smax = (m(ibranch) - m(itree(2,ibranch)))**2
           if (smin .gt. smax) then
              xjac0=-3d0
              goto 222
           endif
           if(samplevar.and.(.not.xexternal))
     &        call sample_get_x(xwgt,x(nbranch-1+(-ibranch)*2),
     &          nbranch-1+(-ibranch)*2,iconfig,
     &          smin/stot,smax/stot)
           if(.not.xexternal)then
             m(ibranch-1)=dsqrt(stot*x(nbranch-1+(-ibranch)*2))
             xjac0 = xjac0 * stot
           else
             m(ibranch-1)=dsqrt((smax-smin)*x(nbranch-1+(-ibranch)*2)+smin)
             xjac0 = xjac0*(smax-smin)
           endif

c           write(*,*) 'Using s',nbranch-1+(-ibranch)*2
           if (m(ibranch-1)**2.lt.smin.or.m(ibranch-1)**2.gt.smax
     $          .or.m(ibranch-1).ne.m(ibranch-1)) then
              xjac0=-1d0
              goto 222
           endif
         else
           if(fksmother.ne.(-ns_channel-1))then
             write(*,*)'Error #501 in one_tree',fksmother,-ns_channel-1
             stop
           endif
           if(fixsch)then
             if(m(-ns_channel-2).ne.sqrtshat_cnt(0))then
               write(*,*)'Error #509 in one_tree',
     #                   m(-ns_channel-2),sqrtshat_cnt(0)
               stop
             endif
           else
             m(-ns_channel-2)=sqrtshat_cnt(0)
           endif
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
c When looping over t-channel branches skip mother -- this must occur only
c when j_fks is initial state
         if(ibranch.eq.fksmother)goto 33
         if(fixsch)then
            write(*,*)'Error #504 in one_tree'
            stop
         endif
         s1  = m(ibranch)**2                        !Total mass available
         ma2 = m(3-compare)**2
         mbq = dot(XP0(0,itree(1,ibranch)),XP0(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mnq = m(ibranch-1)**2
c         write(*,*) 'Enertering yminmax',sqrt(s1),sqrt(m12),sqrt(mnq)
         call yminmax(s1,t,m12,ma2,mbq,mnq,tmin,tmax)


c$$$         tmax_temp = max(tmax,0d0) !This line if want really t freedom
         tmax_temp = tmax

         if(samplevar.and.(.not.xexternal))
     #      call sample_get_x(xwgt,x(-ibranch),-ibranch,iconfig,
     #                        -tmax_temp/stot, -tmin/stot)
         if(.not.xexternal)then
           t = stot*(-x(-ibranch))
           xjac0 = xjac0*stot
         else
           t = (tmax_temp-tmin)*x(-ibranch)+tmin
           xjac0=xjac0*(tmax_temp-tmin)
         endif

         if (t .lt. tmin .or. t .gt. tmax) then
            xjac0=-3d0
            goto 222
         endif
c
c     tmin and tmax set to -s,+s for jacobian because part of jacobian
c     was determined from choosing the point x from the grid based on
c     tmin and tmax.  (ie wgt contains some of the jacobian)
c
         if(samplevar.and.(.not.xexternal))
     &     call sample_get_x(xwgt,x(nbranch+(-ibranch-1)*2),
     &        nbranch+(-ibranch-1)*2,iconfig,0d0,1d0)
         phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
         xjac0 = xjac0*2d0*pi
c
c     Finally generate the momentum. The call is of the form
c     pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c     gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
c
         call gentcms(xp0(0,itree(1,ibranch)),xp0(0,3-compare),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),xp0(0,itree(2,ibranch)),
     &        xp0(0,ibranch),xjac0)

         if (xjac0 .lt. 0d0) then
            write(*,*) 'Failed gentcms',iconfig,ibranch,xjac0
            goto 222
         endif

         xpswgt0 = xpswgt0/(4d0*dsqrt(lambda(s1,ma2,mbq)))
 33      continue
      enddo
c
c     We need to get the momentum of the last external particle.
c     This should just be the sum of p(0,2) (or p(0,1) for inverse
c     t-channel) and the remaining momentum from our last t channel 2->2
c
      do i=0,3
         xp0(i,itree(2,-nbranch)) = xp0(i,-nbranch+1)+xp0(i,3-compare)
      enddo

      endif                     !t-channel stuff


c
c     Now generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
      do i = -nbranch+nt_channel+(nincoming-1),-1
c Loop over all s-channel poles, but skip mother
         if(i.eq.fksmother)goto 44
         ix = nbranch+(-i-1)*2+(2-nincoming)
         if (nt_channel .eq. 0) ix=ix-1

c         write(*,*) 'using costh,phi',ix,ix+1

         if(samplevar.and.(.not.xexternal))
     #      call sample_get_x(xwgt,x(ix),ix,iconfig,0d0,1d0)
         costh= 2d0*x(ix)-1d0
         if(samplevar.and.(.not.xexternal))
     #      call sample_get_x(xwgt,x(ix+1),ix+1,iconfig,0d0,1d0)
         phi  = 2d0*pi*x(ix+1)
         xjac0 = xjac0 * 4d0*pi
         xa2 = m(itree(1,i))*m(itree(1,i))/s(i)
         xb2 = m(itree(2,i))*m(itree(2,i))/s(i)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            xjac0=-8
            goto 222
         endif
         xpswgt0 = xpswgt0*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI)
         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        xp0(0,itree(1,i)),xp0(0,itree(2,i)))
         call boostx(xp0(0,itree(1,i)),xp0(0,i),xp0(0,itree(1,i)))
         call boostx(xp0(0,itree(2,i)),xp0(0,i),xp0(0,itree(2,i)))
 44      continue
      enddo

      if(one_body)then
        if( xp0(0,ionebody).gt.0.d0 .or. 
     #      itree(2,-nbranch).ne.ionebody )then
           write(*,*)'Error #505 in one_tree',xp0(0,ionebody),
     #               itree(2,-nbranch),ionebody
           stop
        endif
        if(xpswgt0.ne.1.d0)then
           write(*,*)'Error #510 in one_tree',xpswgt0
           stop
        endif
c Factor due to the delta function in dphi_1
        xpswgt0=pi/m(ionebody)
c Kajantie's normalization of phase space (compensated below in flux)
        xpswgt0=xpswgt0/(2*pi)
        do i=0,3
           xp0(i,itree(2,-nbranch)) = xp0(i,-nbranch+1)+xp0(i,3-compare)
        enddo
      endif

      if(xp0(0,fksmother).lt.0.d0)then
        if(abs(xp0(0,fksmother)).gt.(1.d-4*sqrtshat_cnt(0)))then
          write(*,*)'Fatal error #1 in one_tree',
     &              fksmother,xp0(0,fksmother)
          stop
        else
          write(*,*)'Error #1 in one_tree',
     &              fksmother,xp0(0,fksmother)
          xjac0=-43d0
          goto 222
        endif
      endif

      imother=0
      do i=1,nexternal-1
         do j=0,3
            if(j_fks.le.nincoming)then
               if (i.lt.i_fks) then
                  p_born(j,i)=xp0(j,i)
                  m_born(i)=m(i)
               else
                  p_born(j,i)=xp0(j,i+1)
                  m_born(i)=m(i+1)
               endif
            else
               if (i.eq.min(i_fks,j_fks)) then
                  imother=i
                  p_born(j,i)=xp0(j,fksmother)
                  m_born(i)=m(j_fks)
               elseif (i.lt.max(i_fks,j_fks))then
                  p_born(j,i)=xp0(j,i)
                  m_born(i)=m(i)
               else
                  p_born(j,i)=xp0(j,i+1)
                  m_born(i)=m(i+1)
               endif
            endif
         enddo
      enddo
      call phspncheck_born(sqrtshat_cnt(0),m_born,p_born)

      if(j_fks.le.nincoming)then
c
c Initial state
c
c Generate E_i and phi_i. Random numbers have the same labels as those
c used by Madgraph to generate the t-channel virtuality (equivalent to a
c polar angle) and the azimuthal angle for the t-channel propagator 
c that corresponds to ibranch = fksmother

c E_i
        ixEi = -fksmother
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixEi),ixEi,iconfig,0d0,1d0) 
c phi_i
        ixpi = nbranch+(-fksmother-1)*2
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixpi),ixpi,iconfig,0d0,1d0)

c Generate y_ij. Random number has the same label as that used by Madgraph 
c to generate the invariant mass in the branching ibranch = fksmother,
c except for special cases

        if((.not.fixsch).or.one_body)then
          ixyij = nbranch-1+(-fksmother)*2
        else
          ixyij = ns_channel
        endif
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixyij),ixyij,iconfig,0d0,1d0)


c Set up the MINT folding:
        ifold_energy=ixEi
        ifold_phi=ixpi
        ifold_yij=ixyij
  
      else
c
c Final state
c
c The following is mostly taken from invar_out3, with (1:4) -> (0:3)
        do i=0,3
          recoilbar(i)=p_born(i,1)+p_born(i,2)-p_born(i,imother)
        enddo
        xmrec2=dot(recoilbar,recoilbar)
        if(xmrec2.lt.0.d0)then
          if(abs(xmrec2).gt.(1.d-4*shat_cnt(0)))then
            write(*,*)'Fatal error #2 in one_tree',xmrec2,imother
            stop
          else
            write(*,*)'Error #2 in one_tree',xmrec2,imother
            xjac0=-44d0
            goto 222
          endif
        endif
  
        if (xmrec2.ne.xmrec2) then
           write (*,*) 'Error in setting up event in one_tree,'//
     &        ' skipping event'
           xjac0=-99
           goto 222
        endif

        call getangles(p_born(0,imother),
     #           th_mother_fks,costh_mother_fks,sinth_mother_fks,
     #           phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)

c Generate E_i and phi_i. Random numbers have the same labels as those
c used by Madgraph to generate the polar and azimuthal angles in the
c branching ibranch = fksmother
        ixEi = nbranch+(-fksmother-1)*2+(2-nincoming)
        if (nt_channel .eq. 0) ixEi=ixEi-1
c E_i
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixEi),ixEi,iconfig,0d0,1d0) 
c phi_i
        ixpi=ixEi+1
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixpi),ixpi,iconfig,0d0,1d0)

c Generate y_ij. Random number has the same label as that used by Madgraph 
c to generate the invariant mass in the branching ibranch = fksmother

        ixyij = -fksmother
        if(samplevar.and.(.not.xexternal))
     #    call sample_get_x(xwgt,x(ixyij),ixyij,iconfig,0d0,1d0)

        if((sqrt(xmrec2)+m(j_fks)).lt.(fksmass-1d-4))then
          write(*,*)'Fatal error #3 in one_tree'
     &        ,sqrt(xmrec2)+m(j_fks),fksmass
          stop
        endif

c Set up the MINT folding:
        ifold_energy=ixEi
        ifold_phi=ixpi
        ifold_yij=ixyij

      endif

c This setting of samplevar is probably obsolete, but not wrong
      samplevar=.false.

 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0

c To save time, uncomment the following, but fix first the setting
c of xjactmp
c$$$      if(nbodyonly.and.icountevts.eq.-100)then
c$$$        xjac = -399
c$$$        goto 112
c$$$      endif

      do i=-max_branch,nexternal
        do j=0,3
          xp(j,i)=xp0(j,i)
        enddo
      enddo

      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi
      
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)

      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     #    ((.not.softtest) .or. 
     #     (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     #    (.not.colltest)  )then
c$$$        y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(ixyij))+1d0
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
        y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(ixyij)**2)+1d0
        y_ij_fks_ev=y_ij_fks
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     #    ((.not.softtest) .or. 
     #     (softtest.and.y_ij_fks_fix.ne.-2.d0)) .and.
     #    (.not.colltest)  )then
        y_ij_fks = y_ij_fks_fix
        y_ij_fks_ev=y_ij_fks
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     #         colltest )then
        y_ij_fks=y_ij_fks_ev
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
        y_ij_fks=y_ij_fks_matrix(icountevts)
      else
        write(*,*)'Error #102 in one_tree',icountevts
        stop
      endif
c$$$      xjac=xjac*2d0
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(ixyij)*2d0

      isolsign=0
      if(m(j_fks).eq.0.d0)then
c
c Parton j is massless
c
c Set isolsign=1 to understand that there is only one fold for massless j_fks
        isolsign=1

        if(j_fks.le.nincoming)then
c
c Parton j is in the initial state
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
          yijdir=idir*y_ij_fks
          costh_i_fks=yijdir
c
          x1bar2=xbjrk_cnt(1,0)**2
          x2bar2=xbjrk_cnt(2,0)**2
          if(1-tau_cnt(0).gt.1.d-5)then
            yij_sol=-sinh(ycm_cnt(0))*(1+tau_cnt(0))/
     #             ( cosh(ycm_cnt(0))*(1-tau_cnt(0)) )
          else
            yij_sol=-ycmhat
          endif
          if(abs(yij_sol).gt.1.d0)then
            write(*,*)'Error #600 in genps_fks',yij_sol,icountevts
            write(*,*)xbjrk_cnt(1,0),xbjrk_cnt(2,0),yijdir
          endif
          if(yijdir.ge.yij_sol)then
            xi1=2*(1+yijdir)*x1bar2/(
     #        sqrt( ((1+x1bar2)*(1-yijdir))**2+16*yijdir*x1bar2 ) +
     #        (1-yijdir)*(1-x1bar2) )
            ximaxtmp=1-xi1
          elseif(yijdir.lt.yij_sol)then
            xi2=2*(1-yijdir)*x2bar2/(
     #        sqrt( ((1+x2bar2)*(1+yijdir))**2-16*yijdir*x2bar2 ) +
     #        (1+yijdir)*(1-x2bar2) )
            ximaxtmp=1-xi2
          else
            write(*,*)'Fatal error #14 in one_tree: unknown option'
            write(*,*)y_ij_fks,yij_sol,idir
            stop
          endif
          if(icountevts.eq.-100)then
            xiimax_ev=ximaxtmp
            xinorm_ev=xiimax_ev
          else
            xiimax_cnt(icountevts)=ximaxtmp
            xinorm_cnt(icountevts)=xiimax_cnt(icountevts)
            if( icountevts.ge.1 .and.
     #          ( (idir.eq.1.and.
     #            abs(ximaxtmp-(1-xbjrk_cnt(1,0))).gt.1.d-5) .or.
     #            (idir.eq.-1.and.
     #            abs(ximaxtmp-(1-xbjrk_cnt(2,0))).gt.1.d-5) ) )then 
              write(*,*)'Fatal error #15 in one_tree'
              write(*,*)ximaxtmp,xbjrk_cnt(1,0),xbjrk_cnt(2,0),idir
              stop
            endif
          endif
        else
c
c Parton j is in the final state
c
          xiimax_ev=1-xmrec2/shat_ev
          xinorm_ev=xiimax_ev
          do i=0,2
            xiimax_cnt(i)=xiimax_ev
            xinorm_cnt(i)=xinorm_ev
          enddo
        endif
c
        if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     #      ((.not.colltest) .or. 
     #       (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     #      (.not.softtest)  )then
          if(icountevts.eq.-100)then
c$$$          xi_i_hat=sstiny+(1-sstiny)*x(ixEi)
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->1
            xi_i_hat=sstiny+(1-sstiny)*x(ixEi)**2
            xi_i_fks=xi_i_hat*xiimax_ev
            xi_i_fks_ev=xi_i_fks
          else
            xi_i_fks=xi_i_hat*xiimax_cnt(icountevts)
            xi_i_fks_cnt(icountevts)=xi_i_fks
          endif
        elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     #      (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     #      (.not.softtest)  )then
          if(icountevts.eq.-100)then
            if(xi_i_fks_fix.lt.xiimax_ev)then
              xi_i_fks=xi_i_fks_fix
            else
              xi_i_fks=xi_i_fks_fix*xiimax_ev
            endif
            xi_i_fks_ev=xi_i_fks
          else
            if(xi_i_fks_fix.lt.xiimax_cnt(icountevts))then
              xi_i_fks=xi_i_fks_fix
            else
              xi_i_fks=xi_i_fks_fix*xiimax_cnt(icountevts)
            endif
            xi_i_fks_cnt(icountevts)=xi_i_fks
          endif
        elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     #           softtest )then
          if(icountevts.eq.-100)then
            if(xi_i_fks_ev.lt.xiimax_ev)then
              xi_i_fks=xi_i_fks_ev
            else
              xjac=-102
              goto 112
            endif
          else
c$$$            if(xi_i_fks_cnt(icountevts).lt.xiimax_cnt(icountevts))then
            if(xi_i_fks_cnt(icountevts)*xi_i_fks_ev.lt.xiimax_cnt(icountevts))then
c$$$              xi_i_fks=xi_i_fks_cnt(icountevts)
              xi_i_fks=xi_i_fks_cnt(icountevts)*xi_i_fks_ev
            else
              xjac=-102
              goto 112
            endif
          endif
        elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
          xi_i_fks=xi_i_fks_matrix(icountevts)
          xi_i_fks_cnt(icountevts)=xi_i_fks
        else
          write(*,*)'Error #101 in one_tree',icountevts
          stop
        endif
c
c remove the following if no importance sampling towards soft singularity
c is performed when integrating over xi_i_hat
        xjac=xjac*2d0*x(ixEi)

c Check that xii is in the allowed range
        if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     #      j_fks.gt.nincoming )then
          if(xi_i_fks.gt.(1-xmrec2/shat_ev))then
            xjac=-101
            goto 112
          endif
        elseif(icountevts.eq.0.or.abs(icountevts).eq.2)then
c May insert here a check on whether xii<xicut, rather than doing it 
c in the cross sections
          continue
        endif

        if(j_fks.le.nincoming)then
          omega=sqrt( (2-xi_i_fks*(1+yijdir))/
     #                (2-xi_i_fks*(1-yijdir)) )
          if(icountevts.eq.-100)then
            tau_ev=tau_cnt(0)/(1-xi_i_fks)
            ycm_ev=ycm_cnt(0)-log(omega)
            shat_ev=tau_ev*stot
            sqrtshat_ev=sqrt(shat_ev)
            xbjrk_ev(1)=xbjrk_cnt(1,0)/(sqrt(1-xi_i_fks)*omega)
            xbjrk_ev(2)=xbjrk_cnt(2,0)*omega/sqrt(1-xi_i_fks)
          elseif(abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
            tau_cnt(icountevts)=tau_cnt(0)/(1-xi_i_fks)
            ycm_cnt(icountevts)=ycm_cnt(0)-log(omega)
            shat_cnt(icountevts)=tau_cnt(icountevts)*stot
            sqrtshat_cnt(icountevts)=sqrt(shat_cnt(icountevts))
            xbjrk_cnt(1,icountevts)=
     #        xbjrk_cnt(1,0)/(sqrt(1-xi_i_fks)*omega)
            xbjrk_cnt(2,icountevts)=
     #        xbjrk_cnt(2,0)*omega/sqrt(1-xi_i_fks)
          endif
        endif

      elseif(m(j_fks).ne.0.d0.and.j_fks.gt.nincoming)then
c
c Parton j is massive
c
        if(colltest .or.
     #     abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
          write(*,*)'Error [coll] in one_tree:'
          write(*,*)
     #      'This parametrization cannot be used in FS coll limits'
          stop
        endif
        xmj=m(j_fks)
        xmj2=xmj**2
        xmjhat=xmj/sqrtshat_ev
        xmhat=sqrt(xmrec2)/sqrtshat_ev
        xim=(1-xmhat**2-2*xmjhat+xmjhat**2)/(1-xmjhat)
        cffA2=1-xmjhat**2*(1-y_ij_fks**2)
        cffB2=-2*(1-xmhat**2-xmjhat**2)
        cffC2=(1-(xmhat-xmjhat)**2)*(1-(xmhat+xmjhat)**2)
        cffDEL2=cffB2**2-4*cffA2*cffC2
        xiBm=(-cffB2-sqrt(cffDEL2))/(2*cffA2)
        ximax=1-(xmhat+xmjhat)**2
        if(xiBm.lt.(xim-1.d-8).or.xim.lt.0.d0.or.xiBm.lt.0.d0.or.
     #     xiBm.gt.(ximax+1.d-8).or.ximax.gt.1.or.ximax.lt.0.d0)then
          write(*,*)'Fatal error #4 in one_tree',xim,xiBm,ximax
          xjac=-1d0
          goto 112
        endif
        if(y_ij_fks.ge.0.d0)then
          xirplus=xim
          xirminus=0.d0
        else
          xirplus=xiBm
          xirminus=xiBm-xim
        endif
        xiimax_ev=xirplus
        xiimax_cnt(0)=xiimax_ev
        xinorm_ev=xirplus+xirminus
        xinorm_cnt(0)=xinorm_ev
        rat_xi=xiimax_ev/xinorm_ev
c
        if( icountevts.eq.-100 .and.
     #      ((.not.colltest) .or. 
     #       (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     #      (.not.softtest)  )then
          xjactmp=1.d0
          xitmp1=x(ixEi)
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
            xi_i_fks=xinorm_ev*xi_i_hat
            isolsign=1
          else
c insert here further importance samplings
            xi_i_hat=xitmp1
            xi_i_fks=-xinorm_ev*xi_i_hat+2*xiimax_ev
            isolsign=-1
          endif
          xi_i_fks_ev=xi_i_fks
        elseif( icountevts.eq.-100 .and.
     #       (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     #      (.not.softtest)  )then
          if(xi_i_fks_fix.lt.xiimax_ev)then
            xi_i_fks=xi_i_fks_fix
          else
            xi_i_fks=xi_i_fks_fix*xiimax_ev
          endif
          xi_i_fks_ev=xi_i_fks
          isolsign=1
        elseif( icountevts.eq.-100 .and. softtest )then
          if(xi_i_fks_ev.lt.xiimax_ev)then
            xi_i_fks=xi_i_fks_ev
          else
            xjac=-102
            goto 112
          endif
          isolsign=1
        elseif(icountevts.eq.0)then
          xi_i_fks=xi_i_fks_matrix(icountevts)
          xi_i_fks_cnt(icountevts)=xi_i_fks
          isolsign=1
        else
          write(*,*)'Error #101 in one_tree',icountevts
          stop
        endif
        xjac=xjac*xjactmp

      else
        write(*,*)'Error #104 in one_tree',j_fks,m(j_fks)
        stop
      endif


      if(isolsign.eq.0)then
        write(*,*)'Fatal error #11 in one_tree',isolsign
        stop
      endif

c Now construct the momenta of the fks partons i_fks and j_fks. Do so only
c for final state, and postpone construction in the case of initial state

      if(j_fks.le.nincoming)then
c
c Parton j is massless and initial state
c
        continue

      elseif(j_fks.gt.nincoming.and.m(j_fks).eq.0.d0)then
c
c Parton j is massless and final state
c
        E_i_fks=xi_i_fks*sqrtshat_ev/2d0
        x3len_i_fks=E_i_fks
        x3len_j_fks=(shat_ev-xmrec2-2*sqrtshat_ev*x3len_i_fks)/
     #              (2*(sqrtshat_ev-x3len_i_fks*(1-y_ij_fks)))
        x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     #                         2*x3len_i_fks*x3len_j_fks*y_ij_fks )
        if(xi_i_fks.lt.qtiny)then
          costh_i_fks=y_ij_fks+shat_ev*(1-y_ij_fks**2)*xi_i_fks/
     #                         (shat_ev-xmrec2)
          if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
        elseif(1-y_ij_fks.lt.qtiny)then
          costh_i_fks=1-(shat_ev*(1-xi_i_fks)-xmrec2)**2*(1-y_ij_fks)/
     #                  (shat_ev-xmrec2)**2
          if(abs(costh_i_fks).gt.1.d0)costh_i_fks=1.d0
        else
          costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)/
     #           (2*x3len_fks_mother*x3len_i_fks)
          if(abs(costh_i_fks).gt.1.d0)then
            if(abs(costh_i_fks).le.(1.d0+1.d-5))then
              costh_i_fks=sign(1.d0,costh_i_fks)
            else
              write(*,*)'Fatal error #5 in one_tree',
     #          costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
              stop
            endif
          endif
        endif
      elseif(j_fks.gt.nincoming.and.m(j_fks).ne.0.d0)then
c
c Parton j is massive
c
        E_i_fks=xi_i_fks*sqrtshat_ev/2d0
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
     #                    (1-xmhat**2+xmjhat**2-xi_i_fks) +
     #                  (2-xi_i_fks)*sqrt(b2m4ac)*isolsign
        x3len_j_fks_den=(2-xi_i_fks*(1-y_ij_fks))*
     #                  (2-xi_i_fks*(1+y_ij_fks))
        x3len_j_fks=sqrtshat_ev*x3len_j_fks_num/x3len_j_fks_den
        if(x3len_j_fks.lt.0.d0)then
          write(*,*)'Fatal error #7 in one_tree',
     #      x3len_j_fks_num,x3len_j_fks_den,xi_i_fks,y_ij_fks
          stop
        endif
        x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     #                         2*x3len_i_fks*x3len_j_fks*y_ij_fks )
        if(xi_i_fks.lt.qtiny)then
          costh_i_fks=y_ij_fks+(1-y_ij_fks**2)*xi_i_fks/sqrt(cffC2)
          if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
        else
          costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)/
     #                (2*x3len_fks_mother*x3len_i_fks)
          if(abs(costh_i_fks).gt.1.d0)then
            write(*,*)'Fatal error #8 in one_tree',
     #                costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
            stop
          endif
        endif

      endif

      if(j_fks.le.nincoming)then
c
c Parton j is initial state
c
        bstfact=sqrt( (2-xi_i_fks*(1-yijdir))*(2-xi_i_fks*(1+yijdir)) )
        shy_tbst=-xi_i_fks*sqrt(1-yijdir**2)/(2*sqrt(1-xi_i_fks))
        chy_tbst=bstfact/(2*sqrt(1-xi_i_fks))
        chy_tbstmo=chy_tbst-1.d0
        xdir_t(1)=-cosphi_i_fks
        xdir_t(2)=-sinphi_i_fks
        xdir_t(3)=zero
c
        shy_lbst=-xi_i_fks*yijdir/bstfact
        chy_lbst=(2-xi_i_fks)/bstfact
c Include the following boost if one wants momenta in the incoming parton
c cm frame rather than in the reduced frame
        if(bst_to_pcms)then
          chy_lbstmo=chy_lbst-1.d0
          xdir_l(1)=zero
          xdir_l(2)=zero
          xdir_l(3)=one
        endif
        do i=3,nexternal
          if(i.ne.i_fks.and.shy_tbst.ne.0.d0)
     #      call boostwdir2(chy_tbst,shy_tbst,chy_tbstmo,xdir_t,
     #                      xp(0,i),xp(0,i))
          if(i.ne.i_fks.and.shy_lbst.ne.0.d0.and.bst_to_pcms)
     #      call boostwdir2(chy_lbst,shy_lbst,chy_lbstmo,xdir_l,
     #                      xp(0,i),xp(0,i))
        enddo
c
        if(icountevts.eq.-100)then
          encmso2=sqrtshat_ev/2.d0
          p_i_fks_ev(0)=encmso2
        else
          encmso2=sqrtshat_cnt(icountevts)/2.d0
          p_i_fks_cnt(0,icountevts)=encmso2
        endif

        E_i_fks=xi_i_fks*encmso2
        sinth_i_fks=sqrt(1-costh_i_fks**2)

        if(.not.bst_to_pcms)then
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
        else
          xp(0,1)=encmso2
          xp(1,1)=0.d0
          xp(2,1)=0.d0
          xp(3,1)=xp(0,1)
c
          xp(0,2)=encmso2
          xp(1,2)=0.d0
          xp(2,2)=0.d0
          xp(3,2)=-xp(0,2)
c
          sinth_i_fks=sqrt(1-yijdir**2)
          xp(0,i_fks)=E_i_fks
          xpifksred(1)=sinth_i_fks*cosphi_i_fks
          xpifksred(2)=sinth_i_fks*sinphi_i_fks
          xpifksred(3)=yijdir
c
          write(*,*)'One_tree: this option is not yet tested'
          stop
        endif

        do j=1,3
          xp(j,i_fks)=E_i_fks*xpifksred(j)
          if(icountevts.eq.-100)then
            p_i_fks_ev(j)=encmso2*xpifksred(j)
          else
            p_i_fks_cnt(j,icountevts)=encmso2*xpifksred(j)
          endif
        enddo


        if(icountevts.eq.-100)then
          do j=1,3
            p_i_fks_ev(j)=encmso2*xpifksred(j)
          enddo
        else
          do j=1,3
            p_i_fks_cnt(j,icountevts)=encmso2*xpifksred(j)
          enddo
        endif

c
c Collinear limit of <ij>/[ij]. See innerpin.m. 
        if( icountevts.eq.-100 .or.
     #     (icountevts.eq.1.and.xij_aor.eq.0) )then
          if(j_fks.eq.1)then
            resAoR0=-exp( 2*ximag*phi_i_fks )
          elseif(j_fks.eq.2)then
            resAoR0=-exp( -2*ximag*phi_i_fks )
          endif
          xij_aor=resAoR0
        endif

c Phase-space factor for (xii,yij,phii) * (tau,ycm)

        if(icountevts.eq.-100)then
          xpswgt=xpswgt*shat_ev
        else
          xpswgt=xpswgt*shat_cnt(icountevts)
        endif
        xpswgt=xpswgt/(4*pi)**3/(1-xi_i_fks)
        xpswgt=abs(xpswgt)

      elseif(j_fks.gt.nincoming)then
c
c Parton j is final state
c
        sinth_i_fks=sqrt(1-costh_i_fks**2)
        xpifksred(1)=sinth_i_fks*cosphi_i_fks
        xpifksred(2)=sinth_i_fks*sinphi_i_fks
        xpifksred(3)=costh_i_fks

        xp(0,i_fks)=E_i_fks
        xp(0,j_fks)=sqrt(x3len_j_fks**2+m(j_fks)**2)
        if(icountevts.eq.-100)then
          p_i_fks_ev(0)=sqrtshat_ev/2d0
        else
          p_i_fks_cnt(0,icountevts)=sqrtshat_ev/2d0
        endif
        do j=1,3
           if(icountevts.eq.-100)then
              p_i_fks_ev(j)=sqrtshat_ev/2d0*xpifksred(j)
           else
              p_i_fks_cnt(j,icountevts)=sqrtshat_ev/2d0*xpifksred(j)
           endif
           xp(j,i_fks)=E_i_fks*xpifksred(j)
           if(j.ne.3)then
             xp(j,j_fks)=-xp(j,i_fks)
           else
             xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
           endif
        enddo
  
        call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     #                    costh_mother_fks,sinth_mother_fks,
     #                    cosphi_mother_fks,sinphi_mother_fks)
        call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     #                    costh_mother_fks,sinth_mother_fks,
     #                    cosphi_mother_fks,sinphi_mother_fks)
        if(icountevts.eq.-100)then
          call rotate_invar(p_i_fks_ev,p_i_fks_ev,
     #                      costh_mother_fks,sinth_mother_fks,
     #                      cosphi_mother_fks,sinphi_mother_fks)
        else
          call rotate_invar(p_i_fks_cnt(0,icountevts),
     #                      p_i_fks_cnt(0,icountevts),
     #                      costh_mother_fks,sinth_mother_fks,
     #                      cosphi_mother_fks,sinphi_mother_fks)
        endif

c Now the xp four vectors of all partons except i_fks and j_fks will be 
c boosted along the direction of the mother; start by redefining the
c mother four momenta
        do i=0,3
          xp(i,fksmother)=xp(i,i_fks)+xp(i,j_fks)
          if( p_born(i,1).ne.xp(i,1) .or.
     #        p_born(i,2).ne.xp(i,2) )then
            write(*,*)'Fatal error #9 in one_tree'
            stop
          endif
          recoil(i)=xp(i,1)+xp(i,2)-xp(i,fksmother)
        enddo

        sumrec=recoil(0)+rho(recoil)

        if(m(j_fks).eq.0.d0)then
c
c Parton j is massless
c
          sumrec2=sumrec**2
          betabst=-(shat_ev-sumrec2)/(shat_ev+sumrec2)
          gammabst=1/sqrt(1-betabst**2)
          shybst=-(shat_ev-sumrec2)/(2*sumrec*sqrtshat_ev)
          chybst=(shat_ev+sumrec2)/(2*sumrec*sqrtshat_ev)
c cosh(y) is very often close to one, so define cosh(y)-1 as well
          chybstmo=(sqrtshat_ev-sumrec)**2/(2*sumrec*sqrtshat_ev)
        else
c
c Parton j is massive
c
          if(xmrec2.lt.1.d-16*shat_ev)then
            expybst=sqrtshat_ev*sumrec/(shat_ev-xmj2)*
     #              (1+xmj2*xmrec2/(shat_ev-xmj2)**2)
          else
            expybst=sumrec/(2*sqrtshat_ev*xmrec2)*
     #              (shat_ev+xmrec2-xmj2-shat_ev*sqrt(cffC2))
          endif
          if(expybst.le.0.d0)then
            write(*,*)'Fatal error #10 in one_tree',expybst
            stop
          endif
          shybst=(expybst-1/expybst)/2.d0
          chybst=(expybst+1/expybst)/2.d0
          chybstmo=chybst-1.d0

        endif

        do j=1,3
          xdir(j)=xp(j,fksmother)/x3len_fks_mother
        enddo
        do i=3,nexternal
          if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     #      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
        enddo

c Collinear limit of <ij>/[ij]. See innerp3.m. 
        if( ( icountevts.eq.-100 .or.
     #       (icountevts.eq.1.and.xij_aor.eq.0) ) .and.
     #      m(j_fks).eq.0.d0 )then
          resAoR0=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c The term O(srt(1-y)) is formally correct but may be numerically large
c Set it to zero
c$$$          resAoR5=-ximag*sqrt(2.d0)*
c$$$       #          sinphi_i_fks*tan(th_mother_fks/2.d0)*
c$$$       #          exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c$$$          xij_aor=resAoR0+resAoR5*sqrt(1-y_ij_fks)
          xij_aor=resAoR0
        endif

c Phase-space factor for (xii,yij,phii)
        veckn=rho(xp(0,j_fks))
        veckbarn=rho(p_born(0,imother))

c Qunatities to be passed to montecarlocounter (event kinematics)
        if(icountevts.eq.-100)then
           veckn_ev=veckn
           veckbarn_ev=veckbarn
           xp0jfks=xp(0,j_fks)
        endif 

        xpswgt=xpswgt*2*shat_ev/(4*pi)**3*veckn/veckbarn/
     #    ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
        xpswgt=abs(xpswgt)
      endif


c At this point, the phase space lacks a factor xi_i_fks, which need be 
c excluded in an NLO computation according to FKS, being taken into 
c account elsewhere
      if(.not.int_nlo)xpswgt=xpswgt*xi_i_fks

      xjac = xjac*xwgt
      if (.not. pass) then
         xjac = -99
         goto 112
      endif
c All done, so check four-momentum conservation
      if(xjac.gt.0.d0)then
        if(icountevts.eq.-100)then
          call phspncheck_nocms(nexternal,sqrtshat_ev,m,xp)
        else
          call phspncheck_nocms(nexternal,sqrtshat_cnt(icountevts),m,xp)
        endif
      endif

      do i=1,nexternal
        do j=0,3
          if(icountevts.eq.-100)then
            p(j,i)=xp(j,i)
          else
            p1_cnt(j,i,icountevts)=xp(j,i)
          endif
        enddo
      enddo

 112  continue

      if (xjac .gt. 0d0 ) then
         if(nincoming.eq.2)then
            if(icountevts.eq.-100)then
              flux  = 1d0 /(2.D0*SQRT(LAMBDA(shat_ev,m(1)**2,m(2)**2)))
            else
              flux  = 1d0 /(2.D0*SQRT(LAMBDA(shat_cnt(icountevts),
     #                                       m(1)**2,m(2)**2)))
            endif
         else                   ! Decays
            if(icountevts.eq.-100)then
               flux = 1d0/(2d0*sqrtshat_ev)
            else
               flux = 1d0/(2d0*sqrtshat_cnt(icountevts))
            endif
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
      else
         if (abs(icountevts).le.2) p1_cnt(0,1,icountevts)=-99
      endif


c Restore ranmar seeds if generation has not been completed
      if(samplevar.and.int_nlo.and.(.not.xexternal))then
        do i=1,97
          ranu(i)=ranu_save(i)
        enddo
        ranc=ranc_save
        rancd=rancd_save
        rancm=rancm_save
        iranmr=iranmr_save
        jranmr=jranmr_save
      endif

      if(icountevts.eq.-100)then
        jac=xjac*savejac
        wgt=xwgt
        pswgt=xpswgt
        if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
          icountevts=-2
        else
          icountevts=0
        endif
c skips counterevents when testing or when integrating over second fold
c for massive j_fks
        if( (.not.int_nlo) .or. isolsign.eq.-1 )icountevts=5 
      else
        jac_cnt(icountevts)=xjac*savejac
        wgt_cnt(icountevts)=xwgt
        pswgt_cnt(icountevts)=xpswgt
        icountevts=icountevts+1
      endif

      if( (icountevts.le.2.and.m(j_fks).eq.0.d0.and.(.not.nbodyonly)) .or.
     #    (icountevts.eq.0.and.m(j_fks).eq.0.d0.and.nbodyonly) .or.
     #    (icountevts.eq.0.and.m(j_fks).ne.0.d0) )then
         goto 111
      else
         if (samplevar .and. jac.lt.0d0 .and. jac_cnt(0).lt.0d0 .and.
     &        ( m(j_fks).ne.0d0 .or.
     &        ( m(j_fks).eq.0d0 .and.
     &          jac_cnt(1).lt.0d0.and.jac_cnt(2).lt.0d0 ) ) .and.
     &        (.not.xexternal) )then
            call sample_get_x(dummy,dummy,ix,iconfig,0d0,1d0)
         endif
c icountevts=5 only when testing, and when integrating over the second fold
c with j_fks massive. The counterevents have been skipped, so make sure their
c momenta are unphysical. Born are physical if event was generated, and must
c stay so for the computation of enhancement factors.
         if(icountevts.eq.5)then
           do i=0,2
             jac_cnt(i)=-299
             p1_cnt(0,1,i)=-99
           enddo
         endif
      endif

      if(.not.xexternal)then
        goodx=.true.
        do i=1,ndim
          goodx=goodx.and.(x(i).ge.0.d0.and.x(i).le.1.d0)
        enddo
        if(.not.goodx)then
          jac=-222
          p(0,1)=-99
          do i=-2,2
            p1_cnt(0,1,i)=-99
            jac_cnt(i)=-222
          enddo
          p_born(0,1)=-99
        endif
      endif

      nocntevents=(jac_cnt(0).le.0.d0) .and.
     #            (jac_cnt(1).le.0.d0) .and.
     #            (jac_cnt(2).le.0.d0)

      call xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,
     #                        p_i_fks_ev,p_i_fks_cnt,
     #                        xi_i_fks_ev,y_ij_fks_ev)

      return

 222  continue

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


      subroutine boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
c chybstmo = chybst-1; if it can be computed analytically it improves
c the numerical accuracy
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xin(0:3),xout(0:3)
      real*8 tmp,en,pz
      integer i
c
      if(abs(xd(1)**2+xd(2)**2+xd(3)**2-1).gt.1.d-6)then
        write(*,*)'Error #1 in boostwdir2',xd
        stop
      endif
c
      en=xin(0)
      pz=xin(1)*xd(1)+xin(2)*xd(2)+xin(3)*xd(3)
      xout(0)=en*chybst-pz*shybst
      do i=1,3
        xout(i)=xin(i)+xd(i)*(pz*chybstmo-en*shybst)
      enddo
c
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


      subroutine gen_s(x,smin,smax,spole,swidth,s,jac,pass)
c*************************************************************************
c     Given a random number x, the limits smin and smax and also
c     any pole spole with swidth, returns s ans jac the jacobian
c     for the transformation.  The jacobian is just multiplied by the
c     new jacobian so if jac=0 on entry jac=0 on exit
c*************************************************************************
      implicit none
c
c     Arguments
c
      double precision smin,smax,spole,swidth,s,jac
      double precision x
      logical pass
c
c     Local
c     
      logical warned0
c
c     Data
c
      data warned0 /.false./
c-----
c  Begin Code
c-----
      pass=.true.
      if (jac .eq. 0 .and. .not. warned0) then
         print*,'Input jacobian 0 in genps'
         warned0 = .true.
      endif
      if (spole .eq. 0d0) then
         s = (smax-smin)*x + smin
         jac = jac*(smax-smin)
      else
         if (spole*spole .lt. smax) then
            CALL TRANSPOLE(spole*spole/smax,spole*swidth/smax,x,s,jac)
            s = s*smax
            jac = jac*smax
         else
            pass=.false.
         endif
      endif
      if (s .gt. smax .or. s .lt. smin) then
         pass = .false.
      endif
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
            write(*,*) 'Error creating momentum in gentcms',pp
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

      p1(0) = MAX((ESUM+ED)*0.5d0,0.d0)
      p1(3) = -(m1*m1+ma2-t-2d0*p1(0)*E_acms)/(2d0*p_acms)
      pt = dsqrt(max(pp*pp-p1(3)*p1(3),0d0))
      p1(1) = pt*cos(phi)
      p1(2) = pt*sin(phi)

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
          write(6,*)'Error #2 in function Lambda:',s,ma2,mb2
        endif
      endif
      LAMBDA=tmp
      RETURN
      END


      DOUBLE PRECISION FUNCTION G(X,Y,Z,U,V,W)
C**************************************************************************
C     This is the G function from Particle Kinematics by
C     E. Byckling and K. Kajantie, Chapter 4 p. 89 eqs 5.23
C     It is used to determine if a set of invarients are physical or not
C**************************************************************************
      implicit none
c
c     Arguments
c
      Double precision x,y,z,u,v,w
c-----
c  Begin Code
c-----
      G = X*Y*(X+Y-Z-U-V-W)+Z*U*(Z+U-X-Y-V-W)+V*W*(V+W-X-Y-Z-U)
     &     +X*Z*W +X*U*V +Y*Z*W +Y*U*W
      end

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

      subroutine ungen_s(x,smin,smax,spole,swidth,s,jac,pass)
c*************************************************************************
c     Given s, the limits smin and smax and also
c     any pole spole with swidth, returns x and jac the jacobian
c     for the transformation.  The jacobian is just multiplied by the
c     new jacobian so if jac=0 on entry jac=0 on exit
c*************************************************************************
      implicit none
c
c     Arguments
c
      double precision smin,smax,spole,swidth,s,jac,x
      logical pass
c
c     Local
c     
      logical warned0
c
c     Data
c
      data warned0 /.false./
c-----
c  Begin Code
c-----
      pass=.true.
      if (jac .eq. 0 .and. .not. warned0) then
         print*,'Input jacobian 0 in genps'
         warned0 = .true.
      endif
      if (spole .eq. 0d0) then
         x = (s-smin)/(smax-smin)
         jac = jac*(smax-smin)
      else
         if (spole*spole .lt. smax) then
            s = s/smax
            CALL UNTRANSPOLE(spole*spole/smax,spole*swidth/smax,x,s,jac)
            s = s*smax
            jac = jac*smax
         else
            pass=.false.
            print*,'Skipping BW pole pass=',pass,spole*spole,smax
         endif
      endif
      if (s .gt. smax .or. s .lt. smin) then
         pass = .false.
      endif
      end


      SUBROUTINE GENCMS(S,X1,X2,X,SMIN,SJACOBI)
C***********************************************************************
C     PICKS PARTON MOMENTUM FRACTIONS X1 AND X2 BY CHOOSING ETA AND TAU
C     X(1) --> TAU = X1*X2
C     X(2) --> ETA = .5*LOG(X1/X2)
C***********************************************************************
      IMPLICIT NONE

C     ARGUMENTS

      DOUBLE PRECISION X1,X2,S,SMIN,SJACOBI
      DOUBLE PRECISION X(2)

C     LOCAL

      DOUBLE PRECISION TAU,TAUMIN,TAUMAX
      DOUBLE PRECISION ETA,ETAMIN,ETAMAX
      logical warned
      data warned/.false./

C------------
C  BEGIN CODE
C------------

      IF (S .LT. SMIN) THEN
         PRINT*,'ERROR CMS ENERGY LESS THAN MINIMUM CMS ENERGY',S,SMIN
         RETURN
      ENDIF

C     TO FLATTEN BRIET WIGNER POLE AT WMASS WITH WWIDTH USE BELOW:
C      CALL TRANSPOLE(REAL(WMASS**2/S),REAL(WMASS*WWIDTH/S),
C     &     X(1),TAU,SJACOBI)

C     IF THERE IS NO S CHANNEL POLE USE BELOW:

      TAUMIN = SMIN/S
      TAUMAX = 1D0
      TAU    = (TAUMAX-TAUMIN)*X(1)+TAUMIN
      SJACOBI=  sjacobi*(TAUMAX-TAUMIN)

C     FROM HERE ON SAME WITH OR WITHOUT POLE
      ETAMIN = .5d0*LOG(TAU)
      ETAMAX = -ETAMIN
      ETA    = (ETAMAX-ETAMIN)*X(2)+ETAMIN
c      if (.not. warned) then
c         write(*,*) 'Fixing eta = 0'
c         warned=.true.
c      endif
c      eta = 0d0

      SJACOBI = SJACOBI*(ETAMAX-ETAMIN)

      X1 = SQRT(TAU)*EXP(ETA)
      X2 = SQRT(TAU)*EXP(-ETA)

      END
