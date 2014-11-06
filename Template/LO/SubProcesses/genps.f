      integer function f_get_nargs(ndim)
c**************************************************************************
c     Returns number of arguments which come from x_to_f_arg
c**************************************************************************
      implicit none
      include 'nexternal.inc'
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
c     COMMON:
c           hel_picked  == Modified integer in gen_ps.inc to pass the
c                          chosen helicity configuration to matrix<i>.f
c
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
c
c     Arguments
c
      integer ndim                             !Number of dimensions(input)
      integer iconfig                          !Configuration (input)
      integer mincfig,maxcfig                  !Range of configurations
      integer invar
      double precision wgt                     !(input and output)
      double precision x(maxdim),p(maxdim)     !x,p (output) [p(0:3,nexternal)]
c
c     Local
c
c
c     External
c     
c
c     Global
c
      INTEGER                    ISUM_HEL
      LOGICAL                    MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
c-----
c  Begin Code
c-----
      call gen_mom(iconfig,mincfig,maxcfig,invar,wgt,x,p)
C     Pick the helicity configuration from the DiscreteSampler if user
C     decided to perform MC over helicity configurations.
      if(ISUM_HEL.ne.0) then
        call sample_get_discrete_x(wgt,hel_picked,iconfig,'Helicity')
      endif
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
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
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
      double precision xtau, dum
      double precision pi1(0:3),pi2(0:3),p0,p3
      save m

      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid
      integer          lwgt(0:maxconfigs,maxinvar)
      logical firsttime

      double precision xprop(3,nexternal),tprop(3,nexternal)
      double precision maxwgt
      integer imatch
      save maxwgt

      integer ninvar, nconfigs
      
c
c     External
c
      double precision lambda,dot,dsig
      logical passcuts
c
c     Global
c
      double precision pmass(nexternal)
      common/to_mass/  pmass

      double precision SMIN
      common/to_smin/ smin

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

      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      save lwgt
      save ndim,nfinal,nbranch,nparticles

      integer jfig,k

      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap      

c     External function
      double precision SumDot
      external SumDot

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
      if (firsttime) then
         firsttime=.false.
         do i=1,nexternal
            m(i)=pmass(i)
         enddo
c        Set stot
         if (nincoming.eq.1) then
            stot=m(1)**2
         else
            m1=m(1)
            m2=m(2)
            if (abs(lpp(1)) .eq. 1 .or. abs(lpp(1)) .eq. 2) m1 = 0.938d0
            if (abs(lpp(2)) .eq. 1 .or. abs(lpp(2)) .eq. 2) m2 = 0.938d0
            if (abs(lpp(1)) .eq. 3) m1 = 0.000511d0
            if (abs(lpp(2)) .eq. 3) m2 = 0.000511d0
            if(ebeam(1).lt.m1) ebeam(1)=m1
            if(ebeam(2).lt.m2) ebeam(2)=m2
            pi1(0)=ebeam(1)
            pi1(3)=sqrt(ebeam(1)**2-m1**2)
            pi2(0)=ebeam(2)
            pi2(3)=-sqrt(ebeam(2)**2-m2**2)
            stot=m1**2+m2**2+2*(pi1(0)*pi2(0)-pi1(3)*pi2(3))
         endif
         write(*,'(x,a,f13.2)') 'Set CM energy to ',sqrt(stot)
c        Start graph mapping
         do i=1,mapconfig(0)
            if (mapconfig(i) .eq. iconfig) this_config=i
         enddo
         write(*,*) 'Mapping Graph',iconfig,' to config',this_config
         iconfig = this_config
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
         if (ndim .lt. 0) ndim = 0   !For 2->1 processes  tjs 5/24/2010
         if (abs(lpp(1)) .ge. 1) ndim=ndim+1
         if (abs(lpp(2)) .ge. 1) ndim=ndim+1
         call set_peaks
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

c     Initialize dsig (needed for subprocess group running mode)
         dum=dsig(0,0,1)

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
C
C     Get fraction of beam energy if pdf's are used
c
      xbk(1)   = 1d0
      xbk(2)   = 1d0
      sjac = 1d0
      if (abs(lpp(1)) .ge. 1 .and. abs(lpp(2)) .ge. 1) then
         call sample_get_x(sjac,x(ndim-1),ndim-1,mincfig,0d0,1d0)
c-----
c tjs 5/24/2010 for 2->1 process
c-------
         xtau = x(ndim-1)
         if(nexternal .eq. 3) then
            x(ndim-1) = pmass(3)*pmass(3)/stot
            sjac=1 / stot     !for delta function in d_tau
         endif

         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
         CALL GENCMS(STOT,Xbk(1),Xbk(2),X(ndim-1), SMIN,SJAC)
         x(ndim-1) = xtau                   !Fix for 2->1 process
c        Set CM rapidity for use in the rap() function
         cm_rap=.5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) = xbk(1)*xbk(2)*stot
      elseif (abs(lpp(1)) .ge. 1) then
         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
         xbk(1) = x(ndim)
c        Set CM rapidity for use in the rap() function
         p0=xbk(1)*ebeam(1)+ebeam(2)
         p3=xbk(1)*ebeam(1)-sqrt(ebeam(2)**2-m2**2)
         cm_rap=.5d0*dlog((p0+p3)/(p0-p3))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) = m2**2+2*xbk(1)*ebeam(1) *
     $                 (ebeam(2)+sqrt(ebeam(2)**2-m2**2))
      elseif (abs(lpp(2)) .ge. 1) then
         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
         xbk(2) = x(ndim)
c        Set CM rapidity for use in the rap() function
         p0=ebeam(1)+xbk(2)*ebeam(2)
         p3=sqrt(ebeam(1)**2-m1**2)-xbk(2)*ebeam(2)
         cm_rap=.5d0*dlog((p0+p3)/(p0-p3))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) = m1**2+2*(ebeam(1)+sqrt(ebeam(1)**2-m1**2))
     $                 * xbk(2)*ebeam(2)
      else
c        Set CM rapidity for use in the rap() function
         p0=ebeam(1) + ebeam(2)
         p3=sqrt(ebeam(1)**2-m1**2)-sqrt(ebeam(2)**2-m2**2)
         cm_rap=.5d0*dlog((p0+p3)/(p0-p3))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) = stot
      endif
c      write(*,*) "shat=",sqrt(s(-nbranch))
      m(-nbranch)  = sqrt(s(-nbranch))
      p(0,-nbranch)= m(-nbranch)
      p(1,-nbranch)= 0d0
      p(2,-nbranch)= 0d0
      p(3,-nbranch)= 0d0

c
c     First Generate Momentum for initial state particles
c
      if(nincoming.eq.2) then
        call mom2cx(m(-nbranch),m(1),m(2),1d0,0d0,p(0,1),p(0,2))
      else
        do i=0,3
          p(i,1)=p(i,-nbranch)
        enddo
        p(3,1)=1e-14 ! For HELAS routine ixxxxx for neg. mass
      endif
      pswgt = 1d0
      jac   = 1d0
      call one_tree(iforest(1,-max_branch,iconfig),mincfig,
     &     nbranch,P,M,S,X,jac,pswgt)
c
c     Add what I think are the essentials
c
         if (jac .gt. 0d0 ) then
            if(nincoming.eq.2)then
               flux  = 1d0 /(2.D0*SQRT(LAMBDA(s(-nbranch),m(1)**2,m(2)**2)))
            else ! Decays
               flux = 1d0/(2d0*sqrt(stot))
            endif
            flux  = flux / (2d0*pi)**(3 * nfinal - 4)
            pwgt(1)=max(sjac*jac*pswgt*wgt,1d-99)
            wgt = pwgt(1)*flux
            do i=1,nparticles
               do j=0,3
                  p1(j,i) = p(j,i)
               enddo
c               write(*,'(i3,4f15.5)') i,(p(j,i),j=0,3)
            enddo
            p1(0,nparticles+1)=xbk(1)
            p1(1,nparticles+1)=xbk(2)
         else
            p1(0,1)=-99
         endif

c
c     comment out everything else
c
      if (.false.) then

      if (jac .gt. 0d0) then
         do i=1,nparticles
            do j=0,3
               p1(j,i) = p(j,i)
            enddo
c            write(*,'(i2,4e15.5)') i,(p(j,i),j=0,3)
         enddo
c         call fill_invarients(nfinal,p1,stot,xx)
         if (abs(lpp(1)) .ge. 1 .and. abs(lpp(1)) .ge. 1) then
            xx(invar)=x(ndim)
            xx(invar-1)=x(ndim-1)
         elseif (abs(lpp(1)) .ge. 1 .or. abs(lpp(1)) .ge. 1) then
            xx(invar)=x(ndim)
         endif
c
c     The next lines take care of redetermining the wgt based on all
c     allowed configurations
c
c         if (passcuts(p(0,1))) then
         if (.true.) then
c            write(*,'(A,5e14.4)') 'Orig',jac,pswgt,wgt,jac*pswgt*wgt

            tprb=0d0
c            write(*,'(a,12f6.2)') 'orig',(x(i),i=1,ndim)

            do jconfig=mincfig,maxcfig

c
c     I'll skip this if there is only one configuration
c

               if (jac .lt. 0) then
                  write(*,*) 'Error did not find wgt ',jconfig,jac
               endif

c               write(*,'(a,12f6.2)') 'final',(x(i),i=1,ndim)

c
c     Now need to set up the x(i)'s
c
               do i=1,ndim
                  if (minvar(i,jconfig) .ne. 0) then
c                    write(*,*)'filling',minvar(i,jconfig),i,jconfig,x(i)
                     xx(minvar(i,jconfig))=x(i)
                  endif
               enddo
c
c     Here is where we put in something for pp etc. Now lets just use 1
c
c               if (pp) then
c                  xx(3*nbranch-3,jconfig)= x


               if (maxcfig-mincfig .gt. 0) then
               
                  pwgt(jconfig)=max(sjac*jac*pswgt*wgt,1d-99)
                  prb(jconfig,jpnt,jplace)=1d0/pwgt(jconfig)
                  tprb = tprb + prb(jconfig,jpnt,jplace)*alpha(jconfig)
               else
                  pwgt(1)=max(sjac*jac*pswgt*wgt,1d-99)
                  prb(1,jpnt,jplace)=1d0/pwgt(1)
                  tprb = tprb + prb(1,jpnt,jplace)
               endif
c               write(*,'(A,5e14.4)') 'Finl',jac,pswgt,wgt,jac*pswgt*wgt



            enddo

            pswgt=1d0
            jac  =1d0/tprb

         else
            jac=-1d0              !Didn't pass the cuts.
         endif
         if (jac .lt. 0) then
            p1(0,1) = -999
            return
         endif
c
c     Flux factor and pi's from delta functions
c
         flux  = 1d0 /(2.D0*SQRT(LAMBDA(s(-nbranch),m(1)**2,m(2)**2)))
         flux  = flux / (2d0*pi)**(3 * nfinal - 4)
c
c     Make sure all the invarients are reasonable
c         

         do i=1,invar
            x(i)=xx(i)
            if (x(i) .gt. 1d0) then
               write(*,'(a,i6,e15.4)') 'Error x>1',i,x(i)
            endif
            fprb(i,jpnt,jplace)=0d0
         enddo

c
c     Determine fractional probs from the different configs this
c     allows me to only include points which were contributed
c     by the projection onto that invarient.
c
c
c     Now normalize fprb
c
         do i=1,invar
c            if (fprb(i,jpnt,jplace) .eq. 0d0) then
            if (tprb .eq. 0d0) then
               fprb(i,jpnt,jplace)=1d0
            else
               fprb(i,jpnt,jplace)=fprb(i,jpnt,jplace)/tprb
            endif
         enddo
         
         
c         write(123,'(2i6,1e15.5)') jpnt,jplace
c         write(123,'(5e15.9)') (fprb(i,jpnt,jplace),i=1,invar) 
c         write(123,'(5e15.9)') (prb(i,jpnt,jplace),i=1,maxcfig) 

c
c     Return the 4 momentum if things worked.
c

c         write(*,'(11f7.4)')(x(i),i=1,invar)

         if (jac .gt. 0d0 ) then
            wgt = jac*flux
            do i=1,nparticles
               do j=0,3
                  p1(j,i) = p(j,i)
               enddo
            enddo
            p1(0,nparticles+1)=xbk(1)
            p1(1,nparticles+1)=xbk(2)
         else
            p1(0,1)=-99
         endif
      else                    !Failed cuts
c         write(*,*) 'Failed cuts'
         p1(0,1)=-99           
      endif
c
c     comment out everything funny here
c
      endif
      end


      subroutine one_tree(itree,iconfig,nbranch,P,M,S,X,jac,pswgt)
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
      include 'nexternal.inc'
      double precision pi            , one
      parameter       (pi=3.1415926d0, one=1d0)
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
      integer nbranch
      double precision x(40) ! ja 3/2/11 21->40 after strange segfault
c
c     Local
c
      logical pass
      integer ibranch,i,ns_channel,nt_channel,ix  !,nerr
c      data nerr/0/
      double precision smin,smax,totmass,totmassin,xa2,xb2,wgt
      double precision costh,phi,tmin,tmax,t
      double precision ma2,mb2,m12,mn2,s1
c
c     External
c
      double precision lambda,dot
c
c     Global
c
      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      include 'run.inc'

c-----
c  Begin Code
c-----
      jac   = 1d0
      pswgt = 1d0
      wgt   = 1d0
      pass = .true.
c-----------
c     Trap for trivial case 2->1
c----------
      if (nexternal .eq. 3 .and. nincoming.ne.1) then
         do i=0,3
            p(i,3) = p(i,1)+p(i,2)
         enddo
         return
      endif
c
c     Determine number of s channel branches, this doesn't count
c     the s channel p1+p2
c
      ns_channel=1
      do while(itree(1,-ns_channel) .ne. 1 .and.ns_channel.lt.nbranch)
         m(-ns_channel)=0d0                 
         ns_channel=ns_channel+1         
      enddo
      ns_channel=ns_channel - 1
      nt_channel=nbranch-ns_channel-1

      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         ns_channel=ns_channel-1
      endif

c
c     Determine masses for all intermediate states.  Starting
c     from outer most (real particle) states
c
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
      if (sqrt(s(-nbranch)) .lt. max(totmass,totmassin)) then
         pass=.false.
         jac = -5d0
         return
      endif

      do ibranch = -1,-ns_channel,-1
         smin = (m(itree(1,ibranch))+m(itree(2,ibranch)))**2
         smax = min((dsqrt(s(-nbranch))-totmass+sqrt(smin))**2, stot)
c     Check for NAN - ja 3/11
         if (smax/stot.eq.smax/stot+1d0) then
            print *,'got NaN: ',smax/stot
            jac = -2
            return
         endif


c         write(*,*) ibranch,sqrt(smin),sqrt(smax)
c
c        Choose the appropriate s given our constraints smin,smax
c     
         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
     &        smin/stot,smax/stot)
         s(ibranch) = x(-ibranch)*stot

c         write(*,*) 'using s',-ibranch

         jac = jac*stot
         if (jac .lt. 0d0 .or. .not. pass) then
            jac = -6
            return
         endif
         if (s(ibranch) .lt. smin) then
            jac=-5
            return
         endif
c
c     Check that s is ok, and fill masses, update totmass
c
         m(ibranch) = sqrt(s(ibranch))
         totmass=totmass+m(ibranch)-
     &        m(itree(1,ibranch))-m(itree(2,ibranch))
         if (totmass .gt. M(-nbranch)) then
            jac = -4
            return
         endif
         if (.not. pass) then
            jac=-9
            return
         endif
      enddo

      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         s(-nbranch+1)=s(-nbranch) 
         m(-nbranch+1)=m(-nbranch)      !Basic s-channel has s_hat 
         p(0,-nbranch+1) = m(-nbranch+1)!and 0 momentum
         p(1,-nbranch+1) = 0d0
         p(2,-nbranch+1) = 0d0
         p(3,-nbranch+1) = 0d0
      endif

c
c     Next do the T-channel branchings
c
c
c     First we need to determine the energy of the remaining particles
c     this is essentially in place of the cos(theta) degree of freedom
c     we have with the s channel decay sequence
c


      if (nt_channel .gt. 0) then               !t-channel stuff exists

      totmass=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totmass=totmass+m(itree(2,ibranch))
      enddo
      m(-ns_channel-1) = dsqrt(S(-nbranch))
      do ibranch = -ns_channel-1,-nbranch+2,-1    !Choose invarient mass
         totmass=totmass-m(itree(2,ibranch))      !for remaining particles
         smin = totmass**2                        !This affects t_min/max
         smax = (m(ibranch) - m(itree(2,ibranch)))**2

         if (smin .gt. smax) then
            jac=-3d0
            return
         endif
         call sample_get_x(wgt,x(nbranch-1+(-ibranch)*2),
     &        nbranch-1+(-ibranch)*2,iconfig,
     &        smin/stot,smax/stot)

         m(ibranch-1)=dsqrt(max(stot*x(nbranch-1+(-ibranch)*2), 0d0))
c         write(*,*) 'Using s',nbranch-1+(-ibranch)*2

         if (m(ibranch-1)**2.lt.smin.or.m(ibranch-1)**2.gt.smax
     $        .or.m(ibranch-1).ne.m(ibranch-1)) then
            jac=-1d0
            return
         endif
         jac = jac * stot
      enddo
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
         s1  = m(ibranch)**2                        !Total mass available
         ma2 = m(2)**2
         mb2 = dot(P(0,itree(1,ibranch)),P(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mn2 = m(ibranch-1)**2
c         write(*,*) 'Enertering yminmax',sqrt(s1),sqrt(m12),sqrt(mn2)
         call yminmax(s1,t,m12,ma2,mb2,mn2,tmin,tmax)
c
c     Call for 0<x<1
c
c         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     &        .5d0*(tmin/stot+1d0),
c     &        .5d0*(tmax/stot+1d0))
c         t   = Stot*(x(-ibranch)*2d0-1d0)
c
c     call for -1<x<1
c

c         write(*,*) 'tmin, tmax',tmin,tmax

         tmax = max(tmax,0d0) !This line if want really t freedom

         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
     $        -tmax/stot, -tmin/stot)
         t = stot*(-x(-ibranch))
c
c     now reset tmax if messed it up for t freedom 3 lines above
c
         call yminmax(s1,t,m12,ma2,mb2,mn2,tmin,tmax) 

c         write(*,*) tmin,t,tmax
c         if (t .eq. 0d0) then
c            jac = -3
c            return
c         endif
         if (t .lt. tmin .or. t .gt. tmax) then
            jac=-3d0
            return
         endif
c
c     tmin and tmax set to -s,+s for jacobian because part of jacobian
c     was determined from choosing the point x from the grid based on
c     tmin and tmax.  (ie wgt contains some of the jacobian)
c
         tmin=-stot
         tmax= stot
         call sample_get_x(wgt,x(nbranch+(-ibranch-1)*2),
     &        nbranch+(-ibranch-1)*2,iconfig,0d0,1d0)
         phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
         jac = jac*(tmax-tmin)*2d0*pi /2d0              ! I need /2d0 if -1<x<1
c
c     Finally generate the momentum. The call is of the form
c     pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c     gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
c
         call gentcms(p(0,itree(1,ibranch)),p(0,2),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),p(0,itree(2,ibranch)),
     &        p(0,ibranch),jac)

         if (jac .lt. 0d0) then
c            nerr=nerr+1
c            if(nerr.le.5)
c     $           write(*,*) 'Failed gentcms',iconfig,ibranch
            return              !Failed, probably due to negative x
         endif

         pswgt = pswgt/(4d0*dsqrt(lambda(s1,ma2,mb2)))
      enddo
c
c     We need to get the momentum of the last external particle.
c     This should just be the sum of p(0,2) and the remaining
c     momentum from our last t channel 2->2
c
      do i=0,3
         p(i,itree(2,-nbranch)) = p(i,-nbranch+1)+p(i,2)
      enddo


      endif                     !t-channel stuff


c
c     Now generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
c      do i = -ns_channel,-1      
c      ix=0
c      if (nt_channel .eq. 0) ix=-1
      do i = -nbranch+nt_channel+(nincoming-1),-1         !Loop over all s-channel poles
         ix = nbranch+(-i-1)*2+(2-nincoming)
         if (nt_channel .eq. 0) ix=ix-1

c         write(*,*) 'using costh,phi',ix,ix+1

         call sample_get_x(wgt,x(ix),ix,iconfig,0d0,1d0)
         costh= 2d0*x(ix)-1d0
         call sample_get_x(wgt,x(ix+1),ix+1,iconfig,0d0,1d0)
         phi  = 2d0*pi*x(ix+1)
         jac = jac * 4d0*pi
         xa2 = m(itree(1,i))*m(itree(1,i))/(s(i)+1d-99)
         xb2 = m(itree(2,i))*m(itree(2,i))/(s(i)+1d-99)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            jac=-8
            return
         endif
         pswgt = pswgt*.5D0*PI*SQRT(MAX(LAMBDA(ONE,XA2,XB2),0d0))/(4.D0*PI)
         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        p(0,itree(1,i)),p(0,itree(2,i)))
         call boostm(p(0,itree(1,i)),p(0,i),m(i),p(0,itree(1,i)))
         call boostm(p(0,itree(2,i)),p(0,i),m(i),p(0,itree(2,i)))
      enddo
      jac = jac*wgt
      if (.not. pass) jac = -99
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
            write(*,*) 'Error gentcms',pp, M1,m2,MD2, ESUM
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
      DOUBLE PRECISION MA2,MB2,S
      LAMBDA=S**2+MA2**2+MB2**2-2d0*S*MA2-2d0*MA2*MB2-2d0*S*MB2
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
      if (ysqr .ge. tiny) then
         yr = dsqrt(ysqr)
      else
c        Probably a problem with negative x selection
c         print*,'Error in yminymax sqrt(-x)',lambda(x,u,v),lambda(x,w,z)
         yr = tiny
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
