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
      double precision x(*),p(*)     !x,p (output) [p(0:3,nexternal)]
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
      use DiscreteSampler
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
      double precision x(*)
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
      logical firsttime

      double precision xprop(3,nexternal),tprop(3,nexternal)
      double precision maxwgt
      integer imatch
      save maxwgt
      double precision R        !random value
      integer ninvar, nconfigs
      double precision tau_m, tau_w, t1, t2
      double precision get_ee_expo

      
c
c     External
c
      double precision lambda,dot,dsig
      logical passcuts
c
c     Global
c
C     Common blocks
      include '../../Source/PDF/pdf.inc'
c      CHARACTER*7         PDLABEL,EPA_LABEL
c      INTEGER       LHAID
c      character*7 pdsublabel(2)
c      COMMON/TO_PDF/LHAID,PDLABEL,EPA_LABEL, pdsublabel

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

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'

      include 'vector.inc'
      include 'run.inc'


      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer tstrategy(lmaxconfigs)
      common/to_forest/ iforest, tstrategy

      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac

      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee
      
      save ndim,nfinal,nbranch,nparticles

      integer jfig,k

      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap      


      double precision x1_ee, x2_ee
      
c     External function
      double precision SumDot
      external SumDot
      logical dummy_boostframe
      external dummy_boostframe
c
c     data
c
      include 'configs.inc'
      data firsttime/.true./
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
         maxwgt=0d0
c         write(*,'(a,12i4)') 'Summing configs',(isym(i),i=1,isym(0))
         nparticles   = nexternal
         nfinal       = nparticles-nincoming
         nbranch      = nparticles-2
         ndim         = 3*nfinal-4
         if (ndim .lt. 0) ndim = 0   !For 2->1 processes  tjs 5/24/2010
         if (abs(lpp(1)) .ge. 1) ndim=ndim+1
         if (abs(lpp(2)) .ge. 1) ndim=ndim+1
c         if (pdlabel.eq.'dressed') ndim = ndim+1
         do i=1,nexternal
            m(i)=pmass(i)
         enddo
         write(*,'(a,12e10.3)') ' Masses:',(m(i),i=1,nparticles)
         call configure_integral(iconfig,mincfig,maxcfig,invar,maxwgt)
      endif                          !First_time

      this_config = iconfig             !Pass iconfig to amplitude routine
C
C     Get fraction of beam energy if pdf's are used
c
      xbk(1)   = 1d0
      xbk(2)   = 1d0
      sjac = 1d0
      if (abs(lpp(1)) .ge. 1 .and. abs(lpp(2)) .ge. 1) then
         if (abs(lpp(1)).eq.9.or.abs(lpp(2)).eq.9)then
            call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
            call sample_get_x(sjac,x(ndim-1),ndim-1,mincfig,0d0,1d0)
            call get_dummy_x1_x2(sjac, Xbk(1), x(ndim-1),pi1, pi2, stot, s(-nbranch))
            if (.not.set_cm_rap)then
               cm_rap=.5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
               set_cm_rap=.true.
            endif
c        lpp1=+-3 or +-4 and lpp2 as well
c        For ISR/beamstrhalung case         
         else if (pdlabel.eq.'dressed') then
            if (spole(ndim-1).gt.0d0.and.swidth(ndim-1).gt.0d0) then
               call ntuple(R,0.0d0,1.0d0,0,iconfig)
               ee_jacobian = 1d0
               call DS_get_point('ee_mc', R , ee_picked, ee_jacobian)
               sjac = sjac *  ee_jacobian
            else
               ee_picked = 1
            endif
            
            if(ee_picked.eq.2) then
               call sample_get_x(sjac,x(ndim-1),ndim-1,mincfig,smin/stot,1d0)               
               xtau = x(ndim-1)
               if(nexternal .eq. 3) then
                  x(ndim-1) = pmass(3)*pmass(3)/stot
                  sjac=1 / stot !for delta function in d_tau
               endif

               call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
               CALL GENCMS_EE(STOT,Xbk(1),Xbk(2),X(ndim-1), SMIN,SJAC)
               x(ndim-1) = xtau !Fix for 2->1 process
c     Set CM rapidity for use in the rap() function
               cm_rap=.5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
               set_cm_rap=.true.
c           Set shat
               s(-nbranch) = xbk(1)*xbk(2)*stot
               x1_ee = Xbk(1)
               x2_ee =  Xbk(2)
            else   
c        ! for dressed ee collisions the generation is different
c        ! wrt the pp case. In the pp case, tau and y_cm are generated, 
c        ! while in the ee case x1 and x2 are generated first.
               call sample_get_x(sjac,x(ndim-1),ndim-1,mincfig,0d0,1d0)
               call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
               call generate_x_ee(x(ndim-1), smin/stot,x1_ee, omx_ee(1), sjac)
               call generate_x_ee(x(ndim), smin/stot/x1_ee,x2_ee, omx_ee(2), sjac)

               s(-nbranch) =  x1_ee * x2_ee * stot
               xbk(1)   = x1_ee
               xbk(2)   = x2_ee
c        ! now we are done. We must call the following function 
c        ! in order to (re-)generate tau and ycm
c        ! from x1 and x2. It also (re-)checks that tau_born 
c        ! is pysical, and otherwise sets xjac0=-1000
               call get_y_from_x12(x1_ee, x2_ee, omx_ee(1), cm_rap) 
               set_cm_rap=.true.
            endif
        ! multiply the jacobian by a multichannel factor if the 
        ! generation with resonances is also possible
           if (spole(ndim-1).gt.0d0.and.swidth(ndim-1).gt.0d0) then
               tau_m = spole(ndim-1)
               tau_w = swidth(ndim-1)
               t1 =  (Max(1e-12,1- x1_ee * x2_ee))**(1d0-2*get_ee_expo())
c               if (dabs(x1_ee * x2_ee-tau_m).lt.5*tau_w)then
                  t2 =  (1d0/(( x1_ee * x2_ee-tau_m)**2 + tau_m*tau_w))
c               else
c                  t2=0d0
c               endif
               
               if (ee_picked.eq.1) then
                  sjac = sjac * t1 / (t1+t2)
               else
                  sjac = sjac * t2 / (t1+t2)
               endif
            endif


         else
c-----
c tjs 5/24/2010 for 2->1 process
c-------
           call sample_get_x(sjac,x(ndim-1),ndim-1,mincfig,0d0,1d0)               
           xtau = x(ndim-1)
            if(nexternal .eq. 3) then
               x(ndim-1) = pmass(3)*pmass(3)/stot
               sjac=1 / stot    !for delta function in d_tau
            endif

            call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
            CALL GENCMS(STOT,Xbk(1),Xbk(2),X(ndim-1), SMIN,SJAC)
            x(ndim-1) = xtau    !Fix for 2->1 process
c           Set CM rapidity for use in the rap() function
            cm_rap=.5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
            set_cm_rap=.true.
c           Set shat
            s(-nbranch) = xbk(1)*xbk(2)*stot
         endif
      elseif (lpp(1).eq.9.or.lpp(2).eq.9) then
         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
        if (lpp(1).eq.9)then
            call get_dummy_x1(sjac, xbk(1), x(ndim), pi1, pi2, stot, s(-nbranch))
            xbk(2) = 1d0
         else
            call get_dummy_x1(sjac, xbk(2), x(ndim), pi1, pi2, stot, s(-nbranch))
            xbk(1) = 1d0
         endif
         if (.not.set_cm_rap)then
            cm_rap=.5d0*dlog(xbk(1)*ebeam(1)/(xbk(2)*ebeam(2)))
            set_cm_rap=.true.
         endif
      elseif (abs(lpp(1)) .ge. 1) then
         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
         xbk(1) = x(ndim)
c        Set CM rapidity for use in the rap() function
         p0=xbk(1)*ebeam(1)+ebeam(2)
         p3=xbk(1)*ebeam(1)-sqrt(ebeam(2)**2-m2**2)
         cm_rap=.5d0*dlog((p0+p3)/(p0-p3))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) = x(ndim)*stot         
      elseif (abs(lpp(2)) .ge. 1) then
         call sample_get_x(sjac,x(ndim),ndim,mincfig,0d0,1d0)
         xbk(2) = x(ndim)

c        Set CM rapidity for use in the rap() function
         p0=ebeam(1)+xbk(2)*ebeam(2)
         p3=sqrt(ebeam(1)**2-m1**2)-xbk(2)*ebeam(2)
         cm_rap=.5d0*dlog((p0+p3)/(p0-p3))
         set_cm_rap=.true.
c        Set shat
         s(-nbranch) =  x(ndim)*stot         
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
      if (lpp(1).eq.9.or.lpp(2).eq.9)then
         if (dummy_boostframe())then
            call mom2cx(m(-nbranch),m(1),m(2),1d0,0d0,p(0,1),p(0,2))
         else
            p(:,1) = pi1(:)
            p(:,2) = pi2(:)
         endif
      else if(nincoming.eq.2) then
        call mom2cx(m(-nbranch),m(1),m(2),1d0,0d0,p(0,1),p(0,2))
      else
        do i=0,3
          p(i,1)=p(i,-nbranch)
        enddo
        p(3,1)=1e-14 ! For HELAS routine ixxxxx for neg. mass
      endif
      pswgt = 1d0
      jac   = 1d0
      call one_tree(iforest(1,-max_branch,iconfig), tstrategy(iconfig),mincfig,
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


      subroutine configure_integral(iconfig,mincfig,maxcfig,invar,maxwgt)
c**************************************************************************
c     inputs  iconfig   == Current configuration working on
c     output  m
c**************************************************************************

      implicit none

      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      include 'vector.inc'
      include 'run.inc'

c     local
      double precision pi1(0:3),pi2(0:3),p0,p3
      double precision dum
      integer i,j,ipole,n
      integer nbranch,ndim,nconfigs
      integer ninvar
      integer nparticles,nfinal
      integer nb_tchannel

c
c     Arguments
c
      integer iconfig,mincfig,maxcfig,invar
      double precision maxwgt
c
c     External
c
      double precision lambda,dot,dsig
      logical passcuts


c
c     global
c
      double precision M(-max_branch:max_particles)

      double precision pmass(nexternal)
      common/to_mass/  pmass

      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar



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
            if (abs(lpp(1)) .eq. 4) m1 = 0.105658d0
            if (abs(lpp(2)) .eq. 4) m2 = 0.105658d0
            if (mass_ion(1).ge.0d0) m1 = mass_ion(1)
            if (mass_ion(2).ge.0d0) m2 = mass_ion(2)
            if(ebeam(1).lt.m1.and.lpp(1).ne.9) ebeam(1)=m1
            if(ebeam(2).lt.m2.and.lpp(2).ne.9) ebeam(2)=m2
            pi1(0)=ebeam(1)
            pi1(3)=sqrt(max(ebeam(1)**2-m1**2, 0d0))
            pi2(0)=ebeam(2)
            pi2(3)=-sqrt(max(ebeam(2)**2-m2**2, 0d0))
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
         if (mincfig.eq.0) then
            iconfig = 1
            nconfigs = mapconfig(mapconfig(0))
            mincfig=1
            maxcfig=mapconfig(0)
         endif
         call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,nexternal,nincoming,nb_tchannel)
c         maxwgt=0d0
c         nparticles   = nexternal
c         nfinal       = nparticles-nincoming
c         nbranch      = nparticles-2
c         ndim         = 3*nfinal-4
c         if (ndim .lt. 0) ndim = 0   !For 2->1 processes  tjs 5/24/2010
c         if (abs(lpp(1)) .ge. 1) ndim=ndim+1
c         if (abs(lpp(2)) .ge. 1) ndim=ndim+1
         call set_peaks
c     Initialize dsig (needed for subprocess group running mode)
         dum=dsig(0,0,1)

      return
      end

      subroutine one_tree(itree,tstrategy,iconfig,nbranch,P,M,S,X,jac,pswgt)
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
      integer tstrategy ! current strategy for t-channel
      integer iconfig                 !Which configuration working on
      double precision P(0:3,-max_branch:max_particles)
      double precision pother(0:3), ptemp(0:3), pboost(0:3), ptemp2(0:3)
      double precision M(-max_branch:max_particles)
      double precision S(-max_branch:0)
c      double precision spole(-max_branch:0),swidth(-max_branch:0)
      double precision jac,pswgt
      integer nbranch
      double precision x(*) ! ja 3/2/11 21->40 after strange segfault
c
c     Local
c
      logical pass
      double precision tmass(-max_branch:-1)
      integer ibranch,i,ns_channel,nt_channel,ix  !,nerr
      integer iopposite ! index for t-channel mapping for the part not handle by itree
c     data nerr/0/
      double precision smin,smax,totmass,totmassin,xa2,xb2,wgt
      double precision costh,phi,tmin,tmax,t
      double precision ma2,mb2,m12,mn2,s1, mi2
c
c     External
c
      double precision lambda,dot
c
c     Global
c
      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      include 'vector.inc'
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
      iopposite = 1
      if (abs(tstrategy).eq.1) then
         iopposite = 2
      endif
      
      do while(itree(1,-ns_channel) .ne. iopposite .and.ns_channel.lt.nbranch)
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

      if (tstrategy.eq.-2.or.tstrategy.eq.-1) then
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
cc       T-channel ping-pong strategy starting with 2
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         
c     No -flipping case:      
c      p(0,itree(ibranch,1)) + p(0,2) -> p(0,ibranch)+ p(0,itree(ibranch,2))
c       -  M(ibranch) is the total mass available (Pa+Pb)^2
c       - M(ibranch-1) is the mass of P2  (all the remaining particles)
c
c     With flipping case
c      p(0,itree(ibranch,1)) + pother -> p(0,ibranch)+ p(0,itree(ibranch,2))
c       - pother = p(0,itree(ibranch,1)) -p(0,itree(ibranch,2))         
c       - M(ibranch) is the total mass available (Pa+Pb)^2
c       - M(ibranch-1) is the mass of P2  (all the remaining particles)      
c
c     This assumes that P(0, ibranch) is set to the T-channel propa (likely)
c      do ibranch = -ns_channel-1,-nbranch,-1
c         totmass=totmass+m(itree(2,ibranch))
c      enddo
      do ibranch=-ns_channel-1,-nbranch+1,-1
c         totmass=totmass-m(itree(2,ibranch))      !for remaining particles
c         smin = totmass**2                        !This affects t_min/max
c         smax = (m(ibranch) - m(itree(2,ibranch)))**2

         if (ibranch.ne.-ns_channel-1)then
            pother(:) = P(:,ibranch+1)
            iopposite = ibranch +1
         else
            pother(:) = p(:,2)
            iopposite = abs(tstrategy)
         endif
         s1  = m(ibranch)**2                        !Total mass available
         ma2 = dot(pother, pother)
         mb2 = dot(P(0,itree(1,ibranch)),P(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mn2 = m(ibranch-1)**2
c
c$$$         write(*,*) itree(1, ibranch), '-----------T----------', itree(2, ibranch), 'm=', m(itree(2, ibranch))
c$$$         write(*,*)        '                       |           '
c$$$         write(*,*)   	   '                       |           '
c$$$         write(*,*)   	   '                       | ',ibranch
c$$$         write(*,*)   	   '                       |           '
c$$$         write(*,*)        '                       |           '
c$$$         if (ibranch.ne.-ns_channel-1)then
c$$$            write(*,*) iopposite, '-----------T---------- m=', m(ibranch-1)
c$$$         else
c$$$            write(*,*) 2, '-----------T---------- m=', m(ibranch-1)
c$$$            write(*,*) m(1), m(2), m(3), m(4), m(5)
c$$$         endif
c$$$         write(*,*) 'Pa', P(0,itree(1, ibranch)),P(1,itree(1, ibranch)),P(2,itree(1, ibranch)),P(3,itree(1, ibranch))
c$$$         if (ibranch.ne.-ns_channel-1) then
c$$$            write(*,*) 'Pb', P(0,iopposite),P(1,iopposite),P(2,iopposite), P(3,iopposite)
c$$$            do i=0,3
c$$$               pother(i) = P(i,itree(1, ibranch)) + P(i,ibranch+1) 
c$$$            enddo
c$$$         else
c$$$            write(*,*) 'Pb', P(0,2),P(1,2),P(2,2),P(3,2)
c$$$            do i=0,3
c$$$               pother(i) = P(i,1) + P(i,2)
c$$$            enddo
c$$$         endif
c$$$         do i=0,3
c$$$            pother(i) = P(i,itree(1, ibranch)) + P(i,iopposite)
c$$$         enddo
c$$$         write(*,*) 'DSQRT(s1) = ', m(ibranch), DSQRT(dot(pother, pother))
c$$$c         if (m(ibranch)**2.ne.dot(pother, pother)) stop 1
c$$$         write(*,*) 'm12= Pd**2 = ', m12 ,DSQRT(m12)
c$$$         write(*,*) 'mn2 = Pc**2 =', mn2, DSQRT(mn2)
         
C     WRITE(*,*) 'Enertering yminmax',sqrt(s1),sqrt(m12),sqrt(mn2)
         
         call yminmax(s1,0d0,m12,ma2,mb2,mn2,tmin,tmax)
c         call yminmax(s1,0d0,m12,ma2,mb2,smax,tmin_temp,tmax_temp)
c         if (tmin_temp.lt.tmin) tmin = tmin_temp
c         if (tmax_temp.gt.tmax) tmax = tmax_temp
         
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

c         write(*,*) 'tmin, tmax/ temp',tmin,tmax, tmin_temp, tmax_temp

c         if (nt_channel.ge.2)then
c            tmin = max(tmin, -stot)
c         endif
c      if ((tmax-tmin)/stot.gt.0.1)then
c            call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $           0d0, 1d0)
c         t = stot*(-x(-ibranch))
      
c      else if (tmax/stot.gt.-0.01.and.tmin/stot.lt.-0.02)then
c         set tmax to 0. The idea is to be sure to be able to hit zero
c         and not to be block by numerical inacuracy
c         tmax = max(tmax,0d0) !This line if want really t freedom
c         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $        0d0, -tmin/stot)
c         t = stot*(-x(-ibranch))

c      else
         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
     $        -tmax/stot, -tmin/stot)
         t = stot*(-x(-ibranch))
c      endif

c      call yminmax(s1,0d0,m12,ma2,mb2,mn2,tmin_temp,tmax_temp)
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
         jac = jac*(tmax-tmin)*2d0*pi /2d0 ! I need /2d0 if -1<x<1

         
c
c     Finally generate the momentum. The call is of the form
c     pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c     gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
c
         if (itree(1,ibranch).gt.-ns_channel-1)then
            mi2 = m(itree(1,ibranch))**2
         else
            mi2 = tmass(itree(1,ibranch))
         endif
         tmass(ibranch) = t
         call gentcms(p(0,itree(1,ibranch)),p(0,iopposite),t,phi,mi2,
     &        m(itree(2,ibranch)),m(ibranch-1),p(0,itree(2,ibranch)),
     &        p(0,ibranch),jac)
c$$$         write(*,*) 'RESULT'
c$$$         write(*,*) 'pa', p(0,itree(1,ibranch)),p(1,itree(1,ibranch)),p(2,itree(1,ibranch)),p(3,itree(1,ibranch))
c$$$         write(*,*) 'pb', p(0,iopposite),p(1,iopposite),p(2,iopposite),p(3,iopposite)
c$$$         write(*,*) '->'
c$$$         write(*,*) 'pc', p(0,itree(2,ibranch)),p(1,itree(2,ibranch)),p(2,itree(2,ibranch)),p(3,itree(2,ibranch))
c$$$         do i =0,3
c$$$            pother(i) = p(i,itree(1,ibranch)) + p(i,iopposite) - p(i,itree(2,ibranch))
c$$$         enddo
c$$$         write(*,*) 'pc', p(0,itree(2,ibranch)),p(1,itree(2,ibranch)),p(2,itree(2,ibranch)),p(3,itree(2,ibranch))
c$$$         write(*,*) 'pd', pother(0), pother(1), pother(2), pother(3) , DSQRT(dot(pother,pother))
c$$$         write(*,*) 'T channel'
c$$$         write(*,*) 'pr', p(0,ibranch),p(1,ibranch),p(2,ibranch),p(3,ibranch)
c$$$         write(*,*) 'pa-pc', p(0,itree(1,ibranch))-p(0,itree(2,ibranch)),p(1,itree(1,ibranch))-p(1,itree(2,ibranch))


         
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
      if (nt_channel.eq.1) then
c$$$         write(*,*) 'need to assign last', itree(2,-nbranch)
c$$$         write(*,*) 'nbranch is at', nbranch
c$$$         do i=-nbranch,nexternal
c$$$            write(*,*) 'p',i, p(0,i),p(1,i),p(2,i),p(3,i)
c$$$         enddo
         do i=0,3
            p(i,itree(2,-nbranch)) = p(i,-nbranch+1)+p(i,2)
         enddo
      else
c$$$                  write(*,*) 'need to assign last', itree(2,-nbranch)
c$$$         write(*,*) 'nbranch is at', nbranch
c$$$         do i=-nbranch,nexternal
c$$$            write(*,*) 'p',i, p(0,i),p(1,i),p(2,i),p(3,i)
c$$$         enddo
         do i=0,3
            p(i,itree(2,-nbranch)) = p(i,-nbranch+1)+p(i,-nbranch+2)
         enddo
      endif


      else if (tstrategy.eq.2.or.tstrategy.eq.1) then
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
cc       T-channel One side eat all strategy ending with 2
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
         ma2 = m(tstrategy)**2
         mb2 = dot(P(0,itree(1,ibranch)),P(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mn2 = m(ibranch-1)**2
c         write(*,*) 'Enertering yminmax',sqrt(s1),sqrt(m12),sqrt(mn2)
         call yminmax(s1,0d0,m12,ma2,mb2,mn2,tmin,tmax)

         if(.false.) then
             write(*,*) itree(1, ibranch), 'a----------T----------', itree(2, ibranch), 'm=', m(itree(2, ibranch))
             write(*,*)        '                       |           '
             write(*,*)   	   '                       |           '
             write(*,*)   	   '                       | ',ibranch
             write(*,*)   	   '                       |           '
             write(*,*)        '                       |           '
             write(*,*) tstrategy, '-----------T---------- m=', m(ibranch-1), ibranch-1
             write(*,*) 'S', dsqrt(s1), 'm_top=', dsqrt(mb2), 'M_bottom', dsqrt(ma2)
             
c     write(*,*) m(1), m(2), m(3), m(4), m(5)
            write(*,*) 'Pa', P(0,itree(1, ibranch)),P(1,itree(1, ibranch)),P(2,itree(1, ibranch)),P(3,itree(1, ibranch))
         endif
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
         if(.false.) then
            ! NOT VALIDATED METHOD, momenta are ok but not jacobian
            
            call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
     $        0d0, 1d0)
            costh= 2d0*x(-ibranch)-1d0
            call sample_get_x(wgt,x(nbranch+(-ibranch-1)*2),
     &        nbranch+(-ibranch-1)*2,iconfig,0d0,1d0)
            phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
            jac = jac * 4d0*pi
                     m12 = m(itree(2,ibranch))**2
         mn2 = m(ibranch-1)**2
         call mom2cx(dsqrt(s1),m(itree(2,ibranch)),m(ibranch-1),costh,phi,
     &        p(0,itree(2,ibranch)),pother)

         I= itree(2,ibranch)
         DO I=0,3
            pboost(I) = P(I,tstrategy) + P(I,itree(1, ibranch))
         ENDDO
         
         call boostm(p(0,itree(2,ibranch)),pboost,m(itree(2,ibranch)),p(0,itree(2,ibranch)))
         call boostm(pother,pboost,m(ibranch),pother)

         do I=0,3
            p(I,ibranch) = pother(i) - p(i, tstrategy)
         enddo
         if(.false.)then
         write(*,*) 'input'
         write(*,*) 'p(tstrategy=',tstrategy,')', p(0,tstrategy), p(1,tstrategy), p(2,tstrategy), p(3,tstrategy)
         I=itree(1, ibranch)
         write(*,*) 'p(',I,',)', p(0,i), p(1,i), p(2,i), p(3,i)
         write(*,*) 'output'
         I= itree(2,ibranch)
         write(*,*) 'p(',I,',)', p(0,i), p(1,i), p(2,i), p(3,i)
         write(*,*) 'pother', pother(0),pother(1),pother(2),pother(3)
         write(*,*) 'check'
         do i=0,3
            ptemp(i) = p(i,tstrategy) + p(i, itree(1, ibranch))
         enddo
         write(*,*) 'pa+pb', ptemp(0), ptemp(1), ptemp(2), ptemp(3), dsqrt(dot(ptemp, ptemp))
         do i=0,3
            ptemp(i) = pother(i) + p(i,itree(2,ibranch))
         enddo
         write(*,*) 'p1+p2', ptemp(0), ptemp(1), ptemp(2), ptemp(3), dsqrt(dot(ptemp, ptemp))
         
         write(*,*) 'Tchannel'
         I = ibranch
         write(*,*) 'p(',I,',)', p(0,i), p(1,i), p(2,i), p(3,i), dot(p(0,I), p(0,I))
         endif
         
         pswgt = pswgt/(4d0*dsqrt(lambda(s1,ma2,mb2)))

      else

c     test of impact of low t part
c         if (nt_channel.ge.2)then
c            tmin = max(tmin,  -stot)
c         endif


         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
     $        -tmax/stot, -tmin/stot)
         t = stot*(-x(-ibranch))
         

c         if ((tmax-tmin)/stot.gt.0.1)then
c            call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $           0d0, 1d0)
c         t = stot*(-x(-ibranch))
c     if (dabs(tmax - tmin)/stot.gt.0.05d0) then
c         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $        0d0,  1d0)
c     set tmax to 0 and tmin to -1 The idea is to avoid dimension correlation
c     the condition ensure a minimum efficiency in the generation of events
c         t = stot*(-x(-ibranch))
c      else if (tmax.gt.-0.01.and.tmin.lt.-0.02)then
c         set tmax to 0. The idea is to be sure to be able to hit zero
c         and not to be block by numerical inacuracy
c         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $        0d0, -tmin/stot)
c         t = stot*(-x(-ibranch))
c      else
c         call sample_get_x(wgt,x(-ibranch),-ibranch,iconfig,
c     $        -tmax/stot, -tmin/stot)
c         t = stot*(-x(-ibranch))
c      endif

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

         if (itree(1,ibranch).gt.-ns_channel-1)then
            mi2 = m(itree(1,ibranch))**2
         else
            mi2 = tmass(itree(1,ibranch))
         endif
         tmass(ibranch) = t
         call gentcms(p(0,itree(1,ibranch)),p(0,tstrategy),t,phi, mi2,
     &        m(itree(2,ibranch)),m(ibranch-1),p(0,itree(2,ibranch)),
     &        p(0,ibranch),jac)

         if (jac .lt. 0d0) then
c            nerr=nerr+1
c            if(nerr.le.5)
c     $           write(*,*) 'Failed gentcms',iconfig,ibranch
            return              !Failed, probably due to negative x
         endif

         pswgt = pswgt/(4d0*dsqrt(lambda(s1,ma2,mb2)))
      endif
      enddo
c
c     We need to get the momentum of the last external particle.
c     This should just be the sum of p(0,2) and the remaining
c     momentum from our last t channel 2->2
c
      do i=0,3
         p(i,itree(2,-nbranch)) = p(i,-nbranch+1)+p(i,tstrategy)
      enddo


      
      else
         write(*,*) 'not supported tstrategy'
         stop 2
      endif

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
c$$$      write(*,*) '****'
c$$$      do i=-nbranch,nexternal
c$$$         write(*,*) 'mass', i, m(i)
c$$$      enddo
c$$$      do i=-nbranch,nexternal
c$$$         write(*,*) 'p',i, p(0,i),p(1,i),p(2,i),p(3,i)
c$$$      enddo
c$$$      do i =0,3
c$$$         pother(i) = p(i,1) + p(i,2) - p(i,3) -p(i,4)
c$$$      enddo
c$$$      write(*,*) 'p5 expected', pother(0), pother(1),pother(2),pother(3)

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

      subroutine gentcms(pa,pb,t,phi,ma2,m1,m2,p1,pr,jac)
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
      double precision t,phi,m1,m2,ma               !inputs
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
C     Y is not used in this formula (called with dummy value)
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

      TAUMIN = 0d0 !SMIN/S !keep scale fix
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

      SUBROUTINE GENCMS_EE(S,X1,X2,X,SMIN,SJACOBI)
C***********************************************************************
C     PICKS PARTON MOMENTUM FRACTIONS X1 AND X2 BY CHOOSING ETA AND TAU
C     X(1) --> TAU = X1*X2
C     X(2) --> one of the bjorken x
C***********************************************************************
      IMPLICIT NONE

C     ARGUMENTS

      DOUBLE PRECISION X1,X2,S,SMIN,SJACOBI
      DOUBLE PRECISION X(2)
C     global

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee
C     FUNCTION
      double precision get_ee_expo
      external get_ee_expo
C     LOCAL

      DOUBLE PRECISION TAU,TAUMIN,TAUMAX

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

      TAUMIN = 0d0 !SMIN/S !keep scale fix
      TAUMAX = 1D0
      TAU    = (TAUMAX-TAUMIN)*X(1)+TAUMIN
      SJACOBI=  sjacobi*(TAUMAX-TAUMIN)

        ! then pick either x1 or x2 and generate it the usual way;
        ! Note that:
        ! - setting xmin=sqrt(tau_born) ensures that the largest 
        !    bjorken x is being generated.
        ! -  there is a jacobian for x1 x2 -> tau x1(2)
        ! -  we must include the factor 1/(1-x)^get_ee_expo,
        !    (x is the bjorken x which is not generated)
        !    because the compute_eepdf function assumes that
        !    this is the case in general
        if (x(2).lt.0.5d0) then
          call generate_x_ee(x(2)*2d0, dsqrt(tau), x1, omx_ee(1), sjacobi)
          x2 = tau / x1
          omx_ee(2) = 1d0 - x2
          sjacobi = sjacobi / x1 * 2d0  / (1d0-x2)**get_ee_expo()
        else
          call generate_x_ee(1d0-2d0*(x(2)-0.5d0), dsqrt(tau), x2, omx_ee(2), sjacobi)
          x1 = tau / x2
          omx_ee(1) = 1d0 - x1
          sjacobi = sjacobi / x2 * 2d0  / (1d0-x1)**get_ee_expo()
       endif
      

      END


C     -----------------------------------------
C     Subroutine to return momenta in a dedicated frame
C     frame_id is the tag of the particle to put at rest
C     frame_id follow the convention of cluster.f (sum 2**(N-1))
C     -----------------------------------------

      subroutine boost_to_frame(P1, frame_id, P2)

      implicit none

      include 'nexternal.inc'

      DOUBLE PRECISION P1(0:3,NEXTERNAL)
      DOUBLE PRECISION P2(0:3,NEXTERNAL)
      DOUBLE PRECISION PBOOST(0:3)
      integer frame_id

      integer ids(nexternal)
      integer i,j
      logical trivial_boost 

c     uncompress
      call mapid(frame_id, ids)
      pboost(:) = 0d0
      p2(:,:) = 0d0
c     avoid trivial boost (check only for  0 0 1 1 1... so sum of the
c     final state. 1 1 0 0 0 .... should not go within this function   
      trivial_boost=.true.
      do i=1,nexternal
        if (i.le.nincoming)then
            if (ids(i).ne.0)then
                trivial_boost=.false.
                EXIT ! stop do loop
            endif
        else
            if (ids(i).ne.1)then
                  trivial_boost=.false.
                  EXIT ! stop do loop
              endif
        endif
        enddo

        if (trivial_boost)then
            p2(:,:) = p1(:,:)
            return
        endif

c     find the boost momenta --sum of particles--
      do i=1,nexternal
       if (ids(i).eq.1)then
            do j=0,3
	           Pboost(j) = Pboost(j) + P1(j,i)
            enddo
         endif
      enddo
      do j=1,3	
          Pboost(j) = -1 * Pboost(j)
      enddo	    
      do i=1, nexternal
         call boostx(p1(0,i), pboost, p2(0,i))
      enddo   
      return
      end

      double precision function get_channel_cut(p, config)
      implicit none

      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'genps.inc'
      include 'maxamps.inc'
      include 'vector.inc' ! defines VECSIZE_MEMMAX
      include 'coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)
c     include 'run.inc'

      double precision p(0:3, nexternal)
      integer config
      

      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer tstrategy(lmaxconfigs)
      common/to_forest/ iforest, tstrategy

      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      common/to_sprop/sprop,tprid

      double precision stot,m1,m2
      common/to_stot/stot,m1,m2

      double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat
      
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config

      double precision      spole(maxinvar),swidth(maxinvar),bwjac
      common/to_brietwigner/spole          ,swidth          ,bwjac

      double precision ptemp(0:3, -nexternal:nexternal)
      integer i,j
      integer d1, d2
      double precision t
      double precision dot
      external dot
      integer ns_channel
      integer nb_tchannel
      integer nbranch
      double precision tmp, tmp2
      
      double precision ZERO
      parameter (ZERO=0d0)
      double precision prmass(-nexternal:0,lmaxconfigs)
      double precision prwidth(-nexternal:0,lmaxconfigs)
      integer pow(-nexternal:0,lmaxconfigs)
      logical first_time
      save prmass,prwidth,pow
      data first_time /.true./

      double precision Mass, Width
      
      include 'configs.inc'

      if(sde_strat.eq.1.and.tmin_for_channel.eq.-1)then
         get_channel_cut = 1d0
         return
      endif
      
      if (first_time) then
         include 'props.inc'
         first_time=.false.
      endif
      
      do i = 1, nexternal
         do j =0,3
            ptemp(j,i) = p(j,i)
            ptemp(j,-i) = 0d0
         enddo
      enddo

      nbranch = nexternal -2
      ns_channel=1
      do while((iforest(1,-ns_channel,config) .ne. 1.and.iforest(1,-ns_channel,config) .ne. 2).and.ns_channel.lt.nbranch)
         ns_channel=ns_channel+1
      enddo
      ns_channel=ns_channel - 1
      nb_tchannel=nbranch-ns_channel-1
c      write(*,*) 'T-channel found: ',nb_tchannel





      
      get_channel_cut = 1.
      if (nb_tchannel.lt.2.and.sde_strat.eq.1)then
         get_channel_cut = 1.
         return
      endif
      
      do i = 1, nexternal-3
         d1 = iforest(1, -i, config)
         d2 = iforest(2, -i, config)
         do j=0,3
            if (d1.gt.0.and.d1.le.nincoming) then
               ptemp(j,-i) = ptemp(j,-i) - ptemp(j, d1)
            else
               ptemp(j,-i) = ptemp(j,-i)+ptemp(j, d1)
            endif
            if (d2.gt.0.and.d2.le.nincoming) then
               ptemp(j,-i) = ptemp(j,-i) - ptemp(j, d2)
            else
               ptemp(j,-i) = ptemp(j,-i)+ptemp(j, d2)
            endif
         enddo
         if (tprid(-i,config).ne.0)then
            if(sde_strat.eq.2)then
               t = dot(ptemp(0,-i), ptemp(0,-i))
               Mass  = prmass(-i, config)
               get_channel_cut = get_channel_cut / (t-Mass**2+stot*1d-10)**2
            endif
c            write(*,*) i, "t, Mass, fact", t, Mass, ((t-Mass)*(t+Mass))**2,get_channel_cut
            t = t/stot 
            if (t.lt.tmin_for_channel)then
                get_channel_cut = get_channel_cut * exp((t-tmin_for_channel)/(t+1))
c               get_channel_cut = get_channel_cut * (t+1)/(1+tmin_for_channel)
c            else if(t.gt.2*tmin_for_channel)then
c               get_channel_cut = get_channel_cut * (2*tmin_for_channel-t)/tmin_for_channel
            endif
         else
            if(sde_strat.eq.2)then
               t = dot(ptemp(0,-i), ptemp(0,-i))
               Mass  = prmass(-i, config)
               Width = prwidth(-i, config)
               tmp = (t-Mass**2)
               tmp2 = Mass*Width
               get_channel_cut = get_channel_cut/(tmp**2 + tmp2**2) 
            endif
c            write(*,*) i, "s, Mass, Width, fact", t, Mass, Width, (((t-Mass)*(t+Mass) )**2 + Width**2*Mass**2), get_channel_cut
         endif
      enddo
c      write(*,*) 'final for config', config, get_channel_cut
      return
      end


      subroutine get_y_from_x12(x1, x2, omx, ycm) 
      implicit none
      double precision x1, x2, omx(2), tau, ycm
      double precision ylim
      double precision tolerance
      parameter (tolerance=1e-3)
      double precision y_settozero
      parameter (y_settozero=1e-12)
      double precision lx1, lx2
      double precision ylim0, ycm0


      ! ycm=-log(tau)/2 ;  ylim = log(x1/x2)/2
      if (1d0-x1.gt.tolerance) then
         lx1 = dlog(x1)
      else if (omx(1).lt.tolerance**5)then
         lx1=0d0
      else
        lx1 = -omx(1)-omx(1)**2/2d0-omx(1)**3/3d0-omx(1)**4/4d0-omx(1)**5/5d0
      endif
      ylim = -0.5d0*lx1
      ycm = 0.5d0*lx1

      if (1d0-x2.gt.tolerance) then
         lx2 = dlog(x2)
      else if (omx(2).lt.tolerance**5)then
         lx2=0d0         
      else
        lx2 = -omx(2)-omx(2)**2/2d0-omx(2)**3/3d0-omx(2)**4/4d0-omx(2)**5/5d0
      endif
      ylim = ylim-0.5d0*lx2
      ycm = ycm-0.5d0*lx2

c      ycmhat = ycm / ylim

      ! this is to prevent numerical inaccuracies
      ! when botn x->1
      if (ylim.lt.y_settozero) then
        ylim = 0d0
        ycm = 0d0
      endif

      return 
      end

      subroutine generate_x_ee(rnd, xmin, x, omx, jac)
      implicit none
      ! generates the momentum fraction with importance
      !  sampling suitable for ee collisions
      ! rnd is generated uniformly in [0,1], 
      ! x is generated according to (1 -rnd)^-expo, starting
      ! from xmin
      ! jac is the corresponding jacobian
      ! omx is 1-x, stored to improve numerical accuracy
      double precision rnd, x, omx, jac, xmin
      double precision expo
      double precision get_ee_expo
      double precision tolerance
      parameter (tolerance=1.d-5)

      expo = get_ee_expo()

      x = 1d0 - rnd ** (1d0/(1d0-expo))
      omx = rnd ** (1d0/(1d0-expo))
      if (x.ge.1d0) then
        if (x.lt.1d0+tolerance) then
          x=1d0
        else
          write(*,*) 'ERROR in generate_x_ee', rnd, x
          stop 1
        endif
      endif
      jac = jac/(1d0-expo) 
      ! then rescale it between xmin and 1
      x = x * (1d0 - xmin) + xmin
      omx = omx * (1d0 - xmin)
      jac = jac * (1d0 - xmin)**(1d0-expo)

      return 
      end


