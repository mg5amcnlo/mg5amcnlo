      program symmetry
c*****************************************************************************
c     Given identical particles, and the configurations. This program identifies
c     identical configurations and specifies which ones can be skipped
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      include 'maxamps.inc'
      
      double precision ZERO
      parameter       (ZERO = 0d0)
      integer   maxswitch
      parameter(maxswitch=99)
      integer lun
      parameter (lun=28)
c
c     Local
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer itree(2,-max_branch:-1)
      integer imatch
      integer use_config(0:lmaxconfigs)
      integer i,j, k, n, nsym
      double precision diff
      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)
c
c     Local for generating amps
c
      double precision p(0:3,99), wgt, x(99), fx
      double precision p1(0:3,99)
      integer ninvar, ndim, iconfig, minconfig, maxconfig
      integer ncall,itmax,nconfigs,ntry, ngraphs
      integer ic(nexternal,maxswitch), jc(12),nswitch
      double precision saveamp(maxamps)
      integer nmatch, ibase
      logical mtc, even

c
c     Global
c
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      Double Precision amp2(maxamps), jamp2(0:maxflow)
      common/to_amps/  amp2,       jamp2
      include 'coupl.inc'
      
c
c     External
c
      logical pass_point
      logical check_swap
      double precision dsig
      external pass_point, dsig
      external check_swap

c      integer icomp
c
c     DATA
c
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'

      multi_channel=.true.

c-----
c  Begin Code
c-----
c      write(*,*) 'Enter compression (0=none, 1=sym, 2=BW, 3=full)'
c      read(*,*) icomp
c      if (icomp .gt. 3 .or. icomp .lt. 0) icomp=0
      if (icomp .eq. 0) then
         write(*,*) 'No compression, summing every diagram and ',
     $        'every B.W.'
      elseif (icomp .eq. 1) then
         write(*,*) 'Using symmetry but summing every B.W. '
      elseif (icomp .eq. 2) then
         write(*,*) 'Assuming B.W. but summing every diagram. '
      elseif (icomp .eq. 3) then
         write(*,*) 'Full compression. Using symmetry and assuming B.W.'
      else
         write(*,*) 'Unknown compression',icomp
         stop
      endif

      call setrun                !Sets up run parameters
      call setpara('param_card.dat',.true.)   !Sets up couplings and masses
      call setcuts               !Sets up cuts 
      call printout
      call run_printout
c
      ndim = 22
      ncall = 10000
      itmax = 10
      ninvar = 35
      nconfigs = 1
      iconfig=1
      include 'props.inc'
      call sample_init(ndim,ncall,itmax,ninvar,nconfigs)

      open(unit=lun,file='symswap.inc',status='unknown')

c     
c     Get momentum configuration
c
      do i = 1,23   !Not needed, but ok
      ntry = 1
      wgt=1d0
      call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
      do while ((.not. pass_point(p) .or. wgt .lt. 0 .or. p(0,1) .le. 0d0) .and. ntry .lt. 1000)
         call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
         ntry=ntry+1
      enddo
      call smatrix(p,fx) 
c      write(*,*) i,ntry,fx
c      do j=1,nexternal
c         write(*,'(2i3,4e15.5)') i,j,(p(k,j),k=0,3)
c      enddo
      
      enddo
      nconfigs = mapconfig(0)
c
c     Get and save base amplitudes
c
      do j = 1 , mapconfig(mapconfig(0))
         saveamp(j)=amp2(j)
      enddo
      do j=1,nexternal
         write(*,'(i4,4e15.5)') j,(p(i,j),i=0,3)
      enddo
c
c     Start using all configurations
c
      do j=1,mapconfig(0)
         use_config(j) = 1
      enddo
c
c     Swap amplitudes looking for matches
c
      nswitch = 1
      do k=1,nexternal
         ic(k,1)=k
      enddo
      nmatch = 0
      mtc=.false.
      nsym = 1
      write(lun,'(a,i2,a$)') '       data (isym(i,',nsym,'),i=1,nexternal) /1 '
      do i=2,nexternal
         write(lun,'(a,i2$)') ",",ic(i,1)
c         write(*,'(a,i2$)') ",",ic(i,1)
      enddo
c      write(*,*)
      write(lun,'(a)') "/"
      call nexper(nexternal-2,ic(3,1),mtc,even)
c      write(*,*) 'mtc',mtc, (ic(i,1),i=1,nexternal)
      do while(mtc)
         call nexper(nexternal-2,ic(3,1),mtc,even)
c         write(*,*) 'mtc',mtc, (ic(i,1),i=1,nexternal)
         do j=3,nexternal
            ic(j,1)=ic(j,1)+2
         enddo
         if (mtc) then
             CALL SWITCHMOM(P,P1,IC(1,1),JC,NEXTERNAL)
c
c     Now check if it is a valid swap to make
c

         if (check_swap(ic(1,1))) then
            write(*,*) 'Good swap', (ic(i,1),i=1,nexternal)
            nsym=nsym+1
            write(lun,'(a,i2,a$)') '       data (isym(i,',nsym,'),i=1,nexternal) /1 '
            do i=2,nexternal
               write(lun,'(a,i2$)') ",",ic(i,1)
            enddo
            write(lun,'(a)') "/"

            call smatrix(p1,fx) 
c            write(*,*) 'dsig=',fx
c         write(*,'(7i6)') i,(ic(k,i),k=1,nexternal)
c         do n = i+1,nswitch
c            CALL SWITCHMOM(P,P1,IC(1,n),JC,NEXTERNAL)
c            call smatrix(p1,fx)
c
c        Look for matches, but only for diagrams < current diagram
c     
         do j=2,mapconfig(0)
            do k=1,j-1
               diff = abs((amp2(mapconfig(j)) - saveamp(mapconfig(k)))/
     $              (amp2(mapconfig(j))+1d-99))
               if (diff .lt. 1d-8 ) then
c                  write(*,*) "Found match graph",mapconfig(j),mapconfig(k),diff
                  if (use_config(j) .gt. 0 ) then  !Not found yet
                     nmatch=nmatch+1
                     if (use_config(k) .gt. 0) then !Match is real config
                        use_config(k)=use_config(k)+use_config(j)
                        use_config(j)=-k
                     else
                        ibase = -use_config(k)
                        use_config(ibase) = use_config(ibase)+use_config(j)
                        use_config(j) = -ibase
                     endif
                  endif
               endif
            enddo
         enddo
         else
            write(*,*) 'Bad swap', (ic(i,1),i=1,nexternal)
         endif   !Good Swap
         endif   !Real Swap
         do j=3,nexternal
            ic(j,1)=ic(j,1)-2
         enddo
      enddo
      write(lun,*) '      data nsym /',nsym,'/'
      close(lun)
      write(*,*) 'Found ',nmatch, ' matches. ',mapconfig(0)-nmatch,
     $     ' channels remain for integration.'
      call write_input(j)
      call write_bash(mapconfig,use_config,pwidth,icomp,iforest,sprop)
      end

      logical function check_swap(ic)
c**************************************************************************
c     check that only identical particles were swapped
c**************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
c
c     Arguments
c
      integer ic(nexternal)
c
c     local
c
      integer i
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
c
c     Process info
c
      include 'leshouche.inc'
c------
c Begin Code
c-----
      check_swap=.true.
      do i=1,nexternal
         if (idup(i,1,1) .ne. idup(ic(i),1,1)) check_swap=.false.
      enddo
      end

      subroutine nexper(n,a,mtc,even)
c*******************************************************************************
c     Gives all permutations for a group
c http://www.cs.sunysb.edu/~algorith/implement/wilf/distrib/processed/nexper_2.f
c next permutation of {1,...,n}. Ref NW p 59.
c*******************************************************************************
      integer a(n),s,d
      logical mtc,even
      if(mtc)goto 10
      nm3=n-3
      do 1 i=1,n
 1        a(i)=i
      mtc=.true.
 5     even=.true.
      if(n.eq.1)goto 8
 6     if(a(n).ne.1.or.a(1).ne.2+mod(n,2))return
      if(n.le.3)goto 8
      do 7 i=1,nm3
      if(a(i+1).ne.a(i)+1)return
 7     continue
 8      mtc=.false.
      return
 10    if(n.eq.1)goto 27
      if(.not.even)goto 20
      ia=a(1)
      a(1)=a(2)
      a(2)=ia
      even=.false.
      goto 6
 20    s=0
      do 26 i1=2,n
 25       ia=a(i1)
      i=i1-1
      d=0
      do 30 j=1,i
 30       if(a(j).gt.ia) d=d+1
      s=d+s
      if(d.ne.i*mod(s,2)) goto 35
 26    continue
 27     a(1)=0
      goto 8
 35    m=mod(s+1,2)*(n+1)
      do 40 j=1,i
      if(isign(1,a(j)-ia).eq.isign(1,a(j)-m))goto 40
      m=a(j)
      l=j
 40    continue
      a(l)=ia
      a(i1)=m
      even=.true.
      return
      end


       subroutine write_input(nconfigs)
c***************************************************************************
c     Writes out input file for approximate calculation based on the
c     number of active configurations
c***************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      integer    maxpara
      parameter (maxpara=1000)
c      integer   npoint_tot,         npoint_min
c      parameter (npoint_tot=50000, npoint_min=1000)
c
c     Arguments
c
      integer nconfigs
c
c     local
c
      integer npoints
      character*20 param(maxpara),value(maxpara)
      integer npara, nreq
c
c     global
c
      logical gridpack
      common/to_gridpack/gridpack
c-----
c  Begin Code
c-----
      call load_para(npara,param,value)
      call get_logical(npara,param,value," gridpack ",gridpack,.false.)

      npoints = min_events_subprocess/nconfigs
      npoints = max(npoints,min_events_channel)
      open (unit=26, file = 'input_app.txt', status='unknown',
     $     err=99)
      if (gridpack) then
         write(26,*) npoints_wu,itmax_wu,
     &     '     !Number of events and iterations'      
         write(26,'(f8.4,a)') acc_wu, '    !Accuracy'
         write(26,*) ' 2       !Grid Adjustment 0=none, 2=adjust'
      else
         write(26,*) npoints,iter_survey,
     &     '     !Number of events and iterations'      
         write(26,*) ' 0.0    !Accuracy'
         write(26,*) ' 2       !Grid Adjustment 0=none, 2=adjust'
      endif
      write(26,*) ' 1       !Suppress Amplitude 1=yes'
      write(26,*) nhel_survey,'       !Helicity Sum/event 0=exact'
      close(26)
      return
 99   close(26)
      write(*,*) 'Error opening input_app.txt'
      end

      subroutine open_bash_file(lun)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
c
c     Constants
c
      include '../../Source/run_config.inc'
c
c     Arguments
c
      integer lun
c
c     Local
c
      character*30 fname
      integer ic

      data ic/0/
c-----
c  Begin Code
c-----
      ic=ic+1
      fname='ajob'
      if (ic .lt. 10) then
         write(fname(5:5),'(i1)') ic
      elseif (ic .lt. 100) then
         write(fname(5:6),'(i2)') ic
      elseif (ic .lt. 1000) then
         write(fname(5:7),'(i3)') ic
      endif
      open (unit=lun, file = fname, status='unknown')
      write(lun,15) '#!/bin/bash'
      write(lun,15) '#PBS -q ' // PBS_QUE
      write(lun,15) '#PBS -o PBS.log'
      write(lun,15) '#PBS -e PBS.err'
      write(lun,15) 'if [[ "$PBS_O_WORKDIR" != "" ]]; then' 
      write(lun,15) '    cd $PBS_O_WORKDIR'
      write(lun,15) 'fi'
      write(lun,15) 'k=run1_app.log'
      write(lun,15) 'script=' // fname
      write(lun,15) 'rm -f wait.$script >& /dev/null'
      write(lun,15) 'touch run.$script'
      write(lun,15) 'echo $script'
      write(lun,'(a$)') 'for i in '
 15   format(a)
      end

      subroutine write_bash(mapconfig,use_config, pwidth, jcomp,iforest,
     $     sprop)
c***************************************************************************
c     Writes out bash commands to run integration over all of the various
c     configurations, but only for "non-identical" configurations.
c     Also labels multiplication factor for each used configuration
c***************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      integer    imax,   ibase
      parameter (imax=max_branch-1, ibase=3)
c
c     Arguments
c
      integer mapconfig(0:lmaxconfigs),use_config(0:lmaxconfigs)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer jcomp

c
c     local
c
      integer i, j, nbw, ic, icode
      integer iarray(imax)
      logical lconflict(-max_branch:nexternal)
      logical done
      logical gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'

c-----
c  Begin Code
c-----
      call open_bash_file(26)
      ic = 0      
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            call bw_conflict(i,iforest(1,-max_branch,i),lconflict,
     $           sprop(-max_branch,i), gForceBW(-max_branch,i))
            nbw=0               !Look for B.W. resonances
            if (jcomp .eq. 0 .or. jcomp .eq. 1 .or. .true.) then
               do j=1,imax
                  iarray(j)=0   !Assume no cuts on BW
               enddo
               do j=1,nexternal-3
c                  write(*,*) 'Width',pwidth(-j,i),j,i
                  if (pwidth(-j,i) .gt. 1d-20) then
                     nbw=nbw+1
                     write(*,*) 'Got bw',-nbw,j
                     if(lconflict(-j).or.gForceBW(-j,i)) then
                        if(lconflict(-j)) write(*,*) 'Got conflict ',-nbw,j
                        if(gForceBW(-j,i)) write(*,*) 'Got forced BW ',-nbw,j
                        iarray(nbw)=1 !Cuts on BW
                        if (nbw .gt. imax) then
                           write(*,*) 'Too many BW w conflicts',nbw,imax
                        endif
                     endif
                  endif
               enddo
            endif
c            do j=1,2**nbw
            done = .false.
            do while (.not. done)
               call enCode(icode,iarray,ibase,imax)
               ic=ic+1
               if (ic .gt. ChanPerJob) then
                  call close_bash_file(26)
                  call open_bash_file(26)
                  ic = 1
               endif
c               write(*,*) 'mapping',ic,mapconfig(i)
               if (mapconfig(i) .lt. 10) then
                  write(26,'(i1$)') mapconfig(i)
               elseif (mapconfig(i) .lt. 100) then
                  write(26,'(i2$)') mapconfig(i)
               elseif (mapconfig(i) .lt. 1000) then
                  write(26,'(i3$)') mapconfig(i)
               elseif (mapconfig(i) .lt. 10000) then
                  write(26,'(i4$)') mapconfig(i)
               endif
               if (icode .eq. 0) then
c                 write(26,'($a)') '.000'
               elseif (icode .lt. 10) then
                  write(26,'(a,i1$)') '.00', icode
               elseif (icode .lt. 100) then
                  write(26,'(a,i2$)') '.0', icode
               elseif (icode .lt. 1000) then
                  write(26,'(a,i3$)') '.', icode
               else
                  write(*,*) 'Error too many B.W. in symmetry.f',icode
                  stop
               endif
               write(26,'(a$)') ' '
               call bw_increment_array(iarray,imax,ibase,gForceBW(-imax,i),done)
            enddo
         endif
      enddo
      call close_bash_file(26)
      if (mapconfig(0) .gt. 9999) then
         write(*,*) 'Only writing first 9999 jobs',mapconfig(0)
      endif
c
c     Now write out the symmetry factors for each graph
c
      open (unit=26, file = 'symfact.dat', status='unknown')
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
c
c        Need to write appropriate number of BW sets this is
c        same thing as above for the bash file
c
            call bw_conflict(i,iforest(1,-max_branch,i),lconflict,
     $           sprop(-max_branch,i), gForceBW(-max_branch,i))
            nbw=0               !Look for B.W. resonances
            if (jcomp .eq. 0 .or. jcomp .eq. 1 .or. .true.) then
               do j=1,imax
                  iarray(j)=0   !Assume no cuts on BW
               enddo
               do j=1,nexternal-3
                  if (pwidth(-j,i) .gt. 1d-20) then
                     nbw=nbw+1
                     write(*,*) 'Got bw',nbw,j
                     if(lconflict(-j).or.gForceBW(-j,i)) then
                        iarray(nbw)=1 !Cuts on BW
                        if (nbw .gt. imax) then
                           write(*,*) 'Too many BW w conflicts',nbw,imax
                        endif
                     endif
                  endif
               enddo
            endif            
            done = .false.
            do while (.not. done)
               call enCode(icode,iarray,ibase,imax)
               if (icode .gt. 0) then
                  write(26,'(f9.3,i6)') mapconfig(i)+real(icode)/1000.,
     $                 use_config(i)
               else
                  write(26,'(2i6)') mapconfig(i),use_config(i)
               endif
               call bw_increment_array(iarray,imax,ibase,gForceBW(-imax,i),done)
            enddo
         else
            write(26,'(2i6)') mapconfig(i),-mapconfig(-use_config(i))
         endif
      enddo
      end


      subroutine BW_Conflict(iconfig,itree,lconflict,sprop,forcebw)
c***************************************************************************
c     Determines which BW conflict
c***************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      double precision zero
      parameter       (zero=0d0)
c      include '../../Source/run_config.inc'
c
c     Arguments
c
      integer itree(2,-max_branch:-1),iconfig
      logical lconflict(-max_branch:nexternal)
      integer sprop(-max_branch:-1)  ! Propagator id
      logical forcebw(-max_branch:-1)  ! Forced BW
c
c     local
c
      integer i,j
      integer iden_part(-max_branch:-1)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagator width
      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagator mass
      double precision pow(-max_branch:-1,lmaxconfigs)    !Not used, in props.inc
      double precision xmass(-max_branch:nexternal)
      include 'maxamps.inc'
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
c
c     Global
c
      include 'coupl.inc'                     !Mass and width info
c-----
c  Begin Code
c-----
      include 'props.inc'   !Propagator mass and width information pmass,pwidth
      write(*,*) 'Checking for BW ',iconfig
c
c     Reset variables
c      
      do i=1,nexternal
         xmass(i) = 0d0
         lconflict(-i) = .false.
      enddo
c
c     Start by determining which propagators are part of the same 
c     chain, or could potentially conflict
c
      i=1
      do while (i .lt. nexternal-2 .and. itree(1,-i) .ne. 1)
         xmass(-i) = xmass(itree(1,-i))+xmass(itree(2,-i))
         if (pwidth(-i,iconfig) .gt. 0d0) then
c     JA 3/31/11 Keep track of identical particles (i.e., radiation vertices)
c     by tracing the particle identity from the external particle.
            if(itree(1,-i).gt.0.and.
     $           sprop(-i).eq.idup(itree(1,-i),1,1).or.
     $         itree(2,-i).gt.0.and.
     $           sprop(-i).eq.idup(itree(2,-i),1,1).or.
     $         itree(1,-i).lt.0.and.(iden_part(itree(1,-i)).ne.0.and.
     $           sprop(-i).eq.iden_part(itree(1,-i)) .or.
     $         forcebw(itree(1,-i)).and.
     $           sprop(-i).eq.sprop(itree(1,-i))).or.
     $         itree(2,-i).lt.0.and.(iden_part(itree(2,-i)).ne.0.and.
     $           sprop(-i).eq.iden_part(itree(2,-i)).or.
     $         forcebw(itree(2,-i)).and.
     $           sprop(-i).eq.sprop(itree(2,-i))))then
               iden_part(-i) = sprop(-i)
            endif
            if (xmass(-i) .gt. pmass(-i,iconfig) .and.
     $           iden_part(-i).eq.0) then !Can't be on shell, and not radiation
               lconflict(-i)=.true.
               write(*,*) "Found Conflict", iconfig,i,
     $              pmass(-i,iconfig),xmass(-i)
            endif
         endif
         xmass(-i) = max(xmass(-i),pmass(-i,iconfig)+3d0*pwidth(-i,iconfig))        
         i=i+1
      enddo
c
c     Mark all daughters of conflicted BW as conflicting
c
      do j=i,1,-1
         if (lconflict(-j)) then 
            lconflict(itree(1,-j)) = .true.
            lconflict(itree(2,-j)) = .true.
            write(*,*) 'Adding conflict ',itree(1,-j),itree(2,-j)
         endif
      enddo
c
c     Only include BW props as conflicting
c
      do j=i,1,-1
         if (lconflict(-j)) then 
            if (pwidth(-j,iconfig) .le. 0 .or. iden_part(-j).gt.0) then 
               lconflict(-j) = .false.
               write(*,*) 'No conflict BW',iconfig,j
            else
               write(*,*) 'Conflicting BW',iconfig,j
            endif
         endif
      enddo                  

      end

      subroutine close_bash_file(lun)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
c
c     Constants
c
c
c     Constants
c
c     Arguments
c
      integer lun
c
c     global
c
      logical gridpack
      common/to_gridpack/gridpack
c
c     local
c
      character*30 fname
      integer ic

      data ic/0/
c-----
c  Begin Code
c-----

      write(lun,'(a)') '; do'
c
c     Now write the commands
c      
      write(lun,20) 'echo $i'
      write(lun,20) 'echo $i >& run.$script'
      write(lun,20) 'j=G$i'
      write(lun,20) 'if [[ ! -e $j ]]; then'
      write(lun,25) 'mkdir $j'
      write(lun,20) 'fi'
      write(lun,20) 'cd $j'
      write(lun,20) 'rm -f ftn25 ftn26 ftn99'
      write(lun,20) 'rm -f $k'
      write(lun,20) 'cat ../input_app.txt >& input_app.txt'
      write(lun,20) 'echo $i >> input_app.txt'
      if (.false.) then
         write(lun,20) 'cp ../../public.sh .'
         write(lun,20) 'qsub -N $1$i public.sh'
      else
         write(lun,20) 'time ../madevent > $k <input_app.txt'
         write(lun,20) 'rm -f ftn25 ftn99'
         if(.not.gridpack) write(lun,20) 'rm -f ftn26'
         write(lun,20) 'cp $k log.txt'
      endif
      write(lun,20) 'cd ../'
      write(lun,15) 'done'
      write(lun,15) 'rm -f run.$script'
      write(lun,15) 'touch done.$script'
 15   format(a)
 20   format(5x,a)
 25   format(10x,a)
      close(lun)
      end



      subroutine bw_increment_array(iarray,imax,ibase,force,done)
c************************************************************************
c     Increments iarray     
c************************************************************************
      implicit none
c
c     Arguments
c
      integer imax          !Input, number of elements in iarray
      integer ibase         !Base for incrementing, 0 is skipped
      logical force(imax)   !Force onshell BW, counting from -imax to -1
      integer iarray(imax)  !Output:Array of values being incremented
      logical done          !Output:Set when no more incrementing

c
c     Global
c
      include 'genps.inc'
      include 'nexternal.inc'

c
c     Local
c
      integer i,j
      logical found
c-----
c  Begin Code
c-----
      found = .false.
      i = 1
      do while (i .le. imax .and. .not. found)
         if (iarray(i) .eq. 0) then    !don't increment this
            i=i+1
         elseif (iarray(i) .lt. ibase-1 .and. .not. force(imax+1-i)) then
            found = .true.
            iarray(i)=iarray(i)+1
         else
            iarray(i)=1
            i=i+1
         endif
      enddo
      done = .not. found
      end

      subroutine store_events()
c**********************************************************************
c     Dummy routine
c**********************************************************************
      end

      double precision function dsig(pp,wgt,imode)
c**********************************************************************
c     Dummy routine
c**********************************************************************
      integer pp,wgt,imode
      dsig=0d0
      return
      end

      subroutine clear_events()
c**********************************************************************
c     Dummy routine
c**********************************************************************
      end

      integer function n_unwgted()
c**********************************************************************
c     Dummy routine
c**********************************************************************
      n_unwgted = 1
      end

