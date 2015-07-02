
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
      include 'run_config.inc'
      include 'maxamps.inc'
      include 'nexternal.inc'
      
      double precision ZERO
      parameter       (ZERO = 0d0)
      integer   maxswitch
      parameter(maxswitch=99)
      integer lun
      parameter (lun=28)
c
c     Local
c
      integer mapconfig(0:lmaxconfigs)
      integer use_config(0:lmaxconfigs)
      integer i,j, npara, nhel_survey
      double precision xdum
      double precision prmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision prwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)
      character*20 param(maxpara),value(maxpara)
      double precision pmass(nexternal)   !External particle mass
      double precision pi1(0:3),pi2(0:3),m1,m2,ebeam(2)
      integer lpp(2)
c
c     Global
c
      include 'coupl.inc'
      logical gridpack
      common/to_gridpack/gridpack
      double precision bwcutoff
      common/to_bwcutoff/bwcutoff
      double precision stot
      common/to_stot/stot
c
c     DATA
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'
      data use_config/0,lmaxconfigs*0/

c-----
c  Begin Code
c-----

c      write(*,*) 'Enter compression (0=none, 1=sym, 2=BW, 3=full)'
c      read(*,*) icomp
c      if (icomp .gt. 3 .or. icomp .lt. 0) icomp=0
c      if (icomp .eq. 0) then
c         write(*,*) 'No compression, summing every diagram and ',
c     $        'every B.W.'
c      elseif (icomp .eq. 1) then
c         write(*,*) 'Using symmetry but summing every B.W. '
c      elseif (icomp .eq. 2) then
c         write(*,*) 'Assuming B.W. but summing every diagram. '
c      elseif (icomp .eq. 3) then
c         write(*,*) 'Full compression. Using symmetry and assuming B.W.'
c      else
c         write(*,*) 'Unknown compression',icomp
c         stop
c      endif

      call load_para(npara,param,value)
      call get_logical(npara,param,value," gridpack ",gridpack,.false.)
      call get_real(npara,param,value," bwcutoff ",bwcutoff,5d0)
      call get_real(npara,param,value," ebeam1 ",ebeam(1),0d0)
      call get_real(npara,param,value," ebeam2 ",ebeam(2),0d0)
      call get_integer(npara,param,value," lpp1 ",lpp(1),0)
      call get_integer(npara,param,value," lpp2 ",lpp(2),0)

      call setpara('%(param_card_name)s' %(setparasecondarg)s)   !Sets up couplings and masses
      include 'pmass.inc'

c     Set stot
      if (nincoming.eq.1) then
         stot=pmass(1)**2
      else
         m1=pmass(1)
         m2=pmass(2)
         if (abs(lpp(1)) .eq. 1 .or. abs(lpp(1)) .eq. 2) m1 = 0.938d0
         if (abs(lpp(2)) .eq. 1 .or. abs(lpp(2)) .eq. 2) m2 = 0.938d0
         if (abs(lpp(1)) .eq. 3) m1 = 0.000511d0
         if (abs(lpp(2)) .eq. 3) m2 = 0.000511d0
         pi1(0)=ebeam(1)
         pi1(3)=sqrt(ebeam(1)**2-m1**2)
         pi2(0)=ebeam(2)
         pi2(3)=-sqrt(ebeam(2)**2-m2**2)
         stot=m1**2+m2**2+2*(pi1(0)*pi2(0)-pi1(3)*pi2(3))
      endif

c      call printout
      include 'props.inc'

c
c     Start reading use_config from symfact.dat written by MG5
c
      open(unit=25, file='symfact_orig.dat', status='old')
      i=0
      do j=1,mapconfig(0)
         do while(i.lt.mapconfig(j))
            read(25,*) xdum, use_config(j)
            i=int(xdum)
         enddo
      enddo
      close(25)

      call write_bash(mapconfig,use_config,prwidth,icomp,iforest,sprop)
      end


      subroutine write_bash(mapconfig,use_config, prwidth, jcomp,iforest,
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
      include 'run_config.inc'
      include 'maxamps.inc'
      integer    imax,   ibase
      parameter (imax=max_branch-1, ibase=3)
c
c     Arguments
c
      integer mapconfig(0:lmaxconfigs),use_config(0:lmaxconfigs)
      double precision prwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(maxsproc,-max_branch:-1,lmaxconfigs)
      integer jcomp

c
c     local
c
      integer i, j, nbw, ic, icode
      integer ncode, nconf, nsym
      double precision dconfig
      character*20 formstr,formstr2,filename
      integer iarray(imax)
      logical lconflict(-max_branch:nexternal)
      logical done,file_exists
      logical failConfig
      external failConfig
      integer gForceBW(-max_branch:-1,lmaxconfigs)  ! Forced BW
      include 'decayBW.inc'

c-----
c  Begin Code
c-----
c     First open symfact file
      open (unit=27, file = 'symfact.dat', status='unknown')
c
c     VA 15/05/2015 It was +3, leading to symfact.dat lines like these
c     112-100
c     So I add here a safety margin 10.
      nsym=int(dlog10(dble(mapconfig(0))))+13

      ic = 0      
c     ncode is number of digits needed for the code
      ncode=int(dlog10(3d0)*(max_particles-3))+1
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            call bw_conflict(i,iforest(1,-max_branch,i),lconflict,
     $           sprop(1,-max_branch,i), gForceBW(-max_branch,i))
            nbw=0               !Look for B.W. resonances
            if (jcomp .eq. 0 .or. jcomp .eq. 1 .or. .true.) then
               do j=1,imax
                  iarray(j)=0   !Assume no cuts on BW
               enddo
               do j=1,nexternal-3
c                  write(*,*) 'Width',prwidth(-j,i),j,ic                 
                  if (prwidth(-j,i) .gt. 0d0 .and. iforest(1,-j,i).ne.1) then
                     nbw=nbw+1
c                     write(*,*) 'Got bw',-nbw,j
c                    JA 4/8/11 don't treat forced BW differently
                     if(lconflict(-j)) then
c                        write(*,*) 'Got conflict ',-nbw,j
                        iarray(nbw)=1 !Cuts on BW
c                        if (nbw .gt. imax) then
c                           write(*,*) 'Too many BW w conflicts',nbw,imax
c                        endif
                     endif
                  endif
               enddo
            endif
c            do j=1,2**nbw
            done = .false.
            do while (.not. done)
               call enCode(icode,iarray,ibase,imax)
               if(failConfig(i,iarray,iforest(1,-max_branch,i),
     $           sprop(1,-max_branch,i),gForceBW(-max_branch,i))) then 
                  goto 100
               endif
               ic=ic+1
               nconf=int(dlog10(dble(mapconfig(i))))+1
c               write(*,*) 'mapping',ic,mapconfig(i),icode               
               if (icode .eq. 0) then
c                 Create format string based on number of digits
                  write(formstr,'(a,i1,a)') '(I',nconf,'$)'
                  write(*,formstr) mapconfig(i)
c                 Write symmetry factors
                  write(formstr2,'(a,i2,a)') '(2i',nsym,')'
                  write(27,formstr2) mapconfig(i),use_config(i)
               else
c                 Create format string based on number of digits
                  dconfig=mapconfig(i)+icode*1d0/10**ncode
                  if(nconf+ncode+1.lt.10) then
                     write(formstr,'(a,i1,a,i1,a)') '(F',nconf+ncode+1,
     $                    '.',ncode,'$)'
                  else
                     write(formstr,'(a,i2,a,i1,a)') '(F',nconf+ncode+1,
     $                    '.',ncode,'$)'
                  endif
                  write(*,formstr) dconfig
c                 Write symmetry factors
                  nconf=int(dlog10(dble(mapconfig(i))))+1
                  if(nconf+ncode+1.lt.10) then
                     write(formstr2,'(a,i1,a,i1,a,i2,a)') '(F',nconf+ncode+1,
     $                    '.',ncode,',i',nsym,')'
                  else
                     write(formstr2,'(a,i2,a,i1,a,i2,a)') '(F',nconf+ncode+1,
     $                    '.',ncode,',i',nsym,')'
                  endif
                  dconfig=mapconfig(i)+icode*1d0/10**ncode
                  write(27,formstr2) dconfig,use_config(i)
               endif
               write(*,'(a$)') ' '
 100           call bw_increment_array(iarray,imax,ibase,done)
            enddo
         else
            write(formstr2,'(a,i2,a)') '(2i',nsym,')'
            write(27,formstr2) mapconfig(i),use_config(i)
         endif
      enddo
      close(27)
      if(ic.eq.0) then
c        Stop generation with error message
         filename='../../error'
         INQUIRE(FILE="../../RunWeb", EXIST=file_exists)
         if(.not.file_exists) filename = '../' // filename
         open(unit=26,file=filename,status='unknown')
         write(26,*)'No Phase Space. Please check particle masses.'
c         write(*,*)'Error: No valid channels found. ',
c     $        'Please check particle masses.'
      endif
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
      include 'maxamps.inc'
      include 'nexternal.inc'
      double precision zero
      parameter       (zero=0d0)
c      include 'run_config.inc'
c
c     Arguments
c
      integer itree(2,-max_branch:-1),iconfig
      logical lconflict(-max_branch:nexternal)
      integer sprop(maxsproc,-max_branch:-1)  ! Propagator id
      integer forcebw(-max_branch:-1) ! Forced BW, for identical particle conflicts
c
c     local
c
      integer i,j,it
      integer iden_part(-nexternal+1:nexternal)
      double precision prwidth(-nexternal:0,lmaxconfigs)  !Propagator width
      double precision prmass(-nexternal:0,lmaxconfigs)   !Propagator mass
      double precision pow(-nexternal:0,lmaxconfigs)    !Not used, in props.inc
      double precision xmass(-max_branch:nexternal)
      double precision pmass(nexternal)   !External particle mass
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      integer ipdg(-nexternal+1:nexternal)
      double precision mtot
      include 'leshouche.inc'
c
c     Global
c
      include 'coupl.inc'                     !Mass and width info
      double precision stot
      common/to_stot/stot

c-----
c  Begin Code
c-----
      include 'props.inc'   !Propagator mass and width information prmass,prwidth
      include 'pmass.inc'   !External particle masses
c      write(*,*) 'Checking for BW in config number ',iconfig
c
c     Reset variables
c      
      do i=1,nexternal
         xmass(i) = pmass(i)
      enddo
      do i=1,nexternal-1
         lconflict(-i) = .false.
      enddo
c     Initialize mtot (needed final-state phase space)
      mtot=0
      do i=nincoming+1,nexternal
          mtot=mtot+xmass(i)
      enddo

c
c     Start by keeping track of identical particles. Only view the outermost
c     identical particle as a BW, unless it is a required BW
c     
      call idenparts(iden_part,itree,sprop,forcebw,
     $               prwidth(-nexternal,iconfig))
c
c     Now determine which propagators are part of the same 
c     chain and could potentially conflict
c
      i=1
      do while (i .lt. nexternal-2 .and. itree(1,-i) .ne. 1)
         xmass(-i) = xmass(itree(1,-i))+xmass(itree(2,-i))
         mtot=mtot-xmass(-i)
         if (prwidth(-i,iconfig) .gt. 0d0) then
            if (xmass(-i) .gt. prmass(-i,iconfig) .and.
     $           iden_part(-i).eq.0) then !Can't be on shell, and not radiation
               lconflict(-i)=.true.
c               write(*,*) "Found Conflict", iconfig,i,
c     $              prmass(-i,iconfig),xmass(-i)
            endif
         endif
         if (iden_part(-i).eq.0) then
            xmass(-i) = max(xmass(-i),prmass(-i,iconfig)+3d0*prwidth(-i,iconfig))        
         endif
         mtot=mtot+xmass(-i)
         i=i+1
      enddo
c
c     Mark all daughters of conflicted BW as conflicting
c
      do j=i,1,-1
         if (lconflict(-j)) then 
            lconflict(itree(1,-j)) = .true.
            lconflict(itree(2,-j)) = .true.
c            write(*,*) 'Adding conflict ',itree(1,-j),itree(2,-j)
         endif
      enddo
c
c     If not enough energy, mark all BWs as conflicting
c
      if(stot.lt.mtot**2)then
c         write(*,*) 'Not enough energy, set all BWs as conflicting'
         do j=i,1,-1
            lconflict(-j) = .true.
         enddo
      endif
c
c     Only include BW props as conflicting, but not if radiation
c
      do j=i,1,-1
         if (lconflict(-j)) then 
            if (prwidth(-j,iconfig) .le. 0 .or. iden_part(-j).gt.0) then 
               lconflict(-j) = .false.
c               write(*,*) 'No conflict BW',iconfig,j
               continue
c            else
c               write(*,*) 'Conflicting BW',iconfig,j
            endif
         endif
      enddo                  

      end

      function failConfig(iconfig,iarray,itree,sprop,forcebw)
c***************************************************************************
c     Determines if the configuration allows integration based on
c     mass relations
c***************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'maxamps.inc'
      include 'nexternal.inc'
      double precision zero
      parameter       (zero=0d0)
      integer    imax
      parameter (imax=max_branch-1)
c
c     Arguments
c
      logical failConfig
      integer iconfig,iarray(imax),itree(2,-max_branch:-1)
      logical lconflict(-max_branch:nexternal)
      integer sprop(maxsproc,-max_branch:-1)  ! Propagator id
      integer forcebw(-max_branch:-1) ! Forced BW, for identical particle conflicts
c
c     local
c
      integer i,j,nbw
      double precision prwidth(-nexternal:0,lmaxconfigs)  !Propagator width
      double precision prmass(-nexternal:0,lmaxconfigs)   !Propagator mass
      double precision pow(-nexternal:0,lmaxconfigs)    !Not used, in props.inc
      double precision xmass(-max_branch:nexternal)
      double precision xwidth(-max_branch:nexternal)
      double precision pmass(nexternal)   !External particle mass
      double precision mtot
      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
c
c     Global
c
      double precision bwcutoff
      common/to_bwcutoff/bwcutoff
      double precision stot
      common/to_stot/stot
      include 'coupl.inc'                     !Mass and width info

c-----
c  Begin Code
c-----
      include 'props.inc'   !Propagator mass and width information prmass,prwidth
      include 'pmass.inc'   !External particle masses
c
c     Reset variables
c      
      do i=1,nexternal
         xmass(i) = pmass(i)
         xwidth(i) = 0
      enddo
      do i=1,nexternal-1
         lconflict(-i) = .false.
      enddo
c     Initialize mtot (needed final-state phase space)
      mtot=0
      do i=nincoming+1,nexternal
          mtot=mtot+xmass(i)
      enddo

c     By default pass
      failConfig=.false.

c
c     Go through
c
      nbw=0
      i=1
      do while (i .lt. nexternal-2 .and. itree(1,-i) .ne. 1)
         xmass(-i) = xmass(itree(1,-i))+xmass(itree(2,-i))
         mtot=mtot-xmass(-i)
         xwidth(-i)=prwidth(-i,iconfig)
         if (xwidth(-i) .gt. 0d0) then
            nbw=nbw+1
            if(forcebw(-i) .eq. 1) then
c               if (iarray(nbw) .ne. 1) then
c                  write(*,*) "fail due to iarray", iarray(nbw)
c                  failConfig=.true.
c                  return
c               endif
               if(xmass(-i).gt.prmass(-i,iconfig)+bwcutoff*xwidth(-i)) then
                  failConfig=.true.
                  return
               else
                  xmass(-i)=max(xmass(-i),prmass(-i,iconfig)-
     $                 bwcutoff*xwidth(-i))
               endif
            else if (iarray(nbw) .eq. 1) then
               if(xmass(-i).gt.prmass(-i,iconfig)+5d0*xwidth(-i)) then
                  failConfig=.true.
                  return
               else
                  xmass(-i)=max(xmass(-i),prmass(-i,iconfig)-5d0*xwidth(-i))
               endif

            endif
         endif
         mtot=mtot+xmass(-i)
         i=i+1
      enddo

c     Fail if too small phase space
      if (stot.lt.mtot**2) then
         failConfig=.true.
      endif

      return
      end

      subroutine bw_increment_array(iarray,imax,ibase,done)
c************************************************************************
c     Increments iarray     
c************************************************************************
      implicit none
c
c     Arguments
c
      integer imax          !Input, number of elements in iarray
      integer ibase         !Base for incrementing, 0 is skipped
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
         elseif (iarray(i) .lt. ibase-1) then
            found = .true.
            iarray(i)=iarray(i)+1
         else
            iarray(i)=1
            i=i+1
         endif
      enddo
      done = .not. found
      end

      subroutine store_events(grid)
c**********************************************************************
c     Dummy routine
c**********************************************************************
      integer grid
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

