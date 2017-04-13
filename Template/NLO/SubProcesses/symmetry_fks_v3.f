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
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      include 'mint.inc'
      
      double precision ZERO
      parameter       (ZERO = 0d0)
      integer   maxswitch
      parameter(maxswitch=99)
c
c     Local
c
      integer itree(2,-max_branch:-1)
      integer imatch
      integer use_config(0:lmaxconfigs)
      integer i,j, k, n, nsym
      double precision diff,rwgt
      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)

      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer fksmother,fksgrandmother,fksaunt
      integer fksconfiguration
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown
      integer nbranch,ns_channel,nt_channel
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
c
c     Local for generating amps
c
      double complex wgt1(2)
      double precision p(0:3,nexternal), wgt, x(99), fx
      double precision p1(0:3,nexternal)
      double precision p1_cnt1(0:3,nexternal,-2:2),p_born1(0:3,nexternal
     &     -1),p_ev_red_save(0:3,nexternal-1)
      double precision p1_cnt_save(0:3,nexternal,-2:2),p_born_save(0:3
     &     ,nexternal-1),p_ev_red1(0:3,nexternal-1)
      integer ninvar, ndim, minconfig, maxconfig
      common/tosigint/ndim
      integer ncall,itmax,nconfigs,ntry, ngraphs
      integer icb(nexternal-1,maxswitch),jc(12),nswitch
      double precision saveamp(maxamps)
      integer nmatch, ibase
      logical mtc, even

c
c     Global
c
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      include 'coupl.inc'

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision p_ev_red(0:3,nexternal-1)
      common/cpevred/p_ev_red

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      logical xexternal
      common /toxexternal/ xexternal


      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      integer run_cluster
      common/c_run_mode/run_cluster

c
c     External
c
      logical passcuts
      logical check_swap
      double precision dsig,fks_sij,dsig2,ran2
      external passcuts, dsig,fks_sij,dsig2
      external check_swap,ran2

c      integer icomp
c
c     DATA
c
      include 'born_conf.inc'
c$$$      integer mapbconf(0:lmaxconfigs)
c$$$      integer b_from_r(lmaxconfigs)
c$$$      integer r_from_b(lmaxconfigs)
c$$$      include 'bornfromreal.inc'
      include 'fks_powers.inc'

      include 'nFKSconfigs.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

c helicity stuff
      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      logical nbody
      common/cnbody/nbody
      logical is_aorg(nexternal)
      common /c_is_aorg/is_aorg

      logical new_point
      common /c_new_point/new_point
c-----
c  Begin Code
c-----
      write (*,'(a)') "Setting-up ajob's..."
      write (*,'(a)') "Give run mode and (adaptive) importance sampler:"
      write (*,'(a)') '"0" for local run, "1" for condor cluster' //
     1 '"2" for multicore'
      read  (*,*) run_cluster
      if (run_cluster.eq.0) then
         write (*,*) "Setting up ajob's for local run"
      elseif(run_cluster.eq.1) then
         write (*,*) "Setting up ajob's for condor cluster run"
      elseif(run_cluster.eq.2) then
         write (*,*) "Setting up ajob's for multicore run"
c     in this case it is the same as for serial run
         run_cluster=0
      else
         write (*,*) "Invalid run mode", run_cluster
      endif

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
      isum_hel=0
      multi_channel=.true.


      nbody=.true.
c Pick a process that is BORN+1GLUON (where the gluon is i_fks).
      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
         if (is_aorg(i_fks)) exit
      enddo
c If there is no fks configuration that has a gluon or photon as i_fks
c (this might happen in case of initial state leptons with
c include_lepton_initiated_processes=False) the Born and virtuals do not
c need to be included, but we still need to set the symmetry
c factors. Hence, simply use the first fks_configuration.
      if (nFKSprocess.gt.fks_configs) nFKSprocess=1
      call leshouche_inc_chooser()
      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts 
      call printout
      call run_printout
      iconfig=1
      ichan=1
      iconfigs(1)=iconfig
      call setfksfactor(.false.)
      
c
      ndim = 55
      ncall = 10000
      itmax = 10
      ninvar = 35
      nconfigs = 1
c      write (*,*) mapconfig(0)

      use_config(0)=0
c
c     Start using all (Born) configurations
c
      do i=1,mapconfig(0)
         use_config(i)=1
      enddo
c     
c     Get momentum configuration
c
c Set xexternal to false to generate new xexternal in the x_to_f_arg subroutine
c$$$      xexternal=.false.
      xexternal=.true.

c Set-up helicities
      call get_helicity(i_fks,j_fks)

      do i = 1,3   !Not needed, but ok
      ntry = 1
      wgt=1d0
      do j=1,ndim
         x(j)=ran2()
      enddo
      new_point=.true.
      call generate_momenta(ndim,iconfig,wgt,x,p)
      do while ((.not.passcuts(p,rwgt) .or. wgt.lt.0 .or. p(0,1).le.0d0
     &           .or. p_born(0,1).le.0d0) .and. ntry.lt.10000)
         do j=1,ndim
            x(j)=ran2()
         enddo
         new_point=.true.
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         ntry=ntry+1
      enddo
      enddo
      write(*,*) 'ntry',ntry
      call set_alphaS(p)
c
c     Get and save base amplitudes
c
      calculatedBorn=.false.
c Call the Born twice to make sure that all common blocks are correctly filled.
      call sborn(p_born,wgt1)
      call sborn(p_born,wgt1)
      do j = 1 , mapconfig(0)
         saveamp(mapconfig(j)) = amp2(mapconfig(j))
      enddo
      write (*,*) 'born momenta'
      do j=1,nexternal-1
         do i=0,3
            p_born_save(i,j)=p_born(i,j)
            p_ev_red_save(i,j)=p_ev_red(i,j)
         enddo
         write(*,'(i4,4e15.5)') j,(p_born(i,j),i=0,3)
      enddo
c
c     Swap amplitudes looking for matches
c
c nexternal is the number for the real configuration. Subtract 1 for the Born.
      nswitch = 1
      do k=1,nexternal-1
         icb(k,1)=k
      enddo
      nmatch = 0
      mtc=.false.
      nsym = 1
      call nexper(nexternal-3,icb(3,1),mtc,even)
c      write(*,*) 'mtc',mtc, (ic(i,1),i=1,nexternal)
      do while(mtc)
         call nexper(nexternal-3,icb(3,1),mtc,even)
c         write(*,*) 'mtc',mtc, (ic(i,1),i=1,nexternal)
         do j=3,nexternal-1
            icb(j,1)=icb(j,1)+2
         enddo
         
c$$$         if (mtc) then
c
c     Now check if it is a valid swap to make
c
         if (check_swap(icb(1,1))) then
            CALL SWITCHMOM(P_born_save,P_born1,ICB(1,1),JC,NEXTERNAL-1)
            do j=1,nexternal-1
               do k=0,3
                  p_born(k,j)=p_born1(k,j)
               enddo
            enddo

            write(*,*) 'Good swap', (icb(i,1),i=1,nexternal-1)
            nsym=nsym+1

            calculatedBorn=.false.
            call sborn(p_born,wgt1)
c$$$            write (*,*) 'saveamp',(saveamp(j),j=1,mapconfig(0))
c$$$            write (*,*) 'new amp',(amp2(j),j=1,mapconfig(0))
c        Look for matches, but only for diagrams < current diagram
c     
         do j=2,mapconfig(0)
            do k=1,j-1
               diff=abs((amp2(mapconfig(j))-saveamp(mapconfig(k)))
     &              /(amp2(mapconfig(j))+1d-99))
               if (diff .lt. 1d-8 ) then
c$$$                  write(*,*) "Found match graph",mapconfig(j),mapconfig(k),diff
                  if (use_config(j) .gt. 0 ) then  !Not found yet
                     nmatch=nmatch+1
                     if (use_config(k) .gt. 0) then !Match is real config
                        use_config(k)=use_config(k)+use_config(j)
                        use_config(j)=-k
                     else
                        ibase = -use_config(k)
                        use_config(ibase) = use_config(ibase)
     &                       +use_config(j)
                        use_config(j) = -ibase
                     endif
                  endif
               endif
            enddo
         enddo
         else
            write(*,*) 'Bad swap', (icb(i,1),i=1,nexternal-1)
         endif   !Good Swap
c$$$         endif   !Real Swap
         do j=3,nexternal-1
            icb(j,1)=icb(j,1)-2
         enddo
      enddo


      write(*,*) 'Found ',nmatch, ' matches. ',mapconfig(0)-nmatch,
     $     ' channels remain for integration.'
      call write_bash(mapconfig,use_config,pwidth,icomp,iforest)

      end

      subroutine store_events()
c**********************************************************************
c     Dummy routine
c**********************************************************************
      end

      subroutine write_symswap
c***********************************************************************
c     This information is used to symmeterize identical particle
c     data
c***********************************************************************
c      implicit none
c      
c      open(unit=lun,file='symswap.inc',status='unknown')
c      write(lun,*) '      data nsym /1/'
c      write(lun,'(a$)') '       data (isym(i,1),i=1,nexternal) /1 '
c      do i=2,nexternal
c         write(lun,'(a,i2$)') ",",i
c      enddo
c      write(lun,'(a)') "/"
c      close(lun)
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
c
c     Arguments
c
      integer ic(nexternal-1)
c
c     local
c
      integer i
c
c     Process info
c
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
c      include 'leshouche.inc'
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
c------
c Begin Code
c-----
      check_swap=.true.
      do i=1,nexternal-1
         if (idup(i,1) .ne. idup(ic(i),1)) check_swap=.false.
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


      integer function n_unwgted()
c
c     dummy routine
c     
      n_unwgted = 1
      end

      subroutine clear_events()
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
      logical gridpack
c-----
c  Begin Code
c-----
      call load_para(npara,param,value)
      gridpack=.false.

      npoints = min_events_subprocess/nconfigs
      npoints = max(npoints,min_events_channel)
      open (unit=26, file = 'input_app.txt', status='unknown',
     $     err=99)
      if (gridpack) then
         write(26,*) npoints_wu,itmax_wu,
     &     '     !Number of events and iterations'      
         write(26,'(f8.4,a)') acc_wu, '    !Accuracy'
         write(26,*) ' 2       !Grid Adjustment 0=none'
      else
         write(26,*) npoints,iter_survey,
     &     '     !Number of events and iterations'      
         write(26,*) ' 0.0    !Accuracy'
         write(26,*) ' 0       !Grid Adjustment 0=none'
      endif
      write(26,*) ' 1       !Suppress Amplitude 1=yes'
      write(26,*) nhel_survey,'       !Helicity Sum/event 0=exact'
      close(26)
      return
 99   close(26)
      write(*,*) 'Error opening input_app.txt'
      end

      subroutine write_bash(mapconfig, use_config, pwidth, jcomp
     &     ,iforest)
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
      include 'nexternal.inc'
      include '../../Source/run_config.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer    imax,   ibase
      parameter (imax=max_branch-1, ibase=3)
c
c     Arguments
c
      integer mapconfig(0:lmaxconfigs)
      integer use_config(0:lmaxconfigs)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer jcomp

c
c     local
c
      integer i, j
      integer iarray(imax)
      logical lconflict(-max_branch:nexternal)
      logical done,j_fks_ini,j_fks_fin,two_jobs
      integer lname
      character*30 fname
      character*2 postfix

c-----
c  Begin Code
c-----
      j_fks_ini=.false.
      j_fks_fin=.false.
      do i=1,fks_configs
         if (fks_j_d(i).le.nincoming) j_fks_ini=.true.
         if (fks_j_d(i).gt.nincoming) j_fks_fin=.true.
      enddo
      if (j_fks_ini.and.j_fks_fin) two_jobs=.true.
      fname='ajob'
      lname=4
      call open_bash_file(26,fname,lname)
      call close_bash_file(26)
      open(unit=26,file='channels.txt',status='unknown')
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            if (two_jobs) then
               postfix='.1'
            else
               postfix='.0'
            endif
 100        continue
            if (mapconfig(i) .lt. 10) then
               write(26,'(x,i1,a2$)') mapconfig(i),postfix
            elseif (mapconfig(i) .lt. 100) then
               write(26,'(x,i2,a2$)') mapconfig(i),postfix
            elseif (mapconfig(i) .lt. 1000) then
               write(26,'(x,i3,a2$)') mapconfig(i),postfix
            elseif (mapconfig(i) .lt. 10000) then
               write(26,'(x,i4,a2$)') mapconfig(i),postfix
            endif
            if (postfix.eq.'.1') then
               postfix='.2'
               goto 100
            endif
         endif
      enddo
      close(26)
      if (mapconfig(0) .gt. 9999) then
         write(*,*) 'ERROR Only writing first 9999 jobs',mapconfig(0)
         stop 1
      endif
c
c     Now write out the symmetry factors for each channel
c
      open (unit=26, file = 'symfact.dat', status='unknown')
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            if (two_jobs) then
               write(26,'(i6,a2,i6)') mapconfig(i),'.1',use_config(i)
               write(26,'(i6,a2,i6)') mapconfig(i),'.2',use_config(i)
            else
               write(26,'(i6,a2,i6)') mapconfig(i),'.0',use_config(i)
            endif
         else
            if (two_jobs) then
               write(26,'(i6,a2,i6)') mapconfig(i),'.1',-mapconfig(
     $              -use_config(i))
               write(26,'(i6,a2,i6)') mapconfig(i),'.2',-mapconfig(
     $              -use_config(i))
            else
               write(26,'(i6,a2,i6)') mapconfig(i),'.0',-mapconfig(
     $              -use_config(i))
            endif
         endif
      enddo
      end
c
c
c Dummy routines
c
c
      subroutine initplot
      end
      subroutine outfun(pp,www,iplot)
      end

      LOGICAL FUNCTION PASSCUTS(P,rwgt)
      real*8 rwgt
      include 'nexternal.inc'
      real*8 p(0:3,nexternal)
      rwgt=1d0
      passcuts=.true.
      RETURN
      END

      subroutine unweight_function(p_born,unwgtfun)
c Dummy function. Should always retrun 1.
      implicit none
      include 'nexternal.inc'
      double precision unwgtfun,p_born(0:3,nexternal-1)
      unwgtfun=1d0
      return
      end

