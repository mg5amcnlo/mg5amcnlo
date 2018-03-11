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
      include 'mint.inc'
      include 'run.inc'
      include 'nFKSconfigs.inc'
      integer mapconfig(0:lmaxconfigs),iforest(2,-max_branch:-1
     $     ,lmaxconfigs),sprop(-max_branch:-1,lmaxconfigs),tprid(
     $     -max_branch:-1,lmaxconfigs)
      include 'born_conf.inc' ! needed for mapconfig
      logical mtc,even
      integer i,j,k,nmatch,ibase,ntry,icb(nexternal-1),jc(nexternal)
     $     ,use_config(0:lmaxconfigs)
      double precision diff,rwgt,p(0:3,nexternal),wgt,x(99),p_born1(0:3
     $     ,nexternal-1),p_born_save(0:3,nexternal-1),saveamp(ngraphs)
      double complex wgt1(2)
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer         ndim
      common/tosigint/ndim
      Double Precision amp2(ngraphs), jamp2(0:ncolor)
      common/to_amps/  amp2,          jamp2
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      logical                calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      logical       nbody
      common/cnbody/nbody
      logical             new_point
      common /c_new_point/new_point
      logical passcuts,check_swap
      double precision ran2
      external passcuts,check_swap,ran2
c-----
c  Begin Code
c-----
      multi_channel=.true.
      nbody=.true.
c Pick a process that is BORN+1GLUON (where the gluon is i_fks).
      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
         if (particle_type(i_fks).eq.8) exit
      enddo
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
      ndim = 3*(nexternal-nincoming)-4
      if (abs(lpp(1)).ge.1) ndim=ndim+1
      if (abs(lpp(2)).ge.1) ndim=ndim+1
      use_config(0)=0
      do i=1,mapconfig(0)
         use_config(i)=1
      enddo
c     
c     Get momentum configuration
c
      ntry = 1
      do j=1,ndim
         x(j)=ran2()
      enddo
      new_point=.true.
      wgt=1d0
      call generate_momenta(ndim,iconfig,wgt,x,p)
      do while ((.not.passcuts(p,rwgt) .or. wgt.lt.0 .or. p(0,1).le.0d0
     $     .or. p_born(0,1).le.0d0) .and. ntry.lt.10000)
         do j=1,ndim
            x(j)=ran2()
         enddo
         new_point=.true.
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         ntry=ntry+1
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
      do j=1, mapconfig(0)
         saveamp(mapconfig(j)) = amp2(mapconfig(j))
      enddo
      write (*,*) 'born momenta'
      do j=1,nexternal-1
         do i=0,3
            p_born_save(i,j)=p_born(i,j)
         enddo
         write(*,'(i4,4e15.5)') j,(p_born(i,j),i=0,3)
      enddo
c
c     Swap amplitudes looking for matches
c
c nexternal is the number for the real configuration. Subtract 1 for the Born.
      do k=1,nexternal-1
         icb(k)=k
      enddo
      nmatch = 0
      mtc=.false.
      call nexper(nexternal-3,icb(3),mtc,even)
      do while(mtc)
         call nexper(nexternal-3,icb(3),mtc,even)
         do j=3,nexternal-1
            icb(j)=icb(j)+2
         enddo
         if (check_swap(icb(1))) then
            write(*,*) 'Good swap', (icb(i),i=1,nexternal-1)
            CALL SWITCHMOM(P_born_save,P_born1,ICB(1),JC,NEXTERNAL-1)
            do j=1,nexternal-1
               do k=0,3
                  p_born(k,j)=p_born1(k,j)
               enddo
            enddo
            calculatedBorn=.false.
            call sborn(p_born,wgt1)
c        Look for matches
            do j=2,mapconfig(0)
               do k=1,j-1
                  diff=abs((amp2(mapconfig(j))-saveamp(mapconfig(k)))
     &                 /(amp2(mapconfig(j))+1d-99))
                  if (diff .gt. 1d-8 ) cycle
                  if (use_config(j) .lt. 0 ) exit ! already found
                  nmatch=nmatch+1
                  if (use_config(k) .gt. 0) then !Match is real config
                     use_config(k)=use_config(k)+use_config(j)
                     use_config(j)=-k
                  else
                     ibase = -use_config(k)
                     use_config(ibase) = use_config(ibase)
     &                    +use_config(j)
                     use_config(j) = -ibase
                  endif
               enddo
            enddo
         else
            write(*,*) 'Bad swap', (icb(i),i=1,nexternal-1)
         endif
         do j=3,nexternal-1
            icb(j)=icb(j)-2
         enddo
      enddo
      write(*,*) 'Found ',nmatch, ' matches. ',mapconfig(0)-nmatch,
     $     ' channels remain for integration.'
      call write_bash(mapconfig,use_config)
      return
      end


      logical function check_swap(ic)
c**************************************************************************
c     check that only identical particles were swapped
c**************************************************************************
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer ic(nexternal-1),i
      integer get_color
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      check_swap=.true.
      do i=1,nexternal-1
         if (i.eq.ic(i)) cycle ! no permutation. Check next particle
         if (idup(i,1) .ne. idup(ic(i),1)) then
            ! permuted particles are not identical
            check_swap=.false.
            return
         endif
         if (abs(get_color(idup(i,1))).gt.1) then
            ! permuted particles are identical, but not colour neutral
            ! (Since we use symmetry when setting up the FKS
            ! configurations, we cannot use symmetry here as well to
            ! reduce the number of integration channels).
            check_swap=.false.
            return
         endif
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


      subroutine write_bash(mapconfig,use_config)
c***************************************************************************
c     Writes out bash commands to run integration over all of the various
c     configurations, but only for "non-identical" configurations.
c     Also labels multiplication factor for each used configuration
c***************************************************************************
      implicit none
      include 'genps.inc'
      integer mapconfig(0:lmaxconfigs),use_config(0:lmaxconfigs),i,lname
      character*30 fname,mname
      fname='ajob'
      lname=4
      mname='mg'
      call open_bash_file(26,fname,lname,mname)
      call close_bash_file(26)
      open(unit=26,file='channels.txt',status='unknown')
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            if (mapconfig(i) .lt. 10) then
               write(26,'(i1$)') mapconfig(i)
            elseif (mapconfig(i) .lt. 100) then
               write(26,'(i2$)') mapconfig(i)
            elseif (mapconfig(i) .lt. 1000) then
               write(26,'(i3$)') mapconfig(i)
            elseif (mapconfig(i) .lt. 10000) then
               write(26,'(i4$)') mapconfig(i)
            endif
            write(26,'(a$)') ' '
         endif
      enddo
      close(26)
      if (mapconfig(0) .gt. 9999) then
         write(*,*) 'ERROR: only writing first 9999 jobs',mapconfig(0)
         stop 1
      endif
      open (unit=26, file = 'symfact.dat', status='unknown')
      do i=1,mapconfig(0)
         if (use_config(i) .gt. 0) then
            write(26,'(2i6)') mapconfig(i),use_config(i)
         else
            write(26,'(2i6)') mapconfig(i),-mapconfig(-use_config(i))
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

      subroutine bias_weight_function(p,ipdg,bias_wgt)
c Dummy function. Should always retrun 1.
      implicit none
      include 'nexternal.inc'
      integer ipdg(nexternal)
      double precision bias_wgt,p(0:3,nexternal)
      bias_wgt=1d0
      return
      end

