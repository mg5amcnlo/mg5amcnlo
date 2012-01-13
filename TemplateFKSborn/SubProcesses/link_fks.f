      program link_fks
c*****************************************************************************
c     Given identical particles, and the configurations. This program identifies
c     identical configurations and specifies which ones can be skipped
c*****************************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'      
      include "nexternal.inc"
      include '../../../Source/run_config.inc'
      
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
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer itree(2,-max_branch:-1)
      integer imatch
      integer use_config(0:lmaxconfigs)
      integer i,j, k, n, nsym,borngraph
      double precision diff
      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'


      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer fksmother,fksgrandmother,fksaunt,i_fks,j_fks
      integer fksconfiguration,mapbconf(0:lmaxconfigs)
      integer r2b(lmaxconfigs),b2r(lmaxconfigs)
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown,non_prop
      integer nbranch,ns_channel,nt_channel
      include "fks.inc"
c
c     Local for generating amps
c
      double precision p(0:3,99), wgt, x(99), fx(2)
      double precision p1(0:3,99)
      integer ninvar, ndim, iconfig, minconfig, maxconfig
      integer ncall,itmax,nconfigs,ntry, ngraphs
      integer ic(nexternal,maxswitch), jc(12),nswitch
      double precision saveamp(maxamps)
      integer nmatch, ibase
      logical mtc, even
c-----
c  Begin Code
c-----

      use_config(0)=0
c Read FKS configuration from file
      open (unit=61,file='config.fks',status='old')
      read(61,'(I2)',err=99,end=99) fksconfiguration
 99   close(61)
c Use the fks.inc include file to set i_fks and j_fks
      i_fks=fks_i(fksconfiguration)
      j_fks=fks_j(fksconfiguration)
      write (*,*) 'FKS configuration number is ',fksconfiguration
      write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks

c Remove diagrams which do not have the correct structure
c for the FKS partons i_fks and j_fks from the list of
c integration channels.
      write (*,*) 'Linking FKS configurations to Born configurations...'
      open (unit=14,file='bornfromreal.inc',status='unknown')
      write (14, '(a)') 'c linked configurations:'
      searchforgranny=.false.
      do i=1,mapconfig(0)
         fksmother=0
c find number of s and t channels
         nbranch=nexternal-2
         ns_channel=1
         do while(iforest(1,-ns_channel,i) .ne. 1 .and.
     &            ns_channel.lt.nbranch)
            ns_channel=ns_channel+1         
         enddo
         ns_channel=ns_channel - 1
         nt_channel=nbranch-ns_channel-1
         call grandmother_fks(i,nbranch,ns_channel,
     &        nt_channel,i_fks,j_fks,searchforgranny,
     &        fksmother,fksgrandmother,fksaunt,
     &        is_beta_cms,is_granny_sch,topdown)
         non_prop=.false.
c Skip diagrams with non-propagating particles:
         do j=1,ns_channel
            if (sprop(-j,i).eq.99) non_prop=.true.
         enddo
         do j=ns_channel+1,ns_channel+nt_channel
            if (tprid(-j,i).eq.99) non_prop=.true.
         enddo
         if (fksmother.ne.0 .and..not.non_prop)then !Found good diagram
            use_config(0) = use_config(0)+1 ! # of good configs found so far
c For each diagrams that contributes to a FKS configuration, there
c must be a corresponding Born diagram. Link the diagrams in
c configs.inc to the Born diagrams in born_conf.inc.
c$$$            call link_to_born(iforest(1,-max_branch,i),i,i_fks,j_fks,
c$$$     &           fksmother, nbranch,mapbconf, r2b(i), biforest)
            call link_to_born2(iforest(1,-max_branch,i),sprop(-max_branch,i),
     &           tprid(-max_branch,i),i,i_fks,j_fks,fksmother,nbranch,
     &           ns_channel,nt_channel,mapbconf, r2b(i), biforest)
            write (14,'(6x,a14,i4,a4,i4,a1)')
     &         'data b_from_r(',mapconfig(i),') / ',mapbconf(r2b(i)),'/'
            write (14,'(6x,a14,i4,a4,i4,a1)')
     &         'data r_from_b(',mapbconf(r2b(i)),') / ',mapconfig(i),'/'
            b2r(r2b(i))=i       ! also need inverse 
            if (topdown) then
               call invert_order_iforest_REAL(nbranch,ns_channel,
     &              nt_channel,i)
               call invert_order_iforest_BORN(nbranch-1,ns_channel,
     &              nt_channel-1,r2b(i))
            endif
            iconfig=mapconfig(i)
         endif
      enddo
      write (14,'(6x,a12)')
     &     'integer mapb'
      write (14,'(6x,a28,i4,a3,i4,a1)') 'data (mapbconf(mapb),mapb=0,',
     &                               mapbconf(0),') / ',mapbconf(0),','
      do i=1,mapbconf(0)
         if (i.lt.mapbconf(0))then
            write (14,'(5x,a1,i4,a1)') '&',mapbconf(i),','
         else
            write (14,'(5x,a1,i4,a1)') '&',mapbconf(i),'/'
         endif
      enddo
      close (14)
      if (use_config(0).ne.mapbconf(0))then
         write (*,*) 'FATAL ERROR 101 in symmetry',
     &        use_config(0),mapbconf(0)
c         stop
      endif
      write (*,*) '...Configurations linked'

      return
      end


c
c
c Dummy routines
c
c
      subroutine outfun(pp,www)
      write(*,*)'"outfun" routine should not be called here'
      stop
      end

      subroutine store_events()
      write(*,*)'"store_events" routine should not be called here'
      stop
      end

      integer function n_unwgted()
      n_unwgted = 1
      write(*,*)'"n_unwgted" function should not be called here'
      stop
      end

      subroutine clear_events()
      write(*,*)'"clear_events" routine should not be called here'
      stop
      end

