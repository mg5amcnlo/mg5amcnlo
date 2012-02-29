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
      include "nexternal.inc"
      include '../../../Source/run_config.inc'
      
      double precision ZERO,one
      parameter       (ZERO = 0d0)
      parameter       (one = 1d0)
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
      integer i,j, k, n, nsym,l,ii,jj
      double precision diff,xi_i_fks
c$$$      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pmass(nexternal)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)

      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer fksmother,fksgrandmother,fksaunt,i_fks,j_fks,compare
      integer fksconfiguration,mapbconf(0:lmaxconfigs)
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown,non_prop
      integer nbranch,ns_channel,nt_channel
      include "fks.inc"
      double precision fxl,limit(15),wlimit(15)
      double precision lxp(0:3,nexternal+1),xp(15,0:3,nexternal+1)
      double precision fks_Sij
      double precision check,tolerance,zh,h_damp
      parameter (tolerance=1.d-4)
      integer kk,ll,bs,bs_min,bs_max,iconfig_in

      integer nsofttests,ncolltests,nerr,imax,iflag,iret
c
c     Local for generating amps
c
      double precision p(0:3,99), wgt, x(99), fx
      double precision p1(0:3,99),xx(maxinvar)
      integer ninvar, ndim, iconfig, minconfig, maxconfig
      integer ncall,itmax,nconfigs,ntry, ngraphs
      integer ic(nexternal,maxswitch), jc(12),nswitch
      double precision saveamp(maxamps)
      integer nmatch, ibase
      logical mtc, even


      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c
c     Global
c
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      include 'coupl.inc'

      double precision hel_fac
      logical calculatedBorn
      integer get_hel,skip
      common/cBorn/hel_fac,calculatedBorn,get_hel,skip


      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt

      logical rotategranny
      common/crotategranny/rotategranny

      logical softtest,colltest
      common/sctests/softtest,colltest
      
      logical xexternal
      common /toxexternal/ xexternal

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      integer b_from_r(lmaxconfigs), r_from_b(lmaxconfigs)
      include 'bornfromreal.inc'

c
c     External
c
      logical pass_point
      logical check_swap
      double precision dsig
      external pass_point, dsig
      external check_swap, fks_Sij

c helicity stuff
      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel

c      integer icomp
c
c     DATA
c
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'configs.inc'
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

      write(*,*)'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*)' Enter -2 to generate them randomly'
      read(*,*)xi_i_fks_fix,y_ij_fks_fix

      write(*,*)'Enter number of tests for soft and collinear limits'
      read(*,*)nsofttests,ncolltests

      write(*,*)'Sum over helicity (0), or random helicity (1)'
      read(*,*) isum_hel

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts 
c$$$      call printout
c$$$      call run_printout
c
      ndim = 22
      ncall = 10000
      itmax = 10
      ninvar = 35
      nconfigs = 1
c      write (*,*) mapconfig(0)

c Read FKS configuration from file
      open (unit=61,file='config.fks',status='old')
      read(61,'(I2)',err=99,end=99) fksconfiguration
 99   close(61)
c Use the fks.inc include file to set i_fks and j_fks
      i_fks=fks_i(fksconfiguration)
      j_fks=fks_j(fksconfiguration)
      write (*,*) 'FKS configuration number is ',fksconfiguration
      write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks


c Set color types of i_fks, j_fks and fks_mother.
      i_type=particle_type(i_fks)
      j_type=particle_type(j_fks)
      if (abs(i_type).eq.abs(j_type)) then
         m_type=8
         if ( (j_fks.le.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &        (j_fks.gt.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
            write(*,*)'Flavour mismatch #1 in setfksfactor',
     &           i_fks,j_fks,i_type,j_type
            stop
         endif
      elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
         m_type=-i_type
      elseif(abs(j_type).eq.3 .and. i_type.eq.8)then
         m_type=j_type
      else
         write(*,*)'Flavour mismatch #2 in setfksfactor',
     &        i_type,j_type,m_type
         stop
      endif

c
c     Start using all (Born) configurations
c
      do i=1,mapbconf(0)
         use_config(i)=1
      enddo


c$$$      include 'props.inc'
      call sample_init(ndim,ncall,itmax,ninvar,nconfigs)

      open(unit=lun,file='symswap.inc',status='unknown')

c     
c     Get momentum configuration
c

c Set xexternal to true to use the x's from external vegas in the
c x_to_f_arg subroutine
      xexternal=.false.
c$$$
c$$$
c$$$      iconfig=1
      
      write(*,*)'  '
      write(*,*)'  '
      write(*,*)'Enter graph number (iconfig), '
     &     //"'0' loops over all graphs"
      read(*,*)iconfig_in
      
      if (iconfig_in.eq.0) then
         bs_min=1
         bs_max=mapbconf(0)
      elseif (iconfig_in.eq.-1) then
         bs_min=1
         bs_max=1
      else
         bs_min=iconfig_in
         bs_max=iconfig_in
      endif

c$$$      write(*,*)'Using iconfig=',iconfig

      do bs=bs_min,bs_max

         wgt=1d0
         ntry=1

         if (iconfig_in.le.0) then
            iconfig=mapconfig(r_from_b(mapbconf(bs)))
            minconfig=mapconfig(r_from_b(mapbconf(bs)))
            maxconfig=mapconfig(r_from_b(mapbconf(bs)))
         else
            iconfig=bs
            minconfig=iconfig
            maxconfig=iconfig
         endif

c$$$      write(*,*)'  '
c$$$      write(*,*)'  '
c$$$      write(*,*)'Enter graph number (iconfig)'
c$$$      read(5,*)iconfig
c$$$
c$$$

      softtest=.false.
      colltest=.false.

      call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &           ) .and. ntry .lt. 1000)
         call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
         calculatedBorn=.false.
         ntry=ntry+1
      enddo

      if (ntry.ge.1000) then
         write (*,*) 'No points passed cuts...'
         write (12,*) 'ERROR: no points passed cuts...'
     &        //' Could not perform ME tests properly',iconfig
      endif


      write (*,*) ''
      write (*,*) ''
      write (*,*) ''

      softtest=.true.
      colltest=.false.
      nerr=0
      imax=10
      do j=1,nsofttests
      call get_helicity(i_fks,j_fks)

         if(nsofttests.le.10)then
           write (*,*) ' '
           write (*,*) ' '
         endif

         xi_i_fks_ev=0.1d0
         ntry=1
         wgt=1d0
         call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
         calculatedBorn=.false.
         do while (( wgt .lt. 0 .or. p(0,1) .le. 0d0) .and. ntry .lt. 1000)
            wgt=1d0
            call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
            calculatedBorn=.false.
            ntry=ntry+1
         enddo
         if(nsofttests.le.10)write (*,*) 'ntry',ntry

c Call the Born to be sure that 'CalculatedBorn' is done correctly
         call sborn(p_born(0,1),wgt)
         call set_cms_stuff(0)
         call sreal(p1_cnt(0,1,0),zero,y_ij_fks_ev,fxl) 
         fxl=fxl*jac_cnt(0)

         call set_cms_stuff(-100)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
         limit(1)=fx*wgt
         wlimit(1)=wgt

         do k=1,nexternal
           do l=0,3
             lxp(l,k)=p1_cnt(l,k,0)
             xp(1,l,k)=p(l,k)
           enddo
         enddo
         do l=0,3
           lxp(l,nexternal+1)=p_i_fks_cnt(l,0)
           xp(1,l,nexternal+1)=p_i_fks_ev(l)
         enddo

         do i=2,imax
            xi_i_fks_ev=xi_i_fks_ev/10d0
            wgt=1d0
            call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
            do k=1,nexternal
              do l=0,3
                xp(i,l,k)=p(l,k)
              enddo
            enddo
            do l=0,3
              xp(i,l,nexternal+1)=p_i_fks_ev(l)
            enddo
         enddo

         if(nsofttests.le.10)then
           write (*,*) 'Soft limit:'
           do i=1,imax
              call xprintout(6,limit(i),fxl)
           enddo
c
           write(80,*)'  '
           write(80,*)'****************************'
           write(80,*)'  '
           do k=1,nexternal+1
             write(80,*)''
             write(80,*)'part:',k
             do l=0,3
               write(80,*)'comp:',l
               do i=1,10
                 call xprintout(80,xp(i,l,k),lxp(l,k))
               enddo
             enddo
           enddo
         else
           iflag=0
           call checkres(limit,fxl,wlimit,jac_cnt(0),xp,lxp,
     #                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
           nerr=nerr+iret
         endif

      enddo
      if(nsofttests.gt.10)then
        write(*,*)'Soft tests done for (real) config',iconfig
        write(*,*)'Failures:',nerr
        write(*,*)'Failures (fraction):',nerr/dfloat(nsofttests)
        if (nerr/dble(nsofttests).gt.0.4d0) then
           write (12,*) 'ERROR soft test ME failed',
     &          iconfig,nerr/dble(nsofttests)
        endif
      endif

      write (*,*) ''
      write (*,*) ''
      write (*,*) ''

c      write (*,*) pmass(-j_fks,iconfig)
      include 'pmass.inc'

      if (pmass(j_fks).ne.0d0) then
         write (*,*) 'No collinear test for massive j_fks'
         ncolltests=0
      endif

      softtest=.false.
      colltest=.true.

c Set rotategranny=.true. to align grandmother along the z axis, when 
c grandmother is not the c.m. system (if granny=cms, this rotation coincides
c with the identity, and the following is harmless).
c WARNING: the setting of rotategranny changes the definition of xij_aor
c in genps_fks_test.f
      rotategranny=.false.

      nerr=0
      imax=10
      do j=1,ncolltests
      call get_helicity(i_fks,j_fks)

         if(ncolltests.le.10)then
           write (*,*) ' '
           write (*,*) ' '
         endif

         y_ij_fks_ev=0.9d0
         ntry=1
         wgt=1d0
         call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
         calculatedBorn=.false.
         do while (( wgt .lt. 0 .or. p(0,1) .le. 0d0) .and. ntry .lt. 1000)
            wgt=1d0
            call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
            calculatedBorn=.false.
            ntry=ntry+1
         enddo
         if(ncolltests.le.10)write (*,*) 'ntry',ntry
         call set_cms_stuff(1)
         call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fxl) 
         fxl=fxl*jac_cnt(1)

         call set_cms_stuff(-100)
         call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
         limit(1)=fx*wgt
         wlimit(1)=wgt

         do k=1,nexternal
           do l=0,3
             lxp(l,k)=p1_cnt(l,k,1)
             xp(1,l,k)=p(l,k)
           enddo
         enddo
         do l=0,3
           lxp(l,nexternal+1)=p_i_fks_cnt(l,1)
           xp(1,l,nexternal+1)=p_i_fks_ev(l)
         enddo

         do i=2,imax
            y_ij_fks_ev=1-0.1d0**i
            wgt=1d0
            call x_to_f_arg(ndim,iconfig,minconfig,maxconfig,ninvar,wgt,x,p)
            calculatedBorn=.false.
            call set_cms_stuff(-100)
            call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx)
            limit(i)=fx*wgt
            wlimit(i)=wgt
            do k=1,nexternal
              do l=0,3
                xp(i,l,k)=p(l,k)
              enddo
            enddo
            do l=0,3
              xp(i,l,nexternal+1)=p_i_fks_ev(l)
            enddo
         enddo
         if(ncolltests.le.10)then
           write (*,*) 'Collinear limit:'
           do i=1,imax
              call xprintout(6,limit(i),fxl)
           enddo
c
           write(80,*)'  '
           write(80,*)'****************************'
           write(80,*)'  '
           do k=1,nexternal+1
             write(80,*)''
             write(80,*)'part:',k
             do l=0,3
               write(80,*)'comp:',l
               do i=1,10
                 call xprintout(80,xp(i,l,k),lxp(l,k))
               enddo
             enddo
           enddo
         else
           iflag=1
           call checkres(limit,fxl,wlimit,jac_cnt(1),xp,lxp,
     #                   iflag,imax,j,nexternal,i_fks,j_fks,iret)
           nerr=nerr+iret
         endif
      enddo
      if(ncolltests.gt.10)then
        write(*,*)'Collinear tests done for (real) config', iconfig
        write(*,*)'Failures:',nerr
        write(*,*)'Failures (fraction):',nerr/dfloat(ncolltests)
        if (nerr/dble(ncolltests).gt.0.4d0) then
           write (12,*) 'ERROR collinear test ME failed',
     &          iconfig,nerr/dble(ncolltests)
        endif
      endif

      enddo

      stop
      return
      end

c
c
c Dummy routines
c
c
      subroutine clear_events()
      end
      subroutine store_events()
      end
      integer function n_unwgted()
      n_unwgted = 1
      end

      subroutine outfun(pp,www)
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),www
c
      write(*,*)'This routine should not be called here'
      stop
      end
