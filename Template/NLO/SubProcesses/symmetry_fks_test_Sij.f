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
      
      double precision ZERO,one
      parameter       (ZERO = 0d0)
      parameter       (one = 1d0)
      integer   maxswitch
      parameter(maxswitch=99)
c
c     Local
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer itree(2,-max_branch:-1)
      integer imatch
      integer i,j, k, n, nsym,l,ii,jj
      double precision diff,xi_i_fks
c$$$      double precision pmass(-max_branch:-1,lmaxconfigs)   !Propagotor mass
      double precision pmass(nexternal)
      double precision pwidth(-max_branch:-1,lmaxconfigs)  !Propagotor width
      integer pow(-max_branch:-1,lmaxconfigs)

      integer biforest(2,-max_branch:-1,lmaxconfigs)
      integer fksmother,fksgrandmother,fksaunt,i_fks,j_fks,compare
      integer fksconfiguration,mapbconf(0:lmaxconfigs)
      integer r2b(lmaxconfigs),b2r(lmaxconfigs)
      logical searchforgranny,is_beta_cms,is_granny_sch,topdown
      integer nbranch,ns_channel,nt_channel
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      double precision fxl(10,10),limit(15,10,10),sumifx(15),check(15)
      double precision checkl,sumifxl
      double precision fks_Sij
      double precision tolerance,zh,h_damp,xmone,softlim,colllim
      parameter (tolerance=1.d-4)
      parameter (xmone=-1.d0)
      integer kk,ll,bs_min,bs_max,iconfig_in,bs

      integer nsofttests,ncolltests,imax,nerr,ierr,iflag,ilim,iret
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
      logical mtc, even,non_prop


      double precision xi_i_fks_fix_save,y_ij_fks_fix_save
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c
c     Global
c
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      include 'coupl.inc'

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn


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

      logical softtest,colltest
      common/sctests/softtest,colltest
      
      logical xexternal
      common /toxexternal/ xexternal

c
c     External
c
      logical check_swap
      double precision dsig,ran2
      external dsig,ran2
      external check_swap, fks_Sij

c      integer icomp
c
c     DATA
c
      integer tprid(-max_branch:-1,lmaxconfigs)
      include 'born_conf.inc'
c-----
c  Begin Code
c-----
      write(*,*)'Enter xi_i, y_ij to be used in coll/soft tests'
      write(*,*)' Enter -2 to generate them randomly'
      read(*,*)xi_i_fks_fix_save,y_ij_fks_fix_save

      write(*,*)'Enter number of tests for soft and collinear limits'
      read(*,*)nsofttests,ncolltests

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts 
c
c Read FKS configuration from file
      open (unit=61,file='config.fks',status='old')
      read(61,'(I2)',err=99,end=99) fksconfiguration
 99   close(61)
c Use the fks.inc include file to set i_fks and j_fks
      i_fks=fks_i(fksconfiguration)
      j_fks=fks_j(fksconfiguration)
      write (*,*) 'FKS configuration number is ',fksconfiguration
      write (*,*) 'FKS partons are: i=',i_fks,'  j=',j_fks


c
      ndim = 22
      ncall = 10000
      itmax = 10
      ninvar = 35
      nconfigs = 1
c     
c     Get momentum configuration
c

c Set xexternal to true to use the x's from external vegas in the
c x_to_f_arg subroutine
      xexternal=.true.
      
      write(*,*)'  '
      write(*,*)'  '
      write(*,*)'Enter graph number (iconfig), '
     &     //"'0' loops over all graphs"
      read(*,*)iconfig_in
      
      if (iconfig_in.eq.0) then
         bs_min=1
         bs_max=mapconfig(0)
      elseif (iconfig_in.eq.-1) then
         bs_min=1
         bs_max=1
      else
         bs_min=iconfig_in
         bs_max=iconfig_in
      endif

      do iconfig=bs_min,bs_max       ! Born configurations

      wgt=1d0
      ntry=1

      softtest=.false.
      colltest=.false.

      do jj=1,ndim
         x(jj)=ran2()
      enddo
      call generate_momenta(ndim,iconfig,wgt,x,p)
      calculatedBorn=.false.
      do while (( wgt.lt.0 .or. p(0,1).le.0d0 .or. p_born(0,1).le.0d0
     &           ) .and. ntry .lt. 1000)
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         calculatedBorn=.false.
         ntry=ntry+1
      enddo

      if (ntry.ge.1000) then
         write (*,*) 'No points passed cuts...'
         write (12,*) 'ERROR: no points passed cuts...'
     &        //' Cannot perform Sij tests properly for config',iconfig
         exit
      endif


      write (*,*) ''
      write (*,*) ''
      write (*,*) ''

      if(particle_type(i_fks).ne.8)goto 111

      nerr=0
      imax=10
      softtest=.true.
      colltest=.false.
      do j=1,nsofttests

         if(nsofttests.le.10)then
           write (*,*) ' '
           write (*,*) ' '
         endif

         y_ij_fks_fix=y_ij_fks_fix_save
         xi_i_fks_fix=0.1d0
         ntry=1
         wgt=1d0
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(nsofttests.le.10)write (*,*) 'ntry',ntry

         checkl=0.d0
         sumifxl=0.d0
         do i=1,imax
           sumifx(i)=0.d0
           check(i)=0.d0
         enddo

         ilim=1
         if(p1_cnt(0,1,0).le.0.d0)ilim=0

         do k=1,fks_configs
           kk = fks_i(k)
           ll = fks_j(k)
           call set_cms_stuff(0)
           fxl(kk,ll)=fks_Sij(p1_cnt(0,1,0),kk,ll,zero,y_ij_fks_ev)
           if(kk.eq.i_fks)sumifxl=sumifxl+fxl(kk,ll)
           checkl=checkl+fxl(kk,ll)

           call set_cms_stuff(-100)
           limit(1,kk,ll)=fks_Sij(p,kk,ll,xi_i_fks_ev,y_ij_fks_ev)
           if(kk.eq.i_fks)sumifx(1)=sumifx(1)+limit(1,kk,ll)
           check(1)=check(1)+limit(1,kk,ll)
         enddo

         do i=2,imax
            xi_i_fks_fix=xi_i_fks_fix/10d0
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.

            do k=1,fks_configs
              kk = fks_i(k)
              ll = fks_j(k)
              call set_cms_stuff(-100)
              limit(i,kk,ll)=fks_Sij(p,kk,ll,xi_i_fks_ev,y_ij_fks_ev)
              if(kk.eq.i_fks)sumifx(i)=sumifx(i)+limit(i,kk,ll)
              check(i)=check(i)+limit(i,kk,ll)
            enddo
         enddo

         if(nsofttests.le.10)then
           write (*,*) 'Soft limit:'
           do k=1,fks_configs
             kk = fks_i(k)
             ll = fks_j(k)
             write(*,*)'  '
             write(*,*)'k,kk,ll=',k,kk,ll
             do i=1,imax
               call xprintout(6,limit(i,kk,ll),fxl(kk,ll))
             enddo
           enddo
           write(*,*)'  '
           write(*,*)'sum over j, i=i_fks'
           do i=1,imax
             call xprintout(6,sumifx(i),sumifxl)
           enddo
           write(*,*)'  '
           write(*,*)'sum over all (check)'
           do i=1,imax
             call xprintout(6,check(i),checkl)
           enddo
         else
           iflag=0
           ierr=0
           do k=1,fks_configs
             kk = fks_i(k)
             ll = fks_j(k)
             if(kk.eq.i_fks)then
               softlim = xmone
             else
               softlim = zero
             endif
             call checksij(limit(1,kk,ll),fxl(kk,ll),softlim,
     #                     sumifx,sumifxl,one,
     #                     check,checkl,tolerance,
     #                     iflag,imax,j,k,kk,ll,
     #                     i_fks,j_fks,ilim,iret)
             ierr=ierr+iret
           enddo
           if(ierr.ne.0)nerr=nerr+1
         endif
        
      enddo
      if(nsofttests.gt.10)then
        write(*,*)'Soft tests done for (Born) config',iconfig
        write(*,*)'Failures:',nerr
        write(*,*)'Failures (fraction):',nerr/dfloat(nsofttests)
        if (nerr/dble(nsofttests).gt.0.4d0) then
           write (12,*) 'ERROR soft test Sij failed',
     &          iconfig,nerr/dble(nsofttests)
        endif
      endif

      write (*,*) ''
      write (*,*) ''
      write (*,*) ''

 111  continue

      include 'pmass.inc'

      if (pmass(j_fks).ne.0d0) then
         write (*,*) 'No collinear test for massive j_fks'
         ncolltests=0
      endif

      softtest=.false.
      colltest=.true.
      nerr=0
      imax=10
      do j=1,ncolltests

         if(ncolltests.le.10)then
           write (*,*) ' '
           write (*,*) ' '
         endif

         y_ij_fks_fix=0.9d0
         xi_i_fks_fix=xi_i_fks_fix_save
         ntry=1
         wgt=1d0
         do jj=1,ndim
            x(jj)=ran2()
         enddo
         call generate_momenta(ndim,iconfig,wgt,x,p)
         do while (( wgt.lt.0 .or. p(0,1).le.0d0) .and. ntry.lt.1000)
            wgt=1d0
            do jj=1,ndim
               x(jj)=ran2()
            enddo
            call generate_momenta(ndim,iconfig,wgt,x,p)
            ntry=ntry+1
         enddo
         if(ncolltests.le.10)write (*,*) 'ntry',ntry
         calculatedBorn=.false.

         checkl=0.d0
         do i=1,imax
           check(i)=0.d0
         enddo

         ilim=1
         if(p1_cnt(0,1,1).le.0.d0)ilim=0

         do k=1,fks_configs
           kk = fks_i(k)
           ll = fks_j(k)
           call set_cms_stuff(1)
           fxl(kk,ll)=fks_Sij(p1_cnt(0,1,1),kk,ll,xi_i_fks_cnt(1),one)
           checkl=checkl+fxl(kk,ll)

           call set_cms_stuff(-100)
           limit(1,kk,ll)=fks_Sij(p,kk,ll,xi_i_fks_ev,y_ij_fks_ev)
           check(1)=check(1)+limit(1,kk,ll)
         enddo

         do i=2,imax
            y_ij_fks_fix=1-0.1d0**i
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            calculatedBorn=.false.

            do k=1,fks_configs
              kk = fks_i(k)
              ll = fks_j(k)
              call set_cms_stuff(-100)
              limit(i,kk,ll)=fks_Sij(p,kk,ll,xi_i_fks_ev,y_ij_fks_ev)
              check(i)=check(i)+limit(i,kk,ll)
            enddo
         enddo
         if(ncolltests.le.10)then
           write (*,*) 'Collinear limit:'
           do k=1,fks_configs
             kk = fks_i(k)
             ll = fks_j(k)
             write(*,*)'  '
             write(*,*)'k,kk,ll=',k,kk,ll
             do i=1,imax
              call xprintout(6,limit(i,kk,ll),fxl(kk,ll))
             enddo
           enddo
           write(*,*)'  '
           write(*,*)'sum over all (check)'
           do i=1,imax
            call xprintout(6,check(i),checkl)
           enddo

           write(*,*)'  '
           if(particle_type(i_fks).eq.8.and.
     #        particle_type(j_fks).eq.8.and.
     #        j_fks.gt.nincoming)then
             zh=p1_cnt(0,i_fks,1)/(p1_cnt(0,i_fks,1)+p1_cnt(0,j_fks,1))
             write(*,*)'H(z)_limit = ',h_damp(zh)
           else
             write(*,*)'H(z) not used, limit must be 1'
           endif

         else
           iflag=1
           ierr=0
           do k=1,fks_configs
             kk = fks_i(k)
             ll = fks_j(k)
             if( (kk.eq.i_fks.and.ll.eq.j_fks) .or.
     #           (kk.eq.j_fks.and.ll.eq.i_fks) )then
               if(particle_type(kk).eq.8.and.
     #            particle_type(ll).eq.8.and.
     #            j_fks.gt.nincoming)then
                 zh=p1_cnt(0,kk,1)/
     #              (p1_cnt(0,kk,1)+p1_cnt(0,ll,1))
                 colllim = h_damp(zh)
               else
                 colllim = one
               endif
             else
               colllim = zero
             endif
             call checksij(limit(1,kk,ll),fxl(kk,ll),colllim,
     #                     sumifx,sumifxl,one,
     #                     check,checkl,tolerance,
     #                     iflag,imax,j,k,kk,ll,
     #                     i_fks,j_fks,ilim,iret)
             ierr=ierr+iret
           enddo
           if(ierr.ne.0)nerr=nerr+1
         endif
      enddo
      if(ncolltests.gt.10)then
        write(*,*)'Collinear tests done for (Born) config',iconfig
        write(*,*)'Failures:',nerr
        write(*,*)'Failures (fraction):',nerr/dfloat(ncolltests)
        if (nerr/dble(ncolltests).gt.0.4d0) then
           write (12,*) 'ERROR collinear test Sij failed',
     &          iconfig,nerr/dble(ncolltests)
        endif
      endif

      enddo

      return
      end



c
c
c Dummy routines
c
c
      subroutine clear_events()
      end
      subroutine initplot
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
