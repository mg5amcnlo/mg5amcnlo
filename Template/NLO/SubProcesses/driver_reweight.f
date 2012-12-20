      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
C
C     CONSTANTS
C
      double precision zero
      parameter       (ZERO = 0d0)
      include 'genps.inc'
      include 'nexternal.inc'
      INTEGER    ITMAX,   NCALL
      common/citmax/itmax,ncall
C
C     LOCAL
C
      integer i,j,l,l1,l2,ndim
      double precision dsig,tot,mean,sigma
      integer npoints
      double precision y,jac,s1,s2,xmin
      external dsig
      character*130 buf
c
c     Global
c
      integer                                      nsteps
      character*40          result_file,where_file
      common /sample_status/result_file,where_file,nsteps
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      real*8          dsigtot(10)
      common/to_dsig/ dsigtot
      integer ngroup
      common/to_group/ngroup
      data ngroup/0/
cc
      include 'run.inc'
      include 'coupl.inc'
      
      integer           iconfig
      common/to_configs/iconfig


      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw

c Vegas stuff
      integer ipole
      common/tosigint/ndim,ipole

      real*8 sigint,res,err,chi2a,res_abs
      external sigint

      integer irestart
      character * 70 idstring
      logical savegrid

      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file

      external initplot

      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint

c For tests
      real*8 fksmaxwgt,xisave,ysave
      common/cfksmaxwgt/fksmaxwgt,xisave,ysave

      integer itotalpoints
      common/ctotalpoints/itotalpoints

c For tests of virtuals
      double precision vobmax,vobmin
      common/cvirt0test/vobmax,vobmin
      double precision vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      common/cvirt1test/vNsumw,vAsumw,vSsumw,vNsumf,vAsumf,vSsumf
      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
      integer ivirtpoints,ivirtpointsExcept
      double precision  virtmax,virtmin,virtsum
      common/cvirt3test/virtmax,virtmin,virtsum,ivirtpoints,
     &     ivirtpointsExcept
      double precision total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min
      common/csum_of_wgts/total_wgt_sum,total_wgt_sum_max,
     &                 total_wgt_sum_min

      logical fixed_order
      double precision wgt2,wgt1,x(99),event_wgt,f_abs,sigintF,wgt
     &     ,max_wgt,www
      common /to_plot/wgt2
      integer nevents
      character*10 MonteCarlo
      integer lunlhe,lunlhe2,lunlhe3,lunlhe4
      parameter (lunlhe=98,lunlhe2=83,lunlhe3=84,lunlhe4=85)
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      character*10 dum
      logical plotEv,plotKin
      common/cEvKinplot/plotEv,plotKin
      logical putonshell,unwgt
      parameter (putonshell=.true.,unwgt=.true.)

      double precision ran2,rnd
      external ran2
      integer ifound
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      character*140 buff

      integer n_mp, n_disc
C-----
C  BEGIN CODE
C-----  
c
c     Read process number
c
      open (unit=lun+1,file='../dname.mg',status='unknown',err=11)
      read (lun+1,'(a130)',err=11,end=11) buf
      l1=index(buf,'P')
      l2=index(buf,'_')
      if(l1.ne.0.and.l2.ne.0.and.l1.lt.l2-1)
     $     read(buf(l1+1:l2-1),*,err=11) ngroup
 11   print *,'Process in group number ',ngroup

      lun = 27
      twgt = -2d0            !determine wgt after first iteration
      open(unit=lun,status='scratch')
      nsteps=2
      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
c     
c     Get user input
c
      write(*,*) "getting user params"
      call get_user_params(ncall,itmax,iconfig,irestart,idstring
     &     ,savegrid,fixed_order)
      if(irestart.eq.1)then
        flat_grid=.true.
      else
        flat_grid=.false.
      endif
c$$$      call setfksfactor(iconfig)
      ndim = 3*(nexternal-2)-4
      if (abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (abs(lpp(2)) .ge. 1) ndim=ndim+1

c Don't proceed if muF1#muF2 (we need to work out the relevant formulae
c at the NLO)
      if( ( fixed_fac_scale .and.
     #       (muF1_over_ref*muF1_ref_fixed) .ne.
     #       (muF2_over_ref*muF2_ref_fixed) ) .or.
     #    ( (.not.fixed_fac_scale) .and.
     #      muF1_over_ref.ne.muF2_over_ref ) )then
        write(*,*)'NLO computations require muF1=muF2'
        stop
      endif

      write(*,*) "about to integrate ", ndim,ncall,itmax,iconfig

      if(doVirtTest)then
        vobmax=-1.d8
        vobmin=1.d8
        vNsumw=0.d0
        vAsumw=0.d0
        vSsumw=0.d0
        vNsumf=0.d0
        vAsumf=0.d0
        vSsumf=0.d0
        nvtozero=0
        virtmax=-1d99
        virtmin=1d99
        virtsum=0d0
      endif
      itotalpoints=0
      ivirtpoints=0
      ivirtpointsExcept=0
      total_wgt_sum=0d0
      total_wgt_sum_max=0d0
      total_wgt_sum_min=0d0

      call addfil(dum)
      call initplot

      open (unit=lunlhe,file='events.lhe',status='old')
      call read_lhef_header(lunlhe,nevents,MonteCarlo)
      call read_lhef_init(lunlhe, IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP
     &     ,NPRUP,XSECUP,XERRUP,XMAXUP,LPRUP)
      wgt2=XSECUP/dble(nevents)

      if (nevents.lt.ncall*itmax) then
         write (*,*) 'Not enough Born events in event file'
         stop
      endif

      if (.not. fixed_order) then
         open(unit=lunlhe2,file='events_NLO.lhe',status='unknown')
      endif

      call fill_MC_mshell()
      res=0d0
      res_abs=0d0
      err=0d0
      max_wgt=1d0
      do i=1,itmax
         do j=1,ncall
            call get_random_numbers(lunlhe,x,wgt1)
            wgt=wgt2/wgt1
            if (fixed_order) then
               event_wgt=sigint(x,wgt)
            else
               event_wgt=sigintF(x,wgt,0,f_abs)
               event_wgt=sigintF(x,wgt,2,f_abs)
               call finalize_event(x,abs(event_wgt),lunlhe2,plotEv
     &              ,putonshell)
            endif
            www=event_wgt/wgt2
            
            write (*,*) 'BBBB weight',i,j,event_wgt,wgt2,www

            if (abs(www).gt.20d0) then
               write (*,*) 'large weight found', www,' xi_i:',x(5)**2
     &              ,' y_ij:',-2*x(6)**2+1
c               call regrid_MC_integer
c               stop
            endif

            if (www.gt.49.5d0) then
               www=49.5d0
            elseif(www.lt.-49.5d0) then
               www=-49.5d0
            endif
            call mfill(2,www,1d0)
            call mfill(3,www,1d0)
            if (abs(event_wgt/wgt2).gt.max_wgt) max_wgt=max_wgt*1.1d0
            res=res+event_wgt
            res_abs=res_abs+f_abs
            err=err+event_wgt**2
         enddo
         call regrid_MC_integer
      enddo
c$$$      max_wgt=20d0
      max_wgt=max_wgt*wgt2
      write (*,*) 'max_wgt=',max_wgt,max_wgt/wgt2
      err=sqrt(err)

      close (lunlhe)
      if (unwgt) then
         rewind(lunlhe2)
         open(unit=lunlhe3,status='unknown')
         ifound=0
         do i=1,itmax*ncall
            call read_lhef_event(lunlhe2,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
            rnd=ran2()
            if (abs(XWGTUP).gt.rnd*max_wgt) then
c               XWGTUP=sign(res_abs/dble(itmax*ncall),XWGTUP)
               ifound=ifound+1
               call write_lhef_event(lunlhe3,
     &              NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
            endif
         enddo
         write (*,*) 'unweighting efficiency',
     &        dble(ifound)/dble(itmax*ncall)
         rewind(lunlhe3)
         open(unit=lunlhe4,file='unweighted_events_NLO.lhe',status
     &        ='unknown')
         call write_header_init(lunlhe4,ifound,res,res_abs,err)
         do i=1,ifound
            call read_lhef_event(lunlhe3,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
            XWGTUP=sign(res_abs/dble(ifound),XWGTUP)
            call write_lhef_event(lunlhe4,
     &           NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &           IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         enddo
         close (lunlhe3)
         write (lunlhe4,'(a)') "</LesHouchesEvents>"
         close (lunlhe4)
      endif
      close (lunlhe2)

      write (*,*) 'number of events',ifound

      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      write(*,*)'Final result:',res,'+/-',err
      write(*,*)'Final result ABS:',res_abs,'+/-',err
      write(*,*)'Maximum weight found:',fksmaxwgt
      write(*,*)'Found for:',xisave,ysave
      write (*,*) '----------------------------------------------------'
      write (*,*) ''

      if(doVirtTest)then
        write(*,*)'  '
        write(*,*)'Statistics for virtuals'
        write(*,*)'max[V/(as/(2*pi)B)]:',vobmax
        write(*,*)'min[V/(as/(2*pi)B)]:',vobmin
        if(vNsumw.ne.0.d0)then
          vAsumw=vAsumw/vNsumw
          vSsumw=vSsumw/vNsumw
          write(*,*)'Weighted:'
          write(*,*)'  average=',vAsumw
          if(vSsumw.lt.(vAsumw**2*0.9999d0))then
            write(*,*)'Error in sigma',vSsumw,vAsumw
          else
            write(*,*)'  std dev=',sqrt(abs(vSsumw-vAsumw**2))
          endif
        else
          write(*,*)'Sum of weights [virt_w] is zero'
        endif
c
        if(vNsumf.ne.0.d0)then
          vAsumf=vAsumf/vNsumf
          vSsumf=vSsumf/vNsumf
          write(*,*)'Flat:'
          write(*,*)'  average=',vAsumf
          if(vSsumf.lt.(vAsumf**2*0.9999d0))then
            write(*,*)'Error in sigma',vSsumf,vAsumf
          else
            write(*,*)'  std dev=',sqrt(abs(vSsumf-vAsumf**2))
          endif
        else
          write(*,*)'Sum of weights [virt_f] is zero'
        endif
c
        if(nvtozero.ne.0)then
          write(*,*)
     &          '# of points (passing cuts) with Born=0 and virt=0:',
     &          nvtozero
        endif
        write (*,*) 'virtual weights directly from BinothLHA.f:'
        if (ivirtpoints.ne.0) then
           write (*,*) 'max(virtual/Born/ao2pi)= ',virtmax
           write (*,*) 'min(virtual/Born/ao2pi)= ',virtmin
           write (*,*) 'avg(virtual/Born/ao2pi)= ',
     &          virtsum/dble(ivirtpoints)
        endif
      endif

      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      if (irestart.eq.1 .or. irestart.eq.3) then
         write (*,*) 'Total points tried:                   ',
     &        ncall*itmax
         write (*,*) 'Total points passing generation cuts: ',
     &        itotalpoints
         write (*,*) 'Efficiency of events passing cuts:    ',
     &        dble(itotalpoints)/dble(ncall*itmax)
      else
         write (*,*)
     &       'Run has been restarted, next line is only for current run'
         write (*,*) 'Total points passing cuts: ',itotalpoints
      endif
      write (*,*) '----------------------------------------------------'
      write (*,*) ''
      write (*,*) ''
      write (*,*) '----------------------------------------------------'
      write (*,*) 'number of except PS points:',ivirtpointsExcept,
     &     'out of',ivirtpoints,'points'
      write (*,*) '   treatment of exceptional PS points:'
      write (*,*) '      maximum approximation:',
     &     total_wgt_sum + dsqrt(total_wgt_sum_max)
      write (*,*) '      minimum approximation:',
     &     total_wgt_sum - dsqrt(total_wgt_sum_min)
      write (*,*) '      taking the max/min average:',total_wgt_sum
      write (*,*) '----------------------------------------------------'
      write (*,*) ''

c Uncomment for getting CutTools statistics
c$$$      call ctsstatistics(n_mp,n_disc)
c$$$      write(*,*) 'n_mp  =',n_mp,'    n_disc=',n_disc

      call mclear
      open(unit=99,file='MADatNLO.top',status='unknown')
      call topout
      close(99)

      end


      function sigint(xx,peso)
c From dsample_fks
      implicit none
      include 'nexternal.inc'
      real*8 sigint,peso,xx(58)
      integer ione
      parameter (ione=1)
      integer ndim
      common/tosigint/ndim
      integer           iconfig
      common/to_configs/iconfig
      integer i
      double precision wgt,dsig,ran2,rnd
      external ran2
      double precision x(99),p(0:3,nexternal)
      include 'nFKSconfigs.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      character*4 abrv
      common /to_abrv/ abrv
      logical nbody
      common/cnbody/nbody
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical sum,firsttime
      parameter (sum=.false.)
      data firsttime /.true./
      integer nFKSprocessBorn
      save nFKSprocessBorn
      double precision vol,sigintR

      double precision wgt1,wgt2
      save wgt2
      integer lunlhe
c
      do i=1,99
         if(i.le.ndim)then
            x(i)=xx(i)
         else
            x(i)=0.d0
         endif
      enddo

      sigint=0d0

c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
         nFKSprocess=fks_configs
         call fks_inc_chooser()
         do while (particle_type(i_fks).ne.8)
            nFKSprocess=nFKSprocess-1
            call fks_inc_chooser()
            if (nFKSprocess.eq.0) then
               write (*,*) 'ERROR in sigint'
               stop
            endif
         enddo
         nFKSprocessBorn=nFKSprocess
         write (*,*) 'Total number of FKS directories is', fks_configs
         write (*,*) 'For the Born we use nFKSprocess  #', nFKSprocess
      endif
         
c
c Compute the Born-like contributions with nbody=.true.
c THIS CAN BE OPTIMIZED
c
      nFKSprocess=nFKSprocessBorn
      nbody=.true.
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setcuts
      call setfksfactor(iconfig)
      wgt=peso
      call generate_momenta(ndim,iconfig,wgt,x,p)
      sigint = sigint+dsig(p,wgt,1d0)
c
c Compute the subtracted real-emission corrections either as an explicit
c sum or a Monte Carlo sum.
c      
      if (abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &     abrv(1:2).eq.'vi') return
      nbody=.false.
      if (sum) then
c THIS CAN BE OPTIMIZED
         do nFKSprocess=1,fks_configs
            call fks_inc_chooser()
            call leshouche_inc_chooser()
            call setcuts
            call setfksfactor(iconfig)
            wgt=peso
            call generate_momenta(ndim,iconfig,wgt,x,p)
            sigint = sigint+dsig(p,wgt,1d0)
         enddo
      else ! Monte Carlo over nFKSprocess
         call get_MC_integer(fks_configs,nFKSprocess,vol)
c THIS CAN BE OPTIMIZED
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         call setcuts
         call setfksfactor(iconfig)
c     The variable 'vol' is the size of the cell for the MC over
c     nFKSprocess. Need to divide by it here to correctly take into
c     account this Jacobian
         wgt=peso/vol
         call generate_momenta(ndim,iconfig,wgt,x,p)
         sigintR = dsig(p,wgt,1d0)
         call fill_MC_integer(nFKSprocess,abs(sigintR)*vol)
         sigint = sigint+ sigintR
      endif
      return
      end


      function sigintF(xx,w,ifl,f_abs)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer i,j
      integer ifl
      integer fold
      common /cfl/fold
      include 'mint.inc'
      real*8 sigintF,xx(ndimmax),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigS,dsigH,f_abs,f(2)
      include 'nexternal.inc'
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      logical Hevents
      common/SHevents/Hevents
      double precision result1,result2,ran2,rnd
      external ran2
      double precision p(0:3,nexternal)
      double precision sigintF_save,f_abs_save
      save sigintF_save,f_abs_save
      double precision x(99),sigintF_without_w,f_abs_without_w
      common /c_sigint/ x,sigintF_without_w,f_abs_without_w
      include 'nFKSconfigs.inc'
      double precision result(0:fks_configs,2)
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      character*4 abrv
      common /to_abrv/ abrv
      logical nbody
      common/cnbody/nbody
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical firsttime
      integer sum
      parameter (sum=1)
      data firsttime /.true./
      integer nFKSprocessBorn
      save nFKSprocessBorn
      double precision vol,sigintR,res,f_tot
      integer proc_map(fks_configs,0:fks_configs),tot_proc
     $     ,i_fks_proc(fks_configs),j_fks_proc(fks_configs)
     $     ,nFKSproc_m
      logical found
      double precision ev_wgt_ratio, born_wgt_ratio
      common /weigths_ev_born/ev_wgt_ratio, born_wgt_ratio
c
      ev_wgt_ratio=0.d0
      born_wgt_ratio=0.d0

      do i=1,99
         if (i.le.ndim) then
            x(i)=xx(i)
         else
            x(i)=-9d99
         endif
      enddo

c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
         nFKSprocess=fks_configs
         call fks_inc_chooser()
         do while (particle_type(i_fks).ne.8)
            write (*,*) i_fks,particle_type(i_fks)
            nFKSprocess=nFKSprocess-1
            call fks_inc_chooser()
            if (nFKSprocess.eq.0) then
               write (*,*) 'ERROR in sigint'
               stop
            endif
         enddo
         nFKSprocessBorn=nFKSprocess
         write (*,*) 'Total number of FKS directories is', fks_configs
         write (*,*) 'For the Born we use nFKSprocess  #', nFKSprocess
c For sum over identical FKS pairs, need to find the identical structures
         if (sum.eq.2) then
            tot_proc=0
            do i=1,fks_configs
               proc_map(i,0)=0
               i_fks_proc(i)=0
               j_fks_proc(i)=0
            enddo
            do nFKSprocess=1,fks_configs
               call fks_inc_chooser()
               found=.false.
               i=1
               do while ( i.le.tot_proc )
                  if (i_fks.eq.i_fks_proc(i)
     &              .and. j_fks.eq.j_fks_proc(i) ) then
                     exit
                  endif
                  i=i+1
               enddo
               proc_map(i,0)=proc_map(i,0)+1
               proc_map(i,proc_map(i,0))=nFKSprocess
               if (i.gt.tot_proc) then
                  tot_proc=tot_proc+1
                  i_fks_proc(tot_proc)=i_fks
                  j_fks_proc(tot_proc)=j_fks
               endif
            enddo
            write (*,*) 'FKS process map:'
            do i=1,tot_proc
               write (*,*) i,'-->',proc_map(i,0),':',
     &              (proc_map(i,j),j=1,proc_map(i,0))
            enddo
         endif
      endif

      f(1)=0d0
      f(2)=0d0
      fold=ifl
      if (ifl.eq.0)then
c
c Compute the Born-like contributions with nbody=.true.
c     
         nFKSprocess=nFKSprocessBorn
         nbody=.true.
         call fks_inc_chooser()
         call leshouche_inc_chooser()
c THIS CAN BE OPTIMIZED
         call setcuts
         call setfksfactor(iconfig)
         wgt=w
         call generate_momenta(ndim,iconfig,wgt,x,p)
         call dsigF(p,wgt,1d0,dsigS,dsigH)
         result(0,1)= dsigS
         result(0,2)= dsigH
         f(1) = f(1)+result(0,1)
         f(2) = f(2)+result(0,2)
c
c Compute the subtracted real-emission corrections either as an explicit
c sum or a Monte Carlo sum.
c      
         do i=1,fks_configs
            result(i,1)=0d0
            result(i,2)=0d0
         enddo

         if (.not.( abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &        abrv(1:2).eq.'vi') ) then
            nbody=.false.
            if (sum.eq.1) then

               do i=1,100
                  
               x(ndim)=ran2()
               x(ndim-1)=ran2()
               x(ndim-2)=ran2()



               do nFKSprocess=1,fks_configs
                  call fks_inc_chooser()
                  call leshouche_inc_chooser()
c THIS CAN BE OPTIMIZED
                  call setcuts
                  call setfksfactor(iconfig)
                  wgt=w
                  call generate_momenta(ndim,iconfig,wgt,x,p)
                  call dsigF(p,wgt,1d0,dsigS,dsigH)
                  result(nFKSprocess,1)= dsigS
                  result(nFKSprocess,2)= dsigH
c$$$                  f(1) = f(1)+result(nFKSprocess,1)
c$$$                  f(2) = f(2)+result(nFKSprocess,2)

c$$$               f(1) = f(1)+abs(result(nFKSprocess,1))
c$$$               f(2) = f(2)+abs(result(nFKSprocess,2))
               f(1) = f(1)+abs(result(nFKSprocess,1)/100d0)
               f(2) = f(2)+abs(result(nFKSprocess,2)/100d0)

               enddo
               nFKSprocess=5

c$$$               write (*,'(i3,5(x,e9.3),x,i3)')i,f(1),f(2)
c$$$     &              ,ev_wgt_ratio,born_wgt_ratio,ev_wgt_ratio
c$$$     &              /born_wgt_ratio,nFKSprocess
               enddo


            elseif(sum.eq.0) then    ! Monte Carlo over nFKSprocess
               call get_MC_integer(fks_configs,nFKSprocess,vol)
               call fks_inc_chooser()
               call leshouche_inc_chooser()
c THIS CAN BE OPTIMIZED
               call setcuts
               call setfksfactor(iconfig)
               wgt=w/vol
               call generate_momenta(ndim,iconfig,wgt,x,p)
               call dsigF(p,wgt,1d0,dsigS,dsigH)
               sigintR = (abs(dsigS)+abs(dsigH))*vol
               call fill_MC_integer(nFKSprocess,sigintR)
               result(nFKSprocess,1)= dsigS
               result(nFKSprocess,2)= dsigH
               f(1) = f(1)+result(nFKSprocess,1)
               f(2) = f(2)+result(nFKSprocess,2)
            elseif(sum.eq.2) then    ! MC over i_fks/j_fks pairs
               call get_MC_integer(tot_proc,nFKSproc_m,vol)
               do i=1,proc_map(nFKSproc_m,0)
                  nFKSprocess=proc_map(nFKSproc_m,i)
                  call fks_inc_chooser()
                  call leshouche_inc_chooser()
c THIS CAN BE OPTIMIZED
                  call setcuts
                  call setfksfactor(iconfig)
                  wgt=w/vol
                  call generate_momenta(ndim,iconfig,wgt,x,p)
                  call dsigF(p,wgt,1d0,dsigS,dsigH)
                  result(nFKSprocess,1)= dsigS
                  result(nFKSprocess,2)= dsigH
                  f(1) = f(1)+result(nFKSprocess,1)
                  f(2) = f(2)+result(nFKSprocess,2)
               enddo
               sigintR=0d0
               do i=1,proc_map(nFKSproc_m,0)
                  sigintR=sigintR+
     &               (abs(result(proc_map(nFKSproc_m,i),1))+
     &                abs(result(proc_map(nFKSproc_m,i),2)))*vol
               enddo
               call fill_MC_integer(nFKSproc_m,sigintR)
            endif
         endif
         sigintF=f(1)+f(2)
         f_abs=abs(f(1))+abs(f(2))

         
c$$$         write (*,*)'BBBBBBBB'
         if (born_wgt_ratio.eq.0d0) then
            sigintF=0d0
            f_abs=0d0
c$$$         elseif (ev_wgt_ratio/born_wgt_ratio.gt.5d0) then
c$$$            sigintF=0d0
c$$$            f_abs=0d0
         endif

         sigintF_save=sigintF
         f_abs_save=f_abs

      elseif(ifl.eq.1) then
         write (*,*) 'Folding not implemented'
         stop
      elseif(ifl.eq.2) then
         sigintF = sigintF_save
         sigintF_without_w=sigintF/w
         f_abs = f_abs_save
         f_abs_without_w=f_abs/w
c Determine if we need to write S or H events according to their
c relative weights
         if (f_abs.gt.0d0) then
            if (sum.eq.1) then
c$$$               write (*,*) 'event generation for "sum"'/
c$$$     $              /' not yet implemented',sum
            elseif (sum.eq.0 .or. sum.eq.2) then
               result1=result(0,1)+result(nFKSprocess,1)
               result2=result(0,2)+result(nFKSprocess,2)
               if (ran2().le.abs(result1)/f_abs) then
                  Hevents=.false.
                  evtsgn=sign(1d0,result1)
                  j=1
               else
                  Hevents=.true.
                  evtsgn=sign(1d0,result2)
                  j=2
               endif
               if (sum.eq.2 .and. .not.( abrv.eq.'born' .or. abrv.eq
     &              .'grid' .or.abrv(1:2).eq.'vi') ) then
                  f_tot=result(0,j)
                  do i=1,proc_map(nFKSproc_m,0)
                     f_tot=f_tot+result(proc_map(nFKSproc_m,i),j)
                  enddo
                  rnd=ran2()
                  i=1
                  res=abs(result(0,j))+
     &                 abs(result(proc_map(nFKSproc_m,1),j))
                  do while (res.lt.rnd*f_tot)
                     i=i+1
                     res=res+abs(result(proc_map(nFKSproc_m,i),j))
                  enddo
                  nFKSprocess=proc_map(nFKSproc_m,i)
                  call fks_inc_chooser()
                  call leshouche_inc_chooser()
c THIS CAN BE OPTIMIZED
                  call setcuts
                  call setfksfactor(iconfig)
               endif
            endif
         endif
      endif
      return
      end

     


c
      subroutine get_user_params(ncall,itmax,iconfig,
     #                           irestart,idstring,savegrid,fixed_order)
c**********************************************************************
c     Routine to get user specified parameters for run
c**********************************************************************
      implicit none
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
c
c     Arguments
c
      integer ncall,itmax,iconfig
c
c     Local
c
      integer i, j
      double precision dconfig
c
c     Global
c
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      double precision    accur
      common /to_accuracy/accur
      integer           use_cut
      common /to_weight/use_cut

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      character*5 abrvinput
      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody

      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
c
c To convert diagram number to configuration
c
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      integer mapconfig(0:lmaxconfigs)
      include 'born_conf.inc'
c
c Vegas stuff
c
      integer irestart,itmp
      character * 70 idstring
      logical savegrid

      character * 80 runstr
      common/runstr/runstr
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
c
c MC counterterm stuff
c
c alsf and besf are the parameters that control gfunsoft
      double precision alsf,besf
      common/cgfunsfp/alsf,besf
c alazi and beazi are the parameters that control gfunazi
      double precision alazi,beazi
      common/cgfunazi/alazi,beazi
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
c Set plotKin=.true. to plot H and S event kinematics (integration steps)
c Set plotEv=.true. to use events for plotting (unweighting phase)
      logical plotEv,plotKin
      common/cEvKinplot/plotEv,plotKin
      logical fixed_order

c-----
c  Begin Code
c-----
      doVirtTest=.true.
      mint=.false.
      unwgt=.false.
      usexinteg=.false.
      plotEV=.false.
      plotKin=.true.
      use_cut=2
      irestart=1
      write(*,'(a)') 'Enter number of events and iterations: '
      read(*,*) ncall,itmax
      write(*,*) 'Number of events and iterations ',ncall,itmax
      write (*,*) 'Fixed order "0" or matching to parton shower "1"?'
      read(*,*) i
      if (i .eq. 0) then
         fixed_order = .true.
         write(*,*) 'Fixed order'
      else
         fixed_order = .false.
         write(*,*) 'Matching to parton shower'
      endif
      
      write(*,*)'Enter alpha, beta for G_soft'
      write(*,*)'  Enter alpha<0 to set G_soft=1 (no ME soft)'
      read(*,*)alsf,besf
      write (*,*) 'for G_soft: alpha=',alsf,', beta=',besf 
      write(*,*)'Enter alpha, beta for G_azi'
      write(*,*)'  Enter alpha>0 to set G_azi=0 (no azi corr)'
      read(*,*)alazi,beazi
      write (*,*) 'for G_azi: alpha=',alazi,', beta=',beazi

      write(*,10) 'Suppress amplitude (0 no, 1 yes)? '
      read(*,*) i
      if (i .eq. 1) then
         multi_channel = .true.
         write(*,*) 'Using suppressed amplitude.'
      else
         multi_channel = .false.
         write(*,*) 'Using full amplitude.'
      endif

      write(*,10) 'Exact helicity sum (0 yes, n = number/event)? '
      read(*,*) i
      if (i .eq. 0) then
         isum_hel = 0
         write(*,*) 'Explicitly summing over helicities'
      else
         isum_hel= i
         write(*,*) 'Summing over',i,' helicities/event'
      endif

      write(*,10) 'Enter Configuration Number: '
      read(*,*) dconfig
      iconfig = int(dconfig)
      do i=1,mapconfig(0)
         if (iconfig.eq.mapconfig(i)) then
            iconfig=i
            exit
         endif
      enddo
      write(*,12) 'Running Configuration Number: ',iconfig

      write(*,*)'enter 1 if you want restart files'
      read (*,*) itmp
      if(itmp.eq.1) then
         savegrid = .true.
      else
         savegrid = .false.
      endif

      write (*,'(a)') 'Monte Carlo for showering. Possible options are'
      write (*,'(a)') 'HERWIG6, HERWIGPP, PYTHIA6Q, PYTHIA6PT, PYTHIA8'
      read (*,*) MonteCarlo
      write (*,*) MonteCarlo

      if(MonteCarlo.ne.'HERWIG6' .and.MonteCarlo.ne.'HERWIGPP' .and.
     #   MonteCarlo.ne.'PYTHIA6Q'.and.MonteCarlo.ne.'PYTHIA6PT'.and.
     #   MonteCarlo.ne.'PYTHIA8')then
        write(*,*)'Take it easy. This MC is not implemented'
        stop
      endif

      abrvinput='     '
      write (*,*) "'all ', 'born', 'real', 'virt', 'novi' or 'grid'?"
      write (*,*) "Enter 'born0' or 'virt0' to perform"
      write (*,*) " a pure n-body integration (no S functions)"
      read(5,*) abrvinput
      if(abrvinput(5:5).eq.'0')then
         write (*,*) 'This is not supported'
         stop
      else
        nbody=.false.
      endif
      abrv=abrvinput(1:4)
c Options are way too many: make sure we understand all of them
      if ( abrv.ne.'all '.and.abrv.ne.'born'.and.abrv.ne.'real'.and.
     &     abrv.ne.'virt'.and.abrv.ne.'novi'.and.abrv.ne.'grid'.and.
     &     abrv.ne.'viSC'.and.abrv.ne.'viLC'.and.abrv.ne.'novA'.and.
     &     abrv.ne.'novB'.and.abrv.ne.'viSA'.and.abrv.ne.'viSB') then
        write(*,*)'Error in input: abrv is:',abrv
        stop
      endif
      if(nbody.and.abrv.ne.'born'.and.abrv(1:2).ne.'vi'
     &     .and. abrv.ne.'grid')then
        write(*,*)'Error in driver: inconsistent input',abrvinput
        stop
      endif

      write (*,*) "doing the ",abrv," of this channel"
      if(nbody)then
        write (*,*) "integration Born/virtual with Sfunction=1"
      else
        write (*,*) "Normal integration (Sfunction != 1)"
      endif

      doVirtTest=doVirtTest.and.abrv(1:2).eq.'vi'
c
c
c     Here I want to set up with B.W. we map and which we don't
c
c$$$      dconfig = dconfig-iconfig
c$$$      if (dconfig .eq. 0) then
c$$$         write(*,*) 'Not subdividing B.W.'
c$$$         lbw(0)=0
c$$$      else
c$$$         lbw(0)=1
c$$$         jconfig=dconfig*1000.1
c$$$         write(*,*) 'Using dconfig=',jconfig
c$$$         call DeCode(jconfig,lbw(1),3,nexternal)
c$$$         write(*,*) 'BW Setting ', (lbw(j),j=1,nexternal-2)
c$$$      endif
 10   format( a)
 12   format( a,i4)
      end
c


c Dummy subroutine (normally used with vegas when resuming plots)
      subroutine resume()
      end



      subroutine get_random_numbers(lunlhe,x,wgt)
      implicit none
      double precision x(99),wgt,wgt_abs
      character*20 buff
      integer lunlhe,ndim,i
      read(lunlhe,'(a)') buff
      if (buff(1:10).ne.'  <event>') then
         write (*,*) 'not a new event #1'
         stop
      endif
      read (lunlhe,*) ndim,wgt,wgt_abs
      if (wgt.ne.wgt_abs) then
         write (*,*) 'weights not equal'
         stop
      endif
      read(lunlhe,*)(x(i),i=1,ndim)
      read(lunlhe,'(a)') buff
      if (buff(1:11).ne.'  </event>') then
         write (*,*) 'not a new event #2'
         stop
      endif
      return
      end
