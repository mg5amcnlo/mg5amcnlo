! with fastjet:
      ! gfortran -o plot_LHE HwU.f plot_LHE.f analysis_HwU_scales.f fjcore.cc fastjetfortran_madfks_core.cc -lstdc++

! without fastjet:
      ! gfortran -o plot_LHE HwU.f plot_LHE.f analysis_HwU_scales.f


      program plot_LHE
      use extra_weights
      implicit none
      include 'nexternal.inc'
      integer ifile,i,j,maxevt
      character*10 MonteCarlo
      logical done
      character*140 filename
      character*50 weights_info(10)
      double precision dummy
      character*1000 buff
c Les Houches Event File info:
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP)
     $     ,ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP(5,MAXNUP)
     $     ,VTIMUP(MAXNUP),SPINUP(MAXNUP)
      DOUBLE PRECISION SCALUP_a(MAXNUP,MAXNUP)
      double precision xwgt(1:3),xwgt_up(1:3),p_up(0:3,nexternal,1:3)
      integer id(1:nexternal)
      write (*,*) 'Give LHE file name'
      read (*,'(a)') filename
      ifile=11
      open(unit=ifile,file=filename,status='OLD')
      
      call read_lhef_header(ifile,maxevt,MonteCarlo)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
c$$$      do
c$$$         read(ifile,*) buff
c$$$         if (index('<event',buff).ne.0) then
c$$$            backspace(ifile
c$$$            exit
c$$$         endif
c$$$      enddo
      
      weights_info(1)="central value               "
      
      call set_error_estimation(0)

      call analysis_begin(1,weights_info)
      do i=1,maxevt
         call read_lhef_event(ifile,NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP
     $        ,AQCDUP,IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff
     $        ,scalup_a)
         if (done) exit
         call fill_wgt_info_from_rwgt_lines
         call recompute_contributions(p_up,xwgt,id)
c$$$         write (*,*) i,sum(xwgt(1:3)),wgtref
         xwgt_up(1:3)=xwgt(1:3)*XWGTUP/wgtref
         istup(1:nincoming)=-1
         istup(nincoming+1:nexternal)=1
         do j=1,3               ! n-body and (n+1)-body
            if (xwgt_up(j).eq.0d0) cycle
c$$$            write (*,*) i,j,xwgt_up(j)
            call plot_event(nexternal,XWGT_UP(j),ID,ISTUP,P_UP(0,1,j)
     $           ,SCALUP,scalup_a,j)
         enddo
         call HwU_add_points
      enddo
      close (ifile)
      
      call finalize_histograms(maxevt)
      call analysis_end(dummy)

      return
      end

      subroutine recompute_contributions(p_up,xwgt,id)
      use weight_lines
      implicit none
      include 'nexternal.inc'
      integer i
      double precision xwgt(1:3),p_up(0:3,nexternal,1:3)
      integer id(1:nexternal)
      p_up(0,1,1:3)=-1d0
      xwgt(1:3)=0
      do i=1,icontr
         if (itype(i).eq.1) then
! real emission (n+1)-body kinematics
            xwgt(1)=xwgt(1)+wgts(1,i)/damp(i)
            if (p_up(0,1,1).eq.-1d0) p_up(0:3,1:nexternal,1)=
     $           momenta_m(0:3,1:nexternal,2,i)
         elseif (itype(i).eq.11) then
! real emission n-body kinematics
c$$$            xwgt(2)=xwgt(2)+wgts(1,i)
            if (p_up(0,1,2).eq.-1d0) p_up(0:3,1:nexternal,2)=
     $           momenta_m(0:3,1:nexternal,1,i)
         elseif (itype(i).eq.2) then
! Born
            xwgt(3)=xwgt(3)+wgts(1,i)
            if (p_up(0,1,3).eq.-1d0) p_up(0:3,1:nexternal,3)=
     $           momenta_m(0:3,1:nexternal,1,i)
         else
! anything else
            xwgt(2)=xwgt(2)+wgts(1,i)
            if (p_up(0,1,2).eq.-1d0) p_up(0:3,1:nexternal,2)=
     $           momenta_m(0:3,1:nexternal,1,i)
         endif
      enddo
      id(1:nexternal)=pdg(1:nexternal,1)
      end

      
      subroutine plot_event(NUP,XWGTUP,IDUP,ISTUP,PUP,SCALUP,scalup_a
     $     ,ibody)
      implicit none
      INTEGER NUP,IDUP(*),ISTUP(*)
      DOUBLE PRECISION XWGTUP,PUP(0:3,*)
      integer nex,i,j
      parameter (nex=10)
      integer nexternal,istatus(nex),ipdg(nex),idummy,ibody
      double precision p(0:4,nex),wgts(10),scalup,scalup_a(10,10)
      wgts(1)=XWGTUP
      nexternal=nup
      do i=1,nup
         p(0:3,i)=pup(0:3,i)
         p(4,i)=-1d0
         istatus(i)=ISTUP(i)
         ipdg(i)=IDUP(i)
      enddo
      call analysis_fill(p,istatus,ipdg,wgts,ibody)
      return
      end
     
      subroutine HwU_write_file
      implicit none
      double precision xnorm
      open (unit=99,file='LHEF.HwU',status='unknown')
      xnorm=1d0
      call HwU_output(99,xnorm)
      close (99)
      return
      end
      

      subroutine fill_wgt_info_from_rwgt_lines
      use weight_lines
      use extra_weights
      implicit none
      include 'nexternal.inc'
      integer i,idum,j,k,momenta_conf(2),ii,n_proc
      icontr=n_ctr_found
      iwgt=1
      n_proc=1
      call weight_lines_allocated(nexternal,icontr,iwgt,n_proc)
      do i=1,icontr
         read(n_ctr_str(i),*)(wgt(j,i),j=1,3),(wgt_ME_tree(j,i),j=1,2)
     $        ,idum,(pdg(j,i),j=1,nexternal),orderstag(i),QCDpower(i)
     $        ,(bjx(j,i),j=1 ,2),(scales2(j,i),j=1,3),g_strong(i)
     $        ,(momenta_conf(j),j=1 ,2),itype(i),nFKS(i),idum,idum,idum
     $        ,wgts(1,i),bias_wgt(i),xi_i(i),y_ij(i),damp(i)
         do ii=1,2
            do j=1,nexternal
               do k=0,3
                  if (momenta_conf(ii).gt.0) then
                     momenta_m(k,j,ii,i)=momenta_str(k,j
     $                                               ,momenta_conf(ii))
                  else
                     momenta_m(k,j,ii,i)=-99d0
                     exit
                  endif
               enddo
            enddo
         enddo
      enddo
      end
