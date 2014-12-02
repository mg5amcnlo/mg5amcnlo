C     HwU: Histograms with Uncertainties
C     By  Rikkert Frederix

c To be called once at the start of each run
      subroutine HwU_inithist(nweights,wgt_info)
      implicit none
      include "HwU.inc"
      integer i,nweights
      character*(*) wgt_info(*)
      do i=1,max_plots
         booked(i)=.false.
      enddo
      nwgts=nweights
      if (nwgts.gt.max_wgts) then
         write (*,*) 'ERROR: increase max_wgts in HwU histogramming'
     $        ,max_wgts,nwgts
         stop 1
      endif
      do i=1,nwgts
         wgts_info(i)=wgt_info(i)
      enddo
      return
      end

c Book the histograms at the start of the run
      subroutine HwU_book(label,title_l,nbin_l,xmin,xmax)
      implicit none
      include "HwU.inc"
      integer label,nbin_l,i,j
      character*(*) title_l
      double precision xmin,xmax
      if (label.gt.max_plots) then
         write (*,*) 'ERROR: increase max_plots in HwU histogramming'
     $        ,max_plots, label
         stop 1
      endif
      if (nbin_l.gt.max_bins) then
         write (*,*) 'ERROR: increase max_bins in HwU histogramming'
     $        ,max_bins,nbin_l
         stop 1
      endif
      booked(label)=.true.
      title(label)=title_l
      nbin(label)=nbin_l
      step(label)=(xmax-xmin)/dble(nbin(label))
      do i=1,nbin(label)
         histxl(label,i)=xmin+step(label)*dble(i-1)
         histxm(label,i)=xmin+step(label)*dble(i)
         do j=1,nwgts
            histy(j,label,i)=0d0
            histy2(label,i)=0d0
            histy_acc(j,label,i)=0d0
            histy_err(label,i)=0d0
         enddo
      enddo
      np=0
      return
      end
      
c Fill the histograms with a point
      subroutine HwU_fill(label,x,wgts)
      implicit none
      include "HwU.inc"
      integer label,i,j,bin
      double precision x, wgts(*)
      bin=int((x-histxm(label,1))/step(label)) +1
      if (bin.lt.1 .or. bin.gt.nbin(label)) return
      do i=1,np
         if ( p_label(i).eq.label .and. p_bin(i).eq.bin) then
            do j=1,nwgts
               p_wgts(j,i)=p_wgts(j,i)+wgts(j)
            enddo
            return
         endif
      enddo
      np=np+1
      if (np.gt.max_points) then
         write (*,*) 'ERROR: increase max_points in HwU histogramming'
     $        ,max_points
         stop 1
      endif
      p_label(np)=label
      p_bin(np)=bin
      do j=1,nwgts
         p_wgts(j,np)=wgts(j)
      enddo
      return
      end

c Call after all correlated contributions for a give phase-space point
      subroutine HwU_add_points
      implicit none
      include "HwU.inc"
      integer i,j
      do i=1,np
         do j=1,nwgts
            histy(j,p_label(i),p_bin(i))=
     $           histy(j,p_label(i),p_bin(i))+p_wgts(j,i)
         enddo
         histy2(p_label(i),p_bin(i))=
     $        histy2(p_label(i),p_bin(i))+p_wgts(1,i)**2
      enddo
      np=0
      return
      end

c Call after every iteration
      subroutine HwU_accum_iter(inclde,nPSpoints)
      implicit none
      include "HwU.inc"
      logical inclde
      integer nPSpoints,i,j,label
      double precision nPSinv,etot,vtot(max_wgts)
      logical firsttime(max_plots)
      data firsttime /max_plots*.true./
      nPSinv = 1d0/dble(nPSpoints)
      do label=1,max_plots
         if (.not. booked(label)) cycle
         if (inclde) then
            do i=1,nbin(label)
               do j=1,nwgts
                  vtot(j)=histy(j,label,i)*nPSinv
               enddo
               etot=sqrt(abs(histy2(label,i)*nPSinv-vtot(1)**2)*nPSinv)
               if (etot.eq.0 .and. vtot(1).eq.0d0) then
                  cycle
               elseif (etot.eq.0) then
                  etot=vtot(1)
               endif
               if (histy_err(label,i).eq.0d0) then
                  do j=1,nwgts
                     histy_acc(j,label,i)=vtot(j)
                  enddo
                  histy_err(label,i)=etot
               else
                  if (etot.eq.0d0) then
                     etot=histy_err(label,i)
                  elseif (histy_err(label,i).eq.0d0) then
                     histy_err(label,i)=etot*1d99
                  endif
                  do j=1,nwgts
                     histy_acc(j,label,i)=(histy_acc(j,label,i)
     $                    /histy_err(label,i)+vtot(j)/etot)/(1d0
     $                    /histy_err(label,i) + 1d0/etot)
                  enddo
                  histy_err(label,i)=1d0/sqrt(1d0/histy_err(label,i)**2
     $                 +1d0/etot**2)
               endif
            enddo
         endif
         do i=1,nbin(label)
            do j=1,nwgts
               histy(j,label,i)=0d0
            enddo
            histy2(label,i)=0d0
         enddo
         firsttime(label)=.false.
      enddo
      return
      end

c Write the histograms to disk at the end of the run
      subroutine HwU_output(unit)
      implicit none
      include "HwU.inc"
      integer unit,i,j,label
      integer max_length
      parameter (max_length=(max_wgts+3)*16)
      character*(max_length) buffer
      do label=1,max_plots
         if (.not. booked(label)) cycle
c title
         write (unit,'(a,a)') '#',title(label)
c column info
         write (buffer( 1:16),'(a)')'            xmin'
         write (buffer(17:32),'(a)')'            xmax'
         write (buffer(33:48),'(1x,a15)') wgts_info(1)(1:15)
         write (buffer(49:64),'(a)')'              dy'
         do j=2,nwgts
            write (buffer((j+2)*16+1:(j+3)*16),'(1x,a15)')
     $           wgts_info(j)(1:15)
         enddo
         write (unit,'(a)') buffer(1:(nwgts+3)*16)
c data
         do i=1,nbin(label)
            write (buffer( 1:16),'(2x,e14.7)') histxl(label,i)
            write (buffer(17:32),'(2x,e14.7)') histxm(label,i)
            write (buffer(33:48),'(2x,e14.7)') histy_acc(1,label,i)
            write (buffer(49:64),'(2x,e14.7)') histy_err(label,i)
            do j=2,nwgts
               write (buffer((j+2)*16+1:(j+3)*16),'(2x,e14.7)')
     $              histy_acc(j,label,i)
            enddo
            write (unit,'(a)') buffer(1:(nwgts+3)*16)
         enddo
c 2 empty lines
         write (unit,'(a)') ''
         write (unit,'(a)') ''
      enddo
      return
      end

c dummy subroutine
      subroutine accum(idummy)
      integer idummy
      end
c dummy subroutine
      subroutine addfil(string)
      character*(*) string
      end
