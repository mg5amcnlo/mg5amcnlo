CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C                                                                          C
C                  HwU: Histograms with Uncertainties                      C
C                   By Rikkert Frederix, Dec. 2014                         C
C                                                                          C
C     Book, fill and write out histograms. Particularly suited for NLO     C
C     computations with correlations between points (ie. event and         C
C     counter-event) and multiple weights for given points (e.g. scale     C
C     and PDF uncertainties through reweighting).                          C
C                                                                          C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      
c To be called once at the start of each run. Initialises the packages
c and sets the number of weights that need to be included for each point.
      subroutine HwU_inithist(nweights,wgt_info)
      implicit none
      include "HwU.inc"
      integer i,nweights
      character*(*) wgt_info(*)
      do i=1,max_plots
         booked(i)=.false.
      enddo
c     Number of weights associated to each point. Note that the first
c     weight should always be the 'central value' and it should not be
c     zero if any of the other weights are non-zero.
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

      subroutine set_error_estimation(input)
c Error estimation
c     input 0: Simply sum all PS points; error is estimated by assuming
c        Binomial statistics: use for unweighted events. (Assumes that
c        all events have equal weight (up to a sign)).
c     input 1: Simply sum all PS points; error is estimated by the
c        variance between the separate iterations. This assumes that
c        each iteration has the same number of PS points (This is the
c        topdrawer default in MG5_aMC).
c     input 2: Sum PS points for a given iteration and error estimate by
c       square root of the sum of the squares. Perform a weighted average
c       iteration-by-iteration
      implicit none
      integer input
      integer error_estimation
      common /HwU_common2/ error_estimation
      if (input.ge.0 .and. input.le.2) then
         error_estimation=input
      else
         write (*,*) 'unknown error estimation',input
         stop 1
      endif
      return
      end
      
      block data HwU
c set the default for the error estimation method. To reduce the size of
c the executable, put the error_estimation variable in a separate common
c block. If we would have included the 'HwU.inc' file here, that
c complete common block seems to be included in the size executable
c (approx. 110 MB).
      integer error_estimation
      common /HwU_common2/ error_estimation
      data error_estimation /1/
      end
      
c Book the histograms at the start of the run. Give a 'label' (an
c integer) that identifies the plot when filling it and a title
c ('title_l') for each plot. Also the number of bins ('nbin_l') and the
c plot range (from 'xmin' to 'xmax') should be given.
      subroutine HwU_book(label,title_l,nbin_l,xmin,xmax)
      implicit none
      include "HwU.inc"
      integer label,nbin_l,i,j
      character*(*) title_l
      double precision xmin,xmax
c     Check that label and number of bins are reasonable
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
c     Compute the bin width
      step(label)=(xmax-xmin)/dble(nbin(label))
      do i=1,nbin(label)
c     Compute the lower and upper bin edges
         histxl(label,i)=xmin+step(label)*dble(i-1)
         histxm(label,i)=xmin+step(label)*dble(i)
c     Set all the bins to zero.
         do j=1,nwgts
            histy(j,label,i)=0d0
            histy_acc(j,label,i)=0d0
         enddo
         histi(label,i)=0
         histy2(label,i)=0d0
         histy_err(label,i)=0d0
      enddo
      np=0
      return
      end
      
c Fill the histograms identified with 'label' with a point at x with
c weights giving by 'wgts'. The dimension of the 'wgts' array should
c always be as large as the number of weights ('nweights') specified in
c the 'HwU_inithist' subroutine. That means that each point should have
c the same number of weights.
      subroutine HwU_fill(label,x,wgts)
      implicit none
      include "HwU.inc"
      integer label,i,j,bin
      double precision x, wgts(*)
c     If central weight is zero do not add this point.
      if (wgts(1).eq.0d0) return
c     Check if point is within plotting range
      if (x.lt.histxl(label,1) .or.
     $     x.gt.histxm(label,nbin(label))) return
c     Compute the bin to fill
      bin=int((x-histxl(label,1))/step(label)) +1
c     To prevent numerical inaccuracy, double check that bin is
c     reasonable
      if (bin.lt.1 .or. bin.gt.nbin(label)) return
c     Check if we already had another (correlated) point in this bin. If
c     so, add the current weight to that point.
      do i=1,np
         if ( p_label(i).eq.label .and. p_bin(i).eq.bin) then
            do j=1,nwgts
               p_wgts(j,i)=p_wgts(j,i)+wgts(j)
            enddo
            return
         endif
      enddo
c     If a new bin, add it to the list of points
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

c Call after all correlated contributions for a give phase-space
c point. I.e., every time you get a new set of random numbers from
c MINT/VEGAS. It adds the current list of points to the histograms. Add
c the squares to compute the statistical uncertainty on the bin. Do the
c second only for the weight corresponding to the 'central value'. In
c this way, correlations between events and counter-events can be
c correctly taken into account.
      subroutine HwU_add_points
      implicit none
      include "HwU.inc"
      integer i,j
      do i=1,np
         do j=1,nwgts
            histy(j,p_label(i),p_bin(i))=
     $           histy(j,p_label(i),p_bin(i))+p_wgts(j,i)
         enddo
         histi(p_label(i),p_bin(i))=histi(p_label(i),p_bin(i))+1
         histy2(p_label(i),p_bin(i))=
     $        histy2(p_label(i),p_bin(i))+p_wgts(1,i)**2
      enddo
      np=0
      return
      end


c *****For plotting during fixed order f(N)LO runs*****
c Call after every iteration. This adds the histograms of the current
c iteration ('histy') to the accumulated results ('histy_acc'), with the
c uncertainty estimate given in 'histy_err' and empties the arrays for
c the current iteration so that they can be filled with the next
c iteration.
      subroutine HwU_accum_iter(inclde,nPSpoints)
      implicit none
      include "HwU.inc"
      logical inclde
      integer nPSpoints,label,i,j
      double precision nPSinv,etot,vtot(max_wgts),niter,y_squared
      data niter /0d0/
      nPSinv = 1d0/dble(nPSpoints)
      if (inclde) niter = niter+1d0
      do label=1,max_plots
         if (.not. booked(label)) cycle
         if (inclde) then
            call accumulate_results(label,nPSinv,niter)
         endif
c     Reset the histo of the current iteration to zero so that they are
c     ready for the next iteration.
         do i=1,nbin(label)
            do j=1,nwgts
               histy(j,label,i)=0d0
            enddo
            histy2(label,i)=0d0
            histi(label,i)=0
         enddo
      enddo
      return
      end

c *****For plotting unweighted events****
c Overwrites the accumulated results ('histy_acc') with the current
c results ('histy') and provides an uncertainty estimate in
c 'histy_err'. This means that during the plotting of events, at
c intermediate stages this function can be called (together with
c HwU_output) to write intermediate plots to disk.
      subroutine finalize_histograms(nPSpoints)
      implicit none
      include "HwU.inc"
      integer label,nPSpoints,i,j
      double precision nPSinv,niter
      nPSinv=1d0/dble(nPSpoints)
      niter=1d0
      do label=1,max_plots
         if (.not. booked(label)) cycle
         do i=1,nbin(label)
c     Set all the bins to zero.
            do j=1,nwgts
               histy_acc(j,label,i)=0d0
            enddo
            histy_err(label,i)=0d0
         enddo
         call accumulate_results(label,nPSinv,niter)
      enddo
      return
      end

c This adds the histograms of the current iteration ('histy') to the
c accumulated results ('histy_acc'), with the uncertainty estimate given
c in 'histy_err'. When error_estimation==2 it adds using a weight
c corresponding to the statistical uncertainties on the 'central value'
c weights ('etot' (from 'histy2') and 'histy_err', respectively), if
c error_estimation==1 it simply averages over the iterations, if
c error_estimation==0 it uses Poisson statistics to compute the
c uncertainty. Note that this means that during the filling of the
c histograms the central value should not be zero if any of the other
c weights are non-zero.
      subroutine accumulate_results(label,nPSinv,niter)
      implicit none
      include "HwU.inc"
      integer label,i,j
      double precision nPSinv,etot,vtot(max_wgts),niter,y_squared
      integer error_estimation
      common /HwU_common2/ error_estimation
      if (error_estimation.eq.2) then
c     Use the weighted average bin-by-bin. This is not really justified
c     for fNLO computations, because for bins with low statistics, the
c     variance computation cannot be trusted.
         do i=1,nbin(label)
c     Skip bin if no entries
            if (histi(label,i).eq.0) cycle
c     Divide weights by the number of PS points. This means that this is
c     now normalised to the total cross section in that bin
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
            enddo
c     Error estimation of the current bin
            etot=sqrt(abs(histy2(label,i)*nPSinv-vtot(1)**2)*nPSinv)
c     Include "Bessel's correction" to have a corrected (even though
c     still biased) estimator of the standard deviation.
            if (histi(label,i).gt.1) then
               etot=etot* sqrt(dble(histi(label,i))
     &                    /(dble(histi(label,i))-1.5d0))
            else
               etot=0.999d49
            endif
c     If the error estimation of the accumulated results is still zero
c     (i.e. no points were added yet, e.g. because it is the first
c     iteration) simply copy the results of this iteration over the
c     accumulated results.
            if (histy_err(label,i).eq.0d0) then
               do j=1,nwgts
                  histy_acc(j,label,i)=vtot(j)
               enddo
               histy_err(label,i)=etot
            else
c     Add the results of the current iteration to the accumulated results
               do j=1,nwgts
                  histy_acc(j,label,i)=(histy_acc(j,label,i)
     &                 /histy_err(label,i)+vtot(j)/etot)/(1d0
     &                 /histy_err(label,i) + 1d0/etot)
               enddo
               histy_err(label,i)=
     $              1d0/sqrt(1d0/histy_err(label,i)**2+1d0/etot**2)
            endif
         enddo
      elseif(error_estimation.eq.1) then
c     simply sum the weights in the bins
         do i=1,nbin(label)
            if (histi(label,i).eq.0 .and.
     &           histy_acc(1,label,i).eq.0d0) cycle
            if (niter.ne.1d0) y_squared=((niter-1)*histy_err(label,i))
     &           **2+(niter-1)*histy_acc(1,label,i)**2
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
               histy_acc(j,label,i)=(histy_acc(j,label,i)*(niter-1d0)
     &              +vtot(j))/niter
            enddo
c     base the error on the variance in the results per iteration. For a
c     small number of iterations, this underestimates the actual
c     uncertainty.
            if (niter.eq.1d0) then
               histy_err(label,i)=0d0
            else
               histy_err(label,i)= sqrt(((y_squared+vtot(1)**2)/niter
     &              -histy_acc(1,label,i)**2)/niter)
            endif
         enddo
      elseif(error_estimation.eq.0) then
c     simply sum the weights in the bins
         do i=1,nbin(label)
            if (histi(label,i).eq.0 .and.
     &           histy_acc(1,label,i).eq.0d0) cycle
            do j=1,nwgts
               vtot(j)=histy(j,label,i)*nPSinv
               histy_acc(j,label,i)=(histy_acc(j,label,i)*(niter-1d0)
     &              +vtot(j))/niter
            enddo
c     Error estimation of the current bin using Poisson statistics,
c     making sure we normalise with the number of iterations.
            if (histi(label,i).ne.0) then
               etot=sqrt(histy2(label,i))*nPSinv
               histy_err(label,i)= sqrt(((niter-1d0)*histy_err(label,i))
     &              **2+etot**2)/niter
            else
               histy_err(label,i)=(niter-1d0)/niter*histy_err(label,i)
            endif
         enddo
      endif
      end
      
c Write the histograms to disk at the end of the run, multiplying the
c output by 'xnorm'
      subroutine HwU_output(unit,xnorm)
      implicit none
      include "HwU.inc"
      integer unit,i,j,label
      integer max_length
      parameter (max_length=(max_wgts+3)*17)
      character*(max_length) buffer
      character*4 str_nbin
      double precision xnorm
c     column info: x_min, x_max, y (central value), dy, {extra
c     weights}.
      write (unit,'(a$)') '##& xmin'
      write (unit,'(a$)') ' & xmax'
      write (unit,'(a$)') ' & '//trim(adjustl(wgts_info(1)))
      write (unit,'(a$)') ' & dy'
      do j=2,nwgts
         write (unit,'(a$)') ' & '//trim(adjustl(wgts_info(j)))
      enddo
      write (unit,'(a)') ''
      write (unit,'(a)') ''
      do label=1,max_plots
         if (.not. booked(label)) cycle
c     title
c        For some weird reason, it is no possible to include directly
c        the integer in two line below with the format '(12a,i4.4,3a,a,2a)'
c        this is why I have to do it in two steps instead.
         write (str_nbin,'(i4.4)') nbin(label)
         write (unit,'(12a,4a,3a,a,2a)') '<histogram> ',str_nbin,
     &                                       ' " ',title(label),' "'
c     data
         do i=1,nbin(label)
           write (buffer( 1:16),'(2x,e14.7)') histxl(label,i)
           write (buffer(17:32),'(2x,e14.7)') histxm(label,i)
           write (buffer(33:48),'(2x,e14.7)') histy_acc(1,label,i)*xnorm
           write (buffer(49:64),'(2x,e14.7)') histy_err(label,i)*xnorm
           do j=2,nwgts
              write (buffer((j+2)*16+1:(j+3)*16),'(2x,e14.7)')
     $             histy_acc(j,label,i)*xnorm
           enddo
           write (unit,'(a)') buffer(1:(nwgts+3)*16)
         enddo
c     2 empty lines after each plot
         write (unit,'(12a)') '<\histogram>'
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
