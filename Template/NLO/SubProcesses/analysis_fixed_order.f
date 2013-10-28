c
c This file contains the default plots for fixed order runs
c
c This uses the hbook package and generates plots in the top-drawer
c format. This format is human-readable. After running, the plots can be
c found in the Events/run_XX/ directory.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called once at the start of each run. Here the
c plots should be declared. 
      implicit none
      character*(*) weights_info(*)
c Initialize the histogramming package (hbook):
      call inithist
c
c Declare the histograms using 'bookup'.
c     o) The first argument is an integer that labels the histogram. In
c     the analysis_end and analysis_fill subroutines this label is used
c     to keep track of the plot. The label should be a number between 1
c     and NPLOTS/4=1250 (can be increased in dbook.inc).
c     o) The second argument is a string that will apear above the plot. 
c     o) The third, forth and fifth arguments are the bin size, the
c     lower edge of the first bin and the upper edge of the last
c     bin. There is a maximum of 100 bins per plot.
c     o) When including scale and/or PDF uncertainties, fill a plot for
c     each weight, and compute the uncertainties from the final set of
c     histograms
      call bookup(1,'total rate',1.0d0,0.5d0,5.5d0)
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(xnorm)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called once at the end of the run. Here the plots
c are written to disk. Note that this is done for each integration
c channel separately. There is an external script that will read the top
c drawer files in each of the integration channels and combines them by
c summing all the bins in a final single top-drawer file to be put in
c the Events/run_XX directory.
      implicit none
      character*14 ytit
      double precision xnorm
      integer i
      include 'dbook.inc'
c Do not touch the folloing lines. These lines make sure that the plots
c will have the correct overall normalisation: cross section (in pb) per
c bin.
      do i=1,NPLOTS
         call mopera(i,'+',i,i,xnorm,0.d0)
         call mfinal(i)
      enddo
      ytit='sigma per bin '
c      
c Here the plots are put in a format to be written to file. Use the
c multitop() function.
c     o) The first argument is the plot label
c     o) The second and third arguments are not used (keep them to the
c     default 3,2)
c     o) Fourth argument is the label for the x-axis
c     o) Fifth argument is the y-axis
c     o) Final argument declares if the y-axis should be a linear 'LIN'
c     or logarithmic 'LOG' scale.
      call multitop(1,3,2,'total rate',ytit,'LIN')
      return
      end



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgt)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called for each n-body and (n+1)-body configuration
c that passes the generation cuts.
      implicit none
c This includes the 'nexternal' parameter that labels the number of
c particles in the (n+1)-body process
      include 'nexternal.inc'
c This is an array which is '-1' for initial state and '1' for final
c state particles
      integer istatus(nexternal)
c This is an array with (simplified) PDG codes for the particles. Note
c that channels that are combined (i.e. they have the same matrix
c elements) are given only 1 set of PDG codes.
      integer iPDG(nexternal)
c The array of the momenta and masses of the particles. The format is
c "E, px, py, pz, mass", while the second dimension loops over the
c particles in the process. Note that these are the (n+1)-body
c particles, for which the n-body have one momenta equal to all zero's
c (this is not necessarily the last particle in the list). If one uses
c IR-safe obserables only, there should be no difficulty in using this.
      double precision p(0:5,nexternal)
c The weight of the current phase-space point
      double precision wgt
c local variables
      double precision var
c
c Fill the histograms here using a call to the mfill() subroutine. The
c first argument is the histogram label, the second is the numerical
c value of the variable to plot for the current phase-space point and
c the final argument is the weight of the current phase-space point.
      var=1d0
      call mfill(1,var,wgt)

      return
      end
