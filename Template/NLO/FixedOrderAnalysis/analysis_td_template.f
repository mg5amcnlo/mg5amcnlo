c
c This file contains the default histograms for fixed order runs: it
c only plots the total rate as an example. It can be used as a template
c to make distributions for other observables.
c
c This uses the hbook package and generates histograms in the top-drawer
c format. This format is human-readable. After running, the histograms
c can be found in the Events/run_XX/ directory.
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_begin(nwgt,weights_info)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called once at the start of each run. Here the
c histograms should be declared. 
c
c Declare the histograms using 'bookup'.
c     o) The first argument is an integer that labels the histogram. In
c     the analysis_end and analysis_fill subroutines this label is used
c     to keep track of the histogram. The label should be a number
c     between 1 and NPLOTS/4=5000 (can be increased in dbook.inc).
c     o) The second argument is a string that will apear above the
c     histogram. Do not use brackets "(" or ")" inside this string.
c     o) The third, forth and fifth arguments are the bin size, the
c     lower edge of the first bin and the upper edge of the last
c     bin. There is a maximum of 100 bins per histogram.
c     o) When including scale and/or PDF uncertainties, declare a
c     histogram for each weight, and compute the uncertainties from the
c     final set of histograms
c
      implicit none
c When including scale and/or PDF uncertainties the total number of
c weights considered is nwgt
      integer nwgt
c In the weights_info, there is an text string that explains what each
c weight will mean. The size of this array of strings is equal to nwgt.
      character*(*) weights_info(*)
c Local variables
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
c Initialize the histogramming package (hbook):
      call inihist
c Fill the c_analysis common block with the number of weights that will
c be computed
      nwgt_analysis=nwgt
c
c loop over all the weights that are computed (depends on run_card
c parameters do_rwgt_scale and do_rwgt_pdf):
      do kk=1,nwgt_analysis
c make sure that there is a separate histogram initialized for each
c weight
         l=(kk-1)*2
c declare (i.e. book) the histograms
         call bookup(l+1,'total rate      '//weights_info(kk),
     &        1.0d0,0.5d0,5.5d0)
         call bookup(l+2,'total rate Born '//weights_info(kk),
     &        1.0d0,0.5d0,5.5d0)
      enddo
      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_end(xnorm)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called once at the end of the run. Here the
c histograms are written to disk. Note that this is done for each
c integration channel separately. There is an external script that will
c read the top drawer files in each of the integration channels and
c combines them by summing all the bins in a final single top-drawer
c file to be put in the Events/run_XX directory.
c      
c The histograms are put in a format to be written to file. Use the
c multitop() subroutine.
c     o) The first argument is the histogram label
c     o) The second and third arguments are not used (keep them to the
c     default 3,2)
c     o) Fourth argument is the label for the x-axis. Do not use
c     brackets "(" or ")" inside this string.
c     o) Fifth argument is the y-axis
c     o) Final argument declares if the y-axis should be a linear 'LIN'
c     or logarithmic 'LOG' scale.
      implicit none
      character*14 ytit
      double precision xnorm
      integer i
c Local variables
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
c This defines NPLOTS:
      include 'dbook.inc'
c Open the topdrawer file to write the histograms (do not remove the
c next to calls)
      call open_topdrawer_file
      call mclear
c Do not touch the folloing 4 lines. These lines make sure that the
c histograms will have the correct overall normalisation: cross section
c (in pb) per bin.
      do i=1,NPLOTS
         call mopera(i,'+',i,i,xnorm,0.d0)
         call mfinal(i)
      enddo
      ytit='sigma per bin '
c Loop over the plots that are declared with bookup
      do kk=1,nwgt_analysis
         l=(kk-1)*2
         do i=1,2
c convert them to a format suitable for writing
            call multitop(l+i,3,2,'total rate',ytit,'LIN')
         enddo
      enddo
c Close the topdrawer file (Do not remove this call)
      call close_topdrawer_file
      return
      end



cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c This subroutine is called for each n-body and (n+1)-body configuration
c that passes the generation cuts. Here the histrograms are filled.
      implicit none
c This includes the 'nexternal' parameter that labels the number of
c particles in the (n+1)-body process
      include 'nexternal.inc'
c This is an array which is '-1' for initial state and '1' for final
c state particles
      integer istatus(nexternal)
c This is an array with (simplified) PDG codes for the particles. Note
c that channels that are combined (i.e. they have the same matrix
c elements) are given only 1 set of PDG codes. This means, e.g., that
c when using a 5-flavour scheme calculation (massless b quark), no
c b-tagging can be applied.
      integer iPDG(nexternal)
c The array of the momenta and masses of the initial and final state
c particles in the lab frame. The format is "E, px, py, pz, mass", while
c the second dimension loops over the particles in the process. Note
c that these are the (n+1)-body particles; for the n-body there is one
c momenta equal to all zero's (this is not necessarily the last particle
c in the list). If one uses IR-safe obserables only, there should be no
c difficulty in using this.
      double precision p(0:4,nexternal)
c The weight of the current phase-space point is wgts(1). If scale
c and/or PDF uncertainties are included through reweighting, the rest of
c the array contains the list of weights in the same order as described
c by the weigths_info strings in analysis_begin
      double precision wgts(*)
c The ibody variable is:
c     ibody=1 : (n+1)-body contribution
c     ibody=2 : n-body contribution (excluding the Born)
c     ibody=3 : Born contribution
c The histograms need to be filled for all these contribution to get the
c physics NLO results. (Note that the adaptive phase-space integration
c is optimized using the sum of the contributions, therefore plotting
c them separately might lead to larger than expected statistical
c fluctuations).
      integer ibody
c local variables
      double precision wgt,var
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
c
c Fill the histograms here using a call to the mfill() subroutine. The
c first argument is the histogram label, the second is the numerical
c value of the variable to plot for the current phase-space point and
c the final argument is the weight of the current phase-space point.
      var=1d0
c loop over all the weights that are computed (depends on run_card
c parameters do_rwgt_scale and do_rwgt_pdf):
      do kk=1,nwgt_analysis
         wgt=wgts(kk)
         l=(kk-1)*2
c always fill the total rate
         call mfill(l+1,var,wgt)
c only fill the total rate for the Born when ibody=3
         if (ibody.eq.3) call mfill(l+2,var,wgt)
      enddo
      return
      end
