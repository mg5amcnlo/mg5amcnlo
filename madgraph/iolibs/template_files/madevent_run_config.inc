c*********************************************************************
c     Parameters to configure running information for MadEvent
c     The default values of these parameters should not need to be
c     changed, unless there is a special need for optimization
c*********************************************************************
c
c     The following parameters are used by symmetry.f in setting up the survey
c
      integer    icomp
      parameter (icomp = 3)   !BW + Symmetry compression  0 == none	
c     The following sets the maximum number of parameters in the run_card.dat
      integer maxpara
      parameter (maxpara=1000)
c
c     The following parameters are used by gen_ximprove.f in running refine
c
      integer    min_events                 !Minimum number of events/iteration
      parameter (min_events = 1000)         !to refine a channel
      integer    max_events                 !Maximum number of events/iteration
      parameter (max_events = 2000)        !to refine a channel
      integer    max_iter                   !Maximum number of iterations
      parameter (max_iter = 9)              !during refinement
      integer    MaxEventsPerJob            !Maximum number of events requested from a single job
      parameter (MaxEventsPerJob=1000)	
c
c     The following are used for parallel running
c
      character*(20) PBS_QUE
      parameter     (PBS_QUE = 'madgraph')

      integer    ChanPerJob
      parameter (ChanPerJob=%(chanperjob)s)  !Number of channels / job for survey

c      integer    max_np
c      parameter (max_np=1)  !Number of channels / job for refine
c
c
c
	double precision trunc_max
        parameter (trunc_max=0.01)

c     Parameter for string buffer length for systematics variations info
        integer s_bufflen
        parameter (s_bufflen=26+3+(max_particles-1)*9+
     $                       2*(max_particles-1)*15)

c     Parameter for string buffer length for clustering info
        integer clus_bufflen
        parameter (clus_bufflen=43)
