c*************************************************************************
c	Parameters used by genps
c*************************************************************************
	include 'nexternal.inc'
c*************************************************************************
c       Parameters used by genps and dsample, you must recompile
c	dsample if you change anything below
c*************************************************************************
      integer    max_particles  , max_branch
      parameter (max_particles=10, max_branch=max_particles-2)
      integer    max_tree    , max_dim
      parameter (max_tree=200, max_dim=max_tree*(max_branch+1))
      integer    ng     , maxdim     , maxinvar     , maxconfigs
      parameter (ng = 96, maxdim = 25, maxinvar= 50 , maxconfigs=10)

      integer    lmaxconfigs
      parameter (lmaxconfigs=7500)       !Used for iforest

      double precision  xgmin,     xgmax
      parameter        (xgmin=-1d0, xgmax=1d0)

      integer    maxproc,     maxamps
      parameter (maxproc=100, maxamps=7500)	

      integer    maxevents         !Requires about 1K/event
      parameter (maxevents=100000)  !Maximum # events to write to disk

c*************************************************************************
c	Parameters used for parrallel running
c*************************************************************************
      integer    max_host   ,maxplace    ,maxpoints    ,maxans
      parameter (max_host=9,maxplace=9,maxpoints=10,maxans=5)
      integer    maxprb
      parameter (maxprb = maxconfigs*maxplace*maxpoints)
      integer    maxfprb
      parameter (maxfprb = maxinvar*maxplace*maxpoints)






