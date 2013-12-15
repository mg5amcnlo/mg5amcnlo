c*************************************************************************
c	Parameters used by genps
c*************************************************************************
c*************************************************************************
c       Parameters used by genps and dsample, you must recompile
c	dsample if you change anything below
c*************************************************************************
      include 'maxparticles.inc'
      integer    max_tree    , max_dim
      parameter (max_tree=200, max_dim=max_tree*(max_branch+1))
      integer    ng     , maxdim     , maxinvar     , maxconfigs
c      parameter (ng = 96, maxdim = 25, maxinvar= 50 , maxconfigs=10)
      parameter (ng = 96, maxdim = 3*(max_particles-2)-1, maxinvar= 4*max_particles, maxconfigs=10)

      include 'maxconfigs.inc'

        
      double precision  xgmin,     xgmax
      parameter        (xgmin=-1d0, xgmax=1d0)

      integer    maxproc,     maxamps
      parameter (maxproc=100, maxamps=7500)	

c      integer    maxproc
c      parameter (maxproc=100)	

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






