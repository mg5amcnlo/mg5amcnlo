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
c      parameter (ng = 43, maxdim = 25, maxinvar= 57 , maxconfigs=20)

      double precision  xgmin,     xgmax
      parameter        (xgmin=-1d0, xgmax=1d0)

      integer    maxevents         !Requires about 1K/event
      parameter (maxevents=200000)  !Maximum # events to write to disk

c*************************************************************************
c	Parameters used for parrallel running
c*************************************************************************
      integer    max_host   ,maxplace    ,maxpoints    ,maxans
      parameter (max_host=9,maxplace=9,maxpoints=10,maxans=5)
c      parameter (max_host=99,maxplace=199,maxpoints=100,maxans=50)
      integer    maxprb
      parameter (maxprb = maxconfigs*maxplace*maxpoints)
      integer    maxfprb
      parameter (maxfprb = maxinvar*maxplace*maxpoints)






