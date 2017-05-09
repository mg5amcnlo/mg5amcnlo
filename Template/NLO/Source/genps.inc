c*************************************************************************
c	Parameters used by genps
c*************************************************************************
c*************************************************************************
c       Parameters used by genps and dsample, you must recompile
c	dsample if you change anything below
c*************************************************************************
      include 'maxparticles.inc'
      include 'maxconfigs.inc'

      integer    maxproc,     maxamps
      parameter (maxproc=100, maxamps=7500)	
