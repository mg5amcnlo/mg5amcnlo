cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c These subroutines allow for a MonteCarlo integration (i.e. sum) over
c integer values with importance sampling.  In aMC@NLO these are used to
c do the Monte Carlo sum over the real-emission FKS directories for a
c given underlying Born.
c
c To use it, simply call the
c get_MC_integer(this_dim,niint_thisd,iint,vol) subroutine. This
c subroutine takes 4 arguments:
c
c this_dim     integer    The label corresponding to the integer one wants
c                         to MC over. By default, allows up to 50 separate
c                         integers. (Can be increased by changing 'maxdim')
c niint_thisd  integer    The range of integers (1 ... 'niint_thisd') one
c                         wants to importance sampling (ie. the number of
c                         intervals). This value has to be kept fixed for
c                         a given 'this_dim' during the integration.
c iint         integer    The randomly pick integer among the 'niint_thisd'
c vol          real*8     The volume related to 'iint'. It's defined such
c                         that the weight in the sum should be divided by
c                         'vol'.
c
c The obtained weight should be accumulated in the
c fill_MC_integer(this_dim,iint,f_abs) subroutine, where 'this_dim' is
c the label corresponding to the integer used, 'iint' is the numerical
c value of the integer picked, and f_abs is the (real*8) value of the
c integrand (without dividing by 'vol'), computed with the 'iint' value.
c Hence, for every call to 'get_MC_integer' there should be a call to
c 'fill_MC_integer'.
c 
c When enough values have been chosen, the integration grids can be
c updated with a call to 'regrid_MC_integer'. This subroutine doesn't
c take any arguments and updates all the integration grids for all
c 'this_dim'. For normal use, this should be updated at the same time
c the VEGAS/MINT/DSAMPLE (or whatever) grids are updated.  After every
c call to 'regrid_MC_integer' the grids are written in the file
c 'grid.MC_integer' and can be used for a subsequent run. To start from
c a run using the existing grids, the 'flat_grid' logical value (in the
c 'to_readgrid' common block) should be set to .false. before the first
c call to get_MC_integer for a given 'this_dim'.
c
c A call to 'reset_MC_grid' resets all the MC integration grids to a
c flat starting point.
c
c A call to 'empty_MC_grid' removes all the accumulated results for the 
c current iteration.
c
c Written by Rikkert Frederix, 2012.
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine get_MC_integer(this_dim,niint_thisd,iint,vol)
      implicit none
      integer iint,i,this_dim
      double precision ran2,rnd,vol
      external ran2
      integer maxdim
      parameter (maxdim=50)
      logical firsttime(maxdim),realfirsttime
      data firsttime/maxdim*.true./
      data realfirsttime/.true./
      character*1 cdum
      integer nintervals(maxdim),maxintervals,niint_thisd
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim)
      double precision grid(0:maxintervals,maxdim),
     &     acc(0:maxintervals,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file
      if (this_dim.lt.1.or.this_dim.gt.maxdim) then
         write (*,*) 'Increase maxdim in MC_integer.f',maxdim,this_dim
      endif
c Set the number of intervales for all the dimensions to zero the very
c first time this subroutine is called
      if (realfirsttime) then
         realfirsttime=.false.
         do i=1,maxdim
            nintervals(i)=0
         enddo
      endif
c For each 'this_dim', if it's the first time this subroutine is called,
c start with a fresh grids, or read grids from filem depending on the
c flat_grid logical parameter.
      if (firsttime(this_dim)) then
         firsttime(this_dim)=.false.
         nintervals(this_dim)=niint_thisd
         if (flat_grid) then
c Flat grid for this dimension
            do i=0,nintervals(this_dim)
               grid(i,this_dim)=dble(i)/nintervals(this_dim)
            enddo
         else
c Read the grid for 'this_dim' from file
            open(unit=52,file='grid.MC_integer',status='old',err=999)
            do i=1,this_dim-1 ! skip the lines not needed for 'this_dim'
               read(52,*,end=999,err=999) cdum
            enddo
            read(52,*,end=999,err=999)
     &           (grid(i,this_dim),i=0,nintervals(this_dim)) ! here is what we want
            do i=this_dim+1,maxdim ! make sure that there are enough lines in this file
               read(52,*,end=999,err=999) cdum
            enddo
            close(52)
            goto 998
c If file not found, give warning and use flat grids for this dimension
 999        write (*,*) 'WARNING: File "grid.MC_integer" not found.'/
     &           /' Using flat grid to start for',this_dim
            close(52)
            do i=0,nintervals(this_dim)
               grid(i,this_dim)=dble(i)/nintervals(this_dim)
            enddo
 998        continue
         endif
c Set the arrays in which we accumulate the results to zero
         do i=0,nintervals(this_dim)
            acc(i,this_dim)=0d0
            ncall(i,this_dim)=0
         enddo
      endif
c
c Take a fresh random number and find the correponding 'iint'
      rnd=ran2()
      iint=0
      do while (rnd .gt. grid(iint,this_dim))
         iint=iint+1
      enddo
c Safety
      if (iint.eq.0 .or. iint.gt.nintervals(this_dim)) then
         write (*,*) 'ERROR in get_MC_integer',iint,nintervals(this_dim)
     &        ,(grid(i,this_dim),i=1,nintervals(this_dim))
         stop
      endif
c Compute the volume 'vol'
      vol=(grid(iint,this_dim)-grid(iint-1,this_dim))
c Increase the array that keeps track of the number of times this iint
c (for 'this_dim') has been picked.
      ncall(iint,this_dim)=ncall(iint,this_dim)+1
      return
      end

      subroutine reset_MC_grid
      implicit none
      integer i,this_dim
      integer maxdim
      parameter (maxdim=50)
      logical firsttime(maxdim)
      integer maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim),nintervals(maxdim)
      double precision grid(0:maxintervals,maxdim),acc(0:maxintervals
     &     ,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      do this_dim=1,maxdim
         do i=0,nintervals(this_dim)
            if (nintervals(this_dim).ne.0) 
     &           grid(i,this_dim)=dble(i)/nintervals(this_dim)
            acc(i,this_dim)=0d0
            ncall(i,this_dim)=0
         enddo
      enddo
      return
      end
            
      subroutine fill_MC_integer(this_dim,iint,f_abs)
      implicit none
      integer iint,this_dim
      double precision f_abs
      integer maxdim
      parameter (maxdim=50)
      integer maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim),nintervals(maxdim)
      double precision grid(0:maxintervals,maxdim),acc(0:maxintervals
     &     ,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      acc(iint,this_dim)=acc(iint,this_dim)+f_abs
      return
      end

      subroutine empty_MC_integer
      implicit none
      integer i,this_dim
      integer maxdim
      parameter (maxdim=50)
      integer nintervals(maxdim),maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim)
      double precision grid(0:maxintervals,maxdim),acc(0:maxintervals
     &     ,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      do this_dim=1,maxdim
         do i=0,nintervals(this_dim)
            acc(i,this_dim)=0d0
            ncall(i,this_dim)=0
         enddo
      enddo
      return
      end

      subroutine regrid_MC_integer
      implicit none
      integer i,ib,this_dim
      double precision tiny
      parameter ( tiny=1d-3 )
      character*101 buff
      integer maxdim
      parameter (maxdim=50)
      logical firsttime(maxdim)
      integer nintervals(maxdim),maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim)
      double precision grid(0:maxintervals,maxdim),acc(0:maxintervals
     &     ,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      do this_dim=1,maxdim
         if (nintervals(this_dim).eq.0) cycle
c
c Compute the accumulated cross section
         ncall(0,this_dim)=0
         do i=1,nintervals(this_dim)
            if(ncall(i,this_dim).ne.0) then
               acc(i,this_dim)=acc(i-1,this_dim)+acc(i,this_dim)/ncall(i
     &              ,this_dim)
               ncall(0,this_dim)=ncall(0,this_dim)+ncall(i,this_dim)
            else
               acc(i,this_dim)=acc(i-1,this_dim)
            endif
         enddo
         if (ncall(0,this_dim).le.max(nintervals(this_dim),10)) then
c Don't update grids if there were too few PS points.
            do i=0,nintervals(this_dim)
               acc(i,this_dim)=0d0
               ncall(i,this_dim)=0
            enddo
            cycle
         endif
c Define the new grids
         if (acc(nintervals(this_dim),this_dim).ne.0d0) then
            do i=0,nintervals(this_dim)
               grid(i,this_dim)=acc(i,this_dim)/acc(nintervals(this_dim)
     &              ,this_dim)
            enddo
         else
c Don't change grids if there was no contribution
            continue
         endif
c
c Make sure that a grid cell is at least of size 'tiny', otherwise it
c will never recover from a "bad iteration"
         do i=1,nintervals(this_dim)
            if (grid(i,this_dim).le.(grid(i-1,this_dim)+tiny)) then
               grid(i,this_dim)=grid(i-1,this_dim)+tiny
            endif
         enddo
         grid(nintervals(this_dim),this_dim)=1d0
         do i=1,nintervals(this_dim)
            if (grid(nintervals(this_dim)-i
     &           ,this_dim).ge.(grid(nintervals(this_dim)-i+1,this_dim)
     &           -tiny)) then
               grid(nintervals(this_dim)-i,this_dim)=1d0-dble(i)*tiny
            else
               exit
            endif
         enddo
c Write grid to a file
      enddo
      open(unit=52,file='grid.MC_integer',status='unknown',err=999)
      do this_dim=1,maxdim
         write(52,*) (grid(i,this_dim),i=0,nintervals(this_dim))
      enddo
      close(52)
c
c Give a nice printout of the grids after the current iteration
      do this_dim=1,maxdim
         if (nintervals(this_dim).eq.0) cycle
         do i=1,101
            buff(i:i)=' '
         enddo
         do i=0,nintervals(this_dim)
            ib=1+int(grid(i,this_dim)*100)
            write (buff(ib:ib),'(i1)') mod(i,10)
         enddo
         write (*,'(i3,a,a)') this_dim,':  ',buff
      enddo
c
c Reset the accumulated results because we start new iteration.
      do this_dim=1,maxdim
         do i=0,nintervals(this_dim)
            acc(i,this_dim)=0d0
            ncall(i,this_dim)=0
         enddo
      enddo
      return
 999  write (*,*) 'Cannot open "grid.MC_integer" file'
      stop
      end
