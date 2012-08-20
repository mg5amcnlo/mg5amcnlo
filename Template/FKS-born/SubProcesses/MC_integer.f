      subroutine get_MC_integer(this_dim,fks_configs,iint,vol)
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
      integer nintervals(maxdim),maxintervals,fks_configs
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
      
      if (realfirsttime) then
         realfirsttime=.false.
         do i=1,maxdim
            nintervals(i)=0
         enddo
      endif
      if (firsttime(this_dim)) then
         firsttime(this_dim)=.false.
         nintervals(this_dim)=fks_configs
         if (flat_grid) then
            call reset_MC_grid
         else
            open(unit=52,file='grid.MC_integer',status='old',err=999)
            do i=1,this_dim-1
               read(52,*) cdum
            enddo
            read(52,*) (grid(i,this_dim),i=0,nintervals(this_dim))
            close(52)
            goto 998
 999        write (*,*) 'WARNING: File "grid.MC_integer" not found.'/
     &           /' Using flat grid to start.'
            do i=0,nintervals(this_dim)
               grid(i,this_dim)=dble(i)/nintervals(this_dim)
            enddo
 998        continue
         endif
         do i=0,nintervals(this_dim)
            acc(i,this_dim)=0d0
            ncall(i,this_dim)=0
         enddo
      endif
c
      rnd=ran2()
      iint=0
      do while (rnd .gt. grid(iint,this_dim))
         iint=iint+1
      enddo
      if (iint.eq.0 .or. iint.gt.nintervals(this_dim)) then
         write (*,*) 'ERROR in get_MC_integer',iint,nintervals(this_dim)
     &        ,(grid(i,this_dim),i=1,nintervals(this_dim))
         stop
      endif
      vol=(grid(iint,this_dim)-grid(iint-1,this_dim))
      ncall(iint,this_dim)=ncall(iint,this_dim)+1
      return
      end

      subroutine reset_MC_grid
      implicit none
      integer i,this_dim
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
         do i=0,nintervals(this_dim)
            grid(i,this_dim)=dble(i)/nintervals(this_dim)
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
      logical firsttime(maxdim)
      integer nintervals(maxdim),maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals,maxdim)
      double precision grid(0:maxintervals,maxdim),acc(0:maxintervals
     &     ,maxdim)
      common/integration_integer/grid,acc,ncall,nintervals
      acc(iint,this_dim)=acc(iint,this_dim)+f_abs
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
c
c      write (*,*) (ncall(i),i=1,nintervals)
c      write (*,*) (acc(i)/ncall(i),i=1,nintervals)
c      write (*,*) (grid(i),i=1,nintervals)
c
c Give a nice printout of the grids used for the current iteration
      do this_dim=1,maxdim
         if (nintervals(this_dim).eq.0) cycle
         do i=1,101
            buff(i:i)=' '
         enddo
         do i=0,nintervals(this_dim)
            ib=1+int(grid(i,this_dim)*100)
            write (buff(ib:ib),'(i1)') mod(i,10)
         enddo
         write (*,*) 'nFKSprocess ',buff
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
            return
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
c Make sure that a grid cell is at least of size 'tiny'
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
         write(52,*) ' ',(grid(i,this_dim),i=0,nintervals(this_dim))
      enddo
      close(52)
c
c Reset the accumalated results because we start new iteration.
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
