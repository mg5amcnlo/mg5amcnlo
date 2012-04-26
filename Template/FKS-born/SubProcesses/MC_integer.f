

      subroutine get_MC_integer(fks_configs,iint,vol)
      implicit none
      integer iint,i
      double precision ran2,rnd,vol
      external ran2
      logical firsttime
      data firsttime/.true./
      integer nintervals,maxintervals,fks_configs
      parameter (maxintervals=200)
      integer ncall(0:maxintervals)
      double precision grid(0:maxintervals),acc(0:maxintervals)
      common/integration_integer/grid,acc,ncall,nintervals
      logical            flat_grid
      common/to_readgrid/flat_grid                !Tells if grid read from file
      if (firsttime) then
         firsttime=.false.
         nintervals=fks_configs
         if (flat_grid) then
            do i=0,nintervals
               grid(i)=dble(i)/nintervals
            enddo
         else
            open(unit=52,file='grid.MC_integer',status='old',err=999)
            read(52,*) (grid(i),i=0,nintervals)
            close(52)
            goto 998
 999        write (*,*) 'WARNING: File "grid.MC_integer" not found.'/
     &           /' Using flat grid to start.'
            do i=0,nintervals
               grid(i)=dble(i)/nintervals
            enddo
 998        continue
         endif
         do i=0,nintervals
            acc(i)=0d0
            ncall(i)=0
         enddo
      endif

      rnd=ran2()
      iint=0
      do while (rnd .gt. grid(iint))
         iint=iint+1
      enddo
      if (iint.eq.0 .or. iint.gt.nintervals) then
         write (*,*) 'ERROR in get_MC_integer',iint,nintervals,grid
         stop
      endif
      vol=(grid(iint)-grid(iint-1))
      ncall(iint)=ncall(iint)+1
      return
      end

      subroutine fill_MC_integer(iint,f_abs)
      implicit none
      integer iint
      double precision f_abs
      integer nintervals,maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals)
      double precision grid(0:maxintervals),acc(0:maxintervals)
      common/integration_integer/grid,acc,ncall,nintervals
      acc(iint)=acc(iint)+f_abs
      return
      end

      subroutine regrid_MC_integer
      implicit none
      integer i,ib
      double precision tiny
      parameter ( tiny=1d-3 )
      character*101 buff
      integer nintervals,maxintervals
      parameter (maxintervals=200)
      integer ncall(0:maxintervals)
      double precision grid(0:maxintervals),acc(0:maxintervals)
      common/integration_integer/grid,acc,ncall,nintervals
c
c      write (*,*) (ncall(i),i=1,nintervals)
c      write (*,*) (acc(i)/ncall(i),i=1,nintervals)
c      write (*,*) (grid(i),i=1,nintervals)
c
c Give a nice printout of the grids used for the current iteration
      do i=1,101
         buff(i:i)=' '
      enddo
      do i=0,nintervals
         ib=1+int(grid(i)*100)
         write (buff(ib:ib),'(i1)') mod(i,10)
      enddo
      write (*,*) 'nFKSprocess ',buff
c
c Compute the accumulated cross section
      ncall(0)=0
      do i=1,nintervals
         if(ncall(i).ne.0) then
            acc(i)=acc(i-1)+acc(i)/ncall(i)
            ncall(0)=ncall(0)+ncall(i)
         else
            acc(i)=acc(i-1)
         endif
      enddo
      if (ncall(0).le.max(nintervals,10)) then
c Don't update grids if there were too few PS points.
         do i=0,nintervals
            acc(i)=0d0
            ncall(i)=0
         enddo
         return
      endif
c Define the new grids
      if (acc(nintervals).ne.0d0) then
         do i=0,nintervals
            grid(i)=acc(i)/acc(nintervals)
         enddo
      else
c Don't change grids if there was no contribution
         continue
      endif
c
c Make sure that a grid cell is at least of size 'tiny'
      do i=1,nintervals
         if (grid(i).le.(grid(i-1)+tiny)) then
            grid(i)=grid(i-1)+tiny
         endif
      enddo
      grid(nintervals)=1d0
      do i=1,nintervals
         if (grid(nintervals-i).ge.(grid(nintervals-i+1)-tiny)) then
            grid(nintervals-i)=1d0-dble(i)*tiny
         else
            exit
         endif
      enddo
c Write grid to a file
      open(unit=52,file='grid.MC_integer',status='unknown',err=999)
      write(52,*) (grid(i),i=0,nintervals)
      close(52)
c
c Reset the accumalated results because we start new iteration.
      do i=0,nintervals
         acc(i)=0d0
         ncall(i)=0
      enddo
      return
 999  write (*,*) 'Cannot open "grid.MC_integer" file'
      stop
      end
