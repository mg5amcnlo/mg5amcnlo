C...GETISSUD performs an interpolation/extrapolation in 3 dimensions by
C...fitting quadratic splines using 4 points in each dimension
      double precision function getissud(ibeam,kfl,x1,x2,pt2)
      implicit none

      include 'sudgrid.inc'

c   Arguments
      integer ibeam,kfl
      double precision x1,x2,pt2
c   Storing values for the interpolation
      double precision smallgrid(4,4),minigrid(4) ! pt2,x1
c   Local variables
      integer ipt2,ix1,ix2,ilo,ihi,i,j,k,kkfl,ipoints
      double precision pt2i,x2i,x1i,minpoint,maxpoint,x(3)
      integer nerr
      data nerr/0/

      getissud=0

      x(1)=log(x2)
      x(2)=x1
      x(3)=log(pt2)

      kkfl=kfl
      if(ibeam.lt.0) kkfl=-kkfl
      if(kkfl.lt.-2) kkfl=iabs(kfl)
      if(iabs(kkfl).eq.21) kkfl=0
      if(kkfl.eq.5) then
        ipoints=2
      else
        ipoints=1
      endif        
      if(kkfl.gt.5) then
        if(nerr.lt.10)
     $     write(*,*)'GETISSUD Warning: flavor ',kfl,' not supported'
        nerr=nerr+1
        getissud=1
        return
      endif

      if(x(1).lt.points(1,ipoints).or.
     $   x(1).gt.points(nx2,ipoints).and.nerr.lt.10)
     $   then
        write(*,*) 'GETISSUD Warning: extrapolation in x2: ',x2
        nerr=nerr+1
      endif

      if(x(2).lt.points(nx2+1,ipoints).or.
     $   x(2).gt.points(nx2+nx1,ipoints)
     $   .and.nerr.lt.10) then
        write(*,*) 'GETISSUD Warning: extrapolation in x1: ',x1
        nerr=nerr+1
      endif

      if(kkfl.eq.5.and.pt2.lt.22.3109)then
        getissud=1d0
        return
      endif

      if(kkfl.eq.5.and.x1.gt.0.6)then
        getissud=0d0
        return
      endif

      if(x(3).lt.points(nx2+nx1+1,ipoints)) then
        write(*,*) 'GETISSUD Error! pt2 = ',exp(x(3)),' < ',
     $     exp(points(nx2+nx1+1,ipoints)),' = min(pt2). Not allowed!'
        write(*,*) 'You need to regenerate grid with new pt2min.'
        stop
      endif

      if(x(3).lt.points(nx2+nx1+1,ipoints).or.
     $   x(3).gt.points(nx2+nx1+npt2,ipoints)
     $   .and.nerr.lt.10) then
        write(*,*) 'GETISSUD Warning: extrapolation in pt2: ',pt2
        nerr=nerr+1
      endif


c   Find nearest points by binary method
c   x2
      ilo=1
      ihi=nx2
      do while(ihi.gt.ilo+1)
        ix2=ilo+(ihi-ilo)/2
        if(x(1).gt.points(ix2,ipoints))then
          ilo=ix2
        else
          ihi=ix2
        endif
      enddo
      if(x(1).lt.points(ix2,ipoints))
     $   ix2=ix2-1
      ix2=max(2,min(ix2,nx2-2))

c      print *,'x2: ',ix2,x(1),(points(i,ipoints),i=ix2-1,ix2+2)

c   x1
      ilo=1
      ihi=nx1
      do while(ihi.gt.ilo+1)
        ix1=ilo+(ihi-ilo)/2
        if(x(2).gt.points(nx2+ix1,ipoints))then
          ilo=ix1
        else
          ihi=ix1
        endif
      enddo
      if(x(2).lt.points(nx2+ix1,ipoints))
     $   ix1=ix1-1
      ix1=max(2,min(ix1,nx1-2))

      do while(kkfl.eq.5.and.
     $   points(nx2+ix1+2,ipoints).gt.0.6)
        ix1=ix1-1
      enddo

c      print *,'x1: ',ix1,x(2),(points(nx2+i,ipoints),i=ix1-1,ix1+2)

c   pt2
      ilo=1
      ihi=npt2
      do while(ihi.gt.ilo+1)
        ipt2=ilo+(ihi-ilo)/2
        if(x(3).gt.points(nx2+nx1+ipt2,ipoints))then
          ilo=ipt2
        else
          ihi=ipt2
        endif
      enddo
      if(x(3).lt.points(nx2+nx1+ipt2,ipoints))
     $   ipt2=ipt2-1
      ipt2=max(2,min(ipt2,npt2-2))

      do while(kkfl.eq.5.and.
     $   exp(points(nx2+nx1+ipt2-1,ipoints)).lt.22.3109)
        ipt2=ipt2+1
      enddo
c      print *,'pt2: ',ipt2,x(3),(points(nx2+nx1+i,ipoints),i=ipt2-1,ipt2+2)
c      print *,'pt: ',ipt2,exp(x(3)/2),
c     $   (exp(points(nx2+nx1+i,ipoints)/2),i=ipt2-1,ipt2+2)

C   Now perform inter-/extra-polation

C   Start with x2, which should have the flattest distribution
C   Calculate sud(x2,ax1,apt2) for the 4x4 apt2 and ax1
C   Then continue with pt2 and calculate sud(x2,ax1,pt2)
C   for the 4 ax1
C   Finally calculate sud(x2,x1,pt2)

      do i=1,4
        do j=1,4
c          print *,'x1,pt:',points(nx2+ix1-2+i,ipoints),
c     $       exp(points(nx2+nx1+ipt2-2+j,ipoints)/2)
          call splint2(sudgrid(ix2-1,ix1-2+i,ipt2-2+j,kkfl),
     $       points(ix2-1,ipoints),4,x(1),smallgrid(j,i))
          smallgrid(j,i)=max(0d0,min(1d0,smallgrid(j,i)))
        enddo
      enddo

      do i=1,4
        call splint2(smallgrid(1,i),
     $     points(nx2+nx1+ipt2-1,ipoints),4,x(3),minigrid(i))
          minigrid(i)=max(0d0,min(1d0,minigrid(i)))
      enddo

      call splint2(minigrid,
     $   points(nx2+ix1-1,ipoints),4,x(2),getissud)
      getissud=max(0d0,min(1d0,getissud))
      
c      print *,'Result: ',getissud

      return
      end


      subroutine splint2(ypoints,xpoints,npoints,x,ans)
      implicit none

C   arguments
      integer npoints
      double precision ypoints(npoints),xpoints(npoints)
      double precision x,ans
C   local variables
      double precision a0,a1,a2,sd
      integer ifail,i

      CALL DLSQP2(npoints,xpoints,ypoints,a0,a1,a2,sd,ifail)

c      print *,'Point, interpolation:'
c      do i=1,npoints
c        print *,exp(xpoints(i)),ypoints(i),
c     $     a0+a1*xpoints(i)+a2*xpoints(i)**2
c      enddo

      ans=a0+a1*x+a2*x**2
c      print *,x,ans

      return
      end
