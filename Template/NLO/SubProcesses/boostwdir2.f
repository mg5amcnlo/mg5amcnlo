      subroutine boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
c chybstmo = chybst-1; if it can be computed analytically it improves
c the numerical accuracy
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xin(0:3),xout(0:3)
      real*8 tmp,en,pz
      integer i
c
      if(abs(xd(1)**2+xd(2)**2+xd(3)**2-1).gt.1.d-6)then
        write(*,*)'Error #1 in boostwdir2',xd
        stop
      endif
c
      en=xin(0)
      pz=xin(1)*xd(1)+xin(2)*xd(2)+xin(3)*xd(3)
      xout(0)=en*chybst-pz*shybst
      do i=1,3
        xout(i)=xin(i)+xd(i)*(pz*chybstmo-en*shybst)
      enddo
c
      return
      end

