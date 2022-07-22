      function derivative(fun,xi,hi,idiri,xmini,xmaxi,erroro)
c returns the derivative of f
c
c fun    (INPUT)= function to differentiate
c xi     (INPUT)= point where to compute the derivative
c hi     (INPUT)= initial stepsize
c idiri  (INPUT)= type of derivative (see below)
c xmini  (INPUT)= lower bound in x
c xmaxi  (INPUT)= upper bound in x
c error  (OUTPUT)= estimated error
c
c idiri =-1 --> right derivative
c idiri = 0 --> both-sides derivative
c idiri = 1 --> left derivative
c idiri = 2 --> higher-order both-sides left derivative
      implicit none
      real*8 derivative,fun,xi,hi,xmini,xmaxi,erroro
      integer idiri
c
      integer idir,i,j,itn,jmax
      parameter(itn=2,jmax=2)
c itn = 2 guarantees a fast code. itns >> 2 gives more precise
c results in extreme configurations, but slows the code down
c linearly
      real*8 x,h,hh,error,con,big,dd,
     &xmin,xmax,tiny,res(0:itn,jmax),er(0:itn)
      parameter(con=3d0,big=1d40)
      parameter(tiny=1.d-8)
      external fun
c
      x=xi
      h=hi
      xmin=xmini
      xmax=xmaxi
      idir=idiri
      if(h.eq.0d0)then
         write(*,*)'Error #1 in function derivative'
         stop
      endif
      if(idir.lt.-1.or.idir.gt.2)then
         write(*,*)'Error #2 in function derivative',idir
         stop
      endif
      if(xmin.ge.xmax)then
         write(*,*)'Error #3 in function derivative',x,xmin,xmax
         stop
      endif
c Set the derivative equal to zero and error equal to one if outside range
      if(x.lt.xmin.or.x.gt.xmax)then
        derivative=0.d0
        erroro=1.d0
        return
      endif
c If close to borders of range, use left or right first-order derivative;
c may include higher orders if implemented from the left or right only
      if(x.le.(xmin+tiny))then
        x=xmin+tiny
        idir=-1
      elseif(x.ge.(xmax-tiny))then
        x=xmax-tiny
        idir=1
      endif
c
      h=min(h,min(xmax-x,x-xmin))
      error=big
      dd=0.d0
      do i=0,itn
         hh=h/1d1**(6d0*i/itn)
         do j=1,jmax
            if(idir.eq.0)then
               res(i,j)=(fun(x+hh)-fun(x-hh))/(2d0*hh)
            elseif(idir.eq.-1)then
               res(i,j)=(fun(x+hh)-fun(x))/hh
            elseif(idir.eq.1)then
               res(i,j)=(fun(x)-fun(x-hh))/hh
            elseif(idir.eq.2)then
c second-order formula: yet higher-order formulae exist
c but they require too many evaluations of f, which slows
c the code down quite significantly
               res(i,j)=(fun(x-2*hh)-8*fun(x-hh)
     &                  -fun(x+2*hh)+8*fun(x+hh))/(12d0*hh)
            endif
            if(j.lt.jmax)hh=hh/con
         enddo
         er(i)=abs(abs(res(i,1))-abs(res(i,2)))/
     &         max(abs(res(i,1)),abs(res(i,2)))
         if(er(i).lt.error.and.er(i).ne.0d0)then
            error=er(i)
            dd=(res(i,1)+res(i,2))/2d0
         endif
      enddo
      derivative=dd
      erroro=error*2*abs(dd)
c
      return
      end
