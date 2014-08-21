      subroutine qlI4DNS41(Y,musq,Ival0)
      implicit none
      include 'qlconstants.f'
c-----Implementation of Eq.~(41) of
c----- %\cite{Denner:1991qq}
c----- %\cite{Denner:1991qq}
c----- \bibitem{Denner:1991qq}
c----- A.~Denner, U.~Nierste and R.~Scharf,
c----- %``A Compact expression for the scalar one loop four point function,''
c----- Nucl.\ Phys.\  B {\bf 367}, 637 (1991).
c----- %%CITATION = NUPHA,B367,637;%%
      double complex Ival0,discr,wlog(2),cln,z(2),k(4,4),lnsum,
     . qlLi2omprod,a,b,c,d,bsq,fourac
      double precision musq,Y(4,4),iep
      integer i,j

      do i=1,4
      do j=1,4
      k(i,j)=dcmplx(2d0*Y(i,j)/musq)
      enddo
      enddo

      a=k(2,4)*k(3,4)
      b=k(1,3)*k(2,4)+k(1,2)*k(3,4)-k(1,4)*k(2,3)
      c=k(1,2)*k(1,3)
      d=k(2,3)
      bsq=b**2
      fourac=4d0*a*c
      discr=sqrt(bsq-fourac)     
 
      if (abs(discr) .lt. 1d-10*max(dble(bsq),dble(fourac))) then
      z(1)=0.5d0*b/a      
      wlog(1)=dcmplx(dreal(cln(z(1),+1d0)))
      Ival0=
     . +k(3,4)*(cln(k(3,4),-1d0)+wlog(1)-cln(k(1,3),-1d0))
     . /(k(3,4)*z(1)-k(1,3))
     . +k(2,4)*(cln(k(2,4),-1d0)+wlog(1)-cln(k(1,2),-1d0))
     . /(k(2,4)*z(1)-k(1,2))
     . -(wlog(1)
     . +cln(k(2,3),-1d0)+cln(k(1,4),-1d0)
     . -cln(k(1,3),-1d0)-cln(k(1,2),-1d0))/z(1)

      Ival0=Ival0/(musq**2*a)
      return
      else
C----wlogi=log(-xi),zi=-xi
      z(1)=0.5d0*(b-discr)/a
      z(2)=0.5d0*(b+discr)/a
      iep=sign(1d0,dreal(d))
C-----z(1) comes with + i*ep*d
C-----z(2) comes with - i*ep*d
      wlog(1)=cln(z(1),+iep)
      wlog(2)=cln(z(2),-iep)
      lnsum=+cln(k(1,2),-1d0)+cln(k(1,3),-1d0)
     .      -cln(k(1,4),-1d0)-cln(k(2,3),-1d0)
      Ival0=czip
      do j=1,2
      iep=-dfloat(2*j-3)*iep
      Ival0=Ival0+dfloat(2*j-3)*(
     . -0.5d0*wlog(j)**2+wlog(j)*lnsum
     . -qlLi2omprod(dble(k(3,4)),dble(k(1,3)),z(j),iep)
     . -qlLi2omprod(dble(k(2,4)),dble(k(1,2)),z(j),iep))

      
      enddo

      Ival0=Ival0/(musq**2*discr)
      endif
      return
      end
