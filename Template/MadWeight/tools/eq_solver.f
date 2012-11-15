      subroutine degree4(a,b,c,d,e,x1,x2,x3,x4)
c
      implicit none
c
c     parameter
c
      double precision pi
      parameter (pi=3.14159265358979323846d0)
c
c     arguments
c
      double precision a,b,c,d,e
      double complex x1,x2,x3,x4
c
c     local
c
      double precision p,q,r,g2,g1,g0
      double precision f3,f2,f1,f0
      double complex y0,y1,y2
      double complex compl_a,compl_b,compl_c
      double precision y

c      write(*,*) 'a: ',a
c      write(*,*) 'b: ',b
c      write(*,*) 'c: ',c
c      write(*,*) 'd: ',d
c      write(*,*) 'e: ',e

      if (b.eq.0d0.and.c.eq.0d0.and.d.eq.0d0.and.e.eq.0d0) then
      x1=dcmplx(0d0)
      x2=dcmplx(0d0)
      x3=dcmplx(0d0)
      x4=dcmplx(0d0)
      return
      elseif (b.eq.0d0.and.c.eq.0d0.and.d.eq.0d0.and.e.ne.0d0) then
      if (e.lt.0d0) then
      x1=dcmplx(dsqrt(dsqrt(-e)))
      x2=dcmplx(dsqrt(dsqrt(-e)))*
     & dcmplx(dcos(pi/2d0),dsin(pi/2d0))
      x3=dcmplx(dsqrt(dsqrt(-e)))
     & *dcmplx(dcos(pi),dsin(pi))
      x4=dcmplx(dsqrt(dsqrt(-e)))
     & *dcmplx(dcos(3d0*pi/2d0),dsin(3d0*pi/2d0))
      return
      elseif (e.gt.0d0) then
      x1=dcmplx(dsqrt(dsqrt(e)))*
     & dcmplx(dcos(pi/4d0),dsin(pi/4d0))
      x2=dcmplx(dsqrt(dsqrt(e)))*
     & dcmplx(dcos(5d0*pi/4d0),dsin(5d0*pi/4d0))
      x3=dcmplx(dsqrt(dsqrt(e)))
     & *dcmplx(dcos(3d0*pi/4d0),dsin(3d0*pi/4d0))
      x4=dcmplx(dsqrt(dsqrt(e)))
     & *dcmplx(dcos(7d0*pi/4d0),dsin(7d0*pi/4d0))
      return
      endif
      endif

      p=-3d0*b**2/(8d0*a**2)+c/a
      q=b**3/(8d0*a**3)-b*c/(2d0*a**2)+d/a
      r=-3d0*b**4/(256d0*a**4)+b**2*c/(16d0*a**3)
     & -d*b/(4d0*a**2)+e/a
c
c     coefficients of the degree 3 polynome P(y)
c
      f3=1d0
      f2=5d0*p/2d0
      f1=2d0*p**2-r
      f0=p**3/2d0-p*r/2d0-q**2/8d0

c      write(*,*) 'f3 ',f3
c      write(*,*) 'f2 ',f2
c      write(*,*) 'f1 ',f1
c      write(*,*) 'f0 ',f0

      call degree3(f3,f2,f1,f0,y0,y1,y2)

c      write(*,*) 'y0 ', y0
      y=dble(y0)
c      write(*,*) 'y ', y
c      write(*,*) 'p ', p
c      write(*,*) 'p+2d0*y ', p+2d0*y

      if ((p+2d0*y).gt.0d0) then 
c      write(*,*) 'p+2d0*y0 ', p+2d0*y
      g2=1d0
      g1=-dsqrt(p+2d0*y)
      g0=p+y+q/(2d0*dsqrt(p+2d0*y))

c       write(*,*) 'g2', g2
c       write(*,*) 'g1', g1
c       write(*,*) 'g0', g0
      call degree2_real(g2,g1,g0,x1,x2)

c      write(*,*) 'x1',x1
c      write(*,*) 'x2',x2

      g2=1d0
      g1=dsqrt(p+2d0*y)
      g0=p+y-q/(2d0*dsqrt(p+2d0*y))

      call degree2_real(g2,g1,g0,x3,x4)

      elseif((p+2d0*y).lt.0d0) then
      compl_a=dcmplx(1d0)
      compl_b=dcmplx(0d0,-dsqrt(-(p+2d0*y)))
      compl_c=dcmplx(p+y,q/(2d0*dsqrt(-(p+2d0*y))))
      call degree2_complex(compl_a,compl_b,compl_c,x1,x2)

      compl_a=dcmplx(1d0)
      compl_b=dcmplx(0d0,dsqrt(-(p+2d0*y)))
      compl_c=dcmplx(p+y,-q/(2d0*dsqrt(-(p+2d0*y))))
      call degree2_complex(compl_a,compl_b,compl_c,x3,x4)
      endif

      x1=x1-b/(4d0*a)
      x2=x2-b/(4d0*a)
      x3=x3-b/(4d0*a)
      x4=x4-b/(4d0*a)

      return
      end


      subroutine degree3(a,b,c,d,x1,x2,x3)
c
      implicit none
c
c     parameter
c
      double precision pi
      parameter (pi=3.14159265358979323846d0)
c
c     arguments
c
      double precision a,b,c,d
      double complex x1,x2,x3
c
c     local
c
      double precision p,q,delta,u,v
      double complex phase
c
c
      if(b.eq.0d0.and.c.eq.0d0.and.d.eq.0d0) then
      x1=dcmplx(0d0) 
      x2=dcmplx(0d0) 
      x3=dcmplx(0d0)
      return 
      elseif (b.eq.0d0.and.c.eq.0d0.and.d.ne.0d0) then
      if (d.lt.0d0) then
      x1=dcmplx( (-d)**(1d0/3d0)  )
      x2=dcmplx((-d)**(1d0/3d0))
     & *dcmplx(dcos(2d0*pi/3d0),dsin(2d0*pi/3d0))
      x3=dcmplx( (-d)**(1d0/3d0)  )
     & *dcmplx(dcos(4d0*pi/3d0),dsin(4d0*pi/3d0))
      return
      elseif (d.gt.0d0) then
      x1=dcmplx( -d**(1d0/3d0)  )
      x2=dcmplx( -d**(1d0/3d0)  )
     & *dcmplx(dcos(2d0*pi/3d0),dsin(2d0*pi/3d0))
      x3=dcmplx( -d**(1d0/3d0)  )
     & *dcmplx(dcos(4d0*pi/3d0),dsin(4d0*pi/3d0))
      return
      endif
      endif
      

      p=-b**2/(3d0*a**2)+c/a
      q=b/(27d0*a)*(2d0*b**2/a**2-9d0*c/a)+d/a      
      delta=q**2+4d0*p**3/27d0

c       write(*,*) 'p: ',p
c       write(*,*) 'q: ',q

      if (delta.gt.0d0) then
c
      if ((-q+dsqrt(delta)).ge.0d0 ) then 
      u=((-q+dsqrt(delta))/2d0)**(1d0/3d0)
      elseif ((-q+dsqrt(delta)).le.0d0) then
      u=-((q-dsqrt(delta))/2d0)**(1d0/3d0)
      endif
c
      if ((-q-dsqrt(delta)).ge.0d0) then
      v=(-(q+dsqrt(delta))/2d0)**(1d0/3d0)
      
      elseif((q+dsqrt(delta)).ge.0d0) then
      v=-((q+dsqrt(delta))/2d0)**(1d0/3d0)
      endif
c
      phase=dcmplx(-1d0/2d0,dsqrt(3d0)/2d0)
      x1=dcmplx(u+v)
      x2=phase*dcmplx(u)+dconjg(phase)*dcmplx(v)
      x3=(phase**2)*dcmplx(u)+dconjg(phase**2)*dcmplx(v)

      elseif (delta.eq.0d0) then
      x1=3d0*q/p
      x2=-3d0*q/2d0*p
      x3=-3d0*q/2d0*p
      
      elseif (delta.lt.0d0) then
      x1=2d0*dsqrt(-p/3d0)*dcos(1d0/3d0*
     & (dacos(-q*dsqrt(27d0/(-p**3))/2d0)))
      x2=2d0*dsqrt(-p/3d0)*dcos(1d0/3d0*
     & (dacos(-q*dsqrt(27d0/(-p**3))/2d0) +2d0*pi))
       x3=2d0*dsqrt(-p/3d0)*dcos(1d0/3d0*
     &  (dacos(-q*dsqrt(27d0/(-p**3))/2d0) +4d0*pi))
      endif

      x1=x1-dcmplx(b/(3d0*a))
      x2=x2-dcmplx(b/(3d0*a))
      x3=x3-dcmplx(b/(3d0*a))
     

      return
      end

      subroutine degree2_real(a,b,c,x1,x2)
c
c     this program compute the roots x1, x2 
c     of the equation aX^2+bX+C=0
c
      implicit none
c
c     argument
c
      double precision a,b,c
      double complex x1,x2
c
c     local
c
      double precision rho

      rho=b**2-4d0*a*c
      if (rho.ge.0d0) then
      x1=dcmplx((-b+sqrt(rho))/(2d0*a))
      x2=dcmplx((-b-sqrt(rho))/(2d0*a))
      elseif (rho.lt.0d0) then 
      x1=dcmplx(-b/(2d0*a),sqrt(-rho)/(2d0*a))
      x2=dcmplx(-b/(2d0*a),-sqrt(-rho)/(2d0*a))
      endif
       return
       end


      subroutine degree2_complex(a,b,c,x1,x2)
c
c     this program compute the roots x1, x2
c     of the equation aX^2+bX+C=0
c
      implicit none
c
c     parameter
c
      double precision zero,pi
      parameter (zero=0d0,pi=3.14159265358979323846d0)
      double complex i_part
      parameter (i_part=(0d0,1d0))
c
c     argument
c
      double complex a,b,c
      double complex x1,x2
c
c     local
c
      double complex rho,j1,j2
      double precision phi,phase1,phase2
      double precision module_rho,real_rho,imag_rho

      rho=b**2/4d0/a**2-c/a
      real_rho=dble(rho+dconjg(rho))/2d0
      imag_rho=dble((rho-dconjg(rho))/(2d0*i_part))
      module_rho=dsqrt(real_rho**2+imag_rho**2)

c      write(*,*) 'rho ', rho
c      write(*,*) 'real_rho ', real_rho
c      write(*,*) 'imag_rho ', imag_rho
c      write(*,*) 'module_rho ', module_rho

      if(real_rho.gt.zero) then
      phi=datan(imag_rho/real_rho)
      else if(real_rho.lt.zero) then
      phi=datan(imag_rho/real_rho)+pi
      else if(real_rho.eq.zero.and.imag_rho.GE.zero) then
      phi=pi/2d0
      else if(real_rho.eq.zero.and.imag_rho.lt.zero) then
      phi=-pi/2d0
      endif
      if(phi.lt.zero) phi=phi+2*pi

c      write(*,*) 'phi', phi
      phase1=phi/2d0
      phase2=phi/2d0+pi

      j1=dcmplx(dcos(phase1),dsin(phase1))
      j2=dcmplx(dcos(phase2),dsin(phase2))


      x1=-b/(2d0*a)+dsqrt(module_rho)*j1
      x2=-b/(2d0*a)+dsqrt(module_rho)*j2

       return
       end

       subroutine solve_2quadeqs(g11,g22,g12,g10,g20,g00,
     & h11,h22,h12,h10,h20,h00,E1,E2,sol)

       implicit none
c
c     arguments
c
      double precision g11,g12,g10,g22,g20,g00
      double precision h11,h12,h10,h22,h20,h00
      double complex E1(4),E2(4)
      logical sol(4)
c
c     local
c
      integer j
      double precision a4,a3,a2,a1,a0
      double complex w2,w1,w0
      double complex x1,x2,x3,x4,root1,root2
c
c       write(*,*) 'g11', g11
c       write(*,*) 'g22', g22
c       write(*,*) 'g12', g12
c       write(*,*) 'g10', g10
c       write(*,*) 'g20', g20
c       write(*,*) 'g00', g00


c       write(*,*) 'h11', h11
c       write(*,*) 'h22', h22
c       write(*,*) 'h12', h12
c       write(*,*) 'h10', h10
c       write(*,*) 'h20', h20
c       write(*,*) 'h00', h00

c
      a0=g11**2*h00**2 - g10*g11*h00*h10 + g00*g11*h10**2 +
     -     g10**2*h00*h11 - 2*g00*g11*h00*h11 - g00*g10*h10*h11 +
     -     g00**2*h11**2
c
      a1=-g11*g12*h00*h10 + g11*g20*h10**2 +
     -     2*g10*g12*h00*h11 - 2*g11*g20*h00*h11 - g00*g12*h10*h11 -
     -     g10*g20*h10*h11 + 2*g00*g20*h11**2 - g10*g11*h00*h12 +
     -     2*g00*g11*h10*h12 - g00*g10*h11*h12 + 2*g11**2*h00*h20 -
     -     g10*g11*h10*h20 + g10**2*h11*h20 - 2*g00*g11*h11*h20
c
      a2=g11*g22*h10**2 + g12**2*h00*h11 - 2*g11*g22*h00*h11 -
     -     g12*g20*h10*h11 - g10*g22*h10*h11 + g20**2*h11**2 +
     -     2*g00*g22*h11**2 - g11*g12*h00*h12 + 2*g11*g20*h10*h12 -
     -     g00*g12*h11*h12 - g10*g20*h11*h12 + g00*g11*h12**2 -
     -     g11*g12*h10*h20 + 2*g10*g12*h11*h20 - 2*g11*g20*h11*h20 -
     -     g10*g11*h12*h20 + g11**2*h20**2 + 2*g11**2*h00*h22 -
     -     g10*g11*h10*h22 + g10**2*h11*h22 - 2*g00*g11*h11*h22
c
      a3=-(g12*g22*h10*h11) + 2*g20*g22*h11**2 +2*g11*g22*h10*h12 -
     -     g12*g20*h11*h12 - g10*g22*h11*h12 + g11*g20*h12**2 +
     -     g12**2*h11*h20 - 2*g11*g22*h11*h20 - g11*g12*h12*h20 -
     -     g11*g12*h10*h22 + 2*g10*g12*h11*h22 - 2*g11*g20*h11*h22 -
     -     g10*g11*h12*h22 + 2*g11**2*h20*h22
c
      a4=g22**2*h11**2 - g12*g22*h11*h12 + g11*g22*h12**2 +
     -     g12**2*h11*h22 - 2*g11*g22*h11*h22 - g11*g12*h12*h22 +
     -     g11**2*h22**2

c        write(*,*) 'a4',a4
c        write(*,*) 'a3',a3
c        write(*,*) 'a2',a2
c        write(*,*) 'a1',a1
c        write(*,*) 'a0',a0


       call degree4(a4,a3,a2,a1,a0,x1,x2,x3,x4)

c       write(*,*) 'x1 ',x1
c       write(*,*) 'x2 ',x2
c       write(*,*) 'x3 ',x3
c       write(*,*) 'x4 ',x4
c
c      reorganizing the roots
c
c       if (x1.eq.x3) then
c       temp=x2
c       x2=x3
c       x3=temp
c       elseif (x1.eq.x4) then
c       temp=x2
c       x2=x4
c       x4=temp
c       endif

       E2(1)=x1
       E2(2)=x2
       E2(3)=x3
       E2(4)=x4


c       write(*,*) 'E2(1) ',E2(1)
c       write(*,*) 'E2(2) ',E2(2)
c       write(*,*) 'E2(3) ',E2(3)
c       write(*,*) 'E2(4) ',E2(4)
c
c      Now, determine E1
c

       sol(1)=.true.
       if((h11*(g10+g12*x1)-g11*(h10+h12*x1)).ne.0d0) then

       E1(1)=(-h11*(g22*x1**2+g20*x1+g00)+g11*(h22*x1**2+h20*x1+h00))
     & /(h11*(g10+g12*x1)-g11*(h10+h12*x1))
c
       elseif ((-h11*(g22*x1**2+g20*x1+g00)+
     & g11*(h22*x1**2+h20*x1+h00)).eq.0d0) then
c
c      in this case, the two equations are not independent
c
       w2=dcmplx(g11)
       w1=g12*x1+g10
       w0=g22*x1**2+g20*x1+g00
       call degree2_complex(w2,w1,w0,root1,root2)
       E1(1)=root1
       else
       write(*,*)  'warning: no solution to the 2 quadr. eqs'
       sol(1)=.false.
       endif


       sol(2)=.true.
       if((h11*(g10+g12*x2)-g11*(h10+h12*x2)).ne.0d0) then
       E1(2)=(-h11*(g22*x2**2+g20*x2+g00)+g11*(h22*x2**2+h20*x2+h00))
     & /(h11*(g10+g12*x2)-g11*(h10+h12*x2))
       elseif ((-h11*(g22*x2**2+g20*x2+g00)+
     & g11*(h22*x2**2+h20*x2+h00)).eq.0d0) then

       w2=dcmplx(g11)
       w1=g12*x2+g10
       w0=g22*x2**2+g20*x2+g00
       call degree2_complex(w2,w1,w0,root1,root2)
       E1(2)=root2
       else
       write(*,*)  'warning: no solution to the 2 quadr. eqs'
       sol(2)=.false.
       endif
c

       sol(3)=.true.
       if((h11*(g10+g12*x3)-g11*(h10+h12*x3)).ne.0d0) then
       E1(3)=(-h11*(g22*x3**2+g20*x3+g00)+g11*(h22*x3**2+h20*x3+h00))
     & /(h11*(g10+g12*x3)-g11*(h10+h12*x3))
       elseif ((-h11*(g22*x3**2+g20*x3+g00)+
     & g11*(h22*x3**2+h20*x3+h00)).eq.0d0) then

       w2=dcmplx(g11)
       w1=g12*x3+g10
       w0=g22*x3**2+g20*x3+g00
       call degree2_complex(w2,w1,w0,root1,root2)
       E1(3)=root1
       else
       write(*,*)  'warning: no solution to the 2 quadr. eqs'
       sol(3)=.false.
       endif

c

       sol(4)=.true.
       if((h11*(g10+g12*x4)-g11*(h10+h12*x4)).ne.0d0) then
       E1(4)=(-h11*(g22*x4**2+g20*x4+g00)+g11*(h22*x4**2+h20*x4+h00))
     & /(h11*(g10+g12*x4)-g11*(h10+h12*x4))
       elseif ((-h11*(g22*x4**2+g20*x4+g00)+
     & g11*(h22*x4**2+h20*x4+h00)).eq.0d0) then

       w2=dcmplx(g11)
       w1=g12*x4+g10
       w0=g22*x4**2+g20*x4+g00
       call degree2_complex(w2,w1,w0,root1,root2)
       E1(4)=root2
       else
       write(*,*)  'warning: no solution to the 2 quadr. eqs'
       sol(4)=.false.
       endif

c       write(*,*) 'E2(1) ',E2(1)
c       write(*,*) 'E2(2) ',E2(2)
c       write(*,*) 'E2(3) ',E2(3)
c       write(*,*) 'E2(4) ',E2(4)

c       write(*,*) 'sol 1 ', sol(1)
c       write(*,*) 'sol 2 ', sol(2)
c       write(*,*) 'sol 3 ', sol(3)
c       write(*,*) 'sol 4 ', sol(4)

c       do j=1,4
c       write(*,*) g11*E1(j)**2+g22*E2(j)**2+g12*E1(j)*E2(j)
c     & +g10*E1(j)+g20*E2(j)+g00
c       enddo
c       write(*,*) 'verif  eq 2 '
c       do j=1,4
c       write(*,*) h11*E1(j)**2+h22*E2(j)**2+h12*E1(j)*E2(j)
c     & +h10*E1(j)+h20*E2(j)+h00
c       enddo


        return
       end
c
