      subroutine upsxxx(t2,sc,gt, xm,xw,t1)

c  Subroutines for graviton phase space integration
c  KEK 2009.11
c
      implicit none
      double complex t1(18), t2(18), sc(3), vertex, tc2(18)
      double complex gt(2)
      double precision xm,xw,xmass

      double complex ft(6,4),ft2(6,4)
      double precision ps1(4), pt2(4),PT(4),PTT2,PG(0:3)
      integer i
      double complex cZero
      double precision rZero, rTwo,Pi
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      double precision PADD
      double precision L_ADD,NADD,MGLOW,MGUP
      external txxxxx

      Pi=dacos(-1.0d0)
      L_ADD=dimag(gt(1))
      NADD=dreal(gt(1))
      MGUP=dimag(gt(2))
      MGLOW=dreal(gt(2))


      ps1(1) = dreal(sc(2))
      ps1(2) = dreal(sc(3))
      ps1(3) = dimag(sc(3))
      ps1(4) = dimag(sc(2))

      pt2(1) = -dreal(t2(17))
      pt2(2) = -dreal(t2(18))
      pt2(3) = -dimag(t2(18))
      pt2(4) = -dimag(t2(17))

      PG(0)=ps1(1)-pt2(1)
      PG(1)=ps1(2)-pt2(2)
      PG(2)=ps1(3)-pt2(3)
      PG(3)=ps1(4)-pt2(4)

      PTT2=PG(0)**2-PG(1)**2-PG(2)**2-PG(3)**2
      xmass=dsqrt(PTT2)

       t1(17) = dcmplx(PG(0),PG(3))
       t1(18) = dcmplx(PG(1), PG(2))

      if(xmass.lt.MGLOW.or.xmass.gt.MGUP) then
      do i=1,16
      t1(i)=dcmplx(0.0d0,0.0d0)
      enddo
      return
      endif

      CALL txxxxx(PG,xmass,INT(t2(1)),+1 , t1)


       if(INT(NADD).eq.2) then
         PADD=2.0d0*Pi
        elseif(INT(NADD).eq.3) then
         PADD=4.0d0*Pi
        elseif(INT(NADD).eq.4) then
         PADD=2.0d0*Pi**2
        elseif(INT(NADD).eq.5) then
          PADD=8.0d0/3.0d0*Pi**2
        elseif(INT(NADD).eq.6) then
           PADD=Pi**3
        else
        print *, "OUT CASE"
        stop
        endif 

 
        do i=1,16
        t1(i)=-1.0d0*t1(i)*
     & dsqrt( 
     & 2.0d0*Pi*8.0d0*Pi  ! to compensate the decay phase factor
     &* PADD/L_ADD**NADD*xmass**(NADD-1)  ! density factor for d=4 case
     &/2.0d0/xmass)   ! dm=dm^2/2/m    
        enddo

      return
      end
