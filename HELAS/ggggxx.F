      subroutine ggggxx(ga,gb,gc,gd,g, vertex)
c
c This subroutine computes the portion of the amplitude of the four-point 
c coupling of 4 massless color octet gauge bosons (gluons) corresponding 
c to the color structure f^{a,b,e} f{c,d,e}. 
c To optain the complete amplitude, this coupling must be called three
c times (once for each color structure) with the following permutations:
c	call ggggxx(ga,gb,gc,gd,g,v1)
c       call ggggxx(ga,gc,gd,gb,g,v2)
c       call ggggxx(ga,gd,gb,gc,g,v3)
c
c	f^{a,b,e} f{c,d,e}*v1+
c	f^{a,c,e} f{d,b,e}*v2+
c	f^{a,d,e} f{b,c,e}*v3
c (See 2.9.1 of the manual for more information).
c                                                                       
c input:                                                                
c       complex ga(0:3)        : Boson with adjoint color index a 
c       complex gb(0:3)        : Boson with adjoint color index b
c       complex gc(0:3)        : Boson with adjoint color index c 
c       complex gd(0:3)        : Boson with adjoint color index d
c       real    g              : coupling of w31 with w-/w+             
c
      implicit none
      double complex ga(6),gb(6),gc(6),gd(6),vertex
      double complex dv1(0:3),dv2(0:3),dv3(0:3),dv4(0:3),
     &     dvertx,v12,v13,v14,v23,v24,v34
      double precision pga(0:3),pgb(0:3),pgc(0:3),pgd(0:3),g

      save dv1,dv2,dv3, dv4
c      save dv1,dv2,dv3,dv4,dvertx,v12,v13,v14,v23,v24,v34

#ifdef HELAS_CHECK
      double precision pm
      double precision epsi
      parameter( epsi = 2.0d-5 )
      double precision rZero
      parameter( rZero = 0.0d0 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      pga(0) = dble( ga(5))
      pga(1) = dble( ga(6))
      pga(2) = dimag(ga(6))
      pga(3) = dimag(ga(5))
      pgb(0) = dble( gb(5))
      pgb(1) = dble( gb(6))
      pgb(2) = dimag(gb(6))
      pgb(3) = dimag(gb(5))
      pgc(0) = dble( gc(5))
      pgc(1) = dble( gc(6))
      pgc(2) = dimag(gc(6))
      pgc(3) = dimag(gc(5))
      pgd(0) = dble( gd(5))
      pgd(1) = dble( gd(6))
      pgd(2) = dimag(gd(6))
      pgd(3) = dimag(gd(5))

      if (  abs(ga(1))+abs(ga(2))
     &     +abs(ga(3))+abs(ga(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : ga in ggggxx is zero vector'
      endif
      if ( abs(ga(5))+abs(ga(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : ga in ggggxx has zero momentum'
      endif
      if (  abs(gb(1))+abs(gb(2))
     &     +abs(gb(3))+abs(gb(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : gb in ggggxx is zero vector'
      endif
      if ( abs(gb(5))+abs(gb(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : gb in ggggxx has zero momentum'
      endif
      if (  abs(gc(1))+abs(gc(2))
     &     +abs(gc(3))+abs(gc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : gc in ggggxx is zero vector'
      endif
      if ( abs(gc(5))+abs(gc(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : gc in ggggxx has zero momentum'
      endif
      if (  abs(gd(1))+abs(gd(2))
     &     +abs(gd(3))+abs(gd(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : gd in ggggxx is zero vector'
      endif
      if ( abs(gd(5))+abs(gd(5)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : gd in ggggxx has zero momentum'
      endif
      pm = max( abs(pga(0)),abs(pgb(0)),abs(pgc(0)),abs(pgd(0)),
     &          abs(pga(1)),abs(pgb(1)),abs(pgc(1)),abs(pgd(1)),
     &          abs(pga(2)),abs(pgb(2)),abs(pgc(2)),abs(pgd(2)),
     &          abs(pga(3)),abs(pgb(3)),abs(pgc(3)),abs(pgd(3)) )
      if (  abs(ga(5)+gb(5)+gc(5)+gd(5))
     &     +abs(ga(6)+gb(6)+gc(6)+gd(6)).ge.pm*epsi) then
         write(stdo,*)
     &        ' helas-error : ga,gb,gc,gd in ggggxx'
         write(stdo,*)
     &        '             : have not balanced momenta'
      endif
      if ( g.eq.rZero ) then
         write(stdo,*) ' helas-error : g in ggggxx is zero coupling'
      endif
#endif

      dv1(0) = dcmplx(ga(1))
      dv1(1) = dcmplx(ga(2))
      dv1(2) = dcmplx(ga(3))
      dv1(3) = dcmplx(ga(4))
      dv2(0) = dcmplx(gb(1))
      dv2(1) = dcmplx(gb(2))
      dv2(2) = dcmplx(gb(3))
      dv2(3) = dcmplx(gb(4))
      dv3(0) = dcmplx(gc(1))
      dv3(1) = dcmplx(gc(2))
      dv3(2) = dcmplx(gc(3))
      dv3(3) = dcmplx(gc(4))
      dv4(0) = dcmplx(gd(1))
      dv4(1) = dcmplx(gd(2))
      dv4(2) = dcmplx(gd(3))
      dv4(3) = dcmplx(gd(4))

      v12 = dv1(0)*dv2(0)-dv1(1)*dv2(1)-dv1(2)*dv2(2)-dv1(3)*dv2(3)
      v13 = dv1(0)*dv3(0)-dv1(1)*dv3(1)-dv1(2)*dv3(2)-dv1(3)*dv3(3)
      v14 = dv1(0)*dv4(0)-dv1(1)*dv4(1)-dv1(2)*dv4(2)-dv1(3)*dv4(3)
      v23 = dv2(0)*dv3(0)-dv2(1)*dv3(1)-dv2(2)*dv3(2)-dv2(3)*dv3(3)
      v24 = dv2(0)*dv4(0)-dv2(1)*dv4(1)-dv2(2)*dv4(2)-dv2(3)*dv4(3)
      v34 = dv3(0)*dv4(0)-dv3(1)*dv4(1)-dv3(2)*dv4(2)-dv3(3)*dv4(3)

      dvertx = v14*v23 -v13*v24

      vertex = dcmplx( dvertx ) * (g*g)

c      if (abs(dvertx) .gt. 1d40) then
c         write(*,*) 'Careful',abs(dvertx)
c         write(*,*) v12,v13,v14
c         write(*,*) v23,v24,v34
c      endif
c
      return
      end
