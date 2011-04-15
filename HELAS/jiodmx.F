      subroutine jiodmx(fi,fo,gc,vmass,vwidth , jio)
c
c This subroutine computes an off-shell vector dipole moment
c (non-renormalizable) current from an external
c fermion pair.  The vector boson propagator is given in Feynman gauge
c for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2,2)        : coupling constants                 gvf
c                              : first index is L,R as normal
c                              : second index is EDM,-MDM
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c
      implicit none
      double complex fi(6), fo(6), jio(6), c0, c1, c2, c3, d
      double complex gc(2,2), gL, gR
      double precision  q(0:3), vmass, vwidth, q2, vm2, dd

      double complex f1122, f12, f21, f12p21, f12m21
      double complex f3344, f34, f43, f34p43, f34m43
      double complex dumL1, dumL2, dumL3, dumL4
      double complex dumR1, dumR2, dumR3, dumR4
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(fi(1))+abs(fi(2))+abs(fi(3))+abs(fi(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fi in jiodmx is zero spinor'
      endif
      if ( abs(fi(5))+abs(fi(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fi in jiodmx has zero momentum'
      endif
      if ( abs(fo(1))+abs(fo(2))+abs(fo(3))+abs(fo(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : fo in jiodmx is zero spinor'
      endif
      if ( abs(fo(5))+abs(fo(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : fo in jiodmx has zero momentum'
      endif
      if ( gc(1,1).eq.cZero .and. gc(2,1).eq.cZero .and.
     &     gc(1,2).eq.cZero .and. gc(2,2).eq.cZero      ) then
         write(stdo,*)
     &        ' helas-error : gc in jiodmx is zero coupling'
      endif
      if ( vmass.lt.rZero ) then
         write(stdo,*) ' helas-error : vmass in jiodmx is negative'
         write(stdo,*) '             : vmass = ',vmass
      endif
      if ( vwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : vwidth in jiodmx is negative'
         write(stdo,*) '             : vwidth = ',vwidth
      endif
#endif

      gL = -gc(1,1) + cImag*gc(1,2)
      gR =  gc(2,1) + cImag*gc(2,2)

      jio(5) = fo(5) - fi(5)
      jio(6) = fo(6) - fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

#ifdef HELAS_CHECK
      if ( abs(jio(5))+abs(jio(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jio in jiodmx has zero momentum'
      endif
      if ( vwidth.eq.rZero .and. q2.eq.vm2 ) then
         write(stdo,*)
     &        ' helas-error : jio in jiodmxq is on vmass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         jio(1) = cZero
         jio(2) = cZero
         jio(3) = cZero
         jio(4) = cZero
         return
      endif
#endif

      f1122  = fo(1)*fi(1) - fo(2)*fi(2)
      f12    = fo(1)*fi(2)
      f21    = fo(2)*fi(1)
      f12p21 = f12 + f21
      f12m21 = f12 - f21
      if ( gc(2,1).ne.cZero .or. gc(2,2).ne.cZero ) then
         f3344  = fo(3)*fi(3) - fo(4)*fi(4)
         f34    = fo(3)*fi(4)
         f43    = fo(4)*fi(3)
         f34p43 = f34 + f43
         f34m43 = f34 - f43
      end if

c note overall (-), since k in vertex is -q above
      dumL1 = -q(2)*f12m21 - cImag*( q(1)*f12p21 + q(3)*f1122 )
      dumL2 =  q(2)*f1122  - cImag*( q(0)*f12p21 - q(3)*f12m21 )
      dumL3 = -q(0)*f12m21 - q(1)*f1122 - q(3)*f12p21
      dumL4 = -q(2)*f12p21 - cImag*( q(0)*f1122  + q(1)*f12m21 )
      if ( gc(2,1).ne.cZero .or. gc(2,2).ne.cZero ) then
         dumR1 =  q(2)*f34m43 + cImag*( q(1)*f34p43 + q(3)*f3344 )
         dumR2 =  q(2)*f3344  + cImag*( q(0)*f34p43 + q(3)*f34m43 )
         dumR3 =  q(0)*f34m43 - q(1)*f3344 - q(3)*f34p43
         dumR4 = -q(2)*f34p43 + cImag*( q(0)*f3344  - q(1)*f34m43 )
      end if

      if ( vmass.ne.rZero ) then

         d = rOne/dcmplx( q2-vm2, vmass*vwidth )

         c0 = gL*dumL1
         c1 = gL*dumL2
         c2 = gL*dumL3
         c3 = gL*dumL4

         if ( gc(2,1).ne.cZero .or.
     &        gc(2,2).ne.cZero      ) then
            c0 = c0 + gR*dumR1
            c1 = c1 + gR*dumR2
            c2 = c2 + gR*dumR3
            c3 = c3 + gR*dumR4
         end if

         jio(1) = c0*d
         jio(2) = c1*d
         jio(3) = c2*d
         jio(4) = c3*d

      else

         dd = rOne/q2

         jio(1) = gL*dumL1
         jio(2) = gL*dumL2
         jio(3) = gL*dumL3
         jio(4) = gL*dumL4

         if ( gc(2,1).ne.cZero .or.
     &        gc(2,2).ne.cZero      ) then
            jio(1) = jio(1) + gR*dumR1
            jio(2) = jio(2) + gR*dumR2
            jio(3) = jio(3) + gR*dumR3
            jio(4) = jio(4) + gR*dumR4
         end if

         jio(1) = jio(1)*dd
         jio(2) = jio(2)*dd
         jio(3) = jio(3)*dd
         jio(4) = jio(4)*dd

      end if
c
      return
      end
