      subroutine jssxxx(s1,s2,gc,vmass,vwidth , jss)
c
c This subroutine computes an off-shell vector current from the vector-
c scalar-scalar coupling.  The coupling is absent in the minimal SM in
c unitary gauge.  The propagator is given in Feynman gauge for a
c massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant (s1 charge)
c       real    vmass          : mass  of output vector v
c       real    vwidth         : width of output vector v
c
c examples of the coupling constant g for susy particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=A   |   v=Z   |   v=W   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the s1 charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex jss(6)         : vector current            j^mu(v:s1,s2)
c     
      implicit none
      double complex s1(3),s2(3),gc,jss(6),dg,adg
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision pp(0:3),pa(0:3),q(0:3),vmass,vwidth
      double precision q2,vm2,mp2,ma2,m2d

      double precision rZero
      parameter( rZero = 0.0d0 )

#ifdef HELAS_CHECK
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( s1(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s1 in jssxxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in jssxxx has zero momentum'
      endif
      if ( abs(s2(1)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : s2 in jssxxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in jssxxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in jssxxx is zero coupling'
      endif
      if ( vmass.lt.rZero ) then
         write(stdo,*) ' helas-error : vmass in jssxxx is negative'
         write(stdo,*) '             : vmass = ',vmass
      endif
      if ( vwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : vwidth in jssxxx is negative'
         write(stdo,*) '             : vwidth = ',vwidth
      endif
#endif

      jss(5) = s1(2)+s2(2)
      jss(6) = s1(3)+s2(3)

      q(0) = dble( jss(5))
      q(1) = dble( jss(6))
      q(2) = dimag(jss(6))
      q(3) = dimag(jss(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      vm2 = vmass**2

#ifdef HELAS_CHECK
      if ( abs(jss(5))+abs(jss(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jss in jssxxx has zero momentum'
      endif
      if ( vwidth.eq.rZero .and. q2.eq.vm2 ) then
         write(stdo,*)
     &        ' helas-error : jss in jssxxx is on vmass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*) 
     &        '             : abs(q)= ',sqrt(abs(q2))
         jss(1) = cZero
         jss(2) = cZero
         jss(3) = cZero
         jss(4) = cZero
         return
      endif
#endif

      if ( vmass.ne.rZero ) then

         dg = gc/dcmplx( q2-vm2, vmass*vwidth )
c  For the running width, use below instead of the above dg.
c         dg = g/dcmplx( q2-vm2, max(vwidth*q2/vmass,rZero) )


         adg = dg*s1(1)*s2(1)

         pp(0) = dble( s1(2))
         pp(1) = dble( s1(3))
         pp(2) = dimag(s1(3))
         pp(3) = dimag(s1(2))
         pa(0) = dble( s2(2))
         pa(1) = dble( s2(3))
         pa(2) = dimag(s2(3))
         pa(3) = dimag(s2(2))
         mp2 = pp(0)**2-(pp(1)**2+pp(2)**2+pp(3)**2)
         ma2 = pa(0)**2-(pa(1)**2+pa(2)**2+pa(3)**2)
         m2d = mp2-ma2

c     Fabio's implementation of the fixed width
         cm2=dcmplx( vm2, -vmass*vwidth )
c     jss(1) = adg*( (pp(0)-pa(0)) - q(0)*m2d/vm2)
c     jss(2) = adg*( (pp(1)-pa(1)) - q(1)*m2d/vm2)
c     jss(3) = adg*( (pp(2)-pa(2)) - q(2)*m2d/vm2)
c     jss(4) = adg*( (pp(3)-pa(3)) - q(3)*m2d/vm2)
         jss(1) = adg*( (pp(0)-pa(0)) - q(0)*m2d/cm2)
         jss(2) = adg*( (pp(1)-pa(1)) - q(1)*m2d/cm2)
         jss(3) = adg*( (pp(2)-pa(2)) - q(2)*m2d/cm2)
         jss(4) = adg*( (pp(3)-pa(3)) - q(3)*m2d/cm2)

      else

         adg = gc*s1(1)*s2(1)/q2

         jss(1) = adg*dble( s1(2)-s2(2))
         jss(2) = adg*dble( s1(3)-s2(3))
         jss(3) = adg*dimag(s1(3)-s2(3))
         jss(4) = adg*dimag(s1(2)-s2(2))

      endif
c
      return
      end
