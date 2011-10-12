      subroutine httsxx(tc1,tc2,sc,g1,g2,mass,width,htts)
c
c- by RF - Mar. 2006 
c
c This subroutine computes an off-shell tensor current from the ttss coupling.
c
c     input:
c          complex tc1(18)           : first incoming tensor particle
c          complex tc2(18)           : second incoming tensor particle
c          complex sc(3)             : Incoming scalar particle
c          complex g1(2)             : coupling constant (Higgs effc. theor)
c          real    g2                : coupling constant (include extra Higgs)
c          real    mass              : mass of the outgoing scalar
c          real    width             : width of the outgoing scalar
c
c     output:
c          complex htts              : off-shell tensor current
c

      implicit none
c--   dimension of the current set to arbitrary length
      INTEGER DIM
      PARAMETER(DIM=18)
c      include 'dimension.inc'
      double complex tc1(DIM),htts(DIM),sc(DIM),tc2(DIM)
      double complex dg,g1(2)

      double precision g2,mass,width,q2,q(0:3)

c The outgoing tensor current is the same as the incoming multiplied by the
c coupling constants and the scalar particles.
c Note that the diagonal tensor terms are always zero because
c the tensor particle is anti-symmetric.

      
      htts(2) = sc(2) + tc2(17) + tc1(17)
      htts(3) = sc(3) + tc2(18) + tc1(18)

 
      if (g1(1).NE.(0D0,0D0)) then

      q(0) = -dble( htts(2))
      q(1) = -dble( htts(3))
      q(2) = -dimag(htts(3))
      q(3) = -dimag(htts(2))

      q2 = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2

      dg = - g1(1)*g2/dcmplx( q2-mass**2, mass*width )

      htts(1)= dg*sc(1)*(
c     &                   + tc1( 1) * tc2( 1)
     &                   - tc1( 2) * tc2( 2)
     &                   - tc1( 3) * tc2( 3)
     &                   - tc1( 4) * tc2( 4)

     &                   - tc1( 5) * tc2( 5)
c     &                   + tc1( 6) * tc2( 6)
     &                   + tc1( 7) * tc2( 7)
     &                   + tc1( 8) * tc2( 8)

     &                   - tc1( 9) * tc2( 9)
     &                   + tc1(10) * tc2(10)
c     &                   + tc1(11) * tc2(11)
     &                   + tc1(12) * tc2(12)

     &                   - tc1(13) * tc2(13)
     &                   + tc1(14) * tc2(14)
     &                   + tc1(15) * tc2(15)
c     &                   + tc1(16) * tc2(16)
     &                                       )


      else
         htts( 1)=(0D0,0D0)
      endif


      return
      end
