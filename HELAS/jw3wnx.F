      subroutine jw3wnx(w1,w2,w3,g1,g2,vmass,vwidth , jw3w)
c
c This subroutine computes an off-shell W+, W-, W3, Z or photon current
c from the four-point gauge boson coupling.  The vector propagator is
c given in Feynman gauge for a photon and in unitary gauge for W and
c Z bosons.  If one sets wmass=0.0, then the ggg-->g current is given
c (see sect 2.9.1 of the manual).
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       complex w3(6)          : third  vector                        w3
c       real    g1             : first  coupling constant
c       real    g2             : second coupling constant
c                                                  (see the table below)
c       real    wmass          : mass  of internal W
c       real    wwidth         : width of internal W
c       real    vmass          : mass  of output W'
c       real    vwidth         : width of output W'
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------------------------------
c   |  w1  |  w2  |  w3  | g1 | g2 |wmass|wwidth|vmass|vwidth || jw3w |
c   -------------------------------------------------------------------
c   |  W-  |  W3  |  W+  | gw |gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  W3  |  W+  | gw |gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   |  W-  |  Z   |  W+  |gwwz|gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  Z   |  W+  |gwwz|gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   |  W-  |  A   |  W+  |gwwa|gwwz|wmass|wwidth|zmass|zwidth ||  Z   |
c   |  W-  |  A   |  W+  |gwwa|gwwa|wmass|wwidth|  0. |  0.   ||  A   |
c   -------------------------------------------------------------------
c   |  W3  |  W-  |  W3  | gw | gw |wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  W3  | gw | gw |wmass|wwidth|wmass|wwidth ||  W-  |
c   |  W3  |  W-  |  Z   | gw |gwwz|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  Z   | gw |gwwz|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  W3  |  W-  |  A   | gw |gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  W3  |  W+  |  A   | gw |gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  Z   |  W-  |  Z   |gwwz|gwwz|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  Z   |  W+  |  Z   |gwwz|gwwz|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  Z   |  W-  |  A   |gwwz|gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  Z   |  W+  |  A   |gwwz|gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   |  A   |  W-  |  A   |gwwa|gwwa|wmass|wwidth|wmass|wwidth ||  W+  |
c   |  A   |  W+  |  A   |gwwa|gwwa|wmass|wwidth|wmass|wwidth ||  W-  |
c   -------------------------------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex jw3w(6)        : W current             j^mu(w':w1,w2,w3)
c     
      implicit none
      double complex w1(6),w2(6),w3(6),jw3w(6)
      double complex dw1(0:3),dw2(0:3),dw3(0:3)
      double complex jj(0:3),j4(0:3),dv,w12,w32,w13,jq
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision g1,g2,vmass,vwidth
      double precision p1(0:3),p2(0:3),p3(0:3),q(0:3)
      double precision dg2,dmv,dwv,mv2,q2

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(w1(1))+abs(w1(2))+abs(w1(3))+abs(w1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w1 in jw3wxx is zero vector'
      endif
      if ( abs(w1(5))+abs(w1(6)).eq.rZero ) then
         write(stdo,*) 
     &        ' helas-error : w1 in jw3wxx has zero momentum' 
      endif 
      if ( abs(w2(1))+abs(w2(2))+abs(w2(3))+abs(w2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w2 in jw3wxx is zero vector'
      endif
      if ( abs(w2(5))+abs(w2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w2 in jw3wxx has zero momentum'
      endif
      if ( abs(w3(1))+abs(w3(2))+abs(w3(3))+abs(w3(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w3 in jw3wxx is zero vector'
      endif
      if ( abs(w3(5))+abs(w3(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w3 in jw3wxx has zero momentum'
      endif
c     Neil edited this file to allow 3-site couplings.
c      if ( g1.eq.rZero ) then
c         write(stdo,*) ' helas-error : g1 in jw3wxx is zero coupling'
c      endif
c      if ( g2.eq.rZero ) then
c         write(stdo,*) ' helas-error : g2 in jw3wxx is zero coupling'
c      endif
c      if ( g1.lt.rZero ) then
c         write(stdo,*)
c     &        ' helas-warn  : g1 in jw3wxx is non-standard coupling'
c         write(stdo,*)
c     &        '             : g1 = ',g1
c      endif
c      if ( g2.lt.rZero ) then
c         write(stdo,*)
c     &        ' helas-warn  : g2 in jw3wxx is non-standard coupling'
c         write(stdo,*)
c     &        '             : g2 = ',g2
c      endif
c     End Neil's edit.
      if ( vmass.lt.rZero ) then
         write(stdo,*) ' helas-error : vmass in jw3wxx is negative'
         write(stdo,*) '             : vmass = ',vmass
      endif
      if ( vwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : vwidth in jw3wxx is negative'
         write(stdo,*) '             : vwidth = ',vwidth
      endif
#endif

      jw3w(5) = w1(5)+w2(5)+w3(5)
      jw3w(6) = w1(6)+w2(6)+w3(6)

      dw1(0) = dcmplx(w1(1))
      dw1(1) = dcmplx(w1(2))
      dw1(2) = dcmplx(w1(3))
      dw1(3) = dcmplx(w1(4))
      dw2(0) = dcmplx(w2(1))
      dw2(1) = dcmplx(w2(2))
      dw2(2) = dcmplx(w2(3))
      dw2(3) = dcmplx(w2(4))
      dw3(0) = dcmplx(w3(1))
      dw3(1) = dcmplx(w3(2))
      dw3(2) = dcmplx(w3(3))
      dw3(3) = dcmplx(w3(4))
      p1(0) = dble(      w1(5))
      p1(1) = dble(      w1(6))
      p1(2) = dble(dimag(w1(6)))
      p1(3) = dble(dimag(w1(5)))
      p2(0) = dble(      w2(5))
      p2(1) = dble(      w2(6))
      p2(2) = dble(dimag(w2(6)))
      p2(3) = dble(dimag(w2(5)))
      p3(0) = dble(      w3(5))
      p3(1) = dble(      w3(6))
      p3(2) = dble(dimag(w3(6)))
      p3(3) = dble(dimag(w3(5)))
      q(0) = -(p1(0)+p2(0)+p3(0))
      q(1) = -(p1(1)+p2(1)+p3(1))
      q(2) = -(p1(2)+p2(2)+p3(2))
      q(3) = -(p1(3)+p2(3)+p3(3))

      q2 = q(0)**2 -(q(1)**2 +q(2)**2 +q(3)**2)
c     Neil edited this to allow 3-site couplings.
c     Now only g1 is important.  g2 does nothing.
c      dg2 = dble(g1)*dble(g2)
c      dg2 = dble(g1)
c     End Neil's edit.

c Benj's modif in order to have FR running
      if(g1.eq.rzero) dg2=g2
      if(g2.eq.rzero) dg2=g1
c End of Benj'S modif

      dmv = dble(vmass)
      dwv = dble(vwidth)
      mv2 = dmv**2

#ifdef HELAS_CHECK
      if ( abs(jw3w(5))+abs(jw3w(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jw3w in jw3wxx has zero momentum'
      endif
      if ( vwidth.eq.rZero .and. q2.eq.mv2 ) then
         write(stdo,*)
     &        ' helas-error : jw3w in jw3wxx is on vmass pole'
         write(stdo,*)
     &        '             : q     = ',
     &        real(q(0)),real(q(1)),real(q(2)),real(q(3))
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(real(q2)))
         jw3w(1) = cZero
         jw3w(2) = cZero
         jw3w(3) = cZero
         jw3w(4) = cZero
         return
      endif
#endif

      if ( vmass.eq.rZero ) then
         dv = rOne/dcmplx( q2 )
      else
         dv = rOne/dcmplx( q2-mv2, dmv*dwv )
      endif

c  For the running width, use below instead of the above dv.
c      dv = rOne/dcmplx( q2-mv2 , max(dwv*q2/dmv,rZero) )

      w12=dw1(0)*dw2(0)-dw1(1)*dw2(1)-dw1(2)*dw2(2)-dw1(3)*dw2(3)
      w32=dw3(0)*dw2(0)-dw3(1)*dw2(1)-dw3(2)*dw2(2)-dw3(3)*dw2(3)

      w13=dw1(0)*dw3(0)-dw1(1)*dw3(1)-dw1(2)*dw3(2)-dw1(3)*dw3(3)
      
      j4(0) = dg2*( dw1(0)*w32 + dw3(0)*w12 - rTwo*dw2(0)*w13 )
      j4(1) = dg2*( dw1(1)*w32 + dw3(1)*w12 - rTwo*dw2(1)*w13 )
      j4(2) = dg2*( dw1(2)*w32 + dw3(2)*w12 - rTwo*dw2(2)*w13 )
      j4(3) = dg2*( dw1(3)*w32 + dw3(3)*w12 - rTwo*dw2(3)*w13 )

      jj(0) = j4(0)
      jj(1) = j4(1)
      jj(2) = j4(2)
      jj(3) = j4(3)

      if ( vmass.ne.rZero ) then

c     Fabio's implementation of the fixed width
         cm2=dcmplx( mv2, -dmv*dwv)
c     jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/mv2
         jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/cm2
         
         jw3w(1) = dcmplx( (jj(0)-jq*q(0))*dv )
         jw3w(2) = dcmplx( (jj(1)-jq*q(1))*dv )
         jw3w(3) = dcmplx( (jj(2)-jq*q(2))*dv )
         jw3w(4) = dcmplx( (jj(3)-jq*q(3))*dv )

      else

         jw3w(1) = dcmplx( jj(0)*dv )
         jw3w(2) = dcmplx( jj(1)*dv )
         jw3w(3) = dcmplx( jj(2)*dv )
         jw3w(4) = dcmplx( jj(3)*dv )
      end if
c
      return
      end
