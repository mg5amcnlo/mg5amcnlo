      subroutine jwwwnx(w1,w2,w3,gwwa,gwwz,wmass,wwidth , jwww)
c
c This subroutine computes an off-shell W+/W- current from the four-
c point gauge boson coupling.  The vector propagators for the output
c W and the internal Z bosons are given in unitary gauge, and that of
c the internal photon is given in Feynman gauge.
c
c input:
c       complex w1(6)          : first  vector                        w1
c       complex w2(6)          : second vector                        w2
c       complex w3(6)          : third  vector                        w3
c       real    gwwa           : coupling constant of W and A       gwwa
c       real    gwwz           : coupling constant of W and Z       gwwz
c       real    zmass          : mass  of internal Z
c       real    zwidth         : width of internal Z
c       real    wmass          : mass  of output W
c       real    wwidth         : width of output W
c
c the possible sets of the inputs are as follows:
c   -------------------------------------------------------------------
c   |  w1  |  w2  |  w3  |gwwa|gwwz|zmass|zwidth|wmass|wwidth || jwww |
c   -------------------------------------------------------------------
c   |  W-  |  W+  |  W-  |gwwa|gwwz|zmass|zwidth|wmass|wwidth ||  W+  |
c   |  W+  |  W-  |  W+  |gwwa|gwwz|zmass|zwidth|wmass|wwidth ||  W-  |
c   -------------------------------------------------------------------
c where all the bosons are defined by the flowing-OUT quantum number.
c
c output:
c       complex jwww(6)        : W current             j^mu(w':w1,w2,w3)
c     
      implicit none
      double complex w1(6),w2(6),w3(6),jwww(6)
      double complex dw1(0:3),dw2(0:3),dw3(0:3),jj(0:3)
      double complex dw,w12,w32,w13,jq
      double complex cm2        ! mass**2- I Gamma mass (Fabio)
      double precision gwwa,gwwz,wmass,wwidth
      double precision p1(0:3),p2(0:3),p3(0:3),q(0:3)
      double precision dgwwa2,dgwwz2,dgw2,dmw,dww,mw2,q2

      double precision rZero, rOne, rTwo
      parameter( rZero = 0.0d0, rOne = 1.0d0, rTwo = 2.0d0 )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(w1(1))+abs(w1(2))+abs(w1(3))+abs(w1(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w1 in jwwwxx is zero vector'
      endif
      if ( abs(w1(5))+abs(w1(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w1 in jwwwxx has zero momentum'
      endif
      if ( abs(w2(1))+abs(w2(2))+abs(w2(3))+abs(w2(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w2 in jwwwxx is zero vector'
      endif
      if ( abs(w2(5))+abs(w2(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w2 in jwwwxx has zero momentum'
      endif
      if ( abs(w3(1))+abs(w3(2))+abs(w3(3))+abs(w3(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : w3 in jwwwxx is zero vector'
      endif
      if ( abs(w3(5))+abs(w3(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : w3 in jwwwxx has zero momentum'
      endif 
c     Neil edited the following to allow 3-site couplings.
c      if ( gwwa.eq.rZero ) then
c         write(stdo,*) ' helas-error : gwwa in jwwwxx is zero coupling'
c      endif
c      if ( gwwz.eq.rZero ) then
c         write(stdo,*) ' helas-error : gwwz in jwwwxx is zero coupling'
c      endif
c      if ( gwwa.lt.rZero .or. gwwa.ge.gwwz ) then
c         write(stdo,*)
c     &  ' helas-warn  : gwwa/gwwz in jwwwxx are non-standard couplings'
c         write(stdo,*)
c     &  '             : gwwa = ',gwwa,'  gwwz = ',gwwz
c      endif
c     End Neil's edit
      if ( wmass.le.rZero ) then
         write(stdo,*) ' helas-error : wmass in jwwwxx is not positive'
         write(stdo,*) '             : wmass = ',wmass
      endif
      if ( wwidth.lt.rZero ) then
         write(stdo,*) ' helas-error : wwidth in jwwwxx is negative'
         write(stdo,*) '             : wwidth = ',wwidth
      endif
#endif



      
      jwww(5) = w1(5)+w2(5)+w3(5)
      jwww(6) = w1(6)+w2(6)+w3(6)

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
c     Neil edited this file to allow implementation of 3-site model.
c     Now, only gwwa matters and is the full coupling squared.
c      dgwwa2 = dble(gwwa)**2
c      dgwwz2 = dble(gwwz)**2
c      dgw2 = dgwwa2+dgwwz2
c      dgw2 = gwwa
c     End Neil's edit

c Benj's modif in order to have FR running
      dgw2=rZero
      if(gwwa.eq.rZero) dgw2=gwwz
      if(gwwz.eq.rZero) dgw2=gwwa
c End of Benj'S modif



      dmw = dble(wmass)
      dww = dble(wwidth)
      mw2 = dmw**2

#ifdef HELAS_CHECK
      if ( abs(jwww(5))+abs(jwww(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : jwww in jwwwxx has zero momentum'
      endif
      if ( wwidth.eq.rZero .and. q2.eq.mw2 ) then
         write(stdo,*)
     &        ' helas-error : jwww in jwwwxx is on wmass pole'
         write(stdo,*)
     &        '             : q     = ',
     &        real(q(0)),real(q(1)),real(q(2)),real(q(3))
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(real(q2)))
         jwww(1) = cZero
         jwww(2) = cZero
         jwww(3) = cZero
         jwww(4) = cZero
         return
      endif
#endif

c Start of Claude'S modif (Removed minus Sign)
      dw = rOne/dcmplx( q2-mw2, dmw*dww )
c End of Claude'S modif

c  For the running width, use below instead of the above dw.
c      dw = -rOne/dcmplx( q2-mw2 , max(dww*q2/dmw,rZero) )

      w12=dw1(0)*dw2(0)-dw1(1)*dw2(1)-dw1(2)*dw2(2)-dw1(3)*dw2(3)
      w32=dw3(0)*dw2(0)-dw3(1)*dw2(1)-dw3(2)*dw2(2)-dw3(3)*dw2(3)

      w13=dw1(0)*dw3(0)-dw1(1)*dw3(1)-dw1(2)*dw3(2)-dw1(3)*dw3(3)

      jj(0) = dgw2*( dw1(0)*w32 + dw3(0)*w12 - rTwo*dw2(0)*w13 )
      jj(1) = dgw2*( dw1(1)*w32 + dw3(1)*w12 - rTwo*dw2(1)*w13 )
      jj(2) = dgw2*( dw1(2)*w32 + dw3(2)*w12 - rTwo*dw2(2)*w13 )
      jj(3) = dgw2*( dw1(3)*w32 + dw3(3)*w12 - rTwo*dw2(3)*w13 )

c     Fabio's implementation of the fixed width
      cm2=dcmplx( mw2, -dmw*dww )
c     jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/mw2
      jq = (jj(0)*q(0)-jj(1)*q(1)-jj(2)*q(2)-jj(3)*q(3))/cm2

      jwww(1) = dcmplx( (jj(0)-jq*q(0))*dw )
      jwww(2) = dcmplx( (jj(1)-jq*q(1))*dw )
      jwww(3) = dcmplx( (jj(2)-jq*q(2))*dw )
      jwww(4) = dcmplx( (jj(3)-jq*q(3))*dw )
c
      return
      end
