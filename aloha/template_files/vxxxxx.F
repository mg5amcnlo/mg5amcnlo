      subroutine vxxxxx(p,vmass,nhel,nsv , vc)
c
c This subroutine computes a VECTOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of vector boson
c       real    vmass          : mass          of vector boson
c       integer nhel = -1, 0, 1: helicity      of vector boson
c                                (0 is forbidden if vmass=0.0)
c       integer nsv  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex vc(6)          : vector wavefunction       epsilon^mu(v)
c     
      implicit none
      double complex vc(6)
      double precision p(0:3),vmass,hel,hel0,pt,pt2,pp,pzpt,emp,sqh
      integer nhel,nsv,nsvahl

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )
      
#ifdef HELAS_CHECK
      double precision p2
      double precision epsi
      parameter( epsi = 2.0d-5 )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
      if ( abs(p(0))+pp.eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in vxxxxx is zero momentum'
      endif
      if ( p(0).le.rZero ) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in vxxxxx has non-positive energy'
         write(stdo,*) 
     &        '             : p(0) = ',p(0)
      endif
      p2 = (p(0)+pp)*(p(0)-pp)
      if ( abs(p2-vmass**2).gt.p(0)**2*2.e-5 ) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in vxxxxx has inappropriate mass'
         write(stdo,*)
     &        '             : p**2 = ',p2,' : vmass**2 = ',vmass**2
      endif
      if ( vmass.ne.rZero ) then
         if ( abs(nhel).gt.1 ) then
            write(stdo,*) ' helas-error : nhel in vxxxxx is not -1,0,1'
            write(stdo,*) '             : nhel = ',nhel
         endif
      else
         if ( abs(nhel).ne.1 ) then
            write(stdo,*) ' helas-error : nhel in vxxxxx is not -1,1'
            write(stdo,*) '             : nhel = ',nhel
         endif
      endif
      if ( abs(nsv).ne.1 ) then
         write(stdo,*) ' helas-error : nsv in vmxxxx is not -1,1'
         write(stdo,*) '             : nsv = ',nsv
      endif
#endif

      sqh = dsqrt(rHalf)
      hel = dble(nhel)
      nsvahl = nsv*dabs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = min(p(0),dsqrt(pt2+p(3)**2))
      pt = min(pp,dsqrt(pt2))

      vc(5) = dcmplx(p(0),p(3))*nsv
      vc(6) = dcmplx(p(1),p(2))*nsv

#ifdef HELAS_CHECK
c nhel=4 option for scalar polarization
      if( nhel.eq.4 ) then
         if( vmass.eq.rZero ) then
            vc(1) = rOne
            vc(2) = p(1)/p(0)
            vc(3) = p(2)/p(0)
            vc(4) = p(3)/p(0)
         else
            vc(1) = p(0)/vmass
            vc(2) = p(1)/vmass
            vc(3) = p(2)/vmass
            vc(4) = p(3)/vmass
         endif
         return
      endif
#endif

      if ( vmass.ne.rZero ) then

         hel0 = rOne-dabs(hel)

         if ( pp.eq.rZero ) then

            vc(1) = dcmplx( rZero )
            vc(2) = dcmplx(-hel*sqh )
            vc(3) = dcmplx( rZero , nsvahl*sqh )
            vc(4) = dcmplx( hel0 )

         else

            emp = p(0)/(vmass*pp)
            vc(1) = dcmplx( hel0*pp/vmass )
            vc(4) = dcmplx( hel0*p(3)*emp+hel*pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc(2) = dcmplx( hel0*p(1)*emp-p(1)*pzpt , 
     &                         -nsvahl*p(2)/pt*sqh       )
               vc(3) = dcmplx( hel0*p(2)*emp-p(2)*pzpt ,  
     &                          nsvahl*p(1)/pt*sqh       )
            else
               vc(2) = dcmplx( -hel*sqh )
               vc(3) = dcmplx( rZero , nsvahl*sign(sqh,p(3)) )
            endif

         endif

      else

         pp = p(0)
         pt = sqrt(p(1)**2+p(2)**2)
         vc(1) = dcmplx( rZero )
         vc(4) = dcmplx( hel*pt/pp*sqh )
         if ( pt.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc(2) = dcmplx( -p(1)*pzpt , -nsv*p(2)/pt*sqh )
            vc(3) = dcmplx( -p(2)*pzpt ,  nsv*p(1)/pt*sqh )
         else
            vc(2) = dcmplx( -hel*sqh )
            vc(3) = dcmplx( rZero , nsv*sign(sqh,p(3)) )
         endif

      endif
c
      return
      end
