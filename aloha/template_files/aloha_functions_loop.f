C###############################################################################
C
C Copyright (c) 2010 The ALOHA Development team and Contributors
C
C This file is a part of the MadGraph5_aMC@NLO project, an application which
C automatically generates Feynman diagrams and matrix elements for arbitrary
C high-energy processes in the Standard Model and beyond.
C
C It is subject to the ALOHA license which should accompany this
C distribution.
C
C###############################################################################
      subroutine ixxxxx(p, fmass, nhel, nsf ,fi)
c
c This subroutine computes a fermion wavefunction with the flowing-IN
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(6)          : fermion wavefunction               |fi>
c
      implicit none
      double complex fi(8),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,ip,im,nh

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
c      if ( abs(p(0))+pp.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx has non-positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      p2 = (p(0)-pp)*(p(0)+pp)
c      if ( abs(p2-fmass**2).gt.p(0)**2*epsi ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx has inappropriate mass'
c         write(stdo,*)
c     &        '             : p**2 = ',p2,' : fmass**2 = ',fmass**2
c      endif
c      if (abs(nhel).ne.1) then
c         write(stdo,*) ' helas-error : nhel in ixxxxx is not -1,1'
c         write(stdo,*) '             : nhel = ',nhel
c      endif
c      if (abs(nsf).ne.1) then
c         write(stdo,*) ' helas-error : nsf in ixxxxx is not -1,1'
c         write(stdo,*) '             : nsf = ',nsf
c      endif
c#endif

c     Convention for trees
c      fi(5) = dcmplx(p(0),p(3))*nsf
c      fi(6) = dcmplx(p(1),p(2))*nsf

c     Convention for loop computations
      fi(1) = dcmplx(p(0),0.D0)*(-nsf)
      fi(2) = dcmplx(p(1),0.D0)*(-nsf)
      fi(3) = dcmplx(p(2),0.D0)*(-nsf)
      fi(4) = dcmplx(p(3),0.D0)*(-nsf)

      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))


         if ( pp.eq.rZero ) then

            sqm(0) = dsqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            ip = (1+nh)/2
            im = (1-nh)/2

            fi(5) = ip     * sqm(ip)
            fi(6) = im*nsf * sqm(ip)
            fi(7) = ip*nsf * sqm(im)
            fi(8) = im     * sqm(im)

         else

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , p(2) )/dsqrt(rTwo*pp*pp3)
            endif

            fi(5) = sfomeg(1)*chi(im)
            fi(6) = sfomeg(1)*chi(ip)
            fi(7) = sfomeg(2)*chi(im)
            fi(8) = sfomeg(2)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fi(5) = dcmplx( rZero )
            fi(6) = dcmplx( rZero )
            fi(7) = chi(1)
            fi(8) = chi(2)
         else
            fi(5) = chi(2)
            fi(6) = chi(1)
            fi(7) = dcmplx( rZero )
            fi(8) = dcmplx( rZero )
         endif
      endif
c
      return
      end


      subroutine ixxxso(p, fmass, nhel, nsf ,fi)
c
c This subroutine computes a fermion wavefunction with the flowing-IN
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(6)          : fermion wavefunction               |fi>
c
      implicit none
      double complex fi(4),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,ip,im,nh

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
c      if ( abs(p(0))+pp.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx has non-positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      p2 = (p(0)-pp)*(p(0)+pp)
c      if ( abs(p2-fmass**2).gt.p(0)**2*epsi ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in ixxxxx has inappropriate mass'
c         write(stdo,*)
c     &        '             : p**2 = ',p2,' : fmass**2 = ',fmass**2
c      endif
c      if (abs(nhel).ne.1) then
c         write(stdo,*) ' helas-error : nhel in ixxxxx is not -1,1'
c         write(stdo,*) '             : nhel = ',nhel
c      endif
c      if (abs(nsf).ne.1) then
c         write(stdo,*) ' helas-error : nsf in ixxxxx is not -1,1'
c         write(stdo,*) '             : nsf = ',nsf
c      endif
c#endif

c     Convention for trees
c      fi(5) = dcmplx(p(0),p(3))*nsf
c      fi(6) = dcmplx(p(1),p(2))*nsf

c$$$c     Convention for loop computations
c$$$      fi(1) = dcmplx(p(0),0.D0)*(-nsf)
c$$$      fi(2) = dcmplx(p(1),0.D0)*(-nsf)
c$$$      fi(3) = dcmplx(p(2),0.D0)*(-nsf)
c$$$      fi(4) = dcmplx(p(3),0.D0)*(-nsf)

      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then

            sqm(0) = dsqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            ip = (1+nh)/2
            im = (1-nh)/2

            fi(1) = ip     * sqm(ip)
            fi(2) = im*nsf * sqm(ip)
            fi(3) = ip*nsf * sqm(im)
            fi(4) = im     * sqm(im)

         else

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , p(2) )/dsqrt(rTwo*pp*pp3)
            endif

            fi(1) = sfomeg(1)*chi(im)
            fi(2) = sfomeg(1)*chi(ip)
            fi(3) = sfomeg(2)*chi(im)
            fi(4) = sfomeg(2)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fi(1) = dcmplx( rZero )
            fi(2) = dcmplx( rZero )
            fi(3) = chi(1)
            fi(4) = chi(2)
         else
            fi(1) = chi(2)
            fi(2) = chi(1)
            fi(3) = dcmplx( rZero )
            fi(4) = dcmplx( rZero )
         endif
      endif
c
      return
      end


      subroutine mp_ixxxxx(p, fmass, nhel, nsf ,fi)
c
c This subroutine computes a fermion wavefunction with the flowing-IN
c fermion number, in QUADRUPLE PRECISIOn
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(6)          : fermion wavefunction               |fi>
c
      implicit none
      complex*32 fi(8),chi(2)
      real*16 p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,ip,im,nh

      real*16 rZero, rHalf, rTwo
      parameter( rZero = 0.0e0_16, rHalf = 0.5e0_16, rTwo = 2.0e0_16 )
c     Convention for loop computations
      fi(1) = cmplx(p(0),rZero,KIND=16)*(-nsf)
      fi(2) = cmplx(p(1),rZero,KIND=16)*(-nsf)
      fi(3) = cmplx(p(2),rZero,KIND=16)*(-nsf)
      fi(4) = cmplx(p(3),rZero,KIND=16)*(-nsf)

      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),sqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then

            sqm(0) = sqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            ip = (1+nh)/2
            im = (1-nh)/2

            fi(5) = ip     * sqm(ip)
            fi(6) = im*nsf * sqm(ip)
            fi(7) = ip*nsf * sqm(im)
            fi(8) = im     * sqm(im)

         else

            sf(1) = REAL(1+nsf+(1-nsf)*nh,KIND=16)*rHalf
            sf(2) = REAL(1+nsf-(1-nsf)*nh,KIND=16)*rHalf
            omega(1) = sqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = cmplx( sqrt(pp3*rHalf/pp), KIND=16 )
            if ( pp3.eq.rZero ) then
               chi(2) = cmplx(-nh ,KIND=16)
            else
               chi(2) = cmplx( nh*p(1) , p(2),KIND=16)/sqrt(rTwo*pp*pp3)
            endif

            fi(5) = sfomeg(1)*chi(im)
            fi(6) = sfomeg(1)*chi(ip)
            fi(7) = sfomeg(2)*chi(im)
            fi(8) = sfomeg(2)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = sqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = cmplx( sqp0p3 ,KIND=16)
         if ( sqp0p3.eq.rZero ) then
            chi(2) = cmplx(-nhel ,KIND=16)*sqrt(rTwo*p(0))
         else
            chi(2) = cmplx( nh*p(1), p(2) ,KIND=16)/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fi(5) = cmplx( rZero ,KIND=16)
            fi(6) = cmplx( rZero ,KIND=16)
            fi(7) = chi(1)
            fi(8) = chi(2)
         else
            fi(5) = chi(2)
            fi(6) = chi(1)
            fi(7) = cmplx( rZero ,KIND=16)
            fi(8) = cmplx( rZero ,KIND=16)
         endif
      endif
c
      return
      end

      subroutine oxxxxx(p,fmass,nhel,nsf , fo)
c
c This subroutine computes a fermion wavefunction with the flowing-OUT
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(6)          : fermion wavefunction               <fo|
c
      implicit none
      double complex fo(8),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,nh,ip,im

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
c      if ( abs(p(0))+pp.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx has non-positive energy'
c         write(stdo,*)
c     &        '         : p(0) = ',p(0)
c      endif
c      p2 = (p(0)-pp)*(p(0)+pp)
c      if ( abs(p2-fmass**2).gt.p(0)**2*epsi ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx has inappropriate mass'
c         write(stdo,*)
c     &        '             : p**2 = ',p2,' : fmass**2 = ',fmass**2
c      endif
c      if ( abs(nhel).ne.1 ) then
c         write(stdo,*) ' helas-error : nhel in oxxxxx is not -1,1'
c         write(stdo,*) '             : nhel = ',nhel
c      endif
c      if ( abs(nsf).ne.1 ) then
c         write(stdo,*) ' helas-error : nsf in oxxxxx is not -1,1'
c         write(stdo,*) '             : nsf = ',nsf
c      endif
c#endif

c     Convention for trees
c      fo(5) = dcmplx(p(0),p(3))*nsf
c      fo(6) = dcmplx(p(1),p(2))*nsf

c     Convention for loop computations
      fo(1) = dcmplx(p(0),0.D0)*(nsf)
      fo(2) = dcmplx(p(1),0.D0)*(nsf)
      fo(3) = dcmplx(p(2),0.D0)*(nsf)
      fo(4) = dcmplx(p(3),0.D0)*(nsf)


      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then

            sqm(0) = dsqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses

            im = nhel * (1+nh)/2
            ip = nhel * -1 * ((1-nh)/2)
            fo(5) = im     * sqm(abs(ip))
            fo(6) = ip*nsf * sqm(abs(ip))
            fo(7) = im*nsf * sqm(abs(im))
            fo(8) = ip     * sqm(abs(im))

         else

            pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , -p(2) )/dsqrt(rTwo*pp*pp3)
            endif

            fo(5) = sfomeg(2)*chi(im)
            fo(6) = sfomeg(2)*chi(ip)
            fo(7) = sfomeg(1)*chi(im)
            fo(8) = sfomeg(1)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), -p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fo(5) = chi(1)
            fo(6) = chi(2)
            fo(7) = dcmplx( rZero )
            fo(8) = dcmplx( rZero )
         else
            fo(5) = dcmplx( rZero )
            fo(6) = dcmplx( rZero )
            fo(7) = chi(2)
            fo(8) = chi(1)
         endif

      endif
c
      return
      end

      subroutine oxxxso(p,fmass,nhel,nsf , fo)
c
c This subroutine computes a fermion wavefunction with the flowing-OUT
c fermion number.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(6)          : fermion wavefunction               <fo|
c
      implicit none
      double complex fo(4),chi(2)
      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,nh,ip,im

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
c      if ( abs(p(0))+pp.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx has non-positive energy'
c         write(stdo,*)
c     &        '         : p(0) = ',p(0)
c      endif
c      p2 = (p(0)-pp)*(p(0)+pp)
c      if ( abs(p2-fmass**2).gt.p(0)**2*epsi ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in oxxxxx has inappropriate mass'
c         write(stdo,*)
c     &        '             : p**2 = ',p2,' : fmass**2 = ',fmass**2
c      endif
c      if ( abs(nhel).ne.1 ) then
c         write(stdo,*) ' helas-error : nhel in oxxxxx is not -1,1'
c         write(stdo,*) '             : nhel = ',nhel
c      endif
c      if ( abs(nsf).ne.1 ) then
c         write(stdo,*) ' helas-error : nsf in oxxxxx is not -1,1'
c         write(stdo,*) '             : nsf = ',nsf
c      endif
c#endif

c     Convention for trees
c      fo(5) = dcmplx(p(0),p(3))*nsf
c      fo(6) = dcmplx(p(1),p(2))*nsf

c$$$c     Convention for loop computations
c$$$      fo(1) = dcmplx(p(0),0.D0)*(nsf)
c$$$      fo(2) = dcmplx(p(1),0.D0)*(nsf)
c$$$      fo(3) = dcmplx(p(2),0.D0)*(nsf)
c$$$      fo(4) = dcmplx(p(3),0.D0)*(nsf)


      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then

            sqm(0) = dsqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            ip = -((1+nh)/2)
            im =  (1-nh)/2

            fo(1) = im     * sqm(im)
            fo(2) = ip*nsf * sqm(im)
            fo(3) = im*nsf * sqm(-ip)
            fo(4) = ip     * sqm(-ip)

         else

            pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = dsqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = dcmplx( dsqrt(pp3*rHalf/pp) )
            if ( pp3.eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = dcmplx( nh*p(1) , -p(2) )/dsqrt(rTwo*pp*pp3)
            endif

            fo(1) = sfomeg(2)*chi(im)
            fo(2) = sfomeg(2)*chi(ip)
            fo(3) = sfomeg(1)*chi(im)
            fo(4) = sfomeg(1)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = dsqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = dcmplx( sqp0p3 )
         if ( sqp0p3.eq.rZero ) then
            chi(2) = dcmplx(-nhel )*dsqrt(rTwo*p(0))
         else
            chi(2) = dcmplx( nh*p(1), -p(2) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fo(1) = chi(1)
            fo(2) = chi(2)
            fo(3) = dcmplx( rZero )
            fo(4) = dcmplx( rZero )
         else
            fo(1) = dcmplx( rZero )
            fo(2) = dcmplx( rZero )
            fo(3) = chi(2)
            fo(4) = chi(1)
         endif

      endif
c
      return
      end

      subroutine mp_oxxxxx(p,fmass,nhel,nsf , fo)
c
c This subroutine computes a fermion wavefunction with the flowing-OUT
c fermion number in quadruple precision.
c
c input:
c       real    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(6)          : fermion wavefunction               <fo|
c
      implicit none
      complex*32 fo(8),chi(2)
      real*16 p(0:3),sf(2),sfomeg(2),omega(2),fmass,
     &     pp,pp3,sqp0p3,sqm(0:1)
      integer nhel,nsf,nh,ip,im

      real*16 rZero, rHalf, rTwo
      parameter( rZero = 0.0e0_16, rHalf = 0.5e0_16, rTwo = 2.0e0_16 )

c     Convention for loop computations
      fo(1) = cmplx(p(0),rZero,KIND=16)*nsf
      fo(2) = cmplx(p(1),rZero,KIND=16)*nsf
      fo(3) = cmplx(p(2),rZero,KIND=16)*nsf
      fo(4) = cmplx(p(3),rZero,KIND=16)*nsf


      nh = nhel*nsf

      if ( fmass.ne.rZero ) then

         pp = min(p(0),sqrt(p(1)**2+p(2)**2+p(3)**2))

         if ( pp.eq.rZero ) then

            sqm(0) = sqrt(abs(fmass)) ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            ip = -((1+nh)/2)
            im =  (1-nh)/2

            fo(5) = im     * sqm(im)
            fo(6) = ip*nsf * sqm(im)
            fo(7) = im*nsf * sqm(-ip)
            fo(8) = ip     * sqm(-ip)

         else

            pp = min(p(0),sqrt(p(1)**2+p(2)**2+p(3)**2))
            sf(1) = real(1+nsf+(1-nsf)*nh,KIND=16)*rHalf
            sf(2) = real(1+nsf-(1-nsf)*nh,KIND=16)*rHalf
            omega(1) = sqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = max(pp+p(3),rZero)
            chi(1) = cmplx( sqrt(pp3*rHalf/pp) ,KIND=16)
            if ( pp3.eq.rZero ) then
               chi(2) = cmplx(-nh ,KIND=16)
            else
               chi(2) = cmplx( nh*p(1) , -p(2) , KIND=16)/
     & sqrt(rTwo*pp*pp3)
            endif

            fo(5) = sfomeg(2)*chi(im)
            fo(6) = sfomeg(2)*chi(ip)
            fo(7) = sfomeg(1)*chi(im)
            fo(8) = sfomeg(1)*chi(ip)

         endif

      else

         if(p(1).eq.0d0.and.p(2).eq.0d0.and.p(3).lt.0d0) then
            sqp0p3 = 0d0
         else
            sqp0p3 = sqrt(max(p(0)+p(3),rZero))*nsf
         end if
         chi(1) = cmplx( sqp0p3 ,KIND=16)
         if ( sqp0p3.eq.rZero ) then
            chi(2) = cmplx(-nhel , KIND=16)*sqrt(rTwo*p(0))
         else
            chi(2) = cmplx( nh*p(1), -p(2) , KIND=16)/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fo(5) = chi(1)
            fo(6) = chi(2)
            fo(7) = cmplx( rZero , KIND=16)
            fo(8) = cmplx( rZero , KIND=16)
         else
            fo(5) = cmplx( rZero , KIND=16)
            fo(6) = cmplx( rZero , KIND=16)
            fo(7) = chi(2)
            fo(8) = chi(1)
         endif

      endif
c
      return
      end

      subroutine pxxxxx(p,tmass,nhel,nst , tc)

c    CP3 2009.NOV

c This subroutine computes a PSEUDOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(18)         : PSEUDOR  wavefunction    epsilon^mu^nu(t)
c
      implicit none
      double precision p(0:3), tmass
      integer nhel, nst
      double complex tc(20)

      double complex ft(6,4), ep(4), em(4), e0(4)
      double precision pt, pt2, pp, pzpt, emp, sqh, sqs
      integer i, j

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )


      tc(5)=NHEL

c     Convention for trees
c      tc(17) = dcmplx(p(0),p(3))*nst
c      tc(18) = dcmplx(p(1),p(2))*nst

c     Convention for loop computations
      tc(1) = dcmplx(p(0),0.D0)*nst
      tc(2) = dcmplx(p(1),0.D0)*nst
      tc(3) = dcmplx(p(2),0.D0)*nst
      tc(4) = dcmplx(p(3),0.D0)*nst

      return
      end

      subroutine mp_pxxxxx(p,tmass,nhel,nst , tc)

c    CP3 2009.NOV

c This subroutine computes a PSEUDOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(18)         : PSEUDOR  wavefunction    epsilon^mu^nu(t)
c
      implicit none
      real*16 p(0:3), tmass
      integer nhel, nst
      complex*32 tc(20)

      complex*32 ft(6,4), ep(4), em(4), e0(4)
      real*16 pt, pt2, pp, pzpt, emp, sqh, sqs
      integer i, j

      real*16 rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0e0_16, rHalf = 0.5e0_16 )
      parameter( rOne = 1.0e0_16, rTwo = 2.0e0_16 )


      tc(5)=NHEL

c     Convention for trees
c      tc(17) = dcmplx(p(0),p(3))*nst
c      tc(18) = dcmplx(p(1),p(2))*nst

c     Convention for loop computations
      tc(1) = cmplx(p(0),0.0e0_16,KIND=16)*nst
      tc(2) = cmplx(p(1),0.0e0_16,KIND=16)*nst
      tc(3) = cmplx(p(2),0.0e0_16,KIND=16)*nst
      tc(4) = cmplx(p(3),0.0e0_16,KIND=16)*nst

      return
      end

      subroutine sxxxxx(p,nss , sc)
c
c This subroutine computes a complex SCALAR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc(3)          : scalar wavefunction                   s
c
      implicit none
      double complex sc(5)
      double precision p(0:3)
      integer nss

      double precision rOne
      parameter( rOne = 1.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      double precision rZero
c      parameter( rZero = 0.0d0 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      if ( abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in sxxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in sxxxxx has non-positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      p2 = p(0)**2-(p(1)**2+p(2)**2+p(3)**2)
c      if ( p2.lt.-p(0)**2*epsi ) then
c         write(stdo,*) ' helas-error : p(0:3) in sxxxxx is spacelike'
c         write(stdo,*) '             : p**2 = ',p2
c      endif
c      if ( abs(nss).ne.1 ) then
c         write(stdo,*) ' helas-error : nss in sxxxxx is not -1,1'
c         write(stdo,*) '             : nss = ',nss
c      endif
c#endif

      sc(5) = dcmplx( rOne )

c     Convention for trees
c      sc(2) = dcmplx(p(0),p(3))*nss
c      sc(3) = dcmplx(p(1),p(2))*nss

c     Convention for loop computations
      sc(1) = dcmplx(p(0),0.D0)*nss
      sc(2) = dcmplx(p(1),0.D0)*nss
      sc(3) = dcmplx(p(2),0.D0)*nss
      sc(4) = dcmplx(p(3),0.D0)*nss
c
      return
      end

      subroutine mp_sxxxxx(p,nss , sc)
c
c This subroutine computes a complex SCALAR wavefunction.
c in quadrupole precision.
c
c input:
c       real    p(0:3)         : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc(3)          : scalar wavefunction                   s
c
      implicit none
      complex*32 sc(5)
      real*16 p(0:3)
      integer nss

      real*16 rOne
      parameter( rOne = 1.0e0_16 )

      sc(5) = cmplx( rOne , KIND=16)

c     Convention for trees
c      sc(2) = dcmplx(p(0),p(3))*nss
c      sc(3) = dcmplx(p(1),p(2))*nss

c     Convention for loop computations
      sc(1) = cmplx(p(0),0.0e0_16, KIND=16)*nss
      sc(2) = cmplx(p(1),0.0e0_16, KIND=16)*nss
      sc(3) = cmplx(p(2),0.0e0_16, KIND=16)*nss
      sc(4) = cmplx(p(3),0.0e0_16, KIND=16)*nss
c
      return
      end

      subroutine txxxxx(p,tmass,nhel,nst , tc)
c
c This subroutine computes a TENSOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(18)         : tensor wavefunction    epsilon^mu^nu(t)
c
      implicit none
      double precision p(0:3), tmass
      integer nhel, nst
      double complex tc(20)

      double complex ft(8,4), ep(4), em(4), e0(4)
      double precision pt, pt2, pp, pzpt, emp, sqh, sqs
      integer i, j

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )

      integer stdo
      parameter( stdo = 6 )

      sqh = sqrt(rHalf)
      sqs = sqrt(rHalf/3.d0)

      pt2 = p(1)**2 + p(2)**2
      pp = min(p(0),sqrt(pt2+p(3)**2))
      pt = min(pp,sqrt(pt2))

c     Convention for trees
c      ft(5,1) = dcmplx(p(0),p(3))*nst
c      ft(6,1) = dcmplx(p(1),p(2))*nst

c     Convention for loop computations
      ft(5,1) = dcmplx(p(0),0.D0)*nst
      ft(6,1) = dcmplx(p(1),0.D0)*nst
      ft(7,1) = dcmplx(p(2),0.D0)*nst
      ft(8,1) = dcmplx(p(3),0.D0)*nst

      if ( nhel.ge.0 ) then
c construct eps+
         if ( pp.eq.rZero ) then
            ep(1) = dcmplx( rZero )
            ep(2) = dcmplx( -sqh )
            ep(3) = dcmplx( rZero , nst*sqh )
            ep(4) = dcmplx( rZero )
         else
            ep(1) = dcmplx( rZero )
            ep(4) = dcmplx( pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh
               ep(2) = dcmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh )
               ep(3) = dcmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh )
            else
               ep(2) = dcmplx( -sqh )
               ep(3) = dcmplx( rZero , nst*sign(sqh,p(3)) )
            endif
         endif
      end if

      if ( nhel.le.0 ) then
c construct eps-
         if ( pp.eq.rZero ) then
            em(1) = dcmplx( rZero )
            em(2) = dcmplx( sqh )
            em(3) = dcmplx( rZero , nst*sqh )
            em(4) = dcmplx( rZero )
         else
            em(1) = dcmplx( rZero )
            em(4) = dcmplx( -pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = -p(3)/(pp*pt)*sqh
               em(2) = dcmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh )
               em(3) = dcmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh )
            else
               em(2) = dcmplx( sqh )
               em(3) = dcmplx( rZero , nst*sign(sqh,p(3)) )
            endif
         endif
      end if

      if ( abs(nhel).le.1 ) then
c construct eps0
         if ( pp.eq.rZero ) then
            e0(1) = dcmplx( rZero )
            e0(2) = dcmplx( rZero )
            e0(3) = dcmplx( rZero )
            e0(4) = dcmplx( rOne )
         else
            emp = p(0)/(tmass*pp)
            e0(1) = dcmplx( pp/tmass )
            e0(4) = dcmplx( p(3)*emp )
            if ( pt.ne.rZero ) then
               e0(2) = dcmplx( p(1)*emp )
               e0(3) = dcmplx( p(2)*emp )
            else
               e0(2) = dcmplx( rZero )
               e0(3) = dcmplx( rZero )
            endif
         end if
      end if

      if ( nhel.eq.2 ) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = ep(i)*ep(j)
            end do
         end do
      else if ( nhel.eq.-2 ) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = em(i)*em(j)
            end do
         end do
      else if (tmass.eq.0) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = 0
            end do
         end do
      else if (tmass.ne.0) then
        if  ( nhel.eq.1 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqh*( ep(i)*e0(j) + e0(i)*ep(j) )
              end do
           end do
        else if ( nhel.eq.0 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqs*( ep(i)*em(j) + em(i)*ep(j)
     &                                + rTwo*e0(i)*e0(j) )
              end do
           end do
        else if ( nhel.eq.-1 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqh*( em(i)*e0(j) + e0(i)*em(j) )
              end do
           end do
        else
           write(stdo,*) 'invalid helicity in TXXXXX'
           stop
        end if
      end if

      tc(5) = ft(1,1)
      tc(6) = ft(1,2)
      tc(7) = ft(1,3)
      tc(8) = ft(1,4)
      tc(9) = ft(2,1)
      tc(10) = ft(2,2)
      tc(11) = ft(2,3)
      tc(12) = ft(2,4)
      tc(13) = ft(3,1)
      tc(14) = ft(3,2)
      tc(15) = ft(3,3)
      tc(16) = ft(3,4)
      tc(17) = ft(4,1)
      tc(18) = ft(4,2)
      tc(19) = ft(4,3)
      tc(20) = ft(4,4)

      tc(1) = ft(5,1)
      tc(2) = ft(6,1)
      tc(3) = ft(7,1)
      tc(4) = ft(8,1)

      return
      end


      subroutine mp_txxxxx(p,tmass,nhel,nst , tc)
c
c This subroutine computes a TENSOR wavefunction.
c
c input:
c       real    p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(18)         : tensor wavefunction    epsilon^mu^nu(t)
c
      implicit none
      real*16 p(0:3), tmass
      integer nhel, nst
      complex*32 tc(20)

      complex*32 ft(8,4), ep(4), em(4), e0(4)
      real*16 pt, pt2, pp, pzpt, emp, sqh, sqs
      integer i, j

      real*16 rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0e0_16, rHalf = 0.5e0_16 )
      parameter( rOne = 1.0e0_16, rTwo = 2.0e0_16 )

      integer stdo
      parameter( stdo = 6 )

      sqh = sqrt(rHalf)
      sqs = sqrt(rHalf/3.0e0_16)

      pt2 = p(1)**2 + p(2)**2
      pp = min(p(0),sqrt(pt2+p(3)**2))
      pt = min(pp,sqrt(pt2))

c     Convention for trees
c      ft(5,1) = dcmplx(p(0),p(3))*nst
c      ft(6,1) = dcmplx(p(1),p(2))*nst

c     Convention for loop computations
      ft(5,1) = cmplx(p(0),0.0e0_16,KIND=16)*nst
      ft(6,1) = cmplx(p(1),0.0e0_16,KIND=16)*nst
      ft(7,1) = cmplx(p(2),0.0e0_16,KIND=16)*nst
      ft(8,1) = cmplx(p(3),0.0e0_16,KIND=16)*nst

      if ( nhel.ge.0 ) then
c construct eps+
         if ( pp.eq.rZero ) then
            ep(1) = cmplx( rZero ,KIND=16)
            ep(2) = cmplx( -sqh ,KIND=16)
            ep(3) = cmplx( rZero , nst*sqh ,KIND=16)
            ep(4) = cmplx( rZero ,KIND=16)
         else
            ep(1) = cmplx( rZero ,KIND=16)
            ep(4) = cmplx( pt/pp*sqh ,KIND=16)
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh
               ep(2) = cmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh ,KIND=16)
               ep(3) = cmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh ,KIND=16)
            else
               ep(2) = cmplx( -sqh ,KIND=16)
               ep(3) = cmplx( rZero , nst*sign(sqh,p(3)) ,KIND=16)
            endif
         endif
      end if

      if ( nhel.le.0 ) then
c construct eps-
         if ( pp.eq.rZero ) then
            em(1) = cmplx( rZero ,KIND=16)
            em(2) = cmplx( sqh ,KIND=16)
            em(3) = cmplx( rZero , nst*sqh ,KIND=16)
            em(4) = cmplx( rZero ,KIND=16)
         else
            em(1) = cmplx( rZero ,KIND=16)
            em(4) = cmplx( -pt/pp*sqh ,KIND=16)
            if ( pt.ne.rZero ) then
               pzpt = -p(3)/(pp*pt)*sqh
               em(2) = cmplx( -p(1)*pzpt , -nst*p(2)/pt*sqh ,KIND=16)
               em(3) = cmplx( -p(2)*pzpt ,  nst*p(1)/pt*sqh ,KIND=16)
            else
               em(2) = cmplx( sqh ,KIND=16)
               em(3) = cmplx( rZero , nst*sign(sqh,p(3)) ,KIND=16)
            endif
         endif
      end if

      if ( abs(nhel).le.1 ) then
c construct eps0
         if ( pp.eq.rZero ) then
            e0(1) = cmplx( rZero ,KIND=16)
            e0(2) = cmplx( rZero ,KIND=16)
            e0(3) = cmplx( rZero ,KIND=16)
            e0(4) = cmplx( rOne ,KIND=16)
         else
            emp = p(0)/(tmass*pp)
            e0(1) = cmplx( pp/tmass ,KIND=16)
            e0(4) = cmplx( p(3)*emp ,KIND=16)
            if ( pt.ne.rZero ) then
               e0(2) = cmplx( p(1)*emp ,KIND=16)
               e0(3) = cmplx( p(2)*emp ,KIND=16)
            else
               e0(2) = cmplx( rZero ,KIND=16)
               e0(3) = cmplx( rZero ,KIND=16)
            endif
         end if
      end if

      if ( nhel.eq.2 ) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = ep(i)*ep(j)
            end do
         end do
      else if ( nhel.eq.-2 ) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = em(i)*em(j)
            end do
         end do
      else if (tmass.eq.0) then
         do j = 1,4
            do i = 1,4
               ft(i,j) = 0
            end do
         end do
      else if (tmass.ne.0) then
        if  ( nhel.eq.1 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqh*( ep(i)*e0(j) + e0(i)*ep(j) )
              end do
           end do
        else if ( nhel.eq.0 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqs*( ep(i)*em(j) + em(i)*ep(j)
     &                                + rTwo*e0(i)*e0(j) )
              end do
           end do
        else if ( nhel.eq.-1 ) then
           do j = 1,4
              do i = 1,4
                 ft(i,j) = sqh*( em(i)*e0(j) + e0(i)*em(j) )
              end do
           end do
        else
           write(stdo,*) 'invalid helicity in TXXXXX'
           stop
        end if
      end if

      tc(5) = ft(1,1)
      tc(6) = ft(1,2)
      tc(7) = ft(1,3)
      tc(8) = ft(1,4)
      tc(9) = ft(2,1)
      tc(10) = ft(2,2)
      tc(11) = ft(2,3)
      tc(12) = ft(2,4)
      tc(13) = ft(3,1)
      tc(14) = ft(3,2)
      tc(15) = ft(3,3)
      tc(16) = ft(3,4)
      tc(17) = ft(4,1)
      tc(18) = ft(4,2)
      tc(19) = ft(4,3)
      tc(20) = ft(4,4)

      tc(1) = ft(5,1)
      tc(2) = ft(6,1)
      tc(3) = ft(7,1)
      tc(4) = ft(8,1)

      return
      end

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
      double complex vc(8)
      double precision p(0:3),vmass,hel,hel0,pt,pt2,pp,pzpt,emp,sqh
      integer nhel,nsv,nsvahl

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision p2
c      double precision epsi
c      parameter( epsi = 2.0d-5 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
c      if ( abs(p(0))+pp.eq.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in vxxxxx is zero momentum'
c      endif
c      if ( p(0).le.rZero ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in vxxxxx has non-positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      p2 = (p(0)+pp)*(p(0)-pp)
c      if ( abs(p2-vmass**2).gt.p(0)**2*2.e-5 ) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in vxxxxx has inappropriate mass'
c         write(stdo,*)
c     &        '             : p**2 = ',p2,' : vmass**2 = ',vmass**2
c      endif
c      if ( vmass.ne.rZero ) then
c         if ( abs(nhel).gt.1 ) then
c            write(stdo,*) ' helas-error : nhel in vxxxxx is not -1,0,1'
c            write(stdo,*) '             : nhel = ',nhel
c         endif
c      else
c         if ( abs(nhel).ne.1 ) then
c            write(stdo,*) ' helas-error : nhel in vxxxxx is not -1,1'
c            write(stdo,*) '             : nhel = ',nhel
c         endif
c      endif
c      if ( abs(nsv).ne.1 ) then
c         write(stdo,*) ' helas-error : nsv in vmxxxx is not -1,1'
c         write(stdo,*) '             : nsv = ',nsv
c      endif
c#endif

      sqh = dsqrt(rHalf)
      hel = dble(nhel)
      nsvahl = nsv*dabs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = min(p(0),dsqrt(pt2+p(3)**2))
      pt = min(pp,dsqrt(pt2))

c     Convention for trees
c      vc(5) = dcmplx(p(0),p(3))*nsv
c      vc(6) = dcmplx(p(1),p(2))*nsv

c     Convention for loop computations
      vc(1) = dcmplx(p(0),0.D0)*nsv
      vc(2) = dcmplx(p(1),0.D0)*nsv
      vc(3) = dcmplx(p(2),0.D0)*nsv
      vc(4) = dcmplx(p(3),0.D0)*nsv

c#ifdef HELAS_CHECK
c nhel=4 option for scalar polarization
c      if( nhel.eq.4 ) then
c         if( vmass.eq.rZero ) then
c            vc(1) = rOne
c            vc(2) = p(1)/p(0)
c            vc(3) = p(2)/p(0)
c            vc(4) = p(3)/p(0)
c         else
c            vc(1) = p(0)/vmass
c            vc(2) = p(1)/vmass
c            vc(3) = p(2)/vmass
c            vc(4) = p(3)/vmass
c         endif
c         return
c      endif
c#endif

      if ( vmass.ne.rZero ) then

         hel0 = rOne-dabs(hel)

         if ( pp.eq.rZero ) then

            vc(5) = dcmplx( rZero )
            vc(6) = dcmplx(-hel*sqh )
            vc(7) = dcmplx( rZero , nsvahl*sqh )
            vc(8) = dcmplx( hel0 )

         else

            emp = p(0)/(vmass*pp)
            vc(5) = dcmplx( hel0*pp/vmass )
            vc(8) = dcmplx( hel0*p(3)*emp+hel*pt/pp*sqh )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc(6) = dcmplx( hel0*p(1)*emp-p(1)*pzpt ,
     &                         -nsvahl*p(2)/pt*sqh       )
               vc(7) = dcmplx( hel0*p(2)*emp-p(2)*pzpt ,
     &                          nsvahl*p(1)/pt*sqh       )
            else
               vc(6) = dcmplx( -hel*sqh )
               vc(7) = dcmplx( rZero , nsvahl*sign(sqh,p(3)) )
            endif

         endif

      else

         pp = p(0)
         pt = sqrt(p(1)**2+p(2)**2)
         vc(5) = dcmplx( rZero )
         vc(8) = dcmplx( hel*pt/pp*sqh )
         if ( pt.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc(6) = dcmplx( -p(1)*pzpt , -nsv*p(2)/pt*sqh )
            vc(7) = dcmplx( -p(2)*pzpt ,  nsv*p(1)/pt*sqh )
        else
            vc(6) = dcmplx( -hel*sqh )
            vc(7) = dcmplx( rZero , nsv*sign(sqh,p(3)) )
         endif

      endif
c
      return
      end

      subroutine mp_vxxxxx(p,vmass,nhel,nsv , vc)
c
c This subroutine computes a VECTOR wavefunction in quadruple precision.
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
      complex*32 vc(8)
      real*16 p(0:3),vmass,hel,hel0,pt,pt2,pp,pzpt,emp,sqh
      integer nhel,nsv,nsvahl

      real*16 rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0e0_16, rHalf = 0.5e0_16 )
      parameter( rOne = 1.0e0_16, rTwo = 2.0e0_16 )

      sqh = sqrt(rHalf)
      hel = real(nhel,KIND=16)
      nsvahl = nsv*abs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = min(p(0),sqrt(pt2+p(3)**2))
      pt = min(pp,sqrt(pt2))

c     Convention for loop computations
      vc(1) = cmplx(p(0),0.e0_16, KIND=16)*nsv
      vc(2) = cmplx(p(1),0.e0_16, KIND=16)*nsv
      vc(3) = cmplx(p(2),0.e0_16, KIND=16)*nsv
      vc(4) = cmplx(p(3),0.e0_16, KIND=16)*nsv

      if ( vmass.ne.rZero ) then

         hel0 = rOne-abs(hel)

         if ( pp.eq.rZero ) then

            vc(5) = cmplx( rZero , KIND=16)
            vc(6) = cmplx(-hel*sqh , KIND=16)
            vc(7) = cmplx( rZero , nsvahl*sqh , KIND=16)
            vc(8) = cmplx( hel0 , KIND=16)

         else

            emp = p(0)/(vmass*pp)
            vc(5) = cmplx( hel0*pp/vmass , KIND=16)
            vc(8) = cmplx( hel0*p(3)*emp+hel*pt/pp*sqh , KIND=16 )
            if ( pt.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc(6) = cmplx( hel0*p(1)*emp-p(1)*pzpt ,
     &                         -nsvahl*p(2)/pt*sqh       , KIND=16)
               vc(7) = cmplx( hel0*p(2)*emp-p(2)*pzpt ,
     &                          nsvahl*p(1)/pt*sqh       , KIND=16)
            else
               vc(6) = cmplx( -hel*sqh , KIND=16)
               vc(7) = cmplx( rZero , nsvahl*sign(sqh,p(3)) , KIND=16)
            endif

         endif

      else

         pp = p(0)
         pt = sqrt(p(1)**2+p(2)**2)
         vc(5) = cmplx( rZero , KIND=16)
         vc(8) = cmplx( hel*pt/pp*sqh , KIND=16)
         if ( pt.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc(6) = cmplx( -p(1)*pzpt , -nsv*p(2)/pt*sqh , KIND=16)
            vc(7) = cmplx( -p(2)*pzpt ,  nsv*p(1)/pt*sqh , KIND=16)
         else
            vc(6) = cmplx( -hel*sqh , KIND=16)
            vc(7) = cmplx( rZero , nsv*sign(sqh,p(3)), KIND=16 )
         endif

      endif
c
      return
      end

      subroutine boostx(p,q , pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momentum p is assumed to be given in the rest frame of q.  pboost is
c the momentum p boosted to the frame in which q is given.  q must be a
c timelike momentum.
c
c input:
c       real    p(0:3)         : four-momentum p in the q rest  frame
c       real    q(0:3)         : four-momentum q in the boosted frame
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

      qq = q(1)**2+q(2)**2+q(3)**2

c#ifdef HELAS_CHECK
c      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in boostx is zero momentum'
c      endif
c      if (abs(q(0))+qq.eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is zero momentum'
c      endif
c      if (p(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      if (q(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : q(0) = ',q(0)
c      endif
c      pp=p(0)**2-p(1)**2-p(2)**2-p(3)**2
c      if (pp.lt.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx is spacelike'
c         write(stdo,*)
c     &        '             : p**2 = ',pp
c      endif
c      if (q(0)**2-qq.le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is not timelike'
c         write(stdo,*)
c     &        '             : q**2 = ',q(0)**2-qq
c      endif
c      if (qq.eq.rZero) then
c         write(stdo,*)
c     &   ' helas-warn  : q(0:3) in boostx has zero spacial components'
c      endif
c#endif

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         m = sqrt(q(0)**2-qq)
         lf = ((q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)+pq)/m
         pboost(1) =  p(1)+q(1)*lf
         pboost(2) =  p(2)+q(2)*lf
         pboost(3) =  p(3)+q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end

      subroutine boostm(p,q,m, pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momentum p is assumed to be given in the rest frame of q.  pboost is
c the momentum p boosted to the frame in which q is given.  q must be a
c timelike momentum.
c
c input:
c       real    p(0:3)         : four-momentum p in the q rest  frame
c       real    q(0:3)         : four-momentum q in the boosted frame
c       real    m        : mass of q (for numerical stability)
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

c#ifdef HELAS_CHECK
c      integer stdo
c      parameter( stdo = 6 )
c      double precision pp
c#endif
c
      qq = q(1)**2+q(2)**2+q(3)**2

c#ifdef HELAS_CHECK
c      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in boostx is zero momentum'
c      endif
c      if (abs(q(0))+qq.eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is zero momentum'
c      endif
c      if (p(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : p(0) = ',p(0)
c      endif
c      if (q(0).le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx has not positive energy'
c         write(stdo,*)
c     &        '             : q(0) = ',q(0)
c      endif
c      pp=p(0)**2-p(1)**2-p(2)**2-p(3)**2
c      if (pp.lt.rZero) then
c         write(stdo,*)
c     &        ' helas-warn  : p(0:3) in boostx is spacelike'
c         write(stdo,*)
c     &        '             : p**2 = ',pp
c      endif
c      if (q(0)**2-qq.le.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in boostx is not timelike'
c         write(stdo,*)
c     &        '             : q**2 = ',q(0)**2-qq
c      endif
c      if (qq.eq.rZero) then
c         write(stdo,*)
c     &   ' helas-warn  : q(0:3) in boostx has zero spacial components'
c      endif
c#endif

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         lf = ((q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)+pq)/m
         pboost(1) =  p(1)+q(1)*lf
         pboost(2) =  p(2)+q(2)*lf
         pboost(3) =  p(3)+q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end

      subroutine momntx(energy,mass,costh,phi , p)
c
c This subroutine sets up a four-momentum from the four inputs.
c
c input:
c       real    energy         : energy
c       real    mass           : mass
c       real    costh          : cos(theta)
c       real    phi            : azimuthal angle
c
c output:
c       real    p(0:3)         : four-momentum
c
      implicit none
      double precision p(0:3),energy,mass,costh,phi,pp,sinth

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

c#ifdef HELAS_CHECK
c      double precision rPi, rTwo
c      parameter( rPi = 3.14159265358979323846d0, rTwo = 2.d0 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
c#ifdef HELAS_CHECK
c      if (energy.lt.mass) then
c         write(stdo,*)
c     &        ' helas-error : energy in momntx is less than mass'
c         write(stdo,*)
c     &        '             : energy = ',energy,' : mass = ',mass
c      endif
c      if (mass.lt.rZero) then
c         write(stdo,*) ' helas-error : mass in momntx is negative'
c         write(stdo,*) '             : mass = ',mass
c      endif
c      if (abs(costh).gt.rOne) then
c         write(stdo,*) ' helas-error : costh in momntx is improper'
c         write(stdo,*) '             : costh = ',costh
c      endif
c      if (phi.lt.rZero .or. phi.gt.rTwo*rPi) then
c         write(stdo,*)
c     &   ' helas-warn  : phi in momntx does not lie on 0.0 thru 2.0*pi'
c         write(stdo,*)
c     &   '             : phi = ',phi
c      endif
c#endif

      p(0) = energy

      if ( energy.eq.mass ) then

         p(1) = rZero
         p(2) = rZero
         p(3) = rZero

      else

         pp = sqrt((energy-mass)*(energy+mass))
         sinth = sqrt((rOne-costh)*(rOne+costh))
         p(3) = pp*costh
         if ( phi.eq.rZero ) then
            p(1) = pp*sinth
            p(2) = rZero
         else
            p(1) = pp*sinth*cos(phi)
            p(2) = pp*sinth*sin(phi)
         endif

      endif
c
      return
      end
      subroutine rotxxx(p,q , prot)
c
c This subroutine performs the spacial rotation of a four-momentum.
c the momentum p is assumed to be given in the frame where the spacial
c component of q points the positive z-axis.  prot is the momentum p
c rotated to the frame where q is given.
c
c input:
c       real    p(0:3)         : four-momentum p in q(1)=q(2)=0 frame
c       real    q(0:3)         : four-momentum q in the rotated frame
c
c output:
c       real    prot(0:3)      : four-momentum p in the rotated frame
c
      implicit none
      double precision p(0:3),q(0:3),prot(0:3),qt2,qt,psgn,qq,p1

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

c#ifdef HELAS_CHECK
c      integer stdo
c      parameter( stdo = 6 )
c#endif
c
      prot(0) = p(0)

      qt2 = q(1)**2 + q(2)**2

c#ifdef HELAS_CHECK
c      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : p(0:3) in rotxxx is zero momentum'
c      endif
c      if (abs(q(0))+abs(q(3))+qt2.eq.rZero) then
c         write(stdo,*)
c     &        ' helas-error : q(0:3) in rotxxx is zero momentum'
c      endif
c      if (qt2+abs(q(3)).eq.rZero) then
c         write(stdo,*)
c     &   ' helas-warn  : q(0:3) in rotxxx has zero spacial momentum'
c      endif
c#endif

      if ( qt2.eq.rZero ) then
          if ( q(3).eq.rZero ) then
             prot(1) = p(1)
             prot(2) = p(2)
             prot(3) = p(3)
          else
             psgn = dsign(rOne,q(3))
             prot(1) = p(1)*psgn
             prot(2) = p(2)*psgn
             prot(3) = p(3)*psgn
          endif
      else
          qq = sqrt(qt2+q(3)**2)
          qt = sqrt(qt2)
          p1 = p(1)
          prot(1) = q(1)*q(3)/qq/qt*p1 -q(2)/qt*p(2) +q(1)/qq*p(3)
          prot(2) = q(2)*q(3)/qq/qt*p1 +q(1)/qt*p(2) +q(2)/qq*p(3)
          prot(3) =          -qt/qq*p1               +q(3)/qq*p(3)
      endif
c
      return
      end

      subroutine mom2cx(esum,mass1,mass2,costh1,phi1 , p1,p2)
c
c This subroutine sets up two four-momenta in the two particle rest
c frame.
c
c input:
c       real    esum           : energy sum of particle 1 and 2
c       real    mass1          : mass            of particle 1
c       real    mass2          : mass            of particle 2
c       real    costh1         : cos(theta)      of particle 1
c       real    phi1           : azimuthal angle of particle 1
c
c output:
c       real    p1(0:3)        : four-momentum of particle 1
c       real    p2(0:3)        : four-momentum of particle 2
c     
      implicit none
      double precision p1(0:3),p2(0:3),
     &     esum,mass1,mass2,costh1,phi1,md2,ed,pp,sinth1

      double precision rZero, rHalf, rOne, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )

c#ifdef HELAS_CHECK
c      double precision rPi
c      parameter( rPi = 3.14159265358979323846d0 )
c      integer stdo
c      parameter( stdo = 6 )
c#endif
cc
c#ifdef HELAS_CHECK
c      if (esum.lt.mass1+mass2) then
c         write(stdo,*)
c     &        ' helas-error : esum in mom2cx is less than mass1+mass2'
c         write(stdo,*)
c     &        '             : esum = ',esum,
c     &        ' : mass1+mass2 = ',mass1,mass2
c      endif
c      if (mass1.lt.rZero) then
c         write(stdo,*) ' helas-error : mass1 in mom2cx is negative'
c         write(stdo,*) '             : mass1 = ',mass1
c      endif
c      if (mass2.lt.rZero) then
c         write(stdo,*) ' helas-error : mass2 in mom2cx is negative'
c         write(stdo,*) '             : mass2 = ',mass2
c      endif
c      if (abs(costh1).gt.1.) then
c         write(stdo,*) ' helas-error : costh1 in mom2cx is improper'
c         write(stdo,*) '             : costh1 = ',costh1
c      endif
c      if (phi1.lt.rZero .or. phi1.gt.rTwo*rPi) then
c         write(stdo,*)
c     &   ' helas-warn  : phi1 in mom2cx does not lie on 0.0 thru 2.0*pi'
c         write(stdo,*) 
c     &   '             : phi1 = ',phi1
c      endif
c#endif

      md2 = (mass1-mass2)*(mass1+mass2)
      ed = md2/esum
      if ( mass1*mass2.eq.rZero ) then
         pp = (esum-abs(ed))*rHalf
      else
         pp = sqrt((md2/esum)**2-rTwo*(mass1**2+mass2**2)+esum**2)*rHalf
      endif
      sinth1 = sqrt((rOne-costh1)*(rOne+costh1))

      p1(0) = max((esum+ed)*rHalf,rZero)
      p1(1) = pp*sinth1*cos(phi1)
      p1(2) = pp*sinth1*sin(phi1)
      p1(3) = pp*costh1

      p2(0) = max((esum-ed)*rHalf,rZero)
      p2(1) = -p1(1)
      p2(2) = -p1(2)
      p2(3) = -p1(3)
c
      return
      end

C###############################################################################
C LOOP related universal subroutines
C###############################################################################

C===============================================================================
C Subroutines to create the external wavefunctions of the L-cut particles 
C===============================================================================

      SUBROUTINE LCUT_F(Q,CFIG,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      COMPLEX*16 W(20)
      
      CALL LCUT_V(Q,CFIG,W)
      END

      SUBROUTINE LCUT_AF(Q,CFIG,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      COMPLEX*16 W(20)
      
      CALL LCUT_V(Q,CFIG,W)
      END

      SUBROUTINE LCUT_V(Q,CFIG,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      COMPLEX*16 W(20)
      
      W(5)=(0.d0,0.d0)
      W(6)=(0.d0,0.d0)
      W(7)=(0.d0,0.d0)
      W(8)=(0.d0,0.d0)
      W(CFIG+4)=(1.d0,0.d0)
      
      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

      SUBROUTINE LCUT_S(Q,CFIG,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      COMPLEX*16 W(20)

      W(5)=(1.D0,0.D0)

      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

      SUBROUTINE LCUT_AS(Q,CFIG,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      COMPLEX*16 W(20)

      W(5)=(1.D0,0.D0)

      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

      SUBROUTINE MP_LCUT_F(Q,CFIG,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      COMPLEX*32 W(20)
      
      CALL MP_LCUT_V(Q,CFIG,W)
      END

      SUBROUTINE MP_LCUT_AF(Q,CFIG,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      COMPLEX*32 W(20)
      
      CALL MP_LCUT_V(Q,CFIG,W)
      END

      SUBROUTINE MP_LCUT_V(Q,CFIG,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      COMPLEX*32 W(20)
      COMPLEX*32 IONE, IZERO
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))
      PARAMETER (IZERO=(0.0e0_16,0.0e0_16))      
      
      W(5)=IZERO
      W(6)=IZERO
      W(7)=IZERO
      W(8)=IZERO
      W(CFIG+4)=IONE

      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

      SUBROUTINE MP_LCUT_AS(Q,CFIG,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      COMPLEX*32 W(20)
      COMPLEX*32 IONE
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))

      W(5)=IONE

      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

      SUBROUTINE MP_LCUT_S(Q,CFIG,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      COMPLEX*32 W(20)
      COMPLEX*32 IONE
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))

      W(5)=IONE

      W(1)=Q(0)
      W(2)=Q(1)
      W(3)=Q(2)
      W(4)=Q(3)

      END

C===============================================================================
C Subroutines to close the lorentz traces of loops, 
C===============================================================================

      SUBROUTINE CLOSE_4(AMPS,RES)
      
      COMPLEX*16 RES
      COMPLEX*16 AMPS(4)

      RES=AMPS(1)+AMPS(2)+AMPS(3)+AMPS(4)

      END

      SUBROUTINE CLOSE_1(AMPS,RES)
      
      COMPLEX*16 RES
      COMPLEX*16 AMPS

      RES=AMPS

      END

      SUBROUTINE MP_CLOSE_4(AMPS,RES)
      
      COMPLEX*32 RES
      COMPLEX*32 AMPS(4)

      RES=AMPS(1)+AMPS(2)+AMPS(3)+AMPS(4)

      END

      SUBROUTINE MP_CLOSE_1(AMPS,RES)
      
      COMPLEX*32 RES
      COMPLEX*32 AMPS

      RES=AMPS

      END

C===============================================================================
C OLD Subroutines to close the lorentz traces of loops, 
c                           OBSOLETE 
C===============================================================================

      SUBROUTINE CLOSE_V(Q,M,AMPS,RES)
      
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      COMPLEX*16 AMPS(4)
      COMPLEX*16 M

      IF (M.NE.(0.D0,0.0D0)) THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      RES=AMPS(1)-AMPS(2)-AMPS(3)-AMPS(4)

      END

c This subroutine is to recreate the fermion propagator with 4 helicities
c only. This has problems with certain configuration of the imaginary
c momentum q, so it is not implemented yet.

      SUBROUTINE CLOSE_F4HEL(Q,M,AMPS,RES)      
      
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES, QNORM
      REAL*8 M
      COMPLEX*16 AMPS(4) 
      COMPLEX*16 PMM, PPM

      PPM=AMPS(1)+AMPS(2)
      PMM=AMPS(3)+AMPS(4)
      write(*,*) 'PPM=',PPM
      write(*,*) 'PMM=',PMM      
      IF (M.NE.0.D0) THEN
        QNORM=SQRT(Q(0)**2-Q(1)**2-Q(2)**2-Q(3)**2)
        write(*,*) 'Q=',Q        
        write(*,*) 'QNORM=',QNORM
        write(*,*) 'M=',M
        RES=(0.D0,0.5D0)*((PPM+PMM)+(PPM-PMM)*(M/QNORM))
        write(*,*) 'RES1=',RES
      ELSE
        RES=(0.D0,0.5D0)*(PPM+PMM)
        write(*,*) 'RES2=',RES
      ENDIF

      END

      SUBROUTINE CLOSE_F(Q,M,AMPS,RES)      
      
      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      COMPLEX*16 M
      COMPLEX*16 AMPS(12) 

      RES=(0.D0,0.D0)
      RES=(Q(0)-Q(3))*AMPS(1)+
     &    (-Q(1)+(0.0d0,1.0d0)*Q(2))*AMPS(2)+
     &    (-Q(1)-(0.0d0,1.0d0)*Q(2))*AMPS(3)+
     &    (Q(0)+Q(3))*AMPS(4)+
     &    (Q(0)+Q(3))*AMPS(5)+
     &    (Q(1)-(0.0d0,1.0d0)*Q(2))*AMPS(6)+
     &    (Q(1)+(0.0d0,1.0d0)*Q(2))*AMPS(7)+
     &    (Q(0)-Q(3))*AMPS(8)
      IF (M.NE.(0.D0,0.D0)) THEN
        RES=RES-M*(AMPS(9)+AMPS(10)+AMPS(11)+AMPS(12))
      ENDIF

      END

      SUBROUTINE CLOSE_S(Q,AMP,RES)

      COMPLEX*16 Q(0:3)
      COMPLEX*16 RES
      COMPLEX*16 AMP

      RES=(1.D0,0.D0)*AMP
      
      END

c     // QUAD PREC VERSIONS OF THE ABOVE //

      SUBROUTINE MP_CLOSE_V(Q,M,AMPS,RES)
      
      COMPLEX*32 Q(0:3)
      COMPLEX*32 RES
      COMPLEX*32 AMPS(4)
      COMPLEX*32 M

      IF (M.NE.(0.0e0_16,0.0e0_16)) THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      RES=AMPS(1)-AMPS(2)-AMPS(3)-AMPS(4)

      END

      SUBROUTINE MP_CLOSE_F(Q,M,AMPS,RES)      
      
      COMPLEX*32 Q(0:3)
      COMPLEX*32 RES
      COMPLEX*32 M
      COMPLEX*32 AMPS(12) 

      RES=(0.0e0_16,0.0e0_16)
      RES=(Q(0)-Q(3))*AMPS(1)+
     &    (-Q(1)+(0.0e0_16,1.0e0_16)*Q(2))*AMPS(2)+
     &    (-Q(1)-(0.0e0_16,1.0e0_16)*Q(2))*AMPS(3)+
     &    (Q(0)+Q(3))*AMPS(4)+
     &    (Q(0)+Q(3))*AMPS(5)+
     &    (Q(1)-(0.0e0_16,1.0e0_16)*Q(2))*AMPS(6)+
     &    (Q(1)+(0.0e0_16,1.0e0_16)*Q(2))*AMPS(7)+
     &    (Q(0)-Q(3))*AMPS(8)
      IF (M.NE.(0.0e0_16,0.0e0_16)) THEN
        RES=RES-M*(AMPS(9)+AMPS(10)+AMPS(11)+AMPS(12))
      ENDIF

      END

      SUBROUTINE MP_CLOSE_S(Q,AMP,RES)

      COMPLEX*32 Q(0:3)
      COMPLEX*32 RES
      COMPLEX*32 AMP

      RES=(1.0e0_16,0.0e0_16)*AMP
      
      END

C===============================================================================
C OLD Subroutines to create the external wavefunctions of the L-cut particles 
C                                 OBSOLETE
C===============================================================================

c This subroutine is to recreate the fermion propagator with 4 helicities
c only. This has problems with certain configuration of the imaginary
c momentum q, so it is not implemented yet.

      SUBROUTINE LCUT_F4HEL(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG,J
      LOGICAL SCD
      COMPLEX*16 M
      COMPLEX*16 W(20)

      IF (CFIG.EQ.1) THEN
        IF (SCD) THEN
C         UBAR, HEL=-1        
          CALL ILXXXX(Q(0),M,-1,1,W(1))
        ELSE
C         U, HEL=-1
          CALL ILXXXX(Q(0),M,-1,-1,W(1))
          do J=1,4
          write(*,*) 'Wcf(',j,',1)=',W(1)   
          enddo
        ENDIF
      ELSEIF (CFIG.EQ.2) THEN
        IF (SCD) THEN
C         UBAR, HEL=1        
          CALL ILXXXX(Q(0),M,1,1,W(1))
        ELSE
C         U, HEL=1        
          CALL ILXXXX(Q(0),M,1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.3) THEN
        IF (SCD) THEN
C         VBAR, HEL=-1,        
          CALL OLXXXX(Q(0),M,-1,-1,W(1))
        ELSE
C         V, HEL=-1        
          CALL OLXXXX(Q(0),M,-1,1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.4) THEN
        IF (SCD) THEN
C         VBAR, HEL=1        
          CALL OLXXXX(Q(0),M,1,-1,W(1))
        ELSE
C         V, HEL=1    
          CALL OLXXXX(Q(0),M,1,1,W(1))
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ENDIF

      END

      SUBROUTINE OLD_LCUT_F(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD
      COMPLEX*16 M
      COMPLEX*16 W(20)

      W(1)=(0.d0,0.d0)
      W(2)=(0.d0,0.d0)
      W(3)=(0.d0,0.d0)
      W(4)=(0.d0,0.d0)

      IF (CFIG.eq.1) then
        IF (SCD) then
          W(3)=(1.d0,0.d0)
        ELSE
          W(1)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.2) then
        IF (SCD) then
          W(4)=(1.d0,0.d0)
        ELSE
          W(1)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.3) then
        IF (SCD) then
          W(3)=(1.d0,0.d0)
        ELSE
          W(2)=(1.d0,0.d0)
        ENDIF  
      ELSEIF (CFIG.eq.4) then
        IF (SCD) then
          W(4)=(1.d0,0.d0)
        ELSE
          W(2)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.5) then
        IF (SCD) then
          W(1)=(1.d0,0.d0)
        ELSE
          W(3)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.6) then
        IF (SCD) then
          W(2)=(1.d0,0.d0)
        ELSE
          W(3)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.7) then
        IF (SCD) then
          W(1)=(1.d0,0.d0)
        ELSE
          W(4)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.8) then
        IF (SCD) then
          W(2)=(1.d0,0.d0)
        ELSE
          W(4)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.9) then
        IF (SCD) then
          W(1)=(1.d0,0.d0)
        ELSE
          W(1)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.10) then
        IF (SCD) then
          W(2)=(1.d0,0.d0)
        ELSE
          W(2)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.11) then
        IF (SCD) then
          W(3)=(1.d0,0.d0)
        ELSE
          W(3)=(1.d0,0.d0)
        ENDIF
      ELSEIF (CFIG.eq.12) then
        IF (SCD) then
          W(4)=(1.d0,0.d0)
        ELSE
          W(4)=(1.d0,0.d0)
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ELSE
        W(5)=Q(0)
        W(6)=Q(1)
        W(7)=Q(2)
        W(8)=Q(3)
      ENDIF

      END

      SUBROUTINE OLD_MP_LCUT_F(Q,M,CFIG,SCD,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD
      COMPLEX*32 M
      COMPLEX*32 W(20)
      COMPLEX*32 IZERO
      PARAMETER (IZERO=(0.0e0_16,0.0e0_16))
      COMPLEX*32 IONE
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))

      W(1)=IZERO
      W(2)=IZERO
      W(3)=IZERO
      W(4)=IZERO

      IF (CFIG.eq.1) then
        IF (SCD) then
          W(3)=IONE
        ELSE
          W(1)=IONE
        ENDIF
      ELSEIF (CFIG.eq.2) then
        IF (SCD) then
          W(4)=IONE
        ELSE
          W(1)=IONE
        ENDIF
      ELSEIF (CFIG.eq.3) then
        IF (SCD) then
          W(3)=IONE
        ELSE
          W(2)=IONE
        ENDIF  
      ELSEIF (CFIG.eq.4) then
        IF (SCD) then
          W(4)=IONE
        ELSE
          W(2)=IONE
        ENDIF
      ELSEIF (CFIG.eq.5) then
        IF (SCD) then
          W(1)=IONE
        ELSE
          W(3)=IONE
        ENDIF
      ELSEIF (CFIG.eq.6) then
        IF (SCD) then
          W(2)=IONE
        ELSE
          W(3)=IONE
        ENDIF
      ELSEIF (CFIG.eq.7) then
        IF (SCD) then
          W(1)=IONE
        ELSE
          W(4)=IONE
        ENDIF
      ELSEIF (CFIG.eq.8) then
        IF (SCD) then
          W(2)=IONE
        ELSE
          W(4)=IONE
        ENDIF
      ELSEIF (CFIG.eq.9) then
        IF (SCD) then
          W(1)=IONE
        ELSE
          W(1)=IONE
        ENDIF
      ELSEIF (CFIG.eq.10) then
        IF (SCD) then
          W(2)=IONE
        ELSE
          W(2)=IONE
        ENDIF
      ELSEIF (CFIG.eq.11) then
        IF (SCD) then
          W(3)=IONE
        ELSE
          W(3)=IONE
        ENDIF
      ELSEIF (CFIG.eq.12) then
        IF (SCD) then
          W(4)=IONE
        ELSE
          W(4)=IONE
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ELSE
        W(5)=Q(0)
        W(6)=Q(1)
        W(7)=Q(2)
        W(8)=Q(3)
      ENDIF

      END

      SUBROUTINE OLD_LCUT_CF(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD
      COMPLEX*16 M
      COMPLEX*16 W(20)

      IF (CFIG.EQ.1) THEN
        IF (SCD) THEN
C         UBAR, HEL=-1        
          CALL ICLXXX(Q(0),M,-1,1,W(1))
        ELSE
C         U, HEL=-1        
          CALL ICLXXX(Q(0),M,-1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.2) THEN
        IF (SCD) THEN
C         UBAR, HEL=1        
          CALL ICLXXX(Q(0),M,1,1,W(1))
        ELSE
C         U, HEL=1        
          CALL ICLXXX(Q(0),M,1,-1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.3) THEN
        IF (SCD) THEN
C         VBAR, HEL=-1,        
          CALL OCLXXX(Q(0),M,-1,-1,W(1))
        ELSE
C         V, HEL=-1        
          CALL OCLXXX(Q(0),M,-1,1,W(1))
        ENDIF
      ELSEIF (CFIG.EQ.4) THEN
        IF (SCD) THEN
C         VBAR, HEL=1        
          CALL OCLXXX(Q(0),M,1,-1,W(1))
        ELSE
C         V, HEL=1    
          CALL OCLXXX(Q(0),M,1,1,W(1))
        ENDIF
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT SPINORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ENDIF

      END

      SUBROUTINE OLD_LCUT_V(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD      
      COMPLEX*16 M
      COMPLEX*16 W(20)

      IF (M.NE.0.D0) THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      W(1)=(0.d0,0.d0)
      W(2)=(0.d0,0.d0)
      W(3)=(0.d0,0.d0)
      W(4)=(0.d0,0.d0)
      IF (CFIG.EQ.1) THEN
        W(1)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.2) THEN
        W(2)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.3) THEN
        W(3)=(1.d0,0.d0)
      ELSEIF (CFIG.EQ.4) THEN
        W(4)=(1.d0,0.d0)
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT VECTORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ELSE
        W(5)=Q(0)
        W(6)=Q(1)
        W(7)=Q(2)
        W(8)=Q(3)
      ENDIF

      END

      SUBROUTINE OLD_MP_LCUT_V(Q,M,CFIG,SCD,W)

      COMPLEX*32 Q(0:3)
      INTEGER CFIG
      LOGICAL SCD      
      COMPLEX*32 M
      COMPLEX*32 W(20)
      COMPLEX*32 IZERO
      PARAMETER (IZERO=(0.0e0_16,0.0e0_16))
      COMPLEX*32 IONE
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))


      IF (M.NE.IZERO) THEN
        STOP 'Massive vector L-cut particle not supported'
      ENDIF
      W(1)=IZERO
      W(2)=IZERO
      W(3)=IZERO
      W(4)=IZERO
      IF (CFIG.EQ.1) THEN
        W(1)=IONE
      ELSEIF (CFIG.EQ.2) THEN
        W(2)=IONE
      ELSEIF (CFIG.EQ.3) THEN
        W(3)=IONE
      ELSEIF (CFIG.EQ.4) THEN
        W(4)=IONE
      ENDIF
C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND L-CUT VECTORS      
      IF (SCD) THEN
        W(5)=-Q(0)
        W(6)=-Q(1)
        W(7)=-Q(2)
        W(8)=-Q(3)
      ELSE
        W(5)=Q(0)
        W(6)=Q(1)
        W(7)=Q(2)
        W(8)=Q(3)
      ENDIF

      END

      SUBROUTINE OLD_LCUT_S(Q,M,CFIG,SCD,W)

      COMPLEX*16 Q(0:3)
      COMPLEX*16 M
      INTEGER CFIG
      LOGICAL SCD      
      COMPLEX*16 W(20)

      W(1)=(1.D0,0.D0)

C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND SCALAR      
      IF (SCD) THEN
        W(2)=-Q(0)
        W(3)=-Q(1)
        W(4)=-Q(2)
        W(5)=-Q(3)
      ELSE
        W(2)=Q(0)
        W(3)=Q(1)
        W(4)=Q(2)
        W(5)=Q(3)
      ENDIF

      END

      SUBROUTINE OLD_MP_LCUT_S(Q,M,CFIG,SCD,W)

      COMPLEX*32 Q(0:3)
      COMPLEX*32 M
      INTEGER CFIG
      LOGICAL SCD      
      COMPLEX*32 W(20)
      COMPLEX*32 IONE
      PARAMETER (IONE=(1.0e0_16,0.0e0_16))

      W(1)=IONE

C     REVERSE THE MOMENTUM IN THE WF FOR THE SECOND SCALAR      
      IF (SCD) THEN
        W(2)=-Q(0)
        W(3)=-Q(1)
        W(4)=-Q(2)
        W(5)=-Q(3)
      ELSE
        W(2)=Q(0)
        W(3)=Q(1)
        W(4)=Q(2)
        W(5)=Q(3)
      ENDIF

      END

C===============================================================================
C Complex-momentum version of the subroutine to create on-shell
C wavefunctions of particles with different spins. OBSOLETE
C===============================================================================
   
C     The subroutine with charge conjugation are not yet implemented
c     Obsolete by now too

      subroutine oclxxx(p,ffmass,nhel,nsf,fo)

      implicit none
      double complex fo(8),p(0:3)
      double precision ffmass
      integer nhel,nsf

      CALL olxxxx(p,ffmass,nhel,nsf,fo)

      end

      subroutine iclxxx(p,ffmass,nhel,nsf,fi)

      implicit none
      double complex fi(8),p(0:3)
      double precision ffmass
      integer nhel,nsf

      CALL ilxxxx(p,ffmass,nhel,nsf,fi)

      end

      subroutine ilxxxx(p,ffmass,nhel,nsf,fi)
c
c This subroutine computes a fermion wavefunction with the flowing-IN
c fermion number and defined with complex ONSHELL momentium.
c
c input:
c       complex    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(8)          : fermion wavefunction               |fi>
c       Note: There are 4 components for the spinor and four for the
c             momentum. 
      implicit none
      double complex fi(8),chi(2), fmass
c      double precision p(0:3),sf(2),sfomeg(2),omega(2),fmass,
c     &     pp,pp3,sqp0p3,sqm(0:1)
      double complex sqm(0:1)
      double precision sf(2),ffmass
      double complex p(0:3), sfomeg(2),omega(2),
     &     pp,pp3,sqp0p3 
      integer nhel,nsf,ip,im,nh

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      

      
c      fi(5) = dcmplx(p(0),p(3))*nsf
c      fi(6) = dcmplx(p(1),p(2))*nsf
      fi(5) = p(0)*nsf
      fi(6) = p(1)*nsf
      fi(7) = p(2)*nsf
      fi(8) = p(3)*nsf

      nh = nhel*nsf

      fmass = sqrt(p(0)**2-p(1)**2-p(2)**2-p(3)**2)

      if ( ffmass.ne.rZero ) then
c special treatment for massless particles.
c         pp = min(p(0),sqrt(p(1)**2+p(2)**2+p(3)**2))
         pp=sqrt(p(1)**2+p(2)**2+p(3)**2)
c for time-like four-momenta we can always think of it as the p_vec^2          
         if ( abs(pp).eq.rZero ) then
c particle at rest.            
            sqm(0) = sqrt(fmass) ! possibility of negative fermion masses
            sqm(1) = sqm(0) ! possibility of negative fermion masses
            ip = (1+nh)/2
            im = (1-nh)/2
            
            fi(1) = ip     * sqm(ip)
            fi(2) = im*nsf * sqm(ip)
            fi(3) = ip*nsf * sqm(im)
            fi(4) = im     * sqm(im)

         else
c standard spinor

            pp=sqrt(p(1)**2+p(2)**2+p(3)**2)
            write(*,*) 'ppre=',pp            
c            if( (dble(p(0)) .lt. 0 .and. dble(pp) .gt. 0) .or.
c     &          (dble(p(0)) .lt. 0 .and. dble(pp) .gt. 0) ) then
c            pp=-pp
c            endif 
            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
c fermion spin using HELAS conventions.            
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0)+pp)
c the omega of the definition.
c            omega(2) = fmass/omega(1)
            omega(2) = sqrt(p(0)-pp)
c the prefactor
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
c            pp3 = max(pp+p(3),rZero)
            pp3=pp+p(3)
            chi(1) = sqrt(pp3*rHalf/pp)
            if ( abs(pp3).eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = ( (nh*p(1)) + ((0d0,1d0)*p(2)) )/
     .sqrt(rTwo*pp*pp3)
            endif
            

c         Write(*,*) 'Chi=',Chi(1),' and ',Chi(2)

            fi(1) = sfomeg(1)*chi(im)
            fi(2) = sfomeg(1)*chi(ip)
c         Write(*,*) 'fi(2)=',fi(2)     
            fi(3) = sfomeg(2)*chi(im)
c         Write(*,*) 'fi(3)=',fi(3)
            fi(4) = sfomeg(2)*chi(ip)
            
         endif
         
      else
         
c         if(zabs(p(1)).eq.0d0.and.zabs(p(2)).eq.0d0.and.
c     .zabs(p(3)).lt.0d0) then
c            sqp0p3 = 0d0
c         else
            sqp0p3 = sqrt(p(0)+p(3))*nsf
c        end if
         chi(1) = sqp0p3
         if ( abs(sqp0p3).eq.rZero ) then
            chi(2) = dcmplx(-nhel )*sqrt(rTwo*p(0))
         else
            chi(2) = ( nh*p(1) + ((0d0,1d0)*p(2) ) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fi(1) = dcmplx( rZero )
            fi(2) = dcmplx( rZero )
            fi(3) = chi(1)
            fi(4) = chi(2)
         else
            fi(1) = chi(2)
            fi(2) = chi(1)
            fi(3) = dcmplx( rZero )
            fi(4) = dcmplx( rZero )
         endif
      endif     

      return
      end

      subroutine olxxxx(p,ffmass,nhel,nsf,fo)
c
c This subroutine computes a fermion wavefunction with the flowing-OUT
c fermion number and defined with complex ONSHELL  momentum.
c
c input:
c       complex    p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(8)          : fermion wavefunction               <fo|
c     
      implicit none
      double complex fo(8),chi(2), fmass

      double complex sqm(0:1)
      double precision sf(2), ffmass

      double complex p(0:3),sfomeg(2),omega(2),
     &     pp,pp3,sqp0p3
      integer nhel,nsf,nh,ip,im

      double precision rZero, rHalf, rTwo
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )


c      fo(5) = dcmplx(p(0),p(3))*nsf
c      fo(6) = dcmplx(p(1),p(2))*nsf
      fo(5) = p(0)*(nsf)
      fo(6) = p(1)*(nsf)
      fo(7) = p(2)*(nsf)
      fo(8) = p(3)*(nsf)      

      nh = nhel*nsf

      fmass=sqrt(p(0)**2-p(1)**2-p(2)**2-p(3)**2)

      if ( ffmass.ne.rZero ) then

c         pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
         pp=sqrt(p(1)**2+p(2)**2+p(3)**2)

         if ( abs(pp).eq.rZero ) then
            
            sqm(0) = sqrt(fmass) ! possibility of negative fermion masses
            sqm(1) = sqm(0) ! possibility of negative fermion masses
            ip = -((1+nh)/2)
            im =  (1-nh)/2
            
            fo(1) = im     * sqm(im)
            fo(2) = ip*nsf * sqm(im)
            fo(3) = im*nsf * sqm(-ip)
            fo(4) = ip     * sqm(-ip)
            
         else
            
c            pp = min(p(0),dsqrt(p(1)**2+p(2)**2+p(3)**2))
            pp=sqrt(p(1)**2+p(2)**2+p(3)**2) !repetition
c            if( (dble(p(0)) .lt. 0 .and. dble(pp) .gt. 0) .or.
c     &          (dble(p(0)) .lt. 0 .and. dble(pp) .gt. 0) ) then
c            pp=-pp
c            endif
            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0)+pp)
c            omega(2) = fmass/omega(1)
            omega(2) = sqrt(p(0)-pp)
            ip = (3+nh)/2
            im = (3-nh)/2
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)
            pp3 = pp+p(3)
            chi(1) = sqrt(pp3*rHalf/pp)
            if ( abs(pp3).eq.rZero ) then
               chi(2) = dcmplx(-nh )
            else
               chi(2) = ( nh*p(1) - ((0d0,1d0)*p(2)) )/
     .sqrt(rTwo*pp*pp3)
            endif
            
            fo(1) = sfomeg(2)*chi(im)
            fo(2) = sfomeg(2)*chi(ip)
            fo(3) = sfomeg(1)*chi(im)
            fo(4) = sfomeg(1)*chi(ip)

         endif
         
      else
         
c         if(zabs(p(1)).eq.0d0.and.zabs(p(2)).eq.0d0.and.zabs(p(3)).lt.0d0) then
c            sqp0p3 = 0d0
c         else
            sqp0p3 = sqrt(p(0)+p(3))*nsf
c         end if
         chi(1) = sqp0p3
         if ( abs(sqp0p3).eq.rZero ) then
            chi(2) = dcmplx(-nhel )*sqrt(rTwo*p(0))
         else
            chi(2) = ( nh*p(1)-((0d0,1d0)*p(2)) )/sqp0p3
         endif
         if ( nh.eq.1 ) then
            fo(1) = chi(1)
            fo(2) = chi(2)
            fo(3) = dcmplx( rZero )
            fo(4) = dcmplx( rZero )
         else
            fo(1) = dcmplx( rZero )
            fo(2) = dcmplx( rZero )
            fo(3) = chi(2)
            fo(4) = chi(1)
         endif
         
      endif
c
      return
      end
