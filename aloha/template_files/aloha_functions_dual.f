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

      subroutine ixxxxx(p,fmass,nhel,nsf,fi)
      use dual_variables
c
c This subroutine computes a fermion wavefunction derivative with the flowing-IN
c fermion number.
c
c input:
c       complex p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fi(8)         : fermion wavefunction derivative      eps_L^alpha . d|fi>/dq^alpha
c
      implicit none
      type(Dual)::fi(8),chi(2)
      type(Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      double precision sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      double precision rZero, rHalf, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

c     Convention for dual computations (same as loop)
      fi(1) = p(0)*(-nsf)
      fi(2) = p(1)*(-nsf)
      fi(3) = p(2)*(-nsf)
      fi(4) = p(3)*(-nsf)
      do i = 5,8
         CALL fi(i)%initZERO()
      enddo

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            ip = (1+nh)/2
            im = (1-nh)/2

            sqm(0) = dsqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fi(5)%comp(0) = ip     * sqm(ip)
            fi(6)%comp(0) = im*nsf * sqm(ip)
            fi(7)%comp(0) = ip*nsf * sqm(im)
            fi(8)%comp(0) = im     * sqm(im)    
            do i = 1, size(p(0))
               fi(5)%comp(i) = oHsqm(ip)*
     &            (+ip*    (p(0)%comp(i) -    p(3)%comp(i))
     &             -im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi(6)%comp(i) = oHsqm(ip)*
     &            (-ip*    (p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*nsf*(p(0)%comp(i) +    p(3)%comp(i)))

               fi(7)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(0)%comp(i) +    p(3)%comp(i))
     &             +im*    (p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi(8)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*    (p(0)%comp(i) -    p(3)%comp(i)))
            enddo

         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0) + pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)
      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = dcmplx(-nh)
            else
               oHsqm(0:1) = (/0d0,0d0/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (dble(nh)*p(1) + ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fi(5) = sfomeg(1)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(0)*chi(ip)
            fi(6) = sfomeg(1)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(0)*chi(im)
            fi(7) = sfomeg(2)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(1)*chi(ip)
            fi(8) = sfomeg(2)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(1)*chi(im)
         endif

      else
         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            sqp0p3 = sqrt(p(0)+p(3))*nsf
            sqp0p3%comp(0) = sqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0).eq.rZero) then
            chi(2) = dcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) + ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fi(7) = chi(1)
            fi(8) = chi(2)
         else
            fi(5) = chi(2)
            fi(6) = chi(1)
         endif
      endif
c
      return
      end subroutine ixxxxx




      subroutine ixxxso(p,fmass,nhel,nsf,fi)
      use dual_variables
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
c       complex fi(4)          : fermion wavefunction               |fi>
c
c
      implicit none
      type(Dual)::fi(4),chi(2)
      type(Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      double precision sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      double precision rZero, rHalf, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

      do i = 1,4
         CALL fi(i)%initZERO()
      enddo

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            ip = (1+nh)/2
            im = (1-nh)/2

            sqm(0) = dsqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fi(1)%comp(0) = ip     * sqm(ip)
            fi(2)%comp(0) = im*nsf * sqm(ip)
            fi(3)%comp(0) = ip*nsf * sqm(im)
            fi(4)%comp(0) = im     * sqm(im)
            do i = 1, size(p(0))
               fi(1)%comp(i) = oHsqm(ip)*
     &            (+ip*    (p(0)%comp(i) -    p(3)%comp(i))
     &             -im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi(2)%comp(i) = oHsqm(ip)*
     &            (-ip*    (p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*nsf*(p(0)%comp(i) +    p(3)%comp(i)))

               fi(3)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(0)%comp(i) +    p(3)%comp(i))
     &             +im*    (p(1)%comp(i) - ci*p(2)%comp(i))) 

               fi(4)%comp(i) = oHsqm(im)*
     &            (+ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))
     &             +im*    (p(0)%comp(i) -    p(3)%comp(i)))
            enddo

         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0) + pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)

            chi(1) = sqrt(rHalf*pp3/pp)
      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = dcmplx(-nh)
            else
               oHsqm(0:1) = (/0d0,0d0/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (dble(nh)*p(1) + ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fi(1) = sfomeg(1)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(0)*chi(ip)
            fi(2) = sfomeg(1)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(0)*chi(im)
            fi(3) = sfomeg(2)*chi(im) 
     &            + (p(1) - nh*ci*p(2))*oHsqm(1)*chi(ip)
            fi(4) = sfomeg(2)*chi(ip) 
     &            - (p(1) - nh*ci*p(2))*oHsqm(1)*chi(im)
         endif
         
      else
         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            sqp0p3%comp(0) = sqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0)%re.eq.rZero) then
            chi(2) = dcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) + ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fi(3) = chi(1)
            fi(4) = chi(2)
         else
            fi(1) = chi(2)
            fi(2) = chi(1)
         endif
      endif
c
      return
      end subroutine ixxxso



      subroutine oxxxxx(p,fmass,nhel,nsf,fo)
      use dual_variables
c
c This subroutine computes a fermion wavefunction derivative with the flowing-OUT
c fermion number.
c
c input:
c       complex p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(8)         : fermion wavefunction derivative       eps_L^alpha . d<fo|/dq^alpha
c
      implicit none
      type(Dual)::fo(8),chi(2)
      type(Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      type(Dual)::p3_tmp
      double precision sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      double precision rZero, rHalf, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

c     Convention for dual computations (same as loop)
      fo(1) = p(0)*(nsf)
      fo(2) = p(1)*(nsf)
      fo(3) = p(2)*(nsf)
      fo(4) = p(3)*(nsf)
      do i = 5,8
         CALL fo(i)%initZERO()
      enddo

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            im = +nhel * (1+nh)/2
            ip = -nhel * (1-nh)/2

            sqm(0) = dsqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fo(5)%comp(0) = im     * sqm(abs(ip))
            fo(6)%comp(0) = ip*nsf * sqm(abs(ip))
            fo(7)%comp(0) = im*nsf * sqm(abs(im))
            fo(8)%comp(0) = ip     * sqm(abs(im))    
            do i = 1, size(p(0))
               fo(5)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(0)%comp(i) +    p(3)%comp(i))
     &             +ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo(6)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*nsf*(p(0)%comp(i) -    p(3)%comp(i)))

               fo(7)%comp(i) = oHsqm(abs(im))*
     &            (+im*nsf*(p(0)%comp(i) -    p(3)%comp(i))
     &             -ip*    (p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo(8)%comp(i) = oHsqm(abs(im))*
     &            (-im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*    (p(0)%comp(i) +    p(3)%comp(i)))
            enddo     
         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)
                      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = dcmplx(-nh)
            else
               oHsqm(0:1) = (/0d0,0d0/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (dble(nh)*p(1) - ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fo(5) = sfomeg(2)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(1)*chi(ip)
            fo(6) = sfomeg(2)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(1)*chi(im)
            fo(7) = sfomeg(1)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(0)*chi(ip)
            fo(8) = sfomeg(1)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(0)*chi(im)
         endif

      else

         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            sqp0p3 = sqrt(p(0)+p(3))*nsf
            sqp0p3%comp(0) = sqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0)%re.eq.rZero) then
            chi(2) = dcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) - ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fo(5) = chi(1)
            fo(6) = chi(2)
         else
            fo(7) = chi(2)
            fo(8) = chi(1)
         endif

      endif
c
      return
      end subroutine oxxxxx


      subroutine oxxxso(p,fmass,nhel,nsf,fo)
      use dual_variables
c
c This subroutine computes a fermion wavefunction derivative with the flowing-OUT
c fermion number.
c
c input:
c       complex p(0:3)         : four-momentum of fermion
c       real    fmass          : mass          of fermion
c       integer nhel = -1 or 1 : helicity      of fermion
c       integer nsf  = -1 or 1 : +1 for particle, -1 for anti-particle
c
c output:
c       complex fo(8)         : fermion wavefunction derivative       eps_L^alpha . d<fo|/dq^alpha
c
      implicit none
      type(Dual)::fo(4),chi(2)
      type(Dual)::p(0:3),omega(2),sfomeg(2),
     &            pp,pp3,sqp0p3
      double precision sf(2),fmass,sqm(0:1),oHsqm(0:1)
      integer nhel,nsf,ip,im,nh
      integer i

      double precision rZero, rHalf, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

      do i = 1,4
         CALL fo(i)%initZERO()
      enddo

      nh = nhel*nsf

      if (fmass.ne.rZero) then
         pp = sqrt(p(1)**2+p(2)**2+p(3)**2)
         pp%comp(0) = min(p(0)%comp(0)%re, pp%comp(0)%re)

         if (pp%comp(0)%re.eq.rZero) then
            im = +nhel * (1+nh)/2
            ip = -nhel * (1-nh)/2

            sqm(0) = dsqrt(abs(fmass))  ! possibility of negative fermion masses
            sqm(1) = sign(sqm(0),fmass) ! possibility of negative fermion masses
            oHsqm(0) = rHalf/sqm(0)     ! normalization for the derivative part
            oHsqm(1) = rHalf/sqm(1)     ! normalization for the derivative part

            fo(1)%comp(0) = im     * sqm(abs(ip))
            fo(2)%comp(0) = ip*nsf * sqm(abs(ip))
            fo(3)%comp(0) = im*nsf * sqm(abs(im))
            fo(4)%comp(0) = ip     * sqm(abs(im))
            do i = 1, size(p(0))
               fo(1)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(0)%comp(i) +    p(3)%comp(i))
     &             +ip*nsf*(p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo(2)%comp(i) = oHsqm(abs(ip))*
     &            (+im*    (p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*nsf*(p(0)%comp(i) -    p(3)%comp(i)))

               fo(3)%comp(i) = oHsqm(abs(im))*
     &            (+im*nsf*(p(0)%comp(i) -    p(3)%comp(i))
     &             -ip*    (p(1)%comp(i) + ci*p(2)%comp(i))) 

               fo(4)%comp(i) = oHsqm(abs(im))*
     &            (-im*nsf*(p(1)%comp(i) - ci*p(2)%comp(i))
     &             +ip*    (p(0)%comp(i) +    p(3)%comp(i)))
            enddo  
         else
            ip = (3+nh)/2
            im = (3-nh)/2

            sf(1) = dble(1+nsf+(1-nsf)*nh)*rHalf
            sf(2) = dble(1+nsf-(1-nsf)*nh)*rHalf
            omega(1) = sqrt(p(0)+pp)
            omega(2) = fmass/omega(1)
            sfomeg(1) = sf(1)*omega(ip)
            sfomeg(2) = sf(2)*omega(im)

            pp3 = pp + p(3)
            pp3%comp(0) = max(pp3%comp(0)%re, rZero)
            chi(1) = sqrt(rHalf*pp3/pp)
                      
            if ((p(1)%comp(0)%re.eq.0d0).and.(p(2)%comp(0)%re.eq.0d0)
     &          .and.(p(3)%comp(0)%re.lt.0d0)) then
               oHsqm(0) = rHalf*sfomeg(1)%comp(0)/pp%comp(0)
               oHsqm(1) = rHalf*sfomeg(2)%comp(0)/pp%comp(0)
               chi(2)%comp(0) = dcmplx(-nh)
            else
               oHsqm(0:1) = (/0d0,0d0/)
               chi(1) = sqrt(rHalf*pp3/pp)
               chi(2) = (dble(nh)*p(1) - ci*p(2))/(sqrt(rTwo*pp*pp3))
            endif

            fo(1) = sfomeg(2)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(1)*chi(ip)
            fo(2) = sfomeg(2)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(1)*chi(im)
            fo(3) = sfomeg(1)*chi(im)
     &            + (p(1) + nh*ci*p(2))*oHsqm(0)*chi(ip)
            fo(4) = sfomeg(1)*chi(ip)
     &            - (p(1) + nh*ci*p(2))*oHsqm(0)*chi(im)
         endif

      else
         if((p(1)%comp(0)%re.ne.0d0).or.(p(2)%comp(0)%re.ne.0d0).or.
     &      (p(3)%comp(0)%re.gt.0d0)) then
            sqp0p3%comp(0) = dsqrt(max(p(0)%comp(0)%re+
     &                        p(3)%comp(0)%re,rZero))*nsf
         end if
         chi(1) = sqp0p3
         if (sqp0p3%comp(0)%re.eq.rZero) then
            chi(2) = dcmplx(-nhel)*sqrt(rTwo*p(0))
         else
            chi(2) = (nh*p(1) - ci*p(2))/sqp0p3
         endif
         if (nh.eq.1) then
            fo(1) = chi(1)
            fo(2) = chi(2)
         else
            fo(3) = chi(2)
            fo(4) = chi(1)
         endif
      endif
c
      return
      end subroutine oxxxso



      subroutine vxxxxx(p,vmass,nhel,nsv,vc)
      use dual_variables
c
c This subroutine computes a VECTOR wavefunction.
c
c input:
c       complex p(0:3)         : four-momentum of vector boson
c       real    fmass          : mass          of vector boson
c       integer nhel = -1, 0, 1: helicity      of bector boson
c       integer nsv  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex vc(8)          : vector wavefunction       epsilon^mu(p)
c
      implicit none
      type(Dual)::vc(8)
      type(Dual)::p(0:3),pp,pt,pt2,pzpt,emp
      double precision vmass,hel,hel0,sqh
      integer nhel,nsv,nsvahl
      integer i

      double precision rZero, rHalf, rOne, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

      sqh = dsqrt(rHalf)
      hel = dble(nhel)
      nsvahl = nsv*dabs(hel)
      pt2 = p(1)**2+p(2)**2
      pp = sqrt(pt2+p(3)**2)
      if ((p(1)%comp(0).ne.(0d0,0d0))
     $     .or.(p(2)%comp(0).ne.(0d0,0d0))) then
         pt = sqrt(pt2)
      else
         CALL pt%initZERO()
      endif

      pp%comp(0) = min(p(0)%comp(0)%re,pp%comp(0)%re)
      pt%comp(0) = min(pp%comp(0)%re,pt%comp(0)%re)

c     Convention for dual computations (same as loop)
      vc(1) = p(0)*nsv
      vc(2) = p(1)*nsv
      vc(3) = p(2)*nsv
      vc(4) = p(3)*nsv
      do i = 5,8
         CALL vc(i)%initZERO()
      enddo

      if (vmass.ne.rZero) then

         hel0 = rOne-dabs(hel)

         if ( pp%comp(0)%re.eq.rZero ) then
            vc(5)%comp(0) = dcmplx( rZero )
            vc(6)%comp(0) = dcmplx(-hel*sqh )
            vc(7)%comp(0) = dcmplx( rZero , nsvahl*sqh )
            vc(8)%comp(0) = dcmplx( hel0 )

         else
            emp = p(0)/(vmass*pp)
            vc(5) = hel0*pp/vmass
            vc(8) = hel0*p(3)*emp+hel*pt/pp*sqh
            if ( pt%comp(0)%re.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh*hel
               vc(6) = hel0*p(1)*emp-p(1)*pzpt
     &                   -ci*nsvahl*p(2)/pt*sqh
               vc(7) = hel0*p(2)*emp-p(2)*pzpt
     &                   +ci*nsvahl*p(1)/pt*sqh
            else
               vc(6)%comp(0) = dcmplx( -hel*sqh )
               vc(7)%comp(0) = dcmplx( rZero , 
     &                   nsvahl*sign(sqh,p(3)%comp(0)%re) )
            endif

         endif

      else
         pp = p(0)
         CALL vc(5)%initZERO()
         vc(8) = hel*pt/pp*sqh
         if ( pt%comp(0)%re.ne.rZero ) then
            pzpt = p(3)/(pp*pt)*sqh*hel
            vc(6) = -p(1)*pzpt-ci*nsv*p(2)/pt*sqh
            vc(7) = -p(2)*pzpt+ci*nsv*p(1)/pt*sqh
        else
            vc(6)%comp(0) = dcmplx( -hel*sqh )
            vc(7)%comp(0) = dcmplx( rZero , 
     &                   nsv*sign(sqh,p(3)%comp(0)%re) )
         endif

      endif
c
      return
      end subroutine vxxxxx



      subroutine sxxxxx(p,nss,sc)
      use dual_variables
c
c This subroutine computes a complex SCALAR wavefunction.
c
c input:
c       complex    p(0:3)      : four-momentum of scalar boson
c       integer nss  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex sc(5)          : scalar wavefunction                   s
c
      implicit none
      type(Dual)::sc(5),p(0:3)
      integer nss
      integer i

      double precision rOne
      parameter( rOne = 1.0d0 )

c     Convention for dual computations (same as loop)
      sc(1) = p(0)*nss
      sc(2) = p(1)*nss
      sc(3) = p(2)*nss
      sc(4) = p(3)*nss
      CALL sc(5)%initZERO()

      sc(5)%comp(0) = dcmplx( rOne )

c
      return
      end subroutine sxxxxx



      subroutine pxxxxx(p,tmass,nhel,nst,tc)
         print*, "The pxxxxx subroutine for PSEUDOR is not (yet) "//
     1           "implementated in dual computation"
         stop
      end



      subroutine txxxxx(p,tmass,nhel,nst,tc)
      use dual_variables
c
c This subroutine computes a TENSOR wavefunction.
c
c input:
c       complex p(0:3)         : four-momentum of tensor boson
c       real    tmass          : mass          of tensor boson
c       integer nhel           : helicity      of tensor boson
c                = -2,-1,0,1,2 : (0 is forbidden if tmass=0.0)
c       integer nst  = -1 or 1 : +1 for final, -1 for initial
c
c output:
c       complex tc(20)         : tensor wavefunction    epsilon^mu^nu(t)
c
      implicit none
      type(Dual)::tc(20),p(0:3)
      double precision  tmass
      integer nhel, nst

      ! double complex ft(8,4), ep(4), em(4), e0(4)
      ! double precision pt, pt2, pp, pzpt, emp, sqh, sqs
      type(Dual)::ft(8,4), ep(4), em(4), e0(4)
      type(Dual)::pt, pt2, pp, pzpt, emp
      double precision sqh, sqs
      integer i, j

      double precision rZero, rHalf, rOne, rTwo
      double complex ci
      parameter( rZero = 0.0d0, rHalf = 0.5d0 )
      parameter( rOne = 1.0d0, rTwo = 2.0d0 )
      parameter( ci = dcmplx(0.0d0,1.0d0) )

      integer stdo
      parameter( stdo = 6 )

      do j = 1, 20
         CALL tc(j)%initZERO()
      enddo
      
      sqh = sqrt(rHalf)
      sqs = sqrt(rHalf/3.d0)

      pt2 = p(1)**2+p(2)**2
      pp = sqrt(pt2+p(3)**2)
      pt = sqrt(pt2)

      pp%comp(0) = min(p(0)%comp(0)%re,pp%comp(0)%re)
      pt%comp(0) = min(pp%comp(0)%re,pt%comp(0)%re)

c     Convention for dual computations (same as loop)
      ft(5,1) = p(0)*nst
      ft(6,1) = p(1)*nst
      ft(7,1) = p(2)*nst
      ft(8,1) = p(3)*nst

      if ( nhel.ge.0 ) then
c construct eps+
         if ( pp%comp(0)%re.eq.rZero ) then
            ep(1)%comp(0) = dcmplx( rZero )
            ep(2)%comp(0) = dcmplx( -sqh )
            ep(3)%comp(0) = dcmplx( rZero , nst*sqh )
            ep(4)%comp(0) = dcmplx( rZero )
         else
            ep(1)%comp(0) = dcmplx( rZero )
            ep(4) = pt/pp*sqh
            if ( pt%comp(0)%re.ne.rZero ) then
               pzpt = p(3)/(pp*pt)*sqh
               ep(2) = -p(1)*pzpt-ci*nst*p(2)/pt*sqh
               ep(3) = -p(2)*pzpt+ci*nst*p(1)/pt*sqh
            else
               ep(2)%comp(0) = dcmplx( -sqh )
               ep(3)%comp(0) = dcmplx( rZero , 
     &                     nst*sign(sqh,p(3)%comp(0)%re) )
            endif
         endif
      end if

      if ( nhel.le.0 ) then
c construct eps-
         if ( pp%comp(0)%re.eq.rZero ) then
            em(1)%comp(0) = dcmplx( rZero )
            em(2)%comp(0) = dcmplx( sqh )
            em(3)%comp(0) = dcmplx( rZero , nst*sqh )
            em(4)%comp(0) = dcmplx( rZero )
         else
            em(1)%comp(0) = dcmplx( rZero )
            em(4) = -pt/pp*sqh
            if ( pt%comp(0)%re.ne.rZero ) then
               pzpt = -p(3)/(pp*pt)*sqh
               em(2) = -p(1)*pzpt-ci*nst*p(2)/pt*sqh
               em(3) = -p(2)*pzpt+ci*nst*p(1)/pt*sqh
            else
               em(2)%comp(0) = dcmplx( sqh )
               em(3)%comp(0) = dcmplx( rZero , 
     &                     nst*sign(sqh,p(3)%comp(0)%re) )
            endif
         endif
      end if

      if ( abs(nhel).le.1 ) then
c construct eps0
         if ( pp%comp(0)%re.eq.rZero ) then
            e0(1)%comp(0) = dcmplx( rZero )
            e0(2)%comp(0) = dcmplx( rZero )
            e0(3)%comp(0) = dcmplx( rZero )
            e0(4)%comp(0) = dcmplx( rOne )
         else
            emp = p(0)/(tmass*pp)
            e0(1) = pp/tmass
            e0(4) = p(3)*emp
            if ( pt%comp(0)%re.ne.rZero ) then
               e0(2) = p(1)*emp
               e0(3) = p(2)*emp
            else
               e0(2)%comp(0) = dcmplx( rZero )
               e0(3)%comp(0) = dcmplx( rZero )
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
               ft(i,j)%comp(0) = 0
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
      end subroutine txxxxx




      subroutine clebsch_gordan(L, Lz, S, Sz, J, Jz, cg)
c
c This subroutine computes the Clebsch-Gordan coefficient <j1,m1,j2,m2|j,m>
c For species with spin<5/2, values are given in a table.
c For higher spins, it computes the values from general formula on the fly.
c
c input:
c       integer L, Lz       : angular momentum and its z-component
c       integer S, Sz       : spin and its z-component
c       integer J, Jz       : total angular momentum and its z-component
c
c output:
c       double precision cg   : value of the Clebsch-Gordan coefficient
c
      implicit none
      integer L, Lz, S, Sz, J, Jz
      double precision cg

      if (J.gt.(L + S)) then
         print*, "ERROR: angular momentum is not conserved!"
         print*, "(L, S, J): (", L, ",", S, ",", J, ")"
         stop
      endif

      if ((abs(Jz).gt.J).or.(abs(Lz).gt.L).or.(abs(Sz).gt.S)) then
         print*, "ERROR: magnetic number overshoots angular number!"
         print*, "(L, Lz): (", L, ",", Lz, ")"
         print*, "(S, Sz): (", S, ",", Sz, ")"
         print*, "(J, Jz): (", J, ",", Jz, ")"
         stop
      endif


      if (Jz.ne.(Lz + Sz)) then
         cg = 0.0d0
      
      elseif ((L.eq.0).or.(S.eq.0)) then
         cg = 1.0d0

      elseif ((L.eq.1).or.(S.eq.1)) then
         if (J.eq.2) then
            if ((Lz.eq.0).and.(Sz.eq.0)) then
               cg = dsqrt(2/3.0d0)
            elseif ((Lz.eq.0).or.(Sz.eq.0)) then
               cg = 1/dsqrt(2.0d0)
            elseif (Lz.eq.Sz) then
               cg = 1
            else 
               cg = 1/dsqrt(6.0d0)
            end if
         elseif (J.eq.1) then
            if ((Lz.eq.1).or.(Sz.eq.-1)) then
               cg = 1/dsqrt(2.0d0)
            elseif ((Lz.eq.-1).or.(Sz.eq.+1)) then
               cg = -1/dsqrt(2.0d0)
            else
               cg = 0.0d0
            end if
         elseif (J.eq.0) then
            if ((Lz.eq.0).and.(Sz.eq.0)) then
               cg = -1/dsqrt(3.0d0)
            else
               cg = 1/dsqrt(3.0d0)
            end if
         else
            cg = 0.0d0
         end if

      
      else
         print*, "Clebsch-Gordan coefficient not implemented for"//
     1            " L=", L, " S=", S
         cg = 0.0d0
      end if


      return
      end subroutine clebsch_gordan

      subroutine onia_proj_dual(fq, fqb, masses, p, m, sz, spin
     &, proj)
      use dual_variables
c
c This subroutine computes the spin projector for an onium states.
c
c input:
c       dual    fq(1:8)        : spinor of first consituent (particle)
c       dual    fqb(1:8)       : spinor of second consituent (anti-particle)
c       real    p(0:3)         : four-momentum of bound state
c       real    m(0:3)         : mass          of bound state
c       integer sz             : z component of the spin
c       integer spin           : spin          of the bound state
c
c output:
c       dual    proj           : value of the projector
c
      implicit none
      type(Dual)::fq(1:8),fqb(1:8),vc(1:8),p(0:3),tmp
      double precision masses(1:2),m
      integer sz,spin
      type(Dual)::proj
      
      double complex ci
      parameter( ci = dcmplx(0.0d0,1.0d0) )
      
      if (spin.eq.0) then
c     spin singlet    

         tmp = -fq(5)*fqb(5)-fq(6)*fqb(6)+fq(7)*fqb(7)+fq(8)*fqb(8)

      elseif (spin.eq.1) then
c     spin triplet
         call vxxxxx(p,m,sz,+1,vc)
         tmp = (fq(5)*fqb(8)-fq(7)*fqb(6))*(vc(6)+ci*vc(7))+
     &         (fq(6)*fqb(7)-fq(8)*fqb(5))*(vc(6)-ci*vc(7))+
     &         (fq(5)*fqb(7)+fq(8)*fqb(6))*(vc(5)+vc(8))+
     &         (fq(6)*fqb(8)+fq(7)*fqb(5))*(vc(5)-vc(8))
      else
         print *,"spin projector not yet implemented"
         stop
      endif

      proj = 0.5d0/SQRT(2d0*masses(1)*masses(2))*tmp

      return
      end





      subroutine boostx(p,q,pboost)
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
c
      qq = q(1)**2+q(2)**2+q(3)**2

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
c
      prot(0) = p(0)

      qt2 = q(1)**2 + q(2)**2

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