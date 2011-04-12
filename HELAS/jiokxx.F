      subroutine jiokxx(fi,fo,g,tmass,twidth , uio)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c coupling.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex g(1)           : coupling constant                 -kappa/8
c       real    g(2)           : fermion mass                        m_f
c
c output:
c       complex uio(18)        : outgoing tensor    T_{mu,vu} = T(mu+4*(vu-1))
c     
      implicit none
      double complex fi(18), fo(18), uio(18),g(2)
      double precision fmass,tmass,twidth

      double complex tc(4,4),tc1(4,4),fgamf(4),ffm
      double complex fkslaf,fgamfk(4,4),prop,denom
      double precision k(4),pp,m2,p(4),eta(4,4)
      integer i,j,ii,jj

      double complex ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )

      fmass = dreal(g(2))
      do i=1,16
         uio(i)=(0d0,0d0)
      enddo

      m2 = tmass**2
      uio(17) = fi(5)-fo(5)
      uio(18) = fi(6)-fo(6)

      p(1) = dble( uio(17))/tmass
      p(2) = dble( uio(18))/tmass
      p(3) = dimag(uio(18))/tmass
      p(4) = dimag(uio(17))/tmass

      k(1) = dble( fi(5)+fo(5))
      k(2) = dble( fi(6)+fo(6))
      k(3) = dimag(fi(6)+fo(6))
      k(4) = dimag(fi(5)+fo(5))

      pp = p(1)**2 - p(2)**2 - p(3)**2 - p(4)**2

      denom = dcmplx(pp*m2 - m2,tmass*twidth)

      fgamf(1) =     fi(3)*fo(1)+fi(4)*fo(2)+fi(1)*fo(3)+fi(2)*fo(4)
      fgamf(2) =     fi(4)*fo(1)+fi(3)*fo(2)-fi(2)*fo(3)-fi(1)*fo(4)
      fgamf(3) =ci*(-fi(4)*fo(1)+fi(3)*fo(2)+fi(2)*fo(3)-fi(1)*fo(4))
      fgamf(4) =     fi(3)*fo(1)-fi(4)*fo(2)-fi(1)*fo(3)+fi(2)*fo(4)
      ffm = 2d0*fmass*(fo(1)*fi(1)+fo(2)*fi(2)+fo(3)*fi(3)+fo(4)*fi(4))
      fkslaf  = k(1)* fgamf(1)-k(2)* fgamf(2)-
     &          k(3)* fgamf(3)-k(4)* fgamf(4)
      do i=1,3
         do j=1+1,4
            eta(i,j)=0d0
            eta(j,i)=eta(i,j)
         enddo
      enddo
      eta(1,1)= 1d0
      eta(2,2)=-1d0
      eta(3,3)=-1d0
      eta(4,4)=-1d0

c vertex
c upper triangle:
      do i=2,4
      do j=1,i-1
      tc(i,j)=fgamf(i)*k(j)+fgamf(j)*k(i)
c lower triangle:
      tc(j,i)=tc(i,j)
      enddo
c diagonal terms:
      tc(i,i)=2d0*(fgamf(i)*k(i)+(fkslaf-ffm))
      enddo
      tc(1,1)=2d0*(fgamf(1)*k(1)-(fkslaf-ffm))
      
c make indices upper indices before contracting with propagator
      do i=2,4
         tc(1,i)=-tc(1,i)
         tc(i,1)=-tc(i,1)
      enddo



c multiply by propagator
      do i=1,4
         do j=1,4
            do ii=1,4
               do jj=1,4
                  uio(i+4*(j-1))=uio(i+4*(j-1))+
     &            (
     &                    (eta(i,ii)-p(i)*p(ii))*(eta(j,jj)-p(j)*p(jj))+
     &                    (eta(i,jj)-p(i)*p(jj))*(eta(j,ii)-p(j)*p(ii))-
     &            2d0/3d0*(eta(i,j)-p(i)*p(j))*(eta(ii,jj)-p(ii)*p(jj))
     &            )
     &            *tc(ii,jj)*g(1)/denom
               enddo
            enddo
         enddo
      enddo

      return
      end
