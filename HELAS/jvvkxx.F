      subroutine jvvkxx(wm,wp,g,tmass,twidth, tc)
c
c This subroutine computes an off-shell tensor boson from two vector
c gauge bosons.
c
c input:
c       complex wm(6)          : vector               flow-in  V
c       complex wp(6)          : vector               flow-out V~
c       complex g(1)           : coupling constant    -kappa/2
c       real    g(2)           : V boson mass          m_V
c       real    tmass          : t boson mass          m_T
c       real    twidth         : t boson width         W_T
c
c output:
c       complex tc(18)        : tensor               KK mode T
c     
      implicit none
      double complex wm(18), wp(18), tc(18),g(2)
      double precision  tmass,twidth,vmass

      double precision pwm(4),pwp(4),preC,m2,p(4),pp,eta(4,4)
      integer i,j,ii,jj
      double complex C(4,4),D(4,4),wpwm,wpeta(4),wmeta(4)
      double complex pwpwm,wppwm,uio(4,4),denom

      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )

      logical firsttime
      data firsttime/.true./

      vmass = dreal(g(2))
      if (firsttime)then
         write(*,*)'----------------------------------------------'
         write(*,*)'Using the jvvkxx HELAS routine. This routine  '
         write(*,*)'is only tested for gg>tt~ (sub)process.       '
         write(*,*)'----------------------------------------------'
         firsttime=.false.
      endif
c
      do i=1,16
         tc(i)=(0d0,0d0)
      enddo

      pwm(1) = dreal(wm(5))
      pwm(2) = dreal(wm(6))
      pwm(3) = dimag(wm(6))
      pwm(4) = dimag(wm(5))
      pwp(1) = dreal(wp(5))
      pwp(2) = dreal(wp(6))
      pwp(3) = dimag(wp(6))
      pwp(4) = dimag(wp(5))

      tc(17) = wm(5)+wp(5)
      tc(18) = wm(6)+wp(6)

      p(1) = -dble( tc(17))/tmass
      p(2) = -dble( tc(18))/tmass
      p(3) = -dimag(tc(18))/tmass
      p(4) = -dimag(tc(17))/tmass

      pp = p(1)**2 - p(2)**2 - p(3)**2 - p(4)**2

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

      m2 = tmass**2
      denom = dcmplx(pp*m2 - m2,tmass*twidth)

      preC = pwm(1)*pwp(1)-pwm(2)*pwp(2)-pwm(3)*pwp(3)-pwm(4)*pwp(4)

      if (vmass.ne.0d0)then
         preC=preC+vmass
      endif

      wpwm  =  wm(1)* wp(1)-  wm(2)* wp(2)-  wm(3)* wp(3)-  wm(4)* wp(4)
      pwpwm =  wm(1)*pwp(1)-  wm(2)*pwp(2)-  wm(3)*pwp(3)-  wm(4)*pwp(4)
      wppwm = pwm(1)* wp(1)- pwm(2)* wp(2)- pwm(3)* wp(3)- pwm(4)* wp(4)

      do i=1,4
         wpeta(i) = -wp(i)
         wmeta(i) = -wm(i)
      enddo
      wpeta(1) = -wpeta(1)
      wmeta(1) = -wmeta(1)

      do i=1,4
         do j=1,4
            C(i,j) = wpeta(i)*wmeta(j)+wpeta(j)*wmeta(i)-eta(i,j)*wpwm
            D(i,j) = eta(i,j)*pwpwm*wppwm
     & -wmeta(i)*pwm(j)*pwpwm-wpeta(i)*pwp(j)*wppwm+wpwm*pwm(i)*pwp(j)
     & -wmeta(j)*pwm(i)*pwpwm-wpeta(j)*pwp(i)*wppwm+wpwm*pwm(j)*pwp(i)
         enddo
      enddo

      do i=1,4
         do j=1,4
            uio(i,j)=preC*C(i,j)+D(i,j)
         enddo
      enddo
c make indices upper indices before contracting with propagator
      do i=2,4
         uio(1,i)=-uio(1,i)
         uio(i,1)=-uio(i,1)
      enddo

c multiply by propagator
      do i=1,4
         do j=1,4
            do ii=1,4
               do jj=1,4
                  tc(i+4*(j-1))=tc(i+4*(j-1))+
     &            (
     &                    (eta(i,ii)-p(i)*p(ii))*(eta(j,jj)-p(j)*p(jj))+
     &                    (eta(i,jj)-p(i)*p(jj))*(eta(j,ii)-p(j)*p(ii))-
     &            2d0/3d0*(eta(i,j)-p(i)*p(j))*(eta(ii,jj)-p(ii)*p(jj))
     &            )
     &            *uio(ii,jj)*g(1)/denom
               enddo
            enddo
         enddo
      enddo


      return
      end
