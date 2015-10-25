c************************************************************************
c  THIS FILE CONTAINS THE DEFINITIONS OF USEFUL FUNCTIONS OF MOMENTA:
c  
c  DOT(p1,p2)         : 4-Vector Dot product
c  R2(p1,p2)          : distance in eta,phi between two particles
c  SumDot(P1,P2,dsign): invariant mass of 2 particles
c  rap(p)             : rapidity of particle in the lab frame (p in CM frame)
C  RAP2(P)            : rapidity of particle in the lab frame (p in lab frame)
c  DELTA_PHI(P1, P2)  : separation in phi of two particles 
c  ET(p)              : transverse energy of particle
c  PT(p)              : transverse momentum of particle
c  DJ(p1,p2)          : y*S (Durham) value for two partons
c  DJB(p1,p2)         : mT^2=m^2+pT^2 for one particle
c  PYJB(p1,p2)        : The Pythia ISR pT^2=(1-x)*Q^2
c  DJ2(p1,p2)         : scalar product squared
c  threedot(p1,p2)    : 3-vector Dot product (accept 4 vector in entry)
c  rho                : |p| in lab frame
c  eta                : pseudo-rapidity
c  phi                : phi
c  four_momentum      : (theta,phi,rho,mass)-> 4 vector
c  four_momentum_set2 : (pt,eta,phi,mass--> 4 vector
c
c************************************************************************

      DOUBLE PRECISION FUNCTION R2(P1,P2)
c************************************************************************
c     Distance in eta,phi between two particles.
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:3),p2(0:3)
c
c     External
c
      double precision rap,DELTA_PHI
      external rap,delta_phi
c-----
c  Begin Code
c-----
      R2 = (DELTA_PHI(P1,P2))**2+(rap(p1)-rap(p2))**2
      RETURN
      END

      DOUBLE PRECISION FUNCTION SumDot(P1,P2,dsign)
c************************************************************************
c     Invarient mass of 2 particles
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:3),p2(0:3),dsign
c
c     Local
c      
      integer i
      double precision ptot(0:3)
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----

      do i=0,3
         ptot(i)=p1(i)+dsign*p2(i)
      enddo
      SumDot = dot(ptot,ptot)
      RETURN
      END

      DOUBLE PRECISION FUNCTION PtDot(P1,P2)
c************************************************************************
c     Pt of 2 particles
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:3),p2(0:3),dsign

c      write (*,*)'Px of particle 1: ',P1(1)
c      write (*,*)'Px of particle 2: ',P2(1)
c      write (*,*)'Py of particle 1: ',P1(2)
c      write (*,*)'Py of particle 2: ',P2(2)
c
      PtDot = (P1(1)+P2(1))**2+(P1(2)+P2(2))**2
      RETURN
      END

      DOUBLE PRECISION  FUNCTION rap(p)
c************************************************************************
c     Returns rapidity of particle with p in the CM frame
c     Note that it only applies to p-p collisions
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision  p(0:3)
c
c     Local
c
      double precision pm
c
c     Global
c
      include 'maxparticles.inc'
      include 'run.inc'

      double precision cm_rap
      logical set_cm_rap
      common/to_cm_rap/set_cm_rap,cm_rap
      data set_cm_rap/.false./

c-----
c  Begin Code
c-----
      if(.not.set_cm_rap) then
         print *,'Need to set cm_rap before calling rap'
         stop
      endif
c      pm=dsqrt(p(1)**2+p(2)**2+p(3)**2)
      
      pm = p(0)
      if (pm.gt.p(3)) then
        rap = .5d0*dlog((pm+p(3))/(pm-p(3)))+cm_rap
      else
        rap = -1d99
      endif
      end
      DOUBLE PRECISION  FUNCTION rap2(p)
c************************************************************************
c     Returns rapidity of particle in the lab frame
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision  p(0:3)
c
c     Local
c
      double precision pm
c
c     Global
c
      include 'maxparticles.inc'
      include 'run.inc'
c-----
c  Begin Code
c-----
c      pm=dsqrt(p(1)**2+p(2)**2+p(3)**2)
      pm = p(0)
      rap2 = .5d0*dlog((pm+p(3))/(pm-p(3)))
      end

      DOUBLE PRECISION FUNCTION DELTA_PHI(P1, P2)
c************************************************************************
c     Returns separation in phi of two particles p1,p2
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p1(0:3),p2(0:3)
c
c     Local
c
      REAL*8 DENOM, TEMP
c-----
c  Begin Code
c-----
      DENOM = SQRT(P1(1)**2 + P1(2)**2) * SQRT(P2(1)**2 + P2(2)**2)
      TEMP = MAX(-0.99999999D0, (P1(1)*P2(1) + P1(2)*P2(2)) / DENOM)
      TEMP = MIN( 0.99999999D0, TEMP)
      DELTA_PHI = ACOS(TEMP)
      END



      double precision function et(p)
c************************************************************************
c     Returns transverse energy of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:3)
c
c     Local
c
      double precision pt
c-----
c  Begin Code
c-----
      pt = dsqrt(p(1)**2+p(2)**2)
      if (pt .gt. 0d0) then
         et = p(0)*pt/dsqrt(pt**2+p(3)**2)
      else
         et = 0d0
      endif
      end

      double precision function pt(p)
c************************************************************************
c     Returns transverse momentum of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:3)
c-----
c  Begin Code
c-----

      pt = dsqrt(p(1)**2+p(2)**2)

      return
      end

      double precision function DJ(p1,p2)
c***************************************************************************
c     Uses Durham algorythm to calculate the y value for two partons
c     If collision type is hh, hadronic jet measure is used
c       y_{ij} = 2min[p_{i,\perp}^2,p_{j,\perp}^2]/S
c                  (cosh(\eta_i-\eta_j)-cos(\phi_1-\phi_2))
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:4),p2(0:4) ! 4 is mass**2
c
c     Global
c
      double precision D
      common/to_dj/D
c
c     Local
c

      include 'maxparticles.inc'
      include 'run.inc'
      include 'cuts.inc'

      double precision pt1,pt2,ptm1,ptm2,eta1,eta2,phi1,phi2,p1a,p2a,costh,sumdot
      integer j
c
c     Functions
c
      double precision DJB

c-----
c  Begin Code
c-----
      if ((lpp(1).eq.0).and.(lpp(2).eq.0)) then
         p1a = dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
         p2a = dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
         if (p1a*p2a .ne. 0d0) then
            costh = (p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3))/(p1a*p2a)
            dj = 2d0*min(p1(0)**2,p2(0)**2)*(1d0-costh) !Durham
c            dj = 2d0*p1(0)*p2(0)*(1d0-costh)    !JADE
         else
            print*,'Warning 0 momentum in Durham algorythm'
            write(*,'(4e15.5)') (p1(j),j=0,3)
            write(*,'(4e15.5)') (p2(j),j=0,3)
            dj = 0d0
         endif
      else
        pt1 = p1(1)**2+p1(2)**2
        pt2 = p2(1)**2+p2(2)**2
        p1a = dsqrt(pt1+p1(3)**2)
        p2a = dsqrt(pt2+p2(3)**2)
        eta1 = 0.5d0*log((p1a+p1(3))/(p1a-p1(3)))
        eta2 = 0.5d0*log((p2a+p2(3))/(p2a-p2(3)))
c     For massless-massive merging, use massless mT
c     to avoid depletion/enhancement of cone around massive particle
c     (only soft divergence)
        if(p1(4).lt.1d0.and.(p2(4).ge.3d0.and.maxjetflavor.gt.4.or.
     $       p2(4).ge.1d0.and.maxjetflavor.gt.3))then
           dj = DJB(p1)*(1d0+1d-6)
        elseif(p2(4).lt.1d0.and.(p1(4).ge.3d0.and.maxjetflavor.gt.4.or.
     $       p1(4).ge.1d0.and.maxjetflavor.gt.3))then
           dj = DJB(p2)*(1d0+1d-6)
        else
           dj = max(p1(4),p2(4))+min(pt1,pt2)*2d0*(cosh(eta1-eta2)-
     &          (p1(1)*p2(1)+p1(2)*p2(2))/dsqrt(pt1*pt2))/D**2
        endif
c        write(*,'(a,5e16.4)')'Mom(1): ',(p1(j),j=1,3),p1(0),p1(4)
c        write(*,'(a,5e16.4)')'Mom(2): ',(p2(j),j=1,3),p2(0),p2(4)
c       print *,'pT1: ',sqrt(pt1),' pT2: ',sqrt(pt2)
c       print *,'deltaR: ',sqrt(2d0*(cosh(eta1-eta2)-
c     &     (p1(1)*p2(1)+p1(2)*p2(2))/dsqrt(pt1*pt2))/D**2),
c     $      ' m: ',sqrt(SumDot(p1,p2,1d0))
c     write(*,*) 'p1  = ',p1(0),',',p1(1),',',p1(2),',',p1(3)
c     write(*,*) 'pm1 = ',pm1,', p1a = ',p1a,'eta1 = ',eta1
c     write(*,*) 'p2  = ',p2(0),',',p2(1),',',p2(2),',',p2(3)
c     write(*,*) 'pm2 = ',pm2,', p2a = ',p2a,'eta2 = ',eta2
c     write(*,*) 'dj = ',dj
      endif
      end
      
      double precision function PYDJ(p1,p2)
c***************************************************************************
c     Uses Durham algorythm to calculate the y value for two partons
c     If collision type is hh, hadronic jet measure is used
c       y_{ij} = 2min[p_{i,\perp}^2,p_{j,\perp}^2]/S
c                  (cosh(\eta_i-\eta_j)-cos(\phi_1-\phi_2))
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:4),p2(0:4) ! 4 is mass**2
c
c     Global
c
      double precision D
      common/to_dj/D
c
c     Local
c

      double precision SumDot
      external SumDot
c-----
c  Begin Code
c-----

      pydj = p1(0)*p2(0)/(p1(0)+p2(0))**2*SumDot(p1,p2,1d0)

      end
      
      double precision function DJ1(p1,p2)
c***************************************************************************
c     Uses single-sided Durham algorythm to calculate the y value for 
c     parton radiated off non-parton
c     If collision type is hh, hadronic jet measure is used
c       y_{ij} = 2min[p_{i,\perp}^2,p_{j,\perp}^2]/S
c                  (cosh(\eta_i-\eta_j)-cos(\phi_1-\phi_2))
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:3),p2(0:3)
c
c     Local
c

      include 'maxparticles.inc'
      include 'run.inc'

      double precision pt1,pt2,ptm1,eta1,eta2,phi1,phi2,p1a,p2a,costh
      integer j
c-----
c  Begin Code
c-----
      if ((lpp(1).eq.0).and.(lpp(2).eq.0)) then
      p1a = dsqrt(p1(1)**2+p1(2)**2+p1(3)**2)
      p2a = dsqrt(p2(1)**2+p2(2)**2+p2(3)**2)
      if (p1a*p2a .ne. 0d0) then
         costh = (p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3))/(p1a*p2a)
         dj1 = 2d0*p1(0)**2*(1d0-costh)   !Durham
c         dj = 2d0*p1(0)*p2(0)*(1d0-costh)    !JADE
      else
         print*,'Warning 0 momentum in Durham algorythm'
         write(*,'(4e15.5)') (p1(j),j=0,3)
         write(*,'(4e15.5)') (p2(j),j=0,3)
         dj1 = 0d0
      endif
      else
        pt1 = p1(1)**2+p1(2)**2
        pt2 = p2(1)**2+p2(2)**2
        p1a = dsqrt(pt1+p1(3)**2)
        p2a = dsqrt(pt2+p2(3)**2)
        eta1 = 0.5d0*log((p1a+p1(3))/(p1a-p1(3)))
        eta2 = 0.5d0*log((p2a+p2(3))/(p2a-p2(3)))
        ptm1 = max((p1(0)-p1(3))*(p1(0)+p1(3)),0d0)
        dj1 = 2d0*ptm1*(cosh(eta1-eta2)-
     &     (p1(1)*p2(1)+p1(2)*p2(2))/dsqrt(pt1*pt2))
c     write(*,*) 'p1  = ',p1(0),',',p1(1),',',p1(2),',',p1(3)
c     write(*,*) 'pm1 = ',pm1,', p1a = ',p1a,'eta1 = ',eta1
c     write(*,*) 'p2  = ',p2(0),',',p2(1),',',p2(2),',',p2(3)
c     write(*,*) 'pm2 = ',pm2,', p2a = ',p2a,'eta2 = ',eta2
c     write(*,*) 'dj = ',dj
      endif
      end
      
      double precision function DJB(p1)
c***************************************************************************
c     Uses kt algorythm to calculate the y value for one parton
c       y_i    = p_{i,\perp}^2/S
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:4)  ! 4 is mass**2
c
c     Local
c
      double precision pm1
      include 'maxparticles.inc'
      include 'run.inc'

c-----
c  Begin Code
c-----
c      pm1=max(p1(0)**2-p1(1)**2-p1(2)**2-p1(3)**2,0d0)
      if ((lpp(1).eq.0).and.(lpp(2).eq.0)) then
c        write(*,*) 'kin_functions.f: Error. No jet measure w.r.t. beam.'
c        djb = 0d0
         djb=max(p1(0),0d0)**2
      else
        djb = (p1(0)-p1(3))*(p1(0)+p1(3)) ! p1(1)**2+p1(2)**2+pm1
c        djb = p1(1)**2+p1(2)**2+p1(4)
      endif
      end

      double precision function PYJB(p2,p1,ppart,z)
c***************************************************************************
c     Calculate the Pythia ISR evolution pT2
c       pTE2 = (1-z)(Q2+m2), Q2=-(p1-p2)**2, z=sred/sprev
c     Note! p1 and p2 must have mass**2 component!
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:4),p2(0:4),ppart(0:3),z
c
c     Local
c
      double precision sred,sprev,Q2,pstar(0:3),pm2
      integer i
      double precision dot,SumDot,PT

c-----
c  Begin Code
c-----
      pm2=0

      if(p1(4).gt.0.or.p2(4).gt.0.and..not.
     $   (p1(4).gt.0.and.p2(4).gt.0)) pm2=max(p1(4),p2(4))
      do i=0,3
        pstar(i)=p1(i)-p2(i)
      enddo
      Q2=-dot(pstar,pstar)+pm2
      if(Q2.lt.0)then
c        print *,'Error in PYJB: Q2 = ',Q2
        PYJB=1d30
        return
      endif
      sprev=SumDot(p1,ppart,1d0)
      sred=SumDot(pstar,ppart,1d0)

      if(sred.lt.1d0)then
        PYJB=1d20
        z=0d0
        return
      endif

      z=sred/sprev
      if(z.gt.1.or.z.lt.0)then
        print *,'Error in PYJB: z = ',z,', sprev = ',sprev,
     $     ', sred = ',sred,', Q2 = ',Q2
        stop
      endif
      PYJB=(1d0-z)*Q2
      end

      double precision function zclus(p2,p1,ppart)
c***************************************************************************
c     Calculate the Pythia ISR evolution pT2
c     z=sred/sprev
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:3),p2(0:3),ppart(0:3)
c
c     Local
c
      double precision sred,sprev,pstar(0:3)
      integer i, nerr
      data nerr/0/
      double precision dot,SumDot

c-----
c  Begin Code
c-----
      do i=0,3
        pstar(i)=p1(i)-p2(i)
      enddo
      sprev=SumDot(p1,ppart,1d0)
      sred=SumDot(pstar,ppart,1d0)

      if(sred.lt.1d0)then
        zclus=0d0
        return
      endif

      zclus=sred/sprev
      if((zclus.gt.1.or.zclus.lt.0).and.nerr.le.10)then
        print *,'Error in zclus: zclus = ',zclus,', sprev = ',sprev,
     $     ', sred = ',sred
        nerr=nerr+1
        if(nerr.eq.10)
     $       print *,'No more zclus errors will be printed'
      endif

      return
      end

      double precision function DJ2(p1,p2)
c***************************************************************************
c     Uses Lorentz
c***************************************************************************
      implicit none
c
c     Arguments
c
      double precision p1(0:3),p2(0:3)
c
c     Local
c
      integer j
c
c     External
c
      double precision dot
c-----
c  Begin Code
c-----
      dj2 = dot(p1,p1)+2d0*dot(p1,p2)+dot(p2,p2)
      return
      end

      subroutine switchmom(p1,p,ic,jc,nexternal)
c**************************************************************************
c     Changes stuff for crossings
c**************************************************************************
      implicit none
      integer nexternal
      integer jc(nexternal),ic(nexternal)
      real*8 p1(0:3,nexternal),p(0:3,nexternal)
      integer i,j
c-----
c Begin Code
c-----
      do i=1,nexternal
         do j=0,3
            p(j,i)=p1(j,ic(i))
         enddo
      enddo
      do i=1,nexternal
         jc(i)=1
      enddo
      jc(ic(1))=-1
      jc(ic(2))=-1
      end

      subroutine switchhel(hel,hel1,ic,nexternal)
c**************************************************************************
c     Changes stuff for crossings
c**************************************************************************
      implicit none
      integer nexternal
      integer ic(nexternal),hel(nexternal),hel1(nexternal)
      integer i
c-----
c Begin Code
c-----
      do i=1,nexternal
         hel1(i)=hel(ic(i))
      enddo
      end

      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)

      if(dabs(dot).lt.1d-6)then ! solve numerical problem 
         dot=0d0
      endif

      end
C*****************************************************************************
C*****************************************************************************
C                      MadWeight function
C*****************************************************************************
C*****************************************************************************

      double precision function threedot(p1,p2)
C****************************************************************************
C     3-Vector  product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      threedot=p1(1)*p2(1)+p1(2)*p2(2)+p1(3)*p2(3)

      end


      double precision function rho(p1)
C****************************************************************************
C     computes rho(p)=dsqrt (p(1)**2+p(2)**2+p(3)**2)
C****************************************************************************
      implicit none
      double precision p1(0:3)
      double precision  threedot
      external  threedot
c
      rho=dsqrt(threedot(p1,p1))

      end

      double precision function theta(p)
c************************************************************************
c     Returns polar angle of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:3)
c-----
c  Begin Code
c-----

      theta=dacos(p(3)/dsqrt(p(1)**2+p(2)**2+p(3)**2))

      return
      end

      double precision function eta(p)
c************************************************************************
c     Returns pseudo rapidity of particle
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision p(0:3)
c
c     external
c
      double precision theta
      external theta
c-----
c  Begin Code
c-----

      eta=-dlog(dtan(theta(p)/2))

      return
      end

      subroutine four_momentum(theta,phi,rho,m,p)
c****************************************************************************
c     modif 3/07/07 : this subroutine defines 4-momentum from theta,phi,rho,m
c     with rho=px**2+py**2+pz**2
c****************************************************************************
c
c     argument
c
      double precision theta,phi,rho,m,p(0:3)
c
      P(1)=rho*dsin(theta)*dcos(phi)
      P(2)=rho*dsin(theta)*dsin(phi)
      P(3)=rho*dcos(theta)
      P(0)=dsqrt(rho**2+m**2)

      return
      end
      subroutine four_momentum_set2(eta,phi,PT,m,p)
c****************************************************************************
c     modif 16/11/06 : this subroutine defines 4-momentum from PT,eta,phi,m
c****************************************************************************
c
c     argument
c
      double precision PT,eta,phi,m,p(0:3)
c
c
c
      P(1)=PT*dcos(phi)
      P(2)=PT*dsin(phi)
      P(3)=PT*dsinh(eta)
      P(0)=dsqrt(p(1)**2+p(2)**2+p(3)**2+m**2)  
      return
      end



      DOUBLE PRECISION  FUNCTION phi(p)
c************************************************************************
c     MODIF 16/11/06 : this subroutine defines phi angle
c                      phi is defined from 0 to 2 pi
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision  p(0:3)
c
c     Parameter
c

      double precision pi,zero
      parameter (pi=3.141592654d0,zero=0d0)
c-----
c  Begin Code
c-----
c 
      if(p(1).gt.zero) then
      phi=datan(p(2)/p(1))
      else if(p(1).lt.zero) then
      phi=datan(p(2)/p(1))+pi
      else if(p(2).GE.zero) then !remind that p(1)=0
      phi=pi/2d0
      else if(p(2).lt.zero) then !remind that p(1)=0
      phi=-pi/2d0
      endif
      if(phi.lt.zero) phi=phi+2*pi
      return
      end

