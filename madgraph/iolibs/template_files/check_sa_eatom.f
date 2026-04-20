      PROGRAM DRIVER
C**************************************************************************
C     THIS IS THE DRIVER FOR CHECKING THE STANDALONE MATRIX ELEMENT.
C     IT USES A SIMPLE PHASE SPACE GENERATOR
C     Fabio Maltoni - 3rd Febraury 2007
C**************************************************************************
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
C     
C     INCLUDE FILES
C     
C---  the include file with the values of the parameters and masses	
      INCLUDE "coupl.inc"
C---  integer nexternal ! number particles (incoming+outgoing) in the me 
      INCLUDE "nexternal.inc" 
C---  particle masses
      REAL*8 PMASS(NEXTERNAL)
      REAL*8 TOTALMASS,PVMIN,PBEAM
C---  integer    n_max_cg
      INCLUDE "ngraphs.inc"     !how many diagrams (could be useful to know...)

C     
C     LOCAL
C     
      INTEGER I,J,K
      REAL*8 P(0:3,NEXTERNAL)   ! four momenta. Energy is the zeroth component.
      REAL*8 SQRTS,MATELEM           ! sqrt(s)= center of mass energy 
      REAL*8 PIN(0:3), POUT(0:3)
      CHARACTER*120 BUFF(NEXTERNAL)
      include "iatom.inc"
C     
C     EXTERNAL
C     
      REAL*8 DOT
      EXTERNAL DOT
      
C-----
C     BEGIN CODE
C-----
C     
C---  INITIALIZATION CALLS
C     
c---  Call to initialize the values of the couplings, masses and widths 
c     used in the evaluation of the matrix element. The primary parameters of the
c     models are read from Cards/param_card.dat. The secondary parameters are calculated
c     in Source/MODEL/couplings.f. The values are stored in common blocks that are listed
c     in coupl.inc .

      call setpara('param_card.dat')  !first call to setup the paramaters
      include "pmass.inc"             !set up masses

      TOTALMASS = 0.0d0
      DO I=NINCOMING+1,NEXTERNAL
        TOTALMASS = TOTALMASS + PMASS(I)
      ENDDO

c---  Now use a simple multipurpose PS generator (RAMBO) just to get a 
c     RANDOM set of four momenta of given masses pmass(i) to be used to evaluate 
c     the MadGraph5_aMC@NLO matrix-element.       
c     Alternatevely, here the user can call or set the four momenta at his will, see below.
c     	
      IF(nincoming.EQ.1) THEN
         SQRTS=PMASS(1)
	 WRITE(*,*)"ERROR: eatom does not work for decay"
	 STOP
      ELSE
         IF(nexternal.EQ.3)THEN
            SQRTS=PMASS(3)
         ELSE
            SQRTS=3d0*MAX(TOTALMASS,PMASS(1)+PMASS(2))
         ENDIF
	 PVMIN=SQRT((SQRTS-PMASS(IATOM))**2-PMASS(3-IATOM)**2)
	 PBEAM=3.7d0*PVMIN
      ENDIF

      call printout()

      CALL GET_EATOM_MOMENTA(IATOM,SQRTS,PBEAM,PMASS,P)	
c
c	  write the information on the four momenta 
c
      write (*,*)
      write (*,*) " Phase space point:"
      write (*,*)
      write (*,*) "-----------------------------------------------------------------------------"
      write (*,*)  "n        E             px             py              pz               m "
      do i=1,nexternal
         write (*,'(i2,1x,5e15.7)') i, P(0,i),P(1,i),P(2,i),P(3,i), 
     . SIGN(1d0,DOT(p(0,i),p(0,i)))*dsqrt(dabs(DOT(p(0,i),p(0,i))))
      enddo
      write (*,*) "-----------------------------------------------------------------------------"

c     
c     Now we can call the matrix element!
c
      CALL SMATRIX(P,MATELEM)
c

      write (*,*) "Matrix element = ", MATELEM, " GeV^",-(2*nexternal-5)	
      write (*,*) "-----------------------------------------------------------------------------"


cc
cc      Copy down here (or read in) the four momenta as a string. 
cc      
cc
c      buff(1)=" 1   0.5630480E+04  0.0000000E+00  0.0000000E+00  0.5630480E+04"
c      buff(2)=" 2   0.5630480E+04  0.0000000E+00  0.0000000E+00 -0.5630480E+04"
c      buff(3)=" 3   0.5466073E+04  0.4443190E+03  0.2446331E+04 -0.4864732E+04"
c      buff(4)=" 4   0.8785819E+03 -0.2533886E+03  0.2741971E+03  0.7759741E+03"
c      buff(5)=" 5   0.4916306E+04 -0.1909305E+03 -0.2720528E+04  0.4088757E+04"
cc
cc      Here the k,E,px,py,pz are read from the string into the momenta array.
cc      k=1,2          : incoming
cc      k=3,nexternal  : outgoing
cc
c      do i=1,nexternal
c         read (buff(i),*) k, P(0,i),P(1,i),P(2,i),P(3,i)
c      enddo
c
cc- print the momenta out
c
c      do i=1,nexternal
c         write (*,'(i2,1x,5e15.7)') i, P(0,i),P(1,i),P(2,i),P(3,i), 
c     .dsqrt(dabs(DOT(p(0,i),p(0,i))))
c      enddo
c
c      CALL SMATRIX(P,MATELEM)
c
c      write (*,*) "-------------------------------------------------"
c      write (*,*) "Matrix element = ", MATELEM, " GeV^",-(2*nexternal-8)	
c      write (*,*) "-------------------------------------------------"

      end
	
	  
	  
	  
      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end


      SUBROUTINE GET_EATOM_MOMENTA(IATOM,ENERGY,PBEAM,PMASS,P)
C---- auxiliary function to change convention between MadGraph5_aMC@NLO and rambo
c---- four momenta. 	  
	  IMPLICIT NONE
	  INCLUDE "nexternal.inc"
C     ARGUMENTS
          INTEGER IATOM
          REAL*8 PBEAM ! three-momentum of the beam
	  REAL*8 ENERGY,PMASS(NEXTERNAL),P(0:3,NEXTERNAL),WGT
C         LOCAL
         INTEGER I
         REAL*8 etot2,m1,m2,PV1,PV2,PV1MIN,PV1THR,PV2MIN,PV2MAX
         REAL*8 STHRESH,R1,R2,E1,ST,CT,phi,Q
         REAL(KIND(1d0)),PARAMETER::EPS=1d-12
         REAL(KIND(1d0)),PARAMETER::PI=3.141592653589793d0
         REAL*8 PT(0:3)

         IF(NINCOMING.NE.2)THEN
            WRITE(*,*)"ERROR: GET_EATOM_MOMENTA does't work for decay"
            STOP
         ENDIF

         IF(IATOM.NE.1.AND.IATOM.NE.2)THEN
            WRITE(*,*)"ERROR: IATOM =/= 1, 2"
            STOP
         ENDIF
         

         ETOT2=energy**2
         m1=pmass(1)
         m2=pmass(2)
         STHRESH=(m1+m2)**2
         IF(ETOT2.LT.STHRESH)THEN
            WRITE(*,*)"ERROR: energy is below threshold"
            STOP
         ENDIF

         IF(IATOM.EQ.2)THEN
            PV1MIN=DSQRT(MAX((DSQRT(ETOT2)-m2)**2-m1**2,0d0))
         ELSE
            PV1MIN=DSQRT(MAX((DSQRT(ETOT2)-m1)**2-m2**2,0d0))
         ENDIF
         IF(PBEAM.LT.PV1MIN)THEN
            WRITE(*,*)"ERROR: PBEAM is below threshold"
            STOP
         ENDIF

!
!     when pv1 >= Sqrt[lam[ETOT2, M1^2, M2^2]]/(2*M2),
!     pv1 + Q >= pv2 >= -pv1 + Q
!
!     when Sqrt[(Sqrt[ETOT2] - M2)^2 - M1^2] <= pv1 < Sqrt[lam[ETOT2, M1^2, M2^2]]/(2*M2)
!     pv1 + Q >= pv2 >= pv1 - Q
!
!     where lam(a,b,c)=a**2+b**2+c**2-2*a*b-2*b*c-2*a*c (Kallen function)
!           E1 = Sqrt[pv1^2+M1^2]
!           Q = Sqrt[(E1+M2)^2-ETOT2] = Sqrt[-ETOT2 + M1^2 + M2^2 + pv1^2 + 2 M2 Sqrt[M1^2 + pv1^2]]
!

         PV1THR=ETOT2**2+m1**4+m2**4-2d0*ETOT2*m1**2
     $        -2d0*ETOT2*m2**2-2d0*m1**2*m2**2
         CALL RANDOM_NUMBER(R1)
         IF(IATOM.EQ.2)THEN
            PV1THR=DSQRT(PV1THR)/(2d0*m2) ! pv10
            E1=SQRT(m1**2+PBEAM**2)
            Q=SQRT((E1+m2)**2-ETOT2)
            IF(PBEAM.LT.PV1THR)THEN               
               PV2MIN=PBEAM-Q
            ELSE
               PV2MIN=Q-PBEAM
            ENDIF
            PV2MAX=PBEAM+PV1MIN
            PV2=(PV2MAX-PV2MIN)*R1+PV2MIN
            CT=(2d0*m2*E1+m1**2+m2**2-ETOT2-PV2**2)/(2d0*PBEAM*PV2)            
         ELSE
            PV1THR=DSQRT(PV1THR)/(2d0*m1) ! pv10
            E1=SQRT(m2**2+PBEAM**2)
            Q=SQRT((E1+m1)**2-ETOT2)
            IF(PBEAM.LT.PV1THR)THEN
               PV2MIN=PBEAM-Q
            ELSE
               PV2MIN=Q-PBEAM
            ENDIF
            PV2MAX=PBEAM+PV1MIN
            PV2=(PV2MAX-PV2MIN)*R1+PV2MIN
            CT=(2d0*m1*E1+m1**2+m2**2-ETOT2-PV2**2)/(2d0*PBEAM*PV2)
         ENDIF
        
         IF(ABS(CT).GT.1d0)THEN
            WRITE(*,*)"ERROR: |cos(theta)|>1"
            STOP
         ENDIF

         ST=DSQRT(1d0-CT**2)
         
         CALL RANDOM_NUMBER(R2)
         phi=2d0*pi*R2

         IF(IATOM.EQ.2)THEN
            P(0,1)=E1
            P(1,1)=0d0
            P(2,1)=0d0
            P(3,1)=PBEAM

            P(0,2)=m2
            P(1,2)=PV2*ST*COS(phi)
            P(2,2)=PV2*ST*SIN(phi)
            P(3,2)=PV2*CT
         ELSE
            P(0,2)=E1
            P(1,2)=0d0
            P(2,2)=0d0
            P(3,2)=-PBEAM

            P(0,1)=m1
            P(1,1)=-PV2*ST*COS(phi)
            P(2,1)=-PV2*ST*SIN(phi)
            P(3,1)=-PV2*CT
         ENDIF

         PT(0:3)=P(0:3,1)+P(0:3,2)

         IF(NEXTERNAL.EQ.3)THEN
            P(0:3,3)=PT(0:3)
         ELSE

            CALL RAMBO_GENERAL(NEXTERNAL-2,PT,PMASS(3:NEXTERNAL),
     $           P(0:3,3:NEXTERNAL),WGT)
         ENDIF

         RETURN
         END
      

      SUBROUTINE RAMBO_GENERAL(N,PT,XM,PF,WGT)
******************************************************************************
*       PHASE-SPACE GENERATOR FOR EATOM                                      *
*    Generate by Claude Code with Claude Opus 4.7                            *
*     Amended by Hua-Sheng Shao on 20 April 2026                             *
*                                                                            *
* RAMBO-like 2->n phase space generator (Kleiss, Stirling, Ellis 1986)       *
* Beam p1  +  off-shell target p2=(ME,p2x,p2y,p2z)  ->  n on-shell particles *
* ME is fixed; p2(1:3) can be any real 3-momentum                            *
*     =>  p2^2 = ME^2 - |p2vec|^2 (off-shell)                                *
*                                                                            *
*                                                                            *
*    N  = NUMBER OF FINAL PARTICLES                                          *
*    PT = TOTAL INITIAL-STATE MOMENTUM (DIM=0:3)                             *
*    XM = PARTICLE MASSES ( DIM=N )                                          *
*    PF = FINAL-STATE PARTICLE MOMENTA ( DIM=(0:3,N) )                      *
*    WGT= WEIGHT OF THE EVENT                                                *
******************************************************************************
      IMPLICIT NONE
      INTEGER,INTENT(IN)::N
      REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::PT
      REAL(KIND(1d0)),INTENT(IN)::XM(N)
      REAL(KIND(1d0)),DIMENSION(0:3,N),INTENT(OUT)::PF
      REAL(KIND(1d0)),INTENT(OUT)::WGT
      REAL(KIND(1d0)),PARAMETER::PI=3.141592653589793d0
      REAL(KIND(1d0))::srt,MR,gam,aa,bq,E0,x
      REAL(KIND(1d0))::xi,f,fp,ct,phi,u,mtot
      REAL(KIND(1d0)),DIMENSION(0:3,N)::q
      REAL(KIND(1d0)),DIMENSION(0:3)::R
      REAL(KIND(1d0)),DIMENSION(3)::bv
      REAL(KIND(1d0)),DIMENSION(4)::rn
      REAL(KIND(1d0)),DIMENSION(N)::Ec,omg
      INTEGER::i,it

      srt = PT(0)**2
      do i=1,3
         srt=srt-PT(i)**2
      enddo
      srt = sqrt(max(srt, 0d0))
      wgt = 0d0
      mtot=0d0
      do i=1,N
         mtot=mtot+XM(i)
      enddo
      if (srt.le.mtot) return   ! below threshold

c --- Step 1: generate n massless isotropic momenta -----------------------
      R = 0d0
      do i = 1, n
         call random_number(rn)
         ct  = 2d0*rn(1) - 1d0
         phi = 2d0*pi*rn(2)
         u   = -log(rn(3)*rn(4)) ! energy ~ Gamma(2) => flat in phase space
         q(0,i) = u
         q(1,i) = u*sqrt(1d0-ct**2)*cos(phi)
         q(2,i) = u*sqrt(1d0-ct**2)*sin(phi)
         q(3,i) = u*ct
         R(0:3) = R(0:3) + q(0:3,i)
      end do

!---  Step 2: boost to rest frame of R, scale so sum q_i = (srt,0,0,0) ---
      MR = R(0)**2
      do i=1,3
         MR = MR - R(i)**2
      enddo
      MR = sqrt(max(MR, 1d-30))
      bv(1:3) = -R(1:3)/MR
      gam = R(0)/MR
      aa = 1d0/(1d0+gam)
      x = srt/MR
      do i = 1, n
         bq = dot_product(bv(1:3),q(1:3,i))
         E0 = q(0,i)
         q(0,i)   = x*(gam*E0 + bq)
         q(1:3,i) = x*(q(1:3,i) + bv*(aa*bq + E0))
      end do

!--- Step 3: mass correction — Newton-Raphson for xi ----------------------
!     Find xi in (0,1] such that  sum_i sqrt(m_i^2 + xi^2*E_i^2) = srt
      Ec(1:N) = q(0,1:N)        ! massless CM energies
      xi = 1d0
      do it = 1, 100
         omg(1:N) = sqrt(XM(1:N)**2 + xi**2*Ec(1:N)**2)
         f   = sum(omg(1:N)) - srt
         fp  = sum(xi*Ec(1:N)**2/omg)
         xi  = xi - f/fp
         if (abs(f).lt.1d-12*srt) exit
      end do
      do i = 1, n
         q(1:3,i) = xi*q(1:3,i)
         q(0,i)   = sqrt(XM(i)**2 + xi**2*Ec(i)**2)
      end do
      omg(1:N) = q(0,1:N)       ! on-shell energies

!---  Step 4: boost from CM rest frame to lab (total momentum = Pt) -------
      bv(1:3) = Pt(1:3)/srt
      gam = Pt(0)/srt
      aa = 1d0/(1d0+gam)
      do i = 1, n
         bq = dot_product(bv(1:3), q(1:3,i))
         E0 = q(0,i)
         q(0,i)   = gam*E0 + bq
         q(1:3,i) = q(1:3,i) + bv*(aa*bq + E0)
      end do
      pf(0:3,1:N) = q(0:3,1:N)

!---  Step 5: weight = (massless RAMBO weight) x (mass-correction Jacobian)
      call RAMBO_GENERAL_weight(n, srt, xi, Ec, omg, wgt)
      
      end subroutine RAMBO_GENERAL

      subroutine RAMBO_GENERAL_weight(n, srt, xi, Ec, omg, wgt)
! Massless part: (pi/2)^(n-1) * srt^(2n-4) / [(n-1)!(n-2)! * (2pi)^(3n-4)]
! Jacobian  J_n: xi^(2n-3) * prod(xi*E_i/omega_i) * srt / sum(xi^2*E_i^2/omega_i)
      implicit none
      integer,intent(in)::n
      real(kind(1d0)),intent(in)::srt,xi
      real(kind(1d0)),dimension(n),intent(in)::Ec,omg
      real(kind(1d0)),intent(out)::wgt
      real(kind(1d0))::logw, jac
      integer::i
      REAL(KIND(1d0)),PARAMETER::PI=3.141592653589793d0

      logw = real(n-1,8)*log(pi/2d0) + real(2*n-4,8)*log(srt)
     $     - real(3*n-4,8)*log(2d0*pi)
      do i = 1, n-1
         logw = logw - log(real(i,8))
      end do                    ! / (n-1)!
      do i = 1, n-2
         logw = logw - log(real(i,8))
      end do                    ! / (n-2)!

      jac = xi**(2*n-3) * product(xi*Ec(1:N)/omg(1:N))
     $     *srt/sum(xi**2*Ec(1:N)**2/omg(1:N))

      wgt = exp(logw) * jac
      return
      
      end subroutine RAMBO_GENERAL_weight






