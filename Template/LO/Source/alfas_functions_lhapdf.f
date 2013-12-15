C
C-----------------------------------------------------------------------------
C
      double precision function alfa(alfa0,qsq )
C
C-----------------------------------------------------------------------------
C
C	This function returns the 1-loop value of alpha.
C
C	INPUT: 
C		qsq   = Q^2
C
C-----------------------------------------------------------------------------
C
      implicit none
      double precision  qsq,alfa0
c
c constants
c
      double precision  One, Three, Pi,zmass
      parameter( One = 1.0d0, Three = 3.0d0 )
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( zmass = 91.188d0 )
cc
      alfa = alfa0 / ( 1.0d0 - alfa0*dlog( qsq/zmass**2 ) /Three /Pi )
ccc
      return
      end

C
C-----------------------------------------------------------------------------
C
      double precision function alfaw(alfaw0,qsq,nh )
C
C-----------------------------------------------------------------------------
C
C	This function returns the 1-loop value of alpha_w.
C
C	INPUT: 
C		qsq = Q^2
C               nh  = # of Higgs doublets
C
C-----------------------------------------------------------------------------
C
      implicit none
      double precision  qsq, alphaw, dum,alfaw0
      integer  nh, nq
c
c	  include
c
	  
c
c constants
c
      double precision  Two, Four, Pi, Twpi, zmass,tmass
      parameter( Two = 2.0d0, Four = 4.0d0 )
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Twpi = 3.0d0*Four*Pi )
      parameter( zmass = 91.188d0,tmass=174d0 )
cc
      if ( qsq.ge.tmass**2 ) then
         nq = 6
      else
         nq = 5
      end if
      dum = ( 22.0d0 - Four*nq - nh/Two ) / Twpi
      alfaw = alfaw0 / ( 1.0d0 + dum*alfaw0*dlog( qsq/zmass**2 ) )
ccc
      return
      end

C-----------------------------------------------------------------------------
C
      DOUBLE PRECISION FUNCTION ALPHAS(Q)
C     wrapper to the lhapdf alphaS
C-----------------------------------------------------------------------------
      IMPLICIT NONE
c
      include 'alfas.inc'
      REAL*8 Q,alphasPDF
      external alphasPDF

      ALPHAS=alphasPDF(Q)

      RETURN
      END

C-----------------------------------------------------------------------------
C
      double precision function mfrun(mf,scale,asmz,nloop)
C
C-----------------------------------------------------------------------------
C
C	This function returns the 2-loop value of a MSbar fermion mass
C       at a given scale.
C
C	INPUT: mf    = MSbar mass of fermion at MSbar fermion mass scale 
C	       scale = scale at which the running mass is evaluated
C	       asmz  = AS(MZ) : this is passed to alphas(scale,asmz,nloop)
C              nloop = # of loops in the evolution
C       
C
C
C	EXTERNAL:      double precision alphas(scale,asmz,nloop)
C                      
C-----------------------------------------------------------------------------
C
      implicit none
C
C     ARGUMENTS
C
      double precision  mf,scale,asmz
      integer           nloop
C
C     LOCAL
C
      double precision  beta0, beta1,gamma0,gamma1
      double precision  A1,as,asmf,l2
      integer  nf
C
C     EXTERNAL
C
      double precision  alphas
      external          alphas
c
c     CONSTANTS
c
      double precision  One, Two, Three, Pi
      parameter( One = 1.0d0, Two = 2.0d0, Three = 3.0d0 )
      parameter( Pi = 3.14159265358979323846d0) 
      double precision tmass
      parameter(tmass=174d0)
cc
C
C
      if ( mf.gt.tmass ) then
         nf = 6
      else
         nf = 5
      end if

      beta0 = ( 11.0d0 - Two/Three *nf )/4d0
      beta1 = ( 102d0  - 38d0/Three*nf )/16d0
      gamma0= 1d0
      gamma1= ( 202d0/3d0  - 20d0/9d0*nf )/16d0
      A1    = -beta1*gamma0/beta0**2+gamma1/beta0
      as    = alphas(scale)
      asmf  = alphas(mf)
      l2    = (1+ A1*as/Pi)/(1+ A1*asmf/Pi)
      
      
      mfrun = mf * (as/asmf)**(gamma0/beta0)

      if(nloop.eq.2) mfrun =mfrun*l2
ccc
      return
      end

