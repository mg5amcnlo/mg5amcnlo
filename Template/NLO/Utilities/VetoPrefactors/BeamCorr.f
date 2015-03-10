       subroutine BeamCorr(x1, x2, mu, mum, muMad, f1, f2, 
     $           ptjmax, alpha, alpham, 
     $           BCorr0, BCorrm0, BCorr1, BCorrm1)
       IMPLICIT NONE
c 
      DOUBLE PRECISION, intent(in)  :: x1, x2, ptjmax
      DOUBLE PRECISION, intent(in)  :: mu, muMad,alpha
      DOUBLE PRECISION, intent(in)  :: alpham, mum
      INTEGER, intent(in)  :: f1, f2       	
      DOUBLE PRECISION, intent(out) :: BCorr0, BCorr1
      DOUBLE PRECISION, intent(out) :: BCorrm0, BCorrm1
C
      DOUBLE PRECISION GetOnePDF,Beam1,Beam2
      DOUBLE PRECISION Beam1m,Beam2m
      CHARACTER prefix*50
C
C  GLOBAL Parameters
C 
      include 'parameters.inc'

      prefix = "Grids/BeRn2014NNLO" ! prefix for the grid files
c--   iset specifies which of our three grids is used 
c--   (0=PDF, 1=B_Rest, 2=B_PT)
c
      Beam1 = (GetOnePDF(prefix,1,x1,mu,f1)+
     $       Log(mu**2/ptjmax**2)*GetOnePDF(prefix,2,x1,mu,f1))

      Beam2 = (GetOnePDF(prefix,1,x2,mu,f2)+
     $       Log(mu**2/ptjmax**2)*GetOnePDF(prefix,2,x2,mu,f2))

      BCorr1= 1/(4.*Pi)*
     $        (Beam1*x1/GetOnePDF(prefix,0,x1,mu,f1)+
     $         Beam2*x2/GetOnePDF(prefix,0,x2,mu,f2))
c   
      BCorr0= GetOnePDF(prefix,0,x1,mu,f1)/
     $        GetOnePDF(prefix,0,x1,muMad,f1)*
     $        GetOnePDF(prefix,0,x2,mu,f2)/
     $        GetOnePDF(prefix,0,x2,muMad,f2)
c
c    Same for the Beam Function at the matching scale:
      Beam1m = (GetOnePDF(prefix,1,x1,mum,f1)+
     $       Log(mum**2/ptjmax**2)*GetOnePDF(prefix,2,x1,mum,f1))

      Beam2m = (GetOnePDF(prefix,1,x2,mum,f2)+
     $       Log(mum**2/ptjmax**2)*GetOnePDF(prefix,2,x2,mum,f2))

      BCorrm1= alpham/(4.*Pi)*
     $        (Beam1m*x1/GetOnePDF(prefix,0,x1,mum,f1)+
     $         Beam2m*x2/GetOnePDF(prefix,0,x2,mum,f2))

      BCorrm0= GetOnePDF(prefix,0,x1,mum,f1)/
     $         GetOnePDF(prefix,0,x1,muMad,f1)*
     $         GetOnePDF(prefix,0,x2,mum,f2)/
     $         GetOnePDF(prefix,0,x2,muMad,f2)

      end subroutine BeamCorr
