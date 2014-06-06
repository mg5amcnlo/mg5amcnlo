       subroutine BeamCorr(x1, x2, mu, muMad, f1, f2, 
     $           ptjmax, alpha, BCorr)
       IMPLICIT NONE
c 
      DOUBLE PRECISION, intent(in)  :: x1, x2, ptjmax
      DOUBLE PRECISION, intent(in)  :: mu, muMad,alpha  
      INTEGER, intent(in)  :: f1, f2       	
      DOUBLE PRECISION, intent(out) :: BCorr
C
      DOUBLE PRECISION GetOnePDF,Beam1,Beam2
      CHARACTER prefix*50
C
C  GLOBAL Parameters
C 
      include 'parameters.inc'

      prefix = "Grids/BeRn2014NNLO" ! prefix for the grid files
c--   iset specifies which of our three grids is used 
c--   (0=PDF, 1=B_Rest, 2=B_PT)

      Beam1 = GetOnePDF(prefix,0,x1,mu,f1)/x1+
     $       alpha/(4.*Pi)*(GetOnePDF(prefix,1,x1,mu,f1)+
     $       Log(mu**2/ptjmax**2)*GetOnePDF(prefix,2,x1,mu,f1))

      Beam2 = GetOnePDF(prefix,0,x2,mu,f2)/x2+
     $       alpha/(4.*Pi)*(GetOnePDF(prefix,1,x2,mu,f2)+
     $       Log(mu**2/ptjmax**2)*GetOnePDF(prefix,2,x2,mu,f2))

      BCorr= Beam1*x1/GetOnePDF(prefix,0,x1,muMad,f1)*
     $       Beam2*x2/GetOnePDF(prefix,0,x2,muMad,f2)

      end subroutine BeamCorr
