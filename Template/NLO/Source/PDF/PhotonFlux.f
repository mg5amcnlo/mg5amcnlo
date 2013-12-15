c/* ********************************************************* */
c/*  Equivalent photon approximation structure function.   * */
c/*     Improved Weizsaecker-Williams formula              * */
c/*   V.M.Budnev et al., Phys.Rep. 15C (1975) 181          * */
c/* ********************************************************* */
c   provided by Tomasz Pierzchala - UCL

      real*8 function epa_electron(x,q2max)
      integer i
      real*8 x,phi_f
      real*8 xin
      real*8 alpha
      real*8 f, q2min,q2max
      real*8 PI
      data PI/3.14159265358979323846/

      data xin/0.511d-3/ !electron mass in GeV

      alpha = .0072992701

C     // x = omega/E = (E-E')/E
      if (x.lt.1) then
         q2min= xin*xin*x*x/(1-x)
         if(q2min.lt.q2max) then 
            f = alpha/2d0/PI*
     &           (2d0*xin*xin*x*(-1/q2min+1/q2max)+
     &           (2-2d0*x+x*x)/x*dlog(q2max/q2min))
            
         else
           f = 0. 
         endif
      else
         f= 0.
      endif
c      write (*,*) x,dsqrt(q2min),dsqrt(q2max),f
      if (f .lt. 0) f = 0
      epa_electron= f

      end

      real*8 function epa_proton(x,q2max)
      integer i
      real*8 x,phi_f
      real*8 xin
      real*8 alpha,qz
      real*8 f, qmi,qma, q2max
      real*8 PI
      data PI/3.14159265358979323846/

      data xin/0.938/ ! proton mass in GeV

      alpha = .0072992701
      qz = 0.71
    
C     // x = omega/E = (E-E')/E
      if (x.lt.1) then
         qmi= xin*xin*x*x/(1-x)
         if(qmi.lt.q2max) then          
            f = alpha/PI*(phi_f(x,q2max/qz)-phi_f(x,qmi/qz))*(1-x)/x
         else
            f=0.
         endif
      else
         f= 0.
      endif
      if (f .lt. 0) f = 0
      epa_proton= f
      end

      real*8 function phi_f(x,qq)
      real*8 x, qq
      real*8 y,qq1,f
      real*8 a,b,c

       a = 7.16
       b = -3.96
       c = .028

      qq1=1+qq
      y= x*x/(1-x)
      f=(1+a*y)*(-log(qq1/qq)+1/qq1+1/(2*qq1*qq1)+1/(3*qq1*qq1*qq1))
      f=f + (1-b)*y/(4*qq*qq1*qq1*qq1);
      f=f+ c*(1+y/4)*(log((qq1-b)/qq1)+b/qq1+b*b/(2*qq1*qq1)+
     $b*b*b/(3*qq1*qq1*qq1))
      phi_f= f
      end
  
