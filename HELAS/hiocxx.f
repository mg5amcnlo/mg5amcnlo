c
c ----------------------------------------------------------------------
c
      subroutine hiocxx(fic,foc,gc,smass,swidth , hioc)                
c                                                                       
c this subroutine computes an off-shell scalar current from an external 
c antifermion pair.                                                     
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex gc(2)          : coupling constants                 gchf
c       real    smass          : mass  of output scalar s               
c       real    swidth         : width of output scalar s               
c                                                                       
c output:                                                               
c       complex hioc(3)        : scalar current           j(<fic|s|foc>)
c                                                                       
      implicit none
      double complex fic(6),foc(6),hioc(3),gc(2),dn
      double precision q(0:3),smass,swidth,q2
c
      hioc(2) = foc(5)-fic(5)
      hioc(3) = foc(6)-fic(6)

      q(0) = dble( hioc(2))
      q(1) = dble( hioc(3))
      q(2) = dimag(hioc(3))
      q(3) = dimag(hioc(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

      dn = -dcmplx( q2-smass**2, smass*swidth )

      hioc(1) = (  gc(1)*(foc(1)*fic(1)+foc(2)*fic(2))
     &           + gc(2)*(foc(3)*fic(3)+foc(4)*fic(4)) )/dn
c
      return
      end
