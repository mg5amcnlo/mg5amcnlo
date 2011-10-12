      subroutine hstlxx(s1,t2,gc,smass,swidth , hst)
c- by RF - Mar. 2006
c
c This subroutine computes an off-shell scalar current from the three-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex t2(3)          : internal particle (scalar)           s2
c       complex gc             : coupling constant                  ghhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hst(3)         : scalar current              j(s':s1,s2)
c     
      implicit none
      INTEGER DIM
      PARAMETER(DIM=18)

      double complex s1(DIM),t2(DIM),hst(DIM),dg
      double precision q(0:3),smass,swidth,q2,gc

      hst(2) = s1(2)+t2(2)
      hst(3) = s1(3)+t2(3)

      q(0) = -dble( hst(2))
      q(1) = -dble( hst(3))
      q(2) = -dimag(hst(3))
      q(3) = -dimag(hst(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)


      dg = gc /dcmplx( q2-smass**2, smass*swidth )
      

      hst(1) = dg*s1(1)*t2(1)

      return
      end
