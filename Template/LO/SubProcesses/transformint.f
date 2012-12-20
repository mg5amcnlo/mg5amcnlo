      subroutine transformint(p,q,ip,jp,kp,ijp,kpp,mass_i,mass_j,mass_k)
************************************************************************
*     Author: NG                                                       *
*     July, 2008.                                                      *
*     Given p (p1 + p2 --> p3 ... px .. p_(nexternal))                 *
*     produce q (q1 + q2 --> q3 ... qx .. q_(nexternal-1))             *
*                                                                      *
*     ip is the emitter, jp the unresolved and kp is the spectator     *
*     after branching.                                                 *
*     ijp and kpp are the emitter and spectator before branching       *
*                                                                      *
*     
************************************************************************
      implicit none 
      include 'nexternal.inc'

      double precision p(0:3,nexternal),q(0:3,nexternal)
      integer j,nu,ip,jp,kp,ijp,kpp
      double precision mass_i,mass_j,mass_k

      do j=1,nexternal
         do nu=0,3
            q(nu,j)=p(nu,j)
         enddo
      enddo

      end

