      subroutine rotxxx(p,q , prot)
c
c This subroutine performs the spacial rotation of a four-momentum.
c the momentum p is assumed to be given in the frame where the spacial
c component of q points the positive z-axis.  prot is the momentum p
c rotated to the frame where q is given.
c
c input:
c       real    p(0:3)         : four-momentum p in q(1)=q(2)=0 frame
c       real    q(0:3)         : four-momentum q in the rotated frame
c
c output:
c       real    prot(0:3)      : four-momentum p in the rotated frame
c     
      implicit none
      double precision p(0:3),q(0:3),prot(0:3),qt2,qt,psgn,qq,p1

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
#endif
c
      prot(0) = p(0)

      qt2 = q(1)**2 + q(2)**2

#ifdef HELAS_CHECK
      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in rotxxx is zero momentum'
      endif
      if (abs(q(0))+abs(q(3))+qt2.eq.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in rotxxx is zero momentum'
      endif
      if (qt2+abs(q(3)).eq.rZero) then
         write(stdo,*)
     &   ' helas-warn  : q(0:3) in rotxxx has zero spacial momentum'
      endif
#endif

      if ( qt2.eq.rZero ) then
          if ( q(3).eq.rZero ) then
             prot(1) = p(1)
             prot(2) = p(2)
             prot(3) = p(3)
          else
             psgn = dsign(rOne,q(3))
             prot(1) = p(1)*psgn
             prot(2) = p(2)*psgn
             prot(3) = p(3)*psgn
          endif
      else
          qq = sqrt(qt2+q(3)**2)
          qt = sqrt(qt2)
          p1 = p(1)
          prot(1) = q(1)*q(3)/qq/qt*p1 -q(2)/qt*p(2) +q(1)/qq*p(3)
          prot(2) = q(2)*q(3)/qq/qt*p1 +q(1)/qt*p(2) +q(2)/qq*p(3)
          prot(3) =          -qt/qq*p1               +q(3)/qq*p(3)
      endif
c
      return
      end
