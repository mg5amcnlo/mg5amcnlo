      subroutine boostx(p,q , pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momentum p is assumed to be given in the rest frame of q.  pboost is
c the momentum p boosted to the frame in which q is given.  q must be a
c timelike momentum.
c
c input:
c       real    p(0:3)         : four-momentum p in the q rest  frame
c       real    q(0:3)         : four-momentum q in the boosted frame
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
      double precision pp
#endif
c
      qq = q(1)**2+q(2)**2+q(3)**2

#ifdef HELAS_CHECK
      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in boostx is zero momentum'
      endif
      if (abs(q(0))+qq.eq.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx is zero momentum'
      endif
      if (p(0).le.rZero) then
         write(stdo,*)
     &        ' helas-warn  : p(0:3) in boostx has not positive energy'
         write(stdo,*)
     &        '             : p(0) = ',p(0)
      endif
      if (q(0).le.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx has not positive energy'
         write(stdo,*)
     &        '             : q(0) = ',q(0)
      endif
      pp=p(0)**2-p(1)**2-p(2)**2-p(3)**2
      if (pp.lt.rZero) then
         write(stdo,*)
     &        ' helas-warn  : p(0:3) in boostx is spacelike'
         write(stdo,*)
     &        '             : p**2 = ',pp
      endif
      if (q(0)**2-qq.le.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx is not timelike'
         write(stdo,*)
     &        '             : q**2 = ',q(0)**2-qq
      endif
      if (qq.eq.rZero) then
         write(stdo,*)
     &   ' helas-warn  : q(0:3) in boostx has zero spacial components'
      endif
#endif

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         m = sqrt(q(0)**2-qq)
         lf = ((q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)+pq)/m
         pboost(1) =  p(1)+q(1)*lf
         pboost(2) =  p(2)+q(2)*lf
         pboost(3) =  p(3)+q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end



      subroutine boostm(p,q,m, pboost)
c
c This subroutine performs the Lorentz boost of a four-momentum.  The
c momentum p is assumed to be given in the rest frame of q.  pboost is
c the momentum p boosted to the frame in which q is given.  q must be a
c timelike momentum.
c
c input:
c       real    p(0:3)         : four-momentum p in the q rest  frame
c       real    q(0:3)         : four-momentum q in the boosted frame
c       real    m              : mass of q (avoid numerical precision problem)
c
c output:
c       real    pboost(0:3)    : four-momentum p in the boosted frame
c
      implicit none
      double precision p(0:3),q(0:3),pboost(0:3),pq,qq,m,lf

      double precision rZero
      parameter( rZero = 0.0d0 )

#ifdef HELAS_CHECK
      integer stdo
      parameter( stdo = 6 )
      double precision pp
#endif

      qq = q(1)**2+q(2)**2+q(3)**2

#ifdef HELAS_CHECK
      if (abs(p(0))+abs(p(1))+abs(p(2))+abs(p(3)).eq.rZero) then
         write(stdo,*)
     &        ' helas-error : p(0:3) in boostx is zero momentum'
      endif
      if (abs(q(0))+qq.eq.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx is zero momentum'
      endif
      if (p(0).le.rZero) then
         write(stdo,*)
     &        ' helas-warn  : p(0:3) in boostx has not positive energy'
         write(stdo,*)
     &        '             : p(0) = ',p(0)
      endif
      if (q(0).le.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx has not positive energy'
         write(stdo,*)
     &        '             : q(0) = ',q(0)
      endif
      pp=p(0)**2-p(1)**2-p(2)**2-p(3)**2
      if (pp.lt.rZero) then
         write(stdo,*)
     &        ' helas-warn  : p(0:3) in boostx is spacelike'
         write(stdo,*)
     &        '             : p**2 = ',pp
      endif
      if (q(0)**2-qq.le.rZero) then
         write(stdo,*)
     &        ' helas-error : q(0:3) in boostx is not timelike'
         write(stdo,*)
     &        '             : q**2 = ',q(0)**2-qq
      endif
      if (qq.eq.rZero) then
         write(stdo,*)
     &   ' helas-warn  : q(0:3) in boostx has zero spacial components'
      endif
#endif

      if ( qq.ne.rZero ) then
         pq = p(1)*q(1)+p(2)*q(2)+p(3)*q(3)
         lf = ((q(0)-m)*pq/qq+p(0))/m
         pboost(0) = (p(0)*q(0)+pq)/m
         pboost(1) =  p(1)+q(1)*lf
         pboost(2) =  p(2)+q(2)*lf
         pboost(3) =  p(3)+q(3)*lf
      else
         pboost(0) = p(0)
         pboost(1) = p(1)
         pboost(2) = p(2)
         pboost(3) = p(3)
      endif
c
      return
      end
