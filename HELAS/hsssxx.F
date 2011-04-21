      subroutine hsssxx(s1,s2,s3,gc,smass,swidth , hsss)
c
c This subroutine computes an off-shell scalar current from the four-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex s3(3)          : third  scalar                        s3
c       complex gc             : coupling constant                 ghhhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hsss(3)        : scalar current           j(s':s1,s2,s3)
c     
      implicit none
      double complex s1(3),s2(3),s3(3),gc,hsss(3),dg
      double precision q(0:3),smass,swidth,q2

#ifdef HELAS_CHECK
      double precision rZero
      parameter( rZero = 0.0d0 )
      double complex cZero
      parameter( cZero = ( 0.0d0, 0.0d0 ) )
      integer stdo
      parameter( stdo = 6 )
#endif
c
#ifdef HELAS_CHECK
      if ( abs(s1(1)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : s1 in hsssxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in hsssxx has zero momentum'
      endif
      if ( abs(s2(1)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : s2 in hsssxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in hsssxx has zero momentum'
      endif
      if ( s3(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s3 in hsssxx is zero scalar'
      endif
      if ( abs(s3(2))+abs(s3(3)).eq.rZero ) then
          write(stdo,*)
     &        ' helas-error : s3 in hsssxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in hsssxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hsssxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hsssxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hsss(2) = s1(2)+s2(2)+s3(2)
      hsss(3) = s1(3)+s2(3)+s3(3)

      q(0) = dble( hsss(2))
      q(1) = dble( hsss(3))
      q(2) = dimag(hsss(3))
      q(3) = dimag(hsss(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hsss(2))+abs(hsss(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hsss in hsssxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hsss in hsssxx is on smass pole'
         write(stdo,*)
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*) 
     &        '             : abs(q)= ',sqrt(abs(q2))
         hsss(1) = cZero
         return
      endif
#endif

      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hsss(1) = dg * s1(1)*s2(1)*s3(1)
c
      return
      end
