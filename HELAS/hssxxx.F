      subroutine hssxxx(s1,s2,gc,smass,swidth , hss)
c
c This subroutine computes an off-shell scalar current from the three-
c scalar coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gc             : coupling constant                  ghhh
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c output:
c       complex hss(3)         : scalar current              j(s':s1,s2)
c     
      implicit none
      double complex s1(3),s2(3),gc,hss(3),dg
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
         write(stdo,*) ' helas-warn  : s1 in hssxxx is zero scalar'
      endif
      if ( abs(s1(2))+abs(s1(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s1 in hssxxx has zero momentum'
      endif
      if ( s2(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : s2 in hssxxx is zero scalar'
      endif
      if ( abs(s2(2))+abs(s2(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : s2 in hssxxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in hssxxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hssxxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hssxxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hss(2) = s1(2)+s2(2)
      hss(3) = s1(3)+s2(3)

      q(0) = dble( hss(2))
      q(1) = dble( hss(3))
      q(2) = dimag(hss(3))
      q(3) = dimag(hss(2))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hss(2))+abs(hss(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hss in hssxxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hss in hssxxx is on smass pole'
         write(stdo,*) 
     &        '             : q     = ',q(0),q(1),q(2),q(3)
         write(stdo,*)
     &        '             : abs(q)= ',sqrt(abs(q2))
         hss(1)=cmplx(rZero)
         return
      endif
#endif

      dg = -gc/dcmplx( q2-smass**2, smass*swidth )

      hss(1) = dg*s1(1)*s2(1)
c
      return
      end
