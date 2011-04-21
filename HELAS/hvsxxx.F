      subroutine hvsxxx(vc,sc,gc,smass,swidth , hvs)
c
c This subroutine computes an off-shell scalar current from the vector-
c scalar-scalar coupling.  The coupling is absent in the minimal SM in
c unitary gauge.
c
c input:
c       complex vc(6)          : input vector                          v
c       complex sc(3)          : input scalar                          s
c       complex gc             : coupling constant (s charge)
c       real    smass          : mass  of OUTPUT scalar s'
c       real    swidth         : width of OUTPUT scalar s'
c
c examples of the coupling constant gc for susy particles are as follows:
c   -----------------------------------------------------------
c   |    s1    | (q,i3) of s1  ||   v=A   |   v=Z   |   v=W   |
c   -----------------------------------------------------------
c   | nu~_l    | (  0  , +1/2) ||   ---   |  gzn(1) |  gwf(1) |
c   | e~_l     | ( -1  , -1/2) ||  gal(1) |  gzl(1) |  gwf(1) |
c   | u~_l     | (+2/3 , +1/2) ||  gau(1) |  gzu(1) |  gwf(1) |
c   | d~_l     | (-1/3 , -1/2) ||  gad(1) |  gzd(1) |  gwf(1) |
c   -----------------------------------------------------------
c   | e~_r-bar | ( +1  ,  0  ) || -gal(2) | -gzl(2) | -gwf(2) |
c   | u~_r-bar | (-2/3 ,  0  ) || -gau(2) | -gzu(2) | -gwf(2) |
c   | d~_r-bar | (+1/3 ,  0  ) || -gad(2) | -gzd(2) | -gwf(2) |
c   -----------------------------------------------------------
c where the sc charge is defined by the flowing-OUT quantum number.
c
c output:
c       complex hvs(3)         : scalar current                j(s':v,s)
c     
      implicit none
      double complex vc(6),sc(3),hvs(3),dg,qvv,qpv,gc
      double precision qv(0:3),qp(0:3),qa(0:3),smass,swidth,q2

      double precision rTwo
      parameter( rTwo = 2.0d0 )

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
      if ( abs(vc(1))+abs(vc(2))+abs(vc(3))+abs(vc(4)).eq.rZero ) then
         write(stdo,*) ' helas-warn  : vc in hvsxxx is zero vector'
      endif
      if ( abs(vc(5))+abs(vc(6)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : vc in hvsxxx has zero momentum'
      endif
      if ( sc(1).eq.cZero ) then
         write(stdo,*) ' helas-warn  : sc in hvsxxx is zero scalar'
      endif
      if ( abs(sc(2))+abs(sc(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : sc in hvsxxx has zero momentum'
      endif
      if ( gc.eq.cZero ) then
         write(stdo,*) ' helas-error : gc in hvsxxx is zero coupling'
      endif
      if ( smass.lt.rZero ) then
         write(stdo,*) ' helas-error : smass in hvsxxx is negative'
         write(stdo,*) '             : smass = ',smass
      endif
      if ( swidth.lt.rZero ) then
         write(stdo,*) ' helas-error : swidth in hvsxxx is negative'
         write(stdo,*) '             : swidth = ',swidth
      endif
#endif

      hvs(2) = vc(5)+sc(2)
      hvs(3) = vc(6)+sc(3)

      qv(0) = dble(  vc(5))
      qv(1) = dble(  vc(6))
      qv(2) = dimag( vc(6))
      qv(3) = dimag( vc(5))
      qp(0) = dble(  sc(2))
      qp(1) = dble(  sc(3))
      qp(2) = dimag( sc(3))
      qp(3) = dimag( sc(2))
      qa(0) = dble( hvs(2))
      qa(1) = dble( hvs(3))
      qa(2) = dimag(hvs(3))
      qa(3) = dimag(hvs(2))
      q2 = qa(0)**2-(qa(1)**2+qa(2)**2+qa(3)**2)

#ifdef HELAS_CHECK
      if ( abs(hvs(2))+abs(hvs(3)).eq.rZero ) then
         write(stdo,*)
     &        ' helas-error : hvs in hvsxxx has zero momentum'
      endif
      if ( swidth.eq.rZero .and. q2.eq.smass**2 ) then
         write(stdo,*)
     &        ' helas-error : hvs in hvsxxx is on smass pole'
         write(stdo,*) 
     &        '             : q     = ',qa(0),qa(1),qa(2),qa(3)
         write(stdo,*) 
     &        '             : abs(q)= ',sqrt(abs(q2))
         hvs(1) = cZero
         return
      endif
#endif

      dg = -gc/dcmplx( q2-smass**2, smass*swidth )
      qvv = qv(0)*vc(1)-qv(1)*vc(2)-qv(2)*vc(3)-qv(3)*vc(4)
      qpv = qp(0)*vc(1)-qp(1)*vc(2)-qp(2)*vc(3)-qp(3)*vc(4)

      hvs(1) = dg*(rTwo*qpv+qvv)*sc(1)
c
      return
      end
