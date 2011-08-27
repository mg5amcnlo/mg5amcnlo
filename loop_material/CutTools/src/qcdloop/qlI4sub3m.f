      subroutine qlI4sub3m(xpi,musq,Ival)
      implicit none
C Calculates divergent integral with three mass
C I_4^{\{D=4-2 \e\}}(m_2^2,\pd^2,\pt^2,m_4^2;s_{12},s_{23};0,m_2^2,m_3^2,m_4^2)
C as well as finite integrals using ff

C     Uses the ordering for the routine xpi wanted by FF
C     psq(1) lies between msq(1) and msq(2) and so on
C     xpi(1-4) = msq(1),msq(2),msq(3),msq(4)
C     xpi(5-8) = psq(1),psq(2),psq(3),psq(4)
C     xpi(9-10) = s12,s23
C     xpi(11) = +xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
C     xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
C     xpi(13) = +xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
      include 'qlconstants.f'
      double precision xpi(13),musq,
     . xpo(13),Y(4,4),Yalt(4,4)
      double complex Ival(-2:0)
      integer j,jsort,swap(13,4),ier,Npt
      logical qlzero
      parameter(Npt=13)
      data swap/
     .      1,2,3,4,5,6,7,8,9,10,11,12,13,
     .      4,1,2,3,8,5,6,7,10,9,11,13,12,
     .      3,4,1,2,7,8,5,6,9,10,11,12,13,
     .      2,3,4,1,6,7,8,5,10,9,11,13,12
     . /

      save swap 

      call qlxpicheck(xpi)

      jsort=0
      do j=1,4
      if (qlzero(xpi(j))) jsort=j
      enddo

      do j=1,Npt
      xpo(swap(j,jsort))=xpi(j)
      enddo

      Ival(-2)=czip
      Ival(-1)=czip

      call qlYcalc(xpo,Y,Yalt) 

C---  divergent three mass box 
      if     ((qlzero(Y(1,1)))
     .   .and.(qlzero(Y(1,2)))
     .   .and.(qlzero(Y(1,4)))) then
      call qlbox16(Y,musq,Ival)
C---  finite three mass box 
      else
      xpo(11) = +xpo(5)+xpo(6)+xpo(7)+xpo(8)-xpo(9)-xpo(10)
      xpo(12) = -xpo(5)+xpo(6)-xpo(7)+xpo(8)+xpo(9)+xpo(10)
      xpo(13) = +xpo(5)-xpo(6)+xpo(7)-xpo(8)+xpo(9)+xpo(10)
      call qlI4fin(Ival(0),xpo,ier)
      endif
      return
      end
