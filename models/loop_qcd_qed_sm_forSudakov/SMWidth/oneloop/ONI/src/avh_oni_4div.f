************************************************************************
* This is the file  avh_oni_4div.f  of the package                     *
*                                                                      *
*                  Oneloop with Numerical Integration                  *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 15-12-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_oni_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_oni_d0m16( rslt ,p2,p3,p12,p23 ,m2,m3,m4 ,smax )
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                     d^(Dim)q
*  * ------ | ------------------------------------------------------
*  * i*pi^2 / q^2 [(q+k1)^2-m2] [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=m2, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
*  * m2,m4 should NOT be identically 0d0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,bt,b2,b3,b4,b0,zc3(0:2)
      double precision smax
      integer
     & avh_oni_un_get
*
c      write(*,*) 'MESSAGE from avh_oni_d0m16: you are calling me' !DEBUG
*
      if (p12.eq.m3) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m16: p12=m3, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
*
      m1 = dcmplx(0d0)
      p1 = m2
      p4 = m4
      bt = dcmplx(1d0)
      b2 = p12-p2+m2
      b3 = 2*p12
      b4 = p12-p3+m4
      b0 = dcmplx(0d0)
      call avh_oni_dfam( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &                  ,smax ,b0,b2,b3,b4,bt )
!      call avh_oni_c0c( zc1 ,p23,p2,p3 ,m4,m2,m3 )
!      call avh_oni_c0c( zc2 ,p12,p3,p4 ,m1,m3,m4 )
      call avh_oni_c0c( zc3 ,p1,p23,p4 ,m1,m2,m4 )
!      call avh_oni_c0c( zc4 ,p1,p2,p12 ,m1,m2,m3 )
      rslt(0) = ( zc3(0)-rslt(0) )/(p12-m3)
      rslt(1) = ( zc3(1)-rslt(1) )/(p12-m3)
      rslt(2) = ( zc3(2)-rslt(2) )/(p12-m3)
      end


      subroutine avh_oni_d0m14(rslt ,p12,p23 ,m2,m4 ,smax)
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                  d^(Dim)q
*  * ------ | -------------------------------------------------
*  * i*pi^2 / q^2 [(q+k1)^2-m2] (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=m2, k2^2=m2, k3^2=m4, (k1+k2+k3)^2=m4
*  * m2,m4 should NOT be identically 0d0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,bt,b2,b3,b4,b0,zc1(0:2),zc3(0:2)
      double precision smax
      integer
     & avh_oni_un_get
*
c      write(*,*) 'MESSAGE from avh_oni_d0m14: you are calling me' !DEBUG
*
      if (p12.eq.dcmplx(0d0)) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m15: p12=0, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
*
      m1 = dcmplx(0d0)
      m3 = m1
      p1 = m2
      p2 = m2
      p3 = m4
      p4 = m4
      bt = dcmplx(2d0)
      b2 = p12
      b3 = 2*p12
      b4 = p12
      b0 = dcmplx(0d0)
      call avh_oni_dfam( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &                  ,smax ,b0,b2,b3,b4,bt )
      call avh_oni_c0c( zc1 ,p23,p2,p3 ,m4,m2,m3 )
!      call avh_oni_c0c( zc2 ,p12,p3,p4 ,m1,m3,m4 )
      call avh_oni_c0c( zc3 ,p1,p23,p4 ,m1,m2,m4 )
!      call avh_oni_c0c( zc4 ,p1,p2,p12 ,m1,m2,m3 )
      rslt(0) = ( zc1(0)+zc3(0)-rslt(0) )/p12
      rslt(1) = ( zc1(1)+zc3(1)-rslt(1) )/p12
      rslt(2) = ( zc1(2)+zc3(2)-rslt(2) )/p12
      end


      subroutine avh_oni_d0m13(rslt ,p2,p3,p4,p12,p23 ,m3,m4 ,smax)
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                  d^(Dim)q
*  * ------ | -------------------------------------------------
*  * i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=p4
*  * m3,m4 should NOT be identically 0d0
*  * p4 should NOT be identical to m4
*  * p2 should NOT be identical to m3
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,bt,b2,b3,b4,b0,zc4(0:2),zc3(0:2),det
      double precision smax
      integer
     & avh_oni_un_get
*
c      write(*,*) 'MESSAGE from avh_oni_d0m13: you are calling me' !DEBUG
*
      if (p12.eq.m3) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m13: p12=m3, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
      if (p23.eq.m4) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m13: p23=m4, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
*
      m1 = dcmplx(0d0)
      m2 = dcmplx(0d0)
      p1 = dcmplx(0d0)
      bt = +p12+p23-p4-p2
      b2 = dcmplx(0d0)
      b3 = p12*(p12+2*p23-p4-p3-p2) + p2*(p3-p4)
      b4 = p4*(p12+p23+p3-p4-2*p2) + p23*(p12-p3)
      b0 = dcmplx(0d0)
      det = (p12-m3)*(p23-m4) - (p2-m3)*(p4-m4)
      call avh_oni_dfam( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &                  ,smax ,b0,b2,b3,b4,bt )
!      call avh_oni_c0c( zc1 ,p23,p2,p3 ,m4,m2,m3 )
!      call avh_oni_c0c( zc2 ,p12,p3,p4 ,m1,m3,m4 )
      call avh_oni_c0c( zc3 ,p1,p23,p4 ,m1,m2,m4 )
      call avh_oni_c0c( zc4 ,p1,p2,p12 ,m1,m2,m3 )
      rslt(0) = ( (p23-p4)*zc3(0) + (p12-p2)*zc4(0) - rslt(0) )/det
      rslt(1) = ( (p23-p4)*zc3(1) + (p12-p2)*zc4(1) - rslt(1) )/det
      rslt(2) = ( (p23-p4)*zc3(2) + (p12-p2)*zc4(2) - rslt(2) )/det
      end


      subroutine avh_oni_d0m8(rslt ,p3,p4,p12,p23 ,m4 ,smax)
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | --------------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=k2^2=0, k3^2=p3, (k1+k2+k3)^2=p4
*  * mm should NOT be identically 0d0
*  * p3 NOR p4 should be identically m4
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,bt,b2,b3,b4,b0,zc1(0:2),zc4(0:2),zc3(0:2),det
      double precision smax
      integer
     & avh_oni_un_get
*
c      write(6,*) 'MESSAGE from avh_oni_d0m8: you are calling me'
*
      if (p12.eq.dcmplx(0d0)) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m8: p12=0, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
      if (p23.eq.m4) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_d0m8: p23=mm, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
*
      m1 = dcmplx(0d0)
      m2 = dcmplx(0d0)
      m3 = dcmplx(0d0)
      p1 = dcmplx(0d0)
      p2 = dcmplx(0d0)
      bt = p12+2*p23-p4-p3
      b2 = dcmplx(0d0)
      b3 = p12*(p12+2*p23-p4-p3)
      b4 = p4*(p12+p23+p3-p4) + p23*(p12-p3)
      b0 = dcmplx(0d0)
      det = p12*(p23-m4)
      call avh_oni_dfam( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &                  ,smax ,b0,b2,b3,b4,bt )
      call avh_oni_c0c( zc1 ,p23,p2,p3 ,m4,m2,m3 )
!      call avh_oni_c0c( zc2 ,p12,p3,p4 ,m1,m3,m4 )
      call avh_oni_c0c( zc3 ,p1,p23,p4 ,m1,m2,m4 )
      call avh_oni_c0c( zc4 ,p1,p2,p12 ,m1,m2,m3 )
      rslt(0) = ((p23-p3)*zc1(0)+(p23-p4)*zc3(0)+p12*zc4(0)-rslt(0))/det
      rslt(1) = ((p23-p3)*zc1(1)+(p23-p4)*zc3(1)+p12*zc4(1)-rslt(1))/det
      rslt(2) = ((p23-p3)*zc1(2)+(p23-p4)*zc3(2)+p12*zc4(2)-rslt(2))/det
      end


      subroutine avh_oni_d0m3( rslt ,p2,p4,p12,p23 ,smax )
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | ---------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
*  *
*  * with  k1^2=k3^2=0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,bt,b2,b3,b4,b0,zc1(0:2),zc2(0:2),zc3(0:2),zc4(0:2),det
      double precision smax
*
      m1 = dcmplx(0d0)
      m2 = dcmplx(0d0)
      m3 = dcmplx(0d0)
      m4 = dcmplx(0d0)
      p1 = dcmplx(0d0)
      p3 = dcmplx(0d0)
      bt = 2*( p12+p23 -p2-p4 )
      b2 = dcmplx(0d0)
      b3 = 2*p12*(p12+p23-p2-p4)
      b4 = 2*p4*(p12+p23-p2-p4)
      b0 = dcmplx(0d0)
      det = p12*p23-p2*p4
      call avh_oni_dfam( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &                  ,smax ,b0,b2,b3,b4,bt )
      call avh_oni_c0c( zc1 ,p23,p2,p3 ,m4,m2,m3 )
      call avh_oni_c0c( zc2 ,p12,p3,p4 ,m1,m3,m4 )
      call avh_oni_c0c( zc3 ,p1,p23,p4 ,m1,m2,m4 )
      call avh_oni_c0c( zc4 ,p1,p2,p12 ,m1,m2,m3 )
      rslt(0) = (  (p23-p2)*zc1(0) + (p12-p4)*zc2(0)
     &           + (p23-p4)*zc3(0) + (p12-p2)*zc4(0) - rslt(0) )/det
      rslt(1) = (  (p23-p2)*zc1(1) + (p12-p4)*zc2(1)
     &           + (p23-p4)*zc3(1) + (p12-p2)*zc4(1) - rslt(1) )/det
      rslt(2) = (  (p23-p2)*zc1(2) + (p12-p4)*zc2(2)
     &           + (p23-p4)*zc3(2) + (p12-p2)*zc4(2) - rslt(2) )/det
      end


      subroutine avh_oni_d0( vald0 ,p1,p2,p3,p4,p12,p23 )
*  ********************************************************************
*  * calculates
*  *               C   /              d^(Dim)q
*  *            ------ | ---------------------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
*  *
*  * The input values (p1,p2,p3,p4,p12,p23) should be real.
*  * If any of these numbers is IDENTICALLY 0d0, the corresponding
*  * IR-singular case is returned.
*  ********************************************************************
      implicit none
      double complex vald0(0:2) ,p1,p2,p3,p4,p12,p23
     &,pp(6),zero,one
      parameter( zero=(0d0,0d0) ,one=(0d0,0d0) )
      double precision
     & gg,ff,hh,smax,small,ap(6)
     &,avh_oni_thrs
      integer
     & base(4),icase,ii,per(6),imax
     &,avh_oni_un_get
      character(3) label(6)
      logical avh_oni_os_get
      data base/8,4,2,1/,label/'p1','p2','p3','p4','p12','p23'/
*
      ff = cdabs(p12*p23) !CDABSyes
      gg = cdabs(p2*p4) !CDABSyes
      hh = cdabs(p1*p3) !CDABSyes
      if (ff.ge.gg.and.ff.ge.hh) then
        pp(1) = p1
        pp(2) = p2
        pp(3) = p3
        pp(4) = p4
        pp(5) = p12
        pp(6) = p23
      elseif (gg.ge.ff.and.gg.ge.hh) then
        pp(1) = p1
        pp(2) = p23
        pp(3) = p3
        pp(4) = p12
        pp(5) = p4
        pp(6) = p2
      else
        pp(1) = p12
        pp(2) = p2
        pp(3) = p23
        pp(4) = p4
        pp(5) = p1
        pp(6) = p3
      endif
      ap(1) = cdabs(pp(1)) !CDABSyes
      smax  = ap(1)
      imax = 1
      do ii=2,6
        ap(ii) = cdabs(pp(ii)) !CDABSyes
        if (ap(ii).gt.smax) then
          smax   = ap(ii)
          imax = ii
        endif
      enddo
      small = avh_oni_thrs(smax)
*
      if (avh_oni_os_get()) then
        if (ap(1).lt.small) ap(1) = 0d0
        if (ap(2).lt.small) ap(2) = 0d0
        if (ap(3).lt.small) ap(3) = 0d0
        if (ap(4).lt.small) ap(4) = 0d0
      endif
*
      icase = 0
      do ii=1,4
      if (ap(ii).gt.0d0) then
        icase = icase + base(ii)
        if (ap(ii).lt.small.and.avh_oni_un_get().gt.0)
     &    write(avh_oni_un_get(),*)
     &    'WARNING from avh_oni_d0: |',label(ii),'/',label(imax),'| ='
     &   ,ap(ii)/smax
      endif
      enddo
      call avh_oni_d0per(icase,per)
*
      if     (icase.eq.0) then
* 0 masses non-zero
        call avh_oni_d0m3( vald0 ,zero      ,zero
     &                           ,pp(per(5)),pp(per(6))
     &                           ,smax )
      elseif (icase.eq.1) then
* 1 mass non-zero
        call avh_oni_d0m3( vald0 ,zero      ,pp(per(4))
     &                           ,pp(per(5)),pp(per(6))
     &                           ,smax )
      elseif (icase.eq.5) then
* 2 opposite masses non-zero
        call avh_oni_d0m3( vald0 ,pp(per(2)),pp(per(4))
     &                           ,pp(per(5)),pp(per(6))
     &                           ,smax )
      elseif (icase.eq.2) then
* 2 neighbour masses non-zero
        call avh_oni_d0m8( vald0 ,pp(per(3)),pp(per(4))
     &                           ,pp(per(5)),pp(per(6))
     &                           ,zero ,smax )
      elseif (icase.eq.3) then
* 3 masses non-zero
        call avh_oni_d0m13( vald0 ,pp(per(2)),pp(per(3)),pp(per(4))
     &                            ,pp(per(5)),pp(per(6))
     &                            ,zero,zero ,smax )
      elseif (icase.eq.4) then
* 4 masses non-zero
        call avh_oni_dfam(vald0 ,pp(per(1)),pp(per(2))
     &                          ,pp(per(3)),pp(per(4))
     &                          ,pp(per(5)),pp(per(6))
     &                          ,zero,zero,zero,zero
     &                          ,smax ,one,zero,zero,zero,zero )
      endif
*
      end

     
      subroutine avh_oni_d0per(icase,per)
*  ********************************************************************
*  * Go through all possibilities of zero (0) and non-zero (1) masses
*  *
*  *   mass: 1234     mass: 1234     mass: 1234     mass: 1234
*  * icase=1 0001  icase= 3 0011  icase= 7 0111  icase= 0 0000 icase->0
*  * icase=2 0010  icase= 6 0110  icase=14 1110  icase=15 1111 icase->4 
*  * icase=4 0100  icase=12 1100  icase=13 1101  icase= 5 0101 icase->5
*  * icase=8 1000  icase= 9 1001  icase=11 1011  icase=10 1010 icase->5
*  *   icase->1      icase->2       icase->3
*  ********************************************************************
      implicit none
      integer icase,per(6)
     &,permtable(6,0:15),casetable(0:15),ii
      data permtable/
     & 1,2,3,4 ,5,6 ! 0, 0 masses non-zero,           no perm
     &,1,2,3,4 ,5,6 ! 1, 1 mass non-zero,             no perm
     &,4,1,2,3 ,6,5 ! 2, 1 mass non-zero,             1 cyclic perm
     &,1,2,3,4 ,5,6 ! 3, 2 neighbour masses non-zero, no perm
     &,3,4,1,2 ,5,6 ! 4, 1 mass   non-zero,           2 cyclic perm's
     &,1,2,3,4 ,5,6 ! 5, 2 opposite masses non-zero,  no perm
     &,4,1,2,3 ,6,5 ! 6, 2 neighbour masses non-zero, 1 cyclic perm
     &,1,2,3,4 ,5,6 ! 7, 3 masses non-zero,           no perm
     &,2,3,4,1 ,6,5 ! 8, 1 mass   non-zero,           3 cyclic perm's
     &,2,3,4,1 ,6,5 ! 9, 2 neighbour masses non-zero, 3 cyclic perm's
     &,4,1,2,3 ,6,5 !10, 2 opposite masses non-zero,  1 cyclic perm
     &,2,3,4,1 ,6,5 !11, 3 masses non-zero,           3 cyclic perm's
     &,3,4,1,2 ,5,6 !12, 2 neighbour masses non-zero, 2 cyclic perm's
     &,3,4,1,2 ,5,6 !13, 3 masses non-zero,           2 cyclic perm's
     &,4,1,2,3 ,6,5 !14, 3 masses non-zero,           1 cyclic perm
     &,1,2,3,4 ,5,6 !15, 4 masses non-zero,           no perm
     &/             ! 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
      data casetable/ 0, 1, 1, 2, 1, 5, 2, 3, 1, 2, 5, 3, 2, 3, 3, 4/
      do ii=1,6
        per(ii) = permtable(ii,icase)
      enddo
      icase = casetable(icase)
      end
