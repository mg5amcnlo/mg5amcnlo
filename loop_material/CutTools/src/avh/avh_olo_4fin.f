************************************************************************
* This is the file  avh_olo_4fin.f  of the package                     *
*                                                                      *
*                               OneLOop                                *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 28-07-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_olo_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_olo_cdm0( rslt ,p1,p2,p3,p4,p12,p23 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with all internal masses
*  * equal zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23
     &,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,qz1,qz2,zero,half,four ,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,avh_olo_sqrt ,avh_olo_li2c2,avh_olo_logc2,avh_olo_logc
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),four=(4d0,0d0))
      double precision
     & hh
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,iz1,iz2 ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_cdm0: you are calling me' !DEBUG
*
      r12 = -p1  !  p1
      r13 = -p12 !  p1+p2
      r14 = -p4  !  p1+p2+p3
      r23 = -p2  !  p2
      r24 = -p23 !  p2+p3
      r34 = -p3  !  p3      
*
      aa = r34*r24
*
      if (r13.eq.zero.or.aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm0: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = r13*r24 + r12*r34 - r14*r23
      cc = r12*r13
      hh = dreal(r23)
      dd = avh_olo_sqrt( bb*bb - four*aa*cc , -dreal(aa)*hh )
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,1)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 , hh)
      call avh_olo_conv( qx2,ix2 ,x2 ,-hh)
      call avh_olo_conv( q12,i12 ,r12,-1d0)
      call avh_olo_conv( q13,i13 ,r13,-1d0)
      call avh_olo_conv( q14,i14 ,r14,-1d0)
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
      rslt(0) = zero
*
      call avh_olo_rat( qss,iss ,q34,i34 ,q13,i13 )
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 / r13
      rslt(0) = rslt(0) + trm
*
      call avh_olo_rat( qss,iss ,q24,i24 ,q12,i12 )
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r24 / r12
      rslt(0) = rslt(0) + trm
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
      call avh_olo_prd( qz2,iz2 ,qx1,ix1 ,qx2,ix2 )
      ss = -avh_olo_logc2( qz1,iz1 ) / x2
      trm = ss * avh_olo_logc( qz2,iz2 ) * half
      rslt(0) = rslt(0) + trm
*
      call avh_olo_prd( qz2,iz2 ,q12,i12 ,q13,i13 )
      call avh_olo_div( qz2,iz2 ,q14,i14 )
      call avh_olo_div( qz2,iz2 ,q23,i23 )
      trm = -ss * avh_olo_logc( qz2,iz2 )
      rslt(0) = rslt(0) + trm
*
      rslt(0) = -rslt(0) / aa
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_cdm1(rslt ,p1,p2,p3,p4,p12,p23 ,m4)
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with one internal mass
*  * non-zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m4
     &,smm,sm4,aa,bb,cc,dd,x1,qx1,x2,qx2,qss,qz1,qz2,zero,half,four
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,trm,oieps ,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc2,avh_olo_logc
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,iz1,iz2 ,avh_olo_un_get
      logical
     & r12zero,r13zero
*
c      write(6,*) 'MESSAGE from avh_olo_cdm1: you are calling me' !DEBUG
*
      sm4 = avh_olo_sqrt(m4,-1d0)
      smm = dcmplx(cdabs(sm4),0d0)
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      r12 = zero
      r13 = zero
      r14 = zero
      r23 = zero
      r24 = zero
      r34 = zero
      if (m4.ne.p4 ) r12 = ( m4-p4 *oieps )/(smm*sm4)
      if (m4.ne.p23) r13 = ( m4-p23*oieps )/(smm*sm4)
      if (m4.ne.p3 ) r14 = ( m4-p3 *oieps )/(smm*sm4)
                     r23 = (   -p1 *oieps )/(smm*smm)
                     r24 = (   -p12*oieps )/(smm*smm)
                     r34 = (   -p2 *oieps )/(smm*smm)
*
      r12zero = (r12.eq.zero)
      r13zero = (r13.eq.zero)
*
      aa = r34*r24
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm1: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = r13*r24 + r12*r34 - r14*r23
      cc = r12*r13 - r23
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 )
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 )
      call avh_olo_conv( q12,i12 ,r12,-1d0)
      call avh_olo_conv( q13,i13 ,r13,-1d0)
      call avh_olo_conv( q14,i14 ,r14,-1d0)
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
      rslt(0) = zero
*
      if (r12zero.and.r13zero) then
        call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_prd( qss,iss ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_mlt( qss,iss ,q34,i34 )
        call avh_olo_mlt( qss,iss ,q24,i24 )
        call avh_olo_div( qss,iss ,q23,i23 )
        call avh_olo_prd( qz2,iz2 ,qss,iss ,qss,iss)
        trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
        rslt(0) = rslt(0) + trm
      else
        if (r13zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qss,iss ,q34,i34 ,q12,i12 )
          call avh_olo_div( qss,iss ,q23,i23 )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        else
          call avh_olo_rat( qss,iss ,q34,i34 ,q13,i13 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 / r13
          rslt(0) = rslt(0) + trm
        endif
        if (r12zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qss,iss ,q24,i24 ,q13,i13 )
          call avh_olo_div( qss,iss ,q23,i23 )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        else
          call avh_olo_rat( qss,iss ,q24,i24 ,q12,i12 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r24 / r12
          rslt(0) = rslt(0) + trm
        endif
        if (.not.r12zero.and..not.r13zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,q12,i12 ,q13,i13 )
          call avh_olo_div( qz2,iz2 ,q23,i23 )
          trm = avh_olo_logc2( qz1,iz1 ) / x2 * avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        endif
      endif
*
      if (r14.ne.zero) then
        call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,q14,i14 )
        call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,q14,i14 )
        trm = -avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r14
        rslt(0) = rslt(0) + trm
      endif
*
      rslt(0) = -rslt(0) / aa / (smm*smm*smm*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_cdm5(rslt ,p1,p2,p3,p4,p12,p23, m2,m4)
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with two opposite internal
*  * masses non-zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m2,m4
      call avh_olo_cdm2(rslt ,p12,p2,p23,p4,p1,p3 ,m2,m4)
      end


      subroutine avh_olo_cdm2(rslt ,p1,p2,p3,p4,p12,p23 ,m3,m4)
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with two adjacent internal
*  * masses non-zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m3,m4
     &,smm,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,qss,trm,qz1,qz2
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d14,zero,half,four,oieps
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc2,avh_olo_logc
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,iz1,iz2 ,avh_olo_un_get
      logical
     & r12zero,r13zero,r24zero,r34zero
*
c      write(6,*) 'MESSAGE from avh_olo_cdm2: you are calling me' !DEBUG
*
      sm3 = avh_olo_sqrt(m3,-1d0)
      sm4 = avh_olo_sqrt(m4,-1d0)
*
      smm = dcmplx(cdabs(sm3),0d0)
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      r12 = zero
      r13 = zero
      q14 = zero
      r23 = zero
      r24 = zero
      r34 = zero
      if (   m4.ne.p4 ) r12 = (    m4-p4 *oieps )/(smm*sm4)
      if (   m4.ne.p23) r13 = (    m4-p23*oieps )/(smm*sm4)
      if (m3+m4.ne.p3 ) q14 = ( m3+m4-p3 *oieps )/(sm3*sm4)
                        r23 = (      -p1 *oieps )/(smm*smm)
      if (   m3.ne.p12) r24 = (    m3-p12*oieps )/(smm*sm3)
      if (   m3.ne.p2 ) r34 = (    m3-p2 *oieps )/(smm*sm3)
*
      r12zero = (r12.eq.zero)
      r13zero = (r13.eq.zero)
      r24zero = (r24.eq.zero)
      r34zero = (r34.eq.zero)
      if (r12zero.and.r24zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm2: m4=p4 and m3=p12, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (r13zero.and.r34zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm2: m4=p23 and m3=p2, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      call avh_olo_rfun( r14,d14 ,q14 )
*
      aa = r34*r24 - r23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm2: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = r13*r24 + r12*r34 - q14*r23
      cc = r12*r13 - r23
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 )
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 )
      call avh_olo_conv( q12,i12 ,r12,-1d0)
      call avh_olo_conv( q13,i13 ,r13,-1d0)
      call avh_olo_conv( q14,i14 ,r14,-1d0)
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,q14,i14 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,q14,i14 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r14
      rslt(0) = -trm
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,q14,i14 )
      call avh_olo_rat( qz2,iz2 ,qx2,ix2 ,q14,i14 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) / r14
      rslt(0) = rslt(0) - trm
*
      if (r12zero.and.r13zero) then
        call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_prd( qss,iss ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_mlt( qss,iss ,q34,i34 )
        call avh_olo_mlt( qss,iss ,q24,i24 )
        call avh_olo_div( qss,iss ,q23,i23 )
        call avh_olo_prd( qz2,iz2 ,qss,iss ,qss,iss)
        trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
        rslt(0) = rslt(0) + trm
      else
        if (r13zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qss,iss ,q34,i34 ,q12,i12 )
          call avh_olo_div( qss,iss ,q23,i23 )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        elseif (.not.r34zero) then
          call avh_olo_rat( qss,iss ,q34,i34 ,q13,i13 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 / r13
          rslt(0) = rslt(0) + trm
        endif
        if (r12zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qss,iss ,q24,i24 ,q13,i13 )
          call avh_olo_div( qss,iss ,q23,i23 )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          call avh_olo_mlt( qz2,iz2 ,qss,iss )
          trm = avh_olo_logc2( qz1,iz1 )/x2*half*avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        elseif (.not.r24zero) then
          call avh_olo_rat( qss,iss ,q24,i24 ,q12,i12 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r24 / r12
          rslt(0) = rslt(0) + trm
        endif
        if (.not.r12zero.and..not.r13zero) then
          call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
          call avh_olo_prd( qz2,iz2 ,q12,i12 ,q13,i13 )
          call avh_olo_div( qz2,iz2 ,q23,i23 )
          trm = avh_olo_logc2( qz1,iz1 ) / x2 * avh_olo_logc( qz2,iz2 )
          rslt(0) = rslt(0) + trm
        endif
      endif
*
      rslt(0) = -rslt(0) / (aa*smm*smm*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_cdm3(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with three internal masses
*  * non-zero.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4
     &,zero,p(6),m(4)
      parameter( zero=(0d0,0d0) )
      integer
     & j,k,ip3(6,6),im3(6,4),ip2(6,6),im2(6,4),ip1(6,6),im1(6,4)
     &,ip4(6,6),im4(6,4),ic(4,6)
*
      data ip4/1,2,5,1,5,2 ,2,1,1,5,2,5 ,3,4,6,3,6,4 ,4,3,3,6,4,6
     &        ,5,5,2,2,1,1 ,6,6,4,4,3,3/
      data im4/1,3,3,2,1,2 ,2,2,1,1,3,3 ,3,1,2,3,2,1 ,4,4,4,4,4,4/
*
      data ip3/1,4,6,1,4,6 ,2,3,3,5,5,2 ,3,2,5,3,2,5 ,4,1,1,6,6,4
     &        ,5,5,2,2,3,3 ,6,6,4,4,1,1/
      data im3/1,1,2,2,4,4 ,2,4,4,1,1,2 ,3,3,3,3,3,3 ,4,2,1,4,2,1/
*
      data ip2/1,2,1,6,2,6 ,2,1,6,1,6,2 ,3,4,3,5,4,5 ,4,3,5,3,5,4
     &        ,5,5,4,4,3,3 ,6,6,2,2,1,1/
      data im2/1,3,1,4,3,4 ,2,2,2,2,2,2 ,3,1,4,1,4,3 ,4,4,3,3,1,1/
*
      data ip1/1,4,1,5,5,4 ,2,3,6,3,2,6 ,3,2,3,6,6,2 ,4,1,5,1,4,5
     &        ,5,5,4,4,1,1 ,6,6,2,2,3,3/
      data im1/1,1,1,1,1,1, 2,4,2,3,3,4 ,3,3,4,4,2,2 ,4,2,3,2,4,3/
*
      data ic/1,2,3,4 ,2,3,4,1 ,3,4,1,2 ,4,1,2,3 ,5,6,5,6 ,6,5,6,5/
*
      p(1) = p1
      p(2) = p2
      p(3) = p3
      p(4) = p4
      p(5) = p12
      p(6) = p23
      m(1) = m1
      m(2) = m2
      m(3) = m3
      m(4) = m4
*
      if     (m1.eq.zero) then
        j = 3
      elseif (m2.eq.zero) then
        j = 4
      elseif (m3.eq.zero) then
        j = 1
      else
        j = 2
      endif
      k = 5
      call avh_olo_cdm33( rslt ,p(ic(j,ip3(k,1))),p(ic(j,ip3(k,2)))
     &                         ,p(ic(j,ip3(k,3))),p(ic(j,ip3(k,4)))
     &                         ,p(ic(j,ip3(k,5))),p(ic(j,ip3(k,6)))
     &       ,m(ic(j,im3(k,1))),m(ic(j,im3(k,2))),m(ic(j,im3(k,4))))
      end

      subroutine avh_olo_cdm31(rslt ,p1,p2,p3,p4,p12,p23 ,m2,m3,m4 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with all internal masses
*  * non-zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m2,m3,m4
     &,sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,tt,qtt,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d23,d24,d34
     &,qz1,qz2,qy1,qy2,zero,one,four,oieps ,avh_olo_sqrt,avh_olo_li2c2 
      parameter(zero=(0d0,0d0),one=(1d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,itt ,avh_olo_un_get
     &,iy1,iy2,iz1,iz2
*
c      write(6,*) 'MESSAGE from avh_olo_cdm31: you are calling me'
*
      sm2 = avh_olo_sqrt( m2 ,-1d0 )
      sm3 = avh_olo_sqrt( m3 ,-1d0 )
      sm4 = avh_olo_sqrt( m4 ,-1d0 )
      sm1 = dcmplx(cdabs(sm2))
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      r12 = zero
      r13 = zero
      r14 = zero
      q23 = zero
      q24 = zero
      q34 = zero
      if (   m2.ne.p1 ) r12 = (      m2 - p1 *oieps )/(sm1*sm2) ! p1
      if (   m3.ne.p12) r13 = (      m3 - p12*oieps )/(sm1*sm3) ! p1+p2
      if (   m4.ne.p4 ) r14 = (      m4 - p4 *oieps )/(sm1*sm4) ! p1+p2+p3
      if (m2+m3.ne.p2 ) q23 = ( m2 + m3 - p2 *oieps )/(sm2*sm3) ! p2
      if (m2+m4.ne.p23) q24 = ( m2 + m4 - p23*oieps )/(sm2*sm4) ! p2+p3
      if (m3+m4.ne.p3 ) q34 = ( m3 + m4 - p3 *oieps )/(sm3*sm4) ! p3
*
      call avh_olo_rfun( r23,d23 ,q23 )
      call avh_olo_rfun( r24,d24 ,q24 )
      call avh_olo_rfun( r34,d34 ,q34 )
*
      aa = r13*r12 - r14*r13/r24
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm4: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = r13*d24 + r12*q34 - r14*q23
      cc = r12/r13 + r24*q34 - r14*r24/r13 - q23
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 )
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 )
      call avh_olo_conv( q12,i12 ,r12,-1d0) ! RE-USING q12
      call avh_olo_conv( q13,i13 ,r13,-1d0) ! RE-USING q13
      call avh_olo_conv( q14,i14 ,r14,-1d0) ! RE-USING q14
      call avh_olo_conv( q23,i23 ,r23,-1d0) ! RE-USING q23
      call avh_olo_conv( q24,i24 ,r24,-1d0) ! RE-USING q24
      call avh_olo_conv( q34,i34 ,r34,-1d0) ! RE-USING q34
*
*
       ss = r12
      qss = q12
      iss = i12
       tt = one/r24
      qtt = one/q24
      itt = -i24
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qtt,itt )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qtt,itt )
      call avh_olo_prd( qy1,iy1 ,qz1,iz1 ,qss,iss )
      call avh_olo_prd( qy2,iy2 ,qz2,iz2 ,qss,iss )
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt * ss
      rslt(0) = trm
*
       ss = r23
      qss = q23
      iss = i23
      call avh_olo_rat( qtt,itt ,q13,i13 ,q24,i24 )
      if (itt.eq.itt/2*2) then
        tt = qtt
      else
        tt = -qtt
      endif
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
       ss = r34
      qss = q34
      iss = i34
       tt = r13
      qtt = q13
      itt = i13
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) + trm
*
       ss = r14
      qss = q14
      iss = i14
      call avh_olo_prd( qy1,iy1 ,qx1,ix1 ,qss,iss )
      call avh_olo_prd( qy2,iy2 ,qx2,ix2 ,qss,iss )
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * ss
      rslt(0) = rslt(0) - trm
*
      rslt(0) = -rslt(0) / (aa*sm1*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end

      subroutine avh_olo_cdm32(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m3,m4 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with three internal masses
*  * non-zero, and m2=0. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m3,m4
     &,sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,tt,qtt,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d13,d14,d24,d34
     &,qz1,qz2,zero,one,four,oieps
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2 
      parameter(zero=(0d0,0d0),one=(1d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,itt ,avh_olo_un_get
     &,iz1,iz2
      logical
     & r12zero,r23zero,r24zero
*
c      write(6,*) 'MESSAGE from avh_olo_cdm32: you are calling me'
*
      sm1 = avh_olo_sqrt( m1 ,-1d0 )
      sm3 = avh_olo_sqrt( m3 ,-1d0 )
      sm4 = avh_olo_sqrt( m4 ,-1d0 )
      sm2 = dcmplx(cdabs(sm1))
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      q12 = zero
      q13 = zero
      q14 = zero
      q23 = zero
      q24 = zero
      q34 = zero
      if (m1   .ne.p1 ) q12 = ( m1      - p1 *oieps )/(sm1*sm2) ! p1
      if (m1+m3.ne.p12) q13 = ( m1 + m3 - p12*oieps )/(sm1*sm3) ! p1+p2
      if (m1+m4.ne.p4 ) q14 = ( m1 + m4 - p4 *oieps )/(sm1*sm4) ! p1+p2+p3
      if (   m3.ne.p2 ) q23 = (      m3 - p2 *oieps )/(sm2*sm3) ! p2
      if (   m4.ne.p23) q24 = (      m4 - p23*oieps )/(sm2*sm4) ! p2+p3
      if (m3+m4.ne.p3 ) q34 = ( m3 + m4 - p3 *oieps )/(sm3*sm4) ! p3
*
      r12 = q12 !call avh_olo_rfun( r12,d12 ,q12 )
      r23 = q23 !call avh_olo_rfun( r23,d23 ,q23 )
      r24 = q24 !call avh_olo_rfun( r24,d24 ,q24 )
      d24 = r24
*
      r12zero = (r12.eq.zero)
      r23zero = (r23.eq.zero)
      r24zero = (r24.eq.zero)
      if (r24zero) then
        if     (r23zero) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cdm32: m4=p23 and m3=p2, returning 0'
          call avh_olo_zero(rslt)
          return
        elseif (r12zero) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cdm32: m4=p23 and m1=p1, returning 0'
          call avh_olo_zero(rslt)
          return
        endif
      endif
*
      call avh_olo_rfun( r13,d13 ,q13 )
      call avh_olo_rfun( r14,d14 ,q14 )
      call avh_olo_rfun( r34,d34 ,q34 )
*
      aa = r13*q12 - q23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm32: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = d13*d24 + q12*q34 - q14*q23
      cc = q12/r13 + r24*q34 - q14*r24/r13 - q23
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q12,i12 ,r12,-1d0) ! RE-USING q12
      call avh_olo_conv( q13,i13 ,r13,-1d0) ! RE-USING q13
      call avh_olo_conv( q14,i14 ,r14,-1d0) ! RE-USING q14
      call avh_olo_conv( q23,i23 ,r23,-1d0) ! RE-USING q23
      call avh_olo_conv( q24,i24 ,r24,-1d0) ! RE-USING q24
      call avh_olo_conv( q34,i34 ,r34,-1d0) ! RE-USING q34
*
      rslt(0) = zero
      if (.not.r24zero) then
        if (.not.r23zero) then
          call avh_olo_rat( qss,iss ,q13,i13 ,q24,i24 )
          call avh_olo_mlt( qss,iss ,q23,i23 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r23 * r13 / r24
          rslt(0) = rslt(0) - trm
        endif
        if (.not.r12zero) then
          call avh_olo_rat( qss,iss ,q12,i12 ,q24,i24 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r12 / r24
          rslt(0) = rslt(0) + trm
        endif
      else
        call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_prd( qz2,iz2 ,q13,i13 ,q23,i23 )
        call avh_olo_div( qz2,iz2 ,q12,i12 )
        trm = -avh_olo_logc2( qz1,iz1 )/x2 * avh_olo_logc( qz2,iz2 )
        rslt(0) = rslt(0) + trm
      endif
*
       ss = r34
      qss = q34
      iss = i34
       tt = r13
      qtt = q13
      itt = i13
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) + trm
*
       ss = r14
      qss = q14
      iss = i14
       tt = one
      qtt = one
      itt = 0
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
      rslt(0) = -rslt(0) / (aa*sm1*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end

      subroutine avh_olo_cdm33(rslt ,p1,p2,p3,p4,p12,p23, m1,m2,m4)
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with three internal masses
*  * non-zero, and m3=0. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m2,m4
     &,sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,qss,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d12,d14,d24
     &,qy1,qy2,qz1,qz2,zero,four,oieps
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2 
      parameter(zero=(0d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,iz1,iz2,iy1,iy2
     &,avh_olo_un_get
      logical
     & r13zero,r23zero,r34zero
*
c      write(6,*) 'MESSAGE from avh_olo_cdm33: you are calling me'
*
      sm1 = avh_olo_sqrt( m1 ,-1d0 )
      sm2 = avh_olo_sqrt( m2 ,-1d0 )
      sm4 = avh_olo_sqrt( m4 ,-1d0 )
      sm3 = dcmplx(cdabs(sm2))
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      q12 = zero
      r13 = zero
      q14 = zero
      r23 = zero
      q24 = zero
      r34 = zero
      if (m1+m2.ne.p1 ) q12 = ( m1 + m2 - p1 *oieps )/(sm1*sm2) ! p1
      if (m1   .ne.p12) r13 = ( m1      - p12*oieps )/(sm1*sm3) ! p1+p2
      if (m1+m4.ne.p4 ) q14 = ( m1 + m4 - p4 *oieps )/(sm1*sm4) ! p1+p2+p3
      if (m2   .ne.p2 ) r23 = ( m2      - p2 *oieps )/(sm2*sm3) ! p2
      if (m2+m4.ne.p23) q24 = ( m2 + m4 - p23*oieps )/(sm2*sm4) ! p2+p3
      if (   m4.ne.p3 ) r34 = (      m4 - p3 *oieps )/(sm3*sm4) ! p3
*
      r13zero = (r13.eq.zero)
      r23zero = (r23.eq.zero)
      r34zero = (r34.eq.zero)
      if (r13zero) then
        if     (r23zero) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cdm33: m4=p4 and m3=p12, returning 0'
          call avh_olo_zero(rslt)
          return
        elseif (r34zero) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_cdm33: m2=p1 and m3=p12, returning 0'
          call avh_olo_zero(rslt)
          return
        endif
      endif
*
      call avh_olo_rfun( r12,d12 ,q12 )
      call avh_olo_rfun( r14,d14 ,q14 )
      call avh_olo_rfun( r24,d24 ,q24 )
*
      aa = r34/r24 - r23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm33: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = -r13*d24 + q12*r34 - q14*r23
      cc = q12*r13 + r24*r34 - q14*r24*r13 - r23
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q12,i12 ,r12,-1d0)
      call avh_olo_conv( q13,i13 ,r13,-1d0)
      call avh_olo_conv( q14,i14 ,r14,-1d0)
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
*
      call avh_olo_rat( qy1,iy1 ,qx1,ix1 ,q24,i24 )
      call avh_olo_rat( qy2,iy2 ,qx2,ix2 ,q24,i24 )
*
      call avh_olo_prd( qz1,iz1 ,qy1,iy1 ,q12,i12 )
      call avh_olo_prd( qz2,iz2 ,qy2,iy2 ,q12,i12 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) / r24 * r12
      rslt(0) = trm
*
      call avh_olo_rat( qz1,iz1 ,qy1,iy1 ,q12,i12 )
      call avh_olo_rat( qz2,iz2 ,qy2,iy2 ,q12,i12 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) / r24 / r12
      rslt(0) = rslt(0) + trm
*
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,q14,i14 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,q14,i14 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r14
      rslt(0) = rslt(0) - trm
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,q14,i14 )
      call avh_olo_rat( qz2,iz2 ,qx2,ix2 ,q14,i14 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) / r14
      rslt(0) = rslt(0) - trm
*
      if (.not.r13zero) then
        if (.not.r23zero) then
          call avh_olo_rat( qss,iss ,q23,i23 ,q13,i13 )
          call avh_olo_div( qss,iss ,q24,i24 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r23 / (r13*r24)
          rslt(0) = rslt(0) - trm
        endif
        if (.not.r34zero) then
          call avh_olo_rat( qss,iss ,q34,i34 ,q13,i13 )
          call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
          call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
          trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 / r13
          rslt(0) = rslt(0) + trm
        endif
      else
        call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
        call avh_olo_rat( qz2,iz2 ,q23,i23 ,q24,i24 )
        call avh_olo_div( qz2,iz2 ,q34,i34 )
        trm = -avh_olo_logc2( qz1,iz1 )/x2 * avh_olo_logc( qz2,iz2 )
        rslt(0) = rslt(0) + trm
      endif
*
      rslt(0) = -rslt(0) / (aa*sm1*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end

      subroutine avh_olo_cdm34(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with the fourth internal 
*  * mass equal zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m2,m3
     &,sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,tt,qtt,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d12,d13,d23,d24
     &,qy1,qy2,qz1,qz2,zero,one,four,oieps
     &,avh_olo_sqrt ,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2
      parameter(zero=(0d0,0d0),one=(1d0,0d0),four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,itt ,avh_olo_un_get
     &,iy1,iy2,iz1,iz2
*
c      write(6,*) 'MESSAGE from avh_olo_cdm34: you are calling me'
*
      sm1 = avh_olo_sqrt( m1 ,-1d0 )
      sm2 = avh_olo_sqrt( m2 ,-1d0 )
      sm3 = avh_olo_sqrt( m3 ,-1d0 )
      sm4 = dcmplx(cdabs(sm1))
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      q12 = zero
      q13 = zero
      q14 = zero
      q23 = zero
      q24 = zero
      q34 = zero
      if (m1+m2.ne.p1 ) q12 = ( m1 + m2 - p1 *oieps )/(sm1*sm2) ! p1
      if (m1+m3.ne.p12) q13 = ( m1 + m3 - p12*oieps )/(sm1*sm3) ! p1+p2
      if (m1   .ne.p4 ) q14 = ( m1      - p4 *oieps )/(sm1*sm4) ! p1+p2+p3
      if (m2+m3.ne.p2 ) q23 = ( m2 + m3 - p2 *oieps )/(sm2*sm3) ! p2
      if (m2   .ne.p23) q24 = ( m2      - p23*oieps )/(sm2*sm4) ! p2+p3
      if (m3   .ne.p3 ) q34 = ( m3      - p3 *oieps )/(sm3*sm4) ! p3
*
      call avh_olo_rfun( r12,d12 ,q12 )
      call avh_olo_rfun( r13,d13 ,q13 )
      call avh_olo_rfun( r23,d23 ,q23 )
      r14 = q14
      r24 = q24
      r34 = q34
      d24 = r24
*
      aa = q34/r24 + r13*q12 - q14*r13/r24 - q23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm4: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = d13*d24 + q12*q34 - q14*q23
      cc = r24*q34 - q14*r24/r13
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q12,i12 ,r12,-1d0) ! RE-USING q12
      call avh_olo_conv( q13,i13 ,r13,-1d0) ! RE-USING q13
      call avh_olo_conv( q14,i14 ,r14,-1d0) ! RE-USING q14
      call avh_olo_conv( q23,i23 ,r23,-1d0) ! RE-USING q23
      call avh_olo_conv( q24,i24 ,r24,-1d0) ! RE-USING q24
      call avh_olo_conv( q34,i34 ,r34,-1d0) ! RE-USING q34
*
*
       ss = r12
      qss = q12
      iss = i12
       tt = one/r24
      qtt = one/q24
      itt = -i24
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = trm
*
       ss = r23
      qss = q23
      iss = i23
      call avh_olo_rat( qtt,itt ,q13,i13 ,q24,i24 )
      if (itt.eq.itt/2*2) then
        tt = qtt
      else
        tt = -qtt
      endif
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
       ss = r34
      qss = q34
      iss = i34
       tt = r13
      qtt = q13
      itt = i13
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qtt,itt )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qtt,itt )
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qss,iss )
      call avh_olo_rat( qy2,iy2 ,qz2,iz2 ,qss,iss )
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt / ss
      rslt(0) = rslt(0) + trm
*
      call avh_olo_mlt( qz1,iz1 ,qss,iss )
      call avh_olo_mlt( qz2,iz2 ,qss,iss )
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 )
      call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 )
      trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc(  qy2,iy2 ) / (2*x2)
      rslt(0) = rslt(0) + trm
*
       ss = r14
      qss = q14
      iss = i14
       tt = one
      qtt = one
      itt = 0
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qtt,itt )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qtt,itt )
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qss,iss )
      call avh_olo_rat( qy2,iy2 ,qz2,iz2 ,qss,iss )
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt / ss
      rslt(0) = rslt(0) - trm
*
      call avh_olo_mlt( qz1,iz1 ,qss,iss )
      call avh_olo_mlt( qz2,iz2 ,qss,iss )
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 )
      call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 )
      trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc(  qy2,iy2 ) / (2*x2)
      rslt(0) = rslt(0) - trm
*
      rslt(0) = -rslt(0) / (aa*sm1*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_cdm4(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*  ********************************************************************
*  * Finite 1-loop scalar 4-point function with all internal masses
*  * non-zero. Based on the formulas from
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
     &,sm1,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,tt,qtt,trm
     &,q12,q13,q14,q23,q24,q34
     &,r12,r13,r14,r23,r24,r34
     &,d12,d13,d14,d23,d24,d34
     &,zero,one,four,oieps ,avh_olo_sqrt 
      parameter(zero=(0d0,0d0),one=(1d0,0d0),four=(4d0,0d0))
      double precision
     & h1,h2,avh_olo_prec
      integer
     & i12,i13,i14,i23,i24,i34 ,ix1,ix2,iss,itt ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_cdm4: you are calling me'
*
      sm1 = avh_olo_sqrt( m1 ,-1d0 )
      sm2 = avh_olo_sqrt( m2 ,-1d0 )
      sm3 = avh_olo_sqrt( m3 ,-1d0 )
      sm4 = avh_olo_sqrt( m4 ,-1d0 )
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
c       ieps = dcmplx(0d0,avh_olo_prec()**2) !DEBUG
      q12 = zero
      q13 = zero
      q14 = zero
      q23 = zero
      q24 = zero
      q34 = zero
      if (m1+m2.ne.p1 ) q12 = ( m1 + m2 - p1 *oieps)/(sm1*sm2) ! p1
      if (m1+m3.ne.p12) q13 = ( m1 + m3 - p12*oieps)/(sm1*sm3) ! p1+p2
      if (m1+m4.ne.p4 ) q14 = ( m1 + m4 - p4 *oieps)/(sm1*sm4) ! p1+p2+p3
      if (m2+m3.ne.p2 ) q23 = ( m2 + m3 - p2 *oieps)/(sm2*sm3) ! p2
      if (m2+m4.ne.p23) q24 = ( m2 + m4 - p23*oieps)/(sm2*sm4) ! p2+p3
      if (m3+m4.ne.p3 ) q34 = ( m3 + m4 - p3 *oieps)/(sm3*sm4) ! p3
*
      call avh_olo_rfun( r12,d12 ,q12 )
      call avh_olo_rfun( r13,d13 ,q13 )
      call avh_olo_rfun( r14,d14 ,q14 )
      call avh_olo_rfun( r23,d23 ,q23 )
      call avh_olo_rfun( r24,d24 ,q24 )
      call avh_olo_rfun( r34,d34 ,q34 )
c      if (dimag(r12).gt.0d0.and.dreal(r12).lt.0d0) write(6,*) 'r12',r12 !DEBUG
c      if (dimag(r13).gt.0d0.and.dreal(r13).lt.0d0) write(6,*) 'r13',r13 !DEBUG ! should
c      if (dimag(r14).gt.0d0.and.dreal(r14).lt.0d0) write(6,*) 'r14',r14 !DEBUG ! never
c      if (dimag(r23).gt.0d0.and.dreal(r23).lt.0d0) write(6,*) 'r23',r23 !DEBUG ! happen
c      if (dimag(r24).gt.0d0.and.dreal(r24).lt.0d0) write(6,*) 'r24',r24 !DEBUG
c      if (dimag(r34).gt.0d0.and.dreal(r34).lt.0d0) write(6,*) 'r34',r34 !DEBUG
*
      aa = q34/r24 + r13*q12 - q14*r13/r24 - q23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_cdm4: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = d13*d24 + q12*q34 - q14*q23
      cc = q12/r13 + r24*q34 - q14*r24/r13 - q23
c     &   + (q23 - r13*q12 - r24*q34 + r13*r24*q14)*ieps
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
*
      h1 = dreal(q23 - r13*q12 - r24*q34 + r13*r24*q14)
      h2 = h1*dreal(aa)*dreal(x1)
      h1 = h1*dreal(aa)*dreal(x2)
*
      call avh_olo_conv( qx1,ix1 ,-x1 ,-h1 ) ! x1 should have im. part
      call avh_olo_conv( qx2,ix2 ,-x2 ,-h2 ) ! x2 should have im. part
      call avh_olo_conv( q12,i12 ,r12,-1d0 ) !-dreal(p1 ))! RE-USING q12
      call avh_olo_conv( q13,i13 ,r13,-1d0 ) !-dreal(p12))! RE-USING q13
      call avh_olo_conv( q14,i14 ,r14,-1d0 ) !-dreal(p4 ))! RE-USING q14
      call avh_olo_conv( q23,i23 ,r23,-1d0 ) !-dreal(p2 ))! RE-USING q23
      call avh_olo_conv( q24,i24 ,r24,-1d0 ) !-dreal(p23))! RE-USING q24
      call avh_olo_conv( q34,i34 ,r34,-1d0 ) !-dreal(p3 ))! RE-USING q34
*
       ss = r12
      qss = q12
      iss = i12
       tt = one/r24
      qtt = one/q24
      itt = -i24
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = trm
*
       ss = r23
      qss = q23
      iss = i23
c      call avh_olo_rat( qtt,itt ,q13,i13 ,q24,i24 ) !DEBUG
c      if (itt.eq.itt/2*2) then
c        tt = qtt
c      else
c        tt = -qtt
c      endif
      tt = r13/r24
      call avh_olo_conv( qtt,itt ,tt,-dreal(r24) )
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
       ss = r34
      qss = q34
      iss = i34
       tt = r13
      qtt = q13
      itt = i13
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) + trm
*
       ss = r14
      qss = q14
      iss = i14
       tt = one
      qtt = one
      itt = 0
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
      rslt(0) = -rslt(0) / (aa*sm1*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end

      subroutine avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2
     &                              ,ss,qss,iss ,tt,qtt,itt)
*  ********************************************************************
*  * Calculates  trm = hh/(x1-x2)  where 
*  *   hh =   Li2(1-x1*t*s) - Li2(1-x2*t*s)
*  *        + Li2(1-x1*t/s) - Li2(1-x2*t/s)
*  *        + log(-x1*t)^2/2 - log(-x2*t)^2/2
*  ********************************************************************
      implicit none
      double complex trm ,qx1 ,qx2 ,ss,qss ,tt,qtt
     &,qz1,qz2 ,qy1,qy2 ,x1,x2
     &,avh_olo_li2c2
      integer ix1 ,ix2 ,iss ,itt
     &,iz1,iz2 ,iy1,iy2
*
      if (ix1.eq.ix1/2*2) then
        x1 = qx1
      else
        x1 = -qx1
      endif
      if (ix2.eq.ix2/2*2) then
        x2 = qx2
      else
        x2 = -qx2
      endif
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qtt,itt ) ! z1 = x1*t
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qtt,itt ) ! z2 = x2*t
*
      call avh_olo_prd( qy1,iy1 ,qz1,iz1 ,qss,iss ) ! y1 = z1*s = x1*t*s
      call avh_olo_prd( qy2,iy2 ,qz2,iz2 ,qss,iss ) ! y2 = z2*s = x2*t*s
*
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt * ss
*
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qss,iss ) ! y1 = z1/s = x1*t/s
      call avh_olo_rat( qy2,iy2 ,qz2,iz2 ,qss,iss ) ! y2 = z2/s = x2*t/s
*
      trm = trm + avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt / ss
*
      end
