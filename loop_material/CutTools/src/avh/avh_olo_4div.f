************************************************************************
* This is the file  avh_olo_4div.f  of the package                     *
*                                                                      *
*                               OneLOop                                *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 31-08-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_olo_hello.f  for more information.      *
*                                                                      *
************************************************************************


      subroutine avh_olo_d0m16(rslt ,p2,p3,p12,p23 ,m2,m3,m4 ,zmu)
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
      double complex p2,p3,p12,p23 ,m2,m3,m4,zmu
      double complex rslt(0:2) ,cp2,cp3,cp12,cp23,cm2,cm3,cm4
     &,sm1,sm2,sm3,sm4
     &,q13,q23,q24,q34,r13,r23,r24,r34,d23,d24,d34,q24sq,qss,qy1,qy2
     &,qz1,qz2,log24,cc,zero,one
     &,avh_olo_sqrt,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      integer
     & i13,i23,i24,i34,iss,iy1,iy2,iz1,iz2,i24sq ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m16: you are calling me'
*
      if (cdabs(m2).gt.cdabs(m4)) then
        cm2 = m2
        cm4 = m4
        cp2 = p2
        cp3 = p3
      else
        cm2 = m4
        cm4 = m2
        cp2 = p3
        cp3 = p2
      endif
      cm3  = m3
      cp12 = p12
      cp23 = p23
*
      if (cp12.eq.cm3) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m16: p12=m3, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      sm1 = avh_olo_sqrt(zmu,-1d0)
      sm2 = avh_olo_sqrt(cm2,-1d0)
      sm3 = avh_olo_sqrt(cm3,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
*
      r13 = (    cm3-cp12)/(sm1*sm3)
      q23 = (cm2+cm3-cp2 )/(sm2*sm3)
      q24 = (cm2+cm4-cp23)/(sm2*sm4)
      q34 = (cm3+cm4-cp3 )/(sm3*sm4)
      call avh_olo_rfun(r23,d23 ,q23)
      call avh_olo_rfun(r24,d24 ,q24)
      call avh_olo_rfun(r34,d34 ,q34)
      call avh_olo_conv(q13,i13 ,r13,-1d0)
      call avh_olo_conv(q23,i23 ,r23,-1d0)
      call avh_olo_conv(q24,i24 ,r24,-1d0)
      call avh_olo_conv(q34,i34 ,r34,-1d0)
*
      if (r24.eq.-one) then 
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m16: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      call avh_olo_prd( q24sq,i24sq ,q24,i24 ,q24,i24 )
*
      call avh_olo_prd( qss,iss ,q23,i23 ,q34,i34 )
      call avh_olo_prd( qy1,iy1 ,qss,iss ,q24,i24 )
      call avh_olo_rat( qy2,iy2 ,qss,iss ,q24,i24 )
*
      call avh_olo_rat( qss,iss ,q23,i23 ,q34,i34 )
      call avh_olo_prd( qz1,iz1 ,qss,iss ,q24,i24 )
      call avh_olo_rat( qz2,iz2 ,qss,iss ,q24,i24 )
*
      call avh_olo_prd( qss,iss ,q13,i13 ,q23,i23 )
      call avh_olo_mlt( qss,iss ,qss,iss )
      call avh_olo_div( qss,iss ,q24,i24 )
*
      cc = one/( sm2*sm4*(cp12-cm3) )
      log24 = avh_olo_logc2(q24,i24)*r24/(one+r24)
      rslt(2) = dcmplx(0d0)
      rslt(1) = -log24
      rslt(0) = log24 * avh_olo_logc( qss,iss )
     &        + avh_olo_li2c2( q24sq,i24sq ,one,0 )*r24
     &        - avh_olo_li2c2( qy1,iy1 ,qy2,iy2 )*r23*r34
     &        - avh_olo_li2c2( qz1,iz1 ,qz2,iz2 )*r23/r34
      rslt(1) = cc*rslt(1)
      rslt(0) = cc*rslt(0)
      end


      subroutine avh_olo_d0m15(rslt ,p2,p3,p12,p23 ,m2,m4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                  d^(Dim)q
*  * ------ | -------------------------------------------------
*  * i*pi^2 / q^2 [(q+k1)^2-m2] (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=m2, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
*  * m2,m4 should NOT be identically 0d0
*  ********************************************************************
      implicit none
      double complex p2,p3,p12,p23 ,m2,m4 ,zmu
      double complex rslt(0:2) ,cp2,cp3,cp12,cp23,cm2,cm4
     &,sm1,sm2,sm3,sm4
     &,q13,q23,q24,q34,r13,r23,r24,r34,d24,q24sq,qss,qz1,qz2
     &,log24,cc,zero,one
     &,avh_olo_sqrt,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      integer
     & i13,i23,i24,i34,iss,iz1,iz2,i24sq ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m15: you are calling me'
*
      if (cdabs(m2-p2).gt.cdabs(m4-p3)) then
        cm2 = m2
        cm4 = m4
        cp2 = p2
        cp3 = p3
      else
        cm2 = m4
        cm4 = m2
        cp2 = p3
        cp3 = p2
      endif
      cp12 = p12
      cp23 = p23
*
      if (cp12.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m15: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      sm1 = avh_olo_sqrt(zmu,-1d0)
      sm2 = avh_olo_sqrt(cm2,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
      sm3 = dcmplx(cdabs(sm2),0d0)
      r13 = (       -cp12)/(sm1*sm3)
      r23 = (cm2    -cp2 )/(sm2*sm3)
      q24 = (cm2+cm4-cp23)/(sm2*sm4)
      r34 = (    cm4-cp3 )/(sm3*sm4)
      call avh_olo_rfun( r24,d24 ,q24)
*
      if (r24.eq.-one) then 
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m15: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 )
      call avh_olo_prd( q24sq,i24sq ,q24,i24 ,q24,i24 )
*
      call avh_olo_rat( qss,iss ,q34,i34 ,q23,i23 )
      call avh_olo_prd( qz1,iz1 ,qss,iss ,q24,i24 )
      call avh_olo_rat( qz2,iz2 ,qss,iss ,q24,i24 )
*
      call avh_olo_rat( qss,iss ,q13,i13 ,q23,i23 )
      call avh_olo_mlt( qss,iss ,qss,iss )
      call avh_olo_div( qss,iss ,q24,i24 )
*
      cc = r24/(sm2*sm4*cp12)
      log24 = avh_olo_logc2( q24,i24 )/(one+r24)
      rslt(2) = dcmplx(0d0)
      rslt(1) = -log24
      rslt(0) = log24 * avh_olo_logc( qss,iss )
     &        + avh_olo_li2c2( q24sq,i24sq ,one,0 )
      if (r34.ne.zero) then
        rslt(0) = rslt(0)
     &          - avh_olo_li2c2( qz1,iz1 ,qz2,iz2 )*r34/(r23*r24)
      endif
      rslt(1) = cc*rslt(1)
      rslt(0) = cc*rslt(0)
      end


      subroutine avh_olo_d0m14(rslt ,cp12,cp23 ,cm2,cm4 ,zmu)
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
      double complex rslt(0:2) ,cp12,cp23,cm2,cm4,zmu
     &,sm2,sm4,q24,r24,d24,q13,zero,one,cc
     &,avh_olo_sqrt,avh_olo_logc2,avh_olo_logc
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      integer
     & i24,i13 ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m14: you are calling me'
*
      if (cp12.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m14: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      sm2 = avh_olo_sqrt(cm2,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
      q24 = (cm2+cm4-cp23)/(sm2*sm4)
      call avh_olo_rfun( r24,d24 ,q24 )
*
      if (r24.eq.-one) then 
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m14: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      call avh_olo_conv( q24,i24 , r24,-1d0 )
      call avh_olo_conv( q13,i13 ,-cp12/zmu,-1d0 )
*
      cc = dcmplx(-2d0)/(sm2*sm4*cp12)
      cc = cc*avh_olo_logc2( q24,i24 )*r24/(one+r24)
*
      rslt(2) = zero
      rslt(1) = cc
      rslt(0) = -cc*avh_olo_logc( q13,i13 )
      end


      subroutine avh_olo_d0m13(rslt ,p2,p3,p4,p12,p23 ,m3,m4 ,zmu)
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
      double complex rslt(0:2) ,p2,p3,p4,p12,p23,m3,m4,zmu
     &,cp2,cp3,cp4,cp12,cp23,cm3,cm4
     &,sm3,sm4,sm1,sm2,r13,r14,r23,r24,r34,d34,q13,q14,q23,q24,q34
     &,qy1,qy2,qz1,qz2,cc ,logd,li2d,loge,li2a,li2b,li2c ,one
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2
      double precision
     & h1,h2
      parameter(one=(1d0,0d0))
      integer
     & i13,i14,i23,i24,i34,iy1,iy2,iz1,iz2
     &,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m13: you are calling me'
*
      if (p12.eq.m3) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m13: p12=m3, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (p23.eq.m4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m13: p23=m4, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      h1 = cdabs((m3-p12)*(m4-p23))
      h2 = cdabs((m3-p2 )*(m4-p4 ))
      if (h1.ge.h2) then
        cp2  = p2
        cp3  = p3
        cp4  = p4
        cp12 = p12
        cp23 = p23
        cm3  = m3
        cm4  = m4
      else
        cp2  = p12
        cp3  = p3
        cp4  = p23
        cp12 = p2
        cp23 = p4
        cm3  = m3
        cm4  = m4
      endif
*
      sm3 = avh_olo_sqrt(cm3,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
      sm1 = avh_olo_sqrt(zmu,-1d0)
      sm2 = sm1
*
      r13 = (    cm3-cp12)/(sm1*sm3)
      r14 = (    cm4-cp4 )/(sm1*sm4)
      r23 = (    cm3-cp2 )/(sm2*sm3)
      r24 = (    cm4-cp23)/(sm2*sm4)
      q34 = (cm3+cm4-cp3 )/(sm3*sm4)
*
      call avh_olo_rfun( r34,d34 ,q34 )
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q14,i14 ,r14,-1d0 )
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 ) 
*
      call avh_olo_prd( qy1,iy1 ,q14,i14 ,q23,i23 )
      call avh_olo_div( qy1,iy1 ,q13,i13 )
      call avh_olo_div( qy1,iy1 ,q24,i24 )
      logd = avh_olo_logc2( qy1,iy1 )/(r13*r24)
*
      li2d = avh_olo_li2c2(qy1,iy1 ,one,0)/(r13*r24)
*
      loge = avh_olo_logc( q13,i13 )
*
      call avh_olo_rat( qy1,iy1 ,q23,i23 ,q24,i24 )
      call avh_olo_rat( qy2,iy2 ,q13,i13 ,q14,i14 )
*
      call avh_olo_prd( qz1,iz1 ,qy1,iy1 ,q34,i34 )
      call avh_olo_prd( qz2,iz2 ,qy2,iy2 ,q34,i34 )
      li2a = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 )*r34/(r14*r24)
*
      call avh_olo_rat( qz1,iz1 ,qy1,iy1 ,q34,i34 )
      call avh_olo_rat( qz2,iz2 ,qy2,iy2 ,q34,i34 )
      li2b = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 )/(r34*r14*r24)
*
      call avh_olo_rat( qz1,iz1 ,q14,i14 ,q24,i24 )
      call avh_olo_rat( qz2,iz2 ,q13,i13 ,q23,i23 )
      li2c = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 )/(r23*r24)
*
      rslt(2) = dcmplx(0d0)
      rslt(1) = logd
      rslt(0) = li2a + li2b + 2*li2c - 2*li2d - 2*logd*loge
      cc = sm1*sm2*sm3*sm4
      rslt(1) = rslt(1)/cc
      rslt(0) = rslt(0)/cc
      end


      subroutine avh_olo_d0m12(rslt ,cp3,cp4,cp12,cp23 ,cm3,cm4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                  d^(Dim)q
*  * ------ | -------------------------------------------------
*  * i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=0, k2^2=m3, k3^2=p3, (k1+k2+k3)^2=p4
*  * m3,m4 should NOT be indentiallcy 0d0
*  * p4 should NOT be identical to m4
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cp3,cp4,cp12,cp23,cm3,cm4,zmu
     &,sm3,sm4,sm1,sm2,r13,r14,r24,r34,d34,q13,q14,q24,q34
     &,qyy,qz1,qz2,cc ,log13,log14,log24,log34,li2a,li2b,li2c
     &,avh_olo_sqrt,avh_olo_li2c,avh_olo_logc
      double precision
     & avh_olo_pi
      integer
     & i13,i14,i24,i34,iyy,iz1,iz2
     &,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m12: you are calling me'
*
      if (cp12.eq.cm3) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m12: p12=m3, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m12: p23=m4, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      sm3 = avh_olo_sqrt(cm3,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
      sm1 = avh_olo_sqrt(zmu,-1d0)
      sm2 = sm1
*
      r13 = (    cm3-cp12)/(sm1*sm3)
      r14 = (    cm4-cp4 )/(sm1*sm4)
      r24 = (    cm4-cp23)/(sm2*sm4)
      q34 = (cm3+cm4-cp3 )/(sm3*sm4)
*
      call avh_olo_rfun( r34,d34 ,q34 )
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q14,i14 ,r14,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 ) 
*
      log13 = avh_olo_logc( q13,i13 ) 
      log14 = avh_olo_logc( q14,i14 ) 
      log24 = avh_olo_logc( q24,i24 ) 
      log34 = avh_olo_logc( q34,i34 ) 
*
      call avh_olo_rat( qyy,iyy ,q14,i14 ,q13,i13 )
      call avh_olo_prd( qz1,iz1 ,qyy,iyy ,q34,i34 )
      call avh_olo_rat( qz2,iz2 ,qyy,iyy ,q34,i34 )
      li2a = avh_olo_li2c( qz1,iz1 )
      li2b = avh_olo_li2c( qz2,iz2 )
*
      call avh_olo_rat( qyy,iyy ,q14,i14 ,q24,i24 )
      li2c = avh_olo_li2c( qyy,iyy )
*
      rslt(2) = dcmplx(0.5d0)
      rslt(1) = log14 - log24 - log13
      rslt(0) = 2*log13*log24 - log14*log14 - log34*log34
     &        - 2*li2c - li2a - li2b
     &        - dcmplx( avh_olo_pi()**2/8d0 )
      cc = (cm3-cp12)*(cm4-cp23) ! = sm1*sm2*sm3*sm4*r13*r24
      rslt(2) = rslt(2)/cc
      rslt(1) = rslt(1)/cc
      rslt(0) = rslt(0)/cc
      end


      subroutine avh_olo_d0m11(rslt ,cp3,cp12,cp23 ,cm3,cm4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                  d^(Dim)q
*  * ------ | -------------------------------------------------
*  * i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3] [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=0, k2^2=m3, k3^2=p3, (k1+k2+k3)^2=m4
*  * m3,m4 should NOT be indentiallcy 0d0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cp3,cp12,cp23,cm3,cm4,zmu
     &,sm3,sm4,sm1,sm2,r13,r24,r34,d34,q13,q24,q34,cc,log13,log24,log34
     &,avh_olo_sqrt,avh_olo_logc
      double precision
     & avh_olo_pi
      integer
     & i13,i24,i34
     &,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_d0m11: you are calling me'
*
      if (cp12.eq.cm3) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m11: p12=m3, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m11: p23=m4, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      sm3 = avh_olo_sqrt(cm3,-1d0)
      sm4 = avh_olo_sqrt(cm4,-1d0)
      sm1 = avh_olo_sqrt(zmu,-1d0)
      sm2 = sm1
*
      r13 = (    cm3-cp12)/(sm1*sm3)
      r24 = (    cm4-cp23)/(sm2*sm4)
      q34 = (cm3+cm4-cp3 )/(sm3*sm4)
*
      call avh_olo_rfun( r34,d34 ,q34 )
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 ) 
*
      log13 = avh_olo_logc( q13,i13 ) 
      log24 = avh_olo_logc( q24,i24 ) 
      log34 = avh_olo_logc( q34,i34 ) 
*
      rslt(2) = dcmplx(1d0)
      rslt(1) = -log13-log24
      rslt(0) = 2*log13*log24 - log34*log34
     &        - dcmplx( avh_olo_pi()**2 * 7d0/12d0 )
      cc = (cm3-cp12)*(cm4-cp23) ! = sm1*sm2*sm3*sm4*r13*r24
      rslt(2) = rslt(2)/cc
      rslt(1) = rslt(1)/cc
      rslt(0) = rslt(0)/cc
      end


      subroutine avh_olo_d0m10(rslt ,p2,p3,p4,p12,p23 ,m4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | --------------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=p4
*  * m4 should NOT be identically 0d0
*  * p2 should NOT be identically 0d0
*  * p4 should NOT be identical to m4
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p2,p3,p4,p12,p23,m4,zmu
     &,cp2,cp3,cp4,cp12,cp23,cm4
     &,r13,r14,r23,r24,r34,q13,q14,q23,q24,q34,qm4,xx,x1,x2
     &,z1,z0,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      double precision
     & h1,h2
      integer
     & ix,i1,i2 ,avh_olo_un_get ,i13,i14,i23,i24,i34,im4
*
c      write(6,*) 'MESSAGE from avh_olo_d0m10: you are calling me'
*
      if (p12.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m10: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (p23.eq.m4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m10: p23=mm, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      h1 = cdabs(p12*(m4-p23))
      h2 = cdabs( p2*(m4-p4 ))
      if (h1.ge.h2) then
        cp2  = p2
        cp3  = p3
        cp4  = p4
        cp12 = p12
        cp23 = p23
        cm4  = m4
      else
        cp2  = p12
        cp3  = p3
        cp4  = p23
        cp12 = p2
        cp23 = p4
        cm4  = m4
      endif
*
      r23 =    -cp2
      r13 =    -cp12
      r34 = cm4-cp3
      r14 = cm4-cp4
      r24 = cm4-cp23
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 )
      call avh_olo_conv( q14,i14 ,r14,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( qm4,im4 ,cm4,-1d0 )
*
      call avh_olo_rat( x1,i1 ,q34,i34 ,qm4,im4 )
      call avh_olo_prd( x2,i2 ,x1,i1 ,q14,i14 )
      call avh_olo_div( x2,i2 ,q13,i13 )
      call avh_olo_mlt( x1,i1 ,q24,i24 )
      call avh_olo_div( x1,i1 ,q23,i23 )
      if (r34.ne.dcmplx(0d0,0d0)) then
        z0 = -avh_olo_li2c2(x1,i1,x2,i2)*r34/(2*cm4*r23)
      else
        z0 = dcmplx(0d0,0d0)
      endif
*
      call avh_olo_rat( x1,i1 ,q23,i23 ,q13,i13 )
      call avh_olo_rat( x2,i2 ,q24,i24 ,q14,i14 )
      call avh_olo_rat( xx,ix ,x1,i1 ,x2,i2 )
      z1 = -avh_olo_logc2(xx,ix)/r24
*
      z0 = z0 - avh_olo_li2c2( x1,i1 ,x2,i2        )/r14
      z0 = z0 + avh_olo_li2c2( xx,ix ,dcmplx(1d0),0)/r24
*
      call avh_olo_rat( x1,i1 ,qm4,im4 ,q24,i24 )
      call avh_olo_rat( x2,i2 ,qm4,im4 ,zmu,0 )
      z0 = z0 + z1*( avh_olo_logc(x1,i1) - avh_olo_logc(x2,i2)/2 )
*
      rslt(2) = dcmplx(0d0)
      rslt(1) = -z1/r13
      rslt(0) = -2*z0/r13
      end

      subroutine avh_olo_d0m9(rslt ,cp2,cp3,cp12,cp23 ,cm4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | --------------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=0, k2^2=p2, k3^2=p3, (k1+k2+k3)^2=m4
*  * m4 should NOT be identically 0d0
*  * p2 should NOT be identically 0d0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cp2,cp3,cp12,cp23,cm4,zmu
     &,logm,log12,log23,li12,li23,z2,z1,z0,cc
     &,r13,r23,r24,r34,q13,q23,q24,q34,qm4,xx
     &,avh_olo_logc,avh_olo_li2c
      double precision
     & const,avh_olo_pi
      integer
     & i13,i23,i24,i34,im4,ix,avh_olo_un_get
      logical init
      data init/.true./
      save init,const
*
c      write(6,*) 'MESSAGE from avh_olo_d0m9: you are calling me'
*
      if (init) then
        init = .false.
        const = avh_olo_pi()**2/24d0
      endif
*
      if (cp12.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m9: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m9: p23=mm, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      r23 =    -cp2
      r13 =    -cp12
      r34 = cm4-cp3
      r24 = cm4-cp23
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( qm4,im4 ,cm4,-1d0 )
*
      call avh_olo_rat( xx,ix ,qm4,im4 ,zmu,0 )
      logm  = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q13,i13 ,q23,i23 )
      log12 = avh_olo_logc( xx,ix )
      li12  = avh_olo_li2c( xx,ix )
      call avh_olo_rat( xx,ix ,q24,i24 ,qm4,im4 )
      log23 = avh_olo_logc( xx,ix )
      call avh_olo_mlt( xx,ix ,q34,i34 )
      call avh_olo_div( xx,ix ,q23,i23 )
      li23  = avh_olo_li2c( xx,ix )
*
      z2 = dcmplx(1d0/2d0)
      z1 = -log12 - log23
      z0 = li23 + 2*li12 + z1*z1 + dcmplx(const)
      cc = dcmplx(1d0)/(r13*r24)
      rslt(2) = cc*z2
      rslt(1) = cc*(z1 - z2*logm)
      rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
      end

      subroutine avh_olo_d0m8(rslt ,cp3,cp4,cp12,cp23 ,cm4 ,zmu)
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
      double complex rslt(0:2) ,cp3,cp4,cp12,cp23,cm4,zmu
     &,z1,z0,cc ,avh_olo_logc,avh_olo_li2c
     &,r13,r14,r24,r34,q13,q14,q24,q34,qm4,xx,x1,x2,x3
      double precision
     & const ,avh_olo_pi
      integer
     & i13,i14,i24,i34,im4,ix,i1,i2,i3,avh_olo_un_get
      logical init
      data init/.true./
      save init,const
*
c      write(6,*) 'MESSAGE from avh_olo_d0m8: you are calling me'
*
      if (init) then
        init = .false.
        const = avh_olo_pi()**2/4d0
      endif
*
      if (cp12.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m8: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m8: p23=mm, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      r13 =    -cp12
      r34 = cm4-cp3
      r14 = cm4-cp4
      r24 = cm4-cp23
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q34,i34 ,r34,-1d0 )
      call avh_olo_conv( q14,i14 ,r14,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( qm4,im4 ,cm4,-1d0 )
*
      call avh_olo_rat( x1,i1 ,q34,i34 ,q24,i24 ) ! mmp3/mmp23
      call avh_olo_rat( x2,i2 ,q14,i14 ,q24,i24 ) ! mmp4/mmp23
      call avh_olo_rat( x3,i3 ,q13,i13 ,zmu,0 )   ! -p12/mu2
*
      call avh_olo_prd( xx,ix ,x1,i1 ,x2,i2 )
      call avh_olo_div( xx,ix ,x3,i3 )
      z1 = avh_olo_logc( xx,ix )
      z0 = -2*( avh_olo_li2c(x1,i1) + avh_olo_li2c(x2,i2) )
*
      call avh_olo_rat( xx,ix ,q24,i24 ,zmu,0 ) ! mmp23/mu2
      z0 = z0 + 2*avh_olo_logc(xx,ix)*avh_olo_logc(x3,i3)
*
      call avh_olo_rat( x1,i1 ,q34,i34 ,zmu,0 ) ! mmp3/mu2
      call avh_olo_rat( x2,i2 ,q14,i14 ,zmu,0 ) ! mmp4/mu2
      z0 = z0 - avh_olo_logc(x1,i1)**2 - avh_olo_logc(x2,i2)**2
*
      call avh_olo_prd( xx,ix ,x1,i1 ,x2,i2 )
      call avh_olo_div( xx,ix ,x3,i3 )
      z0 = z0 + avh_olo_logc(xx,ix)**2/2      
*
      call avh_olo_rat( x1,i1 ,qm4,im4 ,zmu,0 ) ! mm/mu2
      call avh_olo_div( x1,i1 ,xx,ix )
      z0 = z0 + avh_olo_li2c(x1,i1)
*
      cc = dcmplx(1d0)/(r13*r24)
      rslt(2) = cc
      rslt(1) = cc*z1
      rslt(0) = cc*( z0 - dcmplx(const) )
      end

      subroutine avh_olo_d0m7(rslt ,cp4,cp12,cp23 ,cm4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | --------------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=k2^2=0, k3^2=m4, (k1+k2+k3)^2=p4
*  * m3 should NOT be identically 0d0
*  * p4 should NOT be identically m4
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cp4,cp12,cp23,cm4,zmu
     &,r13,r14,r24,q13,q14,q24,qm4,xx
     &,logm,log12,log23,log4,li423,z2,z1,z0,cc
     &,avh_olo_logc,avh_olo_li2c
      double precision
     & const,avh_olo_pi
      integer
     & i13,i14,i24,im4,ix,avh_olo_un_get
      logical init
      data init/.true./
      save init,const
*
c      write(6,*) 'MESSAGE from avh_olo_d0m7: you are calling me'
*
      if (init) then
        init = .false.
        const = avh_olo_pi()**2*(13d0/24d0)
      endif
*
      if (cp12.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m7: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m7: p23=mm, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      r13 =    -cp12
      r14 = cm4-cp4
      r24 = cm4-cp23
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q14,i14 ,r14,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( qm4,im4 ,cm4,-1d0 )
*
      call avh_olo_rat( xx,ix ,qm4,im4 ,zmu,0 )
      logm  = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q13,i13 ,qm4,im4 )
      log12 = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q24,i24 ,qm4,im4 )
      log23 = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q14,i14 ,qm4,im4 )
      log4  = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q14,i14 ,q24,i24 )
      li423 = avh_olo_li2c( xx,ix )
*
      z2 = dcmplx(3d0/2d0)
      z1 = -2*log23 - log12 + log4
      z0 = 2*(log12*log23 - li423) - log4*log4 - dcmplx(const)
      cc = dcmplx(1d0)/(r13*r24)
      rslt(2) = cc*z2
      rslt(1) = cc*(z1 - z2*logm)
      rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
      end

      subroutine avh_olo_d0m6(rslt ,cp12,cp23 ,cm4 ,zmu)
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | --------------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 [(q+k1+k2+k3)^2-m4]
*  *
*  * with  k1^2=k2^2=0, k3^2=(k1+k2+k3)^2=m4
*  * m3 should NOT be identically 0d0
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cp12,cp23,cm4,zmu
     &,r13,r24,q13,q24,qm4,xx
     &,logm,log1,log2,z2,z1,z0,cc ,avh_olo_logc
      double precision
     & const ,avh_olo_pi
      integer
     & i13,i24,im4,ix,avh_olo_un_get
      logical init
      data init/.true./
      save init,const
*
c      write(6,*) 'MESSAGE from avh_olo_d0m6: you are calling me'
*
      if (init) then
        init = .false.
        const = avh_olo_pi()**2/3d0
      endif
*
      if (cp12.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m7: p12=0, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (cp23.eq.cm4) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_d0m7: p23=mm, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      r13 =    -cp12
      r24 = cm4-cp23
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q24,i24 ,r24,-1d0 )
      call avh_olo_conv( qm4,im4 ,cm4,-1d0 )
*
      call avh_olo_rat( xx,ix ,qm4,im4 ,zmu,0 )
      logm = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q13,i13 ,qm4,im4 )
      log1 = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q24,i24 ,qm4,im4 )
      log2 = avh_olo_logc( xx,ix )
*
      z2 = dcmplx(2d0)
      z1 = -2*log2 - log1
      z0 = 2*(log2*log1 - dcmplx(const))
      cc = dcmplx(1d0)/(r13*r24)
      rslt(2) = cc*z2
      rslt(1) = cc*(z1 - z2*logm)
      rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
      end


      subroutine avh_olo_d0m3( rslt ,p2,p4,p5,p6 ,mu2 )
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
      double complex rslt(0:2)
     &,q2,q4,q5,q6,q25,q64,q26,q54,qy,qz,logy
     &,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      double precision p2,p4,p5,p6 ,mu2
      integer
     & i2,i4,i5,i6,i25,i64,i26,i54,iy,iz
*
c      write(6,*) 'MESSAGE from avh_olo_d0m3: you are calling me' !DEBUG
*
      call avh_olo_conv( q2,i2 ,dcmplx(-p2),-1d0 )
      call avh_olo_conv( q4,i4 ,dcmplx(-p4),-1d0 )
      call avh_olo_conv( q5,i5 ,dcmplx(-p5),-1d0 )
      call avh_olo_conv( q6,i6 ,dcmplx(-p6),-1d0 )
      call avh_olo_rat( q25,i25 ,q2,i2 ,q5,i5 )
      call avh_olo_rat( q64,i64 ,q6,i6 ,q4,i4 )
      call avh_olo_rat( q26,i26 ,q2,i2 ,q6,i6 )
      call avh_olo_rat( q54,i54 ,q5,i5 ,q4,i4 )
      call avh_olo_rat( qy,iy ,q26,i26 ,q54,i54 )
      call avh_olo_prd( qz,iz ,q54,i54 ,q2/mu2,i2 )
      call avh_olo_mlt( qz,iz ,q6/mu2,i6 )
      logy = avh_olo_logc2( qy,iy )/dcmplx(p5*p6)
      rslt(1) = logy
      rslt(0) = avh_olo_li2c2( q64,i64 ,q25,i25 )/dcmplx(p4*p5)
     &        + avh_olo_li2c2( q54,i54 ,q26,i26 )/dcmplx(p4*p6)
     &        - avh_olo_li2c2( dcmplx(1d0),0 ,qy,iy )/dcmplx(p5*p6)
     &        - logy*avh_olo_logc( qz,iz )/2
      rslt(2) = dcmplx(0d0)
      rslt(1) = 2*rslt(1)
      rslt(0) = 2*rslt(0)
      end


      subroutine avh_olo_d0m5( rslt ,p2,p3,p4,p5,p6 ,mu2 )
*  ********************************************************************
*  * calculates
*  *
*  *     C   /               d^(Dim)q
*  *  ------ | ---------------------------------------
*  *  i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
*  *
*  * with  k1^2=0
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
     &,q2,q3,q4,q5,q6 ,q25,q64,qy,qz,logy
     &,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      double precision p2,p3,p4,p5,p6 ,mu2
      integer
     & i2,i3,i4,i5,i6,i25,i64,iy,iz
*
c      write(6,*) 'MESSAGE from avh_olo_d0m5: you are calling me' !DEBUG
*
      call avh_olo_conv( q2,i2 ,dcmplx(-p2),-1d0 )
      call avh_olo_conv( q3,i3 ,dcmplx(-p3),-1d0 )
      call avh_olo_conv( q4,i4 ,dcmplx(-p4),-1d0 )
      call avh_olo_conv( q5,i5 ,dcmplx(-p5),-1d0 )
      call avh_olo_conv( q6,i6 ,dcmplx(-p6),-1d0 )
      call avh_olo_rat( q25,i25 ,q2,i2 ,q5,i5 )
      call avh_olo_rat( q64,i64 ,q6,i6 ,q4,i4 )
      call avh_olo_rat( qy,iy ,q25,i25 ,q64,i64 )
      call avh_olo_rat( qz,iz ,q64,i64 ,q3/mu2,i3 )
      call avh_olo_div( qz,iz ,q3/mu2,i3 )
      call avh_olo_mlt( qz,iz ,q2/mu2,i2 )
      call avh_olo_mlt( qz,iz ,q5/mu2,i5 )
      call avh_olo_mlt( qz,iz ,q6/mu2,i6 )
      call avh_olo_mlt( qz,iz ,q6/mu2,i6 )
      logy = avh_olo_logc2( qy,iy )/dcmplx(p5*p6)
      rslt(2) = dcmplx(0d0)
      rslt(1) = logy
      rslt(0) = avh_olo_li2c2( q64,i64 ,q25,i25 )/dcmplx(p4*p5)
     &        - avh_olo_li2c2( dcmplx(1d0),0 ,qy,iy )/dcmplx(p5*p6)
     &        - logy*avh_olo_logc( qz,iz )/4
      rslt(0) = 2*rslt(0)
      end


      subroutine avh_olo_d0( vald0 ,p1,p2,p3,p4,p12,p23 )
*  ********************************************************************
*  * calculates
*  *               C   /              d^(Dim)q
*  *            ------ | ---------------------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2 (q+k1+k2+k3)^2
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
*  *
*  * input:  p1 = k1^2,  p2 = k2^2,  p3 = k3^2,  p4 = (k1+k2+k3)^2,
*  *         p12 = (k1+k2)^2,  p23 = (k2+k3)^2
*  * output: vald0(0) = eps^0   -coefficient
*  *         vald0(1) = eps^(-1)-coefficient
*  *         vald0(2) = eps^(-2)-coefficient
*  *
*  * The input values (p1,p2,p3,p4,p12,p23) should be real.
*  * If any of these numbers is IDENTICALLY 0d0, the corresponding
*  * IR-singular case is returned.
*  ********************************************************************
      implicit none
      double complex vald0(0:2)
     &,pi2,aa,bb,cc,log3,log4,log5,log6,li24,li25,li26
     &,li254,li263,avh_olo_loga,avh_olo_li2a
      double precision p1,p2,p3,p4,p12,p23
     &,mu2,pp(6),gg,ff,hh,arg,smax,ap(6)
     &,avh_olo_thrs,avh_olo_mu_get,avh_olo_pi
      integer
     & base(4),icase,ii,per(6),sf,sgn,imax,i3,i4,i5,i6
     &,avh_olo_un_get
      character(3) label(6)
      logical init ,avh_olo_os_get
      data init/.true./,base/8,4,2,1/
     &    ,label/'p1','p2','p3','p4','p12','p23'/
      save init,pi2
*
      if (init) then
        init = .false.
        pi2  = dcmplx( avh_olo_pi()**2 )
      endif
      mu2 = avh_olo_mu_get()**2
*
      ff = dabs(p12*p23)
      gg = dabs(p2*p4)
      hh = dabs(p1*p3)
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
      ap(1) = dabs(pp(1))
      ff   = ap(1)
      imax = 1
      do ii=2,6
        ap(ii) = dabs(pp(ii))
        if (ap(ii).gt.ff) then
          ff   = ap(ii)
          imax = ii
        endif
      enddo
      smax = avh_olo_thrs(ff)
*
      if (avh_olo_os_get()) then
        if (ap(1).lt.smax) ap(1) = 0d0
        if (ap(2).lt.smax) ap(2) = 0d0
        if (ap(3).lt.smax) ap(3) = 0d0
        if (ap(4).lt.smax) ap(4) = 0d0
      endif
*
      icase = 0
      do ii=1,4
      if (ap(ii).gt.0d0) then
        icase = icase + base(ii)
        if (ap(ii).lt.smax.and.avh_olo_un_get().gt.0)
     &    write(avh_olo_un_get(),*)
     &    'WARNING from avh_olo_d0: |',label(ii),'/',label(imax),'| ='
     &   ,ap(ii)/ff
      endif
      enddo
      call avh_olo_d0per(icase,per)
*
      i3 = 0
      i4 = 0
      i5 = 0
      i6 = 0
      if (-pp(per(3)).lt.0d0) i3 = -1
      if (-pp(per(4)).lt.0d0) i4 = -1
      if (-pp(per(5)).lt.0d0) i5 = -1
      if (-pp(per(6)).lt.0d0) i6 = -1
*
      if     (icase.eq.0) then
* 0 masses non-zero
        gg = 1d0/( pp(per(5)) * pp(per(6)) )
        log5 = avh_olo_loga(-pp(per(5))/mu2, i5 )
        log6 = avh_olo_loga(-pp(per(6))/mu2, i6 )
        aa = dcmplx(4d0)
        bb = -2*(log5 + log6)
        cc = log5**2 + log6**2 
     &     - avh_olo_loga( pp(per(5))/pp(per(6)) , i5-i6 )**2
     &     - 4*pi2/3
      elseif (icase.eq.1) then
* 1 mass non-zero
        gg = 1d0/( pp(per(5)) * pp(per(6)) )
        ff =  gg*( pp(per(5)) + pp(per(6)) - pp(per(4)) )
        log4 = avh_olo_loga(-pp(per(4))/mu2,i4)
        log5 = avh_olo_loga(-pp(per(5))/mu2,i5)
        log6 = avh_olo_loga(-pp(per(6))/mu2,i6)
        sf = idnint(sign(1d0,ff))
        sgn = 0
          arg = pp(per(4))*ff 
          if (arg.lt.0d0) sgn = sf
          li24 = avh_olo_li2a(arg,sgn)
        sgn = 0
          arg = pp(per(5))*ff 
          if (arg.lt.0d0) sgn = sf
          li25 = avh_olo_li2a(arg,sgn)
        sgn = 0
          arg = pp(per(6))*ff 
          if (arg.lt.0d0) sgn = sf
          li26 = avh_olo_li2a(arg,sgn)
        aa = dcmplx(2d0)
        bb = 2*(log4-log5-log6)
        cc = log5**2 + log6**2 - log4**2 
     &     - pi2/2 + 2*(li25 + li26 - li24)
      elseif (icase.eq.2) then
* 2 neighbour masses non-zero
        gg = 1d0/( pp(per(5)) * pp(per(6)) )
        ff =  gg*( pp(per(5)) + pp(per(6)) - pp(per(4)) )
        log3 = avh_olo_loga(-pp(per(3))/mu2,i3)
        log4 = avh_olo_loga(-pp(per(4))/mu2,i4)
        log5 = avh_olo_loga(-pp(per(5))/mu2,i5)
        log6 = avh_olo_loga(-pp(per(6))/mu2,i6)
        li254 = avh_olo_li2a( pp(per(4))/pp(per(5)) , i4-i5 )
        li263 = avh_olo_li2a( pp(per(3))/pp(per(6)) , i3-i6 )
        sf = idnint(sign(1d0,ff))
        sgn = 0
          arg = pp(per(4))*ff 
          if (arg.lt.0d0) sgn = sf
          li24 = avh_olo_li2a(arg,sgn)
        sgn = 0
          arg = pp(per(5))*ff 
          if (arg.lt.0d0) sgn = sf
          li25 = avh_olo_li2a(arg,sgn)
        sgn = 0
          arg = pp(per(6))*ff 
          if (arg.lt.0d0) sgn = sf
          li26 = avh_olo_li2a(arg,sgn)
        aa = dcmplx(1d0)
        bb = log4 + log3 - log5 - 2*log6
        cc = log5**2 + log6**2 - log3**2 - log4**2
     &     +(log3 + log4 - log5)**2/2
     &     - pi2/12 + 2*(li254 - li263 + li25 + li26 - li24)
      elseif (icase.eq.5) then
* 2 opposite masses non-zero
        call avh_olo_d0m3( vald0 ,pp(per(2)),pp(per(4))
     &                           ,pp(per(5)),pp(per(6)) ,mu2 )
        return
      elseif (icase.eq.3) then
* 3 masses non-zero
        call avh_olo_d0m5( vald0 ,pp(per(2)),pp(per(3)),pp(per(4))
     &                           ,pp(per(5)),pp(per(6)) ,mu2 )
        return
      elseif (icase.eq.4) then
* 4 masses non-zero
        call avh_olo_cdm0(vald0 ,dcmplx(pp(per(1))),dcmplx(pp(per(2)))
     &                          ,dcmplx(pp(per(3))),dcmplx(pp(per(4)))
     &                          ,dcmplx(pp(per(5))),dcmplx(pp(per(6))) )
        return
      endif
*
      vald0(0) = gg*cc
      vald0(1) = gg*bb
      vald0(2) = gg*aa
      end

     
      subroutine avh_olo_d0per(icase,per)
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
