************************************************************************
* This is the file  avh_olo_3fin.f  of the package                     *
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


      subroutine avh_olo_cam(rslt ,mm ,zmu)
*  ********************************************************************
*  * The 1-loop scalar 1-point function.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,mm,zmu
     &,qmm,xx,zero ,avh_olo_logc
      parameter(zero=(0d0,0d0))
      integer
     & imm,ix
*
c      write(6,*) 'MESSAGE from avh_olo_cam: you are calling me' !DEBUG
*
      rslt(2) = zero
      if (mm.eq.zero) then
        rslt(1) = zero
        rslt(0) = zero
      else
        call avh_olo_conv( qmm,imm ,mm,-1d0 )
        call avh_olo_rat( xx,ix ,qmm,imm ,zmu,0 )
        rslt(1) = mm
        rslt(0) = mm - mm*avh_olo_logc( xx,ix )
      endif
      end


      subroutine avh_olo_cbm(rslt ,pp,m1i,m2i ,zmu)
*  ********************************************************************
*  * The 1-loop scalar 2-point function. Based on the formulas from
*  * A. Denner, Fortsch.Phys.41:307-420,1993 arXiv:0709.1075 [hep-ph]
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,pp,m1i,m2i,zmu
     &,m1,m2,qpp,qmm,qrr,hh,aa,bb,rr,dd,qz1,qz2,qz3,zero,one,two
     &,avh_olo_logc,avh_olo_logc2,avh_olo_sqrt
      parameter(zero=(0d0,0d0),one=(1d0,0d0),two=(2d0,0d0))
      double precision
     & t1,tt,thrs,avh_olo_prec
      integer
     & imm,ipp,irr,ntrm,ii,iz1,iz2,iz3
      double complex cc(0:20)
      common/avh_olo_b_com/ thrs,ntrm
      logical init
      data init/.true./
      save init
*
c      write(6,*) 'MESSAGE from avh_olo_cbm: you are calling me' !DEBUG
*
      if (init) then
        init = .false.
        if (avh_olo_prec().gt.1d-24) then
          thrs = 0.07d0 ! double precision
          ntrm = 11     ! tested to suit also avh_olo_b11c
        else
          thrs = 0.005d0 ! quadruple precision, not tested
          ntrm = 11      !
        endif
      endif
*
      t1 = cdabs(m1i)
      tt = cdabs(m2i)
      if (t1.lt.tt) then
        m1 = m1i
        m2 = m2i
      else
        m1 = m2i
        m2 = m1i
        tt = t1
      endif
*
      rslt(2) = zero     
      rslt(1) = one
*
      if (m2.eq.zero) then
        if (pp.eq.zero) then
          rslt(1) = zero     
          rslt(0) = zero     
        else
          call avh_olo_conv( qpp,ipp ,-pp/zmu,-1d0 )
          rslt(0) = two - avh_olo_logc( qpp,ipp )
        endif
      else!if(m2.ne.zero)
        tt = cdabs(pp)/tt
        if (m1.eq.zero) then
          call avh_olo_conv( qmm,imm ,m2/zmu,-1d0 )
          if     (pp.eq.zero) then
            rslt(0) = one - avh_olo_logc( qmm,imm )
          elseif (pp.eq.m2) then
            rslt(0) = two - avh_olo_logc( qmm,imm )
          elseif (tt.lt.1d0) then
            hh = m2-pp
            call avh_olo_conv( qpp,ipp ,hh/zmu,-1d0 )
            call avh_olo_rat( qrr,irr ,qpp,ipp  ,qmm,imm )
            rslt(0) = two -         avh_olo_logc( qmm,imm )
     &                    + (hh/pp)*avh_olo_logc( qrr,irr )
          else!if (tt.ge.1d0) then
            hh = m2-pp
            call avh_olo_conv( qpp,ipp ,hh/zmu,-1d0 )
            rslt(0) = two - (m2/pp)*avh_olo_logc( qmm,imm )
     &                    + (hh/pp)*avh_olo_logc( qpp,ipp )
          endif
        else!if(m2.ne.zero)
          if (pp.eq.zero) then
             call avh_olo_conv( qz1,iz1, m1/zmu,-1d0 )
             call avh_olo_conv( qz2,iz2, m2/zmu,-1d0 )
             call avh_olo_rat( qz3,iz3 ,qz1,iz1 ,qz2,iz2 )
             rslt(0) = one + avh_olo_logc2( qz3,iz3 )
     &                     - avh_olo_logc( qz1,iz1 )
          else!if(pp.ne.zero)
            if     (tt.le.thrs) then
              call avh_olo_bexp( cc ,m1,m2 ,zmu ,ntrm)
              rslt(0) = cc(ntrm)
              do ii=ntrm-1,0,-1
                rslt(0) = cc(ii) + pp*rslt(0)
              enddo
            elseif (tt.lt.1d0) then
              hh = avh_olo_sqrt(m1,-1d0)
              bb = avh_olo_sqrt(m2,-1d0)
              aa = hh*bb ! sm1*sm2
              bb = hh/bb ! sm1/sm2
              hh = (m1+m2-pp)/aa
              call avh_olo_conv( qz1,iz1 ,bb,-1d0 ) ! sm1/sm2
              dd = (m2-m1)**2 + ( pp - 2*(m1+m2) )*pp
              dd = avh_olo_sqrt( dd,-1d0 )/aa
              call avh_olo_rfun0( rr ,dd ,hh )
              call avh_olo_conv( qrr,irr ,rr,-1d0 )
              call avh_olo_prd( qz2,iz2 ,qz1,iz1 ,qrr,irr ) ! sm1/sm2 * r
              call avh_olo_conv( qz3,iz3 ,m2/zmu,-1d0 )     ! m2/mu2
              rslt(0) = two - avh_olo_logc( qz3,iz3 )
     &                      + avh_olo_logc( qz1,iz1 )*two*m1/(aa*rr-m1)
     &                + avh_olo_logc2( qz2,iz2 )*dd*aa/(aa*rr-m1+pp)
            else
              hh = avh_olo_sqrt(m1,-1d0)
              bb = avh_olo_sqrt(m2,-1d0)
              aa = hh*bb ! sm1*sm2
              bb = hh/bb ! sm1/sm2
              hh = (m1+m2-pp)/aa
              call avh_olo_conv( qz1,iz1 ,bb,-1d0 ) ! sm1/sm2
              call avh_olo_rfun( rr,dd ,hh )
              call avh_olo_conv( qrr,irr ,rr,-1d0 )
              call avh_olo_prd( qz2,iz2 ,qz1,iz1 ,qrr,irr ) ! sm1/sm2 * r
              call avh_olo_conv( qz1,iz1 ,aa/zmu,-1d0 )     ! sm1*sm2 / mu2
              call avh_olo_conv( qz2,iz2 ,bb,-1d0 )         ! sm1/sm2
              rslt(0) = two - avh_olo_logc( qz1,iz1 )
     &                      + ( avh_olo_logc( qz2,iz2 )*(m2-m1)
     &                         +avh_olo_logc( qrr,irr )*dd*aa   )/pp
            endif
c            call avh_olo_bexp( cc ,m1,m2 ,zmu ,ntrm) !DEBUG
c            hh = cc(ntrm)                            !DEBUG
c            do ii=ntrm-1,0,-1                        !DEBUG
c              hh = cc(ii) + pp*hh                    !DEBUG
c            enddo                                    !DEBUG
c            write(6,'(a4,2d24.16)') 'exp:',hh        !DEBUG
          endif
        endif
      endif
      end


      subroutine avh_olo_ccm0(rslt ,p1,p2,p3)
*  ********************************************************************
*  * Finite 1-loop scalar 3-point function with all internal masses
*  * equal zero. Obtained from the formulas for 4-point functions in
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  * by sending one internal mass to infinity.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3
     &,aa,bb,cc,dd,x1,qx1,x2,qx2,qz1,qz2,qy1,qy2
     &,q23,q24,q34,r23,r24,r34 ,zero,four,one,half
     &,trm ,avh_olo_sqrt ,avh_olo_li2c2,avh_olo_logc2,avh_olo_logc
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),one=(1d0,0d0))
      parameter(four=(4d0,0d0))
      double precision
     & hh
      integer
     & i23,i24,i34 ,ix1,ix2,iz1,iz2,iy1,iy2
*
c      write(6,*) 'MESSAGE from avh_olo_ccm0: you are calling me' !DEBUG
*
      r23 = -p1  ! r23 = (m2+m3-p2 )/(sm2*sm3)  p2
      r24 = -p3  ! r24 = (m2+m4-p23)/(sm2*sm4)  p2+p3
      r34 = -p2  ! r34 = (m3+m4-p3 )/(sm3*sm4)  p3      
*
      aa = r34*r24
      bb = r24 + r34 - r23
      cc = one
      hh = dreal(r23)
      dd = avh_olo_sqrt( bb*bb - four*aa*cc , -dreal(aa)*hh )
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,1)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 , hh)
      call avh_olo_conv( qx2,ix2 ,x2 ,-hh)
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
      rslt(0) = zero
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,q34,i34 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,q34,i34 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34
      rslt(0) = rslt(0) + trm
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,q24,i24 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,q24,i24 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r24
      rslt(0) = rslt(0) + trm
*
      call avh_olo_rat( qy1,iy1 ,qx1,ix1 ,qx2,ix2 )
      call avh_olo_prd( qy2,iy2 ,qx1,ix1 ,qx2,ix2 )
      trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc( qy2,iy2 )*half/x2
      rslt(0) = rslt(0) - trm
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
      qz2 = one/q23
      iz2 = -i23
      trm = avh_olo_logc2( qz1,iz1 ) / x2 * avh_olo_logc( qz2,iz2 )
      rslt(0) = rslt(0) + trm
*
      rslt(0) = rslt(0) / aa
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_ccm1(rslt ,p1i,p2i,p3i ,m3i)
*  ********************************************************************
*  * Finite 1-loop scalar 3-point function with one internal masses
*  * non-zero. Obtained from the formulas for 4-point functions in
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  * by sending one internal mass to infinity.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1i,p2i,p3i ,m3i 
     &,p2,p3,p4,p12,p23,m4
     &,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,qx1,x2,qx2,qss,trm,qz1,qz2,qy1,qy2
     &,q23,q24,q34,r23,r24,r34 ,zero,half,one,four,oieps
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2 
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),one=(1d0,0d0))
      parameter(four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i23,i24,i34 ,ix1,ix2,iss,iz1,iz2,iy1,iy2 ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_ccm1: you are calling me' !DEBUG
*
c      p1 = nul
      p2 = p1i
      p3 = p2i
      p4 = p3i
      p12 = p1i
      p23 = p3i
c      m1 = infinite
c      m2 = m1i = zero
c      m3 = m2i = zero
      m4 = m3i
*
      sm4 = avh_olo_sqrt(m4,-1d0)
      sm3 = dcmplx(cdabs(sm4),0d0)
      sm2 = sm3
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      r24 = zero
      r34 = zero
                     r23 = (   -p2 *oieps )/(sm2*sm3)
      if (m4.ne.p23) r24 = ( m4-p23*oieps )/(sm2*sm4)
      if (m4.ne.p3 ) r34 = ( m4-p3 *oieps )/(sm3*sm4)     
*
      aa = r34*r24 - r23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_ccm1: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = r24/sm3 + r34/sm2 - r23/sm4
      cc = one/(sm2*sm3)
c      hh = dreal(r23)
c      dd = avh_olo_sqrt( bb*bb - four*aa*cc , -dreal(aa)*hh )
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,sm4,0 )
      call avh_olo_rat( qz2,iz2 ,qx2,ix2 ,sm4,0 )
      call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 )
      call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 )
      trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc( qy2,iy2 )*half/x2
      rslt(0) = -trm
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,sm4,0 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,sm4,0 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * sm4
      rslt(0) = rslt(0) - trm
*
*
      if (r34.ne.zero) then
        call avh_olo_prd( qss,iss ,q34,i34 ,sm3,0 )
        call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
        call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
        trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 * sm3
        rslt(0) = rslt(0) + trm
      endif
*
      if (r24.ne.zero) then
        call avh_olo_prd( qss,iss ,q24,i24 ,sm2,0 )
        call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
        call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
        trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r24 * sm2
        rslt(0) = rslt(0) + trm
      endif
*
      call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,qx2,ix2 )
      qz2 = one/sm2
      iz2 = 0
      call avh_olo_div( qz2,iz2 ,sm3,0 )
      call avh_olo_div( qz2,iz2 ,q23,i23 )
      trm = avh_olo_logc2( qz1,iz1 ) / x2 * avh_olo_logc( qz2,iz2 )
      rslt(0) = rslt(0) + trm
*
      rslt(0) = rslt(0) / (aa*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_ccm2(rslt ,p1i,p2i,p3i ,m2i,m3i)
*  ********************************************************************
*  * Finite 1-loop scalar 3-point function with two internal masses
*  * non-zero. Obtained from the formulas for 4-point functions in
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  * by sending one internal mass to infinity.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1i,p2i,p3i ,m2i,m3i
     &,p2,p3,p23,m2,m4,sm2,sm3,sm4 ,aa,bb,cc,dd,x1,x2,qss,trm
     &,q23,r23,q34,r34,q24,r24,d24,qx1,qx2,qy1,qy2,qz1,qz2
     &,zero,half,one,four,oieps
     &,avh_olo_sqrt,avh_olo_li2c2,avh_olo_logc,avh_olo_logc2 
      parameter(zero=(0d0,0d0),half=(0.5d0,0d0),one=(1d0,0d0))
      parameter(four=(4d0,0d0))
      double precision
     & avh_olo_prec
      integer
     & i23,i24,i34 ,ix1,ix2,iss,iz1,iz2,iy1,iy2
     &,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_ccm2: you are calling me' !DEBUG
*
c      p1 = nul
      p2 = p3i
      p3 = p1i
c      p4 = p2i
c      p12 = p3i
      p23 = p2i
c      m1 = infinite
      m2 = m3i
c      m3 = m1i = zero
      m4 = m2i
*
c      sm1 = infinite
      sm2 = avh_olo_sqrt(m2,-1d0)
      sm3 = dcmplx(cdabs(sm2),0d0) !avh_olo_sqrt(m3,-1d0)
      sm4 = avh_olo_sqrt(m4,-1d0)
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      r23 = zero
      q24 = zero
      r34 = zero
      if (m2   .ne.p2 ) r23 = (    m2-p2 *oieps )/(sm2*sm3) ! p2
      if (m2+m4.ne.p23) q24 = ( m2+m4-p23*oieps )/(sm2*sm4) ! p2+p3
      if (m4   .ne.p3 ) r34 = (    m4-p3 *oieps )/(sm3*sm4) ! p3
*
      call avh_olo_rfun( r24,d24 ,q24 )
*
      aa = r34/r24 - r23
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_ccm2: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = -d24/sm3 + r34/sm2 - r23/sm4
      cc = (sm4/sm2 - r24)/(sm3*sm4)
c      hh = dreal(r23 - r24*r34)
c      dd = avh_olo_sqrt( bb*bb - four*aa*cc , -dreal(aa)*hh )
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q23,i23 ,r23,-1d0)
      call avh_olo_conv( q24,i24 ,r24,-1d0)
      call avh_olo_conv( q34,i34 ,r34,-1d0)
*
*
      call avh_olo_rat( qy1,iy1 ,qx1,ix1 ,q24,i24 )
      call avh_olo_rat( qy2,iy2 ,qx2,ix2 ,q24,i24 )
*
      call avh_olo_prd( qz1,iz1 ,qy1,iy1 ,sm2,0 )
      call avh_olo_prd( qz2,iz2 ,qy2,iy2 ,sm2,0 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) / r24 * sm2
      rslt(0) = trm
*
      if (x2.ne.zero) then ! better to put a threshold on cc 
        call avh_olo_rat( qz1,iz1 ,qy1,iy1 ,sm2,0 )
        call avh_olo_rat( qz2,iz2 ,qy2,iy2 ,sm2,0 )
        call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 )
        call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 )
        trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc(  qy2,iy2 )*half/x2
        rslt(0) = rslt(0) + trm
        call avh_olo_rat( qz1,iz1 ,qx1,ix1 ,sm4,0 )
        call avh_olo_rat( qz2,iz2 ,qx2,ix2 ,sm4,0 )
        call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 )
        call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 )
        trm = avh_olo_logc2( qy1,iy1 )*avh_olo_logc(  qy2,iy2 )*half/x2
        rslt(0) = rslt(0) - trm
      endif
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,sm4,0 )
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,sm4,0 )
      trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * sm4
      rslt(0) = rslt(0) - trm
*
*
      if (r23.ne.zero) then
        call avh_olo_prd( qss,iss ,q23,i23 ,sm3,0 )
        call avh_olo_div( qss,iss ,q24,i24 )
        call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
        call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
        trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r23 * sm3 / r24
        rslt(0) = rslt(0) - trm
      endif
*
      if (r34.ne.zero) then
        call avh_olo_prd( qss,iss ,q34,i34 ,sm3,0 )
        call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qss,iss )
        call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qss,iss )
        trm = avh_olo_li2c2( qz1,iz1 ,qz2,iz2 ) * r34 * sm3
        rslt(0) = rslt(0) + trm
      endif
*
      rslt(0) = rslt(0) / (aa*sm2*sm3*sm4)
      rslt(1) = zero
      rslt(2) = zero
      end


      subroutine avh_olo_ccm3( rslt ,p1i,p2i,p3i ,m1i,m2i,m3i )
*  ********************************************************************
*  * Finite 1-loop scalar 3-point function with all internal masses
*  * non-zero. Obtained from the formulas for 4-point functions in
*  * A. Denner, U. Nierste, R. Scharf, Nucl.Phys.B367(1991)637-656
*  * by sending one internal mass to infinity.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1i,p2i,p3i,m1i,m2i,m3i
     &,p1,p2,p3,m1,m2,m3
     &,sm1,sm2,sm3 ,aa,bb,cc,dd,x1,qx1,x2,qx2,ss,qss,tt,qtt,trm
     &,q12,q13,q23,r12,r13,r23,d12,d13,d23 ,zero,one,four ,avh_olo_sqrt
     &,oieps
      parameter(zero=(0d0,0d0),one=(1d0,0d0),four=(4d0,0d0))
      double precision
     & h1,h2,h3,avh_olo_prec
      integer
     & i12,i13,i23,ix1,ix2,iss,itt ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_ccm3: you are calling me' !DEBUG
*
      h1 = -dimag(m1i)
      h2 = -dimag(m2i)
      h3 = -dimag(m3i)
      if (h2.ge.h1.and.h2.ge.h3) then
        p1 = p3i
        p2 = p1i
        p3 = p2i
        m1 = m3i
        m2 = m1i
        m3 = m2i
      else
        p1 = p1i
        p2 = p2i
        p3 = p3i
        m1 = m1i
        m2 = m2i
        m3 = m3i
      endif
*
      sm1 = avh_olo_sqrt(m1,-1d0)
      sm2 = avh_olo_sqrt(m2,-1d0)
      sm3 = avh_olo_sqrt(m3,-1d0)
*
      oieps = dcmplx(1d0,avh_olo_prec()**2)
      q12 = zero
      q13 = zero
      q23 = zero
      if (m1+m2.ne.p1) q12 = ( m1+m2-p1*oieps )/(sm1*sm2) ! p1
      if (m1+m3.ne.p3) q13 = ( m1+m3-p3*oieps )/(sm1*sm3) ! p1+p2 => p12
      if (m2+m3.ne.p2) q23 = ( m2+m3-p2*oieps )/(sm2*sm3) ! p2
*
      call avh_olo_rfun( r12,d12 ,q12 )
      call avh_olo_rfun( r13,d13 ,q13 )
      call avh_olo_rfun( r23,d23 ,q23 )
*
      aa = sm2/sm3 - q23 + r13*(q12 - sm2/sm1)
*
      if (aa.eq.zero) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_ccm3: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      bb = d13/sm2 + q12/sm3 - q23/sm1
      cc = ( sm1/sm3 - one/r13 )/(sm1*sm2)
c      hh = dreal( (r13-sm1/sm3)/(sm1*sm2) )
c      dd = avh_olo_sqrt( bb*bb - four*aa*cc , -dreal(aa)*hh )
      call avh_olo_abc(x1,x2,dd ,aa,bb,cc ,0)
      x1 = -x1
      x2 = -x2
*
      call avh_olo_conv( qx1,ix1 ,x1 ,1d0 ) ! x1 SHOULD HAVE im. part
      call avh_olo_conv( qx2,ix2 ,x2 ,1d0 ) ! x2 SHOULD HAVE im. part
      call avh_olo_conv( q12,i12 ,r12,-1d0) ! RE-USING q12
      call avh_olo_conv( q13,i13 ,r13,-1d0) ! RE-USING q13
      call avh_olo_conv( q23,i23 ,r23,-1d0) ! RE-USING q23
*
       ss = r12
      qss = q12
      iss = i12
       tt = sm2
      qtt = sm2
      itt = 0
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = trm
*
       ss = r23
      qss = q23
      iss = i23
      call avh_olo_prd( qtt,itt ,q13,i13 ,sm2,0 )
      if (itt.eq.itt/2*2) then
        tt = qtt
      else
        tt = -qtt
      endif
      call avh_olo_cdm4_h(trm ,qx1,ix1 ,qx2,ix2 ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
       ss = sm3
      qss = ss
      iss = 0
       tt = r13
      qtt = q13
      itt = i13
      call avh_olo_ccm3_h(trm    ,qx1,ix1 ,x2,qx2,ix2
     &                        ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) + trm
*
       ss = sm1
      qss = ss
      iss = 0
       tt = one
      qtt = one
      itt = 0
      call avh_olo_ccm3_h(trm    ,qx1,ix1 ,x2,qx2,ix2
     &                        ,ss,qss,iss ,tt,qtt,itt)
      rslt(0) = rslt(0) - trm
*
      rslt(0) = rslt(0) / (aa*sm1*sm2*sm3)
      rslt(1) = zero
      rslt(2) = zero
      end

      subroutine avh_olo_ccm3_h(trm    ,qx1,ix1 ,x2,qx2,ix2
     &                              ,ss,qss,iss ,tt,qtt,itt)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex trm    ,qx1 ,x2,qx2 ,ss,qss ,tt,qtt
     &,qz1,qz2 ,qy1,qy2 ,half ,zero
     &,avh_olo_li2c2,avh_olo_logc2,avh_olo_logc
      parameter(half=(0.5d0,0d0),zero=(0d0,0d0))
      integer ix1 ,ix2 ,iss ,itt
     &,iz1,iz2 ,iy1,iy2
*
      call avh_olo_prd( qz1,iz1 ,qx1,ix1 ,qtt,itt ) ! z1 = x1*t
      call avh_olo_prd( qz2,iz2 ,qx2,ix2 ,qtt,itt ) ! z2 = x2*t
*
      call avh_olo_prd( qy1,iy1 ,qz1,iz1 ,qss,iss ) ! y1 = z1*s = x1*t*s
      call avh_olo_prd( qy2,iy2 ,qz2,iz2 ,qss,iss ) ! y2 = z2*s = x2*t*s
*
      trm = avh_olo_li2c2( qy1,iy1 ,qy2,iy2 ) * tt * ss
*
      if (x2.ne.zero) then
        call avh_olo_div( qz1,iz1 ,qss,iss ) ! z1 <- z1/s = x1*t/s
        call avh_olo_div( qz2,iz2 ,qss,iss ) ! z2 <- z2/s = x2*t/s
*
        call avh_olo_rat( qy1,iy1 ,qz1,iz1 ,qz2,iz2 ) ! y1 = z1/z2
        call avh_olo_prd( qy2,iy2 ,qz1,iz1 ,qz2,iz2 ) ! y2 = z1*z2
*
        trm = trm +  avh_olo_logc2( qy1,iy1 )
     &              *avh_olo_logc(  qy2,iy2 ) * half / x2
      endif
      end
