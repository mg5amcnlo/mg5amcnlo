************************************************************************
* This is the file  avh_oni_3fin.f  of the package                     *
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


      subroutine avh_oni_cam(rslt ,mm ,zmu)
*  ********************************************************************
*  * The 1-loop scalar 1-point function.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,mm,zmu
     &,qmm,xx,zero ,avh_oni_logc
      parameter(zero=(0d0,0d0))
      integer
     & imm,ix
*
c      write(6,*) 'MESSAGE from avh_oni_cam: you are calling me' !DEBUG
*
      rslt(2) = zero
      if (mm.eq.zero) then
        rslt(1) = zero
        rslt(0) = zero
      else
        call avh_oni_conv( qmm,imm ,mm,-1d0 )
        call avh_oni_rat( xx,ix ,qmm,imm ,zmu,0 )
        rslt(1) = mm
        rslt(0) = mm - mm*avh_oni_logc( xx,ix )
      endif
      end


      subroutine avh_oni_cbm(rslt ,pp,m1,m2 ,zmu)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,pp,m1,m2,zmu
*********************************************
      double complex          aa,bb,cc     !*
      double precision        ra2,rb,gplus !*
      integer                 order        !*
      common/avh_oni_cbm_com/ aa,bb,cc     !*
     &                       ,ra2,rb,gplus !*
     &                       ,order        !*
*********************************************
      external avh_oni_cbm_i
      double complex avh_oni_cbm_i
      double complex z0,z1,zero
      parameter( zero=(0d0,0d0) )
      if (m1.eq.zero.and.m2.eq.zero.and.pp.eq.zero) then
        rslt(1) = zero     
        rslt(0) = zero     
        return
      endif
      if (cdabs(m2).gt.cdabs(m1)) then !CDABSyes
        aa = m1
        bb = m1+m2-pp
        cc = m2
      else
        aa = m2
        bb = m1+m2-pp
        cc = m1
      endif
      ra2 = dreal(aa)*2d0
      rb  = dreal(bb)
* This value is used to restrain the contour deformation
      gplus = -dimag(aa)
      if (gplus.ne.0d0) then
        gplus = 1d0/gplus
        if (gplus.gt.1d0) gplus = 1d0
      else
        gplus = 1d0
      endif
      z0 = dcmplx(1d0)
      order = 1
      call avh_oni_1dim( z1 ,avh_oni_cbm_i ,'cbm   ' )
      rslt(2) = dcmplx(0d0)
      rslt(1) = z0
      rslt(0) = z1 + cdlog(zmu)*z0 !CDLOGyes
      end

      double complex function avh_oni_cbm_i( xx_in )
*  ********************************************************************
*  * ndim is input, but should be 1
*  * xx is input, and should have dimension at least ndim
*  * ncomp is input, but should be 2
*  * ff is output, and should have dimension at least ncomp
*  ********************************************************************
      implicit none
      double precision xx_in
*********************************************
      double complex          aa,bb,cc     !*
      double precision        ra2,rb,gplus !*
      integer                 order        !*
      common/avh_oni_cbm_com/ aa,bb,cc     !*
     &                       ,ra2,rb,gplus !*
     &                       ,order        !*
*********************************************
      double precision xx,ww,tt,tp,yy,alpha
      parameter( alpha=1d0 )
      double complex dd,zz,vv,one
      parameter( one=(1d0,0d0) )
* Mapping to the full positive axis
      xx = 1d0/xx_in - 1d0
      ww = (1d0+xx)**2
* Function 0 at 0 and 1 at infinity, and its derivative
      tt = xx/(alpha+xx)
      tp = alpha/(alpha+xx)**2
* Imaginary part of the deformed contour (still without tt)
      yy = -(ra2*xx+rb)
* Jacobian matrix dz/dx from contour deformation
      dd = dcmplx(1d0,tp*yy-tt*ra2)
* Deformed contour
      zz = dcmplx(xx,tt*yy)
* Integrand
      vv = zz*(aa*zz + bb) + cc
      if (dimag(vv).gt.0d0) then
        write(6,*) 'ERROR in avh_oni_cbm: Im(V) positive'
      endif
      if (order.eq.1) then
        vv = dcmplx(ww)*dd/(one+zz)**2*( 2*cdlog(one+zz) - cdlog(vv) ) !CDLOGyes
      endif
      avh_oni_cbm_i = vv
      end
        

      subroutine avh_oni_cfam( rslt ,p1,p2,p3 ,m1,m2,m3 ,smax )
*  ********************************************************************
*  * Calculates the finite 1-loop scalar 3-point function with
*  * imaginary parts of the squared masses non-positive, and all
*  * external momenta real.
*  *
*  *               1   /               d^4q
*  *            ------ | ---------------------------------------
*  *            i*pi^2 / [q^2-m1] [(q+k1)^2-m2] [(q+k1+k2)^2-m3]
*  *
*  * input:  p1=k1^2, p2=k2^2, p3=(k1+k2)^2,  m1,m2,m3=squared masses
*  *         uunit: if you put  uunit>0  then estimated integration
*  *         errors and the number of integrand evaluations will be
*  *         send to  unit=uunit
*  *          
*  * output: rslt(0) = value
*  *         rslt(1) = 0
*  *         rslt(2) = 0
*  *
*  * Numerically integrates the 2-dim integrand obtained from the
*  * usual 3-dim Feynman-parametrization by scaling 2 integration
*  * variables with the third before integrating this third one out:
*  *
*  *   d^3x f(x1,x2,x3) dirac(1-x1-x2-x3)
*  *            --> d^2x dy y^2 f(y*x1,y*x2,y) dirac(1-y*(1+x1+x2))
*  *
*  * The result is an integrand living on [0,infinity)^2, for which
*  * the integration contour can be deformed without bothering about
*  * the endpoints at infinity.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,m1,m2,m3
     &,p1i,p2i,p3i,m1i,m2i,m3i,zz
      double precision smax
     &,h1,h2,h3,small
      external avh_oni_cfam_i
***************************************************************
      double complex           q11,q12,q13,q22,q23,q33       !*
      double precision         r11,r12,r13,r22,r23,r33,gplus !*
      common/avh_oni_cfam_com/ q11,q12,q13,q22,q23,q33       !*
     &                        ,r11,r12,r13,r22,r23,r33,gplus !*
***************************************************************
*
      small = 1d-16*smax
*
      h1 = cdabs(m1) !CDABSyes
      h2 = cdabs(m2) !CDABSyes
      h3 = cdabs(m3) !CDABSyes
      if (h3.gt.h1.and.h3.gt.h2) then
!      if (h3.lt.h1.and.h3.lt.h2) then
        p1i = p1
        p2i = p2
        p3i = p3
        m1i = m1
        m2i = m2
        m3i = m3
      elseif (h2.gt.h1.and.h2.gt.h3) then
!      elseif (h2.lt.h1.and.h2.lt.h3) then
        p1i = p3
        p2i = p1
        p3i = p2
        m1i = m3
        m2i = m1
        m3i = m2
      else
        p1i = p2
        p2i = p3
        p3i = p1
        m1i = m2
        m2i = m3
        m3i = m1
      endif
      if (cdabs(m1i).lt.small) m1i = dcmplx(small) !CDABSyes
      if (cdabs(m2i).lt.small) m2i = dcmplx(small) !CDABSyes
      if (cdabs(m3i).lt.small) m3i = dcmplx(small) !CDABSyes
      q11 = m1i
      q22 = m2i
      q33 = m3i
      q12 = (m1i+m2i-p1i)/2
      q23 = (m2i+m3i-p2i)/2
      q13 = (m3i+m1i-p3i)/2
      r11 = dreal(q11)
      r12 = dreal(q12)
      r13 = dreal(q13)
      r22 = dreal(q22)
      r23 = dreal(q23)
      r33 = dreal(q33)
      h1 = -dimag(q11)
      h2 = -dimag(q22)
      h3 = -dimag(q33)
* Assuming that -Im(qij)=(gi+gj)/2 , gplus is the maximum eigenvalue
      gplus = h1+h2 + dsqrt( 2d0*(h1*h1+h2*h2) )
* This value is used to restrain the contour deformation
      if (gplus.ne.0d0) then
        gplus = 1d0/gplus
        if (gplus.gt.1d0) gplus = 1d0
      else
        gplus = 1d0
      endif
*
      call avh_oni_cuba( zz ,avh_oni_cfam_i ,2 ,'cfam  ' )
*
      rslt(0) = zz
      rslt(1) = dcmplx(0d0)
      rslt(2) = dcmplx(0d0)
      end

      subroutine avh_oni_cfam_i(ndim ,xx ,ncomp ,ff)
*  ********************************************************************
*  * ndim is input, but should be 2
*  * xx is input, and should have dimension at least ndim
*  * ncomp is input, but should be 2
*  * ff is output, and should have dimension at least ncomp
*  ********************************************************************
      implicit none
      integer ndim,ncomp
      double precision xx(ndim),ff(ncomp)
     &,x1,x2,y1,y2,t1,t2,t1p,t2p,w1,w2,alpha
      double complex
     & d11,d12,d21,d22,z1,z2,vv,det
***************************************************************
      double complex           q11,q12,q13,q22,q23,q33       !*
      double precision         r11,r12,r13,r22,r23,r33,gplus !*
      common/avh_oni_cfam_com/ q11,q12,q13,q22,q23,q33       !*
     &                        ,r11,r12,r13,r22,r23,r33,gplus !*
***************************************************************
      parameter(alpha=1d0)
!      write(*,*) 'MESSAGE from avh_oni_cfam_i: you are calling me' !DEBUG
*
* Mapping to the full positive quadrant
c      pio2 = avh_oni_pi()/2d0
c      x1 = dtan(pio2*xx(1))
c       w1 = (1d0+x1*x1)*pio2
c      x2 = dtan(pio2*xx(2))
c       w2 = (1d0+x2*x2)*pio2
      x1 = 1d0/xx(1)-1d0
      w1 = (1d0+x1)**2
      x2 = 1d0/xx(2)-1d0
      w2 = (1d0+x2)**2
* Functions 0 at 0 and  gplus  at infinity, and their derivatives
      t1  = gplus * x1/(alpha+x1)
      t1p = gplus * alpha/(alpha+x1)**2
      t2  = gplus * x2/(alpha+x2)
      t2p = gplus * alpha/(alpha+x2)**2
* Imaginary part of the deformed contour (still without t{1,2})
      y1 = -(r11*x1 + r12*x2 + r13)
      y2 = -(r12*x1 + r22*x2 + r23)
* Jacobian matrix dz/dx from contour deformation
      d11 =  dcmplx(1d0,t1p*y1-t1*r11)
       d12 = dcmplx(0d0,      -t1*r12)
      d22 =  dcmplx(1d0,t2p*y2-t2*r22)
       d21 = dcmplx(0d0,      -t2*r12)
      det = d11*d22 - d12*d21
* Deformed contour
      z1 = dcmplx(x1,t1*y1)
      z2 = dcmplx(x2,t2*y2)
* Integrand
      vv = z1*(q11*z1 + q12*z2 + 2*q13)
     &   + z2*(q12*z1 + q22*z2 + 2*q23)
     &   + q33
      if (dimag(vv).gt.0d0) then
        write(6,*) 'ERROR in avh_oni_cfam: Im(V) positive'
      endif
      vv = dcmplx(1d0)/( (dcmplx(1d0)+z1+z2)*vv )
* Full weight
      vv = -vv*dcmplx(w1*w2)*det
      ff(1) = dreal(vv)
      ff(2) = dimag(vv)
      end
