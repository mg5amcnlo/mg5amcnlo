************************************************************************
* This is the file  avh_oni_dfam.f  of the package                     *
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

      subroutine avh_oni_dfam( rslt
     &                        ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 ,smax
     &                        ,c0,c2,c3,c4,ct )
*  ********************************************************************
*  ********************************************************************
      implicit none
************************************************************************
      double complex           q11,q12,q13,q14,q22,q23,q24,q33,q34,q44!*
     &                        ,b0,b2,b3,b4                            !*
     &                        ,a11,a12,a13,a14,a22,a23,a24,a33,a34,a44!*
      double precision         r11,r12,r13,r14,r22,r23,r24,r33,r34,r44!*
     &                        ,gplus                                  !*
      integer                  tnsr                                   !*
      common/avh_oni_dfam_com/ q11,q12,q13,q14,q22,q23,q24,q33,q34,q44!*
     &                        ,b0,b2,b3,b4                            !*
     &                        ,a11,a12,a13,a14,a22,a23,a24,a33,a34,a44!*
     &                        ,r11,r12,r13,r14,r22,r23,r24,r33,r34,r44!*
     &                        ,gplus                                  !*
     &                        ,tnsr                                   !*
************************************************************************
      double complex rslt(0:3) ,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
     &                         ,ct ,c2,c3,c4 ,c0
     &,zz,zero
      parameter( zero=(0d0,0d0) )
      double precision smax
     &,small,h1,h2,h3
      character(6)
     & label
      external avh_oni_dfam_i
*
      small = 1d-16*smax
*
      q11 = m1
      q22 = m2
      q33 = m3
      q44 = m4
      if (cdabs(q11).lt.small) q11 = dcmplx(small)
      if (cdabs(q22).lt.small) q22 = dcmplx(small)
      if (cdabs(q33).lt.small) q33 = dcmplx(small)
      if (cdabs(q44).lt.small) q44 = dcmplx(small)
      q12 = (m1+m2-p1 )/2
      q13 = (m1+m3-p12)/2
      q23 = (m2+m3-p2 )/2
      q14 = (m1+m4-p4 )/2
      q24 = (m2+m4-p23)/2
      q34 = (m3+m4-p3 )/2
*
      if (ct.ne.zero) then
        tnsr = 2
      elseif(c2.ne.zero.or.c3.ne.zero.or.c4.ne.zero) then
        tnsr = 1
      else
        tnsr = 0
      endif
*
      b2 = c2
      b3 = c3
      b4 = c4
      b0 = c0
*
      a11 = -ct*( 2*q11     )
      a22 = -ct*( 2*q22-p1  )
      a33 = -ct*( 2*q33-p12 )
      a44 = -ct*( 2*q44-p4  )
      a12 = -ct*( 2*q12     )
      a13 = -ct*( 2*q13     )
      a14 = -ct*( 2*q14     )
      a23 = -ct*( 2*q23-(p12+p1-p2)/2 )
      a24 = -ct*( 2*q24-(p1+p4-p23)/2 )
      a34 = -ct*( 2*q34-(p12+p4-p3)/2 )
*
      r11 = dreal(q11)
      r12 = dreal(q12)
      r13 = dreal(q13)
      r14 = dreal(q14)
      r22 = dreal(q22)
      r23 = dreal(q23)
      r24 = dreal(q24)
      r33 = dreal(q33)
      r34 = dreal(q34)
      r44 = dreal(q44)
      h1 = -dimag(q11)
      h2 = -dimag(q22)
      h3 = -dimag(q33)
* Assuming that -Im(qij)=(gi+gj)/2 , gplus is the maximum eigenvalue
      gplus = h1+h2+h3 + dsqrt( 3d0*(h1*h1+h2*h2+h3*h3) )
* This value is used to restrain the contour deformation
      if (gplus.ne.0d0) then
        gplus = 1d0/gplus
        if (gplus.gt.1d0) gplus = 1d0
      else
        gplus = 1d0
      endif
*
      label(1:6) = 'dfam  '
      write(label(6:6),'(i1.1)') tnsr
      call avh_oni_cuba( zz ,avh_oni_dfam_i ,3 ,label )
*
      rslt(0) = zz
      rslt(1) = dcmplx(0d0)
      rslt(2) = dcmplx(0d0)
      end

      subroutine avh_oni_dfam_i( ndim ,xx ,ncomp ,ff )
*  ********************************************************************
*  * ndim is input, but should be 3
*  * xx is input, and should have dimension at least ndim
*  * ncomp is input, but should be 2
*  * ff is output, and should have dimension at least ncomp
*  ********************************************************************
      implicit none
      integer ndim,ncomp
      double precision xx(ndim),ff(ncomp)
************************************************************************
      double complex           q11,q12,q13,q14,q22,q23,q24,q33,q34,q44!*
     &                        ,b0,b2,b3,b4                         !*
     &                        ,a11,a12,a13,a14,a22,a23,a24,a33,a34,a44!*
      double precision         r11,r12,r13,r14,r22,r23,r24,r33,r34,r44!*
     &                        ,gplus                                  !*
      integer                  tnsr                                   !*
      common/avh_oni_dfam_com/ q11,q12,q13,q14,q22,q23,q24,q33,q34,q44!*
     &                        ,b0,b2,b3,b4                         !*
     &                        ,a11,a12,a13,a14,a22,a23,a24,a33,a34,a44!*
     &                        ,r11,r12,r13,r14,r22,r23,r24,r33,r34,r44!*
     &                        ,gplus                                  !*
     &                        ,tnsr                                   !*
************************************************************************
      double precision
     & x1,x2,x3,y1,y2,y3,t1,t2,t3,t1p,t2p,t3p,w1,w2,w3,alpha
      double complex bb,ee,vv,ww
     &,d11,d12,d13,d21,d22,d23,d31,d32,d33,z1,z2,z3,det,one,aa
*
      parameter( one=(1d0,0d0) )
      parameter( alpha=1d0 )
*
* Mapping to the full positive octant
      x1 = 1d0/xx(1)-1d0
      w1 = (1d0+x1)**2
      x2 = 1d0/xx(2)-1d0
      w2 = (1d0+x2)**2
      x3 = 1d0/xx(3)-1d0
      w3 = (1d0+x3)**2
* Functions 0 at 0 and  gplus  at infinity, and their derivatives
      t1  = gplus * x1/(alpha+x1)
      t1p = gplus * alpha/(alpha+x1)**2
      t2  = gplus * x2/(alpha+x2)
      t2p = gplus * alpha/(alpha+x2)**2
      t3  = gplus * x3/(alpha+x3)
      t3p = gplus * alpha/(alpha+x3)**2
* Imaginary part of the deformed contour (still without t{1,2,3})
      y1 = -(r11*x1 + r12*x2 + r13*x3 + r14)
      y2 = -(r12*x1 + r22*x2 + r23*x3 + r24)
      y3 = -(r13*x1 + r23*x2 + r33*x3 + r34)
* Jacobian matrix dz/dx from contour deformation
      d11 =  dcmplx(1d0,t1p*y1-t1*r11)
       d12 = dcmplx(0d0,      -t1*r12)
       d13 = dcmplx(0d0,      -t1*r13)
      d22 =  dcmplx(1d0,t2p*y2-t2*r22)
       d21 = dcmplx(0d0,      -t2*r12)
       d23 = dcmplx(0d0,      -t2*r23)
      d33 =  dcmplx(1d0,t3p*y3-t3*r33)
       d31 = dcmplx(0d0,      -t3*r13)
       d32 = dcmplx(0d0,      -t3*r23)
      det = d11*(d22*d33-d23*d32)
     &    - d12*(d21*d33-d23*d31)
     &    + d13*(d21*d32-d22*d31)
* Deformed contour
      z1 = dcmplx(x1,t1*y1)
      z2 = dcmplx(x2,t2*y2)
      z3 = dcmplx(x3,t3*y3)
* Denominator
      vv = z1*(q11*z1 + q12*z2 + q13*z3 + 2*q14)
     &   + z2*(q12*z1 + q22*z2 + q23*z3 + 2*q24)
     &   + z3*(q13*z1 + q23*z2 + q33*z3 + 2*q34)
     &   + q44
      if (dimag(vv).gt.0d0) then
        write(6,*) 'ERROR in avh_oni_dfam_i: Im(V) positive'
      endif
* Jacobian
      ww = dcmplx(w1*w2*w3)*det
* Integrand
      if     (tnsr.eq.0) then
        vv = ww/(vv*vv)
      elseif (tnsr.eq.1) then
        ee = z1 + z2 + z3 + one
        bb = b2*z2 + b3*z3 + b4
        vv = ww*bb/(ee*vv*vv)
      else!if(tnsr.eq.2) then
        ee = z1 + z2 + z3 + one
        bb = b2*z2 + b3*z3 + b4
        aa = z1*(a11*z1 + a12*z2 + a13*z3 + 2*a14)
     &     + z2*(a12*z1 + a22*z2 + a23*z3 + 2*a24)
     &     + z3*(a13*z1 + a23*z2 + a33*z3 + 2*a34)
     &     + a44
        vv = ww*( aa + (b0*ee-bb)*ee )/(ee*ee*vv*vv)
      endif
*
      ff(1) = dreal(vv)
      ff(2) = dimag(vv)
      end
