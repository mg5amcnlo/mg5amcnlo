************************************************************************
* This is the file  avh_oni_real.f  of the package                     *
*                                                                      *
*                  Oneloop with Numerical Integration                  *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 28-07-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_oni_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_oni_a0m( rslt ,mm )
*  ********************************************************************
*  *
*  *           C   / d^(Dim)q
*  * rslt = ------ | --------
*  *        i*pi^2 / (q^2-mm)
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
*  *
*  * input:  mm = mass squared
*  * output: rslt(0) = eps^0   -coefficient
*  *         rslt(1) = eps^(-1)-coefficient
*  *         rslt(2) = eps^(-2)-coefficient
*  *
*  * The input value (mm) should be real.
*  * If this number is IDENTICALLY 0d0, the IR-divergent case is
*  * returned.
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
      double precision mm
*
      call avh_oni_a0c( rslt ,dcmplx(mm) )
      end


      subroutine avh_oni_b0m( rslt ,pp,m1,m2 )
*  ********************************************************************
*  *
*  *           C   /      d^(Dim)q
*  * rslt = ------ | --------------------
*  *        i*pi^2 / [q^2-m1][(q+k)^2-m2]
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
*  *
*  * input:  pp = k^2, m1,m2 = mass squared
*  * output: rslt(0) = eps^0   -coefficient
*  *         rslt(1) = eps^(-1)-coefficient
*  *         rslt(2) = eps^(-2)-coefficient
*  *
*  * The input values (pp,m1,m2) should be real.
*  * If these numbers are IDENTICALLY 0d0, IR-divergent cases are
*  * returned.
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
      double precision pp,m1,m2
*
      call avh_oni_b0c( rslt ,dcmplx(pp),dcmplx(m1),dcmplx(m2) )
      end


      subroutine avh_oni_c0m( rslt ,p1,p2,p3 ,m1,m2,m3 )
*  ********************************************************************
*  * calculates
*  *               C   /               d^(Dim)q
*  *            ------ | ---------------------------------------
*  *            i*pi^2 / [q^2-m1] [(q+k1)^2-m2] [(q+k1+k2)^2-m3]
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps)
*  *             * GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
*  *
*  * input:  p1=k1^2, p2=k2^2, p3=(k1+k2)^2,  m1,m2,m3=squared masses
*  * output: rslt(0) = eps^0   -coefficient
*  *         rslt(1) = eps^(-1)-coefficient
*  *         rslt(2) = eps^(-2)-coefficient
*  *
*  * The input values should be real.
*  * IR-divergent cases are returned ONLY if the appropriate subset
*  * of these numbers are IDENTICAL and/or IDENTICALLY 0d0.
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
     &,cp1,cp2,cp3,cm1,cm2,cm3
      double precision p1,p2,p3,m1,m2,m3
*
      cp1 = dcmplx(p1)
      cp2 = dcmplx(p2)
      cp3 = dcmplx(p3)
      cm1 = dcmplx(m1)
      cm2 = dcmplx(m2)
      cm3 = dcmplx(m3)
      call avh_oni_c0c( rslt ,cp1,cp2,cp3 ,cm1,cm2,cm3 )
      end


      subroutine avh_oni_d0m( rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*  ********************************************************************
*  * calculates
*  *
*  *    C   /                      d^(Dim)q
*  * ------ | --------------------------------------------------------
*  * i*pi^2 / [q^2-m1][(q+k1)^2-m2][(q+k1+k2)^2-m3][(q+k1+k2+k3)^2-m4]
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps)
*  *             * GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
*  *
*  * input:  p1=k1^2, p2=k2^2, p3=k3^2, p4=(k1+k2+k3)^2,
*  *         p12=(k1+k2)^2, p23=(k2+k3)^2,
*  *         m1,m2,m3,m4=squared masses
*  * output: rslt(0) = eps^0   -coefficient
*  *         rslt(1) = eps^(-1)-coefficient
*  *         rslt(2) = eps^(-2)-coefficient
*  *
*  * The input values should be real.
*  * IR-divergent cases are returned ONLY if the appropriate subset
*  * of these numbers are IDENTICAL and/or IDENTICALLY 0d0.
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
     &,cp1,cp2,cp3,cp4,cp12,cp23,cm1,cm2,cm3,cm4
      double precision p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
*
      cp1  = dcmplx(p1)
      cp2  = dcmplx(p2)
      cp3  = dcmplx(p3)
      cp4  = dcmplx(p4)
      cp12 = dcmplx(p12)
      cp23 = dcmplx(p23)
      cm1  = dcmplx(m1)
      cm2  = dcmplx(m2)
      cm3  = dcmplx(m3)
      cm4  = dcmplx(m4)
      call avh_oni_d0c( rslt ,cp1,cp2,cp3,cp4,cp12,cp23
     &                       ,cm1,cm2,cm3,cm4 )
      end


      subroutine avh_oni_printall( unit_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer unit_in ,avh_oni_print
      integer                   nunit
      common/avh_oni_print_com/ nunit
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        nunit = avh_oni_print()
      endif
      nunit = unit_in
      end
*
      integer function avh_oni_print() 
      implicit none
      integer                   nunit
      common/avh_oni_print_com/ nunit
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        nunit = 0
      endif
      avh_oni_print = nunit
      end
