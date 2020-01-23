************************************************************************
* This is the file  avh_oni_3div.f  of the package                     *
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


      subroutine avh_oni_c0m4(rslt ,pp,m2,m3 ,zmu)
*  ********************************************************************
*  * calculates
*  *               C   /             d^(Dim)q
*  *            ------ | ----------------------------------
*  *            i*pi^2 / q^2 [(q+k1)^2-m2] [(q+k1+k2)^2-m3]
*  *
*  * with  k1^2=m2, k2^2=pp, (k1+k2)^2=m3.
*  * m2,m3 should NOT be identically 0d0.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,m2,m3,pp ,zmu
******************************************
      double complex           aa,bb,cc !*
      double precision         ra2,rb   !*
      integer                  order    !*
      common/avh_oni_c0m4_com/ aa,bb,cc !*
     &                        ,ra2,rb   !*
     &                        ,order    !*
******************************************
      external avh_oni_c0m4_i
      double complex avh_oni_c0m4_i
      double complex z0,z1
      if (cdabs(m3).gt.cdabs(m2)) then !CDABSyes
        aa = m2
        bb = m2+m3-pp
        cc = m3
      else
        aa = m3
        bb = m2+m3-pp
        cc = m2
      endif
      ra2 = dreal(aa)*2d0
      rb  = dreal(bb)
      order = 0
      call avh_oni_1dim( z0 ,avh_oni_c0m4_i ,'c0m4_0' )
      order = 1
      call avh_oni_1dim( z1 ,avh_oni_c0m4_i ,'c0m4_1' )
      rslt(2) = dcmplx(0d0)
      rslt(1) = z0/2
      rslt(0) = ( z1 + z0*cdlog(zmu) )/2 !CDLOGyes
      end

      double complex function avh_oni_c0m4_i( xx_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision xx_in
******************************************
      double complex           aa,bb,cc !*
      double precision         ra2,rb   !*
      integer                  order    !*
      common/avh_oni_c0m4_com/ aa,bb,cc !*
     &                        ,ra2,rb   !*
     &                        ,order    !*
******************************************
      double precision xx,ww,tt,tp,yy,alpha
      parameter( alpha=1d0 )
      double complex dd,zz,vv,one
      parameter( one=(1d0,0d0) )
!      write(*,*) 'MESSAGE from avh_oni_c0m4_i: you are calling me' !DEBUG
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
      if (order.eq.0) then
        vv = dcmplx(ww)*dd/vv
      else
        vv = dcmplx(ww)*dd/vv*( 2*cdlog(one+zz)-cdlog(vv) ) !CDLOGyes
      endif
      avh_oni_c0m4_i = vv
      end
        

      subroutine avh_oni_c0m3(rslt ,p2,p3,m3 ,zmu ,smax)
*  ********************************************************************
*  * calculates
*  *               C   /          d^(Dim)q
*  *            ------ | -----------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
*  *
*  * with  p2=k2^2, p3=(k1+k2)^2.
*  * mm should NOT be identically 0d0,
*  * and p2 NOR p3 should be identical to mm. 
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p2,p3,m3,zmu
      double precision smax
******************************************
      double complex           aa,bb,cc !*
      integer                  order    !*
      common/avh_oni_c0m3_com/ aa,bb,cc !*
     &                        ,order    !*
******************************************
      external avh_oni_c0m3_2,avh_oni_c0m3_1
      double complex avh_oni_c0m3_1
      double complex z0,z1,z2
      aa = m3
      bb = m3-p2
      cc = m3-p3
      if (cdabs(aa).gt.1d-16*smax) then !CDABSyes
        order = 0
        call avh_oni_cuba( z0 ,avh_oni_c0m3_2 ,2 ,'c0m3_2' )
      else
        z0 = dcmplx(0d0)
      endif
      order = 0
      call avh_oni_1dim( z1 ,avh_oni_c0m3_1 ,'c0m3_0' )
      order = 1
      call avh_oni_1dim( z2 ,avh_oni_c0m3_1 ,'c0m3_1' )
      rslt(2) = dcmplx(0d0)
      rslt(1) = z1
      rslt(0) = z1*cdlog(zmu) - z0 + z2 !CDLOGyes
      end

      subroutine avh_oni_c0m3_2( ndim ,xx ,ncomp ,ff )
*  ********************************************************************
*  * ndim is input, but should be 1
*  * xx is input, and should have dimension at least ndim
*  * ncomp is input, but should be 2
*  * ff is output, and should have dimension at least ncomp
*  ********************************************************************
      implicit none
      integer ndim,ncomp
      double precision xx(ndim),ff(ncomp)
******************************************
      double complex           aa,bb,cc !*
      integer                  order    !*
      common/avh_oni_c0m3_com/ aa,bb,cc !*
     &                        ,order    !*
******************************************
      double precision x1,w1,x2,w2,y1,y2
      double complex vv,one,d11,d22,det,z1,z2
      parameter( one=(1d0,0d0) )
* Mapping to the full positive quadrant
      x1 = 1d0/xx(1)-1d0
      w1 = (1d0+x1)**2
      x2 = 1d0/xx(2)-1d0
      w2 = (1d0+x2)**2
* Imaginary part of the deformed contour
      y1 = -dreal(aa)*x1
      y2 = -dreal(bb)*x2
* Jacobian matrix dz/dx from contour deformation
      d11 =  dcmplx(1d0,-dreal(aa))
      d22 =  dcmplx(1d0,-dreal(bb))
      det = d11*d22
* Deformed contour
      z1 = dcmplx(x1,y1)
      z2 = dcmplx(x2,y2)
* Integrand ( 1/(a*z1+b*z2+c) - 1/(b*z2+c) )/( z1*(1+z1+z2) )
      if (order.eq.0) then
        vv = -aa/( (one+z1+z2)*(aa*z1+bb*z2+cc)*(bb*z2+cc) )
        vv = dcmplx(w1*w2)*det*vv
      endif
      ff(1) = dreal(vv)
      ff(2) = dimag(vv)
      end

      double complex function avh_oni_c0m3_1( xx_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision xx_in
******************************************
      double complex           aa,bb,cc !*
      integer                  order    !*
      common/avh_oni_c0m3_com/ aa,bb,cc !*
     &                        ,order    !*
******************************************
      double precision xx,ww,yy
      double complex vv,one,det,zz
      parameter( one=(1d0,0d0) )
* Mapping to the full positive quadrant
      xx = 1d0/xx_in-1d0
      ww = (1d0+xx)**2
* Imaginary part of the deformed contour
      yy = -dreal(bb)*xx
* Jacobian matrix dz/dx from contour deformation
      det =  dcmplx(1d0,-dreal(bb))
* Deformed contour
      zz = dcmplx(xx,yy)
* Integrand 
      vv = (bb*zz+cc)*(one+zz)
      if (order.eq.0) then
        vv = dcmplx(ww)*det/vv
      else
        vv = dcmplx(ww)*det/vv*( cdlog(one+zz)-cdlog(bb*zz+cc) ) !CDLOGyes
      endif
      avh_oni_c0m3_1 = vv
      end


      subroutine avh_oni_c0m2(rslt ,p3,m3 ,zmu)
*  ********************************************************************
*  * calculates
*  *               C   /          d^(Dim)q
*  *            ------ | -----------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
*  *
*  * with  k1^2 = 0 , k2^2 = m3  and  (k1+k2)^2 = p3.
*  * mm should NOT be identically 0d0,
*  * and pp should NOT be identical to mm. 
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p3,m3,zmu
********************************************
      double complex           aa,bb,logb !*
      integer                  order      !*
      common/avh_oni_c0m2_com/ aa,bb,logb !*
     &                        ,order      !*
********************************************
      external avh_oni_c0m2_i
      double complex avh_oni_c0m2_i
      double complex logmu2,cc,pi2d12,z0,z1
      double precision avh_oni_pi
      pi2d12 = dcmplx( avh_oni_pi()**2/12 )
*
      aa = m3
      bb = m3-p3
      logb = cdlog( bb - dcmplx(0d0,1d-16*cdabs(bb)) ) !CDLOGyes !CDABSyes
      order = 0
      call avh_oni_1dim( z0 ,avh_oni_c0m2_i ,'c0m2_0' )
      order = 1
      call avh_oni_1dim( z1 ,avh_oni_c0m2_i ,'c0m2_1' )
      logmu2 = cdlog(zmu) !CDLOGyes
      cc     = dcmplx(1d0)/(2*bb)
      rslt(2) = -cc
      rslt(1) = cc*( z0*bb + logb - logmu2 )
      rslt(0) = cc*( logmu2*(z0*bb+logb) + z1*bb + pi2d12
     &                            - (logb**2 + logmu2**2)/2 )
      end

      double complex function avh_oni_c0m2_i( xx_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision xx_in
********************************************
      double complex           aa,bb,logb !*
      integer                  order      !*
      common/avh_oni_c0m2_com/ aa,bb,logb !*
     &                        ,order      !*
********************************************
      double precision xx,ww,yy
      double complex vv,one,det,zz,cc,logc,zero
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
* Mapping to the full positive quadrant
      xx = 1d0/xx_in-1d0
      ww = (1d0+xx)**2
* Imaginary part of the deformed contour
      yy = -dreal(aa)*xx
* Jacobian matrix dz/dx from contour deformation
      det =  dcmplx(1d0,-dreal(aa))
* Deformed contour
      zz = dcmplx(xx,yy)
* Integrand 
      if (order.eq.0) then
        vv = (bb-aa)/( bb*(one+zz)*(aa*zz+bb) )
        vv = dcmplx(ww)*det*vv
      else
        if (zz.eq.zero) zz = dcmplx(1d-16,-1d-16)
        cc = (aa*zz+bb)/bb
        logc = cdlog(cc) !CDLOGyes
        vv = ( 2*cdlog(one+zz)-cdlog(zz) - logb )*(bb-aa) !CDLOGyes
     &     - bb*logc - aa*logc/(cc-one)
        vv = dcmplx(ww)*det*vv/( bb*(one+zz)*(aa*zz+bb) )
      endif
      avh_oni_c0m2_i = vv
      end


      subroutine avh_oni_c0m1(rslt ,cm3 ,zmu)
*  ********************************************************************
*  * calculates
*  *               C   /          d^(Dim)q
*  *            ------ | -----------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 [(q+k1+k2)^2-m3]
*  *
*  * with  k1^2 = (k1+k2)^2 = m3.
*  * mm should NOT be identically 0d0.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,cm3,zmu
     &,avh_oni_logc,zm,xx,qm3
      integer
     & ix,im3
*
c      write(6,*) 'MESSAGE from avh_oni_c0m1: you are calling me'
*
      call avh_oni_conv( qm3,im3 ,cm3,-1d0 )
      call avh_oni_rat( xx,ix ,qm3,im3 ,zmu,0 )
      zm = dcmplx(0.5d0)/cm3
      rslt(2) = dcmplx(0d0)
      rslt(1) = -zm
      rslt(0) = zm*( dcmplx(2d0) + avh_oni_logc(xx,ix) )
      end


      subroutine avh_oni_c0(valc0,p1,p2,p3)
*  ********************************************************************
*  * calculates
*  *               C   /         d^(Dim)q
*  *            ------ | ------------------------
*  *            i*pi^2 / q^2 (q+k1)^2 (q+k1+k2)^2
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
*  *
*  * input:  p1 = k1^2,  p2 = k2^2,  p3 = k3^2
*  * output: valc0(0) = eps^0   -coefficient
*  *         valc0(1) = eps^(-1)-coefficient
*  *         valc0(2) = eps^(-2)-coefficient
*  *
*  * The input values (p1,p2,p3) should be real.
*  * If any of these numbers is IDENTICALLY 0d0, the corresponding
*  * IR-singular case is returned.
*  ********************************************************************
      implicit none
      double complex valc0(0:2)
     &,pi2,zero,log3,avh_oni_loga
      parameter( zero=(0d0,0d0) )
      double precision p1,p2,p3
     &,mu2,pp(3),ap(3),hmax,smax,avh_oni_mu_get,avh_oni_pi,avh_oni_thrs
      integer
     & icase,ii,base(3),per(3),imax,i1,i2,i3 ,avh_oni_un_get
      character(2) label(3)
      logical init ,avh_oni_os_get
      data init/.true./,base/4,2,1/,label/'p1','p2','p3'/
      save init,pi2
*
      if (init) then
        init = .false.
        pi2  = dcmplx( avh_oni_pi()**2 )
      endif
      mu2 = avh_oni_mu_get()**2
*
      pp(1) = p1
      pp(2) = p2
      pp(3) = p3
      ap(1) = dabs(pp(1))
      hmax = ap(1)
      imax = 1
      do ii=2,3
        ap(ii) = dabs(pp(ii))
        if (ap(ii).gt.hmax) then
          hmax = ap(ii)
          imax = ii
        endif
      enddo
      smax = avh_oni_thrs(hmax)
*
      if (avh_oni_os_get()) then
        if (ap(1).lt.smax) ap(1) = 0d0
        if (ap(2).lt.smax) ap(2) = 0d0
        if (ap(3).lt.smax) ap(3) = 0d0
      endif
*
      icase = 0
      do ii=1,3
      if (ap(ii).gt.0d0) then
        icase = icase + base(ii)
        if (ap(ii).lt.smax.and.avh_oni_un_get().gt.0)
     &    write(avh_oni_un_get(),*)
     &    'WARNING from avh_oni_c0: |',label(ii),'/',label(imax),'| ='
     &   ,ap(ii)/hmax
      endif
      enddo
      call avh_oni_c0per(icase,per)
*
      i1 = 0
      i2 = 0
      i3 = 0
      if (-pp(per(1)).lt.0d0) i1 = -1
      if (-pp(per(2)).lt.0d0) i2 = -1
      if (-pp(per(3)).lt.0d0) i3 = -1
*
      if     (icase.eq.0) then
* 0 masses non-zero
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_c0: all external masses equal zero'
        stop
      elseif (icase.eq.1) then
* 1 mass non-zero
        log3 = avh_oni_loga( -pp(per(3))/mu2 , i3 )
        valc0(2) = dcmplx( 1d0/pp(per(3)) )
        valc0(1) = -log3/pp(per(3))
        valc0(0) = ( log3**2/2 - pi2/12 )/pp(per(3))
      elseif (icase.eq.2) then
* 2 masses non-zero
        call avh_oni_c0m3( valc0 ,dcmplx(pp(per(2))),dcmplx(pp(per(3)))
     &                           ,zero ,dcmplx(mu2) ,hmax )
      elseif (icase.eq.3) then
* 3 masses non-zero
        call avh_oni_cfam( valc0 ,dcmplx(p1),dcmplx(p2),dcmplx(p3)
     &                           ,zero,zero,zero ,hmax )
      endif
      end


      subroutine avh_oni_c0per(icase,per)
*  ********************************************************************
*  * Go through all possibilities of zero (0) and non-zero (1) masses
*  *
*  *   mass: 123    mass: 123    mass: 123
*  * icase=1 001  icase=3 011  icase=0 000 icase->0
*  * icase=2 010  icase=6 110  icase=7 111 icase->3 
*  * icase=4 100  icase=5 101
*  *   icase->1     icase->2
*  ********************************************************************
      implicit none
      integer icase,per(3)
     &,permtable(3,0:7),casetable(0:7),ii
      data permtable/
     & 1,2,3 ! 0, 0 masses non-zero, no permutation
     &,1,2,3 ! 1, 1 mass non-zero,   no permutation
     &,3,1,2 ! 2, 1 mass non-zero,   1 cyclic permutation
     &,1,2,3 ! 3, 2 masses non-zero, no permutation
     &,2,3,1 ! 4, 1 mass non-zero,   2 cyclic permutations
     &,2,3,1 ! 5, 2 masses non-zero, 2 cyclic permutations
     &,3,1,2 ! 6, 2 masses non-zero, 1 cyclic permutation
     &,1,2,3 ! 7, 3 masses non-zero, no permutation
     &/             ! 0, 1, 2, 3, 4, 5, 6, 7
      data casetable/ 0, 1, 1, 2, 1, 2, 2, 3/
      do ii=1,3
        per(ii) = permtable(ii,icase)
      enddo
      icase = casetable(icase)
      end


      subroutine avh_oni_zero(rslt)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
      rslt(2) = dcmplx(0d0)
      rslt(1) = dcmplx(0d0)
      rslt(0) = dcmplx(0d0)
      end


      subroutine avh_oni_mu_set(mu_in)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision mu_in
     &,mu ,avh_oni_mu_get
      common/avh_oni_mu_com/ mu
      integer
     & init,avh_oni_un_get
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        call avh_oni_hello
        mu = avh_oni_mu_get()
      endif
      mu = mu_in
      if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &  'MESSAGE from avh_oni_mu_set: scale (mu, not mu^2) set to:',mu
      end
*
      function avh_oni_mu_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision avh_oni_mu_get
     &,mu
      common/avh_oni_mu_com/ mu
      integer
     & init
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        mu = 1d0
      endif
      avh_oni_mu_get = mu
      end


      subroutine avh_oni_unit(un_in)
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer un_in
     &,un ,avh_oni_un_get ,init
      common/avh_oni_un_com/ un
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        un = avh_oni_un_get()
      endif
      un = un_in
      end
*
      function avh_oni_un_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer avh_oni_un_get
     &,un ,init
      common/avh_oni_un_com/ un
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        un = 6
      endif
      avh_oni_un_get = un
      end


      function avh_oni_thrs(xx)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision xx ,avh_oni_thrs,avh_oni_prec
      double precision         thrs_com
      common/avh_oni_thrs_com/ thrs_com
      logical init ,avh_oni_os_get
      data init/.true./
      save init
      if (init) then
        init = .false.
        if (avh_oni_prec().gt.1d-24) then
* * * double precision
          thrs_com = 1d2*avh_oni_prec()
        else
* * * quadruple precision
          thrs_com = 1d4*avh_oni_prec()
        endif
      endif
      if (avh_oni_os_get()) then
        avh_oni_thrs = thrs_com
      else
        avh_oni_thrs = thrs_com*xx
      endif
      end

      subroutine avh_oni_onshell(thrs)
*  ********************************************************************
*  * Set threshold to consider internal masses identical zero and
*  * external squared momenta identical zero or equal to internal
*  * masses, if this leads to an IR divergent case. For example,
*  * if  |p1-m1|<thrs , and  p1=m1  consitutes an IR-divergent
*  * case, then the loop integral is considered to be IR-divergent.
*  * Here  thrs  is the input for this routine.
*  * If this routine is not called,  thrs  will essentially be considerd
*  * identically zero, but warnings will be given when an IR-divergent
*  * case is approached. If this routine is called with thrs=0d0 , then
*  * these warnings will not appear anymore.
*  ********************************************************************
      implicit none
      double precision thrs
     &,avh_oni_thrs
      logical
     & avh_oni_os_get ,init
      integer
     & avh_oni_un_get
      logical                onshell
      common/avh_oni_os_com/ onshell
      double precision         thrs_com
      common/avh_oni_thrs_com/ thrs_com
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_oni_hello
        onshell = avh_oni_os_get()
        thrs_com = avh_oni_thrs(1d0)
      endif
      onshell = .true.
      thrs_com = thrs
      if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &  'MESSAGE from avh_oni_onshell: threshold set to:',thrs_com
      end
*
      function avh_oni_os_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      logical avh_oni_os_get ,init
      logical                onshell
      common/avh_oni_os_com/ onshell
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        onshell = .false.
      endif
      avh_oni_os_get = onshell
      end
