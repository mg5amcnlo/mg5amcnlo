************************************************************************
* This is the file  avh_olo_3div.f  of the package                     *
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


      subroutine avh_olo_c0m4(rslt ,cpp,cm2,cm3 ,zmu)
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
      double complex rslt(0:2) ,cm2,cm3,cpp ,zmu
     &,one,q32,q23sq,z1,z2,cc ,sm2,sm3,q23,r23,d23,qm3
     &,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
     &,avh_olo_sqrt
      parameter(one=(1d0,0d0))
      integer
     & im3,i23,i23sq,i32,i1,i2 ,avh_olo_un_get
*
c      write(6,*) 'MESSAGE from avh_olo_c0m4: you are calling me'
*
      sm2 = avh_olo_sqrt(cm2,-1d0)
      sm3 = avh_olo_sqrt(cm3,-1d0)
      q23 = (cm2+cm3-cpp)/(sm2*sm3)
      call avh_olo_rfun( r23,d23, q23 )
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( qm3,im3 ,cm3/zmu,-1d0 )
*
      if (r23.eq.-one) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_c0m4: threshold singularity, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
*
      call avh_olo_prd(q23sq,i23sq ,q23,i23 ,q23,i23)
*
      call avh_olo_rat( q32,i32 ,sm3,0 ,sm2,0 )
      call avh_olo_prd( z1,i1 ,q32,i32 ,q23,i23 )
      call avh_olo_rat( z2,i2 ,q32,i32 ,q23,i23 )
*
      rslt(2) = dcmplx(0d0)
      cc = avh_olo_logc2(q23,i23) * r23/(one+r23)/(sm2*sm3)
      rslt(1) = -cc
      rslt(0) = cc*( avh_olo_logc(qm3,im3) - avh_olo_logc(q23,i23) )
     &        - avh_olo_li2c2(z1,i1 ,z2,i2) / cm2
     &        + avh_olo_li2c2(q23sq,i23sq ,one,0) * r23/(sm2*sm3)
      end


      subroutine avh_olo_c0m3(rslt ,cp2,cp3,cm3 ,zmu)
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
      double complex rslt(0:2) ,cp2,cp3,cm3,zmu
     &,r13,r23,q13,q23,qm3,xx,x1,x2,log2,log3,li2,logm
     &,avh_olo_logc,avh_olo_logc2,avh_olo_li2c2
      integer
     & i13,i23,im3,ix,i1,i2
*
c      write(6,*) 'MESSAGE from avh_olo_c0m3: you are calling me'
*
      r13 = cm3-cp3
      r23 = cm3-cp2
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( q23,i23 ,r23,-1d0 )
      call avh_olo_conv( qm3,im3 ,cm3,-1d0 )
*
      call avh_olo_rat( xx,ix ,qm3,im3 ,zmu,0 )
      logm = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,q23,i23 ,q13,i13 )
      log2 = avh_olo_logc2( xx,ix )
      call avh_olo_rat( x1,i1 ,q23,i23 ,qm3,im3 )
      call avh_olo_rat( x2,i2 ,q13,i13 ,qm3,im3 )
      li2  = avh_olo_li2c2( x1,i1 ,x2,i2 )
      call avh_olo_prd( xx,ix ,x1,i1 ,x2,i2 )
      log3 = avh_olo_logc( xx,ix )
*
      rslt(2) = dcmplx(0d0)
      rslt(1) = -log2/r13
      rslt(0) = -li2/cm3 - rslt(1)*(log3+logm)
      end

      subroutine avh_olo_c0m2(rslt ,cp3,cm3 ,zmu)
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
      double complex rslt(0:2) ,cp3,cm3,zmu
     &,r13,q13,qm3,xx
     &,logm,logp,li2p,z2,z1,z0,cc ,avh_olo_logc,avh_olo_li2c
      double precision
     & const,avh_olo_pi
      integer
     & i13,im3,ix
      logical init
      data init/.true./
      save init,const
*
c      write(6,*) 'MESSAGE from avh_olo_c0m2: you are calling me'
*
      if (init) then
        init = .false.
        const = avh_olo_pi()**2/24d0
      endif
*
      r13 = cm3-cp3
      call avh_olo_conv( q13,i13 ,r13,-1d0 )
      call avh_olo_conv( qm3,im3 ,cm3,-1d0 )
*
      call avh_olo_rat( xx,ix ,qm3,im3 ,zmu,0 )
      logm = avh_olo_logc( xx,ix )
      call avh_olo_rat( xx,ix ,qm3,im3 ,q13,i13 )
      logp = avh_olo_logc( xx,ix )
      li2p = avh_olo_li2c( xx,ix )
*
      z2 = dcmplx(1d0/2d0)
      z1 = logp
      z0 = dcmplx(const) + logp*logp/2 - li2p
      cc = dcmplx(-1d0)/r13
      rslt(2) = cc*z2
      rslt(1) = cc*(z1 - z2*logm)
      rslt(0) = cc*(z0 + (z2*logm/2-z1)*logm)
      end

      subroutine avh_olo_c0m1(rslt ,cm3 ,zmu)
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
     &,avh_olo_logc,zm,xx,qm3
      integer
     & ix,im3
*
c      write(6,*) 'MESSAGE from avh_olo_c0m1: you are calling me'
*
      call avh_olo_conv( qm3,im3 ,cm3,-1d0 )
      call avh_olo_rat( xx,ix ,qm3,im3 ,zmu,0 )
      zm = dcmplx(0.5d0)/cm3
      rslt(2) = dcmplx(0d0)
      rslt(1) = -zm
      rslt(0) = zm*( dcmplx(2d0) + avh_olo_logc(xx,ix) )
      end


      subroutine avh_olo_c0(valc0,p1,p2,p3)
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
     &,pi2,log2,log3,avh_olo_loga,avh_olo_loga2
      double precision p1,p2,p3
     &,mu2,pp(3),ap(3),hmax,smax,avh_olo_mu_get,avh_olo_pi,avh_olo_thrs
      integer
     & icase,ii,base(3),per(3),imax,i1,i2,i3 ,avh_olo_un_get
      character(2) label(3)
      logical init ,avh_olo_os_get
      data init/.true./,base/4,2,1/,label/'p1','p2','p3'/
      save init,pi2
*
      if (init) then
        init = .false.
        pi2  = dcmplx( avh_olo_pi()**2 )
      endif
      mu2 = avh_olo_mu_get()**2
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
      smax = avh_olo_thrs(hmax)
*
      if (avh_olo_os_get()) then
        if (ap(1).lt.smax) ap(1) = 0d0
        if (ap(2).lt.smax) ap(2) = 0d0
        if (ap(3).lt.smax) ap(3) = 0d0
      endif
*
      icase = 0
      do ii=1,3
      if (ap(ii).gt.0d0) then
        icase = icase + base(ii)
        if (ap(ii).lt.smax.and.avh_olo_un_get().gt.0)
     &    write(avh_olo_un_get(),*)
     &    'WARNING from avh_olo_c0: |',label(ii),'/',label(imax),'| ='
     &   ,ap(ii)/hmax
      endif
      enddo
      call avh_olo_c0per(icase,per)
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
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_c0: all external masses equal zero'
        stop
      elseif (icase.eq.1) then
* 1 mass non-zero
        log3 = avh_olo_loga( -pp(per(3))/mu2 , i3 )
        valc0(2) = dcmplx( 1d0/pp(per(3)) )
        valc0(1) = -log3/pp(per(3))
        valc0(0) = ( log3**2/2 - pi2/12 )/pp(per(3))
      elseif (icase.eq.2) then
* 2 masses non-zero
        log2 = avh_olo_loga( -pp(per(2))/mu2 , i2 )
        log3 = avh_olo_loga( -pp(per(3))/mu2 , i3 )
        valc0(2) = dcmplx(0d0)
        valc0(1) = avh_olo_loga2( pp(per(3))/pp(per(2)) , i3-i2 )
     &           / dcmplx( pp(per(2)) )
        valc0(0) = -valc0(1)*(log3+log2)/2
      elseif (icase.eq.3) then
* 3 masses non-zero
        call avh_olo_ccm0( valc0 ,dcmplx(p1),dcmplx(p2),dcmplx(p3) )
      endif
      end


      subroutine avh_olo_c0per(icase,per)
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


      subroutine avh_olo_b0(valb0,p1)
*  ********************************************************************
*  *
*  *            C   /   d^(Dim)q
*  * valb0 = ------ | ------------
*  *         i*pi^2 / q^2 (q+k1)^2
*  *
*  * with  Dim = 4-2*eps
*  *         C = pi^eps * mu^(2*eps) * exp(gamma_Euler*eps)
*  *
*  * input:  p1 = k1^2
*  * output: valc0(0) = eps^0   -coefficient
*  *         valc0(1) = eps^(-1)-coefficient
*  *         valc0(2) = eps^(-2)-coefficient
*  *
*  * The input value (p1) should be real.
*  * If this number is IDENTICALLY 0d0, the IR-singular case is
*  * returned.
*  ********************************************************************
      implicit none
      double complex valb0(0:2)
     &,avh_olo_loga
      double precision p1
     &,arg,mu2,ap,smax,avh_olo_mu_get,avh_olo_thrs
      integer
     & i1 ,avh_olo_un_get
      logical
     & avh_olo_os_get
*
      mu2  = avh_olo_mu_get()**2
*
      ap = dabs(p1)
      smax = avh_olo_thrs(max(ap,mu2))
      if (avh_olo_os_get().and.ap.lt.smax) ap = 0d0
*
      if (ap.eq.0d0) then
        valb0(2) = dcmplx(0d0)
        valb0(1) = dcmplx(0d0)
        valb0(0) = dcmplx(0d0)
      else
        i1 = 0
        if (-p1.lt.0d0) i1 = -1
        arg = -p1/mu2
        if (ap.lt.smax.and.avh_olo_un_get().gt.0)
     &    write(avh_olo_un_get(),*)
     &    'WARNING from avh_olo_b0: |p1/mu2| =', dabs(arg)
        valb0(2) = dcmplx(0d0)
        valb0(1) = dcmplx(1d0)
        valb0(0) = dcmplx(2d0) - avh_olo_loga(arg,i1)
      endif
      end


      subroutine avh_olo_zero(rslt)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex rslt(0:2)
      rslt(2) = dcmplx(0d0)
      rslt(1) = dcmplx(0d0)
      rslt(0) = dcmplx(0d0)
      end


      subroutine avh_olo_mu_set(mu_in)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision mu_in
     &,mu ,avh_olo_mu_get
      common/avh_olo_mu_com/ mu
      integer
     & init,avh_olo_un_get
      data init/0/
      logical firsttime
      data firsttime/.true./
      save init
*
      if (init.eq.0) then
        init = 1
        call avh_olo_hello
        mu = avh_olo_mu_get()
      endif
      mu = mu_in
      if (avh_olo_un_get().gt.0 .and. firsttime) then
         write(avh_olo_un_get(),*) 'MESSAGE from avh_olo_mu_set:'//
     &        ' scale (mu, not mu^2) set to:',mu
         firsttime=.false.
      endif
      end
*
      function avh_olo_mu_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision avh_olo_mu_get
     &,mu
      common/avh_olo_mu_com/ mu
      integer
     & init
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        mu = 1d0
      endif
      avh_olo_mu_get = mu
      end


      subroutine avh_olo_unit(un_in)
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer un_in
     &,un ,avh_olo_un_get ,init
      common/avh_olo_un_com/ un
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        un = avh_olo_un_get()
      endif
      un = un_in
      end
*
      function avh_olo_un_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer avh_olo_un_get
     &,un ,init
      common/avh_olo_un_com/ un
      data init/0/
      save init
*
      if (init.eq.0) then
        init = 1
        un = 6
      endif
      avh_olo_un_get = un
      end


      function avh_olo_thrs(xx)
*  ********************************************************************
*  ********************************************************************
      implicit none
      double precision xx ,avh_olo_thrs,avh_olo_prec
      double precision         thrs_com
      common/avh_olo_thrs_com/ thrs_com
      logical init ,avh_olo_os_get
      data init/.true./
      save init
      if (init) then
        init = .false.
        if (avh_olo_prec().gt.1d-24) then
* * * double precision
          thrs_com = 1d2*avh_olo_prec()
        else
* * * quadruple precision
          thrs_com = 1d4*avh_olo_prec()
        endif
      endif
      if (avh_olo_os_get()) then
        avh_olo_thrs = thrs_com
      else
        avh_olo_thrs = thrs_com*xx
      endif
      end

      subroutine avh_olo_onshell(thrs)
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
     &,avh_olo_thrs
      logical 
     & avh_olo_os_get ,init
      integer
     & avh_olo_un_get
      logical                onshell
      common/avh_olo_os_com/ onshell
      double precision         thrs_com
      common/avh_olo_thrs_com/ thrs_com
      logical firsttime
      data firsttime/.true./
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_olo_hello
        onshell = avh_olo_os_get()
        thrs_com = avh_olo_thrs(1d0)
      endif
      onshell = .true.
      thrs_com = thrs
      if (avh_olo_un_get().gt.0.and. firsttime) then
         write(avh_olo_un_get(),*)'MESSAGE from avh_olo_onshell:'//
     &        ' threshold set to:',thrs_com
         firsttime=.false.
      endif
      end
*
      function avh_olo_os_get()
*  ********************************************************************
*  ********************************************************************
      implicit none
      logical avh_olo_os_get ,init
      logical                onshell
      common/avh_olo_os_com/ onshell
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        onshell = .false.
      endif
      avh_olo_os_get = onshell
      end
