************************************************************************
* This is the file  avh_oni_cmplx.f  of the package                    *
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

      subroutine avh_oni_a0c(rslt ,mm)
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
*  * Check the comments in  avh_oni_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,mm
     &,avh_oni_logc,qmm,zmu
      double precision
     & mu2,am,small,avh_oni_thrs,avh_oni_mu_get
      integer
     & imm,ii,avh_oni_un_get,avh_oni_print
      logical init ,avh_oni_os_get
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_oni_hello
      endif
*
      mu2 = avh_oni_mu_get()**2
      zmu = dcmplx(mu2)
*
      am = cdabs(mm) !CDABSyes
      small = avh_oni_thrs(max(am,mu2))
      if (avh_oni_os_get().and.am.lt.small) am = 0d0
*
      if (am.eq.0d0) then
        rslt(2) = dcmplx(0d0)
        rslt(1) = dcmplx(0d0)
        rslt(0) = dcmplx(0d0)
      else
        if (am.lt.small.and.avh_oni_un_get().gt.0)
     &    write(avh_oni_un_get(),101)
        call avh_oni_conv( qmm,imm ,mm/zmu,-1d0 )
        rslt(2) = dcmplx(0d0)
        rslt(1) = mm
        rslt(0) = mm - mm*avh_oni_logc(qmm,imm)
      endif
  101 format(' WARNING from avh_oni_a0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_oni_onshell  to cure this.')
*
      ii = avh_oni_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_oni_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_oni_mu_get()
        write(ii,102) '   mm: (',dreal(mm),',',dimag(mm),')'
        write(ii,102) 'a0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(ii,102) 'a0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(ii,102) 'a0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_oni_b0c(rslt ,pp_in,m1_in,m2_in )
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
*  * Check the comments in  avh_oni_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,pp_in,m1_in,m2_in ,pp,m1,m2
     &,zmu,qpp,avh_oni_logc
      double precision
     & h1,am1,am2,am,ap,small,mu2,avh_oni_mu_get,avh_oni_thrs
      integer
     & ipp,ii,iu,avh_oni_un_get,avh_oni_print
      logical init ,avh_oni_os_get
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_oni_hello
      endif
*
      mu2 = avh_oni_mu_get()**2
      zmu = dcmplx(mu2)
      iu = avh_oni_un_get()
*
      pp = pp_in
      m1 = m1_in
      m2 = m2_in
*
      ap = dreal(pp)
      if (dimag(pp).ne.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_oni_b0c: momentum with non-zero imaginary'
     &   ,' part, putting it to zero.'
        pp = dcmplx( ap ,0d0 )
      endif
      ap = dabs(ap)
*
      am1 = dreal(m1)
      h1  = dimag(m1)
      if (h1.gt.0d0) then
        if (iu.gt.0) write(iu,*)
     &     'ERROR in avh_oni_b0c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m1 = dcmplx( am1 ,-h1 )
      endif
      am1 = dabs(am1) + dabs(h1)
*
      am2 = dreal(m2)
      h1  = dimag(m2)
      if (h1.gt.0d0) then
        if (iu.gt.0) write(iu,*)
     &     'ERROR in avh_oni_b0c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m2 = dcmplx( am2 ,-h1 )
      endif
      am2 = dabs(am2) + dabs(h1)
*
      am = max(am1,am2)
      small = avh_oni_thrs(max(ap,am,mu2))
*
      if (avh_oni_os_get().and.ap.lt.small.and.am.lt.small) then
        ap = 0d0
        am = 0d0
      endif
*
      if (am.eq.0d0) then
        if (ap.eq.0d0) then
          rslt(2) = dcmplx(0d0)
          rslt(1) = dcmplx(0d0)
          rslt(0) = dcmplx(0d0)
        else
          if (ap.lt.small.and.iu.gt.0) write(iu,101)
          rslt(2) = dcmplx(0d0)
          rslt(1) = dcmplx(1d0)
          call avh_oni_conv( qpp,ipp ,-pp/zmu,-1d0 )
          rslt(0) = dcmplx(2d0) - avh_oni_logc( qpp,ipp )
        endif
      else
        if (am.lt.small.and.ap.lt.small.and.iu.gt.0) write(iu,101)
        call avh_oni_cbm( rslt ,pp,m1,m2 ,zmu )
      endif
  101 format(' WARNING from avh_oni_b0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_oni_onshell  to cure this.')
*
      ii = avh_oni_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_oni_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_oni_mu_get()
        write(ii,102) '   pp: (',dreal(pp_in),',',dimag(pp_in),')'
        write(ii,102) '   m1: (',dreal(m1_in),',',dimag(m1_in),')'
        write(ii,102) '   m2: (',dreal(m2_in),',',dimag(m2_in),')'
        write(ii,102) 'b0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(ii,102) 'b0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(ii,102) 'b0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_oni_c0c(rslt ,p1,p2,p3,m1,m2,m3)
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
*  * Check the comments in  avh_oni_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,m1,m2,m3
     &,zmu,mm(3),pp(3),s1,s2,s3,r1,r2,r3,cnst,zero
      double precision 
     & mu2,h1,h2,smax,small,avh_oni_mu_get,avh_oni_thrs,avh_oni_pi
     &,ap(3),am(3),s1r2,s2r3,s3r3,as1
      integer
     & icase,ii,base(3),ll(3) ,iu,avh_oni_un_get,avh_oni_print
      logical init ,avh_oni_os_get
      parameter( zero=(0d0,0d0) )
      data init/.true./,base/4,2,1/
      save init,cnst
      if (init) then
        init = .false.
        call avh_oni_hello
        cnst = dcmplx(avh_oni_pi()**2/12d0)
      endif
*
      mu2 = avh_oni_mu_get()**2
      zmu = dcmplx(mu2)
      iu = avh_oni_un_get()
*
      pp(1) = p1
      pp(2) = p2
      pp(3) = p3
      mm(1) = m1
      mm(2) = m2
      mm(3) = m3
*
      smax = 0d0
      do ii=1,3
        ap(ii) = dreal(pp(ii))
        if (dimag(pp(ii)).ne.0d0) then
          if (iu.gt.0) write(iu,*)
     &      'ERROR in avh_oni_c0c: momentum with non-zero imaginary'
     &     ,' part, putting it to zero.'
          pp(ii) = dcmplx( ap(ii) ,0d0 )
        endif
        ap(ii) = dabs(ap(ii))
        if (ap(ii).gt.smax) smax = ap(ii)
      enddo
      h2 = 1d99
      do ii=1,3
        am(ii) = dreal(mm(ii))
        h1     = dimag(mm(ii))
        if (h1.gt.0d0) then
          if (iu.gt.0) write(iu,*)
     &       'ERROR in avh_oni_c0c: mass-squared has positive imaginary'
     &      ,' part, switching its sign.'
          mm(ii) = dcmplx( am(ii) ,-h1 )
        endif
        am(ii) = dabs(am(ii)) + dabs(h1)
        if (am(ii).gt.smax) smax = am(ii)
        if (am(ii).gt.0d0.and.am(ii).lt.h2) h2 = am(ii)
      enddo
      if (smax.eq.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_oni_c0c: all input equal zero, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
      if (mu2.gt.smax) smax = mu2
      small = avh_oni_thrs(smax)
      if (h2.lt.small.and.iu.gt.0) write(iu,101)
*
      if (avh_oni_os_get()) then
      do ii=1,3
        if (ap(ii).lt.small) ap(ii) = 0d0
        if (am(ii).lt.small) am(ii) = 0d0
      enddo
      endif
*
      icase = 0
      do ii=1,3
        if (am(ii).gt.0d0) icase = icase + base(ii)
      enddo
      call avh_oni_c0per(icase,ll)
      s1 = pp(ll(1))
      s2 = pp(ll(2))
      s3 = pp(ll(3))
      r1 = mm(ll(1))
      r2 = mm(ll(2))
      r3 = mm(ll(3))
      as1 = ap(ll(1))
*
      s1r2 = cdabs(s1-r2) !CDABSyes
      s2r3 = cdabs(s2-r3) !CDABSyes
      s3r3 = cdabs(s3-r3) !CDABSyes
      if (avh_oni_os_get()) then
        if (s1r2.lt.small) s1r2 = 0d0
        if (s2r3.lt.small) s2r3 = 0d0
        if (s3r3.lt.small) s3r3 = 0d0
      endif
*
!      write(6,*) smax !DEBUG
      if     (icase.eq.3) then
* 3 non-zero internal masses
        call avh_oni_cfam( rslt ,s1,s2,s3 ,r1,r2,r3 ,smax )
      elseif (icase.eq.2) then
* 2 non-zero internal masses
        if     (s1r2.ne.0d0.or.s3r3.ne.0d0) then
          if (s1r2.lt.small.and.s3r3.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_cfam( rslt ,s1,s2,s3 ,zero,r2,r3 ,smax )
        else
          call avh_oni_c0m4( rslt ,s2 ,r2,r3 ,zmu )
        endif
      elseif (icase.eq.1) then
* 1 non-zero internal mass
        if     (as1.ne.0d0) then
          if (as1.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_cfam( rslt ,s1,s2,s3, zero,zero,r3 ,smax )
        elseif (s2r3.ne.0d0) then
          if (s2r3.lt.small.and.iu.gt.0) write(iu,101)
          if   (s3r3.ne.0d0) then
            if (s3r3.lt.small.and.iu.gt.0) write(iu,101)
            call avh_oni_c0m3( rslt ,s2,s3 ,r3 ,zmu ,smax )
          else
            call avh_oni_c0m2( rslt ,s2 ,r3 ,zmu )
          endif
        elseif (s3r3.ne.0d0) then
          if (s3r3.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_c0m2( rslt ,s3 ,r3 ,zmu )
        else
          call avh_oni_c0m1( rslt ,r3 ,zmu )
        endif
      else
* 0 non-zero internal masses
        call avh_oni_c0( rslt ,dreal(s1),dreal(s2),dreal(s3) )
      endif
  101 format(' WARNING from avh_oni_c0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_oni_onshell  to cure this.')
* exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
      rslt(0) = rslt(0) + cnst*rslt(2)
*
      ii = avh_oni_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_oni_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_oni_mu_get()
        write(ii,102) '  p1 : (',dreal(p1),',',dimag(p1),')'
        write(ii,102) '  p2 : (',dreal(p2),',',dimag(p2),')'
        write(ii,102) '  p3 : (',dreal(p3),',',dimag(p3),')'
        write(ii,102) '  m1 : (',dreal(m1),',',dimag(m1),')'
        write(ii,102) '  m2 : (',dreal(m2),',',dimag(m2),')'
        write(ii,102) '  m3 : (',dreal(m3),',',dimag(m3),')'
        write(ii,102) 'c0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(ii,102) 'c0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(ii,102) 'c0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_oni_d0c(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4)
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
*  * Check the comments in  avh_oni_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
     &,s1,s2,s3,s4,s12,s23,r1,r2,r3,r4 ,mm(4),pp(6),zero,one,cnst
      parameter( zero=(0d0,0d0) ,one=(1d0,0d0) )
      double precision
     & mu2,h1,h2,smax,small,avh_oni_mu_get,avh_oni_thrs,avh_oni_pi
     &,ap(6),am(4),ar2,as1,as2,s1r2,s2r2,s2r3,s3r4,s4r4
      integer
     & icase,ii,base(4),ll(6),ncm ,iu,avh_oni_un_get,avh_oni_print
      logical init ,avh_oni_os_get
      data init/.true./,base/8,4,2,1/
      save init,cnst
      if (init) then
        init = .false.
        call avh_oni_hello
        cnst = dcmplx(avh_oni_pi()**2/12d0)
      endif
*
      mu2 = avh_oni_mu_get()**2
      iu = avh_oni_un_get()
*
      call avh_oni_rot4( pp,mm ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*
      smax = 0d0
      do ii=1,6
        ap(ii) = dreal(pp(ii))
        if (dimag(pp(ii)).ne.0d0) then
          if (iu.gt.0) write(iu,*)
     &      'ERROR in avh_oni_d0c: momentum with non-zero imaginary'
     &     ,' part, putting it to zero.'
          pp(ii) = dcmplx( ap(ii) ,0d0 )
        endif
        ap(ii) = dabs(ap(ii))
        if (ap(ii).gt.smax) smax = ap(ii)
      enddo
      h2 = 1d99
      do ii=1,4
        am(ii) = dreal(mm(ii))
        h1     = dimag(mm(ii))
        if (h1.gt.0d0) then
          if (iu.gt.0) write(iu,*)
     &       'ERROR in avh_oni_d0c: mass-squared has positive imaginary'
     &      ,' part, switching its sign.'
          mm(ii) = dcmplx( am(ii) ,-h1 )
        endif
        am(ii) = dabs(am(ii)) + dabs(h1)
        if (am(ii).gt.smax) smax = am(ii)
        if (am(ii).gt.0d0.and.am(ii).lt.h2) h2 = am(ii)
      enddo
      if (smax.eq.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_oni_d0c: all input equal zero, returning 0'
        call avh_oni_zero(rslt)
        return
      endif
      if (mu2.gt.smax) smax = mu2
      small = avh_oni_thrs(smax)
      if (h2.lt.small.and.iu.gt.0)  write(iu,101)
*
      if (avh_oni_os_get()) then
      do ii=1,4
        if (ap(ii).lt.small) ap(ii) = 0d0
        if (am(ii).lt.small) am(ii) = 0d0
      enddo
      endif
*
      ncm = 0
      do ii=1,4
        if (am(ii).ne.0d0.and.dimag(mm(ii)).ne.0d0) ncm = ncm + 1
      enddo
*
      icase = 0
      do ii=1,4
        if (am(ii).gt.0d0) icase = icase + base(ii)
      enddo
      call avh_oni_d0per(icase,ll)
      s1  = pp(ll(1))
      s2  = pp(ll(2))
      s3  = pp(ll(3))
      s4  = pp(ll(4))
      s12 = pp(ll(5))
      s23 = pp(ll(6))
      r1 = mm(ll(1))
      r2 = mm(ll(2))
      r3 = mm(ll(3))
      r4 = mm(ll(4))
      ar2 = am(ll(2))
      as1 = ap(ll(1))
      as2 = ap(ll(2))
*
      s1r2 = dabs(dreal(s1-r2)) + dabs(dimag(s1-r2))
      s2r2 = dabs(dreal(s2-r2)) + dabs(dimag(s2-r2))
      s2r3 = dabs(dreal(s2-r3)) + dabs(dimag(s2-r3))
      s3r4 = dabs(dreal(s3-r4)) + dabs(dimag(s3-r4))
      s4r4 = dabs(dreal(s4-r4)) + dabs(dimag(s4-r4))
      if (avh_oni_os_get()) then
        if (s1r2.lt.small) s1r2 = 0d0
        if (s2r2.lt.small) s2r2 = 0d0
        if (s2r3.lt.small) s2r3 = 0d0
        if (s3r4.lt.small) s3r4 = 0d0
        if (s4r4.lt.small) s4r4 = 0d0
      endif
*
      if     (icase.eq.4) then
* 4 non-zero internal masses
        call avh_oni_dfam( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4
     &                    ,smax ,one,zero,zero,zero,zero )
      elseif (icase.eq.3) then
* 3 non-zero internal masses
        if (ar2.lt.small.and.iu.gt.0) write(iu,101)
        if (s1r2.ne.0d0.or.s4r4.ne.0d0) then
          if (s1r2.lt.small.and.s4r4.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_dfam( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4
     &                      ,smax ,one,zero,zero,zero,zero )
        else
          call avh_oni_d0m16( rslt ,s2,s3,s12,s23 ,r2,r3,r4 ,smax )
        endif
      elseif (icase.eq.5) then
* 2 non-zero internal masses, opposite case
        if     (s1r2.ne.0d0.or.s4r4.ne.0d0) then
          if (s1r2.lt.small.and.s4r4.lt.small.and.iu.gt.0) write(iu,101)
          if (s2r2.ne.0d0.or.s3r4.ne.0d0) then
            if(s2r2.lt.small.and.s3r4.lt.small.and.iu.gt.0)write(iu,101)
            call avh_oni_dfam( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4
     &                        ,smax ,one,zero,zero,zero,zero )
          else
            call avh_oni_d0m16( rslt ,s1,s4,s12,s23 ,r2,r3,r4 ,smax )
          endif
        elseif (s2r2.ne.0d0.or.s3r4.ne.0d0) then
          if (s2r2.lt.small.and.s3r4.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_d0m16( rslt ,s2,s3,s12,s23 ,r2,r3,r4 ,smax )
        else
          call avh_oni_d0m14( rslt ,s12,s23 ,r2,r4 ,smax )
        endif
      elseif (icase.eq.2) then
* 2 non-zero internal masses, adjacent case
        if (as1.ne.0d0) then
          if (as1.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_dfam( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4
     &                      ,smax ,one,zero,zero,zero,zero )
        else
          call avh_oni_d0m13( rslt ,s2,s3,s4,s12,s23 ,r3,r4 ,smax )
        endif
      elseif (icase.eq.1) then
* 1 non-zero internal mass
        if (as1.ne.0d0) then
          if (as1.lt.small.and.iu.gt.0) write(iu,101)
          if (as2.ne.0d0) then
            if (as2.lt.small.and.iu.gt.0) write(iu,101)
            call avh_oni_dfam( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4
     &                        ,smax ,one,zero,zero,zero,zero )
          else
            call avh_oni_d0m13( rslt ,s1,s4,s3,s12,s23 ,zero,r4 ,smax)
          endif
        elseif (as2.ne.0d0) then
          if (as2.lt.small.and.iu.gt.0) write(iu,101)
          call avh_oni_d0m13( rslt ,s2,s3,s4,s12,s23 ,zero,r4 ,smax )
        else
          call avh_oni_d0m8( rslt ,s3,s4,s12,s23 ,r4 ,smax )
        endif
      else
* 0 non-zero internal mass
        call avh_oni_d0( rslt ,s1,s2,s3,s4,s12,s23 )
      endif
  101 format(' WARNING from avh_oni_d0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_oni_onshell  to cure this.')
!* exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
!      rslt(0) = rslt(0) + cnst*rslt(2) !CHANGE
*
      ii = avh_oni_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_oni_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_oni_mu_get()
        write(ii,102) '  p1 : (',dreal(p1),',',dimag(p1),')'
        write(ii,102) '  p2 : (',dreal(p2),',',dimag(p2),')'
        write(ii,102) '  p3 : (',dreal(p3),',',dimag(p3),')'
        write(ii,102) '  p4 : (',dreal(p4),',',dimag(p4),')'
        write(ii,102) '  p12: (',dreal(p12),',',dimag(p12),')'
        write(ii,102) '  p23: (',dreal(p23),',',dimag(p23),')'
        write(ii,102) '  m1 : (',dreal(m1),',',dimag(m1),')'
        write(ii,102) '  m2 : (',dreal(m2),',',dimag(m2),')'
        write(ii,102) '  m3 : (',dreal(m3),',',dimag(m3),')'
        write(ii,102) '  m4 : (',dreal(m4),',',dimag(m4),')'
        write(ii,102) 'd0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(ii,102) 'd0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(ii,102) 'd0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_oni_rot4(pp,mm ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4)
*  ********************************************************************
*  * Rotate kinematics to one of the following forms
*  *   p12,p23 ,p1,p2,p3,p4
*  *            +  +  +  + , p1 largest of p1,p2,p3,p4
*  *    +   -   -  -  +  + 
*  *            -  +  -  + 
*  ********************************************************************
      implicit none
      double complex pp(6),mm(4) ,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
      double precision
     & rp(6),small,scl,avh_oni_prec
      integer
     & ii,jj,per(6,4)
      data per/1,2,3,4,5,6  ,4,1,2,3,6,5  ,3,4,1,2,5,6  ,2,3,4,1,6,5/
*
      rp(1) = dreal(p1)
      rp(2) = dreal(p2)
      rp(3) = dreal(p3)
      rp(4) = dreal(p4)
      rp(5) = dreal(p12)
      rp(6) = dreal(p23)
*
      jj = 1
      do ii=2,6
        if ( dabs(rp(ii)).gt.dabs(rp(jj)) ) jj = ii
      enddo
      scl = dabs( rp(jj) )
      small = scl*avh_oni_prec()*1d2
*
      jj = 1
!      if     (rp(5).ge.0d0.and.rp(6).ge.0d0) then
      if (     rp(1).ge.0d0.and.rp(2).ge.0d0
     &    .and.rp(3).ge.0d0.and.rp(4).ge.0d0 ) then
        do ii=2,4
          if ( rp(ii).gt.rp(jj) ) jj = ii
        enddo
      elseif (rp(5).ge.0d0.and.rp(6).lt.0d0) then
        if (     min(rp(3),rp(4)) .lt.-small
     &      .or. max(rp(1),rp(2)) .gt. small ) jj = 3
      elseif (rp(5).lt.0d0.and.rp(6).ge.0d0) then
        jj = 2
        if (     min(rp(1),rp(4)) .lt.-small
     &      .or. max(rp(2),rp(3)) .gt. small ) jj = 4
      else
        if (     min(rp(2),rp(4)) .lt.-small
     &      .or. max(rp(1),rp(3)) .gt. small ) jj = 2
      endif
*
c      write(6,*) 'WARNING from avh_oni_cmplx: jj put to 1' !DEBUG
c      jj = 1 !DEBUG
      pp(per(1,jj)) = p1
      pp(per(2,jj)) = p2
      pp(per(3,jj)) = p3
      pp(per(4,jj)) = p4
      pp(per(5,jj)) = p12
      pp(per(6,jj)) = p23
      mm(per(1,jj)) = m1
      mm(per(2,jj)) = m2
      mm(per(3,jj)) = m3
      mm(per(4,jj)) = m4
      end
