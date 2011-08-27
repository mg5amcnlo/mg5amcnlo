************************************************************************
* This is the file  avh_olo_cmplx.f  of the package                    *
*                                                                      *
*                               OneLOop                                *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 09-09-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_olo_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_olo_a0c(rslt ,mm)
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
*  * Check the comments in  avh_olo_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,mm
     &,avh_olo_logc,qmm,zmu
      double precision
     & mu2,am,smax,avh_olo_thrs,avh_olo_mu_get
      integer
     & imm,ii,avh_olo_un_get,avh_olo_print
      logical init ,avh_olo_os_get
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_olo_hello
      endif
*
      mu2 = avh_olo_mu_get()**2
      zmu = dcmplx(mu2)
*
      am = cdabs(mm)
      smax = avh_olo_thrs(max(am,mu2))
      if (avh_olo_os_get().and.am.lt.smax) am = 0d0
*
      if (am.eq.0d0) then
        rslt(2) = dcmplx(0d0)
        rslt(1) = dcmplx(0d0)
        rslt(0) = dcmplx(0d0)
      else
        if (am.lt.smax.and.avh_olo_un_get().gt.0)
     &    write(avh_olo_un_get(),101)
        call avh_olo_conv( qmm,imm ,mm/zmu,-1d0 )
        rslt(2) = dcmplx(0d0)
        rslt(1) = mm
        rslt(0) = mm - mm*avh_olo_logc(qmm,imm)
      endif
  101 format(' WARNING from avh_olo_a0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_olo_onshell  to cure this.')
*
      ii = avh_olo_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_olo_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_olo_mu_get()
        write(ii,102) '   mm: (',dreal(mm),',',dimag(mm),')'
        write(ii,102) 'a0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(ii,102) 'a0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(ii,102) 'a0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_olo_b0c(rslt ,pp_in,m1_in,m2_in )
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
*  * Check the comments in  avh_olo_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,pp_in,m1_in,m2_in ,pp,m1,m2
     &,zmu,qpp,avh_olo_logc
      double precision
     & h1,am1,am2,am,ap,smax,mu2,avh_olo_mu_get,avh_olo_thrs
      integer
     & ipp,iu,avh_olo_un_get,avh_olo_print
      logical init ,avh_olo_os_get
      data init/.true./
      save init
*
      if (init) then
        init = .false.
        call avh_olo_hello
      endif
*
      mu2 = avh_olo_mu_get()**2
      zmu = dcmplx(mu2)
      iu = avh_olo_un_get()
*
      pp = pp_in
      m1 = m1_in
      m2 = m2_in
*
      ap = dreal(pp)
      if (dimag(pp).ne.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_olo_b0c: momentum with non-zero imaginary'
     &   ,' part, putting it to zero.'
        pp = dcmplx( ap ,0d0 )
      endif
      ap = dabs(ap)
*
      am1 = dreal(m1)
      h1  = dimag(m1)
      if (h1.gt.0d0) then
        if (iu.gt.0) write(iu,*)
     &     'ERROR in avh_olo_b0c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m1 = dcmplx( am1 ,-h1 )
      endif
      am1 = dabs(am1) + dabs(h1)
*
      am2 = dreal(m2)
      h1  = dimag(m2)
      if (h1.gt.0d0) then
        if (iu.gt.0) write(iu,*)
     &     'ERROR in avh_olo_b0c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m2 = dcmplx( am2 ,-h1 )
      endif
      am2 = dabs(am2) + dabs(h1)
*
      am = max(am1,am2)
      smax = avh_olo_thrs(max(ap,am,mu2))
*
      if (avh_olo_os_get().and.ap.lt.smax.and.am.lt.smax) then
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
          if (ap.lt.smax.and.iu.gt.0) write(iu,101)
          rslt(2) = dcmplx(0d0)
          rslt(1) = dcmplx(1d0)
          call avh_olo_conv( qpp,ipp ,-pp/zmu,-1d0 )
          rslt(0) = dcmplx(2d0) - avh_olo_logc( qpp,ipp )
        endif
      else
        if (am.lt.smax.and.ap.lt.smax.and.iu.gt.0) write(iu,101)
        call avh_olo_cbm( rslt ,pp,m1,m2 ,zmu )
      endif
  101 format(' WARNING from avh_olo_b0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_olo_onshell  to cure this.')
*
      iu = avh_olo_print()
      if (iu.gt.0) then
        write(iu,'(a7,d39.32)') 'onshell',avh_olo_thrs(1d0)
        write(iu,'(a2,d39.32)') 'mu',avh_olo_mu_get()
        write(iu,102) '   pp: (',dreal(pp_in),',',dimag(pp_in),')'
        write(iu,102) '   m1: (',dreal(m1_in),',',dimag(m1_in),')'
        write(iu,102) '   m2: (',dreal(m2_in),',',dimag(m2_in),')'
        write(iu,102) 'b0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(iu,102) 'b0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(iu,102) 'b0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_olo_c0c(rslt ,p1,p2,p3,m1,m2,m3)
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
*  * Check the comments in  avh_olo_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,m1,m2,m3
     &,zmu,mm(3),pp(3),s1,s2,s3,r1,r2,r3,cnst
      double precision 
     & mu2,h1,h2,smax,avh_olo_mu_get,avh_olo_thrs,avh_olo_pi
     &,ap(3),am(3),s1r2,s2r3,s3r3,as1
      integer
     & icase,ii,iu,base(3),ll(3) ,avh_olo_un_get,avh_olo_print
      logical init ,avh_olo_os_get
      data init/.true./,base/4,2,1/
      save init,cnst
      if (init) then
        init = .false.
        call avh_olo_hello
        cnst = dcmplx(avh_olo_pi()**2/12d0)
      endif
*
      mu2 = avh_olo_mu_get()**2
      zmu = dcmplx(mu2)
      iu = avh_olo_un_get()
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
     &      'ERROR in avh_olo_c0c: momentum with non-zero imaginary'
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
     &       'ERROR in avh_olo_c0c: mass-squared has positive imaginary'
     &      ,' part, switching its sign.'
          mm(ii) = dcmplx( am(ii) ,-h1 )
        endif
        am(ii) = dabs(am(ii)) + dabs(h1)
        if (am(ii).gt.smax) smax = am(ii)
        if (am(ii).gt.0d0.and.am(ii).lt.h2) h2 = am(ii)
      enddo
      if (smax.eq.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_olo_c0c: all input equal zero, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (mu2.gt.smax) smax = mu2
      smax = avh_olo_thrs(smax)
      if (h2.lt.smax.and.iu.gt.0)
     &  write(iu,101)
*
      if (avh_olo_os_get()) then
      do ii=1,3
        if (ap(ii).lt.smax) ap(ii) = 0d0
        if (am(ii).lt.smax) am(ii) = 0d0
      enddo
      endif
*
      icase = 0
      do ii=1,3
        if (am(ii).gt.0d0) icase = icase + base(ii)
      enddo
      call avh_olo_c0per(icase,ll)
      s1 = pp(ll(1))
      s2 = pp(ll(2))
      s3 = pp(ll(3))
      r1 = mm(ll(1))
      r2 = mm(ll(2))
      r3 = mm(ll(3))
      as1 = ap(ll(1))
*
      s1r2 = cdabs(s1-r2)
      s2r3 = cdabs(s2-r3)
      s3r3 = cdabs(s3-r3)
      if (avh_olo_os_get()) then
        if (s1r2.lt.smax) s1r2 = 0d0
        if (s2r3.lt.smax) s2r3 = 0d0
        if (s3r3.lt.smax) s3r3 = 0d0
      endif
*
      if     (icase.eq.3) then
* 3 non-zero internal masses
        call avh_olo_ccm3( rslt ,s1,s2,s3 ,r1,r2,r3 )
      elseif (icase.eq.2) then
* 2 non-zero internal masses
        if     (s1r2.ne.0d0.or.s3r3.ne.0d0) then
          if (s1r2.lt.smax.and.s3r3.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_ccm2( rslt ,s1,s2,s3 ,r2,r3 )
        else
          call avh_olo_c0m4( rslt ,s2 ,r2,r3 ,zmu )
        endif
      elseif (icase.eq.1) then
* 1 non-zero internal mass
        if     (as1.ne.0d0) then
          if (as1.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_ccm1( rslt ,s1,s2,s3, r3 )
        elseif (s2r3.ne.0d0) then
          if (s2r3.lt.smax.and.iu.gt.0) write(iu,101)
          if   (s3r3.ne.0d0) then
            if (s3r3.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_c0m3( rslt ,s2,s3 ,r3 ,zmu )
          else
            call avh_olo_c0m2( rslt ,s2 ,r3 ,zmu )
          endif
        elseif (s3r3.ne.0d0) then
          if (s3r3.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_c0m2( rslt ,s3 ,r3 ,zmu )
        else
          call avh_olo_c0m1( rslt ,r3 ,zmu )
        endif
      else
* 0 non-zero internal masses
        call avh_olo_c0( rslt ,dreal(s1),dreal(s2),dreal(s3) )
      endif
  101 format(' WARNING from avh_olo_c0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_olo_onshell  to cure this.')
* exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
      rslt(0) = rslt(0) + cnst*rslt(2)
*
      iu = avh_olo_print()
      if (iu.gt.0) then
        write(iu,'(a7,d39.32)') 'onshell',avh_olo_thrs(1d0)
        write(iu,'(a2,d39.32)') 'mu',avh_olo_mu_get()
        write(iu,102) '  p1 : (',dreal(p1),',',dimag(p1),')'
        write(iu,102) '  p2 : (',dreal(p2),',',dimag(p2),')'
        write(iu,102) '  p3 : (',dreal(p3),',',dimag(p3),')'
        write(iu,102) '  m1 : (',dreal(m1),',',dimag(m1),')'
        write(iu,102) '  m2 : (',dreal(m2),',',dimag(m2),')'
        write(iu,102) '  m3 : (',dreal(m3),',',dimag(m3),')'
        write(iu,102) 'c0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(iu,102) 'c0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(iu,102) 'c0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_olo_d0c(rslt ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4)
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
*  * Check the comments in  avh_olo_onshell  to find out how this
*  * routines decides when to return IR-divergent cases.
*  ********************************************************************
      implicit none
      double complex rslt(0:2) ,p1,p2,p3,p4,p12,p23,m1,m2,m3,m4
     &,zmu,s1,s2,s3,s4,s12,s23,r1,r2,r3,r4 ,mm(4),pp(6),zero,cnst
      parameter( zero=(0d0,0d0) )
      double precision
     & mu2,h1,h2,smax,avh_olo_mu_get,avh_olo_thrs,avh_olo_pi
     &,ap(6),am(4),ar2,as1,as2,s1r2,s2r2,s2r3,s3r4,s4r4
      integer
     & icase,ii,iu,base(4),ll(6),ncm ,avh_olo_un_get,avh_olo_print
      logical init ,avh_olo_os_get,avh_olo_kin
      data init/.true./,base/8,4,2,1/
      save init,cnst
      if (init) then
        init = .false.
        call avh_olo_hello
        cnst = dcmplx(avh_olo_pi()**2/12d0)
      endif
*
      mu2 = avh_olo_mu_get()**2
      zmu = dcmplx(mu2)
      iu = avh_olo_un_get()
*
      call avh_olo_rot4( pp,mm ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4 )
*
      smax = 0d0
      do ii=1,6
        ap(ii) = dreal(pp(ii))
        if (dimag(pp(ii)).ne.0d0) then
          if (iu.gt.0) write(iu,*)
     &      'ERROR in avh_olo_d0c: momentum with non-zero imaginary'
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
     &       'ERROR in avh_olo_d0c: mass-squared has positive imaginary'
     &      ,' part, switching its sign.'
          mm(ii) = dcmplx( am(ii) ,-h1 )
        endif
        am(ii) = dabs(am(ii)) + dabs(h1)
        if (am(ii).gt.smax) smax = am(ii)
        if (am(ii).gt.0d0.and.am(ii).lt.h2) h2 = am(ii)
      enddo
      if (smax.eq.0d0) then
        if (iu.gt.0) write(iu,*)
     &    'ERROR in avh_olo_d0c: all input equal zero, returning 0'
        call avh_olo_zero(rslt)
        return
      endif
      if (mu2.gt.smax) smax = mu2
      smax = avh_olo_thrs(smax)
      if (h2.lt.smax.and.iu.gt.0) write(iu,101)
*
      if (avh_olo_os_get()) then
      do ii=1,4
        if (ap(ii).lt.smax) ap(ii) = 0d0
        if (am(ii).lt.smax) am(ii) = 0d0
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
      call avh_olo_d0per(icase,ll)
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
      if (avh_olo_os_get()) then
        if (s1r2.lt.smax) s1r2 = 0d0
        if (s2r2.lt.smax) s2r2 = 0d0
        if (s2r3.lt.smax) s2r3 = 0d0
        if (s3r4.lt.smax) s3r4 = 0d0
        if (s4r4.lt.smax) s4r4 = 0d0
      endif
*
      if     (icase.eq.4) then
* 4 non-zero internal masses
        if (ncm.ge.1) then
          call avh_olo_cd0( rslt ,pp,mm ,ap )
        else
          if (avh_olo_kin(1)) then
            call avh_olo_cd0( rslt ,pp,mm ,ap )
          else
            call avh_olo_cdm4( rslt ,s1,s2,s3,s4,s12,s23 ,r1,r2,r3,r4 )
          endif
        endif
      elseif (icase.eq.3) then
* 3 non-zero internal masses
        if (ar2.lt.smax.and.iu.gt.0) write(iu,101)
        if (s1r2.ne.0d0.or.s4r4.ne.0d0) then
          if (s1r2.lt.smax.and.s4r4.lt.smax.and.iu.gt.0) write(iu,101)
          if (ncm.ge.1) then
            call avh_olo_cd0( rslt ,pp,mm ,ap )
          else
            if (avh_olo_kin(1)) then
              call avh_olo_cd0( rslt ,pp,mm ,ap )
            else
              call avh_olo_cdm3(rslt,pp(1),pp(2),pp(3),pp(4),pp(5),pp(6)
     &                              ,mm(1),mm(2),mm(3),mm(4) )
            endif
          endif
        else
          call avh_olo_d0m16( rslt ,s2,s3,s12,s23 ,r2,r3,r4 ,zmu )
        endif
      elseif (icase.eq.5) then
* 2 non-zero internal masses, opposite case
        if     (s1r2.ne.0d0.or.s4r4.ne.0d0) then
          if (s1r2.lt.smax.and.s4r4.lt.smax.and.iu.gt.0) write(iu,101)
          if     (s2r2.ne.0d0.or.s3r4.ne.0d0) then
            if (s2r2.lt.smax.and.s3r4.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_cdm5( rslt ,s1,s2,s3,s4,s12,s23 ,r2,r4 )
          else                             ! s1=/=r2/s4=/=r4,s2=r2,s3=r4
            call avh_olo_d0m15( rslt ,s1,s4,s12,s23 ,r2,r4 ,zmu )
          endif
        elseif (s2r2.ne.0d0.or.s3r4.ne.0d0) then ! s1=r2,s4=r4,s2=/=r2/s3=/=r4
          if (s2r2.lt.smax.and.s3r4.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_d0m15( rslt ,s2,s3,s12,s23 ,r2,r4 ,zmu )
        else                               ! s1=r2,s4=r4,s2=r2,s3=r4
          call avh_olo_d0m14( rslt ,s12,s23 ,r2,r4 ,zmu )
        endif
      elseif (icase.eq.2) then
* 2 non-zero internal masses, adjacent case
        if     (as1.ne.0d0) then
          if (as1.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_cdm2( rslt ,s1,s2,s3,s4,s12,s23 ,r3,r4 )
        elseif (s2r3.ne.0d0) then
          if (s2r3.lt.smax.and.iu.gt.0) write(iu,101)
          if     (s4r4.ne.0d0) then ! s1=0,s2=/=r3,s4=/=r4
            if (s4r4.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_d0m13( rslt ,s2,s3,s4,s12,s23 ,r3,r4 ,zmu )
          else                   ! s1=0,s2=/=r3,s4=r4 
            call avh_olo_d0m12( rslt ,s3,s2,s23,s12 ,r4,r3 ,zmu )
          endif
        elseif (s4r4.ne.0d0) then ! s1=0,s2=r3,s4=/=r4
          if (s4r4.lt.smax.and.iu.gt.0) write(iu,101)
          call avh_olo_d0m12( rslt ,s3,s4,s12,s23 ,r3,r4 ,zmu )
        else                   ! s1=0,s2=r3,s4=r4
          call avh_olo_d0m11( rslt ,s3,s12,s23 ,r3,r4 ,zmu )
        endif
      elseif (icase.eq.1) then
* 1 non-zero internal mass
        if     (as1.ne.0d0) then
          if (as1.lt.smax.and.iu.gt.0) write(iu,101)
          if      (as2.ne.0d0) then
            if (as2.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_cdm1( rslt ,s1,s2,s3,s4,s12,s23 ,r4 )
          else
            if     (s3r4.ne.0d0) then ! s1=/=0,s2=0,s3=/=r4
              if (s3r4.lt.smax.and.iu.gt.0) write(iu,101)
              call avh_olo_d0m10( rslt ,s1,s4,s3,s12,s23 ,r4 ,zmu )
            else                   ! s1=/=0,s2=0,s3=r4
              call avh_olo_d0m9( rslt ,s1,s4,s12,s23 ,r4 ,zmu )
            endif
          endif
        elseif (as2.ne.0d0) then
          if (as2.lt.smax.and.iu.gt.0) write(iu,101)
          if      (s4r4.ne.0d0) then ! s1=0,s2=/=0,s4=/=r4
            if (s4r4.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_d0m10( rslt ,s2,s3,s4,s12,s23 ,r4 ,zmu )
          else                    ! s1=0,s2=/=0,s4=r4
            call avh_olo_d0m9( rslt ,s2,s3,s12,s23 ,r4 ,zmu )
          endif
        else
          if     (s3r4.ne.0d0) then
            if (s3r4.lt.smax.and.iu.gt.0) write(iu,101)
            if     (s4r4.ne.0d0) then ! s1=0,s2=0,s3=/=r4,s4=/=r4
              if (s4r4.lt.smax.and.iu.gt.0) write(iu,101)
              call avh_olo_d0m8( rslt ,s3,s4,s12,s23 ,r4 ,zmu )
            else                   ! s1=0,s2=0,s3=/=r4,s4=r4
              call avh_olo_d0m7( rslt ,s3,s12,s23 ,r4 ,zmu )
            endif
          elseif (s4r4.ne.0d0) then ! s1=0,s2=0,s3=r4,s4=/=r4
            if (s4r4.lt.smax.and.iu.gt.0) write(iu,101)
            call avh_olo_d0m7( rslt ,s4,s12,s23 ,r4 ,zmu )
          else                   ! s1=0,s2=0,s3=r4,s4=r4
            call avh_olo_d0m6( rslt ,s12,s23 ,r4 ,zmu )
          endif
        endif
      else
* 0 non-zero internal mass
        call avh_olo_d0(rslt ,dreal(s1),dreal(s2),dreal(s3),dreal(s4)
     &                       ,dreal(s12),dreal(s23) )
      endif
  101 format(' WARNING from avh_olo_d0c: it seems you forgot'
     &      ,' to put some input explicitly on shell.'
     &      ,' You may  call avh_olo_onshell  to cure this.')
* exp(eps*gamma_EULER) -> GAMMA(1-2*eps)/GAMMA(1-eps)^2/GAMMA(1+eps)
      rslt(0) = rslt(0) + cnst*rslt(2)
*
      iu = avh_olo_print()
      if (iu.gt.0) then
        write(iu,'(a7,d39.32)') 'onshell',avh_olo_thrs(1d0)
        write(iu,'(a2,d39.32)') 'mu',avh_olo_mu_get()
        write(iu,102) '  p1 : (',dreal(p1),',',dimag(p1),')'
        write(iu,102) '  p2 : (',dreal(p2),',',dimag(p2),')'
        write(iu,102) '  p3 : (',dreal(p3),',',dimag(p3),')'
        write(iu,102) '  p4 : (',dreal(p4),',',dimag(p4),')'
        write(iu,102) '  p12: (',dreal(p12),',',dimag(p12),')'
        write(iu,102) '  p23: (',dreal(p23),',',dimag(p23),')'
        write(iu,102) '  m1 : (',dreal(m1),',',dimag(m1),')'
        write(iu,102) '  m2 : (',dreal(m2),',',dimag(m2),')'
        write(iu,102) '  m3 : (',dreal(m3),',',dimag(m3),')'
        write(iu,102) '  m4 : (',dreal(m4),',',dimag(m4),')'
        write(iu,102) 'd0c 2: (',dreal(rslt(2)),',',dimag(rslt(2)),')'
        write(iu,102) 'd0c 1: (',dreal(rslt(1)),',',dimag(rslt(1)),')'
        write(iu,102) 'd0c 0: (',dreal(rslt(0)),',',dimag(rslt(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_olo_rot4(pp,mm ,p1,p2,p3,p4,p12,p23 ,m1,m2,m3,m4)
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
     & rp(6),small,scl,avh_olo_prec
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
      small = scl*avh_olo_prec()*1d2
      call avh_olo_kin_set(0)
*
      jj = 1
      if (     rp(1).ge.-small.and.rp(2).ge.-small
     &    .and.rp(3).ge.-small.and.rp(4).ge.-small ) then
        call avh_olo_kin_set(1)
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
c      write(6,*) 'WARNING from avh_olo_cmplx: jj put to 1' !DEBUG
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


      subroutine avh_olo_kin_set(ikin)
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer ikin
      logical                 kin(3)
      common/avh_olo_kin_com/ kin
      kin(1) = .false.
      kin(2) = .false.
      kin(3) = .false.
      if (1.le.ikin.and.ikin.le.3) kin(ikin) = .true.
      end
*
      logical function avh_olo_kin(ikin)
      integer ikin
      logical                 kin(3)
      common/avh_olo_kin_com/ kin
      avh_olo_kin = kin(ikin)
      end
