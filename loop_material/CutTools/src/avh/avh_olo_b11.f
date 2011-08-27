************************************************************************
* This is the file  avh_olo_b11.f  of the package                      *
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

      subroutine avh_olo_b11c( b11,b00,b1,b0 ,pp_in,m0_in,m1_in )
*  ********************************************************************
*  * Return the Passarino-Veltman functions b11,b00,b1,b0 , for
*  *
*  *      C   /      d^(Dim)q
*  *   ------ | -------------------- = b0
*  *   i*pi^2 / [q^2-m0][(q+p)^2-m1]
*  *
*  *      C   /    d^(Dim)q q^mu
*  *   ------ | -------------------- = p^mu b1
*  *   i*pi^2 / [q^2-m0][(q+p)^2-m1]
*  *
*  *      C   /  d^(Dim)q q^mu q^nu
*  *   ------ | -------------------- = g^{mu,nu} b00 + p^mu p^nu b11
*  *   i*pi^2 / [q^2-m0][(q+p)^2-m1]
*  *
*  ********************************************************************
      implicit none
      double complex b11(0:2),b00(0:2),b1(0:2),b0(0:2),a1(0:2),a0(0:2)
     &,pp_in,m0_in,m1_in,pp,m0,m1,zmu
     &,ff,gg,c1,c2 ,zero,one,two,three,four,six
      parameter(zero=(0d0,0d0),one=(1d0,0d0),two=(2d0,0d0))
      parameter(three=(3d0,0d0),four=(4d0,0d0),six=(6d0,0d0))
      double precision
     & hh,rr,ap,am0,am1,smax,mu2,thrs
     &,avh_olo_mu_get,avh_olo_thrs
      integer
     & ntrm,ii,avh_olo_un_get,avh_olo_print
      double complex cc(0:20)
      logical init ,avh_olo_os_get
      common/avh_olo_b_com/ thrs,ntrm
      data init/.true./
      save init
*
      mu2 = avh_olo_mu_get()**2
      zmu = dcmplx(mu2)
      pp = pp_in
      m0 = m0_in
      m1 = m1_in
*
      if (init) then
        init = .false.
        call avh_olo_hello
        call avh_olo_cbm(b0 ,pp,m0,m1 ,zmu)
      endif
*
      ap = dreal(pp)
      if (dimag(pp).ne.0d0) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_b11c: momentum with non-zero imaginary'
     &   ,' part, putting it to zero.'
        pp = dcmplx( ap ,0d0 )
      endif
      ap = dabs(ap)
*
      am1 = dreal(m1)
      hh  = dimag(m1)
      if (hh.gt.0d0) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &     'ERROR in avh_olo_b11c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m1 = dcmplx( am1 ,-hh )
      endif
      am1 = dabs(am1) + dabs(hh)
*
      am0 = dreal(m0)
      hh  = dimag(m0)
      if (hh.gt.0d0) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &     'ERROR in avh_olo_b11c: mass-squared has positive imaginary'
     &    ,' part, switching its sign.'
        m0 = dcmplx( am0 ,-hh )
      endif
      am0 = dabs(am0) + dabs(hh)
*
      rr = max(am0,am1)
      smax = avh_olo_thrs(max(rr,ap,mu2))
      if (avh_olo_os_get().and.ap.lt.smax.and.rr.lt.smax) then
        ap = 0d0
        rr = 0d0
      endif
*
      if (rr.eq.0d0) then
        if (ap.eq.0d0) then
          call avh_olo_zero( b0 )
          call avh_olo_zero( b1 )
          call avh_olo_zero( b00 )
          call avh_olo_zero( b11 )
          return
        endif
        rr = 1d0+thrs
      else
        rr = ap/rr
      endif
*
      ff = pp - m1 + m0
      gg = m0 + m1 - pp/three
      b0(2)  = zero
      b1(2)  = zero
      b00(2) = zero
      b11(2) = zero
      b0(1)  = one
      b1(1)  = -one/two
      b00(1) = gg/four
      b11(1) = one/three
      call avh_olo_cam( a1 ,m0 ,zmu )
      call avh_olo_cam( a0 ,m1 ,zmu )
*
      if (rr.le.thrs) then
c        write(6,*) 'expansion' !DEBUG
        call avh_olo_bexp( cc ,m0,m1 ,zmu ,ntrm)
        c2 = cc(ntrm)
        do ii=ntrm-1,2,-1
          c2 = cc(ii) + pp*c2
        enddo
        c1 = cc(1) + pp*c2
        b0(0)  = cc(0) + pp*c1
        b1(0)  = -( cc(0) + ff*c1 )/two
        b00(0) = ( a0(0) + ff*b1(0) + two*m0*b0(0) + gg )/six
        b11(0) = cc(0) + (ff+m0-m1)*cc(1) + ff*ff*c2 - m0*c1
        b11(0) = ( b11(0) + one/six )/three
      else
        call avh_olo_cbm(b0 ,pp,m0,m1 ,zmu)
        b1(0)  = ( a1(0) - a0(0) - ff*b0(0) )/(two*pp)
        b00(0) = ( a0(0) + ff*b1(0) + two*m0*b0(0) + gg )/six
        b11(0) = ( a0(0) - two*ff*b1(0) - m0*b0(0) - gg/two )/(three*pp)
      endif
*
      ii = avh_olo_print()
      if (ii.gt.0) then
        write(ii,'(a7,d39.32)') 'onshell',avh_olo_thrs(1d0)
        write(ii,'(a2,d39.32)') 'mu',avh_olo_mu_get()
        write(ii,102) '   pp: (',dreal(pp_in),',',dimag(pp_in),')'
        write(ii,102) '   m0: (',dreal(m0_in),',',dimag(m0_in),')'
        write(ii,102) '   m1: (',dreal(m1_in),',',dimag(m1_in),')'
        write(ii,102) 'b11 2: (',dreal(b11(2)),',',dimag(b11(2)),')'
        write(ii,102) 'b11 1: (',dreal(b11(1)),',',dimag(b11(1)),')'
        write(ii,102) 'b11 0: (',dreal(b11(0)),',',dimag(b11(0)),')'
        write(ii,102) 'b00 2: (',dreal(b00(2)),',',dimag(b00(2)),')'
        write(ii,102) 'b00 1: (',dreal(b00(1)),',',dimag(b00(1)),')'
        write(ii,102) 'b00 0: (',dreal(b00(0)),',',dimag(b00(0)),')'
        write(ii,102) ' b1 2: (',dreal(b1(2)),',',dimag(b1(2)),')'
        write(ii,102) ' b1 1: (',dreal(b1(1)),',',dimag(b1(1)),')'
        write(ii,102) ' b1 0: (',dreal(b1(0)),',',dimag(b1(0)),')'
        write(ii,102) ' b0 2: (',dreal(b0(2)),',',dimag(b0(2)),')'
        write(ii,102) ' b0 1: (',dreal(b0(1)),',',dimag(b0(1)),')'
        write(ii,102) ' b0 0: (',dreal(b0(0)),',',dimag(b0(0)),')'
  102   format(a8,d39.32,a1,d39.32,a1)
      endif
*
      end


      subroutine avh_olo_bexp(cc ,m1i,m2i ,zmu ,ntrm)
*  ********************************************************************
*  * Returns the first 1+ntrm  terms of the expansion in p^2 of the
*  * finite part of the 1-loop scalar 2-point function 
*  ********************************************************************
      implicit none
      integer ntrm
      double complex cc(0:ntrm) ,m1i,m2i,zmu
     &,m1,m2,qm1,qm2,qzz,zz,oz,xx,logz,tt(ntrm) ,zero,one ,avh_olo_logc
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      integer
     & init ,im1,im2,izz ,ii,nn ,avh_olo_un_get
      double precision
     & rr,thrs ,aa(8),avh_olo_prec
      data init/0/
      save init,thrs,nn,aa
*
c      write(6,*) 'MESSAGE from avh_olo_bexp: you are calling me' !DEBUG
*
      if (init.eq.0) then
        init = 1
        if (avh_olo_prec().gt.1d-24) then
          thrs = 0.01d0 ! double precision
          nn = 7        !
        else
          thrs = 0.0001d0 ! quadruple precision, not tested
          nn = 7          !
        endif
        do ii=1,nn
          aa(ii) = 1d0/dble(ii*(ii+1)) 
        enddo
      endif
*
      if (cdabs(m1i).le.cdabs(m2i)) then
        m1 = m1i
        m2 = m2i
      else
        m1 = m2i
        m2 = m1i
      endif
*
      if (m2.eq.zero) then
*
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_bexp: m1=m2=0, returning 0'
        do ii=0,ntrm
          cc(ii) = zero
        enddo
*
      else
*
        call avh_olo_conv( qm1,im1 ,m1,-1d0 ) 
        call avh_olo_conv( qm2,im2 ,m2,-1d0 )
        call avh_olo_rat( qzz,izz ,qm1,im1 ,qm2,im2 )
        if (izz.eq.izz/2*2) then
          zz = qzz
        else
          zz = -qzz
        endif
*
        if (m1.eq.zero) then
          cc(0) = one - avh_olo_logc( qm2/zmu,im2 )
        else
          oz = one-zz
          rr = cdabs(oz)
          if (rr.lt.thrs) then
            xx = dcmplx(aa(nn))
            do ii=nn-1,1,-1
              xx = dcmplx(aa(ii)) + oz*xx
            enddo
            xx = oz*xx
          else
            logz = avh_olo_logc( qzz,izz )
            xx = zz*logz + oz
            xx = xx/oz
          endif
          cc(0) = xx - avh_olo_logc( qm2/zmu,im2 )
        endif
*
        zz = one-zz
        xx = one
        call avh_olo_bexp1(tt ,ntrm,zz)
        do ii=1,ntrm
          xx = xx*m2
          cc(ii) = tt(ii)/(ii*xx)
        enddo
*
      endif
      end


      subroutine avh_olo_bexp1(tt ,ntrm,zz)
*  ********************************************************************
*  * Returns  tt(n) = int( ( x*(1-x)/(1-zz*x) )^n , x=0..1 )
*  * for  n=1...ntrm  and  |zz|=<1
*  *
*  * Gives at least 2 correct digits (4 at quad.) for tt(ntrm),
*  * and increasingly more digits for tt(i<ntrm)
*  *
*  * Uses recursion on integrals of the type
*  *    int( x^m * (1-x)^n / (1-z*x)^n , x=0..1 )
*  * and
*  *    int( x^m * (1-x)^n / (1+y*x)^(n+2) , x=0..1 )
*  * where  y = z/(1-z)
*  * The latter integrals are related to the original ones via the
*  * substitution  x <- 1-x  followed by  x <- (1-x)/(1+y*x)
*  ********************************************************************
      implicit none
      integer ntrm
     &,nn,nmax
      parameter(nmax=20)
      double complex tt(ntrm) ,zz
     &,tu(ntrm),tv(ntrm) ,tt0,tu0,tv0,yy,y2,oy ,zero,one
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      double precision
     & rr,thrs(nmax),thrsd(nmax),thrsq(nmax),avh_olo_prec
      integer
     & ii,jj,avh_olo_un_get
      logical
     & init
      data thrsd/  5d-5,  5d-3,  5d-2, 0.1d0,0.15d0,0.20d0,0.30d0,0.40d0
     &          ,0.50d0,0.60d0,0.65d0,0.68d0,0.72d0,0.74d0,0.76d0,0.78d0
     &          ,0.80d0,0.82d0,0.83d0,0.84d0/
      data thrsq/ 1d-10, 5d-5,1d-4,1d-3,7d-3,2d-2,4d-2,7d-2,0.1d0,0.13d0
     &          ,0.17d0,0.20d0,0.25d0,0.30d0,0.34d0,0.38d0,0.42d0,0.44d0
     &          ,0.47d0,0.50d0/
      data init/.true./
      save init,thrs
*
      if (init) then
        init = .false.
        if (avh_olo_prec().gt.1d-24) then
          do ii=1,nmax           !
            thrs(ii) = thrsd(ii) ! double precision
          enddo                  !
        else
          do ii=1,nmax           !
            thrs(ii) = thrsq(ii) ! quadruple precision
          enddo                  !
        endif
      endif
*
      rr = dreal(zz)
      nn = ntrm
      if (nn.lt.1) nn = 1
      if (nn.gt.20) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &     'WARNING from avh_olo_bexp1:'
     &    ,' ntrm =',nn,' > nmax =',nmax,', using ntrm=nmax'
        nn = nmax
        do ii=nn+1,ntrm
          tt(ii) = zero
        enddo
      endif
*
      if (zz.eq.one) then
c        write(6,'(a16,i4)') 'simple expansion',nn !DEBUG
        do ii=1,nn
          tt(ii) = dcmplx(1d0/dble(ii+1))
        enddo
      elseif (rr.lt.thrs(nn)) then
* * Backward recursion, number of correct decimals constant, so need
* * full precision from the start
c        write(6,'(a8,i4,d24.16)') 'Backward',nn,rr !DEBUG
        call avh_olo_bexp2(tt(nn),tv(nn-1),tu(nn-1) ,nn,zz)
        do ii=nn-1,2,-1
          jj = ii+1
          tt(ii) = 2*tv(ii) - dcmplx(dble(ii)/jj)*zz*tt(jj)
          tu(ii-1) = dcmplx(2d0+1d0/ii)*tt(ii) - zz*tu(ii)
          tv(ii-1) = (one-zz)*tu(ii-1) + zz*( 2*tt(ii) - zz*tu(ii) )
        enddo
        tt(1) = 2*tv(1) - dcmplx(0.5d0)*zz*tt(2)
      else
* * Foreward recursion, number of correct decimals decreases
c        write(6,'(a8,i4,d24.16)') 'Foreward',nn,rr !DEBUG
        yy = zz/(one-zz)
        y2 = yy*yy
        oy = one+yy ! one/(one-zz)
        tt0 = one-zz ! 1/(1+y)
        tu0 = ( oy*cdlog(oy)-yy )/( y2*oy )
        tv0 = tt0/2
        tt(1) = ( tt0-2*tu0 )/( 2*yy )
        tv(1) = ( tv0 - 3*tt(1) )/( 3*yy )
        tu(1) = ( oy*tu0 - 2*yy*tt(1) - tv0 )/y2
        do ii=2,nn
          jj = ii-1
          tt(ii) = ii*( tt(jj)-2*tu(jj) )/( (ii+1)*yy )
          tv(ii) = ( ii*tv(jj) - (ii+ii+1)*tt(ii) )/( (ii+2)*yy )
          tu(ii) = ( oy*tu(jj) - 2*yy*tt(ii) - tv(jj) )/y2
        enddo
        yy = oy
        do ii=1,nn
          oy = oy*yy
          tt(ii) = oy*tt(ii)
        enddo
      endif
      end


      subroutine avh_olo_bexp2(ff,fa,fb ,nn_in,zz)
*  ********************************************************************
*  * ff = Beta(nn+1,nn+1) * 2F1(nn  ,nn+1;2*nn+2;zz)
*  * fa = Beta(nn+1,nn  ) * 2F1(nn-1,nn+1;2*nn+1;zz)
*  * fb = Beta(nn  ,nn+1) * 2F1(nn  ,nn  ;2*nn+1;zz)
*  ********************************************************************
      implicit none
      double complex ff,fa,fb ,zz
      integer nn_in
     &,nn,aa,bb,cc,ii,ntrm,init,nmax ,avh_olo_un_get
      parameter(nmax=100)
      double precision
     & ac0,bc0,ai,bi,ci,ac,bc,qq(0:nmax),qa(0:nmax),qb(0:nmax),gg,ga
     &,logprec,avh_olo_prec
      data nn/0/
      save nn,qq,qa,qb,gg,ga,logprec
      if (nn.ne.nn_in) then
        init = 1
        nn = nn_in
        aa = nn-1
        bb = nn
        cc = nn+nn+1
        qq(0) = 1d0
        qa(0) = 1d0
        qb(0) = 1d0
        ac0 = dble(aa)/dble(cc)
        bc0 = dble(bb)/dble(cc)
        ntrm = nmax
        do ii=1,ntrm
          ai = dble(aa+ii)
          bi = dble(bb+ii)
          ci = dble(cc+ii)
          ac = ai/ci
          bc = bi/ci
          qq(ii) = qq(ii-1) * ai*bc  / ii
          qa(ii) = qa(ii-1) * ac0*bi / ii
          qb(ii) = qb(ii-1) * ai*bc0 / ii
          ac0 = ac
          bc0 = bc
        enddo
        ai = 1d0
        do ii=2,nn-1
          ai = ai*dble(ii)
        enddo
        ci = ai
        cc = nn+nn
        do ii=nn,cc
          ci = ci*dble(ii)
        enddo
        bi = ai*dble(nn)
        gg = bi*bi/(ci*dble(cc+1))
        ga = ai*bi/ci
        logprec = dlog(avh_olo_prec())
      endif
*
      ai = cdabs(zz)
      if (ai.gt.0d0) then
        ntrm = 1 + int(logprec/dlog(ai))
      else
        ntrm = 1
      endif
      if (ntrm.gt.nmax) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'WARNING from avh_olo_bexp2:'
     &   ,' ntrm =',ntrm,' > nmax =',nmax,', putting ntrm=nmax'
        ntrm = nmax
      endif
c      write(6,*) 'ntrm',ntrm !DEBUG
*
      ff = dcmplx(qq(ntrm))
      fa = dcmplx(qa(ntrm))
      fb = dcmplx(qb(ntrm))
      do ii=ntrm-1,0,-1
        ff = dcmplx(qq(ii)) + ff*zz
        fa = dcmplx(qa(ii)) + fa*zz
        fb = dcmplx(qb(ii)) + fb*zz
      enddo
      ff = dcmplx(gg)*ff
      fa = dcmplx(ga)*fa
      fb = dcmplx(ga)*fb
      end
