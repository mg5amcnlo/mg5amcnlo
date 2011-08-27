************************************************************************
* This is the file  avh_olo_func.f  of the package                     *
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


      function avh_olo_sqrt(xx,sgn)
*  ********************************************************************
*  * Returns the square-root of xx .
*  * If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
*  * imaginary and has the same sign as  sgn .
*  ********************************************************************
      implicit none
      double complex avh_olo_sqrt,xx
     &,zz
      double precision sgn
     &,xim,xre
      xim = dimag(xx)
      if (xim.eq.0d0) then
        xre = dreal(xx)
        if (xre.ge.0d0) then
          zz = dcmplx(dsqrt(xre),0d0)
        else
          zz = dcmplx(0d0,dsign(dsqrt(-xre),sgn))
        endif
      else
        zz = cdsqrt(xx)
      endif
      avh_olo_sqrt = zz
      end


      subroutine avh_olo_abc( x1,x2 ,dd ,aa,bb,cc ,imode )
*  ********************************************************************
*  * Returns the solutions  x1,x2  to the equation  aa*x^2+bb*x+cc=0
*  * Also returns  dd = aa*(x1-x2)
*  * If  imode=/=0  it uses  dd  as input as value of  sqrt(b^2-4*a*c)
*  ********************************************************************
      implicit none
      double complex x1,x2,dd ,aa,bb,cc
     &,zero,two,four,qq,hh
      parameter(zero=(0d0,0d0),two=(2d0,0d0),four=(4d0,0d0))
      double precision
     & r1,r2
      integer imode
     &,avh_olo_un_get
*
      if (aa.eq.zero) then
        if (bb.eq.zero) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_abc: no solutions, returning 0'
          x1 = zero
          x2 = zero
          dd = zero
        else
c          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
c     &      'WARNING from avh_olo_abc: only one solution, putting x2=x1'
c     &     ,' and dd=bb'
          x1 = -cc/bb
          x2 = x1
          dd = bb
        endif
      elseif (cc.eq.zero) then
        dd = -bb
        x1 = dd/aa
        x2 = zero
      else
        if (imode.eq.0) dd = cdsqrt(bb*bb - four*aa*cc)
        qq = -bb+dd
        hh = -bb-dd
        r1 = cdabs(qq)
        r2 = cdabs(hh)
        if (r1.ge.r2) then
          x1 = qq/(two*aa)
          x2 = (two*cc)/qq
        else
          qq = hh
          x2 = qq/(two*aa)
          x1 = (two*cc)/qq
        endif
      endif
      end


      subroutine avh_olo_rfun(rr,dd ,qq)
*  ********************************************************************
*  * Returns  rr  such that  qq = rr + 1/rr  and  Im(rr)  has the same
*  * sign as  Im(qq) .
*  * If  Im(qq)  is zero, then  Im(rr)  is negative or zero.
*  * If  Im(rr)  is zero, then  |rr| > 1/|rr| .
*  * Also returns  dd = rr - 1/rr .
*  ********************************************************************
      implicit none
      double complex rr,dd ,qq
     &,two,four,r2
      double precision
     & aa,bb
      integer
     & ir,ik
      parameter(two=(2d0,0d0),four=(4d0,0d0))
      dd = cdsqrt(qq*qq-four)
      rr = qq+dd
      r2 = qq-dd
      aa = cdabs(rr)
      bb = cdabs(r2)
      if (bb.gt.aa) then
        rr = r2
        dd = -dd
      endif
      aa = dimag(qq)
      bb = dimag(rr)
      if (aa.eq.0d0) then
        if (bb.le.0d0) then
          rr = rr/two
        else
          rr = two/rr
          dd = -dd
        endif
      else
        ik = idnint(dsign(1d0,aa))
        ir = idnint(dsign(1d0,bb))
        if (ir.eq.ik) then
          rr = rr/two
        else
          rr = two/rr
          dd = -dd
        endif
      endif
      end

      subroutine avh_olo_rfun0(rr ,dd,qq)
*  ********************************************************************
*  * Like rfun, but now  dd  is input, which may get a minus sign
*  ********************************************************************
      implicit none
      double complex rr,dd ,qq
     &,two,four,r2
      double precision
     & aa,bb
      integer
     & ir,ik
      parameter(two=(2d0,0d0),four=(4d0,0d0))
      rr = qq+dd
      r2 = qq-dd
      aa = cdabs(rr)
      bb = cdabs(r2)
      if (bb.gt.aa) then
        rr = r2
        dd = -dd
      endif
      aa = dimag(qq)
      bb = dimag(rr)
      if (aa.eq.0d0) then
        if (bb.le.0d0) then
          rr = rr/two
        else
          rr = two/rr
          dd = -dd
        endif
      else
        ik = idnint(dsign(1d0,aa))
        ir = idnint(dsign(1d0,bb))
        if (ir.eq.ik) then
          rr = rr/two
        else
          rr = two/rr
          dd = -dd
        endif
      endif
      end


      subroutine avh_olo_conv(zz,iz ,xx,sgn)
*  ********************************************************************
*  * Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
*  * is positive. If  Im(x)=0  and  Re(x)<0  then  iz  becomes the
*  * sign of  sgn .
*  ********************************************************************
      implicit none
      double complex zz,xx
      double precision sgn
      integer iz
      double precision
     & xre,xim
      xre = dreal(xx)
      if (xre.ge.0d0) then
        zz = xx
        iz = 0
      else
        xim = dimag(xx)
        if (xim.eq.0d0) then
          zz = dcmplx(-xre,0d0)
          iz = idnint(dsign(1d0,sgn))
        else
          zz = -xx
          iz = idnint(dsign(1d0,xim)) ! xim = -Im(zz)
        endif
      endif
      end


      function avh_olo_sheet(xx,ix)
*  ********************************************************************
*  * Returns the number of the Riemann-sheet (times 2) for the complex
*  * number  xx*exp(ix*imag*pi) . The real part of xx is assumed to be
*  * positive or zero. Examples:
*  * xx=1+imag, ix=-1 -> ii= 0 
*  * xx=1+imag, ix= 1 -> ii= 2 
*  * xx=1-imag, ix=-1 -> ii=-2 
*  * xx=1-imag, ix= 1 -> ii= 0 
*  * xx=1     , ix= 1 -> ii= 0  convention that log(-1)=pi on
*  * xx=1     , ix=-1 -> ii=-2  the principal Riemann-sheet
*  ********************************************************************
      implicit none
      double complex xx
      integer avh_olo_sheet ,ix
     &,ii,jj
      double precision
     & xim
      ii = ix/2*2
      jj = ix-ii
      xim = dimag(xx)
      if (xim.le.0d0) then ! also xim=0 <==> log(-1)=pi, not -pi
        if (jj.eq.-1) ii = ii-2
      else
        if (jj.eq. 1) ii = ii+2
      endif
      avh_olo_sheet = ii
      end


      subroutine avh_olo_prd(zz,iz ,yy,iy ,xx,ix)
*  ********************************************************************
*  * Return the product  zz  of  yy  and  xx  
*  * keeping track of (the multiple of pi of) the phase  iz  such that
*  * the real part of  zz  remains positive 
*  ********************************************************************
      implicit none
      double complex zz,yy,xx
      integer iz,iy,ix
      zz = yy*xx
      iz = iy+ix
      if (dreal(zz).lt.0d0) then
        iz = iz + idnint(dsign(1d0,dimag(xx)))
        zz = -zz
      endif
      end

      subroutine avh_olo_rat(zz,iz ,yy,iy ,xx,ix)
*  ********************************************************************
*  * Return the ratio  zz  of  yy  and  xx  
*  * keeping track of (the multiple of pi of) the phase  iz  such that
*  * the real part of  zz  remains positive 
*  ********************************************************************
      implicit none
      double complex zz,yy,xx
      integer iz,iy,ix
      zz = yy/xx
      iz = iy-ix
      if (dreal(zz).lt.0d0) then
        iz = iz - idnint(dsign(1d0,dimag(xx)))
        zz = -zz
      endif
      end

      subroutine avh_olo_mlt(yy,iy ,xx,ix)
*  ********************************************************************
*  * Multiply  yy  with  xx  keeping track of (the multiple of pi of) 
*  * the phase  iy  such that the real part of  yy  remains positive 
*  ********************************************************************
      implicit none
      double complex yy,xx
      integer iy,ix
      yy = yy*xx
      iy = iy+ix
      if (dreal(yy).lt.0d0) then
        iy = iy + idnint(dsign(1d0,dimag(xx)))
        yy = -yy
      endif
      end

      subroutine avh_olo_div(yy,iy ,xx,ix)
*  ********************************************************************
*  * Divide  yy  by  xx  keeping track of (the multiple of pi of) 
*  * the phase  iy  such that the real part of  yy  remains positive 
*  ********************************************************************
      implicit none
      double complex yy,xx
      integer iy,ix
      yy = yy/xx
      iy = iy-ix
      if (dreal(yy).lt.0d0) then
        iy = iy - idnint(dsign(1d0,dimag(xx)))
        yy = -yy
      endif
      end


      function avh_olo_li2c2(x1,ip1 ,x2,ip2)
*  ********************************************************************
*  * avh_olo_li2c2 = ( li2(x1,ip1) - li2(x2,ip2) )/(x1-x2)
*  *
*  *                        /1    ln(1-zz*t)
*  * where  li2(x1,ip1) = - |  dt ----------
*  *                        /0        t
*  * with  zz = 1 - ( |Re(x1)| + imag*Im(x1) )*exp(imag*pi*ip1)
*  * and similarly for li2(x2,ip2)
*  ********************************************************************
      implicit none
      double complex x1,x2  ,avh_olo_li2c2
     &,x1r,x2r,delta ,xx,xr,omx,del,hh,ff(0:20),zz ,one
     &,avh_olo_li2c,avh_olo_logc2
      parameter(one=(1d0,0d0))
      double precision 
     & thrs,thrs1,pi,avh_olo_pi,avh_olo_prec
      integer ip1,ip2
     &,ix,ih,init,nmax,ii ,avh_olo_un_get
      data init/0/
      save init,nmax,pi,thrs,thrs1
*
      if (init.eq.0) then
        init = 1
        pi = avh_olo_pi()
        thrs1 = 1d0*avh_olo_prec()
        if (avh_olo_prec().gt.1d-24) then
          thrs = 0.11d0 ! double precision
          nmax = 12
        else
          thrs = 0.008d0 ! quadruple precision
          nmax = 12
        endif
      endif
*
      if (ip1.eq.ip1/2*2) then
        x1r = dcmplx( dabs(dreal(x1)), dimag(x1))
      else
        x1r = dcmplx(-dabs(dreal(x1)),-dimag(x1))
      endif     
      if (ip2.eq.ip2/2*2) then
        x2r = dcmplx( dabs(dreal(x2)), dimag(x2))
      else
        x2r = dcmplx(-dabs(dreal(x2)),-dimag(x2))
      endif
      delta = x1r-x2r
*
      if (ip1.ne.ip2) then !OLD
        if (delta.eq.dcmplx(0d0)) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_li2c2: ip1,ip2,delta=',ip1,ip2,delta
          avh_olo_li2c2 = dcmplx(0d0)
        else
          avh_olo_li2c2 = ( avh_olo_li2c(x1,ip1)
     &                     -avh_olo_li2c(x2,ip2) )/delta
        endif
      else
        if (cdabs(delta/x1).gt.thrs) then
          avh_olo_li2c2 = ( avh_olo_li2c(x1,ip1)
     &                     -avh_olo_li2c(x2,ip2) )/delta
        else
          xx  = x1
          xr  = x1r
          omx = one-xr
          del = delta
          hh = one-x2r
          if (cdabs(hh).gt.cdabs(omx)) then
            xx = x2
            xr = x2r
            omx = hh
            del = -delta
          endif
          if (cdabs(omx).lt.thrs1) then
            zz = -one-omx/2-del/4
          else
            ih = ip1 - ip1/2*2
            ff(0) = avh_olo_logc2(xx,ih)
            hh = -one
            do ii=1,nmax
              hh = -hh/xr
              ff(ii) = ( hh/ii + ff(ii-1) )/omx
            enddo
            zz = ff(nmax)/(nmax+1)
            do ii=nmax-1,0,-1
              zz = ff(ii)/(ii+1) - zz*del
            enddo
          endif
          ih = ip1-ih
          if (ih.ne.0) then
            omx = one-x1r
            call avh_olo_conv( xx,ix ,(one-x2r)/omx,0d0 )
            zz = zz + dcmplx(0d0,-ih*pi)*avh_olo_logc2( xx,ix )/omx
          endif
          avh_olo_li2c2 = zz
        endif
      endif
      end

      function avh_olo_logc2(xx,iph)
*  ********************************************************************
*  * log(xx)/(1-xx)
*  * with  log(xx) = log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex xx ,avh_olo_logc2
     &,omx,avh_olo_logc
      double precision
     & thrs,avh_olo_prec
      integer iph
     &,init ,avh_olo_un_get
      data init/0/
      save init,thrs
*
      if (init.eq.0) then
        init = 1
        thrs = 1d1*avh_olo_prec()
      endif
*
      if (iph.eq.iph/2*2) then
        omx = dcmplx(1d0-dabs(dreal(xx)),-dimag(xx))
      else
        omx = dcmplx(1d0+dabs(dreal(xx)), dimag(xx))
      endif
*
      if (iph.ne.0) then
        if (omx.eq.dcmplx(0d0)) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_logc2: 1-xx,iph=',omx,iph
          avh_olo_logc2 = dcmplx(0d0)
        else
          avh_olo_logc2 = avh_olo_logc(xx,iph)/omx
        endif
      else
        if (cdabs(omx).lt.thrs) then
          avh_olo_logc2 = dcmplx(-1d0)-omx/2
        else
          avh_olo_logc2 = avh_olo_logc(xx,iph)/omx
        endif
      endif
      end

      function avh_olo_li2c(xx,iph)
*  ********************************************************************
*  *                  /1    ln(1-zz*t)
*  * avh_olo_li2c = - |  dt ---------- 
*  *                  /0        t
*  * with  zz = 1 - ( |Re(xx)| + imag*Im(xx) )*exp(imag*pi*iph)
*  * Examples:
*  *   In order to get the dilog of  1+imag  use  xx=1+imag, iph= 0
*  *   In order to get the dilog of  1-imag  use  xx=1-imag, iph= 0
*  *   In order to get the dilog of -1+imag  use  xx=1-imag, iph= 1
*  *   In order to get the dilog of -1-imag  use  xx=1+imag, iph=-1
*  * Add multiples of  2  to  iph  in order to get the result on
*  * different Riemann-sheets.
*  ********************************************************************
      implicit none
      double complex xx ,avh_olo_li2c
     &,yy,lyy,loy,zz,z2,liox,zero,one,avh_olo_li2a
      parameter(zero=(0d0,0d0),one=(1d0,0d0))
      double precision
     & bb(36),pi,pi2o6 ,rex,imx,avh_olo_pi,avh_olo_bern,avh_olo_prec
      integer iph
     &,init,nn,ii,iyy
      logical
     & x_gt_1 , y_lt_h
      data init/0/
      save init,nn,bb,pi,pi2o6
*
      if (init.eq.0) then
        init = 1
        pi = avh_olo_pi()
        pi2o6  = pi**2/6d0
        if (avh_olo_prec().gt.1d-24) then
          nn = 18 ! double precision
        else
          nn = 36 ! quadruple precision
        endif
        do ii=1,nn
          bb(ii) = avh_olo_bern(ii)
        enddo
      endif
*     
      rex = dreal(xx)
      imx = dimag(xx)
*     
      if (imx.eq.0d0) then
        liox = avh_olo_li2a(rex,iph)
      else
        rex = dabs(rex)
*
        if (iph.eq.iph/2*2) then
          yy = dcmplx(rex,imx)
          iyy = iph
        else
          yy = dcmplx(-rex,-imx)
* * Notice that  iyy=iph/2*2  does not deal correctly with the
* * situation when  iph-iph/2*2 = sign(Im(xx)) . The following does:
          iyy = iph + idnint(sign(1d0,imx))
        endif
*
        x_gt_1 = (cdabs(xx).gt.1d0)
        if (x_gt_1) then
          yy = one/yy
          iyy = -iyy
        endif
        lyy = cdlog(yy)
        loy = cdlog(one-yy)
*
        y_lt_h = (dreal(yy).lt.0.5d0)
        if (y_lt_h) then
          zz = -loy
        else
          zz = -lyy
        endif
*
        z2 = zz*zz
        liox = dcmplx(bb(nn))
        do ii=nn,4,-2
          liox = dcmplx(bb(ii-2)) + liox*z2/(ii*(ii+1))
        enddo
        liox = dcmplx(bb(1)) + liox*zz/3
        liox = zz + liox*z2/2
*
        if (y_lt_h) liox = dcmplx(pi2o6) - liox - loy*lyy
*
        liox = liox - loy*dcmplx(0d0,pi*iyy)
*
        if (x_gt_1) liox = -liox - (lyy + dcmplx(0d0,iyy*pi))**2/2
      endif
      avh_olo_li2c = liox
      end

      function avh_olo_logc(xx,iph)
*  ********************************************************************
*  * Returns  log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex xx ,avh_olo_logc
      double precision
     & pi ,avh_olo_pi
      integer iph
     &,init ,avh_olo_un_get
      data init/0/
      save init,pi
*
      if (init.eq.0) then
        init = 1
        pi = avh_olo_pi()
      endif
*
      if (xx.eq.dcmplx(0d0)) then
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'ERROR in avh_olo_logc: xx =',xx
        avh_olo_logc = dcmplx(0d0)
      else
        avh_olo_logc = cdlog( dcmplx(dabs(dreal(xx)),dimag(xx)) )
     &               + dcmplx(0d0,iph*pi)
      endif
      end


      function avh_olo_li2a2(x1,ip1 ,x2,ip2)
*  ********************************************************************
*  * avh_olo_li2a2 = ( li2(x1,ip1) - li2(x2,ip2) )/(x1-x2)
*  *
*  *                        /1    ln(1-zz*t)
*  * where  li2(x1,ip1) = - |  dt ----------
*  *                        /0        t
*  * with  zz = 1 - |x1|*exp(imag*pi*ip1)  and similarly for li2(x2,ip2)
*  ********************************************************************
      implicit none
      double complex avh_olo_li2a2
     &,ff(0:20),zz,avh_olo_li2a,avh_olo_loga2
      double precision x1,x2,delta
     &,thrs,thrs1,x1r,x2r,xx,omx,del,hh,avh_olo_prec
      integer ip1,ip2
     &,init,nmax,ii ,avh_olo_un_get
      data init/0/
      save init,nmax,thrs,thrs1
*
      if (init.eq.0) then
        init = 1
        thrs1 = 1d0*avh_olo_prec()
        if (avh_olo_prec().gt.1d-24) then
          thrs = 0.11d0 ! double precision
          nmax = 12
        else
          thrs = 0.008d0 ! quadruple precision
          nmax = 12
        endif
      endif
*
      if (ip1.eq.ip1/2*2) then
        x1r = dabs(x1)
      else
        x1r = -dabs(x1)
      endif     
      if (ip2.eq.ip2/2*2) then
        x2r = dabs(x2)
      else
        x2r = -dabs(x2)
      endif
      delta = x1r-x2r
*      
      if (ip1.ne.ip2) then
        if (delta.eq.0d0) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_li2a2: ip1,ip2,delta=',ip1,ip2,delta
          avh_olo_li2a2 = dcmplx(0d0)
        else
          avh_olo_li2a2 = ( avh_olo_li2a(x1,ip1)
     &                     -avh_olo_li2a(x2,ip2) )/dcmplx(delta)
        endif
      else
        if (dabs(delta/x1).gt.thrs) then
          avh_olo_li2a2 = ( avh_olo_li2a(x1,ip1)
     &                     -avh_olo_li2a(x2,ip2) )/dcmplx(delta)
        else
          xx  = x1r
          omx = 1d0-xx
          del = delta
          hh = 1d0-x2r
          if (dabs(hh).gt.dabs(omx)) then
            xx = x2r
            omx = hh
            del = -delta
          endif
          if (dabs(omx).lt.thrs1) then
            zz = dcmplx(-1d0-omx/2-del/4)
          else
            ff(0) = avh_olo_loga2(xx,ip1) ! ip1=ip2
            hh = -1d0
            do ii=1,nmax
              hh = -hh/xx
              ff(ii) = ( dcmplx(hh/ii) + ff(ii-1) )/dcmplx(omx)
            enddo
            zz = ff(nmax)/(nmax+1)
            do ii=nmax-1,0,-1
              zz = ff(ii)/(ii+1) - zz*dcmplx(del)
            enddo
          endif
          avh_olo_li2a2 = zz
        endif
      endif
      end

      function avh_olo_loga2(xx,iph)
*  ********************************************************************
*  * log(xx)/(1-xx)  with  xx = log|xx| + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex avh_olo_loga2
     &,avh_olo_loga
      double precision xx,omx
     &,thrs,avh_olo_prec
      integer iph
     &,init ,avh_olo_un_get
      data init/0/
      save init,thrs
*
      if (init.eq.0) then
        init = 1
        thrs = 1d1*avh_olo_prec()
      endif
*
      if (iph.eq.iph/2*2) then
        omx = 1d0-dabs(xx)
      else
        omx = 1d0+dabs(xx)
      endif
*
      if (iph.ne.0) then
        if (omx.eq.0d0) then
          if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &      'ERROR in avh_olo_loga2: 1-xx,iph=',omx,iph
          avh_olo_loga2 = dcmplx(0d0)
        else
          avh_olo_loga2 = avh_olo_loga(xx,iph)/dcmplx(omx)
        endif
      else
        if (dabs(omx).lt.thrs) then
          avh_olo_loga2 = dcmplx(-1d0-omx/2d0)
        else
          avh_olo_loga2 = avh_olo_loga(xx,iph)/dcmplx(omx)
        endif
      endif
      end


      function avh_olo_li2a(xx,iph)
*  ********************************************************************
*  *                  /1    ln(1-zz*t)
*  * avh_olo_li2a = - |  dt ---------- 
*  *                  /0        t
*  * with  zz = 1 - |xx|*exp(imag*pi*iph)
*  * Examples:
*  *   In order to get the dilog of  1.1  use  xx=1.1, iph=0
*  *   In order to get the dilog of -1.1  use  xx=1.1, iph=1
*  * Add multiples of  2  to  iph  in order to get the result on
*  * different Riemann-sheets.
*  ********************************************************************
      implicit none
      double complex avh_olo_li2a
     &,cliox
      double precision xx
     &,bb(30),pi,pi2o6,rr,yy,lyy,loy,zz,z2,liox
     &,avh_olo_pi,avh_olo_bern,avh_olo_prec
      integer iph
     &,init,nn,ii,ntwo,ione
      logical
     & positive , r_gt_1 , y_lt_h
      data init/0/
      save init,nn,bb,pi,pi2o6
*
      if (init.eq.0) then
        init = 1
        pi = avh_olo_pi()
        pi2o6  = pi**2/6d0
        if (avh_olo_prec().gt.1d-24) then
          nn = 16 ! double precision
        else
          nn = 30 ! quadruple precision
        endif
        do ii=1,nn
          bb(ii) = avh_olo_bern(ii)
        enddo
      endif
*     
      rr = dabs(xx)
      ntwo = iph/2*2
      ione = iph - ntwo
      positive = (ione.eq.0)
*     
      if     (rr.eq.0d0) then
        cliox = dcmplx(pi2o6,0d0)
      elseif (rr.eq.1d0.and.positive) then
        cliox = dcmplx(0d0,0d0)
      else
        yy  = rr
        lyy = dlog(rr)
        if (.not.positive) yy = -yy
*
        r_gt_1 = (rr.gt.1d0)
        if (r_gt_1) then
          yy   = 1d0/yy
          lyy  = -lyy
          ntwo = -ntwo
          ione = -ione
        endif
        loy = dlog(1d0-yy) ! log(1-yy) is always real
*
        y_lt_h = (yy.lt.0.5d0)
        if (y_lt_h) then
          zz = -loy ! log(1-yy) is real
        else
          zz = -lyy ! yy>0.5 => log(yy) is real
        endif
*
        z2 = zz*zz
        liox = bb(nn)
        do ii=nn,4,-2
          liox = bb(ii-2) + liox*z2/(ii*(ii+1))
        enddo
        liox = bb(1) + liox*zz/3
        liox = zz + liox*z2/2
*
        cliox = dcmplx(liox)
*
        if (y_lt_h) then
          cliox = dcmplx(pi2o6) - cliox - dcmplx(loy*lyy , loy*pi*ione)
        endif
*
        cliox = cliox + dcmplx( 0d0 , -loy*pi*ntwo )
*
        if (r_gt_1) cliox = -cliox - dcmplx( -lyy , iph*pi )**2/2
      endif
      avh_olo_li2a = cliox
      end


      function avh_olo_loga(xx,iph)
*  ********************************************************************
*  * log( |xx|*exp(imag*pi*iph) ) = log|xx| + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex avh_olo_loga
      double precision xx
     &,rr,pi ,avh_olo_pi
      integer iph
     &,init ,avh_olo_un_get
      data init/0/
      save init,pi
*
      if (init.eq.0) then
        init = 1
        pi = avh_olo_pi()
      endif
*
      rr = dabs(xx)
      if (rr.eq.0d0.and.avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &  'ERROR in avh_olo_loga: |xx|=',rr
      avh_olo_loga = dcmplx(dlog(rr),iph*pi)
      end


      function avh_olo_bern(ii)
*  ********************************************************************
*  * the first nn Bernoulli numbers
*  ********************************************************************
      implicit none
      double precision avh_olo_bern
     &,bern(40)
      integer ii
     &,init,nn,jj ,avh_olo_un_get
      data init/0/
      save init,bern
*
      parameter(nn=36)
*
      if (init.eq.0) then
        init = 1
        do jj=3,nn-1,2
          bern(jj) = 0d0
        enddo
        bern( 1) = -1d0/2d0
        bern( 2) =  1d0/6d0
        bern( 4) = -1d0/30d0
        bern( 6) =  1d0/42d0
        bern( 8) = -1d0/30d0
        bern(10) =  5d0/66d0
        bern(12) = -691d0/2730d0
        bern(14) =  7d0/6d0
        bern(16) = -3617d0/510d0
        bern(18) =  43867d0/798d0
        bern(20) = -174611d0/330d0
        bern(22) =  854513d0/138d0
        bern(24) = -236364091d0/2730d0
        bern(26) =  8553103d0/6d0
        bern(28) = -23749461029d0/870d0
        bern(30) =  8615841276005d0/14322d0
        bern(32) = -7709321041217d0/510d0
        bern(34) =  2577687858367d0/6d0
        bern(36) = -26315271553053477373d0/1919190d0
        bern(38) =  2929993913841559d0/6d0
        bern(40) = -261082718496449122051d0/13530d0
      endif
      if (ii.le.nn) then
        avh_olo_bern = bern(ii)
      else
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &   'ERROR in avh_olo_bern: bernoulli(',ii,') not yet implemented'
      endif
      end


      function avh_olo_pi()
*  ********************************************************************
*  * the number  pi=3.14...
*  ********************************************************************
      implicit none
      double precision avh_olo_pi
     &,pi
      integer
     & init
      data init/0/
      save init,pi
*
      if (init.eq.0) then
        init = 1
        pi = 4*datan(1d0)
      endif
      avh_olo_pi = pi
      end

 
      function avh_olo_prec()
*  ********************************************************************
*  * the smallest number  prec  satisfying  1+prec = dexp(dlog(1+prec))
*  ********************************************************************
      implicit none
      double precision avh_olo_prec
     &,prec,xx,yy
      integer
     & init ,avh_olo_un_get
      data init/0/
      save init,prec
*
      if (init.eq.0) then
        init = 1
        xx = 1d0
        yy = xx
        do while (xx.eq.yy)
          prec = xx
          xx = xx/2
          yy = -1d0+dexp(dlog(1d0+xx))
        enddo
        if (avh_olo_un_get().gt.0) write(avh_olo_un_get(),*)
     &    'MESSAGE from avh_olo_prec: precision set to',prec
      endif
      avh_olo_prec = prec
      end
