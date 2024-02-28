************************************************************************
* This is the file  avh_oni_func.f  of the package                     *
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


      subroutine avh_oni_conv(zz,iz ,xx,sgn)
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


      subroutine avh_oni_rat(zz,iz ,yy,iy ,xx,ix)
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


      function avh_oni_logc(xx,iph)
*  ********************************************************************
*  * Returns  log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex xx ,avh_oni_logc
      double precision
     & pi ,avh_oni_pi
      integer iph
     &,init ,avh_oni_un_get
      data init/0/
      save init,pi
*
      if (init.eq.0) then
        init = 1
        pi = avh_oni_pi()
      endif
*
      if (xx.eq.dcmplx(0d0)) then
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'ERROR in avh_oni_logc: xx =',xx
        avh_oni_logc = dcmplx(0d0)
      else
        avh_oni_logc = cdlog( dcmplx(dabs(dreal(xx)),dimag(xx)) ) !CDLOGyes
     &               + dcmplx(0d0,iph*pi)
      endif
      end


      function avh_oni_loga2(xx,iph)
*  ********************************************************************
*  * log(xx)/(1-xx)  with  xx = log|xx| + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex avh_oni_loga2
     &,avh_oni_loga
      double precision xx,omx
     &,thrs,avh_oni_prec
      integer iph
     &,init ,avh_oni_un_get
      data init/0/
      save init,thrs
*
      if (init.eq.0) then
        init = 1
        thrs = 1d1*avh_oni_prec()
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
          if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &      'ERROR in avh_oni_loga2: 1-xx,iph=',omx,iph
          avh_oni_loga2 = dcmplx(0d0)
        else
          avh_oni_loga2 = avh_oni_loga(xx,iph)/dcmplx(omx)
        endif
      else
        if (dabs(omx).lt.thrs) then
          avh_oni_loga2 = dcmplx(-1d0-omx/2d0)
        else
          avh_oni_loga2 = avh_oni_loga(xx,iph)/dcmplx(omx)
        endif
      endif
      end


      function avh_oni_loga(xx,iph)
*  ********************************************************************
*  * log( |xx|*exp(imag*pi*iph) ) = log|xx| + imag*pi*iph
*  ********************************************************************
      implicit none
      double complex avh_oni_loga
      double precision xx
     &,rr,pi ,avh_oni_pi
      integer iph
     &,init ,avh_oni_un_get
      data init/0/
      save init,pi
*
      if (init.eq.0) then
        init = 1
        pi = avh_oni_pi()
      endif
*
      rr = dabs(xx)
      if (rr.eq.0d0.and.avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &  'ERROR in avh_oni_loga: |xx|=',rr
      avh_oni_loga = dcmplx(dlog(rr),iph*pi)
      end


      function avh_oni_pi()
*  ********************************************************************
*  * the number  pi=3.14...
*  ********************************************************************
      implicit none
      double precision avh_oni_pi
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
      avh_oni_pi = pi
      end

 
      function avh_oni_prec()
*  ********************************************************************
*  * the smallest number  prec  satisfying  1+prec = dexp(dlog(1+prec))
*  ********************************************************************
      implicit none
      double precision avh_oni_prec
     &,prec,xx,yy
      integer
     & init ,avh_oni_un_get
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
        if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &    'MESSAGE from avh_oni_prec: precision set to',prec
      endif
      avh_oni_prec = prec
      end
