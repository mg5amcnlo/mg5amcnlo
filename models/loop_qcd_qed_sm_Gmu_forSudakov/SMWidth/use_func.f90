MODULE use_func
! This is the file  use_func.f90  of the package Master_One_Loop,
! which is a modification of the package avh_olo_func.f in OneLoop-1.1
!                    Ref: arXiv:1007.4716 [hep-ph]
! for the evaluation of 1-loop scalar 1-,2-,3- and 4-point functions.
! Including following functions:
!      func_sqrt,func_abc,func_rfun,func_rfun0,func_conv,riemann_sheet,func_prd,
!      func_rat,func_mlt,func_div,func_logc,func_logc2,func_li2c,func_li2c2,
!      func_loga,func_loga2,func_li2a,func_li2a2,func_bern,func_pi,func_eulergamma,
!      func_prec,os_get,func_onshell,func_thrs,func_unit,unit_get,mu_get,mu_set,
!      func_zero,func_eta2,func_eta3,func_plnr,func_r0fun,func_r1fun
!      (func_s3fun,func_tfun,func_t1fun,func_t13fun),func_print,func_printall
!  
!   author: Hua-Sheng Shao
!   Physics school of Peking University
!
!USE msimsl
IMPLICIT NONE
SAVE
INTEGER,PARAMETER::SGL=SELECTED_REAL_KIND(p=6)
INTEGER,PARAMETER::DBL=SELECTED_REAL_KIND(p=13)
LOGICAL,PUBLIC::onshell
REAL(KIND=DBL),PUBLIC::thrs_com
INTEGER,PUBLIC::un,nunit
REAL(KIND=DBL),PUBLIC::mu

CONTAINS
    SUBROUTINE func_printall( unit_in )
!  ********************************************************************
!  ********************************************************************
    IMPLICIT NONE
    INTEGER,INTENT(IN)::unit_in
!	INTEGER::func_print
    LOGICAL::init=.TRUE.
    SAVE init
    IF (init) THEN
       init = .FALSE.
       nunit = func_print()
    ENDIF
    nunit = unit_in
    END SUBROUTINE func_printall

    INTEGER FUNCTION func_print() 
!************************************************************************
!************************************************************************
    IMPLICIT NONE
    LOGICAL::init=.TRUE.
    SAVE init
    IF (init) THEN
       init = .FALSE.
       nunit = 0
    ENDIF
    func_print = nunit
    END FUNCTION func_print

   FUNCTION func_sqrt(xx,sgn)
!  ********************************************************************
! Returns the square-root of xx .
! If  Im(xx)  is equal zero and  Re(xx)  is negative, the result is
! imaginary and has the same sign as  sgn .
!  ********************************************************************
   IMPLICIT NONE
   COMPLEX(KIND=DBL)::func_sqrt
   COMPLEX(KIND=DBL),INTENT(IN)::xx
   REAL(KIND=DBL),INTENT(IN)::sgn
   COMPLEX(KIND=DBL)::zz
   REAL(KIND=DBL)::xim,xre
   xim = DIMAG(xx)
   outer:IF (xim.EQ.0d0) THEN
      xre = DREAL(xx)
      inner:IF (xre.GE.0d0) THEN
        zz = DCMPLX(DSQRT(xre),0d0)
      ELSE inner
        zz = DCMPLX(0d0,DSIGN(DSQRT(-xre),sgn))
      ENDIF inner
   ELSE outer
      zz = CDSQRT(xx)
   ENDIF outer
      func_sqrt = zz
   END FUNCTION func_sqrt

   SUBROUTINE func_abc( x1,x2 ,dd ,aa,bb,cc ,imode )
!  ********************************************************************
! Returns the solutions  x1,x2  to the equation  aa*x^2+bb*x+cc=0
! Also returns  dd = aa*(x1-x2)
! If  imode=/=0  it uses  dd  as input as value of  sqrt(b^2-4*a*c)
!  ********************************************************************
   IMPLICIT NONE
   COMPLEX(KIND=DBL),INTENT(OUT)::x1,x2
   COMPLEX(KIND=DBL),INTENT(INOUT)::dd
   COMPLEX(KIND=DBL),INTENT(IN)::aa,bb,cc
   COMPLEX(KIND=DBL)::qq,hh
   COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0),two=(2d0,0d0),four=(4d0,0d0)
   REAL(KIND=DBL)::r1,r2
   INTEGER,INTENT(IN)::imode
!   INTEGER,EXTERNAL::unit_get
   outer:IF (aa.EQ.zero) THEN
       inner:IF (bb.EQ.zero) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_abc: no solutions, returning 0'
          x1 = zero
          x2 = zero
          dd = zero
       ELSE inner
          x1 = -cc/bb
          x2 = x1
          dd = bb
       END IF inner
    ELSEIF (cc.EQ.zero) THEN
       dd = -bb
       x1 = dd/aa
       x2 = zero
    ELSE
       IF (imode.EQ.0) dd = CDSQRT(bb*bb - four*aa*cc)
       qq = -bb+dd
       hh = -bb-dd
       r1 = CDABS(qq)
       r2 = CDABS(hh)
       IF (r1.GE.r2) THEN
         x1 = qq/(two*aa)
         x2 = (two*cc)/qq
       ELSE
         qq = hh
         x2 = qq/(two*aa)
         x1 = (two*cc)/qq
       ENDIF
    ENDIF outer
    END SUBROUTINE func_abc

	SUBROUTINE func_rfun(rr,dd ,qq)
!  ********************************************************************
! Returns  rr  such that  qq = rr + 1/rr  and  Im(rr)  has the same
! sign as  Im(qq) .
! If  Im(qq)  is zero, then  Im(rr)  is negative or zero.
! If  Im(rr)  is zero, then  |rr| > 1/|rr| .
! Also returns  dd = rr - 1/rr .
!  ********************************************************************
    IMPLICIT NONE
    COMPLEX(KIND=DBL),INTENT(OUT):: rr,dd
	COMPLEX(KIND=DBL),INTENT(IN)::qq
    COMPLEX(KIND=DBL)::r2
    REAL(KIND=DBL)::aa,bb
    INTEGER::ir,ik
    COMPLEX(KIND=DBL),PARAMETER::two=(2d0,0d0),four=(4d0,0d0)
    dd = CDSQRT(qq*qq-four)
    rr = qq+dd
    r2 = qq-dd
    aa = CDABS(rr)
    bb = CDABS(r2)
    swap:IF (bb.GT.aa) THEN                      !|rr| > 1/|rr|
      rr = r2
      dd = -dd
    ENDIF swap
    aa = DIMAG(qq)
    bb = DIMAG(rr)
    IF (aa.EQ.0d0) THEN                          ! Im(qq)  is zero        
       IF (bb.LE.0d0) THEN
          rr = rr/two
       ELSE
          rr = two/rr
          dd = -dd
       ENDIF
    ELSE
       ik = IDNINT(DSIGN(1d0,aa))
       ir = IDNINT(DSIGN(1d0,bb))
       IF (ir.EQ.ik) THEN                        !Im(rr)  has the same sign as  Im(qq)
          rr = rr/two
       ELSE
          rr = two/rr
          dd = -dd
       ENDIF
    ENDIF
    END SUBROUTINE func_rfun

	SUBROUTINE func_rfun0(rr,dd,qq)
!  ********************************************************************
! Like rfun, but now  dd  is input, which may get a minus sign
!  ********************************************************************
    IMPLICIT NONE
    COMPLEX(KIND=DBL),INTENT(OUT)::rr
	COMPLEX(KIND=DBL),INTENT(INOUT)::dd
	COMPLEX(KIND=DBL),INTENT(IN)::qq
    COMPLEX(KIND=DBL)::r2
    REAL(KIND=DBL)::aa,bb
    INTEGER::ir,ik
    COMPLEX(KIND=DBL),PARAMETER::two=(2d0,0d0),four=(4d0,0d0)
    rr = qq+dd
    r2 = qq-dd
    aa = CDABS(rr)
    bb = CDABS(r2)
    IF (bb.GT.aa) THEN
      rr = r2
      dd = -dd
    ENDIF
    aa = DIMAG(qq)
    bb = DIMAG(rr)
    IF (aa.EQ.0d0) THEN
      IF (bb.LE.0d0) THEN
         rr = rr/two
      ELSE
         rr = two/rr
         dd = -dd
      ENDIF
    ELSE
      ik = IDNINT(DSIGN(1d0,aa))
      ir = IDNINT(DSIGN(1d0,bb))
      IF (ir.EQ.ik) THEN
        rr = rr/two
      ELSE
        rr = two/rr
        dd = -dd
      ENDIF
    ENDIF
    END SUBROUTINE func_rfun0

	SUBROUTINE func_conv(zz,iz ,xx,sgn)
!  ********************************************************************
! Determine  zz,iz  such that  xx = zz*exp(iz*imag*pi)  and  Re(zz)
! is positive. If  Im(xx)=0  and  Re(xx)<0  then  iz  becomes the
! sign of  sgn .
!  ********************************************************************
    IMPLICIT NONE
    COMPLEX(KIND=DBL),INTENT(OUT)::zz
	COMPLEX(KIND=DBL),INTENT(IN)::xx
    REAL(KIND=DBL),INTENT(IN)::sgn
    INTEGER,INTENT(OUT)::iz
    REAL(KIND=DBL)::xre,xim
    xre = DREAL(xx)
    IF (xre.GE.0d0) THEN
      zz = xx
      iz = 0
    ELSE
      xim = DIMAG(xx)
      IF (xim.EQ.0d0) THEN
          zz = DCMPLX(-xre,0d0)
          iz = IDNINT(DSIGN(1d0,sgn))
      ELSE
          zz = -xx
          iz = IDNINT(DSIGN(1d0,xim))
      ENDIF
   ENDIF
   END SUBROUTINE func_conv

   FUNCTION riemann_sheet(xx,ix)
!  ********************************************************************
!   Returns the number of the Riemann-sheet (times 2) for the complex
!   number  xx*exp(ix*imag*pi) . The real part of xx is assumed to be
!   positive or zero. Examples:
!   xx=1+imag, ix=-1 -> ii= 0 
!   xx=1+imag, ix= 1 -> ii= 2 
!   xx=1-imag, ix=-1 -> ii=-2 
!   xx=1-imag, ix= 1 -> ii= 0 
!   xx=1     , ix= 1 -> ii= 0  convention that log(-1)=pi on
!   xx=1     , ix=-1 -> ii=-2  the principal Riemann-sheet
!  ********************************************************************
    IMPLICIT NONE
    COMPLEX(KIND=DBL),INTENT(IN)::xx
    INTEGER,INTENT(IN)::ix
	INTEGER::riemann_sheet,ii,jj
    REAL(KIND=DBL)::xim
    ii = ix/2*2
    jj = ix-ii
    xim = DIMAG(xx)
    IF (xim.LE.0d0) THEN
       IF (jj.EQ.-1) ii = ii-2
    ELSE
       IF (jj.EQ. 1) ii = ii+2
    ENDIF
      riemann_sheet = ii
    END FUNCTION riemann_sheet

	SUBROUTINE func_prd(zz,iz ,yy,iy ,xx,ix)
!  ********************************************************************
! Return the product  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase  iz  such that
! the real part of  zz  remains positive 
!  ********************************************************************
    IMPLICIT NONE
    COMPLEX(KIND=DBL),INTENT(OUT)::zz
	COMPLEX(KIND=DBL),INTENT(IN)::yy,xx
    INTEGER,INTENT(OUT)::iz
	INTEGER,INTENT(IN)::iy,ix
    zz = yy*xx
    iz = iy+ix
    IF (DREAL(zz).LT.0d0) THEN
        iz = iz + IDNINT(DSIGN(1d0,DIMAG(xx)))
        zz = -zz
    ENDIF
    END SUBROUTINE func_prd

	SUBROUTINE func_rat(zz,iz ,yy,iy ,xx,ix)
!  ********************************************************************
! Return the ratio  zz  of  yy  and  xx  
! keeping track of (the multiple of pi of) the phase  iz  such that
! the real part of  zz  remains positive 
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(OUT)::zz
	 COMPLEX(KIND=DBL),INTENT(IN)::yy,xx
     INTEGER,INTENT(OUT)::iz
	 INTEGER,INTENT(IN)::iy,ix
     zz = yy/xx
     iz = iy-ix
     IF (DREAL(zz).LT.0d0) THEN
        iz = iz - IDNINT(DSIGN(1d0,DIMAG(xx)))
        zz = -zz
     ENDIF
     END SUBROUTINE func_rat

	 SUBROUTINE func_mlt(yy,iy ,xx,ix)
!  ********************************************************************
! Multiply  yy  with  xx  keeping track of (the multiple of pi of) 
! the phase  iy  such that the real part of  yy  remains positive 
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(INOUT)::yy
	 COMPLEX(KIND=DBL),INTENT(IN)::xx
     INTEGER,INTENT(INOUT)::iy
	 INTEGER,INTENT(IN)::ix
     yy = yy*xx
     iy = iy+ix
     IF (DREAL(yy).LT.0d0) THEN
        iy = iy + IDNINT(DSIGN(1d0,DIMAG(xx)))
        yy = -yy
     ENDIF
     END SUBROUTINE func_mlt

	 SUBROUTINE func_div(yy,iy ,xx,ix)
!  ********************************************************************
! Divide  yy  by  xx  keeping track of (the multiple of pi of) 
! the phase  iy  such that the real part of  yy  remains positive 
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(INOUT)::yy
	 COMPLEX(KIND=DBL),INTENT(IN)::xx
     INTEGER,INTENT(INOUT)::iy
	 INTEGER,INTENT(IN)::ix
     yy = yy/xx
     iy = iy-ix
     IF (DREAL(yy).LT.0d0) THEN
        iy = iy - IDNINT(DSIGN(1d0,DIMAG(xx)))
        yy = -yy
     ENDIF
     END SUBROUTINE func_div

	  FUNCTION func_logc(xx,iph)
!  ********************************************************************
! Returns  log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL),INTENT(IN)::xx
	  COMPLEX(KIND=DBL)::func_logc
      REAL(KIND=DBL)::pi
!	  REAL(KIND=DBL),EXTERNAL::func_pi
      INTEGER,INTENT(IN)::iph
      INTEGER::init=0
!	  INTEGER,EXTERNAL::unit_get
      SAVE init,pi

      IF (init.EQ.0) THEN
        init = 1
        pi = func_pi()
      ENDIF

      IF (xx.EQ.DCMPLX(0d0)) THEN
        IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_logc: xx =',xx
        func_logc = DCMPLX(0d0)
      ELSE
        func_logc = CDLOG( DCMPLX(DABS(DREAL(xx)),DIMAG(xx)) )+ DCMPLX(0d0,iph*pi)
      ENDIF
      END FUNCTION func_logc

	  FUNCTION func_logc2(xx,iph)
!  ********************************************************************
! log(xx)/(1-xx)
! with  log(xx) = log( |Re(xx)| + imag*Im(xx) ) + imag*pi*iph
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(IN)::xx
	 COMPLEX(KIND=DBL)::func_logc2
     COMPLEX(KIND=DBL)::omx
!	 COMPLEX(KIND=DBL),EXTERNAL::func_logc
     REAL(KIND=DBL)::thrs
!	 REAL(KIND=DBL),EXTERNAL::func_prec
     INTEGER,INTENT(IN)::iph
     INTEGER::init=0
!	 INTEGER,EXTERNAL::unit_get
     SAVE init,thrs

     IF (init.EQ.0) THEN
        init = 1
        thrs = 1d1*func_prec()
     ENDIF

     IF (iph.EQ.iph/2*2) THEN
        omx = DCMPLX(1d0-DABS(DREAL(xx)),-DIMAG(xx))
     ELSE
        omx = DCMPLX(1d0+DABS(DREAL(xx)), DIMAG(xx))
     ENDIF

     outer:IF (iph.NE.0) THEN
        inner:IF (omx.EQ.DCMPLX(0d0)) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_logc2: 1-xx,iph=',omx,iph
          func_logc2 = DCMPLX(0d0)
        ELSE
          func_logc2 = func_logc(xx,iph)/omx
        ENDIF inner
     ELSE
        IF (CDABS(omx).LT.thrs) THEN
          func_logc2 = DCMPLX(-1d0)-omx/2
        ELSE
          func_logc2 = func_logc(xx,iph)/omx
        ENDIF
     ENDIF outer
     END FUNCTION func_logc2
	 
	 FUNCTION func_li2c(xx,iph)
!  ********************************************************************
!                /1    ln(1-zz*t)
! func_li2c = - |  dt ---------- 
!               /0         t
! with  zz = 1 - ( |Re(xx)| + imag*Im(xx) )*exp(imag*pi*iph)
! Examples:
!   In order to get the dilog of  1+imag  use  xx=1+imag, iph= 0
!   In order to get the dilog of  1-imag  use  xx=1-imag, iph= 0
!   In order to get the dilog of -1+imag  use  xx=1-imag, iph= 1
!   In order to get the dilog of -1-imag  use  xx=1+imag, iph=-1
!   Add multiples of  2  to  iph  in order to get the result on
!   different Riemann-sheets.
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(IN)::xx
	 COMPLEX(KIND=DBL)::func_li2c
     COMPLEX(KIND=DBL)::yy,lyy,loy,zz,z2,liox
!	 COMPLEX(KIND=DBL),EXTERNAL::func_li2a
     COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0),one=(1d0,0d0)
     REAL(KIND=DBL),DIMENSION(36)::bb
	 REAL(KIND=DBL)::pi,pi2o6 ,rex,imx
!	 REAL(KIND=DBL),EXTERNAL::func_pi,func_bern,func_prec
     INTEGER,INTENT(IN)::iph
     INTEGER::init=0,nn,ii,iyy
     LOGICAL::x_gt_1 , y_lt_h
     SAVE init,nn,bb,pi,pi2o6

     IF (init.EQ.0) THEN
        init = 1
        pi = func_pi()
        pi2o6  = pi**2/6d0
        IF (func_prec().GT.1d-24) THEN
          nn = 18 ! double precision
        ELSE
          nn = 36 ! quadruple precision
        ENDIF
        DO ii=1,nn
          bb(ii) = func_bern(ii)
        ENDDO
     ENDIF
     
     rex = DREAL(xx)
     imx = DIMAG(xx)
     
     IF (imx.EQ.0d0) THEN
        liox = func_li2a(rex,iph)
     ELSE
        rex = DABS(rex)

        IF (iph.EQ.iph/2*2) THEN
          yy = DCMPLX(rex,imx)
          iyy = iph
        ELSE
          yy = DCMPLX(-rex,-imx)
          iyy = iph + IDNINT(SIGN(1d0,imx))
        ENDIF

        x_gt_1 = (CDABS(xx).GT.1d0)
        IF (x_gt_1) THEN
          yy = one/yy
          iyy = -iyy
        ENDIF
        lyy = CDLOG(yy)
        loy = CDLOG(one-yy)

        y_lt_h = (DREAL(yy).LT.0.5d0)
        IF (y_lt_h) THEN
          zz = -loy
        ELSE
          zz = -lyy
        ENDIF

        z2 = zz*zz
        liox = DCMPLX(bb(nn))
        DO ii=nn,4,-2
          liox = DCMPLX(bb(ii-2)) + liox*z2/(ii*(ii+1))
        ENDDO
        liox = DCMPLX(bb(1)) + liox*zz/3
        liox = zz + liox*z2/2

        !li2(x)=-li2(1-x)+pi^2/6-log(x)log(1-x)
		IF (y_lt_h) liox = DCMPLX(pi2o6) - liox - loy*lyy        
		!the connection between different Rienman sheets
        liox = liox - loy*DCMPLX(0d0,pi*iyy)
		!li2(1-x)=-li2(1-1/x)-log(x)**2/2
        IF (x_gt_1) liox = -liox - (lyy + DCMPLX(0d0,iyy*pi))**2/2
		 
      ENDIF
      func_li2c = liox
      END FUNCTION func_li2c 

	 FUNCTION func_li2c2(x1,ip1 ,x2,ip2)
!  ********************************************************************
! avh_olo_li2c2 = ( li2(x1,ip1) - li2(x2,ip2) )/(x1-x2)
!
!                        /1    ln(1-zz*t)
! where  li2(x1,ip1) = - |  dt ----------
!                        /0        t
! with  zz = 1 - ( |Re(x1)| + imag*Im(x1) )*exp(imag*pi*ip1)
! and similarly for li2(x2,ip2)
!  ********************************************************************
     IMPLICIT NONE
     COMPLEX(KIND=DBL),INTENT(IN)::x1,x2 
	 COMPLEX(KIND=DBL)::func_li2c2
     COMPLEX(KIND=DBL)::x1r,x2r,delta ,xx,xr,omx,del,hh,zz 
!	 COMPLEX(KIND=DBL),EXTERNAL::func_li2c,func_logc2
	 COMPLEX(KIND=DBL),DIMENSION(0:20)::ff
     COMPLEX(KIND=DBL),PARAMETER::one=(1d0,0d0)
     REAL(KIND=DBL)::thrs,thrs1,pi
!	 REAL(KIND=DBL),EXTERNAL::func_pi,func_prec
     INTEGER,INTENT(IN)::ip1,ip2
     INTEGER::ix,ih,init=0,nmax,ii
!	 INTEGER,EXTERNAL::unit_get
     SAVE init,nmax,pi,thrs,thrs1
     IF (init.EQ.0) THEN
        init = 1
        pi = func_pi()
        thrs1 = 1d0*func_prec()
        IF (func_prec().GT.1d-24) THEN
          thrs = 0.11d0                                  ! double precision
          nmax = 12
        ELSE
          thrs = 0.008d0                                 ! quadruple precision
          nmax = 12
        ENDIF
     ENDIF

     IF (ip1.EQ.ip1/2*2) THEN
        x1r = DCMPLX( DABS(DREAL(x1)), DIMAG(x1))
     ELSE
        x1r = DCMPLX(-DABS(DREAL(x1)),-DIMAG(x1))
     ENDIF  
	    
     IF (ip2.EQ.ip2/2*2) THEN
        x2r = DCMPLX( DABS(DREAL(x2)), DIMAG(x2))
     ELSE
        x2r = DCMPLX(-DABS(DREAL(x2)),-DIMAG(x2))
     ENDIF

     delta = x1r-x2r
     outer:IF (ip1.NE.ip2) THEN !OLD
        IF (delta.EQ.DCMPLX(0d0)) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_li2c2: ip1,ip2,delta=',ip1,ip2,delta
          func_li2c2=DCMPLX(0d0)
        ELSE
          func_li2c2=( func_li2c(x1,ip1)-func_li2c(x2,ip2) )/delta
        ENDIF
     ELSE outer
        inner:IF (CDABS(delta/x1).GT.thrs) THEN
          func_li2c2 = ( func_li2c(x1,ip1)-func_li2c(x2,ip2) )/delta
        ELSE inner
          xx  = x1
          xr  = x1r
          omx = one-xr
          del = delta
          hh = one-x2r
          IF (CDABS(hh).GT.CDABS(omx)) THEN
            xx = x2
            xr = x2r
            omx = hh
            del = -delta
          ENDIF
          IF (CDABS(omx).LT.thrs1) THEN
            zz = -one-omx/2-del/4
          ELSE
            ih = ip1 - ip1/2*2
            ff(0) = func_logc2(xx,ih)
            hh = -one
            DO ii=1,nmax
              hh = -hh/xr
              ff(ii) = ( hh/ii + ff(ii-1) )/omx
            ENDDO

            zz = ff(nmax)/(nmax+1)
            DO ii=nmax-1,0,-1
              zz = ff(ii)/(ii+1) - zz*del
            ENDDO
          ENDIF

          ih = ip1-ih
          IF (ih.NE.0) THEN
            omx = one-x1r
            CALL func_conv( xx,ix ,(one-x2r)/omx,0d0 )
            zz = zz + DCMPLX(0d0,-ih*pi)*func_logc2( xx,ix )/omx
          ENDIF
          func_li2c2 = zz
        ENDIF inner
      ENDIF outer
      END FUNCTION func_li2c2

	  FUNCTION func_loga(xx,iph)
!  ********************************************************************
! log( |xx|*exp(imag*pi*iph) ) = log|xx| + imag*pi*iph
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL) func_loga
      REAL(KIND=DBL),INTENT(IN)::xx
      REAL(KIND=DBL)::rr,pi
!	  REAL(KIND=DBL),EXTERNAL::func_pi
      INTEGER,INTENT(IN)::iph
      INTEGER::init=0
!	  INTEGER,EXTERNAL::unit_get
      SAVE init,pi

      IF (init.EQ.0) THEN
        init = 1
        pi = func_pi()
      ENDIF

      rr = DABS(xx)
      IF (rr.EQ.0d0.AND.unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_loga: |xx|=',rr
      func_loga = DCMPLX(DLOG(rr),iph*pi)
      END FUNCTION func_loga

	  FUNCTION func_loga2(xx,iph)
!  ********************************************************************
! log(xx)/(1-xx)  with  xx = log|xx| + imag*pi*iph
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL)::func_loga2
!      COMPLEX(KIND=DBL),EXTERNAL::func_loga
      REAL(KIND=DBL),INTENT(IN)::xx
	  REAL(KIND=DBL)::omx,thrs
!	  REAL(KIND=DBL),EXTERNAL::func_prec
      INTEGER,INTENT(IN)::iph
      INTEGER::init=0 
!	  INTEGER,EXTERNAL::unit_get
      SAVE init,thrs

      IF (init.EQ.0) THEN
        init = 1
        thrs = 1d1*func_prec()
      ENDIF

      IF (iph.EQ.iph/2*2) THEN
        omx = 1d0-DABS(xx)
      ELSE
        omx = 1d0+DABS(xx)
      ENDIF

      IF (iph.NE.0) THEN
        IF (omx.EQ.0d0) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_loga2: 1-xx,iph=',omx,iph
          func_loga2 = DCMPLX(0d0)
        ELSE
          func_loga2 = func_loga(xx,iph)/DCMPLX(omx)
        ENDIF
      ELSE
        IF (DABS(omx).LT.thrs) THEN
          func_loga2 = DCMPLX(-1d0-omx/2d0)
        ELSE
          func_loga2 = func_loga(xx,iph)/DCMPLX(omx)
        ENDIF
      ENDIF
      END FUNCTION func_loga2

	  FUNCTION func_li2a(xx,iph)
!  ********************************************************************
!                  /1    ln(1-zz*t)
! func_li2a = -    |  dt ---------- 
!                  /0        t
! with  zz = 1 - |xx|*exp(imag*pi*iph)
! Examples:
!   In order to get the dilog of  1.1  use  xx=1.1, iph=0
!   In order to get the dilog of -1.1  use  xx=1.1, iph=1
!   Add multiples of  2  to  iph  in order to get the result on
!   different Riemann-sheets.
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL)::func_li2a
      COMPLEX(KIND=DBL)::cliox
      REAL(KIND=DBL),INTENT(IN)::xx
      REAL(KIND=DBL),DIMENSION(30)::bb
	  REAL(KIND=DBL)::pi,pi2o6,rr,yy,lyy,loy,zz,z2,liox
!      REAL(KIND=DBL),EXTERNAL::func_pi,func_bern,func_prec
      INTEGER,INTENT(IN)::iph
      INTEGER::init=0,nn,ii,ntwo,ione
      LOGICAL::positive , r_gt_1 , y_lt_h
      SAVE init,nn,bb,pi,pi2o6

      IF (init.EQ.0)THEN
        init = 1
        pi = func_pi()
        pi2o6  = pi**2/6d0
        IF (func_prec().GT.1d-24) THEN
          nn = 16 ! double precision
        ELSE
          nn = 30 ! quadruple precision
        ENDIF
        DO ii=1,nn
          bb(ii) = func_bern(ii)
        ENDDO
      ENDIF
     
      rr = DABS(xx)
      ntwo = iph/2*2
      ione = iph - ntwo
      positive = (ione.EQ.0)
     
      IF     (rr.EQ.0d0) THEN
        cliox = DCMPLX(pi2o6,0d0)
      ELSEIF (rr.EQ.1d0.AND.positive) THEN
        cliox = DCMPLX(0d0,0d0)
      ELSE
        yy  = rr
        lyy = DLOG(rr)
        IF (.NOT.positive) yy = -yy

        r_gt_1 = (rr.GT.1d0)
        IF (r_GT_1) THEN
          yy   = 1d0/yy
          lyy  = -lyy
          ntwo = -ntwo
          ione = -ione
        ENDIF
        loy = DLOG(1d0-yy) ! log(1-yy) is always real

        y_lt_h = (yy.LT.0.5d0)
        IF (y_lt_h) THEN
          zz = -loy ! log(1-yy) is real
        ELSE
          zz = -lyy ! yy>0.5 => log(yy) is real
        ENDIF

        z2 = zz*zz
        liox = bb(nn)
        DO ii=nn,4,-2
          liox = bb(ii-2) + liox*z2/(ii*(ii+1))
        ENDDO
        liox = bb(1) + liox*zz/3
        liox = zz + liox*z2/2

        cliox = DCMPLX(liox)

        IF (y_lt_h) THEN
          cliox = DCMPLX(pi2o6) - cliox - DCMPLX(loy*lyy , loy*pi*ione)
        ENDIF

        cliox = cliox + DCMPLX( 0d0 , -loy*pi*ntwo )

        IF (r_gt_1) cliox = -cliox - DCMPLX( -lyy , iph*pi )**2/2
      ENDIF
      func_li2a = cliox
      END FUNCTION func_li2a

	  FUNCTION func_li2a2(x1,ip1 ,x2,ip2)
!  ********************************************************************
! func_li2a2 = ( li2(x1,ip1) - li2(x2,ip2) )/(x1-x2)
!
!                        /1    ln(1-zz*t)
! where  li2(x1,ip1) = - |  dt ----------
!                        /0        t
! with  zz = 1 - |x1|*exp(imag*pi*ip1)  and similarly for li2(x2,ip2)
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL):: func_li2a2
      COMPLEX(KIND=DBL),DIMENSION(0:20)::ff
	  COMPLEX(KIND=DBL)::zz
!	  COMPLEX(KIND=DBL),EXTERNAL::func_li2a,func_loga2
      REAL(KIND=DBL),INTENT(IN):: x1,x2 
	  REAL(KIND=DBL)::delta,thrs,thrs1,x1r,x2r,xx,omx,del,hh
!	  REAL(KIND=DBL),EXTERNAL::func_prec
      INTEGER,INTENT(IN)::ip1,ip2
      INTEGER::init=0,nmax,ii 
!	  INTEGER,EXTERNAL::unit_get
      SAVE init,nmax,thrs,thrs1

      IF (init.EQ.0) THEN
        init = 1
        thrs1 = 1d0*func_prec()
        IF (func_prec().GT.1d-24) THEN
          thrs = 0.11d0 ! double precision
          nmax = 12
        ELSE
          thrs = 0.008d0 ! quadruple precision
          nmax = 12
        ENDIF
      ENDIF

      IF (ip1.EQ.ip1/2*2) THEN
        x1r = DABS(x1)
      ELSE
        x1r = -DABS(x1)
      ENDIF     
      IF (ip2.EQ.ip2/2*2) THEN
        x2r = DABS(x2)
      ELSE
        x2r = -DABS(x2)
      ENDIF
      delta = x1r-x2r

      outer:IF (ip1.NE.ip2) THEN
        IF (delta.EQ.0d0) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_li2a2: ip1,ip2,delta=',ip1,ip2,delta
          func_li2a2 = DCMPLX(0d0)
        ELSE
          func_li2a2 = ( func_li2a(x1,ip1) -func_li2a(x2,ip2) )/DCMPLX(delta)
        ENDIF
      ELSE outer
        inner:IF (DABS(delta/x1).GT.thrs) THEN
          func_li2a2 = ( func_li2a(x1,ip1)-func_li2a(x2,ip2) )/DCMPLX(delta)
        ELSE inner
          xx  = x1r
          omx = 1d0-xx
          del = delta
          hh = 1d0-x2r
          IF (DABS(hh).GT.DABS(omx)) THEN
            xx = x2r
            omx = hh
            del = -delta
          ENDIF
          IF (DABS(omx).LT.thrs1) THEN
            zz = DCMPLX(-1d0-omx/2-del/4)
          ELSE
            ff(0) = func_loga2(xx,ip1) ! ip1=ip2
            hh = -1d0
            DO ii=1,nmax
              hh = -hh/xx
              ff(ii) = ( DCMPLX(hh/ii) + ff(ii-1) )/DCMPLX(omx)
            ENDDO
            zz = ff(nmax)/(nmax+1)
            DO ii=nmax-1,0,-1
              zz = ff(ii)/(ii+1) - zz*DCMPLX(del)
            ENDDO
          ENDIF
          func_li2a2 = zz
        ENDIF inner
      ENDIF outer
      END FUNCTION func_li2a2

	  FUNCTION func_bern(ii)
!  ********************************************************************
! the first nn Bernoulli numbers
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL)::func_bern
      REAL(KIND=DBL),DIMENSION(40)::bern
      INTEGER,INTENT(IN)::ii
      INTEGER::init=0,jj 
!	  INTEGER,EXTERNAL::unit_get
	  INTEGER,PARAMETER::nn=36
      SAVE init,bern

      IF (init.EQ.0) THEN
        init = 1
        DO jj=3,nn-1,2
          bern(jj) = 0d0
        ENDDO
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
      ENDIF
      if (ii.LE.nn) THEN
        func_bern = bern(ii)
      ELSE
        IF (unit_get().GT.0) WRITE(unit_get(),*)'ERROR in func_bern: bernoulli(',ii &
		,') not yet implemented'
      ENDIF
      END FUNCTION func_bern

	  FUNCTION func_pi()
!  ********************************************************************
! the number  pi=3.14...
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL)::func_pi
      REAL(KIND=DBL)::pi
      INTEGER::init=0
      SAVE init,pi

      IF (init.EQ.0) THEN
        init = 1
        pi = 4*DATAN(1d0)
      ENDIF
      func_pi = pi
      END FUNCTION func_pi

	  FUNCTION func_eulergamma()
!  ********************************************************************
! the number  eulergamma=0.577216...
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL)::func_eulergamma
      REAL(KIND=DBL)::eulerg
      INTEGER::init=0
      SAVE init,eulerg

      IF (init.EQ.0) THEN
        init = 1
        eulerg = 0.57721566490153286061d0
 !       eulerg = -DPSI(1d0)
      ENDIF
      func_eulergamma = eulerg
      END FUNCTION func_eulergamma

	  FUNCTION func_prec()
!  ********************************************************************
! the smallest number  prec  satisfying  1+prec = dexp(dlog(1+prec))
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL)::func_prec,prec,xx,yy
      INTEGER::init=0 
!	  INTEGER,EXTERNAL::unit_get
      SAVE init,prec

      IF (init.EQ.0) THEN
        init = 1
        xx = 1d0
        yy = xx
        DO WHILE (xx.EQ.yy)
          prec = xx
          xx = xx/2
          yy = -1d0+DEXP(DLOG(1d0+xx))
        ENDDO
        IF (unit_get().GT.0) WRITE(unit_get(),*)'MESSAGE from func_prec: precision set to',prec
      ENDIF
      func_prec = prec
      END FUNCTION func_prec

	  SUBROUTINE func_onshell(thrs)
!  ********************************************************************
! Set threshold to consider internal masses identical zero and
! external squared momenta identical zero or equal to internal
! masses, if this leads to an IR divergent case. For example,
! if  |p1-m1|<thrs , and  p1=m1  consitutes an IR-divergent
! case, then the loop integral is considered to be IR-divergent.
! Here  thrs  is the input for this routine.
! If this routine is not called,  thrs  will essentially be considerd
! identically zero, but warnings will be given when an IR-divergent
! case is approached. If this routine is called with thrs=0d0 , then
! these warnings will not appear anymore.
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL),INTENT(IN)::thrs
!      REAL(KIND=DBL),EXTERNAL::func_thrs
!      LOGICAL,EXTERNAL::os_get
	  LOGICAL::init=.TRUE.
!      INTEGER,EXTERNAL::unit_get    
      SAVE init

      IF (init) THEN
        init = .FALSE.
!       call func_hello
        onshell =os_get()
        thrs_com = func_thrs(1d0)
      ENDIF
      onshell = .TRUE.
      thrs_com = thrs
      IF (unit_get().GT.0) WRITE(unit_get(),*) &
       'MESSAGE from func_onshell: threshold set to:',thrs_com
     END SUBROUTINE func_onshell

	  FUNCTION os_get()
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      LOGICAL::os_get ,init=.TRUE.
      SAVE init

      IF (init) THEN
        init = .FALSE.
        onshell = .FALSE.
      ENDIF
      os_get = onshell
      END FUNCTION os_get

	  FUNCTION func_thrs(xx)
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL),INTENT(IN)::xx
	  REAL(KIND=DBL)::func_thrs
!	  REAL(KIND=DBL),EXTERNAL::func_prec
      LOGICAL::init=.TRUE.
!	  LOGICAL,EXTERNAL::os_get
      SAVE init
      IF (init) THEN
        init = .FALSE.
        IF (func_prec().GT.1d-24) THEN
!double precision
          thrs_com = 1d2*func_prec()
        ELSE
!quadruple precision
          thrs_com = 1d4*func_prec()
        ENDIF
      ENDIF
      IF (os_get()) THEN
        func_thrs = thrs_com
      ELSE
        func_thrs = thrs_com*xx
      ENDIF
      END FUNCTION func_thrs

	  FUNCTION unit_get()
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      INTEGER::unit_get,init=0
      SAVE init

      IF (init.EQ.0) THEN
        init = 1
        un = 6
      ENDIF
      unit_get = un
      END FUNCTION unit_get

	  SUBROUTINE func_unit(un_in)
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      INTEGER,INTENT(IN)::un_in
!      INTEGER,EXTERNAL::unit_get
	  INTEGER::init=0
      SAVE init

      IF (init.EQ.0) THEN
        init = 1
        un = unit_get()
      ENDIF
      un = un_in
      END SUBROUTINE func_unit

	  FUNCTION mu_get()
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL)::mu_get
      INTEGER::init=0
      SAVE init

      IF (init.EQ.0) THEN
        init = 1
        mu = 1d0
      ENDIF
      mu_get = mu
      END FUNCTION mu_get

	  SUBROUTINE mu_set(mu_in)
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      REAL(KIND=DBL),INTENT(IN)::mu_in
!      REAL(KIND=DBL),EXTERNAL::mu_get
      INTEGER::init=0
!	  INTEGER,EXTERNAL::unit_get
      SAVE init

      IF (init.EQ.0) THEN
        init = 1
!       CALL func_hello
        mu = mu_get()
      ENDIF
      mu = mu_in
      IF (unit_get().GT.0) WRITE(unit_get(),*) &
       'MESSAGE from mu_set: scale (mu, not mu^2) set to:',mu
      END SUBROUTINE mu_set

	  SUBROUTINE func_zero(rslt)
!  ********************************************************************
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL),DIMENSION(0:2),INTENT(OUT)::rslt
      rslt(2) = DCMPLX(0d0)
      rslt(1) = DCMPLX(0d0)
      rslt(0) = DCMPLX(0d0)
      END SUBROUTINE func_zero

	  INTEGER FUNCTION func_eta3( aa,sa ,bb,sb ,cc,sc )
!  ********************************************************************
!     theta(-Im(a))*theta(-Im(b))*theta( Im(c))
!   - theta( Im(a))*theta( Im(b))*theta(-Im(c))
! where a,b,c are interpreted as a+i|eps|sa, b+i|eps|sb, c+i|eps|sc
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL),INTENT(IN)::aa,bb,cc
      REAL(KIND=DBL),INTENT(IN)::sa,sb,sc
      REAL(KIND=DBL)::ima,imb,imc
      ima = DIMAG(aa)
      imb = DIMAG(bb)
      imc = DIMAG(cc)
      IF (ima.EQ.0d0) ima = sa
      IF (imb.EQ.0d0) imb = sb
      IF (imc.EQ.0d0) imc = sc
      ima = DSIGN(1d0,ima)
      imb = DSIGN(1d0,imb)
      imc = DSIGN(1d0,imc)
      IF (ima.EQ.imb.AND.ima.NE.imc) THEN
        func_eta3 = IDNINT(imc)
      ELSE
        func_eta3 = 0
      ENDIF
      END FUNCTION func_eta3
 
      INTEGER FUNCTION func_eta2( aa,sa ,bb,sb )
!  ********************************************************************
! The same as  func_eta3, but with  c=a*b, so that
!   2*pi*imag*eta(a,b) = log(a*b) - log(a) - log(b)
!  ********************************************************************
      IMPLICIT NONE
      COMPLEX(KIND=DBL),INTENT(IN)::aa,bb
      REAL(KIND=DBL),INTENT(IN)::sa,sb
      REAL(KIND=DBL)::rea,reb,ima,imb,imab
      rea = DREAL(aa)
      reb = DREAL(bb)
      ima = DIMAG(aa)
      imb = DIMAG(bb)
      imab = rea*imb + reb*ima
      IF (ima.EQ.0d0) ima = sa
      IF (imb.EQ.0d0) imb = sb
      IF (imab.EQ.0d0) imab = DSIGN(rea,sb) + DSIGN(reb,sa)
      ima  = DSIGN(1d0,ima)
      imb  = DSIGN(1d0,imb)
      imab = DSIGN(1d0,imab)
      IF (ima.EQ.imb.AND.ima.NE.imab) THEN
        func_eta2 = IDNINT(imab)
      ELSE
        func_eta2 = 0
      ENDIF
      END FUNCTION func_eta2

	  FUNCTION func_plnr( y1,y2 ,p1,p2 ,aa,bb,cc )
!  ********************************************************************
!                   /   a    \          /   a    \
!            p1*log |--------| - p2*log |--------| 
!                   \ b*y1+c /          \ b*y2+c /
! 2*pi*imag* -------------------------------------
!                           y1 - y2
! 
! p1,p2 are logical, to be interpreted as 0,1 in the formula above 
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_plnr
      COMPLEX(KIND=DBL),INTENT(IN)::y1,y2 ,aa,bb,cc
      LOGICAL,INTENT(IN)::p1,p2
      COMPLEX(KIND=DBL)::x1,q1,x2,q2,xx,rslt,twopii
!      COMPLEX(KIND=DBL),EXTERNAL::func_logc,func_logc2
!      REAL(KIND=DBL),EXTERNAL::func_pi
      INTEGER::i1,i2,ii
!      INTEGER::unit_get

      IF (p1) THEN
        x1 = bb*y1 + cc
        xx = aa/x1
        IF (DIMAG(xx).EQ.0d0) THEN
          IF (unit_get().GT.0) WRITE(unit_get(),*) &
           'ERROR in func_plnr: aa/x1 has zero imaginary part'
        ENDIF
        CALL func_conv( q1,i1 ,xx,0d0 )
      ENDIF
      IF (p2) THEN
        x2 = bb*y2 + cc
        xx = aa/x2
        IF (DIMAG(xx).EQ.0d0) THEN
           IF (unit_get().GT.0) WRITE(unit_get(),*) &
           'ERROR in func_plnr: aa/x2 has zero imaginary part'
        ENDIF
        CALL func_conv( q2,i2 ,xx,0d0 )
      ENDIF
      IF (p1) THEN
        IF (p2) THEN
          CALL func_rat( xx,ii ,q2,i2 ,q1,i1 )
          twopii = DCMPLX(0d0,2*func_pi())
          rslt = func_logc2( xx,ii ) * twopii*bb/x2 !RIGHT
        ELSE
          twopii = DCMPLX(0d0,2*func_pi())
          rslt = func_logc( q1,i1 ) * twopii/(y1-y2)
        ENDIF
      ELSEIF (p2) THEN
        twopii = DCMPLX(0d0,2*func_pi())
        rslt = func_logc( q2,i2 ) * twopii/(y2-y1) ! minus sign
      ELSE
        rslt = DCMPLX(0d0)
      ENDIF
      func_plnr = rslt
      END FUNCTION func_plnr

	  FUNCTION func_r0fun( y1,y2 )
!  ********************************************************************
!      / 1-y1 \       / 1-y2 \
!  log |------| - log |------| 
!      \  -y1 /       \  -y2 /
!  ---------------------------
!            y1 - y2
!
! y1,y2 should have non-zero imaginary parts
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_r0fun
      COMPLEX(KIND=DBL),INTENT(IN)::y1,y2
      COMPLEX(KIND=DBL)::q1,q2,qq,oy1,oy2,log1,log2
!      COMPLEX(KIND=DBL),EXTERNAL::func_logc2,func_logc
      COMPLEX(KIND=DBL),PARAMETER::one=(1d0,0d0)
      INTEGER::i1,i2,ii
      CALL func_conv( q1,i1 ,-y1,0d0 )
      CALL func_conv( q2,i2 ,-y2,0d0 )
      CALL func_rat( qq,ii ,q2,i2 ,q1,i1 )
      log1 = func_logc2( qq,ii )/y1              ! log((-y2)/(-y1))/(y1-y2)
      oy1 = one-y1
      oy2 = one-y2
      CALL func_conv( q1,i1 ,oy1,0d0 )
      CALL func_conv( q2,i2 ,oy2,0d0 )
      CALL func_rat( qq,ii ,q2,i2 ,q1,i1 )
      log2 = func_logc2( qq,ii )/oy1              ! -log((1-y2)/(1-y1))/(y1-y2)
      func_r0fun = log1 + log2
      END FUNCTION func_r0fun

	  FUNCTION func_r1fun( zz,y1,y2,fy1y2 )
!  ********************************************************************
! calculates  ( R1(y1,z) - R1(y2,z) )/( y1 - y2 )
! where
!                          /     / 1-y \       / 1-z \ \
!      R1(y,z) = ln(y-z) * | log |-----| - log |-----| |
!                          \     \ -y  /       \ -z  / / 
!
!                      /    y-z \       /    y-z \
!                - Li2 |1 - ----| + Li2 |1 - ----|
!                      \    -z  /       \    1-z /
!
!                                     / 1-y1 \       / 1-y2 \
!                                 log |------| - log |------| 
! input fy1y2 should be equal to      \  -y1 /       \  -y2 /
!                                 ---------------------------
!                                           y1 - y2
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_r1fun
      COMPLEX(KIND=DBL),INTENT(IN)::y1,y2,zz,fy1y2
      COMPLEX(KIND=DBL)::q1z,q2z,qq,rslt,q1,q2,q3,q4,oz,trm
!      COMPLEX(KIND=DBL),EXTERNAL::func_logc,func_r0fun,func_logc2 &
!	                              ,func_li2c2,func_li2c
      COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0) ,one=(1d0,0d0)
      REAL(KIND=DBL)::h12,hz1,hz2,hzz,hoz,pi
!      REAL(KIND=DBL),EXTERNAL::func_pi
      INTEGER::i1z,i2z,iq,i1,i2,i3,i4
      LOGICAL::zzsmall,ozsmall

      pi = func_pi()

      oz = one-zz
      h12 = CDABS(y1-y2)
      hz1 = CDABS(y1-zz)
      hz2 = CDABS(y2-zz)
      hzz = CDABS(zz)
      hoz = CDABS(oz)
      CALL func_conv( q1z,i1z ,y1-zz,0d0 )
      CALL func_conv( q2z,i2z ,y2-zz,0d0 )

      zzsmall = .FALSE.
      ozsmall = .FALSE.
      IF(hzz.LT.hz1.AND.hzz.LT.hz2.AND.hzz.LT.hoz) THEN    ! |z| < |y1-z|,|y2-z|,|1-z|
        zzsmall = .TRUE.
        CALL func_rat( q1,i1 ,q1z,i1z ,q2z,i2z )            ! q1 = (y1-z)/(y2-z)
        CALL func_prd( q2,i2 ,q1z,i1z ,q2z,i2z )            ! q2 = (y1-z)*(y2-z)
        CALL func_conv( q3,i3 ,(y2-one)/y2,0d0 )            ! q3 = (y2-1)/y2
        CALL func_conv( q4,i4 ,oz,0d0 )                     ! q4 = 1-z
        rslt = fy1y2*func_logc( q1z,i1z ) &
               - (func_logc( q2,i2 )/2    &
               +func_logc( q3,i3 )        &
               -func_logc( q4,i4 ) )*func_logc2( q1,i1 )/(y2-zz)
      ELSEIF (hoz.LT.hz1.AND.hoz.LT.hz2) THEN                ! |1-z| < |y1-z|,|y2-z|,|z|
        ozsmall = .TRUE.
        CALL func_rat( q1,i1 ,q1z,i1z ,q2z,i2z )             ! q1 = (y1-z)/(y2-z)
        CALL func_prd( q2,i2 ,q1z,i1z ,q2z,i2z )             ! q2 = (y1-z)*(y2-z)
        CALL func_conv( q3,i3 ,(y2-one)/y2,0d0 )             ! q3 = (y2-1)/y2
        CALL func_conv( q4,i4 ,-zz,0d0 )                     ! q4 = -z
        rslt = fy1y2*func_logc( q1z,i1z )   &
               - (-func_logc( q2,i2 )/2     &
               +func_logc( q3,i3 )          &
               +func_logc( q4,i4 ) )*func_logc2( q1,i1 )/(y2-zz)
      ELSEIF (h12.LE.hz2.AND.hz2.LE.hz1) THEN                 ! |y1-y2| < |y2-z| < |y1-z|
        CALL func_rat( qq,iq, q1z,i1z ,q2z,i2z )              ! qq = (y1-z)/(y2-z)
        rslt = fy1y2*func_logc( q1z,i1z ) &
               - func_r0fun( y2,zz )*func_logc2( qq,iq )        
      ELSEIF (h12.LE.hz1.AND.hz1.LE.hz2) THEN                 ! |y1-y2| < |y1-z| < |y2-z|
        CALL func_rat( qq,iq, q2z,i2z ,q1z,i1z )              ! qq = (y2-z)/(y1-z)
        rslt = fy1y2*func_logc( q2z,i2z ) &
               - func_r0fun( y1,zz )*func_logc2( qq,iq )        
      ELSE                                                    ! |y2-z|,|y1-z| < |y1-y2|
        rslt = zero
        IF (hz1.NE.0d0) rslt = rslt + (y1-zz)*func_logc( q1z,i1z ) & !|y1-z|>0
                               *func_r0fun( y1,zz )
        IF (hz2.NE.0d0) rslt = rslt - (y2-zz)*func_logc( q2z,i2z ) &
                               *func_r0fun( y2,zz )
        rslt = rslt/(y1-y2)
      ENDIF

      IF (zzsmall) THEN                                        ! |z| < |y1-z|,|y2-z|
        CALL func_conv( qq ,iq ,-zz,0d0 )
        CALL func_rat( q1,i1 ,qq,iq ,q1z,i1z )                 ! (-z)/(y1-z)
        CALL func_rat( q2,i2 ,qq,iq ,q2z,i2z )                 ! (-z)/(y2-z)
        trm = func_li2c( q1,i1 ) - func_li2c( q2,i2 )
        rslt = rslt + trm/(y1-y2)
      ELSE
        CALL func_conv( qq ,iq ,-zz,0d0 )
        CALL func_rat( q1,i1 ,q1z,i1z ,qq,iq )                  ! (y1-z)/(-z)
        CALL func_rat( q2,i2 ,q2z,i2z ,qq,iq )                  ! (y2-z)/(-z)
        rslt = rslt + func_li2c2( q1,i1 ,q2,i2 )/zz
      ENDIF

      IF (ozsmall) THEN                                         ! |1-z| < |y1-z|,|y2-z|
        CALL func_conv( qq ,iq ,oz,0d0 )
        CALL func_rat( q1,i1 ,qq,iq ,q1z,i1z )                  ! (1-z)/(y1-z)
        CALL func_rat( q2,i2 ,qq,iq ,q2z,i2z )                  ! (1-z)/(y2-z)
        trm = func_li2c( q1,i1 ) - func_li2c( q2,i2 )
        rslt = rslt - trm/(y1-y2)
      ELSE
        CALL func_conv( qq,iq ,oz,0d0 )
        CALL func_rat( q1,i1 ,q1z,i1z ,qq,iq )                   ! (y1-z)/(1-z)
        CALL func_rat( q2,i2 ,q2z,i2z ,qq,iq )                   ! (y2-z)/(1-z)
        rslt = rslt + func_li2c2( q1,i1 ,q2,i2 )/oz
      ENDIF
      func_r1fun = rslt
      END FUNCTION func_r1fun

	  FUNCTION func_s3fun( y1i,y2i ,dd,ee ,aa,bb,cin )
!  ********************************************************************
! Calculate
!            ( S3(y1i) - S3(y2i) )/( y1i - y2i )
! where
!               /1    ee * ln( aa*x^2 + bb*x + cc )
!       S3(y) = |  dx -----------------------------
!               /0           ee*x - y - dd
!
! y1i,y2i should have a non-zero imaginary part
! similary to (B.1) in Nucl.Phys.B 153 (1979) 365-401
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_s3fun
      COMPLEX(KIND=DBL),INTENT(IN)::y1i,y2i ,dd,ee ,aa,bb,cin
      COMPLEX(KIND=DBL)::y1,y2,fy1y2,qq,z1,z2,rslt,cc
!      COMPLEX(KIND=DBL),EXTERNAL::func_logc,func_r1fun,func_r0fun
      COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0) ,one=(1d0,0d0)
      INTEGER::iq,ieta
!	  INTEGER,EXTERNAL::unit_get,eta3
      REAL(KIND=DBL)::rea,reb,rez1,rez2,imz1,imz2,small,thrs,simc,twopi
!      REAL(KIND=DBL),EXTERNAL::func_prec,func_pi
      LOGICAL::init=.TRUE.
      SAVE init,small,thrs,twopi
      IF (init) THEN
        init = .FALSE.
        small = func_prec()**2
        IF (func_prec().GT.1d-24) THEN
          thrs = 1d3*func_prec()
        ELSE
          thrs = 1d6*func_prec()
        ENDIF
        twopi = 2*func_pi()
      ENDIF
      IF (ee.EQ.zero) THEN
        func_s3fun = zero
        RETURN
      ENDIF

      cc = cin
      rea = CDABS(aa)
      reb = CDABS(bb)
      simc = CDABS(cc)
      IF (simc.LT.thrs*MIN(rea,reb)) cc = zero

      simc = DIMAG(cc)
      IF (simc.EQ.0d0) THEN
        simc = DIMAG(bb)
        IF (simc.EQ.0d0) simc = -1d0
      ENDIF
      simc = DSIGN(1d0,simc)

      y1 = (dd+y1i)/ee
      y2 = (dd+y2i)/ee
      IF (DIMAG(y1).EQ.0d0) THEN
        IF (unit_get().GT.0) WRITE(unit_get(),*) &
         'ERROR in func_s3fun: y1 has zero imaginary part'
      ENDIF
      IF (DIMAG(y2).EQ.0d0) THEN
        IF (unit_get().GT.0) WRITE(unit_get(),*) &
         'ERROR in func_s3fun: y2 has zero imaginary part'
      ENDIF
      fy1y2 = func_r0fun( y1,y2 )

      IF(aa.NE.zero) THEN
        CALL func_abc( z1,z2 ,qq ,aa,bb,cc ,0 )
        rea  = DSIGN(1d0,DREAL(aa))
        rez1 = DREAL(z1)
        rez2 = DREAL(z2) 
        imz1 = DIMAG(z1)                                     ! sign(Im(a*z1*z2)) = simc
        imz2 = DIMAG(z2)
        IF(imz1.EQ.0d0) imz1 = simc*rea*DSIGN(1d0,rez2)*DABS(small*rez1)
        if(imz2.EQ.0d0) imz2 = simc*rea*DSIGN(1d0,rez1)*DABS(small*rez2)
        z1 = DCMPLX( rez1,imz1 )
        z2 = DCMPLX( rez2,imz2 )
        ieta = func_eta3( -z1,-imz1 ,-z2,-imz2 ,zero,simc*rea )
        CALL func_conv( qq,iq ,aa,simc )
        rslt = fy1y2 * ( func_logc( qq,iq )  &
               + DCMPLX(0d0,twopi*DBLE(ieta)) ) &
               + func_r1fun( z1,y1,y2,fy1y2 ) &
               + func_r1fun( z2,y1,y2,fy1y2 )
      ELSEIF (bb.NE.zero) THEN
        z1 = -cc/bb                                   ! - i|eps|Re(b)
        reb  = DREAL(bb)
        rez1 = DREAL(z1)
        imz1 = DIMAG(z1)
        IF (DABS(imz1).EQ.0d0) THEN
          imz1 = -simc*reb*DABS(small*rez1/reb)
          z1 = DCMPLX( rez1,imz1 )
        ENDIF
        CALL func_conv( qq,iq ,bb,simc )
        ieta = func_eta3( bb,simc ,-z1,-imz1 ,cc,simc )
        rslt = fy1y2 * ( func_logc( qq,iq ) &
                + DCMPLX(0d0,twopi*DBLE(ieta)) ) &
                + func_r1fun( z1,y1,y2,fy1y2 )
      ELSEIF (cc.NE.zero) THEN
        CALL func_conv( qq,iq ,cc,simc )
        rslt = func_logc( qq,iq )*fy1y2

      ELSE                                               !if (aa=bb=cc=0)

        IF (unit_get().GT.0) WRITE(unit_get(),*) &
         'ERROR in func_s3fun: cc equal zero, returning 0'
        rslt = zero
      ENDIF
      func_s3fun = rslt/ee
      END FUNCTION func_s3fun

	  FUNCTION func_t13fun( aa,cc,gg,hh ,dd,ee,ff,jj )
!  ********************************************************************
! /1   /x                             y
! | dx |  dy -----------------------------------------------------
! /0   /0    (gy^2 + hxy + dx + jy + f)*(ax^2 + cxy + dx + ey + f)
!
! jj should have negative imaginary part
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_t13fun
      COMPLEX(KIND=DBL),INTENT(IN)::aa,cc,dd,ee,ff,gg,hh,jj
      COMPLEX(KIND=DBL)::kk,ll,nn,y1,y2,sdnt,rslt,ieps
!      COMPLEX(KIND=DBL),EXTERNAL::func_s3fun
      COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0) ,one=(1d0,0d0)
      REAL(KIND=DBL)::small
!      REAL(KIND=DBL),EXTERNAL::func_prec
!
!      write(6,*) 'MESSAGE from func_t13fun: you are calling me' !DEBUG
!
      small = func_prec()**2
      ieps = DCMPLX(0d0,small*DABS(DREAL(ff)))
      kk = hh*aa - cc*gg
      ll = aa*dd + hh*ee - dd*gg - cc*jj
      nn = dd*(ee - jj) + (hh - cc)*(ff-ieps)
      CALL func_abc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
      rslt =      - func_s3fun( y1,y2 ,zero,one ,aa   ,ee+cc,dd+ff )
      rslt = rslt + func_s3fun( y1,y2 ,zero,one ,gg   ,jj+hh,dd+ff )
      rslt = rslt - func_s3fun( y1,y2 ,zero,one ,gg+hh,dd+jj,ff )
      rslt = rslt + func_s3fun( y1,y2 ,zero,one ,aa+cc,ee+dd,ff )
      func_t13fun = rslt/kk
      END FUNCTION func_t13fun


      FUNCTION func_t1fun( aa,cc,gg,hh ,dd,ee,ff,jj )
!  ********************************************************************
! /1   /x                         1
! | dx |  dy ----------------------------------------------
! /0   /0    (g*x + h*y + j)*(a*x^2 + c*xy + d*x + e*y + f)
!
! jj should have negative imaginary part
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_t1fun
      COMPLEX(KIND=DBL),INTENT(IN)::aa,cc,gg,hh ,dd,ee,ff,jj
      COMPLEX(KIND=DBL)::kk,ll,nn,y1,y2,sdnt,rslt,ieps
!      COMPLEX(KIND=DBL),EXTERNAL::func_s3fun
      COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0) ,one=(1d0,0d0)
      REAL(KIND=DBL)::small
!      REAL(KIND=DBL),EXTERNAL::func_prec
!
!      write(6,*) 'MESSAGE from func_t1fun: you are calling me' !DEBUG
!
      small = func_prec()**2
      ieps = DCMPLX(0d0,small*DABS(DREAL(ff)))
      kk = hh*aa - cc*gg
      ll = hh*dd - cc*jj - ee*gg
      nn = hh*(ff-ieps) - ee*jj
      CALL func_abc( y1,y2 ,sdnt ,kk,ll,nn ,0 )
      rslt =      - func_s3fun( y1,y2 ,zero,one ,aa+cc,dd+ee,ff )
      rslt = rslt + func_s3fun( y1,y2 ,zero,one ,zero ,gg+hh,jj )
      rslt = rslt - func_s3fun( y1,y2 ,zero,one ,zero ,gg   ,jj )
      rslt = rslt + func_s3fun( y1,y2 ,zero,one ,aa   ,dd   ,ff )
      func_t1fun = rslt/kk
      END FUNCTION func_t1fun


      FUNCTION func_tfun( aa,bb,cc ,gin,hin,dd,ee,ff ,jin )
!  ********************************************************************
! /1   /x                             1
! | dx |  dy ------------------------------------------------------
! /0   /0    (g*x + h*y + j)*(a*x^2 + b*y^2 + c*xy + d*x + e*y + f)
!  ********************************************************************
      IMPLICIT NONE
	  COMPLEX(KIND=DBL)::func_tfun
      COMPLEX(KIND=DBL),INTENT(IN)::aa,bb,cc ,gin,hin ,dd,ee,ff,jin
      COMPLEX(KIND=DBL)::gg,hh,jj,beta,ll,nn,kk,y1,y2,sdnt,rslt,ieps
	  COMPLEX(KIND=DBL),DIMENSION(2)::zz,tmpa,tmpb,tmpc,kiz
      COMPLEX(KIND=DBL),DIMENSION(2,2)::yy
!      COMPLEX(KIND=DBL),EXTERNAL::func_t1fun,func_s3fun,func_plnr
      COMPLEX(KIND=DBL),PARAMETER::zero=(0d0,0d0) ,one=(1d0,0d0)
      REAL(KIND=DBL)::sj,ab1,ab2,ac1,ac2,abab,acac,abac,det,ap1,ap2 &
	                 ,apab,apac,x1(2,2),x2(2,2),xmin,small
!      REAL(KIND=DBL),EXTERNAL::func_prec
      INTEGER::iz,iy,izmin
      LOGICAL,DIMENSION(2,2)::pp
	  LOGICAL::p1,p2
!
!      write(6,*) 'MESSAGE from func_tfun: you are calling me' !DEBUG
!
      sj = DIMAG(jin)
      IF (sj.EQ.0d0) THEN
        sj = -1d0
      ELSE
        sj = DSIGN(1d0,DIMAG(jin))
      ENDIF
      gg = -sj*gin
      hh = -sj*hin
      jj = -sj*jin
      IF(bb.EQ.zero) THEN
        func_tfun = -sj*func_t1fun( aa,cc,gg,hh ,dd,ee,ff,jj )
        RETURN
      ELSEIF (aa.EQ.zero) THEN
        func_tfun = -sj*func_t1fun( bb+cc,-cc,-gg-hh,gg &
                    ,-dd-ee-2*(bb+cc),dd+cc,dd+ee+bb+cc+ff,gg+hh+jj )
        RETURN
      ENDIF
      small = func_prec()**2
      ieps = DCMPLX(0d0,small*DABS(DREAL(ff)))
      CALL func_abc( zz(1),zz(2) ,sdnt ,bb,cc,aa ,0 )
      IF (CDABS(zz(1)).GT.CDABS(zz(2))) THEN
        beta = zz(1)
        zz(1) = zz(2)
        zz(2) = beta
      ENDIF
      DO iz=1,2
        beta = zz(iz)
        tmpa(iz) = gg + beta*hh
        tmpb(iz) = cc + 2*beta*bb
        tmpc(iz) = dd + beta*ee
        kiz(iz) =        bb*tmpa(iz)               - hh*tmpb(iz)
        ll      =        ee*tmpa(iz) - hh*tmpc(iz) - jj*tmpb(iz)
        nn      = (ff-ieps)*tmpa(iz) - jj*tmpc(iz)
        CALL func_abc( yy(iz,1),yy(iz,2) ,sdnt ,kiz(iz),ll,nn ,0 )
        IF (DABS(DIMAG(beta)).NE.0d0) THEN
          ab1 = DREAL(-beta)
          ab2 = DIMAG(-beta)
          ac1 = ab1+1d0                                      !dreal(one-beta)
          ac2 = ab2                                          !dimag(one-beta)
          abab = ab1*ab1 + ab2*ab2
          acac = ac1*ac1 + ac2*ac2
          abac = ab1*ac1 + ab2*ac2
          det = abab*acac - abac*abac
          DO iy=1,2
            ap1 = DREAL(yy(iz,iy))
            ap2 = DIMAG(yy(iz,iy))
            apab = ap1*ab1 + ap2*ab2
            apac = ap1*ac1 + ap2*ac2
            x1(iz,iy) = ( acac*apab - abac*apac )/det
            x2(iz,iy) = (-abac*apab + abab*apac )/det
          ENDDO
        ELSE
          DO iy=1,2
            x1(iz,iy) = -1d0
            x2(iz,iy) = -1d0
          ENDDO
        ENDIF
      ENDDO
      xmin = 1d0
      izmin = 2
      DO iz=1,2
         DO iy=1,2
            IF ( x1(iz,iy).GE.0d0.AND.x2(iz,iy).GE.0d0 &
               .AND.x1(iz,iy)+x2(iz,iy).le.1d0 ) THEN
                pp(iz,iy) = .TRUE.
                IF (x1(iz,iy).LT.xmin) THEN
                   xmin = x1(iz,iy)
                   izmin = iz
                ENDIF
                IF (x2(iz,iy).LT.xmin) THEN
                   xmin = x2(iz,iy)
                   izmin = iz
                ENDIF
            ELSE
                pp(iz,iy) = .FALSE.
            ENDIF
         ENDDO
      ENDDO
      iz = izmin+1
      IF (iz.EQ.3) iz = 1
      beta = zz(iz)
!      write(6,*) '-----> beta  ',beta !DEBUG
      kk = kiz(iz)
      y1 = yy(iz,1)
      y2 = yy(iz,2)
      p1 = pp(iz,1)
      p2 = pp(iz,2)
      rslt = &
      + func_s3fun( y1,y2 ,beta ,one      ,zero    ,hh   ,gg+jj ) &
      - func_s3fun( y1,y2 ,zero ,one-beta ,zero    ,gg+hh,   jj ) &
      + func_s3fun( y1,y2 ,zero ,   -beta ,zero    ,gg   ,   jj ) &
      - func_s3fun( y1,y2 ,beta ,one      ,bb      ,cc+ee,aa+dd+ff ) &
      + func_s3fun( y1,y2 ,zero ,one-beta ,aa+bb+cc,dd+ee,ff       ) &
      - func_s3fun( y1,y2 ,zero ,   -beta ,aa      ,dd   ,ff       ) 
      sdnt = func_plnr( y1,y2 ,p1,p2, tmpa(iz),tmpb(iz),tmpc(iz) )
      IF (DIMAG(beta).LE.0d0) THEN !RIGHT
        rslt = rslt + sdnt         !RIGHT
      ELSE                         !RIGHT
        rslt = rslt - sdnt         !RIGHT
      ENDIF                        !RIGHT
      func_tfun = -sj*rslt/kk
      END FUNCTION func_tfun

END MODULE use_func
