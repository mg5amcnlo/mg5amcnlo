*###[ ffzli2:
	subroutine ffzli2(zdilog,zlog,cx,lreal,ier)
***#[*comment:***********************************************************
*									*
*	Computes the dilogarithm (Li2, Sp) for any (complex) cx 	*
*	to a precision precc.  It assumes that cx is already in the	*
*	area |cx|<=1, Re(cx)<=1/2.  As it is available it also returns	*
*	log(1-cx) = zlog.						*
*									*
*	Input:	cx	(complex)					*
*		lreal	(logical) indicates whether only the real part  *
*				  is needed				*
*									*
*	Output:	zdilog	(complex) Li2(cx)				*
*		zlog	(complex) log(1-cx) = -Li1(cx)			*
*		ier	(integer) 0=OK,1=num,2=err			*
*									*
*	Calls:	log,zfflo1,(d/a)imag,real/dble				*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	logical lreal
	DOUBLE COMPLEX cx,zlog,zdilog
*
*	local variables
*
	DOUBLE PRECISION xprec,bdn02,bdn05,bdn10,bdn15,bdn20,
     +		xi,xr,xdilog,xlog,x,absc,xa,a,ffbnd
	DOUBLE COMPLEX cc,cz,cz2,zfflo1
	save xprec,bdn02,bdn05,bdn10,bdn15,bdn20
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations: 
*  #[ initialisations:
	data xprec /-1./
	if ( xprec .ne. precc ) then
	    xprec = precc
	    bdn02 = ffbnd(1,2,bf)
	    bdn05 = ffbnd(1,5,bf)
	    bdn10 = ffbnd(1,10,bf)
	    bdn15 = ffbnd(1,15,bf)
	    bdn20 = ffbnd(1,19,bf)
*		we don't have bf(21) ...
	endif
*  #] initialisations: 
*  #[ check input:
*	(throw out later)
	if ( ltest .and. absc(cx).gt.1.5 .or. DBLE(cx).gt..75 ) then
	    call fferr(30,ier)
	    print *,'cx = ',cx
	endif
*  #] check input:
*  #[ exceptional cases:
	xi  = DIMAG(cx)
	xr  = DBLE(cx)
	if ( xi .eq. 0) then
	    call ffxli2(xdilog,xlog,xr,ier)
	    zdilog = xdilog
	    zlog = xlog
	    return
	endif
	xa = abs(xi) + abs(xr)
	if ( xa .lt. precc ) then
	    zdilog = cx
	    zlog = -cx
	    return
	endif
*  #] exceptional cases: 
*  #[ get log,dilog:
	if ( xa .lt. xloss**2 ) then
	    zlog = zfflo1(cx,ier)
	else
	    zlog = log(1-cx)
	endif
	cz = -zlog
	if ( absc(cz) .lt. xclog2 ) then
	    zdilog = cz
	else
	cz2 = cz*cz
	a = xa**2
	if ( lwarn .and. a .gt. bdn20 ) then
	    call ffwarn(61,ier,precc,abs(bf(20))*a**20)
	endif
	if ( a .gt. bdn15 ) then
	    zdilog = cz2*(DBLE(bf(16)) + cz2*(DBLE(bf(17))
     +		   + cz2*(DBLE(bf(18)) + cz2*(DBLE(bf(19))
     +		   + cz2*(DBLE(bf(20)))))))
	else
	    zdilog = 0
	endif
	if ( a .gt. bdn10 ) then
	    zdilog = cz2*(DBLE(bf(11)) + cz2*(DBLE(bf(12))
     +		   + cz2*(DBLE(bf(13)) + cz2*(DBLE(bf(14))
     +		   + cz2*(DBLE(bf(15)) + zdilog)))))
	endif
	if ( a .gt. bdn05 ) then
	    zdilog = cz2*(DBLE(bf(6)) + cz2*(DBLE(bf(7))
     +		   + cz2*(DBLE(bf(8)) + cz2*(DBLE(bf(9))
     +		   + cz2*(DBLE(bf(10)) + zdilog)))))
	endif
	if ( a .gt. bdn02 ) then
	    zdilog = cz2*(DBLE(bf(3)) + cz2*(DBLE(bf(4))
     +		   + cz2*(DBLE(bf(5)) + zdilog)))
	endif
*	watch the powers of z.
	zdilog = cz + cz2*(DBLE(bf(1)) + cz*(DBLE(bf(2)) + zdilog))
	endif
*  #] get log,dilog: 
*  #[ check for numerical problems:
*
*	if we just need the real part the dominant term is xi^2/4
*
	if ( lreal .and. abs(DBLE(zdilog)) .lt. xloss*xi**2/4 ) then
*	    think of something more intelligent later ...
	    x = DBLE(zdilog)
	    if ( lwarn ) call ffwarn(151,ier,x,xi**2/4)
	endif
*  #] check for numerical problems: 
*###] ffzli2:
	end
*###[ ffzzdl:
	subroutine ffzzdl(zdilog,ipi12,zlog,cx,ier)
***#[*comment:***************************************************
*								*
*	Computes the dilogarithm (Li2, Sp) for any (complex) cx *
*	to about 15 significant figures. This can be improved	*
*	by adding more of the bf's. For real cx > 1 an error is	*
*	generated as the imaginary part is undefined then.	*
*	For use in ffcdbd zlog = log(1-cx) is also calculated	*
*								*
*	Input:	cx	(complex)				*
*								*
*	Output:	zdilog	(complex) Li2(cx) mod factors pi^2/12	*
*		ipi12	(integer) these factors			*
*		zlog	(complex) log(1-cx)			*
*								*
*	Calls:	log,zfflo1,(d/a)imag,real/dble			*
*								*
***#]*comment:*************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi12,ier
	DOUBLE COMPLEX zdilog,zlog,cx
*
*	local variables
*
	integer jsgn
	DOUBLE PRECISION xprec,bdn02,bdn05,bdn10,bdn15,bdn20,
     +		xi,xr,s1,s2,xa,a,absc,ffbnd
	DOUBLE COMPLEX cfact,cx1,cy,cz,cz2,zfflo1,c
	save xprec,bdn02,bdn05,bdn10,bdn15,bdn20
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations: 
*  #[ initialisations:
	data xprec /-1./
	if ( xprec .ne. precc ) then
	    xprec = precc
	    bdn02 = ffbnd(1,2,bf)
	    bdn05 = ffbnd(1,5,bf)
	    bdn10 = ffbnd(1,10,bf)
	    bdn15 = ffbnd(1,15,bf)
	    bdn20 = ffbnd(1,19,bf)
	endif
*  #] initialisations: 
*  #[ debug:
*	if ( lwrite ) print *,'ffzzdl(',cx,')'
*  #] debug: 
*  #[ exceptional cases:
	xi  = DIMAG(cx)
	xr  = DBLE(cx)
	if ( xi .eq. 0 ) then
	    if ( xr .gt. 1 ) call fferr(31,ier)
	    call ffzxdl(zdilog,ipi12,zlog,xr,1,ier)
	    return
	endif
	if ( abs(xi) .lt. xalog2 ) then
	    s1 = 0
	else
	    s1 = xi**2
	endif
	if ( abs(xr) .lt. xalog2 ) then
	    s2 = 0
	else
	    s2 = xr**2
	endif
	xa = sqrt(s1 + s2)
	if ( xa .lt. precc ) then
	    zdilog = cx
	    zlog = -cx
	    ipi12 = 0
	    return
	endif
*  #] exceptional cases: 
*  #[ transform to |x|<1, Re(x) < 0.5:
	if ( xr .le. x05) then
	    if (xa .gt. 1) then
		if ( 1/xa .lt. xalogm ) then
		    cfact = 0
		elseif ( 1/xa .lt. xclogm ) then
		    cx1 = cx*DBLE(1/xa)
		    cfact = log(-cx1) + log(DBLE(xa))
		else
		    cfact = log(-cx)
		endif
		cy = - cfact**2/2
		ipi12 = -2
		if ( xa*xloss**2 .gt. 1) then
		    if ( 1/xa .lt. xclogm ) then
			cx1 = cx*DBLE(1/xa)
			cx1 = 1/cx1
			cx1 = cx1*DBLE(1/xa)
		    else
			cx1 = 1/cx
		    endif
		    cz = -zfflo1(cx1,ier)
		else
		    cz = -log(1-1/cx)
		endif
		zlog = log(1-cx)
		jsgn = -1
	    else
		cy = 0
		ipi12 = 0
		if ( xa .lt. xloss**2 ) then
		    zlog = zfflo1(cx,ier)
		else
		    zlog = log(1-cx)
		endif
		cz = -zlog
		jsgn = 1
	    endif
	else
	    if (xa .le. sqrt(2*xr)) then
		cz = -log(cx)
		if ( abs(xr-1) + abs(xi) .lt. xclogm ) then
		    if ( lwarn )
     +			call ffwarn(65,ier,abs(1-xr)+abs(xi),xclogm)
		    cy = 0
		else
		    zlog = log(1-cx)
		    cy = cz*zlog
		endif
		ipi12 = 2
		jsgn = -1
	    else
		if ( 1/xa .lt. xalogm ) then
		    cfact = 0
		elseif ( 1/xa .lt. xclogm ) then
		    cx1 = cx*DBLE(1/xa)
		    cfact = log(-cx1) + log(DBLE(xa))
		else
		    cfact = log(-cx)
		endif
		cy = - cfact**2/2
		ipi12 = -2
		if ( xa*xloss .gt. 1) then
		    if ( 1/xa .lt. xclogm ) then
			cx1 = cx*DBLE(1/xa)
			cx1 = 1/cx1
			cx1 = cx1*DBLE(1/xa)
		    else
			cx1 = 1/cx
		    endif
		    cz = -zfflo1(cx1,ier)
		else
		    cz = -log(1-1/cx)
		endif
		zlog = log(1-cx)
		jsgn = -1
	    endif
	endif
*  #] transform to |x|<1, Re(x) < 0.5: 
*  #[ get dilog:
	if ( absc(cz) .lt. xclogm ) then
	    zdilog = cz
	else
	cz2 = cz*cz
	a = DBLE(cz)**2 + DIMAG(cz)**2
	if ( lwarn .and. a .gt. bdn20 ) then
	    call ffwarn(67,ier,precc,abs(bf(20))*a**20)
	endif
	if ( a .gt. bdn15 ) then
	    zdilog = cz2*(DBLE(bf(16)) + cz2*(DBLE(bf(17))
     +		   + cz2*(DBLE(bf(18)) + cz2*(DBLE(bf(19))
     +		   + cz2*(DBLE(bf(20)))))))
	else
	    zdilog = 0
	endif
	if ( a .gt. bdn10 ) then
	    zdilog = cz2*(DBLE(bf(11)) + cz2*(DBLE(bf(12))
     +		   + cz2*(DBLE(bf(13)) + cz2*(DBLE(bf(14))
     +		   + cz2*(DBLE(bf(15)) + zdilog)))))
	endif
	if ( a .gt. bdn05 ) then
	    zdilog = cz2*(DBLE(bf(6)) + cz2*(DBLE(bf(7))
     +		   + cz2*(DBLE(bf(8)) + cz2*(DBLE(bf(9))
     +		   + cz2*(DBLE(bf(10)) + zdilog)))))
	endif
	if ( a .gt. bdn02 ) then
	    zdilog = cz2*(DBLE(bf(3)) + cz2*(DBLE(bf(4))
     +		   + cz2*(DBLE(bf(5)) + zdilog)))
	endif
*	watch the powers of z.
	zdilog = cz + cz2*(DBLE(bf(1)) + cz*(DBLE(bf(2)) + zdilog))
	endif
	if(jsgn.eq.1)then
	    zdilog =  zdilog + cy
	else
	    zdilog = -zdilog + cy
	endif
*  #] get dilog: 
*###] ffzzdl: 
	end
*###[ zfflog:
	DOUBLE COMPLEX function zfflog(cx,ieps,cy,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the complex logarithm of cx.  The following cases	*
*	are treted separately:						*
*		|cx| too small:		give warning and return 0	*
*					(for Absoft, Apollo DN300)	*
*		Im(cx) = 0, Re(cx) < 0:	take sign according to ieps	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
*
*	arguments
*
	implicit none
	integer ieps,ier
	DOUBLE COMPLEX cx,cy
*
*	local variables
*
	DOUBLE COMPLEX c,ctroep
	DOUBLE PRECISION absc,xa,xlog1p
*
*	common blocks, statement function
*
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations: 
*  #[ check input:
	if ( lwarn .and. absc(cx-1) .lt. xloss ) then
	    call ffwarn(128,ier,absc(cx-1),x1)
	endif
*  #] check input: 
*  #[ calculations:
	xa = absc(cx)
	if ( xa .lt. xalogm ) then
	    if ( cx .ne. 0 ) call fferr(23,ier)
	    zfflog = 0
	elseif ( DBLE(cx) .lt. 0 .and. DIMAG(cx) .eq. 0 ) then
*     +		 abs(DIMAG(cx)) .lt. precc*abs(DBLE(cx)) ) then
	    xlog1p = log(-DBLE(cx))
*	    checked imaginary parts 19-May-1988
	    if ( abs(ieps) .eq. 1 ) then
		if ( ieps*DBLE(cy) .lt. 0 ) then
		    zfflog = DCMPLX(xlog1p,-pi)
		elseif ( ieps*DBLE(cy) .gt. 0 ) then
		    zfflog = DCMPLX(xlog1p,pi)
		else
		    call fferr(51,ier)
		    zfflog = DCMPLX(xlog1p,pi)
		endif
	    elseif ( ieps .ge. 2 .and. ieps .le. 3 ) then
		zfflog = DCMPLX(xlog1p,-pi)
	    elseif ( ieps .le. -2 .and. ieps .ge. -3 ) then
		zfflog = DCMPLX(xlog1p,pi)
	    else
		call fferr(51,ier)
		zfflog = DCMPLX(xlog1p,pi)
	    endif
	    if ( ltest .and. DIMAG(cx) .ne. 0 ) then
		if ( DIMAG(zfflog) .gt. 0 .neqv. DIMAG(cx) .gt. 0 )
     +		    call fferr(56,ier)
	    endif
	elseif ( xa .lt. xclogm .or. 1/xa .lt. xclogm ) then
	    ctroep = cx*DBLE(1/xa)
	    zfflog = log(ctroep) + DBLE(log(xa))
	else
*	    print *,'zfflog: neem log van ',cx
	    zfflog = log(cx)
	endif
*  #] calculations: 
*###] zfflog: 
	end
*###[ zfflo1:
	DOUBLE COMPLEX function zfflo1(cx,ier)
***#[*comment:***************************************************
*	calculates log(1-x) for |x|<.14 in a faster way to ~15	*
*	significant figures.					*
***#]*comment:*************************************************** 
*  #[ declarations:
	implicit none
	integer ier
	DOUBLE COMPLEX cx,c,zfflog
	DOUBLE PRECISION xprec,bdn01,bdn05,bdn10,bdn15,bdn19,
     +		absc,xa,ffbnd
	save xprec,bdn01,bdn05,bdn10,bdn15,bdn19
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations: 
*  #[ initialisations:
	data xprec /-1./
	if ( precc .ne. xprec ) then
	    xprec = precc
*	    determine the boundaries for 1,5,10,15 terms
	    bdn01 = ffbnd(1,1,xninv)
	    bdn05 = ffbnd(1,5,xninv)
	    bdn10 = ffbnd(1,10,xninv)
	    bdn15 = ffbnd(1,15,xninv)
	    bdn19 = ffbnd(1,19,xninv)
	endif
*  #] initialisations: 
*  #[ calculations:
	xa = absc(cx)
	if ( xa .gt. bdn19 ) then
	    if ( lwarn .and. xa .lt. xloss ) call ffwarn(63,ier,xa,x1)
	    c = cx-1
	    xa = absc(c)
	    if ( lwarn .and. xa .lt. xloss ) call ffwarn(133,ier,xa,x1)
	    zfflo1 = zfflog(1-cx,0,c0,ier)
	    return
	endif
	if ( xa .gt. bdn15 ) then
	    zfflo1 = cx*( DBLE(xninv(16)) + cx*( DBLE(xninv(17))
     +		   + cx*( DBLE(xninv(18)) + cx*( DBLE(xninv(19))
     +		   + cx*( DBLE(xninv(20)) )))))
	else
	    zfflo1 = 0
	endif
	if ( xa .gt. bdn10 ) then
	    zfflo1 = cx*( DBLE(xninv(11)) + cx*( DBLE(xninv(12))
     +		   + cx*( DBLE(xninv(13)) + cx*( DBLE(xninv(14))
     +		   + cx*( DBLE(xninv(15)) + zfflo1 )))))
	endif
	if ( xa .gt. bdn05 ) then
	    zfflo1 = cx*( DBLE(xninv(6)) + cx*( DBLE(xninv(7))
     +		   + cx*( DBLE(xninv(8)) + cx*( DBLE(xninv(9))
     +		   + cx*( DBLE(xninv(10)) + zfflo1 )))))
	endif
	if ( xa .gt. bdn01 ) then
	    zfflo1 = cx*( DBLE(xninv(2)) + cx*( DBLE(xninv(3))
     +		   + cx*( DBLE(xninv(4)) + cx*( DBLE(xninv(5))
     +		   + zfflo1 ))))
	endif
	zfflo1 = - cx*( DBLE(xninv(1)) + zfflo1 )
*  #] calculations: 
*###] zfflo1: 
	end
*###[ zfflo2:
	DOUBLE COMPLEX function zfflo2(x,ier)
***#[*comment:***************************************************
*	calculates log(1-x)+x for |x|<.14 in a faster way to	*
*	~15 significant figures.				*
***#]*comment:*************************************************** 
*  #[ declarations:
	implicit none
	integer ier,ier0
	DOUBLE COMPLEX x,d1,zfflo1,cc
	DOUBLE PRECISION bdn01,bdn05,bdn10,bdn15,bdn18,xprec,xa,xheck,
     +		ffbnd,absc
	save xprec,bdn01,bdn05,bdn10,bdn15,bdn18
	include 'ff.h'
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations: 
*  #[ initialisation:
	data xprec /-1./
	if ( xprec .ne. precc ) then
	    xprec = precx
	    precx = precc
*	    determine the boundaries for 1,5,10,15 terms
	    bdn01 = ffbnd(1,1,xninv(2))
	    bdn05 = ffbnd(1,5,xninv(2))
	    bdn10 = ffbnd(1,10,xninv(2))
	    bdn15 = ffbnd(1,15,xninv(2))
	    bdn18 = ffbnd(1,18,xninv(2))
	    precx = xprec
	    xprec = precc
	endif
*  #] initialisation: 
*  #[ calculations:
	xa = absc(x)
	if ( xa .gt. bdn18 ) then
	    zfflo2 = zfflo1(x,ier) + x
	    if ( lwarn .and. absc(zfflo2).lt.xloss*abs(x) )
     +		call ffwarn(234,ier,absc(zfflo2),absc(x))
	    return
	endif
	if ( xa .gt. bdn15 ) then
	    zfflo2 = x*( DBLE(xninv(17)) + x*( DBLE(xninv(18)) +
     +		x*( DBLE(xninv(19)) + x*( DBLE(xninv(20)) ))))
	else
	    zfflo2 = 0
	endif
	if ( xa .gt. bdn10 ) then
	    zfflo2 = x*( DBLE(xninv(12)) + x*( DBLE(xninv(13)) +
     +		x*( DBLE(xninv(14)) + x*( DBLE(xninv(15)) +
     +		x*( DBLE(xninv(16)) + zfflo2 )))))
	endif
	if ( xa .gt. bdn05 ) then
	    zfflo2 = x*( DBLE(xninv(7)) + x*( DBLE(xninv(8)) +
     +		x*( DBLE(xninv(9)) +x*( DBLE(xninv(10)) +
     +		x*( DBLE(xninv(11)) + zfflo2 )))))
	endif
	if ( xa .gt. bdn01 ) then
	    zfflo2 = x*( DBLE(xninv(3)) + x*( DBLE(xninv(4)) +
     +		x*( DBLE(xninv(5)) + x*( DBLE(xninv(6)) + zfflo2 ))))
	endif
	zfflo2 = - x**2*( DBLE(xninv(2)) + zfflo2 )
*  #] calculations: 
*  #[ check output:
	if ( ltest ) then
	    ier0 = ier
	    d1 = zfflo1(x,ier0) + x
	    xheck = absc(d1-zfflo2)
	    if ( xloss*abs(xheck) .gt. precc ) print *,'zfflo2: error:',
     +		' answer is not OK',d1,zfflo2,xheck
	endif
*  #] check output: 
*###] zfflo2: 
	end
*###[ zfflo3:
	DOUBLE COMPLEX function zfflo3(x,ier)
***#[*comment:***************************************************
*	calculates log(1-x)+x+x^2/2 for |x|<.14 in a faster 	*
*	way to ~15 significant figures.				*
***#]*comment:*************************************************** 
*  #[ declarations:
	implicit none
	integer ier,ier0
	DOUBLE COMPLEX x,d1,zfflo2,cc
	DOUBLE PRECISION bdn01,bdn05,bdn10,bdn15,xprec,xa,xheck,ffbnd,
     +		absc
	save xprec,bdn01,bdn05,bdn10,bdn15
	include 'ff.h'
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations: 
*  #[ initialisation:
	data xprec /-1./
	if ( xprec .ne. precx ) then
	    xprec = precx
	    precx = precc
*	    determine the boundaries for 1,5,10,15 terms
	    bdn01 = ffbnd(1,1,xninv(3))
	    bdn05 = ffbnd(1,5,xninv(3))
	    bdn10 = ffbnd(1,10,xninv(3))
	    bdn15 = ffbnd(1,15,xninv(3))
	    precx = xprec
	    xprec = precc
	endif
*  #] initialisation: 
*  #[ calculations:
	xa = absc(x)
	if ( xa .gt. bdn15 ) then
	    zfflo3 = zfflo2(x,ier) + x**2/2
	    if ( lwarn .and. absc(zfflo3).lt.xloss*absc(x**2)/2 )
     +		call ffwarn(235,ier,absc(zfflo3),absc(x**2/2))
	    return
	endif
	if ( xa .gt. bdn10 ) then
	    zfflo3 = x*( DBLE(xninv(13)) + x*( DBLE(xninv(14)) +
     +		x*( DBLE(xninv(15)) + x*( DBLE(xninv(16)) +
     +		x*( DBLE(xninv(17)) )))))
	else
	    zfflo3 = 0
	endif
	if ( xa .gt. bdn05 ) then
	    zfflo3 = x*( DBLE(xninv(8)) + x*( DBLE(xninv(9)) +
     +		x*( DBLE(xninv(10)) + x*( DBLE(xninv(11)) +
     +		x*( DBLE(xninv(12)) + zfflo3 )))))
	endif
	if ( xa .gt. bdn01 ) then
	    zfflo3 = x*( DBLE(xninv(4)) + x*( DBLE(xninv(5)) +
     +		x*( DBLE(xninv(6)) + x*( DBLE(xninv(7)) + zfflo3 ))))
	endif
	zfflo3 = - x**3*( DBLE(xninv(3)) + zfflo3 )
*  #] calculations: 
*  #[ check output:
	if ( ltest ) then
	    ier0 = ier
	    d1 = zfflo2(x,ier0) + x**2/2
	    xheck = absc(d1-zfflo3)
	    if ( xloss*abs(xheck) .gt. precc ) print *,'zfflo3: error:',
     +		' answer is not OK',d1,zfflo3,xheck
	endif
*  #] check output: 
*###] zfflo3: 
	end
*###[ zff0li:
	DOUBLE COMPLEX function zff0li(r2)
***#[*comment:***********************************************************
*									*
*	computes complex value z such that  abs(z)**2 = r2 < 1		*
*	and Re(Li2(z))=0						*
*	written by P.Noguiero (Lisboa)					*									*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE PRECISION r2
*
*	local variables
*
	integer i
	DOUBLE PRECISION c1(30),c2(30),zr,zx
	save c1,c2
*
*	common blocks
*
*  #] declarations: 
*  #[ data:
*
	data c1(1) / 0.2500000000000000 /
	data c1(2) / -1.0416666666666667D-02 /
	data c1(3) / 1.2152777777777778D-03 /
	data c1(4) / -2.1959738756613757D-04 /
	data c1(5) / 4.9439553020282187D-05 /
	data c1(6) / -1.2675094665654561D-05 /
	data c1(7) / 3.5389820153701292D-06 /
	data c1(8) / -1.0493857656770419D-06 /
	data c1(9) / 3.2537695998679074D-07 /
	data c1(10) / -1.0442280388149559D-07 /
	data c1(11) / 3.4441733990714665D-08 /
	data c1(12) / -1.1615493272944038D-08 /
	data c1(13) / 3.9902649974583553D-09 /
	data c1(14) / -1.3922421108836989D-09 /
	data c1(15) / 4.9225507537640102D-10 /
	data c1(16) / -1.7605266995285916D-10 /
	data c1(17) / 6.3596990550536869D-11 /
	data c1(18) / -2.3176654407515461D-11 /
	data c1(19) / 8.5124040210417827D-12 /
	data c1(20) / -3.1483106624053104D-12 /
	data c1(21) / 1.1717062820424101D-12 /
	data c1(22) / -4.3854323145311313D-13 /
	data c1(23) / 1.6498013217746003D-13 /
	data c1(24) / -6.2356193829603354D-14 /
	data c1(25) / 2.3669242668432088D-14 /

	data c2(1) / 0.2500000000000000 /
	data c2(2) / 5.2083333333333333D-02 /
	data c2(3) / 6.4236111111111111D-03 /
	data c2(4) / 4.7484705687830688D-04 /
	data c2(5) / 1.4303971009700176D-05 /
	data c2(6) / -1.1031735071448613D-06 /
	data c2(7) / -1.6930087449913219D-07 /
	data c2(8) / -9.5437325895661167D-09 /
	data c2(9) / -1.1765492620111313D-10 /
	data c2(10) / 1.5727493777091249D-11 /
	data c2(11) / 6.7654901409698409D-13 /
	data c2(12) / -4.6807758765169774D-15 /
	data c2(13) / -2.4871711489610564D-15 /
	data c2(14) / -1.3622942781034796D-16 /
	data c2(15) / 3.8201988176071429D-17 /
	data c2(16) / -3.2258659308514033D-19 /
	data c2(17) / -4.5613496077409173D-19 /
	data c2(18) / 5.1177130568324641D-20 /
	data c2(19) / 5.4099028875697205D-22 /
	data c2(20) / -8.5181489051619174D-22 /
	data c2(21) / 9.6732395493921367D-23 /
	data c2(22) / 2.1141447009853665D-24 /
	data c2(23) / -1.8622848688015854D-24 /
	data c2(24) / 1.9077807703926496D-25 /
	data c2(25) / 8.0274683356039559E-27 /
*
*  #] data: 
*  #[ work:
*
	if ( abs(r2).le.0.1d0)t h en
	    zx = 0
	    do i=10,1,-1
		zx = r2*(zx+c1(i))
	    enddo
	elseif ( abs(r2).le.0.5d0 ) then
	    zx = 0
	    do i=20,1,-1
		zx = r2*(zx+c1(i))
	    enddo
	elseif ( abs(r2).le.1.0d0 ) then
	    zr = 2*log(1 + r2/2)
	    zx = 0
	    do i=13,1,-1
		zx = zr*(zx+c2(i))
	    enddo
	else
	    print *,'zff0li: error: argumnet must <= 1, not ',r2
	    zx = 0
	endif

	zff0li = DCMPLX(zx,sqrt(r2-zx*zx))
*
*  #] work: 
*###] zff0li: 
	end

