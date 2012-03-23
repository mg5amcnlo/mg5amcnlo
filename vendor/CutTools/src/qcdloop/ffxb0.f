*###[ ffxb0:
	subroutine ffxb0(cb0,d0,xmu,xp,xma,xmb,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the the two-point function (cf 't Hooft and Veltman)	*
*	we include an overall factor 1/(i*pi^2)	relative to FormF	*
*									*
*	Input:	d0	(real)	  infinity arising from renormalization	*
*		xmu	(real)	  renormalization mass			*
*		xp	(real)	  k2, in B&D metric			*
*		xma	(real)	  mass2					*
*		xmb	(real)	  mass2					*
*									*
*	Output:	cb0	(complex) B0, the two-point function,		*
*		ier	(integer) # of digits lost, if >=100: error	*
*									*
*	Calls:	ffxb0p							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cb0
	DOUBLE PRECISION d0,xmu,xp,xma,xmb
*
*	local variables
*
	integer ier0
	DOUBLE COMPLEX cb0p,c
	DOUBLE PRECISION dmamb,dmap,dmbp,xm,absc
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffxb0: nevent,id = ',nevent,id,' input:'
	    print *,'xma,xmb,xp,ier = ',xma,xmb,xp,ier
	endif
	if ( ltest ) then
	    if ( xma .lt. 0 .or. xmb .lt. 0 ) then
		print *,'ffxb0: error: xma,b < 0: ',xma,xmb
		stop
	    endif
	endif
*  #] check input:
*  #[ get differences:
	ier0 = 0
	dmamb = xma - xmb
	dmap = xma - xp
	dmbp = xmb - xp
	if ( lwarn ) then
	    if ( abs(dmamb) .lt. xloss*abs(xma) .and. xma .ne. xmb )
     +		call ffwarn(97,ier0,dmamb,xma)
	    if ( abs(dmap) .lt. xloss*abs(xp) .and. xp .ne. xma )
     +		call ffwarn(98,ier0,dmap,xp)
	    if ( abs(dmbp) .lt. xloss*abs(xp) .and. xp .ne. xmb )
     +		call ffwarn(99,ier0,dmbp,xp)
	endif
*  #] get differences:
*  #[ calculations:
	call ffxb0p(cb0p,xp,xma,xmb,dmap,dmbp,dmamb,ier)
	if ( xma .eq. 0 ) then
	    if ( xmb .eq. 0 ) then
		xm = x1
	    else
		xm = xmb**2
	    endif
	elseif ( xmb .eq. 0 ) then
	    xm = xma**2
	else
	    xm = xma*xmb
	endif
	if ( xmu .ne. 0 ) xm = xm/xmu**2
	if ( abs(xm) .gt. xalogm ) then
	    cb0 = DBLE(d0 - log(xm)/2) - cb0p
	    if ( lwarn .and. absc(cb0).lt.xloss*max(abs(d0),absc(cb0p)))
     +		call ffwarn(150,ier,absc(cb0),max(abs(d0),absc(cb0p)))
	else
	    call fferr(4,ier)
	    cb0 = DBLE(d0) - cb0p
	endif
	if ( lwrite ) print *,'B0 = ',cb0,ier
*  #] calculations:
*###] ffxb0:
	end
*###[ ffxb0p:
	subroutine ffxb0p(cb0p,xp,xma,xmb,dmap,dmbp,dmamb,ier)
***#[*comment:***********************************************************
*									*
*	calculates the two-point function (see 't Hooft and		*
*	Veltman) for all possible cases: masses equal, unequal,		*
*	equal to zero.							*
*									*
*	Input:	xp	(real) p.p, in B&D metric			*
*		xma	(real) mass2,					*
*		xmb	(real) mass2,					*
*		dm[ab]p	(real) xm[ab] - xp				*
*		dmamb	(real) xma - xmb				*
*									*
*	Output:	cb0p	(complex) B0, the two-point function, minus	*
*				  log(xm1*xm2)/2, delta and ipi^2	*
*		ier	(integer) 0=ok, 1=numerical problems, 2=error	*
*									*
*	Calls:	ffxb0q.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cb0p
	DOUBLE PRECISION xp,xma,xmb,dmap,dmbp,dmamb
*
*	local variables
*
	integer i,initeq,initn1,iflag,jsign,init
	DOUBLE PRECISION ax,ay,ffbnd,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn1,bdn101,bdn105,bdn110,bdn115,bdn120,
     +		xprnn2,bdn201,bdn205,bdn210,bdn215,bdn220,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515,
     +		absc
	DOUBLE PRECISION xcheck,xm,dmp,xm1,xm2,dm1m2,dm1p,
     +		dm2p,s,s1,s1a,s1b,s1p,s2,s2a,s2b,s2p,x,y,som,
     +		xlam,slam,xlogmm,alpha,alph1,xnoe,xpneq(30),
     +		xpnn1(30),xx,xtel,dfflo1
	DOUBLE COMPLEX cs2a,cs2b,cs2p,c,cx
	save initeq,initn1,init,xpneq,xpnn1,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn1,bdn101,bdn105,bdn110,bdn115,bdn120,
     +		xprnn2,bdn201,bdn205,bdn210,bdn215,bdn220,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data xprceq /-1./
	data xprcn1 /-1./
	data xprnn2 /-1./
	data xprcn3 /-1./
	data xprcn5 /-1./
	data initeq /0/
	data initn1 /0/
	data init /0/
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ check input:
	if (ltest) then
	    xcheck = xma - xmb - dmamb
	    if ( abs(xcheck) .gt. precx*max(abs(xma),abs(xmb),abs(
     +			dmamb))/xloss ) then
		print *,'ffxb0q: input not OK, dmamb <> xma-xmb',xcheck
	    endif
	    xcheck = -xp + xma - dmap
	    if ( abs(xcheck) .gt. precx*max(abs(xp),abs(xma),abs(
     +			dmap))/xloss ) then
		print *,'ffxb0q: input not OK, dmap <> xma - xp',xcheck
	    endif
	    xcheck = -xp + xmb - dmbp
	    if ( abs(xcheck) .gt. precx*max(abs(xp),abs(xmb),abs(
     +			dmbp))/xloss ) then
		print *,'ffxb0q: input not OK, dmbp <> xmb - xp',xcheck
	    endif
	endif
*  #] check input:
*  #[ fill some dotproducts:
	if ( ldot ) then
	    call ffdot2(fpij2,xp,xma,xmb,dmap,dmbp,dmamb,ier)
	endif
*  #] fill some dotproducts:
*  #[ which case:
*
*	sort according to the type of masscombination encountered:
*	100: both masses zero, 200: one equal to zero, 300: both equal
*	400: rest.
*
	if ( xma .eq. 0 ) then
		if ( xmb .eq. 0 ) then
			goto 100
		endif
		xm = xmb
		dmp = dmbp
		goto 200
	endif
	if ( xmb .eq. 0 ) then
		xm = xma
		dmp = dmap
		goto 200
	elseif ( dmamb .eq. 0 ) then
		xm = xma
		dmp = dmap
		goto 300
	elseif ( xma .gt. xmb ) then
		xm2 = xma
		xm1 = xmb
		dm1m2 = -dmamb
		dm1p = dmbp
		dm2p = dmap
	else
		xm1 = xma
		xm2 = xmb
		dm1m2 = dmamb
		dm1p = dmap
		dm2p = dmbp
	endif
	goto 400
*  #] which case:
*  #[ both masses equal to zero:
  100	continue
	if ( xp .lt. -xalogm ) then
	    cb0p = log(-xp) - 2
	elseif ( xp .gt. xalogm ) then
	    cb0p = DCMPLX( DBLE(log(xp) - 2), DBLE(-pi) )
	else
	    cb0p = 0
	    call fferr(7,ier)
	endif
	return
*  #] both masses equal to zero:
*  #[ one mass equal to zero:
  200	continue
*
*	special case xp = 0
*
	if ( xp .eq. 0 ) then
	    cb0p = -1
	    goto 990
*
*	special case xp = xm
*
	elseif ( dmp.eq.0 ) then
	    cb0p = -2
	    goto 990
	endif
*
*	Normal case:
*
	s1 = xp/xm
	if ( abs(s1) .lt. xloss ) then
	    s = dfflo1(s1,ier)
	else
	    s = log(abs(dmp/xm))
	endif
	s = -s*dmp/xp
	cb0p = s - 2
	if ( xp .gt. xm )
     +		cb0p = cb0p - DCMPLX(DBLE(x0),DBLE(-(dmp/xp)*pi))
	if ( lwarn .and. absc(cb0p) .lt. xloss*x2 )
     +		call ffwarn(14,ier,absc(cb0p),x2)
	goto 990
*  #] one mass equal to zero:
*  #[ both masses equal:
  300	continue
*
*	Both masses are equal.	Not only this speeds up things, some
*	cancellations have to be avoided as well.
*
*	first a special case
*
	if ( abs(xp) .lt. 8*xloss*xm ) then
* -#[	    taylor expansion:
*
*	    a Taylor expansion seems appropriate as the result will go
*	    as k^2 but seems to go as 1/k !!
*
*--#[	    data and bounds:
	    if ( initeq .eq. 0 ) then
		initeq = 1
		xpneq(1) = x1/6
		do 1 i=2,30
		    xpneq(i) = - xpneq(i-1)*DBLE(i-1)/DBLE(2*(2*i+1))
    1		continue
	    endif
	    if (xprceq .ne. precx ) then
*
*		calculate the boundaries for the number of terms to be
*		included in the taylorexpansion
*
		xprceq = precx
		bdeq01 = ffbnd(1,1,xpneq)
		bdeq05 = ffbnd(1,5,xpneq)
		bdeq11 = ffbnd(1,11,xpneq)
		bdeq17 = ffbnd(1,17,xpneq)
		bdeq25 = ffbnd(1,25,xpneq)
	    endif
*--#]	    data and bounds:
	    x = -xp/xm
	    ax = abs(x)
	    if ( lwarn .and. ax .gt. bdeq25 ) then
		call ffwarn(15,ier,precx,abs(xpneq(25))*ax**25)
	    endif
	    if ( ax .gt. bdeq17 ) then
		som = x*(xpneq(18) + x*(xpneq(19) + x*(xpneq(20) +
     +		x*(xpneq(21) + x*(xpneq(22) + x*(xpneq(23) +
     +		x*(xpneq(24) + x*(xpneq(25) ))))))))
	    else
		som = 0
	    endif
	    if ( ax .gt. bdeq11 ) then
		som = x*(xpneq(12) + x*(xpneq(13) + x*(xpneq(14) +
     +		x*(xpneq(15) + x*(xpneq(16) + x*(xpneq(17) + som ))))
     +		))
	    endif
	    if ( ax .gt. bdeq05 ) then
		som = x*(xpneq(6) + x*(xpneq(7) + x*(xpneq(8) + x*(
     +		xpneq(9) + x*(xpneq(10) + x*(xpneq(11) + som ))))))
	    endif
	    if ( ax .gt. bdeq01 ) then
		som = x*(xpneq(2) + x*(xpneq(3) + x*(xpneq(4) + x*(
     +		xpneq(5) + som ))))
	    endif
	    cb0p = x*(xpneq(1)+som)
	    if (lwrite) then
		print *,'ffxb0q: m1 = m2, Taylor expansion in ',x
	    endif
	    goto 990
* -#]	    taylor expansion:
	endif
* -#[	normal case:
*
*	normal case
*
	call ffxlmb(xlam,-xp,-xm,-xm,dmp,dmp,x0,ier)
	if ( xlam .ge. 0 ) then
*	    cases 1,2 and 4
	    slam = sqrt(xlam)
	    s2a = dmp + xm
	    s2 = s2a + slam
	    if ( abs(s2) .gt. xloss*slam ) then
*		looks fine
		jsign = 1
	    else
		s2 = s2a - slam
		jsign = -1
	    endif
	    ax = abs(s2/(2*xm))
	    if ( ax .lt. xalogm ) then
		if ( lwarn ) call ffwarn(16,ier,ax,xalogm)
		s = 0
	    elseif( ax-1 .lt. .1 .and. s2 .gt. 0 ) then
*		In this case a quicker and more accurate way is to
*		calculate log(1-x).
		s2 = (xp - slam)
*		the following line is superfluous.
		if ( lwarn .and. abs(s2) .lt. xloss*slam )
     +			call ffwarn(17,ier,s2,slam)
		s = -slam/xp*dfflo1(s2/(2*xm),ier)
	    else
*		finally the normal case
		s = -slam/xp*log(ax)
		if ( jsign .eq. -1 ) s = -s
	    endif
	    if ( xp .gt. 2*xm ) then
*		in this case ( xlam>0, so xp>(2*m)^2) ) there also
*		is an imaginary part
		y = -pi*slam/xp
	    else
		y = 0
	    endif
	else
*	    the root is complex (k^2 between 0 and (2*m1)^2)
	    slam = sqrt(-xlam)
	    s = 2*slam/xp*atan2(xp,slam)
	    y = 0
	endif
	if (lwrite) print *,'s =   ',s
	xx = s - 2
	if ( lwarn .and. abs(xx).lt.xloss*2 ) call ffwarn(18,ier,xx,x2)
	cb0p = DCMPLX(DBLE(xx),DBLE(y))
	goto 990
* -#]	normal case:
*
*  #] both masses equal:
*  #[ unequal nonzero masses:
* -#[	get log(xm2/xm1):
  400	continue
	x = xm2/xm1
	if ( 1 .lt. xalogm*x ) then
	    call fferr(8,ier)
	    xlogmm = 0
	elseif ( abs(x-1) .lt. xloss ) then
	    xlogmm = dfflo1(dm1m2/xm1,ier)
	else
	    xlogmm = log(x)
	endif
* -#]	get log(xm2/xm1):
* -#[	xp = 0:
*
*	first a special case
*
	if ( xp .eq. 0 ) then
	    s2 = ((xm2+xm1) / dm1m2)*xlogmm
	    s = - s2 - 2
*	    save the factor 1/2 for the end
	    if (lwrite) print *,'s = ',s/2
*	    save the factor 1/2 for the end
	    if ( abs(s) .lt. xloss*2 ) then
*		Taylor expansions: choose which one
		x = dm1m2/xm1
		ax = abs(x)
		if ( ax .lt. .15 .or. precx .gt. 1.E-8 .and. ax
     +			.lt. .3 ) then
*
*		    This is the simple Taylor expansion 'n1'
*
*--#[		    data and bounds:
*		    get the coefficients of the taylor expansion
		    if ( initn1 .eq. 0 ) then
			initn1 = 1
			do 410 i = 1,30
  410			    xpnn1(i) = DBLE(i)/DBLE((i+1)*(i+2))
		    endif
*		    determine the boundaries for 1,5,10,15 terms
		    if ( xprcn1 .ne. precx ) then
			xprcn1 = precx
			bdn101 = ffbnd(1,1,xpnn1)
			bdn105 = ffbnd(1,5,xpnn1)
			bdn110 = ffbnd(1,10,xpnn1)
			bdn115 = ffbnd(1,15,xpnn1)
			bdn120 = ffbnd(1,20,xpnn1)
		    endif
*--#]		    data and bounds:
*		    calculate:
		    if ( lwarn .and. ax .gt. bdn120 )
     +			call ffwarn(19,ier,precx,abs(xpnn1(20))*ax**19)
		    if ( ax .gt. bdn115 ) then
			s = x*(xpnn1(16) + x*(xpnn1(17) + x*(xpnn1(18) +
     +			    x*(xpnn1(19) + x*(xpnn1(20)) ))))
		    else
			s = 0
		    endif
		    if ( ax .gt. bdn110 ) then
			s = x*(xpnn1(11) + x*(xpnn1(12) + x*(xpnn1(13) +
     +			    x*(xpnn1(14) + x*(xpnn1(15)) + s))))
		    endif
		    if ( ax .gt. bdn105 ) then
			s = x*(xpnn1(6) + x*(xpnn1(7) + x*(xpnn1(8) + x*
     +			      (xpnn1(9) + x*(xpnn1(10) + s)))))
		    endif
		    if ( ax .gt. bdn101 ) then
			s = x*(xpnn1(2) + x*(xpnn1(3) + x*(xpnn1(4) + x*
     +			(xpnn1(5) +s))))
		    endif
		    s = x*x*(xpnn1(1) + s)
		    if (lwrite) then
			print *,'ffxb0q: xp = 0, simple Taylor exp'
			print *,'	 in ',x
			print *,'	 gives s ',s/2
		    endif
		else
*
*		    This is the more complicated Taylor expansion 'fc'
*
*  #[		    bounds:
*		    determine the boundaries for 1,5,10,15 terms for
*		    the exponential taylor expansion, assuming it
*		    starts at n=2.
*
		    if ( xprnn2 .ne. precx ) then
			xprnn2 = precx
			bdn201 = ffbnd(4,1,xinfac)
			bdn205 = ffbnd(4,5,xinfac)
			bdn210 = ffbnd(4,10,xinfac)
			bdn215 = ffbnd(4,15,xinfac)
			bdn220 = ffbnd(4,20,xinfac)
		    endif
*  #]		    bounds:
*		    calculate:
		    y = 2*x/(2-x)
		    ay = abs(y)
		    if ( lwarn .and. ay .gt. bdn220 )
     +			call ffwarn(20,ier,precx,xinfac(23)*ay**23)
		    if ( ay .gt. bdn220 ) then
			s = y*(xinfac(19) + y*(xinfac(20) + y*(xinfac(
     +				      21) + y*(xinfac(22) + y*(xinfac(
     +				      23) )))))
		    else
			s = 0
		    endif
		    if ( ay .gt. bdn215 ) then
			s = y*(xinfac(14) + y*(xinfac(15) + y*(xinfac(
     +				      16) + y*(xinfac(17) + y*(xinfac(
     +				      18) + s)))))
		    endif
		    if ( ay .gt. bdn210 ) then
			s = y*(xinfac(9) + y*(xinfac(10) + y*(xinfac(11)
     +			  + y*(xinfac(12) + y*(xinfac(13) + s)))))
		    endif
		    if ( ay .gt. bdn205 ) then
			s = y*(xinfac(5) + y*(xinfac(6) + y*(xinfac(7) +
     +			    y*(xinfac(8) + s))))
		    endif
		    s = (1-x)*y**4*(xinfac(4)+s)
		    s = x*y**2*(1+y)/12 - s
		    s = - 2*dfflo1(s,ier)/y
		    if (lwrite) then
			print *,'ffxb0q: xp = 0, other Taylor expansion'
			print *,'	 in ',y
			print *,'	 s = ',s/2
		    endif
		endif
	    endif
	    cb0p = s/2
	    goto 990
	endif
* -#]	xp = 0:
* -#[	normal case:
*
*	proceeding with the normal case
*
	call ffxlmb(xlam,-xp,-xm2,-xm1,dm2p,dm1p,dm1m2,ier)
	if ( xlam .gt. 0 ) then
*	    cases k^2 < -(m2+m1)^2 or k^2 > -(m2-m1)^2:
*--#[	    first try:
*	    first try the normal way
	    iflag = 0
	    slam = sqrt(xlam)
	    s2a = dm2p + xm1
	    s2 = s2a + slam
	    if ( abs(s2) .gt. xloss*slam ) then
*		looks fine
		jsign = 1
	    else
		s2 = s2a - slam
		jsign = -1
	    endif
	    s2 = s2**2/(4*xm1*xm2)
	    if ( abs(s2) .lt. xalogm ) then
		call fferr(9,ier)
		s2 = 0
	    elseif ( abs(s2-1) .lt. xloss ) then
		if ( jsign.eq.1 ) then
		    if ( lwrite ) print *,'s2 was ',-slam/(2*xp)*log(s2)
		    s2 = -slam*(s2a+slam)/(2*xm1*xm2)
		    s2 = -slam/(2*xp)*dfflo1(s2,ier)
		else
		    if ( lwrite ) print *,'s2 was ',+slam/(2*xp)*log(s2)
		    s2 = +slam*(s2a-slam)/(2*xm1*xm2)
		    s2 = +slam/(2*xp)*dfflo1(s2,ier)
		endif
		if ( lwrite ) print *,'s2 is  ',s2,jsign
	    else
		s2 = -slam/(2*xp)*log(s2)
		if ( jsign .eq. -1 ) s2 = -s2
	    endif
	    s1 = -dm1m2*xlogmm/(2*xp)
	    xx = s1+s2-2
	    if (lwrite) then
		print *,'ffxb0q: lam>0, first try, xx  = ',xx,s1,s2,-2
	    endif
*--#]	    first try:
	    if ( abs(xx) .lt. xloss*max(abs(s1),abs(s2)) ) then
*--#[		second try:
*		this is unacceptable, try a better solution
		s1a = dm1m2 + slam
		if (lwrite) print *,'s1 = ',-s1a/(2*xp),slam/(2*xp)
		if ( abs(s1a) .gt. xloss*slam ) then
*		    (strangely) this works
		    s1 = -s1a/(2*xp)
		else
*		    by division a more accurate form can be found
		    s1 = ( -xp/2 + xm1 + xm2 ) / ( slam - dm1m2 )
		    if (lwrite) print *,'s1+= ',s1
		endif
		s1 = s1*xlogmm
		if ( abs(xp) .lt. xm2 ) then
		    s2a = xp - dm1m2
		else
		    s2a = xm2 - dm1p
		endif
		s2 = s2a - slam
		if (lwrite) print *,'s2 = ',s2/(2*xm2),slam/(2*xm2)
		if ( abs(s2) .gt. xloss*slam ) then
*		    at least reasonable
		    s2 = s2 / (2*xm2)
		else
*		    division again
		    s2 = (2*xp) / (s2a+slam)
		    if (lwrite) print *,'s2+= ',s2
		endif
		if ( abs(s2) .lt. .1 ) then
*		    choose a quick way to get the logarithm
		    s2 = dfflo1(s2,ier)
		else
		    s2a = abs(1-s2)
		    s2 = log(s2a)
		endif
		s2 = -(slam/xp)*s2
		xx = s1 + s2 - 2
		if (lwrite) then
		    print *,'ffxb0q: lam>0, second try, xx  = ',xx
		    alpha = slam/(slam-dm1m2)
		    alph1 = -dm1m2/(slam-dm1m2)
		    print *,'	     alpha = ',alpha
		    print *,'	     s1 = ',s1,',- 2alph1 = ',s1-2*alph1
		    print *,'	     s2 = ',s2,',- 2alpha = ',s2-2*alpha
		endif
*--#]		second try:
		if ( abs(xx) .lt. xloss**2*max(abs(s1),abs(s2)) ) then
*--#[		    third try:
*		    (we accept two times xloss because that's the same
*		    as in this try)
*		    A Taylor expansion might work.  We expand
*		    inside the logs. Only do the necessary work.
*
		    alpha = slam/(slam-dm1m2)
		    alph1 = -dm1m2/(slam-dm1m2)
*
*		    First s1:
*
		    s1p = s1 - 2*alph1
		    if ( abs(s1p) .lt. xloss*abs(s1) ) then
* -#[			bounds:
*			determine the boundaries for 1,5,10,15 terms
			if ( xprcn3 .ne. precx ) then
			    xprcn3 = precx
			    bdn301 = ffbnd(3,1,xinfac)
			    bdn305 = ffbnd(3,5,xinfac)
			    bdn310 = ffbnd(3,10,xinfac)
			    bdn315 = ffbnd(3,15,xinfac)
			endif
* -#]			bounds:
			xnoe = -xp + 2*xm1 + 2*xm2
			x = 4*dm1m2/xnoe
			ax = abs(x)
			if ( lwarn .and. ax .gt. bdn315 ) then
			    call ffwarn(21,ier,precx,xinfac(17)*ax**14)
			endif
			if ( ax .gt. bdn310 ) then
			    s1a = x*(xinfac(13) + x*(xinfac(14) + x*(
     +				     xinfac(15) + x*(xinfac(16) + x*(
     +				     xinfac(17))))))
			else
			    s1a = 0
			endif
			if ( ax .gt. bdn305 ) then
			    s1a = x*(xinfac(8) + x*(xinfac(9) + x*(
     +				   xinfac(10) + x*(xinfac(11) + x*(
     +				   xinfac(12) + s1a)))))
			endif
			if ( ax .gt. bdn301 ) then
			    s1a = x*(xinfac(4) + x*(xinfac(5) + x*(
     +				     xinfac(6) + x*(xinfac(7) + s1a))))
			endif
			s1a = x**3 *(xinfac(3) + s1a) *xm2/xm1
			s1b = dm1m2*(4*dm1m2**2 - xp*(4*xm1-xp))/
     +				(xm1*xnoe**2)
			s1p = s1b - s1a
			if ( lwarn .and. abs(s1p).lt.xloss*abs(s1a) )
     +			    call ffwarn(22,ier,s1p,s1a)
			s1p = xnoe*dfflo1(s1p,ier)/(slam - dm1m2)/2
			if (lwrite) then
			    print *,'ffxb0q: Taylor exp. of s1-2(1-a)'
			    print *,'	     in x = ',x
			    print *,'	     gives s1p = ',s1p
			endif
		    endif
*
*		    next s2:
*
  490		    s2p = s2 - 2*alpha
		    if ( abs(s2p) .lt. xloss*abs(s2) ) then
* -#[			bounds:
*			determine the boundaries for 1,5,10,15 terms
			if ( xprcn5 .ne. precx ) then
			    xprcn5 = precx
			    bdn501 = ffbnd(4,1,xinfac)
			    bdn505 = ffbnd(4,5,xinfac)
			    bdn510 = ffbnd(4,10,xinfac)
			    bdn515 = ffbnd(4,15,xinfac)
			endif
* -#]			bounds:
			xnoe = slam - dm1m2
			x = 2*xp/xnoe
			ax = abs(x)
			if ( ax .gt. bdn515 ) then
*			    do not do the Taylor expansion
			    if ( lwarn ) call ffwarn(23,ier,s2p,s2) 
			    goto 495
			endif
			if ( ax .gt. bdn510 ) then
			    s2a = x*(xinfac(14) + x*(xinfac(15) + x*(
     +				     xinfac(16) + x*(xinfac(17) + x*(
     +				     xinfac(18))))))
			else
			    s2a = 0
			endif
			if ( ax .gt. bdn505 ) then
			    s2a = x*(xinfac(9) + x*(xinfac(10) + x*(
     +				    xinfac(11) + x*(xinfac(12) + x*(
     +				    xinfac(13) + s2a)))))
			endif
			if ( ax .gt. bdn501 ) then
			    s2a = x*(xinfac(5) + x*(xinfac(6) + x*(
     +				     xinfac(7) + x*(xinfac(8) + s2a))))
			endif
			s2a = x**4*(xinfac(4)+s2a)*(1-2*xp/(xnoe+xp))
			s2b = -2*xp**3 *(-2*xp - xnoe)/(3*(xnoe+xp)*
     +			    xnoe**3)
			s2p = s2b - s2a
			if ( lwarn .and. abs(s2p).lt.xloss*abs(s2a) )
     +				call ffwarn(24,ier,s2p,s2a)
			s2p = -slam/xp*dfflo1(s2p,ier)
			if (lwrite) then
			    print *,'ffxb0q: Taylor expansion of s2-2a'
			    print *,'	     in x = ',x
			    print *,'	     gives s2p = ',s2p
			endif
		    endif
*
*		    finally ...
*
  495		    xx = s1p + s2p
		    if ( lwarn .and. abs(xx) .lt. xloss*abs(s1p) ) then
			call ffwarn(25,ier,xx,s1p)
		    endif
*--#]		    third try:
		endif
	    endif
	    if ( xp .gt. xm1+xm2 ) then
*--#[		imaginary part:
*		in this case ( xlam>0, so xp>(m1+m2)^2) ) there also
*		is an imaginary part
		y = -pi*slam/xp
	    else
		y = 0
*--#]		imaginary part:
	    endif
	 else
*	    the root is complex (k^2 between -(m1+m2)^2 and -(m2-m1)^2)
*--#[	    first try:
	    slam = sqrt(-xlam)
	    xnoe = dm2p + xm1
	    s1 = -(dm1m2/(2*xp))*xlogmm
	    s2 = (slam/xp)*atan2(slam,xnoe)
	    xx = s1 + s2 - 2
	    if (lwrite) then
		print *,'ffxb0q: lam<0, first try, xx  = ',xx
		alpha = -xlam/(2*xp*xnoe)
		alph1 = -(xp**2-dm1m2**2)/(2*xp*xnoe)
		print *,'	 alpha = ',alpha
		print *,'	 s1 = ',s1,' - 2alph1 = ',s1-2*alph1
		print *,'	 s2 = ',s2,' - 2alpha = ',s2-2*alpha
	    endif
*--#]	    first try:
	    if ( abs(xx) .lt. xloss**2*max(abs(s1),abs(s2)) ) then
*--#[		second try:
*		Again two times xloss as we'll accept that in the next
*		step as well.
*
		xtel = dm1m2**2 - xp**2
		alpha = -xlam/(2*xp*xnoe)
		alph1 = xtel/(2*xp*xnoe)
*
*		try a taylor expansion on the terms.  First s1:
*
		s1p = s1 - 2*alph1
		if ( abs(s1p) .lt. xloss*abs(s1) ) then
* -#[		    bounds:
*		    determine the boundaries for 1,5,10,15 terms
		    if ( xprcn3 .ne. precx ) then
			xprcn3 = precx
			bdn301 = ffbnd(3,1,xinfac)
			bdn305 = ffbnd(3,5,xinfac)
			bdn310 = ffbnd(3,10,xinfac)
			bdn315 = ffbnd(3,15,xinfac)
		    endif
* -#]		    bounds:
		    x = 2*xtel/(dm1m2*xnoe)
		    ax = abs(x)
		    if ( ax .gt. bdn315 ) then
*			do not do the Taylor expansion
			if ( lwarn ) call ffwarn(21,ier,s1p,s1)
			goto 590
		    endif
		    if ( ax .gt. bdn310 ) then
			s1a = x*(xinfac(13) + x*(xinfac(14) + x*(
     +				 xinfac(15) + x*(xinfac(16) + x*(
     +				 xinfac(17))))))
		    else
			s1a = 0
		    endif
		    if ( ax .gt. bdn305 ) then
			s1a = x*(xinfac(8) + x*(xinfac(9) + x*(
     +				 xinfac(10) + x*(xinfac(11) + x*(
     +				 xinfac(12) + s1a)))))
		    endif
		    if ( ax .gt. bdn301 ) then
			s1a = x*(xinfac(4) + x*(xinfac(5) + x*(
     +				 xinfac(6) + x*(xinfac(7) + s1a))))
		    endif
		    s1a = x**3 *(xinfac(3) + s1a) *xm2/xm1
		    s1b = (dm1m2**3*(dm1m2**2-2*xp*xm1) + xp**2*(4*
     +			dm1m2*xm1**2-dm1m2**2*(dm1m2+2*xm1))-2*xm2*
     +			xp**3*(dm1m2+xp))/(xm1*dm1m2**2*xnoe**2)
		    s1p = s1b - s1a
		    if ( lwarn .and. abs(s1p) .lt. xloss*abs(s1a) )
     +			call ffwarn(22,ier,s1p,s1a)
		    s1p = -dm1m2*dfflo1(s1p,ier)/(2*xp)
		    if (lwrite) then
			print *,'ffxb0q: Taylor expansion of s1-2(1-a)'
			print *,'	 in x = ',x
			print *,'	 gives s1p = ',s1p
		    endif
		endif
*
*		next s2:
*
  590		continue
		s2p = s2 - 2*alpha
		if ( abs(s2p) .lt. xloss*abs(s2) ) then
* -#[		    bounds:
*		    determine the boundaries for 1,5,10,15 terms
		    if ( xprcn3 .ne. precx ) then
			xprcn3 = precx
			bdn301 = ffbnd(3,1,xinfac)
			bdn305 = ffbnd(3,5,xinfac)
			bdn310 = ffbnd(3,10,xinfac)
			bdn315 = ffbnd(3,15,xinfac)
		    endif
* -#]		    bounds:
		    cx = DCMPLX(DBLE(x0),DBLE(-slam/xnoe))
		    ax = absc(cx)
		    if ( ax .gt. bdn315 ) then
			if ( lwarn ) call ffwarn(23,ier,s2p,s2)
			goto 600
		    endif
		    if ( ax .gt. bdn310 ) then
			cs2a = cx*(DBLE(xinfac(13)) + cx*(DBLE(xinfac(14
     +			  )) + cx*(DBLE(xinfac(15)) + cx*(DBLE(xinfac(16
     +			  )) + cx*(DBLE(xinfac(17)))))))
		    else
			cs2a = 0
		    endif
		    if ( ax .gt. bdn305 ) then
			cs2a = cx*(DBLE(xinfac(8)) + cx*(DBLE(xinfac(9))
     +			   + cx*(DBLE(xinfac(10)) + cx*(DBLE(xinfac(11))
     +			   + cx*(DBLE(xinfac(12)) + cs2a)))))
		    endif
		    if ( ax .gt. bdn301 ) then
			cs2a = cx*(DBLE(xinfac(4)) + cx*(DBLE(xinfac(5))
     +			     + cx*(DBLE(xinfac(6)) + cx*(DBLE(xinfac(7))
     +			     + cs2a))))
		    endif
		    cs2a = cx**3*(DBLE(xinfac(3))+cs2a)*
     +				DCMPLX(DBLE(xnoe),DBLE(slam))
		    cs2b = DCMPLX(DBLE(xnoe-xlam/xnoe/2),
     +				 -DBLE(slam**3/xnoe**2/2))
		    cs2p = cs2b + cs2a
		    if ( lwarn .and. absc(cs2p) .lt. xloss*absc(cs2a) )
     +			call ffwarn(24,ier,absc(cs2p),absc(cs2b))
		    s2p = slam*atan2(DIMAG(cs2p),DBLE(cs2p))/xp
		    if (lwrite) then
			print *,'ffxb0q: Taylor expansion of s2-2a'
			print *,'	 in x = ',cx
			print *,'	 gives s2p = ',s2p
		    endif
		endif
  600		continue
		xx = s1p + s2p
		if ( lwarn .and. abs(xx) .lt. xloss*abs(s1p) ) then
		    call ffwarn(25,ier,xx,s1p)
		endif
*--#]		second try:
	    endif
	    y = 0
	endif
	cb0p = DCMPLX(DBLE(xx),DBLE(y))
	goto 990
* -#]	normal case:
*  #] unequal nonzero masses:
*  #[ debug:
  990	continue
	if (lwrite) then
	    print *,'cb0p  = ',cb0p,ier
	endif
*  #] debug:
*###] ffxb0p:
	end
*###[ ffxlmb:
	subroutine ffxlmb(xlambd,a1,a2,a3,a12,a13,a23,ier)
***#[*comment:***********************************************************
*	calculate in a numerically stable way				*
*	 lambda(a1,a2,a3) =						*
*		a1**2 + a2**2 + a3**2 - 2*a2*a3 - 2*a3*a1 - 2*a1*a2	*
*	aij = ai - aj are required for greater accuracy at times	*
*	ier is the usual error flag.					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xlambd,a1,a2,a3,a12,a13,a23
*
*	local variables
*
	DOUBLE PRECISION aa1,aa2,aa3,aa12,aa13,aa23,
     +		xcheck,a,aff,asq
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ calculations:
	aa1 = abs(a1)
	aa2 = abs(a2)
	aa3 = abs(a3)
	aa12 = abs(a12)
	aa13 = abs(a13)
	aa23 = abs(a23)
	if (ltest) then
*	    xcheck input
	    xcheck = a1 - a2 - a12
	    if ( xloss*abs(xcheck) .gt. precx*max(aa1,aa2,aa12) )
     +		print *,'ffxlmb: input not OK, a12 /= a1 - a2',a12,a1,
     +		a2,xcheck
	    xcheck = a1 - a3 - a13
	    if ( xloss*abs(xcheck) .gt. precx*max(aa1,aa3,aa13) )
     +		print *,'ffxlmb: input not OK, a13 /= a1 - a3',a13,a3,
     +		a3,xcheck
	    xcheck = a2 - a3 - a23
	    if ( xloss*abs(xcheck) .gt. precx*max(aa2,aa3,aa23) )
     +		print *,'ffxlmb: input not OK, a23 /= a2 - a3',a23,a2,
     +		a3,xcheck
	endif
*
*	first see if there are input parameters with opposite sign:
*
	if ( a1 .lt. 0 .and. a2 .gt. 0 .or.
     +	     a1 .gt. 0 .and. a2 .lt. 0 ) then
	    goto 12
	elseif ( a1 .lt. 0 .and. a3 .gt. 0 .or.
     +		 a1 .gt. 0 .and. a3 .lt. 0 ) then
	    goto 13
*
*	all have the same sign, choose the smallest 4*ai*aj term
*
	elseif ( aa1 .gt. aa2 .and. aa1 .gt. aa3 ) then
	    goto 23
	elseif ( aa2 .gt. aa3 ) then
	    goto 13
	else
	    goto 12
	endif
   12	continue
	if ( aa1 .gt. aa2 ) then
	    a = a13 + a2
	else
	    a = a1 + a23
	endif
	aff = 4*a1*a2
	goto 100
   13	continue
	if ( aa1 .gt. aa3 ) then
	    a = a12 + a3
	else
	    a = a1 - a23
	endif
	aff = 4*a1*a3
	goto 100
   23	continue
	if ( aa2 .gt. aa3 ) then
	    a = a12 - a3
	else
	    a = a13 - a2
	endif
	aff = 4*a2*a3
  100	continue
	asq = a**2
	xlambd = asq - aff
	if ( lwarn .and. abs(xlambd) .lt. xloss*asq )
     +		call ffwarn(69,ier,xlambd,asq)
*  #] calculations:
*###] ffxlmb:
	end
*###[ ffclmb:
	subroutine ffclmb(clambd,cc1,cc2,cc3,cc12,cc13,cc23,ier)
***#[*comment:***********************************************************
*	calculate in cc numerically stable way				*
*	 lambda(cc1,cc2,cc3) =						*
*	cc1**2 + cc2**2 + cc3**2 - 2*cc2*cc3 - 2*cc3*cc1 - 2*cc1*cc2	*
*	cij = ci - cj are required for greater accuracy at times	*
*	ier is the usual error flag.					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX clambd,cc1,cc2,cc3,cc12,cc13,cc23
*
*	local variables
*
	DOUBLE PRECISION aa1,aa2,aa3,aa12,aa13,aa23,absc
	DOUBLE COMPLEX check,cc,cff,csq,c
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ calculations (rather old style ...):
	aa1 = absc(cc1)
	aa2 = absc(cc2)
	aa3 = absc(cc3)
	aa12 = absc(cc12)
	aa13 = absc(cc13)
	aa23 = absc(cc23)
	if (ltest) then
*	    check input
	    check = cc1 - cc2 - cc12
	    if ( xloss*absc(check) .gt. precc*max(aa1,aa2,aa12) ) then
		print *,'ffclmb: input not OK, cc12 /= cc1 - cc2',check
	    endif
	    check = cc1 - cc3 - cc13
	    if ( xloss*absc(check) .gt. precc*max(aa1,aa3,aa13) ) then
		print *,'ffclmb: input not OK, cc13 /= cc1 - cc3',check
	    endif
	    check = cc2 - cc3 - cc23
	    if ( xloss*absc(check) .gt. precc*max(aa2,aa3,aa23) ) then
		print *,'ffclmb: input not OK, cc23 /= cc2 - cc3',check
	    endif
	endif
*
*	first see if there are input parameters with opposite sign:
*
	if ( DBLE(cc1) .lt. 0 .and. DBLE(cc2) .gt. 0 .or.
     +	     DBLE(cc1) .gt. 0 .and. DBLE(cc2) .lt. 0 ) then
	    goto 12
	elseif ( DBLE(cc1) .lt. 0 .and. DBLE(cc3) .gt. 0 .or.
     +		 DBLE(cc1) .gt. 0 .and. DBLE(cc3) .lt. 0 ) then
	    goto 13
*
*	all have the same sign, choose the smallest 4*ci*cj term
*
	elseif ( aa1 .gt. aa2 .and. aa1 .gt. aa3 ) then
	    goto 23
	elseif ( aa2 .gt. aa3 ) then
	    goto 13
	else
	    goto 12
	endif
   12	continue
	if ( aa1 .gt. aa2 ) then
	    cc = cc13 + cc2
	else
	    cc = cc1 + cc23
	endif
	cff = 4*cc1*cc2
	goto 100
   13	continue
	if ( aa1 .gt. aa3 ) then
	    cc = cc12 + cc3
	else
	    cc = cc1 - cc23
	endif
	cff = 4*cc1*cc3
	goto 100
   23	continue
	if ( aa2 .gt. aa3 ) then
	    cc = cc12 - cc3
	else
	    cc = cc13 - cc2
	endif
	cff = 4*cc2*cc3
  100	continue
	csq = cc**2
	clambd = csq - cff
	if ( lwarn .and. absc(clambd) .lt. xloss*absc(csq) )
     +		call ffwarn(68,ier,absc(clambd),absc(csq))
*  #] calculations (rather old style ...):
*###] ffclmb:
	end
*###[ ffdot2:
	subroutine ffdot2(piDpj,xp,xma,xmb,dmap,dmbp,dmamb,ier)
***#[*comment:***********************************************************
*									*
*	Store the 3 dotproducts in the common block ffdot.		*
*									*
*	Input:	see ffxb0p						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION piDpj(3,3),xp,xma,xmb,dmap,dmbp,dmamb
*
*	local variables
*
	integer ier0,ier1
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
*	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ work:
	ier1 = ier
	piDpj(1,1) = xma
	piDpj(2,2) = xmb
	piDpj(3,3) = xp
	if ( abs(dmap) .lt. abs(dmbp) ) then
		piDpj(1,2) = (dmap + xmb)/2
	else
		piDpj(1,2) = (dmbp + xma)/2
	endif
	piDpj(2,1) = piDpj(1,2)
	if ( lwarn .and. abs(piDpj(1,2)) .lt. xloss*min(xma,xmb)/2
     +			) then
		call ffwarn(24,ier1,piDpj(1,2),min(xma,xmb)/2)
	endif
	if ( abs(dmamb) .lt. abs(dmbp) ) then
		piDpj(1,3) = (-dmamb - xp)/2
	else
		piDpj(1,3) = (dmbp - xma)/2
	endif
	piDpj(3,1) = piDpj(1,3)
	if ( lwarn .and. abs(piDpj(1,3)) .lt. xloss*min(xma,
     +			abs(xp))/2) then
     		ier0 = ier
		call ffwarn(25,ier0,piDpj(1,3),min(xma,abs(xp))/2)
		ier1 = max(ier0,ier1)
	endif
	if ( abs(dmamb) .lt. abs(dmap) ) then
		piDpj(2,3) = (-dmamb + xp)/2
	else
		piDpj(2,3) = (-dmap + xmb)/2
	endif
	piDpj(3,2) = piDpj(2,3)
	if ( lwarn .and. abs(piDpj(2,3)) .lt. xloss*min(xmb,
     +			abs(xp))/2) then
     		ier0 = ier
		call ffwarn(25,ier0,piDpj(2,3),min(xmb,abs(xp))/2)
		ier1 = max(ier0,ier1)
	endif
	ier = ier1
*  #] work:
*###] ffdot2:
	end
