*###[ ffxdb0:
	subroutine ffxdb0(cdb0,cdb0p,xp,xma,xmb,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the the derivative of the two-point function with	*
*	respect to p2 and the same times p2 (one is always well-defined)*
*									*
*	Input:	xp	(real)	  k2, in B&D metric			*
*		xma	(real)	  mass2					*
*		xmb	(real)	  mass2					*
*									*
*	Output:	cdb0	(complex) dB0/dxp				*
*		cdb0p	(complex) xp*dB0/dxp				*
*		ier	(integer) # of digits lost, if >=100: error	*
*									*
*	Calls:	ffxdba							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cdb0,cdb0p
	DOUBLE PRECISION xp,xma,xmb
*
*	local variables
*
	integer ier0
	DOUBLE PRECISION dmamb,dmap,dmbp
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffxdb0: input:'
	    print *,'xma,xmb,xp,ier = ',xma,xmb,xp,ier
	endif
	if ( ltest ) then
	    if ( xma .lt. 0 .or. xmb .lt. 0 ) then
		print *,'ffxdb0: error: xma,b < 0: ',xma,xmb
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
	call ffxdbp(cdb0,cdb0p,xp,xma,xmb,dmap,dmbp,dmamb,ier)
	if ( lwrite ) print *,'B0'' = ',cdb0,cdb0p,ier
*  #] calculations:
*###] ffxdb0:
	end
*###[ ffxdbp:
	subroutine ffxdbp(cdb0,cdb0p,xp,xma,xmb,dmap,dmbp,dmamb,ier)
***#[*comment:***********************************************************
*									*
*	calculates the derivatives of the two-point function		*
*	Veltman) for all possible cases: masses equal, unequal,		*
*	equal to zero.							*
*									*
*	Input:	xp	(real) p.p, in B&D metric			*
*		xma	(real) mass2,					*
*		xmb	(real) mass2,					*
*		dm[ab]p	(real) xm[ab] - xp				*
*		dmamb	(real) xma - xmb				*
*									*
*	Output:	cdb0	(complex) B0' = dB0/dxp				*
*		cdb0p	(complex) xp*dB0/dxp				*
*		ier	(integer) 0=ok,>0=numerical problems,>100=error	*
*									*
*	Calls:	ffxdbp.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cdb0,cdb0p
	DOUBLE PRECISION xp,xma,xmb,dmap,dmbp,dmamb
*
*	local variables
*
	integer i,initeq,jsign,initir
	DOUBLE PRECISION ax,ffbnd,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515,
     +		xprec0,bdn001,bdn005,bdn010,bdn015,bdn020
	DOUBLE PRECISION xcheck,xm,dmp,xm1,xm2,dm1m2,dm1p,
     +		dm2p,s,s1,s1a,s1b,s1p,s2,s2a,s2b,s2p,x,y,som,
     +		xlam,slam,xlogmm,alpha,alph1,xnoe,xpneq(30),
     +		xx,dfflo1,dfflo3,d1,d2,diff,h,a,b,c,d,beta,
     +		betm2n,xmax,s1c,s1d,s1e,s1f,s3
	DOUBLE COMPLEX cc,zxfflg
	save initeq,xpneq,initir,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515,
     +		xprec0,bdn001,bdn005,bdn010,bdn015,bdn020
*
*	common blocks
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	data
*
	data xprceq /-1./
	data xprec0 /-1./
	data xprcn3 /-1./
	data xprcn5 /-1./
	data initeq /0/
*
*  #] declarations:
*  #[ check input:
	if (ltest) then
	    xcheck = xma - xmb - dmamb
	    if ( abs(xcheck) .gt. precx*max(abs(xma),abs(xmb),abs(
     +			dmamb))/xloss ) then
		print *,'ffxdbp: input not OK, dmamb <> xma-xmb',xcheck
	    endif
	    xcheck = -xp + xma - dmap
	    if ( abs(xcheck) .gt. precx*max(abs(xp),abs(xma),abs(
     +			dmap))/xloss ) then
		print *,'ffxdbp: input not OK, dmap <> xma - xp',xcheck
	    endif
	    xcheck = -xp + xmb - dmbp
	    if ( abs(xcheck) .gt. precx*max(abs(xp),abs(xmb),abs(
     +			dmbp))/xloss ) then
		print *,'ffxdbp: input not OK, dmbp <> xmb - xp',xcheck
	    endif
	endif
*  #] check input:
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
	if ( xp.ne.0 ) cdb0 = -1/xp
	cdb0p = -1
	return
*  #] both masses equal to zero:
*  #[ one mass equal to zero:
  200	continue
*
*	special case xp = 0
*
	if ( xp .eq. 0 ) then
	    cdb0p = 0
	    cdb0 = 1/(2*xm)
	    goto 990
*
*	special case xp = xm
*
	elseif ( dmp.eq.0 ) then
	    if ( lsmug ) then
		if ( DBLE(cmipj(1,3)).lt.DBLE(cmipj(2,3)) ) then
		    cdb0p = -1 - log(cmipj(1,3)*DBLE(1/xm))
		else
		    cdb0p = -1 - log(cmipj(2,3)*DBLE(1/xm))
		endif
	    else
		if ( initir.eq.0 ) then
		    initir = 1
		    print *,'ffxdb0: IR divergent B0'', using cutoff ',
     +		    	delta 
		endif
		if ( delta.eq.0 ) then
		    call fferr(74,ier)
		    cdb0p = 0
		else
		    cdb0p = -1 + log(xm/delta)/2
		endif
	    endif
	    cdb0 = cdb0p*(1/DBLE(xp))
	    goto 990
	endif
*
*	Normal case:
*
	x = xp/xm
	ax = abs(x)
	if ( ax .lt. xloss ) then
* 	#[ Taylor expansion:
	    if ( xprec0 .ne. precx ) then
		xprec0 = precx
		bdn001 = ffbnd(2,1,xninv)
		bdn005 = ffbnd(2,5,xninv)
		bdn010 = ffbnd(2,10,xninv)
		bdn015 = ffbnd(2,15,xninv)
		bdn020 = ffbnd(2,20,xninv)
	    endif
	    if ( lwarn .and. ax .gt. bdn020 ) then
	    	call ffwarn(15,ier,precx,xninv(21)*ax**20)
	    endif
	    if ( ax .gt. bdn015 ) then
		som = x*(xninv(17) + x*(xninv(18) + x*(xninv(19) +
     +		      x*(xninv(20) + x*(xninv(21) )))))
	    else
		som = 0
	    endif
	    if ( ax .gt. bdn010 ) then
		som = x*(xninv(12) + x*(xninv(13) + x*(xninv(14) +
     +		      x*(xninv(15) + x*(xninv(16) + som )))))
	    endif
	    if ( ax .gt. bdn005 ) then
		som = x*(xninv(7) + x*(xninv(8) + x*(xninv(9) +
     +		      x*(xninv(10) + x*(xninv(11) + som )))))
	    endif
	    if ( ax .gt. bdn001 ) then
		som = x*(xninv(3) + x*(xninv(4) + x*(xninv(5) +
     +		      x*(xninv(6) + som ))))
	    endif
	    cdb0p = x*(xninv(2) + som)
	    if ( lwrite ) then
		print *,'cdb0p = ',cdb0p
		print *,'verg   ',-1 - xm/xp*dfflo1(x,ier),1
	    endif
* 	#] Taylor expansion:
	else
* 	#[ short formula:
	    s = log(abs(dmp/xm))
	    cdb0p = -(1 + s*xm/xp)
	    if ( xp.gt.xm ) cdb0p = cdb0p+DCMPLX(DBLE(0),DBLE(xm/xp*pi))
* 	#] short formula:
	endif
	cdb0 = cdb0p*(1/DBLE(xp))
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
		    xpneq(i) = - xpneq(i-1)*DBLE(i)/DBLE(2*(2*i+1))
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
	    cdb0p = -x*(xpneq(1)+som)
	    if (lwrite) then
		print *,'ffxdbp: m1 = m2, Taylor expansion in ',x
		print *,'cdb0p = ',cdb0p
	    endif
	    if ( xp.ne.0 ) then
		cdb0 = cdb0p*(1/DBLE(xp))
	    else
		cdb0 = xpneq(1)/xm
	    endif
	    goto 990
* -#]	    taylor expansion:
	endif
* -#[	normal case:
*
*	normal case
*
	call ffxlmb(xlam,-xp,-xm,-xm,dmp,dmp,x0,ier)
	if ( xlam .eq. 0 ) then
	    call fferr(86,ier)
	    return
	elseif ( xlam .gt. 0 ) then
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
		s = 2*xm/slam*dfflo1(s2/(2*xm),ier)
	    else
*		finally the normal case
		s = 2*xm/slam*log(ax)
		if ( jsign .eq. -1 ) s = -s
	    endif
	    if ( xp .gt. 2*xm ) then
*		in this case ( xlam>0, so xp>(2*m)^2) ) there also
*		is an imaginary part
		y = pi*2*xm/slam
	    else
		y = 0
	    endif
	else
*	    the root is complex (k^2 between 0 and (2*m1)^2)
	    slam = sqrt(-xlam)
	    s = 4*xm/slam*atan2(xp,slam)
	    y = 0
	endif
	if (lwrite) print *,'s =   ',s
	xx = s - 1
	if ( lwarn .and. abs(xx).lt.xloss ) call ffwarn(18,ier,xx,x1)
	cdb0p = DCMPLX(DBLE(xx),DBLE(y))
	cdb0 = cdb0p*(1/DBLE(xp))
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
*
*	    repaired 19-nov-1993, see b2.frm
*
	    s1 = xm1*xm2*xlogmm/dm1m2**3
	    s2 = (xm1+xm2)/(2*dm1m2**2)
	    s = s1 + s2
	    if ( abs(s) .lt. xloss**2*s2 ) then
*
*		second try
*
		h = dfflo3(dm1m2/xm1,ier)
		s1 = -xm1*h/dm1m2**2
		s2 = 1/(2*xm1)
		s3 = xm1**2*h/dm1m2**3
		s = s1 + s2 + s3
		if ( abs(s) .lt. xloss*max(abs(s2),abs(s3)) ) then
		    call ffwarn(228,ier,s,s2)
		endif
	    endif
	    cdb0 = s
	    cdb0p = 0
	    goto 990
	endif
* -#]	xp = 0:
* -#[	normal case:
*
*	proceeding with the normal case
*
	call ffxlmb(xlam,-xp,-xm2,-xm1,dm2p,dm1p,dm1m2,ier)
	diff = xlam + xp*(dm2p+xm1)
	if ( lwrite ) print *,'diff = ',diff
	if ( abs(diff) .lt. xloss*xlam ) then
	    h = dm1m2**2 - xp*(xm1+xm2)
	    if ( lwrite ) print *,'diff+= ',h
	    if ( abs(h) .lt. xloss*dm1m2**2 ) then
		if ( dm1m2**2 .lt. abs(xlam) ) diff = h
		if ( lwarn ) then
		    call ffwarn(221,ier,diff,min(dm1m2**2,abs(xlam)))
		endif
	    endif
	endif
	if ( xlam .eq. 0 ) then
	    call fferr(86,ier)
	    return
	elseif ( xlam .gt. 0 ) then
*	    cases k^2 < -(m2+m1)^2 or k^2 > -(m2-m1)^2:
*--#[	    first try:
*	    first try the normal way
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
		    if (lwrite) print *,'s2 ',-diff/(2*slam*xp)*log(s2)
		    s2 = -slam*(s2a+slam)/(2*xm1*xm2)
		    s2 = -diff/(2*slam*xp)*dfflo1(s2,ier)
		else
		    ier = ier + 50
		    print *,'ffxdb0: untested: s2 better in first try'
		    if (lwrite) print *,'s2 ',+diff/(2*slam*xp)*log(s2)
		    s2 = +slam*(s2a-slam)/(2*xm1*xm2)
		    s2 = +diff/(2*slam*xp)*dfflo1(s2,ier)
		endif
		if ( lwrite ) print *,'s2+ ',s2,jsign
	    else
		s2 = -diff/(2*slam*xp)*log(s2)
		if ( jsign .eq. -1 ) s2 = -s2
	    endif
	    s1 = -dm1m2*xlogmm/(2*xp)
	    xx = s1+s2-1
	    if (lwrite) then
		print *,'ffxdbp: lam>0, first try, xx  = ',xx,s1,s2,-1
	    endif
*--#]	    first try:
	    if ( abs(xx) .lt. xloss**2*max(abs(s1),abs(s2)) ) then
*--#[		second try:
*		this is unacceptable, try a better solution
		s1a = diff + slam*dm1m2
		if (lwrite) print *,'s1 = ',-s1a/(2*xp*slam),diff/
     +			(2*xp*slam)
		if ( abs(s1a) .gt. xloss*diff ) then
*		    this works
		    s1 = -s1a/(2*xp*slam)
		else
*		    by division a more accurate form can be found
		    s1 = -2*xm1*xm2*xp/(slam*(diff - slam*dm1m2))
		    if (lwrite) print *,'s1+= ',s1
		endif
		s = s1
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
		elseif ( s2.eq.1 ) then
		    print *,'ffxdbp: error: arg log would be 0!'
		    print *,'        xp,xma,xmb = ',xp,xma,xmb
		    goto 600
		else
		    h = abs(1-s2)
		    s2 = zxfflg(h,0,c0,ier)
		endif
		s2 = -diff/(slam*xp)*s2
		xx = s1 + s2 - 1
		if (lwrite) then
		    print *,'ffxdbp: lam>0, 2nd try, xx  = ',xx,s1,s2,-1
		endif
*--#]		second try:
		if ( abs(xx) .lt. xloss**2*max(abs(s1),abs(s2)) ) then
*--#[		    third try:
*		    (we accept two times xloss because that's the same
*		    as in this try)
*		    A Taylor expansion might work.  We expand
*		    inside the logs. Only do the necessary work.
*
*		#[ split up 1:
		    xnoe = s2a+slam
		    a = 1
		    b = 2/xnoe-1/xp
		    c = -4/(xp*xnoe)
		    d = sqrt((2/xnoe)**2 + 1/xp**2)
		    call ffroot(d1,d2,a,b,c,d,ier)
		    if ( xp.gt.0 ) then
			beta = d2
		    else
			beta = d1
		    endif
		    alpha = beta*diff/slam
		    alph1 = 1-alpha
		    if ( alph1 .lt. xloss ) then
			s1a = 4*xp**2*xm1*xm2/(slam*dm1m2*(diff-slam*
     +				dm1m2))
			s1b = -diff/slam*4*xm1*xp/(dm1m2*xnoe*(2*xp-
     +				xnoe))
			b = -1/xp
			c = -(2/xnoe)**2
			call ffroot(d1,d2,a,b,c,d,ier)
			if ( xp.gt.0 ) then
			    betm2n = d2
			else
			    betm2n = d1
			endif
			d1 = s1a + s1b - diff/slam*betm2n
			if ( lwrite ) then
			    print *,'alph1    = ',d1,s1a,s1b,-diff/slam*
     +				betm2n
			    print *,'verg       ',1-alpha
			endif
			xmax = max(abs(s1a),abs(s1b))
			if ( xmax .lt. 1 ) then
			    alph1 = d1
			else
			    xmax = 1
			endif
			if ( lwarn .and. abs(alph1).lt.xloss*xmax ) then
			    call ffwarn(222,ier,alph1,xmax)
			    if ( lwrite ) print *,'d1,s1a,s2b,... = ',
     +				d1,s1a,s1b,diff/slam*betm2n
			endif
		    else
			betm2n = beta - 2/xnoe
		    endif
		    if ( lwrite ) then
			print *,'     s1 - alph1 = ',s1-alph1
			print *,'     s2 - alpha = ',s2-alpha
		    endif
*		#] split up 1:
*		#[ s2:
*
*		    first s2:
*
  490		    continue
		    s2p = s2 - alpha
		    if ( abs(s2p) .lt. xloss*abs(s2) ) then
* -#[			bounds:
*			determine the boundaries for 1,5,10,15 terms
			if ( xprcn5 .ne. precx ) then
			    xprcn5 = precx
			    bdn501 = ffbnd(3,1,xinfac)
			    bdn505 = ffbnd(3,5,xinfac)
			    bdn510 = ffbnd(3,10,xinfac)
			    bdn515 = ffbnd(3,15,xinfac)
			endif
* -#]			bounds:
			x = beta*xp
			ax = abs(x)
			if ( lwarn .and. ax .gt. bdn515 ) then
*			    do not do the Taylor expansion
			    call ffwarn(23,ier,s2p,s2)
			    goto 495
			endif
			if ( ax .gt. bdn510 ) then
			    s2a = x*(xinfac(13) + x*(xinfac(14) + x*(
     +				     xinfac(15) + x*(xinfac(16) + x*(
     +				     xinfac(17))))))
			else
			    s2a = 0
			endif
			if ( ax .gt. bdn505 ) then
			    s2a = x*(xinfac(8) + x*(xinfac(9) + x*(
     +				    xinfac(10) + x*(xinfac(11) + x*(
     +				    xinfac(12) + s2a)))))
			endif
			if ( ax .gt. bdn501 ) then
			    s2a = x*(xinfac(4) + x*(xinfac(5) + x*(
     +				     xinfac(6) + x*(xinfac(7) + s2a))))
			endif
			s2a = x**3*(xinfac(3)+s2a)
			s2b = 2*xp/xnoe*(s2a + x**2/2)
			s2p = s2b - s2a
			if ( lwarn .and. abs(s2p).lt.xloss*abs(s2a) )
     +				call ffwarn(24,ier,s2p,s2a)
			s2p = -diff/(xp*slam)*dfflo1(s2p,ier)
			if (lwrite) then
			    print *,'ffxdbp: Taylor expansion of s2-a'
			    print *,'	     in x = ',x
			    print *,'	     gives s2p = ',s2p
			endif
		    endif
*		#] s2:
*		#[ s1:
*
*		    next s1:
*
  495		    continue
		    s1p = s1 - alph1
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
*
			x = slam*(diff-slam*dm1m2)*alph1/(2*xp*xm1*xm2)
			h = (2*xp*(xm1+xm2) - xp**2)/(slam-dm1m2)
			ax = abs(x)
			if ( lwarn .and. ax .gt. bdn315 ) then
*			    do not do the Taylor expansion
			    call ffwarn(21,ier,s1p,s1)
			    goto 500
			endif
*
*			see form job gets1.frm
*
			s1b = diff*(diff-slam*dm1m2)*betm2n/(2*xp*xm1*
     +				xm2)
			s1c = 1/(xm1*xnoe*(2*xp-xnoe))*(
     +				xp*( 4*xp*xm2 + 2*dm1m2**2/xm2*(xp-h) +
     +				2*dm1m2*(3*xp-h) - 8*dm1m2**2 )
     +				- 2*dm1m2**3/xm2*(3*xp-h)
     +				+ 4*dm1m2**4/xm2
     +				)
			if ( lwrite ) then
			    print *,'s1c was ',-2*xp/dm1m2 + 2*diff*
     +				(diff-slam*dm1m2)/(xm2*dm1m2*xnoe*(2*xp-
     +				xnoe)) + dm1m2/xm1
			    print *,'  en is ',s1c
			    print *,'s1b+s1c was ',dm1m2/xm1-x
			    print *,'      en is ',s1b+s1c
			endif
			s1d = x*dm1m2/xm1
			s1e = -x**2/2
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
			s1a = -x**3 *(xinfac(3) + s1a)
			s1f = dm1m2/xm1*(x**2/2 - s1a)
			s1p = s1e + s1d + s1c + s1b + s1a + s1f
			xmax = max(abs(s1a),abs(s1b),abs(s1c),abs(s1d),
     +				abs(s1e))
			if ( lwarn .and. abs(s1p).lt.xloss*xmax ) then
			    call ffwarn(223,ier,s1p,xmax)
			    if ( lwrite )
     +				print *,'s1p,s1e,s1d,s1c,s1b,s1a,s1f = '
     +				,s1p,s1e,s1d,s1c,s1b,s1a,s1f
			endif
			s1p = s*dfflo1(s1p,ier)
			if (lwrite) then
			    print *,'s1a = ',s1a
			    print *,'s1b = ',s1b
			    print *,'s1c = ',s1c
			    print *,'s1d = ',s1d
			    print *,'s1e = ',s1e
			    print *,'s1f = ',s1f
			    print *,'s   = ',s
			    print *,'ffxdbp: Taylor exp. of s1-(1-a)'
			    print *,'        in x = ',x
			    print *,'        gives s1p = ',s1p
			    print *,'        verg        ',s*log(xm2/xm1
     +				*exp(x))
			endif
		    endif
*		#] s1:
*
*		    finally ...
*
  500		    continue
		    xx = s1p + s2p
		    if ( lwarn .and. abs(xx) .lt. xloss*abs(s1p) ) then
			call ffwarn(25,ier,xx,s1p)
		    endif
*--#]		    third try:
		endif
	    endif
  600	    continue
	    if ( xp .gt. xm1+xm2 ) then
*--#[		imaginary part:
*		in this case ( xlam>0, so xp>(m1+m2)^2) ) there also
*		is an imaginary part
		y = -pi*diff/(slam*xp)
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
	    s2 = -diff/(slam*xp)*atan2(slam,xnoe)
	    xx = s1 + s2 - 1
	    if (lwrite) then
		print *,'ffxdbp: lam<0, first try, xx  = ',xx,s1,s2,-1
*		alpha = -xlam/(2*xp*xnoe)
*		alph1 = -(xp**2-dm1m2**2)/(2*xp*xnoe)
*		print *,'	 alpha = ',alpha
*		print *,'	 s1 = ',s1,' - 2alph1 = ',s1-2*alph1
*		print *,'	 s2 = ',s2,' - 2alpha = ',s2-2*alpha
	    endif
*--#]	    first try:
	    if ( lwarn .and. abs(xx).lt.xloss**2*max(abs(s1),abs(s2)) )
     +			then
		call ffwarn(224,ier,xx,max(abs(s1),abs(s2)))
	    endif
	    y = 0
	endif
  590	continue
	cdb0p = DCMPLX(DBLE(xx),DBLE(y))
	cdb0 = cdb0p*(1/DBLE(xp))
	goto 990
* -#]	normal case:
*  #] unequal nonzero masses:
*  #[ debug:
  990	continue
	if (lwrite) then
	    print *,'cdb0  = ',cdb0,cdb0p
	endif
*  #] debug:
*###] ffxdbp:
	end
