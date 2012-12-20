*--#[ log:
*	$Id: ffcxr.f,v 1.2 1995/11/10 19:04:24 gj Exp $
*	$Log: ffcxr.f,v $
c Revision 1.2  1995/11/10  19:04:24  gj
c Added nicer logging header...
c
*--#] log: 
*###[ ffcxr:
	subroutine ffcxr(crr,ipi12,y,y1,z,z1,dyz,ld2yzz,d2yzz,zz,zz1,
     +		ldy2z,dy2z,ieps,ier)
***#[*comment:***********************************************************
*									*
*	calculates R as defined in appendix b:				*
*									*
*		   /1    log(x-z+i*eps) - log(y-z+i*eps)		*
*	r(y,z)  =  \ dx  -----------------------------------		*
*		   /0		      x-y				*
*									*
*	    = li2(y/(y-z)+i*eps') - li2((y-1)/(y-z)+i*eps')		*
*									*
*	y,z are real, ieps integer denoting the sign of i*eps.		*
*	factors pi^2/12 are passed in the integer ipi12.		*
*									*
*	Input:	y	(real)						*
*		y1	(real)		1-y				*
*		z	(real)						*
*		z1	(real)		1-z				*
*		dyz	(real)		y-z				*
*									*
*		ld2yzz	(logical)	if .TRUE. also defined are:	*
*		d2yzz	(real)		2*y - z^+ - z^-			*
*		zz	(real)		the other z-root		*
*		zz1	(real)		1 - zz				*
*									*
*		ieps	(integer)	if +/-1 denotes sign imaginary	*
*					part of	argument logs		*
*		ieps	(integer)	if +/-2 denotes sign imaginary	*
*					part of	argument dilogs		*
*									*
*	Output	crr	(complex)	R modulo factors pi^2/12	*
*		ipi12	(integer)	these factors			*
*		ier	(intger)	0=ok, 1=num prob, 2=error	*
*									*
*	Calls:	ffxli2,(test: ffzxdl),dfflo1,zxfflg			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi12,ieps,ier
	logical ld2yzz,ldy2z
	DOUBLE PRECISION y,y1,z,z1,dyz,d2yzz,zz,zz1,dy2z(3)
	DOUBLE COMPLEX crr(7)
*
*	local variables
*
        integer i,iclas1,iclas2,iteken,ieps1,ieps2,ipi121,ipi122,ierdum
	logical taylor
	DOUBLE PRECISION xheck,fact,xx1,xx2,xx1p,xx2p,arg2,arg3,
     +		xli1,xli2,xli3,xlo1,xlo2,xlo3,xhill,xlog1,
     +		xlog2p,xx1n,d2,d21,d2n,d21n1,term,tot,xlia,xtroep,xli4,
     +		xlo4,rloss,som,xmax
	DOUBLE COMPLEX cr,cr1,clog1p,clog2p,ctroep,cli1,cli2
	DOUBLE PRECISION dfflo1
	DOUBLE COMPLEX zxfflg
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffcxr: input:'
	    print *,'  y    = ',y,y1
	    print *,'  z    = ',z,z1
	    print *,'  dyz  = ',dyz
	    if ( ld2yzz ) then
		print *,'  d2yzz= ',d2yzz
		print *,'  zz   = ',zz,zz1
	    endif
	    if ( ldy2z ) then
		print *,'  dy2z = ',dy2z(1),dy2z(3)
	    endif
	    print *,'  z->z - eps*',ieps
	endif
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    xheck = y + y1 - 1
	    if ( rloss*abs(xheck).gt.precx*max(abs(y),abs(y1),x1) ) then
		print *,'ffcxr: error: 1-y <> y1',y,y1,xheck,ier
	    endif
	    xheck = z + z1 - 1
	    if ( rloss*abs(xheck).gt.precx*max(abs(z),abs(z1),x1) ) then
		print *,'ffcxr: error: 1-z <> z1',z,z1,xheck,ier
	    endif
	    xheck = dyz - y + z
	    if ( rloss*abs(xheck).gt.precx*max(abs(z),abs(y),abs(dyz)) )
     +			then
		print *,'ffcxr: error: dyz<>y-z',dyz,y,z,xheck,ier
	    endif
	    if ( ld2yzz ) then
		xheck = d2yzz-2*y+z+zz
		if ( rloss*abs(xheck).gt.precx*max(abs(d2yzz),abs(2*y),
     +			abs(z),abs(zz)) ) then
		    print *,'ffcxr: error: d2yzz<>2y-z-zz',d2yzz,2*y,z,
     +			zz,xheck,ier
		endif
		xheck = zz + zz1 - 1
		if ( rloss*abs(xheck) .gt. precx*max(abs(zz),abs(zz1),
     +			x1)) then
		    print *,'ffcxr: error: 1-zz <> zz1',zz,zz1,xheck
		endif
	    endif
	    if ( ldy2z ) then
		xheck = dy2z(1)-y+2*z
		if ( rloss*abs(xheck).gt.precx*max(abs(dy2z(1)),abs(y),
     +			abs(2*z)) ) then
		    print *,'ffcxr: error: dy2z<>y-2z',dy2z(1),y,2*z,
     +			xheck,ier
		endif
		xheck = dy2z(3)-y1+2*z1
		if ( rloss*abs(xheck).gt.precx*max(abs(dy2z(2)),abs(y1),
     +			abs(2*z1)) ) then
		    print *,'ffcxr: error: dy2z1<>y1-2z1',dy2z(3),y1,
     +			2*z1,xheck,ier
		endif
	    endif
	    if ( abs(ieps).gt.2 ) then
		print*,'ffcxr: ieps is not -2,..2 ',ieps
	    endif
	endif
*  #] check input:
*  #[ groundwork:
	taylor = .FALSE.
*
*	get the arguments
*
	if ( dyz .eq. 0 ) then
	    if ( lwarn ) call ffwarn(51,ier,dyz,x1)
	    return
	endif
	fact = 1/dyz
	xx1 = y * fact
	xx2 = - y1 * fact
*
*  #] groundwork:
*  #[ which area?:
*
*	determine the area:	1 = [-1+xloss,1/2]
*				2 = (1/2,2-xloss]
*				3 = [2+xloss,->) U (<-,-1-xloss]
*				4 = [-1-xloss,-1+xloss]
*				5 = [2-xloss,2+xloss]
*
	if ( xx1 .lt. -1-xloss/2 ) then
	    iclas1 = 3
	    xx1p = 1/xx1
	elseif( xx1 .lt. -1+xloss/2 ) then
	    if ( ld2yzz ) then
		iclas1 = 4
	    else
		iclas1 = 1
	    endif
	    xx1p = xx1
	elseif( xx1 .le. x05 ) then
	    iclas1 = 1
	    xx1p = xx1
	elseif ( xx1 .lt. 2-xloss ) then
	    iclas1 = 2
	    xx1p = -z*fact
	elseif ( ldy2z .and. xx1 .lt. 2+xloss ) then
	    iclas1 = 5
	    xx1p = dy2z(1)*fact
	else
	    iclas1 = 3
	    xx1p = 1/xx1
	endif
	if ( xx2 .lt. -1-xloss/2 ) then
	    iclas2 = 3
	    xx2p = 1/xx2
	elseif( xx2 .lt. -1+xloss/2 ) then
	    if ( ld2yzz ) then
		iclas2 = 4
	    else
		iclas2 = 1
	    endif
	    xx2p = xx2
	elseif ( xx2 .le. x05 ) then
	    iclas2 = 1
	    xx2p = xx2
	elseif ( xx2 .lt. 2-xloss ) then
	    iclas2 = 2
	    xx2p = z1*fact
	elseif ( ldy2z .and. xx2 .lt. 2+xloss ) then
	    iclas2 = 5
	    xx2p = -dy2z(3)*fact
	else
	    iclas2 = 3
	    xx2p = 1/xx2
	endif
*
*	throw together if they are close
*
	if ( iclas1 .ne. iclas2 .and. abs(xx1-xx2) .lt. 2*xloss )
     +		then
*	    we don't want trouble with iclasn = 4,5
	    if ( iclas1 .eq. 4 ) then
		iclas1 = 1
	    elseif ( iclas1 .eq. 5 ) then
		iclas1 = 3
		xx1p = 1/xx1
	    endif
	    if ( iclas2 .eq. 4 ) then
		iclas2 = 1
	    elseif ( iclas2 .eq. 5 ) then
		iclas2 = 3
		xx2p = 1/xx2
	    endif
	    if ( iclas1 .eq. iclas2 ) goto 5
*	    go on
	    if ( iclas1 .le. iclas2 ) then
		iclas2 = iclas1
		if ( iclas1 .eq. 1 ) then
		    xx2p = xx2
		else
		    xx2p = z1*fact
		endif
	    else
		iclas1 = iclas2
		if ( iclas1 .eq. 1 ) then
		    xx1p = xx1
		else
		    xx1p = -z*fact
		endif
	    endif
	endif
*  #] which area?:
*  #[ calculations:
    5	if ( iclas1 .eq. iclas2 .and.
     +		abs(xx1p-xx2p) .lt. 2*xloss*max(abs(xx1p),abs(xx2p))
     +		.and. iclas1 .ne. 5 ) then
*		      |----->temporary!
*	    Close together:
* -#[	    handle dilog's:
	    if ( abs(xx2p) .gt. xloss ) then
*--#[		Hill identity:
*
*		Use the Hill identity to get rid of the cancellations.
*
*
*	    first get the arguments:
*
		if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
		    d2 = 1/y
		    arg2 = 1/z1
		    arg3 = arg2/xx1p
		elseif ( iclas1 .eq. 2 ) then
		    d2 = 1/z
		    arg2 = 1/y1
		    arg3 = arg2/xx1p
		elseif ( iclas1 .eq. 3 ) then
		    d2 = 1/y1
		    arg3 = 1/z1
		    arg2 = arg3*xx1p
		endif
		call ffxli2(xli1,xlo1,d2,ier)
		call ffxli2(xli2,xlo2,arg2,ier)
		call ffxli2(xli3,xlo3,arg3,ier)
		if ( abs(xx2p) .lt. xloss ) then
		    xlog2p = dfflo1(xx2p,ier)
		else
		    xlog2p = zxfflg(1-xx2p,0,x1,ier)
		endif
		xhill = xlo1*xlog2p
*--#]		Hill identity:
	    else
*--#[		Taylor expansion:
*
*		if the points are close to zero do a Taylor
*		expansion of the first and last dilogarithm
*
*			Li2(xx1p) - Li2(xx2p)
*			  = sum xx1p^i ( 1-(1-d2)^i ) /i^2
*
*		with d2 = 1-xx2p/xx1p = ...
*
		if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
		    d2 = 1/y
		elseif ( iclas1 .eq. 2 ) then
		    d2 = 1/z
		elseif ( iclas1 .eq. 3 ) then
		    d2 = 1/y1
		endif
*		flag to the print section that we did a Taylor expansion
		if ( lwrite ) taylor = .TRUE.
		d21 = 1-d2
		d21n1 = 1
		xx1n = xx1p
		d2n = d2
		tot = xx1p*d2
*		check for possible underflow on the next line
		if ( abs(xx1p) .lt. xalog2 ) goto 51
		do 50 i=2,20
		    xx1n = xx1n*xx1p
		    d21n1 = d21n1*d21
		    d2n = d2n + d2*d21n1
		    term = xx1n*d2n*xn2inv(i)
		    tot = tot + term
		    if ( abs(term) .le. precx*abs(tot) ) goto 51
   50		continue
		if ( lwarn ) call ffwarn(55,ier,abs(tot),abs(term))
   51		continue
		xli1 = tot
		xli2 = 0
		xli3 = 0
		xhill = 0
*		for the eta+transformation section we also need
		if ( iclas1 .ne. 1 ) then
		    if ( abs(d2) .lt. xloss ) then
			xlo1 = dfflo1(d2,ier)
		    else
			xlo1 = zxfflg(d21,0,x1,ier)
		    endif
		endif
		if ( iclas1 .eq. 2 ) xlo2 = dfflo1(1/y1,ier)
*--#]		Taylor expansion:
	    endif
*
* -#]	    handle dilog's:
* -#[	    handle transformation terms:
	    if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
*
*		no transformation was made.
*
*		crr(5) = 0
*		crr(6) = 0
	    elseif ( iclas1 .eq. 2 ) then
*
*		we tranformed to 1-x for both dilogs
*
		if ( abs(xx1p) .lt. xloss ) then
		    xlog1 = dfflo1(xx1p,ier)
		else
		    xlog1 = zxfflg(xx1,0,x1,ier)
		endif
		crr(5) = xlo1*xlog1
		clog2p = zxfflg(xx2p,ieps,-y1,ier)
*		if ( abs(xx2p) .lt. xalogm ) then
*		    if ( lwarn .and. xx2p .ne. 0 ) call ffwarn(53,ier,xx2p,xalogm)
*		    clog2p = 0
*		elseif ( xx2p .gt. 0 ) then
*		    clog2p = log(xx2p)
*		else
*		    xlog2p = log(-xx2p)
*		    checked imaginary parts 19-May-1988
*		    if ( abs(ieps) .eq. 1 ) then
*			if ( y1*ieps .gt. 0 ) then
*			    clog2p = DCMPLX(xlog2p,-pi)
*			else
*			    clog2p = DCMPLX(xlog2p,pi)
*			endif
*		    elseif ( ieps .eq. 2 ) then
*			clog2p = DCMPLX(xlog2p,-pi)
*		    else
*			clog2p = DCMPLX(xlog2p,pi)
*		    endif
*		endif
		crr(6) = -DBLE(xlo2)*clog2p
		if (lwrite) then
		    clog1p = zxfflg(xx1p,ieps,y,ier)
		endif
	    elseif ( iclas1 .eq. 3 ) then
*
*		we transformed to 1/x for both dilogs
*
		clog2p = zxfflg(-xx2p,-ieps,-y1,ier)
*		if ( abs(xx2p) .lt. xalogm ) then
*		    if ( lwarn ) call ffwarn(53,ier,xx2p,xalogm)
*		    clog2p = 0
*		elseif ( xx2p .lt. 0 ) then
*		    clog2p = log(-xx2p)
*		else
*		    xlog2p = log(xx2p)
*		    checked imaginary parts 19-May-1988
*		    if ( abs(ieps) .eq. 1 ) then
*			if ( ieps*y1 .gt. 0 ) then
*			    clog2p = DCMPLX(xlog2p,pi)
*			else
*			    clog2p = DCMPLX(xlog2p,-pi)
*			endif
*		    elseif ( ieps .eq. 2 ) then
*			clog2p = DCMPLX(xlog2p,-pi)
*		    else
*			clog2p = DCMPLX(xlog2p,pi)
*		    endif
*		endif
		crr(5) = DBLE(xlo1)*(clog2p - DBLE(xlo1)/2)
*		crr(6) = 0
		if (lwrite) then
		    clog1p = zxfflg(xx1p,ieps,y,ier)
		endif
	    endif
* -#]	    handle transformation terms:
* -#[	    add up and print out:
	    if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
		crr(1) = xli1
		crr(2) = xli2
		crr(3) = - xli3
		crr(4) = xhill
	    else
		crr(1) = - xli1
		crr(2) = - xli2
		crr(3) = xli3
		crr(4) = - xhill
	    endif
*	    crr(7) = 0
*	    ipi12 = 0
	    if ( lwrite ) then
		if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
		    iteken = 1
		else
		    iteken = -1
		endif
		if ( iclas1 .eq. 1 .or. iclas1 .eq. 4 ) then
		    cr = DBLE(xli1+xli2-xli3+xhill) + crr(5) + crr(6)
		else
		    cr = DBLE(-xli1-xli2+xli3-xhill) + crr(5) + crr(6)
		endif
		print *,'ffcxr: Close together'
		print *,'       oorspronkeijk:',xx1
		print *,'                    :',xx2
		print *,'       iclas = ',iclas1
		print *,'       Li2''s:',xli1*iteken
		if ( .not.taylor ) then
		print *,'            :',xli2*iteken
		print *,'            :',-xli3*iteken
		endif
		print *,'       log''s:',xhill*iteken
		print *,'       eta''s:',crr(5)
		print *,'            :',crr(6)
		print '(a,2g24.15,2i3)','    cr is dus:',cr,ipi12,ier
	    endif
* -#]	    add up and print out:
	else
*	    Normal case:
* -#[	    handle dilogs:
*
*	    the dilogs will not come close together so just go on
*	    only the special case xx1p ~ -1 needs special attention
*	    - and the special case xx1 ~ 2 also needs special attention
*
	    if ( iclas1 .eq. 4 ) then
		d2 = d2yzz + zz
		xmax = abs(d2yzz)
		if ( abs(d2) .lt. xloss*xmax ) then
		    if ( lwrite ) print *,'d2  = ',d2,xmax
		    som = y + dyz
		    if ( lwrite ) print *,'d2+ = ',som,abs(y)
		    if ( abs(y).lt.xmax ) then
			d2 = som
			xmax = abs(y)
		    endif
		    if ( lwarn .and. abs(d2) .lt. xloss*xmax ) then
			call ffwarn(58,ier,d2,xmax)
		    endif
		endif
		d2 = d2/dyz
		fact = 1/(2-d2)
		call ffxli2(xli1,xlo1,d2*fact,ier)
		call ffxli2(xli3,xlo3,-d2*fact,ier)
		call ffxli2(xli4,xlo4,d2,ier)
	    elseif ( iclas1 .eq. 5 ) then
		call ffxl22(xli1,xx1p,ier)
		ipi12 = ipi12 + 3
	    else
		call ffxli2(xli1,xlo1,xx1p,ier)
	    endif
	    if ( iclas2 .eq. 4 ) then
		if ( iclas1 .eq. 4 ) call fferr(26,ier)
		d2 = d2yzz - zz1
		xmax = abs(d2yzz)
		if ( abs(d2) .lt. xloss*xmax ) then
		    if ( lwrite ) print *,'d2  = ',d2,xmax
		    som = dyz - y1
		    if ( lwrite ) print *,'d2+ = ',som,abs(y1)
		    if ( abs(y1).lt.xmax ) then
			d2 = som
			xmax = abs(y1)
		    endif
		    if ( lwarn .and. abs(d2) .lt. xloss*xmax ) then
			call ffwarn(59,ier,d2,xmax)
		    endif
		endif
		d2 = d2/dyz
		fact = 1/(2-d2)
		call ffxli2(xli2,xlo2,d2*fact,ier)
		call ffxli2(xli3,xlo3,-d2*fact,ier)
		call ffxli2(xli4,xlo4,d2,ier)
	    elseif ( iclas2 .eq. 5 ) then
		call ffxl22(xli2,xx2p,ier)
		ipi12 = ipi12 - 3
	    else
		call ffxli2(xli2,xlo2,xx2p,ier)
	    endif
* -#]	    handle dilogs:
* -#[	    handle transformation terms xx1:
*
*	    transformation of c1
*
	    if ( iclas1 .eq. 1 ) then
		crr(1) = xli1
	    elseif( iclas1 .eq. 2 ) then
		crr(1) = -xli1
		ipi12 = ipi12 + 2
		clog1p = zxfflg(xx1p,ieps,y,ier)
		crr(3) = - DBLE(xlo1)*clog1p
	    elseif ( iclas1 .eq. 3 ) then
		crr(1) = -xli1
		ipi12 = ipi12 - 2
		clog1p = zxfflg(-xx1p,-ieps,y,ier)
		crr(3) = - clog1p**2/2
	    elseif ( iclas1 .eq. 4 ) then
		crr(1) = xli1
*		Note that this sum does not cause problems as d2<<1
		crr(3) = DBLE(-xli3-xli4) + DBLE(xlo4)*
     +			zxfflg(fact,0,x0,ier)
		ipi12 = ipi12 - 1
		if ( lwrite ) then
		    print *,'Check iclas1 = 4'
		    print '(a,2g14.8)','Nu:   ',crr(1)+crr(3)
		    call ffxli2(xlia,xtroep,xx1p,ier)
		    print '(a,2g14.8)','Eerst:',xlia+pi12,x0
		endif
	    elseif ( iclas1 .eq. 5 ) then
		crr(1) = xli1
*		supply an imaginary part
		clog1p = zxfflg(-1/xx1,-ieps,y,ier)
		xtroep = -DIMAG(clog1p)*DBLE(clog1p)
		crr(3) = DCMPLX(x0,xtroep)
	    else
		call fferr(26,ier)
	    endif
* -#]	    handle transformation terms xx1:
* -#[	    handle transformation terms xx2:
*
*	    transformation of c2
*
	    if ( iclas2 .eq. 1 ) then
		crr(2) = -xli2
	    elseif( iclas2 .eq. 2 ) then
		crr(2) = +xli2
		ipi12 = ipi12 - 2
		clog2p = zxfflg(xx2p,ieps,-y1,ier)
		crr(4) = + DBLE(xlo2)*clog2p
	    elseif ( iclas2 .eq. 3 ) then
		crr(2) = +xli2
		ipi12 = ipi12 + 2
		clog2p = zxfflg(-xx2p,-ieps,-y1,ier)
		crr(4) = clog2p**2/2
	    elseif ( iclas2 .eq. 4 ) then
		crr(2) = -xli2
*		Note that this sum does not cause problems as d2<<1
		crr(4) = DBLE(xli3+xli4) - DBLE(xlo4)*
     +			zxfflg(fact,0,x0,ier)
		ipi12 = ipi12 + 1
		if ( lwrite ) then
		    print *,'Check iclas2 = 4'
		    print '(a,2g14.8)','Nu:   ',-DBLE(xli2)+crr(4)
		    call ffxli2(xlia,xtroep,xx2p,ier)
		    print '(a,2g14.8)','Eerst:',-xlia-pi12,x0
		endif
	    elseif ( iclas2 .eq. 5 ) then
		crr(2) = -xli2
*		supply an imaginary part
		clog2p = zxfflg(-1/xx2,-ieps,-y1,ier)
		xtroep = DIMAG(clog2p)*DBLE(clog2p)
		crr(4) = DCMPLX(x0,xtroep)
	    else
		call fferr(28,ier)
	    endif
* -#]	    handle transformation terms xx2:
* -#[	    sum and print:
	    if ( lwrite ) then
		cr = crr(1) + crr(2) + crr(3) + crr(4) + crr(5) + crr(6)
		print *,'ffcxr: Normal case'
		print *,'	oorspronkelijk:',xx1
		print *,'	iclas1 = ',iclas1
		if(iclas1.ne.1)print *,'                   nu:',xx1p
		print *,'	Li21 :',crr(1)
		if(iclas1.ne.1)print *,'       tran1:',crr(3)
		if(crr(5).ne.0)print *,'             :',crr(5)
		if(crr(6).ne.0)print *,'             :',crr(6)
		print *,'	oorspronkelijk:',xx2
		print *,'	iclas2 = ',iclas2
		if(iclas2.ne.1)print *,'                   nu:',xx2p
		print *,'	Li22 :',-crr(2)
		if(iclas2.ne.1)print *,'        tran2:',-crr(4)
		if(crr(5).ne.0)print *,'             :',-crr(5)
		if(crr(6).ne.0)print *,'             :',-crr(6)
		print '(a,2g24.15,2i6)','    cr is dus:',cr,ipi12,ier
		if(ipi12.ne.0)print '(a,2g24.15)','             =',
     +			cr+ipi12*DBLE(pi12)
	    endif
* -#]	    sum and print:
	endif
*  #] calculations:
*  #[ debug:
	if ( lwrite ) then
	    if ( abs(ieps) .eq. 1 ) then
		if ( y .lt. 0 ) then
		    ieps1 = ieps
		else
		    ieps1 = -ieps
		endif
		if ( y1 .lt. 0 ) then
		    ieps2 = -ieps
		else
		    ieps2 = ieps
		endif
	    else
		ieps1 = ieps
		ieps2 = ieps
	    endif
	    ierdum = 0
	    call ffzxdl(cli1,ipi121,ctroep,xx1,ieps1,ierdum)
	    call ffzxdl(cli2,ipi122,ctroep,xx2,ieps2,ierdum)
	    cr1 = cli1 - cli2 + (ipi121-ipi122)*DBLE(pi12)
	    print '(a,2g24.15,i6)','    verg. cr1:',cr1,ierdum
	endif
*  #] debug:
*###] ffcxr:
	end
