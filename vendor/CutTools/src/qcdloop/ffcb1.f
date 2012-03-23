*###[ ffcb1:
	subroutine ffcb1(cb1,cb0,ca0i,xp,xm1,xm2,piDpj,ier)
***#[*comment:***********************************************************
*									*
*	Calculate   1	/	   d^n Q Q(mu)				*
*		 ------	| ------------------------ = B1*p(mu)		*
*		 i pi^2	/ (Q^2-m1^2)((Q+p)^2-m2^2)			*
*									*
*	Input:	cb0	   complex	scalar twopoint function	*
*		ca0i(2)	   complex	scalar onepoint function with	*
*						m1,m2			*
*		xp	   complex	p.p in B&D metric		*
*		xm1,2	   complex	m_1^2,m_2^2			*
*		piDpj(3,3) complex	dotproducts between s1,s2,p	*
*		ier	   integer	digits lost so far		*
*	Output:	cb1	   complex	B1				*
*		ier	   integer	digits lost			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX xp,xm1,xm2,piDpj(3,3)
	DOUBLE COMPLEX cb1,cb0,ca0i(2)
*
*	local variables
*
	integer ier0,i,j
	DOUBLE COMPLEX dm1p,dm2p,dm1m2,cc
	DOUBLE PRECISION rm1,rm2,rp,rpiDpj(3,3),sprec,absc
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ real case:
	if ( DIMAG(xm1).eq.0 .and. DIMAG(xm2).eq.0 ) then
	    rm1 = DBLE(xm1)
	    rm2 = DBLE(xm2)
	    rp  = DBLE(xp)
	    do 20 j=1,3
		do 10 i=1,3
		    rpiDpj(i,j) = DBLE(piDpj(i,j))
   10		continue
   20	    continue
	    sprec = precx
	    precx = precc
	    call ffxb1(cb1,cb0,ca0i,rp,rm1,rm2,rpiDpj,ier)
	    precx = sprec
	    return
	endif
*  #] real case:
*  #[ get differences:
	ier0 = 0
	dm1m2 = xm1 - xm2
	dm1p = xm1 - xp
	dm2p = xm2 - xp
	if ( lwarn ) then
	    if ( abs(dm1m2) .lt. xloss*abs(xm1) .and. xm1 .ne. xm2 )
     +		call ffwarn(97,ier0,absc(dm1m2),absc(xm1))
	    if ( abs(dm1p) .lt. xloss*abs(xp) .and. xp .ne. xm1 )
     +		call ffwarn(98,ier0,absc(dm1p),absc(xp))
	    if ( abs(dm2p) .lt. xloss*abs(xp) .and. xp .ne. xm2 )
     +		call ffwarn(99,ier0,absc(dm2p),absc(xp))
	endif
*  #] get differences:
*  #[ call ffcb1a:
	call ffcb1a(cb1,cb0,ca0i,xp,xm1,xm2,dm1p,dm2p,dm1m2,piDpj,ier)
*  #] call ffcb1a:
*###] ffcb1:
	end
*###[ ffcb1a:
	subroutine ffcb1a(cb1,cb0,ca0i,xp,xm1,xm2,dm1p,dm2p,dm1m2,piDpj,
     +		ier)
***#[*comment:***********************************************************
*									*
*	Calculate   1	/	   d^n Q Q(mu)				*
*		 ------	| ------------------------ = B1*p(mu)		*
*		 i pi^2	/ (Q^2-m1^2)((Q+p)^2-m2^2)			*
*									*
*	Input:	cb0	   complex	scalar twopoint function	*
*		ca0i(2)	   complex	scalar onepoint function with	*
*						m1,m2			*
*		xp	   complex	p.p in B&D metric		*
*		xm1,2	   complex	m_1^2,m_2^2			*
*		piDpj(3,3) complex	dotproducts between s1,s2,p	*
*		ier	   integer	digits lost so far		*
*	Output:	cb1	   complex	B1				*
*		ier	   integer	digits lost			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX xp,xm1,xm2,dm1p,dm2p,dm1m2,piDpj(3,3)
	DOUBLE COMPLEX cb1,cb0,ca0i(2)
*
*	local variables
*
	integer i,j,ithres,init
	logical lneg,lreal
	DOUBLE PRECISION xmax,absc,bnd101,bnd105,bnd110,bnd115,ax,cprec,
     +		xprec,xmxp,rloss
	DOUBLE COMPLEX s,s1,h,slam,xma,xmb,x,small,dmbma,clam,clogm,
     +		ts2Dp,xlo3,xlogm,cqiqj(3,3),cqi(3),xnul
	DOUBLE COMPLEX cs(5),cc,csom
	DOUBLE PRECISION ffbnd
	DOUBLE COMPLEX zfflo1,zfflo3
	DOUBLE PRECISION rm1,rm2,rp,rm1m2,rm1p,rm2p,rpiDpj(3,3),sprec
	save cprec,bnd101,bnd105,bnd110,bnd115,init
*FOR ABSOFT ONLY
*	DOUBLE COMPLEX csqrt
*	external csqrt
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data cprec /0./
*
*  #] declarations:
*  #[ the real cases:
*
	if ( DIMAG(xm1) .eq. 0 .and. DIMAG(xm2) .eq. 0 ) then
	    lreal = .TRUE.
	    if ( lwrite ) print *,'ffcb1a: real masses'
	elseif ( nschem.le.4 ) then
	    lreal = .TRUE.
	    if ( lwrite .or. init.eq.0 ) then
		init = 1
		print *,'ffcb1a: nschem <= 4, ignoring complex masses:',
     +			nschem
	    endif
	elseif ( nschem.le.6 ) then
	    if ( lwrite .or. init.eq.0 ) then
		init = 1
		print *,'ffcb1a: nschem = 5,6 complex masses near ',
     +			'threshold: ',nschem
	    endif
	    cqi(1) = xm1
	    cqi(2) = xm2
	    cqi(3) = xp
	    cqiqj(1,2) = dm1m2
	    cqiqj(2,1) = -cqiqj(1,2)
	    cqiqj(1,3) = dm1p
	    cqiqj(3,1) = -cqiqj(1,3)
	    cqiqj(2,3) = dm2p
	    cqiqj(3,2) = -cqiqj(2,3)
	    cqiqj(1,1) = 0
	    cqiqj(2,2) = 0
	    cqiqj(3,3) = 0
	    call ffthre(ithres,cqi,cqiqj,3,1,2,3)
	    if ( ithres.eq.0 .or. ithres.eq.1 .and. nschem.eq.5 ) then
		lreal = .TRUE.
		if ( lwrite ) print *,'ffcb1a: no threshold'
	    else
		if ( lwrite ) print *,'ffcb1a: found threshold'
		lreal = .FALSE.
	    endif
	else
	    lreal = .FALSE.
	endif
	if ( lreal ) then
	    rm1 = DBLE(xm1)
	    rm2 = DBLE(xm2)
	    rp  = DBLE(xp)
	    rm1p  = DBLE(dm1p)
	    rm2p  = DBLE(dm2p)
	    rm1m2 = DBLE(dm1m2)
	    do 20 j=1,3
		do 10 i=1,3
		    rpiDpj(i,j) = DBLE(piDpj(i,j))
   10		continue
   20	    continue
	    sprec = precx
	    precx = precc
	    call ffxb1a(cb1,cb0,ca0i,rp,rm1,rm2,rm1p,rm2p,rm1m2,rpiDpj,
     +		ier)
	    precx = sprec
	    return
	endif
*  #] the real cases:
*  #[ check input:
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    xmax = max(absc(xm1),absc(xm2),abs(DBLE(xp)))
	    xnul = 2*piDpj(1,2) - xm1 - xm2 + xp
	    if ( rloss*absc(xnul) .gt. precc*xmax ) print *,
     +		'ffcb1a: error: s1.s2 wrong: ',2*piDpj(1,2),xm1+xm2-xp,
     +		xnul,ier
	    xnul = 2*piDpj(1,3) + xm1 - xm2 + xp
	    if ( rloss*absc(xnul) .gt. precc*xmax ) print *,
     +		'ffcb1a: error: s1.p wrong: ',2*piDpj(1,3),-xm1+xm2-xp,
     +		xnul,ier
	    xnul = 2*piDpj(2,3) + xm1 - xm2 - xp
	    if ( rloss*absc(xnul) .gt. precc*xmax ) print *,
     +		'ffcb1a: error: s2.p wrong: ',2*piDpj(2,3),-xm1+xm2+xp,
     +		xnul,ier
	endif
*  #] check input:
*  #[ p^2 != 0:
	if ( DBLE(xp) .ne. 0 ) then
* 	#[ normal case:
	    if ( dm1m2 .ne. 0 ) then
		cs(1) = -ca0i(2)
		cs(2) = +ca0i(1)
	    else
		cs(1) = 0
		cs(2) = 0
	    endif
	    cs(3) = +2*piDpj(1,3)*cb0
	    cb1 = cs(1) + cs(2) + cs(3)
	    xmax = max(absc(cs(2)),absc(cs(3)))
	    if ( absc(cb1) .ge. xloss*xmax ) goto 110
* 	#] normal case:
* 	#[ almost equal masses:
	    if ( absc(dm1m2) .le. xloss*absc(xm1) ) then
		if ( lwrite ) print *,'Using algorithms for dm1m2 small'
		cs(2) = dm1m2/xm1*cs(2)
		cs(1) = -xm2*zfflo1(-dm1m2/xm2,ier)
		if ( lwrite ) print *,'cb1 was',cb1,xmax
		cb1 = cs(1) + cs(2) + cs(3)
		xmax = max(absc(cs(2)),absc(cs(3)))
		if ( lwrite ) print *,'cb1 is ',cb1,xmax
		if ( absc(cb1) .ge. xloss*xmax ) goto 110
*		for the perfectionist (not me (today)):
*		if d0=0 and mu~m1(~m2), then the terms of order
*		(m1^2-m2^2) also cancel.  To patch this I need d0 and mu
	    endif
* 	#] almost equal masses:
* 	#[ p2 -> 0:
	    if ( xloss**2*max(absc(xm1),absc(xm2)) .gt. absc(xp) ) then
		if ( DBLE(xm2).gt.DBLE(xm1) ) then
		    xma = xm1
		    xmb = xm2
		    dmbma = -dm1m2
		    ts2Dp = +2*piDpj(2,3)
		    lneg = .FALSE.
		else
		    xma = xm2
		    xmb = xm1
		    dmbma = +dm1m2
		    ts2Dp = -2*piDpj(1,3)
		    lneg = .TRUE.
		endif
	    else
		goto 100
	    endif
*
*	    We found a situation in which p2 is much smaller than
*	    the masses.
*
	    if ( lwrite ) print *,'Using algorithms for p2 small'
	    if ( xma.eq.0 ) then
		clogm = 1
	    elseif ( absc(dmbma) .gt. xloss*absc(xmb) ) then
		clogm = log(xmb/xma)
	    else
		clogm = zfflo1(-dmbma/xma,ier)
	    endif
	    clam =  (dmbma-xp)**2 - 4*xma*xp
	    slam = sqrt(clam)
	    small = xp*(-2*(xma+xmb) + xp)/(slam+dmbma)
	    if ( lwrite ) then
		print *,'small = ',small
		print *,'vgl     ',slam-dmbma,slam
	    endif
	    cs(1) = clogm*xma*(4*xmb*(small-xp) + (small-xp)**2)/(2*
     +			(slam+dmbma)*(slam+2*piDpj(1,2)))
	    if ( lwrite ) then
		print *,'cs(1) = ',cs(1)
		print *,'vgl     ',
     +		+xma*clogm*(DBLE(x05)+(xma+xmb-xp/2)/(slam-xma+xmb))
     +		+xmb*clogm*(DBLE(x05)-(xma+xmb-xp/2)/(slam-xma+xmb))
	    endif
	    if ( cprec.ne.precc ) then
		cprec = precc
		xprec = precx
		precx = precc
		bnd101 = ffbnd(2,1,xinfac)
		bnd105 = ffbnd(2,5,xinfac)
		bnd110 = ffbnd(2,10,xinfac)
		bnd115 = ffbnd(2,15,xinfac)
		precx = xprec
	    endif
	    x = xp/slam
	    if ( lwrite ) print *,'Taylor expansion in ',x
	    ax = absc(x)
	    if ( lwarn .and. ax.gt.bnd115 )
     +		call ffwarn(220,ier,precc,xinfac(16)*ax**14)
	    if ( ax.gt.bnd110 ) then
		s = x*(DBLE(xinfac(12)) + x*(DBLE(xinfac(13)) +
     +		    x*(DBLE(xinfac(14)) + x*(DBLE(xinfac(15)) +
     +		    x*(DBLE(xinfac(16)) )))))
	    else
		s = 0
	    endif
	    if ( ax.gt.bnd105 ) then
		s = x*(DBLE(xinfac(7)) + x*(DBLE(xinfac(8)) +
     +		    x*(DBLE(xinfac(9)) + x*(DBLE(xinfac(10)) +
     +		    x*(DBLE(xinfac(11) + s) )))))
	    endif
	    if ( ax.gt.bnd101) then
		s = x*(DBLE(xinfac(3)) + x*(DBLE(xinfac(4)) +
     +		    x*(DBLE(xinfac(5)) + x*(DBLE(xinfac(6)) + s))))
	    endif
	    s = x**2*(DBLE(x05) + s)
	    s1 = 2*xp/(ts2Dp + slam)*(s + x)
	    h = -4*xp**2*xmb/(slam*(slam+ts2Dp)**2) - s + s1
	    if ( lwarn .and. absc(h) .lt. xloss*max(absc(s),absc(s1)) )
     +			then
		call ffwarn(221,ier,absc(h),max(absc(s),absc(s1)))
	    endif
	    if ( lwrite ) then
		print *,'arg ',h
		print *,'vgl ',1-(1-2*xp/(xp+dmbma+slam))*exp(xp/
     +			slam)
	    endif
	    if ( absc(h) .lt. .1 ) then
		cs(2) = dmbma*slam/xp*zfflo1(h,ier)
	    else
		print *,'ffcb1: warning: I thought this was small: ',h
		print *,'       cp,cma,cmb = ',xp,xma,xmb
		cs(2) = dmbma*slam/xp*log(1-h)
***		goto 100
	    endif
	    if ( lneg ) then
		cs(1) = -cs(1)
		cs(2) = -cs(2)
	    endif
	    cs(3) = -xp*cb0
	    if ( lwrite ) print *,'cb1 was',cb1,xmax
	    cb1 = cs(1) + cs(2) + cs(3)
	    xmax = max(absc(cs(2)),absc(cs(3)))
	    if ( lwrite ) then
		print *,'cb1 is ',cb1,xmax
		print *,'cs = ',(cs(i),i=1,3)
	    endif
	    if ( absc(cb1) .gt. xloss*xmax) goto 110
* 	#] p2 -> 0:
* 	#[ give up:
*
*	    give up...
*
  100	    continue
	    if ( lwarn ) then
		call ffwarn(167,ier,absc(cb1),xmax)
		if ( lwrite ) then
		    print *,'cs(i)      = ',(cs(i),i=1,3)
		    print *,'xp,xm1,xm2 = ',xp,xm1,xm2
		endif
	    endif
  110	    continue
* 	#] give up:
	    cb1 = cb1/(2*xp)
*  #] p^2 != 0:
*  #[ p^2=0, m1 != m2:
	elseif ( dm1m2 .ne. 0 ) then
	    cs(1) = +xm2/(2*dm1m2**2)*(ca0i(2)+xm2/2)
	    cs(2) = -xm1/(2*dm1m2**2)*(ca0i(1)+xm1/2)
	    cs(3) = +ca0i(2)/dm1m2
	    cb1 = cs(1) + cs(2) + cs(3)
	    xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)))
	    if ( absc(cb1).ge.xloss**2*xmax ) goto 120
	    if ( lwrite ) then
		print *,'cb1 = ',cb1,xmax
		print *,'with cs '
		print '(i3,2e30.16)',(i,cs(i),i=1,3)
	    endif
*
*	    m1 ~ m2, see b21.frm
*
	    if ( absc(dm1m2).lt.xloss*absc(xm1) ) then
		xlogm = zfflo1(dm1m2/xm1,ier)
	    else
		xlogm = log(xm2/xm1)
	    endif
	    cs(1) = -(xm1/dm1m2)/2
	    cs(2) = -xlogm/2*(xm1/dm1m2)**2
	    cs(3) = +1/DBLE(4) - ca0i(1)/(2*xm1)
	    cs(4) = xlogm/2
	    csom = cs(1) + cs(2) + cs(3) + cs(4)
	    xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)))
	    if ( lwrite ) then
		print *,'cb1+= ',csom,xmxp
		print *,'with cs '
		print '(i3,2e30.16)',(i,cs(i),i=1,4)
	    endif
	    if ( xmxp.lt.xmax ) then
		xmax = xmxp
		cb1 = csom
		if ( absc(cb1).gt.xloss**2*xmax ) goto 120
	    endif
*
*	    better
*
	    xlo3 = zfflo3(dm1m2/xm1,ier)
	    cs(1) = -(dm1m2/xm1)**2/4
	    cs(2) = -(dm1m2/xm1)/2
	    cs(3) = -xlo3/(dm1m2/xm1)**2/2
	    cs(4) = xlo3/2
	    cs(5) = 1/DBLE(2) - ca0i(1)/(2*xm1)
	    csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
	    xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),absc(cs(5)))
	    if ( lwrite ) then
		print *,'cb1+= ',csom,xmxp
		print *,'with cs '
		print '(i3,2e30.16)',(i,cs(i),i=1,5)
	    endif
	    if ( xmxp.lt.xmax ) then
		xmax = xmxp
		cb1 = csom
		if ( absc(cb1).gt.xloss**2*xmax ) goto 120
	    endif
*
*	    give up
*
	    if ( lwarn ) then
		if ( absc(cb1) .lt. xloss*xmax )
     +			call ffwarn(167,ier,absc(cb1),xmax)
	    endif
  120	    continue
*  #] p^2=0, m1 != m2:
*  #[ p^2=0, m1 == m2:
	else
	    cb1 = -cb0/2
	endif
*  #] p^2=0, m1 == m2:
*###] ffcb1a:
	end
