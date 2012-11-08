*###[ ffxc1:
	subroutine ffxc1(cc1i,cc0,cb0i,xpi,piDpj,del2,ier)
***#[*comment:***********************************************************
*									*
*	calculate the C1(mu) = C11*p1(mu) + C12*p2(mu) numerically	*
*									*
*	Input:	cc0	   complex	scalar threepoint function	*
*		cb0i(3)	   complex	scalar twopoint functions	*
*						without m1,m2,m3	*
*						(=with p2,p3,p1)	*
*		xpi(6)	   real		masses (1-3), momenta^2 (4-6)	*
*		piDpj(6,6) real		dotproducts as in C0		*
*		del2	   real		overall determinant		*
*		ier	   integer	digits lost so far		*
*	Output:	cc1i(2)	   complex	C11,C12				*
*		ier	   integer	number of dgits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xpi(6),piDpj(6,6),del2
	DOUBLE COMPLEX cc1i(2),cc0,cb0i(3)
*
*	local variables
*
	integer i,j,ier0
	DOUBLE PRECISION xmax,absc,xnul,xlosn,mc1i(2),mc0,mb0i(3)
	DOUBLE PRECISION dpipj(6,6),piDpjp(6,6)
	DOUBLE COMPLEX cc
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffxc1: input:'
	    print *,'xpi  = ',xpi
	    print *,'del2 = ',del2
	endif
	if ( ltest ) then
	    xlosn = xloss*DBLE(10)**(-1-mod(ier,50))
	    do 1 i=1,6
		if ( xpi(i) .ne. piDpj(i,i) ) then
		    print *,'ffxc1: error: xpi and piDpj do not agree'
		endif
    1	    continue
	    do 4 i=1,6
		do 3 j=1,6
		    dpipj(j,i) = xpi(j) - xpi(i)
    3		continue
    4	    continue
	    ier0 = 0
	    call ffdot3(piDpjp,xpi,dpipj,6,ier0)
	    do 7 i=1,6
		do 6 j=1,6
		    xnul = piDpj(j,i) - piDpjp(j,i)
		    if ( xlosn*abs(xnul) .gt. precx*abs(piDpjp(j,i)) )
     +			print *,'piDpj(',j,i,') not correct, cmp:',
     +			piDpj(j,i),piDpjp(j,i),xnul
    6		continue
    7	    continue
	    xnul = del2 - xpi(4)*xpi(5) + piDpj(4,5)**2
	    xmax = max(abs(del2),abs(xpi(4)*xpi(5)))
	    if ( xlosn*abs(xnul) .gt. precx*xmax ) then
		print *,'ffxc1: error: del2 != pi(4)*pi(5)-pi.pj(4,5)^2'
     +			,del2,xpi(4)*xpi(5),piDpj(4,5)**2,xnul
	    endif
	    i = 0
	    ltest = .FALSE.
	    call ffxb0(cc,x0,x1,xpi(4),xpi(1),xpi(2),i)
	    if ( xlosn*absc(cc-cb0i(3)) .gt. precc*absc(cc) ) print *,
     +		'cb0i(3) not right: ',cb0i(3),cc,cb0i(3)-cc
	    call ffxb0(cc,x0,x1,xpi(5),xpi(2),xpi(3),i)
	    if ( xlosn*absc(cc-cb0i(1)) .gt. precc*absc(cc) ) print *,
     +		'cb0i(1) not right: ',cb0i(1),cc,cb0i(1)-cc
	    call ffxb0(cc,x0,x1,xpi(6),xpi(3),xpi(1),i)
	    if ( xlosn*absc(cc-cb0i(2)) .gt. precc*absc(cc) ) print *,
     +		'cb0i(2) not right: ',cb0i(2),cc,cb0i(2)-cc
	    call ffxc0(cc,xpi,ier0)
	    if ( xlosn*absc(cc-cc0) .gt. precc*absc(cc) ) print *,
     +		'cc0 not right: ',cc0,cc,cc0-cc
	    ltest = .TRUE.
	endif
*  #] check input:
*  #[ call ffxc1a:
*
	mc0 = absc(cc0)*DBLE(10)**mod(ier,50)
	mb0i(1) = absc(cb0i(1))*DBLE(10)**mod(ier,50)
	mb0i(2) = absc(cb0i(2))*DBLE(10)**mod(ier,50)
	mb0i(3) = absc(cb0i(3))*DBLE(10)**mod(ier,50)
	call ffxc1a(cc1i,mc1i,cc0,mc0,cb0i,mb0i,xpi,piDpj,del2,ier)
*
*  #] call ffxc1a:
*###] ffxc1:
	end
*###[ ffxc1a:
	subroutine ffxc1a(cc1i,mc1i,cc0,mc0,cb0i,mb0i,xpi,piDpj,del2,
     +		ier)
***#[*comment:***********************************************************
*									*
*	calculate the C1(mu) = C11*p1(mu) + C12*p2(mu) numerically	*
*									*
*	Input:	cc0	   complex	scalar threepoint function	*
*		mc0	   real		maximal partial sum in C0	*
*		cb0i(3)	   complex	scalar twopoint functions	*
*						without m1,m2,m3	*
*						(=with p2,p3,p1)	*
*		mb0i(3)	   real		maxoimal partial sum in B0i	*
*		xpi(6)	   real		masses (1-3), momenta^2 (4-6)	*
*		piDpj(6,6) real		dotproducts as in C0		*
*		del2	   real		overall determinant		*
*		ier	   integer	digits lost so far		*
*	Output:	cc1i(2)	   complex	C11,C12				*
*		mc1i(2)	   real		maximal partial sum in C11,C12	*
*		ier	   integer	number of dgits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION mc1i(2),mc0,mb0i(3),xpi(6),piDpj(6,6),del2
	DOUBLE COMPLEX cc1i(2),cc0,cb0i(3)
*
*	local variables
*
	integer i,ier0,ier1
	DOUBLE PRECISION xmax,absc,del2s2,dpipj(6,6),ms(5)
	DOUBLE COMPLEX cs(5),cc
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
*  #[ debug input:
	if ( lwrite ) then
	    print *,'ffxc1: input, ier = ',ier
	    print *,'cc0     = ',cc0,mc0
	    print *,'cb0i(1) = ',cb0i(1),mb0i(1)
	    print *,'cb0i(2) = ',cb0i(2),mb0i(2)
	    print *,'cb0i(3) = ',cb0i(3),mb0i(3)
	    print *,'xpi = ',xpi
	    print *,'del2= ',del2
	endif
	if ( del2.eq.0 ) then
	    call fferr(92,ier)
	    return
	endif
*  #] debug input:
*  #[ calculations:
*   C1 =
*       + p1(mu)*Del2^-1 * (  - 1/2*B(p1)*p1.p2 - 1/2*B(p2)*p2.p2 - 1/2*B(p3)*
*         p2.p3 - C*p1.p2*p2.s1 + C*p1.s1*p2.p2 )
*
*       + p2(mu)*Del2^-1 * ( 1/2*B(p1)*p1.p1 + 1/2*B(p2)*p1.p2 + 1/2*B(p3)*
*         p1.p3 + C*p1.p1*p2.s1 - C*p1.p2*p1.s1 );
*
	cs(1) = - cb0i(1)*DBLE(piDpj(5,5))
	cs(2) = - cb0i(2)*DBLE(piDpj(6,5))
	cs(3) = - cb0i(3)*DBLE(piDpj(4,5))
	cs(4) = - 2*cc0*DBLE(piDpj(1,5)*piDpj(4,5))
	cs(5) = + 2*cc0*DBLE(piDpj(1,4)*piDpj(5,5))
	ms(1) = mb0i(1)*abs(piDpj(5,5))
	ms(2) = mb0i(2)*abs(piDpj(6,5))
	ms(3) = mb0i(3)*abs(piDpj(4,5))
	ms(4) = 2*mc0*abs(piDpj(1,5)*piDpj(4,5))
	ms(5) = 2*mc0*abs(piDpj(1,4)*piDpj(5,5))
*	exceptions
	if ( xpi(2).eq.xpi(3) .and. xpi(4).eq.xpi(6) ) then
	    if ( lwrite ) print *,'special case m1=m3,p5=p6'
	    cs(2) = + cb0i(2)*DBLE(xpi(5))
	    cs(3) = 0
	    ms(2) = + mb0i(2)*xpi(5)
	    ms(3) = 0
	endif
*	more to come?
*
	cc1i(1) = 0
	mc1i(1) = 0
	xmax = 0
	do 10 i=1,5
	    cc1i(1) = cc1i(1) + cs(i)
	    xmax = max(xmax,absc(cs(i)))
	    mc1i(1) = max(mc1i(1),ms(i))
   10	continue
	ier0 = ier
	if ( lwarn .and. absc(cc1i(1)) .lt. xloss*xmax ) then
	    call ffwarn(163,ier0,absc(cc1i(1)),xmax)
	    if ( lwrite ) then
		print *,'cs(i),ms(i) = '
		print '(i2,3g16.8)',(i,cs(i),ms(i),i=1,5)
		print '(a2,3g16.8)','+ ',cc1i(1),mc1i(1)
	    endif
	endif
	cc1i(1) = cc1i(1)*DBLE(1/(2*del2))
	mc1i(1) = mc1i(1)*abs(1/(2*del2))
*
	cs(1) = + cb0i(1)*DBLE(piDpj(5,4))
	cs(2) = + cb0i(2)*DBLE(piDpj(6,4))
	cs(3) = + cb0i(3)*DBLE(piDpj(4,4))
*	invalidate dpipj
	dpipj(1,1) = 1
	ier1 = ier
	call ffdl2p(del2s2,xpi,dpipj,piDpj, 4,5,6, 1,2,3, 6,ier1)
	cs(4) = + 2*cc0*DBLE(del2s2)
	ms(1) = mb0i(1)*abs(piDpj(5,4))
	ms(2) = mb0i(2)*abs(piDpj(6,4))
	ms(3) = mb0i(3)*abs(piDpj(4,4))
	ms(4) = 2*mc0*abs(del2s2)*DBLE(10)**mod(ier1-ier,50)
*
	cc1i(2) = 0
	mc1i(2) = 0
	xmax = 0
	do 20 i=1,4
	    cc1i(2) = cc1i(2) + cs(i)
	    xmax = max(xmax,absc(cs(i)))
	    mc1i(2) = max(mc1i(2),ms(i))
   20	continue
	if ( lwarn .and. absc(cc1i(2)) .lt. xloss*xmax ) then
	    call ffwarn(163,ier0,absc(cc1i(2)),xmax)
	    if ( lwrite ) then
		print *,'cs(i),ms(i) = '
		print '(i2,3g16.8)',(i,cs(i),ms(i),i=1,4)
		print '(a2,3g16.8)','+ ',cc1i(2),mc1i(2)
	    endif
	endif
	cc1i(2) = cc1i(2)*DBLE(1/(2*del2))
	mc1i(2) = mc1i(2)*abs(1/(2*del2))
	ier = max(ier0,ier1)
*
*  #] calculations:
*  #[ print output:
	if ( lwrite ) then
	    print *,'ffxc1: results:'
	    print *,'C11 = ',cc1i(1),mc1i(1),ier
	    print *,'C12 = ',cc1i(2),mc1i(2),ier
	endif
*  #] print output:
*###] ffxc1a:
	end
