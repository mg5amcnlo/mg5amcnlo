*###[ ffxd1:
	subroutine ffxd1(cd1i,cd0,cc0i,xpi,piDpj,del3,del2i,ier)
***#[*comment:***********************************************************
*									*
*	calculate the D1(mu) = D11*p1(mu) + D12*p2(mu) + D13*p3(mu)	*
*	numerically							*
*									*
*	Input:	cd0	     complex	scalar fourpoint function	*
*		cc0i(4)	     complex	scalar threepoint functions	*
*						without s1,s2,s3,s4	*
*		xpi(13)	     real	masses (1-4), momenta^2 (5-10)	*
*		piDpj(10,10) real	dotproducts as in D0		*
*		del3	     real	overall determinant		*
*		del2i(4)     real	minors as in cc0i		*
*		ier	     integer	digits lost so far		*
*	Output:	cd1i(3)	     complex	D11,D12,D13			*
*		ier	     integer	number of dgits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xpi(13),piDpj(10,10),del3,del2i(4)
	DOUBLE COMPLEX cd1i(3),cd0,cc0i(4)
*
*	local variables
*
	DOUBLE PRECISION md1i(3),md0,mc0i(4)
	integer i,j,ier0
	logical wasnul(3)
	DOUBLE PRECISION xmax,absc,xnul,xlosn
	DOUBLE PRECISION dpipj(10,13),piDpjp(10,10),s(6),som
	DOUBLE COMPLEX cc
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
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffxd1: input:'
	    print *,'xpi  = ',xpi
	    print *,'del3 = ',del3
	endif
	if ( ltest ) then
	    xlosn = xloss*DBLE(10)**(-mod(ier,50))
	    do 1 i=1,6
		if ( xpi(i) .ne. piDpj(i,i) ) then
		    print *,'ffxd1: error: xpi and piDpj do not agree'
		endif
    1	    continue
	    if ( xpi(11).eq.0 ) then
		xpi(11) = xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
		wasnul(1) = .TRUE.
	    else
		wasnul(1) = .FALSE.
	    endif
	    if ( xpi(12).eq.0 ) then
		xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
		wasnul(2) = .TRUE.
	    else
		wasnul(2) = .FALSE.
	    endif
	    if ( xpi(13).eq.0 ) then
		xpi(13) = xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
		wasnul(3) = .TRUE.
	    else
		wasnul(3) = .FALSE.
	    endif
	    do 4 i=1,13
		do 3 j=1,10
		    dpipj(j,i) = xpi(j) - xpi(i)
    3		continue
    4	    continue
	    ier0 = ier
	    call ffdot4(piDpjp,xpi,dpipj,10,ier0)
	    if ( wasnul(1) ) xpi(11) = 0
	    if ( wasnul(2) ) xpi(12) = 0
	    if ( wasnul(3) ) xpi(13) = 0
	    do 7 i=1,10
		do 6 j=1,10
		    xnul = piDpj(j,i) - piDpjp(j,i)
		    if ( xlosn*abs(xnul) .gt. precx*abs(piDpjp(j,i)) )
     +			print *,'piDpj(',j,i,') not correct, cmp:',
     +			piDpj(j,i),piDpjp(j,i),xnul
    6		continue
    7	    continue
	    s(1) = + piDpj(5,5)*piDpj(6,6)*piDpj(7,7)
	    s(2) = - piDpj(5,5)*piDpj(6,7)*piDpj(7,6)
	    s(3) = - piDpj(5,6)*piDpj(6,5)*piDpj(7,7)
	    s(4) = + piDpj(5,6)*piDpj(6,7)*piDpj(7,5)
	    s(5) = + piDpj(5,7)*piDpj(6,5)*piDpj(7,6)
	    s(6) = - piDpj(5,7)*piDpj(6,6)*piDpj(7,5)
	    som = s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
	    xmax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +		abs(s(5)),abs(s(6)))
	    xnul = del3-som
	    if ( xloss*abs(xnul) .gt. precx*xmax ) print *,
     +		'ffxd1: error: del3 is not correct',del3,som,xmax
	endif
*  #] check input:
*  #[ call ffxd1a:
*
	md0 = absc(cd0)*DBLE(10)**mod(ier,50)
	mc0i(1) = absc(cc0i(1))*DBLE(10)**mod(ier,50)
	mc0i(2) = absc(cc0i(2))*DBLE(10)**mod(ier,50)
	mc0i(3) = absc(cc0i(3))*DBLE(10)**mod(ier,50)
	mc0i(4) = absc(cc0i(4))*DBLE(10)**mod(ier,50)
	call ffxd1a(cd1i,md1i,cd0,md0,cc0i,mc0i,xpi,piDpj,del3,del2i,
     +		ier)
*
*  #] call ffxd1a:
*###] ffxd1:
	end
*###[ ffxd1a:
	subroutine ffxd1a(cd1i,md1i,cd0,md0,cc0i,mc0i,xpi,piDpj,del3,
     +		del2i,ier)
***#[*comment:***********************************************************
*									*
*	calculate the D1(mu) = D11*p1(mu) + D12*p2(mu) + D13*p3(mu)	*
*	numerically							*
*									*
*	Input:	cd0	     complex	scalar fourpoint function	*
*		md0	     real	maximum partial sum in D0	*
*		cc0i(4)	     complex	scalar threepoint functions	*
*						without s1,s2,s3,s4	*
*		mc0i(4)	     real	maximum partial sum in C0i	*
*		xpi(13)	     real	masses (1-4), momenta^2 (5-10)	*
*		piDpj(10,10) real	dotproducts as in D0		*
*		del3	     real	overall determinant		*
*		del2i(4)     real	minors as in cc0i		*
*		ier	     integer	digits lost so far		*
*	Output:	cd1i(3)	     complex	D11,D12,D13			*
*		md1i(3)	     real	maximum partial sum in D1i	*
*		ier	     integer	number of dgits lost		*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xpi(13),piDpj(10,10),del3,del2i(4)
	DOUBLE PRECISION md1i(3),md0,mc0i(4)
	DOUBLE COMPLEX cd1i(3),cd0,cc0i(4)
*
*	local variables
*
	integer i,ier0,ier1,ier2
	DOUBLE PRECISION xmax,absc,del2,del2sa,dl3q,ms(5),mdelsa
	DOUBLE COMPLEX cs(5),cc
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations: 
*  #[ Form-ula:
*	see the Form job D1.frm
*   D1 =
*       + p1(mu)*Del3^-1 * (  - 1/2*C(s1)*p2.p2*p3.p3 + 1/2*C(s1)*p2.p3^2 + 1/2
*         *C(s2)*p2.p3*p3.p4 - 1/2*C(s2)*p2.p4*p3.p3 + 1/2*C(s3)*p1.p2*p3.p4 -
*         1/2*C(s3)*p1.p3*p2.p4 + 1/2*C(s4)*p1.p2*p2.p3 - 1/2*C(s4)*p1.p3*p2.p2
*          + D*delta(s1,p2,p3,p1,p2,p3) - D*delta(s1,p3,p2,p1,p2,p3) )
*
*       + p2(mu)*Del3^-1 * ( 1/2*C(s1)*p1.p2*p3.p3 - 1/2*C(s1)*p1.p3*p2.p3 - 1/
*         2*C(s2)*p1.p3*p3.p4 + 1/2*C(s2)*p1.p4*p3.p3 - 1/2*C(s3)*p1.p1*p3.p4
*          + 1/2*C(s3)*p1.p3*p1.p4 - 1/2*C(s4)*p1.p1*p2.p3 + 1/2*C(s4)*p1.p2*
*         p1.p3 - D*delta(s1,p1,p3,p1,p2,p3) + D*delta(s1,p3,p1,p1,p2,p3) )
*
*       + p3(mu)*Del3^-1 * (  - 1/2*C(s1)*p1.p2*p2.p3 + 1/2*C(s1)*p1.p3*p2.p2
*          + 1/2*C(s2)*p1.p3*p2.p4 - 1/2*C(s2)*p1.p4*p2.p3 + 1/2*C(s3)*p1.p1*
*         p2.p4 - 1/2*C(s3)*p1.p2*p1.p4 + 1/2*C(s4)*p1.p1*p2.p2 - 1/2*C(s4)*
*         p1.p2^2 + D*delta(s1,p1,p2,p1,p2,p3) - D*delta(s1,p2,p1,p1,p2,p3) );
*
*  #] Form-ula:
*  #[ D11:
	if ( lwrite ) print *,'ffxd1: D11'
	cs(1) = - cc0i(1)*DBLE(del2i(1))
	ms(1) = mc0i(1)*abs(del2i(1))
	if ( lwrite ) print *,'ffdl2i 1'
	ier1 = ier
	call ffdl2i(del2,piDpj,10, 6,7,10,+1,7,8,9,+1,ier1)
	cs(2) = + cc0i(2)*DBLE(del2)
	ms(2) = mc0i(2)*abs(del2)*DBLE(10)**mod(ier1-ier,50)
	if ( lwrite ) print *,'ffdl2i 2'
	ier0 = ier
	call ffdl2i(del2,piDpj,10, 6,7,10,+1,8,5,10,-1,ier0)
	ier1 = max(ier1,ier0)
	cs(3) = - cc0i(3)*DBLE(del2)
	ms(3) = mc0i(3)*abs(del2)*DBLE(10)**mod(ier0-ier,50)
	if ( lwrite ) print *,'ffdl2i 3'
	ier0 = ier
	call ffdl2i(del2sa,piDpj,10, 6,7,10,+1,5,6,9,-1,ier0)
	ier1 = max(ier1,ier0)
	cs(4) = + cc0i(4)*DBLE(del2sa)
	mdelsa = abs(del2sa)*DBLE(10)**mod(ier0-ier,50)
	ms(4) = mc0i(4)*mdelsa
	ier0 = ier
	call ffdl3q(dl3q,piDpj, 1,6,7, 0,10,0, 0,-1,0, 0,+1,0, ier0)
	ier1 = max(ier1,ier0)
	cs(5) = + 2*cd0*DBLE(dl3q)
	ms(5) = 2*md0*abs(dl3q)*DBLE(10)**mod(ier0-ier,50)

	cd1i(1) = 0
	xmax = 0
	md1i(1) = 0
	do 10 i=1,5
	    cd1i(1) = cd1i(1) + cs(i)
	    xmax = max(xmax,absc(cs(i)))
	    md1i(1) = max(md1i(1),ms(i))
   10	continue
	if ( lwarn .and. absc(cd1i(1)) .lt. xloss*xmax ) then
	    call ffwarn(164,ier1,absc(cd1i(1)),xmax)
	    if ( awrite .or. lwrite ) then
		print *,'cs  = ',cs
		print *,'D11 = ',cd1i(1),xmax
		print *,'ms  = ',ms
	    endif
	endif
	cd1i(1) = cd1i(1)*DBLE(1/(2*del3))
	md1i(1) = md1i(1)*abs(1/(2*del3))
	ier2 = ier1
*
*  #] D11:
*  #[ D12:
*
	if ( lwrite ) print *,'ffxd1: D12'
	ier1 = ier
	call ffdl2t(del2,piDpj,7,5, 6,7,10,-1,-1, 10,ier1)
	cs(1) = - cc0i(1)*DBLE(del2)
	ms(1) = mc0i(1)*abs(del2)*DBLE(10)**mod(ier-ier1,50)
	ier0 = ier
	call ffdl2t(del2,piDpj,7,5, 7,8,9,-1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	cs(2) = + cc0i(2)*DBLE(del2)
	ms(2) = mc0i(2)*abs(del2)*DBLE(10)**mod(ier-ier0,50)
	ier0 = ier
	call ffdl2t(del2,piDpj,7,5, 8,5,10,+1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	cs(3) = - cc0i(3)*DBLE(del2)
	ms(3) = mc0i(3)*abs(del2)*DBLE(10)**mod(ier-ier0,50)
	ier0 = ier
	call ffdl2t(del2,piDpj,7,5, 5,6,9,+1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	cs(4) = + cc0i(4)*DBLE(del2)
	ms(4) = mc0i(4)*abs(del2)*DBLE(10)**mod(ier-ier0,50)
	ier0 = ier
	call ffdl3q(dl3q,piDpj, 1,7,5, 0,0,2, 0,0,-1, 0,0,+1, ier0)
	ier1 = max(ier1,ier0)
	cs(5) = + 2*cd0*DBLE(dl3q)
	ms(5) = 2*md0*abs(dl3q)*DBLE(10)**mod(ier-ier0,50)

	cd1i(2) = 0
	xmax = 0
	md1i(2) = 0
	do 20 i=1,5
	    cd1i(2) = cd1i(2) + cs(i)
	    xmax = max(xmax,absc(cs(i)))
	    md1i(2) = max(md1i(2),ms(i))
   20	continue
	if ( lwarn .and. absc(cd1i(2)) .lt. xloss*xmax ) then
	    call ffwarn(164,ier1,absc(cd1i(2)),xmax)
	    if ( lwrite .or. awrite ) then
		print *,'cs  = ',cs
		print *,'D12 = ',cd1i(2),xmax
		print *,'ms  = ',ms
	    endif
	endif
	cd1i(2) = cd1i(2)*DBLE(1/(2*del3))
	md1i(2) = md1i(2)*abs(1/(2*del3))
	ier2 = max(ier2,ier1)
*
*  #] D12:
*  #[ D13:
*
	if ( lwrite ) print *,'ffxd1: D13'
	cs(1) = - cc0i(1)*DBLE(del2sa)
	ms(1) = mc0i(1)*mdelsa
	if ( lwrite ) print *,'ffdl2i 1'
	ier1 = ier
	call ffdl2i(del2,piDpj,10, 5,6,9,-1,7,8,9,+1,ier1)
	cs(2) = + cc0i(2)*DBLE(del2)
	ms(2) = mc0i(2)*abs(del2)*DBLE(10)**mod(ier-ier1,50)
	if ( lwrite ) print *,'ffdl2i 2'
	ier0 = ier
	call ffdl2i(del2,piDpj,10, 5,6,9,-1,8,5,10,-1,ier0)
	ier1 = max(ier1,ier0)
	cs(3) = - cc0i(3)*DBLE(del2)
	ms(3) = mc0i(3)*abs(del2)*DBLE(10)**mod(ier-ier0,50)
	cs(4) = + cc0i(4)*DBLE(del2i(4))
	ms(4) = mc0i(4)*abs(del2i(4))
	ier0 = ier
	call ffdl3q(dl3q,piDpj, 1,5,6, 2,9,0, -1,-1,0, +1,-1,0, ier0)
	ier1 = max(ier1,ier0)
	cs(5) = + 2*cd0*DBLE(dl3q)
	ms(5) = 2*md0*abs(dl3q)*DBLE(10)**mod(ier-ier0,50)

	cd1i(3) = 0
	xmax = 0
	md1i(3) = 0
	do 30 i=1,5
	    cd1i(3) = cd1i(3) + cs(i)
	    xmax = max(xmax,absc(cs(i)))
	    md1i(3) = max(md1i(3),ms(i))
   30	continue
	if ( lwarn .and. absc(cd1i(3)) .lt. xloss*xmax ) then
	    call ffwarn(164,ier1,absc(cd1i(3)),xmax)
	    if ( lwrite .or. awrite ) then
		print *,'cs  = ',cs
		print *,'D13 = ',cd1i(3),xmax
		print *,'ms  = ',ms
	    endif
	endif
	cd1i(3) = cd1i(3)*DBLE(1/(2*del3))
	md1i(3) = md1i(3)*abs(1/(2*del3))
	ier2 = max(ier2,ier1)
*
*	fidel3 is the error on del3, but only when del3=fdel3
*
	if ( fdel3.eq.del3 ) then
	    ier2 = max(ier2,fidel3)
	    do 40 i=1,3
	    	md1i(i) = md1i(i)*DBLE(10**mod(fidel3,50))
   40	    continue
	endif
	ier = ier2
*
*  #] D13:
*  #[ print output:
	if ( lwrite ) then
	    print *,'ffxd1: results:'
	    print *,'D11 = ',cd1i(1),md1i(1),ier
	    print *,'D12 = ',cd1i(2),md1i(2),ier
	    print *,'D13 = ',cd1i(3),md1i(3),ier
	endif
*  #] print output:
*###] ffxd1:
	end
