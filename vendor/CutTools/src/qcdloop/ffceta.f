	subroutine ffceta(ceta,ipi,cpi,a,y,z,dyz,alpha,dha,ii,ier)
***#[*comment:***********************************************************
*									*
*	get the eta terms associated with the S_i as used in the	*
*	complex 4point function, see s.frm. EXPERIMENTAL.		*
*									*
*	Input:	cpi          complex	p_i^2 UNtransformed, hence real	*
*		a(3)	     complex	a(1)=A_{i+1}/(A_{i+1}-A_i),	*
*						a(3)=1-a(1)		*
*		z(4)	     complex	z roots				*
*		y(4)	     complex	y roots				*
*		dyz(2,2)     complex	y-z				*
*		alpha(3)     complex	alpha of shift (only when ii=2)	*
*		dha	     complex	h-a				*
*		ii	     integer	i=1,2,3 for S1,S2,S3		*
*					(h=1,alpha,0 for S1,S2,S3)	*
*									*
*	Output:	ceta	     complex	output				*
*		ipi	     integer	factors i*pi			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi,ier,ii
	DOUBLE COMPLEX ceta,cpi,a(3),z(4),y(4),dyz(2,2),
     +		dha,alpha(3)
*
*	local variables
*
	integer i,n,ns,ier0,ier1,n19a
	parameter(ns=21)
	DOUBLE PRECISION absc,xmax,xnul
	DOUBLE COMPLEX s(ns),c,zz,v(2:4),w(4),dvw(2:2,2),
     +		d1az(2),d1ay,daw(2),dav(2:2),dhw(2)
	integer nffeta,nffet1
	DOUBLE COMPLEX zfflog,zfflo1
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
	    print *,'ffceta: eta terms for S',ii
	    print *,'cpi= ',cpi
	    print *,'a  = ',a
	    print *,'y  = ',y
	    print *,'z  = ',z
	    print *,'dyz= ',dyz
	endif
	if ( ltest ) then
	    if ( ii .eq. 1 ) then
		if ( dha .ne. a(3) ) print *,'ffceta: error: dha!=1-a ',
     +			dha,a(3)
	    elseif ( ii .eq. 2 ) then
		xnul = absc(dha - alpha(1) + a(1))
		if ( xloss*abs(xnul) .gt. precc*max(absc(dha),
     +			absc(alpha(1)),absc(a(1))) ) print *,
     +			'ffceta: error: dha!=alpha-a ',dha,alpha(1),
     +			a(1),xnul
	    elseif ( ii .eq. 3 ) then
		if ( dha .ne. -a(1) ) print *,'ffceta: error: dha!=-a ',
     +			dha,a(1)
	    else
		print *,'ffceta: error: ii != 1,2,3: ',ii
	    endif
	endif
*  #] check input:
*  #[ get differences a and y,z:
*
	ier1 = 0
	ier0 = 0
	if ( absc(a(1)) .lt. absc(a(3)) ) then
	    d1ay = y(4) - a(1)
	    xmax = absc(a(1))
	else
	    d1ay = a(3) - y(2)
	    xmax = absc(a(3))
	endif
	if ( absc(d1ay) .lt. xloss*xmax ) then
	    call ffwarn(175,ier0,absc(d1ay),xmax)
	    if ( lwrite ) print *,'    a,y,1-a,1-y,1-a-y = ',
     +		a(1),y(2),a(3),y(4),d1ay
	    ier1 = max(ier1,ier0)
	endif
	do 2 i=1,2
	    ier0 = 0
	    if ( absc(a(1)) .lt. absc(a(3)) ) then
		d1az(i) = z(i+2) - a(1)
		xmax = absc(a(1))
	    else
		d1az(i) = a(3) - z(i)
		xmax = absc(a(3))
	    endif
	    if ( absc(d1az(i)) .lt. xloss*xmax ) then
		call ffwarn(176,ier0,absc(d1az(i)),xmax)
		if ( lwrite ) print *,'    a,z,1-a,1-z,1-a-z = ',
     +			a(1),z(i),a(3),z(i+2),d1az(i)
		ier1 = max(ier1,ier0)
	    endif
    2	continue
	ier = ier + ier1
*
*  #] get differences a and y,z:
*  #[ get untransformed roots:
*
	v(2) = -a(1)*y(2)/d1ay
	v(4) = +a(3)*y(4)/d1ay
	w(1) = -a(1)*z(1)/d1az(1)
	w(2) = -a(1)*z(2)/d1az(2)
	w(3) = +a(3)*z(3)/d1az(1)
	w(4) = +a(3)*z(4)/d1az(2)
	dvw(2,1) = -a(1)*a(3)*dyz(2,1)/(d1ay*d1az(1))
	dvw(2,2) = -a(1)*a(3)*dyz(2,2)/(d1ay*d1az(2))
	dav(2) = a(1)*a(3)/d1ay
	daw(1) = a(1)*a(3)/d1az(1)
	daw(2) = a(1)*a(3)/d1az(2)
*
	if ( ii .eq. 1 ) then
	    dhw(1) = w(3)
	    dhw(2) = w(4)
	elseif ( ii .eq. 2 ) then
	    if ( absc(alpha(1)) .lt. absc(alpha(3)) ) then
		dhw(1) = alpha(1) - w(1)
		dhw(2) = alpha(1) - w(2)
	    else
		dhw(1) = w(3) - alpha(3)
		dhw(2) = w(4) - alpha(3)
	    endif
	    xmax = min(absc(alpha(1)),absc(alpha(3)))
	    ier0 = 0
	    if ( absc(dhw(1)) .lt. xloss*xmax )
     +		call ffwarn(173,ier0,absc(dhw(1)),xmax)
	    ier1 = 0
	    if ( absc(dhw(2)) .lt. xloss*xmax )
     +		call ffwarn(174,ier1,absc(dhw(2)),xmax)
	    ier = ier + max(ier0,ier1)
	elseif ( ii .eq. 3 ) then
	    dhw(1) = w(1)
	    dhw(2) = w(2)
	else
	    print *,'ffceta: error: ii != 1,2,3 ',ii
	    stop
	endif
*
	if ( lwrite ) then
	    print *,'v   = ',v
	    print *,'w   = ',w
	    print *,'dvw = ',dvw
	    print *,'dav = ',dav
	    print *,'daw = ',daw
	endif
*  #] get untransformed roots:
*  #[ zero:
	ipi = 0
	do 10 i=1,ns
	    s(i) = 0
   10	continue
	ier1 = 0
*  #] zero:
*  #[ from form:

*   Scompl =

	if ( lwrite ) print *,'log number 1'
	ier0 = 0
	n =
     +  + nffeta( - w(1), - w(2),ier0)
     +  - nffeta(dvw(2,1),1/(dhw(1)),ier0)
     +  - nffeta(dvw(2,1),dvw(2,2),ier0)
     +  - nffeta(dvw(2,2),1/(dhw(2)),ier0)
     +  + 2*nffeta( - dav(2),1/(dha),ier0)
     +  - nffet1(DCMPLX(DBLE(cpi),-DBLE(x1)),dvw(2,1)*dvw(2,2),
     *	        	dvw(2,1)*dvw(2,2)*cpi,ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(v(2))*v(4),99,c0,ier0)
	    s(1) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 2'
	ier0 = 0
	n =
     +  + nffeta(1/(a(1))*w(1), - 1/(w(1))/(dav(2))*a(1)*dvw(2,1),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(a(1))*daw(1),99,c0,ier0)
	    s(2) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 3'
	ier0 = 0
	n =
     +  + nffeta(1/(a(1))*w(2), - 1/(w(2))/(dav(2))*a(1)*dvw(2,2),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(a(1))*daw(2),99,c0,ier0)
	    s(3) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 4'
	ier0 = 0
	n =
     +  - nffeta( - w(1), - w(2),ier0)
     +  - nffeta(a(3),1/(daw(1)),ier0)
     +  - nffeta(a(3),1/(daw(2)),ier0)
     +  - nffeta(dvw(2,1), - 1/(dav(2)),ier0)
     +  + nffeta(dvw(2,1),1/(dhw(1)),ier0)
     +  + nffeta(dvw(2,1),dvw(2,2),ier0)
     +  - nffeta(dvw(2,2), - 1/(dav(2)),ier0)
     +  + nffeta(dvw(2,2),1/(dhw(2)),ier0)
     +  - 2*nffeta( - dav(2),1/(dha),ier0)
     +  + nffet1(DCMPLX(DBLE(cpi),-DBLE(x1)),dvw(2,1)*dvw(2,2),
     *		dvw(2,1)*dvw(2,2)*cpi,ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(a(1))*a(3),99,c0,ier0)
	    s(4) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 5'
	ier0 = 0
	n =
     +  - nffeta(1/(a(3))*w(3), - 1/(w(3))/(dav(2))*a(3)*dvw(2,1),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(-1/(a(3))*daw(1),99,c0,ier0)
	    s(5) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 6'
	ier0 = 0
	n =
     +  - nffeta(1/(a(3))*w(4), - 1/(w(4))/(dav(2))*a(3)*dvw(2,2),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(-1/(a(3))*daw(2),99,c0,ier0)
	    s(6) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 7'
	ier0 = 0
	n =
     +  + nffeta( - 1/(dvw(2,1))*w(1), - 1/(w(1))/(dav(2))*a(1)*dvw(2,
     *		1),ier0)
     +  + nffeta( - w(1),1/(dvw(2,1)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(dvw(2,1))*v(2),99,c0,ier0)
	    s(7) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 8'
	ier0 = 0
	n =
     +  - nffeta(1/(dvw(2,1))*w(3), - 1/(w(3))/(dav(2))*a(3)*dvw(2,1),
     *		ier0)
     +  - nffeta(w(3),1/(dvw(2,1)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(dvw(2,1))*v(4),99,c0,ier0)
	    s(8) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 9'
	ier0 = 0
	n =
     +  + nffeta( - 1/(dvw(2,2))*w(2), - 1/(w(2))/(dav(2))*a(1)*dvw(2,
     *		2),ier0)
     +  + nffeta( - w(2),1/(dvw(2,2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(dvw(2,2))*v(2),99,c0,ier0)
	    s(9) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 10'
	ier0 = 0
	n =
     +  - nffeta(1/(dvw(2,2))*w(4), - 1/(w(4))/(dav(2))*a(3)*dvw(2,2),
     *		ier0)
     +  - nffeta(w(4),1/(dvw(2,2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(dvw(2,2))*v(4),99,c0,ier0)
	    s(10) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 11'
	ier0 = 0
	n =
     +  - 2*nffeta( - a(1), - 1/(dav(2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(dav(2))*v(2),99,c0,ier0)
	    s(11) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 12'
	ier0 = 0
	n =
     +  + 2*nffeta(a(3), - 1/(dav(2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(dav(2))*v(4),99,c0,ier0)
	    s(12) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 13'
	ier0 = 0
	n =
     +  - nffeta( - 1/(a(1))*a(3),1/(dav(2))*a(1),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(dav(2))*dvw(2,1),99,c0,ier0)
	    s(13) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 14'
	ier0 = 0
	n =
     +  - nffeta( - 1/(a(1))*a(3),1/(dav(2))*a(1),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(dav(2))*dvw(2,2),99,c0,ier0)
	    s(14) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 15'
	ier0 = 0
	n =
     +  - nffeta( - w(1),1/(daw(1)),ier0)
	if ( n .ne. 0 ) then
	    c = -w(1)/daw(1)
	    if ( absc(c) .lt. xloss ) then
		zz = zfflo1(c,ier0)
	    else
		zz = zfflog(1/(daw(1))*a(1),99,c0,ier0)
	    endif
	    s(15) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 16'
	ier0 = 0
	n =
     +  + nffeta(w(3),1/(daw(1)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(daw(1))*a(3),99,c0,ier0)
	    s(16) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 17'
	ier0 = 0
	n =
     +  - nffeta( - w(2),1/(daw(2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(1/(daw(2))*a(1),99,c0,ier0)
	    s(17) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 18'
	ier0 = 0
	n =
     +  + nffeta(w(4),1/(daw(2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog( - 1/(daw(2))*a(3),99,c0,ier0)
	    s(18) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 19'
	ier0 = 0
	n =
     +  + nffeta( - a(1),1/(daw(1)),ier0)
     +  + nffeta( - a(1),1/(daw(2)),ier0)
     +  - nffeta(a(3),1/(daw(1)),ier0)
     +  - nffeta(a(3),1/(daw(2)),ier0)
	if ( n .ne. 0 ) then
	    if ( DBLE(a(1)) .lt. 0 ) then
	    	zz = zfflog( - a(1),99,c0,ier0)
	    else
	    	zz = zfflog(a(1),99,c0,ier0)
	    	if ( DIMAG(a(1)) .gt. 0 ) then
	    	    ipi = ipi - n
	    	elseif ( DIMAG(a(1)) .lt. 0 ) then
	    	    ipi = ipi + n
	    	endif
	    endif
	    s(19) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 20'
	ier0 = 0
	n =
     +  - nffeta( - a(1),1/(daw(1)),ier0)
     +  + nffeta(a(3),1/(daw(1)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(daw(1),99,c0,ier0)
	    s(20) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 21'
	ier0 = 0
	n =
     +  - nffeta( - a(1),1/(daw(2)),ier0)
     +  + nffeta(a(3),1/(daw(2)),ier0)
	if ( n .ne. 0 ) then
	    zz = zfflog(daw(2),99,c0,ier0)
	    s(21) = n*zz
	endif
	ier1 = max(ier1,ier0)

	if ( lwrite ) print *,'log number 22'
	ier0 = 0
	n =
     +  + nffeta( - a(1),1/(daw(1)),ier0)**2
     +  + nffeta( - a(1),1/(daw(2)),ier0)**2
     +  - nffeta(a(3),1/(daw(1)),ier0)**2
     +  - nffeta(a(3),1/(daw(2)),ier0)**2
	if ( n .ne. 0 ) then
	    ipi = ipi + n
	endif
	ier1 = max(ier1,ier0)

*  #[ from form:
*  #[ add:
	ceta = 0
	xmax = 0
	do 20 i=1,ns
	    ceta = ceta + s(i)
	    xmax = max(xmax,absc(s(i)))
   20	continue
	ier = ier + ier1
	if ( absc(ceta) .lt. xloss*xmax ) then
	    call ffwarn(172,ier,absc(ceta),xmax)
	endif
*  #] add:
*  #[ debug:
	if ( lwrite ) then
	    print *,'ffceta: eta terms for complex 4point function'
	    do 900 i=1,ns
		print '(i2,2g18.6)',i,s(i)
  900	    continue
	    if ( ipi .ne. 0 ) print '(a,2g18.6)','pi',ipi*c2ipi/2
	    print *,'---------------- +'
	    print '(2x,2g18.6,i4,g18.6)',ceta,ipi,xmax
	    if ( ipi .ne. 0 ) print '(a,3g18.6)','= ',ceta+ipi*c2ipi/2
	endif
*  #] debug:
	end
