*###[ ffcxs3:
	subroutine ffcxs3(cs3,ipi12,y,z,dyz,d2yzz,dy2z,xpi,piDpj,ii,ns,
     +					isoort,ier)
***#[*comment:***********************************************************
*									*
*	calculates the s3 as defined in appendix b.			*
*		(ip = ii+3, is1 = ii, is2 = ii+1)			*
*									*
*		  log( xk*y^2 + (-xk+xm1-xm2)*y + xm2 - i*eps ) 	*
*	     /1 				  - log( ... ) |y=yi	*
*	s3 = \ dy --------------------------------------------------	*
*	     /0 			y - yi				*
*									*
*	    = r(yi,y-,+) + r(yi,y+,-)					*
*									*
*	with y+- the roots of the argument of the logarithm.		*
*	the sign of the argument to the logarithms in r is passed	*
*	in ieps 							*
*									*
*	input:	y(4),z(4)	(real)	roots in form (z-,z+,1-z-,1-z+)	*
*		dyz(2,2),d2yzz,	(real)	y() - z(), y+ - z- - z+		*
*		dy2z(4)		(real)	y() - 2z()			*
*		xpi	(real(ns))	p(i).p(i) (B&D metric)	i=1,3	*
*					m(i)^2 = si.si		i=4,6	*
*		ii	(integer)	xk = xpi(ii+3) etc		*
*		ns	(integer)	size of arrays			*
*		isoort	(integer)	returns kind of action taken	*
*		cs3	(complex)(20)	assumed zero.			*
*		ccy	(complex)(3)	if i0 != 0: complex y		*
*									*
*	output: cs3	(complex)	mod factors pi^2/12, in array	*
*		ipi12	(integer)	these factors			*
*		ier	(integer)	0=ok 1=inaccurate 2=error	*
*									*
*	calls:	ffcrr,ffcxr,real/dble,DCMPLX,log,ffadd1,ffadd2,ffadd3	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(2),ii,ns,isoort(2),ier
	DOUBLE COMPLEX cs3(20)
	DOUBLE PRECISION y(4),z(4),dyz(2,2),d2yzz,dy2z(4),
     +		xpi(ns),piDpj(ns,ns)
*
*	local variables:
*
	integer i,ip,ieps(2),ipi12p(2),ier0,i2,i3
	DOUBLE COMPLEX c,csum,cs3p(14)
	DOUBLE PRECISION yy,yy1,zz,zz1,dyyzz,xdilog,xlog,x00(3)
	DOUBLE PRECISION absc,xmax
	logical ld2yzz
*
*	common blocks
*
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*  #] declarations:
*  #[ get counters:
	if ( ltest .and. ns .ne. 6 )
     +	    print *,'ffcxs3: error: only for ns=6, not ',ns
	ip = ii+3
	if ( isoort(2) .ne. 0 ) then
	    if ( (z(2).gt.z(1) .or.  z(1).eq.z(2) .and. z(4).lt.z(3) )
     +			.eqv. xpi(ip) .gt. 0 ) then
		ieps(1) = +1
		ieps(2) = -1
	    else
		ieps(1) = -1
		ieps(2) = +1
	    endif
	else
	    if ( piDpj(ip,ii) .gt. 0 ) then
		ieps(1) = +1
	    else
		ieps(1) = -1
	    endif
	endif
	i2 = mod(ii,3) + 1
	i3 = mod(i2,3) + 1
*  #] get counters:
*  #[ special case |z| >> |y|:
	if ( xpi(ip).lt.0 .and. max(abs(y(2)),abs(y(4))) .lt.
     +		xloss*min(abs(z(1)), abs(z(2)))/2 ) then
*
*	    we will obtain cancellations of the type Li_2(x) + Li_2(-x)
*	    with x small.
*
	    if ( lwrite ) then
		print *,'ffcxs3: special case |z| >> |y|'
		print *,'  y,y1  = ',y(2),y(4)
		print *,'  z,z1- = ',z(1),z(3)
		print *,'  z,z1+ = ',z(2),z(4)
	    endif
	    yy = dyz(2,1)/d2yzz
	    yy1 = dyz(2,2)/d2yzz
	    if ( y(2) .eq. 0 ) goto 10
	    zz = z(2)*yy/y(2)
	    zz1 = 1-zz
	    if ( lwarn .and. abs(zz) .lt. xloss )
     +		call ffwarn(44,ier,abs(zz),x1)
	    dyyzz = dyz(2,2)*yy/y(2)
	    call ffcxr(cs3(1),ipi12(1),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,0,ier)
   10	    continue
	    if ( y(4) .eq. 0 ) goto 30
	    zz = yy*z(4)/y(4)
	    zz1 = 1-zz
	    if ( lwarn .and. abs(zz) .lt. xloss )
     +		call ffwarn(44,ier,abs(zz),x1)
	    dyyzz = -yy*dyz(2,2)/y(4)
	    call ffcxr(cs3(8),ipi12(2),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,0,ier)
	    do 20 i=8,14
   20		cs3(i) = -cs3(i)
   30	    continue
*	    And now the remaining Li_2(x^2) terms
	    call ffxli2(xdilog,xlog,(y(2)/dyz(2,1))**2,ier)
	    cs3(15) = +xdilog/2
	    call ffxli2(xdilog,xlog,(y(4)/dyz(2,1))**2,ier)
	    cs3(16) = -xdilog/2
	    if ( lwrite ) then
		lwrite = .FALSE.
		ipi12p(1) = 0
		ipi12p(2) = 0
		ier0 = 0
		do 40 i=1,14
   40		    cs3p(i) = 0
		call ffcxr(cs3p(1),ipi12p(1),y(2),y(4),z(1),z(3),
     +		    dyz(2,1),.FALSE.,x0,x0,x0,.FALSE.,x00,ieps(1),ier0)
		call ffcxr(cs3p(8),ipi12p(2),y(2),y(4),z(2),z(4),
     +		    dyz(2,2),.FALSE.,x0,x0,x0,.FALSE.,x00,ieps(2),ier0)
		csum = 0
		xmax = 0
		do 50 i=1,14
		    csum = csum + cs3p(i)
		    xmax = max(xmax,absc(csum))
   50		continue
		csum = csum + (ipi12p(1)+ipi12(2))*DBLE(pi12)
		print '(a,3g20.10,3i3)','cmp',csum,xmax,ipi12p,ier0
		lwrite = .TRUE.
	    endif
	    goto 900
	endif
*  #] special case |z| >> |y|:
*  #[ normal:
	if ( xpi(ip) .eq. 0 ) then
	    ld2yzz = .FALSE.
	else
	    ld2yzz = .TRUE.
	endif
	if ( lwrite ) print *,  'ieps = ',ieps
	if ( isoort(1) .ne. 0 ) call ffcxr(cs3(1),ipi12(1),y(2),y(4),
     +	    z(1),z(3),dyz(2,1),ld2yzz,d2yzz,z(2),z(4),.TRUE.,dy2z(1),
     +	    ieps(1),ier)
	if ( isoort(2) .ne. 0 ) then
	    if ( mod(isoort(2),10) .eq. 2 ) then
*		both roots are equal: multiply by 2
		if ( lwrite ) print *,'ffcxs3: skipped next R as it ',
     +			'is the conjugate'
		do 60 i=1,7
		    cs3(i) = 2*DBLE(cs3(i))
   60		continue
		ipi12(1) = 2*ipi12(1)
	    else
		call ffcxr(cs3(8),ipi12(2),y(2),y(4),z(2),z(4),dyz(2,2),
     +		    ld2yzz,d2yzz,z(1),z(3),.TRUE.,dy2z(2),ieps(2),ier)
	    endif
	endif
*
*  #] normal:
*  #[ print output:
  900	if (lwrite) then
	    print *,'  cs3 ='
	    do 905 i=1,20
	    	if ( cs3(i).ne.0 ) print '(i3,2g20.10,1x)',i,cs3(i)
  905	    continue
	    print '(a3,2g20.10,1x)','pi ',(ipi12(1)+ipi12(2))*pi12
	    print *,'+-----------'
	    csum = 0
	    do 910 i=1,20
  910	    csum = csum + cs3(i)
	    csum = csum+(ipi12(1)+ipi12(2))*DBLE(pi12)
	    print '(a,2g20.10)','Si ',csum
	    print *,'  ipi12,ier=',ipi12,ier
	    print *,' '
	endif
*  #] print output:
*###] ffcxs3:
	end
*###[ ffcs3:
	subroutine ffcs3(cs3,ipi12,cy,cz,cdyz,cd2yzz,cpi,cpiDpj,ii,ns,
     +		isoort,ier)
***#[*comment:***********************************************************
*									*
*	calculates the s3 as defined in appendix b.			*
*									*
*		  log( cpi(ii+3)*y^2 + (cpi(ii+3)+cpi(ii)-cpi(ii+1))*y	*
*	     /1 		     +	cpi(ii+1))  - log( ... ) |y=cyi	*
*	s3 = \ dy ----------------------------------------------------	*
*	     /0 			y - cyi				*
*									*
*	   = r(cyi,cy+) + r(cyi,cy-) +  ( eta(-cy-,-cy+) -		*
*		eta(1-cy-,1-cy+) - eta(...) )*log(1-1/cyi)		*
*									*
*	with y+- the roots of the argument of the logarithm.		*
*									*
*	input:	cy(4)	 (complex)  cy(1)=y^-,cy(2)=y^+,cy(i+2)=1-cy(1)	*
*		cz(4)	 (complex)  cz(1)=z^-,cz(2)=z^+,cz(i+2)=1-cz(1)	*
*		cpi(6)   (complex)  masses & momenta (B&D)		*
*		ii	 (integer)  position of cp,cma,cmb in cpi	*
*		ns	 (integer)  size of arrays			*
*		isoort(2)(integer)  returns the kind of action taken	*
*		cs3	 (complex)(14)	assumed zero.			*
*									*
*	output: cs3	 (complex)  mod factors ipi12			*
*		ipi12(2) (integer)  these factors			*
*		ier	 (integer)  0=ok, 1=numerical problems, 2=error	*
*									*
*	calls:	ffcrr,DIMAG,DBLE,zfflog					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(2),ii,ns,isoort(2),ier
	DOUBLE COMPLEX cs3(20),cpi(ns),cpiDpj(ns,ns)
	DOUBLE COMPLEX cy(4),cz(4),cdyz(2,2),cd2yzz
*
*	local variables:
*
	integer i,ip,ieps(2),ieps0,ni(4),ipi12p(2),ier0,ntot,i2,i3
	logical ld2yzz
	DOUBLE COMPLEX c,csum,zdilog,zlog,cyy,cyy1,czz,czz1,cdyyzz
     +		,cs3p(14)
	DOUBLE PRECISION absc,xmax,y,y1,z,z1,dyz,d2yzz,zz,zz1,
     +		x00(3),sprec
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ get ieps:
	if ( ltest ) then
	    if ( ns .ne. 6 ) then
		print *,'ffcs3: error: only for ns=6, not ',ns
		stop
	    endif
	endif
	ip = ii+3
	call ffieps(ieps,cz(1),cpi(ip),cpiDpj(ip,ii),isoort)
	i2 = mod(ii,3) + 1
	i3 = mod(i2,3) + 1
*  #] get ieps:
*  #[ special case |cz| >> |cy|:
	if ( isoort(2) .ne. 0 .and. max(absc(cy(2)),absc(cy(4))) .lt.
     +		xloss*min(absc(cz(1)),absc(cz(2)))/2 ) then
*
*	    we will obtain cancellations of the type Li_2(x) + Li_2(-x)
*	    with x small.
*
	    if ( lwrite ) print *,'Special case |cz| >> |cy|'
	    cyy = cdyz(2,1)/cd2yzz
	    cyy1 = cdyz(2,2)/cd2yzz
	    if ( absc(cy(2)) .lt. xclogm ) then
		if ( DIMAG(cy(2)) .eq. 0 .and. abs(DBLE(cy(2))) .gt.
     +			xalogm ) then
		    czz = cz(2)*cyy*DCMPLX(1/DBLE(cy(2)))
		    cdyyzz = cyy*cdyz(2,2)*DCMPLX(1/DBLE(cy(2)))
		elseif ( cy(2) .eq. 0 .and. cz(2) .ne. 0 .and. cyy
     +			.ne. 0 ) then
*		    the answer IS zero
		    goto 30
		else
*		    the answer is rounded off to zero
		    if (lwarn) call ffwarn(42,ier,absc(cy(2)),xclogm)
		endif
	    else
		czz = cz(2)*cyy/cy(2)
		cdyyzz = cyy*cdyz(2,2)/cy(2)
	    endif
	    czz1 = 1-czz
	    if ( lwarn .and. absc(czz) .lt. xloss )
     +		call ffwarn(43,ier,absc(czz),x1)
	    if ( isoort(1) .eq. -10 ) then
*		no eta terms.
		ieps0 = 99
	    else
*		do not know the im part
		ieps0 = 0
	    endif
	    call ffcrr(cs3(1),ipi12(1),cyy,cyy1,czz,czz1,cdyyzz,.FALSE.,
     +		c0,c0,c0,-1,ieps0,ier)
   30	    continue
	    if ( absc(cy(4)) .lt. xclogm ) then
		if ( DIMAG(cy(4)) .eq. 0 .and. abs(DBLE(cy(4))) .gt.
     +			xalogm ) then
		    czz = cz(4)*cyy*DCMPLX(1/DBLE(cy(4)))
		    cdyyzz = -cyy*cdyz(2,2)*DCMPLX(1/DBLE(cy(4)))
		elseif ( cy(4) .eq. 0 .and. cz(4) .ne. 0 .and. cyy
     +			.ne. 0 ) then
*		    the answer IS zero
		    goto 50
		else
*		    the answer is rounded off to zero
		    if (lwarn) call ffwarn(42,ier,absc(cy(4)),xclogm)
		endif
	    else
		czz = cz(4)*cyy/cy(4)
		cdyyzz = -cyy*cdyz(2,2)/cy(4)
	    endif
	    czz1 = 1-czz
	    if ( lwarn .and. absc(czz) .lt. xloss )
     +		call ffwarn(43,ier,absc(czz),x1)
	    call ffcrr(cs3(8),ipi12(2),cyy,cyy1,czz,czz1,cdyyzz,.FALSE.,
     +		c0,c0,c0,-1,ieps0,ier)
	    do 40 i=8,14
		cs3(i) = -cs3(i)
   40	    continue
   50	    continue
*
*	    And now the remaining Li_2(x^2) terms
*	    stupid Gould NP1
*
	    c = cy(2)*cy(2)/(cdyz(2,1)*cdyz(2,1))
	    call ffzli2(zdilog,zlog,c,.FALSE.,ier)
	    cs3(15) = +zdilog/2
*	    stupid Gould NP1
	    c = cy(4)*cy(4)/(cdyz(2,1)*cdyz(2,1))
	    call ffzli2(zdilog,zlog,c,.FALSE.,ier)
	    cs3(16) = -zdilog/2
	    if ( lwrite ) then
		lwrite = .FALSE.
		ipi12p(1) = 0
		ipi12p(2) = 0
		ier0 = 0
		do 60 i=1,14
		    cs3p(i) = 0
   60		continue
		call ffcrr(cs3p(1),ipi12p(1),cy(2),cy(4),cz(1),
     +			cz(3),cdyz(2,1),.TRUE.,cd2yzz,cz(2),
     +			cz(4),isoort(1),ieps(1),ier0)
		call ffcrr(cs3p(8),ipi12p(2),cy(2),cy(4),cz(2),
     +			cz(4),cdyz(2,2),.TRUE.,cd2yzz,cz(1),
     +			cz(3),isoort(2),ieps(2),ier0)
		csum = 0
		xmax = 0
		do 70 i=1,14
		    csum = csum + cs3p(i)
		    xmax = max(xmax,absc(csum))
   70		continue
		csum = csum + (ipi12p(1)+ipi12(2))*DBLE(pi12)
		print '(a,3g20.10,3i3)','cmp',csum,xmax,ipi12p,ier0
		lwrite = .TRUE.
	    endif
	    goto 900
	endif
*  #] special case |cz| >> |cy|:
*  #[ normal:
	if ( isoort(2) .eq. 0 ) then
	    ld2yzz = .FALSE.
	else
	    ld2yzz = .TRUE.
	endif
	if ( isoort(1) .eq. 0 ) then
*	    do nothing
	elseif ( mod(isoort(1),10).eq.0 .or. mod(isoort(1),10).eq.-1
     +		.or. mod(isoort(1),10).eq.-3 ) then
	    call ffcrr(cs3(1),ipi12(1),cy(2),cy(4),cz(1),cz(3),
     +		cdyz(2,1),ld2yzz,cd2yzz,cz(2),cz(4),isoort(1),
     +		ieps(1),ier)
	elseif ( mod(isoort(1),10) .eq. -5 .or. mod(isoort(1),10) .eq.
     +		-6 ) then
	    y = DBLE(cy(2))
	    y1 = DBLE(cy(4))
	    z = DBLE(cz(1))
	    z1 = DBLE(cz(3))
	    dyz = DBLE(cdyz(2,1))
	    d2yzz = DBLE(cd2yzz)
	    zz = DBLE(cz(2))
	    zz1 = DBLE(cz(4))
	    sprec = precx
	    precx = precc
	    call ffcxr(cs3(1),ipi12(1),y,y1,z,z1,dyz,ld2yzz,d2yzz,zz,zz1
     +				,.FALSE.,x00,ieps(1),ier)
	    precx = sprec
	else
	    call fferr(12,ier)
	endif
	if ( isoort(2) .eq. 0 ) then
*	    do nothing
	elseif ( mod(isoort(2),5) .eq. 0 ) then
	    if ( lwrite ) print *,'ffcs3: skipped next R as it is the ',
     +		'conjugate'
	    do 100 i=1,7
  100		cs3(i) = 2*DBLE(cs3(i))
	    ipi12(1) = 2*ipi12(1)
	elseif ( mod(isoort(2),10).eq.-1 .or. mod(isoort(1),10).eq.-3 )
     +		then
	    call ffcrr(cs3(8),ipi12(2),cy(2),cy(4),cz(2),cz(4),
     +		cdyz(2,2),ld2yzz,cd2yzz,cz(1),cz(3),isoort(2),
     +		ieps(2),ier)
	elseif ( mod(isoort(2),10) .eq. -6 ) then
	    y = DBLE(cy(2))
	    y1 = DBLE(cy(4))
	    z = DBLE(cz(2))
	    z1 = DBLE(cz(4))
	    dyz = DBLE(cdyz(2,2))
	    d2yzz = DBLE(cd2yzz)
	    zz = DBLE(cz(1))
	    zz1 = DBLE(cz(3))
	    sprec = precx
	    precx = precc
	    call ffcxr(cs3(8),ipi12(2),y,y1,z,z1,dyz,ld2yzz,d2yzz,zz,zz1
     +				,.FALSE.,x00,ieps(2),ier)
	    precx = sprec
	else
	    call fferr(13,ier)
	endif
*  #] normal:
*  #[ eta's:
	if ( mod(isoort(1),10).eq.-5 .or. mod(isoort(1),10).eq.-6 )
     +		then
	    if ( mod(isoort(2),10).ne.-5 .and. mod(isoort(1),10).ne.-6
     +			) then
		print *,'ffcxs3: error: I assumed both would be real!'
		ier = ier + 50
	    endif
*	    we called ffcxr - no eta's
	elseif ( DIMAG(cpi(ip)).eq.0 ) then
	    call ffgeta(ni,cz(1),cdyz(1,1),cd2yzz,
     +		cpi(ip),cpiDpj(ii,ip),ieps,isoort,ier)
	    if ( lwrite ) print *,'ffcs3: eta''s are ',ni
	    ntot = ni(1) + ni(2) + ni(3) + ni(4)
	    if ( ntot .ne. 0 ) call ffclgy(cs3(15),ipi12(2),ntot,
     +		cy(1),cz(1),cd2yzz,ier)
	else
*
*	    cpi(ip) is really complex (occurs in transformed
*	    4pointfunction)
*
	    print *,'THIS PART IS NOT READY ',
     +			'and should not be reached'
	    stop
	endif
*  #] eta's:
*  #[ print output:
  900	if (lwrite) then
	    print *,'  cs3 ='
	    do 905 i=1,20
	    	if ( cs3(i).ne.0 ) print '(i3,2g20.10,1x)',i,cs3(i)
  905	    continue
	    print '(a3,2g20.10,1x)','pi ',(ipi12(1)+ipi12(2))*pi12
	    print *,'+-----------'
	    csum = 0
	    do 910 i=1,20
  910	    csum = csum + cs3(i)
	    csum = csum+(ipi12(1)+ipi12(2))*DBLE(pi12)
	    print '(a,2g20.10)','Si ',csum
	    print *,'  ipi12,ier=',ipi12,ier
	endif
*  #] print output:
*###] ffcs3:
	end
*###[ ffclgy:
	subroutine ffclgy(cs3,ipi12,ntot,cy,cz,cd2yzz,ier)
***#[*comment:***********************************************************
*									*
*	calculates the the difference of two S's with cy(3,4),cz(3,4),	*
*	cy(4)cz(3)-cy(3)cz(4) given.  Note the difference with ffdcs4,	*
*	in which the cy's are the same and only the cz's different.	*
*	Here both can be different.	Also we skip an intermediat	*
*	level.								*
*									*
*	input:	cy(4)	     (complex)	cy,1-cy in S with s3,s4		*
*		cz(4)	     (complex)	cz,1-cz in S with s3,s4		*
*		cdyz(2,2)    (complex)	cy - cz				*
*		cd2yzz	     (complex)	2*cy - cz+ - cz-		*
*		cdyzzy(4)    (complex)	cy(i,4)*cz(i,4)-cy(i,3)*cz(i,4)	*
*		cpiDpj(6,6)  (complex)	usual				*
*		cs3	     (complex)	assumed zero.			*
*									*
*	output: cs3	     (complex)	mod factors pi^2/12, in array	*
*		ipi12	     (integer)	these factors			*
*		isoort	     (integer)	returns kind of action taken	*
*		ier	     (integer)	number of digits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cs3
	DOUBLE COMPLEX cy(4),cz(4),cd2yzz
	integer ipi12,ntot,ier
*
*	local variables
*
	integer ipi
	DOUBLE COMPLEX c,cc,clogy,c2y1,zfflog,zfflo1,csum
	DOUBLE PRECISION absc
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ calculations:
	ipi = 0
	if ( 1 .lt. xloss*absc(cy(2)) ) then
	    clogy = zfflo1(1/cy(2),ier)
	else
	    if ( absc(cy(2)) .lt. xclogm .or. absc(cy(4)) .lt. xclogm )
     +			then
		if ( ntot .ne. 0 ) call fferr(15,ier)
		clogy = 0
	    else
		c = -cy(4)/cy(2)
		if ( DBLE(c) .gt. -abs(DIMAG(c)) ) then
		    clogy = zfflog(c,0,c0,ier)
		else
*		    take out the factor 2*pi^2
		    cc = c+1
		    if ( absc(cc) .lt. xloss ) then
			c2y1 = -cd2yzz - cz(1) + cz(4)
			if ( absc(c2y1) .lt. xloss*max(absc(cz(1)),
     +				absc(cz(4))) ) then
			    c2y1 = -cd2yzz - cz(2) + cz(3)
			    if ( lwarn .and. absc(c2y1) .lt. xloss*max(
     +				absc(cz(2)),absc(cz(3))) ) call ffwarn(
     +				56,ier,absc(c2y1),absc(cy(2)))
			endif
			csum = -c2y1/cy(2)
			clogy = zfflo1(csum,ier)
			if ( lwrite ) then
			    print *,'c  = ',c
			    print *,'c+ = ',-1+csum
			endif
		    else
			csum = 0
			clogy = zfflog(-c,0,c0,ier)
		    endif
		    if ( DIMAG(c) .lt. -precc*absc(c) .or.
     +			 DIMAG(csum) .lt. -precc*absc(csum) ) then
			ipi = -1
		    elseif ( DIMAG(c) .gt. precc*absc(c) .or.
     +			     DIMAG(csum) .gt. precc*absc(csum) ) then
			ipi = +1
		    else
			call fferr(51,ier)
			ipi = 0
		    endif
		endif
	    endif
	endif
	if ( ltest .and. cs3 .ne. 0 ) then
	    print *,'ffclgy: error: cs3 al bezet! ',cs3
	endif
	cs3 = cs3 + ntot*c2ipi*clogy
	if ( ipi .ne. 0 ) then
	    ipi12 = ipi12 - 24*ntot*ipi
	endif
*  #] calculations:
*###] ffclgy:
	end
*###[ ffieps:
	subroutine ffieps(ieps,cz,cp,cpDs,isoort)
***#[*comment:***********************************************************
*									*
*	Get the ieps prescription in such a way that it is compatible	*
*	with the imaginary part of cz if non-zero, compatible with the	*
*	real case if zero.						*
*									*
*	Input:	cz	complex(4)	the roots z-,z+,1-z-,1-z+	*
*		cp	complex		p^2				*
*		cpDs	complex		p.s				*
*		isoort	integer(2)	which type of Ri		*
*									*
*	Output:	ieps	integer(2)	z -> z-ieps*i*epsilon		*
*					will give correct im part	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ieps(2),isoort(2)
	DOUBLE COMPLEX cp,cpDs,cz(4)
*
*  #] declarations:
*  #[ work:
	if ( DIMAG(cp) .ne. 0 ) then
*	    do not calculate ANY eta terms, we'll do them ourselves.
	    ieps(1) = 99
	    ieps(2) = 99
	elseif ( isoort(2) .ne. 0 ) then
	    if ( DIMAG(cz(1)) .lt. 0 ) then
		ieps(1) = +1
		if ( DIMAG(cz(2)) .lt. 0 ) then
		    ieps(2) = +1
		else
		    ieps(2) = -1
		endif
	    elseif ( DIMAG(cz(1)) .gt. 0 ) then
		ieps(1) = -1
		if ( DIMAG(cz(2)) .le. 0 ) then
		    ieps(2) = +1
		else
		    ieps(2) = -1
		endif
	    else
		if ( DIMAG(cz(2)) .lt. 0 ) then
		    ieps(1) = -1
		    ieps(2) = +1
		elseif ( DIMAG(cz(2)) .gt. 0 ) then
		    ieps(1) = +1
		    ieps(2) = -1
		else
		    if ( (DBLE(cz(2)).gt.DBLE(cz(1))
     +			.or.  (DBLE(cz(1)).eq.DBLE(cz(2))
     +				.and. DBLE(cz(4)).lt.DBLE(cz(3)))
     +			) .eqv. DBLE(cp).gt.0 ) then
			ieps(1) = +1
			ieps(2) = -1
		    else
			ieps(1) = -1
			ieps(2) = +1
		    endif
		endif
	    endif
	else
	    if ( DIMAG(cz(1)) .lt. 0 ) then
		ieps(1) = +1
	    elseif ( DIMAG(cz(1)) .gt. 0 ) then
		ieps(1) = -1
	    elseif ( DBLE(cpDs) .gt. 0 ) then
		ieps(1) = +1
	    else
		ieps(1) = -1
	    endif
	    ieps(2) = -9999
	endif
*  #] work:
*###] ffieps:
	end
*###[ ffgeta:
	subroutine ffgeta(ni,cz,cdyz,cd2yzz,cp,cpDs,ieps,isoort,ier)
***#[*comment:***********************************************************
*									*
*	Get the eta terms which arise from splitting up			*
*	log(p2(x-z-)(x-z+)) - log(p2(y-z-)(y-z+))			*
*									*
*	Input:	cz	complex(4)	the roots z-,z+,1-z-,1-z+	*
*		cdyz	complex(2,2)	y-z				*
*		cd2yzz	complex(2)	2y-(z-)-(z+)			*
*		cp	complex		p^2				*
*		cpDs	complex		p.s				*
*		ieps	integer(2)	the assumed im part if Im(z)=0	*
*		isoort	integer(2)	which type of Ri		*
*									*
*	Output:	ni	integer(4)	eta()/(2*pi*i)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ni(4),ieps(2),isoort(2),ier
	DOUBLE COMPLEX cp,cpDs,cz(4),cdyz(2,2),cd2yzz
*
*	local variables
*
	integer i,nffeta,nffet1
	DOUBLE COMPLEX cmip
*
*	common
*
	include 'ff.h'
*
*  #] declarations:
*  #[ complex  masses or imaginary roots:
*
*	only complex because of complex roots in y or z
*	[checked and in agreement with ieps definition 23-sep-1991]
*
	if ( lwrite ) print *,'ffgeta: isoort = ',isoort
*
*	isoort = +1:        y is real, z is real
*	isoort = -1-n*10:   y is complex, possibly z as well
*	isoort = -3-n*10:   y,z complex, (y-z-)*(y-z+) real
*	isoort = 0:         y is complex, one z root only
*	isoort = -10-n*10:  y is real, z is complex
*	isoort = -5,6-n*10: y,z real
*
	if ( isoort(1) .gt. 0 ) then
*
*	    really a real case
*
	    ni(1) = 0
	    ni(2) = 0
	    ni(3) = 0
	    ni(4) = 0
	elseif ( mod(isoort(1),10) .ne. 0 .and. isoort(2) .ne. 0 ) then
	    cmip = DCMPLX(DBLE(x0),-DBLE(cp))
*
*	    ni(1) = eta(p2,(x-z-)(x-z+)) = 0 by definition (see ni(3))
*	    ni(2) = eta(x-z-,x-z+)
*
	    ni(1) = 0
	    if ( ieps(1) .gt. 0 .neqv. ieps(2) .gt. 0 ) then
		ni(2) = 0
	    else
		ni(2) = nffet1(-cz(1),-cz(2),cmip,ier)
		if ( cz(3).ne.0 .and. cz(4).ne.0 ) then
		    i = nffet1(cz(3),cz(4),cmip,ier)
		    if ( i .ne. ni(2) ) call fferr(53,ier)
		endif
	    endif
*
*	    ni(3) compensates for whatever convention we chose in ni(1)
*	    ni(4) = -eta(y-z-,y-z+)
*
***	    if ( DBLE(cd2yzz).eq.0 .and. ( DIMAG(cz(1)).eq.0 .and.
***     +		 DIMAG(cz(2)).eq.0 .or. DBLE(cdyz(2,1)).eq.0 .and.
***     +		 DBLE(cdyz(2,2)) .eq. 0 ) ) then
	    if ( mod(isoort(1),10).eq.-3 ) then
*		follow the i*epsilon prescription as (y-z-)(y-z+) real
		ni(3) = 0
		if ( ltest ) then
		    if ( DIMAG(cdyz(2,1)).eq.0 .or. DIMAG(cdyz(2,2))
     +		    .eq.0 ) print *,'ffgeta: error: calling nffet1',
     +		    ' with im(y-z-)=im(y-z+)=0: ',cdyz(2,1),cdyz(2,2)
		endif
		ni(4) = -nffet1(cdyz(2,1),cdyz(2,2),cmip,ier)
	    else
		if ( DBLE(cp) .lt. 0 .and. DIMAG(cdyz(2,1)*
     +			cdyz(2,2)) .lt. 0 ) then
		    ni(3) = -1
		else
		    ni(3) = 0
		endif
		ni(4) = -nffeta(cdyz(2,1),cdyz(2,2),ier)
	    endif
	elseif ( (mod(isoort(1),10).eq.-1 .or. mod(isoort(1),10).eq.-3)
     +		.and. isoort(2) .eq. 0 ) then
	    ni(1) = 0
	    if ( DIMAG(cz(1)) .ne. 0 ) then
		ni(2) = nffet1(-cpDs,-cz(1),DCMPLX(DBLE(0),
     +			DBLE(-1)),ier)
	    else
		ni(2) = nffet1(-cpDs,DCMPLX(DBLE(0),DBLE(1)),
     +			DCMPLX(DBLE(0),DBLE(-1)),ier)
	    endif
	    ni(3) = 0
	    ni(4) = -nffeta(-cpDs,cdyz(2,1),ier)
	else
	    ni(1) = 0
	    ni(2) = 0
	    ni(3) = 0
	    ni(4) = 0
	endif
*  #] complex  masses or imaginary roots:
*###] ffgeta:
	end
