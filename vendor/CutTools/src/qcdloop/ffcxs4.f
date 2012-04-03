*	$Id: ffcxs4.f,v 1.3 1995/10/17 06:55:09 gj Exp $
*	$Log: ffcxs4.f,v $
c Revision 1.3  1995/10/17  06:55:09  gj
c Fixed ieps error in ffdcrr (ffcxs4.f), added real case in ffcrr, debugging
c info in ffxd0, and warned against remaining errors for del2=0 in ffrot4
c (ffxd0h.f)
c
c Revision 1.2  1995/10/06  09:17:22  gj
c Found stupid typo in ffxc0p which caused the result to be off by pi^2/3 in
c some equal-mass cases.  Added checks to ffcxs4.f ffcrr.f.
c
*###[ ffcxs4:
	subroutine ffcxs4(cs3,ipi12,w,y,z,dwy,dwz,dyz,d2yww,d2yzz,
     +					xpi,piDpj,ii,ns,isoort,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the 8 Spence functions = 4 R's = 2 dR's		*
*									*
*									*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(4),ii,ns,isoort(4),ier
	DOUBLE COMPLEX cs3(40)
	DOUBLE PRECISION w(4),y(4),z(4),dwy(2,2),dwz(2,2),dyz(2,2),
     +		d2yww,d2yzz,xpi(ns),piDpj(ns,ns),x00(3)
*
*	local variables:
*
	integer iepz(2),iepw(2)
	logical ld2yzz,ld2yww
*	DOUBLE COMPLEX c
*	DOUBLE PRECISION absc
*
*	common blocks
*
	include 'ff.h'
*	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations: 
*  #[ groundwork:
	if ( ltest .and. ns .ne. 6 )
     +	    print *,'ffcxs4: error: only for ns=6, not ',ns
	if ( isoort(2) .eq. 0 ) then
	    ld2yzz = .FALSE.
	else
	    ld2yzz = .TRUE.
	endif
	if ( isoort(4) .eq. 0 ) then
	    ld2yww = .FALSE.
	else
	    ld2yww = .TRUE.
	endif
	if ( isoort(2) .ne. 0 ) then
	    if ( z(2) .gt. z(1) .eqv. xpi(ii+3) .gt. 0 ) then
		iepz(1) = +1
		iepz(2) = -1
	    else
		iepz(1) = -1
		iepz(2) = +1
	    endif
	else
	    print *,'ffcxs4: error: untested algorithm'
	    if ( piDpj(ii,ii+3) .gt. 0 ) then
		iepz(1) = +1
	    else
		iepz(1) = -1
	    endif
	endif
	if ( isoort(4) .ne. 0 ) then
	    if ( w(2) .gt. w(1) .eqv. xpi(5) .gt. 0 ) then
		iepw(1) = 1
		iepw(2) = -1
	    else
		iepw(1) = -1
		iepw(2) = 1
	    endif
	else
	    print *,'ffcxs4: error: untested algorithm'
	    if ( piDpj(2,5) .gt. 0 ) then
		iepw(1) = +1
	    else
		iepw(1) = -1
	    endif
	endif
*  #] groundwork: 
*  #[ zm and wp:
	if ( isoort(4) .eq. 0 ) then
	    if (lwrite) print *,'ffcxs4: to ffcxr(zm)'
	    call ffcxr(cs3(1),ipi12(1),y(2),y(4),z(1),z(3),dyz(2,1),
     +		ld2yzz,d2yzz,z(2),z(4),.FALSE.,x00,iepz(1),ier)
	else
	    if (lwrite) print *,'ffcxs4: to ffdcxr(zm,wp)'
	    if ( .not. ( dwz(2,1).eq.0 .and. iepz(1).eq.iepw(2) ) )
     +	    	call ffdcxr(cs3( 1),ipi12(1),y(2),y(4),z(1),z(3),
     +			z(2),z(4),d2yzz,w(2),w(4),w(1),w(3),d2yww,
     +			dyz(2,1),dwy(2,2),dwz(2,1),iepz(1),iepw(2),ier)
	endif
*  #] zm and wp: 
*  #[ zp and wm:
	if ( isoort(2) .eq. 0 ) then
	    if (lwrite) print *,'ffcxs4: to ffcxr(wm)'
	    call ffcxr(cs3(1),ipi12(1),y(2),y(4),w(1),w(3),-dwy(1,2),
     +		ld2yww,d2yww,w(2),w(4),.FALSE.,x00,iepw(1),ier)
	else
	    if (lwrite) print *,'ffcxs4: to ffdcxr(zp,wm)'
	    if ( .not. ( dwz(1,2).eq.0 .and. iepz(2).eq.iepw(1) ) )
     +	    	call ffdcxr(cs3(21),ipi12(3),y(2),y(4),z(2),z(4),
     +			z(1),z(3),d2yzz,w(1),w(3),w(2),w(4),d2yww,
     +			dyz(2,2),dwy(1,2),dwz(1,2),iepz(2),iepw(1),ier)
	endif
*  #] zp and wm: 
*###] ffcxs4: 
	end
*###[ ffcs4:
	subroutine ffcs4(cs3,ipi12,cw,cy,cz,cdwy,cdwz,cdyz,cd2yww,cd2yzz
     +			,cpi,cpiDpj,cp2p,cetami,ii,ns,isoort,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the 8 Spence functions = 4 R's = 2 dR's		*
*									*
*									*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(4),ii,ns,isoort(4),ier
	DOUBLE COMPLEX cs3(40)
	DOUBLE COMPLEX cw(4),cy(4),cz(4),cdwy(2,2),cdwz(2,2),cdyz(2,2)
	DOUBLE COMPLEX cd2yww,cd2yzz,cpi(ns),cp2p,cpiDpj(ns,ns),
     +		cetami(6)
*
*	local variables:
*
	logical ld2yzz,ld2yww
	integer i,j,ip,iepz(2),iepw(2),nz(4),nw(4),ntot,i2pi
	DOUBLE COMPLEX c,cc,clogy,c2y1,cdyw(2,2)
	DOUBLE COMPLEX zfflo1,zfflog
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
*  #[ get counters:
	if ( ltest ) then
	    if ( ns .ne. 6 ) then
	    	print *,'ffcs4: error: only for ns=6, not ',ns
	    	stop
	    endif
	    do i=1,4
	    	if ( ipi12(i).ne.0 ) then
	    	    print *,'ffcs4: error: ipi12(',i,') non-zero! ',
     +	    	    	ipi12(i)
	    	endif
	    enddo
	endif
	ip = ii+3
	if ( isoort(2) .eq. 0 ) then
	    ld2yzz = .FALSE.
	else
	    ld2yzz = .TRUE.
	endif
	if ( isoort(4) .eq. 0 ) then
	    ld2yww = .FALSE.
	else
	    ld2yww = .TRUE.
	endif
	call ffieps(iepz,cz,cpi(ip),cpiDpj(ip,ii),isoort)
	call ffieps(iepw,cw,cp2p,cpiDpj(ip,ii),isoort(3))
	if ( isoort(4) .eq. 0 ) then
	    print *,'ffcs4: error: case not implemented'
	    ier = ier + 50
	endif
*  #] get counters: 
*  #[ R's:
	if ( isoort(4) .eq. 0 ) then
	    call ffcrr(cs3(1),ipi12(1),cy(2),cy(4),cz(1),cz(3),cdyz(2,1)
     +		,ld2yzz,cd2yzz,cz(2),cz(4),isoort(4),iepz(1),ier)
	else
	    if (lwrite) print *,'ffcs4: to ffdcrr(zm,wp)'
	    if ( .not. ( cdwz(2,1).eq.0 .and. iepz(1).eq.iepw(2) ) )
     +	    call ffdcrr(cs3( 1),ipi12(1),cy(2),cy(4),cz(1),cz(3),cz(2),
     +		cz(4),cd2yzz,cw(2),cw(4),cw(1),cw(3),cd2yww,cdyz(2,1),
     +		cdwy(2,2),cdwz(2,1),isoort(4),iepz(1),iepw(2),ier)
	endif
	if ( isoort(2) .eq. 0 ) then
	    call ffcrr(cs3(1),ipi12(1),cy(2),cy(4),cw(1),cw(3),-cdwy(1,2
     +	    	),ld2yww,cd2yww,cw(2),cw(4),isoort(2),iepw(1),ier)
	else
	    if (lwrite) print *,'ffcs4: to ffdcrr(zp,wm)'
	    if ( .not. ( cdwz(1,2).eq.0 .and. iepz(2).eq.iepw(1) ) )
     +	    call ffdcrr(cs3(21),ipi12(3),cy(2),cy(4),cz(2),cz(4),cz(1),
     +		cz(3),cd2yzz,cw(1),cw(3),cw(2),cw(4),cd2yww,cdyz(2,2),
     +		cdwy(1,2),cdwz(1,2),iepz(2),isoort(2),iepw(1),ier)
	endif
*  #] R's: 
*  #[ eta's:
	if ( DIMAG(cpi(ip)) .eq. 0 ) then
	    call ffgeta(nz,cz,cdyz,cd2yzz,
     +		cpi(ip),cpiDpj(ii,ip),iepz,isoort,ier)
	    do 120 i=1,2
		do 110 j=1,2
		    cdyw(i,j) = cdwy(j,i)
  110		continue
  120	    continue
	    call ffgeta(nw,cw,cdyw,cd2yww,
     +		cp2p,cpiDpj(ii,ip),iepw,isoort(3),ier)
	else
	    print *,'ffcs4: error: not ready for complex D0 yet'
	endif
	ntot = nz(1)+nz(2)+nz(3)+nz(4)-nw(1)-nw(2)-nw(3)-nw(4)
	if ( ntot .ne. 0 ) then
	    i2pi = 0
	    if ( 1/absc(cy(2)) .lt. xloss ) then
		clogy = zfflo1(1/cy(2),ier)
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
     +			     absc(cz(4))) ) then
			    c2y1 = -cd2yzz - cz(2) + cz(3)
			    if ( lwarn .and. absc(c2y1) .lt. xloss*max(
     +				absc(cz(2)),absc(cz(3))) ) then
				call ffwarn(134,ier,absc(c2y1),
     +					absc(cy(2)))
			    endif
			endif
			if ( lwrite ) then
			    print *,'1+c         = ',1+c
			    print *,'-c2y1/cy(2) = ',-c2y1/cy(2)
			endif
			clogy = zfflo1(-c2y1/cy(2),ier)
		    else
			clogy = zfflog(-c,0,c0,ier)
		    endif
		    if ( DIMAG(c) .lt. 0 ) then
			i2pi = -1
		    elseif ( DIMAG(c) .gt. 0 ) then
			i2pi = +1
		    else
			call fferr(51,ier)
			i2pi = 0
		    endif
		    ipi12(2) = ipi12(2) - ntot*24*i2pi
		endif
	    endif
	    if ( cs3(40) .ne. 0 ) print *,'ffcs4: error: cs3(40) != 0'
	    cs3(40) = ntot*c2ipi*clogy
	endif
	if ( lwrite ) then
	    print *,'eta''s:'
	    print *,'nzi  :',nz
	    print *,'nwi  :',nw
	    print *,'total:',ntot*c2ipi*clogy
	    if ( i2pi .ne. 0 ) print *,'     +',-ntot*24*i2pi*pi12
	    print *,'     =',ntot,' *( ',c2ipi*clogy,' + ',24*i2pi*pi12,
     +		')'
	endif
*  #] eta's: 
*###] ffcs4:
	end
*###[ ffdcxr:
	subroutine ffdcxr(cs3,ipi12,y,y1,z,z1,zp,zp1,d2yzz,
     +			w,w1,wp,wp1,d2yww,dyz,dwy,dwz,iepsz,iepsw,ier)
***#[*comment:***********************************************************
*									*
*	Calculate							*
*									*
*		R(y,z,iepsz) - R(y,w,iepsw)				*
*									*
*	Input:								*
*		a = [yzw]	(real)		see definition		*
*		a1 = 1 - a	(real)					*
*		dab = a - b	(real)					*
*		ieps[zw]	(integer)	sign of imaginary part	*
*						of argument logarithm	*
*		cs3(20)		(complex)	assumed zero		*
*									*
*	Output:								*
*		cs3(20)		(complex)	the results, not added	*
*		ipi12(2)	(integer)	factors pi^2/12		*
*									*
*	Calls:	ffcxr							*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(2),iepsz,iepsw,ier
	DOUBLE COMPLEX cs3(20)
	DOUBLE PRECISION y,z,w,y1,z1,w1,dyz,dwy,dwz,zp,zp1,d2yzz,wp,wp1,
     +		d2yww
*
*	local variables:
*
	integer i,ieps,ipi12p(2),ier1,ier2,isign,inorm
	logical again
	DOUBLE PRECISION yy,yy1,zz,zz1,dyyzz,xx1,xx1n,term,tot,d2,d3,
     +		d21,d31,d2n,d3n,d21n1,d31n1,dw,xlogy,x00(3)
	DOUBLE COMPLEX csum,csum1,csum2,cs3p(20),chulp
	DOUBLE PRECISION dfflo1
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
*	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
	inorm = 0
*  #] declarations: 
*  #[ groundwork:
	if ( dwz .eq. 0 .and. iepsz .eq. iepsw ) return
	if ( dyz .eq. 0 ) then
	    call fferr(75,ier)
	    return
	endif
	xx1 = y/dyz
	dw = dwz/dyz
	if ( xx1 .le. x05 .or. xx1 .gt. 2 ) then
	    d2 = 1/y
	    dw = dw*y/w
	else
	    d2 = 1/z1
	endif
	again = .FALSE.
  123	continue
*  #] groundwork: 
*  #[ trivial case:
	if ( dw .eq. 0 ) then
	    if ( lwrite ) print *,'  Trivial case'
*  #] trivial case: 
*  #[ normal case:
	elseif ( abs(dw) .gt. xloss .or. again ) then
*	    nothing's the matter
	    if ( lwrite ) print *,'  Normal case'
	    inorm = 1
	    call ffcxr(cs3( 1),ipi12(1),y,y1,z,z1,dyz,
     +		.TRUE.,d2yzz,zp,zp1,.FALSE.,x00,iepsz,ier)
	    call ffcxr(cs3(11),ipi12(2),y,y1,w,w1,-dwy,
     +		.TRUE.,d2yww,wp,wp1,.FALSE.,x00,iepsw,ier)
	    do 10 i=11,20
   10		cs3(i) = -cs3(i)
	    ipi12(2) = -ipi12(2)
*  #] normal case: 
*  #[ only cancellations in w, not in y:
	elseif ( abs(d2) .gt. xloss ) then
*	    there are no cancellations the other way:
	    if ( lwrite ) print *,'  Cancellations one way, turned Rs'
	    if ( iepsz .ne. iepsw .and. ( y/dyz .gt. 1 .or.-y/dwy .gt.
     +		1 ) ) then
		again = .TRUE.
		if ( lwrite ) then
		    print *,'ffdcxr: problems with ieps, solvable,'
		    print *,'        but for the moment just call the ',
     +		    	'normal case'
		endif
		again = .TRUE.
		goto 123
*		call fferr(21,ier)
	    endif
	    yy = dwy/dwz
	    zz = yy*z/y
	    yy1 = dyz/dwz
	    zz1 = yy1*w/y
	    dyyzz = yy*dyz/y
	    if ( y .lt. 0 ) then
		ieps = iepsz
	    else
		ieps = -iepsz
	    endif
	    call ffcxr(cs3( 1),ipi12(1),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,2*ieps,ier)
	    zz = yy*z1/y1
	    zz1 = yy1*w1/y1
	    dyyzz = -yy*dyz/y1
	    if ( y1 .gt. 0 ) then
		ieps = iepsz
	    else
		ieps = -iepsz
	    endif
	    call ffcxr(cs3(11),ipi12(2),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,2*ieps,ier)
	    do 20 i=11,20
		cs3(i) = -cs3(i)
   20	    continue
	    ipi12(2) = -ipi12(2)
*  #] only cancellations in w, not in y: 
*  #[ Hill identity:
	elseif (  ( 1 .gt. xloss*abs(y) .or. abs(xx1) .gt. xloss )
     +	    .and. ( 1 .gt. xloss*abs(z) .or. abs(z/dyz) .gt. xloss )
     +	    .and. ( 1 .gt. xloss*abs(y) .or. abs(dyz/y) .gt. xloss )
     +		) then
*	    do a Hill identity on the y,y-1 direction
	    if ( lwrite ) print *,'  Hill identity to split z,w'
	    yy = -y*w1/dwy
	    yy1 = w*y1/dwy
	    zz = -z*w1/dwz
	    zz1 = w*z1/dwz
	    dyyzz = -w*w1*(dyz/(dwy*dwz))
	    if ( y*dwz .gt. 0 .eqv. (y+dwz) .gt. 0 ) then
		ieps = 2*iepsw
	    else
		ieps = -2*iepsw
	    endif
	    call ffcxr(cs3( 1),ipi12(1),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,ieps,ier)
	    yy = w1
	    yy1 = w
	    zz = -w1*z/dwz
	    zz1 = w*z1/dwz
	    dyyzz = w*w1/dwz
	    call ffcxr(cs3( 9),ipi12(2),yy,yy1,zz,zz1,dyyzz,.FALSE.,x0,
     +			x0,x0,.FALSE.,x00,ieps,ier)
	    do 30 i=9,16
   30		cs3(i) = -cs3(i)
	    ipi12(2) = -ipi12(2)
*	    the extra logarithms ...
	    if ( 1 .lt. xloss*abs(w) ) then
		chulp = dfflo1(1/w,ier)
	    elseif ( w1 .lt. 0 .or. w .lt. 0 ) then
		chulp = log(-w1/w)
	    else
		chulp = DCMPLX(DBLE(log(w1/w)),DBLE(-iepsw*pi))
	    endif
	    cs3(20) = -DBLE(dfflo1(dwz/dwy,ier))*chulp
*  #] Hill identity: 
*  #[ Taylor expansion:
	elseif ( (w.lt.0..or.w1.lt.0) .and. (z.lt.0..or.z1.lt.0) ) then
*	    do a Taylor expansion
	    if ( abs(xx1) .lt. xloss ) then
		if ( lwrite ) print *,'ffdcxr: Taylor expansion, normal'
		d3 = dwz/dwy
*		isign = 1
		xx1n = xx1
		d2n = d2
		d3n = d3
		d21 = 1-d2
		d21n1 = 1
		d31 = 1-d3
		d31n1 = 1
		tot = xx1*d2*d3
		do 50 i=2,20
		    xx1n = xx1n*xx1
		    d21n1 = d21n1*d21
		    d31n1 = d31n1*d31
		    d2n = d2n + d2*d21n1
		    d3n = d3n + d3*d31n1
		    term = xx1n*d2n*d3n*xn2inv(i)
		    tot = tot + term
		    if ( abs(term) .le. precx*abs(tot) ) goto 51
   50		continue
		if ( lwarn ) call ffwarn(46,ier,tot,term)
   51		continue
*		if ( isign .eq. 1 ) then
		    cs3(1) = tot
*		else
*		    cs3(1) = -tot
*		endif
	    elseif ( abs(z/dyz) .lt. xloss ) then
		if ( lwrite ) print *,'  Normal case'
		inorm = 1
		call ffcxr(cs3( 1),ipi12(1),y,y1,z,z1,dyz,
     +		     .TRUE.,d2yzz,zp,zp1,.FALSE.,x00,iepsz,ier)
		call ffcxr(cs3(11),ipi12(2),y,y1,w,w1,-dwy,
     +		     .TRUE.,d2yww,wp,wp1,.FALSE.,x00,iepsw,ier)
		do 110 i=11,20
  110		    cs3(i) = -cs3(i)
*		if ( lwrite ) print *,'ffdcxr: Taylor expansion, 1-x'
*		print *,'NOT YET READY !!!'
*		ier = ier + 100
*		yy = y1*dwz/(z1*dwy)
*		if ( abs(yy) .lt. xloss ) then
*		    cs3(10) = -dfflo1(1/y,ier)*dfflo1(yy,ier)
*		else
*		    yy1 = -w1*dyz/(z1*dwy)
*		    if ( yy1 .gt. xalogm ) then
*			cs3(10) = -dfflo1(1/y,ier)*log(yy1)
*		    elseif ( yy1 .gt. -xalogm ) then
*			if ( lwarn ) call ffwarn(80,ier,yy1,xalogm)
*		    else
*			xlogy = log(-yy1)
*			if ( lwarn .and. iepsz.ne.iepsw )
*     +				call ffwarn(81,ier,x1,x1)
*			if ( (w1+dyz)*dwz*y1*iepsz .lt. 0 ) then
*			    cs3(10) = -dfflo1(1/y,ier)*DCMPLX(DBLE(xlogy),DBLE(pi))
*			else
*			    cs3(10) = -dfflo1(1/y,ier)*DCMPLX(DBLE(xlogy),DBLE(-pi))
*			endif
*		    endif
*		endif
*		cs3(11) = -dfflo1(1/z,ier)*dfflo1(dwz/dwy,ier)
*		yy = dwz/(w*z1)
*		if ( abs(yy) .lt. xloss ) then
*		    cs3(12) = -dfflo1(w/dwy,ier)*dfflo1(yy,ier)
*		else
*		    yy1 = z*w1/(w*z1)
*		    if ( yy1 .gt. xalogm ) then
*			cs3(12) = -dfflo1(w/dwy,ier)*log(yy1)
*		    elseif (  yy .gt. -xalogm ) then
*			if ( lwarn ) call ffwarn(80,ier,yy,xalogm)
*		    else
*			xlogy = log(-yy1)
*			if ( lwarn .and. iepsz.ne.iepsw )
*     +				call ffwarn(81,ier,x1,x1)
*			if ( dwz*(dwz+1)*ieps .gt. 0 ) then
*			    cs3(12) = -dfflo1(w/dwy,ier)*DCMPLX(DBLE(xlogy),DBLE(pi))
*			else
*			    cs3(12) =-dfflo1(w/dwy,ier)*DCMPLX(DBLE(xlogy),DBLE(-pi))
*			endif
*		    endif
*		endif
*		isign = -1
*		xx1 = -z/dyz
*		d2 = 1/z
*		d3 = dwz/dwy
	    else
		if ( lwrite ) print *,'ffdcxr: Taylor expansion, 1/x'
		call fferr(22,ier)
		return
	    endif
	else
	    if ( lwrite ) print *,'Not clear, take normal route'
	    inorm = 1
	    call ffcxr(cs3( 1),ipi12(1),y,y1,z,z1,dyz,.FALSE.,x0,x0,x0,
     +				.FALSE.,x00,iepsz,ier)
	    call ffcxr(cs3(11),ipi12(2),y,y1,w,w1,-dwy,.FALSE.,x0,x0,x0,
     +				.FALSE.,x00,iepsw,ier)
	    do 40 i=11,20
   40		cs3(i) = -cs3(i)
	    ipi12(2) = -ipi12(2)
	endif
*  #] Taylor expansion: 
*  #[ debug output:
        if ( lwrite ) then
	    csum = 0
	    do 900 i=1,20
		csum = csum + cs3(i)
		print '(i2,2g16.8)',i,cs3(i)
  900	    continue
	    print '(a)','---------------------------------'
	    print '(2x,2g16.8,2i3)',csum,ipi12
	    print '(a,i3)','ier = ',ier
	    if ( inorm .eq. 0 ) then
	    lwrite = .FALSE.
	    ier1 = 0
	    ier2 = 0
	    do 905 i=1,20
  905		cs3p(i) = 0
	    ipi12p(1) = 0
	    ipi12p(2) = 0
	    call ffcxr(cs3p( 1),ipi12p(1),y,y1,z,z1,dyz,.FALSE.,x0,x0,
     +				x0,.FALSE.,x00,iepsz,ier1)
	    call ffcxr(cs3p(11),ipi12p(2),y,y1,w,w1,-dwy,.FALSE.,x0,x0,
     +				x0,.FALSE.,x00,iepsw,ier2)
	    csum1 = 0
	    do 910 i=1,10
  910		csum1 = csum1 + cs3p(i)
	    csum2 = 0
	    do 920 i=11,20
  920		csum2 = csum2 - cs3p(i)
	    csum = csum1 + csum2 + (ipi12p(1)-ipi12p(2))*DBLE(pi12)
	    print *,'cmp with:'
	    print '(i2,2g16.8,i3)',1,csum1,ier1
	    print '(i2,2g16.8,i3)',2,csum2,ier2
	    print *,'------------------+'
	    print '(2x,2g16.8,3i3)',csum1+csum2,ipi12p,max(ier1,ier2)
	    print '(2x,2g16.8,3i3)',csum
	    lwrite = .TRUE.
	    endif
	endif
*  #] debug output: 
*###] ffdcxr: 
	end
*###[ ffdcrr:
	subroutine ffdcrr(cs3,ipi12,cy,cy1,cz,cz1,czp,czp1,cd2yzz,cw,cw1
     +		,cwp,cwp1,cd2yww,cdyz,cdwy,cdwz,isoort,iepsz,iepsw,ier)
***#[*comment:***********************************************************
*									*
*	Calculate							*
*									*
*		R(cy,cz,iepsz) - R(cy,cw,iepsw)				*
*									*
*	Input:								*
*		a = [yzw]	(real)		see definition		*
*		a1 = 1 - a	(real)					*
*		dab = a - b	(real)					*
*		ieps[zw]	(integer)	sign of imaginary part	*
*						of argument logarithm	*
*		cs3(20)		(complex)	assumed zero		*
*									*
*	Output:								*
*		cs3(20)		(complex)	the results, not added	*
*		ipi12(2)	(integer)	factors pi^2/12		*
*									*
*	Calls:	ffcrr							*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(2),isoort,iepsz,iepsw,ier
	DOUBLE COMPLEX cs3(20)
	DOUBLE COMPLEX cy,cz,czp,cw,cwp,cy1,cz1,czp1,cw1,cwp1,
     +		cdyz,cdwy,cdwz,cd2yzz,cd2yww
*
*	local variables:
*
	integer i,ieps,ieps1,ieps2,ipi12p(2),ier1,ier2,isign,inorm,i2pi,
     +		nffeta,nffet1,n1,n2,n3,n4,n5,n6
	logical ld2yyz
	DOUBLE COMPLEX cyy,cyy1,czz,czz1,cdyyzz,chulp,zfflo1,zfflog,
     +		cc1,cdw,cc1n,cterm,ctot,cd2,cd3,
     +		cd21,cd31,cd2n,cd3n,cd21n1,cd31n1,
     +		cc2,cfactz,cfactw,czzp,czzp1,cd2yyz
	DOUBLE COMPLEX csum,csum1,csum2,cs3p(20),c,check
	DOUBLE PRECISION absc,xlosn
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
	inorm = 0
*  #] declarations: 
*  #[ check input:
	if ( ltest ) then
	    xlosn = xloss*DBLE(10)**(-1-mod(ier,50))
	    check = cd2yzz - 2*cy + cz + czp
	    if ( xlosn*absc(check) .gt. precc*max(2*absc(cy),absc(cz),
     +			absc(czp)) ) then
		print *,'ffdcrr: error:  cd2yzz != 2*cy - cz - czp:',
     +			cd2yzz,cy,cz,czp,check
	    endif
	    check = cd2yww - 2*cy + cw + cwp
	    if ( xlosn*absc(check) .gt. precc*max(2*absc(cy),absc(cw),
     +			absc(cwp)) ) then
		print *,'ffdcrr: error:  cd2yww != 2*cy - cw - cwp:',
     +			cd2yww,cy,cw,cwp,check
	    endif
	endif
*  #] check input:
*  #[ groundwork:
	if ( cdwz .eq. 0 ) then
	    if ( abs(DIMAG(cz)) .gt. precc*abs(DBLE(cz)) .or.
     +		iepsz .eq. iepsw ) return
	    if ( DBLE(cz) .ge. 0 .and. DBLE(cz1) .ge. 0 ) return
	    call fferr(76,ier)
	    return
	endif
	if ( cdyz .eq. 0 ) then
	    call fferr(77,ier)
	    return
	endif
	cc1 = cy/cdyz
	cdw = cdwz/cdyz
	if ( DBLE(cc1) .le. x05 .or. abs(cc1-1) .gt. 1 ) then
	    cd2 = 1/cy
	    cdw = cdw*cy/cw
	else
	    cd2 = 1/cz1
	endif
*  #] groundwork:
*  #[ trivial case:
	if ( absc(cdw) .eq. 0 ) then
	    if ( lwrite ) print *,'  Trivial case'
*  #] trivial case: 
*  #[ normal case:
*
*	if no cancellations are expected OR the imaginary signs differ
*	and are significant
*
	elseif ( absc(cdw) .gt. xloss .or. (iepsz.ne.iepsw .and.
     +		(DBLE(cy/cdyz).gt.1 .or. DBLE(-cy1/cdyz).gt.1) ) ) then
*	    nothing's the matter
	    if ( lwrite ) print *,'ffdcrr: Normal case'
	    inorm = 1
*	    special case to avoid bug found 15-oct=1995
	    if ( iepsz.eq.iepsw ) then
	    	if ( DIMAG(cz).eq.0 .and. DIMAG(cz1).eq.0 ) then
	    	    print *,'ffdcrr: flipping sign iepsz'
	    	    iepsz = -iepsz
	    	elseif ( DIMAG(cw).eq.0 .and. DIMAG(cw1).eq.0 ) then
	    	    print *,'ffdcrr: flipping sign iepsw'
	    	    iepsw = -iepsw
	    	else
	    	    print *,'ffdcrr: error: missing eta terms!'
	    	    ier = ier + 100
	    	endif
	    endif
	    call ffcrr(cs3(1),ipi12(1),cy,cy1,cz,cz1,cdyz,.TRUE.,
     +		cd2yzz,czp,czp1,isoort,iepsz,ier)
	    call ffcrr(cs3(8),ipi12(2),cy,cy1,cw,cw1,-cdwy,.TRUE.,
     +		cd2yww,cwp,cwp1,isoort,iepsw,ier)
	    do 10 i=8,14
		cs3(i) = -cs3(i)
   10	    continue
	    ipi12(2) = -ipi12(2)
*  #] normal case:
*  #[ only cancellations in cw, not in cy:
	elseif ( absc(cd2) .gt. xloss ) then
*	    there are no cancellations the other way:
	    if ( lwrite ) print *,'ffdcrr: Cancellations one way, ',
     +		'turned Rs'
	    cyy = cdwy/cdwz
	    czz = cz*cyy/cy
	    cyy1 = cdyz/cdwz
	    czz1 = cyy1*cw/cy
	    cdyyzz = cdyz*cyy/cy
	    if ( DBLE(cy) .gt. 0 ) then
		ieps1 = -3*iepsz
	    else
		ieps1 = +3*iepsz
	    endif
*	    Often 2y-z-z is relevant, but 2*yy-zz-zz is not, solve by
*	    introducing zzp.
	    czzp = czp*cyy/cy
	    cd2yyz = cd2yzz*cyy/cy
	    czzp1 = 1 - czzp
	    if ( absc(czzp1) .lt. xloss ) then
*		later try more possibilities
		ld2yyz = .FALSE.
	    else
		ld2yyz = .TRUE.
	    endif
	    call ffcrr(cs3(1),ipi12(1),cyy,cyy1,czz,czz1,cdyyzz,
     +		ld2yyz,cd2yyz,czzp,czzp1,isoort,ieps1,ier)
	    czz = cyy*cz1/cy1
	    czz1 = cyy1*cw1/cy1
	    if ( DBLE(-cy1) .gt. 0 ) then
		ieps2 = -3*iepsz
	    else
		ieps2 = +3*iepsz
	    endif
	    cdyyzz = -cyy*cdyz/cy1
	    czzp = czp1*cyy/cy1
	    cd2yyz = -cd2yzz*cyy/cy1
	    czzp1 = 1 - czzp
	    if ( absc(czzp1) .lt. xloss ) then
*		later try more possibilities
		ld2yyz = .FALSE.
	    else
		ld2yyz = .TRUE.
	    endif
	    call ffcrr(cs3(8),ipi12(2),cyy,cyy1,czz,czz1,cdyyzz,
     +		.TRUE.,cd2yyz,czzp,czzp1,isoort,ieps2,ier)
	    do 20 i=8,14
		cs3(i) = -cs3(i)
   20	    continue
	    ipi12(2) = -ipi12(2)
*	    eta terms (are not calculated in ffcrr as ieps = 3)
	    cfactz = 1/cdyz
	    if ( DIMAG(cz) .eq. 0 ) then
		if ( DIMAG(cy) .eq. 0 ) then
		    n1 = 0
		    n2 = 0
		else
		    n1 = nffet1(DCMPLX(DBLE(0),DBLE(iepsz)),cfactz,
     +			-cz*cfactz,ier)
		    n2 = nffet1(DCMPLX(DBLE(0),DBLE(iepsz)),cfactz,
     +			cz1*cfactz,ier)
		endif
	    else
		n1 = nffeta(-cz,cfactz,ier)
		n2 = nffeta(cz1,cfactz,ier)
	    endif
	    cfactw = -1/cdwy
	    if ( DIMAG(cw) .eq. 0 ) then
		if ( DIMAG(cy) .eq. 0 ) then
		    n4 = 0
		    n5 = 0
		else
		    n4 = nffet1(DCMPLX(DBLE(0),DBLE(iepsw)),cfactw,
     +			-cw*cfactw,ier)
		    n5 = nffet1(DCMPLX(DBLE(0),DBLE(iepsw)),cfactw,
     +			cw1*cfactw,ier)
		endif
	    else
		n4 = nffeta(-cw,cfactw,ier)
		n5 = nffeta(cw1,cfactw,ier)
	    endif
*
*	    we assume that cs3(15-17) are not used, this is always true
*
	    n3 = 0
	    n6 = 0
	    if ( n1.eq.n4 ) then
		if ( n1.eq.0 ) then
*		    nothing to do
		else
		    cc1 = cdwz/cdyz
		    if ( absc(cc1) .lt. xloss ) then
			cs3(15) = n1*c2ipi*zfflo1(cc1,ier)
		    else
			cc1 = -cdwy/cdyz
			cs3(15) = n1*c2ipi*zfflog(cc1,0,c0,ier)
		    endif
		    cc1 = cy*cfactz
		    cc2 = cy*cfactw
		    if ( DIMAG(cc1).eq.0 .or. DIMAG(cc2).eq.0 ) then
			n3 = 0
		    else
			n3 = nffeta(cc1,1/cc2,ier)
		    endif
		    if ( n3.ne.0 ) then
			print *,'ffdcrr: error: untested algorithm'
			ier = ier + 50
			ipi12(1) = ipi12(1) + 4*12*n1*n3
		    endif
		endif
	    else
		cc1 = cy*cfactz
		cc2 = cy*cfactw
		cs3(15) = (n1*zfflog(cc1,ieps1,c0,ier) + 
     +			   n4*zfflog(cc2,ieps1,c0,ier))*c2ipi
	    endif
	    if ( n2.eq.n5 ) then
		if ( n2.eq.0 ) then
*		    nothing to do
		else
		    cc1 = cdwz/cdyz
		    if ( absc(cc1) .lt. xloss ) then
			cs3(16) = n2*c2ipi*zfflo1(cc1,ier)
		    else
			cc1 = -cdwy/cdyz
			cs3(16) = n2*c2ipi*zfflog(cc1,0,c0,ier)
		    endif
		    cc1 = -cy1*cfactz
		    cc2 = -cy1*cfactw
		    if ( DIMAG(cc1).eq.0 .or. DIMAG(cc2).eq.0 ) then
			n6 = 0
		    else
			n6 = nffeta(cc1,1/cc2,ier)
		    endif
		    if ( n6.ne.0 ) then
			print *,'ffdcrr: error: untested algorithm'
			ier = ier + 50
			ipi12(2) = ipi12(2) + 4*12*n2*n6
		    endif
		endif
	    else
		cc1 = -cy1*cfactz
		cc2 = -cy1*cfactw
		cs3(15) = (n2*zfflog(cc1,ieps2,c0,ier) + 
     +			   n5*zfflog(cc2,ieps2,c0,ier))*c2ipi
	    endif
	    if ( lwrite ) then
		print *,'  eta''s z are :',n1,n2,n3
		print *,'  eta''s w are :',n4,n5,n6
	    endif
*  #] only cancellations in cw, not in cy: 
*  #[ Hill identity:
	elseif (  ( 1.gt.xloss*absc(cy) .or. absc(cc1).gt.xloss )
     +	    .and. ( 1.gt.xloss*absc(cz) .or. absc(cz/cdyz).gt.xloss )
     +	    .and. ( 1.gt.xloss*absc(cy) .or. absc(cdyz/cy).gt.xloss )
     +		) then
*	    do a Hill identity on the cy,cy-1 direction
	    if ( lwrite ) print *,'ffdcrr: Hill identity to split cz,cw'
	    cyy = -cy*cw1/cdwy
	    cyy1 = cw*cy1/cdwy
	    czz = -cz*cw1/cdwz
	    czz1 = cw*cz1/cdwz
	    cdyyzz = -cw*cw1*(cdyz/(cdwy*cdwz))
	    ieps = -2*iepsz
	    call ffcrr(cs3(1),ipi12(1),cyy,cyy1,czz,czz1,cdyyzz,
     +		.FALSE.,c0,c0,c0,isoort,ieps,ier)
	    cyy = cw1
	    cyy1 = cw
	    czz = -cw1*cz/cdwz
	    czz1 = cw*cz1/cdwz
	    cdyyzz = cw*cw1/cdwz
	    call ffcrr(cs3(8),ipi12(2),cyy,cyy1,czz,czz1,cdyyzz,
     +		.FALSE.,c0,c0,c0,isoort,0,ier)
	    do 30 i=8,14
   30		cs3(i) = -cs3(i)
	    ipi12(2) = -ipi12(2)
*	    the extra logarithms ...
	    if ( 1 .lt. xloss*absc(cw) ) then
		chulp = zfflo1(1/cw,ier)
	    else
		chulp = zfflog(-cw1/cw,0,c0,ier)
	    endif
	    cs3(15) = -zfflo1(cdwz/cdwy,ier)*chulp
*  #] Hill identity: 
*  #[ Taylor expansion:
	else
*	    Do a Taylor expansion
	    if ( absc(cc1) .lt. xloss ) then
		if ( lwrite ) print *,'ffdcrr: Taylor expansion, normal'
		cd3 = cdwz/cdwy
*		isign = 1
		cc1n = cc1
		cd2n = cd2
		cd3n = cd3
		cd21 = 1-cd2
		cd21n1 = 1
		cd31 = 1-cd3
		cd31n1 = 1
		ctot = cc1*cd2*cd3
		do 50 i=2,20
		    cc1n = cc1n*cc1
		    cd21n1 = cd21n1*cd21
		    cd31n1 = cd31n1*cd31
		    cd2n = cd2n + cd2*cd21n1
		    cd3n = cd3n + cd3*cd31n1
		    cterm = cc1n*cd2n*cd3n*DBLE(xn2inv(i))
		    ctot = ctot + cterm
		    if ( absc(cterm) .lt. precc*absc(ctot) ) goto 51
   50		continue
		if ( lwarn ) call ffwarn(45,ier,absc(ctot),absc(cterm))
   51		continue
*		if ( isign .eq. 1 ) then
		    cs3(1) = ctot
*		else
*		    cs3(1) = -ctot
*		endif
	    elseif ( absc(cz/cdyz) .lt. xloss ) then
		if ( lwrite ) print *,'ffdcrr: Normal case'
		inorm = 1
		call ffcrr(cs3(1),ipi12(1),cy,cy1,cz,cz1,cdyz,.TRUE.,
     +			cd2yzz,czp,czp1,isoort,iepsz,ier)
		call ffcrr(cs3(8),ipi12(2),cy,cy1,cw,cw1,-cdwy,.TRUE.,
     +			cd2yww,cwp,cwp1,isoort,iepsw,ier)
		do 110 i=8,14
  110			cs3(i) = -cs3(i)
		ipi12(2) = -ipi12(2)
*		if ( lwrite ) print *,'ffdcrr: Taylor expansion, 1-x'
*		print *,'NOT YET READY !!'
*		ier = ier + 100
*		cyy = cy1*cdwz/(cz1*cdwy)
*		if ( absc(cyy) .lt. xloss ) then
*		    cs3(10) = -zfflo1(1/cy,ier)*zfflo1(cyy,ier)
*		else
*		    cyy1 = -cw1*cdyz/(cz1*cdwy)
*		    cs3(10) = -zfflo1(1/cy,ier)*zfflog(cyy1,0,cy,ier)
*		endif
*		cs3(11) = -zfflo1(1/cz,ier)*zfflo1(cdwz/cdwy,ier)
*		cyy = cdwz/(cw*cz1)
*		if ( absc(cyy) .lt. xloss ) then
*		    cs3(12) = -zfflo1(cw/cdwy,ier)*zfflo1(cyy,ier)
*		else
*		    cyy1 = cz*cw1/(cw*cz1)
*		    cs3(12) = -zfflo1(cw/cdwy,ier)*zfflog(cyy1,0,c0,ier)
*		endif
*		isign = -1
*		cc1 = -cz/cdyz
*		cd2 = 1/cz
*		cd3 = cdwz/cdwy
	    else
		if ( lwrite ) print *,'ffdcrr: Taylor expansion, 1/x'
		call fferr(20,ier)
		return
	    endif
	endif
*  #] Taylor expansion: 
*  #[ debug output:
        if ( lwrite ) then
	    csum = 0
	    do 900 i=1,20
		csum = csum + cs3(i)
		print '(i2,2g16.8)',i,cs3(i)
  900	    continue
	    print '(a)','---------------------------------'
	    print '(2x,2g16.8,2i3)',csum,ipi12
	    print '(a,2g16.8)','= ',csum+(ipi12(1)+ipi12(2))*DBLE(pi12)
	    print '(a,i3)','ier = ',ier
	    if ( inorm .eq. 0 ) then
		lwrite = .FALSE.
		ier1 = 0
		ier2 = 0
		do 905 i=1,14
  905			cs3p(i) = 0
		ipi12p(1) = 0
		ipi12p(2) = 0
		call ffcrr(cs3p(1),ipi12p(1),cy,cy1,cz,cz1,cdyz,
     +			.TRUE.,cd2yzz,czp,czp1,isoort,iepsz,ier1)
		call ffcrr(cs3p(8),ipi12p(2),cy,cy1,cw,cw1,-cdwy,
     +			.TRUE.,cd2yww,cwp,cwp1,isoort,iepsw,ier2)
		csum1 = 0
		do 910 i=1,7
  910			csum1 = csum1 + cs3p(i)
		csum2 = 0
		do 920 i=8,14
  920			csum2 = csum2 - cs3p(i)
		print *,'cmp with:'
		print '(i2,2g16.8,i2)',1,csum1,ier1
		print '(i2,2g16.8,i2)',2,csum2,ier2
		print *,'------------------+'
		print '(2x,2g16.8,3i3)',csum1+csum2,ipi12p,
     +			max(ier1,ier2)
		print '(a,2g16.8,3i3)','= ',csum1+csum2+
     +			(ipi12p(1)-ipi12p(2))*DBLE(pi12)
		lwrite = .TRUE.
	    endif
	endif
*  #] debug output: 
*###] ffdcrr:
	end
