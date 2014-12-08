*###[ ffcxyz:
	subroutine ffcxyz(cy,cz,cdyz,cd2yzz,ivert,sdelpp,sdelps,etalam,
     +			etami,delps,xpi,piDpj,isoort,ldel2s,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*	cz(1,2) = (-p(ip1).p(is2) +/- sdelpp)/xpi(ip1)			*
*	cy(1,2) = (-p(ip1).p(is2) +/- sdisc)/xpi(ip1)			*
*			disc = slam1 + 4*eta*xpi(ip)/slam		*
*									*
*	cy(3,4) = 1-cy(1,2)						*
*	cz(3.4) = 1-cz(1,2)						*
*	cdyz(i,j) = cy(i) - cz(j)					*
*									*
*	Input:	ivert		(integer)	1,2 of 3		*
*		sdelpp		(real)		sqrt(lam(p1,p2,p3))/2	*
*		sdelps		(real)		sqrt(-lam(p,ma,mb))/2	*
*		etalam		(real)		det(si.sj)/det(pi.pj)	*
*		etami(6)	(real)		si.si - etalam		*
*		xpi(ns)		(real)		standard		*
*		piDpj(ns,ns)	(real)		standard		*
*		ns		(integer)	dim of xpi,piDpj	*
*									*
*	Output:	cy(4),cz(4),cdyz(4,4)	(complex)	see above	*
*									*
*	Calls:	??							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ivert,isoort(2),ns,ier
	logical ldel2s
	DOUBLE COMPLEX cy(4),cz(4),cdyz(2,2),cd2yzz
	DOUBLE PRECISION sdelpp,sdelps,etalam,etami(6),delps,xpi(ns),
     +		piDpj(ns,ns)
*
*	local variables:
*
	integer i,j,ip1,ip2,ip3,is1,is2,is3
	DOUBLE COMPLEX c
	DOUBLE PRECISION absc,y(4)
	DOUBLE PRECISION delps1,disc,hulp,xlosn
*
*	common blocks:
*
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ set up pointers:
	if ( ldel2s .and. ivert .ne. 1 ) goto 100
	is1 = ivert
	is2 = ivert+1
	if ( is2 .eq. 4 ) is2 = 1
	is3 = ivert-1
	if ( is3 .eq. 0 ) is3 = 3
	ip1 = is1 + 3
*	ip2 = is2 + 3
*	ip3 = is3 + 3
	isoort(1) = -10
	isoort(2) = -10
*  #] set up pointers:
*  #[ test input:
	if ( ltest .and. xpi(ip1) .eq. 0 ) then
	    call fferr(47,ier)
	    return
	endif
*  #] test input:
*  #[ get cypm,czpm:
	hulp = sdelps/xpi(ip1)
	cz(1) = DCMPLX(piDpj(ip1,is2)/xpi(ip1),-hulp)
	cz(2) = DCMPLX(piDpj(ip1,is2)/xpi(ip1),+hulp)
	disc = delps/sdelpp
	call ffroot(y(1),y(2),xpi(ip1),piDpj(ip1,is2),etami(is2),disc,
     +								ier)
	cy(1) = y(1)
	cy(2) = y(2)
*  #] get cypm,czpm:
*  #[ get cypm1,czpm1:
	if ( xpi(is1) .eq. xpi(is2) ) then
	    cy(4) = cy(1)
	    cy(3) = cy(2)
	    cz(4) = cz(1)
	    cz(3) = cz(2)
	else
	    cz(3) = 1 - cz(1)
	    cz(4) = 1 - cz(2)
	    if ( absc(cz(3)).lt.xloss .or. absc(cz(4)).lt.xloss ) then
		cz(3) =DCMPLX(-piDpj(ip1,is1)/xpi(ip1),+hulp)
		cz(4) =DCMPLX(-piDpj(ip1,is1)/xpi(ip1),-hulp)
	    endif
	    y(3) = 1 - y(1)
	    y(4) = 1 - y(2)
	    if ( abs(y(3)) .lt. xloss .or. abs(y(4)) .lt. xloss ) then
		call ffroot(y(4),y(3),xpi(ip1),-piDpj(ip1,is1),
     +						etami(is1),disc,ier)
	    endif
	    cy(3) = y(3)
	    cy(4) = y(4)
	endif
*  #] get cypm1,czpm1:
*  #[ get cdypzp, cdypzm:
	cdyz(2,1) = DCMPLX(disc/xpi(ip1),+hulp)
	cdyz(2,2) = DCMPLX(disc/xpi(ip1),-hulp)
	cdyz(1,1) = -cdyz(2,2)
	cdyz(1,2) = -cdyz(2,1)
	cd2yzz = 2*disc/xpi(ip1)
	goto 200
*  #] get cdypzp, cdypzm:
*  #[ special case, get indices:
  100	continue
	if ( ivert.eq.2 ) then
	    is1 = 2
	    ip1 = 5
	else
	    is1 = 1
	    ip1 = 6
	endif
	isoort(1) = -100
	isoort(2) = -100
*  #] special case, get indices:
*  #[ get cypm,czpm:
*
*	special case del2s = 0, hence the roots are not the real roots
*	but z_2'' = (z_2'-1)/delta, z''_3 = -z'_3/delta
*
	hulp = sdelps/xpi(3)
	disc = delps/sdelpp
	if ( ivert .eq. 3 ) then
	    hulp = -hulp
	    disc = -disc
	endif
	cz(1) = DCMPLX(piDpj(is1,3)/xpi(3),-hulp)
	cz(2) = DCMPLX(piDpj(is1,3)/xpi(3),+hulp)
	call ffroot(y(1),y(2),xpi(3),piDpj(is1,3),etami(is1),disc,ier)
	cy(1) = y(1)
	cy(2) = y(2)
*  #] get cypm,czpm:
*  #[ get cypm1,czpm1:
	cz(3) = 1 - cz(1)
	cz(4) = 1 - cz(2)
	if ( absc(cz(3)).lt.xloss .or. absc(cz(4)).lt.xloss ) then
	    if ( lwrite ) print *,'cz(3,4) = ',cz(3),cz(4)
	    if ( ivert.eq.2 ) then
		cz(3) =DCMPLX(piDpj(ip1,3)/xpi(3),+hulp)
		cz(4) =DCMPLX(piDpj(ip1,3)/xpi(3),-hulp)
	    else
		cz(3) =DCMPLX(-piDpj(ip1,3)/xpi(3),+hulp)
		cz(4) =DCMPLX(-piDpj(ip1,3)/xpi(3),-hulp)
	    endif
	    if ( lwrite ) print *,'cz(3,4)+= ',cz(3),cz(4)
	endif
	y(3) = 1 - y(1)
	y(4) = 1 - y(2)
	if ( abs(y(3)) .lt. xloss .or. abs(y(4)) .lt. xloss ) then
	    if ( lwrite ) print *,'y(3,4) = ',y(3),y(4)
	    if ( ivert .eq. 2 ) then
		call ffroot(y(4),y(3),xpi(3),piDpj(ip1,3),etami(ip1),
     +							disc,ier)
	    else
		call ffroot(y(4),y(3),xpi(3),-piDpj(ip1,3),etami(ip1),
     +							disc,ier)
	    endif
	    if ( lwrite ) print *,'y(3,4)+= ',y(3),y(4)
	endif
	cy(3) = y(3)
	cy(4) = y(4)
*  #] get cypm1,czpm1:
*  #[ get cdypzp, cdypzm:
	cdyz(2,1) = DCMPLX(disc/xpi(3),+hulp)
	cdyz(2,2) = DCMPLX(disc/xpi(3),-hulp)
	cdyz(1,1) = -cdyz(2,2)
	cdyz(1,2) = -cdyz(2,1)
	cd2yzz = 2*disc/xpi(3)
*  #] get cdypzp, cdypzm:
*  #[ test output:
  200	continue
	if ( ltest ) then
	    xlosn = xloss**2*DBLE(10)**(-mod(ier,50))
	    do 99 i=1,2
		if ( xlosn*absc(cy(i)+cy(i+2)-1) .gt. precc*max(absc(
     +		    cy(i)),absc(cy(i+2)),x1)) print *,'ffcxyz: error: ',
     +		    'cy(',i+2,')<>1-cy(',i,'):',cy(i+2),cy(i),cy(i+2)+
     +		    cy(i)-1
		if ( xlosn*absc(cz(i)+cz(i+2)-1) .gt. precc*max(absc(
     +		    cz(i)),absc(cz(i+2)),x1)) print *,'ffcxzz: error: ',
     +		    'cz(',i+2,')<>1-cz(',i,'):',cz(i+2),cz(i),cz(i+2)+
     +		    cz(i)-1
		do 98 j=1,2
		    if ( xlosn*absc(cdyz(i,j)-cy(i)+cz(j)) .gt. precc*
     +			max(absc(cdyz(i,j)),absc(cy(i)),absc(cz(j))) )
     +			print *,'ffcxyz: error: cdyz(',i,j,') <> cy(',i,
     +			')-cz(',j,'):',cdyz(i,j),cy(i),cz(j),cdyz(i,j)-
     +			cy(i)+cz(j)
   98		continue
   99	    continue
	endif
*  #] test output:
*###] ffcxyz:
	end
*###[ ffcdwz:
	subroutine ffcdwz(cdwz,cw,cz,i1,j1,l,calpha,calph1,cpi,cdpipj,
     +					cpiDpj,csdeli,csdel2,ns,ier)
***#[*comment:***********************************************************
*									*
*	Recalculate cdwz(i1,j1) = cw(i1) - cz(j1)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer i1,j1,l,ns,ier
	DOUBLE COMPLEX cdwz(2,2),cw(4),cz(4),calpha,calph1,cpi(ns)
	DOUBLE COMPLEX cdpipj(ns,ns),cpiDpj(ns,ns),csdeli(3),csdel2
*
*	local variables:
*
	integer i,n
	DOUBLE COMPLEX cs(8),csum,cfac,c,cddel
	DOUBLE PRECISION xmax,absc,afac
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ calculations:
	if ( l .eq. 1 ) then
	    if ( j1 .eq. 1 ) then
		if ( absc(csdeli(1)+csdel2) .lt. xloss*absc(csdel2) )
     +								then
*		    for example in e-> e g* with eeg loop
*		    first get the difference of csdeli(1) and csdel2:
		    cs(1) = cpi(4)*cdpipj(2,5)
		    cs(2) = -cpiDpj(4,3)*cpiDpj(4,2)
		    cs(3) = cpiDpj(4,3)*cpiDpj(4,5)
		    csum = cs(1)+cs(2)+cs(3)
		    xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)))
		    if ( absc(csum) .lt. xloss*xmax ) then
			if ( lwrite ) print *,'ffcdwz: canc in cddel'
			ier = 1
			goto 5
		    endif
		    cddel = csum/(csdel2-csdeli(1))
		    if ( i1 .eq. 1 ) then
			cs(1) = cpi(4)*csdeli(2)
		    else
			cs(1) = -cpi(4)*csdeli(2)
		    endif
		    cs(2) = cddel*cpiDpj(4,2)
		    cs(3) = -cpiDpj(4,3)*csdeli(1)
		    cs(4) = cpiDpj(4,3)*cpiDpj(4,5)
		    cs(5) = -cpi(4)*cpiDpj(5,3)
		    cs(6) = -cddel*csdel2
		    n = 6
		else
		    if ( lwrite ) print *,'ffcdwz: ',
     +			'cannot handle this case yet'
		    ier = ier + 100
		    goto 5
		endif
		csum = 0
		xmax = 0
		do 1 i=1,n
		    csum = csum + cs(i)
		    xmax = max(xmax,absc(cs(i)))
    1		continue
		if ( absc(csum) .lt. xloss*xmax ) then
		    if ( lwrite ) print *,'ffcdwz: still cancellations',
     +			' in cdwz(',i1,j1,l,'): ',csum,xmax
		    ier = ier + 1
		endif
		if (lwrite) print *,'  cdwz(',i1,j1,l,')  =',cdwz(i1,j1)
     +				,min(absc(cw(i1)),absc(cw(i1+2)))
		cdwz(i1,j1) = csum/calph1/cpi(4)/cpi(5)
		if ( cdwz(i1,j1) .eq. 0 .and. csum .ne. 0 ) then
		    print *,'?#$&!! cdwz = 0 but csum != 0, try again'
		    afac = 1/absc(csum)
		    csum = csum*DBLE(afac)
		    cdwz(i1,j1) = csum/calph1/cpi(4)/cpi(5)
		    afac = 1/afac
		    cdwz(i1,j1) = cdwz(i1,j1)*DBLE(afac)
		endif
		if (lwrite) print *,'  cdwz(',i1,j1,l,')+ =',cdwz(i1,j1)
     +				,xmax/absc(calph1*cpi(4)*cpi(5))
	    else
		if ( lwrite ) print *,'ffcdwz:  warning: cannot handle',
     +			' this case cdwz(',i1,j1,l,') yet'
		ier = ier + 100
	    endif
    5	    continue
	elseif ( l .eq. 3 ) then
	    if ( (i1.eq.2 .and. j1.eq.1) .or. (i1.eq.1 .and. j1.eq.2 ) )
     +			then
		cfac = 1/(csdeli(2) + csdeli(3))
		cs(1) = cdpipj(6,5)*cz(j1)
		cs(2) = -calph1*cpi(5)*cz(j1+2)
		if ( max(absc(cdpipj(2,1)),absc(cdpipj(5,6))) .lt.
     +		     max(absc(cdpipj(2,6)),absc(cdpipj(5,1))) ) then
		    cs(3) = cdpipj(2,1)/2
		    cs(4) = cdpipj(5,6)/2
		else
		    cs(3) = cdpipj(2,6)/2
		    cs(4) = cdpipj(5,1)/2
		endif
		cs(5) = cpiDpj(4,3)*cpiDpj(5,3)*cfac
		cs(6) = -cpiDpj(4,3)*cpiDpj(6,3)*cfac
		cs(7) = cpi(3)*cdpipj(5,6)*cfac
		if ( i1 .eq. 1 ) then
		    csum = cs(1)+cs(2)+cs(3)+cs(4) - (cs(5)+cs(6)+cs(7))
		else
		    csum = cs(1)+cs(2)+cs(3)+cs(4) + cs(5)+cs(6)+cs(7)
		endif
		xmax = absc(cs(1))
		do 10 i=2,7
		    xmax = max(xmax,absc(cs(i)))
   10		continue
		if ( absc(csum) .lt. xloss*xmax ) then
*		    this result is not used if it is not accurate (see
*		    ffxc0p)
		    if ( lwrite ) then
			call ffwarn(78,ier,absc(csum),xmax)
		    else
			ier = ier + 1
		    endif
		    xmax = xmax/absc(calpha*cpi(5))
		    if ( xmax .lt. min(absc(cz(j1)),absc(cz(j1+2))) )
     +								then
			if (lwrite) print *,'  cdwz(',i1,j1,l,')  = ',
     +			    cdwz(i1,j1),min(absc(cw(i1)),absc(cw(i1+2)))
			cdwz(i1,j1) = csum/(calpha*cpi(5))
			if (lwrite) print *,'  cdwz(',i1,j1,l,')+ = ',
     +						cdwz(i1,j1),xmax
		    endif
		else
		    if (lwrite) print *,'  cdwz(',i1,j1,l,')  = ',
     +			    cdwz(i1,j1),min(absc(cw(i1)),absc(cw(i1+2)))
		    cdwz(i1,j1) = csum/(calpha*cpi(5))
		    if (lwrite) print *,'  cdwz(',i1,j1,l,')+ = ',
     +				cdwz(i1,j1),xmax/absc(calpha*cpi(5))
		endif
	    else
		if ( lwrite ) print *,'ffcdwz:  warning: cannot handle',
     +			' this case cdwz(',i1,j1,l,') yet'
		ier = ier + 100
	    endif
	else
	    if ( lwrite ) print *,'ffcdwz:  error: l <> 1 or 3 but ',l
	    ier = ier + 100
	endif
*  #] calculations:
*  #[ test output:
	if ( ltest .and. ier .eq. 0 ) then
	    if ( xloss**2*absc(cdwz(i1,j1)-cw(i1)+cz(j1)) .gt. precc*
     +		max(absc(cdwz(i1,j1)),absc(cw(i1)),absc(cz(j1))) )
     +		print *,'ffcdwz:  error: cdwz(',i1,j1,l,') <> cw - cz :'
     +		,cdwz(i1,j1),cw(i1),cz(j1),cw(i1)-cz(j1),
     +		cdwz(i1,j1)-cw(i1)+cz(j1)
	    if ( xloss**2*absc(cdwz(i1,j1)+cw(i1+2)-cz(j1+2)) .gt.
     +		precc*max(absc(cdwz(i1,j1)),absc(cw(i1+2)),
     +		absc(cz(j1+2))) ) print *,'ffcdwz:  error: cdwz(',i1,j1,
     +		l,') <> cz1- cw1:',cdwz(i1,j1),cz(i1+2),cw(j1+2),
     +		cz(i1+2)-cw(j1+2),cdwz(i1,j1)+cw(i1+2)-cz(j1+2)
	    endif
*  #] test output:
*###] ffcdwz:
	end
