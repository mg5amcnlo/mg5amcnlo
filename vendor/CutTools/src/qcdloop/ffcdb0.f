*###[ ffcdb0:
	subroutine ffcdb0(cdb0,cdb0p,cp,cma,cmb,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the derivative of the two-point function with	*
*	respect to p2, plus the same times p2.				*
*									*
*	Input:	cp	(complex) k2, in B&D metric			*
*		cma	(complex) mass2					*
*		cmb	(complex) mass2					*
*									*
*	Output:	cdb0	(complex) dB0/dxp				*
*		cdb0p	(complex) cp*dB0/dxp				*
*		ier	(integer) # of digits lost, if >=100: error	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cdb0,cdb0p
	DOUBLE COMPLEX cp,cma,cmb
*
*	local variables
*
	integer ier0
	DOUBLE COMPLEX cmamb,cmap,cmbp,cc
	DOUBLE PRECISION xp,xma,xmb,absc
*
*	common
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
	    print *,'ffcdb0: input:'
	    print *,'cma,cmb,cp,ier = ',cma,cmb,cp,ier
	endif
	if ( ltest ) then
	    if ( DBLE(cma) .lt. 0 .or. DBLE(cmb) .lt. 0 ) then
		print *,'ffcdb0: error: Re(cma,b) < 0: ',cma,cmb
		stop
	    endif
	    if ( DIMAG(cma) .gt. 0 .or. DIMAG(cmb) .gt. 0 ) then
		print *,'ffcdb0: error: Im(cma,b) > 0: ',cma,cmb
		stop
	    endif
	    if ( DIMAG(cp) .ne. 0 ) then
		print *,'ffcdb0: error: Im(cp) != 0: ',cp
		stop
	    endif
	endif
	if ( DIMAG(cma).eq.0 .and. DIMAG(cmb).eq.0 ) then
	    xma = DBLE(cma)
	    xmb = DBLE(cmb)
	    xp = DBLE(cp)
	    if ( lwrite ) print *,'ffcdb0: calling real case'
	    call ffxdb0(cdb0,cdb0p,xp,xma,xmb,ier)
	    return
	endif
*  #] check input: 
*  #[ get differences:
	ier0 = 0
	cmamb = cma - cmb
	cmap = cma - cp
	cmbp = cmb - cp
	if ( lwarn ) then
	    if ( absc(cmamb) .lt. xloss*absc(cma) .and. cma .ne. cmb )
     +		call ffwarn(94,ier0,absc(cmamb),absc(cma))
	    if ( absc(cmap) .lt. xloss*absc(cp) .and. cp .ne. cma )
     +		call ffwarn(95,ier0,absc(cmap),absc(cp))
	    if ( absc(cmbp) .lt. xloss*absc(cp) .and. cp .ne. cmb )
     +		call ffwarn(96,ier0,absc(cmbp),absc(cp))
	endif
*  #] get differences: 
*  #[ calculations:
	call ffcdbp(cdb0,cdb0p,cp,cma,cmb,cmap,cmbp,cmamb,ier)
	if ( lwrite ) then
	    print *,'   B0'' = ',cdb0,ier
	    print *,'cp*B0'' = ',cdb0p,ier
	endif
*  #] calculations: 
*###] ffcdb0: 
	end
*###[ ffcdbp:
	subroutine ffcdbp(cdb0,cdb0p,cp,cma,cmb,cmap,cmbp,cmamb,ier)
***#[*comment:***********************************************************
*									*
*	calculates the derivatives of the two-point function		*
*									*
*	Input:	cp	(complex) p.p, in B&D metric			*
*		cma	(complex) mass2,				*
*		cmb	(complex) mass2,				*
*		dm[ab]p	(complex) cm[ab] - cp				*
*		cmamb	(complex) cma - cmb				*
*									*
*	Output:	cdb0	(complex) B0' = dB0/dxp				*
*		cdb0p	(complex) cp*B0'				*
*		ier	(integer) 0=ok,>0=numerical problems,>100=error	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cdb0,cdb0p
	DOUBLE COMPLEX cp,cma,cmb,cmap,cmbp,cmamb
*
*	local variables
*
	integer i,initeq,jsign,init,ithres,initir,n1,n2,nffet1
	logical lreal
	DOUBLE PRECISION ax,ffbnd,ffbndc,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515,
     +		xprec0,bdn001,bdn005,bdn010,bdn015,bdn020,
     +		absc,xmax,prcsav
	DOUBLE COMPLEX xcheck,cm,cdmp,cm1,cm2,cm1m2,cdm1p,
     +		cdm2p,s,s1,s1a,s1b,s1p,s2,s2a,s2b,s2p,s3,cx,som,
     +		clam,slam,xlogmm,alpha,alph1,xnoe,xpneq(30),
     +		zfflo1,zfflo3,d1,d2,diff,h,a,b,c,d,beta,
     +		betm2n,s1c,s1d,s1e,s1f,cqi(3),cqiqj(3,3),zm,zp
	DOUBLE COMPLEX cc
	DOUBLE PRECISION xp,xma,xmb,dmamb,dmap,dmbp,sprec
	save initeq,xpneq,init,initir,
     +		xprceq,bdeq01,bdeq05,bdeq11,bdeq17,bdeq25,
     +		xprcn3,bdn301,bdn305,bdn310,bdn315,
     +		xprcn5,bdn501,bdn505,bdn510,bdn515,
     +		xprec0,bdn001,bdn005,bdn010,bdn015,bdn020
*for ABSOFT only
*	DOUBLE COMPLEX csqrt
*	external csqrt
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
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations: 
*  #[ check input:
	if (ltest) then
	    xcheck = cma - cmb - cmamb
	    if ( absc(xcheck) .gt. precc*max(absc(cma),absc(cmb),absc(
     +			cmamb))/xloss ) then
		print *,'ffcdbp: input not OK, cmamb <> cma-cmb',xcheck
	    endif
	    xcheck = -cp + cma - cmap
	    if ( absc(xcheck) .gt. precc*max(absc(cp),absc(cma),absc(
     +			cmap))/xloss ) then
		print *,'ffcdbp: input not OK, cmap <> cma - cp',xcheck
	    endif
	    xcheck = -cp + cmb - cmbp
	    if ( absc(xcheck) .gt. precc*max(absc(cp),absc(cmb),absc(
     +			cmbp))/xloss ) then
		print *,'ffcdbp: input not OK, cmbp <> cmb - cp',xcheck
	    endif
	endif
*  #] check input: 
*  #[ the real cases:
*
	if ( DIMAG(cma) .eq. 0 .and. DIMAG(cmb) .eq. 0 ) then
	    lreal = .TRUE.
	elseif ( nschem.le.2 ) then
	    lreal = .TRUE.
	    if ( init.eq.0 ) then
		init = 1
		print *,'ffcb0: nschem <= 2, ignoring complex masses: ',
     +			nschem
	    endif
	elseif ( nschem.le.4 ) then
	    if ( init.eq.0 ) then
		init = 1
		print *,'ffcdbp: nschem = 3,4 complex masses near ',
     +			'singularity: ',nschem
	    endif
	    if ( abs(DBLE(cma)) .lt. -xloss*DIMAG(cmb)
     +		.and. abs(DBLE(cmbp)) .le. -nwidth*DIMAG(cmb)
     +	    .or. abs(DBLE(cmb)) .lt. -xloss*DIMAG(cma)
     +		.and. abs(DBLE(cmap)) .le. -nwidth*DIMAG(cma) ) then
		lreal = .FALSE.
	    else
		lreal = .TRUE.
	    endif
	elseif ( nschem.le.6 ) then
	    if ( init.eq.0 ) then
		init = 1
		print *,'ffcdbp: nschem = 5,6 complex masses near ',
     +			'threshold: ',nschem
	    endif
	    cqi(1) = cma
	    cqi(2) = cmb
	    cqi(3) = cp
	    cqiqj(1,2) = cmamb
	    cqiqj(2,1) = -cqiqj(1,2)
	    cqiqj(1,3) = cmap
	    cqiqj(3,1) = -cqiqj(1,3)
	    cqiqj(2,3) = cmbp
	    cqiqj(3,2) = -cqiqj(2,3)
	    cqiqj(1,1) = 0
	    cqiqj(2,2) = 0
	    cqiqj(3,3) = 0
	    call ffthre(ithres,cqi,cqiqj,3,1,2,3)
	    if ( ithres.eq.0 .or. ithres.eq.1 .and. nschem.eq.5 ) then
		lreal = .TRUE.
	    else
		lreal = .FALSE.
	    endif
	else
	    lreal = .FALSE.
	endif
	if ( lreal ) then
	    xp = DBLE(cp)
	    xma = DBLE(cma)
	    xmb = DBLE(cmb)
	    dmap = DBLE(cmap)
	    dmbp = DBLE(cmbp)
	    dmamb = DBLE(cmamb)
	    sprec = precx
	    precx = precc
	    if ( lwrite ) print *,'ffcdbp: to real case'
	    call ffxdbp(cdb0,cdb0p,xp,xma,xmb,dmap,dmbp,dmamb,ier)
	    precx = sprec
	    return
	endif
*
*  #] the real cases: 
*  #[ which case:
*
*	sort according to the type of masscombination encountered:
*	100: both masses zero, 200: one equal to zero, 300: both equal
*	400: rest.
*
	if ( cma .eq. 0 ) then
		if ( cmb .eq. 0 ) then
			goto 100
		endif
		cm = cmb
		cdmp = cmbp
		goto 200
	endif
	if ( cmb .eq. 0 ) then
		cm = cma
		cdmp = cmap
		goto 200
	elseif ( cmamb .eq. 0 ) then
		cm = cma
		cdmp = cmap
		goto 300
	elseif ( DBLE(cma) .gt. DBLE(cmb) ) then
		cm2 = cma
		cm1 = cmb
		cm1m2 = -cmamb
		cdm1p = cmbp
		cdm2p = cmap
	else
		cm1 = cma
		cm2 = cmb
		cm1m2 = cmamb
		cdm1p = cmap
		cdm2p = cmbp
	endif
	goto 400
*  #] which case: 
*  #[ both masses equal to zero:
  100	continue
	if ( cp.ne.0 ) cdb0 = -1/cp
	cdb0p = -1
	return
*  #] both masses equal to zero: 
*  #[ one mass equal to zero:
  200	continue
*
*	special case cp = 0
*
	if ( cp .eq. 0 ) then
	    cdb0p = 0
	    cdb0 = 1/(2*cm)
	    goto 990
*
*	special case cp = cm
*
	elseif ( cdmp.eq.0 ) then
	    if ( initir.eq.0 ) then
		initir = 1
		print *,'ffcdbd: IR divergent B0'', using cutoff ',delta
	    endif
	    if ( delta.eq.0 ) then
		call fferr(74,ier)
		cdb0p = 0
	    else
		cdb0p = -1 + log(cm/DBLE(delta))/2
	    endif
	    cdb0 = cdb0p/cp
	    goto 990
	endif
*
*	Normal case:
*
	cx = cp/cm
	ax = absc(cx)
	if ( ax .lt. xloss ) then
* 	#[ Taylor expansion:
	    if ( xprec0 .ne. precx ) then
		xprec0 = precc
		prcsav = precx
		precx = precc
		bdn001 = ffbnd(2,1,xninv)
		bdn005 = ffbnd(2,5,xninv)
		bdn010 = ffbnd(2,10,xninv)
		bdn015 = ffbnd(2,15,xninv)
		bdn020 = ffbnd(2,20,xninv)
		precx = prcsav
	    endif
	    if ( lwarn .and. ax .gt. bdn020 ) then
		call ffwarn(15,ier,precc,xninv(21)*ax**20)
	    endif
	    if ( ax .gt. bdn015 ) then
		som = cx*(DBLE(xninv(17)) + cx*(DBLE(xninv(18))
     +		    + cx*(DBLE(xninv(19)) + cx*(DBLE(xninv(20))
     +		    + cx*(DBLE(xninv(21)) )))))
	    else
		som = 0
	    endif
	    if ( ax .gt. bdn010 ) then
		som = cx*(DBLE(xninv(12)) + cx*(DBLE(xninv(13))
     +		    + cx*(DBLE(xninv(14)) + cx*(DBLE(xninv(15))
     +		    + cx*(DBLE(xninv(16)) + som )))))
	    endif
	    if ( ax .gt. bdn005 ) then
		som = cx*(DBLE(xninv(7)) + cx*(DBLE(xninv(8))
     +		    + cx*(DBLE(xninv(9)) + cx*(DBLE(xninv(10))
     +		    + cx*(DBLE(xninv(11)) + som )))))
	    endif
	    if ( ax .gt. bdn001 ) then
		som = cx*(DBLE(xninv(3)) + cx*(DBLE(xninv(4))
     +		    + cx*(DBLE(xninv(5)) + cx*(DBLE(xninv(6)) + som ))))
	    endif
	    cdb0p = cx*(DBLE(xninv(2)) + som)
	    if ( lwrite ) then
		print *,'cdb0p = ',cdb0p
		print *,'verg    ',-1 - cm/cp*zfflo1(cx,ier),1
	    endif
* 	#] Taylor expansion: 
	else
* 	#[ short formula:
	    s = log(cdmp/cm)
	    cdb0p = -(1 + s*cm/cp)
	    if ( lwarn .and. absc(cdb0p).lt.xloss ) then
		call ffwarn(13,ier,absc(cdb0p),x1)
	    endif
* 	#] short formula: 
	endif
	cdb0 = cdb0p/cp
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
	if ( absc(cp) .lt. 8*xloss*absc(cm) ) then
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
		bdeq01 = ffbndc(1,1,xpneq)
		bdeq05 = ffbndc(1,5,xpneq)
		bdeq11 = ffbndc(1,11,xpneq)
		bdeq17 = ffbndc(1,17,xpneq)
		bdeq25 = ffbndc(1,25,xpneq)
	    endif
*--#]	    data and bounds: 
	    cx = -cp/cm
	    ax = absc(cx)
	    if ( lwarn .and. ax .gt. bdeq25 ) then
		call ffwarn(13,ier,precc,abs(xpneq(25))*ax**25)
	    endif
	    if ( ax .gt. bdeq17 ) then
		som = cx*(xpneq(18) + cx*(xpneq(19) + cx*(xpneq(20) +
     +		cx*(xpneq(21) + cx*(xpneq(22) + cx*(xpneq(23) +
     +		cx*(xpneq(24) + cx*(xpneq(25) ))))))))
	    else
		som = 0
	    endif
	    if ( ax .gt. bdeq11 ) then
		som = cx*(xpneq(12) + cx*(xpneq(13) + cx*(xpneq(14) +
     +		cx*(xpneq(15) + cx*(xpneq(16) + cx*(xpneq(17) + som ))))
     +		))
	    endif
	    if ( ax .gt. bdeq05 ) then
		som = cx*(xpneq(6) + cx*(xpneq(7) + cx*(xpneq(8) + cx*(
     +		xpneq(9) + cx*(xpneq(10) + cx*(xpneq(11) + som ))))))
	    endif
	    if ( ax .gt. bdeq01 ) then
		som = cx*(xpneq(2) + cx*(xpneq(3) + cx*(xpneq(4) + cx*(
     +		xpneq(5) + som ))))
	    endif
	    cdb0p = -cx*(xpneq(1)+som)
	    if (lwrite) then
		print *,'ffcdbp: m1 = m2, Taylor expansion in ',cx
		print *,'cdb0p = ',cdb0p
	    endif
	    if ( cp.ne.0 ) then
		cdb0 = cdb0p*(1/DBLE(cp))
	    else
		cdb0 = xpneq(1)/cm
	    endif
	    goto 990
* -#]	    taylor expansion: 
	endif
* -#[	normal case:
*
*	normal case
*
	if ( lwrite ) print*,'ffcdb0: equal masses, normal case'
	call ffclmb(clam,-cp,-cm,-cm,cdmp,cdmp,c0,ier)
	slam = sqrt(clam)
	call ffcoot(zm,zp,c1,c05,cm/cp,slam/(2*cp),ier)
	if ( lwrite ) print *,' zm,zp = ',zm,zp
	s1 = zp/zm
	if( abs(s1-1) .lt. xloss ) then
*	    In this case a quicker and more accurate way is to
*	    calculate log(1-cx).
	    print *,'Not tested, probably wrong'
	    ier = ier + 50
	    if ( lwrite ) print *,' arg log1 = ',1-s1
	    s2 = (cp - slam)
	    if ( lwrite ) print *,' arg log1+= ',-2*slam/s2
	    if ( absc(s2) .lt. xloss*absc(cp) ) then
		s2 = -slam*(cp+slam)/(4*cp*cm)
		if ( lwrite ) print *,' arg log1*= ',s2
	    else
		s2 = -2*slam/s2
	    endif
	    s = -2*cm/slam*zfflo1(s2/(2*cm),ier)
	else
*	    finally the normal case
	    s = -2*cm/slam*log(s1)
	endif
*
*	eta terms
*
	n1 = nffet1(zp,1/zm,s1,ier)
	n2 = nffet1(-zp,-1/zm,s1,ier)
	if ( lwrite .and. (n1.ne.0 .or. n2.ne.0) ) then
	    print *,'ffcb0: eta terms: n1,n2 = ',n1,n2
	endif
	if (lwrite) print *,'s =   ',s
	if ( n1+n2 .ne. 0 ) then
	    s1 = cm/slam*c2ipi*(n1+n2)
	    s = s + s1
	    if ( lwrite ) then
		print *,'eta''s: ',s1
		print *,'sum  : ',s
	    endif
	endif
	cdb0p = s - 1
	cdb0 = cdb0p/cp
	if ( lwarn .and. absc(cdb0p).lt.xloss )
     +		call ffwarn(233,ier,absc(cdb0),x1)
	goto 990
* -#]	normal case: 
*
*  #] both masses equal: 
*  #[ unequal nonzero masses:
  400	continue
* -#[	get log(cm2/cm1):
	cx = cm2/cm1
	c = cx-1
	if ( 1 .lt. xclogm*absc(cx) ) then
	    call fferr(8,ier)
	    xlogmm = 0
	elseif ( absc(c) .lt. xloss ) then
	    xlogmm = zfflo1(cm1m2/cm1,ier)
	else
	    xlogmm = log(cx)
	endif
* -#]	get log(cm2/cm1): 
* -#[	cp = 0:
*
*	first a special case
*
	if ( cp .eq. 0 ) then
*
*	    repaired 19-nov-1993, see b2.frm
*
	    s1 = cm1*cm2*xlogmm/cm1m2**3
	    s2 = (cm1+cm2)/(2*cm1m2**2)
	    s = s1 + s2
	    if ( absc(s) .lt. xloss**2*absc(s2) ) then
*
*		second try
*
		h = zfflo3(cm1m2/cm1,ier)
		s1 = -cm1*h/cm1m2**2
		s2 = 1/(2*cm1)
		s3 = cm1**2*h/cm1m2**3
		s = s1 + s2 + s3
		if ( absc(s) .lt. xloss*max(absc(s2),absc(s3)) ) then
		    call ffwarn(234,ier,absc(s),absc(s2))
		endif
	    endif
	    cdb0 = s
	    cdb0p = 0
	    goto 990
	endif
* -#]	cp = 0: 
* -#[	normal case:
*
*	proceeding with the normal case
*
	call ffclmb(clam,-cp,-cm2,-cm1,cdm2p,cdm1p,cm1m2,ier)
	diff = clam + cp*(cdm2p+cm1)
	if ( absc(diff) .lt. xloss*absc(clam) ) then
	    if ( lwrite ) print *,'diff = ',diff
	    h = cm1m2**2 - cp*(cm1+cm2)
	    if ( lwrite ) print *,'diff+= ',h
	    if ( absc(h) .lt. xloss*absc(cm1m2)**2 ) then
		if ( absc(cm1m2)**2 .lt. absc(clam) ) diff = h
		call ffwarn(235,ier,absc(diff),min(absc(cm1m2)**2,
     +			absc(clam)))
	    endif
	endif
*--#[	first try:
*	first try the normal way
	slam = sqrt(clam)
	if ( lwrite ) then
	    print *,'clam = ',clam
	    print *,'slam = ',slam
	endif
	if ( abs(DBLE(cm1)) .lt. abs(DBLE(cm2)) ) then
	    s2a = cm1 + cdm2p
	else
	    s2a = cm2 + cdm1p
	endif
	s2 = s2a + slam
	if ( absc(s2) .gt. xloss*absc(slam) ) then
*	    looks fine
	    jsign = 1
	else
	    s2 = s2a - slam
	    jsign = -1
	endif
	s2 = s2/sqrt(4*cm1*cm2)
	if ( lwrite ) print *,'  arg log s2 = ',s2
	if ( absc(s2) .lt. xclogm ) then
	    call fferr(9,ier)
	    s2 = 0
	elseif ( absc(s2-1) .lt. xloss ) then
	    ier = ier + 50
	    print *,'ffcdb0: untested: s2 better in first try'
	    if ( jsign.eq.1 ) then
		if ( lwrite ) print *,'s2 ',-diff/(2*slam*cp)*2*log(s2)
		s2 = -slam*(s2a+slam)/(2*cm1*cm2)
		s2 = -diff/(2*slam*cp)*zfflo1(s2,ier)
	    else
		if ( lwrite ) print *,'s2 ',+diff/(2*slam*cp)*2*log(s2)
		s2 = +slam*(s2a-slam)/(2*cm1*cm2)
		s2 = +diff/(2*slam*cp)*zfflo1(s2,ier)
	    endif
	    if ( lwrite ) print *,'s2+ ',s2,jsign
	else
	    s2 = -diff/(2*slam*cp)*2*log(s2)
	    if ( jsign .eq. -1 ) s2 = -s2
	endif
	s1 = -cm1m2*xlogmm/(2*cp)
	cdb0p = s1+s2-1
	if (lwrite) then
	    print *,'ffcdbp: first try, cdb0p = ',cdb0p,s1,s2,-1
	endif
*--#]	first try: 
	if ( absc(cdb0p) .lt. xloss**2*max(absc(s1),absc(s2)) ) then
*--#[		second try:
*		this is unacceptable, try a better solution
		s1a = diff + slam*cm1m2
		if (lwrite) print *,'s1 = ',-s1a/(2*cp*slam),diff/
     +			(2*cp*slam)
		if ( absc(s1a) .gt. xloss*absc(diff) ) then
*		    this works
		    s1 = -s1a/(2*cp*slam)
		else
*		    by division a more accurate form can be found
		    s1 = -2*cm1*cm2*cp/(slam*(diff - slam*cm1m2))
		    if (lwrite) print *,'s1+= ',s1
		endif
		s = s1
		s1 = s1*xlogmm
		if ( abs(DBLE(cp)).lt.abs(DBLE(cm2)) ) then
		    s2a = cp - cm1m2
		else
		    s2a = cm2 - cdm1p
		endif
		s2 = s2a - slam
		if (lwrite) print *,'s2 = ',s2/(2*cm2),slam/(2*cm2)
		if ( absc(s2) .gt. xloss*absc(slam) ) then
*		    at least reasonable
		    s2 = s2 / (2*cm2)
		else
*		    division again
		    s2 = (2*cp) / (s2a+slam)
		    if (lwrite) print *,'s2+= ',s2
		endif
		if ( absc(s2) .lt. .1 ) then
*		    choose a quick way to get the logarithm
		    s2 = zfflo1(s2,ier)
		else
		    s2 = log(1-s2)
		endif
		s2 = -diff/(slam*cp)*s2
		cdb0p = s1 + s2 - 1
		if (lwrite) then
		    print *,'ffcdbp: 2nd try, cdb0p  = ',cdb0p,s1,s2,-1
		endif
*--#]		second try: 
		if ( absc(cdb0p) .lt. xloss**2*max(absc(s1),absc(s2)) )
     +			then
*--#[		    third try:
*		    (we accept two times xloss because that's the same
*		    as in this try)
*		    A Taylor expansion might work.  We expand
*		    inside the logs. Only do the necessary work.
*
*		#[ split up 1:
		    xnoe = s2a+slam
		    a = 1
		    b = 2/xnoe-1/cp
		    c = -4/(cp*xnoe)
		    d = sqrt(cp**(-2) + (2/xnoe)**2)
		    call ffcoot(d1,d2,a,b,c,d,ier)
		    if ( DBLE(cp).gt.0 ) then
			beta = d2
		    else
			beta = d1
		    endif
		    alpha = beta*diff/slam
		    alph1 = 1-alpha
		    if ( absc(alph1) .lt. xloss ) then
			s1a = 4*cp**2*cm1*cm2/(slam*cm1m2*(diff-slam*
     +				cm1m2))
			s1b = -diff/slam*4*cm1*cp/(cm1m2*xnoe*(2*cp-
     +				xnoe))
			b = -1/cp
			c = -(2/xnoe)**2
			call ffcoot(d1,d2,a,b,c,d,ier)
			if ( DBLE(cp).gt.0 ) then
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
			xmax = max(absc(s1a),absc(s1b))
			if ( xmax .lt. 1 ) then
			    alph1 = d1
			else
			    xmax = 1
			endif
			if ( absc(alph1) .lt. xloss*xmax )
     +				call ffwarn(236,ier,absc(alph1),xmax)
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
  490		    s2p = s2 - alpha
		    if ( absc(s2p) .lt. xloss*absc(s2) ) then
* -#[			bounds:
*			determine the boundaries for 1,5,10,15 terms
			if ( xprcn5 .ne. precx ) then
			    xprcn5 = precc
			    prcsav = precx
			    precx = precc
			    bdn501 = ffbnd(3,1,xinfac)
			    bdn505 = ffbnd(3,5,xinfac)
			    bdn510 = ffbnd(3,10,xinfac)
			    bdn515 = ffbnd(3,15,xinfac)
			    precx = prcsav
			endif
* -#]			bounds: 
			cx = beta*cp
			ax = absc(cx)
			if ( lwarn .and. ax .gt. bdn515 ) then
			    call ffwarn(13,ier,absc(s2p),absc(s2))
			    goto 495
			endif
			if ( ax .gt. bdn510 ) then
			   s2a = cx*(DBLE(xinfac(13)) + cx*(DBLE(xinfac(
     +			   14))+ cx*(DBLE(xinfac(15)) + cx*(DBLE(xinfac(
     +			   16))+ cx*(DBLE(xinfac(17)))))))
			else
			    s2a = 0
			endif
			if ( ax .gt. bdn505 ) then
			   s2a = cx*(DBLE(xinfac( 8)) + cx*(DBLE(xinfac(
     +			    9))+ cx*(DBLE(xinfac(10)) + cx*(DBLE(xinfac(
     +			   11))+ cx*(DBLE(xinfac(12)) + s2a)))))
			endif
			if ( ax .gt. bdn501 ) then
			    s2a =cx*(DBLE(xinfac(4))+cx*(DBLE(xinfac(5))
     +				+cx*(DBLE(xinfac(6))+cx*(DBLE(xinfac(7))
     +				+ s2a))))
			endif
			s2a = cx**3*(DBLE(xinfac(3))+s2a)
			s2b = 2*cp/xnoe*(s2a + cx**2/2)
			s2p = s2b - s2a
			if ( lwarn .and. absc(s2p).lt.xloss*absc(s2a) )
     +				call ffwarn(237,ier,absc(s2p),absc(s2a))
			s2p = -diff/(cp*slam)*zfflo1(s2p,ier)
			if (lwrite) then
			    print *,'ffcdbp: Taylor expansion of s2-a'
			    print *,'	     in cx = ',cx
			    print *,'	     gives s2p = ',s2p
			endif
		    endif
*		#] s2: 
*		#[ s1:
*
*		    next s1:
*
  495		    s1p = s1 - alph1
		    if ( absc(s1p) .lt. xloss*absc(s1) ) then
* -#[			bounds:
*			determine the boundaries for 1,5,10,15 terms
			if ( xprcn3 .ne. precx ) then
			    xprcn3 = precc
			    prcsav = precx
			    precx = precc
			    bdn301 = ffbnd(3,1,xinfac)
			    bdn305 = ffbnd(3,5,xinfac)
			    bdn310 = ffbnd(3,10,xinfac)
			    bdn315 = ffbnd(3,15,xinfac)
			    precx = prcsav
			endif
* -#]			bounds: 
*
			cx = slam*(diff-slam*cm1m2)*alph1/(2*cp*cm1*cm2)
			ax = absc(cx)
			if ( lwarn .and. ax .gt. bdn315 ) then
			    call ffwarn(238,ier,absc(s1p),absc(s1))
			    goto 496
			endif
			h = (2*cp*(cm1+cm2) - cp**2)/(slam-cm1m2)
*
*			see form job gets1.frm
*
			s1b = diff*(diff-slam*cm1m2)*betm2n/(2*cp*cm1*
     +				cm2)
			s1c = 1/(cm1*xnoe*(2*cp-xnoe))*(
     +				cp*( 4*cp*cm2 + 2*cm1m2**2/cm2*(cp-h) +
     +				2*cm1m2*(3*cp-h) - 8*cm1m2**2 )
     +				- 2*cm1m2**3/cm2*(3*cp-h)
     +				+ 4*cm1m2**4/cm2
     +				)
			if ( lwrite ) then
			    print *,'s1c was ',-2*cp/cm1m2 + 2*diff*
     +				(diff-slam*cm1m2)/(cm2*cm1m2*xnoe*(2*cp-
     +				xnoe)) + cm1m2/cm1
			    print *,'  en is ',s1c
			    print *,'s1b+s1c was ',cm1m2/cm1-cx
			    print *,'      en is ',s1b+s1c
			endif
			s1d = cx*cm1m2/cm1
			s1e = -cx**2/2
			if ( ax .gt. bdn310 ) then
			   s1a = cx*(DBLE(xinfac(13)) + cx*(DBLE(xinfac(
     +			   14))+ cx*(DBLE(xinfac(15)) + cx*(DBLE(xinfac(
     +			   16))+ cx*(DBLE(xinfac(17)))))))
			else
			    s1a = 0
			endif
			if ( ax .gt. bdn305 ) then
			   s1a = cx*(DBLE(xinfac( 8)) + cx*(DBLE(xinfac(
     +			    9))+ cx*(DBLE(xinfac(10)) + cx*(DBLE(xinfac(
     +			   11))+ cx*(DBLE(xinfac(12)) + s1a)))))
			endif
			if ( ax .gt. bdn301 ) then
			    s1a =cx*(DBLE(xinfac(4))+cx*(DBLE(xinfac(5))
     +				+cx*(DBLE(xinfac(6))+cx*(DBLE(xinfac(7))
     +				+s1a))))
			endif
			s1a = -cx**3 *(DBLE(xinfac(3)) + s1a)
			s1f = cm1m2/cm1*(cx**2/2 - s1a)
			s1p = s1e + s1d + s1c + s1b + s1a + s1f
			xmax = max(absc(s1a),absc(s1b),absc(s1c),
     +				absc(s1d),absc(s1e))
			if ( lwarn .and. absc(s1p).lt.xloss*xmax ) then
			    call ffwarn(239,ier,absc(s1p),xmax)
			endif
			s1p = s*zfflo1(s1p,ier)
			if (lwrite) then
			    print *,'s1a = ',s1a
			    print *,'s1b = ',s1b
			    print *,'s1c = ',s1c
			    print *,'s1d = ',s1d
			    print *,'s1e = ',s1e
			    print *,'s1f = ',s1f
			    print *,'s   = ',s
			    print *,'ffcdbp: Taylor exp. of s1-(1-a)'
			    print *,'        in cx = ',cx
			    print *,'        gives s1p = ',s1p
			    print *,'        verg        ',s*log(cm2/cm1
     +				*exp(cx))
			endif
		    endif
*		#] s1: 
*
*		    finally ...
*
  496		    cdb0p = s1p + s2p
		    if ( lwarn .and. absc(cdb0p) .lt. xloss*absc(s1p) )
     +				then
			call ffwarn(240,ier,absc(cdb0p),absc(s1p))
		    endif
*--#]		    third try: 
		endif
	endif
	cdb0 = cdb0p*(1/DBLE(cp))
	goto 990
* -#]	normal case: 
*  #] unequal nonzero masses: 
*  #[ debug:
  990	continue
	if (lwrite) then
	    print *,'cdb0   = ',cdb0
	    print *,'cdb0p  = ',cdb0p
	endif
*  #] debug: 
*###] ffcdbp: 
	end
