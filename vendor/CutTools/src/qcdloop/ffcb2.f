*###[ ffcb2:
	subroutine ffcb2(cb2p,cb2d,cb1,cb0,ca0i,xp,xma,xmb,piDpj,ier)
***#[*comment:***********************************************************
*									*
*	Calculate    1   /    d^n Q Q(mu) Q(nu)				*
*		  ------ | ------------------------			*
*		  i pi^2 / (Q^2-ma^2)((Q+p)^2-mb^2)			*
*						 p mu			*
*		    = B2p*p(mu)*p(nu) + B2d*delta    /p^2		*
*						 p nu			*
*									*
*	Input:	cb1	   complex	vector twopoint function	*
*		cb0	   complex	scalar twopoint function	*
*		ca0i(2)	   complex	scalar onepoint function with	*
*						m1,m2			*
*		xp	   complex	p.p in B&D metric		*
*		xma,2	   complex	m_1^2,m_2^2			*
*		piDpj(3,3) complex	dotproducts between s1,s2,p	*
*		ier	   integer	digits lost so far		*
*	Output:	cb2p	   complex	coefficient of p(mu)*p(nu)	*
*		cb2d	   complex	coefficient of delta()/p^2	*
*		ier	   integer	digits lost			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX xp,xma,xmb,piDpj(3,3)
	DOUBLE COMPLEX cb2p,cb2d,cb1,cb0,ca0i(2)
*
*	local variables
*
	integer ier0,i,j
	DOUBLE COMPLEX dmap,dmbp,dmamb,cc
	DOUBLE PRECISION absc
	DOUBLE PRECISION rm1,rm2,rp,rpiDpj(3,3),sprec
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
	if ( DIMAG(xma).eq.0 .and. DIMAG(xmb).eq.0 ) then
	    rm1 = DBLE(xma)
	    rm2 = DBLE(xmb)
	    rp  = DBLE(xp)
	    do 20 j=1,3
		do 10 i=1,3
		    rpiDpj(i,j) = DBLE(piDpj(i,j))
   10		continue
   20	    continue
	    sprec = precx
	    precx = precc
	    call ffxb2(cb2p,cb2d,cb1,cb0,ca0i,rp,rm1,rm2,rpiDpj,
     +	    	ier)
	    precx = sprec
	    return
	endif
*  #] real case:
*  #[ get differences:
	ier0 = 0
	dmamb = xma - xmb
	dmap = xma - xp
	dmbp = xmb - xp
	if ( lwarn ) then
	    if ( absc(dmamb) .lt. xloss*absc(xma) .and. xma .ne. xmb )
     +		call ffwarn(97,ier0,absc(dmamb),absc(xma))
	    if ( absc(dmap) .lt. xloss*absc(xp) .and. xp .ne. xma )
     +		call ffwarn(98,ier0,absc(dmap),absc(xp))
	    if ( absc(dmbp) .lt. xloss*absc(xp) .and. xp .ne. xmb )
     +		call ffwarn(99,ier0,absc(dmbp),absc(xp))
	endif
*  #] get differences:
*  #[ call ffcb2a:
	call ffcb2a(cb2p,cb2d,cb1,cb0,ca0i,xp,xma,xmb,dmap,dmbp,dmamb,
     +		piDpj,ier)
*  #] call ffcb2a:
*###] ffcb2:
	end
*###[ ffcb2a:
	subroutine ffcb2a(cb2p,cb2d,cb1,cb0,ca0i,xp,xma,xmb,
     +		dmap,dmbp,dmamb,piDpj,ier)
***#[*comment:***********************************************************
*									*
*	see ffcb2, plus differences.					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX xp,xma,xmb,dmap,dmbp,dmamb,piDpj(3,3)
	DOUBLE COMPLEX cb2p,cb2d,cb1,cb0,ca0i(2)
*
*	local variables
*
	integer i,j,ier0,init,ithres
	logical llogmm,lreal
	DOUBLE PRECISION absc,xmax,xmxp,rloss
	DOUBLE PRECISION rm1,rm2,rp,rm1p,rm2p,rm1m2,rpiDpj(3,3),sprec
	DOUBLE COMPLEX delsp,xlam,xlo3,xlogmm,zfflo1,zfflo3
	DOUBLE COMPLEX cc,cs(6),cb21,cb22,csom
	DOUBLE COMPLEX cqi(3),cqiqj(3,3),qiDqj(3,3)
	save init
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
	if ( DIMAG(xma).eq.0 .and. DIMAG(xmb).eq.0 ) then
	    lreal = .TRUE.
	    if ( lwrite ) print *,'ffcb2a: real masses'
	elseif ( nschem.le.4 ) then
	    lreal = .TRUE.
	    if ( lwrite .or. init.eq.0 ) then
	    	init = 1
	    	print *,'ffcb2a: nschem <= 4, ignoring complex masses:',
     +	    		nschem
	    endif
	elseif ( nschem.le.6 ) then
	    if ( lwrite .or. init.eq.0 ) then
	    	init = 1
	    	print *,'ffcb2a: nschem = 5,6 complex masses near ',
     +	    		'threshold: ',nschem
	    endif
	    cqi(1) = xma
	    cqi(2) = xmb
	    cqi(3) = xp
	    cqiqj(1,2) = dmamb
	    cqiqj(2,1) = -cqiqj(1,2)
	    cqiqj(1,3) = dmap
	    cqiqj(3,1) = -cqiqj(1,3)
	    cqiqj(2,3) = dmbp
	    cqiqj(3,2) = -cqiqj(2,3)
	    cqiqj(1,1) = 0
	    cqiqj(2,2) = 0
	    cqiqj(3,3) = 0
	    call ffthre(ithres,cqi,cqiqj,3,1,2,3)
	    if ( ithres.eq.0 .or. ithres.eq.1 .and. nschem.eq.5 ) then
		lreal = .TRUE.
		if ( lwrite ) print *,'ffcb2a: no threshold'
	    else
		if ( lwrite ) print *,'ffcb2a: found threshold'
		lreal = .FALSE.
	    endif
	else
	    lreal = .FALSE.
	endif
	if ( lreal ) then
	    rm1 = DBLE(xma)
	    rm2 = DBLE(xmb)
	    rp  = DBLE(xp)
	    rm1p  = DBLE(dmap)
	    rm2p  = DBLE(dmbp)
	    rm1m2 = DBLE(dmamb)
	    do 20 j=1,3
		do 10 i=1,3
		    rpiDpj(i,j) = DBLE(piDpj(i,j))
   10		continue
   20	    continue
	    sprec = precx
	    precx = precc
	    call ffxb2a(cb2p,cb2d,cb1,cb0,ca0i,rp,rm1,rm2,rm1p,rm2p,
     +	    	rm1m2,rpiDpj,ier)
	    precx = sprec
	    return
	endif
*  #] real case:
*  #[ test input:
	if ( ltest ) then
	    ier0 = ier
	    call ffcot2(qiDqj,xp,xma,xmb,dmap,dmbp,dmamb,ier0)
	    rloss = xloss*DBLE(10)**(-mod(ier0,50))
	    do 40 j=1,3
		do 30 i=1,3
		    if ( rloss*absc(piDpj(i,j)-qiDqj(i,j)).gt.precc*
     +		    	absc(qiDqj(i,j)) ) print *,'ffcb2a: ',
     +			'error: piDpj(',i,j,') wrong: ',piDpj(i,j),
     +			qiDqj(i,j),piDpj(i,j)-qiDqj(i,j),ier0
   30		continue
   40	    continue
	endif
*  #] test input:
*  #[ p^2 != 0:
	if ( xp .ne. 0 ) then
* 	#[ normal case:
	    call ffclmb(xlam,-xp,-xmb,-xma,dmbp,dmap,dmamb,ier)
	    delsp = -xlam/4
*
*	    the first one is simple...
*
	    cs(1) = 2*cb1*piDpj(1,3)
	    cs(2) = ca0i(2)
	    cb2p = cs(1) + cs(2)
	    if ( absc(cb2p) .lt. xloss*absc(cs(2)) ) then
	    	if ( lwarn ) call ffwarn(214,ier,absc(cb2p),absc(cs(2)))
	    endif
*
*	    the next one ain't.
*
	    cs(1) = ca0i(2)
	    cs(2) = 2*xma*cb0
	    cs(3) = -2*piDpj(1,3)*cb1
	    cs(4) = xma+xmb
	    cs(5) = -xp/3
	    cb2d = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
	    xmax = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),absc(cs(5)))
	    if ( absc(cb2d) .ge. xloss*xmax ) goto 110
	    if ( lwrite ) then
		print '(a,2e30.16,e12.4)','cb2d  = ',cb2d,xmax
		print *,'with cs '
		print '(i3,2e30.16)',(i,cs(i),i=1,5)
	    endif
	    if ( lwarn ) then
		call ffwarn(214,ier,absc(cb2d),xmax)
		if ( lwrite ) then
		    print *,'xp,xma,xmb = ',xp,xma,xmb
		endif
	    endif
  110	    continue
* 	#] give up:
	    cb2p = cb2p*DBLE(1/(2*xp))
	    cb2d = cb2d*(1/DBLE(6))
*  #] p^2 != 0:
*  #[ p^2=0:
	elseif ( dmamb .ne. 0 ) then
	    if ( init.eq.0 ) then
		init = 1
		print *,' '
		print *,'ffcb2a: note: in this case p^2=0 B21 is ',
     +			'returned rather than B2p which is undefined'
		print *,' '
	    endif
	    if ( dmamb .ne. 0 ) then
*		#[ B21:
		llogmm = .FALSE.
*
*		B21 (see thesis, b21.frm)
*
		cs(1) = xma**2/3/dmamb**3*ca0i(1)
		cs(2) = (-xma**2 + xma*xmb - xmb**2/3)/dmamb**3*ca0i(2)
		cs(3) = (5*xma**3/18 - xma*xmb**2/2 + 2*xmb**3/9)
     +			/dmamb**3
		cb21 = cs(1)+cs(2)+cs(3)
		xmax = max(absc(cs(2)),absc(cs(3)))
		if ( absc(cb21).gt.xloss**2*xmax ) goto 160
		if ( lwrite ) then
		    print *,'cb21 = ',cb21,xmax
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,3)
		endif
*
*		ma ~ mb
*
		if ( absc(dmamb).lt.xloss*absc(xma) ) then
		    xlogmm = zfflo1(dmamb/xma,ier)
		else
		    xlogmm = log(xmb/xma)
		endif
		llogmm = .TRUE.
		cs(1) = (xma/dmamb)/6
		cs(2) = (xma/dmamb)**2/3
		cs(3) = (xma/dmamb)**3*xlogmm/3
		cs(4) = -2/DBLE(9) + ca0i(1)/(3*xma)
		cs(5) = -xlogmm/3
		csom = cs(1)+cs(2)+cs(3)+cs(4)+cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)))
		if ( lwrite ) then
		    print *,'cb21+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,5)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb21 = csom
		    if ( absc(cb21).gt.xloss**2*xmax ) goto 160
		endif
*
*		and last try
*
		xlo3 = zfflo3(dmamb/xma,ier)
		cs(1) = (dmamb/xma)**2/6
		cs(2) = (dmamb/xma)/3
		cs(3) = xlo3/(3*(dmamb/xma)**3)
*same		cs(4) = -2/DBLE(9) + ca0i(1)/(3*xma)
		cs(5) = -xlo3/3
		csom = cs(1)+cs(2)+cs(3)+cs(4)+cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)))
		if ( lwrite ) then
		    print *,'cb21+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,5)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb21 = csom
		    if ( absc(cb21).gt.xloss**2*xmax ) goto 160
		endif
*
*		give up
*
		if ( lwarn ) then
		    call ffwarn(227,ier,absc(cb21),xmax)
		    if ( lwrite ) then
			print *,'xp,xma,xmb = ',xp,xma,xmb
		    endif
		endif
  160		continue
*		#] B21:
*		#[ B22:
*
*		B22
*
		cs(1) = +xma/(4*dmamb)*ca0i(1)
		cs(2) = -xmb/(4*dmamb)*ca0i(2)
		cs(3) = (xma+xmb)/8
		cb22 = cs(1) + cs(2) + cs(3)
		xmax = max(absc(cs(2)),absc(cs(3)))
		if ( absc(cb22).gt.xloss*xmax ) goto 210
		if ( lwrite ) then
		    print *,'cb22 = ',cb22,xmax
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,3)
		endif
*
*		second try, close together
*
		if ( .not.llogmm ) then
		    if ( abs(dmamb).lt.xloss*absc(xma) ) then
		    	xlogmm = zfflo1(dmamb/xma,ier)
		    else
		    	xlogmm = log(xmb/xma)
		    endif
		endif
		cs(1) = dmamb*( -1/DBLE(8) - ca0i(1)/(4*xma) )
		cs(2) = dmamb*xlogmm/4
		cs(3) = xma*(xma/dmamb)/4*xlogmm
		cs(4) = xma*( 1/DBLE(4) + ca0i(1)/(2*xma) )
		cs(5) = -xma*xlogmm/2
		csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5))) 
		if ( lwrite ) then
		    print *,'cb22+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,2)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb22 = csom
		endif
		if ( absc(cb22).gt.xloss*xmax ) goto 210
*
*		give up
*
		if ( lwarn ) then
		    call ffwarn(214,ier,absc(cb22),xmax)
		    if ( lwrite ) then
			print *,'xp,xma,xmb = ',xp,xma,xmb
		    endif
		endif
  210		continue
*		#] B22:
	    else
*
*		ma=mb: simple
*
		cb21 = cb0/3
		cb22 = xma/2*(cb0 + 1)
	    endif
	    cb2d = cb22
	    cb2p = cb21
	endif
*  #] p^2=0:
*  #[ debug output:
	if ( lwrite ) then
	    print *,'ffcb2: cb2p = ',cb2p,ier
	    print *,'       cb2d = ',cb2d,ier
	endif
*  #] debug output:
*###] ffcb2a:
	end
