*###[ ffcb2p:
	subroutine ffcb2p(cb2i,cb1,cb0,ca0i,cp,xm1,xm2,piDpj,ier)
***#[*comment:***********************************************************
*									*
*	Compute the PV B2, the coefficients of p(mu)p(nu) and g(mu,nu)	*
*	of 1/(ipi^2)\int d^nQ Q(mu)Q(nu)/(Q^2-m_1^2)/((Q+p)^2-m_2^2)	*
*	originally based on aaxbx by Andre Aeppli.			*
*									*
*	Input:	cb1	   complex	vector two point function	*
*		cb0	   complex	scalar two point function	*
*		ca0i(2)	   complex	scalar onepoint function with	*
*						m1,m2			*
*		cp	   complex	p.p in B&D metric		*
*		xm1,2	   complex	m_1^2,m_2^2			*
*		piDpj(3,3) complex	dotproducts between s1,s2,p	*
*		ier	   integer	digits lost so far		*
*									*
*	Output:	cb2i(2)	   complex	B21,B22: coeffs of p*p, g in B2	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cp,xm1,xm2,piDpj(3,3)
	DOUBLE COMPLEX cb2i(2),cb1,cb0,ca0i(2)
	DOUBLE PRECISION rm1,rm2,rp,rpiDpj(3,3),sprec
*
*	local variables
*
	integer i,j
	DOUBLE COMPLEX dm1p,dm2p,dm1m2
*
*	common blocks
*
*
	include 'ff.h'
*
*  #] declarations:
*  #[ real case:
	if ( DIMAG(xm1).eq.0 .and. DIMAG(xm2).eq.0 ) then
	    rm1 = DBLE(xm1)
	    rm2 = DBLE(xm2)
	    rp  = DBLE(cp)
	    do 20 j=1,3
		do 10 i=1,3
		    rpiDpj(i,j) = DBLE(piDpj(i,j))
   10		continue
   20	    continue
	    sprec = precx
	    precx = precc
	    call ffxb2p(cb2i,cb1,cb0,ca0i,rp,rm1,rm2,rpiDpj,ier)
	    precx = sprec
	    return
	endif
*  #] real case:
*  #[ work:
*
	dm1p = xm1 - cp
	dm2p = xm2 - cp
	dm1m2= xm1 - xm2
	call ffcb2q(cb2i,cb1,cb0,ca0i,cp,xm1,xm2,dm1p,dm2p,dm1m2,
     +		piDpj,ier)
*
*  #] work:
*###] ffcb2p:
	end
*###[ ffcb2q:
	subroutine ffcb2q(cb2i,cb1,cb0,ca0i,cp,xm1,xm2,dm1p,dm2p,dm1m2,
     +		piDpj,ier)
***#[*comment:***********************************************************
*									*
*	Compute the PV B2, the coefficients of p(mu)p(nu) and g(mu,nu)	*
*	of 1/(ipi^2)\int d^nQ Q(mu)Q(nu)/(Q^2-m_1^2)/((Q+p)^2-m_2^2)	*
*	originally based on aaxbx by Andre Aeppli.			*
*									*
*	Input:	cb1	   complex	vector two point function	*
*		cb0	   complex	scalar two point function	*
*		ca0i(2)	   complex	scalar onepoint function with	*
*						m1,m2			*
*		cp	   complex	p.p in B&D metric		*
*		xm1,2	   complex	m_1^2,m_2^2			*
*		piDpj(3,3) complex	dotproducts between s1,s2,p	*
*		ier	   integer	digits lost so far		*
*									*
*	Output:	cb2i(2)	   complex	B21,B22: coeffs of p*p, g in B2	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cp,xm1,xm2,dm1p,dm2p,dm1m2,piDpj(3,3)
	DOUBLE COMPLEX cb2i(2),cb1,cb0,ca0i(2)
*
*	local variables
*
	integer i,j,ier0,ier1,ithres,init
	logical lreal,llogmm
	DOUBLE PRECISION xmax,xmxsav,absc,rloss,xmxp
	DOUBLE PRECISION rm1,rm2,rp,rm1p,rm2p,rm1m2,rpiDpj(3,3),sprec
	DOUBLE COMPLEX cs(14),cc,slam,xlo3,csom,clam,xlogmm,zfflo1,alp,
     +		bet,xnoe,xnoe2,zfflo3
	DOUBLE COMPLEX cqi(3),cqiqj(3,3),qiDqj(3,3)
	save init
* for Absoft only
*	external csqrt
*	DOUBLE COMPLEX csqrt
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
*  #[ real cases:
	if ( DIMAG(xm1).eq.0 .and. DIMAG(xm2).eq.0 ) then
	    lreal = .TRUE.
	    if ( lwrite ) print *,'ffcb2q: real masses'
	elseif ( nschem.le.4 ) then
	    lreal = .TRUE.
	    if ( lwrite .or. init.eq.0 ) then
		init = 1
		print *,'ffcb2q: nschem <= 4, ignoring complex masses:',
     +			nschem
	    endif
	elseif ( nschem.le.6 ) then
	    if ( lwrite .or. init.eq.0 ) then
		init = 1
		print *,'ffcb2q: nschem = 5,6 complex masses near ',
     +			'threshold: ',nschem
	    endif
	    cqi(1) = xm1
	    cqi(2) = xm2
	    cqi(3) = cp
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
		if ( lwrite ) print *,'ffcb2q: no threshold'
	    else
		if ( lwrite ) print *,'ffcb2q: found threshold'
		lreal = .FALSE.
	    endif
	else
	    lreal = .FALSE.
	endif
	if ( lreal ) then
	    rm1 = DBLE(xm1)
	    rm2 = DBLE(xm2)
	    rp  = DBLE(cp)
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
	    call ffxb2q(cb2i,cb1,cb0,ca0i,rp,rm1,rm2,rm1p,rm2p,
     +		rm1m2,rpiDpj,ier)
	    precx = sprec
	    return
	endif
*  #] real cases:
*  #[ test input:
	if ( ltest ) then
	    ier0 = ier
	    call ffcot2(qiDqj,cp,xm1,xm2,dm1p,dm2p,dm1m2,ier0)
	    rloss = xloss*DBLE(10)**(-mod(ier0,50))
	    do 40 j=1,3
		do 30 i=1,3
		    if ( rloss*absc(piDpj(i,j)-qiDqj(i,j)).gt.precc*
     +			absc(qiDqj(i,j)) ) print *,'ffcb2q: ',
     +			'error: piDpj(',i,j,') wrong: ',piDpj(i,j),
     +			qiDqj(i,j),piDpj(i,j)-qiDqj(i,j),ier0
   30		continue
   40	    continue
	endif
*  #] test input:
*  #[ normal case:
	ier0 = ier
	ier1 = ier
*
*	with thanks to Andre Aeppli, off whom I stole the original
*
	if ( DBLE(cp) .ne. 0) then
	    cs(1) = ca0i(2)
	    cs(2) = xm1*cb0
	    cs(3) = 2*piDpj(1,3)*cb1
	    cs(4) = (xm1+xm2)/2
	    cs(5) = -cp/6
	    cb2i(1) = cs(1) - cs(2) + 2*cs(3) - cs(4) - cs(5)
	    cb2i(2) = cs(1) + 2*cs(2) - cs(3) + 2*cs(4) + 2*cs(5)
	    xmax = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),absc(cs(5)))
	    xmxsav = xmax
	    if ( absc(cb2i(1)) .ge. xloss*xmax ) goto 100
	    if ( lwrite ) then
		print *,'cb2i(1) = ',cb2i(1),xmax
		print *,'with cs '
		print '(i3,2e30.16)',1,cs(1),2,-cs(2),3,2*cs(3),4,
     +			-cs(4),5,-cs(5)
	    endif
*  #] normal case:
*  #[   improve: m1=m2:
*
*	    a relatively simple case: dm1m2 = 0 (bi0.frm)
*
	    if ( dm1m2.eq.0 ) then
		slam = sqrt(cp**2-4*xm1*cp)
		xlo3 = zfflo3((cp-slam)/(2*xm1),ier)
		cs(1) = cp*(-1/DBLE(3) + slam/(4*xm1))
		cs(2) = cp**2*(-slam/(4*xm1**2) - 3/(4*xm1))
		cs(3) = cp**3/(4*xm1**2)
		cs(4) = cp/xm1*ca0i(1)
		cs(5) = xlo3/cp*(-xm1*slam)
		cs(6) = xlo3*slam
		csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5) + cs(6)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)),absc(cs(6)))
		if ( lwrite ) then
		    print *,'cb2i(1)+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,6)
		endif
		if ( xmxp.lt.xmax ) then
		    cb2i(1) = csom
		    xmax = xmxp
		endif
		if ( absc(cb2i(1)).ge.xloss**2*xmax ) goto 100
	    endif
*  #]   improve: m1=m2: 
*  #[   improve: |cp| < xm1 < xm2:
*
*	    try again (see bi.frm)
*
	    clam =  4*(piDpj(1,3)**2 - xm1*cp)
	    if ( xm1.eq.0 .or. xm2.eq.0 ) then
		xlogmm = 0
	    elseif ( absc(dm1m2).lt.xloss*absc(xm1) ) then
		xlogmm = zfflo1(dm1m2/xm1,ier)
	    else
		xlogmm = log(xm2/xm1)
	    endif
	    if ( abs(DBLE(cp)).lt.xloss*absc(xm2) .and.
     +			DBLE(xm1).lt.DBLE(xm2) ) then
		slam = sqrt(clam)
		alp = (2*xm1*xm2/(2*piDpj(1,2)+slam) + xm1)/(slam-dm1m2)
*		bet = [xm2-xm1-cp-slam]
		bet = 4*xm1*cp/(2*piDpj(1,3)+slam)
		cs(1) = cp/xm2*ca0i(2)
		cs(2) = xlogmm*bet*(-2*xm1**2*xm2 - 2*xm1**3)
     +		/((-dm1m2+slam)*(2*piDpj(1,2)+slam)*(2*piDpj(1,3)+slam))
		cs(3) = xlogmm*(-4*cp*xm1**3)
     +		/((-dm1m2+slam)*(2*piDpj(1,2)+slam)*(2*piDpj(1,3)+slam))
		xnoe = 1/(2*piDpj(2,3)+slam)
		xnoe2 = xnoe**2
		cs(4) = xnoe2*xm1*bet*(cp-4*xm2)
		cs(5) = xnoe2*xm1*2*cp*xm2
		cs(6) = xnoe2*xm1**2*bet
		cs(7) = xnoe2*xm1**2*4*cp
		cs(8) = xnoe2*bet*(cp*xm2+3*xm2**2)
		cs(9) = xnoe2*(-6*cp*xm2**2)
		cs(10)= cp*(7/6.d0 - 2*xm1*slam*xnoe2 +
     +			4*xm2*slam*xnoe2 - 2*slam*xnoe)
		cs(11)= cp**2*( -2*slam*xnoe2 )
		xlo3 = zfflo3(2*cp*xnoe,ier)
		cs(12) = xlo3*dm1m2**2*slam/cp**2
		cs(13) = xlo3*(xm1 - 2*xm2)*slam/cp
		cs(14) = xlo3*slam
		csom = 0
		xmxp = 0
		do 50 i=1,14
		    csom = csom + cs(i)
		    xmxp = max(xmxp,absc(cs(i)))
   50		continue
		if ( lwrite ) then
		    print *,'cb2i(1)+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,14)
		endif
		if ( xmxp.lt.xmax ) then
		    cb2i(1) = csom
		    xmax = xmxp
		endif
		if ( absc(cb2i(1)).ge.xloss**2*xmax ) goto 100
	    endif
*  #]   improve: |cp| < xm1 < xm2:
*  #[   improve: |cp| < xm2 < xm1:
	    if ( abs(DBLE(cp)).lt.xloss*absc(xm1) .and. 
     +	    		DBLE(xm2).lt.DBLE(xm1) ) then
		slam = sqrt(clam)
		alp = (2*xm2*xm1/(2*piDpj(1,2)+slam) + xm2)/(slam+dm1m2)
*		bet = [xm1-xm2-cp-slam]
		bet = 4*xm2*cp/(-2*piDpj(2,3)+slam)
		xnoe = 1/(-2*piDpj(1,3)+slam)
		xnoe2 = xnoe**2
		cs(1) = cp/xm1*ca0i(1)
		cs(2) = -xlogmm*bet*(12*cp*xm1*xm2+6*cp*xm2**2-
     +		6*cp**2*xm2-2*xm1*xm2**2-2*xm2**3)
     +		/((dm1m2+slam)*(2*piDpj(1,2)+slam)*(-2*piDpj(2,3)+slam))
		cs(3) = -xlogmm*(-24*cp*xm1**2*xm2-4*cp*xm2**3+36*
     +		cp**2*xm1*xm2+12*cp**2*xm2**2-12*cp**3*xm2)
     +		/((dm1m2+slam)*(2*piDpj(1,2)+slam)*(-2*piDpj(2,3)+slam))
		cs(4) = xnoe2*xm2*bet*(cp-4*xm1)
		cs(5) = xnoe2*xm2*(-10*cp*xm1)
		cs(6) = xnoe2*xm2**2*bet
		cs(7) = xnoe2*xm2**2*4*cp
		cs(8) = xnoe2*bet*(cp*xm1+3*xm1**2)
		cs(9) = xnoe2*6*cp*xm1**2
		cs(10)= cp*(7/6.d0 - 2*xm1*slam*xnoe2 +
     +			4*xm2*slam*xnoe2 - 2*slam*xnoe)
		cs(11)= cp**2*( -2*slam*xnoe2 )
		xlo3 = zfflo3(2*cp*xnoe,ier)
		cs(12) = xlo3*dm1m2**2*slam/cp**2
		cs(13) = xlo3*(xm1 - 2*xm2)*slam/cp
		cs(14) = xlo3*slam
		csom = 0
		xmxp = 0
		do 60 i=1,14
		    csom = csom + cs(i)
		    xmxp = max(xmxp,absc(cs(i)))
   60		continue
		if ( lwrite ) then
		    print *,'cb2i(1)-= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,14)
		endif
		if ( xmxp.lt.xmax ) then
		    cb2i(1) = csom
		    xmax = xmxp
		endif
		if ( absc(cb2i(1)).ge.xloss**2*xmax ) goto 100
	    endif
*  #]   improve: |cp| < xm2 < xm1:
*  #[   wrap up:
	    if ( lwarn ) then
		call ffwarn(225,ier0,absc(cb2i(1)),xmax)
		if ( lwrite ) then
		    print *,'cp,xm1,xm2 = ',cp,xm1,xm2
		endif
	    endif
  100	    continue
	    xmax = xmxsav
	    if ( absc(cb2i(2)) .lt. xloss**2*xmax ) then
		if ( lwrite ) then
		    print *,'cb2i(2) = ',cb2i(2),xmax
		    print *,'with cs '
		    print '(i3,2e30.16)',1,cs(1),2,2*cs(2),3,-cs(3),
     +			4,2*cs(4)
		endif
*
		if ( lwarn ) then
		    call ffwarn(226,ier1,absc(cb2i(2)),xmax)
		endif
  110		continue
		if ( lwrite ) print *,'cb2i(2)+= ',cb2i(2)
	    endif
	    cb2i(1) = DBLE(1/(3*cp)) * cb2i(1)
	    cb2i(2) = DBLE(1/6.d0)   * cb2i(2)
*  #]   wrap up:
*  #[ cp=0, m1!=m2:
	elseif (dm1m2 .ne. 0) then
*		#[ B21:
		llogmm = .FALSE.
*
*		B21 (see thesis, b21.frm)
*
		cs(1) = xm1**2/3/dm1m2**3*ca0i(1)
		cs(2) = (-xm1**2 + xm1*xm2 - xm2**2/3)/dm1m2**3*ca0i(2)
		cs(3) = (5*xm1**3/18 - xm1*xm2**2/2 + 2*xm2**3/9)
     +			/dm1m2**3
		cb2i(1) = cs(1)+cs(2)+cs(3)
		xmax = max(absc(cs(2)),absc(cs(3)))
		if ( absc(cb2i(1)).gt.xloss**2*xmax ) goto 160
		if ( lwrite ) then
		    print *,'cb2i(1) = ',cb2i(1),xmax
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,3)
		endif
*
*		ma ~ mb
*
		if ( absc(dm1m2).lt.xloss*absc(xm1) ) then
		    xlogmm = zfflo1(dm1m2/xm1,ier)
		else
		    xlogmm = log(xm2/xm1)
		endif
		llogmm = .TRUE.
		cs(1) = (xm1/dm1m2)/6
		cs(2) = (xm1/dm1m2)**2/3
		cs(3) = (xm1/dm1m2)**3*xlogmm/3
		cs(4) = -2/DBLE(9) + ca0i(1)/(3*xm1)
		cs(5) = -xlogmm/3
		csom = cs(1)+cs(2)+cs(3)+cs(4)+cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)))
		if ( lwrite ) then
		    print *,'cb2i(1)+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,5)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb2i(1) = csom
		    if ( absc(cb2i(1)).gt.xloss**2*xmax ) goto 160
		endif
*
*		and last try
*
		xlo3 = zfflo3(dm1m2/xm1,ier)
		cs(1) = (dm1m2/xm1)**2/6
		cs(2) = (dm1m2/xm1)/3
		cs(3) = xlo3/(3*(dm1m2/xm1)**3)
*same		cs(4) = -2/DBLE(9) + ca0i(1)/(3*xm1)
		cs(5) = -xlo3/3
		csom = cs(1)+cs(2)+cs(3)+cs(4)+cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)))
		if ( lwrite ) then
		    print *,'cb2i(1)+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,5)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb2i(1) = csom
		    if ( absc(cb2i(1)).gt.xloss**2*xmax ) goto 160
		endif
*
*		give up
*
		if ( lwarn ) then
		    call ffwarn(225,ier,absc(cb2i(1)),xmax)
		    if ( lwrite ) then
			print *,'cp,xm1,xm2 = ',cp,xm1,xm2
		    endif
		endif
  160		continue
*		#] B21:
*		#[ B22:
*
*		B22
*
		cs(1) = +xm1/(4*dm1m2)*ca0i(1)
		cs(2) = -xm2/(4*dm1m2)*ca0i(2)
		cs(3) = (xm1+xm2)/8
		cb2i(2) = cs(1) + cs(2) + cs(3)
		xmax = max(absc(cs(2)),absc(cs(3)))
		if ( absc(cb2i(2)).gt.xloss*xmax ) goto 210
		if ( lwrite ) then
		    print *,'cb2i(2) = ',cb2i(2),xmax
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,3)
		endif
*
*		second try, close together
*
		if ( .not.llogmm ) then
		    if ( abs(dm1m2).lt.xloss*absc(xm1) ) then
			xlogmm = zfflo1(dm1m2/xm1,ier)
		    else
			xlogmm = log(xm2/xm1)
		    endif
		endif
		cs(1) = dm1m2*( -1/DBLE(8) - ca0i(1)/(4*xm1) )
		cs(2) = dm1m2*xlogmm/4
		cs(3) = xm1*(xm1/dm1m2)/4*xlogmm
		cs(4) = xm1*( 1/DBLE(4) + ca0i(1)/(2*xm1) )
		cs(5) = -xm1*xlogmm/2
		csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)))
		if ( lwrite ) then
		    print *,'cb2i(2)+= ',csom,xmxp
		    print *,'with cs '
		    print '(i3,2e30.16)',(i,cs(i),i=1,2)
		endif
		if ( xmxp.lt.xmax ) then
		    xmax = xmxp
		    cb2i(2) = csom
		endif
		if ( absc(cb2i(2)).gt.xloss*xmax ) goto 210
*
*		give up
*
		if ( lwarn ) then
		    call ffwarn(226,ier,absc(cb2i(2)),xmax)
		    if ( lwrite ) then
			print *,'cp,xm1,xm2 = ',cp,xm1,xm2
		    endif
		endif
  210		continue
*		#] B22:
*  #] cp=0, m1!=m2:
*  #[ cp=0, m1==m2:
	else
*
*	    taken over from ffxb2a, which in turns stem from my thesis GJ
*
	    cb2i(1) = cb0/3
	    cb2i(2) = xm1/2*(cb0 + 1)
	endif
*  #] cp=0, m1==m2:
*  #[ finish up:
	ier = max(ier0,ier1)
*  #] finish up:
*###] ffcb2q:
	end
