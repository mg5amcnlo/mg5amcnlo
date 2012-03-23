*###[ ffxb2p:
	subroutine ffxb2p(cb2i,cb1,cb0,ca0i,xp,xm1,xm2,piDpj,ier)
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
*		xp	   real		p.p in B&D metric		*
*		xm1,2	   real		m_1^2,m_2^2			*
*		piDpj(3,3) real		dotproducts between s1,s2,p	*
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
	DOUBLE PRECISION xp,xm1,xm2,piDpj(3,3)
	DOUBLE COMPLEX cb2i(2),cb1,cb0,ca0i(2)
*
*	local variables
*
	DOUBLE PRECISION dm1p,dm2p,dm1m2
*
*  #] declarations:
*  #[ work:
*
	dm1p = xm1 - xp
	dm2p = xm2 - xp
	dm1m2= xm1 - xm2
	call ffxb2q(cb2i,cb1,cb0,ca0i,xp,xm1,xm2,dm1p,dm2p,dm1m2,
     +		piDpj,ier)
*
*  #] work:
*###] ffxb2p:
	end
*###[ ffxb2q:
	subroutine ffxb2q(cb2i,cb1,cb0,ca0i,xp,xm1,xm2,dm1p,dm2p,dm1m2,
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
*		xp	   real		p.p in B&D metric		*
*		xm1,2	   real		m_1^2,m_2^2			*
*		piDpj(3,3) real		dotproducts between s1,s2,p	*
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
	DOUBLE PRECISION xp,xm1,xm2,dm1p,dm2p,dm1m2,piDpj(3,3)
	DOUBLE COMPLEX cb2i(2),cb1,cb0,ca0i(2)
*
*	local variables
*
	integer i,j,ier0,ier1
	logical llogmm
	DOUBLE PRECISION xmax,absc,xlam,slam,alp,bet,xmxp,dfflo3,xlo3,
     +		xmxsav,xnoe,xnoe2,xlogmm,dfflo1,rloss,
     +		qiDqj(3,3)
	DOUBLE COMPLEX cs(16),cc,csom,clo2,clo3,zfflo2,zfflo3
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
*  #[ test input:
	if ( ltest ) then
	    ier0 = ier
	    call ffdot2(qiDqj,xp,xm1,xm2,dm1p,dm2p,dm1m2,ier0)
	    rloss = xloss*DBLE(10)**(-mod(ier0,50))
	    do 20 j=1,3
		do 10 i=1,3
		    if ( rloss*abs(piDpj(i,j)-qiDqj(i,j)).gt.precx*
     +			abs(piDpj(i,j))) print *,'ffxb2q: error: piDpj('
     +			,i,j,') wrong: ',piDpj(i,j),qiDqj(i,j),
     +			piDpj(i,j)-qiDqj(i,j),ier0
   10		continue
   20	    continue
	endif
*  #] test input:
*  #[ normal case:
	ier0 = ier
	ier1 = ier
*
*	with thanks to Andre Aeppli, off whom I stole the original
*
	if ( xp .ne. 0) then
	    cs(1) = ca0i(2)
	    cs(2) = DBLE(xm1)*cb0
	    cs(3) = DBLE(2*piDpj(1,3))*cb1
	    cs(4) = (xm1+xm2)/2
	    cs(5) = -xp/6
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
	    if ( dm1m2.eq.0 .and. xm1.ne.0 ) then
		if ( xp.lt.0 ) then
		    slam = sqrt(xp**2-4*xm1*xp)
		    xlo3 = dfflo3((xp-slam)/(2*xm1),ier)
		    cs(1) = xp*(-1/DBLE(3) + slam/(4*xm1))
		    cs(2) = xp**2*(-slam/(4*xm1**2) - 3/(4*xm1))
		    cs(3) = xp**3/(4*xm1**2)
		    cs(4) = DBLE(xp/xm1)*ca0i(1)
		    cs(5) = xlo3/xp*(-xm1*slam)
		    cs(6) = xlo3*slam
		else
		    slam = isgnal*sqrt(-xp**2+4*xm1*xp)
		    clo3 = zfflo3(DCMPLX(DBLE(xp/(2*xm1)),
     +			DBLE(-slam/(2*xm1))),ier)
		    cs(1) = DBLE(xp)*DCMPLX(-1/DBLE(3),
     +			DBLE(slam/(4*xm1)))
		    cs(2) = DBLE(xp**2)*DCMPLX(DBLE(-3/(4*xm1)),
     +			DBLE(-slam/(4*xm1**2)))
		    cs(3) = DBLE(xp**3/(4*xm1**2))
		    cs(4) = DBLE(xp/xm1)*ca0i(1)
		    cs(5) = clo3*DCMPLX(DBLE(0),DBLE(-xm1*slam/xp))
		    cs(6) = clo3*DCMPLX(DBLE(0),DBLE(slam))
		endif
		csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5) + cs(6)
		xmxp = max(absc(cs(2)),absc(cs(3)),absc(cs(4)),
     +			absc(cs(5)),absc(cs(6)))
*		
*		get rid of noise in the imaginary part
*	
		if ( xloss*abs(DIMAG(csom)).lt.precc*abs(DBLE(csom)) ) 
     +			csom = DCMPLX(DBLE(csom),DBLE(0))
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
*  #[   improve: |xp| < xm1 < xm2:
*
*	    try again (see bi.frm)
*
	    xlam =  4*(piDpj(1,3)**2 - xm1*xp)
	    if ( xm1.eq.0 .or. xm2.eq.0 ) then
	    	xlogmm = 0
	    elseif ( abs(dm1m2).lt.xloss*xm1 ) then
	    	xlogmm = dfflo1(dm1m2/xm1,ier)
	    else
	    	xlogmm = log(xm2/xm1)
	    endif
	    if ( xlam.gt.0 .and. abs(xp).lt.xloss*xm2 .and.
     +			xm1.lt.xm2 ) then
		slam = sqrt(xlam)
		alp = (2*xm1*xm2/(2*piDpj(1,2)+slam) + xm1)/(slam-dm1m2)
*		bet = [xm2-xm1-xp-slam]
		bet = 4*xm1*xp/(2*piDpj(1,3)+slam)
		cs(1) = DBLE(xp/xm2)*ca0i(2)
		cs(2) = xlogmm*bet*(-2*xm1**2*xm2 - 2*xm1**3)
     +		/((-dm1m2+slam)*(2*piDpj(1,2)+slam)*(2*piDpj(1,3)+slam))
		cs(3) = xlogmm*(-4*xp*xm1**3)
     +		/((-dm1m2+slam)*(2*piDpj(1,2)+slam)*(2*piDpj(1,3)+slam))
		xnoe = 1/(2*piDpj(2,3)+slam)
		xnoe2 = xnoe**2
		cs(4) = xnoe2*xm1*bet*(xp-4*xm2)
		cs(5) = xnoe2*xm1*2*xp*xm2
		cs(6) = xnoe2*xm1**2*bet
		cs(7) = xnoe2*xm1**2*4*xp
		cs(8) = xnoe2*bet*(xp*xm2+3*xm2**2)
		cs(9) = xnoe2*(-6*xp*xm2**2)
		cs(10)= xp*(7/6.d0 - 2*xm1*slam*xnoe2 +
     +			4*xm2*slam*xnoe2 - 2*slam*xnoe)
		cs(11)= xp**2*( -2*slam*xnoe2 )
		xlo3 = dfflo3(2*xp*xnoe,ier)
		cs(12) = xlo3*dm1m2**2*slam/xp**2
		cs(13) = xlo3*(xm1 - 2*xm2)*slam/xp
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
*  #]   improve: |xp| < xm1 < xm2:
*  #[   improve: |xp| < xm2 < xm1:
	    if ( xlam.gt.0 .and. abs(xp).lt.xloss*xm1 .and.
     +			xm2.lt.xm1 ) then
		slam = sqrt(xlam)
		alp = (2*xm2*xm1/(2*piDpj(1,2)+slam) + xm2)/(slam+dm1m2)
*		bet = [xm1-xm2-xp-slam]
		bet = 4*xm2*xp/(-2*piDpj(2,3)+slam)
		xnoe = 1/(-2*piDpj(1,3)+slam)
		xnoe2 = xnoe**2
		cs(1) = DBLE(xp/xm1)*ca0i(1)
		cs(2) = -xlogmm*bet*(12*xp*xm1*xm2+6*xp*xm2**2-
     +		6*xp**2*xm2-2*xm1*xm2**2-2*xm2**3)
     +		/((dm1m2+slam)*(2*piDpj(1,2)+slam)*(-2*piDpj(2,3)+slam))
		cs(3) = -xlogmm*(-24*xp*xm1**2*xm2-4*xp*xm2**3+36*
     +		xp**2*xm1*xm2+12*xp**2*xm2**2-12*xp**3*xm2)
     +		/((dm1m2+slam)*(2*piDpj(1,2)+slam)*(-2*piDpj(2,3)+slam))
		cs(4) = xnoe2*xm2*bet*(xp-4*xm1)
		cs(5) = xnoe2*xm2*(-10*xp*xm1)
		cs(6) = xnoe2*xm2**2*bet
		cs(7) = xnoe2*xm2**2*4*xp
		cs(8) = xnoe2*bet*(xp*xm1+3*xm1**2)
		cs(9) = xnoe2*6*xp*xm1**2
		cs(10)= xp*(7/6.d0 - 2*xm1*slam*xnoe2 +
     +			4*xm2*slam*xnoe2 - 2*slam*xnoe)
		cs(11)= xp**2*( -2*slam*xnoe2 )
		xlo3 = dfflo3(2*xp*xnoe,ier)
		cs(12) = xlo3*dm1m2**2*slam/xp**2
		cs(13) = xlo3*(xm1 - 2*xm2)*slam/xp
		cs(14) = xlo3*slam
		csom = 0
		xmxp = 0
		do 60 i=1,14
		    csom = csom + cs(i)
		    xmxp = max(xmxp,absc(cs(i)))
   60		continue
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
*  #]   improve: |xp| < xm2 < xm1:
*  #[   wrap up:
	    if ( lwarn ) then
		call ffwarn(225,ier0,absc(cb2i(1)),xmax)
		if ( lwrite ) then
		    print *,'xp,xm1,xm2 = ',xp,xm1,xm2
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
	    cb2i(1) = DBLE(1/(3*xp)) * cb2i(1)
	    cb2i(2) = DBLE(1/6.d0)   * cb2i(2)
*  #]   wrap up:
*  #[ xp=0, m1!=m2:
	elseif (dm1m2 .ne. 0) then
*		#[ old code:
*		first calculate B21
*	    cs(1) = +DBLE(xm1*xm1/dm1m2) * ca0i(1)
*	    cs(2) = -     xm1*xm1/dm1m2  * xm1
*	    cs(3) = -DBLE((3*xm1**2-3*xm1*xm2+xm2**2)/dm1m2) * ca0i(2)
*	    cs(4) = +     (3*xm1**2-3*xm1*xm2+xm2**2)/dm1m2  * xm2
*	    cs(5) = (11*xm1**2-7*xm1*xm2+2*xm2**2)/6
**
*	    cb2i(2) = cs(1)+cs(2)+cs(3)+cs(4)+cs(5)
*	    if ( lwarn  ) then
*		xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)),
*     +			absc(cs(4)),absc(cs(5)))
*		if ( absc(cb2i(2)) .lt. xloss*xmax )
*     +			call ffwarn(298,ier0,absc(cb2i(2)),xmax)
*	    endif
*	    cb2i(1)=1/(3*dm1m2**2) * cb2i(2)
*		 B22 in the same way as with xp diff from zero
*		 18-nov-1993 fixed sign error in cs(2) GJ
*	    cs(1) = ca0i(2)
*	    cs(2) =+DBLE(2*xm1)*cb0
*	    cs(3) = DBLE(dm1m2)*cb1
*	    cs(4) = xm1+xm2
*	    cb2i(2) = cs(1) + cs(2) + cs(3) + cs(4)
*	    if ( lwarn ) then
*		xmax = max(absc(cs(1)),absc(cs(3)),absc(cs(4)))
*		if ( absc(cb2i(2)) .lt. xloss*xmax )
*     +			call ffwarn(298,ier1,absc(cb2i(2)),xmax)
*	    endif
*	    cb2i(2) = cb2i(2)/6
*		#] old code:
*		#[ B21:
		llogmm = .FALSE.
*
*		B21 (see thesis, b21.frm)
*
		cs(1) = DBLE(xm1**2/3/dm1m2**3)*ca0i(1)
		cs(2) = DBLE((-xm1**2 + xm1*xm2 - xm2**2/3)/dm1m2**3)*
     +			ca0i(2)
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
		if ( abs(dm1m2).lt.xloss*xm1 ) then
		    xlogmm = dfflo1(dm1m2/xm1,ier)
		else
		    xlogmm = log(xm2/xm1)
		endif
		llogmm = .TRUE.
		cs(1) = (xm1/dm1m2)/6
		cs(2) = (xm1/dm1m2)**2/3
		cs(3) = (xm1/dm1m2)**3*xlogmm/3
		cs(4) = -2/DBLE(9) + ca0i(1)*DBLE(1/(3*xm1))
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
		xlo3 = dfflo3(dm1m2/xm1,ier)
		cs(1) = (dm1m2/xm1)**2/6
		cs(2) = (dm1m2/xm1)/3
		cs(3) = xlo3/(3*(dm1m2/xm1)**3)
*same		cs(4) = -2/DBLE(9) + ca0i(1)*DBLE(1/(3*xm1))
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
			print *,'xp,xm1,xm2 = ',xp,xm1,xm2
		    endif
		endif
  160		continue
*		#] B21:
*		#[ B22:
*
*		B22
*
		cs(1) = +DBLE(xm1/(4*dm1m2))*ca0i(1)
		cs(2) = -DBLE(xm2/(4*dm1m2))*ca0i(2)
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
		    if ( abs(dm1m2).lt.xloss*xm1 ) then
		    	xlogmm = dfflo1(dm1m2/xm1,ier)
		    else
		    	xlogmm = log(xm2/xm1)
		    endif
		endif
		cs(1) = dm1m2*( -1/DBLE(8) - ca0i(1)*DBLE(1/(4*xm1)) )
		cs(2) = dm1m2*xlogmm/4
		cs(3) = xm1*(xm1/dm1m2)/4*xlogmm
		cs(4) = xm1*( 1/DBLE(4) + ca0i(1)*DBLE(1/(2*xm1)) )
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
			print *,'xp,xm1,xm2 = ',xp,xm1,xm2
		    endif
		endif
  210		continue
*		#] B22:
*  #] xp=0, m1!=m2:
*  #[ xp=0, m1==m2:
	else
*
*	    taken over from ffxb2a, which in turns stem from my thesis GJ
*
	    cb2i(1) = cb0/3
	    cb2i(2) = DBLE(xm1/2)*(cb0 + 1)
	endif
*  #] xp=0, m1==m2:
*  #[ finish up:
	ier = max(ier0,ier1)
*  #] finish up:
*###] ffxb2q:
	end
