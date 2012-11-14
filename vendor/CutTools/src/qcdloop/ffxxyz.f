*###[ ffxxyz:
	subroutine ffxxyz(y,z,dyz,d2yzz,dy2z,ivert,sdel2p,sdel2s,etalam,
     +		etami,delps,xpi,dpipj,piDpj,isoort,ldel2s,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*	z(1,2) = (-p(ip1).p(is2) +/- sdel2s)/xpi(ip1)			*
*	y(1,2) = (-p(ip1).p(is2) +/- sdisc)/xpi(ip1)			*
*			disc = del2s + etaslam*xpi(ip1)			*
*									*
*	y(3,4) = 1-y(1,2)						*
*	z(3,4) = 1-z(1,2)						*
*	dyz(i,j) = y(i) - z(j)						*
*	d2yzz = y(2) - z(1) - z(2)					*
*	dy2z(j) = y(2) - 2*z(j)						*
*									*
*	Input:	ivert		(integer)	defines the vertex	*
*		sdel2p		(real)		sqrt(lam(p1,p2,p3))/2	*
*		sdel2s		(real)		sqrt(lam(p,ma,mb))/2	*
*		etalam		(real)		det(si.sj)/det(pi.pj)	*
*		etami(6)	(real)		si.si - etalam		*
*		xpi(ns)		(real)		standard		*
*		piDpj(ns,ns)	(real)		standard		*
*		ns		(integer)	dim of xpi,piDpj	*
*									*
*	Output:	y(4),z(4),dyz(4,4)	(real)		see above	*
*									*
*	Calls:	fferr,ffroot						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ivert,ns,ier,isoort(2)
	logical ldel2s
	DOUBLE PRECISION y(4),z(4),dyz(2,2),d2yzz,dy2z(4),
     +		sdel2p,sdel2s,etalam,etami(6),delps,xpi(ns),
     +		dpipj(ns,ns),piDpj(ns,ns)
*
*	local variables:
*
	integer i,j,n,ip1,ip2,ip3,is1,is2,is3,iwarn,ier0,ier1
	DOUBLE PRECISION delps1,disc,xheck,xlosn,hulp,s,smax,som(51),
     +		xmax
	DOUBLE PRECISION t1,t2,t4,t5,t8,t3,t7,t9,t12,t14,t21,t23,t24,
     +	t28,t6,t35,t44,t42,t36,t55,t41,t19,t59,t25,t69,t82,t75,t84,t92,
     +	t31,t98,t74,t101,t89,t106,t112,t113,t13,t117,t126,t127,t129,
     +	t130,t133,t128,t132,t134,t137,t139,t146,t148,t149,t153,t131,
     +	t160,t171,t169,t161,t182,t168,t144,t186,t150,t208,t201,t210,
     +	t219,t156,t225,t200,t228,t215,t233,t239,t240,t138,t244
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ set up pointers:
	if ( ldel2s .and. ivert .ne. 1 ) goto 100
	is1 = ivert
	is2 = ivert+1
	if ( is2 .eq. 4 ) is2 = 1
	is3 = ivert-1
	if ( is3 .eq. 0 ) is3 = 3
	ip1 = is1 + 3
	ip2 = is2 + 3
	ip3 = is3 + 3
*  #] set up pointers:
*  #[ xk = 0:
	if ( xpi(ip1) .eq. 0 ) then
	    isoort(2) = 0
	    if ( piDpj(is1,ip1) .eq. 0 ) then
		isoort(1) = 0
		if ( lwrite ) print *,'ffxxyz: xk=0, ma=mb -> S3 =0'
		return
	    endif
	    isoort(1) = 1
	    y(1) = etami(is2) / piDpj(is1,ip1) /2
	    y(2) = y(1)
	    y(3) = - etami(is1) / piDpj(is1,ip1) /2
	    y(4) = y(3)
	    z(1) = xpi(is2) / piDpj(is1,ip1) /2
	    z(2) = z(1)
	    z(3) = - xpi(is1) / piDpj(is1,ip1) /2
	    z(4) = z(3)
	    dyz(1,1) = - etalam / piDpj(is1,ip1) /2
	    dyz(1,2) = dyz(1,1)
	    dyz(2,1) = dyz(1,1)
	    dyz(2,2) = dyz(1,1)
	    if ( ltest ) then
*		check whether we have the correct root ...
		ier0 = ier
		call ffdl2p(delps1,xpi,dpipj,piDpj,
     +			ip1,ip2,ip3,is1,is2,is3,ns,ier0)
		disc = delps1/sdel2p
		xheck = piDpj(ip1,is2) + disc
		if ( xloss*abs(xheck) .gt. precx*max(abs(piDpj(ip1,
     +			is2)),abs(disc)) ) call fferr(37,ier)
	    endif
	    ier1 = ier
	    do 10 i=1,3,2
		dy2z(i) = y(i) - 2*z(i)
		smax = abs(y(i))
		if ( lwarn .and. abs(dy2z(i)) .lt. xloss*smax ) then
		    ier0 = ier
		    call ffwarn(152,ier0,dy2z(i),smax)
		    ier1 = max(ier1,ier0)
		    if ( lwrite ) print *,'special case xk = 0'
		endif
		dy2z(i+1) = dy2z(i)
   10	    continue
	    ier = ier1
	    return
	endif
*  #] xk = 0:
*  #[ get y(1,2),z(1,2):
	if ( sdel2s .eq. 0 ) then
	    isoort(1) = 2
	    isoort(2) = 2
	    z(1) = piDpj(ip1,is2)/xpi(ip1)
	    z(2) = z(1)
	else
	    isoort(1) = 1
	    isoort(2) = 1
	    call ffroot(z(1),z(2),xpi(ip1),piDpj(ip1,is2),xpi(is2),
     +							sdel2s,ier)
	endif
*	if ( ltest ) then
*	    call ffdl2p(delps1,xpi,dpipj,piDpj,
*     +		ip1,ip2,ip3,is1,is2,is3,ns,ier)
*	    if ( delps .ne. delps1 ) print *,'ffxxyz: error: delps != ',
*     +		'delps1: ',delps,delps1
*	endif
	disc = delps/sdel2p
	ier0 = ier
	call ffroot(y(1),y(2),xpi(ip1),piDpj(ip1,is2),etami(is2),disc,
     +								ier)
*  #] get y(1,2),z(1,2):
*  #[ get y(3,4),z(3,4):
*	if ( xpi(is1) .eq. xpi(is2) ) then
*	    y(4) = y(1)
*	    y(3) = y(2)
*	    z(4) = z(1)
*	    z(3) = z(2)
*	else
	    if ( isoort(1) .eq. 2 ) then
		z(3) = -piDpj(ip1,is1)/xpi(ip1)
		z(4) = z(3)
	    else
		z(3) = 1-z(1)
		z(4) = 1-z(2)
		if ( abs(z(3)) .lt. xloss .or. abs(z(4)) .lt. xloss )
     +		    call ffroot(z(4),z(3),xpi(ip1),-piDpj(ip1,is1),
     +						xpi(is1),sdel2s,ier)
	    endif
	    y(3) = 1-y(1)
	    y(4) = 1-y(2)
	    if ( abs(y(3)) .lt. xloss .or. abs(y(4)) .lt. xloss ) then
		call ffroot(y(4),y(3),xpi(ip1),-piDpj(ip1,is1),
     +						etami(is1),disc,ier)
	    endif
*	endif
*  #] get y(3,4),z(3,4):
*  #[ get dyz:
*	Note that dyz(i,j) only exists for i,j=1,2!
	if ( isoort(1) .eq. 2 ) then
	    dyz(2,1) = disc/xpi(ip1)
	    dyz(2,2) = dyz(2,1)
	elseif ( disc .gt. 0 .eqv. sdel2s .gt. 0 ) then
	    dyz(2,1) = ( disc + sdel2s )/xpi(ip1)
	    dyz(2,2) = etalam/(xpi(ip1)*dyz(2,1))
	else
	    dyz(2,2) = ( disc - sdel2s )/xpi(ip1)
	    dyz(2,1) = etalam/(xpi(ip1)*dyz(2,2))
	endif
	dyz(1,1) = -dyz(2,2)
	dyz(1,2) = -dyz(2,1)
	d2yzz = 2*disc/xpi(ip1)
*
*	these are very rarely needed, but ...
*
	iwarn = 0
	ier1 = ier
	do 20 i=1,4
	    j = 2*((i+1)/2)
	    dy2z(i) = y(j) - 2*z(i)
	    smax = abs(y(j))
	    if ( abs(dy2z(i)) .lt. xloss*smax ) then
		if ( lwrite ) print *,'  dy2z(',i,') = ',dy2z(i),smax
		if ( i/2 .eq. 1 ) then
		    s = -y(j-1) - 2*sdel2s/xpi(ip1)
		else
		    s = -y(j-1) + 2*sdel2s/xpi(ip1)
		endif
		if ( lwrite ) print *,'  dy2z(',i,')+= ',s,y(j-1)
		if ( abs(y(j-1)) .lt. smax ) then
		    dy2z(i) = s
		    smax = abs(y(j-1))
		endif
		if ( abs(dy2z(i)) .lt. xloss*smax ) then
		    if ( iwarn .ne. 0 ) then
			if ( lwarn ) then
			    ier0 = ier
			    call ffwarn(152,ier0,dy2z(i),smax)
			    ier1 = max(ier1,ier0)
			    if ( lwrite ) print *,'iwarn = ',i
			endif
		    else
			iwarn = i
			xmax = smax
		    endif
		endif
	    endif
   20	continue
	if ( iwarn .ne. 0 ) then
*
*	    we should import the differences, but later...
*
	    if ( abs(dpipj(is3,ip1)) .lt. xloss*xpi(is3)
     +		.and. abs(dpipj(is1,is2)) .lt. xloss*abs(xpi(ip1))) then
*
*		give it another try - multiply roots (see dy2z.frm)
*
		if ( iwarn.lt.3 ) then
*prod1=
*	som(1)=+160*xpi(ip1)*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2*
*     +	dpipj(is2,is1)**2
*	som(2)=-40*xpi(ip1)*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)**3
*	som(3)=-32*xpi(ip1)*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)**3
*	som(4)=+9*xpi(ip1)*xpi(ip2)**2*dpipj(is2,is1)**4
*	som(5)=-128*xpi(ip1)*xpi(is2)*piDpj(ip1,ip2)**3*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)
*	som(6)=-128*xpi(ip1)*xpi(is2)*piDpj(ip1,ip2)**4*dpipj(is2,
*     +	is1)
*	som(7)=+256*xpi(ip1)*xpi(is2)**2*piDpj(ip1,ip2)**4
*	som(8)=-16*xpi(ip1)*piDpj(ip1,ip2)**2*piDpj(ip2,is2)**2*
*     +	dpipj(is2,is1)**2
*	som(9)=+96*xpi(ip1)*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*dpipj(is2,
*     +	is1)**2
*	som(10)=+128*xpi(ip1)**2*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)*piDpj(
*     +	ip2,is2)*dpipj(is2,is1)
*	som(11)=+320*xpi(ip1)**2*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2*
*     +	dpipj(is2,is1)
*	som(12)=-512*xpi(ip1)**2*xpi(ip2)*xpi(is2)**2*piDpj(ip1,ip2)**2
*	som(13)=-120*xpi(ip1)**2*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)**2
*	som(14)=-48*xpi(ip1)**2*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)**2
*	som(15)=+40*xpi(ip1)**2*xpi(ip2)*piDpj(ip2,is2)**2*dpipj(is2,
*     +	is1)**2
*	som(16)=-96*xpi(ip1)**2*xpi(ip2)**2*xpi(is2)*dpipj(is2,is1)**2
*	som(17)=+36*xpi(ip1)**2*xpi(ip2)**2*dpipj(is2,is1)**3
*	som(18)=+128*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**2*piDpj(ip2,
*     +	is2)**2
*	som(19)=-128*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**3*piDpj(ip2,
*     +	is2)
*	som(20)=-64*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**4
*	som(21)=-32*xpi(ip1)**2*piDpj(ip1,ip2)*piDpj(ip2,is2)**3*
*     +	dpipj(is2,is1)
*	som(22)=-32*xpi(ip1)**2*piDpj(ip1,ip2)**2*piDpj(ip2,is2)**2*
*     +	dpipj(is2,is1)
*	som(23)=+96*xpi(ip1)**2*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*
*     +	dpipj(is2,is1)
*	som(24)=+128*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)*piDpj(
*     +	ip2,is2)
*	som(25)=+160*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2
*	som(26)=-128*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip2,is2)**2
*	som(27)=+32*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(28)=-120*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)
*	som(29)=-32*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)
*	som(30)=-16*xpi(ip1)**3*xpi(ip2)*piDpj(ip2,is1)*piDpj(ip2,
*     +	is2)**2
*	som(31)=+80*xpi(ip1)**3*xpi(ip2)*piDpj(ip2,is2)**2*dpipj(is2,
*     +	is1)
*	som(32)=-192*xpi(ip1)**3*xpi(ip2)**2*xpi(is2)*dpipj(is2,is1)
*	som(33)=+256*xpi(ip1)**3*xpi(ip2)**2*xpi(is2)**2
*	som(34)=+54*xpi(ip1)**3*xpi(ip2)**2*dpipj(is2,is1)**2
*	som(35)=-16*xpi(ip1)**3*xpi(ip3)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(36)=+8*xpi(ip1)**3*xpi(ip3)*piDpj(ip2,is1)*piDpj(ip2,is2)**2
*	som(37)=+16*xpi(ip1)**3*xpi(is2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(38)=-8*xpi(ip1)**3*xpi(is2)*piDpj(ip2,is1)*piDpj(ip2,is2)**2
*	som(39)=-16*xpi(ip1)**3*piDpj(ip1,ip2)*piDpj(ip2,is1)*piDpj(ip2,
*     +	is2)*dpipj(is3,ip1)
*	som(40)=+8*xpi(ip1)**3*piDpj(ip2,is1)*piDpj(ip2,is2)**2*
*     +	dpipj(is3,ip1)
*	som(41)=-40*xpi(ip1)**4*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,is2)
*	som(42)=-8*xpi(ip1)**4*xpi(ip2)*piDpj(ip1,ip2)**2
*	som(43)=+40*xpi(ip1)**4*xpi(ip2)*piDpj(ip2,is2)**2
*	som(44)=-96*xpi(ip1)**4*xpi(ip2)**2*xpi(is2)
*	som(45)=+36*xpi(ip1)**4*xpi(ip2)**2*dpipj(is2,is1)
*	som(46)=+9*xpi(ip1)**5*xpi(ip2)**2
*	som(47)=-8*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,is1)**4
*	som(48)=-64*xpi(is2)*piDpj(ip1,ip2)**4*dpipj(is2,is1)**2
*	som(49)=+32*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*dpipj(is2,is1)**3
*	print '(7g20.12)',(som(i),i=1,49)
*
*	optimized by Maple (see ffxxyz.map)
*
	t1 = xpi(ip1)
	t2 = xpi(ip2)
	t3 = t1*t2
	t4 = xpi(is2)
	t5 = piDpj(ip1,ip2)
	t6 = t5**2
	t7 = t4*t6
	t8 = dpipj(is2,is1)
	t9 = t8**2
	som(1) = 160*t3*t7*t9
	t12 = piDpj(ip2,is2)
	t13 = t5*t12
	t14 = t9*t8
	som(2) = -40*t3*t13*t14
	som(3) = -32*t3*t6*t14
	t19 = t2**2
	t21 = t9**2
	som(4) = 9*t1*t19*t21
	t23 = t1*t4
	t24 = t6*t5
	t25 = t24*t12
	som(5) = -128*t23*t25*t8
	t28 = t6**2
	som(6) = -128*t23*t28*t8
	t31 = t4**2
	som(7) = 256*t1*t31*t28
	t35 = t12**2
	t36 = t35*t9
	som(8) = -16*t1*t6*t36
	som(9) = 96*t1*t24*t12*t9
	t41 = t1**2
	t42 = t41*t2
	t44 = t13*t8
	som(10) = 128*t42*t4*t44
	som(11) = 320*t42*t7*t8
	som(12) = -512*t42*t31*t6
	som(13) = -120*t42*t13*t9
	som(14) = -48*t42*t6*t9
	som(15) = 40*t42*t36
	t55 = t41*t19
	som(16) = -96*t55*t4*t9
	som(17) = 36*t55*t14
	t59 = t41*t4
	som(18) = 128*t59*t6*t35
	som(19) = -128*t59*t25
	som(20) = -64*t59*t28
	som(21) = -32*t41*t5*t35*t12*t8
	t69 = t35*t8
	som(22) = -32*t41*t6*t69
	som(23) = 96*t41*t24*t12*t8
	t74 = t41*t1
	t75 = t74*t2
	som(24) = 128*t75*t4*t5*t12
	som(25) = 160*t75*t7
	som(26) = -128*t75*t4*t35
	t82 = piDpj(ip2,is1)
	t84 = t5*t82*t12
	som(27) = 32*t75*t84
	som(28) = -120*t75*t44
	som(29) = -32*t75*t6*t8
	t89 = t82*t35
	som(30) = -16*t75*t89
	som(31) = 80*t75*t69
	t92 = t74*t19
	som(32) = -192*t92*t4*t8
	som(33) = 256*t92*t31
	som(34) = 54*t92*t9
	t98 = t74*xpi(ip3)
	som(35) = -16*t98*t84
	som(36) = 8*t98*t89
	t101 = t74*t4
	som(37) = 16*t101*t84
	som(38) = -8*t101*t89
	t106 = dpipj(is3,ip1)
	som(39) = -16*t74*t5*t82*t12*t106
	som(40) = 8*t74*t82*t35*t106
	t112 = t41**2
	t113 = t112*t2
	som(41) = -40*t113*t13
	som(42) = -8*t113*t6
	som(43) = 40*t113*t35
	t117 = t112*t19
	som(44) = -96*t117*t4
	som(45) = 36*t117*t8
	som(46) = 9*t112*t1*t19
	som(47) = -8*t2*t6*t21
	som(48) = -64*t4*t28*t9
	som(49) = 32*t25*t14
*	print '(7g20.12)',(som(i),i=1,49)
	n=49
		else
*prod3=
*	som(1)=+160*xpi(ip1)*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2*
*     +	dpipj(is2,is1)**2
*	som(2)=-40*xpi(ip1)*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)**3
*	som(3)=-88*xpi(ip1)*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)**3
*	som(4)=+9*xpi(ip1)*xpi(ip2)**2*dpipj(is2,is1)**4
*	som(5)=-128*xpi(ip1)*xpi(is2)*piDpj(ip1,ip2)**3*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)
*	som(6)=-256*xpi(ip1)*xpi(is2)*piDpj(ip1,ip2)**4*dpipj(is2,is1)
*	som(7)=+256*xpi(ip1)*xpi(is2)**2*piDpj(ip1,ip2)**4
*	som(8)=-16*xpi(ip1)*piDpj(ip1,ip2)**2*piDpj(ip2,is2)**2*dpipj(
*     +	is2,is1)**2
*	som(9)=+64*xpi(ip1)*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*dpipj(is2,
*     +	is1)**2
*	som(10)=+80*xpi(ip1)*piDpj(ip1,ip2)**4*dpipj(is2,is1)**2
*	som(11)=+128*xpi(ip1)**2*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)*piDpj(
*     +	ip2,is2)*dpipj(is2,is1)
*	som(12)=+576*xpi(ip1)**2*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2*
*     +	dpipj(is2,is1)
*	som(13)=-512*xpi(ip1)**2*xpi(ip2)*xpi(is2)**2*piDpj(ip1,ip2)**2
*	som(14)=-88*xpi(ip1)**2*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)**2
*	som(15)=-192*xpi(ip1)**2*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)**2
*	som(16)=+40*xpi(ip1)**2*xpi(ip2)*piDpj(ip2,is2)**2*dpipj(is2,
*     +	is1)**2
*	som(17)=-96*xpi(ip1)**2*xpi(ip2)**2*xpi(is2)*dpipj(is2,is1)**2
*	som(18)=+60*xpi(ip1)**2*xpi(ip2)**2*dpipj(is2,is1)**3
*	som(19)=+128*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**2*piDpj(ip2,
*     +	is2)**2
*	som(20)=-128*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**3*piDpj(ip2,
*     +	is2)
*	som(21)=-64*xpi(ip1)**2*xpi(is2)*piDpj(ip1,ip2)**4
*	som(22)=-32*xpi(ip1)**2*piDpj(ip1,ip2)*piDpj(ip2,is2)**3*
*     +	dpipj(is2,is1)
*	som(23)=+64*xpi(ip1)**2*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*
*     +	dpipj(is2,is1)
*	som(24)=+32*xpi(ip1)**2*piDpj(ip1,ip2)**4*dpipj(is2,is1)
*	som(25)=+128*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)*piDpj(
*     +	ip2,is2)
*	som(26)=+160*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip1,ip2)**2
*	som(27)=-128*xpi(ip1)**3*xpi(ip2)*xpi(is2)*piDpj(ip2,is2)**2
*	som(28)=+32*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(29)=-88*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is2)*dpipj(is2,is1)
*	som(30)=-88*xpi(ip1)**3*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,
*     +	is1)
*	som(31)=-16*xpi(ip1)**3*xpi(ip2)*piDpj(ip2,is1)*piDpj(ip2,
*     +	is2)**2
*	som(32)=+48*xpi(ip1)**3*xpi(ip2)*piDpj(ip2,is2)**2*dpipj(is2,
*     +	is1)
*	som(33)=-320*xpi(ip1)**3*xpi(ip2)**2*xpi(is2)*dpipj(is2,is1)
*	som(34)=+256*xpi(ip1)**3*xpi(ip2)**2*xpi(is2)**2
*	som(35)=+118*xpi(ip1)**3*xpi(ip2)**2*dpipj(is2,is1)**2
*	som(36)=-16*xpi(ip1)**3*xpi(ip3)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(37)=+8*xpi(ip1)**3*xpi(ip3)*piDpj(ip2,is1)*piDpj(ip2,is2)**2
*	som(38)=+16*xpi(ip1)**3*xpi(is2)*piDpj(ip1,ip2)*piDpj(ip2,
*     +	is1)*piDpj(ip2,is2)
*	som(39)=-8*xpi(ip1)**3*xpi(is2)*piDpj(ip2,is1)*piDpj(ip2,is2)**2
*	som(40)=-16*xpi(ip1)**3*piDpj(ip1,ip2)*piDpj(ip2,is1)*piDpj(ip2,
*     +	is2)*dpipj(is3,ip1)
*	som(41)=+8*xpi(ip1)**3*piDpj(ip2,is1)*piDpj(ip2,is2)**2*
*     +	dpipj(is3,ip1)
*	som(42)=-40*xpi(ip1)**4*xpi(ip2)*piDpj(ip1,ip2)*piDpj(ip2,is2)
*	som(43)=-8*xpi(ip1)**4*xpi(ip2)*piDpj(ip1,ip2)**2
*	som(44)=+40*xpi(ip1)**4*xpi(ip2)*piDpj(ip2,is2)**2
*	som(45)=-96*xpi(ip1)**4*xpi(ip2)**2*xpi(is2)
*	som(46)=+60*xpi(ip1)**4*xpi(ip2)**2*dpipj(is2,is1)
*	som(47)=+9*xpi(ip1)**5*xpi(ip2)**2
*	som(48)=-8*xpi(ip2)*piDpj(ip1,ip2)**2*dpipj(is2,is1)**4
*	som(49)=-64*xpi(is2)*piDpj(ip1,ip2)**4*dpipj(is2,is1)**2
*	som(50)=+32*piDpj(ip1,ip2)**3*piDpj(ip2,is2)*dpipj(is2,is1)**3
*	som(51)=+32*piDpj(ip1,ip2)**4*dpipj(is2,is1)**3
*	print '(7g20.12)',(som(i),i=1,51)
*
*	optimized by Maple (see ffxxyz.map)
*
	t126 = xpi(ip1)
	t127 = xpi(ip2)
	t128 = t126*t127
	t129 = xpi(is2)
	t130 = piDpj(ip1,ip2)
	t131 = t130**2
	t132 = t129*t131
	t133 = dpipj(is2,is1)
	t134 = t133**2
	som(1) = 160*t128*t132*t134
	t137 = piDpj(ip2,is2)
	t138 = t130*t137
	t139 = t134*t133
	som(2) = -40*t128*t138*t139
	som(3) = -88*t128*t131*t139
	t144 = t127**2
	t146 = t134**2
	som(4) = 9*t126*t144*t146
	t148 = t126*t129
	t149 = t131*t130
	t150 = t149*t137
	som(5) = -128*t148*t150*t133
	t153 = t131**2
	som(6) = -256*t148*t153*t133
	t156 = t129**2
	som(7) = 256*t126*t156*t153
	t160 = t137**2
	t161 = t160*t134
	som(8) = -16*t126*t131*t161
	som(9) = 64*t126*t149*t137*t134
	som(10) = 80*t126*t153*t134
	t168 = t126**2
	t169 = t168*t127
	t171 = t138*t133
	som(11) = 128*t169*t129*t171
	som(12) = 576*t169*t132*t133
	som(13) = -512*t169*t156*t131
	som(14) = -88*t169*t138*t134
	som(15) = -192*t169*t131*t134
	som(16) = 40*t169*t161
	t182 = t168*t144
	som(17) = -96*t182*t129*t134
	som(18) = 60*t182*t139
	t186 = t168*t129
	som(19) = 128*t186*t131*t160
	som(20) = -128*t186*t150
	som(21) = -64*t186*t153
	som(22) = -32*t168*t130*t160*t137*t133
	som(23) = 64*t168*t149*t137*t133
	som(24) = 32*t168*t153*t133
	t200 = t168*t126
	t201 = t200*t127
	som(25) = 128*t201*t129*t130*t137
	som(26) = 160*t201*t132
	som(27) = -128*t201*t129*t160
	t208 = piDpj(ip2,is1)
	t210 = t130*t208*t137
	som(28) = 32*t201*t210
	som(29) = -88*t201*t171
	som(30) = -88*t201*t131*t133
	t215 = t208*t160
	som(31) = -16*t201*t215
	som(32) = 48*t201*t160*t133
	t219 = t200*t144
	som(33) = -320*t219*t129*t133
	som(34) = 256*t219*t156
	som(35) = 118*t219*t134
	t225 = t200*xpi(ip3)
	som(36) = -16*t225*t210
	som(37) = 8*t225*t215
	t228 = t200*t129
	som(38) = 16*t228*t210
	som(39) = -8*t228*t215
	t233 = dpipj(is3,ip1)
	som(40) = -16*t200*t130*t208*t137*t233
	som(41) = 8*t200*t208*t160*t233
	t239 = t168**2
	t240 = t239*t127
	som(42) = -40*t240*t138
	som(43) = -8*t240*t131
	som(44) = 40*t240*t160
	t244 = t239*t144
	som(45) = -96*t244*t129
	som(46) = 60*t244*t133
	som(47) = 9*t239*t126*t144
	som(48) = -8*t127*t131*t146
	som(49) = -64*t129*t153*t134
	som(50) = 32*t150*t139
	som(51) = 32*t153*t139
*	print '(7g20.12)',(som(i),i=1,51)
	n=51
		endif
*
		s = 0
		smax = 0
		do 30 j=1,n
		    s = s + som(j)
		    smax = max(smax,som(j))
   30		continue
		if ( iwarn .lt. 3 ) then
		    hulp = 1/(16*xpi(ip1)**3*sdel2p**4*dy2z(3-iwarn)*
     +			(y(1)-2*z(1))*(y(1)-2*z(2)))
		else
		    hulp = 1/(16*xpi(ip1)**3*sdel2p**4*dy2z(7-iwarn)*
     +			(y(3)-2*z(3))*(y(3)-2*z(4)))
		endif
		s = s*hulp
		smax = smax*hulp
		if ( lwrite ) print *,'  dy2z(',iwarn,')++=',s,smax
		if ( smax .lt. xmax ) then
		    dy2z(iwarn) = s
		    xmax = smax
		endif
	    else
		n=0
	    endif
	    if ( lwarn .and. abs(dy2z(iwarn)) .lt. xloss*xmax ) then
		ier0 = ier
		call ffwarn(152,ier0,dy2z(iwarn),xmax)
		ier1 = max(ier1,ier0)
		if ( lwrite ) then
		    print *,'n = ',n
		    print *,'xpi = ',xpi
		    print *,'cs = '
		    print '(i3,g24.12)',(i,som(i),i=1,n)
		endif
	    endif
	endif
	ier = ier1
*
	goto 200
*  #] get dyz:
*  #[ special case, get indices:
  100	continue
	if ( ivert.eq.2 ) then
	    is1 = 2
	    ip1 = 5
	else
	    is1 = 1
	    ip1 = 6
	endif
*  #] special case, get indices:
*  #[ xk = 0:
	if ( xpi(ip1) .eq. 0 ) then
	    call fferr(88,ier)
	endif
*  #] xk = 0:
*  #[ get ypm,zpm:
*
*	special case del2s = 0, hence the roots are not the real roots
*	but z_2'' = (z_2'-1)/delta, z''_3 = -z'_3/delta
*
	hulp = sdel2s
	disc = delps/sdel2p
	if ( ivert .eq. 3 ) then
	    hulp = -hulp
	    disc = -disc
	endif
	if ( sdel2s .eq. 0 ) then
	    isoort(1) = 102
	    isoort(2) = 102
	    z(1) = piDpj(is1,3)/xpi(3)
	    z(2) = z(1)
	else
	    isoort(1) = 101
	    isoort(2) = 101
	    call ffroot(z(1),z(2),xpi(3),piDpj(is1,3),xpi(is1),hulp,ier)
	endif
	call ffroot(y(1),y(2),xpi(3),piDpj(is1,3),etami(is1),disc,ier)
*  #] get ypm,zpm:
*  #[ get ypm1,zpm1:
	z(3) = 1 - z(1)
	z(4) = 1 - z(2)
	if ( abs(z(3)).lt.xloss .or. abs(z(4)).lt.xloss ) then
	    if ( lwrite ) print *,'z(3,4) = ',z(3),z(4)
	    if ( ivert.eq.2 ) then
		call ffroot(z(4),z(3),xpi(3),piDpj(ip1,3),xpi(ip1),hulp,
     +								ier)
	    else
		call ffroot(z(4),z(3),xpi(3),-piDpj(ip1,3),xpi(ip1),hulp
     +								,ier)
	    endif
	    if ( lwrite ) print *,'z(3,4)+= ',z(3),z(4)
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
*  #] get ypm1,zpm1:
*  #[ get dypzp, dypzm:
	if ( isoort(1) .eq. 2 ) then
	    dyz(2,1) = disc/xpi(3)
	    dyz(2,2) = dyz(2,1)
	elseif ( disc .gt. 0 .eqv. sdel2s .gt. 0 ) then
	    dyz(2,1) = ( disc + hulp )/xpi(3)
	    dyz(2,2) = etalam/(xpi(3)*dyz(2,1))
	else
	    dyz(2,2) = ( disc - hulp )/xpi(3)
	    dyz(2,1) = etalam/(xpi(3)*dyz(2,2))
	endif
	dyz(1,1) = -dyz(2,2)
	dyz(1,2) = -dyz(2,1)
	d2yzz = 2*disc/xpi(3)
*
*	these are very rarely needed, but ...
*
	do 220 i=1,4
	    j = 2*((i+1)/2)
	    dy2z(i) = y(j) - 2*z(i)
	    smax = abs(y(j))
*	    do not know whether this is correct! 29-mar-1990
*	    if ( abs(dy2z(i)) .lt. xloss*smax ) then
*		if ( lwrite ) print *,'  dy2z(',i,') = ',dy2z(i),smax
*		if ( i/2 .eq. 1 ) then
*		    s = -y(j-1) - 2*hulp/xpi(3)
*		else
*		    s = -y(j-1) + 2*hulp/xpi(3)
*		endif
*		if ( abs(y(j-1)) .lt. smax ) then
*		    dy2z(i) = s
*		    smax = abs(y(j-1))
*		endif
*		if ( lwrite ) print *,'  dy2z(',i,')+= ',s,y(j-1)
		if ( lwarn .and. abs(dy2z(i)) .lt. xloss*smax ) then
		    call ffwarn(152,ier,dy2z(i),abs(y(j-1)))
		endif
*	    endif
  220	continue
*  #] get dypzp, dypzm:
*  #[ test output:
  200	continue
	if ( ltest ) then
	    xlosn = xloss**2*DBLE(10)**(-mod(ier,50))
	    do 99 i=1,2
		xheck = y(i)+y(i+2)-1
		if ( xlosn*abs(xheck) .gt. precx*max(abs(y(i)),
     +		    abs(y(i+2)),x1) ) print *,'ffxxyz: error: ',
     +		    'y(',i+2,')<>1-y(',i,'):',y(i+2),y(i),xheck
		xheck = z(i)+z(i+2)-1
		if ( xlosn*abs(xheck) .gt. precx*max(abs(z(i)),
     +		    abs(z(i+2)),x1) ) print *,'ffxxyz: error: ',
     +		    'z(',i+2,')<>1-z(',i,'):',z(i+2),z(i),xheck
		xheck = dy2z(i)-y(2)+2*z(i)
		if ( xlosn*abs(xheck) .gt. precx*max(abs(y(2)),
     +		    abs(2*z(i))) ) print *,'ffxxyz: error: ',
     +		    'dy2z(',i,')<>y(2)-2*z(',i,'):',dy2z(i),y(2),2*z(i),
     +		    xheck
		xheck = dy2z(i+2)-y(4)+2*z(i+2)
		if ( xlosn*abs(xheck) .gt. precx*max(abs(y(4)),
     +		    abs(2*z(i+2)))) print *,'ffxxyz: error: ',
     +		    'dy2z(',i+2,')<>y(4)-2*z(',i+2,'):',dy2z(i+2),y(4),
     +		    2*z(i+2),xheck
		do 98 j=1,2
		    if ( xlosn*abs(dyz(i,j)-y(i)+z(j)) .gt. precx*max(
     +			abs(dyz(i,j)),abs(y(i)),abs(z(j))) ) print *,
     +			'ffxxyz: error: dyz(',i,j,') <> y(',i,')-z(',j,
     +			'):',dyz(i,j),y(i),z(j),dyz(i,j)-y(i)+z(j)
   98		continue
   99	    continue
	    if ( xlosn*abs(d2yzz-2*y(2)+z(1)+z(2)) .gt. precx*max(abs(
     +		d2yzz),2*abs(y(2)),abs(z(1)),abs(z(2))) ) print *,
     +		'ffxxyz: error: d2yzz <> 2*y(2)+z(1)+z(2):',d2yzz,2*
     +		y(2),z(1),z(2),d2yzz-2*y(2)+z(1)+z(2)
	endif
*  #] test output:
*###] ffxxyz:
	end
*###[ ffdwz:
	subroutine ffdwz(dwz,w,z,i1,j1,l,alpha,alph1,xpi,dpipj,piDpj,
     +							sdel2i,ns,ier)
***#[*comment:***********************************************************
*									*
*	Recalculate dwz(i1,j1) = w(i1) - z(j1)				*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer i1,j1,l,ns,ier
	DOUBLE PRECISION dwz(2,2),w(4),z(4)
	DOUBLE PRECISION alpha,alph1,xpi(ns),dpipj(ns,ns),piDpj(ns,ns),
     +		sdel2i(3)
*
*	local variables:
*
	DOUBLE PRECISION s(8),sum,fac,xmax
	integer i
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ calculations:
	if ( l .eq. 1 ) then
	    if ( lwrite ) print *,'ffdwz:  warning: cannot handle',
     +			' this case dwz(',i1,j1,l,') yet'
	    ier = ier + 100
	elseif ( l .eq. 3 ) then
	    if ( (i1.eq.2 .and. j1.eq.1) .or. (i1.eq.1 .and. j1.eq.2) )
     +			then
		fac = x1/(sdel2i(2) + sdel2i(3))
		s(1) = dpipj(6,5)*z(j1)
		s(2) = -alph1*xpi(5)*z(j1+2)
		if ( max(abs(dpipj(2,1)),abs(dpipj(5,6))) .lt.
     +		     max(abs(dpipj(2,6)),abs(dpipj(5,1))) ) then
		    s(3) = x05*dpipj(2,1)
		    s(4) = x05*dpipj(5,6)
		else
		    s(3) = x05*dpipj(2,6)
		    s(4) = x05*dpipj(5,1)
		endif
		s(5) = piDpj(4,3)*piDpj(5,3)*fac
		s(6) = -piDpj(4,3)*piDpj(6,3)*fac
		s(7) = xpi(3)*dpipj(5,6)*fac
		if ( i1 .eq. 1 ) then
		    sum = s(1)+s(2)+s(3)+s(4) - (s(5)+s(6)+s(7))
		else
		    sum = s(1)+s(2)+s(3)+s(4) + s(5)+s(6)+s(7)
		endif
		xmax = abs(s(1))
		do 10 i=2,7
		    xmax = max(xmax,abs(s(i)))
   10		continue
		if ( abs(sum) .lt. xloss*xmax ) then
*		    this result is not used if it is not accurate (see
*		    ffxc0p)
		    if ( lwrite ) then
			call ffwarn(79,ier,sum,xmax)
		    else
			ier = ier + 1
		    endif
		    xmax = xmax/abs(alpha*xpi(5))
*		    if ( xmax .lt. min(abs(z(j1)),abs(z(j1+2))) ) then
			if (lwrite) print *,'  dwz(',i1,j1,l,')  = ',
     +				dwz(i1,j1),min(abs(z(j1)),abs(z(j1+2)))
			dwz(i1,j1) = sum/(alpha*xpi(5))
			if (lwrite) print *,'  dwz(',i1,j1,l,')+ = ',
     +					dwz(i1,j1),xmax/(alpha*xpi(5))
*		    endif
		else
		    if (lwrite) print *,'  dwz(',i1,j1,l,')  = ',
     +							dwz(i1,j1)
		    dwz(i1,j1) = sum/(alpha*xpi(5))
		    if (lwrite) print *,'  dwz(',i1,j1,l,')+ = ',
     +							dwz(i1,j1)
		endif
	    else
		if ( lwrite ) print *,'ffdwz:  warning: cannot handle',
     +			' this case dwz(',i1,j1,l,') yet'
		ier = ier + 100
	    endif
	endif
*  #] calculations:
*  #[ test output:
	if ( ltest .and. ier .eq. 0 ) then
	    if ( xloss*abs(dwz(i1,j1)-w(i1)+z(j1)) .gt. precx*max(
     +		abs(dwz(i1,j1)),abs(w(i1)),abs(z(j1))) ) print *,
     +		'ffdwz:  error: dwz(',i1,j1,l,') <> w - z :',
     +		dwz(i1,j1),w(i1),z(j1),dwz(i1,j1)-w(i1)+z(j1)
	    if ( xloss*abs(dwz(i1,j1)+w(i1+2)-z(j1+2)) .gt. precx*max(
     +		abs(dwz(i1,j1)),abs(w(i1+2)),abs(z(j1+2))) ) print *,
     +		'ffdwz:  error: dwz(',i1,j1,l,') <> z1 - w1 :',
     +		dwz(i1,j1),z(i1+2),w(j1+2),dwz(i1,j1)+w(i1+2)-z(j1+2)
	    endif
*  #] test output:
*###] ffdwz:
	end
