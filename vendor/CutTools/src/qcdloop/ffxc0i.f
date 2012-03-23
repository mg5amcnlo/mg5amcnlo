*--#[ log:
*	$Id: ffxc0i.f,v 1.3 1996/06/03 12:11:43 gj Exp $
*	$Log: ffxc0i.f,v $
c Revision 1.3  1996/06/03  12:11:43  gj
c Added an error message for ffxc0j with zero masses, which is ill-defined.
c
c Revision 1.2  1995/12/01  15:04:40  gj
c Fixed a ridiculous bug: wrong sign for p4^2=0, m2<m1.
c
*--#] log:
*###[ ffxc0i:
	subroutine ffxc0i(cc0,xpi,dpipj,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the infrared finite part of a infrared divergent	*
*	threepoint function with the factor ipi^2.  The cutoff		*
*	parameter is assumed to be in a common block /ffcut/. (ugly)	*
*									*
*	Input:	xpi(6)		(real)	pi.pi (B&D)			*
*		dpipj(6,6)	(real)	xpi(i)-xpi(j)			*
*		/ffcut/delta	(real)	cutoff (either foton mass**2 or	*
*						radiation limit).	*
*	Output: cc0	(complex)	C0, the threepoint function.	*
*		ier	(integer)	usual error flag		*
*	Calls:								*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cc0
	DOUBLE PRECISION xpi(6),dpipj(6,6)
*
*	local variables
*
	integer init,ipi12,i,ilogi(3),irota,n
	integer j,inew(6,6)
	DOUBLE COMPLEX cs(15),csum,c,clogi(3)
	DOUBLE PRECISION xqi(6),dqiqj(6,6),qiDqj(6,6),sdel2,xmax,absc,
     +		dum66(6,6),del2
	save init,inew,ilogi
*
*	common blocks etc
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	data
*
	data init /0/
	data inew /1,2,3,4,5,6,
     +		   2,3,1,5,6,4,
     +		   3,1,2,6,4,5,
     +		   1,3,2,6,5,4,
     +		   3,2,1,5,4,6,
     +		   2,1,3,4,6,5/
	data ilogi /3*0/
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	initialisations
*
	do 1 i=1,15
	    cs(i) = 0
    1	continue
	ipi12 = 0
*  #] declarations:
*  #[ check input:
	if ( init .eq. 0 .and. .not.lsmug ) then
	    init = 1
	    print *,'ffxc0i: infra-red divergent threepoint function, ',
     +		'working with a cutoff ',delta
	endif
	if ( .not.lsmug .and. delta .eq. 0 ) then
	    call fferr(59,ier)
	    return
	endif
	if ( lwrite ) then
*	    print input
	    print *,'ffxc0i: infrared divergent threepoint function'
	    if ( .not.lsmug ) then
		print *,'  cutoff parameter:',delta
	    endif
	endif
*  #] check input:
*  #[ groundwork:
*
*	rotate to xpi(3)=0, xpi(1)=xpi(6), xpi(2)=xpi(5)
*
	call ffrot3(irota,xqi,dqiqj,qiDqj,xpi,dpipj,dum66,6,3,3,ier)
*
*	get some dotproducts
*
	if ( ldot ) then
	    call ffdot3(qiDqj,xqi,dqiqj,6,ier)
	    do 5 i=1,6
		do 4 j=1,6
		    fpij3(j,i) = qiDqj(inew(i,irota),inew(j,irota))
    4		continue
    5	    continue
	else
	if ( abs(xqi(4)) .lt. xqi(1) ) then
	    qiDqj(4,1) = dqiqj(2,1) - xqi(4)
	    xmax = abs(xqi(4))
	else
	    qiDqj(4,1) = dqiqj(2,4) - xqi(1)
	    xmax = xqi(1)
	endif
	if ( lwarn .and. abs(qiDqj(4,1)) .lt. xloss*xmax )
     +		call ffwarn(156,ier,qiDqj(4,1),xmax)
	qiDqj(4,1) = qiDqj(4,1)/2
	qiDqj(1,4) = qiDqj(4,1)

	if ( abs(xqi(4)) .lt. xqi(2) ) then
	    qiDqj(4,2) = dqiqj(2,1) + xqi(4)
	    xmax = abs(xqi(4))
	else
	    qiDqj(4,2) = xqi(2) - dqiqj(1,4)
	    xmax = xqi(2)
	endif
	if ( lwarn .and. abs(qiDqj(4,2)) .lt. xloss*xmax )
     +		call ffwarn(156,ier,qiDqj(4,2),xmax)
	qiDqj(4,2) = qiDqj(4,2)/2
	qiDqj(2,4) = qiDqj(4,2)

	if ( (xqi(1)) .lt. (xqi(2)) ) then
	    qiDqj(1,2) = xqi(1) + dqiqj(2,4)
	    xmax = xqi(1)
	else
	    qiDqj(1,2) = xqi(2) + dqiqj(1,4)
	    xmax = xqi(2)
	endif
	if ( lwarn .and. abs(qiDqj(1,2)) .lt. xloss*xmax )
     +		call ffwarn(156,ier,qiDqj(1,2),xmax)
	qiDqj(1,2) = qiDqj(1,2)/2
	qiDqj(2,1) = qiDqj(1,2)

	qiDqj(1,1) = xqi(1)
	qiDqj(2,2) = xqi(2)
	qiDqj(4,4) = xqi(4)
	endif
*  #] groundwork:
*  #[ calculations:
*
	call ffdel2(del2,qiDqj,6,1,2,4,1,ier)
	if ( ldot ) fdel2 = del2
*
*	the case del2=0 is hopeless - this is really a two-point function
*
	if ( del2 .eq. 0 ) then
	    call fferr(58,ier)
	    return
	endif
*
*	we cannot yet handle the complex case
*
	if ( del2 .gt. 0 ) then
	    call fferr(83,ier)
	    return
	endif
*
	sdel2 = isgnal*sqrt(-del2)
*
	call ffxc0j(cs,ipi12,sdel2,clogi,ilogi,xqi,dqiqj,qiDqj,
     +		delta,3,ier)
*  #] calculations:
*  #[ sum:
*
*	Sum
*
	xmax = 0
	csum = 0
	if ( .not.lsmug ) then
	    n = 10
	else
	    n = 15
	endif
	do 10 i=1,n
	    csum = csum + cs(i)
	    xmax = max(xmax,absc(csum))
   10	continue
	csum = csum + ipi12*DBLE(pi12)
	if ( lwarn .and. 2*absc(csum) .lt. xloss*xmax ) then
	    call ffwarn(157,ier,absc(csum),xmax)
	endif
	cc0 = -csum*DBLE(1/(2*sdel2))
*  #] sum:
*  #[ debug:
  900	continue
	if (lwrite) then
	    print '(a)','cs(i) = '
	    print '(i3,2g20.10,1x)',(i,cs(i),i=1,n)
	    print '(a3,2g20.10,1x)','pi ',ipi12*pi12
	    print '(a)','+-----------'
	    print '(a3,2g20.10,1x)','som  :',csum
	    print '(a)',' '
	    print *,'cc0  :',cc0,ier
	endif
*  #] debug:
*###] ffxc0i:
	end
*###[ ffxc0j:
	subroutine ffxc0j(cs,ipi12,sdel2i,clogi,ilogi,
     +					xpi,dpipj,piDpj,delta,npoin,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the infrared finite part of a infrared divergent	*
*	threepoint function with the factor ipi^2.			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi12,ilogi(3),npoin,ier
	DOUBLE COMPLEX cs(15),clogi(3)
	DOUBLE PRECISION xpi(6),dpipj(6,6),piDpj(6,6),delta,sdel2i
*
*	local variables
*
	integer i,ieps,ieps1,n,ier0
	DOUBLE COMPLEX clog1,clog2,cdum(2),cel3,cdyzm,cdyzp,cli,chulp,
     +		carg1,carg2,chulp1
	DOUBLE COMPLEX zfflog,zxfflg,cc
	DOUBLE PRECISION del2,zm,zp,zm1,zp1,sdel2,hulp,xheck,dum(3),
     +		dfflo1,dyzp,dyzm,wm,wp,absc,arg1,arg2,del3
*
*	common blocks etc
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    call ffxhck(xpi,dpipj,6,ier)
	endif
	if ( lwrite ) then
	    print '(a)','  ##[ ffxc0j:'
	    print *,'ffxc0j: input: '
	    print *,'xpi   = ',xpi
	    if ( .not.lsmug ) then
		print *,'delta = ',delta
	    else
		print *,'cmipj(2,2) = ',cmipj(2,2)
		print *,'cmipj(1,3) = ',cmipj(1,3)
	    endif
	endif
*  #] check input:
*  #[ get determinants, roots, ieps:
*
	if ( lsmug ) then
	    del3 = (- DBLE(xpi(1))*DBLE(cmipj(2,2))**2
     +		    - DBLE(xpi(2))*DBLE(cmipj(1,3))**2
     +		    + 2*DBLE(piDpj(1,2))*DBLE(cmipj(2,2))*
     +				DBLE(cmipj(1,3)) )/4
	    if ( nschem .ge. 3 ) then
		cel3 = (- DBLE(xpi(1))*cmipj(2,2)**2
     +			- DBLE(xpi(2))*cmipj(1,3)**2
     +			+ 2*DBLE(piDpj(1,2))*cmipj(2,2)*cmipj(1,3) )/4
	    else
		cel3 = DBLE(del3)
	    endif
	    if ( lwrite ) print *,'cel3 = ',cel3
	endif
	del2 = -sdel2i**2
*
*	the routine as it stands can not handle sdel2<0.
*	the simplest solution seems to be to switch to sdel2>0 for
*	the time being - we calculate a complete 3point function so it
*	should not be a problem (just a sign).  Of course this spoils a
*	good check on the correctness.
*
	sdel2 = abs(sdel2i)
	if ( sdel2i .gt. 0 .and. lwrite ) print *,
     +		'ffxc0j: cannot handle sdel2>0, switched to sdel2<0'
*
	if ( xpi(4).eq.0 ) then
	    zm = xpi(2)/dpipj(2,1)
	    zm1 = -xpi(1)/dpipj(2,1)
	else
	    call ffroot(zm,zp,xpi(4),piDpj(4,2),xpi(2),sdel2,ier)
	    if ( dpipj(1,2) .ne. 0 ) then
		call ffroot(zp1,zm1,xpi(4),-piDpj(4,1),xpi(1),sdel2,ier)
	    else
		zm1 = zp
		zp1 = zm
	    endif
	endif
	if ( lwrite ) then
	    print *,'ffxc0j: found roots:'
	    print *,'  zm = ',zm,zm1
	    if ( xpi(4).ne.0 ) print *,'  zp = ',zp,zp1
	endif
	if ( ltest ) then
	    xheck = zm + zm1 - 1
	    if ( xloss*abs(xheck) .gt. precx*max(x1,abs(zm)) ) print *,
     +		'ffxc0j: zm + zm1 <> 1: ',zm,zm1,xheck
	    if ( xpi(4).ne.0 ) then
		xheck = zp + zp1 - 1
		if ( xloss*abs(xheck) .gt. precx*max(x1,abs(zp)) )
     +		    print *,'ffxc0j: zp + zp1 <> 1: ',zp,zp1,xheck
	    endif
	endif

*	imag sign ok 30-oct-1989
	ieps = -1
	if ( xpi(4).ne.0 ) dyzp = -2*sdel2/xpi(4)
*
*  #] get determinants, roots, ieps:
*  #[ the finite+divergent S1:
*
	if ( xpi(4).ne.0 ) then
	    call ffcxr(cs(1),ipi12,zm,zm1,zp,zp1,dyzp,
     +		.FALSE.,x0,x0,x0,.FALSE.,dum,ieps,ier)
	endif
*
*	Next the divergent piece
*
	if ( .not.lsmug ) then
*
*	    Here we dropped the term log(lam/delta)*log(-zm/zm1)
*
	    if ( abs(zm1) .gt. 1/xloss ) then
		clog1 = dfflo1(1/zm1,ier)
	    elseif ( zm.ne.0 ) then
		clog1 = zxfflg(-zm/zm1,-2,x0,ier)
	    else
	        call fferr(97,ier)
	        return
	    endif
	    hulp = zm*zm1*4*del2/delta**2
*
*	    14-jan-1994: do not count when this is small, this was 
*	    meant to be so by the user carefully adjusting delta
*
	    ier0 = ier
	    if ( hulp.eq.0 ) call fferr(97,ier)
	    clog2 = zxfflg(hulp,2,x0,ier0)
	    cs(8) = -clog1*clog2/2
	    if ( lwrite ) then
*		print *,'arg1 = ',-zm/zm1,1/zm1
		print *,'log1 = ',clog1
*		print *,'arg2 = ',hulp
		print *,'log2 = ',clog2
		print *,'cs(8)= ',cs(8)
	    endif
	else
*
*	    checked 4-aug-1992, but found Yet Another Bug 30-sep-1992
*
	    cdyzm = cel3*DBLE(1/(-2*sdel2*del2))
	     dyzm = del3/(-2*sdel2*del2)
	    carg1 = +cdyzm*DBLE(1/zm)
	     arg1 = +dyzm/zm
	    clog1 = zfflog(-carg1,+ieps,DCMPLX(DBLE(zm),DBLE(0)),ier)
	    if (DIMAG(cdyzm) .lt. 0 .and. arg1 .gt. 0 ) then
		clog1 = clog1 - c2ipi
		if ( lwrite ) then
		    print *,'added -2*pi*i to log1 S1'
		    print *,' arg1,zm = ',arg1,zm
		    print *,'carg1    = ',carg1
		endif
*		ier = ier + 50
	    endif
	    cs(8) = -clog1**2/2
	    carg2 = -cdyzm*DBLE(1/zm1)
	     arg2 = -dyzm/zm1
	    clog2 = zfflog(-carg2,ieps,DCMPLX(DBLE(-zm1),DBLE(0)),ier)
	    if ( DIMAG(cdyzm) .lt. 0 .and. arg2 .gt. 0 ) then
		clog2 = clog2 + c2ipi
		if ( lwrite ) then
		    print *,'added +2*pi*i to log2 S1'
		    print *,' arg2,zm = ',arg2,zm
		    print *,'carg2    = ',carg2
		endif
	    endif
	    cs(9) = +clog2**2/2
	    if ( lwrite ) then
		print *,'y=zm = ',zm,zm1
		if ( xpi(4).ne.0 ) print *,'  zp = ',zp,zp1
		print *,'cdyzm= ',cdyzm
		print *,'arg1 = ',1/carg1
		print *,'log1 = ',clog1
		print *,'cs(8)= ',cs(8)
		print *,'arg2 = ',1/carg2
		print *,'log2 = ',clog2
		print *,'cs(9)= ',cs(9)
		print *,'ipi12= ',ipi12
		print *,'S1   = ',cs(1)+cs(2)+cs(3)+cs(4)+cs(5)+cs(6)+
     +			cs(7)+cs(8)+cs(9)+ipi12*DBLE(pi12)
		print *,' '
	    endif
	endif
*  #] the finite+divergent S1:
*  #[ log(1) for npoin=4:
	if ( npoin .eq. 4 ) then
	    if ( ilogi(1) .eq. -999 ) then
		if ( .not.lsmug ) then
		    hulp = xpi(4)*delta/(4*del2)
		    ier0 = ier
		    if ( hulp.eq.0 ) call fferr(97,ier)
		    clogi(1) = -zxfflg(abs(hulp),0,x0,ier0)
		    if ( hulp .lt. 0 ) then
			if ( xpi(4) .gt. 0 ) then
			    ilogi(1) = -1
			else
			    ilogi(1) = +1
			endif
			if ( ltest ) then
			    print *,'ffxc0j: I am not 100% sure of the',
     +				' terms pi^2, please check against the',
     +				' limit lam->0 (id=',id,')'
			    ier = ier + 50
			endif
		    else
			ilogi(1) = 0
		    endif
		else
		    if ( xpi(4).eq.0 ) then
			print *,'ffxc0i: cannot handle t=0 yet, sorry'
			print *,'Please regularize with a small mass'
			stop
		    endif
		    chulp = -cdyzm*DBLE(1/dyzp)
		    chulp1 = 1+chulp
		    if ( absc(chulp1) .lt. xloss )
     +			call ffwarn(129,ier,absc(chulp1),x1)
		    call ffxclg(clogi(1),ilogi(1),chulp,chulp1,dyzp,
     +			ier)
		endif
	    endif
	endif
*  #] log(1) for npoin=4:
*  #[ the log(lam) Si:
	if ( .not.lsmug ) then
*
*	    Next the divergent S_i (easy).
*	    The term -2*log(lam/delta)*log(xpi(2)/xpi(1)) has been discarded
*	    with lam the photon mass (regulator).
*	    If delta = sqrt(xpi(1)*xpi(2)) the terms cancel as well
*
	    if ( dpipj(1,2).ne.0 .and. xloss*abs(xpi(1)*xpi(2)-delta**2)
     +	    		.gt.precx*delta**2 ) then
		if ( xpi(1) .ne. delta ) then
		    ier0 = ier
		    if ( xpi(1).eq.0 ) call fferr(97,ier)
		    cs(9) = -zxfflg(xpi(1)/delta,0,x0,ier0)**2 /4
		endif
		if ( xpi(2) .ne. delta ) then
		    ier0 = ier
		    if ( xpi(2).eq.0 ) call fferr(97,ier)
		    cs(10) = zxfflg(xpi(2)/delta,0,x0,ier0)**2 /4
		endif
	    endif
	    if ( lwrite ) then
		print *,'cs(9)= ',cs(9)
		print *,'cs(10)=',cs(10)
	    endif
*  #] the log(lam) Si:
*  #[ the logs for A_i<0:
	    if ( npoin.eq.4 ) then
		clogi(2) = 0
		ilogi(2) = 0
		clogi(3) = 0
		ilogi(3) = 0
	    endif
*  #] the logs for A_i<0:
*  #[ the off-shell S3:
	else
*
*	    the divergent terms in the offshell regulator scheme - not
*	    quite as easy
*	    wm = p3.p2/sqrtdel - 1 = -s1.s2/sqrtdel - 1
*	    wp = p3.p2/sqrtdel + 1 = -s1.s2/sqrtdel + 1
*	    Note that we took the choice sdel2<0 in S1 when
*	    \delta^{p1 s2}_{p1 p2} < 0 by using yp=zm
*
	    wm = -1 - piDpj(1,2)/sdel2
	    wp = wm + 2
	    if ( lwrite ) print *,'wm,wp = ',wm,wp
	    if ( abs(wm) .lt. abs(wp) ) then
		wm = -xpi(5)*xpi(6)/(del2*wp)
		if ( lwrite ) print *,'wm+   = ',wm
	    else
		wp = -xpi(5)*xpi(6)/(del2*wm)
		if ( lwrite ) print *,'wp+   = ',wp
	    endif
*
*	    the im sign
*
	    if ( -DBLE(cmipj(1,3)) .gt. 0 ) then
		ieps = -1
	    else
		ieps = +1
	    endif
*
	    if ( nschem .lt. 3 .or. DIMAG(cmipj(1,3)).eq.0 .and.
     +			DIMAG(cmipj(2,2)).eq.0 ) then
*  #[		real case:
		if ( lwrite ) print *,'ffxc0i: Real S3'
*
*		first z-,z+
*
		dyzp = -DBLE(cmipj(1,3))*DBLE(wm)/(2*DBLE(xpi(6))) -
     +			DBLE(cmipj(2,2))/(2*DBLE(sdel2))
		dyzm = -DBLE(cmipj(1,3))*DBLE(wp)/(2*DBLE(xpi(6))) -
     +			DBLE(cmipj(2,2))/(2*DBLE(sdel2))
*
*		the (di)logs
*
		clog1 = zxfflg(-dyzp,-ieps,x1,ier)
		cs(10) = -clog1**2/2
		ipi12 = ipi12 - 4
		clog2 = zxfflg(-dyzm,+ieps,x1,ier)
		cs(11) = -clog2**2/2
		ipi12 = ipi12 - 2
		hulp = dyzp/dyzm
		if ( dyzp .lt. 0 ) then
		    ieps1 = -ieps
		else
		    ieps1 = +ieps
		endif
		call ffzxdl(cli,i,cdum(1),hulp,+ieps1,ier)
		cs(12) = -cli
		ipi12 = ipi12 - i
*
*		the log for npoin=4
*
		if ( npoin.eq.4 ) then
		    if ( ilogi(3) .eq. -999 ) then
			if ( DBLE(cmipj(1,3)) .eq. 0 ) then
			    chulp = -1
			    chulp1 = 0
			elseif ( dyzp .lt. dyzm ) then
			    chulp = -dyzm/dyzp
			    chulp1 = +DBLE(cmipj(1,3))/DBLE(xpi(6)*dyzp)
			else
			    chulp = -dyzp/dyzm
			    chulp1 = -DBLE(cmipj(1,3))/DBLE(xpi(6)*dyzm)
			endif
			call ffxclg(clogi(3),ilogi(3),chulp,chulp1,dyzp,
     +				ier)
		    endif
		endif
*
*		and some debug output:
*
		if ( lwrite ) then
		    print *,'z      = 1,0'
		    print *,'y-zm   = ',dyzm
		    print *,'y-zp   = ',dyzp
		    print *,'+Li2(y/(y-zp))    = ',cs(10)
		    print *,'+Li2(y/(y-zm))    = ',cs(11)
		    print *,'-Li2((y-1)/(y-zm))= ',cs(12)
		    print *,'ipi12   = ',ipi12
		endif
*  #]		real case:
	    else
*  #[		complex case:
		if ( lwrite ) print *,'ffxc0i: Complex S3'
*
*		first z+
*
		cdyzp = -cmipj(1,3)*DBLE(wm)/(2*DBLE(xpi(6))) -
     +			cmipj(2,2)/(2*DBLE(sdel2))
		clog1 = zfflog(-cdyzp,-ieps,c1,ier)
		if ( ieps*DIMAG(cdyzp).lt.0.and.DBLE(cdyzp).gt.0 ) then
		    if ( lwrite ) then
			print *,'added ',-ieps,'*2*pi*i to log1 S3'
			print *,'carg1    = ',-cdyzp
			print *,'clog1 was  ',clog1
			print *,'clog1 is   ',clog1 - ieps*c2ipi
		    endif
		    clog1 = clog1 - ieps*c2ipi
		else
		    if ( lwrite ) then
			print *,'carg1    = ',-cdyzp
			print *,'clog1 is   ',clog2
		    endif
		endif
		cs(10) = -clog1**2/2
		ipi12 = ipi12 - 4
*
*		now z-
*
		cdyzm = -cmipj(1,3)*DBLE(wp)/(2*DBLE(xpi(6))) -
     +			cmipj(2,2)/(2*DBLE(sdel2))
		clog2 = zfflog(-cdyzm,+ieps,c1,ier)
		if ( ieps*DIMAG(cdyzm).gt.0.and.DBLE(cdyzm).gt.0 ) then
		    if ( lwrite ) then
			print *,'added ',ieps,'*2*pi*i to log2 S3'
			print *,'carg2    = ',-cdyzm
			print *,'clog2 was  ',clog2
			print *,'clog2 is   ',clog2 + ieps*c2ipi
		    endif
		    clog2 = clog2 + ieps*c2ipi
*		    ier = ier + 50
		else
		    if ( lwrite ) then
			print *,'carg2    = ',-cdyzm
			print *,'clog2 is   ',clog2
		    endif
		endif
		cs(11) = -clog2**2/2
		ipi12 = ipi12 - 2
*
*		the dilog
*
		chulp = cdyzp/cdyzm
		hulp = DBLE(cdyzp)/DBLE(cdyzm)
		if ( DBLE(cdyzp) .lt. 0 ) then
		    ieps1 = -ieps
		else
		    ieps1 = +ieps
		endif
		if ( DIMAG(chulp) .eq. 0 ) then
		    hulp = DBLE(chulp)
		    call ffzxdl(cli,i,cdum(1),hulp,+ieps1,ier)
		else
		    call ffzzdl(cli,i,cdum(1),chulp,ier)
		    if ( hulp.gt.1 .and. ieps1*DIMAG(chulp).lt.0 ) then
			if ( lwrite ) then
			    print *,'addded 2ipi*log(z) to Li'
			    print *,'chulp = ',chulp
			    print *,'cli was ',cli
			    print *,'cli is  ',cli +
     +				ieps1*c2ipi*zfflog(chulp,0,c0,ier)
			    call ffzxdl(cdum(2),i,cdum(1),hulp,+ieps1,
     +				ier)
			    print *,'vgl   ',cdum(2)
			endif
			cli = cli + ieps1*c2ipi*zfflog(chulp,0,c0,ier)
		    endif
		endif
		cs(12) = -cli
		ipi12 = ipi12 - i
*
*		the log for npoin=4
*
		if ( npoin.eq.4 ) then
		    if ( ilogi(3) .eq. -999 ) then
			if ( cmipj(1,3) .eq. 0 ) then
			    chulp = -1
			    chulp1 = 0
			elseif ( DBLE(cdyzp) .lt. DBLE(cdyzm) ) then
			    chulp = -cdyzm/cdyzp
			    chulp1 = +cmipj(1,3)/cdyzp*DBLE(1/xpi(6))
			else
			    chulp = -cdyzp/cdyzm
			    chulp1 = -cmipj(1,3)/cdyzm*DBLE(1/xpi(6))
			endif
			dyzp = DBLE(cdyzp)
			call ffxclg(clogi(3),ilogi(3),chulp,chulp1,dyzp,
     +				ier)
		    endif
		endif
*
*		and some debug output:
*
		if ( lwrite ) then
		    print *,'z      = 1,0'
		    print *,'y-zm   = ',cdyzm
		    print *,'y-zp   = ',cdyzp
		    print *,'+Li2(y/(y-zp))    = ',cs(10)
		    print *,'+Li2(y/(y-zm))    = ',cs(11)
		    print *,'-Li2((y-1)/(y-zm))= ',cs(12)
		    print *,'ipi12   = ',ipi12
		endif
*  #]		complex case:
	    endif
*  #] the off-shell S3:
*  #[ the off-shell S2:
*
*	    the im sign
*
	    if ( -DBLE(cmipj(2,2)) .gt. 0 ) then
		ieps = -1
	    else
		ieps = +1
	    endif
*
	    if ( nschem .lt. 3 ) then
*  #[		real case:
		if ( lwrite ) print *,'ffxc0i: Real S2'
*
*		first z-
*
		dyzm = -DBLE(cmipj(2,2))*DBLE(wp)/(2*DBLE(xpi(5))) -
     +			DBLE(cmipj(1,3))/(2*DBLE(sdel2))
		clog1 = zxfflg(+dyzm,-ieps,x1,ier)
		cs(13) = +clog1**2/2
		ipi12 = ipi12 + 4
*
*		now z+
*
		dyzp = -DBLE(cmipj(2,2))*DBLE(wm)/(2*DBLE(xpi(5))) -
     +			DBLE(cmipj(1,3))/(2*DBLE(sdel2))
		clog2 = zxfflg(+dyzp,+ieps,x1,ier)
		cs(14) = +clog2**2/2
		ipi12 = ipi12 + 2
		hulp = dyzm/dyzp
		if ( dyzm .lt. 0 ) then
		    ieps1 = -ieps
		else
		    ieps1 = +ieps
		endif
		call ffzxdl(cli,i,cdum(1),hulp,-ieps1,ier)
		cs(15) = +cli
		ipi12 = ipi12 + i
*
*		the log for npoin=4
*
		if ( npoin.eq.4 ) then
		    if ( ilogi(2) .eq. -999 ) then
			if ( DBLE(cmipj(2,2)) .eq. 0 ) then
			    chulp = -1
			    chulp1 = 0
			elseif ( dyzp .lt. dyzm ) then
			    chulp = -dyzm/dyzp
			    chulp1 = +DBLE(cmipj(2,2))/DBLE(xpi(5)*dyzp)
			elseif ( dyzp .gt. dyzm ) then
			    chulp = -dyzp/dyzm
			    chulp1 = -DBLE(cmipj(2,2))/DBLE(xpi(5)*dyzm)
			endif
			call ffxclg(clogi(2),ilogi(2),chulp,chulp1,dyzp,
     +				ier)
		    endif
		endif
*
*		and some debug output:
*
		if ( lwrite ) then
		    print *,'z      = 0,1'
		    print *,'y-zm   = ',dyzm
		    print *,'y-zp   = ',dyzp
		    print *,'-Li2((y-1)/(y-zm))= ',cs(13)
		    print *,'-Li2((y-1)/(y-zp))= ',cs(14)
		    print *,'+Li2(y/(y-zp))    = ',cs(15)
		    print *,'ipi12   = ',ipi12
		endif
*  #]		real case:
	    else
*  #[		complex case:
		if ( lwrite ) print *,'ffxc0i: Complex S2'
*
*		first z-
*
		cdyzm = -cmipj(2,2)*DBLE(wp)/(2*DBLE(xpi(5))) -
     +			cmipj(1,3)/(2*DBLE(sdel2))
		clog1 = zfflog(+cdyzm,-ieps,c1,ier)
		if ( DBLE(cdyzm).lt.0.and.ieps*DIMAG(cdyzm).gt.0 ) then
		    if ( lwrite ) print *,'added 2*i*pi to log1'
		    clog1 = clog1 - ieps*c2ipi
		endif
		cs(13) = +clog1**2/2
		ipi12 = ipi12 + 4
*
*		now z+
*
		cdyzp = -cmipj(2,2)*DBLE(wm)/(2*DBLE(xpi(5))) -
     +			cmipj(1,3)/(2*DBLE(sdel2))
		clog2 = zfflog(+cdyzp,+ieps,c1,ier)
		if ( DBLE(cdyzp).lt.0.and.ieps*DIMAG(cdyzp).lt.0 ) then
		    if ( lwrite ) then
			print *,'added ',ieps,'*2*pi*i to log2 S2'
			print *,'carg1    = ',+cdyzp
		    endif
		    clog2 = clog2 + ieps*c2ipi
		endif
		cs(14) = +clog2**2/2
		ipi12 = ipi12 + 2
*		
*		and ghe dilog
*		
		chulp = cdyzm/cdyzp
		hulp = DBLE(dyzm)/DBLE(dyzp)
		if ( DBLE(cdyzm) .lt. 0 ) then
		    ieps1 = -ieps
		else
		    ieps1 = +ieps
		endif
		if ( DIMAG(chulp ) .eq. 0 ) then
		    hulp = DBLE(chulp)
		    call ffzxdl(cli,i,cdum(1),hulp,-ieps1,ier)
		else
		    call ffzzdl(cli,i,cdum(1),chulp,ier)
		    if ( hulp.gt.1 .and. ieps1*DIMAG(chulp).gt.0 ) then
			if ( lwrite ) then
			    print *,'addded 2ipi*log(z) to Li'
			    print *,'chulp = ',chulp
			    print *,'cli was ',cli
			    print *,'cli is  ',cli -
     +				ieps1*c2ipi*zfflog(chulp,0,c0,ier)
			    call ffzxdl(cdum(2),i,cdum(1),hulp,-ieps1,
     +				ier)
			    print *,'vgl   ',cdum(2)
			endif
			cli = cli - ieps1*c2ipi*zfflog(chulp,0,c0,ier)
		    endif
		endif
		cs(15) = +cli
		ipi12 = ipi12 + i
*
*		the log for npoin=4
*
		if ( npoin.eq.4 ) then
		    if ( ilogi(2) .eq. -999 ) then
			if ( cmipj(2,2) .eq. 0 ) then
			    chulp = -1
			    chulp1 = 0
			elseif ( DBLE(cdyzp) .lt. DBLE(cdyzm) ) then
			    chulp = -cdyzm/cdyzp
			    chulp1 = +cmipj(2,2)/cdyzp*DBLE(1/xpi(5))
			elseif ( DBLE(cdyzp) .gt. DBLE(cdyzm) ) then
			    chulp = -cdyzp/cdyzm
			    chulp1 = -cmipj(2,2)/cdyzm*DBLE(1/xpi(5))
			endif
			dyzp = DBLE(cdyzp)
			call ffxclg(clogi(2),ilogi(2),chulp,chulp1,dyzp,
     +				ier)
		    endif
		endif
*
*		and some debug output:
*
		if ( lwrite ) then
		    print *,'z      = 0,1'
		    print *,'y-zm   = ',cdyzm
		    print *,'y-zp   = ',cdyzp
		    print *,'-Li2((y-1)/(y-zm))= ',cs(13)
		    print *,'-Li2((y-1)/(y-zp))= ',cs(14)
		    print *,'+Li2(y/(y-zp))    = ',cs(15)
		    print *,'ipi12   = ',ipi12
		endif
*  #]		complex case:
	    endif
	endif
*  #] the off-shell S2:
*  #[ sdel2<0!:
	if ( sdel2i.gt.0 .neqv. xpi(4).eq.0.and.xpi(1).gt.xpi(2) ) then
	    if ( .not.lsmug ) then
		n = 10
	    else
		n = 15
	    endif
	    do 10 i=1,n
		cs(i) = -cs(i)
   10	    continue
	    ipi12 = -ipi12
	    if ( npoin.eq.4 ) then
	    	do 20 i=1,3
		    ilogi(i) = -ilogi(i)
		    clogi(i) = -clogi(i)
   20	    	continue
	    endif
	endif
	if ( lwrite ) print '(a)','  ##] ffxc0j:'
*  #] sdel2<0!:
*###] ffxc0j:
	end
*###[ ffxclg:
	subroutine ffxclg(clg,ilg,chulp,chulp1,dyzp,ier)
***#[*comment:***********************************************************
*									*
*	compute the extra logs for npoin=4 given chulp=-cdyzm/cdyzp	*
*	all flagchecking has already been done.				*
*									*
*	Input:	chulp	(complex)	see above			*
*		chulp1	(complex)	1+chulp (in case chulp ~ -1)	*
*		dyzp	(real)		(real part of) y-z+ for im part	*
*	Output:	clg	(complex)	the log				*
*		ilg	(integer)	factor i*pi split off clg	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ilg,ier
	DOUBLE PRECISION dyzp
	DOUBLE COMPLEX clg,chulp,chulp1
*
*	local variables
*
	DOUBLE PRECISION hulp,hulp1,dfflo1
	DOUBLE COMPLEX zxfflg,zfflog,zfflo1,check
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ check:
	if ( ltest ) then
	    check = c1 + chulp - chulp1
	    if ( xloss*abs(check) .gt. precc*max(abs(c1),abs(chulp)) )
     +		print *,'ffxclg: error: chulp1 != 1+chulp: ',chulp1,
     +		c1+chulp,check
	endif
*  #] check:
*  #[ work:
*
	if ( DIMAG(chulp) .eq. 0 ) then
	    hulp = DBLE(chulp)
	    hulp1 = DBLE(chulp1)
	    if ( abs(hulp1) .lt. xloss ) then
		clg = DBLE(dfflo1(hulp1,ier))
	    else
		clg = zxfflg(abs(hulp),0,x0,ier)
	    endif
	    if ( hulp .lt. 0 ) then
		if ( dyzp.lt.0 ) then
		    ilg = +1
		else
		    ilg = -1
		endif
	    else
		ilg = 0
	    endif
	    if ( lwrite ) print *,'clg(real) = ',clg+c2ipi*ilg/2
	else
*
*	    may have to be improved
*
	    if ( abs(DBLE(chulp1))+abs(DIMAG(chulp1)) .lt. xloss ) then
		clg = zfflo1(chulp1,ier)
	    else
		clg = zfflog(chulp,0,c0,ier)
	    endif
	    ilg = 0
	    if ( DBLE(chulp) .lt. 0 ) then
		if ( dyzp.lt.0 .and. DIMAG(clg).lt.0 ) then
		    if ( lwrite ) print *,'ffxclg: added -2*pi to log'
		    ilg = +2
		elseif ( dyzp.gt.0 .and. DIMAG(clg).gt.0 ) then
		    if ( lwrite ) print *,'ffxclg: added +2*pi to log'
		    ilg = -2
		endif
	    endif
	    if ( lwrite ) print *,'clg(cmplx)= ',clg+c2ipi*ilg/2
	endif
*  #] work:
*###] ffxclg:
	end
