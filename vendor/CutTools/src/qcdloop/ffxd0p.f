*(##[ ffxd0p:
	subroutine ffxd0p(cs4,ipi12,isoort,cfac,xpi,dpipj,piDpj,
     +		xqi,dqiqj,qiDqj,ai,daiaj,ldel2s,ier)
***#[*comment:***********************************************************
*									*
*	calculate D0/pi^2/(A1*A2*A3*A4/dt3t4)				*
*									*
*		     = C0(t1,t2,t3) - C0(t1,t2,t4)			*
*									*
*	The transformed momenta of the fourpoint functions are		*
*	input.								*
*									*
*	Input:	xpi(10)		untransformed fourpoint momenta		*
*		dpipj(10,10)	differences of xpi			*
*		piDpj(10,10)	dotproducts of xpi			*
*		xqi(10)		transformed fourpoint momenta		*
*		dqiqj(10,10)	differences of xqi			*
*		qiDqj(10,10)	dotproducts of xqi			*
*		ai(4)		the transformation parameters		*
*		daiaj(4,4)	their deifferences			*
*		ldel2s		if .TRUE. we took out factors delta	*
*									*
*	Output:	cs4(170)	not added (assumed 0 on input)		*
*		cfac		the factor of cs4 from C0 (ie lam(pi))	*
*		ier		0=ok 1=inaccurate 2=error		*
*									*
*	Calls:	ffxc0p,ffpi34,ffxhck,ffdl3m,ffdel2,ffdel3,...		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cs4(175),cfac
	integer ipi12(26),isoort(16),ier
	logical ldel2s
	DOUBLE PRECISION xpi(10),dpipj(10,10),piDpj(10,10),
     +		xqi(10),dqiqj(10,10),qiDqj(10,10),ai(4),daiaj(4,4)
*
*	local variables
*
	integer i,j,k,ip,jp,m,ilogi(6),ii(6,2),jj(6,2),ier0,ier1,
     +		is1,is2
	DOUBLE COMPLEX c,clogi(6),cipi
	DOUBLE PRECISION xpi3(6,3:4),dpipj3(6,6,3:4),piDpj3(6,6,3:4),
     +		absc,del2,del2s(3,3:4),del3(3:4),del3mi(6,3:4),
     +		del4,etalam(3:4),etami(6,3:4),ddel2s(2:3),delpsi(3,3:4),
     +		alph(3),blph(3),sdel2,hulp,som,s(4),smax,xmax
	DOUBLE COMPLEX cpi(6,3:4),cpiDpj(6,6,3:4),cdpipj(6,6,3:4),
     +		cetalm(3:4),cetami(6,3:4),calph(3),csdel2,
     +		cel2s(3,3:4),celpsi(3,3:4),zqi(10),zqiDqj(10,10),
     +		zdqiqj(10,10),cddl2s(2:3),cqi3(6,3:4),cqiqj3(6,6,3:4),
     +		cqiDqj3(6,6,3:4)
	logical lcroot,lb
	save ii,jj
*
*	common blocks:
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	statement function:
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data ii/1,2,3,5,6,9,1,2,3,5,6,9/
	data jj/1,2,4,5,10,8,1,2,4,5,10,8/
*
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
*	    call ffxhck(xpi,dpipj,10,ier)
*	    call ffxhck(xqi,dqiqj,10,ier)
*	    if ( ier .ne. 0 ) print *,'(input tested by ffxd0p)'
	endif
*  #] check input:
*  #[ preparation:
*	Note that the piDpj3(,,3) contain now the threepoint function
*	with s3, (,,4) with s4 (and NOT *without* as before)
	call ffpi43(xpi3(1,3),dpipj3(1,1,3),piDpj3(1,1,3),
     +		xqi,dqiqj,qiDqj,7-3,ier)
	call ffpi43(xpi3(1,4),dpipj3(1,1,4),piDpj3(1,1,4),
     +		xqi,dqiqj,qiDqj,7-4,ier)
*
*	set the logarithms to be calculated to -999
*
	do 40 i=1,6
	    clogi(i) = 0
	    ilogi(i) = 0
   40	continue
	if ( ai(1) .lt. 0 .neqv. ai(2) .lt. 0 ) then
	    ilogi(1) = -999
	    ilogi(4) = -999
	endif
	if ( ai(2) .lt. 0 .neqv. ai(3) .lt. 0 ) then
	    ilogi(2) = -999
	endif
	if ( ai(3) .lt. 0 .neqv. ai(1) .lt. 0 ) then
	    ilogi(3) = -999
	endif
	if ( ai(2) .lt. 0 .neqv. ai(4) .lt. 0 ) then
	    ilogi(5) = -999
	endif
	if ( ai(4) .lt. 0 .neqv. ai(1) .lt. 0 ) then
	    ilogi(6) = -999
	endif
*
*  #] preparation:
*  #[ determinants:
*
*	some determinants
*
	if ( lwrite ) print '(a)','  ##[ determinants:'
*
*	note that not all errors are additive, only when a previous
*	result is used as input do we need to add ther ier's, otherwise
*	we can take the maximum value to get a decent estimate of the
*	number of digits lost.
*
	ier1 = ier
	if ( .not.ldel2s ) then
	    ier0 = ier
	    call ffdel2(del2,qiDqj,10, 5,6,9, 0,ier0)
	    ier1 = max(ier1,ier0)
	else
	    s(1) = xqi(5)*xqi(3)
	    s(2) = qiDqj(5,3)**2
	    del2 = s(1) - s(2)
	    if ( abs(del2) .lt. xloss*s(2) ) ier1 = 100
	endif
	if ( ier1 .ne. ier ) then
	    ier0 = ier
	    call ffdel4(del4,xpi,piDpj,10,ier0)
	    if ( ldel2s ) then
		hulp = -(ai(1)*ai(2)*ai(3)*ai(4)/xqi(3))**2 * del4
	    else
		hulp = -(2*ai(1)*ai(2)*ai(3)*ai(4)/dqiqj(3,4))**2 * del4
	    endif
	    if ( lwrite ) then
		print *,'del2 was :',del2
		print *,'  and is :',hulp
	    endif
	    del2 = hulp
	    ier1 = ier0
	    fdel4s = del4
	else
	    if ( ldel2s ) then
		fdel4s = -del2*(xqi(3)/ai(1)*ai(2)*ai(3)*ai(4))**2
	    else
		fdel4s=-del2*(dqiqj(3,4)/(2*ai(1)*ai(2)*ai(3)*ai(4)))**2
	    endif
	endif
	if ( del2 .gt. 0 ) then
*	    use complex routines
*	    call fferr(44,ier)
	    lcroot = .TRUE.
	    sdel2 = isgnal*sqrt(del2)
	    csdel2 = DCMPLX(x0,sdel2)
	elseif ( del2 .eq. 0 ) then
	    call fferr(45,ier)
	    if ( ltest ) then
		print *,'ffxd0p: error: del2 = 0'
		print *,'xqi = ',xqi,ier
		return
	    endif
	else
	    lcroot = .FALSE.
	    sdel2 = isgnal*sqrt(-del2)
	endif
	ier0 = ier
	call ffdl3s(del3(3),xpi,piDpj,ii,10,ier0)
	ier1 = max(ier0,ier1)
	if ( lwrite ) print *,'del3s(untransformed) 3 = ',del3(3)
	ier0 = ier
	call ffdl3s(del3(4),xpi,piDpj,jj,10,ier0)
	ier1 = max(ier1,ier0)
	if ( lwrite ) print *,'del3s(untransformed) 4 = ',del3(4)
	del3(3) = ai(1)**2*ai(2)**2*ai(3)**2*del3(3)
	del3(4) = ai(1)**2*ai(2)**2*ai(4)**2*del3(4)
	do 108 m=3,4
	    ier0 = ier
	    if ( .not.ldel2s ) then
		call ffdl3m(del3mi(1,m),.TRUE.,del3(m),del2,xpi3(1,m)
     +			,dpipj3(1,1,m),piDpj3(1,1,m), 6, 4,5,6,1,3,ier0)
	    else
*
*		the special case del2s = 0.  Note that del3mi(i) and
*		del3mi(i+3) are used in S_{i-1}.
*
		call ffdl3m(del3mi(1,m),.FALSE.,x0,x0,xpi3(1,m),
     +		   dpipj3(1,1,m),piDpj3(1,1,m), 6, 4,3,0, 1,2,ier0)
		ier1= max(ier1,ier0)
		ier0 = ier
		call ffdl3m(del3mi(5,m),.FALSE.,x0,x0,xpi3(1,m),
     +		   dpipj3(1,1,m),piDpj3(1,1,m), 6, 4,3,0, 5,2,ier0)
		del3mi(3,m) = 0
		del3mi(4,m) = 0
	    endif
	    ier1 = max(ier1,ier0)
	    do 105 i=1,3
		j = i+1
		if ( j .eq. 4 ) j = 1
		ip = i
		jp = j
		if ( m .eq. 4 ) then
		    if ( jp .eq. 3 ) jp = 4
		    if ( ip .eq. 3 ) ip = 4
		endif
		if ( i.eq.1 .and. m.eq.4 ) then
		    del2s(1,4) = del2s(1,3)
		else
		    ier0 = ier
		    call ffdel2(del2s(i,m),piDpj,10,inx(ip,jp),ip,
     +				jp,1,ier0)
		    del2s(i,m) = ai(ip)**2*ai(jp)**2*del2s(i,m)
		    ier1 = max(ier1,ier0)
		endif
		k = i-1
		if ( k .eq. 0 ) k = 3
		ier0 = ier
		if ( .not.ldel2s ) then
		    call ffdl2p(delpsi(i,m),xpi3(1,m),dpipj3(1,1,m),
     +			piDpj3(1,1,m),i+3,j+3,k+3,i,j,k,6,ier0)
		else
		    call ffdl2t(delpsi(i,m),qiDqj, m,5, ip,jp,inx(ip,jp)
     +			,+1,+1, 10,ier0)
		endif
		ier1 = max(ier1,ier0)
		etami(i,m) = del3mi(i,m)/del2
		if ( ldel2s .and. i.gt.1 )
     +			etami(i+3,m) = del3mi(i+3,m)/del2
  105	    continue
	    etalam(m) = del3(m)/del2
  108	continue
*
*	the error analysis
*
	ier = ier1
*
*	get alpha,1-alpha
*
	if ( .not. lcroot ) then
	    if ( .not.ldel2s ) then
		if ( xpi3(5,3).eq.0 .and. (piDpj3(5,6,3).gt.0 .eqv.
     +			sdel2.gt.0) ) then
		    alph(1) = -xpi3(6,3)/(piDpj3(5,6,3)+sdel2)
		    alph(3) = -xpi3(4,3)/(piDpj3(5,4,3)-sdel2)
		    lb = .FALSE.
		else
		    lb = .TRUE.
		    call ffroot(blph(1),alph(1),xpi3(5,3),
     +			-piDpj3(5,6,3),xpi3(6,3),sdel2,ier)
		    call ffroot(alph(3),blph(3),xpi3(5,3),
     +			-piDpj3(5,4,3),xpi3(4,3),sdel2,ier)
		endif
*		We cannot change the sign as it is fixed by the choice
*		of sign in fftrans (sqrt(delta(s3,s4))) WRONG
*		if ( l4also .and. ( alph(1) .gt. 1 .or. alph(1) .lt. 0
*     +		   ) .and. abs(blph(1)-x05) .lt. abs(alph(1)-x05) ) then
*		    alph(1) = blph(1)
*		    alph(3) = blph(3)
*		    sdel2 = -sdel2
*		    isgnal = -isgnal
*		endif
	    else
		alph(1) = 1
		alph(3) = 0
	    endif
	    cfac = 2*sdel2
	    if (lwrite) then
		print *,'slam   = ',2*sdel2
		print *,'del2s3 = ',(del2s(i,3),i=1,3)
		print *,'del2s4 = ',(del2s(i,4),i=1,3)
		print *,'del2ps3= ',(delpsi(i,3),i=1,3)
		print *,'del2ps4= ',(delpsi(i,4),i=1,3)
		print *,'del3mi3= ',(del3mi(i,3),i=1,3)
		print *,'del3mi4= ',(del3mi(i,4),i=1,3)
		print *,'etami3 = ',(etami(i,3),i=1,3)
		print *,'etami4 = ',(etami(i,4),i=1,3)
		print *,'eta3   = ',-4*del3(3)
		print *,'eta4   = ',-4*del3(4)
		print *,'alpha  = ',alph(1),alph(3)
		print *,'ier    = ',ier
	    endif
	else
	    do 4 k=3,4
		do 3 i=1,6
		    cpi(i,k) = xpi3(i,k)
		    do 2 j=1,6
			cdpipj(j,i,k) = dpipj3(j,i,k)
			cpiDpj(j,i,k) = piDpj3(j,i,k)
    2		    continue
    3		continue
    4	    continue
	    if ( .not.ldel2s ) then
		call ffcoot(c,calph(1),cpi(5,3),-cpiDpj(5,6,3),
     +			cpi(6,3),csdel2,ier)
		call ffcoot(calph(3),c,cpi(5,3),-cpiDpj(5,4,3),
     +			cpi(4,3),csdel2,ier)
	    else
		calph(1) = 1
		calph(3) = 0
	    endif
	    cfac = 2*csdel2
	    if (lwrite) then
		print *,'slam   =',cfac
		print *,'eta3   =',-4*del3(3)
		print *,'eta4   =',-4*del3(4)
		print *,'alpha  =',calph(1),calph(3)
		print *,'ier    = ',ier
	    endif
	endif
	if ( lwrite ) print '(a)','  ##] determinants:'
*  #] determinants:
*  #[ convert to complex:
	if ( lcroot ) then
	    do 110 k=3,4
		cetalm(k) = etalam(k)
		do 109 i=1,3
		    cel2s(i,k) = del2s(i,k)
		    celpsi(i,k) = delpsi(i,k)
		    cetami(i,k) = etami(i,k)
  109		continue
  110	    continue
	endif
*  #] convert to complex:
*  #[ simple case:
	if ( ldel2s .or. abs(dqiqj(3,4)) .lt. xloss*abs(xqi(3)) ) then
	    if ( .not.lsmug .and. (ldel2s .or. ldc3c4) ) goto 500
	    if ( lwrite ) print *,'Expect cancellations of ',
     +		abs(dqiqj(3,4)/xqi(3))
	endif
*
*	and the calculations
*
	ier0 = ier
	ier1 = ier
	if ( lcroot ) then
	    call ffcc0p(cs4( 1),ipi12(1),isoort(1),clogi(1),ilogi(1),
     +		cpi(1,3),cdpipj(1,1,3),cpiDpj(1,1,3),csdel2,cel2s(1,3),
     +		cetalm(3),cetami(1,3),celpsi(1,3),calph,4,ier0)
	    call ffcc0p(cs4(81),ipi12(9),isoort(9),clogi(4),ilogi(4),
     +		cpi(1,4),cdpipj(1,1,4),cpiDpj(1,1,4),csdel2,cel2s(1,4),
     +		cetalm(4),cetami(1,4),celpsi(1,4),calph,4,ier1)
	else
	    if ( lsmug ) call ffsm43(xpi3(1,3),7-3)
	    call ffxc0p(cs4( 1),ipi12(1),isoort(1),clogi(1),ilogi(1),
     +		xpi3(1,3),dpipj3(1,1,3),piDpj3(1,1,3),sdel2,del2s(1,3),
     +		etalam(3),etami(1,3),delpsi(1,3),alph,4,ier0)
	    if ( lsmug ) call ffsm43(xpi3(1,4),7-4)
	    call ffxc0p(cs4(81),ipi12(9),isoort(9),clogi(4),ilogi(4),
     +		xpi3(1,4),dpipj3(1,1,4),piDpj3(1,1,4),sdel2,del2s(1,4),
     +		etalam(4),etami(1,4),delpsi(1,4),alph,4,ier1)
	endif
	ier = max(ier0,ier1)
	goto 600
*  #] simple case:
*  #[ cancellations:
  500	continue
*
*	There are cancellations between the dilogarithms or the vertex
*	is on threshold.
*	we need the differences ddel2s(i) = del2s(i,3)-del2s(i,4)
*
	do 510 i=2,3
	    if ( i .eq. 2 ) then
		j = 2
	    else
		j = 1
	    endif
	    ddel2s(i) = del2s(i,3) - del2s(i,4)
	    xmax = abs(del2s(i,3))
	    if ( abs(ddel2s(i)) .ge. xloss*xmax ) goto 510
	    if ( lwrite ) print *,'ddel2s(',i,')  = ',ddel2s(i),
     +		abs(del2s(i,3))
*
*	    Very first try with transformation
*
	    s(1) = (ai(3)+ai(4))*daiaj(3,4)*del2s(i,3)/ai(3)**2
	    s(2) = ai(j)**2*ai(4)**2*xpi(j)*dpipj(3,4)
	    s(3) = ai(j)**2*ai(4)**2*piDpj(j,7)*piDpj(j,3)
	    s(4) = ai(j)**2*ai(4)**2*piDpj(j,7)*piDpj(j,4)
	    som = s(1) + s(2) + s(3) + s(4)
	    smax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)))
	    if ( lwrite ) print *,'ddel2s(',i,')+ = ',som,
     +						s(1),s(2),s(3),s(4)
	    if ( abs(som) .ge. xloss*smax ) goto 510
	    if ( smax .lt. xmax ) then
		ddel2s(i) = som
		xmax = smax
	    endif
**
*	    first try (tested, but not needed)
**
*	    s(1) = xqi(j)*dqiqj(3,4)
*	    s(2) = qiDqj(7,j)*qiDqj(j,3)
*	    s(3) = qiDqj(7,j)*qiDqj(j,4)
*	    som = s(1) + s(2) + s(3)
*	    smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	    if ( abs(som) .ge. xloss*smax ) goto 510
*	    if ( lwrite ) print *,' ddel2s(i)  = ',som,s(1),s(2),s(3)
*	    if ( smax .lt. xmax ) then
*		ddel2s(i) = som
*		xmax = smax
*	    endif
**
*	    second try (tested, but not needed)
**
*	    s(1) = xqi(inx(j,3))*dqiqj(3,4)
*	    s(2) = -isgn(j,3)*qiDqj(7,4)*qiDqj(inx(j,3),3)
*	    s(3) = -isgn(j,4)*qiDqj(7,4)*qiDqj(inx(j,4),4)
*	    som = s(1) + s(2) + s(3)
*	    smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	    if ( lwrite ) print *,' ddel2s(i)+ = ',som,s(1),s(2),s(3)
*	    if ( abs(som) .ge. xloss*smax ) goto 510
*	    if ( smax .lt. xmax ) then
*		ddel2s(i) = som
*		xmax = smax
*	    endif
*
*	    maybe insert something intelligent later ...
*
	    if ( lwarn ) call ffwarn(139,ier,ddel2s(i),xmax)
  510	continue
	if ( .not. lcroot ) then
	    call ffdxc0(cs4,ipi12,isoort,clogi,ilogi,xpi3,dpipj3,piDpj3,
     +		xqi,dqiqj,qiDqj,sdel2,del2s,etalam,etami,delpsi,alph,
     +		ddel2s,ldel2s,4,ier)
	else
	    cddl2s(2) = ddel2s(2)
	    cddl2s(3) = ddel2s(3)
	    do 530 i=1,10
		zqi(i) = xqi(i)
		do 520 j=1,10
		    zdqiqj(j,i) = dqiqj(j,i)
		    zqiDqj(j,i) = qiDqj(j,i)
  520		continue
  530	    continue
	    call ffdcc0(cs4,ipi12,isoort,clogi,ilogi,cpi,cdpipj,cpiDpj,
     +		zqi,zdqiqj,zqiDqj,csdel2,cel2s,cetalm,cetami,celpsi,
     +		calph,cddl2s,ldel2s,4,ier)
	endif
  600	continue
*  #] cancellations:
*  #[ Ai<0 terms:
	cipi = DCMPLX(x0,pi)
	if ( ai(3) .lt. 0 .neqv. ai(4) .lt. 0 ) then
*	    we need the S term
	    if ( ai(1) .lt. 0 .eqv. ai(2) .lt. 0 ) then
		if ( lcroot ) then
		   call ffcxra(cs4(167),ipi12(23),xqi,qiDqj,sdel2,1,ier)
		else
*		   call ffxtro(cs4(167),ipi12(23),xqi,qiDqj,sdel2,1,ier)
		   call ffxtra(cs4(167),ipi12(23),xqi,qiDqj,sdel2,1,ier)
		endif
	    else
		if ( lcroot ) then
		   call ffcxra(cs4(167),ipi12(23),xqi,qiDqj,sdel2,2,ier)
		   call ffcxra(cs4(169),ipi12(25),xqi,qiDqj,sdel2,3,ier)
		else
		   call ffxtra(cs4(167),ipi12(23),xqi,qiDqj,sdel2,2,ier)
		   call ffxtra(cs4(169),ipi12(25),xqi,qiDqj,sdel2,3,ier)
*		   call ffxtro(cs4(167),ipi12(23),xqi,qiDqj,sdel2,2,ier)
*		   call ffxtro(cs4(169),ipi12(25),xqi,qiDqj,sdel2,3,ier)
		endif
	    endif
	endif
*
*	The normal correction terms
*
	if ( ai(1) .lt. 0 .neqv. ai(2) .lt. 0 ) then
	    cs4(161) = -cipi*clogi(1)
	    ipi12(17) = 12*ilogi(1)
	    if ( ilogi(1) .eq. -999 ) call fferr(46,ier)
	    cs4(164) = cipi*clogi(4)
	    ipi12(20) = -12*ilogi(4)
	    if ( ilogi(4) .eq. -999 ) call fferr(46,ier)
	endif
	if ( ai(2) .lt. 0 .neqv. ai(3) .lt. 0 ) then
	    cs4(162) = -cipi*clogi(2)
	    ipi12(18) = 12*ilogi(2)
	    if ( ilogi(2) .eq. -999 ) call fferr(46,ier)
	endif
	if ( ai(3) .lt. 0 .neqv. ai(1) .lt. 0 ) then
	    cs4(163) = -cipi*clogi(3)
	    ipi12(19) = 12*ilogi(3)
	    if ( ilogi(3) .eq. -999 ) call fferr(46,ier)
	endif
	if ( ai(2) .lt. 0 .neqv. ai(4) .lt. 0 ) then
	    cs4(165) = cipi*clogi(5)
	    ipi12(21) = -12*ilogi(5)
	    if ( ilogi(5) .eq. -999 ) call fferr(46,ier)
	endif
	if ( ai(4) .lt. 0 .neqv. ai(1) .lt. 0 ) then
	    cs4(166) = cipi*clogi(6)
	    ipi12(22) = -12*ilogi(6)
	    if ( ilogi(6) .eq. -999 ) call fferr(46,ier)
	endif
	if ( lwrite ) print *,'signs Ai: ',(nint(sign(x1,ai(i))),i=1,4)
*  #] Ai<0 terms:
*###] ffxd0p:
	end
*###[ ffpi43:
	subroutine ffpi43(xpi3,dpipj3,piDpj3,xpi,dpipj,piDpj,imiss,ier)
***#[*comment:***********************************************************
*									*
*	Fill the threepoint arrays xpi3 and dpipj3 with masses from the	*
*	the fourpoint array xpi with leg imiss cut out.			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE PRECISION xpi3(6),dpipj3(6,6),piDpj3(6,6)
	DOUBLE PRECISION xpi(10),dpipj(10,10),piDpj(10,10)
	integer imiss,ier
*
*	local variables
*
	integer i,j
	integer iinx(6,4)
	DOUBLE PRECISION xmin,xmax,a
	save iinx
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
*  #] declarations:
*  #[ calculations:
*	if ( lscale ) then
*	    xmax = abs(xpi(iinx(1,imiss)))
*	    xmin = xmax
*	    do 5 i=2,6
*		a = abs(xpi(iinx(i,imiss)))
*		xmax = max(xmax,a)
*		xmin = min(xmin,a)
*    5	    continue
*	    scale = (xmax*sqrt(xmin))**(-2/3.)
*	else
*	    scale = 1
*	endif
	do 20 i=1,6
	    xpi3(i) = xpi(iinx(i,imiss))
	    do 10 j=1,6
		dpipj3(j,i) = dpipj(iinx(j,imiss),iinx(i,imiss))
		piDpj3(j,i) = piDpj(iinx(j,imiss),iinx(i,imiss))
   10	    continue
   20	continue
*	call ffxhck(xpi3,dpipj3,6,ier)
*	if ( lscale .and. lwrite ) then
*	    print *,'ffpi43: scaled momenta:'
*	    print *,xpi3
*	endif
*  #] calculations:
*###] ffpi43:
	end
*###[ ffxtra:
	subroutine ffxtra(cs4,ipi12,xqi,qiDqj,sdel2,ii,ier)
***#[*comment:***********************************************************
*									*
*	calculate the extra terms S_ii^{\infty\prime}, put them in	*
*	cs4 and ipi12.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi12(3),ii,ier
	DOUBLE COMPLEX cs4(3)
	DOUBLE PRECISION xqi(10),qiDqj(10,10),sdel2
*
*	local variables
*
	integer i,ip(5)
	DOUBLE PRECISION x(2,3),dfflo1,s,s1
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data ip/5,6,8,5,6/
*  #] declarations:
*  #[ calculations:
	if ( ii .eq. 3 ) return
	do 10 i=1,3
	    if ( ii .eq. 1 .and. i .eq. 2 ) goto 10
	    call ffroot(x(1,i),x(2,i),xqi(ip(i)),-qiDqj(ip(i),
     +		ip(i+1)),xqi(ip(i+1)),sdel2,ier)
	    s = -x(2,i)/x(1,i)
	    if ( lwrite ) then
		print *,'s = ',s
	    endif
	    if ( abs(s-1) .lt. xloss ) then
		if ( lwrite ) then
		print *,'s''=',1+2*qiDqj(ip(i),ip(i+1))/(xqi(ip(i))*
     +				x(1,i))
		endif
		s1 = dfflo1(-2*qiDqj(ip(i),ip(i+1))/(xqi(ip(i))*x(1,i)),
     +				ier)
	    elseif ( s .gt. 0 ) then
		s1 = log(s)
	    else
		if ( abs(s+1) .lt. xloss ) then
		    if ( lwrite ) then
			print *,'s''=',-1-2*sdel2/(xqi(ip(i))*x(1,i))
		    endif
		    s1 = dfflo1(-2*sdel2/(xqi(ip(i))*x(1,i)),ier)
		else
		    s1 = log(-s)
		endif
*		also here an minus sign (-i*pi*log(-(p.p-sqrt)/(p.p+sqrt)))
		if ( qiDqj(ip(i),ip(i+1))*xqi(ip(i))*sdel2 .gt. 0 ) then
		    ipi12(i) = +12
		else
		    ipi12(i) = -12
		endif
*		ier = ier + 50
*		print *,'ffxtra: imaginary part may well be wrong -> ',
*     +			'n*pi^2 fout'
*		print *,'        ipi12(i) = ',ipi12(i)
*		print *,'        qiDqj    = ',qiDqj(ip(i),ip(i+1))
*		print *,'        qi^2     = ',xqi(ip(i))
	    endif
*	    there is an overall minus compared with Veltman
	    cs4(i) = DCMPLX(x0,-pi*s1)
	    if ( sdel2 .lt. 0 ) then
		cs4(i) = -cs4(i)
		ipi12(i) = -ipi12(i)
	    endif
	    if ( ii .ne. 1 ) then
		cs4(i) = -cs4(i)
		ipi12(i) = -ipi12(i)
	    endif
	    if ( i .eq. 2 ) then
		cs4(i) = 2*cs4(i)
		ipi12(i) = 2*ipi12(i)
	    endif
   10	continue
*  #] calculations:
*  #[ debug:
	if ( lwrite ) then
	    print *,'ffxtra: ii    = ',ii
	    print *,'        sdel2 = ',sdel2
	    print *,'        x     = ',x
	    print *,'        cs4   = ',cs4
	    print *,'        ipi12 = ',ipi12
	endif
*  #] debug:
*###] ffxtra:
	end
*###[ ffcxra:
	subroutine ffcxra(cs4,ipi12,xqi,qiDqj,sdel2,ii,ier)
***#[*comment:***********************************************************
*									*
*	calculate the extra terms S_ii^{\infty\prime}, put them in	*
*	cs4 and ipi12 for qi real but sdel2 complex.			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipi12(3),ii,ier
	DOUBLE COMPLEX cs4(3)
	DOUBLE PRECISION xqi(10),qiDqj(10,10),sdel2
*
*	local variables
*
	integer i,ip(5)
	DOUBLE COMPLEX x(2,3),zfflo1,s,s1,c
	DOUBLE PRECISION absc
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data ip/5,6,8,5,6/
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ calculations:
	if ( ii .eq. 3 ) return
	do 10 i=1,3
	    if ( ii .eq. 1 .and. i .eq. 2 ) goto 10
	    x(1,i) = DCMPLX(-qiDqj(ip(i),ip(i+1))/xqi(ip(i)),
     +			-sdel2/xqi(ip(i)))
	    x(2,i) = DCMPLX(-qiDqj(ip(i),ip(i+1))/xqi(ip(i)),
     +			+sdel2/xqi(ip(i)))
	    s = -x(2,i)/x(1,i)
	    if ( lwrite ) then
		print *,'s = ',s
	    endif
	    c = s-1
	    if ( absc(c) .lt. xloss ) then
		if ( lwrite ) then
		print *,'s''=',1+DBLE(2*qiDqj(ip(i),ip(i+1))/xqi(ip(i)))
     +				/x(1,i)
		endif
		s1 = zfflo1(DBLE(-2*qiDqj(ip(i),ip(i+1))/xqi(ip(i)))/
     +				x(1,i),ier)
	    elseif ( abs(s+1) .lt. xloss ) then
		if ( lwrite ) then
		    print *,'s''=',-1+DCMPLX(x0,2*sdel2/xqi(ip(i)))/
     +			x(1,i)
		endif
		s1 = zfflo1(DCMPLX(x0,-2*sdel2/xqi(ip(i)))/x(1,i),ier)
		if ( DIMAG(c).gt.0 ) then
		    ipi12(i) = +12
		else
		    ipi12(i) = -12
		endif
	    else
		s1 = log(s)
	    endif
*	    there is an overall minus compared with Veltman
	    cs4(i) = DCMPLX(pi*DIMAG(s1),-pi*DBLE(s1))
	    if ( ii .ne. 1 ) then
		cs4(i) = -cs4(i)
		ipi12(i) = -ipi12(i)
	    endif
	    if ( sdel2 .lt. 0 ) then
		cs4(i) = -cs4(i)
		ipi12(i) = -ipi12(i)
	    endif
	    if ( i .eq. 2 ) then
		cs4(i) = 2*cs4(i)
		ipi12(i) = 2*ipi12(i)
	    endif
   10	continue
*  #] calculations:
*  #[ debug:
	if ( lwrite ) then
	    print *,'ffcxra: ii    = ',ii
	    print *,'        sdel2 = ',sdel2
	    print *,'        x     = ',x
	    print *,'        cs4   = ',cs4
	    print *,'        ipi12 = ',ipi12
	endif
*  #] debug:
*###] ffcxra:
	end
*###[ ffsm43:
	subroutine ffsm43(xpi3,imiss)
***#[*comment:***********************************************************
*									*
*	Distribute the smuggled 4point momenta to the 3point smuggled	*
*	momenta.  Note that because of the common block smuggling this	*
*	cannot be included in ffpi43.					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer imiss
	DOUBLE PRECISION xpi3(6)
*
*	local variables
*
	integer i,j,iinx(6,4)
	save iinx
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
*
*  #] declarations:
*  #[ parcel out:
	if ( lsmug ) then
*
*	    parcel out the smuggled diffs
*
	    do 30 i=1,3
		j = mod(i,3)+1
		if ( xpi3(j) .eq. 0 ) then
		    cmipj(i,i) = c2sisj(iinx(i,imiss),iinx(j,imiss))
		elseif ( xpi3(i) .eq. 0 ) then
		    cmipj(j,i) = c2sisj(iinx(i,imiss),iinx(j,imiss))
		endif
   30	    continue
	endif
*  #] parcel out:
*)##] ffsm43:
	end
