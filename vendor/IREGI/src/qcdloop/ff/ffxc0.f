*	$Id: ffxc0.f,v 1.5 1996/08/15 09:36:47 gj Exp $
*###[ ffxc0:
	subroutine ffxc0(cc0,xpi,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the threepoint function closely following		*
*	recipe in 't Hooft & Veltman, NP B(183) 1979.			*
*	Bjorken and Drell metric is used nowadays!			*
*									*
*	    p2	| |							*
*		v |							*
*		 / \							*
*	      m2/   \m3 						*
*	p1     /     \	p3						*
*	->    /  m1   \ <-						*
*	------------------------					*
*									*
*		1   /			     1				*
*	    = ----- \d^4Q----------------------------------------	*
*	      ipi^2 /	 [Q^2-m1^2][(Q+p1)^2-m2^2][(Q-p3)^2-m3^2]	*
*									*
*	If the function is infra-red divergent (p1=m2,p3=m3,m1=0 or	*
*	cyclic) the function is calculated with a user-supplied cutoff	*
*	delta in the common block /ffcut/.				*
*									*
*	Input:	xpi	(real)		i=1,3: mass^2, i=4,6: pi.pi	*
*	Output: cc0	(complex)	C0, the threepoint function.	*
*		ier	(integer)	0=ok, 1=inaccurate, 2=error	*
*	Calls:	ffxc0p,ffxb0p						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cc0
	DOUBLE PRECISION xpi(6)
	integer ier
*
*	local variables:
*
	integer i,j,ier0
	DOUBLE PRECISION dpipj(6,6)
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ convert input:
	if ( lwrite ) then
	    print *,'ffxc0: input = ',xpi
	endif
	if ( lwarn ) then
	    do 30 i=1,6
		dpipj(i,i) = 0
		do 29 j = i+1,6
		    dpipj(j,i) = xpi(j) - xpi(i)
		    dpipj(i,j) = - dpipj(j,i)
		    if ( abs(dpipj(j,i)) .lt. xloss*abs(xpi(i)) .and.
     +			xpi(i) .ne. xpi(j) ) then
			ier0 = 0
			call ffwarn(87,ier0,dpipj(j,i),xpi(i))
			if ( lwrite ) print *,'between xpi(',i,
     +				') and xpi(',j,')'
		    endif
   29		continue
   30	    continue
	else
	    do 40 i=1,6
		do 39 j = 1,6
		    dpipj(j,i) = xpi(j) - xpi(i)
   39		continue
   40	    continue
	endif
*  #] convert input:
*  #[ call ffxc0a:
	call ffxc0a(cc0,xpi,dpipj,ier)
*  #] call ffxc0a:
*###] ffxc0:
	end
*###[ ffxc0a:
	subroutine ffxc0a(cc0,xpi,dpipj,ier)
***#[*comment:***********************************************************
*									*
*	See ffxc0.							*
*									*
*	Input:	xpi	(real)		i=1,3: mass^2, i=4,6: pi.pi	*
*		dpipj	(real)		= xpi(i) - xpi(j)		*
*	Output: cc0	(complex)	C0, the threepoint function.	*
*		ier	(integer)	0=ok, 1=inaccurate, 2=error	*
*	Calls:	ffxc0p,ffxb0p						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cc0
	DOUBLE PRECISION xpi(6),dpipj(6,6)
	integer ier
*
*	local variables:
*
	logical ljust
	integer i,j,inew(6,6),idotsa,ier0
*	DOUBLE COMPLEX cs,cs1,cs2
	DOUBLE COMPLEX c
	DOUBLE PRECISION xqi(6),dqiqj(6,6),qiDqj(6,6),absc,delta0,
     +		dum66(6,6),rloss,xnul,xmax
	save inew,delta0
*
*	common blocks:
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	memory
*
	integer iermem(memory),ialmem(memory),memind,ierini
	DOUBLE PRECISION xpimem(6,memory),dl2mem(memory)
	DOUBLE COMPLEX cc0mem(memory)
	save memind,iermem,ialmem,xpimem,dl2mem,cc0mem
	data memind /0/
*
*	statement function:
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data delta0 /0./
	data inew /1,2,3,4,5,6,
     +		   2,3,1,5,6,4,
     +		   3,1,2,6,4,5,
     +		   1,3,2,6,5,4,
     +		   3,2,1,5,4,6,
     +		   2,1,3,4,6,5/
*  #] declarations:
*  #[ initialisations:
	if ( lmem .and. memind .eq. 0 ) then
	    do 2 i=1,memory
		do 1 j=1,6
		    xpimem(j,i) = 0
    1		continue
		ialmem(i) = 0
    2	    continue
	endif
	idsub = 0
	ljust = .FALSE.
*  #] initialisations:
*  #[ check input:
	if ( ltest ) then
*	    I have had too many problems here ...
	    if ( abs(isgnal) .ne. 1 ) then
		print *,'ffxc0: error: ab(isgnal) /= 1!',isgnal
		if ( isgnal .eq. 0 ) then
		    isgnal = 1
		else
		    isgnal = sign(1,isgnal)
		endif
	    endif
*	    check input dot products if present
	    if ( idot.gt.0 ) then
	    	ier0 = ier
	    	idotsa = idot
	    	idot = 0
	    	call ffdot3(qiDqj,xpi,dpipj,6,ier0)
	    	idot = idotsa
	    	rloss = xloss**2*DBLE(10)**(-mod(ier0,50))
	    	if ( idot.le.2 ) then
		    do 20 i=4,6
		        do 10 j=4,6
			    xnul = abs(fpij3(j,i)-qiDqj(j,i))
			    xmax = abs(qiDqj(j,i))
			    if ( rloss*xnul .gt. precx*xmax ) print *,
     +			    'ffxc0a: error: input dotproduct piDpj(',j,
     +			    i,') wrong:',fpij3(j,i),qiDqj(j,i),xnul,ier0
   10		    	continue
   20		    continue
	       	else
		    do 40 i=1,6
		    	do 30 j=1,6
			    xnul = abs(fpij3(j,i)-qiDqj(j,i))
			    xmax = abs(qiDqj(j,i))
			    if ( rloss*xnul .gt. precx*xmax ) print *,
     +			    'ffxc0a: error: input dotproduct piDpj(',j,
     +			    i,') wrong:',fpij3(j,i),qiDqj(j,i),xnul,ier0
   30		    	continue
   40		    continue
	    	endif
	    endif
	endif
*  #] check input:
*  #[ handel special cases:
*
*	The infrared divergent diagrams are calculated in ffxc0i:
*
	if ( dpipj(2,4).eq.0 .and. dpipj(3,6).eq.0 .and. xpi(1).eq.0
     +	.or. dpipj(3,5).eq.0 .and. dpipj(1,4).eq.0 .and. xpi(2).eq.0
     +	.or. dpipj(1,6).eq.0 .and. dpipj(2,5).eq.0 .and. xpi(3).eq.0 )
     +	then
	    call ffxc0i(cc0,xpi,dpipj,ier)
	    return
	endif
**
*	The general case cannot handle xpi=0, pj=pk. These are simple
*	though.
**
*	goto 50
*	if ( xpi(4) .eq. 0 .and. dpipj(5,6) .eq. 0 .and.
*     +		dpipj(1,2) .ne. 0 ) then
*	    call ffxb0p(cs1,-xpi(5),xpi(1),xpi(3),dpipj(1,6),dpipj(3,5),
*     +			dpipj(1,3),ier)
*	    call ffxb0p(cs2,-xpi(5),xpi(2),xpi(3),dpipj(2,5),dpipj(3,5),
*     +			dpipj(2,3),ier)
*	    cs = cs1 - cs2
*	    cc0 = cs/dpipj(1,2)
*	elseif ( xpi(6) .eq. 0 .and. dpipj(4,5) .eq. 0 .and.
*     +		dpipj(3,1) .ne. 0 ) then
*	    call ffxb0p(cs1,-xpi(4),xpi(3),xpi(2),dpipj(3,5),dpipj(2,4),
*     +			dpipj(3,2),ier)
*	    call ffxb0p(cs2,-xpi(4),xpi(1),xpi(2),dpipj(1,4),dpipj(2,4),
*     +			dpipj(1,2),ier)
*	    cs = cs1 - cs2
*	    cc0 = cs/dpipj(3,1)
*	elseif ( xpi(5) .eq. 0 .and. dpipj(6,4) .eq. 0 .and.
*     +		dpipj(2,3) .ne. 0 ) then
*	    call ffxb0p(cs1,-xpi(6),xpi(2),xpi(1),dpipj(2,4),dpipj(1,6),
*     +			dpipj(2,1),ier)
*	    call ffxb0p(cs2,-xpi(6),xpi(3),xpi(1),dpipj(3,6),dpipj(1,6),
*     +			dpipj(3,1),ier)
*	    cs = cs1 - cs2
*	    cc0 = cs/dpipj(2,3)
*	else
*	    goto 50
*	endif
**
*	common piece - excuse my style
**
*	print *,'ffcc0: WARNING: this algorithm has not yet been tested'
*	if ( lwarn .and. absc(cs) .lt. xloss*absc(cs1) )
*     +		call ffwarn(28,ier,absc(cs),absc(cs1))
**
*	debug output
**
*	if (lwrite) then
*	    print *,'simple case xpi=0,xpj=xpk, two twopoint functions:'
*	    print *,cs1,cs2
*	    print *,'result: cc0=',cc0,ier
*	endif
*	return
*   50	continue
*  #] handel special cases:
*  #[ rotate to alpha in (0,1):
	call ffrot3(irota3,xqi,dqiqj,qiDqj,xpi,dpipj,dum66,6,2,3,ier)
*  #] rotate to alpha in (0,1):
*  #[ look in memory:
	ierini = ier+ner
	if ( lmem .and. delta .eq. delta0 ) then
	    do 70 i=1,memory
		do 60 j=1,6
		    if ( xqi(j) .ne. xpimem(j,i) ) goto 70
   60		continue
		if ( ialmem(i) .ne. isgnal ) goto 70
*		we found an already calculated masscombination ..
*		(maybe check differences as well)
		if ( lwrite ) print *,'ffxc0: using previous result'
		cc0 = cc0mem(i)
		ier = ier+iermem(i)
		if ( ldot ) then
		    fdel2 = dl2mem(i)
*		    we forgot to recalculate the stored quantities
		    ljust = .TRUE.
		    goto 71
		endif
		return
   70	    continue
*	    if ( lwrite ) print *,'ffxc0: not found in memory'
	elseif ( lmem ) then
	    delta0 = delta
	endif
   71	continue
*  #] look in memory:
*  #[ dot products:
	call ffdot3(qiDqj,xqi,dqiqj,6,ier)
*
*	save dotproducts for tensor functions if requested
*
	if ( ldot ) then
	    do 75 i=1,6
		do 74 j=1,6
		    fpij3(j,i) = qiDqj(inew(i,irota3),inew(j,irota3))
   74		continue
   75	    continue
	    if ( irota3 .gt. 3 ) then
*
*		the sign of the s's has been changed!
*
		do 77 i=1,3
		    do 76 j=4,6
			fpij3(j,i) = -fpij3(j,i)
			fpij3(i,j) = -fpij3(i,j)
   76		    continue
   77		continue
	    endif
	endif
	if ( ljust ) return
*  #] dot products:
*  #[ call ffxc0b:
	call ffxc0b(cc0,xqi,dqiqj,qiDqj,ier)
*  #] call ffxc0b:
*  #[ add to memory:
	if ( lmem ) then
	    memind = memind + 1
	    if ( memind .gt. memory ) memind = 1
	    do 200 j=1,6
		xpimem(j,memind) = xqi(j)
  200	    continue
	    cc0mem(memind) = cc0
	    iermem(memind) = ier+ner-ierini
	    ialmem(memind) = isgnal
	    dl2mem(memind) = fdel2
	endif
*  #] add to memory:
*###] ffxc0a:
	end
*###[ ffxc0b:
	subroutine ffxc0b(cc0,xqi,dqiqj,qiDqj,ier)
***#[*comment:***********************************************************
*									*
*	See ffxc0.							*
*									*
*	Input:	xpi	(real)		i=1,3: mass^2, i=4,6: pi.pi	*
*		dpipj	(real)		= xpi(i) - xpi(j)		*
*	Output: cc0	(complex)	C0, the threepoint function.	*
*		ier	(integer)	0=ok, 1=inaccurate, 2=error	*
*	Calls:	ffxc0p,ffxb0p						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cc0
	DOUBLE PRECISION xqi(6),dqiqj(6,6),qiDqj(6,6)
	integer ier
*
*	local variables:
*
	integer nerr
	parameter(nerr=6)
	integer isoort(8),ipi12(8),i,j,k,ipi12t,ilogi(3),ier0,ieri(nerr)
	DOUBLE COMPLEX cs3(80),cs,cs1,cs2,c,clogi(3),cslam,cetalm,
     +		cetami(6),cel2s(3),clamp,calph(3),cblph(3),csdel2,
     +		cqi(6),cdqiqj(6,6),cqiDqj(6,6),celpsi(3),cdum(3),
     +		cdum2(3,3)
	DOUBLE PRECISION del2,del2s(3),del3,delpsi(3),
     +		del3mi(3)
	DOUBLE PRECISION xmax,absc,alph(3),etalam,etami(6),sdel2,
     +		xlamp,eta,blph(3)
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
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'input: xqi,dqiqj'
	    do 1 i=1,6
	    	print *,xqi(i),(dqiqj(i,j),j=1,6)
    1	    continue
	endif
*  #] check input:
*  #[ calculations:
*
*	some determinants
*
	if ( lwrite ) print '(a)','  ##[ determinants:'
	do 98 i = 1,nerr
	    ieri(i) = 0
   98	continue
	call ffdel2(del2,qiDqj, 6, 4,5,6, 1,ier)
	if ( lwrite ) print *,'ffxc0: del2 = ',del2
	if ( ldot ) fdel2 = del2
	if ( del2 .gt. 0 ) then
*	    shouldn't occur ...
*	    12-10-1993 three spacelike momenta are OK
	    if ( .not.(xqi(4).lt.0 .and. xqi(5).lt.0 .and. xqi(6).lt.0)
     +	    		) then
     		call fferr(41,ier)
     	    	print *,'xpi = ',xqi
     	    endif
	elseif ( del2 .eq. 0 ) then
	    call fferr(42,ier)
	    return
	endif
	call ffdel3(del3,xqi,qiDqj,6,ier)
	call ffdl3m(del3mi,.TRUE.,del3,del2,xqi,dqiqj,qiDqj,6, 4,5,6,
     +							1,3,ier)
	do 101 i=1,3
	    j = i+1
	    if ( j .eq. 4 ) j = 1
	    call ffdel2(del2s(i),qiDqj,6, i+3,i,j, 1,ieri(i))
	    k = i-1
	    if ( k .eq. 0 ) k = 3
	    call ffdl2p(delpsi(i),xqi,dqiqj,qiDqj,i+3,j+3,k+3,i,j,k,6,
     +							ieri(i+3))
  101	continue
	ier0 = 0
	do 99 i=1,nerr
	    ier0 = max(ier0,ieri(i))
   99	continue
	ier = ier + ier0
*
*	initialize cs3:
*
	do 80 i=1,80
	    cs3(i) = 0
   80	continue
	do 90 i=1,8
	    ipi12(i) = 0
   90	continue
	do 100 i=1,3
	    clogi(i) = 0
	    ilogi(i) = 0
  100	continue
*  #[ complex case:
*	in case of three spacelike momenta or unphysical real ones
	if ( del2 .gt. 0 ) then
	    do 102 i=1,3
		cel2s(i) = del2s(i)
		celpsi(i) = delpsi(i)
		cetami(i) = del3mi(i)/del2
  102	    continue
	    do 104 i=1,6
		cqi(i) = xqi(i)
		do 103 j=1,6
		    cdqiqj(j,i) = dqiqj(j,i)
		    cqiDqj(j,i) = qiDqj(j,i)
  103		continue
  104	    continue
	    cetalm = del3/del2
	    csdel2 = isgnal*DCMPLX(x0,sqrt(del2))
*
*	    get alpha,1-alpha
*
	    call ffcoot(cblph(1),calph(1),cqi(5),-cqiDqj(5,6),cqi(6),
     +							csdel2,ier)
	    call ffcoot(calph(3),cblph(3),cqi(5),-cqiDqj(5,4),cqi(4),
     +							csdel2,ier)
	    cslam = 2*csdel2
	    if (lwrite) then
		print '(a)','  ##[ get roots: (ffxc0)'
		print *,'cslam   =',2*csdel2
		ier0 = ier
*		call ffclmb(clamp,cqi(4),cqi(5),cqi(6),cdqiqj(4,5),
*     +					cdqiqj(4,6),cdqiqj(5,6),ier0)
*		print *,'cslamp  =',sqrt(clamp)
		print *,'ceta    =',-4*del3
*		call ffeta(eta,xpi,dpipj,6,ier0)
*		print *,'cetap   =',eta
		print *,'cetalam =',cetalm
		print *,'calpha  = ',calph(1),calph(3)
	    endif
	    if ( lwrite ) print '(a)','  ##] determinants:'
	    call ffcc0p(cs3,ipi12,isoort,clogi,ilogi,cqi,cdqiqj,cqiDqj,
     +		csdel2,cel2s,cetalm,cetami,celpsi,calph,3,ier)
	    goto 109
	endif
*  #] complex case:
	etalam = del3/del2
	do 106 i=1,3
	    etami(i) = del3mi(i)/del2
  106	continue
	if ( abs(isgnal).ne.1 ) then
	    print *,'ffxc0b: error: isgnal should be +/-1, not ',isgnal
	    print *,'        forgot to call ffini?'
	    call ffini
	endif
	sdel2 = isgnal*sqrt(-del2)
*
*	get alpha,1-alpha
*
	call ffroot(blph(1),alph(1),xqi(5),-qiDqj(5,6),xqi(6),sdel2,ier)
	call ffroot(alph(3),blph(3),xqi(5),-qiDqj(5,4),xqi(4),sdel2,ier)
	if ( l4also .and. ( alph(1) .gt. 1 .or. alph(1) .lt. 0 ) .and.
     +		abs(blph(1)-x05) .lt. abs(alph(1)-x05) ) then
	    alph(1) = blph(1)
	    alph(3) = blph(3)
	    sdel2 = -sdel2
	    isgnal = -isgnal
	endif
	cslam = 2*sdel2
	if (lwrite) then
	    print '(a)','  ##[ get roots:'
	    print *,'slam   =',2*sdel2
*	    ier0 = ier
*	    call ffxlmb(xlamp,xqi(4),xqi(5),xqi(6),dqiqj(4,5),
*     +					dqiqj(4,6),dqiqj(5,6),ier0)
*	    print *,'slamp  =',sqrt(xlamp)
	    print *,'eta    =',-4*del3
*	    call ffeta(eta,xpi,dpipj,6,ier0)
*	    print *,'etap   =',eta
	    print *,'etalam =',etalam
	    print *,'alpha = ',alph(1),alph(3)
	endif
	if ( lwrite ) print '(a)','  ##] determinants:'
*
*	and the calculations
*
	call ffxc0p(cs3,ipi12,isoort,clogi,ilogi,xqi,dqiqj,qiDqj,
     +			sdel2,del2s,etalam,etami,delpsi,alph,3,ier)
*
*	sum'em up:
*
  109	continue
	cs = 0
	xmax = 0
	do 110 i=1,80
*	    if ( cs3(i) .ne. 0 ) then
		cs = cs + cs3(i)
		xmax = max(xmax,absc(cs))
*	    endif
  110	continue
	ipi12t = 0
	do 120 i=1,8
	    ipi12t = ipi12t + ipi12(i)
  120	continue
	cs = cs + ipi12t*DBLE(pi12)
*
*	Check for cancellations in the final adding up (give a fctor 2)
*
	if ( lwarn .and. 2*absc(cs) .lt. xloss*xmax )
     +		call ffwarn(29,ier,absc(cs),xmax)
*
*	Check for a sum close to the minimum of the range (underflow
*	problems)
*
	if ( lwarn .and. absc(cs) .lt. xalogm/precc )
     +		call ffwarn(120,ier,absc(cs),xalogm/precc)
*
*	A imaginary component less than precc times the real part is
*	zero (may be removed)
*
	if ( abs(DIMAG(cs)) .lt. precc*abs(DBLE(cs)) )
     +	    cs = DCMPLX(DBLE(cs))
*
*	Finally ...
*
	cc0 = - cs/cslam
*  #] calculations:
*  #[ debug:
	if(lwrite)then
*	    print '(a)','  ##[ all terms: '
*	    print *,'s3''s :'
* 1000	    format(g12.6,1x,g12.6,1x,g12.6,1x,g12.6,1x,g12.6,1x,
*     +		g12.6,1x,g12.6,1x,g12.6)
*	    print 1000,(cs3(i),cs3(i+20),cs3(i+40),cs3(i+60),i=1,20)
	    print *,'ipi12: ',ipi12
	    print *,'isoort:' ,isoort
*	    print '(a)','  ##] all terms: '
	    print *,'som  :',cs,ipi12t,ier
	    print *,'cc0  :',cc0
	endif
*  #] debug:
*###] ffxc0b:
	end
*###[ ffrot3:
	subroutine ffrot3(irota,xqi,dqiqj,qiDqj,xpi,dpipj,piDpj,ns,
     +		iflag,npoin,ier)
***#[*comment:***********************************************************
*									*
*	rotates the arrays xpi, dpipj into xqi,dqiqj so that		*
*	xpi(6),xpi(4) suffer the strongest outside cancellations and	*
*	xpi(6) > xpi(4) if iflag = 1, so that xpi(5) largest and xpi(5)	*
*	and xpi(6) suffer cancellations if iflag = 2.			*
*	if iflag = 3 make xqi(3)=0.					*
*	If npoin=4, rotate piDpj into qiDqj as well.			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer irota,ns,iflag,ier,npoin
	DOUBLE PRECISION xpi(6),dpipj(6,6),piDpj(6,6),xqi(6),dqiqj(6,6),
     +		qiDqj(6,6)
*
*	local variables
*
	DOUBLE PRECISION a1,a2,a3,xpimax
	DOUBLE COMPLEX chulp(3,3)
	integer i,j,inew(6,6)
	save inew
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data inew /1,2,3,4,5,6,
     +		   2,3,1,5,6,4,
     +		   3,1,2,6,4,5,
     +		   1,3,2,6,5,4,
     +		   3,2,1,5,4,6,
     +		   2,1,3,4,6,5/
*  #] declarations:
*  #[ check input:
	if ( ltest .and. ns .ne. 6 ) print *,'ffrot3: error: ns /= 6'
*  #] check input:
*  #[ get largest cancellation:
	if ( iflag .eq. 1 ) then
	    a1 = abs(dpipj(6,4))/max(abs(xpi(6)+xpi(4)),xalogm)
	    a2 = abs(dpipj(5,4))/max(abs(xpi(5)+xpi(4)),xalogm)
	    a3 = abs(dpipj(5,6))/max(abs(xpi(6)+xpi(5)),xalogm)
	    if ( a1 .le. a2 .and. a1 .le. a3 ) then
		irota = 1
		if ( abs(xpi(6)) .lt. abs(xpi(4)) ) then
		    irota = 4
		endif
	    elseif ( a2 .le. a3 ) then
		irota = 3
		if ( abs(xpi(4)) .lt. abs(xpi(5)) ) then
		    irota = 6
		endif
	    else
		irota = 2
		if ( abs(xpi(5)) .lt. abs(xpi(6)) ) then
		    irota = 5
		endif
	    endif
	elseif ( iflag .eq. 2 ) then
	    xpimax = max(xpi(4),xpi(5),xpi(6))
	    if ( xpimax .eq. 0 ) then
		if ( xpi(5) .ne. 0 ) then
		    irota = 1
		elseif ( xpi(4) .ne. 0 ) then
		    irota = 2
		elseif ( xpi(6) .ne. 0 ) then
		    irota = 3
		else
		    call fferr(40,ier)
		    irota = 1
		endif
	    elseif ( xpi(5) .eq. xpimax ) then
		if ( xpi(4) .le. xpi(6) ) then
		    irota = 1
		else
		    irota = 4
		endif
	    elseif ( xpi(4) .eq. xpimax ) then
		if ( xpi(5) .ge. xpi(6) ) then
		    irota = 2
		else
		    irota = 5
		endif
	    else
		if ( xpi(4) .ge. xpi(6) ) then
		    irota = 3
		else
		    irota = 6
		endif
	    endif
	elseif ( iflag .eq. 3 ) then
	    if ( dpipj(2,4).eq.0 .and. dpipj(3,6).eq.0 .and.
     +			xpi(1).eq.0 ) then
		irota = 3
	    elseif ( dpipj(1,6).eq.0 .and. dpipj(2,5).eq.0 .and.
     +			xpi(3).eq.0 ) then
		irota = 1
	    elseif ( dpipj(3,5).eq.0 .and. dpipj(1,4).eq.0 .and.
     +			xpi(2).eq.0 ) then
		irota = 2
	    else
		call fferr(35,ier)
	    	irota = 1
	    endif
	else
	    call fferr(35,ier)
	    irota = 1
	endif
	if ( lwrite ) print *,'ffrot3: rotated over ',irota,' positions'
*  #] get largest cancellation:
*  #[ rotate:
	do 20 i=1,6
	    xqi(inew(i,irota)) = xpi(i)
	    do 10 j=1,6
		dqiqj(inew(i,irota),inew(j,irota)) = dpipj(i,j)
   10	    continue
   20	continue
*
*	when called in a 4pointfunction we already have the dotproducts
*
	if ( npoin .eq. 4 ) then
	    do 80 j=1,6
	    	do 70 i=1,6
	    	    qiDqj(inew(i,irota),inew(j,irota)) = piDpj(i,j)
   70	    	continue
   80	    continue
	endif
*DEBUG	if ( iflag .eq. 3 .and. lsmug ) then
	if ( lsmug ) then
*	    
*	    do not forget to rotate the smuggled differences
*	    
	    do 40 j=1,3
	    	do 30 i=1,3
	    	    chulp(i,j) = cmipj(i,j)
   30	    	continue
   40	    continue
	    do 60 j=1,3
	    	do 50 i=1,3
	    	    cmipj(inew(i,irota),inew(j+3,irota)-3) = chulp(i,j)
   50	    	continue
   60	    continue
	endif
*  #] rotate:
*  #[ test output:
	if ( ltest ) then
	    call ffxhck(xqi,dqiqj,6,ier)
	    if ( iflag .eq. 3 .and. xqi(3) .ne. 0 ) print *,
     +		'ffrot3: IR divergent C0 rotated wrongly!',xqi
	endif
*  #] test output:
*###] ffrot3:
	end
*###[ ffdot3:
	subroutine ffdot3(piDpj,xpi,dpipj,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate the dotproducts pi.pj with				*
*									*
*		pi = si		i1=1,3					*
*		pi = p(i-3)	i1=4,6					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ns,ier
	DOUBLE PRECISION xpi(6),dpipj(6,6),piDpj(6,6)
*
*	locals
*
	integer is1,is2,is3,ip1,ip2,ip3,i,j,ier0,ier1,inew(6,6)
	DOUBLE PRECISION xheck,xlosn
	save inew
*
*	rest
*
	include 'ff.h'
*
*	data
*
	data inew /1,2,3,4,5,6,
     +		   2,3,1,5,6,4,
     +		   3,1,2,6,4,5,
     +		   1,3,2,6,5,4,
     +		   3,2,1,5,4,6,
     +		   2,1,3,4,6,5/
*
*  #] declarations:
*  #[ check input:
	if ( ns .ne. 6 ) print *,'ffdot3: error: ns /= 6 '
	if ( ltest ) call ffxhck(xpi,dpipj,6,ier)
*  #] check input:
*  #[ copy if known:
*
	if ( idot.ge.3 ) then
	    do 2 i=1,6
	    	do 1 j=1,6
	    	    piDpj(inew(j,irota3),inew(i,irota3)) = fpij3(j,i)
    1	    	continue
    2	    continue
	    if ( irota3 .gt. 3 ) then
*
*		the sign of the s's has been changed!
*
		do 4 i=1,3
		    do 3 j=4,6
			piDpj(j,i) = -piDpj(j,i)
			piDpj(i,j) = -piDpj(i,j)
    3		    continue
    4		continue
	    endif
	    return
	endif
*
*  #] copy if known:
*  #[ calculations:
	ier1 = ier
	do 10 is1=1,3
	    is2 = is1 + 1
	    if ( is2 .eq. 4 ) is2 = 1
	    is3 = is2 + 1
	    if ( is3 .eq. 4 ) is3 = 1
	    ip1 = is1 + 3
	    ip2 = is2 + 3
	    ip3 = is3 + 3
*
*	    pi.pj, si.sj
*
	    piDpj(is1,is1) = xpi(is1)
	    piDpj(ip1,ip1) = xpi(ip1)
*
*	    si.s(i+1)
*
	    if ( xpi(is2) .le. xpi(is1) ) then
		piDpj(is1,is2) = (dpipj(is1,ip1) + xpi(is2))/2
	    else
		piDpj(is1,is2) = (dpipj(is2,ip1) + xpi(is1))/2
	    endif
	    if ( lwarn ) then
		ier0 = ier
		if ( abs(piDpj(is1,is2)) .lt. xloss*min(xpi(is1),
     +			xpi(is2))/2 ) call ffwarn(100,ier0,piDpj(is1,
     +			is2),min(xpi(is1),xpi(is2))/2)
		ier1 = max(ier1,ier0)
	    endif
	    piDpj(is2,is1) = piDpj(is1,is2)
*
*	    pi.si
*
	    if ( abs(xpi(ip1)) .le. xpi(is1) ) then
		piDpj(ip1,is1) = (dpipj(is2,is1) - xpi(ip1))/2
	    else
		piDpj(ip1,is1) = (dpipj(is2,ip1) - xpi(is1))/2
	    endif
	    piDpj(is1,ip1) = piDpj(ip1,is1)
	    if ( lwarn ) then
		ier0 = ier
		if ( abs(piDpj(ip1,is1)) .lt. xloss*min(abs(xpi(ip1)),
     +			xpi(is1))/2) call ffwarn(101,ier0,
     +			piDpj(ip1,is1),min(abs(xpi(ip1)),xpi(is1))/2)
		ier1 = max(ier1,ier0)
	    endif
*
*	    pi.s(i+1)
*
	    if ( abs(xpi(ip1)) .le. xpi(is2) ) then
		piDpj(ip1,is2) = (dpipj(is2,is1) + xpi(ip1))/2
	    else
		piDpj(ip1,is2) = (dpipj(ip1,is1) + xpi(is2))/2
	    endif
	    piDpj(is2,ip1) = piDpj(ip1,is2)
	    if ( lwarn ) then
		ier0 = ier
		if ( abs(piDpj(ip1,is2)) .lt. xloss*min(abs(xpi(ip1)),
     +			xpi(is2))/2) call ffwarn(102,ier0,
     +			piDpj(ip1,is2),min(abs(xpi(ip1)),xpi(is2))/2)
		ier1 = max(ier1,ier0)
	    endif
*
*	    pi.s(i+2)
*
	    if ( min(abs(dpipj(is2,is1)),abs(dpipj(ip3,ip2))) .le.
     +		 min(abs(dpipj(ip3,is1)),abs(dpipj(is2,ip2))) ) then
		piDpj(ip1,is3) = (dpipj(ip3,ip2) + dpipj(is2,is1))/2
	    else
		piDpj(ip1,is3) = (dpipj(ip3,is1) + dpipj(is2,ip2))/2
	    endif
	    piDpj(is3,ip1) = piDpj(ip1,is3)
	    if ( lwarn ) then
		ier0 = ier
		if ( abs(piDpj(ip1,is3)) .lt. xloss*min(abs(dpipj(ip3,
     +			ip2)),abs(dpipj(ip3,is1)))/2 ) call ffwarn(103,
     +			ier0,piDpj(ip1,is3),min(abs(dpipj(ip3,ip2)),
     +			abs(dpipj(ip3,is1)))/2)
		ier1 = max(ier1,ier0)
	    endif
*
*	    pi.p(i+1)
*
	    if ( idot.le.0 ) then
	    	if ( abs(xpi(ip2)) .le. abs(xpi(ip1)) ) then
		    piDpj(ip1,ip2) = (dpipj(ip3,ip1) - xpi(ip2))/2
	    	else
		    piDpj(ip1,ip2) = (dpipj(ip3,ip2) - xpi(ip1))/2
	    	endif
	    	piDpj(ip2,ip1) = piDpj(ip1,ip2)
	    	if ( lwarn ) then
		    ier0 = ier
		    if ( abs(piDpj(ip1,ip2)) .lt.
     +			xloss*min(abs(xpi(ip1)),abs(xpi(ip2)))/2 ) call
     +			ffwarn(104,ier0,piDpj(ip1,ip2),
     +			min(abs(xpi(ip1)),abs(xpi(ip2)))/2)
		    ier1 = max(ier1,ier0)
	    	endif
	    else
	    	piDpj(inew(ip2,irota3),inew(ip1,irota3)) = 
     +	    		fpij3(ip1,ip2)
     		piDpj(inew(ip1,irota3),inew(ip2,irota3)) = 
     +     		piDpj(inew(ip2,irota3),inew(ip1,irota3))
	    endif
   10	continue
	ier = ier1
*
*  #] calculations:
*  #[ check:
	if ( ltest ) then
	    xlosn = xloss*DBLE(10)**(-2-mod(ier,50))
	    do 20 i = 1,6
		xheck = piDpj(i,4) + piDpj(i,5) + piDpj(i,6)
		if ( xlosn*abs(xheck) .gt. precx*max(abs(piDpj(i,4)),
     +			abs(piDpj(i,5)),abs(piDpj(i,6))) ) print *,
     +			'ffdot3: error: dotproducts with p(',i,
     +			') wrong: ',xheck,(piDpj(i,j),j=4,6)
   20	    continue
	endif
*  #] check:
*###] ffdot3:
	end
*###[ ffxc0r:
	subroutine ffxc0r(cc0,xpi,ier)
***#[*comment:***********************************************************
*									*
*	Tries all 2 permutations of the 3pointfunction			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ier
	DOUBLE PRECISION xpi(6),xqi(6)
	DOUBLE COMPLEX cc0,cc0p
	integer inew(6,2),irota,ier1,i,j,icon,ialsav,init
	logical lcon
	parameter (icon=3)
	save inew,init,lcon
	include 'ff.h'
	data inew /1,2,3,4,5,6,
     +		   1,3,2,6,5,4/
	data init /0/
*  #] declarations:
*  #[ open console for some activity on screen:
	if ( init .eq. 0 ) then
	    init = 1
	    if ( lwrite ) then
		open(icon,file='CON:',status='old',err=11)
		lcon = .TRUE.
		goto 13
	    endif
   11	    continue
	    lcon = .FALSE.
   13	    continue
	endif
*  #] open console for some activity on screen:
*  #[ calculations:
	cc0 = 0
	ier = 999
	ialsav = isgnal
	do 30 j = -1,1,2
	    do 20 irota=1,2
		do 10 i=1,6
		    xqi(inew(i,irota)) = xpi(i)
   10		continue
		print '(a,i1,a,i2)','---#[ rotation ',irota,': isgnal ',
     +			isgnal
		if (lcon) write(icon,'(a,i1,a,i2)')'rotation ',irota,',
     +			isgnal ',isgnal
		ier1 = 0
		ner = 0
		id = id + 1
		isgnal = ialsav
		call ffxc0(cc0p,xqi,ier1)
		ier1 = ier1 + ner
		print '(a,i1,a,i2)','---#] rotation ',irota,': isgnal ',
     +			isgnal
		print '(a,2g28.16,i3)','c0 = ',cc0p,ier1
		if (lcon) write(icon,'(a,2g28.16,i3)')'d0 = ',cc0p,ier1
		if ( ier1 .lt. ier ) then
		    cc0 = cc0p
		    ier = ier1
		endif
   20	    continue
	    ialsav = -ialsav
   30	continue
*  #] calculations:
*###] ffxc0r:
	end
