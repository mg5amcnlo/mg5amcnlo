*###[ ffcel3:
	subroutine ffcel3(del3,xpi,piDpj,ns,ier)
***#[*comment:***********************************************************
*									*
*	Calculate del3(piDpj) = det(si.sj)	with			*
*	the momenta as follows:						*
*	p(1-3) = s(i)							*
*	p(4-6) = p(i)							*
*									*
*	Input:	xpi(ns)		(real)	m^2(i),i=1,3; p^2(i-3),i=4,10	*
*		piDpj(ns,ns)	(real)					*
*		ns		(integer)				*
*		ier		(integer)				*
*									*
*	Output:	del3		(real)	det(si.sj)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ier
	DOUBLE COMPLEX del3,xpi(6),piDpj(6,6)
*
*	local variables:
*
	integer mem,nperm
	parameter(mem=10,nperm=16)
	integer i,jj(6),iperm(3,nperm),imem,memarr(mem,3),memind,inow
	DOUBLE COMPLEX s(6),del3p,cc
	DOUBLE PRECISION xmax,xmaxp,absc,rloss
	save iperm,memind,memarr,inow
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations:
*  #[ data:
	data memind /0/
	data memarr /mem*0,mem*0,mem*1/
	data inow /1/
*
*	these are all permutations that give a non-zero result with the
*	correct sign.  This list was generated with getperm3.
*
	data iperm/
     +		1,2,3,  1,2,5,  1,6,2,  1,4,3,
     +		1,3,5,  1,4,5,  1,6,4,  1,5,6,
     +		2,4,3,  2,3,6,  2,4,5,  2,6,4,
     +		2,5,6,  3,4,5,  3,6,4,  3,5,6/
*  #] data:
*  #[ check input:
	if ( ltest .and. ns .ne. 6 ) then
	    print *,'ffcel3: error: only for ns = 6, not ',ns
	    stop
	endif
*  #] check input:
*  #[ starting point in memory?:
*
*	see if we know were to start, if not: go on as last time
*
	do 5 i=1,mem
	    if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) ) then
		inow = memarr(i,3)
		goto 6
	    endif
    5	continue
    6	continue
*  #] starting point in memory?:
*  #[ calculations:
	imem = inow
	del3 = 0
	xmax = 0

   10	continue

	jj(1) = iperm(1,inow)
	jj(3) = iperm(2,inow)
	jj(5) = iperm(3,inow)

	jj(2) = iperm(1,inow)
	jj(4) = iperm(2,inow)
	jj(6) = iperm(3,inow)

	s(1) = +piDpj(jj(1),jj(2))*piDpj(jj(3),jj(4))*piDpj(jj(5),jj(6))
	s(2) = +piDpj(jj(1),jj(4))*piDpj(jj(3),jj(6))*piDpj(jj(5),jj(2))
	s(3) = +piDpj(jj(1),jj(6))*piDpj(jj(3),jj(2))*piDpj(jj(5),jj(4))
	s(4) = -piDpj(jj(1),jj(2))*piDpj(jj(3),jj(6))*piDpj(jj(5),jj(4))
	s(5) = -piDpj(jj(1),jj(6))*piDpj(jj(3),jj(4))*piDpj(jj(5),jj(2))
	s(6) = -piDpj(jj(1),jj(4))*piDpj(jj(3),jj(2))*piDpj(jj(5),jj(6))

	del3p = 0
	xmaxp = 0
	do 20 i=1,6
	    del3p = del3p + s(i)
	    xmaxp = max(xmaxp,absc(s(i)))
   20	continue
	if ( absc(del3p) .lt. xloss*xmaxp ) then
	    if ( lwrite ) print *,'del3+',inow,' = ',del3p,xmaxp
	    if ( inow .eq. imem .or. xmaxp .lt. xmax ) then
		del3 = del3p
		xmax = xmaxp
	    endif
	    inow = inow + 1
	    if ( inow .gt. nperm ) inow = 1
	    if ( inow .eq. imem ) then
		if ( lwarn ) call ffwarn(72,ier,absc(del3),xmax)
		goto 800
	    endif
	    goto 10
	endif
	if ( inow .ne. imem ) then
	    if ( lwrite ) print *,'del3+',inow,' = ',del3p,xmaxp
	endif
	del3 = del3p
	xmax = xmaxp
*  #] calculations:
*  #[ into memory:
  800	continue
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
*  #] into memory:
*  #[ check output:
	if ( ltest ) then

	    s(1) = +piDpj(1,1)*piDpj(2,2)*piDpj(3,3)
	    s(2) = +piDpj(1,2)*piDpj(2,3)*piDpj(3,1)
	    s(3) = +piDpj(1,3)*piDpj(2,1)*piDpj(3,2)
	    s(4) = -piDpj(1,1)*piDpj(2,3)*piDpj(3,2)
	    s(5) = -piDpj(1,3)*piDpj(2,2)*piDpj(3,1)
	    s(6) = -piDpj(1,2)*piDpj(2,1)*piDpj(3,3)

	    del3p = 0
	    xmaxp = 0
	    do 820 i=1,6
		del3p = del3p + s(i)
		xmaxp = max(xmaxp,absc(s(i)))
  820	    continue
	    cc = del3p-del3
	    rloss = xloss*DBLE(10)**(-mod(ier,50))
	    if ( rloss*absc(cc) .gt. precc*xmaxp ) then
		print *,'ffcel3: error: result does not agree with',
     +			' normal case'
		print *,'result: ',del3,xmax
		print *,'normal: ',del3p,xmaxp
		print *,'diff.:  ',del3-del3p
	    endif
	endif
*  #] check output:
*###] ffcel3:
	end
*(##[ ffcl3s:
	subroutine ffcl3s(dl3s,xpi,piDpj,ii,ns,ier)
***#[*comment:***********************************************************
*									*
*	Calculate dl3s(piDpj) = det(si.sj)	with			*
*	the momenta indicated by the indices ii(1-6,1), ii(1-6,2)	*
*	as follows:							*
*	p(|ii(1,)|-|ii(3,)|) = s(i)					*
*	p(|ii(4,)|-|ii(6,)|) = p(i) = sgn(ii())*(s(i+1) - s(i))		*
*									*
*	At this moment (26-apr-1990) only the diagonal is tried		*
*									*
*	Input:	xpi(ns)		(real)	m^2(i),i=1,3; p^2(i-3),i=4,10	*
*		piDpj(ns,ns)	(real)					*
*		ii(6,2)		(integer)	see above		*
*		ns		(integer)				*
*		ier		(integer)				*
*									*
*	Output:	dl3s		(real)	det(si.sj)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ii(6,2),ns,ier
	DOUBLE COMPLEX dl3s,xpi(ns),piDpj(ns,ns)
*
*	local variables:
*
	integer mem,nperm
	parameter(mem=10,nperm=16)
	integer i,j,jj(6),jsgn,iperm(3,nperm),imem,memarr(mem,3),
     +		memind,inow
	DOUBLE PRECISION xmax,xmaxp,xlosn,absc,rloss
	DOUBLE COMPLEX s(6),dl3sp,xhck,cc
	save iperm,memind,memarr,inow
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ data:
*
	data memind /0/
	data memarr /mem*0,mem*0,mem*1/
	data inow /1/
*
*	these are all permutations that give a non-zero result with the
*	correct sign.  This list was generated with getperm3.
*
	data iperm/
     +		1,2,3,  1,2,5,  1,6,2,  1,4,3,
     +		1,3,5,  1,4,5,  1,6,4,  1,5,6,
     +		2,4,3,  2,3,6,  2,4,5,  2,6,4,
     +		2,5,6,  3,4,5,  3,6,4,  3,5,6/
*  #] data:
*  #[ test input:
	if ( ltest ) then
	    if ( lwrite ) then
		print *,'ffcl3s: input: ii(,1) = ',(ii(i,1),i=1,6)
		print *,'               ii(,2) = ',(ii(i,2),i=1,6)
	    endif
	    xlosn = xloss*DBLE(10)**(-mod(ier,50)-1)
	    do 3 j=1,2
	    do 1 i=1,6
		if ( abs(ii(i,j)) .gt. ns ) print *,'ffcl3s: error: ',
     +			'|ii(i,j)| > ns: ',ii(i,j),ns
		if ( abs(ii(i,j)) .eq. 0 ) print *,'ffcl3s: error: ',
     +			'|ii(i,j)| = 0: ',ii(i,j)
    1	    continue
	    do 2 i=1,6

		xhck = piDpj(abs(ii(i,j)),ii(1,j))
     +		     - piDpj(abs(ii(i,j)),ii(2,j))
     +		     + sign(1,ii(4,j))*piDpj(abs(ii(i,j)),abs(ii(4,j)))
		xmax = max(absc(piDpj(abs(ii(i,j)),ii(1,j))),
     +			   absc(piDpj(abs(ii(i,j)),ii(2,j))))
		if ( xlosn*absc(xhck).gt.precc*xmax ) print *,'ffcl3s:'
     +		  ,' error: dotproducts 124 with ',i,' do not add to 0:'
     +		  ,piDpj(abs(ii(i,j)),ii(1,j)),
     +		  piDpj(abs(ii(i,j)),ii(2,j)),
     +		  piDpj(abs(ii(i,j)),abs(ii(4,j))),xhck

		xhck = piDpj(abs(ii(i,j)),ii(2,j))
     +		     - piDpj(abs(ii(i,j)),ii(3,j))
     +		     +sign(1,ii(5,j))*piDpj(abs(ii(i,j)),abs(ii(5,j)))
		xmax = max(absc(piDpj(abs(ii(i,j)),ii(2,j))),
     +			absc(piDpj(abs(ii(i,j)),ii(3,j))))
		if ( xlosn*absc(xhck).gt.precc*xmax ) print *,'ffcl3s:'
     +		  ,' error: dotproducts 235 with ',i,' do not add to 0:'
     +		  ,piDpj(abs(ii(i,j)),ii(2,j)),
     +		  piDpj(abs(ii(i,j)),ii(3,j)),
     +		  piDpj(abs(ii(i,j)),abs(ii(5,j))),xhck

		xhck = piDpj(abs(ii(i,j)),ii(3,j))
     +		     - piDpj(abs(ii(i,j)),ii(1,j))
     +		     + sign(1,ii(6,j))*piDpj(abs(ii(i,j)),abs(ii(6,j)))
		xmax = max(absc(piDpj(abs(ii(i,j)),ii(3,j))),
     +			absc(piDpj(abs(ii(i,j)),ii(1,j))))
		if ( xlosn*absc(xhck).gt.precc*xmax ) print *,'ffcl3s:'
     +		  ,' error: dotproducts 316 with ',i,' do not add to 0:'
     +		  ,piDpj(abs(ii(i,j)),ii(3,j)),
     +		  piDpj(abs(ii(i,j)),ii(1,j)),
     +		  piDpj(abs(ii(i,j)),abs(ii(6,j))),xhck

		xhck = sign(1,ii(4,j))*piDpj(abs(ii(i,j)),abs(ii(4,j)))
     +		     + sign(1,ii(5,j))*piDpj(abs(ii(i,j)),abs(ii(5,j)))
     +		     + sign(1,ii(6,j))*piDpj(abs(ii(i,j)),abs(ii(6,j)))
		xmax = max(absc(piDpj(abs(ii(i,j)),abs(ii(4,j)))),
     +			   absc(piDpj(abs(ii(i,j)),abs(ii(5,j)))))
		if ( xlosn*absc(xhck).gt.precc*xmax ) print *,'ffcl3s:'
     +		  ,' error: dotproducts 456 with ',i,' do not add to 0:'
     +		  ,piDpj(abs(ii(i,j)),abs(ii(4,j))),
     +		  piDpj(abs(ii(i,j)),abs(ii(5,j))),
     +		  piDpj(abs(ii(i,j)),abs(ii(6,j))),xhck

    2	    continue
    3	    continue
	    do 4 i=1,ns
	    	xhck = piDpj(i,i) - xpi(i)
	    	xmax = abs(xpi(i))
		if ( xlosn*absc(xhck).gt.precc*xmax ) print *,'ffcl3s:'
     +		  ,' error: xpi(',i,') != piDpj(',i,i,') :',xpi(i),
     +		  piDpj(i,i),xhck
    4	    continue
	endif
*  #] test input:
*  #[ starting point in memory?:
*
*	see if we know were to start, if not: go on as last time
*
	do 5 i=1,mem
	    if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) ) then
		inow = memarr(i,3)
		goto 6
	    endif
    5	continue
    6	continue
*  #] starting point in memory?:
*  #[ calculations:
	imem = inow
	dl3s = 0
	xmax = 0

   10	continue

	jj(1) = abs(ii(iperm(1,inow),1))
	jj(3) = abs(ii(iperm(2,inow),1))
	jj(5) = abs(ii(iperm(3,inow),1))

	jj(2) = abs(ii(iperm(1,inow),2))
	jj(4) = abs(ii(iperm(2,inow),2))
	jj(6) = abs(ii(iperm(3,inow),2))

	jsgn =  sign(1,ii(iperm(1,inow),1))
     +		*sign(1,ii(iperm(2,inow),1))
     +		*sign(1,ii(iperm(3,inow),1))
     +		*sign(1,ii(iperm(1,inow),2))
     +		*sign(1,ii(iperm(2,inow),2))
     +		*sign(1,ii(iperm(3,inow),2))

	s(1) = +piDpj(jj(1),jj(2))*piDpj(jj(3),jj(4))*piDpj(jj(5),jj(6))
	s(2) = +piDpj(jj(1),jj(4))*piDpj(jj(3),jj(6))*piDpj(jj(5),jj(2))
	s(3) = +piDpj(jj(1),jj(6))*piDpj(jj(3),jj(2))*piDpj(jj(5),jj(4))
	s(4) = -piDpj(jj(1),jj(2))*piDpj(jj(3),jj(6))*piDpj(jj(5),jj(4))
	s(5) = -piDpj(jj(1),jj(6))*piDpj(jj(3),jj(4))*piDpj(jj(5),jj(2))
	s(6) = -piDpj(jj(1),jj(4))*piDpj(jj(3),jj(2))*piDpj(jj(5),jj(6))

	dl3sp = 0
	xmaxp = 0
	do 20 i=1,6
	    dl3sp = dl3sp + s(i)
	    xmaxp = max(xmaxp,absc(s(i)))
   20	continue
	if ( absc(dl3sp) .lt. xloss*xmaxp ) then
	    if ( lwrite ) print *,'dl3s+',inow,' = ',dl3sp,xmaxp
	    if ( inow .eq. imem .or. xmaxp .lt. xmax ) then
		dl3s = jsgn*dl3sp
		xmax = xmaxp
	    endif
	    inow = inow + 1
	    if ( inow .gt. nperm ) inow = 1
	    if ( inow .eq. imem ) then
		if ( lwarn ) call ffwarn(85,ier,absc(dl3s),xmax)
		goto 800
	    endif
	    goto 10
	endif
	if ( inow .ne. imem ) then
	    if ( lwrite ) print *,'dl3s+',inow,' = ',dl3sp,xmaxp
	endif
	dl3s = jsgn*dl3sp
	xmax = xmaxp
*  #] calculations:
*  #[ into memory:
  800	continue
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
*  #] into memory:
*  #[ check output:
	if ( ltest ) then

	    s(1) = +piDpj(ii(1,1),ii(1,2))*piDpj(ii(2,1),ii(2,2))*
     +						piDpj(ii(3,1),ii(3,2))
	    s(2) = +piDpj(ii(1,1),ii(2,2))*piDpj(ii(2,1),ii(3,2))*
     +						piDpj(ii(3,1),ii(1,2))
	    s(3) = +piDpj(ii(1,1),ii(3,2))*piDpj(ii(3,1),ii(2,2))*
     +						piDpj(ii(2,1),ii(1,2))
	    s(4) = -piDpj(ii(1,1),ii(1,2))*piDpj(ii(2,1),ii(3,2))*
     +						piDpj(ii(3,1),ii(2,2))
	    s(5) = -piDpj(ii(1,1),ii(3,2))*piDpj(ii(2,1),ii(2,2))*
     +						piDpj(ii(3,1),ii(1,2))
	    s(6) = -piDpj(ii(1,1),ii(2,2))*piDpj(ii(2,1),ii(1,2))*
     +						piDpj(ii(3,1),ii(3,2))

	    dl3sp = 0
	    xmaxp = 0
	    do 820 i=1,6
		dl3sp = dl3sp + s(i)
		xmaxp = max(xmaxp,absc(s(i)))
  820	    continue
	    rloss = xloss*DBLE(10)**(-mod(ier,50))
	    if ( rloss*absc(dl3sp-dl3s) .gt. precc*xmaxp ) then
		print *,'ffcl3s: error: result does not agree with',
     +			' normal case'
		print *,'result: ',dl3s,xmax
		print *,'normal: ',dl3sp,xmaxp
		print *,'diff.:  ',dl3s-dl3sp
	    endif
	endif
*  #] check output:
*)##] ffcl3s:
	end
