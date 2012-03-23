*###[ ffdel4:
	subroutine ffdel4(del4,xpi,piDpj,ns,ier)
***#[*comment:***********************************************************
*									*
*	Calculate del4(piDpj) = det(si.sj)	with			*
*	the momenta as follows:						*
*	p(1-4) = s(i)							*
*	p(4-10) = p(i)							*
*									*
*	Input:	xpi(ns)		(real)	m^2(i),i=1,3; p^2(i-3),i=4,10	*
*		piDpj(ns,ns)	(real)					*
*		ns		(integer)				*
*		ier		(integer)				*
*									*
*	Output:	del4		(real)	det(si.sj)			*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ier
	DOUBLE PRECISION del4,xpi(10),piDpj(10,10)
*
*	local variables:
*
	integer mem,nperm
	parameter(mem=10,nperm=125)
	integer i,jj(8),iperm(4,nperm),imem,jmem,memarr(mem,4),memind,
     +		inow,jnow,icount
	DOUBLE PRECISION s(24),xmax,del4p,xmaxp,rloss
	save iperm,memind,memarr,inow,jnow
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations: 
*  #[ data:
	data memind /0/
	data memarr /mem*0,mem*0,mem*1,mem*1/
	data inow /1/
	data jnow /1/
*
*	these are all permutations that give a non-zero result with the
*	correct sign.  This list was generated with getperm4.
*	(note: this used to be well-ordened, but then it had more than
*	19 continuation lines)
*
	data iperm/
     +	1,2,3,4,1,2,3,7,1,2,8,3,1,2,3,10,1,2,6,4,1,2,4,7,1,2,4,9,1,2,6,7
     +	,1,2,8,6,1,2,6,10,1,2,7,8,1,2,7,9,1,2,10,7,1,2,9,8,1,2,10,9,1,3,
     +	4,5,1,3,6,4,1,3,10,4,1,3,7,5,1,3,5,8,1,3,10,5,1,3,6,7,1,3,8,6,1,
     +	3,6,10,1,3,10,7,1,3,8,10,1,4,5,6,1,4,7,5,1,4,9,5,1,4,6,7,1,4,6,9
     +	,1,4,6,10,1,4,10,7,1,4,10,9,1,5,6,7,1,5,8,6,1,5,6,10,1,5,7,8,1,5
     +	,7,9,1,5,10,7,1,5,9,8,1,5,10,9,1,6,8,7,1,6,9,7,1,6,8,9,1,6,8,10,
     +	1,6,9,10,1,7,10,8,1,7,10,9,1,8,9,10,2,3,4,5,2,3,8,4,2,3,9,4,2,3,
     +	7,5,2,3,5,8,2,3,10,5,2,3,8,7,2,3,9,7,2,3,8,9,2,3,8,10,2,3,9,10,2
     +	,4,5,6,2,4,7,5,2,4,9,5,2,4,6,8,2,4,6,9,2,4,8,7,2,4,9,7,2,4,8,9,2
     +	,5,6,7,2,5,8,6,2,5,6,10,2,5,7,8,2,5,7,9,2,5,10,7,2,5,9,8,2,5,10,
     +	9,2,6,8,7,2,6,9,7,2,6,8,9,2,6,8,10,2,6,9,10,2,7,10,8,2,7,10,9,2,
     +	8,9,10,3,4,5,6,3,4,8,5,3,4,9,5,3,4,5,10,3,4,6,8,3,4,6,9,3,4,10,8
     +	,3,4,10,9,3,5,6,7,3,5,8,6,3,5,6,10,3,5,7,8,3,5,7,9,3,5,10,7,3,5,
     +	9,8,3,5,10,9,3,6,8,7,3,6,9,7,3,6,8,9,3,6,8,10,3,6,9,10,3,7,10,8,
     +	3,7,10,9,3,8,9,10,4,5,6,7,4,5,8,6,4,5,6,10,4,5,7,8,4,5,7,9,4,5,1
     +	0,7,4,5,9,8,4,5,10,9,4,6,8,7,4,6,9,7,4,6,8,9,4,6,8,10,4,6,9,10,4
     +	,7,10,8,4,7,10,9,4,8,9,10/
*  #] data: 
*  #[ check input:
	if ( ltest .and. ns .ne. 10 ) then
	    print *,'ffdel4: error: only for ns = 10, not ',ns
	    stop
	endif
*  #] check input: 
*  #[ get starting point from memory:
*
*	see if we know were to start, if not: go on as last time
*
	do 5 i=1,mem
	    if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) ) then
		inow = memarr(i,3)
		jnow = memarr(i,4)
		if ( lwrite ) print *,'ffcel4: from memory: ',id,idsub,
     +			inow,jnow
		goto 6
	    endif
    5	continue
    6	continue
*  #] get starting point from memory: 
*  #[ calculations:
	imem = inow
	jmem = jnow
	del4 = 0
	xmax = 0
	icount = 0

   10	continue

	jj(1) = iperm(1,inow)
	jj(3) = iperm(2,inow)
	jj(5) = iperm(3,inow)
	jj(7) = iperm(4,inow)

	jj(2) = iperm(1,jnow)
	jj(4) = iperm(2,jnow)
	jj(6) = iperm(3,jnow)
	jj(8) = iperm(4,jnow)

	s( 1) = +piDpj(jj(1),jj(2))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(7),jj(8))
	s( 2) = +piDpj(jj(1),jj(4))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(7),jj(8))
	s( 3) = +piDpj(jj(1),jj(6))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(7),jj(8))
	s( 4) = -piDpj(jj(1),jj(2))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(7),jj(8))
	s( 5) = -piDpj(jj(1),jj(6))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(7),jj(8))
	s( 6) = -piDpj(jj(1),jj(4))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(7),jj(8))

	s( 7) = -piDpj(jj(1),jj(2))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(7),jj(6))*piDpj(jj(5),jj(8))
	s( 8) = -piDpj(jj(1),jj(4))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(7),jj(2))*piDpj(jj(5),jj(8))
	s( 9) = -piDpj(jj(1),jj(6))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(7),jj(4))*piDpj(jj(5),jj(8))
	s(10) = +piDpj(jj(1),jj(2))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(7),jj(4))*piDpj(jj(5),jj(8))
	s(11) = +piDpj(jj(1),jj(6))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(7),jj(2))*piDpj(jj(5),jj(8))
	s(12) = +piDpj(jj(1),jj(4))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(7),jj(6))*piDpj(jj(5),jj(8))

	s(13) = -piDpj(jj(1),jj(2))*piDpj(jj(7),jj(4))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(3),jj(8))
	s(14) = -piDpj(jj(1),jj(4))*piDpj(jj(7),jj(6))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(3),jj(8))
	s(15) = -piDpj(jj(1),jj(6))*piDpj(jj(7),jj(2))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(3),jj(8))
	s(16) = +piDpj(jj(1),jj(2))*piDpj(jj(7),jj(6))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(3),jj(8))
	s(17) = +piDpj(jj(1),jj(6))*piDpj(jj(7),jj(4))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(3),jj(8))
	s(18) = +piDpj(jj(1),jj(4))*piDpj(jj(7),jj(2))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(3),jj(8))

	s(19) = -piDpj(jj(7),jj(2))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(1),jj(8))
	s(20) = -piDpj(jj(7),jj(4))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(1),jj(8))
	s(21) = -piDpj(jj(7),jj(6))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(1),jj(8))
	s(22) = +piDpj(jj(7),jj(2))*piDpj(jj(3),jj(6))*
     +		piDpj(jj(5),jj(4))*piDpj(jj(1),jj(8))
	s(23) = +piDpj(jj(7),jj(6))*piDpj(jj(3),jj(4))*
     +		piDpj(jj(5),jj(2))*piDpj(jj(1),jj(8))
	s(24) = +piDpj(jj(7),jj(4))*piDpj(jj(3),jj(2))*
     +		piDpj(jj(5),jj(6))*piDpj(jj(1),jj(8))

	del4p = 0
	xmaxp = 0
	do 20 i=1,24
	    del4p = del4p + s(i)
	    xmaxp = max(xmaxp,abs(s(i)))
   20	continue
	if ( abs(del4p) .lt. xloss*xmaxp ) then
	    if ( lwrite ) print *,'del4+',icount,' = ',del4p,xmaxp,inow,
     +		jnow
	    if ( inow .eq. imem .or. xmaxp .lt. xmax ) then
		del4 = del4p
		xmax = xmaxp
	    endif
*	    as the list is ordered we may have more luck stepping
*	    through with large steps
	    inow = inow + 43
	    jnow = jnow + 49
	    if ( inow .gt. nperm ) inow = inow - nperm
	    if ( jnow .gt. nperm ) jnow = jnow - nperm
	    icount = icount + 1
	    if ( icount.gt.15 .or. inow.eq.imem .or. jnow.eq.jmem
     +			) then
		if ( lwarn ) call ffwarn(143,ier,del4,xmax)
		goto 800
	    endif
	    goto 10
	endif
	if ( inow.ne.imem) then
	    if ( lwrite ) print *,'del4+',icount,' = ',del4p,xmaxp,inow,
     +		jnow
	endif
	del4 = del4p
	xmax = xmaxp
*  #] calculations: 
*  #[ into memory:
	if ( lwrite ) print *,'ffcel4: into memory: ',id,idsub,inow,jnow
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
	memarr(memind,4) = jnow
  800	continue
*  #] into memory: 
*  #[ check output:
	if ( ltest ) then
*
	    s( 1) = +piDpj(1,1)*piDpj(2,2)*piDpj(3,3)*piDpj(4,4)
	    s( 2) = +piDpj(1,2)*piDpj(2,3)*piDpj(3,1)*piDpj(4,4)
	    s( 3) = +piDpj(1,3)*piDpj(2,1)*piDpj(3,2)*piDpj(4,4)
	    s( 4) = -piDpj(1,1)*piDpj(2,3)*piDpj(3,2)*piDpj(4,4)
	    s( 5) = -piDpj(1,3)*piDpj(2,2)*piDpj(3,1)*piDpj(4,4)
	    s( 6) = -piDpj(1,2)*piDpj(2,1)*piDpj(3,3)*piDpj(4,4)
	    s( 7) = -piDpj(1,1)*piDpj(2,2)*piDpj(4,3)*piDpj(3,4)
	    s( 8) = -piDpj(1,2)*piDpj(2,3)*piDpj(4,1)*piDpj(3,4)
	    s( 9) = -piDpj(1,3)*piDpj(2,1)*piDpj(4,2)*piDpj(3,4)
	    s(10) = +piDpj(1,1)*piDpj(2,3)*piDpj(4,2)*piDpj(3,4)
	    s(11) = +piDpj(1,3)*piDpj(2,2)*piDpj(4,1)*piDpj(3,4)
	    s(12) = +piDpj(1,2)*piDpj(2,1)*piDpj(4,3)*piDpj(3,4)
	    s(13) = -piDpj(1,1)*piDpj(4,2)*piDpj(3,3)*piDpj(2,4)
	    s(14) = -piDpj(1,2)*piDpj(4,3)*piDpj(3,1)*piDpj(2,4)
	    s(15) = -piDpj(1,3)*piDpj(4,1)*piDpj(3,2)*piDpj(2,4)
	    s(16) = +piDpj(1,1)*piDpj(4,3)*piDpj(3,2)*piDpj(2,4)
	    s(17) = +piDpj(1,3)*piDpj(4,2)*piDpj(3,1)*piDpj(2,4)
	    s(18) = +piDpj(1,2)*piDpj(4,1)*piDpj(3,3)*piDpj(2,4)
	    s(19) = -piDpj(4,1)*piDpj(2,2)*piDpj(3,3)*piDpj(1,4)
	    s(20) = -piDpj(4,2)*piDpj(2,3)*piDpj(3,1)*piDpj(1,4)
	    s(21) = -piDpj(4,3)*piDpj(2,1)*piDpj(3,2)*piDpj(1,4)
	    s(22) = +piDpj(4,1)*piDpj(2,3)*piDpj(3,2)*piDpj(1,4)
	    s(23) = +piDpj(4,3)*piDpj(2,2)*piDpj(3,1)*piDpj(1,4)
	    s(24) = +piDpj(4,2)*piDpj(2,1)*piDpj(3,3)*piDpj(1,4)
*
	    del4p = 0
	    xmaxp = 0
	    do 820 i=1,24
		del4p = del4p + s(i)
		xmaxp = max(xmaxp,abs(s(i)))
  820	    continue
	    rloss = xloss*DBLE(10)**(-mod(ier,50)-1)
	    if ( rloss*abs(del4p-del4) .gt. precx*xmaxp ) then
		print *,'ffdel4: error: result does not agree with',
     +			' normal case'
		print *,'result: ',del4,xmax
		print *,'normal: ',del4p,xmaxp
		print *,'diff.:  ',del4-del4p,ier
	    endif
	endif
*  #] check output: 
*###] ffdel4: 
	end
*###[ ffdl3p:
	subroutine ffdl3p(dl3p,piDpj,ns,ii,jj,ier)
***#[*comment:***********************************************************
*	calculate in a numerically stable way				*
*									*
*	     p1  p2  p3							*
*	delta								*
*	     p1' p2' p3'						*
*									*
*	with pn = xpi(ii(n)), p4 = -p1-p2-p3, p5 = -p1-p2, p6 = p2+p3	*
*	with pn'= xpi(jj(n)), p4'= etc.       (when ns=15 p5=p1+p2)	*
*									*
*	Input:	piDpj	real(ns,ns)	dotpruducts			*
*		ns	integer		either 10 or 15			*
*		ii,jj	integer(6)	location of pi in piDpj		*
*		ier	integer		number of digits lost so far	*
*	Output:	dl3p	real		see above			*
*		ier	integer		number of digits lost so far	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ii(6),jj(6),ier
	DOUBLE PRECISION dl3p,piDpj(ns,ns)
*
*	local variables
*
	integer i,j,k,l,iperm(3,16),ii1,ii2,ii3,jj1,jj2,jj3,i0
	logical lsymm
	DOUBLE PRECISION s(6),som,xmax,smax,xheck,xlosn,trylos
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data iperm /1,2,3, 2,4,3, 3,4,1, 4,2,1,
     +		    1,2,6, 6,4,3, 3,1,6, 2,4,6,
     +		    2,5,3, 5,4,1, 1,3,5, 2,4,5,
     +		    1,6,5, 2,5,6, 3,6,5, 4,5,6/
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffdl3p: indices are'
	    print *,ii
	    print *,jj
	endif
	if ( ltest ) then
	    if ( ns .ne. 10 .and. ns .ne. 15 ) print *,'ffdl3p: error:',
     +		' only tested for ns=10,15'
	    xlosn = xloss**2*10.d0**(-mod(ier,50))
	    do 10 i=1,ns
		xheck = +piDpj(i,ii(1))+piDpj(i,ii(2))
     +			+piDpj(i,ii(3))+piDpj(i,ii(4))
		xmax = max(abs(piDpj(i,ii(1))),abs(piDpj(i,ii(2))),
     +			   abs(piDpj(i,ii(3))),abs(piDpj(i,ii(4))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta i1234 do not add to 0:',
     +			i,piDpj(i,ii(1)),piDpj(i,ii(2)),piDpj(i,ii(3)),
     +			piDpj(i,ii(4)),xheck,ier
		xheck = piDpj(i,ii(6))-piDpj(i,ii(2))-piDpj(i,ii(3))
		xmax = max(abs(piDpj(i,ii(6))),abs(piDpj(i,ii(2))),
     +			   abs(piDpj(i,ii(3))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta i623 do not add to 0:',
     +			i,piDpj(i,ii(6)),piDpj(i,ii(2)),piDpj(i,ii(3)),
     +			xheck,ier
		if ( ns .eq. 10 ) then
		    xheck = piDpj(i,ii(5))+piDpj(i,ii(1))+piDpj(i,ii(2))
		else
		    xheck = piDpj(i,ii(5))-piDpj(i,ii(1))-piDpj(i,ii(2))
		endif
		xmax = max(abs(piDpj(i,ii(5))),abs(piDpj(i,ii(1))),
     +			   abs(piDpj(i,ii(2))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta i512 do not add to 0:',
     +			i,piDpj(i,ii(5)),piDpj(i,ii(1)),piDpj(i,ii(2)),
     +			xheck,ier
		xheck = +piDpj(i,jj(1))+piDpj(i,jj(2))
     +			+piDpj(i,jj(3))+piDpj(i,jj(4))
		xmax = max(abs(piDpj(i,jj(1))),abs(piDpj(i,jj(2))),
     +			   abs(piDpj(i,jj(3))),abs(piDpj(i,jj(4))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta j1234 do not add to 0:',
     +			i,piDpj(i,jj(1)),piDpj(i,jj(2)),piDpj(i,jj(3)),
     +			piDpj(i,jj(4)),xheck,ier
		xheck = piDpj(i,jj(6))-piDpj(i,jj(2))-piDpj(i,jj(3))
		xmax = max(abs(piDpj(i,jj(6))),abs(piDpj(i,jj(2))),
     +			   abs(piDpj(i,jj(3))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta j623 do not add to 0:',
     +			i,piDpj(i,jj(6)),piDpj(i,jj(2)),piDpj(i,jj(3)),
     +			xheck,ier
		if ( ns .eq. 10 ) then
		    xheck = piDpj(i,jj(5))+piDpj(i,jj(1))+piDpj(i,jj(2))
		else
		    xheck = piDpj(i,jj(5))-piDpj(i,jj(1))-piDpj(i,jj(2))
		endif
		xmax = max(abs(piDpj(i,jj(5))),abs(piDpj(i,jj(1))),
     +			   abs(piDpj(i,jj(2))))
		if ( xlosn*xheck .gt. precx*xmax ) print *,
     +			'ffdl3p: error: momenta j512 do not add to 0:',
     +			i,piDpj(i,jj(5)),piDpj(i,jj(1)),piDpj(i,jj(2)),
     +			xheck,ier
   10	    continue
	endif
*  #] check input:
*  #[ calculations:
	if ( ii(1).eq.jj(1) .and. ii(2).eq.jj(2) .and. ii(3).eq.jj(3) )
     +		then
*
*	    symmetric - fewer possibilities
*
	    lsymm = .TRUE.
	else
	    lsymm = .FALSE.
	endif
*
*	try all (8.5,16)*16 permutations
*
	xmax = 0
	trylos = 1
	do 101 l=1,16
	    if ( lsymm ) then
		i0 = l
	    else
		i0 = 1
	    endif
	    do 100 i=i0,16
		ii1 = ii(iperm(1,i))
		ii2 = ii(iperm(2,i))
		ii3 = ii(iperm(3,i))
		j = i+l-1
		if ( j .gt. 16 ) j=j-16
		jj1 = jj(iperm(1,j))
		jj2 = jj(iperm(2,j))
		jj3 = jj(iperm(3,j))
		s(1) = +piDpj(ii1,jj1)*piDpj(ii2,jj2)*piDpj(ii3,jj3)
		s(2) = +piDpj(ii2,jj1)*piDpj(ii3,jj2)*piDpj(ii1,jj3)
		s(3) = +piDpj(ii3,jj1)*piDpj(ii1,jj2)*piDpj(ii2,jj3)
		s(4) = -piDpj(ii1,jj1)*piDpj(ii3,jj2)*piDpj(ii2,jj3)
		s(5) = -piDpj(ii3,jj1)*piDpj(ii2,jj2)*piDpj(ii1,jj3)
		s(6) = -piDpj(ii2,jj1)*piDpj(ii1,jj2)*piDpj(ii3,jj3)
		som = 0
		smax = 0
		do 80 k=1,6
		    som = som + s(k)
		    smax = max(smax,abs(som))
   80		continue
		if ( ns .eq. 15 .and. (i.gt.8 .neqv. j.gt.8) )
     +		    som = -som
		if ( i .eq. 1 .or. smax .lt. xmax ) then
		    dl3p = som
		    xmax = smax
		endif
		if ( lwrite ) then
		    print *,'dl3p = +',i-1+16*(l-1),' = ',som,smax
		endif
		if ( abs(dl3p) .ge. xloss*smax ) goto 110
*		give up a bit more easily if I have tried many times
		if ( trylos*abs(dl3p) .ge. xloss*smax ) goto 109
		trylos = trylos*1.3
  100	    continue
  101	continue
  109	continue
	if ( lwarn ) call ffwarn(138,ier,dl3p,xmax)
  110	continue
*  #] calculations:
*###] ffdl3p:
	end
