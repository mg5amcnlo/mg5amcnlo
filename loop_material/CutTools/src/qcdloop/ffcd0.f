*	$Id: ffcd0.f,v 1.3 1995/12/08 10:50:35 gj Exp $
*###[ ffcd0:
	subroutine ffcd0(cd0,cpi,ier)
***#[*comment:***********************************************************
*									*
*		         1   /						*
*	calculate d0 = ----- \dq [(q^2 + 2*s_1.q)*(q^2 + 2*s2.q)	*
*		       ipi^2 /      *(q^2 + 2*s3.q)*(q^2 + 2*s4.q)]^-1	*
*									*
*	      |p9							*
*	\p8   V	   p7/							*
*	 \	    /							*
*	  \________/							*
*	  |   m4   |							*
*    =	  |	   |	/____						*
*	m1|	   |m3	\ p10						*
*	  |	   |		all momenta are incoming		*
*	  |________|							*
*	  /  m2	   \							*
*	 /	    \							*
*	/p5	   p6\							*
*									*
*									*
*	input:	cpi(1-10)   (complex) 1-4: m_i^2, 5-10 p_i^2 (B&D metric)
*		cpi(11)=u   (complex)  u=p5.p5+..-p9.p9-p10.10 or 0	*
*		cpi(12)=v   (complex)  v=-p5.p5+p6.p6-p7.p7+.. or 0	*
*		cpi(13)=w   (complex)  w=p5.p5-p6.p6+p7.p7-p8.p8+..	*
*	output:	cd0	    (complex)  D0				*
*		ier	    (integer)  <50: #digits lost, >100: error	*
*	calls:	ffcd0a,ffxd0						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	DOUBLE COMPLEX cd0,cpi(13)
	integer ier
*
*	local variables
*
	DOUBLE COMPLEX c,cs,cfac
	integer i,j,ier0,init
	logical luvw(3)
	DOUBLE PRECISION absc,absr,xpi(13),xmax,sprec
	DOUBLE COMPLEX cpipj(10,13)
	save init
*
*	common blocks:
*
	include 'ff.h'
*
*	statement functions:
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
	absr(c) = abs(DBLE(c))
*
*	data
*
	data init /0/
*  #] declarations:
*  #[ the real case:
*
	if ( nschem.ge.2 ) then
	    if ( DIMAG(cpi(1)).ne.0 .or. DIMAG(cpi(2)).ne.0 .or.
     +		 DIMAG(cpi(3)).ne.0 .or. DIMAG(cpi(4)).ne.0 ) goto 40
	elseif ( init .eq. 0 ) then
	    init = 1
	    print *,'ffcd0: disregarding complex masses, nschem= ',
     +			nschem
	endif
	do 10 i=1,13
	    xpi(i) = DBLE(cpi(i))
   10	continue
	sprec = precx
	precx = precc
	call ffxd0(cd0,xpi,ier)
	if ( ldot ) then
	    ier0 = 0
	    call ffcif4(cpi,luvw,cpipj,ier0)
	    call ffcod4(cpi,cpipj)
	    if ( luvw(1) ) cpi(11) = 0
	    if ( luvw(2) ) cpi(12) = 0
	    if ( luvw(3) ) cpi(13) = 0
	endif
	precx = sprec
	return
   40	continue
*
*  #] the real case:
*  #[ check input:
*
	idsub = 0
	if ( ltest ) then
	    do 50 i=1,4
		if ( DIMAG(cpi(i)) .gt. 0 ) call fferr(50,ier)
   50	    continue
	    do 60 i=5,13
		if ( DIMAG(cpi(i)) .ne. 0 ) call fferr(50,ier)
   60	    continue
	endif
	if ( lwrite ) then
	    print *,'ffcd0: input = ',cpi
	endif
*
*  #] check input:
*  #[ call ffcif4,ffcd0a:
*
	call ffcif4(cpipj,luvw,cpi,ier)
	call ffcd0b(cs,cfac,cpi,cpipj,0,ier)
	cd0 = cs*cfac
*
*	restore the zeros for u,v,w as we have calculated them
*	ourselves and the user is unlikely to do this...
*
	if ( luvw(1) ) cpi(11) = 0
	if ( luvw(2) ) cpi(12) = 0
	if ( luvw(3) ) cpi(13) = 0
*
*  #] call ffcif4,ffcd0a:
*###] ffcd0:
	end
*###[ ffcd0a:
	subroutine ffcd0a(cd0,cpi,cpipj,ier)
***#[*comment:***********************************************************
*									*
*	Dummy routine.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cd0,cpi(13),cpipj(10,13)
*
*	local variables
*
	DOUBLE COMPLEX cs,cfac
*
*  #] declarations:
*  #[ call ffcd0b:
*
	call ffcd0b(cs,cfac,cpi,cpipj,0,ier)
	cd0 = cs*cfac
*
*  #] call ffcd0b:
*###] ffcd0a:
	end
*###[ ffcd0b:
	subroutine ffcd0b(cs,cfac,cpi,cpipj,ndiv,ier)
***#[*comment:***********************************************************
*									*
*	See ffcd0, the differences between the input parameters are	*
*	also input.  This routines has the big nschem switchyard.	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,ndiv
	DOUBLE COMPLEX cs,cfac,cpi(13),cpipj(10,13)
*
*	local variables
*
	integer ier0,i,j,initlo,iir(2,4),ithres(4,4)
	logical ldone
	DOUBLE PRECISION xpi(13),dpipj(10,13),sprec,absc
	DOUBLE COMPLEX cc
	save initlo
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
*	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data initlo /0/
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    ier0 = 0
	    call ffchck(cpi,cpipj,10,ier0)
	    if ( ier0 .ne. 0 ) print *,'called from ffcd0b'
	endif
*  #] check input:
*  #[ the real case:
*
	if ( DIMAG(cpi(1)).eq.0 .and. DIMAG(cpi(2)).eq.0 .and.
     +	     DIMAG(cpi(3)).eq.0 .and. DIMAG(cpi(4)).eq.0 .or.
     +	     nschem.le.1 ) then
	    if ( initlo.eq.0 .and. nschem.le.1 ) then
		initlo = 1
		print *,'ffcd0b: disregarding all complex masses'
	    endif
	    if ( onshel .and. ndiv.gt.0 ) then
		cs = 0
		cfac = 1
		return
	    endif
	    do 5 i=1,13
		xpi(i) = DBLE(cpi(i))
		do 4 j=1,10
		    dpipj(j,i) = DBLE(cpipj(j,i))
    4		continue
    5	    continue
	    sprec = precx
	    precx = precc
	    call ffxd0b(cs,cfac,xpi,dpipj,ndiv,ier)
	    if ( ldot ) call ffcod4(cpi,cpipj)
	    precx = sprec
	    return
	endif
*
*  #] the real case:
*  #[ handle poles-only approach:
	if (  nschem.le.6 ) then
	    if ( initlo .eq. 0 ) then
		initlo = 1
		if ( nschem.eq.2 ) then
		    print *,'ffcd0b: disregarding complex masses ',
     +			'except in linearly divergent terms'
		elseif ( nschem.eq.3 ) then
		    print *,'ffcd0b: undefined nschem=3'
		elseif ( nschem.eq.4 ) then
		    print *,'ffcd0b: using the scheme in which ',
     +			'complex masses are used everywhere when ',
     +			'there is a divergent log'
		elseif ( nschem.eq.5 ) then
		    print *,'ffcd0b: using the scheme in which ',
     +			'complex masses are used everywhere when ',
     +			'there is a divergent or almost divergent log'
		elseif ( nschem.eq.6 ) then
		    print *,'ffcd0b: using the scheme in which ',
     +			'complex masses are used everywhere when ',
     +			'there is a singular log'
		elseif ( nschem.eq.7 ) then
		    print *,'ffcd0b: using complex masses'
		endif
		if ( nschem.ge.3 ) then
		    print *,'ffcd0b: switching to complex when on',
     +			'shell or |p^2-Re(m^2)| < ',nwidth,'*|Im(m^2)|'
		endif
	    endif
*
*	    ffcdir computes all linearly onshell singular cases,
*	    returns 0 if ndiv too large
*	    and returns the other IR divergences in iir as a bonus
*
	    call ffcdir(cs,cfac,ldone,iir,cpi,cpipj,4,ndiv,ier)
	    if ( ldone ) return
*
*	    use a subtraction method, or just the real case.
*	    for both we need the real vars.
*
	    sprec = precx
	    precx = precc
	    do 16 i=1,13
		xpi(i) = DBLE(cpi(i))
		do 15 j=1,10
		    dpipj(j,i) = DBLE(cpipj(j,i))
   15		continue
   16	    continue
	    if ( nschem.le.3 .or. iir(1,1).eq.0 .and. nschem.eq.4 ) then
	    else
*
*	    finally, search for threshold terms
*
		do 20 i=1,3
		    do 19 j=i+1,4
			call ffthre(ithres(j,i),cpi,cpipj,10,i,j,
     +				inx(i,j))
			ithres(i,j) = ithres(j,i)
   19		    continue
   20		continue
	    endif
	    call ffcd0c(cs,cfac,cpi,cpipj,xpi,dpipj,iir,ithres,ier)
	    precx = sprec
*
*  #] handle poles-only approach:
*  #[ complex case:
	else
	    print *,'ffcd0b: complex D0 not implemented'
	    stop
	endif
*  #] complex case:
*###] ffcd0b:
	end
*###[ ffcd0c:
	subroutine ffcd0c(cs,cfac,cpi,cpipj,xpi,dpipj,iir,ithres,ier)
***#[*comment:***********************************************************
*									*
*	computes the complex D0 by adding and subtracting the complex	*
*	C0 (for ir divergences in iir) and B0 (for threshold effects in	*
*	ithres).							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer iir(2,4),ithres(4,4),ier
	DOUBLE PRECISION xpi(13),dpipj(10,13)
	DOUBLE COMPLEX cs,cfac,cpi(13),cpipj(10,13)
*
*	local variables
*
	integer ir,i,i1,j,j1,ier1,ier0,nscsav,ij,notij(4,4),
     +		notijk(4,4,4),k,l
	logical ldotsa,lwrisa
	DOUBLE PRECISION xpi3(6),dpipj3(6,6),absc,a(2),p1,p2,xmax,del2
	DOUBLE COMPLEX csr,cc0r,cc0c,cb0r,cb0c,cpi3(6),cpipj3(6,6),cc
	save notij,notijk
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data notij/0,3,2,2, 3,0,1,1, 2,1,0,1, 2,1,1,0/
	data notijk/
     +	0,0,0,0,0,0,4,3,0,4,0,2,0,3,2,0,0,0,4,3,0,0,0,0,4,0,0,1,3,0,1,0,
     +	0,4,0,2,4,0,0,1,0,0,0,0,2,1,0,0,0,3,2,0,3,0,1,0,2,1,0,0,0,0,0,0/
*
*  #] declarations:
*  #[ D0:
*
	if ( lwrite ) print *,'ffcd0c: calling ffxd0b'
	ier1 = ier
	ldotsa = ldot
	ldot = .TRUE.
	lwrisa = lwrite
*	lwrite = .FALSE.
	call ffxd0b(csr,cfac,xpi,dpipj,0,ier1)
	lwrite = lwrisa
	cs = csr
*
*  #] D0:
*  #[ bookkeeping:
	if ( nschem.eq.5 .or. nschem.eq.6 ) then
*
*	    first weed out the thresholds already included in the IR terms
*
	    if ( lwrite ) then
	    	print '(a)','ffcd0c: before comp iir ithres was '
	    	print '(4i3)',ithres
	    endif
	    do 15 i=1,2
		if ( iir(i,1).eq.0 ) goto 16
		ithres(iir(i,1),iir(i,2)) = 0
		ithres(iir(i,2),iir(i,1)) = 0
		ithres(iir(i,1),iir(i,3)) = 0
		ithres(iir(i,3),iir(i,1)) = 0
   15	    continue
   16	    continue
	    if ( lwrite ) then
	    	print '(a)','ffcd0c: after comp iir ithres is '
	    	print '(4i3)',ithres
	    endif
*
*	    next - we need a complete complex C0 for a (m,m,0) type
*	    vertex; the B0 does not contain the 2i\pi^2 jump
*
	    ir = 1
	    if ( iir(1,1) .ne. 0 ) ir = 2
	    if ( iir(2,1) .ne. 0 ) goto 19
	    do 18 i=1,3
		do 17 j=i+1,4
		    if ( ithres(i,j).eq.2 ) then
			if ( xpi(i).lt.xpi(j) ) then
			    iir(ir,1) = i
			    iir(ir,2) = j
			else
			    iir(ir,1) = j
			    iir(ir,2) = i
			endif
			k = notij(i,j)
			l = notijk(i,j,k)
			if ( abs(fpij4(iir(ir,1),k)) .lt.
     +			     abs(fpij4(iir(ir,1),l)) ) then
			    iir(ir,3) = k
			    iir(ir,4) = l
			else
			    iir(ir,3) = l
			    iir(ir,4) = k
			endif
*			throw out the vertices connected with this C0
			ithres(i,j) = 0
			ithres(j,i) = 0
			ithres(iir(ir,1),iir(ir,3)) = 0
			ithres(iir(ir,3),iir(ir,1)) = 0
			if ( lwrite ) then
			    print *,'ffcd0c: thresold of (m,m,0) type'
			    print *,xpi(i),xpi(j),xpi(inx(i,j))
			    print *,'made into C0 ',iir(ir,1),iir(ir,2),
     +				iir(ir,3),iir(ir,4)
			endif
			ir = ir + 1
		    endif
   17		continue
   18	    continue
   19	    continue
	endif
*
*  #] bookkeeping:
*  #[ IR:
*
*	get the IR parts correct
*
	if ( nschem.le.3 ) goto 31
	do 30 ir=1,2
	    if ( iir(ir,1).eq.0 ) goto 31
	    do 25 i=1,3
		i1 = mod(i,3) + 1
		xpi3(i)   = xpi(iir(ir,i))
		cpi3(i)   = cpi(iir(ir,i))
		xpi3(i+3) = xpi(inx(iir(ir,i),iir(ir,i1)))
		cpi3(i+3) = cpi(inx(iir(ir,i),iir(ir,i1)))
		do 24 j=1,3
		    j1 = mod(j,3) + 1
		    dpipj3(j,i) = dpipj(iir(ir,j),iir(ir,i))
		    cpipj3(j,i) = cpipj(iir(ir,j),iir(ir,i))
		    dpipj3(j,i+3) = dpipj(iir(ir,j),
     +				inx(iir(ir,i),iir(ir,i1)))
		    dpipj3(i+3,j) = -dpipj3(j,i+3)
		    cpipj3(j,i+3) = cpipj(iir(ir,j),
     +				inx(iir(ir,i),iir(ir,i1)))
		    cpipj3(i+3,j) = -cpipj3(j,i+3)
		    dpipj3(j+3,i+3) = dpipj(
     +				inx(iir(ir,j),iir(ir,j1)),
     +				inx(iir(ir,i),iir(ir,i1)))
		    cpipj3(j+3,i+3) = cpipj(
     +				inx(iir(ir,j),iir(ir,j1)),
     +				inx(iir(ir,i),iir(ir,i1)))
   24		continue
   25	    continue
	    ier0 = ier
	    if ( lwrite ) then
		print *,'ffcd0c: calling ffxc0a'
		print *,'xpi3 = ',xpi3
	    endif
	    ldot = .TRUE.
	    call ffxc0a(cc0r,xpi3,dpipj3,ier0)
	    ier1 = max(ier1,ier0)
	    del2 = fdel2
	    if ( lwrite ) then
		print *,'ffcd0c: calling ffcc0a'
		print *,'cpi3 = ',cpi3
	    endif
	    nscsav = nschem
	    nschem = 7
	    ldot = .FALSE.
	    call ffcc0a(cc0c,cpi3,cpipj3,ier0)
	    nschem = nscsav
	    ier1 = max(ier1,ier0)
	    if ( ltest .and. xloss*absc(cc0r-cc0c) .lt.
     +		precc*absc(cc0r) ) print *,'ffcd0c: ',
     +		'unnecessary subtraction!!',iir
	    p1 = 1/dpipj(inx(iir(ir,4),iir(ir,1)),iir(ir,4))
	    if ( lwrite ) then
		print *,'Compare propagator and improved factor'
		print *,'p1                       = ',p1
		print *,'sqrt(-del2/fdel4s)/2     = ',sqrt(-del2/fdel4s)
     +			/2
		print *,'sqrt(-del2)*DBLE(cfac)/2 = ',sqrt(-del2)*
     +			DBLE(cfac)*2
	    endif
*	    this can not cause problems because p1 flips sign when the
*	    function is linearly divergent, and that region should
*	    never come here
	    p1 = sign(sqrt(-del2/fdel4s)/2,p1)
	    cc0r=cc0r/cfac*DBLE(p1)
	    cc0c=cc0c/cfac*DBLE(p1)
	    cs = cs - cc0r + cc0c
	    if ( lwarn .and. absc(cs) .lt. xloss*max(absc(cc0r),
     +			absc(cc0c)) ) then
	      call ffwarn(211,ier1,absc(cs),max(absc(cc0r),absc(cc0c)))
	    endif
	    if ( lwrite ) then
	    	print *
		if ( ir.eq.1 ) print *,'csr  = ',csr
		print *,'subtracted IR divergence ',iir(ir,1),iir(ir,2),
     +			iir(ir,3)
		print *,'cc0r = ',cc0r
		print *,'cc0c = ',cc0c
		if ( ir.eq.2 .or. iir(2,1).eq.0 ) then
		    print *,'-------------'
		    print *,'cs   = ',cs
		endif
	    endif
   30	continue
   31	continue
*
*  #] IR:
*  #[ threshold:
*
*	and the threshold terms
*
	if ( nschem.le.4 ) goto 41
*
*	next add and subtract the complex/real B0
*
	do 40 i=1,3
	    do 39 j=i+1,4
		if ( ithres(j,i).eq.0 ) goto 39
		ij = inx(i,j)
		if ( xpi(ij) .lt. 0 ) then
		    if ( lwrite ) print *,'ffcd0c: error: ',
     +			'cannot handle pseudothresholds yet: ',xpi(i),
     +			xpi(j),xpi(ij)
		    goto 39
		endif
		if ( ltest .and. DIMAG(cpi(i)).eq.0 .and. 
     +			DIMAG(cpi(j)).eq.0 ) then
		    print *,'ffcd0c: error: threshold without complex '
     +		    	//'masses: ',i,cpi(i),j,cpi(j)
		    goto 39
		endif
*
*		else just add and subtract the B0
*
		if ( lwrite ) print *,'ffcd0c: calling ffxb0p ',i,j,ij
		ier0 = ier
		call ffxb0p(cb0r,xpi(ij),xpi(i),xpi(j),dpipj(i,ij),
     +			dpipj(j,ij),dpipj(i,j),ier0)
		ier1 = max(ier1,ier0)
*
		if ( lwrite ) print *,'ffcd0c: calling ffcb0p ',i,j,ij
		ier0 = ier
		call ffcb0p(cb0c,cpi(ij),cpi(i),cpi(j),cpipj(i,ij),
     +			cpipj(j,ij),cpipj(i,j),ier0)
		ier1 = max(ier1,ier0)
*
*		get the coefficients which are given by Q=a*p(1)
*
		a(1) = xpi(i)*(xpi(j)*dpipj(j,i) - xpi(ij)*(xpi(i)+
     +			xpi(j)))/(xpi(ij)*(xpi(i)+xpi(j))**2)
*		a(2) = 1+a(1):
		a(2) = xpi(j)*(xpi(i)*dpipj(j,i) + xpi(ij)*(xpi(i)+
     +			xpi(j)))/(xpi(ij)*(xpi(i)+xpi(j))**2)
		if ( lwrite ) print *,'a,1+a,1 = ',a(1),a(2),a(2)-a(1)
*
		k = notij(i,j)
		if ( abs(a(1)) .lt. abs(a(2)) ) then
		    p1 = dpipj(inx(i,k),k) + a(1)**2*xpi(ij) +
     +			2*a(1)*fpij4(ij,inx(i,k)) *
     +			isgn(i,j)*isgn(i,k)
		    xmax = max(abs(dpipj(inx(i,k),k)),a(1)**2*xpi(ij))
		    if ( lwrite ) print *,'p1 = ',p1,xmax
		else
		    p1 = dpipj(inx(j,k),k) + a(2)**2*xpi(ij) +
     +			2*a(2)*fpij4(ij,inx(j,k)) *
     +			isgn(i,j)*isgn(j,k)
		    xmax = max(abs(dpipj(inx(j,k),k)),a(2)**2*xpi(ij))
		    if ( lwrite ) print *,'p1 = ',p1,xmax
		endif
		if ( abs(p1) .lt. xloss*xmax )
     +			call ffwarn(212,ier1,p1,xmax)
*
		l = notijk(i,j,k)
		if ( abs(a(1)) .lt. abs(a(2)) ) then
		    p2 = dpipj(inx(i,l),l) + a(1)**2*xpi(ij) +
     +			2*a(1)*fpij4(ij,inx(i,l)) *
     +			isgn(i,j)*isgn(i,l)
		    xmax = max(abs(dpipj(inx(i,l),l)),a(1)**2*xpi(ij))
		    if ( lwrite ) print *,'p2 = ',p2,xmax
		else
		    p2 = dpipj(inx(j,l),l) + a(2)**2*xpi(ij) +
     +			2*a(2)*fpij4(ij,inx(j,l)) *
     +			isgn(i,j)*isgn(j,l)
		    xmax = max(abs(dpipj(inx(j,l),l)),a(2)**2*xpi(ij))
		    if ( lwrite ) print *,'p2 = ',p2,xmax
		endif
		if ( abs(p2) .lt. xloss*xmax )
     +			call ffwarn(213,ier1,p2,xmax)
*
		cb0r = cb0r/cfac/(p1*p2)
		cb0c = cb0c/cfac/(p1*p2)
*		minus because we computed B', not B
		cs = cs + cb0r - cb0c
		if ( lwrite ) then
		    print *,'subtracted threshold ',i,j
		    print *,'csr  = ',csr
		    print *,'cb0r = ',-cb0r
		    print *,'diff   ',csr+cb0r
		    print *,'cb0c = ',-cb0c
		endif
   39	    continue
   40	continue
   41	continue
*  #] threshold:
*  #[ dotproducts:
*
*	and the dot products if requested
*
	ldot = ldotsa
	if ( ldot ) then
	    call ffcod4(cpi,cpipj)
	endif
*  #] dotproducts:
*  #[ finito:
*
*	clean up
*
	ier = ier1
	if ( lwrite ) then
	    print *,'cs,cfac :',cs,cfac,ier
	    print *,'cd0     :',cs*cfac,ier
	endif
*  #] finito:
*###] ffcd0c:
	end
*###[ ffcod4:
	subroutine ffcod4(cpi,cpipj)
***#[*comment:***********************************************************
*									*
*	Convert real dorproducts into complex ones, adding the		*
*	imaginary parts where appropriate.				*
*	For the time being just recompute them...			*
*									*
*	Input:	cpi(13)			complex		m^2, p^2	*
*		cpipj(10,13)		complex		diffs		*
*									*
*	Output:	/ffcots/cfpij4(10,10)	complex		p.p complex	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX cpi(13),cpipj(10,13)
*
*	local variables
*
	integer i,j,ier0,ii(6)
	DOUBLE PRECISION piDpj(10,10),sprec
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ compute dotproducts and determinants:
*
	ier0 = 0
	call ffcot4(cfpij4,cpi,cpipj,10,ier0)
	call ffcel4(cfdl4s,cpi,cfpij4,10,ier0)
	if ( abs(idot).lt.2 ) then
	    if ( onshel ) then
*		we have to recompute the overall \Delta_3
		do 10 i=1,6
		    ii(i) = i+4
   10		continue
		do 30 i=1,10
		    do 20 j=1,10
			piDpj(j,i) = DBLE(cfpij4(j,i))
   20		    continue
   30		continue
*		this prec-juggling should not be necessary, but it is...
		sprec = precx
		precx = precc
		call ffdl3p(fodel3,piDpj,10,ii,ii,ier0)
		precx = sprec
	    else
		fodel3 = fdel3
	    endif
	endif
*
*  #] compute dotproducts and determinants:
*###] ffcod4:
	end
*###[ ffcif4:
	subroutine ffcif4(cpipj,luvw,cpi,ier)
***#[*comment:***********************************************************
*									*
*	Compute the elements 11-13 in xpi and the differences cpipj	*
*	Note that the digits lost in cpipj are not counted towards	*
*	the total.							*
*									*
*	Input:	cpi(1:10)	complex		masses, momenta^2	*
*									*
*	Output:	cpi(11:13)	complex		u and similar vars v,w	*
*		luvw(3)		logical		TRUE if xpi(10+i) has	*
*						been computed here	*
*		cpipj(10,13)	complex		xpi(i) - xpi(j)		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	logical luvw(3)
	DOUBLE COMPLEX cpi(13),cpipj(10,13)
*
*	local variables
*
	integer i,j,ier0,ier1
	DOUBLE PRECISION xmax,absr,absc
	DOUBLE COMPLEX cc
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absr(cc) = abs(DBLE(cc))
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ get differences:
*	simulate the differences in the masses etc..
	if ( lwrite ) print *,'ffcif4: input cpi: ',cpi
	if ( cpi(11)  .eq. 0 ) then
	    cpi(11) = cpi(5)+cpi(6)+cpi(7)+cpi(8)-cpi(9)-cpi(10)
	    if ( lwarn ) then
		xmax = max(absr(cpi(5)),absr(cpi(6)),absr(cpi(7)),
     +		       absr(cpi(8)),absr(cpi(9)),absr(cpi(10)))
		if ( absr(cpi(11)) .lt. xloss*xmax )
     +		    call ffwarn(153,ier,absr(cpi(11)),xmax)
	    endif
	    luvw(1) = .TRUE.
	else
	    luvw(1) = .FALSE.
	endif
	if ( cpi(12)  .eq. 0 ) then
	    cpi(12) = -cpi(5)+cpi(6)-cpi(7)+cpi(8)+cpi(9)+cpi(10)
	    if ( lwarn ) then
		xmax = max(absr(cpi(5)),absr(cpi(6)),absr(cpi(7)),
     +		       absr(cpi(8)),absr(cpi(9)),absr(cpi(10)))
		if ( absr(cpi(12)) .lt. xloss*xmax )
     +		    call ffwarn(154,ier,absr(cpi(12)),xmax)
	    endif
	    luvw(2) = .TRUE.
	else
	    luvw(2) = .FALSE.
	endif
	if ( cpi(13)  .eq. 0 ) then
	    cpi(13) = cpi(5)-cpi(6)+cpi(7)-cpi(8)+cpi(9)+cpi(10)
	    if ( lwarn ) then
		xmax = max(absr(cpi(5)),absr(cpi(6)),absr(cpi(7)),
     +		       absr(cpi(8)),absr(cpi(9)),absr(cpi(10)))
		if ( absr(cpi(13)) .lt. xloss*xmax )
     +		    call ffwarn(155,ier,absr(cpi(13)),xmax)
	    endif
	    luvw(3) = .TRUE.
	else
	    luvw(3) = .FALSE.
	endif
	if ( lwarn ) then
	    do 80 i=1,13
		if ( i .le. 10 ) cpipj(i,i) = 0
		do 70 j=1,min(i-1,10)
		    cpipj(j,i) = cpi(j) - cpi(i)
		    if ( i .le. 10 ) then
			cpipj(i,j) = -cpipj(j,i)
		    endif
*		    we do not need the differences of s,t,u,v,w accurately
		    if ( i .gt. 8 .and. j .gt. 8 ) goto 70
		    if ( absc(cpipj(j,i)) .lt. xloss*absc(cpi(i))
     +					.and. cpi(i) .ne. cpi(j) ) then
			ier0 = 0
			call ffwarn(135,ier0,absc(cpipj(j,i)),
     +				absc(cpi(i)))
			if ( lwrite ) print *,'between cpi(',i,
     +				') and cpi(',j,')'
		    endif
   70		continue
   80	    continue
	else
	    do 100 i=1,13
		do 90 j=1,10
		    cpipj(j,i) = cpi(j) - cpi(i)
   90		continue
  100	    continue
	endif
*  #] get differences:
*###] ffcif4:
	end
