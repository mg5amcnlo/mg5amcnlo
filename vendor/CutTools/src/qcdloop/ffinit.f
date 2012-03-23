*	$Id: ffinit.f,v 1.9 1996/04/26 10:39:03 gj Exp $
*###[ ffini:
	subroutine ffini
***#[*comment:***********************************************************
*	calculate a lot of commonly-used constants in the common block	*
*	/ffcnst/. also set the precision, maximum loss of digits and	*
*	the minimum value the logarithm accepts in /prec/.		*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer i,j,init,ioldp(13,12),isgrop(10,12),ji
	save init
	DOUBLE PRECISION s,sold
	DOUBLE COMPLEX cs
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
	data init /0/
	data ioldp/1,2,3,4, 5,6,7,8,9,10, 11,12,13,
     +		   4,1,2,3, 8,5,6,7,10,9, 11,13,12,
     +		   3,4,1,2, 7,8,5,6,9,10, 11,12,13,
     +		   2,3,4,1, 6,7,8,5,10,9, 11,13,12,
     +		   4,2,3,1, 10,6,9,8,7,5, 12,11,13,
     +		   1,3,2,4, 9,6,10,8,5,7, 12,11,13,
     +		   1,2,4,3, 5,10,7,9,8,6, 13,12,11,
     +		   1,4,3,2, 8,7,6,5,9,10, 11,13,12,
     +		   3,4,2,1, 7,10,5,9,6,8, 13,12,11,
     +		   2,3,1,4, 6,9,8,10,5,7, 12,13,11,
     +		   4,2,1,3, 10,5,9,7,8,6, 13,11,12,
     +		   1,3,4,2, 9,7,10,5,8,6, 13,11,12/
	data isgrop/
     +		+1,+1,+1,+1, +1,+1,+1,+1, +1,+1,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,-1,
     +		+1,+1,+1,+1, +1,+1,+1,+1, +1,-1,
     +		+1,+1,+1,+1, -1,+1,+1,-1, +1,-1,
     +		+1,+1,+1,+1, -1,-1,+1,+1, -1,+1,
     +		+1,+1,+1,+1, +1,+1,-1,+1, +1,+1,
     +		+1,+1,+1,+1, -1,-1,-1,-1, +1,-1,
     +		+1,+1,+1,+1, -1,+1,+1,+1, -1,-1,
     +		+1,+1,+1,+1, +1,+1,+1,-1, +1,-1,
     +		+1,+1,+1,+1, -1,+1,+1,-1, -1,-1,
     +		+1,+1,+1,+1, -1,-1,+1,+1, -1,-1/
*  #] declarations:
*  #[ check:
*	check whether tehre is anything to do
	if ( init .ne. 0 ) return
	init = 1
	print *,'===================================================='
	print *,'  FF 2.0, a package to evaluate one-loop integrals'
	print *,'written by G. J. van Oldenborgh, NIKHEF-H, Amsterdam'
	print *,'===================================================='
	print *,'for the algorithms used see preprint NIKHEF-H 89/17,'
	print *,'''New Algorithms for One-loop Integrals'', by G.J. van'
	print *,'Oldenborgh and J.A.M. Vermaseren, published in '
	print *,'Zeitschrift fuer Physik C46(1990)425.'
	print *,'===================================================='
*  #] check:
*  #[ precision etc:
	lwrite = .TRUE.
	nevent = -1
*
*	the loss of accuracy in any single subtraction at which
*	(timeconsuming) corrective action is to be taken is
*
	xloss = 0.125
*
*	the precision to which real calculations are done is
*
	precx = 1
	sold = 0
	do 1 i=1,1000
	    precx = precx/2
	    s = exp(log(1+precx))
	    if ( s .eq. sold ) goto 2
	    sold = s
    1	continue
    2	continue
	precx = precx*8
*	(take three bits for safety)
	if ( lwrite ) print *,'ffini: precx = ',precx
*
*	the precision to which complex calculations are done is
*
	precc = 1
	sold = 0
	do 3 i=1,1000
	    precc = precc/2
	    cs = exp(log(DCMPLX(1+precc,x0)))
	    if ( DBLE(cs) .eq. sold ) goto 4
	    sold = DBLE(cs)
    3	continue
    4	continue
	precc = precc*8
*	(take three bits for safety)
	if ( lwrite ) print *,'ffini: precc = ',precc
*
*	for efficiency tke them equal if they are not too different
*
	if ( precx/precc .lt. 4 .and. precx/precc .gt. .25 ) then
	    precx = max(precc,precx)
	    precc = max(precc,precx)
	endif
*
*	and the minimum value the logarithm accepts without complaining
*	about arguments zero is (DOUBLE PRECISION cq DOUBLE COMPLEX)
*
	s = 1
	xalogm = 1
	do 5 i=1,10000
	    s = s/2
	    if ( 2*abs(s) .ne. xalogm ) goto 6
	    xalogm = abs(s)
    5	continue
    6	continue
	if ( xalogm.eq.0 ) xalogm = 1d-308
	if ( lwrite ) print *,'ffini: xalogm = ',xalogm
	s = 1
	xclogm = abs(DCMPLX(s))
	do 7 i=1,10000
	    s = s/2
	    if ( 2*abs(DCMPLX(s)) .ne. xclogm ) goto 8
	    xclogm = abs(DCMPLX(s))
    7	continue
    8	continue
	if ( xclogm.eq.0 ) xclogm = 1d-308
	if ( lwrite ) print *,'ffini: xclogm = ',xclogm
*
*	These values are for Absoft, Apollo fortran (68000):
*	xalogm = 1.D-308
*	xclogm = 1.D-18
*	These values are for VAX g_float
*	xalogm = 1.D-308
*	xclogm = 1.D-308
*	These values are for Gould fort (because of div_zz)
*	xalogm = 1.D-75
*	xclogm = 1.D-36
	xalog2 = sqrt(xalogm)
	xclog2 = sqrt(xclogm)
*  #] precision etc:
*  #[ constants:
*
*	calculate the coefficients of the series expansion
*	li2(x) = sum bn*z^n/(n+1)!, z = -log(1-x), bn are the
*	bernouilli numbers (zero for odd n>1).
*
	bf(1) = - 1.D+0/4.D+0
	bf(2) = + 1.D+0/36.D+0
	bf(3) = - 1.D+0/36.D+2
	bf(4) = + 1.D+0/21168.D+1
	bf(5) = - 1.D+0/108864.D+2
	bf(6) = + 1.D+0/52690176.D+1
	bf(7) = - 691.D+0/16999766784.D+3
	bf(8) = + 1.D+0/1120863744.D+3
	bf(9) = - 3617.D+0/18140058832896.D+4
	bf(10) = + 43867.D+0/97072790126247936.D+3
	bf(11) = - 174611.D+0/168600109166641152.D+5
	bf(12) = + 77683.D+0/32432530090601152512.D+4
	bf(13) = - 236364091.D+0/4234560341829359173632.D+7
	bf(14) = + 657931.D+0/5025632054039239458816.D+6
	bf(15) = - 3392780147.D+0/109890470493622010006470656.D+7
	bf(16)=+172.3168255201D+0/2355349904102724211909.3102313472D+6
	bf(17)=-770.9321041217D+0/4428491985594062112714.2791446528D+8
	bf(18)=( 0.4157635644614046176D-28)
	bf(19)=(-0.9962148488284986022D-30)
	bf(20)=( 0.2394034424896265390D-31)
*
*	inverses of integers:
*
	do 10 i=1,30
	    xninv(i) = x1/i
	    xn2inv(i) = x1/(i*i)
   10	continue
*
*	inverses of faculties of integers:
*
	xinfac(1) = x1
	do 20 i=2,30
	    xinfac(i) = xinfac(i-1)/i
   20	continue
*
*	inx: p(inx(i,j)) = isgn(i,j)*(s(i)-s(j))
*
	inx(1,1) = -9999
	inx(2,1) = 5
	inx(3,1) = 9
	inx(4,1) = 8
	inx(1,2) = 5
	inx(2,2) = -9999
	inx(3,2) = 6
	inx(4,2) = 10
	inx(1,3) = 9
	inx(2,3) = 6
	inx(3,3) = -9999
	inx(4,3) = 7
	inx(1,4) = 8
	inx(2,4) = 10
	inx(3,4) = 7
	inx(4,4) = -9999
	isgn(1,1) = -9999
	isgn(2,1) = +1
	isgn(3,1) = -1
	isgn(4,1) = -1
	isgn(1,2) = -1
	isgn(2,2) = -9999
	isgn(3,2) = +1
	isgn(4,2) = +1
	isgn(1,3) = +1
	isgn(2,3) = -1
	isgn(3,3) = -9999
	isgn(4,3) = +1
	isgn(1,4) = +1
	isgn(2,4) = -1
	isgn(3,4) = -1
	isgn(4,4) = -9999
	do 40 i=1,12
	    do 30 j=1,13
		iold(j,i) = ioldp(j,i)
   30	    continue
	    do 35 j=1,10
		isgrot(j,i) = isgrop(j,i)
   35	    continue
   40	continue
	inx5(1,1) = -9999
	inx5(1,2) =  6
	inx5(1,3) = 11
	inx5(1,4) = 14
	inx5(1,5) = 10
	inx5(2,1) =  6
	inx5(2,2) = -9999
	inx5(2,3) =  7
	inx5(2,4) = 12
	inx5(2,5) = 15
	inx5(3,1) = 11
	inx5(3,2) =  7
	inx5(3,3) = -9999
	inx5(3,4) =  8
	inx5(3,5) = 13
	inx5(4,1) = 14
	inx5(4,2) = 12
	inx5(4,3) =  8
	inx5(4,4) = -9999
	inx5(4,5) =  9
	inx5(5,1) = 10
	inx5(5,2) = 15
	inx5(5,3) = 13
	inx5(5,4) =  9
	inx5(5,5) = -9999
*	isgn5 is not yet used.
	do i=1,5
	    do j=1,5
		isgn5(i,j) = -9999
	    enddo
	enddo
*
	inx6(1,1) = -9999
	inx6(1,2) =  7
	inx6(1,3) = 13
	inx6(1,4) = 19
	inx6(1,5) = 17
	inx6(1,6) = 12
	inx6(2,1) =  7
	inx6(2,2) = -9999
	inx6(2,3) =  8
	inx6(2,4) = 14
	inx6(2,5) = 20
	inx6(2,6) = 18
	inx6(3,1) = 13
	inx6(3,2) =  8
	inx6(3,3) = -9999
	inx6(3,4) =  9
	inx6(3,5) = 15
	inx6(3,6) = 21
	inx6(4,1) = 19
	inx6(4,2) = 14
	inx6(4,3) =  9
	inx6(4,4) = -9999
	inx6(4,5) = 10
	inx6(4,6) = 16
	inx6(5,1) = 17
	inx6(5,2) = 20
	inx6(5,3) = 15
	inx6(5,4) = 10
	inx6(5,5) = -9999
	inx6(5,6) = 11
	inx6(6,1) = 12
	inx6(6,2) = 18
	inx6(6,3) = 21
	inx6(6,4) = 16
	inx6(6,5) = 11
	inx6(6,6) = -9999
*	isgn6 is used.
	do i=1,6
	    do j=1,6
		ji = j-i
		if ( ji.gt.+3 ) ji = ji - 6
		if ( ji.lt.-3 ) ji = ji + 6
		if ( ji.eq.0 ) then
		    isgn6(j,i) = -9999
		elseif ( abs(ji).eq.3 ) then
		    if ( i.lt.0 ) then
			isgn6(j,i) = -1
		    else
			isgn6(j,i) = +1
		    endif
		elseif ( ji.gt.0 ) then
		    isgn6(j,i) = +1
		elseif ( ji.lt.0 ) then
		    isgn6(j,i) = -1
		else
		    print *,'ffini: internal error in isgn6'
		    stop
		endif
	    enddo
	enddo
*
*  #] constants:
*  #[ defaults for flags:
	nevent = 0
*
*	the debugging flags.
*
	lwrite = .FALSE.
	ltest = .FALSE.
	lwarn = .TRUE.
	ldc3c4 = .FALSE.
	l4also = .FALSE.
	lmem = .FALSE.
	ldot = .FALSE.
	idot = 0
*
*	Specify which root to take in cases were two are possible
*	it may be advantageous to change this to -1 (debugging hook)
*
	isgn34 = 1
	isgnal = 1
*
*	the cutoff has to be initialized because of the memory mechansim
*
	delta = 0
*
*	the scheme used for the complex scalar functions:
*
*	nschem = 1: do not use the complex mass at all
*		 2: only use the complex mass in linearly divergent terms
*		 3: also use the complex mass in divergent logs UNDEFINED
*		 4: use the complex mass in the C0 if there are
*		    divergent logs
*		 5: include the almost-divergent threshold terms from
*		    (m,m,0) vertices
*		 6: include the (s-m^2)*log(s-m^2) threshold terms from
*		    (m1+m2),m1,m2) vertices
*		 7: full complex computation
*	(only in the ffz... functions):
*	onshel = .FALSE.: use the offshell p^2 everywhere
*		 .TRUE.:  use the onshell p^2 except in complex parts
*
	nschem = 7
	onshel = .TRUE.
*
*	the precision wanted in the complex D0 (and hence E0) when
*	nschem=7, these are calculated via Taylor exoansion in the real
*	one and hence expensive.
*
	reqprc = 1.e-8
*
*	in some schemes, for onshel=.FALSE.,
*	when |p^2-Re(m^2)| < nwidth*|Im(m^2)| special action is taken
*
	nwidth = 5
*
*	a flag to indicate the validity of differences smuggled to the
*	IR routines in the C0 (ff internal only)
*
	lsmug = .FALSE.
*
*  #] defaults for flags:
*###] ffini:
	end
*###[ ffexi:
	subroutine ffexi
***#[*comment:***********************************************************
*	check a lot of commonly-used constants in the common block	*
*	/ffcnst/.							*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer i,ier
	include 'ff.h'
*  #] declarations:
*  #[ checks:
*
*	calculate the coefficients of the series expansion
*	li2(x) = sum bn*z^n/(n+1)!, z = -log(1-x), bn are the
*	bernouilli numbers (zero for odd n>1).
*
	if ( bf(1) .ne. - 1.D+0/4.D+0 )
     +	print *,'ffexi: error: bf(1) is corrupted'
	if ( bf(2) .ne. + 1.D+0/36.D+0 )
     +	print *,'ffexi: error: bf(2) is corrupted'
	if ( bf(3) .ne. - 1.D+0/36.D+2 )
     +	print *,'ffexi: error: bf(3) is corrupted'
	if ( bf(4) .ne. + 1.D+0/21168.D+1 )
     +	print *,'ffexi: error: bf(4) is corrupted'
	if ( bf(5) .ne. - 1.D+0/108864.D+2 )
     +	print *,'ffexi: error: bf(5) is corrupted'
	if ( bf(6) .ne. + 1.D+0/52690176.D+1 )
     +	print *,'ffexi: error: bf(6) is corrupted'
	if ( bf(7) .ne. - 691.D+0/16999766784.D+3 )
     +	print *,'ffexi: error: bf(7) is corrupted'
	if ( bf(8) .ne. + 1.D+0/1120863744.D+3 )
     +	print *,'ffexi: error: bf(8) is corrupted'
	if ( bf(9) .ne. - 3617.D+0/18140058832896.D+4 )
     +	print *,'ffexi: error: bf(9) is corrupted'
	if ( bf(10) .ne. + 43867.D+0/97072790126247936.D+3 )
     +	print *,'ffexi: error: bf(10) is corrupted'
	if ( bf(11) .ne. - 174611.D+0/168600109166641152.D+5 )
     +	print *,'ffexi: error: bf(11) is corrupted'
	if ( bf(12) .ne. + 77683.D+0/32432530090601152512.D+4 )
     +	print *,'ffexi: error: bf(12) is corrupted'
	if ( bf(13) .ne. - 236364091.D+0/4234560341829359173632.D+7 )
     +	print *,'ffexi: error: bf(13) is corrupted'
	if ( bf(14) .ne. + 657931.D+0/5025632054039239458816.D+6 )
     +	print *,'ffexi: error: bf(14) is corrupted'
	if ( bf(15) .ne. -3392780147.D+0/109890470493622010006470656.D+7
     +	) print *,'ffexi: error: bf(15) is corrupted'
	if ( bf(16).ne.+172.3168255201D+0/2355349904102724211909.3102313
     +	472D+6 )
     +	print *,'ffexi: error: bf(16) is corrupted'
	if ( bf(17).ne.-770.9321041217D+0/4428491985594062112714.2791446
     +	528D+8 )
     +	print *,'ffexi: error: bf(17) is corrupted'
	if ( bf(18).ne.( 0.4157635644614046176D-28) )
     +	print *,'ffexi: error: bf(18) is corrupted'
	if ( bf(19).ne.(-0.9962148488284986022D-30) )
     +	print *,'ffexi: error: bf(19) is corrupted'
	if ( bf(20).ne.( 0.2394034424896265390D-31) )
     +	print *,'ffexi: error: bf(20) is corrupted'
*
*	inverses of integers:
*
	do 10 i=1,20
	    if ( abs(xninv(i)-x1/i) .gt. precx*xninv(i) ) print *,
     +		'ffexi: error: xninv(',i,') is not 1/',i,': ',
     +		xninv(i),xninv(i)-x1/i
   10	continue
*
*  #] checks:
*  #[ print summary of errors and warning:
	ier = 0
	call fferr(999,ier)
*  #] print summary of errors and warning:
*###] ffexi:
	end
*###[ fferr:
	subroutine fferr(nerr,ierr)
***#[*comment:***********************************************************
*									*
*	generates an errormessage #nerr	with severity 2			*
*	nerr=999 gives a frequency listing of all errors		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer nmax
	parameter (nmax=100)
	integer nerr,ierr,ifile
	character*80 error(nmax),error1
	logical locwrt
	integer noccur(nmax),init,i,ier,inone,nnerr,nomore
	save error,noccur,init,locwrt,nomore
	include 'ff.h'
*  #] declarations:
*  #[ data:
	data locwrt /.TRUE./
	data nomore /-1/
	data noccur /nmax*0/
	data init /0/
	if ( init.eq.0 ) then
	    init = 1
	    do 1 i=1,nmax
		error(i) =
     +		'fferr:  error:   illegal value for ierr'
    1	    continue
	    call ffopen(ifile,'fferr.dat',ier)
	    if ( ier .ne. 0 ) goto 100
	    rewind(ifile)
	    read(ifile,'(a)')error1
	    read(ifile,'(a)')error1
	    do 90 i=1,10000
		read(ifile,'(i4,a80)',end=110,err=110)ier,error1
		if ( ier .lt. 1 .or. ier .gt. nmax ) then
		    print '(a,i3)','fferr:  error: wild error number ',
     +								ier
		    print '(a,a)','>>> ',error1
		    goto 90
		endif
		error(ier) = error1
   90	    continue
	    goto 110
  100	    continue
	    print '(a)',
     +		'fferr:  warning cannot open fferr.dat with error texts'
  110	    continue
	    close(ifile)
	endif
*  #] data:
*  #[ nerr=999:
	if ( nerr .eq. 999 ) then
*	    print out total numbers...
	    print '(a)',' '
	    print '(a)','total number of errors and warnings'
	    print '(a)','==================================='
	    inone = 1
	    do 10 i=1,nmax
		if ( noccur(i) .gt. 0 ) then
		    print '(a,i8,a,i3,a,a)','fferr: ',noccur(i),
     +			' times ',i,': ',error(i)
		    noccur(i) = 0
		    inone = 0
		endif
   10	    continue
	    if ( inone.eq.1 ) print '(a)','fferr: no errors'
	    if ( lwarn ) then
		call ffwarn(999,ierr,x1,x1)
	    else
		print '(a)','the warning system has been disabled'
	    endif
	    print '(a)',' '
	    return
	endif
*  #] nerr=999:
*  #[ print error:
	if ( nerr .lt. 1 .or. nerr .gt. nmax ) then
	    nnerr = nmax
	else
	    nnerr = nerr
	endif
	noccur(nnerr) = noccur(nnerr) + 1
	ierr = ierr + 100

	if ( nevent .eq. nomore ) return

	if ( locwrt ) then
	    print '(a,i6,a,i6,a,i8)','fferr: id nr ',id,'/',idsub,
     +		', event nr ',nevent
	    print '(a,i6,a,a)','error nr',nerr,': ',error(nnerr)
	endif

	if ( nerr .eq. 100 ) then
*	    we found a matherror - discard all errors from now till next
*	    event
	    nomore = nevent
	endif

*  #] print error:
*###] fferr:
	end
*###[ ffwarn:
	subroutine ffwarn(nerr,ierr,som,xmax)
***#[*comment:***********************************************************
*									*
*	The warning routine.  A warning is aloss of precision greater	*
*	than xloss (which is default set in ffini), whenever in a	*
*	subtraction the result is smaller than xloss*max(operands) this	*
*	routine is called.  Now the strategy is to remember these	*
*	warnings until a 998 message is obtained; then all warnings of	*
*	the previous event are printed.  The rationale is that one	*
*	makes this call if too much preciasion is lost only.		*
*	nerr=999 gives a frequency listing of all warnings		*
*									*
*	Input:	nerr	integer	the id of the warning message, see the	*
*				file ffwarn.dat or 998 or 999		*
*		ierr	integer	the usual error flag: number of digits	*
*				lost so far				*
*		som	real	the result of the addition		*
*		xmax	real	the largest operand			*
*									*
*	Output:	ierr	integer	is raised by the number of digits lost	*
*				the tolerated loss of xloss		*
*									*
*	NOTE: This routine needs a file ffwarn.dat with the warning	*
*	texts, it is very system dependent where to pick it up		*
*	set the PATH variable to your own taste.			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer nmax
	parameter (nmax=300)
*
*	arguments
*
	integer nerr,ierr
	DOUBLE PRECISION som,xmax
*
*	local variables
*
	integer memmax
	parameter (memmax = 1000)
	character*80 warn(nmax),warn1
	integer noccur(nmax),init,i,ier,inone,nnerr,ilost,
     +		nermem(memmax),losmem(memmax),idmem(memmax),
     +		idsmem(memmax),laseve,imem,ifile
	DOUBLE PRECISION xlosti(nmax),xlost
	save warn,noccur,init,xlosti,nermem,losmem,idmem,idsmem,
     +		laseve,imem
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*  #] declarations:
*  #[ data:
	data noccur /nmax*0/
	data init /0/
	if ( init.eq.0 .and. nerr.ne.999 ) then
	    init = 1
	    do 1 i=1,nmax
		warn(i) =
     +  	'ffwarn:  warning:   illegal value for ierr'
		xlosti(i) = 0
    1	    continue
            call ffopen(ifile,'ffwarn.dat',ier)
	    if ( ier.ne.0 ) goto 100
	    rewind(ifile)
	    read(ifile,'(a)')warn1
	    read(ifile,'(a)')warn1
	    do 90 i=1,10000
		read(ifile,'(i4,a80)',end=110,err=110)ier,warn1
		if ( warn1.eq.' ' ) goto 90
		if ( ier.lt.1 .or. ier.gt.nmax ) then
		    print '(a,i3)','ffwarn: error: wild warning number '
     +								,ier
		    print '(a,a)','>>> ',warn1
		    goto 90
		endif
		warn(ier) = warn1
   90	    continue
	    goto 110
  100	    continue
	    print '(a)',
     +	    'ffwarn: warning cannot open ffwarn.dat with warning texts'
  110	    continue
	    close(ifile)
	    laseve = -1
	    imem = 1
	endif
*  #] data:
*  #[ nerr=999:
	if ( nerr.eq.999 ) then
*	    print out total numbers...
	    inone = 1
	    do 10 i=1,nmax
		if ( noccur(i) .gt. 0 ) then
		    print '(a,i8,a,i3,a,a)','ffwarn: ',noccur(i),
     +			' times ',i,': ',warn(i)
		    print '(a,g12.3,a)',
     +			'     (lost at most a factor ',xlosti(i),')'
		    noccur(i) = 0
		    xlosti(i) = 0
		    inone = 0
		endif
   10	    continue
	    if ( inone.eq.1 ) print '(a)','ffwarn: no warnings'
	    return
	endif
*  #] nerr=999:
*  #[ print warning:
	if ( nerr .eq. 998 ) then
	    if ( nevent .ne. laseve ) return
	    do 20 i=1,imem-1
		if ( nermem(i).ne.0 ) then
		    print '(a,i6,a,i6,a,i8)','ffwarn: id nr ',idmem(i),
     +			'/',idsmem(i),', event nr ',nevent
		    print '(a,i6,a,a)','warning nr ',nermem(i),': ',
     +			warn(nermem(i))
		    print '(a,i3,a)','     (lost ',losmem(i),' digits)'
		endif
   20	    continue
	    imem = 1
	    return
	endif
*  #] print warning:
*  #[ collect warnings:
*
*	bring in range
*
	if ( nerr .lt. 1 .or. nerr .gt. nmax ) then
	    nnerr = nmax
	else
	    nnerr = nerr
	endif
*
*	bookkeeping
*
	noccur(nnerr) = noccur(nnerr) + 1
	if ( som .ne. 0 ) then
	    xlost = abs(xmax/som)
	elseif ( xmax .ne. 0 ) then
	    xlost = 1/precx
	else
	    xlost = 1
	endif
	xlosti(nnerr) = max(xlosti(nnerr),xlost)
	if ( xlost*xloss .gt. xalogm ) then
	    ilost = 1 + int(abs(log10(xlost*xloss)))
	else
	    ilost = 0
	endif
	ierr = ierr + ilost
*
*	nice place to stop when debugging
*
	if ( ilost.ge.10 ) then
	    ilost = ilost + 1 - init
	endif
*
*	add to memory
*
	if ( laseve .ne. nevent ) then
	    imem = 1
	    laseve = nevent
	endif
	if ( imem .le. memmax ) then
	    idmem(imem) = id
	    idsmem(imem) = idsub
	    nermem(imem) = nerr
	    losmem(imem) = ilost
	    imem = imem + 1
	endif
*
*	print directly if lwrite TRUE
*
	if ( awrite .or. lwrite ) then
	    imem = imem - 1
	    print '(a,i6,a,i6,a,i8)','ffwarn: id nr ',idmem(imem),'/',
     +			idsmem(imem),', event nr ',nevent
	    print '(a,i6,a,a)','warning nr ',nermem(imem),': ',
     +			warn(nnerr)
	    print '(a,i3,a)','     (lost ',losmem(imem),' digits)'
	endif
*  #] collect warnings:
*###] ffwarn:
	end
*###[ ffopen:
	subroutine ffopen(ifile,name,ier)
*
*	opens a data file and returns the unit number.
*
	implicit none
*
*	arguments
*
	integer ifile,ier
	character*(*) name
*
	logical lopen
	character*128 path,fullname
*
	include 'ff.h'
*
	ier = 0
	do 10 ifile = 10,100
	    inquire(ifile,opened=lopen)
	    if ( .not.lopen ) goto 20
   10	continue
   20	continue
*
*	Adjust PATH to suit your own directory structure
*	I could use a getenv() here, but that may not work
*	on PC/Mac/...
*	VMS users: use something like the following lines instead
*	fullname = 'USR$LOCAL[GEERT]'//name
*	open(ifile,file=fullname,status='OLD',READONLY,err=100)
*
*	first try - my home directory
	path = '/user/gj/lib/'
	fullname = path(1:index(path,' ')-1)//name
	open(ifile,file=fullname,status='OLD',err=30)
	return
   30	continue
*	second try - the system directory
	path = '/usr/local/ff/'
	fullname = path(1:index(path,' ')-1)//name
	open(ifile,file=fullname,status='OLD',err=40)
	return
*	file could not be found
   40	continue
	print *,'ffopen: error: could not open ',fullname
	print *,'        adjust path in ffopen (ffinit.f)'
	ier = -1
*###] ffopen:
	end
*###[ ffbnd:
	DOUBLE PRECISION function ffbnd(n1,n2,array)
*************************************************************************
*									*
*	calculate bound = (precx*|a(n1)/a(n1+n2)|^(1/n2) which is the	*
*	maximum value of x in a series expansion sum_(i=n1)^(n1+n2)	*
*	a(i)*x(i) to give a result of accuracy precx (actually of |next	*
*	term| < prec							*
*									*
*************************************************************************
	implicit none
	integer n1,n2
	DOUBLE PRECISION array(n1+n2)
	include 'ff.h'
	if ( array(n1+n2) .eq. 0 ) then
	   print *,'ffbnd: fatal: array not intialized; did you call ',
     +		'ffini?'
	   stop
	endif
	ffbnd = (precx*abs(array(n1)/array(n1+n2)))**(1/DBLE(n2))
*###] ffbnd:
	end
*###[ ffbndc:
	DOUBLE PRECISION function ffbndc(n1,n2,carray)
*************************************************************************
*									*
*	calculate bound = (precc*|a(n1)/a(n1+n2)|^(1/n2) which is the	*
*	maximum value of x in a series expansion sum_(i=n1)^(n1+n2)	*
*	a(i)*x(i) to give a result of accuracy precc (actually of |next	*
*	term| < prec							*
*									*
*************************************************************************
	implicit none
	integer n1,n2
	DOUBLE COMPLEX carray(n1+n2)
	include 'ff.h'
	if ( carray(n1+n2) .eq. 0 ) then
	   print *,'ffbnd: fatal: array not intialized; did you call ',
     +		'ffini?'
	   stop
	endif
	ffbndc = (precc*abs(carray(n1)/carray(n1+n2)))**(1/DBLE(n2))
*###] ffbndc:
	end
*###[ ffroot:
	subroutine ffroot(xm,xp,a,b,c,d,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the roots of the equation				*
*		a*x^2 - 2*b*x + c = 0					*
*	given by							*
*		x = (b +/- d )/a	xp*xm = c/a			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ier
	DOUBLE PRECISION xm,xp,a,b,c,d
*
*	local variables:
*
	DOUBLE PRECISION s1,s2,s3,rloss
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( a .eq. 0 ) then
	    call fferr(39,ier)
	    if ( b.gt.0 .eqv. d.gt.0 ) then
		xp = 1/xalogm
		xm = c/(b+d)
	    else
		xp = c/(b-d)
		xm = 1/xalogm
	    endif
	    return
	endif
*	if ( lwrite ) print *,'ffroot: a,b,c,d = ',a,b,c,d
*  #] check input:
*  #[ calculations:
	if ( d .eq. 0 ) then
	    xm = b / a
	    xp = xm
	elseif ( b .gt. 0 .eqv.  d .gt. 0 ) then
	    xp = ( b + d ) / a
	    xm = c / (a*xp)
	else
	    xm = ( b - d ) / a
	    xp = c / (a*xm)
	endif
*  #] calculations:
*  #[ test output:
	if ( ltest ) then
	    rloss = xloss*DBLE(10)**(-2-mod(ier,50))
	    if ( xm .ne. 0 ) then
		s1 = a*xm
		s2 = 2*b
		s3 = c/xm
		if ( rloss*abs(s1-s2+s3) .gt. precx*max(abs(s1),abs(s2),
     +			abs(s3)) ) then
		    print *,'ffroot: error: xm not root! ',s1,s2,s3,
     +			s1-s2+s3,ier
		endif
	    endif
	    if ( xp .ne. 0 ) then
		s1 = a*xp
		s2 = 2*b
		s3 = c/xp
		if ( rloss*abs(s1-s2+s3) .gt. precx*max(abs(s1),abs(s2),
     +			abs(s3)) ) then
		    print *,'ffroot: error: xp not root! ',s1,s2,s3,
     +			s1-s2+s3,ier
		endif
	    endif
	endif
*  #] test output:
*###] ffroot:
	end
*###[ ffcoot:
	subroutine ffcoot(xm,xp,a,b,c,d,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the roots of the equation				*
*		a*x^2 - 2*b*x + c = 0					*
*	given by							*
*		x = (b +/- d )/a	xp*xm = c/a			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ier
	DOUBLE COMPLEX xm,xp,a,b,c,d
*
*	local variables:
*
	DOUBLE COMPLEX s1,s2,s3,cc
	DOUBLE PRECISION absc,rloss
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*  #] declarations:
*  #[ check input:
	if ( a .eq. 0 ) then
	    call fferr(38,ier)
	    if ( DBLE(b).gt.0 .eqv. DBLE(d).gt.0 ) then
		xp = 1/xclogm
		xm = c/(b+d)
	    else
		xp = c/(b-d)
		xm = 1/xclogm
	    endif
	    return
	endif
*	if ( lwrite ) print *,'ffroot: a,b,c,d = ',a,b,c,d
*  #] check input:
*  #[ calculations:
	cc = b+d
	if ( d .eq. 0 ) then
	    xm = b / a
	    xp = xm
	elseif ( absc(cc) .gt. xloss*absc(d) ) then
	    xp = ( b + d ) / a
	    xm = c / (a*xp)
	else
	    xm = ( b - d ) / a
	    xp = c / (a*xm)
	endif
*  #] calculations:
*  #[ test output:
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    if ( absc(xm) .gt. xclogm ) then
		s1 = a*xm
		s2 = 2*b
		s3 = c/xm
		cc = s1-s2+s3
		if ( rloss*absc(cc).gt.precc*max(absc(s1),absc(
     +			s2),absc(s3)) ) print *,
     +			'ffcoot: error: xm not root! ',s1,s2,s3,s1-s2+s3
	    endif
	    if ( absc(xp) .gt. xclogm ) then
		s1 = a*xp
		s2 = 2*b
		s3 = c/xp
		cc = s1-s2+s3
		if ( rloss*absc(cc).gt.precc*max(absc(s1),absc(
     +			s2),absc(s3)) ) print *,
     +			'ffcoot: error: xp not root! ',s1,s2,s3,s1-s2+s3
	    endif
	endif
*  #] test output:
*###] ffcoot:
	end
*###[ ffxhck:
	subroutine ffxhck(xpi,dpipj,ns,ier)
***#[*comment:***********************************************************
*									*
*	check whether the differences dpipj are compatible with xpi	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ns,ier
	DOUBLE PRECISION xpi(ns),dpipj(ns,ns)
	integer i,j
	DOUBLE PRECISION xheck,rloss
	include 'ff.h'
*  #] declarations:
*  #[ calculations:
	if ( ier.lt.0 ) then
	    print *,'ffxhck: error: ier < 0 ',ier
	    ier=0
	endif
	rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	do 20 i=1,ns
	    do 10 j=1,ns
		xheck = dpipj(j,i) - xpi(j) + xpi(i)
		if ( rloss*abs(xheck) .gt. precx*max(abs(dpipj(j,i)),
     +			abs(xpi(j)),abs(xpi(i))) ) then
		    print *,'ffxhck: error: dpipj(',j,i,') <> xpi(',j,
     +			') - xpi(',i,'):',dpipj(j,i),xpi(j),xpi(i),
     +			xheck,ier
		    if ( lwrite ) ier = ier + 100
		endif
   10	    continue
   20	continue
*  #] calculations:
*###] ffxhck:
	end
*###[ ffchck:
	subroutine ffchck(cpi,cdpipj,ns,ier)
***#[*comment:***********************************************************
*									*
*	check whether the differences cdpipj are compatible with cpi	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ns,ier
	DOUBLE COMPLEX cpi(ns),cdpipj(ns,ns),c
	integer i,j
	DOUBLE COMPLEX check
	DOUBLE PRECISION absc,rloss
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ calculations:
	if ( ier.lt.0 ) then
	    print *,'ffchck: error: ier < 0 ',ier
	    ier=0
	endif
	rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	do 20 i=1,ns
	    do 10 j=1,ns
		check = cdpipj(j,i) - cpi(j) + cpi(i)
		if ( rloss*absc(check) .gt. precc*max(absc(
     +			cdpipj(j,i)),absc(cpi(j)),absc(cpi(i))) ) then
		    print *,'ffchck: error: cdpipj(',j,i,') <> cpi(',j,
     +		       ') - cpi(',i,'):',cdpipj(j,i),cpi(j),cpi(i),
     +		       check,ier
		    if ( lwrite ) ier = ier + 100
		endif
   10	    continue
   20	continue
*  #] calculations:
*###] ffchck:
	end
*###[ nffeta:
	integer function nffeta(ca,cb,ier)
***#[*comment:***********************************************************
*	calculates							*
*									*
*	eta(a,b)/(2*i*pi) = ( thIm(-a)*thIm(-b)*thIm(a*b)		*
*				- thIm(a)*thIm(b)*thIm(-a*b) )		*
*									*
*	with thIm(a) = theta(Im(a))					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ier
	DOUBLE COMPLEX ca,cb
	DOUBLE PRECISION a,b,ab,rab
	include 'ff.h'
*  #] declarations:
*  #[ calculations:
	a = DIMAG(ca)
	b = DIMAG(cb)
	if ( a*b .lt. 0 ) then
	    nffeta = 0
	    return
	endif
	rab = DBLE(ca)*DBLE(cb) - a*b
	ab = DBLE(ca)*b + a*DBLE(cb)
	if ( abs(ab) .lt. precc*abs(DBLE(ca)*b) ) then
	    call fferr(32,ier)
	    if ( lwrite ) print *,'a,b = ',ca,cb,
     +		' (no precision left in DIMAG(ab)=',ab,')'
	endif
	if ( a .lt. 0 .and. b .lt. 0 .and. ab .gt. 0 ) then
	    nffeta = 1
	elseif ( a .gt. 0 .and. b .gt. 0 .and. ab .lt. 0 ) then
	    nffeta = -1
	elseif ( a .eq. 0 .and. DBLE(ca) .le. 0 .or.
     +		 b .eq. 0 .and. DBLE(cb) .le. 0 .or.
     +		 ab .eq. 0 .and. rab .le. 0 ) then
	    call fferr(32,ier)
	    if ( ltest .or. lwrite ) print *,'a,b = ',ca,cb
	    nffeta = 0
	else
	    nffeta = 0
	endif
*  #] calculations:
*###] nffeta:
	end
*###[ nffet1:
	integer function nffet1(ca,cb,cc,ier)
***#[*comment:***********************************************************
*	calculates the same eta with three input variables		*
*									*
*	et1(a,b)/(2*i*pi) = ( thIm(-a)*thIm(-b)*thIm(c)			*
*				- thIm(a)*thIm(b)*thIm(-c) )		*
*									*
*	with thIm(a) = theta(Im(a))					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ier
	DOUBLE COMPLEX ca,cb,cc,c
	DOUBLE PRECISION a,b,ab,abp,absc
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ check input:
	if ( ltest .and. DIMAG(ca)*DIMAG(cb) .gt. 0 .and. DBLE(ca)*DBLE(
     +		cb) .ne. 0 ) then
	    ab = DIMAG(cc)
	    abp = DIMAG(ca*cb)
	    if ( xloss*abs(abp) .lt. precc*absc(ca)*absc(cb) )
     +		abp = 0
	    if ( ab .gt. 0 .and. abp .lt. 0 .or. ab .lt. 0 .and. abp
     +		.gt. 0  ) then
		print *,'nffet1: error:  sgn im(ca*cb) != sgn im(cc): ',
     +			ab,abp
	    endif
	endif
*  #] check input:
*  #[ calculations:
	a = DIMAG(ca)
	b = DIMAG(cb)
	if ( a .gt. 0 .neqv. b .gt. 0 ) then
	    nffet1 = 0
	    return
	endif
	ab = DIMAG(cc)
	if ( a .lt. 0 .and. b .lt. 0 .and. ab .gt. 0 ) then
	    nffet1 = 1
	elseif ( a .gt. 0 .and. b .gt. 0 .and. ab .lt. 0 ) then
	    nffet1 = -1
	elseif ( a .eq. 0 .and. DBLE(ca) .le. 0 .or.
     +		 b .eq. 0 .and. DBLE(cb) .le. 0 .or.
     +		 ab .eq. 0 .and. DBLE(cc) .le. 0 ) then
	    call fferr(33,ier)
	    if ( ltest.or.lwrite ) print *,'a,b,ab = ',ca,cb,cc
	    nffet1 = 1
	else
	    nffet1 = 0
	endif
*  #] calculations:
*###] nffet1:
	end
*###[ ffcayl:
	subroutine ffcayl(cs,z,coeff,n,ier)
***#[*comment:***********************************************************
*									*
*	Do a Taylor expansion in z with real coefficients coeff(i)	*
*									*
*	Input:	z		complex					*
*		coeff(n)	real					*
*		n		integer					*
*									*
*	Output	cs		complex		\sum_{i=1} z^i coeff(i)	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer n,ier
	DOUBLE PRECISION coeff(n)
	DOUBLE COMPLEX z,cs
*
*	local variables
*
	integer i
	DOUBLE PRECISION absc
	DOUBLE COMPLEX c,zi,csi
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*  #] declarations:
*  #[ work:
	cs = z*DBLE(coeff(1))
	if ( absc(z) .lt. precc ) return
	zi = z
	do 10 i=2,n
	    zi = zi*z
	    csi = zi*DBLE(coeff(i))
	    cs = cs + csi
	    if ( absc(csi) .lt. precc*absc(cs) ) goto 20
   10	continue
	call ffwarn(9,ier,precc,absc(csi))
   20	continue
*  #] work:
*###] ffcayl:
	end
*###[ fftayl:
	subroutine fftayl(s,z,coeff,n,ier)
***#[*comment:***********************************************************
*									*
*	Do a Taylor expansion in z with real coefficients coeff(i)	*
*									*
*	Input:	z		real					*
*		coeff(n)	real					*
*		n		integer					*
*									*
*	Output	cs		real		\sum_{i=1} z^i coeff(i)	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer n,ier
	DOUBLE PRECISION coeff(n),z,s
*
*	local variables
*
	integer i
	DOUBLE PRECISION zi,si
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ work:
	s = coeff(1)*z
	if ( abs(z) .lt. precx ) return
	zi = z
	do 10 i=2,n
	    zi = zi*z
	    si = coeff(i)*zi
	    s = s + si
	    if ( abs(si) .lt. precx*abs(s) ) goto 20
   10	continue
	call ffwarn(9,ier,precx,si)
   20	continue
*  #] work:
*###] fftayl:
	end

