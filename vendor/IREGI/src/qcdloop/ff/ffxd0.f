*--#[ log:
*	$Id: ffxd0.f,v 1.4 1996/01/22 13:32:52 gj Exp $
*	$Log: ffxd0.f,v $
c Revision 1.4  1996/01/22  13:32:52  gj
c Added sanity check on ier; if it is larger than 16 some routines will not
c compute anything.
c
c Revision 1.3  1995/11/28  13:37:47  gj
c Found wrong sign in ffcdna, fixed typo in ffcrp.
c Killed first cancellation in ffcdna - more to follow
c Added warnings to ffwarn.dat; slightly changed debug output in ffxd0.f
c
c Revision 1.2  1995/10/17  06:55:12  gj
c Fixed ieps error in ffdcrr (ffcxs4.f), added real case in ffcrr, debugging
c info in ffxd0, and warned against remaining errors for del2=0 in ffrot4
c (ffxd0h.f)
c
*--#] log:
*###[ ffxd0:
	subroutine ffxd0(cd0,xpi,ier)
***#[*comment:***********************************************************
*									*
*			  1   /						*
*	calculate cd0 =	----- \dq [(q^2 + 2*s_1.q)*(q^2 + 2*s2.q)	*
*			ipi^2 /      *(q^2 + 2*s3.q)*(q^2 + 2*s4.q)]^-1	*
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
*	following the two-three-point-function method in 't hooft &	*
*	veltman. this is only valid if there is a lambda(pij,mi,mj)>0	*
*									*
*	Input:	xpi = mi^2	   (real)  i=1,4			*
*		xpi = pi.pi	   (real)  i=5,8 (note: B&D metric)	*
*		xpi(9)=s	   (real)  (=p13)			*
*		xpi(10)=t	   (real)  (=p24)			*
*		xpi(11)=u	   (real)  u=p5.p5+..-p9.p9-p10.10 or 0	*
*		xpi(12)=v	   (real)  v=-p5.p5+p6.p6-p7.p7+.. or 0	*
*		xpi(13)=w	   (real)  w=p5.p5-p6.p6+p7.p7-p8.p8+.. *
*	output:	cd0		  (complex)				*
*		ier		  (integer) <50:lost # digits 100=error	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE PRECISION xpi(13)
	DOUBLE COMPLEX cd0
	integer ier
*
*	local variables
*
	logical luvw(3)
	DOUBLE PRECISION dpipj(10,13)
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ call ffdif4, ffxd0a:
*
	call ffdif4(dpipj,luvw,xpi,ier)
	call ffxd0a(cd0,xpi,dpipj,ier)
*
*	restore the zeros for u,v,w as we have calculated them
*	ourselves and the user is unlikely to do this...
*
	if ( luvw(1) ) xpi(11) = 0
	if ( luvw(2) ) xpi(12) = 0
	if ( luvw(3) ) xpi(13) = 0
*
*  #] call ffdif4, ffxd0a:
*###] ffxd0:
	end
*###[ ffxd0a:
	subroutine ffxd0a(cd0,xpi,dpipj,ier)
*
*	glue routine which calls ffxd0b with ndiv=0
*
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xpi(13),dpipj(10,13)
	DOUBLE COMPLEX cd0
*
*	locals
*
	DOUBLE COMPLEX cs,cfac
*
*	and go!
*
	call ffxd0b(cs,cfac,xpi,dpipj,0,ier)
	cd0 = cs*cfac
*
*###] ffxd0a:
	end
*###[ ffxd0b:
	subroutine ffxd0b(cs,cfac,xpi,dpipj,ndiv,ier)
***#[*comment:***********************************************************
*									*
*			  1   /						*
*	calculate cd0 =	----- \dq [(q^2 + 2*s_1.q)*(q^2 + 2*s2.q)	*
*			ipi^2 /      *(q^2 + 2*s3.q)*(q^2 + 2*s4.q)]^-1	*
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
*	following the two-three-point-function method in 't hooft &	*
*	veltman. this is only valid if there is a lambda(pij,mi,mj)>0	*
*									*
*	Input:	xpi = mi^2	   (real)  i=1,4			*
*		xpi = pi.pi	   (real)  i=5,8 (note: B&D metric)	*
*		xpi(9)=s	   (real)  (=p13)			*
*		xpi(10)=t	   (real)  (=p24)			*
*		xpi(11)=u	   (real)  u=p5.p5+..-p9.p9-p10.10	*
*		xpi(12)=v	   (real)  v=-p5.p5+p6.p6-p7.p7+..	*
*		xpi(13)=w	   (real)  w=p5.p5-p6.p6+p7.p7-p8.p8+.. *
*		dpipj(10,13)	   (real)  = pi(i) - pi(j)		*
*	output:	cs,cfac		  (complex) cd0 = cs*cfac		*
*		ier		  (integr) 0=ok 1=inaccurate 2=error	*
*	calls:	ffcxs3,ffcxr,ffcrr,...					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ndiv,ier
	DOUBLE PRECISION xpi(13),dpipj(10,13)
	DOUBLE COMPLEX cs,cfac
*
*	local variables
*
	integer i,j,itype,ini2ir,ier2,idone,ier0,ii(6),idotsa
	logical ldel2s
	DOUBLE COMPLEX c,cs1,cs2
	DOUBLE PRECISION absc,xmax,xpip(13),dpipjp(10,13),piDpjp(10,10),
     +		qiDqj(10,10),del2s,delta0,xnul,rloss,vgl
	save ini2ir,delta0
*
*	common blocks:
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	memory
*
	integer iermem(memory),ialmem(memory),memind,ierini,nscsav,
     +		isgnsa
	logical onssav
	DOUBLE PRECISION xpimem(10,memory),dl4mem(memory)
	DOUBLE COMPLEX csmem(memory),cfcmem(memory)
	save memind,iermem,ialmem,xpimem,dl4mem,nscsav,onssav,csmem,
     +		cfcmem
*
*	statement function:
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data memind /0/
	data ini2ir /0/
	data delta0 /0./
*
*  #] declarations:
*  #[ initialisations:
	cs = 0
	cfac = 1
	idsub = 0
	idone = 0
*  #] initialisations:
*  #[ check input if dotproducts are input:
*
	if ( ltest .and. idot.gt.0 ) then
	    if ( lwrite ) print *,'ffxd0b: checking input dotproducts'
	    ier0 = ier
	    idotsa = idot
	    idot = 0
	    call ffdot4(qiDqj,xpi,dpipj,10,ier0)
	    idot = idotsa
	    rloss = xloss*DBLE(10)**(-2-mod(ier0,50))
	    if ( idot.le.2 ) then
		do 20 i=5,10
		    do 10 j=5,10
			xnul = fpij4(j,i)-qiDqj(j,i)
			xmax = abs(qiDqj(j,i))
			if ( abs(rloss*xnul) .gt. precx*xmax ) print *,
     +			    'ffxd0b: error: input dotproduct piDpj(',j,
     +			    i,') wrong: ',fpij4(j,i),qiDqj(j,i),xnul,
     +			    ier0
   10		    continue
   20		continue
	    else
		do 40 i=1,10
		    do 30 j=1,10
			xnul = fpij4(j,i)-qiDqj(j,i)
			xmax = abs(qiDqj(j,i))
			if ( abs(rloss*xnul) .gt. precx*xmax ) print *,
     +			    'ffxd0b: error: input dotproduct piDpj(',j,
     +			    i,') wrong:',fpij4(j,i),qiDqj(j,i),xnul,ier0
   30		    continue
   40		continue
	    endif
	endif
	if ( ltest ) then
	    if ( abs(idot).ge.2 ) then
		if ( lwrite ) print *,'ffxd0b: checking input fdel3 ',
     +			fdel3,ier0
		if ( idot.lt.0 ) then
		    ier0 = ier
		    idotsa = idot
		    idot = 0
		    call ffdot4(qiDqj,xpi,dpipj,10,ier0)
		    idot = idotsa
		endif
		ii(1) =  5
		ii(2) =  6
		ii(3) =  7
		ii(4) =  8
		ii(5) =  9
		ii(6) = 10
		call ffdl3p(vgl,qiDqj,10,ii,ii,ier0)
		rloss = xloss**2*DBLE(10)**(-mod(ier0,50))
		xnul = fdel3 - vgl
		xmax = abs(vgl)
		if ( abs(rloss*xnul).gt.precx*xmax ) print *,
     +			'ffxd0b: error: input del3p wrong: ',fdel3,vgl,
     +			xnul,ier0
	    endif
	    if ( idot.ge.4 ) then
		if ( lwrite ) print *,'ffxd0b: checking input fdel4s'
		call ffdel4(vgl,xpi,qiDqj,10,ier0)
		xnul = fdel4s - vgl
		xmax = abs(vgl)
		if ( abs(rloss*xnul).gt.precx*xmax ) print *,
     +			'ffxd0b: error: input del4s wrong: ',fdel4s,vgl,
     +			xnul,ier0
	    endif
	endif
*
*  #] check input if dotproducts are input:
*  #[ check for IR 4point function:
*
	call ffxdir(cs,cfac,idone,xpi,dpipj,4,ndiv,ier)
	if ( idone .le. 0 .and. ndiv .gt. 0 ) then
	    if ( lwrite ) print *,'ffxd0b: at most log divergence'
	    cs = 0
	    cfac = 1
	    ier = 0
	    return
	endif
	if ( idone .gt. 0 ) then
	    return
	endif
*
*  #] check for IR 4point function:
*  #[ rotate to calculable position:
	call ffrot4(irota4,del2s,xpip,dpipjp,piDpjp,xpi,dpipj,qiDqj,4,
     +		itype,ier)
	if ( itype .lt. 0 ) then
	    print *,'ffxd0b:  error:  Cannot handle this ',
     +		' masscombination yet:'
	    print *,(xpi(i),i=1,13)
	    return
	endif
	if ( itype .eq. 1 ) then
	    ldel2s = .TRUE.
	    isgnal = +1
	else
	    ldel2s = .FALSE.
	endif
*  #] rotate to calculable position:
*  #[ treat doubly IR divergent case:
	if ( itype .eq. 2 ) then
*
*	    double IR divergent diagram, i.e. xpi(3)=xpi(4)=xpi(7)=0
*
	    if ( ini2ir .eq. 0 ) then
		ini2ir = 1
		print *,'ffxd0b: using the log(lam) prescription to'
		print *,'        regulate the 2 infrared poles to match'
		print *,'        with soft gluon massive, lam^2 =',delta
	    endif
	    if ( ltest .and. idone .ne. 2 ) then
		print *,'ffxd0: error: itype=2 but idone != 2'
	    endif
	    ier2 = 0
	    call ffx2ir(cs1,cs2,xpip,dpipjp,ier2)
	    del2s = -delta**2/4
*
*	    correct for the wrongly treated IR pole
*
	    cs = cs + (cs1 + cs2)/cfac
	    ier = max(ier,ier2)
	    xmax = max(absc(cs1),absc(cs2))/absc(cfac)
	    if ( absc(cs) .lt. xloss*xmax )
     +		call ffwarn(172,ier,absc(cs),xmax)
	    if ( .not.ldot ) return
	endif
*
*  #] treat doubly IR divergent case:
*  #[ look in memory:
	ierini = ier
	isgnsa = isgnal
*
*	initialise memory
*
	if ( lmem .and. idone .eq. 0 .and. (memind .eq. 0 .or. nschem
     +		.ne. nscsav .or. (onshel .neqv. onssav) ) ) then
	    memind = 0
	    nscsav = nschem
	    onssav = onshel
	    do 2 i=1,memory
		do 1 j=1,10
		    xpimem(j,i) = 0
    1		continue
		ialmem(i) = 0
    2	    continue
	endif
*
	if ( lmem .and. idone .eq. 0 .and. delta .eq. delta0 ) then
	    do 150 i=1,memory
		do 130 j=1,10
		    if ( xpip(j) .ne. xpimem(j,i) ) goto 150
  130		continue
*		we use ialmem(i)==0 to signal that both are covered as
*		the sign was flipped during the computation
		if ( ialmem(i).ne.isgnal .and. ialmem(i).ne.0 ) goto 150
*		we found an already calculated masscombination ..
*		(maybe check differences as well)
		if ( lwrite ) print *,'ffxd0b: using previous result'
		cs = csmem(i)
		cfac = cfcmem(i)
		ier = ier+iermem(i)
		if ( ldot ) then
		    fdel4s = dl4mem(i)
*		    we forgot to calculate the dotproducts
		    idone = 1
		    goto 51
		endif
		return
  150	    continue
*	    if ( lwrite ) print *,'ffxd0b: not found in memory'
	elseif ( lmem ) then
	    delta0 = delta
	endif
   51	continue
*  #] look in memory:
*  #[ get dotproducts:
*
*	Calculate the dotproducts (in case it comes out of memory the
*	error is already included in ier)
*
	ier0 = ier
	call ffgdt4(piDpjp,xpip,dpipjp,xpi,dpipj,itype,ier0)
	if ( idone .gt. 0 ) return
	ier = ier0
	if ( ier.ge.100 ) then
	    cs = 0
	    cfac = 1
	    return
	endif
*
*  #] get dotproducts:
*  #[ calculations:
*
	call ffxd0e(cs,cfac,xmax, .FALSE.,ndiv,xpip,dpipjp,piDpjp,del2s,
     +		ldel2s,ier)
*
*	Finally ...
*	Check for cancellations in the final adding up
*
	if ( lwarn .and. 2*absc(cs) .lt. xloss*xmax )
     +		call ffwarn(84,ier,absc(cs),xmax)
*
*  #] calculations:
*  #[ add to memory:
*
*	memory management :-)
*
	if ( lmem ) then
	    memind = memind + 1
	    if ( memind .gt. memory ) memind = 1
	    do 200 j=1,10
		xpimem(j,memind) = xpip(j)
  200	    continue
	    csmem(memind)  = cs
	    cfcmem(memind) = cfac
	    iermem(memind) = ier-ierini
	    ialmem(memind) = isgnal
	    dl4mem(memind) = fdel4s
	    if ( isgnal.ne.isgnsa ) then
		ialmem(memind) = 0
	    endif
	endif
*  #] add to memory:
*###] ffxd0b:
	end
*###[ ffxd0e:
	subroutine ffxd0e(cs,cfac,xmax,lir,ndiv,xpip,dpipjp,piDpjp,
     +		del2s,ldel2s,ier)
***#[*comment:***********************************************************
*									*
*	Break in the calculation of D0 to allow the E0 to tie in in a	*
*	logical position.  This part gets untransformed momenta but	*
*	rotated momenta in and gives the D0 (in two pieces) and the	*
*	maximum term back.						*
*									*
*	Input	xpip	real(13)					*
*		dpipjp	real(10,13)					*
*		piDpjp	real(10,10)					*
*		del2s	real						*
*		ldel2s	logical						*
*		lir	logical		if TRUE it can still be IR-div	*
*		ndiv	integer		number of required divergences	*
*									*
*	Output:	cs	complex		the fourpoint function without	*
*					overall factor (sum of dilogs)	*
*		cfac	complex		this overall factor		*
*		xmax	real		largest term in summation	*
*		ier	integer		usual error flag		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ndiv,ier
	logical lir,ldel2s
	DOUBLE PRECISION xpip(13),dpipjp(10,13),piDpjp(10,10),xmax,del2s
	DOUBLE COMPLEX cs,cfac
*
*	local variables
*
	DOUBLE COMPLEX c,cs4(175),cs3(2)
	logical laai
	integer i,j,ier0,itime,maxlos,init,isoort(16),ipi12(26),
     +		ipi123(2),ipi12t,idone
	DOUBLE PRECISION absc,sdel2s,ai(4),daiaj(4,4),aai(4),
     +		dt3t4,xqi(10),dqiqj(10,10),qiDqj(10,10),xfac
	save maxlos
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function:
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data init /0/
*  #] declarations:
*  #[ check for IR 4point function:
	if ( lir ) then
*
	    ier0 = ier
	    call ffxdir(cs,cfac,idone,xpip,dpipjp,4,0,ier)
	    if ( idone .le. 0 .and. ndiv .gt. 0 ) then
		if ( lwrite ) print *,'ffxd0e: at most log divergence'
		cs = 0
		cfac = 1
		xmax = 0
		ier = 0
		return
	    endif
	    if ( idone .gt. 0 ) then
		xmax = abs(cs)*10d0**(-mod((ier0-ier),50))
		return
	    endif
	endif
*
*  #] check for IR 4point function:
*  #[ init:
*
*	initialize cs4:
*
	do 80 i=1,175
	    cs4(i) = 0
   80	continue
	do 90 i=1,26
	    ipi12(i) = 0
   90	continue
	cs = 0
*
*	check ier for sanity
*
	if ( ltest ) then
	    if ( ier.lt.0 .or. mod(ier,50).gt.20 ) then
		print *,'ffxd0e: error: found ier = ',ier
		print *,'        are you sure I lost THAT many digits?'
		print *,'        please check that ier is set to 0 '//
     +			'before calling FF!'
	    endif
	endif
*
*  #] init:
*  #[ transform the masses and momenta:
	itime = 1
   25	continue
*
*	Transform with the A's of gerard 't hooft's transformation:
*
	if ( lwrite ) print '(a)','  ##[ transform momenta:'
*
*	NOTE: for some odd reason I cannot vary isgnal,isgn34
*	independently!
*
	isgn34 = isgnal
	sdel2s = isgn34*sqrt(-del2s)
	ier0 = ier
	call ffai(ai,daiaj,aai,laai,del2s,sdel2s,xpip,dpipjp,piDpjp,
     +		ier0)
	if ( ier0 .ge. 100 ) goto 70
	call fftran(ai,daiaj,aai,laai,xqi,dqiqj,qiDqj,del2s,sdel2s,
     +		xpip,dpipjp,piDpjp,ier0)
	if ( ier0 .ge. 100 ) goto 70
	if ( .not.ldel2s ) then
	    dt3t4 = -2*ai(3)*ai(4)*sdel2s
	    if ( ltest ) then
		if ( xloss*abs(dt3t4-xqi(3)+xqi(4)) .gt. precx*max(
     +			abs(dt3t4),abs(xqi(3)),abs(xqi(4))) ) then
		    print *,'ffxd0a:  error:  dt3t4 <> t3 - t4',dt3t4,
     +			xqi(3),xqi(4),dt3t4-xqi(3)+xqi(4)
		endif
	    endif
	    if ( dt3t4 .eq. 0 ) then
*		don't know what to do...
		call fferr(85,ier)
		return
	    endif
	else
*	    this value is modulo the delta of xpip(4)=xpip(3)(1+2delta)
	    dt3t4 = -2*ai(4)**2*xpip(3)
	endif

   70	continue
	if ( lwrite ) print '(a)','  ##] transform momenta:'
*
*	If we lost too much accuracy try the other root...
*	(to do: build in a mechanism for remembering this later)
*
	if ( init .eq. 0 ) then
	    init = 1
*	    go ahead if we have half the digits left
	    maxlos = -int(log10(precx))/2
	    if ( lwrite ) print *,'ffxd0a: redo trans if loss > ',maxlos
	endif
	if ( ier0-ier .gt. maxlos ) then
	    if ( itime .eq. 1 ) then
		itime = 2
		if ( ier0-ier .ge. 100 ) itime = 100
		isgnal = -isgnal
		if ( lwrite ) print *,'ffxd0a: trying other root, ier=',
     +			ier0
		goto 25
	    else
		if ( ier0-ier .lt. 100 ) then
*		    it does not make any sense to go on, but do it anyway
		    if ( lwrite ) print *,'ffxd0a:  both roots rotten ',
     +			'going on'
		elseif ( itime.eq.100 ) then
		    if ( lwrite ) print *,'ffxd0a:  both roots rotten ',
     +			'giving up'
		    call fferr(72,ier)
		    cfac = 1
		    return
		elseif ( itime.le.2 ) then
*		    the first try was better
		    isgnal = -isgnal
		    itime = 3
		    goto 25
		endif
	    endif
	endif
	ier = ier0
*  #] transform the masses and momenta:
*  #[ calculations:
	call ffxd0p(cs4,ipi12,isoort,cfac,xpip,dpipjp,piDpjp,
     +		xqi,dqiqj,qiDqj,ai,daiaj,ldel2s,ier)
	xfac = -ai(1)*ai(2)*ai(3)*ai(4)/dt3t4
*
*	see the note at the end of this section about the sign
*
	if ( DIMAG(cfac) .eq. 0 ) then
	    cfac = xfac/DBLE(cfac)
	else
	    cfac = DBLE(xfac)/cfac
	endif
*
*	sum'em up:
*
	cs3(1) = 0
	cs3(2) = 0
	xmax = 0
	do 110 i=1,80
	    cs3(1) = cs3(1) + cs4(i)
	    xmax = max(xmax,absc(cs3(1)))
  110	continue
	do 111 i=81,160
	    cs3(2) = cs3(2) + cs4(i)
	    xmax = max(xmax,absc(cs3(2)))
  111	continue
	cs = cs3(1) - cs3(2)
	do 112 i=161,175
	    cs = cs + cs4(i)
	    xmax = max(xmax,absc(cs))
  112	continue
	ipi123(1) = 0
	ipi123(2) = 0
	do 113 i=1,8
	    ipi123(1) = ipi123(1) + ipi12(i)
  113	continue
	do 114 i=9,16
	    ipi123(2) = ipi123(2) + ipi12(i)
  114	continue
	ipi12t = ipi123(1) - ipi123(2)
	do 120 i=17,26
	    ipi12t = ipi12t + ipi12(i)
  120	continue
	cs = cs + ipi12t*DBLE(pi12)
*
*	Check for a sum close to the minimum of the range (underflow
*	problems)
*
	if ( lwarn .and. absc(cs) .lt. xalogm/precc .and. cs .ne. 0 )
     +		call ffwarn(119,ier,absc(cs),xalogm/precc)
*
*	If the imaginary part is very small it most likely is zero
*	(can be removed, just esthetically more pleasing)
*
	if ( abs(DIMAG(cs)) .lt. precc*abs(DBLE(cs)) )
     +		cs = DCMPLX(DBLE(cs))
*
*	it is much nicer to have the sign of cfac fixed, say positive
*
	if ( DBLE(cfac) .lt. 0 .or. (DBLE(cfac) .eq. 0 .and. DIMAG(cfac)
     +		.lt. 0 ) ) then
	    cfac = -cfac
	    cs = -cs
	endif
*  #] calculations:
*  #[ debug:
	if(lwrite)then
*	    print *,'s3''s :'
*	    print *,' '
*	    print 1004,(cs4(i),cs4(i+20),cs4(i+40),cs4(i+60),i=1,20)
*	    print *,' '
*	    print 1004,(cs4(i+80),cs4(i+100),cs4(i+120),cs4(i+140),i=
*     +		1,20)
*	    print *,' '
	    print *,'C3:'
	    do i=1,80
		if ( cs4(i).ne.0 ) print '(i4,2g16.8)',i,cs4(i)
	    enddo
	    print *,'C4:'
	    do i=81,160
		if ( cs4(i).ne.0 ) print '(i4,2g16.8)',i,cs4(i)
	    enddo
	    print *,'Threepoint functions:'
	    print '(a,2g24.14,i3)','C3  = ',cs3(1),ipi123(1)
	    print '(a,2g24.14,i3)','C4  = ',cs3(2),ipi123(2)
	    print '(a,2g24.14,i3)','sum = ',cs3(1)-cs3(2),
     +		ipi123(1)-ipi123(2)
	    if ( ipi123(1) .ne. ipi123(2) ) print '(a,2g24.14)',
     +		'    = ',cs3(1)-cs3(2)+(ipi123(1)-ipi123(2))*DBLE(pi12)
	    print *,'Correction terms for Ai negative'
	    print 1013,(cs4(160+i),ipi12(16+i),
     +			cs4(161+i),ipi12(17+i),
     +			cs4(162+i),ipi12(18+i),i=1,4,3)
	    c = 0
	    j = 0
	    do 803 i=1,6
		j = j + ipi12(16+i)
		c = c + cs4(160+i)
  803	    continue
	    print '(a,2g24.14,i3)','sum = ',c,j
	    if ( j .ne. 0 ) print '(a,2g24.14)','    = ',
     +		c+j*DBLE(pi12)
	    print *,'S of ''t Hooft and Veltman'
	    print 1012,(cs4(166+i),ipi12(22+i),
     +			cs4(167+i),ipi12(23+i),i=1,3,2)
	    c = 0
	    j = ipi12(23)+ipi12(24)+ipi12(25)+ipi12(26)
	    do 804 i=1,6
		c = c + cs4(166+i)
  804	    continue
	    print '(a,2g24.14,i3)','sum = ',c,j
	    if ( j .ne. 0 ) print '(a,2g24.14)','    = ',
     +		c+j*DBLE(pi12)
*	    print *,' '
*	    print *,'ipi12: ',ipi12
*	    print *,'isoort:',isoort
	    print '(a,2g24.14,2i6)','som  : ',cs-ipi12t*DBLE(pi12),
     +		ipi12t,ier
	    if ( ipi12t .ne. 0 ) print '(a,2g24.14)','som = ',cs
	    print *,'fac  :',cfac
	    print *,'cd0  :',cs*cfac
 1012	    format(g12.6,1x,g12.6,i4,2x,g12.6,1x,g12.6,i4)
 1013	    format(g12.6,1x,g12.6,i4,2x,g12.6,1x,g12.6,i4,2x,g12.6,1x,
     +        g12.6,i4)
 1004	    format(g12.6,1x,g12.6,2x,g12.6,1x,g12.6,2x,g12.6,1x,g12.6,
     +        2x,g12.6,1x,g12.6)
	endif
*  #] debug:
*###] ffxd0e:
	end
*###[ ffxd0r:
	subroutine ffxd0r(cd0,xpi,ier)
***#[*comment:***********************************************************
*									*
*	Tries all 12 permutations of the 4pointfunction			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
	integer ier
	DOUBLE PRECISION xpi(13),xqi(13)
	DOUBLE COMPLEX cd0,cd0p
	integer inew(13,6),irota,ier0,ier1,i,j,icon,ialsav,init
	logical lcon
	parameter (icon=3)
	save inew,init,lcon
	include 'ff.h'
	data inew /1,2,3,4,5,6,7,8,9,10,11,12,13,
     +		   4,1,2,3,8,5,6,7,10,9,11,13,12,
     +		   3,4,1,2,7,8,5,6,9,10,11,12,13,
     +		   2,3,4,1,6,7,8,5,10,9,11,13,12,
     +		   4,2,3,1,10,6,9,8,7,5,12,11,13,
     +		   1,3,2,4,9,6,10,8,5,7,12,11,13/
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
	cd0 = 0
	ier0 = ier
	ier = 999
	ialsav = isgnal
	do 30 j = -1,1,2
	    do 20 irota=1,6
		do 10 i=1,13
		    xqi(inew(i,irota)) = xpi(i)
   10		continue
		ier1 = ier0
		ner = 0
		id = id + 1
		isgnal = ialsav
		print '(a,i1,a,i2)','---#[ rotation ',irota,': isgnal ',
     +			isgnal
		if (lcon) write(icon,'(a,i1,a,i2)')'rotation ',irota,',
     +			isgnal ',isgnal
		call ffxd0(cd0p,xqi,ier1)
		ier1 = ier1 + ner
		print '(a,i1,a,i2,a)','---#] rotation ',irota,
     +			': isgnal ',isgnal,' '
		print '(a,2g28.16,i3)','d0 = ',cd0p,ier1
		if (lcon) write(icon,'(a,2g28.16,i3)')'d0 = ',cd0p,ier1
		if ( ier1 .lt. ier ) then
		    cd0 = cd0p
		    ier = ier1
		endif
   20	    continue
	    ialsav = -ialsav
   30	continue
*  #] calculations:
*###] ffxd0r:
	end
*###[ ffxd0d:
	subroutine ffxd0d(cd0,xpi,piDpj,del3p,del4s,info,ier)
***#[*comment:***********************************************************
*									*
*	Entry point to the four point function with dotproducts given.	*
*	Necessary to avoid cancellations near the borders of phase	*
*	space.								*
*									*
*	Input:	xpi(13)	      real	1-4: mi^2, 5-10: pi^2,s,t	*
*					optional: 11:u, 12:v, 13:w	*
*		info	      integer	0: no extra info		*
*					1: piDpj(i,j), i,j>4 is defined	*
*					2: del3p is also defined	*
*					3: all piDpj are given		*
*					4: del4s is also given		*
*		piDpj(10,10)  real	pi.pj in B&D metric;		*
*					1-4:si.sj=(m_i^2+m_j^2-p_ij^2)/2*
*					cross: si.pjk=si.pj-si.pk	*
*					5-10: pi.pj			*
*		del3p	      real	det(pi.pj)			*
*		del4s	      real	det(si.sj) (~square overall fac)*
*		ier	      integer	#digits accuracy lost in input	*
*	Output:	cd0	      complex	D0				*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer info,ier
	DOUBLE PRECISION xpi(13),piDpj(10,10),del3p,del4s
	DOUBLE COMPLEX cd0
*
*	local vars
*
	integer i,j
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ hide information in common blocks:
*
	idot = info
	if ( idot.ne.0 ) then
	    if ( idot.gt.0 .and. idot.le.2 ) then
		do 20 i=5,10
		    do 10 j=5,10
			fpij4(j,i) = piDpj(j,i)
   10		    continue
   20		continue
	    elseif ( idot.ge.3 ) then
		do 40 i=1,10
		    do 30 j=1,10
			fpij4(j,i) = piDpj(j,i)
   30		    continue
   40		continue
	    endif
	    if ( abs(idot).ge.2 ) then
		fdel3 = del3p
	    endif
	    if ( abs(idot).ge.4 ) then
		fdel4s = del4s
	    endif
	endif
*
*  #] hide information in common blocks:
*  #[ call ffxd0:
*
	call ffxd0(cd0,xpi,ier)
*
*	invalidate all the common blocks for the next call
*
	idot = 0
*
*  #] call ffxd0:
*###] ffxd0d:
	end
*###[ ffdif4:
	subroutine ffdif4(dpipj,luvw,xpi,ier)
***#[*comment:***********************************************************
*									*
*	Compute the elements 11-13 in xpi and the differences dpipj	*
*	Note that the digits lost in dpipj are not counted towards	*
*	the total.							*
*									*
*	Input:	xpi(1:10)	real		masses, momenta^2	*
*									*
*	Output:	xpi(11:13)	real		u and similar vars v,w	*
*		luvw(3)		logical		TRUE if xpi(10+i) has	*
*						been computed here	*
*		dpipj(10,13)	real		xpi(i) - xpi(j)		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	logical luvw(3)
	DOUBLE PRECISION xpi(13),dpipj(10,13)
*
*	local variables
*
	integer i,j,ier1,ier0
	DOUBLE PRECISION xmax
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ get differences:
*	simulate the differences in the masses etc..
	if ( lwrite ) print *,'ffdif4: input xpi: ',xpi
	ier1 = ier
	if ( xpi(11)  .eq. 0 ) then
	    xpi(11) = xpi(5)+xpi(6)+xpi(7)+xpi(8)-xpi(9)-xpi(10)
	    if ( lwarn ) then
		xmax = max(abs(xpi(5)),abs(xpi(6)),abs(xpi(7)),
     +		       abs(xpi(8)),abs(xpi(9)),abs(xpi(10)))
		if ( abs(xpi(11)) .lt. xloss*xmax )
     +		    call ffwarn(153,ier1,xpi(11),xmax)
	    endif
	    luvw(1) = .TRUE.
	else
	    luvw(1) = .FALSE.
	endif
	if ( xpi(12)  .eq. 0 ) then
	    xpi(12) = -xpi(5)+xpi(6)-xpi(7)+xpi(8)+xpi(9)+xpi(10)
	    if ( lwarn ) then
		ier0 = ier
		xmax = max(abs(xpi(5)),abs(xpi(6)),abs(xpi(7)),
     +		       abs(xpi(8)),abs(xpi(9)),abs(xpi(10)))
		if ( abs(xpi(12)) .lt. xloss*xmax )
     +		    call ffwarn(154,ier0,xpi(12),xmax)
		ier1 = max(ier1,ier0)
	    endif
	    luvw(2) = .TRUE.
	else
	    luvw(2) = .FALSE.
	endif
	if ( xpi(13)  .eq. 0 ) then
	    if ( max(abs(xpi(5)),abs(xpi(7))) .gt.
     +		 max(abs(xpi(9)),abs(xpi(10))) ) then
		xpi(13) = -xpi(12) + 2*(xpi(9)+xpi(10))
	    else
		xpi(13) = -xpi(11) + 2*(xpi(5)+xpi(7))
	    endif
*	    xpi(13) = xpi(5)-xpi(6)+xpi(7)-xpi(8)+xpi(9)+xpi(10)
	    if ( lwarn ) then
		ier0 = ier
		xmax = 2*min(max(abs(xpi(5)),abs(xpi(7))),
     +			     max(abs(xpi(9)),abs(xpi(10))))
		if ( abs(xpi(13)) .lt. xloss*xmax )
     +		    call ffwarn(155,ier0,xpi(13),xmax)
		ier1 = max(ier1,ier0)
	    endif
	    luvw(3) = .TRUE.
	else
	    luvw(3) = .FALSE.
	endif
	if ( lwarn ) then
	    do 10 i=1,13
		if ( i .le. 10 ) dpipj(i,i) = 0
		do 9 j=1,min(i-1,10)
		    dpipj(j,i) = xpi(j) - xpi(i)
		    if ( i .le. 10 ) then
			dpipj(i,j) = -dpipj(j,i)
		    endif
*		    we do not need the differences of s,t,u,v,w accurately
		    if ( i .gt. 8 .and. j .gt. 8 ) goto 9
		    if ( abs(dpipj(j,i)) .lt. xloss*abs(xpi(i))
     +					.and. xpi(i) .ne. xpi(j) ) then
			ier0 = ier
			call ffwarn(121,ier0,dpipj(j,i),xpi(i))
			if ( lwrite ) print *,'between xpi(',i,
     +				') and xpi(',j,')'
		    endif
    9		continue
   10	    continue
	    ier = ier1
	else
	    do 20 i=1,13
		do 19 j=1,10
		    dpipj(j,i) = xpi(j) - xpi(i)
   19		continue
   20	    continue
	endif
*  #] get differences:
*###] ffdif4:
	end
