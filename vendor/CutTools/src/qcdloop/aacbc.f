
* file aacbc.for  16-jul-1990

*###[ aacbc :
	subroutine aacbc(caxi,cbxi,acbxi,d0,xmu,cp,cma,cmb,level,ier)
***#[ comment:***********************************************************
*									*
*	Calculation of two point formfactors for complex arguments.	*
*	Calls ffcb0, ffcb1, ffcb2p, ffcdb0.				*
*									*
*	Input:	cp,cma,cmb   complex	p^2 (B&D), ma^2, mb^2		*
*		d0,xmu	     real	renormalisation constants	*
*		level	     integer	rank of tensor(integral)	*
*	/aaflag/aderiv	     logical	whether or not to compute B0'	*
*									*
*	Output:	caxi(2)	     complex	A0(i) (with ma, mb resp.)	*
*		cbxi(4)	     complex	B0,B11,B21,B22			*
*					B1 = B11*p			*
*					B2 = B21*p*p + B21*g		*
*		acbxi(2)     complex	B0',B11'(not computed)		*
*									*
***#] comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION d0,xmu
	DOUBLE COMPLEX cp,cma,cmb
	DOUBLE COMPLEX caxi(2),cbxi(4),acbxi(2)
*
*	local variables
*
	DOUBLE PRECISION maxi(2),mbxi(4),mabxi(2)
*  #] declarations:
*  #[ call ffcbc:
	call ffcbc(caxi,maxi,cbxi,mbxi,acbxi,mabxi,d0,xmu,cp,cma,cmb,
     +		level,ier)
*  #] call ffcbc:
*###] aacbc :
	end
*###[ ffcbc :
	subroutine ffcbc(caxi,maxi,cbxi,mbxi,acbxi,mabxi,
     +		d0,xmu,cp,cma,cmb,level,ier)
***#[ comment:***********************************************************
*									*
*	Calculation of two point formfactors with more accurate errors	*
*	Calls ffcb0, ffcb1, ffcb2p, ffcdb0.				*
*									*
*	Input:	cp,cma,cmb   complex	p^2 (B&D), ma^2, mb^2		*
*		d0,xmu	     real	renormalisation constants	*
*		level	     integer	rank of tensor(integral)	*
*	/aaflag/aderiv	     logical	whether or not to compute B0'	*
*									*
*	Output:	caxi(2)	     complex	A0(i) (with ma, mb resp.)	*
*		maxi(2)	     real	maximal partial sum in A0i	*
*		cbxi(4)	     complex	B0,B11,B21,B22			*
*					B1 = B11*p			*
*					B2 = B21*p*p + B21*g		*
*		mbxi(4)	     real	maximal partial sum in B0...	*
*		acbxi(2)     complex	B0',B11'(not computed)		*
*		mabxi(2)     real	maximal partial sum in B0'	*
*									*
***#] comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION maxi(2),mbxi(4),mabxi(2),d0,xmu
	DOUBLE COMPLEX cp,cma,cmb
	DOUBLE COMPLEX caxi(2),cbxi(4),acbxi(2)
*
*	local variables
*
	integer i,ier0,ier1
	DOUBLE PRECISION big,absc,xma,xmb,xp
	DOUBLE COMPLEX acb0p,cc
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ really real?:
*
	if ( DIMAG(cma).eq.0 .and. DIMAG(cmb).eq.0 .and. DIMAG(cp).eq.0
     +		) then
	    xp  = DBLE(cp)
	    xma = DBLE(cma)
	    xmb = DBLE(cmb)
	    if ( awrite ) print *,'ffcbc: calling ffxbx'
	    call ffxbx(caxi,maxi,cbxi,mbxi,acbxi,mabxi,
     +		d0,xmu,xp,xma,xmb,level,ier)
	    return
	endif
*
*  #] really real?:
*  #[ init:
*
*	initialization to nonsense to prevent use of uncomputed vars
*
	big = 1/(1d20*xclogm)
	if ( ltest ) then
	    do 10 i=1,2
		caxi(i) = big
   10	    continue
	    do 11 i=1,4
		cbxi(i) = big
   11	    continue
	    do 12 i=1,2
		acbxi(i) = big
   12	    continue
	endif
*  #] init:
*  #[ level 0 : B0
*
*	B0
*
	ldot = .TRUE.
	ier1 = ier
	call ffcb0(cbxi(1),d0,xmu,cp,cma,cmb,ier1)
*	note that this may be off by a fctor 1/xloss
	mbxi(1) = absc(cbxi(1))*DBLE(10)**mod(ier1,50)
	if ( awrite ) then
	    print *,' '
	    print *,'ffcbc : level 0: id,nevent ',id,nevent
	    print *,'B0 =',cbxi(1),mbxi(1),ier1
	    print *,'cfpij2 = '
	    print '(6g12.6)',cfpij2
	endif
	if (level .eq. 0  .and. .NOT. aderiv ) goto 990
*  #] level 0 :
*  #[ level 1/2 : B0':
	if (aderiv) then
	    ier0 = ier
	    call ffcdb0(acbxi(1),acb0p,cp,cma,cmb,ier0)
	    mabxi(1) = absc(acbxi(1))*DBLE(10)**mod(ier0,50)
	    ier1 = max(ier1,ier0)
	    if ( lwarn .and. atest ) then
		if ( abs(cp*acbxi(1)-acb0p) .gt. precc*abs(acb0p) )
     +		    print *,'ffcbc: error: B0'' not consistent: ',
     +		    cp*acbxi(1),acb0p,cp*acbxi(1)-acb0p,ier0
	    endif
	    if ( awrite ) then
		print *,'AB0 =',acbxi(1),mabxi(1),ier0
		print *,'AB11= not yet implemented'
	    endif
	endif

	if ( level .eq. 0 ) return
*  #] level 1/2 : B0'
*  #[ level 1 : B11
*
*	first get the needed A0's
*
	ier0 = ier
	call ffca0(caxi(1),d0,xmu,cma,ier0)
	maxi(1) = absc(caxi(1))*DBLE(10)**mod(ier0,50)
	ier1 = max(ier1,ier0)
	ier0 = ier
	call ffca0(caxi(2),d0,xmu,cmb,ier0)
	maxi(2) = absc(caxi(2))*DBLE(10)**mod(ier0,50)
	ier1 = max(ier1,ier0)
	ier = ier1
*
*	and get the B11
*
	call ffcb1(cbxi(2),cbxi(1),caxi,cp,cma,cmb,cfpij2,ier1)
	mbxi(2) = absc(cbxi(2))*DBLE(10)**mod(ier1,50)
*
*	debug output
*
	if ( awrite ) then
	    print *,' '
	    print *,'ffcbc : level 1: id,nevent ',id,nevent
	    print *,'B11 = ',cbxi(2),mbxi(2),ier1
	    print *,'  A0(1) =',caxi(1),maxi(1)
	    print *,'  A0(2) =',caxi(2),maxi(2)
	endif
*
*	finished?
*
	if (level .eq. 1  ) goto 990
*
*  #] level 1 :
*  #[ level 2 : B21,B22
*
*	just a simple call...
*
	call ffcb2p(cbxi(3),cbxi(2),cbxi(1),caxi,cp,cma,cmb,cfpij2,ier1)
	mbxi(3) = absc(cbxi(3))*DBLE(10)**mod(ier1,50)
	mbxi(4) = absc(cbxi(4))*DBLE(10)**mod(ier1,50)
*
*	debug output
*
	if ( awrite ) then
	    print *,' '
	    print *,'ffcbc : level 2: id,nevent ',id,nevent
	    print *,'B21 = ',cbxi(3),ier1
	    print *,'B22 = ',cbxi(4),ier1
	endif
*
	if (level .eq. 2) goto 990
*
*  #] level 2 :
	print *,'ffcbc: error: level ',level,' not supported'
	stop

  990	continue
	ier = max(ier1,ier)
*###] ffcbc :
	end
