
* file aaxbx.for  16-jul-1990

*###[ aaxbx :
	subroutine aaxbx(caxi,cbxi,acbxi,d0,xmu,xp,xma,xmb,level,ier)
***#[ comment:***********************************************************
*									*
*	Calculation of two point formfactors.				*
*	Calls ffxb0, ffxb1, ffxb2p, ffxdb0.				*
*									*
*	Input:	xp,xma,xmb   real	p^2 (B&D), ma^2, mb^2		*
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
	DOUBLE PRECISION xp,xma,xmb,d0,xmu
	DOUBLE COMPLEX caxi(2),cbxi(4),acbxi(2)
*
*	local variables
*
	DOUBLE PRECISION maxi(2),mbxi(4),mabxi(2)
*  #] declarations:
*  #[ call ffxbx:
	call ffxbx(caxi,maxi,cbxi,mbxi,acbxi,mabxi,d0,xmu,xp,xma,xmb,
     +		level,ier)
*  #] call ffxbx:
*###] aaxbx :
	end
*###[ ffxbx :
	subroutine ffxbx(caxi,maxi,cbxi,mbxi,acbxi,mabxi,
     +		d0,xmu,xp,xma,xmb,level,ier)
***#[ comment:***********************************************************
*									*
*	Calculation of two point formfactors with more accurate errors	*
*	Calls ffxb0, ffxb1, ffxb2p, ffxdb0.				*
*									*
*	Input:	xp,xma,xmb   real	p^2 (B&D), ma^2, mb^2		*
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
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION maxi(2),mbxi(4),mabxi(2),xp,xma,xmb,d0,xmu
	DOUBLE COMPLEX caxi(2),cbxi(4),acbxi(2)
*
*	local variables
*
	integer i,ier0,ier1
	DOUBLE PRECISION big
	DOUBLE COMPLEX acb0p,absc
	DOUBLE COMPLEX cc
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
*
*  #] declarations :
*  #[ level 0 : B0
*
*	B0
*
	ldot = .TRUE.
	ier1 = ier
	call ffxb0(cbxi(1),d0,xmu,xp,xma,xmb,ier1)
*	note that this may be off by a fctor 1/xloss
	mbxi(1) = absc(cbxi(1))*DBLE(10)**mod(ier1,50)
	if ( awrite ) then
	    print *,' '
	    print *,'ffxbx : level 0: id,nevent ',id,nevent
	    print *,'B0 =',cbxi(1),mbxi(1),ier1
	endif
	if (level .eq. 0  .and. .NOT. aderiv ) goto 990
*  #] level 0 :
*  #[ level 1/2 : B0':
	if (aderiv) then
	    ier0 = ier
	    call ffxdb0(acbxi(1),acb0p,xp,xma,xmb,ier0)
	    mabxi(1) = absc(acbxi(1))*DBLE(10)**mod(ier0,50)
	    ier1 = max(ier1,ier0)
	    if ( lwarn .and. atest ) then
		if ( abs(xp*acbxi(1)-acb0p) .gt. precc*abs(acb0p) )
     +		    print *,'ffxbx: error: B0'' not consistent: ',
     +		    xp*acbxi(1),acb0p,xp*acbxi(1)-acb0p,ier0
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
	call ffxa0(caxi(1),d0,xmu,xma,ier0)
	maxi(1) = absc(caxi(1))*DBLE(10)**mod(ier0,50)
	ier1 = max(ier1,ier0)
	ier0 = ier
	call ffxa0(caxi(2),d0,xmu,xmb,ier0)
	maxi(2) = absc(caxi(2))*DBLE(10)**mod(ier0,50)
	ier1 = max(ier1,ier0)
	ier = ier1
*
*	and get the B11
*
	call ffxb1(cbxi(2),cbxi(1),caxi,xp,xma,xmb,fpij2,ier1)
	mbxi(2) = absc(cbxi(2))*DBLE(10)**mod(ier1,50)
*
*	debug output
*
	if ( awrite ) then
	    print *,' '
	    print *,'ffxbx : level 1: id,nevent ',id,nevent
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
	call ffxb2p(cbxi(3),cbxi(2),cbxi(1),caxi,xp,xma,xmb,fpij2,ier1)
	mbxi(3) = absc(cbxi(3))*DBLE(10)**mod(ier1,50)
	mbxi(4) = absc(cbxi(4))*DBLE(10)**mod(ier1,50)
*
*	debug output
*
	if ( awrite ) then
	    print *,' '
	    print *,'ffxbx : level 2: id,nevent ',id,nevent
	    print *,'B21 = ',cbxi(3),ier1
	    print *,'B22 = ',cbxi(4),ier1
	endif
*
	if (level .eq. 2) goto 990
*
*  #] level 2 :
	print *,'ffxbx: error: level ',level,' not supported'
	stop

  990	continue
	ier = max(ier1,ier)
*###] ffxbx :
	end
