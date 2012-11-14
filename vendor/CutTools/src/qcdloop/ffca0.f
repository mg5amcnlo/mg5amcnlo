*###[ ffca0:
	subroutine ffca0(ca0,d0,xmm,cm,ier)
***#[*comment:***********************************************************
*									*
*	calculates the one-point function (see 't Hooft and		*
*	Veltman) for complex mass					*
*									*
*	Input:	d0	(real) infinity, result of the			*
*			renormalization procedure, the final		*
*			answer should not depend on it.			*
*		xmm	(real) arbitrary mass2, the final answer	*
*			should not depend on this either.		*
*		cm	(complex) mass2, re>0, im<0.			*
*									*
*	Output:	ca0	(complex) A0, the one-point function,		*
*		ier	0 (OK)						*
*									*
*	Calls:	log.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX ca0,cm
	DOUBLE PRECISION d0,xmm
*
*	local variables
*
	DOUBLE COMPLEX cmu,clogm,c
	DOUBLE PRECISION absc,xm
*
*	common blocks etc
*
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*  #] declarations:
*  #[ the real case:
*	
*	adapted to log-and-pole scheme 25-mar-1992
*	
	if ( DIMAG(cm) .eq. 0 .or. nschem .lt. 7 ) then
	    xm = DBLE(cm)
	    call ffxa0(ca0,d0,xmm,xm,ier)
	    return
	endif
*  #] the real case:
*  #[ "calculations":
	if ( xmm .ne. 0 ) then
	    cmu = cm/DBLE(xmm)
	else
	    cmu = cm
	endif
	if ( absc(cmu) .gt. xclogm ) then
	    clogm = log(cmu)
	else
	    clogm = 0
	    if ( cmu .ne. c0 ) call fferr(1,ier)
	endif
	ca0 = - cm * ( clogm - 1 - DBLE(d0) )
*  #] "calculations":
*  #[ debug:
	if (lwrite) then
	    print *,'d0  = ',d0
	    print *,'xmm = ',xmm
	    print *,'cm  = ',cm
	    print *,'ca0 = ',ca0
	endif
*  #] debug:
*###] ffca0:
	end
*###[ ffxa0:
	subroutine ffxa0(ca0,d0,xmm,xm,ier)
***#[*comment:***********************************************************
*									*
*	calculates the one-point function (see 't Hooft and		*
*	Veltman) for real mass						*
*									*
*	Input:	d0	(real) infinity, result of the			*
*			renormalization procedure, the final		*
*			answer should not depend on it.			*
*		xmm	(real) arbitrary mass2, the final answer	*
*			should not depend on this either.		*
*		xm	(real) mass2,					*
*									*
*	Output:	ca0	(complex) A0, the one-point function,		*
*		ier	0 (ok)						*
*									*
*	Calls:	log.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX ca0
	DOUBLE PRECISION d0,xmm,xm
*
*	local variables
*
	DOUBLE PRECISION xmu,xlogm
*
*	common blocks etc
*
	
	include 'ff.h'
*  #] declarations:
*  #[ "calculations":
	if ( xmm .ne. 0 ) then
	    xmu = xm/xmm
	else
	    xmu = xm
	endif
	if ( xmu .gt. xalogm ) then
	    xlogm = log(xmu)
	else
	    xlogm = 0
	    if ( xmu .ne. 0 ) call fferr(2,ier)
	endif
	ca0 = -(xm*(xlogm - 1 - d0))
*  #] "calculations":
*  #[ debug:
	if (lwrite) then
	    print *,'d0  = ',d0
	    print *,'xmm = ',xmm
	    print *,'xm  = ',xm
	    print *,'ca0 = ',ca0
	endif
*  #] debug:
*###] ffxa0:
	end
*###[ ffza0:
	subroutine ffza0(za0,d0,xmm,cm,xm,ndiv,ier)
***#[*comment:***********************************************************
*									*
*	calculates the one-point function (see 't Hooft and		*
*	Veltman) for complex mass in some on-shell scheme		*
*									*
*	Input:	d0	(real) infinity, result of the			*
*			renormalization procedure, the final		*
*			answer should not depend on it.			*
*		xmm	(real) arbitrary mass2, the final answer	*
*			should not depend on this either.		*
*		cm	(complex) mass2, re>0, im<0.			*
*		xm	(real) mass2, used instead of cm if onshel=true	*
*		ndiv	(integer) if >0 return 0 (the number of 	*
*			divergences the A0 should contain)		*
*									*
*	Output:	za0	(complex) A0, the one-point function,		*
*		ier	0 (OK)						*
*									*
*	Calls:	log.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ndiv,ier
	DOUBLE COMPLEX za0,cm
	DOUBLE PRECISION d0,xmm,xm
*
*	common blocks etc
*
	include 'ff.h'
*
*  #] declarations:
*  #[ preliminaries:
*
*	as the A0 cannot contain any on-shell singularities, return 
*	zero when one asks for one.
*
	if ( onshel .and. ndiv .gt. 0 ) then
	    za0 = 0
	    return
	endif
*
*  #] preliminaries:
*  #[ "work":
	if ( nschem.lt.7 ) then
	    call ffxa0(za0,d0,xmm,xm,ier)
	else
	    call ffca0(za0,d0,xmm,cm,ier)
	endif
*  #] "work":
*###] ffza0:
	end

