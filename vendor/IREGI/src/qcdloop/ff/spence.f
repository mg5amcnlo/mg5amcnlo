*###[ SPENCE:
	DOUBLE COMPLEX function SPENCE(z)
***#[*comment:***********************************************************
*									*
*	Interface to the FF dilogarithms compatible with the FormF 	*
*	SPENCE function.  All error propagation is lost and the terms	*
*	pi^2/12 are added.						*
*									*
*	Input:	z	complex		cannot lie on the real axis for	*
*					Re(z)>1				*
*	Output:	SPENCE	complex		Sp(z) = Li2(z) = \sum z^n/n^2	*
*					= \int_0^z log(1-x)/x dx	*
*	Calls: 	ffzzdl							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE COMPLEX z
*
*	local variables
*
	integer init,ipi12,ier
	DOUBLE COMPLEX zdilog,zdum
	save init
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ initialisations:
	data init /0/
	if ( init .eq. 0 ) then
	    init = 1
	    call ffini
	endif
*  #] initialisations:
*  #[ work:
	ier = 0
	call ffzzdl(zdilog,ipi12,zdum,z,ier)
	SPENCE = zdilog + ipi12*pi12
*  #] work:
*###] SPENCE:
	end

