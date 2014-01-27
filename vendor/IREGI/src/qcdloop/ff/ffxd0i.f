*###[ ffx2ir:
	subroutine ffx2ir(cs1,cs2,xpip,dpipjp,ier)
***#[*comment:***********************************************************
*									*
*	Get the terms to correct for the second IR pole which is	*
*	treated incorrectly if the first one is regulated with a small	*
*	mass lam and they are adjacent.  It is assumed that xpi(3)=	*
*	xpi(4)=xpi(7)=0, xpi(1)=xpi(8), xpi(2)=xpi(6).  The correction	*
*	terms are							*
*									*
*	cs1 = -C0(m2^2,0,lam^2;m2^2,0,p10^2)/(s-m1^2)			*
*	cs2 = +C0(m2^2,lam^2,0;m2^2,0,p10^2)/(s-m1^2)			*
*									*
*	when xpi(4)=lam^2=delta is taken in the D0,			*
*									*
*	cs1 = -C0(lam^2,0,m1^2;0,m1^2,p9^2)/(t-m2^2)			*
*	cs2 = +C0(0,lam^2,m1^2;0,m1^2,p9^2)/(t-m2^2)			*
*									*
*	when xpi(3)=lam^2.  Not yet tested.				*
*									*
*				10-oct-1991 Geert Jan van Oldenborgh	*
*									*
*	Input:	xpip(13)	(real)	usual 4point pi.pi		*
*		dpipjp(10,13)	(real)	xpip(i) - xpip(j)		*
*	output:	xpip(13)	(real)	usual 4point pi.pi modified	*
*		dpipjp(10,13)	(real)	xpip(i) - xpip(j)  modified	*
*		cs1,cs2		(complex)				*
*		ier		(integer)				*
*	calls:	ffxc0							*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cs1,cs2
	DOUBLE PRECISION xpip(13),dpipjp(10,13)
*
*	local vars
*
	integer itest,ier0,ier1,i,j,iinx(6,4)
	DOUBLE COMPLEX cc0
	DOUBLE PRECISION xpi3(6),dpipj3(6,6)
	save itest,iinx
*
*	common
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	data
*
*	3=put mass on xpi(3)
*	4=put mass on xpi(4)
	data itest /4/
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
*
*  #] declarations: 
*  #[ check input:
	if ( ltest ) then
	    if ( xpip(3).ne.0 .or. xpip(4).ne.0 .or. xpip(7).ne.0 )
     +		print *,'ffx2ir: wrong input: vertex (3,4,7) not all 0',
     +		xpip(3),xpip(4),xpip(7)
	    ier0 = 0
	    call ffxhck(xpip,dpipjp,10,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffx2ir: error: input wrong'
	endif
*  #] check input: 
*  #[ work 3:
	if ( itest .eq. 3 ) then
	if ( lwrite ) then
	    print *,'ffx2ir: giving xpi(3) a mass ',delta
	endif
*
*	modify xpip,dpipjp
*
	xpip(3) = delta
	do 10 i=1,10
	    dpipjp(i,3) = dpipjp(i,3) - delta
   10	continue
	do 20 i=1,13
	    dpipjp(3,i) = dpipjp(3,i) + delta
   20	continue
*
*	call first C0
*
	do 120 i=1,6
	    xpi3(i) = xpip(iinx(i,2))
	    do 110 j=1,6
		dpipj3(j,i) = dpipjp(iinx(j,2),iinx(i,2))
  110	    continue
  120	continue
	idsub = idsub + 1
	ier1 = 0
	if ( lwrite ) print *,'ffx2ir: calling first C0'
	call ffxc0a(cc0,xpi3,dpipj3,ier1)
	cs1 = -cc0/DBLE(dpipjp(9,2))
*
*	call second C0
*
	xpi3(2) = 0
	xpi3(3) = delta
	do 130 i=1,6
	    dpipj3(i,2) = dpipj3(i,2) + delta
	    dpipj3(i,3) = dpipj3(i,3) - delta
  130	continue
	do 140 i=1,6
	    dpipj3(2,i) = dpipj3(2,i) - delta
	    dpipj3(3,i) = dpipj3(3,i) + delta
  140	continue
	idsub = idsub + 1
	ier0 = 0
	if ( lwrite ) print *,'ffx2ir: calling second C0'
	call ffxc0a(cc0,xpi3,dpipj3,ier0)
	cs2 = +cc0/DBLE(dpipjp(9,2))
	ier1 = max(ier1,ier0)
	ier = ier + ier1
*  #] work 3: 
*  #[ work 4:
	elseif ( itest .eq. 4 ) then
	if ( lwrite ) then
	    print *,'ffx2ir: giving xpi(4) a mass ',delta
	endif
*
*	modify xpip,dpipjp
*
	xpip(4) = delta
	do 210 i=1,10
	    dpipjp(i,4) = dpipjp(i,4) - delta
  210	continue
	do 220 i=1,13
	    dpipjp(4,i) = dpipjp(4,i) + delta
  220	continue
*
*	call first C0
*
	do 320 i=1,6
	    xpi3(i) = xpip(iinx(i,1))
	    do 310 j=1,6
		dpipj3(j,i) = dpipjp(iinx(j,1),iinx(i,1))
  310	    continue
  320	continue
	idsub = idsub + 1
	ier1 = 0
	if ( lwrite ) print *,'ffx2ir: calling first C0'
	call ffxc0a(cc0,xpi3,dpipj3,ier1)
	cs1 = -cc0/DBLE(dpipjp(10,1))
*
*	call second C0
*
	xpi3(3) = 0
	xpi3(2) = delta
	do 330 i=1,6
	    dpipj3(i,3) = dpipj3(i,3) + delta
	    dpipj3(i,2) = dpipj3(i,2) - delta
  330	continue
	do 340 i=1,6
	    dpipj3(3,i) = dpipj3(3,i) - delta
	    dpipj3(2,i) = dpipj3(2,i) + delta
  340	continue
	idsub = idsub + 1
	ier0 = 0
	if ( lwrite ) print *,'ffx2ir: calling second C0'
	call ffxc0a(cc0,xpi3,dpipj3,ier0)
	cs2 = +cc0/DBLE(dpipjp(10,1))
	ier1 = max(ier1,ier0)
	ier = ier + ier1
*  #] work 4: 
*  #[ error:
	else
	print *,'ffx2ir: error: itest should be either 3 or 4!',itest
	endif
*  #] error: 
*  #[ print:
	if ( lwrite ) then
	    print *,'   cs1 = ',cs1
	    print *,'   cs2 = ',cs2
	endif
*  #] print: 
*###] ffx2ir: 
	end
