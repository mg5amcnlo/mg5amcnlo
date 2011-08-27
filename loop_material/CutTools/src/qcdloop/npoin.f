*###[ NPOIN:
	subroutine NPOIN(npoint)
***#[*comment:***********************************************************
*									*
*	entry point to the AA and FF routines compatible with Veltman's	*
*	NPOIN for FormF.						*
*									*
*	Input:	npoin	integer		specifies which function	*
*		DEL	real		infinity			*
*		PX(1-6)	real		momenta squared (Pauli metric)	*
*		RM(2-4)	real		masses squared			*
*									*
*	Output:	B0,B0PM,B1,B1PM,B2	complex		if npoint=2	*
*		C0,C1,C2,C3		complex		if npoint=3	*
*		D0,D1,D2,D3,D4		complex		if npoint=4	*
*		(all in blank common)					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer npoint
*
*	local variables
*
	integer init,i,l2,l3,l4,ier
	DOUBLE PRECISION xmu,xpc(6),xpd(13)
	DOUBLE COMPLEX cab(2),cbi(4),acbi(2),cac(3),cbc(12),cci(13),
     +		cbd(12),ccd(28),cdi(24)
	save init,l2,l3,l4
*
*	common blocks
*
	DOUBLE COMPLEX B0,B0PM,B1,B1PM,B2,CC0,CC1,CC2,CC3,D0,D1,D2,D3,D4
	DOUBLE PRECISION PX(6),RM(4),DEL
	common PX,RM,DEL,
     +		B0,B0PM,B1,B1PM,B2(2),CC0,CC1(2),CC2(4),CC3(6),
     +		D0,D1(3),D2(7),D3(13),D4(22)
	include 'ff.h'
	include 'aa.h'
*
*	data
*
	data xmu /0.D0/
	data l2,l3,l4 /2,3,3/
	data init /0/
*  #] declarations:
*  #[ initialisations:
	if ( init.eq.0 ) then
	    init = 1
	    do 10 i=1,22
		D4(i) = 0
   10	    continue
	    print *,'NPOIN: warning: D4 is not yet supported'
	    print *,'NPOIN: warning: B1'' seems also not yet supported'
	    call ffini
	endif
	ier = 0
	nevent = nevent + 1
*  #] initialisations:
*  #[ 2point:
	if ( npoint.eq.2 ) then
	    aderiv = .TRUE.
	    call aaxbx(cab,cbi,acbi,del,xmu,-PX(1),RM(1),RM(2),l2,ier)
      	    B0     = cipi2*cbi(1)
      	    B1     = cipi2*cbi(2)
      	    B2(1)  = cipi2*cbi(3)
      	    B2(2)  =-cipi2*cbi(4)
	    B0PM   = cipi2*acbi(1)
	    B1PM   = cipi2*acbi(2)
*  #] 2point:
*  #[ 3point:
	elseif ( npoint.eq.3 ) then
	    xpc(1) = RM(1)
	    xpc(2) = RM(2)
  	    xpc(3) = RM(3)
 	    xpc(4) =-PX(1)
  	    xpc(5) =-PX(2)
  	    xpc(6) =-PX(5)
  	    call aaxcx(cac,cbc,cci,del,xmu,xpc,l3,ier)
  	    CC0     =-cipi2*cci(1)
  	    CC1(1)  =-cipi2*cci(2)
  	    CC1(2)  =-cipi2*cci(3)
  	    CC2(1)  =-cipi2*cci(4)
  	    CC2(2)  =-cipi2*cci(5)
  	    CC2(3)  =-cipi2*cci(6)
  	    CC2(4)  =+cipi2*cci(7)
  	    CC3(1)  =-cipi2*cci(8)
  	    CC3(2)  =-cipi2*cci(9)
  	    CC3(3)  =-cipi2*cci(10)
  	    CC3(4)  =-cipi2*cci(11)
  	    CC3(5)  =+cipi2*cci(12)
  	    CC3(6)  =+cipi2*cci(13)
*  #] 3point:
*  #[ 4point:
	elseif ( npoint.eq.4 ) then
	    xpd(1) = RM(1)
	    xpd(2) = RM(2)
	    xpd(3) = RM(3)
	    xpd(4) = RM(4)
	    xpd(5) =-PX(1)
	    xpd(6) =-PX(2)
	    xpd(7) =-PX(3)
	    xpd(8) =-PX(4)
	    xpd(9) =-PX(5)
	    xpd(10)=-PX(6)
	    xpd(11)= 0.D0
	    xpd(12)= 0.D0
	    xpd(13)= 0.D0
	    call aaxdx(cbd,ccd,cdi,del,xmu,xpd,l4,ier)
	    D0     = cipi2*cdi(1)
	    D1(1)  = cipi2*cdi(2)
	    D1(2)  = cipi2*cdi(3)
	    D1(3)  = cipi2*cdi(4)
	    D2(1)  = cipi2*cdi(5)
	    D2(2)  = cipi2*cdi(6)
	    D2(3)  = cipi2*cdi(7)
	    D2(4)  = cipi2*cdi(8)
	    D2(5)  = cipi2*cdi(9)
	    D2(6)  = cipi2*cdi(10)
	    D2(7)  =-cipi2*cdi(11)
	    D3(1)  = cipi2*cdi(12)
	    D3(2)  = cipi2*cdi(13)
	    D3(3)  = cipi2*cdi(14)
	    D3(4)  = cipi2*cdi(15)
	    D3(5)  = cipi2*cdi(16)
	    D3(6)  = cipi2*cdi(17)
	    D3(7)  = cipi2*cdi(18)
	    D3(8)  = cipi2*cdi(19)
	    D3(9)  = cipi2*cdi(20)
	    D3(10) = cipi2*cdi(21)
	    D3(11) =-cipi2*cdi(22)
	    D3(12) =-cipi2*cdi(23)
	    D3(13) =-cipi2*cdi(24)
*  #] 4point:
*  #[ finish:
	else
	    print *,'NPOIN: error: npoint should be 2,3 or 4; not ',
     +	    	npoint
	    stop
	endif
	if ( ier .gt. 10 ) then
	    print *,'NPOIN: warning: more than 10 digits lost: ',ier
	    print *,'npoint = ',npoint
	    print *,'RM = ',RM
	    print *,'PX = ',PX
	    if ( ltest ) call ffwarn(998,ier,x0,x0)
	endif
*  #] finish:
*###] NPOIN:
      	end
*###[ AA0:
	DOUBLE COMPLEX function AA0(XM,DEL)
***#[*comment:***********************************************************
*									*
*	provides an interface to FF compatible with FormF by M. Veltman	*
*									*
*	Input:	XM	real		mass				*
*		DEL	real		infinity			*
*									*
*	Output:	A0	complex						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE PRECISION XM,DEL
*
*	my variables
*
	DOUBLE COMPLEX ca0
	integer ier,init
	save init
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data init /0/
*  #] declarations:
*  #[ initialisations:
	if ( init .eq. 0 ) then
	    init = 1
	    call ffini
	endif
*  #] initialisations:
*  #[ calculations:
	nevent = nevent + 1
	ier = 0
	call ffxa0(ca0,DEL,x0,XM,ier)
	AA0 = -ca0*cipi2
*  #] calculations:
*###] AA0:
	end
*###[ ALIJ:
	DOUBLE PRECISION function ALIJ(P22,P12,P1P2,P20,P10,DELE,PM2)
	DOUBLE PRECISION P22,P12,P1P2,P20,P10,DELE,PM2
	print *,'ALIJ: error: not implemented'
*	stupid fort!
	ALIJ = 0
*###] ALIJ:
	end
