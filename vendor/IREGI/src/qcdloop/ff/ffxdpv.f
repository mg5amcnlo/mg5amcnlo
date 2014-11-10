*###[ ffxdpv:
	subroutine ffxdpv(cd0,cd1,cd2,cd3,cd4,xpi,degree,ier)
***#[*comment:***********************************************************
*									*
*	Compute the scalar and tensor functions D0-D(degree) in the	*
*	Passarino-Veltman scheme, i.e. with basis p1-p3 and d_(mu,nu)).	*
*									*
*	Input:	xpi(13)	(real)		1-4: mi^2, 5-10: p(i-4)^2	*
*					11-13: either 0 or u,v,w	*
*		degree	(integer)	0-4				*
*									*
*	Output:	ier			number of digits lost in the	*
*					least stable result		*
*		cd0	(complex)	D0				*
*	   only when degree>0:						*
*		cd1(3)	(complex)	coeffs of p1,p2,p3		*
*	   only when degree>1:						*
*		cd2(7)	(complex)	..				*
*	   only when degree>2:						*
*		cd3(13)	(complex)	...				*
*	   only when degree>3:						*
*		cd4(22)	(complex)	...				*
*									*
*	Note: at this moment (28-feb-1993) only D1 and D2 are coded.	*
*	I am undecided as yet about whether to include the Ci.		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer degree,ier
	DOUBLE PRECISION xpi(13)
	DOUBLE COMPLEX cd0,cd1(3),cd2(7),cd3(13),cd4(22)
*
*	local variables
*
	integer i,j,k,ier0,ier1,inx43(6,4),sgn43(6,4),isgnsa,idotsa
	DOUBLE PRECISION del2pi(4),d0,xmu,absc
	DOUBLE PRECISION h,del3sp(4),del2ij,xpi3(6),dl2pij(6,6)
	DOUBLE COMPLEX cd4pppp(3,3,3,3),cd4ppdel(3,3),cd4deldel,
     +		cd3ppp(3,3,3),cd3pdel(3),cd2pp(3,3),cd2del,
     +		cc0i(4),cb0ij(4,4),ca0i(4),cc
	save inx43,sgn43
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data inx43 /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
     	data sgn43 /+1,+1,+1,+1,+1,-1,
     + 		    +1,+1,+1,-1,+1,+1,
     + 		    +1,+1,+1,+1,+1,+1,
     + 		    +1,+1,+1,+1,+1,+1/
*
*	statement function
*	
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ write input:
	if ( lwrite ) then
	    print *,'ffxdpv: input:'
	    print *,'xpi  = ',xpi
	    print *,'degree ',degree
	endif
*  #] write input:
*  #[ get scalar functions and determinants:
*
	ldot = .TRUE.
	isgnsa = isgnal
*
*	the D0
*
	ier0 = ier
	call ffxd0(cd0,xpi,ier0)
	isgnal = isgnsa
	ier1 = ier0
*
*	the C0s
*
	do 40 i=1,4
	    do 30 j=1,6
		xpi3(j) = xpi(inx43(j,i))
*		distribute dotproducts
		do 25 k=1,6
		    fpij3(k,j) = fpij4(inx43(k,i),inx43(j,i))*
     +		    	sgn43(k,i)*sgn43(j,i)
   25		continue
   30	    continue
	    ier0 = ier
	    idotsa = idot
	    idot = max(idot,3)
	    call ffxc0(cc0i(i),xpi3,ier0)
	    idot = idotsa
	    isgnal = isgnsa
	    ier1 = max(ier1,ier0)
	    del2pi(i) = fdel2
   40	continue
*
*	the B0s
*
	if ( degree .lt. 2 ) goto 80
	do 60 i=1,3
	    do 50 j=i+1,4
		ier0 = ier
		call ffxb0(cb0ij(i,j),x0,x0,xpi(inx(i,j)),xpi(i),
     +			xpi(j),ier0)
		cb0ij(j,i) = cb0ij(i,j)
		ier1 = max(ier1,ier0)
   50	    continue
   60	continue
*
*	the A0s
*
	if ( degree .lt. 3 ) goto 80
	do 70 i=1,4
	    ier0 = ier
	    call ffxa0(ca0i(i),x0,x0,xpi(i),ier0)
	    ier1 = max(ier1,ier0)
   70	continue
   80	continue
	ier = ier1
*
*  #] get scalar functions and determinants:
*  #[ call ffxdi:
	call ffxdi(cd4pppp,cd4ppdel,cd4deldel, cd3ppp,cd3pdel,
     +		cd2pp,cd2del, cd1, dl2pij, cd0,cc0i,cb0ij,ca0i, fdel4s,
     +		fdel3, del2pi, xpi,fpij4, x0,x0, degree, ier)
*  #] call ffxdi:
*  #[ convert to PV conventions:
*
	ier1 = ier
	cd2(1) = cd2pp(1,1) - DBLE(del2pi(1))*cd2del
	if ( lwarn .and. absc(cd2(1)).lt.xloss*absc(cd2pp(1,1)) ) then
	    call ffwarn(229,ier1,absc(cd2(1)),absc(cd2pp(1,1)))
	endif
	cd2(2) = cd2pp(1,2) + DBLE(dl2pij(2,4))*cd2del
	if ( lwarn .and. absc(cd2(2)).lt.xloss*absc(cd2pp(1,2)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(2)),absc(cd2pp(1,2)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(3) = cd2pp(1,3) - DBLE(dl2pij(1,4))*cd2del
	if ( lwarn .and. absc(cd2(3)).lt.xloss*absc(cd2pp(1,3)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(3)),absc(cd2pp(1,3)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(4) = cd2pp(2,2) - DBLE(xpi(5)*xpi(7)-fpij4(5,7)**2)*cd2del
	if ( lwarn .and. absc(cd2(4)).lt.xloss*absc(cd2pp(2,2)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(4)),absc(cd2pp(2,2)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(5) = cd2pp(2,3) + DBLE(dl2pij(1,2))*cd2del
	if ( lwarn .and. absc(cd2(5)).lt.xloss*absc(cd2pp(2,3)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(5)),absc(cd2pp(2,3)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(6) = cd2pp(3,3) - DBLE(del2pi(4))*cd2del
	if ( lwarn .and. absc(cd2(6)).lt.xloss*absc(cd2pp(3,3)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(6)),absc(cd2pp(3,3)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(7) = DBLE(fdel3)*cd2del
*
*  #] convert to PV conventions:
*  #[ print output:
	if ( lwrite ) then
	    print *,'PV D1 = '
	    print '(6e20.13)',cd1
	    if ( degree .lt. 2 ) return
	    print *,'PV D2 = '
	    print '(6e20.13)',cd2
	endif
*  #] print output:
*###] ffxdpv:
	end
*###[ ffxdpd:
	subroutine ffxdpd(cd0,cd1,cd2,cd3,cd4,xpi,piDpj,del3p,del4s,
     +		info,degree,ier)
***#[*comment:***********************************************************
*									*
*	Compute the scalar and tensor functions D0-D(degree) in the	*
*	Passarino-Veltman scheme, i.e. with basis p1-p3 and d_(mu,nu)).	*
*									*
*	Input:	xpi(13)	     real	1-4: mi^2, 5-10: p(i-4)^2	*
*					11-13: either 0 or u,v,w	*
*		piDpj(10,10) real	dotproducts pi.pj		*
*		del3	     real	det(pi.pj)			*
*		info	     integer	0: piDpj, del3 invalid		*
*					1: piDpj(6:10,6:10) defined	*
*					2: del3p also			*
*					3: rest of piDpj (internal) also*
*					4: del4s = det(si.sj) also	*
*		degree	     integer	0-4: which tensor functions 	*
*									*
*	Output:	see ffxdpv
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer info,degree,ier
	DOUBLE PRECISION xpi(13),piDpj(10,10),del3p,del4s
	DOUBLE COMPLEX cd0,cd1(3),cd2(7),cd3(13),cd4(22)
*
*	local vars
*	
	integer i,j
*
*	common
*	
	include 'ff.h'
*
*  #] declarations:
*  #[ hide information in common blocks:
*
	idot = info
	if ( idot.ne.0 ) then
	    if ( idot.le.2 ) then
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
*  #[ call ffxdpv:
*
	call ffxdpv(cd0,cd1,cd2,cd3,cd4,xpi,degree,ier)
	idot = 0
*
*  #] call ffxdpv:
*###] ffxdpd:
	end
