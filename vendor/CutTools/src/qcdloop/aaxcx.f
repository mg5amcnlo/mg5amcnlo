*###[ aaxcx :
	subroutine aaxcx(caxi,cbxi,ccxi,d0,xmm,xpi,level,ier)
***#[*comment:***********************************************************
*									*
*	Calculation of formfactors resulting from decvert.sub		*
*	or decvert.frm (up to third rank)				*
*	21-dec-1993: switched to ffxc1 for C1, added numerical checks.	*
*	Definitions:							*
*       C0 = 1/(i pi^2)*\int d^4 Q					*
*		1/((Q^2-m_1^2)((Q+p1)^2-m2^2)((Q-p3)^2-m3^2))		*
*	C1 = 1/(i pi^2)*\int d^n Q Q(mu)/(...)				*
*	   = C11*p1 + C12*p2						*
*	C2 = C21*p1*p1 + C22*p2*p2 + C23*(p1*p2+p2*p1) + C24*g		*
*	C3 = C31*p1*p1*p1 + C32*p2*p2*p2 + C33*(p1*p1*p2 + p1*p2*p1 +	*
*	     p2*p1*p1) + C34*(p1*p2*p2 + p2*p1*p2 + p1*p2*p2) + C35*	*
*	     (p1*g + g*p1 + 'g*p1*g') + C36*(p2*g + g*p2 + 'g*p2*g')	*
*									*
*	Input:	xpi	  the same as in Geert Jan's routines		*
*		level	  rank of tensor(integral)			*
*	Output: caxi(3)  : ca0i 		     i=1,2,3		*
*		cbxi(12) : (cb0i,cb11i,cb21i,cb22i)  i=1,2,3		*
*		ccxi(13) : cc0,cc1(2),cc2(4),cc3(6)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(6),d0,xmm
	DOUBLE COMPLEX caxi(3),cbxi(12),ccxi(13)
*
*	local variables
*
	DOUBLE PRECISION maxi(3),mbxi(12),mcxi(13)
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*  #] declarations :
*  #[ call ffxcx:
*
	call ffxcx(caxi,maxi,cbxi,mbxi,ccxi,mcxi,d0,xmm,xpi,level,ier)
*
*  #] call ffxcx:
*###] aaxcx :
	end
*###[ ffxcx:
	subroutine ffxcx(caxi,maxi,cbxi,mbxi,ccxi,mcxi,d0,xmm,xpi,level,
     +		ier)
***#[*comment:***********************************************************
*									*
*	Calculation of three point form factors with more accurate	*
*	error estimates.  Calls ffxc1, the rest is still here.		*
*	Definitions:							*
*       C0 = 1/(i pi^2)*\int d^4 Q					*
*		1/((Q^2-m_1^2)((Q+p1)^2-m2^2)((Q-p3)^2-m3^2))		*
*	C1 = 1/(i pi^2)*\int d^n Q Q(mu)/(...)				*
*	   = C11*p1 + C12*p2						*
*	C2 = C21*p1*p1 + C22*p2*p2 + C23*(p1*p2+p2*p1) + C24*g		*
*	C3 = C31*p1*p1*p1 + C32*p2*p2*p2 + C33*(p1*p1*p2 + p1*p2*p1 +	*
*	     p2*p1*p1) + C34*(p1*p2*p2 + p2*p1*p2 + p1*p2*p2) + C35*	*
*	     (p1*g + g*p1 + 'g*p1*g') + C36*(p2*g + g*p2 + 'g*p2*g')	*
*									*
*	Input:	xpi(6)	     real	m_i^2 (1:3), p_{i-3}^2 (4:6)	*
*		d0,xmu	     real	renormalisation constants	*
*		level	     integer	rank of tensor (integral)	*
*	Output: caxi(3)	     complex	A0(m_i^2)			*
*		maxi(3)	     real	max term in sum to caxi()	*
*		cbxi(12)     complex	3x(B0,B11,B21,B22)(p_i^2)	*
*		mbxi(12)     real	max term in sum to cbxi()	*
*		ccxi(13)     complex	C0,C1(2),C2(4),C3(6)		*
*		mcxi(13)     real	max term in sum to ccxi()	*
*	Note that if level<3 some of these are not defined.		*
*									*
***#]*comment:***********************************************************
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION maxi(3),mbxi(12),mcxi(13),xpi(6),d0,xmm
	DOUBLE COMPLEX caxi(3),cbxi(12),ccxi(13)
*
*	local variables
*
	integer i,bl,ier0,ier1
	logical adesav
	DOUBLE PRECISION absc,ma0i(6),mabxi(2),big
	DOUBLE COMPLEX acbxi(2),ca0i(6),cc,cc0
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
*  #] declarations :
*  #[ initialisations:
*
*	initialization to nonsense
*
	big = 1/(1d20*xclogm)
	if ( ltest ) then
	    do 10 i=1,3
		caxi(i) = big
   10	    continue
	    do 20 i=1,12
		cbxi(i) = big
   20	    continue
	    do 30 i=1,13
		ccxi(i) = big
   30	    continue
	endif
*
*  #] initialisations:
*  #[ get C0:
*
*	C0-function
*
	ldot=.TRUE.
	ier1 = ier
	call ffxc0(ccxi(1),xpi,ier1)
	if ( ier1.gt.10 ) then
	    if ( ltest ) then
		print *,'ffxcx: id = ',id,', nevent = ',nevent
		print *,'ffxcx: lost ',ier1,' digits in C0 with isgnal '
     +			,isgnal,', trying other roots, isgnal ',-isgnal
		print *,'       if OK (no further messages) adding this'
     +			,' to your code will improve speed'
	    endif
	    isgnal = -isgnal
	    ier0 = ier
	    call ffxc0(cc0,xpi,ier0)
	    isgnal = -isgnal
	    if ( ier0.lt.ier1 ) then
	    	ier1 = ier0
	    	ccxi(1) = cc0
	    endif
	endif
	if ( ier1 .gt. 10 ) then
	    print *,'ffxcx: id = ',id,', nevent = ',nevent
	    print *,'ffxcx: error: C0 not stable, lost ',ier1,' digits'
	    print *,'       please contact author (t19@nikhef.nl)'
	    print *,'xpi = ',xpi
	endif
*	note that we may have lost another factor xloss**3 or so
	mcxi(1) = absc(ccxi(1))*DBLE(10)**mod(ier1,50)
	if ( awrite ) then
*	 #[ for debugging: imported stuff from ff
	    print *,' '
	    print *,'ffxcx : level 0 '
	    print *,'C0 =',ccxi(1),mcxi(1),ier1
	    print *,'used:',( xpi(i),i=1,3 )
	    print *,'     ',( xpi(i),i=4,6 )
	    print *,'imported stuff via ff.h:'
	    print *,'kin det = ',fdel2
	    print *,'dotpr1,1= ',fpij3(4,4)
	    print *,'dotpr2,2= ',fpij3(5,5)
	    print *,'dotpr1,2= ',fpij3(4,5)
*	 #] for debugging:
	endif

	if ( level.eq.0 ) goto 990
*
*  #] get C0:
*  #[ need B-functions till b-level=(level-1):
	bl=level-1
	if ( awrite ) then
	    print '(a,i1)','  ##[ B-function output: up to level ',bl
	endif
	adesav = aderiv
	aderiv = .FALSE.
	ier0 = ier
	call ffxbx( ca0i(1),ma0i(1),cbxi(1),mbxi(1),acbxi,mabxi,
     +		d0,xmm,xpi(5),xpi(2),xpi(3),bl,ier0)
	ier1 = max(ier1,ier0)
	ier0 = ier
	call ffxbx( ca0i(3),ma0i(3),cbxi(5),mbxi(5),acbxi,mabxi,
     +		d0,xmm,xpi(6),xpi(1),xpi(3),bl,ier0)
	ier1 = max(ier1,ier0)
	ier0 = ier
	call ffxbx( ca0i(5),ma0i(5),cbxi(9),mbxi(9),acbxi,mabxi,
     +		d0,xmm,xpi(4),xpi(1),xpi(2),bl,ier0)
	ier1 = max(ier1,ier0)
	aderiv = adesav
	if ( awrite ) then
	    print '(a)','  ##] B-function output:'
	endif
*		symmetry in A0(i,j)
	caxi(1)=ca0i(1)
	caxi(2)=ca0i(2)
	caxi(3)=ca0i(3)
	maxi(1)=ma0i(1)
	maxi(2)=ma0i(2)
	maxi(3)=ma0i(3)
	if ( lwarn .and. atest ) then
	    if ((ca0i(4)-ca0i(2)) .ne. 0. .or.
     +		(ca0i(5)-ca0i(3)) .ne. 0. .or.
     +		(ca0i(6)-ca0i(1)) .ne. 0. ) then
		print *,'error in A0-calculations in aaxbx.for'
	    endif
	endif
*  #] need B-functions till b-level=(level-1):
*  #[ break to let ffzcz tie in:
	call ffxcxp(caxi,maxi,cbxi,mbxi,ccxi,mcxi,xpi,level,ier1)
*  #] break to let ffzcz tie in:
  990	ier = ier1
	end
	subroutine ffxcxp(caxi,maxi,cbxi,mbxi,ccxi,mcxi,xpi,level,ier)
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION maxi(3),mbxi(12),mcxi(13),xpi(6)
	DOUBLE COMPLEX caxi(3),cbxi(12),ccxi(13)
*
*	local variables
*
	integer i,j,ier1,ier2
	DOUBLE PRECISION xi3(3),f1,f2,absc,xmax,R1m,R2m,R3m,R4m,
     +		R5m,R6m,R11m,R12m,R13m,R14m,R15m,R16m,R17m,R18m
	DOUBLE PRECISION mb0i(3),mb11i(3),mxy(2),mb21i(3),mb22i(3)
	DOUBLE COMPLEX R1,R2,R3,R4,R5,R6,R11,R12,R13,R14,R15,R16,R17,
     +		R18,cb0i(3),cb11i(3),cb21i(3),cb22i(3),cc,cxy(2)
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
*  #] declarations :
*  #[ kinematical quantities for 3pv-red :
*	       inverse kinematical matrix xi3  (2X2)
	ier2 = ier
	call aaxi3(xi3,xpi,ier2)
	ier2 = ier2 - ier
*
*	       f-functions:
	f1 = 2*fpij3(1,4)
	f2 = 2*fpij3(1,5)
*  #] kinematical quantities for 3pv-red :
*  #[ level 1 : C11,C12,B0(I)
*		need 3 diff B0(I)-functions,I=1,2,3
	cb0i(1)=cbxi(1)
	cb0i(2)=cbxi(5)
	cb0i(3)=cbxi(9)
	mb0i(1)=mbxi(1)
	mb0i(2)=mbxi(5)
	mb0i(3)=mbxi(9)
	call ffxc1a(ccxi(2),mcxi(2),ccxi(1),mcxi(1),cb0i,mb0i,
     +		xpi,fpij3,fdel2,ier)
	if ( awrite ) then
	    print *,'GEERT JANs-scheme:'
	    print *,'C11=',ccxi(2),mcxi(2),ier
	    print *,'C12=',ccxi(3),mcxi(3),ier
	    print *,' '
	endif
	if ( lwarn .and. atest ) then
*		PV-reduction
	    R1=( f1*ccxi(1)+cb0i(2)-cb0i(1) )/2
	    R2=( f2*ccxi(1)+cb0i(3)-cb0i(2) )/2
	    R1m=max(abs(f1)*mcxi(1),mb0i(2),mb0i(1))/2
	    R2m=max(abs(f2)*mcxi(1),mb0i(3),mb0i(2))/2
	    cxy(1)=xi3(1)*R1+xi3(3)*R2
	    cxy(2)=xi3(3)*R1+xi3(2)*R2
	    mxy(1)=max(abs(xi3(1))*R1m,abs(xi3(3))*R2m)
	    mxy(2)=max(abs(xi3(3))*R1m,abs(xi3(2))*R2m)
	    if ( xloss*absc(ccxi(2)-cxy(1)) .gt. precc*
     +		max(mcxi(2),mxy(1)) )
     +		print *,'ffxcxp: error: FF C11 disagrees with PV: ',
     +		ccxi(2),cxy(1),ccxi(2)-cxy(1),ier
	    if ( xloss*absc(ccxi(3)-cxy(2)) .gt. precc*
     +		max(mcxi(3),mxy(2)) )
     +		print *,'ffxcxp: error: FF C12 disagrees with PV: ',
     +		ccxi(3),cxy(2),ccxi(3)-cxy(2),ier
	    if (awrite) then
		print *,' '
		print *,'ffxcxp : level 1: id,nevent ',id,nevent
		print *,'C11=',ccxi(2)
		print *,'C12=',ccxi(3)
	    endif
	endif
*
	if ( level.eq.1 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,3
		    if ( absc(ccxi(i)).ne.0 ) then
			xmax = max(xmax,mcxi(i)/absc(ccxi(i)))
		    elseif ( mcxi(i).ne.0 ) then
			xmax = max(xmax,1/precc)
		    endif
		enddo
		ier1 = int(log10(xmax))
		if ( awrite ) print *,'ier = ',ier1
	    else
		ier1 = 0
	    endif
	    goto 990
	endif
*
*  #] level 1 :
*  #[ level 2 : C21,C22,C23,C24,B11(I),A0(I,J)
*		need 3 diff B1-functions and 3 diff A0-fuctions
	do 12 i=1,3
	    j=(i+1)+(i-1)*3
	    cb11i(i) = cbxi(j)
	    mb11i(i) = mbxi(j)
   12	continue
*		PV-reduction
	ccxi(7)=1/4.d0 + 1/2.d0*xpi(1)*ccxi(1) -
     +		1/4.d0*( f1*ccxi(2)+f2*ccxi(3)-cb0i(1)  )
	ier1 = ier
	if ( lwarn ) then
***	    c7max = max(x1,2*xpi(1)*absc(ccxi(1)),absc(f1*ccxi(2)),
***     +		absc(f2*ccxi(3)),absc(cb0i(1)))/4
***	    if ( absc(ccxi(7)) .lt. xloss*c7max ) then
***		call ffwarn(293,ier1,absc(ccxi(7)),c7max)
***	    endif
	    mcxi(7) = max(x1,2*xpi(1)*mcxi(1),abs(f1)*mcxi(2),
     +		abs(f2)*mcxi(3),mb0i(1))/4
	endif
	R3=( f1*ccxi(2) + cb11i(2) + cb0i(1)  )/2 - ccxi(7)
	R4=( f2*ccxi(2) + cb11i(3) - cb11i(2) )/2
	R5=( f1*ccxi(3) + cb11i(2) - cb11i(1) )/2
	R6=( f2*ccxi(3)            - cb11i(2) )/2 - ccxi(7)
	ccxi(4)=xi3(1)*R3 + xi3(3)*R4
	ccxi(5)=xi3(3)*R5 + xi3(2)*R6
	ccxi(6)=xi3(3)*R3 + xi3(2)*R4
	if ( lwarn ) then
***	    R3m = max(absc(f1*ccxi(2)),absc(cb11i(2)),absc(cb0i(1)),
***     +		2*c7max)/2
***	    R4m = max(absc(f2*ccxi(2)),absc(cb11i(3)),absc(cb11i(2)))/2
***	    R5m = max(absc(f1*ccxi(3)),absc(cb11i(2)),absc(cb11i(1)))/2
***	    R6m = max(absc(f2*ccxi(3)),absc(cb11i(2)),2*c7max)/2
***	    xmax = max(abs(xi3(1))*R3m,abs(xi3(3))*R4m)
***	    if ( absc(ccxi(4)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(292,ier0,absc(ccxi(4)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi3(3))*R5m,abs(xi3(2))*R6m)
***	    if ( absc(ccxi(5)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(291,ier0,absc(ccxi(5)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi3(3))*R3m,abs(xi3(2))*R4m)
***	    if ( absc(ccxi(6)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(290,ier0,absc(ccxi(6)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
*
*	    for the whole chain
*
	    R3m = max(abs(f1)*mcxi(2),mb11i(2),mb0i(1),2*mcxi(7))/2
	    R4m = max(abs(f2)*mcxi(2),mb11i(3),mb11i(2))/2
	    R5m = max(abs(f1)*mcxi(3),mb11i(2),mb11i(1))/2
	    R6m = max(abs(f2)*mcxi(3),mb11i(2),2*mcxi(7))/2
	    mcxi(4) = max(abs(xi3(1))*R3m,abs(xi3(3))*R4m)
	    mcxi(5) = max(abs(xi3(3))*R5m,abs(xi3(2))*R6m)
	    mcxi(6) = max(abs(xi3(3))*R3m,abs(xi3(2))*R4m)
	endif
	if ( lwarn .and. atest ) then
	    cxy(1) = xi3(1)*R5 + xi3(3)*R6
	    mxy(1) = abs(xi3(1))*R5m + abs(xi3(3))*R6m
	    if ( xloss*absc(cxy(1)-ccxi(6)).gt.precc*max(mcxi(6),mxy(1))
     +			) then
		print *,'ffxcxp: error: id/nevent ',id,'/',nevent
		print *,'redundancy check at level 2 failed: '
		print *,cxy(1),mxy(1)
		print *,ccxi(6),mcxi(6)
		print *,absc(cxy(1)-ccxi(6))
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'ffxcxp : level 2: id,nevent ',id,nevent
	    print *,'C21=',ccxi(4),mcxi(4)
	    print *,'C22=',ccxi(5),mcxi(5)
	    print *,'C23=',ccxi(6),mcxi(6)
	    print *,'    ',cxy(1),mxy(1)
	    print *,'C24=',ccxi(7),mcxi(7)
	endif

	if ( level.eq.2 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,7
		    if ( absc(ccxi(i)).ne.0 ) then
			xmax = max(xmax,mcxi(i)/absc(ccxi(i)))
		    elseif ( mcxi(i).ne.0 ) then
			xmax = max(xmax,1/precc)
		    endif
		enddo
		ier1 = int(log10(xmax))
		if ( awrite ) print *,'ier = ',ier1
	    else
		ier1 = 0
	    endif
	    goto 990
	endif

*  #] level 2 :
*  #[ level 3 : C31,C32,C33,C34,C35,C36,B21(I),B22(I)
	do 13 i=1,3
	    j = (i+1)+(i-1)*3
	    cb21i(i)=cbxi(j+1)
	    cb22i(i)=cbxi(j+2)
	    mb21i(i)=mbxi(j+1)
	    mb22i(i)=mbxi(j+2)
   13	continue
*		PV-reduction
	R17=( f1*ccxi(7)+cb22i(2)-cb22i(1) )/2
	R18=( f2*ccxi(7)+cb22i(3)-cb22i(2) )/2
	ccxi(12)=xi3(1)*R17+xi3(3)*R18
	ccxi(13)=xi3(3)*R17+xi3(2)*R18
	if ( lwarn ) then
***	    R17m = max(abs(f1)*c7max,absc(cb22i(2)),absc(cb22i(1)))/2
***	    R18m = max(abs(f2)*c7max,absc(cb22i(3)),absc(cb22i(2)))/2
***	    c12max = max(abs(xi3(1))*R17m,abs(xi3(3))*R18m)
***	    if ( absc(ccxi(12)).lt.xloss*c12max ) then
***		ier0 = ier
***		call ffwarn(289,ier0,absc(ccxi(12)),c12max)
***		ier1 = max(ier1,ier0)
***	    endif
***	    c13max = max(abs(xi3(3))*R17m,abs(xi3(2))*R18m)
***	    if ( absc(ccxi(13)).lt.xloss*c13max ) then
***		ier0 = ier
***		call ffwarn(288,ier0,absc(ccxi(13)),c13max)
***		ier1 = max(ier1,ier0)
***	    endif
	    R17m = max(abs(f1)*mcxi(7),mb22i(2),mb22i(1))/2
	    R18m = max(abs(f2)*mcxi(7),mb22i(3),mb22i(2))/2
	    mcxi(12) = max(abs(xi3(1))*R17m,abs(xi3(3))*R18m)
	    mcxi(13) = max(abs(xi3(3))*R17m,abs(xi3(2))*R18m)
	endif
	R11=( f1*ccxi(4)+cb21i(2)-cb0i(1)  )/2 - 2*ccxi(12)
	R12=( f2*ccxi(4)+cb21i(3)-cb21i(2) )/2
	R13=( f1*ccxi(5)+cb21i(2)-cb21i(1) )/2
	R14=( f2*ccxi(5)         -cb21i(2) )/2 - 2*ccxi(13)
	R15=( f1*ccxi(6)+cb21i(2)+cb11i(1) )/2 - ccxi(13)
	R16=( f2*ccxi(6)         -cb21i(2) )/2 - ccxi(12)
	ccxi(8) =xi3(1)*R11 + xi3(3)*R12
	ccxi(9) =xi3(3)*R13 + xi3(2)*R14
	ccxi(10)=xi3(3)*R11 + xi3(2)*R12
	ccxi(11)=xi3(1)*R13 + xi3(3)*R14
	if ( lwarn ) then
***	    R11m = max(absc(f1*ccxi(4)),absc(cb21i(2)),absc(cb0i(1)),
***     +		2*c12max)/2
***	    R12m = max(absc(f2*ccxi(4)),absc(cb21i(3)),absc(cb21i(2)))/2
***	    R13m = max(absc(f1*ccxi(5)),absc(cb21i(2)),absc(cb21i(1)))/2
***	    R14m = max(absc(f2*ccxi(5)),absc(cb21i(2)),4*c13max)/2
***	    R15m = max(absc(f1*ccxi(6)),absc(cb21i(2)),absc(cb11i(1)),
***     +		2*c13max)/2
***	    R16m = max(absc(f2*ccxi(6)),absc(cb21i(2)),2*c12max)/2
***	    xmax = max(abs(xi3(1))*R11m,abs(xi3(3))*R12m)
***	    if ( absc(ccxi(8)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(287,ier0,absc(ccxi(8)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi3(3))*R13m,abs(xi3(2))*R14m)
***	    if ( absc(ccxi(9)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(286,ier0,absc(ccxi(9)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi3(3))*R11m,abs(xi3(2))*R12m)
***	    if ( absc(ccxi(10)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(285,ier0,absc(ccxi(10)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi3(1))*R13m,abs(xi3(3))*R14m)
***	    if ( absc(ccxi(11)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(284,ier0,absc(ccxi(11)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
*
*	    for the whole chain
*
	    R11m = max(abs(f1)*mcxi(4),mb21i(2),mb0i(1),2*mcxi(12))/2
	    R12m = max(abs(f2)*mcxi(4),mb21i(3),mb21i(2))/2
	    R13m = max(abs(f1)*mcxi(5),mb21i(2),mb21i(1))/2
	    R14m = max(abs(f2)*mcxi(5),mb21i(2),4*mcxi(13))/2
	    R15m = max(abs(f1)*mcxi(6),mb21i(2),mb11i(1),2*mcxi(13))/2
	    R16m = max(abs(f2)*mcxi(6),mb21i(2),2*mcxi(12))/2
	    mcxi(8) = max(abs(xi3(1))*R11m,abs(xi3(3))*R12m)
	    mcxi(9) = max(abs(xi3(3))*R13m,abs(xi3(2))*R14m)
	    mcxi(10)= max(abs(xi3(3))*R11m,abs(xi3(2))*R12m)
	    mcxi(11)= max(abs(xi3(1))*R13m,abs(xi3(3))*R14m)
	endif
*	redundancy check
	if ( lwarn .and. atest ) then
	    cxy(1) = xi3(1)*R15 + xi3(3)*R16
	    cxy(2) = xi3(3)*R15 + xi3(2)*R16
	    mxy(1) = abs(xi3(1))*R15m + abs(xi3(3))*R16m
	    mxy(2) = abs(xi3(3))*R15m + abs(xi3(2))*R16m
	    if ( xloss*absc(cxy(1)-ccxi(10)).gt.precc*max(mxy(1),
     +			mcxi(10))
     +	    .or. xloss*absc(cxy(2)-ccxi(11)).gt.precc*max(mxy(2),
     +			mcxi(11)) ) then
		print *,'ffxcxp: error: id/nevent ',id,'/',nevent
		print *,'redundancy check at level 3 failed: '
		print *,cxy(1),mxy(1)
		print *,ccxi(10),mcxi(10)
		print *,absc(cxy(1)-ccxi(10))
		print *,cxy(2),mxy(2)
		print *,ccxi(11),mcxi(11)
		print *,absc(cxy(1)-ccxi(11))
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'ffxcxp: level 3: id,nevent ',id,nevent
	    print *,'C31=',ccxi(8),mcxi(8)
	    print *,'C32=',ccxi(9),mcxi(9)
	    print *,'C33=',ccxi(10),mcxi(10)
	    print *,'    ',cxy(1),mxy(1)
	    print *,'C34=',ccxi(11),mcxi(11)
	    print *,'    ',cxy(2),mxy(2)
	    print *,'C35=',ccxi(12),mcxi(12)
	    print *,'C36=',ccxi(13),mcxi(13)
	endif

	if ( level.eq.3 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,13
		    if ( absc(ccxi(i)).ne.0 ) then
			xmax = max(xmax,mcxi(i)/absc(ccxi(i)))
		    elseif ( mcxi(i).ne.0 ) then
			xmax = max(xmax,1/precc)
		    endif
		enddo
		ier1 = int(log10(xmax))
		if ( awrite ) print *,'ier = ',ier1
	    else
		ier1 = 0
	    endif
	    goto 990
	endif

*  #] level 3 :
*  #[ end:
	print *,'ffxcxp: level ',level,' not supported.'
	stop
  990	continue
	ier = ier1 + ier2
*  #] end:
*###] ffxcx:
	end
