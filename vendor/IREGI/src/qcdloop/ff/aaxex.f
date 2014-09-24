
* file aaxex  23-sep-1990

*###[ aaxex:
	subroutine aaxex(ccxi,cdxi,cexi,d0,xmm,xpi,level,ier)
***#[*comment:***********************************************************
*									*
*	Calculation of four point tensor integrals.  Just a wrapper	*
*	for ffxdx nowadays, see there for the real description.		*
*									*
*	Input:	xpi(20)	  real	   the same as in FF			*
*		level	  integer  rank of tensor integral		*
*	Output:	ccxi(30)  complex  cc0(1),cc1( ),[cc2( ),cc3( ) ]  i,j	*
*		cdxi(55)  complex  cd0(1),cd1(3),cd2(7),[cd3(13)]	*
*					i=1,2,3,4,5			*
*		cexi(35)  complex  ce0(1),ce1(4),ce2(10),ce3(20)	*
*		ier	  integer  FF error flag			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(20),d0,xmm
	DOUBLE COMPLEX ccxi(30),cdxi(55),cexi(35)
*
*	local variables
*
	DOUBLE COMPLEX caxi(5),cbxi(30)
	DOUBLE PRECISION maxi(5),mbxi(30),mcxi(30),mdxi(55),mexi(35)
*
*  #] declarations:
*  #[ call ffxex:
*
	call ffxex(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,cexi,mexi,
     +		d0,xmm,xpi,level,ier)
*
*  #] call ffxex:
*###] aaxex:
	end
*###[ ffxex:
	subroutine ffxex(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,cexi,
     +		mexi,d0,xmm,xpi,level,ier)
***#[*comment:***********************************************************
*									*
*	Calculation of five-point formfactors, notation is defined in	*
*	adapt.prc and covdec5.prc (WW), leaving out the d_ terms	*
*	adapted by GJvO 4-apr-1995					*
*									*
*	cexi(1)		E0						*
*	cexi(2-5)       E1i, coeff of pi				*
*	cexi(6-15)	E2i, coeff of p1p1,p2p2,p3p3,p4p4,(p1p2),	*
*			(p1p3),(p1p4),(p2p3),(p2p4),(p3p4)		*
*	cexi(16-35)	E2i, coeff of p1p1p1,p2p2p2,p3p3p3,p4p4p4,	*
*			(p1p1p2),(p1p1p3),(p1p1p4),(p2p2p1),(p2p2p3),	*
*			(p2p2p4),(p3p3p1),(p3p3p2),(p3p3p4),(p4p4p1),	*
*			(p4p4p2),(p4p4p3),(p1p2p3),(p1p2p4),(p1p3p4),	*
*			(p2p3p4)					*
*	cdxi(55)	5*(D0(1),D1i(3),D2i(7)), s_i missing		*
*	ccxi(30)	15*(C0(1),C1i(2))  ??????			*
*	cbxi(30)	not used yet					*
*	caxi(5)		not used yet					*
*	m[abcde]xi()	if c[abcde]xi() were written as a sum of stable	*
*			terms, the largest term in that sum, i.e., the	*
*			accuracy of c[abcde]xi() is precc*m[abcde]xi	*
*									*
*	Input:	xpi(20)	  real	   the same as in FF			*
*		level	  integer  rank of tensor integral		*
*	Output:	ccxi(30)  complex  cc0(1),cc1( ),[cc2( ),cc3( ) ]  i,j	*
*		cdxi(55)  complex  cd0(1),cd1(3),cd2(7),[cd3(13)]	*
*					i=1,2,3,4,5			*
*		cexi(35)  complex  ce0(1),ce1(5),ce2(10),ce3(20)	*
*		ier	  integer  FF error flag			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(20),d0,xmm
	DOUBLE COMPLEX caxi(5),cbxi(30),ccxi(30),cdxi(55),cexi(35)
	DOUBLE PRECISION maxi(5),mbxi(30),mcxi(30),mdxi(55),mexi(35)
*
*	local variables
*
	integer i,j,dl,iplace(11,5),ier0,ier1
	DOUBLE PRECISION xpj(13),absc,big
        DOUBLE COMPLEX cd0i(5),cdxj(120),ccxj(140),cbxj(60),caxj(20),cc
	DOUBLE PRECISION mdxj(120),mcxj(140),mbxj(60),maxj(20)
	save iplace
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*	statement functions
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data iplace /
     +          2,3,4,5, 07,08,09,15, 12,13, 17,
     +          1,3,4,5, 11,08,09,10, 14,13, 18,
     +          1,2,4,5, 06,12,09,10, 14,15, 19,
     +          1,2,3,5, 06,07,13,10, 11,15, 20,
     +          1,2,3,4, 06,07,08,14, 11,12, 16/
*
*  #] declarations:
*  #[ init:
*
*	initialize to something ridiculous so that one immediately 
*	notices when it is accidentally used...
*
	big = 1/(1d20*xclogm)
	do i=1,5
	    caxi(i) = big
	enddo
	do i=1,30
	    cbxi(i) = big
	enddo
	do i=1,30
	    ccxi(i) = big
	enddo
	do i=1,55
	    cdxi(i) = big
	enddo
	do i=1,35
	    cexi(i) = big
	enddo
*
*  #] init:
*  #[ level 0: E0, and kinematical quantities for 5 point PV-red
*	E0-function (ff)
*
	ldot = .TRUE.
	ier1 = ier
	call ffxe0(cexi(1),cd0i,xpi,ier1)
	mexi(1) = absc(cexi(1))*DBLE(10)**mod(ier1,50)
	do i=1,5
	    cdxi(1+11*(i-1)) = cd0i(i)
	enddo
*
	if ( awrite ) then
	    print *,'    '
	    print *,'aaxex : level 0, imported from ff '
	    print *,'E0    = ',cexi(1),mexi(1)
	    print *,'D0(1) = ',cd0i(1)
	    print *,'D0(2) = ',cd0i(2)
	    print *,'D0(3) = ',cd0i(3)
	    print *,'D0(4) = ',cd0i(4)
	    print *,'D0(5) = ',cd0i(5)
	    print *,'xpi used:'
	    do i =1,15
		print *,i,xpi(i)
	    enddo
	    print *,'imported stuff via ff.h:'
	    print *,'    kin determinat = ',fdel4
	    print *,'dotpr(1,1)= ',fpij5(6,6)
	    print *,'dotpr(2,2)= ',fpij5(7,7)
	    print *,'dotpr(1,2)= ',fpij5(6,7)
	endif
*
	if (level .eq. 0) return
*
*  #] level 0: E0, and kinematical quantities for 5 point PV-red
*  #[ need D-functions till d-level=(level-1):
	dl=level-1
*
*	go trough the 5 different cancellation patterns
*
	if ( awrite ) then
	   print *,'     '
	   print *,'------>underlying D-functions up to level:',dl
	endif
	xpj(12) = 0
	xpj(13) = 0
	do j=1,5
*	    D(j) is the D0 with leg j missing.
	    do i=1,11
		xpj(i) = xpi(iplace(i,j))
	    enddo
*	    note that we recompute the D0 (or get it from cache)
	    ier0 = ier
	    call ffxdx(caxj(1+4*(j-1)),maxj(1+4*(j-1)),cbxj(1+12*(j-1))
     +		,mbxj(1+12*(j-1)),ccxj(1+28*(j-1)),mcxj(1+28*(j-1)),
     +		cdxj(1+24*(j-1)),mdxj(1+24*(j-1)),d0,xmm,xpj,dl,ier0)
	    ier1 = max(ier1,ier0)
	enddo
	ier = ier1
	if ( awrite ) then
	    print *,' '
	    print *,'ier = ',ier
	    print *,'---->end of D-function output--------------------'
	endif
	if ( atest ) then
*	    (although these should come from cache) (but don't yet!)
	    do i=1,5
		if ( xloss*10d0**(-mod(ier,50))*absc(cd0i(i)-
     +			cdxj(1+24*(i-1))) .gt. precx*absc(cd0i(i)) )
     +			then
		    print *,'aaxex: error: D0 does not agree with ',
     +			'recomputed: ',cd0i(i),cdxj(1+24*(i-1)),ier
		endif
	    enddo
	endif
*
*  #] need D-functions till d-level=(level-1):
*  #[ break to let ffzez tie in:
*
*	convert ??xj to ??xi
*
	call ffeji(cdxi,mcxi,ccxi,mcxi,cbxi,mbxi,caxi,maxi,
     +		cdxj,mdxj,ccxj,mcxj,cbxj,mbxj,caxj,maxj,level)
*
*	and call the real routine for the rest
*
	call ffxexp(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,cexi,mexi,
     +		xpi,level,ier)
*
*  #] break to let ffzez tie in:
	end
	subroutine ffxexp(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,cexi,
     +		mexi,xpi,level,ier)
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(20),d0,xmm
	DOUBLE COMPLEX caxi(5),cbxi(30),ccxi(30),cdxi(55),cexi(35)
	DOUBLE PRECISION maxi(5),mbxi(30),mcxi(30),mdxi(55),mexi(35)
*
*	local variables
*
	integer i,j,ier0,ier1,ij2k(4,4),m2ijk(3,20)
	DOUBLE PRECISION xi5(10),f1,f2,f3,f4,absc
        DOUBLE COMPLEX R(70),cd0i(5),cd1ij(3,5),ce2ij(4,4),ce3ijk(4,4,4)
     +		,cd2ijk(3,3,5),cd2i(5),cxy(70),cc,rg(4),cexj(39)
	DOUBLE PRECISION mdxj(120),mcxj(140),mbxj(60),maxj(20),
     +		del3ij(5,5),del4i(4)
	save ij2k,m2ijk
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*	statement functions
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data ij2k / 6,10,11,12,
     +		   10, 7,13,14,
     +		   11,13, 8,15,
     +		   12,14,15, 9/
	data m2ijk /1,1,1,  2,2,2,
     +		    3,3,3,  4,4,4,
     +		    1,1,2,  1,1,3,
     +		    1,1,4,  2,2,1,
     +		    2,2,3,  2,2,4,
     +		    3,3,1,  3,3,2,
     +		    3,3,4,  4,4,1,
     +		    4,4,2,  4,4,3,
     +		    1,2,3,  1,2,4,
     +		    1,3,4,  2,3,4/
*
*  #] declarations:
*  #[ kinematical quatities for 5pv-red:
*
*	inverse kinematical matrix xi5  (4X4)
*
	call aaxi5(xi5,ier)
*
*	AAs f-functions:
*
	f1 = 2*fpij5(1,6)
	f2 = 2*fpij5(1,7)
	f3 = 2*fpij5(1,8)
	f4 = 2*fpij5(1,9)
*
*  #] kinematical quatities for 5pv-red:
*  #[ level 1: E11,E12,E13,E14,D0(I)
	do i=1,5
	    cd0i(i) = cdxi(1+11*(i-1))
	enddo
	call ffxe1(cexi(2),cexi(1),del3ij,del4i,cd0i,xpi,fpij5,fdel4,
     +		ier)
	if ( atest ) then
*
*	PV-reduction
*
	    R(1) = (f1*cexi(1) + cd0i(2) - cd0i(1))/2
	    R(2) = (f2*cexi(1) + cd0i(3) - cd0i(2))/2
	    R(3) = (f3*cexi(1) + cd0i(4) - cd0i(3))/2
	    R(4) = (f4*cexi(1) + cd0i(5) - cd0i(4))/2
	    cxy(2)=xi5(1)*R(1)+xi5(5)*R(2)+xi5(6) *R(3)+xi5(7) *R(4)
	    cxy(3)=xi5(5)*R(1)+xi5(2)*R(2)+xi5(8) *R(3)+xi5(9) *R(4)
	    cxy(4)=xi5(6)*R(1)+xi5(8)*R(2)+xi5(3) *R(3)+xi5(10)*R(4)
	    cxy(5)=xi5(7)*R(1)+xi5(9)*R(2)+xi5(10)*R(3)+xi5(4) *R(4)
	    do i=2,5
		if ( xloss*10d0**(-mod(ier,50))*absc(cexi(i)-cxy(i))
     +			.gt. precc*absc(cxy(i)) ) then
		    print *,'aaxex: error: E1 from ffxe1 is not correct'
     +			,i,cexi(i),cxy(i),ier
		endif
	    enddo
	endif
	if ( awrite ) then
	    print *,'     '
	    print *,'aaxex: level 1: id,nevent ',id,nevent
	    print *,'E11=',cexi(2),ier
	    print *,'E12=',cexi(3),ier
	    print *,'E13=',cexi(4),ier
	    print *,'E14=',cexi(5),ier
	endif

	if ( level.le.1 ) return
*
*  #] level 1:
*  #[ level 2: E21,E22,E23,E24,E25,E26,E27,E28,E29,E210
*
*	D11(I),D12(I),D13(I) need 5 diff D1-functions
*
	do i=1,5
	    j = 2 + (i-1)*11
	    cd1ij(1,i)=cdxi(j)
	    cd1ij(2,i)=cdxi(j+1)
	    cd1ij(3,i)=cdxi(j+2)
	enddo
*
*	GJ reduction:
*
	call ffxe2(ce2ij,cexi(2),cexi(1),cd1ij,cd0i,xpi,fpij5,
     +          del3ij,del4i,fdel4,ier)
	if ( atest ) then
*
*	PV-reduction
*
	R(11) = (f1*cexi(2) + cd1ij(1,2) + cd0i(1) )/2
	R(12) = (f2*cexi(2) + cd1ij(1,3) - cd1ij(1,2))/2
	R(13) = (f3*cexi(2) + cd1ij(1,4) - cd1ij(1,3))/2
	R(14) = (f4*cexi(2) + cd1ij(1,5) - cd1ij(1,4))/2
*
	R(15) = (f1*cexi(3) + cd1ij(1,2) - cd1ij(1,1))/2
	R(16) = (f2*cexi(3) + cd1ij(2,3) - cd1ij(1,2))/2
	R(17) = (f3*cexi(3) + cd1ij(2,4) - cd1ij(2,3))/2
	R(18) = (f4*cexi(3) + cd1ij(2,5) - cd1ij(2,4))/2
*
	R(19) = (f1*cexi(4) + cd1ij(2,2) - cd1ij(2,1))/2
	R(20) = (f2*cexi(4) + cd1ij(2,3) - cd1ij(2,2))/2
	R(21) = (f3*cexi(4) + cd1ij(3,4) - cd1ij(2,3))/2
	R(22) = (f4*cexi(4) + cd1ij(3,5) - cd1ij(3,4))/2
*
	R(23) = (f1*cexi(5) + cd1ij(3,2) - cd1ij(3,1))/2
	R(24) = (f2*cexi(5) + cd1ij(3,3) - cd1ij(3,2))/2
	R(25) = (f3*cexi(5) + cd1ij(3,4) - cd1ij(3,3))/2
	R(26) = (f4*cexi(5)            - cd1ij(3,4))/2
*
	cexi(6) = xi5(1)*R(11)+xi5(5)*R(12)+xi5(6) *R(13)+xi5(7) *R(14)
	cexi(7) = xi5(5)*R(15)+xi5(2)*R(16)+xi5(8) *R(17)+xi5(9) *R(18)
	cexi(8) = xi5(6)*R(19)+xi5(8)*R(20)+xi5(3) *R(21)+xi5(10)*R(22)
	cexi(9) = xi5(7)*R(23)+xi5(9)*R(24)+xi5(10)*R(25)+xi5(4) *R(26)
	cexi(10)= xi5(5)*R(11)+xi5(2)*R(12)+xi5(8) *R(13)+xi5(9) *R(14)
	cexi(11)= xi5(6)*R(11)+xi5(8)*R(12)+xi5(3) *R(13)+xi5(10)*R(14)
	cexi(12)= xi5(7)*R(11)+xi5(9)*R(12)+xi5(10)*R(13)+xi5(4) *R(14)
	cexi(13)= xi5(6)*R(15)+xi5(8)*R(16)+xi5(3) *R(17)+xi5(10)*R(18)
	cexi(14)= xi5(7)*R(15)+xi5(9)*R(16)+xi5(10)*R(17)+xi5(4) *R(18)
	cexi(15)= xi5(7)*R(19)+xi5(9)*R(20)+xi5(10)*R(21)+xi5(4) *R(22)
*
	if ( atest ) then
*
*	    redundancy check
*
	cxy(10) = xi5(1)*R(15)+xi5(5)*R(16)+xi5(6)*R(17)+xi5(7) *R(18)
	cxy(11) = xi5(1)*R(19)+xi5(5)*R(20)+xi5(6)*R(21)+xi5(7) *R(22)
	cxy(12) = xi5(1)*R(23)+xi5(5)*R(24)+xi5(6)*R(25)+xi5(7) *R(26)
	cxy(13) = xi5(5)*R(19)+xi5(2)*R(20)+xi5(8)*R(21)+xi5(9) *R(22)
	cxy(14) = xi5(5)*R(23)+xi5(2)*R(24)+xi5(8)*R(25)+xi5(9) *R(26)
	cxy(15) = xi5(6)*R(23)+xi5(8)*R(24)+xi5(3)*R(25)+xi5(10)*R(26)
	    do i=10,15
		if ( absc(cxy(i)-cexi(i))  .gt. .1d-3  ) then
		    print *,'redundancy check at level 2 failed'
		endif
	    enddo
*###] :
	endif
*
*	check against GJ
*
	do i=1,4
	    do j=1,4
		if ( xloss*10d0**(-mod(ier,50))*absc(ce2ij(i,j) -
     +			cexi(ij2k(i,j))) .gt. precc*absc(ce2ij(i,j)) )
     +			then
		    print *,'ffxdxp: error: GJ does not agree with PV:',
     +			i,j,ce2ij(i,j),cexi(ij2k(i,j)),ce2ij(i,j) -
     +			cexi(ij2k(i,j)),ier
		endif
	    enddo
	enddo
	endif
*
*	copy to AAs arrays
*
	do i=1,4
	    do j=1,4
		cexi(ij2k(j,i)) = ce2ij(j,i)
	    enddo
	enddo
*
	if ( awrite ) then
	    print *,'     '
	    print *,'aaxex: level 2: id,nevent ',id,nevent
	    print *,'E21 =',cexi(6),ier
	    print *,'E22 =',cexi(7),ier
	    print *,'E23 =',cexi(8),ier
	    print *,'E24 =',cexi(9),ier
	    print *,'E25 =',cexi(10),ier
	    print *,'E26 =',cexi(11),ier
	    print *,'E27 =',cexi(12),ier
	    print *,'E28 =',cexi(13),ier
	    print *,'E28 =',cexi(14),ier
	    print *,'E210=',cexi(15),ier
	endif
*
	if (level .le. 2) return
*
*  #] level 2:
*###[ : level 3 :
*
*	first recast the D2's to a more useful form
*
	do 15 i=1,5
	    j = 5 +(i-1)*11
	    cd2ijk(1,1,i) = cdxi(j)
	    cd2ijk(2,2,i) = cdxi(j+1)
	    cd2ijk(3,3,i) = cdxi(j+2)
	    cd2ijk(1,2,i) = cdxi(j+3)
	    cd2ijk(2,1,i) = cdxi(j+3)
	    cd2ijk(1,3,i) = cdxi(j+4)
	    cd2ijk(3,1,i) = cdxi(j+4)
	    cd2ijk(2,3,i) = cdxi(j+5)
	    cd2ijk(3,2,i) = cdxi(j+5)
	    cd2i(i) = cdxi(j+6)
 15	continue
*
*	FF function
*
	call ffxe3(ce3ijk,ce2ij,cexi(2),cexi(1), cd2ijk,cd2i,cd1ij,cd0i,
     +		xpi,fpij5, del3ij,del4i,fdel4, ier)
*
*	copy FF structs to AA
*
	do i=16,35
	    cexi(i) = ce3ijk(m2ijk(1,i),m2ijk(2,i),m2ijk(3,i))
	enddo
	if ( atest ) then
*
*
*		PV-reduction
*	g-terms
	rg(1)=1/2d0*( f1*cexi(6)+f2*cexi(10)+f3*cexi(11)+f4*cexi(12) )
	rg(2)=1/2d0*( f1*cexi(10)+f2*cexi(7)+f3*cexi(13)+f4*cexi(14) )
	rg(3)=1/2d0*( f1*cexi(11)+f2*cexi(13)+f3*cexi(8)+f4*cexi(15) )
	rg(4)=1/2d0*( f1*cexi(12)+f2*cexi(14)+f3*cexi(15)+f4*cexi(9) )
*
	cexj(36)=xpi(1)*cexj(2)-1/2d0*cd0i(1) -rg(1)
	cexj(37)=xpi(1)*cexj(3)+1/2d0*cd1ij(1,1)-rg(2)
	cexj(38)=xpi(1)*cexj(4)+1/2d0*cd1ij(2,1)-rg(3)
	cexj(39)=xpi(1)*cexj(5)+1/2d0*cd1ij(3,1)-rg(4)
*
*	terms ~pipi
*	1)
	R(31)=1/2d0*(f1*cexj(6)+cd2ijk(1,1,2)-cd0i(1)) -2*cexj(36)
	R(32)=1/2d0*(f2*cexj(6)+cd2ijk(1,1,3)-cd2ijk(1,1,2))
	R(33)=1/2d0*(f3*cexj(6)+cd2ijk(1,1,4)-cd2ijk(1,1,3))
	R(34)=1/2d0*(f4*cexj(6)+cd2ijk(1,1,5)-cd2ijk(1,1,4))
*	2)
	R(35)=1/2d0*(f1*cexj(7)+cd2ijk(1,1,2)-cd2ijk(1,1,1))
	R(36)=1/2d0*(f2*cexj(7)+cd2ijk(2,2,3)-cd2ijk(1,1,2)) -2*cexj(37)
	R(37)=1/2d0*(f3*cexj(7)+cd2ijk(2,2,4)-cd2ijk(2,2,3))
	R(38)=1/2d0*(f4*cexj(7)+cd2ijk(2,2,5)-cd2ijk(2,2,4))
*	3)
	R(39)=1/2d0*(f1*cexj(8)+cd2ijk(2,2,2)-cd2ijk(2,2,1))
	R(40)=1/2d0*(f2*cexj(8)+cd2ijk(2,2,3)-cd2ijk(2,2,2))
	R(41)=1/2d0*(f3*cexj(8)+cd2ijk(3,3,4)-cd2ijk(2,2,3)) -2*cexj(38)
	R(42)=1/2d0*(f4*cexj(8)+cd2ijk(3,3,5)-cd2ijk(3,3,4))
*	4)
	R(43)=1/2d0*(f1*cexj(9)+cd2ijk(3,3,2)-cd2ijk(3,3,1))
	R(44)=1/2d0*(f2*cexj(9)+cd2ijk(3,3,3)-cd2ijk(3,3,2))
	R(45)=1/2d0*(f3*cexj(9)+cd2ijk(3,3,4)-cd2ijk(3,3,3))
	R(46)=1/2d0*(f4*cexj(9)         -cd2ijk(3,3,4) ) -2*cexj(39)
*
*	terms ~p1pi
*	1)
	R(47)=1/2d0*(f1*cexj(10)+cd2ijk(1,1,2)+cd1ij(1,1)) -cexj(37)
	R(48)=1/2d0*(f2*cexj(10)+cd2ijk(1,2,3)-cd2ijk(1,1,2)) -cexj(36)
	R(49)=1/2d0*(f3*cexj(10)+cd2ijk(1,2,4)-cd2ijk(1,2,3))
	R(50)=1/2d0*(f4*cexj(10)+cd2ijk(1,2,5)-cd2ijk(1,2,4))
*	2)
	R(51)=1/2d0*(f1*cexj(11)+cd2ijk(1,2,2)+cd1ij(2,1) ) -cexj(38)
	R(52)=1/2d0*(f2*cexj(11)+cd2ijk(1,2,3)-cd2ijk(1,2,2))
	R(53)=1/2d0*(f3*cexj(11)+cd2ijk(1,3,4)-cd2ijk(1,2,3)) -cexj(36)
	R(54)=1/2d0*(f4*cexj(11)+cd2ijk(1,3,5)-cd2ijk(1,3,4))
*	3)
	R(55)=1/2d0*(f1*cexj(12)+cd2ijk(1,3,2)+cd1ij(3,1) ) -cexj(39)
	R(56)=1/2d0*(f2*cexj(12)+cd2ijk(1,3,3)-cd2ijk(1,3,2))
	R(57)=1/2d0*(f3*cexj(12)+cd2ijk(1,3,4)-cd2ijk(1,3,3))
	R(58)=1/2d0*(f4*cexj(12)         -cd2ijk(1,3,4) ) -cexj(36)
*
*	terms ~p2pi
*	1)
	R(59)=1/2d0*(f1*cexj(13)+cd2ijk(1,2,2)-cd2ijk(1,2,1))
	R(60)=1/2d0*(f2*cexj(13)+cd2ijk(2,2,3)-cd2ijk(1,2,2)) -cexj(38)
	R(61)=1/2d0*(f3*cexj(13)+cd2ijk(2,3,4)-cd2ijk(2,2,3)) -cexj(37)
	R(62)=1/2d0*(f4*cexj(13)+cd2ijk(2,3,5)-cd2ijk(2,3,4))
*	2)
	R(63)=1/2d0*(f1*cexj(14)+cd2ijk(1,3,2)-cd2ijk(1,3,1))
	R(64)=1/2d0*(f2*cexj(14)+cd2ijk(2,3,3)-cd2ijk(1,3,2)) -cexj(39)
	R(65)=1/2d0*(f3*cexj(14)+cd2ijk(2,3,4)-cd2ijk(2,3,3))
	R(66)=1/2d0*(f4*cexj(14)         -cd2ijk(2,3,4) ) -cexj(37)
*
*	terms ~p3pi
*	1)
	R(67)=1/2d0*(f1*cexj(15)+cd2ijk(2,3,2)-cd2ijk(2,3,1))
	R(68)=1/2d0*(f2*cexj(15)+cd2ijk(2,3,3)-cd2ijk(2,3,2))
	R(69)=1/2d0*(f3*cexj(15)+cd2ijk(3,3,4)-cd2ijk(2,3,3)) -cexj(39)
	R(70)=1/2d0*(f4*cexj(15)         -cd2ijk(3,3,4) ) -cexj(38)

	cexj(16)=xi5(1)*R(31)+xi5(5)*R(32)+xi5(6)*R(33)+xi5(7) *R(34)
	cexj(17)=xi5(5)*R(35)+xi5(2)*R(36)+xi5(8)*R(37)+xi5(9) *R(38)
	cexj(18)=xi5(6)*R(39)+xi5(8)*R(40)+xi5(3)*R(41)+xi5(10)*R(42)
	cexj(19)=xi5(7)*R(43)+xi5(9)*R(44)+xi5(10)*R(45)+xi5(4)*R(46)
	cexj(20)=xi5(5)*R(31)+xi5(2)*R(32)+xi5(8)*R(33)+xi5(9) *R(34)
	cexj(21)=xi5(6)*R(31)+xi5(8)*R(32)+xi5(3)*R(33)+xi5(10)*R(34)
	cexj(22)=xi5(7)*R(31)+xi5(9)*R(32)+xi5(10)*R(33)+xi5(4)*R(34)
	cexj(23)=xi5(1)*R(35)+xi5(5)*R(36)+xi5(6)*R(37)+xi5(7) *R(38)
	cexj(24)=xi5(6)*R(35)+xi5(8)*R(36)+xi5(3)*R(37)+xi5(10)*R(38)
	cexj(25)=xi5(7)*R(35)+xi5(9)*R(36)+xi5(10)*R(37)+xi5(4)*R(38)
	cexj(26)=xi5(1)*R(39)+xi5(5)*R(40)+xi5(6)*R(41)+xi5(7) *R(42)
	cexj(27)=xi5(5)*R(39)+xi5(2)*R(40)+xi5(8)*R(41)+xi5(9) *R(42)
	cexj(28)=xi5(7)*R(39)+xi5(9)*R(40)+xi5(10)*R(41)+xi5(4)*R(42)
	cexj(29)=xi5(1)*R(43)+xi5(5)*R(44)+xi5(6)*R(45)+xi5(7) *R(46)
	cexj(30)=xi5(5)*R(43)+xi5(2)*R(44)+xi5(8)*R(45)+xi5(9) *R(46)
	cexj(31)=xi5(6)*R(43)+xi5(8)*R(44)+xi5(3)*R(45)+xi5(10)*R(46)
	cexj(32)=xi5(6)*R(47)+xi5(8)*R(48)+xi5(3)*R(49)+xi5(10)*R(50)
	cexj(33)=xi5(7)*R(47)+xi5(9)*R(48)+xi5(10)*R(49)+xi5(4)*R(50)
	cexj(34)=xi5(7)*R(51)+xi5(9)*R(52)+xi5(10)*R(53)+xi5(4)*R(54)
	cexj(35)=xi5(7)*R(59)+xi5(9)*R(60)+xi5(10)*R(61)+xi5(4)*R(62)
*

*###[ : redundancy check
	cxy(20)=xi5(1)*R(47)+xi5(5)*R(48)+xi5(6)*R(49)+xi5(7) *R(50)
	cxy(21)=xi5(1)*R(51)+xi5(5)*R(52)+xi5(6)*R(53)+xi5(7) *R(54)
	cxy(22)=xi5(1)*R(55)+xi5(5)*R(56)+xi5(6)*R(57)+xi5(7) *R(58)
	cxy(23)=xi5(5)*R(47)+xi5(2)*R(48)+xi5(8)*R(49)+xi5(9) *R(50)
	cxy(24)=xi5(5)*R(59)+xi5(2)*R(60)+xi5(8)*R(61)+xi5(9) *R(62)
	cxy(25)=xi5(5)*R(63)+xi5(2)*R(64)+xi5(8)*R(65)+xi5(9) *R(66)
	cxy(26)=xi5(6)*R(51)+xi5(8)*R(52)+xi5(3)*R(53)+xi5(10)*R(54)
	cxy(27)=xi5(6)*R(59)+xi5(8)*R(60)+xi5(3)*R(61)+xi5(10)*R(62)
	cxy(28)=xi5(6)*R(67)+xi5(8)*R(68)+xi5(3)*R(69)+xi5(10)*R(70)
	cxy(29)=xi5(7)*R(55)+xi5(9)*R(56)+xi5(10)*R(57)+xi5(4)*R(58)
	cxy(30)=xi5(7)*R(63)+xi5(9)*R(64)+xi5(10)*R(65)+xi5(4)*R(66)
	cxy(31)=xi5(7)*R(67)+xi5(9)*R(68)+xi5(10)*R(69)+xi5(4)*R(70)
	cxy(32)=xi5(5)*R(51)+xi5(2)*R(52)+xi5(8)*R(53)+xi5(9) *R(54)
	cxy(32)=xi5(1)*R(59)+xi5(5)*R(60)+xi5(6)*R(61)+xi5(7) *R(62)
	cxy(33)=xi5(5)*R(55)+xi5(2)*R(56)+xi5(8)*R(57)+xi5(9) *R(58)
	cxy(33)=xi5(1)*R(63)+xi5(5)*R(64)+xi5(6)*R(65)+xi5(7) *R(66)
	cxy(34)=xi5(6)*R(55)+xi5(8)*R(56)+xi5(3)*R(57)+xi5(10)*R(58)
	cxy(34)=xi5(1)*R(67)+xi5(5)*R(68)+xi5(6)*R(69)+xi5(7) *R(70)
	cxy(35)=xi5(6)*R(63)+xi5(8)*R(64)+xi5(3)*R(65)+xi5(10)*R(66)
	cxy(35)=xi5(5)*R(67)+xi5(2)*R(68)+xi5(8)*R(69)+xi5(9) *R(70)
	do 16 i=20,35
	   if ( absc(cxy(i)-cexj(i)) .gt. .1d-3  ) then
	      print *,'ffxdxp: redundancy check at level 3 failed ',
     +	      	i,cxy(i),cexj(i),cxy(i)-cexj(i),ier
	      return
	   endif
 16	continue
*	
*	check against FF
*	
	do 17 i=16,35
	    if ( xloss*10d0**(-mod(ier,50))*absc(cexi(i)-cexj(i)) .gt. 
     +	    		precc*absc(cexi(i)) ) then
	    	print *,'ffxexp: error: FF disagrees with PV: ',i,
     +	    		cexi(i),cexj(i),cexi(i)-cexj(i),ier
	    endif
   17	continue
*###] :
	endif
	if ( awrite ) then
*###[ : print level 3
	    print *,'     '
	    print *,'aaxex : level 3 '
	    print *,'E31 =',cexi(16)
	    print *,'E32 =',cexi(17)
	    print *,'E33 =',cexi(18)
	    print *,'E34 =',cexi(19)
	    print *,'E35 =',cexi(20)
	    print *,'E36 =',cexi(21)
	    print *,'E37 =',cexi(22)
	    print *,'E38 =',cexi(23)
	    print *,'E39 =',cexi(24)
	    print *,'E310=',cexi(25)
	    print *,'E311=',cexi(26)
	    print *,'E312=',cexi(27)
	    print *,'E313=',cexi(28)
	    print *,'E314=',cexi(29)
	    print *,'E315=',cexi(30)
	    print *,'E316=',cexi(31)
	    print *,'E317=',cexi(32)
	    print *,'E318=',cexi(33)
	    print *,'E319=',cexi(34)
	    print *,'E320=',cexi(35)
*###] :
	endif
*
	if (level .eq. 3) return
*
*###] : level 3 :
	if (level .gt. 3) then
	    print *,'higher than third rank not yet implemented'
	    stop
	endif

*###] ffxex:
	end
*###[ ffeji:
	subroutine ffeji(cdxi,mdxi,ccxi,mcxi,cbxi,mbxi,caxi,maxi,
     +		cdxj,mdxj,ccxj,mcxj,cbxj,mbxj,caxj,maxj,level)
***#[*comment:***********************************************************
*									*
*	Copy the fourpoint arrays [cm][dcba]xj to the five point arrays	*
*	[cm][dcba]xi.							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
*
	integer level
	DOUBLE PRECISION mdxi(55),mcxi(30),mbxi(30),maxi(5),
     +		mdxj(120),mcxj(140),mbxj(60),maxj(20)
	DOUBLE COMPLEX cdxi(55),ccxi(30),cbxi(30),caxi(5),
     +		cdxj(120),ccxj(140),cbxj(60),caxj(20)
*
	integer i,j,k
*
	include 'aa.h'
*
*  #] declarations:
*  #[ copy necessary parts to E0 arrays:
*
*	1)D-output: reduce the array cdxj(5*24) to cdxi(5*11)
*		    d's are calculated only to (level-1)
	do j=1,5
	    do i=1,11
		cdxi(i+(j-1)*11) = cdxj(i+(j-1)*24)
	    enddo
	enddo
*
*	2)C-output: reduce the array ccxj(20*7) to ccxi(10*3)
*		    c's are calculated only to (level-2)
*
	do j=1,4
	    do i=1,3
		ccxi(0+i+(j-1)*3) = ccxj(0+i+(j-1)*7)
	    enddo
	enddo
	do j=1,3
	    do i=1,3
		ccxi(12+i+(j-1)*3) = ccxj(35+i+(j-1)*7)
	    enddo
	enddo
	do j=1,2
	    do i=1,3
		ccxi(21+i+(j-1)*3) = ccxj(70+i+(j-1)*7)
	    enddo
	enddo
	ccxi(28) = ccxj(106)
	ccxi(29) = ccxj(107)
	ccxi(30) = ccxj(108)
*
*	check the symmetry in B0(i,j)
*	if ( atest ) then
*	    do 13 i=1,4
*		j=4+i
*		k=8+i
*		if (    ( cbxj(i)     - cbxj(i+1*12) ) .ne. 0. .or.
*     +			( cbxj(j)     - cbxj(i+2*12) ) .ne. 0. .or.
*     +			( cbxj(k)     - cbxj(i+3*12) ) .ne. 0. .or.
*     +			( cbxj(j+1*12)- cbxj(j+2*12) ) .ne. 0. .or.
*     +			( cbxj(k+1*12)- cbxj(j+3*12) ) .ne. 0. .or.
*     +			( cbxj(k+2*12)- cbxj(k+3*12) ) .ne. 0. ) then
*		    print *,'error in B0-calculations in aaxcx.for'
*		endif
* 13	    continue
*	endif
*
*  #] copy necessary parts to E0 arrays:
*###] ffeji:
	end
