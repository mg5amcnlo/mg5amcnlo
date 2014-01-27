*###[ aaxdx :
	subroutine aaxdx(cbxi,ccxi,cdxi,d0,xmm,xpi,level,ier)
***#[*comment:***********************************************************
*									*
*	Calculation of four point tensor integrals.  Just a wrapper	*
*	for ffxdx nowadays, see there for the real description.		*
*									*
*	Input:								*
*		xpi	  the same as in Geert Jan's routines		*
*		level	  rank of tensor(integral)			*
*	Output: 							*
*		cbxi(12)  cb0(1),cb1(1),[cb2(2)]	 x 6		*
*		ccxi(28)  cc0(1),cc1(2),cc2(4),[cc3(6)]  x 4		*
*		cdxi(24)  cd0(1),cd1(3),cd2(7),cd3(13)			*
*									*
***#]*comment:***********************************************************
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(13),d0,xmm
	DOUBLE COMPLEX cbxi(12),ccxi(28),cdxi(24)
*
*	local variables
*
	DOUBLE COMPLEX caxi(4)
	DOUBLE PRECISION maxi(4),mbxi(12),mcxi(28),mdxi(24)
*
*  #] declarations :
*  #[ call ffxdx:
*
	call ffxdx(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,d0,xmm,xpi,
     +		level,ier)
*
*  #] call ffxdx:
*###] aaxdx :
	end
*###[ ffxdx:
	subroutine ffxdx(caxi,maxi,cbxi,mbxi,ccxi,mcxi,cdxi,mdxi,d0,xmm,
     +		xpi,level,ier)
***#[*comment:***********************************************************
*									*
*	Calculation of four point form factors with more accurate	*
*	error estimates.  Calls ffxd1, the rest is still here.		*
*	Definitions:							*
*       D0 = 1/(i pi^2)*\int d^4 Q 1/((Q^2-m_1^2)((Q+p1)^2-m2^2)	*
*		((Q+p1+p2)^2-m3^2))((Q-p4)^2-m4^2))			*
*	D1 = 1/(i pi^2)*\int d^n Q Q(mu)/(...)				*
*	   = D11*p1 + D12*p2 + D13*p3					*
*	D2 = D21*p1*p1 + D22*p2*p2 + D23*p3*p3 + D24*(p1*p2+p2*p1) +	*
*	     D25*(p1*p3+p3*p1) + D26*(p2*p3+p3*p2) + D27*g		*
*	D3 = D31*p1*p1*p1 + D32*p2*p2*p2 + D33*p3*p3*p3 + D34*(p1*p1*p2+*
*	     p1*p2*p1+p2*p1*p1) + D35*(p1*p1*p3+p1*p3*p1+p3*p1*p1) + D36*
*	     *(p1*p2*p2+p2*p1*p2+p1*p2*p2) + D37*(p1*p3*p3+p3*p1*p3+p1*	*
*	     p3*p3) + D38*(p2*p2*p3+p2*p3*p2+p3*p2*p2) + D39*(p2*p3*p3+	*
*	     p3*p2*p3+p2*p3*p3) + D310*(p1*p2*p3+p2*p3*p1+p3*p1*p2+p1*p3*
*	     *p2+p3*p2*p1+p2*p1*p3) + D311*(p1*g+g*p1+'g*p1*g') + D312*	*
*	     (p2*g+g*p2+'g*p2*g') + D313*(p3*g+g*p3+'g*p3*g')		*
*	D4 has not yet been implemented					*
*									*
*	Input:	xpi(13)	     real	m_i^2 (1:4), p_{i-4}^2 (4:8),s,t*
*					optionally u,v,w (see ffxd0a)	*
*		d0,xmm	     real	renormalisation constants	*
*		level	     integer	rank of tensor (integral)	*
*	Output: caxi(4)	     complex	A0(m_i^2) only when level>3	*
*		maxi(12)     real	max term in sum to caxi()	*
*		cbxi(12)     complex	6x(B0,B11,B21,B22)(p_i^2)	*
*		mbxi(12)     real	max term in sum to cbxi()	*
*		ccxi(28)     complex	4x(C0,C1(2),C2(4))		*
*		mcxi(28)     real	max term in sum to ccxi()	*
*		cdxi(24)     complex	D0,D1(3),D2(7),D3(13)		*
*		mdxi(24)     real	max term in sum to cdxi()	*
*	Note that if level<3 some of these are not defined.		*
*									*
***#]*comment:***********************************************************
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(13),d0,xmm
	DOUBLE COMPLEX caxi(4),cbxi(12),ccxi(28),cdxi(24)
	DOUBLE PRECISION maxi(4),mbxi(12),mcxi(28),mdxi(24)
*
*	local variables
*
	integer i,j,cl,ier0,ier1,iinx(6,4)
	DOUBLE PRECISION xpi3(6),fdel2i(4),absc,big
	DOUBLE COMPLEX caxj(12),cbxj(48),ccxj(52),cc,cd0
	DOUBLE PRECISION maxj(12),mbxj(48),mcxj(52)
	save iinx
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
*	data
*
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
*
*  #] declarations :
*  #[ initialisations:
*
*	initialize to something ridiculous so that one immediately
*	notices when it is accidentally used...
*
	big = 1/(1d20*xclogm)
	do 8 i=1,4
	    caxi(i) = big
    8	continue
	do 9 i=1,12
	    cbxi(i) = big
    9	continue
	do 10 i=1,28
	    ccxi(i) = big
   10	continue
	do 11 i=1,24
	    cdxi(i) = big
   11	continue
*
*  #] initialisations:
*  #[ get D0:
*	D0-function (ff)
*	   futhermore dotpr and determinants are delivered by ff
	ldot = .TRUE.
	ier1 = ier
	call ffxd0(cdxi(1),xpi,ier1)
	if ( ier1.gt.10 ) then
	    if ( ltest ) then
		print *,'ffxdx: id = ',id,', nevent = ',nevent
		print *,'ffxdx: lost ',ier1,' digits in D0 with isgnal '
     +			,isgnal,', trying other roots, isgnal ',-isgnal
		print *,'       if OK (no further messages) adding this'
     +			,' to your code will improve speed'
	    endif
	    isgnal = -isgnal
	    ier0 = ier
	    call ffxd0(cd0,xpi,ier0)
	    isgnal = -isgnal
	    if ( ier0.lt.ier1 ) then
	    	ier1 = ier0
	    	cdxi(1) = cd0
	    endif
	endif
	if ( awrite ) then
	    print *,'    '
	    print *,'ffxdx : level 0: id,nevent ',id,nevent
	    print *,'D0 =',cdxi(1),ier1
	endif
	if ( ier1 .gt. 10 ) then
	    print *,'ffxdx: id = ',id,', nevent = ',nevent
	    print *,'ffxdx: error: D0 not stable, lost ',ier1,' digits'
	    print *,'       please try another permutation or contact',
     +		' author (t19@nikhef.nl)'
	    print *,'xpi = ',xpi
	endif
*	note that we may have lost another factor xloss**3 or so
	mdxi(1) = absc(cdxi(1))*DBLE(10)**mod(ier1,50)
*
	if (level .eq. 0) goto 990
*
*  #] get D0:
*  #[ need C-functions till c-level=(level-1):
	if ( level.gt.3 ) then
	    print *,'ffxdx: error: higher than third rank ',
     +		'not yet implemented'
	    stop
	endif
	cl = level-1
*	go trough the 4 different cancellation patterns
	if ( awrite ) then
	    print '(a,i1)','####[ C-function output: to level ',cl
	endif
	do 100 i=1,4
	    do 60 j=1,6
		xpi3(j) = xpi(iinx(j,i))
   60	    continue
	    ier0 = ier
	    call ffxcx( caxj(3*i-2),maxj(3*i-2),cbxj(12*i-11),
     +		mbxj(12*i-11),ccxj(13*i-12),mcxj(13*i-12),d0,xmm,xpi3,
     +		cl,ier0)
	    ier1 = max(ier1,ier0)
	    fdel2i(i)=fdel2
  100	continue
	if ( awrite ) then
	    print '(a)','####] C-function output:'
	endif
*  #] need C-functions till c-level=(level-1):
*  #[ break to let ffzdz tie in:
*
*	convert ??xj to ??xi
*
	call ffdji(ccxi,mcxi,cbxi,mbxi,caxi,maxi,
     +		ccxj,mcxj,cbxj,mbxj,caxj,maxj,level)
*
*	and call the real routine for the rest
*
	call ffxdxp(caxj,maxj,cbxj,mbxj,ccxj,mcxj,cdxi,mdxi,xpi,fdel2i,
     +		level,ier1)
*  #] break to let ffzdz tie in:
  990	ier = ier1
	end
	subroutine ffxdxp(caxj,maxj,cbxj,mbxj,ccxj,mcxj,cdxi,mdxi,xpi,
     +		fdel2i,level,ier)
*  #[ declarations :
	implicit none
*
*	arguments
*
	integer ier,level
	DOUBLE PRECISION xpi(13)
	DOUBLE COMPLEX caxj(12),cbxj(48),ccxj(52),cdxi(24)
	DOUBLE PRECISION maxj(12),mbxj(48),mcxj(52),mdxi(24),fdel2i(4)
*
*	local variables
*
	integer i,j,ier0,ier1,ier2,iinx(6,4),bij(12)
	DOUBLE PRECISION xi4(6),f1,f2,f3,absc,xmax,Rm(20:55),
     +		d11max,d22max,d23max,d24max,d0,xmm
	DOUBLE PRECISION dl2pij(6,6),del3p
	DOUBLE PRECISION mc0i(4),mxy(3),mc21i(4),mc22i(4),mc23i(4),
     +		mc24i(4),mc11i(4),mc12i(4),md1i(3)
	DOUBLE COMPLEX R20,R21,R22,R30,R31,R32,R33,R34,R35,R36,R37,R38,
     +		R41,R42,R43,R44,R45,R46,R47,R48,R49,R50,R51,R52,R53,R54,
     +		R55,cd1i(3),cc0i(4),cc11i(4),cc12i(4),
     +		cc21i(4),cc22i(4),cc23i(4),cc24i(4),cc,cxy(3)
	DOUBLE COMPLEX cd4pppp(3,3,3,3),cd4ppdel(3,3),cd4deldel,
     +		cd3ppp(3,3,3),cd3pdel(3),cd2pp(3,3),cd2del,
     +		cb0ij(4,4),ca0i(4),cd2(7)
	save iinx,bij
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*
*	data
*
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
	data bij /1,2,5,6,9,10,17,18,21,22,33,34/
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations :
*  #[ kinematical quantities for 4pv-red :
*	if ( abs(fdel3)  .lt. 1.d-6 ) then
*	    print *,'kinematical det = 0, PV-scheme breaks down'
*	    print *,fdel3
*	    goto 990
*	endif
	if ( atest ) then
	    del3p =
     +  - xpi(5)*xpi(5)*xpi(7) + xpi(5)*xpi(6)*xpi(7) + xpi(5)*xpi(6)*
     + xpi(8) - xpi(5)*xpi(6)*xpi(9) - xpi(5)*xpi(7)*xpi(7) + xpi(5)*
     + xpi(7)*xpi(8) + xpi(5)*xpi(7)*xpi(9) + xpi(5)*xpi(7)*xpi(10) -
     + xpi(5)*xpi(8)*xpi(10) + xpi(5)*xpi(9)*xpi(10) - xpi(6)*xpi(6)*
     + xpi(8) + xpi(6)*xpi(7)*xpi(8) - xpi(6)*xpi(7)*xpi(10) - xpi(6)*
     + xpi(8)*xpi(8) + xpi(6)*xpi(8)*xpi(9) + xpi(6)*xpi(8)*xpi(10) +
     + xpi(6)*xpi(9)*xpi(10) - xpi(7)*xpi(8)*xpi(9) + xpi(7)*xpi(9)*
     + xpi(10) + xpi(8)*xpi(9)*xpi(10) - xpi(9)*xpi(9)*xpi(10) -
     + xpi(9)*xpi(10)*xpi(10)
	    del3p = del3p/4
	    xmax = max(abs(xpi(5)),abs(xpi(6)),abs(xpi(7)),abs(xpi(8)),
     +		abs(xpi(9)),abs(xpi(10)))**3
	    if ( abs(del3p-fdel3).gt.1d-12*xmax ) then
		print *,'ffxdxp: fdel3 wrong: ',fdel3,del3p,fdel3-del3p,
     +			xmax
	    endif
	endif
*
*	inverse kinematical matrix xi4  (3X3)
*
	ier2 = ier
	call aaxi4(xi4,ier2)
	ier2 = ier2 - ier
*
*	    f-functions:
	f1 = 2*fpij4(1,5)
	f2 = 2*fpij4(1,6)
	f3 = 2*fpij4(1,7)
*
*  #] kinematical quantities for 4pv-red :
*  #[ level 1 : D11,D12,D13,C0(I)
*		need 4 diff C0(I)-functions,I=1,2,3
	cc0i(1)=ccxj(1)
	cc0i(2)=ccxj(14)
	cc0i(3)=ccxj(27)
	cc0i(4)=ccxj(40)
	mc0i(1)=mcxj(1)
	mc0i(2)=mcxj(14)
	mc0i(3)=mcxj(27)
	mc0i(4)=mcxj(40)
	ier1 = ier
	call ffxd1a(cdxi(2),mdxi(2),cdxi(1),mdxi(1),cc0i,mc0i,
     +		xpi,fpij4,fdel3,fdel2i,ier1)
	if ( awrite ) then
	    print *,'GEERT JANs-scheme: id,nevent ',id,nevent
	    print *,'D11=',cdxi(2),mdxi(2),ier1
	    print *,'D12=',cdxi(3),mdxi(3),ier1
	    print *,'D13=',cdxi(4),mdxi(4),ier1
	endif
*
	if ( lwarn .and. atest ) then
*	    PV-reduction
	    R20 = ( f1*cdxi(1)+cc0i(2)-cc0i(1) )/2
	    R21 = ( f2*cdxi(1)+cc0i(3)-cc0i(2) )/2
	    R22 = ( f3*cdxi(1)+cc0i(4)-cc0i(3) )/2
	    Rm(20) = ( max(abs(f1)*mdxi(1),mc0i(2),mc0i(1)) )/2
	    Rm(21) = ( max(abs(f2)*mdxi(1),mc0i(3),mc0i(2)) )/2
	    Rm(22) = ( max(abs(f3)*mdxi(1),mc0i(4),mc0i(3)) )/2
	    cd1i(1)=xi4(1)*R20+xi4(4)*R21+xi4(5)*R22
	    cd1i(2)=xi4(4)*R20+xi4(2)*R21+xi4(6)*R22
	    cd1i(3)=xi4(5)*R20+xi4(6)*R21+xi4(3)*R22
	    md1i(1)=abs(xi4(1))*Rm(20)+abs(xi4(4))*Rm(21)+
     +		abs(xi4(5))*Rm(22)
	    md1i(2)=abs(xi4(4))*Rm(20)+abs(xi4(2))*Rm(21)+
     +		abs(xi4(6))*Rm(22)
	    md1i(3)=abs(xi4(5))*Rm(20)+abs(xi4(6))*Rm(21)+
     +		abs(xi4(3))*Rm(22)
	    do 139 i=1,3
		if ( xloss**2*absc(cdxi(i+1)-cd1i(i)).gt.precc*max(
     +		    mdxi(i+1),md1i(i)) ) print *,'ffxdx: error: FF D1',
     +		    i,' disagrees with PV:',cdxi(i+1),cd1i(i),
     +		    cdxi(i+1)-cd1i(i),max(mdxi(i+1),md1i(i))
  139	    continue
	    if (awrite) then
		print *,' '
		print *,'ffxdx : level 1: id,nevent ',id,nevent
		print *,'D11=',cd1i(1)
		print *,'D12=',cd1i(2)
		print *,'D13=',cd1i(3)
	    endif
	endif
*
	if ( level.eq.1 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,4
		    if ( absc(cdxi(i)).ne.0 ) then
			xmax = max(xmax,mdxi(i)/absc(cdxi(i)))
		    elseif ( mdxi(i).ne.0 ) then
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
*  #[ level 2 : D21,D22,D23,D24,D25,D26,D27,C11(I),C12(I)
*	    need 4 diff C1-functions
	ier = ier1
	do 14 i=1,4
	    j = 2 + (i-1)*13
	    cc11i(i) = ccxj(j)
	    cc12i(i) = ccxj(j+1)
	    mc11i(i) = mcxj(j)
	    mc12i(i) = mcxj(j+1)
   14	continue
*		PV-reduction
	cdxi(11)=-( f1*cdxi(2)+f2*cdxi(3)+f3*cdxi(4)-cc0i(1) )/2
     +		 +xpi(1)*cdxi(1)
	if ( lwarn ) then
***	    d11max = max(absc(f1*cdxi(2)),absc(f2*cdxi(3)),
***     +		absc(f3*cdxi(4)),absc(cc0i(1)),2*absc(xpi(1)*cdxi(1)))/2
***	    if ( absc(cdxi(11)).lt.xloss*d11max ) then
***		ier0 = ier
***		call ffwarn(283,ier0,absc(cdxi(11)),d11max)
***		ier1 = max(ier1,ier0)
***	    endif
	    mdxi(11) = max(abs(f1)*mdxi(2),abs(f2)*mdxi(3),
     +		abs(f3)*mdxi(4),mc0i(1),2*xpi(1)*mdxi(1))/2
	endif
	R30=( f1*cdxi(2) + cc11i(2) + cc0i(1)  )/2 - cdxi(11)
	R31=( f2*cdxi(2) + cc11i(3) - cc11i(2) )/2
	R32=( f3*cdxi(2) + cc11i(4) - cc11i(3) )/2
	R33=( f1*cdxi(3) + cc11i(2) - cc11i(1) )/2
	R34=( f2*cdxi(3) + cc12i(3) - cc11i(2) )/2 - cdxi(11)
	R35=( f3*cdxi(3) + cc12i(4) - cc12i(3) )/2
	R36=( f1*cdxi(4) + cc12i(2) - cc12i(1) )/2
	R37=( f2*cdxi(4) + cc12i(3) - cc12i(2) )/2
	R38=( f3*cdxi(4)            - cc12i(3) )/2 - cdxi(11)
	cdxi(5) = xi4(1)*R30+xi4(4)*R31+xi4(5)*R32
	cdxi(6) = xi4(4)*R33+xi4(2)*R34+xi4(6)*R35
	cdxi(7) = xi4(5)*R36+xi4(6)*R37+xi4(3)*R38
	cdxi(8) = xi4(4)*R30+xi4(2)*R31+xi4(6)*R32
	cdxi(9) = xi4(5)*R30+xi4(6)*R31+xi4(3)*R32
	cdxi(10)= xi4(5)*R33+xi4(6)*R34+xi4(3)*R35
	if ( lwarn ) then
***	    Rm(30)=max(absc(f1*cdxi(2)),absc(cc11i(2)),absc(cc0i(1)),
***     +		2*d11max)/2
***	    Rm(31)=max(absc(f2*cdxi(2)),absc(cc11i(3)),absc(cc11i(2)))/2
***	    Rm(32)=max(absc(f3*cdxi(2)),absc(cc11i(4)),absc(cc11i(3)))/2
***	    Rm(33)=max(absc(f1*cdxi(3)),absc(cc11i(2)),absc(cc11i(1)))/2
***	    Rm(34)=max(absc(f2*cdxi(3)),absc(cc12i(3)),absc(cc11i(2)),
***     +		2*d11max)/2
***	    Rm(35)=max(absc(f3*cdxi(3)),absc(cc12i(4)),absc(cc12i(3)))/2
***	    Rm(36)=max(absc(f1*cdxi(4)),absc(cc12i(2)),absc(cc12i(1)))/2
***	    Rm(37)=max(absc(f2*cdxi(4)),absc(cc12i(3)),absc(cc12i(2)))/2
***	    Rm(38)=max(absc(f3*cdxi(4)),absc(cc12i(3)),2*d11max)/2
***	    xmax = max(abs(xi4(1))*Rm(30),abs(xi4(4))*Rm(31),abs(xi4(5))
***     +		*Rm(32))
***	    if ( absc(cdxi(5)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(282,ier0,absc(cdxi(5)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi4(4))*Rm(33),abs(xi4(2))*Rm(34),abs(xi4(6))
***     +		*Rm(35))
***	    if ( absc(cdxi(6)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(281,ier0,absc(cdxi(6)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi4(5))*Rm(36),abs(xi4(6))*Rm(37),abs(xi4(3))
***     +		*Rm(38))
***	    if ( absc(cdxi(7)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(280,ier0,absc(cdxi(7)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi4(4))*Rm(30),abs(xi4(2))*Rm(31),abs(xi4(6))
***     +		*Rm(32))
***	    if ( absc(cdxi(8)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(279,ier0,absc(cdxi(8)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi4(5))*Rm(30),abs(xi4(6))*Rm(31),abs(xi4(3))
***     +		*Rm(32))
***	    if ( absc(cdxi(9)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(278,ier0,absc(cdxi(9)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax = max(abs(xi4(5))*Rm(33),abs(xi4(6))*Rm(34),abs(xi4(3))
***     +		*Rm(35))
***	    if ( absc(cdxi(10)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(277,ier0,absc(cdxi(10)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
*
*	    the maximum values of the whole value (not only in this step)
*
	    Rm(30) = max(abs(f1)*mdxi(2),mc11i(2),mc0i(1),2*mdxi(11))/2
	    Rm(31) = max(abs(f2)*mdxi(2),mc11i(3),mc11i(2))/2
	    Rm(32) = max(abs(f3)*mdxi(2),mc11i(4),mc11i(3))/2
	    Rm(33) = max(abs(f1)*mdxi(3),mc11i(2),mc11i(1))/2
	    Rm(34) = max(abs(f2)*mdxi(3),mc12i(3),mc11i(2),2*mdxi(11))/2
	    Rm(35) = max(abs(f3)*mdxi(3),mc12i(4),mc12i(3))/2
	    Rm(36) = max(abs(f1)*mdxi(4),mc12i(2),mc12i(1))/2
	    Rm(37) = max(abs(f2)*mdxi(4),mc12i(3),mc12i(2))/2
	    Rm(38) = max(abs(f3)*mdxi(4),mc12i(3),2*mdxi(11))/2
	    mdxi(5) = max(abs(xi4(1))*Rm(30),abs(xi4(4))*Rm(31),
     +		abs(xi4(5))*Rm(32))
	    mdxi(6) = max(abs(xi4(4))*Rm(33),abs(xi4(2))*Rm(34),
     +		abs(xi4(6))*Rm(35))
	    mdxi(7) = max(abs(xi4(5))*Rm(36),abs(xi4(6))*Rm(37),
     +		abs(xi4(3))*Rm(38))
	    mdxi(8) = max(abs(xi4(4))*Rm(30),abs(xi4(2))*Rm(31),
     +		abs(xi4(6))*Rm(32))
	    mdxi(9) = max(abs(xi4(5))*Rm(30),abs(xi4(6))*Rm(31),
     +		abs(xi4(3))*Rm(32))
	    mdxi(10)= max(abs(xi4(5))*Rm(33),abs(xi4(6))*Rm(34),
     +		abs(xi4(3))*Rm(35))
	endif
*	redundancy check
	if ( lwarn .and. atest ) then
	    cxy(1) = xi4(1)*R33+xi4(4)*R34+xi4(5)*R35
	    cxy(2) = xi4(1)*R36+xi4(4)*R37+xi4(5)*R38
	    cxy(3) = xi4(4)*R36+xi4(2)*R37+xi4(6)*R38
	    mxy(1) = abs(xi4(1))*Rm(33)+abs(xi4(4))*Rm(34)+abs(xi4(5))*
     +		Rm(35)
	    mxy(2) = abs(xi4(1))*Rm(36)+abs(xi4(4))*Rm(37)+abs(xi4(5))*
     +		Rm(38)
	    mxy(3) = abs(xi4(4))*Rm(36)+abs(xi4(2))*Rm(37)+abs(xi4(6))*
     +		Rm(38)
	    if ( xloss*absc(cxy(1)-cdxi(8)) .gt.precc*max(mxy(1),
     +			mdxi(8))
     +	    .or. xloss*absc(cxy(2)-cdxi(9)) .gt.precc*max(mxy(2),
     +			mdxi(9))
     +	    .or. xloss*absc(cxy(3)-cdxi(10)).gt.precc*max(mxy(3),
     +			mdxi(10)) ) then
		print *,'ffxdx: error: id/nevent ',id,'/',nevent
		print *,'redundancy check at level 2 failed: '
		print *,cxy(1),cdxi(8),absc(cxy(1)-cdxi(8)),
     +			max(mxy(1),mdxi(8))
		print *,cxy(2),cdxi(9),absc(cxy(2)-cdxi(9)),
     +			max(mxy(2),mdxi(9))
		print *,cxy(3),cdxi(10),absc(cxy(3)-cdxi(20)),
     +			max(mxy(3),mdxi(10))
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'ffxdx : level 2: id,nevent ',id,nevent
	    print *,'D21=',cdxi(5),mdxi(5),ier1
	    print *,'D22=',cdxi(6),mdxi(6),ier1
	    print *,'D23=',cdxi(7),mdxi(7),ier1
	    print *,'D24=',cdxi(8),mdxi(8),ier1
	    print *,'    ',cxy(1),mxy(1)
	    print *,'D25=',cdxi(9),mdxi(9),ier1
	    print *,'    ',cxy(2),mxy(2)
	    print *,'D26=',cdxi(10),mdxi(10),ier1
	    print *,'    ',cxy(3),mxy(3)
	    print *,'D27=',cdxi(11),mdxi(11),ier1
	endif
*	this goes wrong in the case of complex masses - no way out yet...
	if ( awrite .and. .FALSE. ) then
	    d0 = 0
	    xmm = 0
	    if ( awrite ) print *,'calling ffxdi with ier = ',ier
*	    the order of the B0s can be deduced from the C0 -> B0 chain
	    cb0ij(1,2) = cbxj(33)
	    cb0ij(1,3) = cbxj(21)
	    cb0ij(1,4) = cbxj(17)
	    cb0ij(2,1) = cbxj(33)
	    cb0ij(2,3) = cbxj( 9)
	    cb0ij(2,4) = cbxj( 5)
	    cb0ij(3,1) = cbxj(21)
	    cb0ij(3,2) = cbxj( 9)
	    cb0ij(3,4) = cbxj( 1)
	    cb0ij(4,1) = cbxj(17)
	    cb0ij(4,2) = cbxj( 5)
	    cb0ij(4,3) = cbxj( 1)
*	    the A0s are not used for the moment
	    call ffxdi(cd4pppp,cd4ppdel,cd4deldel, cd3ppp,cd3pdel,
     +		cd2pp,cd2del, cd1i, dl2pij, cdxi(1),cc0i,cb0ij,ca0i,
     +		fdel4s,fdel3,fdel2i, xpi,fpij4, d0,xmm, 2, ier)
*  #[ convert to PV conventions:
*
	ier1 = ier
	cd2(1) = cd2pp(1,1) - DBLE(fdel2i(1))*cd2del
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
	cd2(6) = cd2pp(3,3) - DBLE(fdel2i(4))*cd2del
	if ( lwarn .and. absc(cd2(6)).lt.xloss*absc(cd2pp(3,3)) ) then
	    ier0 = ier
	    call ffwarn(229,ier0,absc(cd2(6)),absc(cd2pp(3,3)))
	    ier1 = max(ier1,ier0)
	endif
	cd2(7) = DBLE(fdel3)*cd2del
*
*  #] convert to PV conventions:
	    if ( awrite ) then
		print *,'ffxdi gives'
		print *,'D11 = ',cd1i(1),ier1
		print *,'D12 = ',cd1i(2),ier1
		print *,'D13 = ',cd1i(3),ier1
		print *,'D21 = ',cd2(1),ier1
		print *,'D22 = ',cd2(4),ier1
		print *,'D23 = ',cd2(6),ier1
		print *,'D24 = ',cd2(2),ier1
		print *,'D25 = ',cd2(3),ier1
		print *,'D26 = ',cd2(5),ier1
		print *,'D27 = ',cd2(7),ier1
	    endif
	endif
*
	if ( level.eq.2 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,11
		    if ( absc(cdxi(i)).ne.0 ) then
			xmax = max(xmax,mdxi(i)/absc(cdxi(i)))
		    elseif ( mdxi(i).ne.0 ) then
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
*  #] level 2 :
*  #[ level 3 : D31,D32,D33,D34,D35,D36,D37,D38,D39,D310,D311,D312,D313
*		C21(I),C22(I),C23(I),C11(I),C12(I)
*		need 4 diff C2-functions
	do 15 i=1,4
	    j = 4 +(i-1)*13
	    cc21i(i)=ccxj(j)
	    cc22i(i)=ccxj(j+1)
	    cc23i(i)=ccxj(j+2)
	    cc24i(i)=ccxj(j+3)
	    mc21i(i)=mcxj(j)
	    mc22i(i)=mcxj(j+1)
	    mc23i(i)=mcxj(j+2)
	    mc24i(i)=mcxj(j+3)
   15	continue
*		PV-reduction
	R53=( f1*cdxi(11) + cc24i(2) - cc24i(1) )/2
	R54=( f2*cdxi(11) + cc24i(3) - cc24i(2) )/2
	R55=( f3*cdxi(11) + cc24i(4) - cc24i(3) )/2
	cdxi(22) = xi4(1)*R53+xi4(4)*R54+xi4(5)*R55
	cdxi(23) = xi4(4)*R53+xi4(2)*R54+xi4(6)*R55
	cdxi(24) = xi4(5)*R53+xi4(6)*R54+xi4(3)*R55
	if ( lwarn ) then
***	    Rm(53)=max(abs(f1)*d11max,absc(cc24i(2)),absc(cc24i(1)))/2
***	    Rm(54)=max(abs(f2)*d11max,absc(cc24i(3)),absc(cc24i(2)))/2
***	    Rm(55)=max(abs(f3)*d11max,absc(cc24i(4)),absc(cc24i(3)))/2
***	    d22max = max(abs(xi4(1))*Rm(53),abs(xi4(4))*Rm(54),
***     +		abs(xi4(5))*Rm(55))
***	    if ( absc(cdxi(22)).lt.xloss*d22max ) then
***		ier0 = ier
***		call ffwarn(276,ier0,absc(cdxi(22)),d22max)
***		ier1 = max(ier1,ier0)
***	    endif
***	    d23max = max(abs(xi4(4))*Rm(53),abs(xi4(2))*Rm(54),
***     +		abs(xi4(6))*Rm(55))
***	    if ( absc(cdxi(23)).lt.xloss*d23max ) then
***		ier0 = ier
***		call ffwarn(275,ier0,absc(cdxi(23)),d23max)
***		ier1 = max(ier1,ier0)
***	    endif
***	    d24max = max(abs(xi4(5))*Rm(53),abs(xi4(6))*Rm(54),
***     +		abs(xi4(3))*Rm(55))
***	    if ( absc(cdxi(24)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(274,ier0,absc(cdxi(24)),d24max)
***		ier1 = max(ier1,ier0)
***	    endif
*
*	    and again for the whole thing
*
	    Rm(53)=max(abs(f1)*mdxi(11),mc24i(2),mc24i(1))/2
	    Rm(54)=max(abs(f2)*mdxi(11),mc24i(3),mc24i(2))/2
	    Rm(55)=max(abs(f3)*mdxi(11),mc24i(4),mc24i(3))/2
	    mdxi(22) = max(abs(xi4(1))*Rm(53),abs(xi4(4))*Rm(54),
     +		abs(xi4(5))*Rm(55))
	    mdxi(23) = max(abs(xi4(4))*Rm(53),abs(xi4(2))*Rm(54),
     +		abs(xi4(6))*Rm(55))
	    mdxi(24) = max(abs(xi4(5))*Rm(53),abs(xi4(6))*Rm(54),
     +		abs(xi4(3))*Rm(55))
	endif
*
	R41=( f1*cdxi(5) + cc21i(2) - cc0i(1)  )/2-2*cdxi(22)
	R42=( f2*cdxi(5) + cc21i(3) - cc21i(2) )/2
	R43=( f3*cdxi(5) + cc21i(4) - cc21i(3) )/2
	R44=( f1*cdxi(6) + cc21i(2) - cc21i(1) )/2
	R45=( f2*cdxi(6) + cc22i(3) - cc21i(2) )/2-2*cdxi(23)
	R46=( f3*cdxi(6) + cc22i(4) - cc22i(3) )/2
	R47=( f1*cdxi(7) + cc22i(2) - cc22i(1) )/2
	R48=( f2*cdxi(7) + cc22i(3) - cc22i(2) )/2
	R49=( f3*cdxi(7)            - cc22i(3) )/2-2*cdxi(24)
	R50=( f1*cdxi(8) + cc21i(2) + cc11i(1) )/2-cdxi(23)
	R51=( f2*cdxi(8) + cc23i(3) - cc21i(2) )/2-cdxi(22)
	R52=( f3*cdxi(8) + cc23i(4) - cc23i(3) )/2
	cdxi(12) = xi4(1)*R41+xi4(4)*R42+xi4(5)*R43
	cdxi(13) = xi4(4)*R44+xi4(2)*R45+xi4(6)*R46
	cdxi(14) = xi4(5)*R47+xi4(6)*R48+xi4(3)*R49
	cdxi(15) = xi4(4)*R41+xi4(2)*R42+xi4(6)*R43
	cdxi(16) = xi4(5)*R41+xi4(6)*R42+xi4(3)*R43
	cdxi(17) = xi4(1)*R44+xi4(4)*R45+xi4(5)*R46
	cdxi(18) = xi4(1)*R47+xi4(4)*R48+xi4(5)*R49
	cdxi(19) = xi4(5)*R44+xi4(6)*R45+xi4(3)*R46
	cdxi(20) = xi4(4)*R47+xi4(2)*R48+xi4(6)*R49
	cdxi(21) = xi4(5)*R50+xi4(6)*R51+xi4(3)*R52
	if ( lwarn ) then
***	    Rm(41)=max(absc(f1*cdxi(5)),absc(cc21i(2)),absc(cc0i(1)),
***     +		4*d22max)/2
***	    Rm(42)=max(absc(f2*cdxi(5)),absc(cc21i(3)),absc(cc21i(2)))/2
***	    Rm(43)=max(absc(f3*cdxi(5)),absc(cc21i(4)),absc(cc21i(3)))/2
***	    Rm(44)=max(absc(f1*cdxi(6)),absc(cc21i(2)),absc(cc21i(1)))/2
***	    Rm(45)=max(absc(f2*cdxi(6)),absc(cc22i(3)),absc(cc21i(2)),
***     +		4*d23max)/2
***	    Rm(46)=max(absc(f3*cdxi(6)),absc(cc22i(4)),absc(cc22i(3)))/2
***	    Rm(47)=max(absc(f1*cdxi(7)),absc(cc22i(2)),absc(cc22i(1)))/2
***	    Rm(48)=max(absc(f2*cdxi(7)),absc(cc22i(3)),absc(cc22i(2)))/2
***	    Rm(49)=max(absc(f3*cdxi(7)),absc(cc22i(3)),4*d24max)/2
***	    Rm(50)=max(absc(f1*cdxi(8)),absc(cc21i(2)),absc(cc11i(1)),
***     +		2*d23max)/2
***	    Rm(51)=max(absc(f2*cdxi(8)),absc(cc23i(3)),absc(cc21i(2)),
***     +		2*d22max)/2
***	    Rm(52)=max(absc(f3*cdxi(8)),absc(cc23i(4)),absc(cc23i(3)))/2
***	    xmax=max(abs(xi4(1))*Rm(41),abs(xi4(4))*Rm(42),
***     +		abs(xi4(5))*Rm(43))
***	    if ( absc(cdxi(12)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(273,ier0,absc(cdxi(12)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(4))*Rm(44),abs(xi4(2))*Rm(45),
***     +		abs(xi4(6))*Rm(46))
***	    if ( absc(cdxi(13)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(272,ier0,absc(cdxi(13)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(5))*Rm(47),abs(xi4(6))*Rm(48),
***     +		abs(xi4(3))*Rm(49))
***	    if ( absc(cdxi(14)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(271,ier0,absc(cdxi(14)),xmax)
***		if ( awrite ) then
***		print *,'xi4(5)*R47,xi4(6)*R48,xi4(3)*R49,cdxi(14) = '
***		print *,xi4(5)*R47,xi4(6)*R48,xi4(3)*R49,cdxi(14)
***		print *,xi4(5)*Rm(47),xi4(6)*Rm(48),xi4(3)*Rm(49),xmax
***		endif
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(4))*Rm(41),abs(xi4(2))*Rm(42),
***     +		abs(xi4(6))*Rm(43))
***	    if ( absc(cdxi(15)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(270,ier0,absc(cdxi(15)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(5))*Rm(41),abs(xi4(6))*Rm(42),
***     +		abs(xi4(3))*Rm(43))
***	    if ( absc(cdxi(16)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(269,ier0,absc(cdxi(16)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(1))*Rm(44),abs(xi4(4))*Rm(45),
***     +		abs(xi4(5))*Rm(46))
***	    if ( absc(cdxi(17)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(268,ier0,absc(cdxi(17)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(1))*Rm(47),abs(xi4(4))*Rm(48),
***     +		abs(xi4(5))*Rm(49))
***	    if ( absc(cdxi(18)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(267,ier0,absc(cdxi(18)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(5))*Rm(44),abs(xi4(6))*Rm(45),
***     +		abs(xi4(3))*Rm(46))
***	    if ( absc(cdxi(19)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(266,ier0,absc(cdxi(19)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(4))*Rm(47),abs(xi4(2))*Rm(48),
***     +		abs(xi4(6))*Rm(49))
***	    if ( absc(cdxi(20)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(265,ier0,absc(cdxi(20)),xmax)
***		if ( awrite ) then
***		print *,'xi4(4)*R47,xi4(2)*R48,xi4(6)*R49,cdxi(20) = '
***		print *,xi4(4)*R47,xi4(2)*R48,xi4(6)*R49,cdxi(20)
***		print *,xi4(4)*Rm(47),xi4(2)*Rm(48),xi4(6)*Rm(49),xmax
***		endif
***		ier1 = max(ier1,ier0)
***	    endif
***	    xmax=max(abs(xi4(5))*Rm(50),abs(xi4(6))*Rm(51),
***     +		abs(xi4(3))*Rm(52))
***	    if ( absc(cdxi(21)).lt.xloss*xmax ) then
***		ier0 = ier
***		call ffwarn(264,ier0,absc(cdxi(21)),xmax)
***		ier1 = max(ier1,ier0)
***	    endif
*
*	    again the whole thing, not just this step
*
	    Rm(41) = max(abs(f1)*mdxi(5),mc21i(2),mc0i(1),4*mdxi(22))/2
	    Rm(42) = max(abs(f2)*mdxi(5),mc21i(3),mc21i(2))/2
	    Rm(43) = max(abs(f3)*mdxi(5),mc21i(4),mc21i(3))/2
	    Rm(44) = max(abs(f1)*mdxi(6),mc21i(2),mc21i(1))/2
	    Rm(45) = max(abs(f2)*mdxi(6),mc22i(3),mc21i(2),4*mdxi(23))/2
	    Rm(46) = max(abs(f3)*mdxi(6),mc22i(4),mc22i(3))/2
	    Rm(47) = max(abs(f1)*mdxi(7),mc22i(2),mc22i(1))/2
	    Rm(48) = max(abs(f2)*mdxi(7),mc22i(3),mc22i(2))/2
	    Rm(49) = max(abs(f3)*mdxi(7),mc22i(3),4*mdxi(24))/2
	    Rm(50) = max(abs(f1)*mdxi(8),mc21i(2),mc11i(1),2*mdxi(23))/2
	    Rm(51) = max(abs(f2)*mdxi(8),mc23i(3),mc21i(2),2*mdxi(22))/2
	    Rm(52) = max(abs(f3)*mdxi(8),mc23i(4),mc23i(3))/2
	    mdxi(12) = max(abs(xi4(1))*Rm(41),abs(xi4(4))*Rm(42),
     +		abs(xi4(5))*Rm(43))
	    mdxi(13) = max(abs(xi4(4))*Rm(44),abs(xi4(2))*Rm(45),
     +		abs(xi4(6))*Rm(46))
	    mdxi(14) = max(abs(xi4(5))*Rm(47),abs(xi4(6))*Rm(48),
     +		abs(xi4(3))*Rm(49))
	    mdxi(15) = max(abs(xi4(4))*Rm(41),abs(xi4(2))*Rm(42),
     +		abs(xi4(6))*Rm(43))
	    mdxi(16) = max(abs(xi4(5))*Rm(41),abs(xi4(6))*Rm(42),
     +		abs(xi4(3))*Rm(43))
	    mdxi(17) = max(abs(xi4(1))*Rm(44),abs(xi4(4))*Rm(45),
     +		abs(xi4(5))*Rm(46))
	    mdxi(18) = max(abs(xi4(1))*Rm(47),abs(xi4(4))*Rm(48),
     +		abs(xi4(5))*Rm(49))
	    mdxi(19) = max(abs(xi4(5))*Rm(44),abs(xi4(6))*Rm(45),
     +		abs(xi4(3))*Rm(46))
	    mdxi(20) = max(abs(xi4(4))*Rm(47),abs(xi4(2))*Rm(48),
     +		abs(xi4(6))*Rm(49))
	    mdxi(21) = max(abs(xi4(5))*Rm(50),abs(xi4(6))*Rm(51),
     +		abs(xi4(3))*Rm(52))
	endif
*	redundancy check
	if ( lwarn .and. atest ) then
	    cxy(1) = xi4(1)*R50+xi4(4)*R51+xi4(5)*R52
	    cxy(2) = xi4(4)*R50+xi4(2)*R51+xi4(6)*R52
	    mxy(1) = abs(xi4(1))*Rm(50)+abs(xi4(4))*Rm(51)+ abs(xi4(5))*
     +		Rm(52)
	    mxy(2) = abs(xi4(4))*Rm(50)+abs(xi4(2))*Rm(51)+ abs(xi4(6))*
     +		Rm(52)
	    if ( xloss*absc(cxy(1)-cdxi(15)).gt.precc*max(mxy(1),
     +			mdxi(15))
     +	    .or. xloss*absc(cxy(2)-cdxi(17)).gt.precc*max(mxy(2),
     +			mdxi(17))  ) then
		print *,'ffxdx: error: id/nevent ',id,'/',nevent
		print *,'redundancy check at level 3 failed: '
		print *,cxy(1),cdxi(15),absc(cxy(1)-cdxi(15)),
     +			max(mxy(1),mdxi(15))
		print *,cxy(2),cdxi(17),absc(cxy(2)-cdxi(17)),
     +			max(mxy(2),mdxi(17))
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'ffxdx : level 3: id,nevent ',id,nevent
	    print *,'D31 =',cdxi(12),mdxi(12),ier1
	    print *,'D32 =',cdxi(13),mdxi(13),ier1
	    print *,'D33 =',cdxi(14),mdxi(14),ier1
	    print *,'D34 =',cdxi(15),mdxi(15),ier1
	    print *,'     ',cxy(1)  ,mxy(1)
	    print *,'D35 =',cdxi(16),mdxi(16),ier1
	    print *,'D36 =',cdxi(17),mdxi(17),ier1
	    print *,'     ',cxy(2)  ,mxy(2)
	    print *,'D37 =',cdxi(18),mdxi(18),ier1
	    print *,'D38 =',cdxi(19),mdxi(19),ier1
	    print *,'D39 =',cdxi(20),mdxi(20),ier1
	    print *,'D310=',cdxi(21),mdxi(21),ier1
	    print *,'D311=',cdxi(22),mdxi(22),ier1
	    print *,'D312=',cdxi(23),mdxi(23),ier1
	    print *,'D313=',cdxi(24),mdxi(24),ier1
	endif
*
	if ( level.eq.3 ) then
	    if ( lwarn ) then
		xmax = 0
		do i=1,24
		    if ( absc(cdxi(i)).ne.0 ) then
			xmax = max(xmax,mdxi(i)/absc(cdxi(i)))
		    elseif ( mdxi(i).ne.0 ) then
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
*  #] level 3 :
*  #[ end:
	print *,'ffxdx: level ',level,' not supported.'
	stop
  990	continue
	ier = ier1 + ier2
*  #] end:
*###] ffxdx:
	end
*###[ ffdji:
	subroutine ffdji(ccxi,mcxi,cbxi,mbxi,caxi,maxi,
     +		ccxj,mcxj,cbxj,mbxj,caxj,maxj,level)
***#[*comment:***********************************************************
*									*
*	Renumber the [mc][abc]xj arrays into the [mc][abc]xi arrays.	*
*	Note: the A's are not yet used and not yet renumbered!		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer level
	DOUBLE PRECISION mcxi(28),mbxi(12),maxi(4),
     +		mcxj(52),mbxj(48),maxj(12)
	DOUBLE COMPLEX ccxi(28),cbxi(12),caxi(4),
     +		ccxj(52),cbxj(48),caxj(12)
*
*	local variables
*
	integer i,j,k,bij(12),beq(6,2)
	save bij,beq
*
*	common
*
	include 'aa.h'
	include 'ff.h'
*
*	data
*
	data bij /1,2,5,6,9,10,17,18,21,22,33,34/
	data beq / 0, 4, 8,16,20,32,
     +		  12,24,36,28,40,44/
*
*  #] declarations:
*  #[ renumber:
*	output preparation
*	   1)C-output: reduce the array ccxj(4*13) to ccxi(4*7)
*		       c's are calculated only to (level-1)
	do 130 j=1,4
	    do 131 i=1,7
		ccxi(i+(j-1)*7) = ccxj(i+(j-1)*13)
		mcxi(i+(j-1)*7) = mcxj(i+(j-1)*13)
 131	    continue
 130	continue
*	   2)B-output: reduce the array cbxj(12*4) to cbxi(6*2)
*		       b's are calculated only to (level-2)
	do i=1,12
	    cbxi(i) = cbxj(bij(i))
	    mbxi(i) = mbxj(bij(i))
	enddo
*	check the symmetry in B0(i,j)
	if ( atest ) then
	    do 13 i=1,4
		do 12 j=1,6
		    if ( xloss*abs(cbxj(i+beq(j,1))-cbxj(i+beq(j,2)))
     +		    		.gt. precc*abs(cbxj(i+beq(j,1))) ) then
			print *,'ffxdji: cbxj(',i+beq(j,1),') != cbxj(',
     +				i+beq(j,2),'): ',cbxj(i+beq(j,1)),
     +				cbxj(i+beq(j,2)),cbxj(i+beq(j,1))-
     +				cbxj(i+beq(j,2))
		    endif
   12		continue
   13	    continue
	endif
*  #] renumber:
*###] ffdji:
	end
