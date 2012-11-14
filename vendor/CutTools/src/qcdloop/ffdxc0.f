*###[ ffdxc0:
	subroutine ffdxc0(cs3,ipi12,isoort,clogi,ilogi,xpi,dpipj,piDpj,
     +		xqi,dqiqj,qiDqj,sdel2,del2s,etalam,etami,delpsi,alph,
     +		ddel2s,ldel2s,npoin,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the difference of two threepoint functions		*
*	C(3,...a) - C(4,...b)						*
*	For this we not only calculate the roots of the three-point	*
*	function y,z(1-4,3-4,1-3) but also the combinations		*
*									*
*		yzzy = y(,4,)*z(,3,) - z(,4,)*y(,3,)			*
*	and								*
*		yyzz = y(,4,) - z(,4,) - y(,3,) + z(,3,)		*
*									*
*	This is done explicitly for most special cases, so a lot of	*
*	lines of code result.  This may be shortened with a smart use	*
*	of indices, however, it is readable now.			*
*									*
*	Input:	xpi(6,3:4)	(real)	transformed mi,pi squared in Ci	*
*		dpipj(6,6,3:4)	(real)	xpi(i)-xpi(j)			*
*		piDpj(6,6,3:4)	(real)	pi(i).pi(j)			*
*		xqi(10,10)	(real)	transformed mi,pi squared in D	*
*		dqiqj(10,10)	(real)	xqi(i)-xqi(j)			*
*		qiDqj(10,10)	(real)	qi(i).qi(j)			*
*		sdel2		(real)	sqrt(delta_{p_1 p_2}^{p_1 p_2})	*
*		del2s(3,3:4)	(real)	delta_{p_i s_i}^{p_i s_i}	*
*		etalam(3:4)	(real)	delta_{s_1 s_2 s_3}^{s_1 s_2 s_3}
*					  /delta_{p_1 p_2}^{p_1 p_2}	*
*		etami(6,3:4)	(real)	m_i^2 - etalam			*
*		ddel2s(2:3)	(real)	del2s(i,3) - del2s(i,4)		*
*		alph(3)		(real)	alph(1)=alpha, alph(3)=1-alpha	*
*		ldel2s	 (logical)	indicates yes/no limit del2s->0	*
*									*
*	Output: cs3	 (complex)(160)	C0(3)-C0(4), not yet summed.	*
*		ipi12	 (integer)(6)	factors pi^2/12, not yet summed	*
*		slam	 (complex)	lambda(p1,p2,p3).		*
*		isoort	 (integer)(16)	indication of he method used	*
*		clogi	 (complex)(6)	log(-dyz(2,1,i)/dyz(2,2,i))	*
*		ilogi	 (integer)(6)	factors i*pi in this		*
*		ier	 (integer)	0=ok, 1=inaccurate, 2=error	*
*									*
*	Calls:	...							*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(16),isoort(16),ilogi(6),npoin,ier
	logical ldel2s
	DOUBLE COMPLEX cs3(160),clogi(6)
	DOUBLE PRECISION xqi(10),dqiqj(10,10),qiDqj(10,10),
     +		xpi(6,3:4),dpipj(6,6,3:4),piDpj(6,6,3:4),
     +		sdel2,del2s(3,3:4),etalam(3:4),etami(6,3:4),alph(3),
     +		ddel2s(2:3),delpsi(3,3:4)
*
*	local variables:
*
	integer i,j,k,l,ip,ier0,ii,ifirst,ieri(12),idone(6)
	logical lcompl
	DOUBLE COMPLEX c,csom,chck,cs(5),csdeli(3,3:4),csdel2,
     +		cy(4,3:4,3),cz(4,3:4,3),cdyz(2,2,3:4,3),cd2yzz(3:4,3),
     +		cpi(6,3:4),cpiDpj(6,6,3:4),cdyzzy(4,3),cdyyzz(2,3)
	DOUBLE PRECISION sdel2i(3,3:4),s(5),som,smax,absc,dfflo1,xhck,
     +		rloss,y(4,3:4,3),z(4,3:4,3),dyz(2,2,3:4,3),d2yzz(3:4,3),
     +		dy2z(4,3:4,3),dyzzy(4,3),dsdel2,xmax
	DOUBLE COMPLEX zxfflg,zfflog,zfflo1
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
*	    call ffxhck(xpi(1,3),dpipj(1,1,3),6,ier)
*	    call ffxhck(xpi(1,4),dpipj(1,1,4),6,ier)
	    call ffxhck(xqi,dqiqj,10,ier)
	endif
*  #] check input:
*  #[ get y,z-roots:
	lcompl = .FALSE.
	if ( lwrite ) print '(a)','  ##[ get roots:'
	do 20 k=3,4
	do 10 i=1,3
*
*	get roots (y,z) and flag what to do: 0=nothing, 1=normal,
*	-1=complex
*
	ip = i+3
*	first get the roots
	if ( del2s(i,k) .le. 0 ) then
*	    real case
	    sdel2i(i,k) = sqrt(-del2s(i,k))
*	    then handle the special case Si = 0
	    if ( xpi(ip,k) .eq. 0 ) then
		if ( i .eq. 1 .and. alph(3) .eq. 0 .or.
     +		     i .eq. 3 .and. alph(1) .eq. 0 ) then
		    isoort(2*i-1+8*(k-3)) = 0
		    isoort(2*i+8*(k-3)) = 0
		    goto 10
		endif
	    endif
	    call ffxxyz(y(1,k,i),z(1,k,i),dyz(1,1,k,i),d2yzz(k,i),
     +		dy2z(1,k,i),i,sdel2,sdel2i(i,k),etalam(k),etami(1,k),
     +		delpsi(i,k),xpi(1,k),dpipj(1,1,k),piDpj(1,1,k),
     +		isoort(2*i-1+8*(k-3)),ldel2s,6,ier)
	else
*	    complex case
	    sdel2i(i,k) = sqrt(del2s(i,k))
	    csdeli(i,k) = DCMPLX(x0,sdel2i(i,k))
	    lcompl = .TRUE.
	    call ffcxyz(cy(1,k,i),cz(1,k,i),cdyz(1,1,k,i),cd2yzz(k,i),i,
     +		sdel2,sdel2i(i,k),etalam(k),etami(1,k),delpsi(i,k),xpi(
     +		1,k),piDpj(1,1,k),isoort(2*i-1+8*(k-3)),ldel2s,6,ier)
	endif
   10	continue
   20	continue
*  #] get y,z-roots:
*  #[ convert to complex if necessary:
	do 60 i=2,3
	    l = 2*i-1
	    if ( isoort(l).gt.0 .and. isoort(l+8).lt.0 ) then
		k = 3
*		we get  -5, -105 if they have equal roots, isoort=+2
*			-6, -106 if they have unequal roots, isoort=+1
		if ( .not.ldel2s ) then
		    isoort(l) = isoort(l)-7
		    isoort(l+1) = isoort(l+1)-7
		else
		    isoort(l) = isoort(l)-207
		    isoort(l+1) = isoort(l+1)-207
		endif
	    elseif ( isoort(l).lt.0 .and. isoort(l+8).gt.0 ) then
		k = 4
		if ( .not.ldel2s ) then
		    isoort(l+8) = isoort(l+8)-7
		    isoort(l+9) = isoort(l+9)-7
		else
		    isoort(l+8) = isoort(l+8)-207
		    isoort(l+9) = isoort(l+9)-207
		endif
	    else
		k = 0
	    endif
	    if ( k .ne. 0 ) then
		if ( lwrite ) print *,'ffdxc0: converting i,k=',i,k,
     +			' to complex'
		do 30 j=1,4
		    cy(j,k,i) = y(j,k,i)
		    cz(j,k,i) = z(j,k,i)
   30		continue
		do 50 j=1,2
		    do 40 l=1,2
			cdyz(l,j,k,i) = dyz(l,j,k,i)
   40		    continue
   50		continue
		cd2yzz(k,i) = d2yzz(k,i)
		csdeli(i,k) = sdel2i(i,k)
	    endif
   60	continue
*  #] convert to complex if necessary:
*  #[ get differences:
*
*	the only important differences are y4z3-z3y4 and (1-y4)(1-z3)-
*	(1-y3)(1-z4)
*
	do 100 i=1,12
	    ieri(i) = 0
  100	continue
*  #[	vertices (1):
	som = qiDqj(7,2)/sdel2
	if ( isoort(1) .ge. 0 ) then
*	    Note that the isoorts are equal as the vertex is equal.
*
*	    flag if we have a cancellation
*
	    if ( abs(som) .lt. xloss ) then
		isoort(1) = isoort(1) + 10
		isoort(9) = isoort(9) + 10
	    endif
	    do 110 k=1,4
		dyzzy(k,1) = som*z(k,3,1)
		if ( k .gt. 2 ) dyzzy(k,1) = -dyzzy(k,1)
  110	    continue
	else
	    if ( abs(som) .lt. xloss ) then
		isoort(1) = isoort(1) - 10
		isoort(9) = isoort(9) - 10
	    endif
	    do 120 k=1,4
		cdyzzy(k,1) = DBLE(som)*cz(k,3,1)
		if ( k .gt. 2 ) cdyzzy(k,1) = -cdyzzy(k,1)
  120	    continue
	    cdyyzz(1,1) = som
	    cdyyzz(2,1) = som
	    if ( lwrite ) then
		print *,'cdyyzz(11) =',cy(2,4,1)-cy(2,3,1),
     +			absc(cy(2,4,1))
		print *,'cdyyzz(11)+=',cdyyzz(1,1)
	    endif
	endif
*  #]	vertices (1):
*  #[	vertices (2):
	if ( isoort(3) .ge. 0 ) then
*  #[	    real case: (note that this implies isoort(11)>0)
	    ifirst = 0
	    do 150 j=1,2
	    do 140 k=1,2
		ii = 2*(j-1) + k
		dyzzy(ii,2) = y(2*j,4,2)*z(ii,3,2)-y(2*j,3,2)*z(ii,4,2)
		xmax = abs(y(2*j,4,2)*z(ii,3,2))
		if ( abs(dyzzy(ii,2)) .ge. xmax ) goto 140
		isoort(3) = isoort(3) + 10
		isoort(11) = isoort(11) + 10
 1000		format(a,i1,a,g22.14,g12.4)
		if ( lwrite ) print 1000,'dyzzy(',ii,'2)  = ',
     +			dyzzy(ii,2),xmax
		if ( ldel2s ) then
		    print *,'ffdxc0: not ready for del2s=0, real case'
		    goto 130
		endif
		if ( ifirst .le. 0 ) then
		    if ( ddel2s(2) .eq. 0 ) then
			dsdel2 = 0
		    else
			dsdel2 = ddel2s(2)/(sdel2i(2,3)+sdel2i(2,4))
		    endif
		endif
		if ( ifirst .le. 1 ) then
		    if ( j .eq. 1 ) then
			s(1) = xqi(6)*qiDqj(7,4)*qiDqj(5,4)/sdel2
			s(2) = -qiDqj(7,4)*sdel2i(2,3)
			s(3) = +qiDqj(6,4)*dsdel2
		    else
			s(1) = xqi(6)*qiDqj(7,2)*qiDqj(5,2)/sdel2
			s(2) = -qiDqj(7,2)*sdel2i(2,3)
			s(3) = +qiDqj(6,2)*dsdel2
		    endif
		endif
		if ( ifirst .le. 0 ) then
		    ifirst = 2
		    s(4) = -qiDqj(5,10)*qiDqj(7,4)*sdel2i(2,3)/sdel2
		    s(5) = delpsi(2,3)*dsdel2/sdel2
		endif
		if ( k .eq. 1 ) then
		    som = s(1) + s(2) + s(3) + s(4) + s(5)
		else
		    som = s(1) - s(2) - s(3) - s(4) - s(5)
		endif
		smax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +			abs(s(5)))/xqi(6)**2
		if ( lwrite ) then
		    print 1000,'dyzzy(',ii,'2)+ = ',som/xqi(6)**2,smax
*		    print *,(s(i)/xqi(6)**2,i=1,5)
		endif
		if ( smax .lt. xmax ) then
		    dyzzy(ii,2) = som/xqi(6)**2
		    xmax = smax
		endif
  130		continue
		if ( lwarn .and. abs(dyzzy(ii,2)) .lt. xloss*xmax ) then
		    call ffwarn(140,ieri(2*k+j-2),dyzzy(ii,2),xmax)
		endif
  140	    continue
	    ifirst = ifirst - 1
  150	    continue
*  #]	    real case:
	else
*  #[	    complex case:
	    ifirst = 0
	    do 180 j=1,2
	    do 170 k=1,2
		ii = 2*(j-1) + k
		cdyzzy(ii,2) = cy(2*j,4,2)*cz(ii,3,2)-cy(2*j,3,2)*
     +			cz(ii,4,2)
		xmax = absc(cy(2*j,4,2)*cz(ii,3,2))
		if ( absc(cdyzzy(ii,2)) .ge. xmax ) goto 170
		isoort(3) = isoort(3) - 10
		isoort(11) = isoort(11) - 10
 1002		format(a,i1,a,2g22.14,g12.4)
		if ( lwrite ) print 1002,'cdyzzy(',ii,'2)  =',
     +						cdyzzy(ii,2),xmax
		if ( ldel2s ) then
		    ip = 3
		else
		    ip = 6
		endif
		if ( mod(isoort(3),10).ne.0 .or. mod(isoort(11),10).ne.0
     +								) then
*
*		    one of the roots is really real
*
		    if ( ifirst .le. 0 ) then
			csdel2=DBLE(ddel2s(2))/(csdeli(2,3)+csdeli(2,4))
		    endif
		    if ( ifirst .le. 1 ) then
			if ( j .eq. 1 .neqv. ldel2s ) then
			    if ( .not.ldel2s ) then
				cs(1)=xqi(6)*qiDqj(7,4)*qiDqj(5,4)/sdel2
				cs(2) = -DBLE(qiDqj(7,4))*csdeli(2,3)
				cs(3) = +DBLE(qiDqj(6,4))*csdel2
			    else
				cs(1)=-xqi(3)*qiDqj(5,10)*qiDqj(7,2)/
     +								sdel2
				cs(2) = -DBLE(qiDqj(7,2))*csdeli(2,3)
				cs(3) = -DBLE(qiDqj(6,3))*csdel2
			    endif
			else
			    cs(1) = xqi(ip)*qiDqj(7,2)*qiDqj(5,2)/sdel2
			    cs(2) = -DBLE(qiDqj(7,2))*csdeli(2,3)
			    cs(3) = +DBLE(qiDqj(ip,2))*csdel2
			endif
		    endif
		    if ( ifirst .le. 0 ) then
			ifirst = 2
			if ( .not.ldel2s ) then
			    cs(4) = -DBLE(qiDqj(5,10)*qiDqj(7,4)/sdel2)*
     +				csdeli(2,3)
			else
			    cs(4) = -DBLE(qiDqj(5,3)*qiDqj(7,2)/sdel2)*
     +				csdeli(2,3)
			endif
			cs(5) = DBLE(delpsi(2,3)/sdel2)*csdel2
		    endif
		else
*
*		    both roots are complex
*
		    if ( ifirst .eq. 0 ) then
			dsdel2 = -ddel2s(2)/(sdel2i(2,3)+sdel2i(2,4))
			csdel2 = DCMPLX(x0,dsdel2)
		    endif
		    if ( ifirst .le. 1 ) then
			if ( j .eq. 1 .neqv. ldel2s ) then
			    if ( .not.ldel2s ) then
				cs(1)=xqi(6)*qiDqj(7,4)*qiDqj(5,4)/sdel2
				cs(2)=-DCMPLX(x0,qiDqj(7,4)*sdel2i(2,3))
				cs(3)=+DCMPLX(x0,qiDqj(6,3)*dsdel2)
			    else
				cs(1)=-xqi(3)*qiDqj(5,10)*qiDqj(7,2)/
     +								sdel2
				cs(2)=-DCMPLX(x0,qiDqj(7,2)*sdel2i(2,3))
				cs(3)=-DCMPLX(x0,qiDqj(6,3)*dsdel2)
			    endif
			else
			    cs(1) = xqi(ip)*qiDqj(7,2)*qiDqj(5,2)/sdel2
			    cs(2) = -DCMPLX(x0,qiDqj(7,2)*sdel2i(2,3))
			    cs(3) = +DCMPLX(x0,qiDqj(ip,2)*dsdel2)
			endif
		    endif
		    if ( ifirst .eq. 0 ) then
			ifirst = 2
			if ( .not.ldel2s ) then
			    cs(4) = -DCMPLX(x0,qiDqj(5,10)*qiDqj(7,4)*
     +						sdel2i(2,3)/sdel2)
			else
			    cs(4) = -DCMPLX(x0,qiDqj(5,3)*qiDqj(7,2)*
     +						sdel2i(2,3)/sdel2)
			endif
			cs(5) = DCMPLX(x0,delpsi(2,3)*dsdel2/sdel2)
		    endif
		endif
		if ( k .eq. 1 ) then
		    csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
		else
		    csom = cs(1) - cs(2) - cs(3) - cs(4) - cs(5)
		endif
		smax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)),
     +			absc(cs(4)),absc(cs(5)))/xqi(ip)**2
		if ( lwrite ) then
		    print 1002,'cdyzzy(',ii,'2)+ =',csom/DBLE(xqi(ip))**
     +			2,smax
***		    print *,(cs(i)/DBLE(xqi(ip))**2,i=1,5)
		endif
		if ( smax .lt. xmax ) then
		    cdyzzy(ii,2) = csom/DBLE(xqi(ip))**2
		    xmax = smax
		endif
		if ( lwarn .and. absc(cdyzzy(ii,2)).lt.xloss*xmax ) then
		  call ffwarn(140,ieri(2*k+j-2),absc(cdyzzy(ii,2)),xmax)
		endif
  170	    continue
*
*	    get cdyyzz
*
	    if ( ldel2s ) then
		cdyyzz(j,2) = cdyz(2,j,4,2) - cdyz(2,j,3,2)
		xmax = absc(cdyz(2,j,4,2))
		if ( absc(cdyyzz(j,2)) .ge. xloss*xmax ) goto 175
		if ( lwrite ) print 1002,'cdyyzz(',j,'2) =',cdyyzz(j,2),
     +			xmax
		if ( ifirst .le. 0 ) then
		    if ( mod(isoort( 3),10).ne.0 .or.
     +			 mod(isoort(11),10).ne.0 ) then
			csdel2=DBLE(ddel2s(2))/(csdeli(2,3)+csdeli(2,4))
		    else
			dsdel2 = -ddel2s(2)/(sdel2i(2,3)+sdel2i(2,4))
			csdel2 = DCMPLX(x0,dsdel2)
		    endif
		endif
		cs(2) = csdel2/DBLE(xqi(3))
		cs(1) = qiDqj(5,3)*qiDqj(7,2)/(sdel2*xqi(3))
		if ( j .eq. 1 ) then
		    csom = cs(1) + cs(2)
		else
		    csom = cs(1) - cs(2)
		endif
		smax = absc(cs(1))
		if ( lwrite ) print 1002,'cdyyzz(',j,'2)+=',csom,smax
		if ( smax .lt. xmax ) then
		    cdyyzz(j,2) = csom
		    xmax = smax
		endif
		if ( lwarn .and. absc(cdyyzz(j,2)).lt.xloss*xmax ) then
		    call ffwarn(147,ieri(7+j),absc(cdyyzz(j,2)),xmax)
		endif
	    endif
*
*	    bookkeeping
*
  175	    continue
	    ifirst = ifirst - 1
  180	    continue
*  #]	    complex case:
	endif
*  #]	vertices (2):
*  #[	vertices (3):
	if ( isoort(5) .ge. 0 ) then
*  #[	    real case: (note that this implies isoort(15)>0)
	    ifirst = 0
	    do 210 j=1,2
	    do 200 k=1,2
		ii = 2*(j-1) + k
		dyzzy(ii,3) = y(2*j,4,3)*z(ii,3,3)-y(2*j,3,3)*z(ii,4,3)
		xmax = abs(y(2*j,4,3)*z(ii,3,3))
		if ( abs(dyzzy(ii,3)) .ge. xmax ) goto 200
		isoort(5) = isoort(5) + 10
		isoort(13) = isoort(13) + 10
		if ( lwrite ) print 1000,'dyzzy(',ii,'3)  = ',
     +			dyzzy(ii,3),xmax
		if ( ldel2s ) then
		    print *,'ffdxc0: not ready for del2s=0, real case'
		    goto 190
		endif
		if ( ifirst .le. 0 ) then
		    if ( ddel2s(2) .eq. 0 ) then
			dsdel2 = 0
		    else
			dsdel2 = ddel2s(3)/(sdel2i(3,3)+sdel2i(3,4))
		    endif
		endif
		if ( ifirst .le. 1 ) then
		    if ( j .eq. 1 ) then
			s(1) = xqi(8)*qiDqj(7,1)*qiDqj(5,1)/sdel2
			s(2) = +qiDqj(7,1)*sdel2i(3,3)
			s(3) = +qiDqj(9,1)*dsdel2
		    else
			s(1) = xqi(8)*qiDqj(7,4)*qiDqj(5,4)/sdel2
			s(2) = +qiDqj(7,4)*sdel2i(3,3)
			s(3) = +qiDqj(9,4)*dsdel2
		    endif
		endif
		if ( ifirst .le. 0 ) then
		    ifirst = 2
		    s(4) = -qiDqj(5,9)*qiDqj(7,1)*sdel2i(3,3)/sdel2
		    s(5) = delpsi(3,3)*dsdel2/sdel2
		endif
		if ( k .eq. 1 ) then
		    som = s(1) + s(2) + s(3) + s(4) + s(5)
		else
		    som = s(1) - s(2) - s(3) - s(4) - s(5)
		endif
		smax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +			abs(s(5)))/xqi(8)**2
		if ( lwrite ) then
		    print 1000,'dyzzy(',ii,'3)+ = ',som/xqi(8)**2,smax
***		    print *,(s(i)/xqi(8)**2,i=1,5)
		endif
		if ( smax .lt. xmax ) then
		    dyzzy(ii,3) = som/xqi(8)**2
		    xmax = smax
		endif
  190		continue
		if ( lwarn .and. abs(dyzzy(ii,3)) .lt. xloss*xmax ) then
		    call ffwarn(140,ieri(2*k+j+2),dyzzy(ii,3),xmax)
		endif
  200	    continue
	    ifirst = ifirst - 1
  210	    continue
*  #]	    real case:
	else
*  #[	    complex case:
	    ifirst = 0
	    do 240 j=1,2
	    do 230 k=1,2
		ii = 2*(j-1) + k
		cdyzzy(ii,3) = cy(2*j,4,3)*cz(ii,3,3)-cy(2*j,3,3)*
     +			cz(ii,4,3)
		xmax = absc(cy(2*j,4,3)*cz(ii,3,3))
		if ( absc(cdyzzy(ii,3)) .ge. xmax ) goto 230
		isoort(5) = isoort(5) - 10
		isoort(13) = isoort(13) - 10
		if ( lwrite ) print 1002,'cdyzzy(',ii,'3)  =',
     +						cdyzzy(ii,3),xmax
		if ( ldel2s ) then
		    ip = 3
		else
		    ip = 8
		endif
		if ( mod(isoort(3),10).ne.0 .or. mod(isoort(11),10).ne.0
     +								) then
*
*		    one of the roots is really real
*
		    if ( ifirst .le. 0 ) then
			csdel2=DBLE(ddel2s(3))/(csdeli(3,3)+csdeli(3,4))
		    endif
		    if ( ifirst .le. 1 ) then
			if ( j .eq. 1 ) then
			    cs(1) = xqi(ip)*qiDqj(7,1)*qiDqj(5,1)/sdel2
			    cs(2) = +DBLE(qiDqj(7,1))*csdeli(3,3)
			    if ( .not.ldel2s ) then
				cs(3) = +DBLE(qiDqj(9,1))*csdel2
			    else
				cs(3) = +DBLE(qiDqj(3,1))*csdel2
			    endif
			else
			    if ( .not.ldel2s ) then
				cs(1) = xqi(ip)*qiDqj(7,4)*qiDqj(5,4)/
     +								sdel2
				cs(2) = DBLE(qiDqj(7,4))*csdeli(3,3)
			    else
				cs(1) = xqi(ip)*qiDqj(7,1)*qiDqj(5,9)/
     +								sdel2
				cs(2) = DBLE(qiDqj(7,1))*csdeli(3,3)
			    endif
			    cs(3) = +DBLE(qiDqj(9,3))*csdel2
			endif
			if ( ldel2s ) cs(3) = -cs(3)
		    endif
		    if ( ifirst .le. 0 ) then
			ifirst = 2
			if ( .not.ldel2s ) then
			    cs(4) = -DBLE(qiDqj(5,9)*qiDqj(7,1)/sdel2)*
     +				csdeli(3,3)
			else
			    cs(4) = DBLE(qiDqj(5,4)*qiDqj(7,1)/sdel2)*
     +				csdeli(3,3)
			endif
			cs(5) = DBLE(delpsi(3,3)/sdel2)*csdel2
		    endif
		else
*
*		    both roots are complex
*
		    if ( ifirst .eq. 0 ) then
			dsdel2 = -ddel2s(3)/(sdel2i(3,3)+sdel2i(3,4))
			csdel2 = DCMPLX(x0,dsdel2)
		    endif
		    if ( ifirst .le. 1 ) then
			if ( j .eq. 1 ) then
			    cs(1) = xqi(ip)*qiDqj(7,1)*qiDqj(5,1)/sdel2
			    cs(2) = +DCMPLX(x0,qiDqj(7,1)*sdel2i(3,3))
			    if ( .not.ldel2s ) then
				cs(3) = +DCMPLX(x0,qiDqj(9,1)*dsdel2)
			    else
				cs(3) = +DCMPLX(x0,qiDqj(3,1)*dsdel2)
			    endif
			else
			    if ( .not.ldel2s ) then
				cs(1) = xqi(ip)*qiDqj(7,4)*qiDqj(5,4)/
     +								sdel2
				cs(2) =DCMPLX(x0,qiDqj(7,4)*sdel2i(3,3))
			    else
				cs(1) = xqi(ip)*qiDqj(7,1)*qiDqj(5,9)/
     +								sdel2
				cs(2) =DCMPLX(x0,qiDqj(7,1)*sdel2i(3,3))
			    endif
			    cs(3) = +DCMPLX(x0,qiDqj(9,3)*dsdel2)
			endif
			if ( ldel2s ) cs(3) = -cs(3)
		    endif
		    if ( ifirst .le. 0 ) then
			ifirst = 2
			if ( .not.ldel2s ) then
			    cs(4) = -DCMPLX(x0,qiDqj(5,9)*qiDqj(7,1)*
     +						sdel2i(3,3)/sdel2)
			else
			    cs(4) = DCMPLX(x0,qiDqj(5,4)*qiDqj(7,1)*
     +						sdel2i(3,3)/sdel2)
			endif
			cs(5) = DCMPLX(x0,delpsi(3,3)*dsdel2/sdel2)
		    endif
		endif
		if ( k .eq. 1 ) then
		    csom = cs(1) + cs(2) + cs(3) + cs(4) + cs(5)
		else
		    csom = cs(1) - cs(2) - cs(3) - cs(4) - cs(5)
		endif
		smax =max(absc(cs(1)),absc(cs(2)),absc(cs(3)),
     +			absc(cs(4)),absc(cs(5)))/xqi(ip)**2
		if ( lwrite ) then
		    print 1002,'cdyzzy(',ii,'3)+ =',csom/DBLE(xqi(ip))**
     +			2,smax
***		    print *,(cs(i)/DBLE(xqi(ip))**2,i=1,5)
		endif
		if ( smax .lt. xmax ) then
		    cdyzzy(ii,3) = csom/DBLE(xqi(ip))**2
		    xmax = smax
		endif
		if ( lwarn .and. absc(cdyzzy(ii,3)).lt.xloss*xmax ) then
		  call ffwarn(140,ieri(2*k+j+2),absc(cdyzzy(ii,3)),xmax)
		endif
  230	    continue
*
*	    get cdyyzz
*
	    if ( ldel2s ) then
		cdyyzz(j,3) = cdyz(2,j,4,3) - cdyz(2,j,3,3)
		xmax = absc(cdyz(2,j,4,3))
		if ( absc(cdyyzz(j,3)) .ge. xloss*xmax ) goto 235
		if ( lwrite ) print 1002,'cdyyzz(',j,'3) =',cdyyzz(j,3),
     +			xmax
		if ( ifirst .le. 0 ) then
		    if ( mod(isoort( 5),10).ne.0 .or.
     +			 mod(isoort(13),10).ne.0 ) then
			csdel2=DBLE(ddel2s(3))/(csdeli(3,3)+csdeli(3,4))
		    else
			dsdel2 = -ddel2s(3)/(sdel2i(3,3)+sdel2i(3,4))
			csdel2 = DCMPLX(x0,dsdel2)
		    endif
		endif
		cs(2) = -csdel2/DBLE(xqi(3))
		cs(1) = qiDqj(5,3)*qiDqj(7,1)/(sdel2*xqi(3))
		if ( j .eq. 1 ) then
		    csom = cs(1) + cs(2)
		else
		    csom = cs(1) - cs(2)
		endif
		smax = absc(cs(1))
		if ( lwrite ) print 1002,'cdyyzz(',j,'3)+=',csom,smax
		if ( smax .lt. xmax ) then
		    cdyyzz(j,3) = csom
		    xmax = smax
		endif
		if ( lwarn .and. absc(cdyyzz(j,3)).lt.xloss*xmax ) then
		    call ffwarn(147,ieri(9+j),absc(cdyyzz(j,3)),xmax)
		endif
	    endif
*
*	    bookkeeping
*
  235	    continue
	    ifirst = ifirst - 1
  240	    continue
*  #]	    complex case:
	endif
*  #]	vertices (3):
	ier0 = 0
	do 250 i = 1,12
	   ier0 = max(ier0,ieri(i))
  250	continue
	ier = ier + ier0
*  #] get differences:
*  #[ check differences:
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    do 300 i=1,3
		if ( isoort(2*i-1) .ne. isoort(2*i+7) ) goto 300
		do 290 j=1,2
		    if ( isoort(2*i-1) .ge. 0 ) then
			xhck = dyzzy(j,i) - y(2,4,i)*z(j,3,i)
     +				      + z(j,4,i)*y(2,3,i)
			if ( rloss*abs(xhck) .gt. precx*max(
     +				abs(y(2,4,i)*z(j,3,i)),
     +				abs(z(j,4,i)*y(2,3,i))) ) print *,
     +			'ffdxc0: error: ','dyzzy(',j,i,') <> terms, ',
     +				dyzzy(j,i),y(2,4,i)*z(j,3,i),
     +				z(j,4,i)*y(2,3,i),xhck
			xhck = dyzzy(j+2,i) - y(4,4,i)*z(j+2,3,i)
     +				       + z(j+2,4,i)*y(4,3,i)
			if ( rloss*abs(xhck) .gt. precx*max(
     +				abs(y(4,4,i)*z(j+2,3,i)),
     +				abs(z(j+2,4,i)*y(4,3,i))) ) print *,
     +			'ffdxc0: error: ','dyzzy(',j+2,i,') <> terms, ',
     +				dyzzy(j+2,i),y(4,4,i)*z(j+2,3,i),
     +				z(j+2,4,i)*y(4,3,i),xhck
		    else
			chck = cdyzzy(j,i) - cy(2,4,i)*cz(j,3,i)
     +				      + cz(j,4,i)*cy(2,3,i)
			if ( rloss*absc(chck) .gt. precc*max(
     +				abs(cy(2,4,i)*cz(j,3,i)),
     +				abs(cz(j,4,i)*cy(2,3,i))) ) print *,
     +			'ffdxc0: error: ','cdyzzy(',j,i,') <> terms, ',
     +				cdyzzy(j,i),cy(2,4,i)*cz(j,3,i),
     +				cz(j,4,i)*cy(2,3,i),chck
			chck = cdyzzy(j+2,i) - cy(4,4,i)*cz(j+2,3,i)
     +				       + cz(j+2,4,i)*cy(4,3,i)
			if ( rloss*absc(chck) .gt. precc*max(
     +				abs(cy(4,4,i)*cz(j+2,3,i)),
     +				abs(cz(j+2,4,i)*cy(4,3,i))) ) print *,
     +			'ffdxc0: error: ','cdyzzy(',j+2,i,') <> terms,',
     +				cdyzzy(j+2,i),cy(4,4,i)*cz(j+2,3,i),
     +				cz(j+2,4,i)*cy(4,3,i),chck
		    endif
  290		continue
  300	    continue
	endif
*  #] check differences:
*  #[ write output:
	if ( lwrite ) then
	    print *,'ffdxc0: found roots:'
	    do 320 k=3,4
	    do 310 i=1,3
		print *,'  k = ',i
		if ( isoort(2*i+8*(k-3)) .gt. 0 ) then
		    print *,'  ym,ym1 = ',y(1,k,i),y(3,k,i),
     +			' (not used)'
		    print *,'  yp,yp1 = ',y(2,k,i),y(4,k,i)
		    print *,'  zm,zm1 = ',z(1,k,i),z(3,k,i)
		    print *,'  zp,zp1 = ',z(2,k,i),z(4,k,i)
		elseif ( isoort(2*i+8*(k-3)) .eq. 0 ) then
		    if ( isoort(2*i-1+8*(k-3)) .eq. 0 ) then
			print *,'  no roots, all is zero'
		    else
			print *,'  yp,yp1 = ',y(2,k,i),y(4,k,i)
			print *,'  zp,zp1 = ',z(2,k,i),z(4,k,i)
		    endif
		else
		    print *,'  cym,cym1 = ',cy(1,k,i),cy(3,k,i),
     +			'(not used)'
		    print *,'  cyp,cyp1 = ',cy(2,k,i),cy(4,k,i)
		    print *,'  czm,czm1 = ',cz(1,k,i),cz(3,k,i)
		    print *,'  czp,czp1 = ',cz(2,k,i),cz(4,k,i)
		endif
  310	    continue
  320	    continue
	endif
	if ( lwrite ) print '(a)','  ##] get roots:'
*  #] write output:
*  #[ logarithms for 4point function:
*
*	Not yet made stable ...
*
	if ( npoin .eq. 4 ) then
	if ( lwrite ) print '(a)','  ##[ logarithms for Ai<0:'
	do 420 i = 1,3
	do 410 k = 3,4
	    ii = i+3*(k-3)
	    if ( ilogi(ii) .ne. -999 ) then
		idone(ii) = 0
		goto 410
	    endif
	    l = 2*i+8*(k-3)-1
	    if ((isoort(l).gt.0 .or. mod(isoort(l),10).le.-5) .and.
     +		(isoort(l+1).ge.0 .or. mod(isoort(l+1),10).le.-5)) then
*  #[		real case:
*
*		the real case (isoort=-5,-6: really real but complex for ffdcs)
*
		s(1) = -dyz(2,1,k,i)/dyz(2,2,k,i)
		if ( lwrite ) then
*		    fantasize imag part, but suppress error message
		    clogi(ii) = zxfflg(s(1),1,x1,ier0)
		    print *,'clogi  = ',clogi(ii)
		endif
		if ( abs(s(1)-1) .lt. xloss ) then
		    clogi(ii) = dfflo1(d2yzz(k,i)/dyz(2,2,k,i),ier)
		    ilogi(ii) = 0
		else
		    if ( abs(s(1)+1) .lt. xloss ) then
			clogi(ii) = dfflo1(-2*sdel2i(i,k)/(xpi(i+3,k)*
     +				dyz(2,2,k,i)),ier)
		    else
			clogi(ii) = zxfflg(abs(s(1)),0,x0,ier)
		    endif
		    if ( dyz(2,2,k,i).gt.0 .and. dyz(2,1,k,i).gt.0 )
     +								then
			ilogi(ii) = -1
		    elseif ( dyz(2,1,k,i).lt.0 .and. dyz(2,2,k,i).lt.0)
     +								then
			ilogi(ii) = +1
		    else
			ilogi(ii) = 0
		    endif
*		    in case del2s=0 and i=3 we pick up a minus sign, I think
		    if ( ldel2s .and. i .eq. 3 ) ilogi(ii) = -ilogi(ii)
		endif
		if ( lwrite ) print *,'clogi+ = ',clogi(ii)+
     +			DCMPLX(x0,pi)*ilogi(ii)
		idone(ii) = 1
*  #]		real case:
	    elseif ( isoort(l) .lt. 0 ) then
*  #[		complex case:
*		for stability split the unit circle up in 4*pi/2
*		(this may have to be improved to 8*pi/4...)
*
		ier0 = 0
		if ( lwrite ) then
		    if ( abs(DBLE(cdyz(2,1,k,i))) .lt. xalog2 .or.
     +			abs(DIMAG(cdyz(2,2,k,i))) .lt. xalog2 ) then
			csom = -DCMPLX(DBLE(cdyz(2,1,k,i))/xalog2,DIMAG(
     +				cdyz(2,1,k,i))/xalog2) /DCMPLX(DBLE(cdyz
     +				(2,2,k,i))/xalog2,DIMAG(cdyz(2,2,k,i))/
     +				xalog2)
		    else
			csom = -cdyz(2,1,k,i)/cdyz(2,2,k,i)
		    endif
		    clogi(ii)=zfflog(csom,0,c0,ier0)
		    print *,'isoort = ',isoort(2*i-1)
		    print *,'cdyz(2,1) = ',cdyz(2,1,k,i)
		    print *,'cdyz(2,2) = ',cdyz(2,2,k,i)
		    print *,'clogi = ',clogi(ii)
		endif
		if ( DBLE(cdyz(2,1,k,i)) .gt. abs(DIMAG(cdyz(2,1,k,i))))
     +								then
		    som =2*atan2(DIMAG(cdyz(2,1,k,i)),DBLE(
     +							cdyz(2,1,k,i)))
		    clogi(ii) = DCMPLX(x0,som)
		    if ( DIMAG(cdyz(2,1,k,i)) .gt. 0 ) then
			ilogi(ii) = -1
		    else
			ilogi(ii) = +1
		    endif

		elseif ( DBLE(cdyz(2,1,k,i)) .lt.
     +			-abs(DIMAG(cdyz(2,1,k,i))) ) then
		    if ( DIMAG(cdyz(2,1,k,i)) .eq. 0 ) then
			call fferr(82,ier)
			print *,'isoort = ',isoort(l),isoort(l+1)
		    endif
		    som = 2*atan2(-DIMAG(cdyz(2,1,k,i)),-DBLE(
     +							cdyz(2,1,k,i)))
		    clogi(ii) = DCMPLX(x0,som)
		    if ( DIMAG(cdyz(2,1,k,i)) .gt. 0 ) then
			ilogi(ii) = +1
		    else
			ilogi(ii) = -1
		    endif
		else
		    s(1) = -DBLE(cdyz(2,1,k,i))
		    s(2) = DIMAG(cdyz(2,1,k,i))
		    som = 2*atan2(s(1),s(2))
		    clogi(ii) = DCMPLX(x0,som)
		    ilogi(ii) = 0
		endif
		if ( lwrite ) print *,'clogi+= ',clogi(ii)+
     +			DCMPLX(x0,pi)*ilogi(ii)
		idone(ii) = 1
*  #]		complex case:
	    endif
*	    Note that we generate an error if isoort(l)=0
	    if ( lwrite ) then
		print *,'ffdxc0: ',ii,': ',clogi(ii),' + ',ilogi(ii),
     +			'*i*pi'
	endif
  410	continue
	if ( idone(ii) .ne. 0 .and. idone(ii-3) .ne. 0 .and.
     +	     absc(clogi(ii)-clogi(ii-3)).lt.xloss*absc(clogi(ii)) .and. 
     +	     ilogi(ii).eq.ilogi(ii-3) ) then
*  #[	    subtract more smartly:
	    if ( isoort(l).gt.0 .and. isoort(l+1).ge.0 ) then
		if ( lwrite ) print *,'ffdxc0: extra logs not ready ',
     +			'in the real case'
		goto 420
	    else
		cs(1) = cdyzzy(1,i)
		cs(2) = cdyzzy(2,i)
		if ( i .eq. 1 ) then
		    cs(3) = 0
		else
		    if ( lwrite ) print *,'ffdxc0: extra logs not ',
     +			'ready for i <>1'
		    goto 420
		endif
		csom = cs(1) - cs(2) + cs(3)
		xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)))
*		change this to "no warning and quit" later
		if ( lwarn .and. absc(csom) .lt. xmax ) then
		    goto 420
***		    call ffwarn(148,ier,absc(csom),xmax)
		endif
		if ( lwrite ) print *,'som was : ',clogi(ii-3)-clogi(ii)
		c = csom/(cdyz(2,2,3,i)*cdyz(2,1,4,i))
		c = zfflo1(c,ier)
		if ( lwrite ) print *,'som is  : ',c
*		
*		the log is never much bigger than 1, so demand at least 
*		accuracy to 0.1; this will catch all i*pi errors
*		
		if ( abs(clogi(ii-3)-clogi(ii)-c).gt.0.1 ) then
		    print *,'ffdxc0: error in smart logs: ',clogi(ii-3)-
     +		    	clogi(ii),c,' not used'
		    goto 420
		endif
		clogi(ii-3) = c
		clogi(ii) = 0
	    endif
*  #]	    subtract more smartly:
	endif
  420	continue
*	An algorithm to obtain the sum of two small logarithms more
*	accurately has been put in ffcc0p, not yet here
	if ( lwrite ) print '(a)','  ##] logarithms for Ai<0:'
	endif
*  #] logarithms for 4point function:
*  #[ real case integrals:
	if ( .not. lcompl ) then
*	    normal case
	    do 510 i=1,3
		if ( lwrite ) print '(a,i1,a)','  ##[ dxs nr ',i,':'
		j = 2*i-1
		if ( isoort(j) .eq. 0 ) then
		    if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j),isoort(j+1)
		    endif
		    if ( isoort(j+8) .eq. 0 ) then
			if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j+8),isoort(j+9)
			endif
		    else
			call ffcxs3(cs3(20*i+61),ipi12(j+8),y(1,4,i),
     +			   z(1,4,i),dyz(1,1,4,i),d2yzz(4,i),dy2z(1,4,i),
     +			   xpi(1,4),piDpj(1,1,4),i,6,isoort(j+8),ier)
		    endif
		elseif ( isoort(j+8) .eq. 0 ) then
		    if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j),isoort(j+1)
		    endif
		    call ffcxs3(cs3(20*i-19),ipi12(j),y(1,3,i),
     +			   z(1,3,i),dyz(1,1,3,i),d2yzz(3,i),dy2z(1,3,i),
     +			   xpi(1,3),piDpj(1,1,3),i,6,isoort(j),ier)
		else
		    call ffdcxs(cs3(20*i-19),ipi12(j),y(1,3,i),z(1,3,i),
     +			dyz(1,1,3,i),d2yzz(3,i),dy2z(1,3,i),dyzzy(1,i),
     +			xpi,piDpj,i,6,isoort(j),ier)
		endif
		if ( lwrite ) print '(a,i1,a)','  ##] dxs nr ',i,':'
  510	    continue
	    isoort(7) = 0
	    isoort(8) = 0
*  #] real case integrals:
*  #[ complex case integrals:
	else
*	    convert xpi
	    do 540 k=3,4
*not	    cetami(1,k) = etami(1,k)
*used	    cetami(3,k) = etami(3,k)
	    do 530 i=1,6
		cpi(i,k) = xpi(i,k)
		do 520 j=1,6
		    cpiDpj(j,i,k) = piDpj(j,i,k)
  520		continue
  530	    continue
  540	    continue
	    do 550 i=1,3
		if ( lwrite ) print '(a,i1,a)','  ##[ dcs nr ',i,':'
		j = 2*i-1
		if ( isoort(j) .eq. 0 ) then
		    if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j),isoort(j+1)
		    endif
		    if ( isoort(j+8) .eq. 0 ) then
			if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j+8),isoort(j+9)
			endif
		    else
			call ffcxs3(cs3(20*i+61),ipi12(j+8),y(1,4,i),
     +			   z(1,4,i),dyz(1,1,4,i),d2yzz(4,i),dy2z(1,4,i),
     +			   xpi(1,4),piDpj(1,1,4),i,6,isoort(j+8),ier)
		    endif
		elseif ( isoort(j+8) .eq. 0 ) then
		    if ( lwrite ) then
			print *,'ffdxc0: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j),isoort(j+1)
		    endif
		    call ffcxs3(cs3(20*i-19),ipi12(j),y(1,3,i),
     +			   z(1,3,i),dyz(1,1,3,i),d2yzz(3,i),dy2z(1,3,i),
     +			   xpi(1,3),piDpj(1,1,3),i,6,isoort(j),ier)
		elseif ( isoort(j) .gt. 0 ) then
		    if ( isoort(j+8) .gt. 0 ) then
			call ffdcxs(cs3(20*i-19),ipi12(j),y(1,3,i),
     +			   z(1,3,i),dyz(1,1,3,i),d2yzz(3,i),dy2z(1,3,i),
     +			   dyzzy(1,i),xpi,piDpj,i,6,isoort(j),ier)
		    else
			print *,'ffdxc0: error: should not occur!'
			call ffcxs3(cs3(20*i-19),ipi12(j),y(1,3,i),
     +			   z(1,3,i),dyz(1,1,3,i),d2yzz(3,i),dy2z(1,3,i),
     +			   xpi(1,3),piDpj(1,1,3),i,6,isoort(j),ier)
			call ffcs3(cs3(20*i+61),ipi12(j+8),cy(1,4,i),
     +			    cz(1,4,i),cdyz(1,1,4,i),cd2yzz(4,i),
     +			    cpi(1,4),cpiDpj(1,1,4),i,6,isoort(j+8),ier)
		    endif
		else
		    if ( isoort(j+8) .lt. 0 ) then
			call ffdcs(cs3(20*i-19),ipi12(j),cy(1,3,i),
     +			    cz(1,3,i),cdyz(1,1,3,i),cd2yzz(3,i),
     +			    cdyzzy(1,i),cdyyzz(1,i),cpi,cpiDpj,
     +			    i,6,isoort(j),ier)
		    else
			print *,'ffdxc0: error: should not occur!'
			call ffcs3(cs3(20*i-19),ipi12(j),cy(1,3,i),
     +			    cz(1,3,i),cdyz(1,1,3,i),cd2yzz(3,i),
     +			    cpi(1,3),cpiDpj(1,1,3),i,6,isoort(j),ier)
			call ffcxs3(cs3(20*i+61),ipi12(j+8),y(1,4,i),
     +			   z(1,4,i),dyz(1,1,4,i),d2yzz(4,i),dy2z(1,4,i),
     +			   xpi(1,4),piDpj(1,1,4),i,6,isoort(j+8),ier)
		    endif
		endif
		if ( lwrite ) print '(a,i1,a)','  ##] dcs nr ',i,':'
  550	    continue
	    isoort(7) = 0
	    isoort(8) = 0
	endif
	return
*  #] complex case integrals:
*###] ffdxc0:
	end
