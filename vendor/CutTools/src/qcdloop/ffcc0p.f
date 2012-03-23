*###[ ffcc0p:
	subroutine ffcc0p(cs3,ipi12,isoort,clogi,ilogi,cpi,cpipj,
     +		cpiDpj,sdel2,cel2si,etalam,etami,delpsi,alpha,npoin,ier)
***#[*comment:***********************************************************
*									*
*	Calculates the threepoint function closely following		*
*	recipe in 't Hooft & Veltman, NP B(183) 1979.			*
*	Bjorken and Drell metric is used nowadays!			*
*									*
*	    p2	^ |							*
*		| |							*
*		 / \							*
*	      m2/   \m3 						*
*	p1     /     \	p3						*
*	<-    /  m1   \ ->						*
*	------------------------					*
*									*
*	Input:	cpi(1-3)   (complex)	pi squared (,2=untransformed	*
*					when npoin=4)			*
*		cpi(4-6)   (complex)	internal mass squared		*
*		cpipj(6,6)   (complex)	cpi(i)-cpi(j)			*
*		cpiDpj(6,6)   (complex)	pi(i).pi(j)			*
*									*
*	Output: cs3	 (complex)(48)	C0, not yet summed.		*
*		ipi12	 (integer)(3)	factors pi^2/12, not yet summed	*
*		cslam	 (complex)	lambda(p1,p2,p3).		*
*		isoort	 (integer)(3)	indication of he method used	*
*		ier	 (integer)	0=ok, 1=inaccurate, 2=error	*
*									*
*	Calls:	ffcel2,ffcoot,ffccyz,ffcdwz,ffcs3,ffcs4			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(8),isoort(8),ilogi(3),npoin,ier
	DOUBLE COMPLEX cs3(80),clogi(3),cpi(6),cpipj(6,6),
     +		cpiDpj(6,6),sdel2,cel2si(3),etalam,etami(6),
     +		delpsi(3),alpha(3)
*
*	local variables:
*
	integer i,j,k,ip,ierw,jsoort(8),iw,ismall(3),ier0
	logical l4,l4pos
	DOUBLE COMPLEX c,cs,zfflog,cs1,cs2,cs4,ci
	DOUBLE COMPLEX cy(4,3),cz(4,3),cw(4,3),cdyz(2,2,3),
     +		cdwy(2,2,3),cdwz(2,2,3),cd2yzz(3),cd2yww(3)
	DOUBLE COMPLEX csdl2i(3)
*	DOUBLE COMPLEX cyp,cym,ca,cb,cc,cd
	DOUBLE COMPLEX zfflo1
	DOUBLE PRECISION absc
*FOR ABSOFT ONLY
*	DOUBLE COMPLEX csqrt
*	external csqrt
*
*	common blocks:
*
	include 'ff.h'
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    ier0 = 0
	    call ffchck(cpi,cpipj,6,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffcc0p: error: ',
     +		'transformed momenta wrong'
	endif
*  #] check input:
*  #[ get roots etc:
*  #[   get z-roots:
	if ( npoin .ne. 3 ) then
	    l4pos = .FALSE.
	else
	    l4pos = l4also
	endif
	if ( lwrite ) print '(a)','  ##[ get roots: (ffcc0p)'
	do 10 i=1,3
*
*	    get roots (y,z)
*
	    ip = i+3
*	    first get the roots
	    j = i+1
	    if ( j .eq. 4 ) j = 1
	    csdl2i(i) = sqrt(-cel2si(i))
	    if ( cpi(ip) .eq. 0 ) then
		if ( i .eq. 1 .and. alpha(3) .eq. 0 .or.
     +		     i .eq. 3 .and. alpha(1) .eq. 0 ) then
		    isoort(2*i-1) = 0
		    isoort(2*i) = 0
		    l4pos = .FALSE.
		    goto 10
		endif
	    endif
	    call ffccyz(cy(1,i),cz(1,i),cdyz(1,1,i),cd2yzz(i),i,
     +		sdel2,csdl2i(i),etalam,etami,delpsi(i),
     +		cpi,cpiDpj,isoort(2*i-1),6,ier)
   10	continue
*	if ( lwrite ) then
*	    print *,'cy(1)  = ',cy(2,1)
*	    print *,'vgl    = ',cy(4,2)/alpha(3)
*	    print *,'cy(3)1 = ',cy(4,3)
*	    print *,'vgl    = ',cy(2,2)/alpha(1)
*	endif
*  #]   get z-roots:
*  #[   get w-roots:
*
*	get w's:
*
	ierw = 0
	l4 = .FALSE.
	if ( isoort(4) .eq. 0 ) then
	    call fferr(10,ierw)
	    goto 90
	endif
	do 70 iw = 1,3,2
	if ( .not. l4pos .or. alpha(4-iw) .eq. 0 ) then
	    jsoort(2*iw-1) = 0
	    jsoort(2*iw) = 0
	    l4pos = .FALSE.
	else
	jsoort(2*iw-1) = -1
	jsoort(2*iw) = -1
	cd2yww(iw) = -cd2yzz(2)/alpha(4-iw)
	do 20 j=1,2
	    cw(j+iw-1,iw) = cz(j+3-iw,2)/alpha(4-iw)
	    cw(j+3-iw,iw) = 1 - cw(j+iw-1,iw)
	    if ( absc(cw(j+3-iw,iw)) .lt. xloss ) then
		if (lwrite) print *,'   cw(',j+3-iw,iw,')  = ',
     +			cw(j+3-iw,iw),x1
		cs = cz(j+iw-1,2) - alpha(iw)
		if ( absc(cs) .lt. xloss*absc(alpha(iw)) ) then
		    ierw = 1
		    goto 70
		endif
		cw(j+3-iw,iw) = cs/alpha(4-iw)
		if (lwrite) print *,'   cw(',j+3-iw,iw,')+ = ',
     +			cw(j+3-iw,iw),absc(alpha(iw))/absc(alpha(4-iw))
	    endif
	    cdwy(j,2,iw) = cdyz(2,j,2)/alpha(4-iw)
	    do 15 i=1,2
		cdwz(j,i,iw) = cw(j,iw) - cz(i,iw)
		if ( absc(cdwz(j,i,iw)) .ge. xloss*absc(cw(j,iw)) )
     +								goto 14
		if ( lwrite ) print *,'  cdwz(',j,i,iw,')  = ',
     +			cdwz(j,i,iw),absc(cw(j,iw))
		cdwz(j,i,iw) = cz(i+2,iw) - cw(j+2,iw)
		if ( lwrite ) print *,'  cdwz(',j,i,iw,')+ = ',
     +			cdwz(j,i,iw),absc(cw(j+2,iw))
		if ( absc(cdwz(j,i,iw)) .ge. xloss*absc(cw(j+2,iw)) )
     +								goto 14
		cdwz(j,i,iw) = cdwy(j,2,iw) + cdyz(2,i,iw)
		if ( lwrite ) print *,'  cdwz(',j,i,iw,')++= ',
     +			cdwz(j,i,iw),absc(cdwy(j,2,iw))
		if ( absc(cdwz(j,i,iw)) .ge. xloss*absc(cdwy(j,2,iw)) )
     +								goto 14
		l4 = .TRUE.
		call ffcdwz(cdwz(1,1,iw),cw(1,iw),cz(1,iw),j,i,iw,
     +			alpha(1),alpha(3),cpi,cpipj,cpiDpj,csdl2i,
     +			sdel2,6,ierw)
   14		continue
   15	    continue
   20	continue
	endif
   70	continue
*  #]   get w-roots:
*  #[   write output:
	if ( lwrite ) then
	    print *,'ffcc0p: found roots:'
	    do 85 i=1,3
		print *,'  k = ',i
		if ( isoort(2*i) .ne. 0 ) then
		    print *,'  cym,cym1 = ',cy(1,i),cy(3,i),
     +			'(not used)'
		    print *,'  cyp,cyp1 = ',cy(2,i),cy(4,i)
		    print *,'  czm,czm1 = ',cz(1,i),cz(3,i)
		    print *,'  czp,czp1 = ',cz(2,i),cz(4,i)
		    if ( i .ne. 2 .and. l4pos ) then
			print *,'  cwm,cwm1 = ',cw(1,i),cw(3,i)
			print *,'  cwp,cwp1 = ',cw(2,i),cw(4,i)
		    endif
		else
		    if ( isoort(2*i-1) .eq. 0 ) then
			print *,'  no roots, all is zero'
		    else
			print *,'  cyp,cyp1 = ',cy(2,i),
     +				cy(4,i)
			print *,'  czp,czp1 = ',cz(2,i),
     +				cz(4,i)
			if ( i .ne. 2 .and. jsoort(2*i-1) .ne. 0 ) then
			    print *,'  cwm,cwm1 = ',cw(1,i),cw(3,i)
			    print *,'  cwp,cwp1 = ',cw(2,i),cw(4,i)
			endif
		    endif
		endif
   85	    continue
   86	    continue
	    print '(a)','  ##] get roots:'
	endif
*  #]   write output:
*  #[   which case:
   90	if ( l4 ) then
	    if ( DIMAG(alpha(1)) .ne. 0 ) then
		if ( lwrite ) print *,'ffcc0p: cannot handle unphysical'
     +			,' momenta in 16 dilogs (yet)'
		l4pos = .FALSE.
	    elseif ( ierw .ge. 1 ) then
		l4pos = .FALSE.
	    else
		ier = max(ier,ierw)
	    endif
	endif
*  #]   which case:
*  #] get roots etc:
*  #[ logarithms for 4point function:
	if ( npoin .eq. 4 ) then
	if ( lwrite ) print '(a)','  ##[ logarithms for Ai<0:'
	do 95 i = 1,3
	    ismall(i) = 0
	    if ( ilogi(i) .ne. -999 ) goto 95
	    if ( isoort(2*i) .ne. 0 ) then
*		maybe add sophisticated factors i*pi later
		c = -cdyz(2,1,i)/cdyz(2,2,i)
		if ( lwrite ) then
*		    fantasize imag part, but suppress error message
		    print *,'c = ',c
		    clogi(i) = zfflog(c,1,c1,ier0)
		    print *,'clogi  = ',clogi(i)
		endif
		if ( absc(c-1) .lt. xloss ) then
		    cs = cd2yzz(i)/cdyz(2,2,i)
		    clogi(i) = zfflo1(cs,ier)
		    ilogi(i) = 0
		    ismall(i) = 1
		    if ( lwrite ) then
			print *,'c = ',c
			print *,'c+= ',1-cs
		    endif
		elseif ( DBLE(c) .gt. 0 ) then
		    clogi(i) = zfflog(c,0,c0,ier)
		    ilogi(i) = 0
		else
		    if ( absc(c+1) .lt. xloss ) then
			cs = -2*csdl2i(i)/cdyz(2,2,i)/
     +				DBLE(cpi(i+3))
			clogi(i) = zfflo1(cs,ier)
			ismall(i) = -1
			if ( lwrite ) then
			    print *,'c = ',c
			    print *,'c+= ',-1+cs
			endif
		    else
			cs = 0
			clogi(i) = zfflog(-c,0,c0,ier)
		    endif
		    if ( DIMAG(c).lt.0 .or. DIMAG(cs).lt.0 ) then
			ilogi(i) = -1
		    elseif ( DIMAG(c).gt.0 .or. DIMAG(cs).gt.0 ) then
			ilogi(i) = +1
		    elseif ( DBLE(cdyz(2,2,i)) .eq. 0 ) then
			ilogi(i)=-nint(sign(DBLE(x1),DBLE(cpi(i+3))))
			ier = ier + 50
			print *,'doubtful imaginary part ',ilogi(i)
		    endif
		    if ( abs(DIMAG(c)).lt.precc*absc(c) .and.
     +			abs(DIMAG(cs)).lt.precc*absc(cs) ) then
			print *,'ffcc0p: error: imaginary part doubtful'
			ier = ier + 50
		    endif
		endif
		if ( lwrite ) then
		    print *,'clogi+ = ',i,clogi(i),ilogi(i)
		    if ( ilogi(i).ne.0 ) then
			print *,'       = ',i,clogi(i)+DCMPLX(x0,pi)*
     +				ilogi(i)
		    endif
		endif
	    endif
   95	continue
	do 96 i=1,3
	    j = i + 1
	    if ( j .eq. 4 ) j = 1
	    if ( abs(ismall(i)+ismall(j)) .eq. 2 .and. absc(clogi(i)+
     +			clogi(j)) .lt. xloss*absc(clogi(i)) ) then
		print *,'eerst: ',clogi(i)+clogi(j)
*		assume that we got here because of complex sqrt(-delta)
		ci = DCMPLX(DBLE(0),DBLE(1))
		cs1=-2*ci*DIMAG(cy(2,i))*csdl2i(j)/DBLE(cpi(j+3))/
     +			(cdyz(2,2,i)*cdyz(2,2,j))
		cs2=-2*ci*DIMAG(cy(2,j))*csdl2i(i)/DBLE(cpi(i+3))/
     +			(cdyz(2,2,i)*cdyz(2,2,j))
		cs = cs1 + cs2
		if ( absc(cs) .lt. xloss*absc(cs1) ) then
		    if ( lwrite ) print *,'Eerste poging:',cs,cs1,cs2
		    k = j+1
		    if ( k .eq. 4 ) k = 1
		    cs1 = cpipj(j+3,i+3)*cpi(j)
		    cs2 = cpiDpj(k+3,j)*cpiDpj(j+3,j)
		    cs4 = -cpiDpj(k+3,j)*cpiDpj(i+3,j)
		    cs = cs1 + cs2 + cs4
		    if ( lwrite ) then
			print *,'csdl2i(i)-csdl2i(j) = ',
     +			    csdl2i(i)-csdl2i(j),absc(csdl2i(i))
			print *,'csdl2i(i)-csdl2i(j)+= ',cs/
     +				(csdl2i(i)+csdl2i(j))
		    endif
		    if ( absc(cs) .lt. xloss*max(absc(cs1),absc(cs2),
     +				absc(cs4)) ) then
			print *,'ffcc0p: cancellations in delj-deli'
			goto 96
		    endif
		    cs1=ci*DIMAG(cy(2,j))*cs/(csdl2i(i)+csdl2i(j))
		    call ffcl2t(cs2,cpiDpj,k+3,j,4,5,6,+1,-1,6,ier)
		    cs2 = -cs2*csdl2i(j)/sdel2/DBLE(cpi(j+3))
		    cs = cs1 + cs2
		    if ( lwrite ) print *,'Tweede poging:',cs,cs1,cs2
		    if ( absc(cs) .lt. xloss*absc(cs1) ) then
			print *,'ffcc0p: cancellations in extra terms'
			goto 96
		    endif
		    cs = -2*cs/DBLE(cpi(i+3))/(cdyz(2,2,i)*
     +			cdyz(2,2,j))
		endif
		clogi(i) = zfflo1(cs,ier)
		clogi(j) = 0
		print *,'nu:    ',clogi(i)+clogi(j)
	    endif
   96	continue
	if ( lwrite ) print '(a)','  ##] logarithms for Ai<0:'
	endif
*  #] logarithms for 4point function:
*  #[ integrals:
	if ( .not. l4 .or. .not. l4pos ) then
*	    normal case
	    do 200 i=1,3
		if ( lwrite ) print '(a,i1,a)','  ##[ s3 nr ',i,':'
		j = 2*i-1
		if ( isoort(2*i-1) .eq. 0 ) then
		    if ( lwrite ) then
			print *,'ffcc0p: xk=0, ma=mb/Si-0 -> S3 = 0'
			print *,'isoort:',isoort(j),isoort(j+1)
		    endif
		else
		    call ffcs3(cs3(20*i-19),ipi12(2*i-1),cy(1,i),
     +			cz(1,i),cdyz(1,1,i),cd2yzz(i),cpi,cpiDpj,
     +			i,6,isoort(j),ier)
		endif
		if ( lwrite ) print '(a,i1,a)','  ##] s3 nr ',i,':'
  200	    continue
	    isoort(7) = 0
	    isoort(8) = 0
	else
	    if ( lwrite ) print '(a)','  ##[ s4 nr 1:'
	    isoort(3) = jsoort(1)
	    isoort(4) = jsoort(2)
	    call ffcs4(cs3(1),ipi12(1),cw(1,1),cy(1,1),
     +			cz(1,1),cdwy(1,1,1),cdwz(1,1,1),cdyz(1,1,1),
     +			cd2yww(1),cd2yzz(1),cpi,cpiDpj,
     +			cpi(5)*alpha(3)**2,etami,1,6,isoort(1),ier)
	    if ( lwrite ) print '(a)','  ##] s4 nr 1:'
	    if ( lwrite ) print '(a)','  ##[ s4 nr 2:'
	    isoort(7) = jsoort(5)
	    isoort(8) = jsoort(6)
	    call ffcs4(cs3(41),ipi12(1),cw(1,3),cy(1,3),
     +			cz(1,3),cdwy(1,1,3),cdwz(1,1,3),cdyz(1,1,3),
     +			cd2yww(3),cd2yzz(3),cpi,cpiDpj,
     +			cpi(5)*alpha(1)**2,etami,3,6,isoort(5),ier)
	    if ( lwrite ) print '(a)','  ##] s4 nr 2:'
	endif
*  #] integrals:
*###] ffcc0p:
	end
*###[ ffccyz:
	subroutine ffccyz(cy,cz,cdyz,cd2yzz,ivert,csdelp,csdels,etalam,
     +				etami,delps,xpi,piDpj,isoort,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*	cz(1,2) = (-p(ip1).p(is2) +/- csdelp)/xpi(ip1)			*
*	cy(1,2) = (-p(ip1).p(is2) +/- sdisc)/xpi(ip1)			*
*			cdisc = csdels + etaslam*xpi(ip1)		*
*									*
*	cy(3,4) = 1-cy(1,2)						*
*	cz(3,4) = 1-cz(1,2)						*
*	cdyz(i,j) = cy(i) - cz(j)					*
*									*
*	Input:	ivert		(integer)	defines the vertex	*
*		csdelp		(complex)	sqrt(lam(p1,p2,p3))/2	*
*		csdels		(complex)	sqrt(lam(p,ma,mb))/2	*
*		etalam		(complex)	det(si.sj)/det(pi.pj)	*
*		etami(6)	(complex)	si.si - etalam		*
*		xpi(ns)		(complex)	standard		*
*		piDpj(ns,ns)	(complex)	standard		*
*		ns		(integer)	dim of xpi,piDpj	*
*									*
*	Output:	cy(4),cz(4),cdyz(4,4)	(complex)	see above	*
*		ier		(integer)	usual error flag	*
*									*
*	Calls:	fferr,ffroot						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ivert,ns,ier,isoort(2)
	DOUBLE COMPLEX cy(4),cz(4),cdyz(2,2),cd2yzz,csdelp,csdels
	DOUBLE COMPLEX etalam,etami(6),delps,xpi(6),piDpj(6,6)
*
*	local variables:
*
	integer i,j,ip1,ip2,ip3,is1,is2,is3,ier0
	DOUBLE COMPLEX cverg,cdisc,c,check,dpipj(6,6)
	DOUBLE PRECISION absc,rloss
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*  #] declarations:
*  #[ set up pointers:
	if ( lwrite ) print *,'ffccyz: ivert = ',ivert
	if ( ltest .and. ns .ne. 6 ) then
	    print *,'ffccyz: error: ns != 6 !!',ns
	    stop
	endif
	is1 = ivert
	is2 = ivert+1
	if ( is2 .eq. 4 ) is2 = 1
	is3 = ivert-1
	if ( is3 .eq. 0 ) is3 = 3
	ip1 = is1 + 3
	ip2 = is2 + 3
	ip3 = is3 + 3
*  #] set up pointers:
*  #[ check input:
	if ( ltest ) then
	    ier0 = ier
	    dpipj(1,1) = 1
	    call ffcl2p(cverg,xpi,dpipj,piDpj,ip1,ip2,ip3,is1,is2,is3,6,
     +		ier0)
	    rloss = xloss*DBLE(10)**(-mod(ier0,50))
	    if ( rloss*absc(cverg-delps).gt.precc*absc(cverg) ) print *,
     +		'ffccyz: error: delps <> cverg',delps,cverg,delps-cverg
	    ier0 = ier
	    call ffcel2(cverg,piDpj,6,ip1,ip2,ip3,1,ier0)
	    rloss = xloss*DBLE(10)**(-mod(ier0,50))
	    if ( rloss*absc(cverg+csdelp**2) .gt. precc*absc(cverg) )
     +		print *,'ffccyz: error: csdelp**2 incorrect ',
     +		csdelp**2,-cverg,csdelp**2+cverg
	    ier0 = ier
	    call ffcel3(cverg,xpi,piDpj,6,ier0)
	    check = etami(is2)-xpi(is2)-cverg/csdelp**2
	    rloss = xloss**2*DBLE(10)**(-mod(ier0,50))
	    if ( rloss*absc(check) .gt. precc*max(absc(etami(is2)),
     +		absc(xpi(is2)),absc(cverg/csdelp**2)) ) print *,
     +		'ffccyz: error: etami(',is2,') incorrect ',
     +		etami(is2),xpi(is2)+cverg/csdelp**2,check,ier0
	endif
*  #] check input:
*  #[ xk = 0:
	if ( xpi(ip1) .eq. 0 ) then
	    isoort(2) = 0
	    if ( piDpj(is1,ip1) .eq. 0 ) then
		isoort(1) = 0
		if (lwrite) print *,'  ck = 0, cm1 = cm2, so cs3 = 0'
		return
	    endif
	    if ( DIMAG(etalam).ne.0 ) then
		isoort(1) = -1
	    else
		isoort(1) = -3
	    endif
	    cy(1) = etami(is2) / piDpj(is1,ip1) /2
	    cy(2) = cy(1)
	    cy(3) = - etami(is1) / piDpj(is1,ip1) /2
	    cy(4) = cy(3)
	    cz(1) = xpi(is2) / piDpj(is1,ip1) /2
	    cz(2) = cz(1)
	    cz(3) = - xpi(is1) / piDpj(is1,ip1) /2
	    cz(4) = cz(3)
	    cdyz(1,1) = - etalam / piDpj(is1,ip1) /2
	    cdyz(1,2) = cdyz(1,1)
	    cdyz(2,1) = cdyz(1,1)
	    cdyz(2,2) = cdyz(1,1)
	    if ( ltest ) then
*		check whether we have the correct root ...
		call ffcl2p(cverg,xpi,dpipj,piDpj,ip1,ip2,ip3,
     +					is1,is2,is3,6,ier)
		cdisc = cverg/csdelp
		check = piDpj(ip1,is2) + cdisc
		if ( xloss*absc(check) .gt. precc*max(absc(piDpj(
     +			ip1,is2)),absc(cdisc)) ) then
		    call fferr(36,ier)
		    if ( lwrite ) then
			print *,'piDpj(',ip1,is2,') = ',piDpj(ip1,is2)
			print *,'cdisc = ',cdisc
			print *,'diff  = ',check
		    endif
		endif
	    endif
	    return
	endif
*  #] xk = 0:
*  #[ get cy(1,2),cz(1,2):
	if ( DIMAG(etalam).ne.0 ) then
	    isoort(1) = -1
	    isoort(2) = -1
	else
	    isoort(1) = -3
	    isoort(2) = -3
	endif
	call ffcoot(cz(1),cz(2),xpi(ip1),piDpj(ip1,is2),xpi(is2),
     +							csdels,ier)
	cdisc = delps/csdelp
	call ffcoot(cy(1),cy(2),xpi(ip1),piDpj(ip1,is2),etami(is2),
     +							cdisc,ier)
*  #] get cy(1,2),cz(1,2):
*  #[ get cy(3,4),cz(3,4):
	cz(4) = 1-cz(2)
	cz(3) = 1-cz(1)
	if ( absc(cz(3)) .lt. xloss .or. absc(cz(4)) .lt. xloss ) then
	    call ffcoot(cz(4),cz(3),xpi(ip1),-piDpj(ip1,is1),
     +						xpi(is1),csdels,ier)
	endif
*	the imaginary part may not be accurate in these cases, take
*	some precautions:
	if ( cz(3) .eq. 0 ) cz(1) = 1
	if ( cz(4) .eq. 0 ) cz(2) = 1
	if ( DIMAG(cz(1)).eq.0 )
     +		cz(1) = DCMPLX(DBLE(cz(1)),-DIMAG(cz(3)))
	if ( DIMAG(cz(2)).eq.0 )
     +		cz(2) = DCMPLX(DBLE(cz(2)),-DIMAG(cz(4)))
	if ( DIMAG(cz(1)) .gt. 0 .neqv. DIMAG(cz(3)) .lt. 0 ) then
	    if ( abs(DBLE(cz(1))) .ge. abs(DBLE(cz(3))) ) then
		cz(1) = DCMPLX(DBLE(cz(1)),-DIMAG(cz(3)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part z1 changed to -z3'
	    else
		cz(3) = DCMPLX(DBLE(cz(3)),-DIMAG(cz(1)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part z3 changed to -z1'
	    endif
	endif
	if ( DIMAG(cz(2)) .gt. 0 .neqv. DIMAG(cz(4)) .lt. 0 ) then
	    if ( abs(DBLE(cz(2))) .ge. abs(DBLE(cz(4))) ) then
		cz(2) = DCMPLX(DBLE(cz(2)),-DIMAG(cz(4)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part z2 changed to -z4'
	    else
		cz(4) = DCMPLX(DBLE(cz(4)),-DIMAG(cz(2)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part z4 changed to -z2'
	    endif
	endif
	cy(4) = 1-cy(2)
	cy(3) = 1-cy(1)
	if ( absc(cy(3)) .lt. xloss .or. absc(cy(4)) .lt. xloss ) then
	    call ffcoot(cy(4),cy(3),xpi(ip1),-piDpj(ip1,is1),
     +						etami(is1),cdisc,ier)
	endif
	if ( cy(3) .eq. 0 ) cy(1) = 1
	if ( cy(4) .eq. 0 ) cy(2) = 1
	if ( DIMAG(cy(1)).eq.0 )
     +		cy(1) = DCMPLX(DBLE(cy(1)),-DIMAG(cy(3)))
	if ( DIMAG(cy(2)).eq.0 )
     +		cy(2) = DCMPLX(DBLE(cy(2)),-DIMAG(cy(4)))
	if ( DIMAG(cy(1)) .gt. 0 .neqv. DIMAG(cy(3)) .lt. 0 ) then
	    if ( abs(DBLE(cy(1))) .ge. abs(DBLE(cy(3))) ) then
		cy(1) = DCMPLX(DBLE(cy(1)),-DIMAG(cy(3)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part y1 changed to -y3'
	    else
		cy(3) = DCMPLX(DBLE(cy(3)),-DIMAG(cy(1)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part y3 changed to -y1'
	    endif
	endif
	if ( DIMAG(cy(2)) .gt. 0 .neqv. DIMAG(cy(4)) .lt. 0 ) then
	    if ( abs(DBLE(cy(2))) .ge. abs(DBLE(cy(4))) ) then
		cy(2) = DCMPLX(DBLE(cy(2)),-DIMAG(cy(4)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part y2 changed to -y4'
	    else
		cy(4) = DCMPLX(DBLE(cy(4)),-DIMAG(cy(2)))
		if ( lwrite ) print *,'ffccyz: comment: imaginary ',
     +			'part y4 changed to -y2'
	    endif
	endif
*  #] get cy(3,4),cz(3,4):
*  #[ get cdyz:
*	Note that cdyz(i,j) only exists for i,j=1,2!
	if ( absc(cdisc+csdels) .gt. xloss*absc(cdisc) ) then
	    cdyz(2,1) = ( cdisc + csdels )/xpi(ip1)
	    cdyz(2,2) = etalam/(xpi(ip1)*cdyz(2,1))
	else
	    cdyz(2,2) = ( cdisc - csdels )/xpi(ip1)
	    cdyz(2,1) = etalam/(xpi(ip1)*cdyz(2,2))
	endif
	cdyz(1,1) = -cdyz(2,2)
	cdyz(1,2) = -cdyz(2,1)
	cd2yzz = 2*cdisc/xpi(ip1)
*  #] get cdyz:
*  #[ test output:
	if ( ltest ) then
	    rloss = xloss*DBLE(10)**(-1-mod(ier,50))
	    do 99 i=1,2
		if ( rloss*absc(cy(i)+cy(i+2)-1) .gt. precc*max(absc(
     +		    cy(i)),absc(cy(i+2)),x1)) print *,'ffccyz: error: ',
     +		    'cy(',i+2,')<>1-cy(',i,'):',cy(i+2),cy(i),cy(i+2)+
     +		    cy(i)-1
		if ( rloss*absc(cz(i)+cz(i+2)-1) .gt. precc*max(absc(
     +		    cz(i)),absc(cz(i+2)),x1)) print *,'ffccyz: error: ',
     +		    'cz(',i+2,')<>1-cz(',i,'):',cz(i+2),cz(i),cz(i+2)+
     +		    cz(i)-1
		do 98 j=1,2
		    if ( rloss*absc(cdyz(i,j)-cy(i)+cz(j)) .gt. precc*
     +			max(absc(cdyz(i,j)),absc(cy(i)),absc(cz(j))) )
     +			print *,'ffccyz: error: cdyz(',i,j,') <> cy(',
     +			i,')-','cz(',j,'):',cdyz(i,j),cy(i),cz(j),
     +			cdyz(i,j)-cy(i)+cz(j)
   98		continue
   99	    continue
	    if ( rloss*absc(cd2yzz-2*cy(2)+cz(1)+cz(2)) .gt. precc*max(
     +		absc(cd2yzz),x2*absc(cy(2)),absc(cz(1)),absc(cz(2))) )
     +		print *,'ffccyz: error: cd2yzz <> 2*cy(2)+cz(1)+cz(2):',
     +		cd2yzz,2*cy(2),cz(1),cz(2),cd2yzz-2*cy(2)+cz(1)+cz(2)
	endif
*  #] test output:
*###] ffccyz:
	end
