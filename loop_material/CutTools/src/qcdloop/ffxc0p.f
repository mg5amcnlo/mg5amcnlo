*	$Id: ffxc0p.f,v 1.3 1995/10/06 09:17:26 gj Exp $
*	$Log: ffxc0p.f,v $
c Revision 1.3  1995/10/06  09:17:26  gj
c Found stupid typo in ffxc0p which caused the result to be off by pi^2/3 in
c some equal-mass cases.  Added checks to ffcxs4.f ffcrr.f.
c
*###[ ffxc0p:
	subroutine ffxc0p(cs3,ipi12,isoort,clogi,ilogi,xpi,dpipj,piDpj,
     +		sdel2,del2s,etalam,etami,delpsi,alph,npoin,ier)
***#[*comment:***********************************************************
*									*
*	DOUBLE PRECISIONLY calculates the threepoint function closely following	*
*	recipe in 't Hooft & Veltman, NP B(183) 1979.			*
*	Bjorken and Drell metric is used nowadays!			*
*									*
*	    p2	^ |							*
*		| |							*
*		 / \							*
*	      m2/   \m3							*
*	p1     /     \	p3						*
*	<-    /  m1   \ ->						*
*	------------------------					*
*									*
*	Input:	xpi(1-3)     (real)	pi squared			*
*		xpi(4-6)     (real)	internal mass squared		*
*		dpipj(6,6)   (real)	xpi(i)-xpi(j)			*
*		piDpj(6,6)   (real)	pi(i).pi(j)			*
*		sdel2	     (real)	sqrt(delta_{p_1 p_2}^{p_1 p_2})	*
*		del2s(3)     (real)	delta_{p_i s_i}^{p_i s_i}	*
*		etalam	     (real)	delta_{s_1 s_2 s_3}^{s_1 s_2 s_3}
*					  /delta_{p_1 p_2}^{p_1 p_2}	*
*		etami(6)     (real)	m_i^2 - etalam			*
*		alph(3)	     (real)	alph(1)=alpha, alph(3)=1-alpha	*
*									*
*	Output: cs3(80)	     (complex)	C0, not yet summed.		*
*		ipi12(8)     (integer)	factors pi^2/12, not yet summed	*
*		slam	     (complex)	lambda(p1,p2,p3).		*
*		isoort(8)    (integer)	indication of he method used	*
*		clogi(3)     (complex)	log(-dyz(2,1,i)/dyz(2,2,i))	*
*		ilogi(3)     (integer)	factors i*pi in this		*
*		ier	     (integer)	number of digits inaccurate in	*
*					answer				*
*									*
*	Calls:	ffdel3,ffdel3m,ffroot,ffxxyz,ffcxyz,ffdwz,ffcdwz,	*
*		ffcxs3,ffcs3,ffcxs4,ffcs4				*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ipi12(8),isoort(8),ilogi(3),npoin,ier
	DOUBLE COMPLEX cs3(80),clogi(3)
	DOUBLE PRECISION xpi(6),dpipj(6,6),piDpj(6,6),sdel2,del2s(3),
     +		etalam,etami(6),delpsi(3),alph(3)
*
*	local variables:
*
	integer i,j,k,m,ip,jsoort(8),ierw,iw,ier0,ier1,irota,
     +		ilogip(3)
	logical l4,lcompl,lcpi,l4pos
	DOUBLE COMPLEX c,cs,calph(3),csdl2i(3),csdel2
	DOUBLE COMPLEX cy(4,3),cz(4,3),cw(4,3),cdyz(2,2,3),cdwy(2,2,3),
     +		cdwz(2,2,3),cd2yzz(3),cd2yww(3)
	DOUBLE COMPLEX cpi(6),cdpipj(6,6),cpiDpj(6,6),cetami(6),
     +		clogip(3)
	DOUBLE PRECISION y(4,3),z(4,3),w(4,3),dyz(2,2,3),dwy(2,2,3),
     +		dwz(2,2,3),d2yzz(3),d2yww(3),dy2z(4,3)
	DOUBLE PRECISION sdel2i(3),s1,s2
	DOUBLE PRECISION absc,s,xqi(6),dqiqj(6,6),qiDqj(6,6)
	DOUBLE PRECISION dfflo1
	DOUBLE COMPLEX zxfflg,zfflog
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
	    call ffxhck(xpi,dpipj,6,ier)
	    do i=1,8
	    	if ( ipi12(i).ne.0 ) then
	    	    print *,'ffxc0p: error: ipi12(',i,') != 0: ',
     +	    	    	ipi12(i)
	    	endif
	    enddo
	endif
*  #] check input:
*  #[ IR case:
*
*	but only the off-shell regulator case - the log(lam) has been
*	caught before
*
	if ( lsmug ) then
	    do 5 i=1,3
		if ( xpi(i) .eq. 0 ) then
		    j = mod(i,3)+1
		    k = mod(j,3)+1
		    if ( piDpj(i,j).eq.0 .and. piDpj(i,k).eq.0 ) then
			call ffrot3(irota,xqi,dqiqj,qiDqj,
     +				xpi,dpipj,piDpj,6,3,4,ier)
			if ( lwrite ) print *,'ffxc0p: rotated over ',
     +				irota
			if ( npoin.eq.4 ) call ffrt3p(clogip,ilogip,
     +				irota,clogi,ilogi,+1)
			call ffxc0j(cs3(1),ipi12(1),sdel2,clogip,ilogip,
     +				xqi,dqiqj,qiDqj,x0,4,ier)
			if ( npoin.eq.4 ) call ffrt3p(clogi,ilogi,irota,
     +				clogip,ilogip,-1)
			return
		    endif
		endif
    5	    continue
	endif
*  #] IR case:
*  #[ get roots etc:
*  #[   get z-roots:
*	if ( npoin .eq. 3 ) then
	    l4pos = l4also
*	else
*	    l4pos = .FALSE.
*	endif
	lcompl = .FALSE.
	if ( lwrite ) print '(a)','  ##[ get roots:'
	ier1 = ier
	do 10 i=1,3
*
*	get roots (y,z,w) and flag what to do: 0=nothing, 1=normal,
*	-1=complex
*
	    ip = i+3
*	    first get the roots
	    ier0 = ier
	    if ( del2s(i) .le. 0 ) then
*		real case
		sdel2i(i) = sqrt(-del2s(i))
		csdl2i(i) = sdel2i(i)
*		then handle the special case Si = 0
		if ( xpi(ip) .eq. 0 ) then
		    if ( i .eq. 1 .and. alph(3) .eq. 0 .or.
     +			 i .eq. 3 .and. alph(1) .eq. 0 ) then
			isoort(2*i-1) = 0
			isoort(2*i) = 0
			l4pos = .FALSE.
			goto 10
		    endif
		endif
		call ffxxyz(y(1,i),z(1,i),dyz(1,1,i),d2yzz(i),dy2z(1,i),
     +			i,sdel2,sdel2i(i),etalam,etami,delpsi(i),xpi,
     +			dpipj,piDpj,isoort(2*i-1),.FALSE.,6,ier0)
	    else
*		complex case
		sdel2i(i) = sqrt(del2s(i))
		csdl2i(i) = DCMPLX(x0,sdel2i(i))
		lcompl = .TRUE.
		call ffcxyz(cy(1,i),cz(1,i),cdyz(1,1,i),cd2yzz(i),i,
     +			sdel2,sdel2i(i),etalam,etami,delpsi(i),xpi,
     +			piDpj,isoort(2*i-1),.FALSE.,6,ier0)
	    endif
	    ier1 = max(ier1,ier0)
   10	continue
	ier = ier1
*  #]   get z-roots:
*  #[   get w-roots:
*
*	get w's:
*
	ierw = ier
	l4 = .FALSE.
	lcpi = .FALSE.
	if ( isoort(4) .eq. 0 ) then
*	    no error message; just bail out
	    ierw = ierw + 100
	    goto 90
	endif
	do 70 iw = 1,3,2
	if ( .not. l4pos .or. alph(4-iw) .eq. 0 ) then
	    jsoort(2*iw-1) = 0
	    jsoort(2*iw) = 0
	    l4pos = .FALSE.
	else
	if ( isoort(4) .gt. 0 .and. isoort(2*iw) .ge. 0 ) then
	    jsoort(2*iw-1) = 1
	    jsoort(2*iw) = 1
	    d2yww(iw) = -d2yzz(2)/alph(4-iw)
	    do 20 j=1,2
		w(j+iw-1,iw) = z(j+3-iw,2)/alph(4-iw)
		w(j+3-iw,iw) = 1 - w(j+iw-1,iw)
		if ( abs(w(j+3-iw,iw)) .lt. xloss ) then
		    if ( lwrite ) print *,'   w(',j+3-iw,iw,')  = ',
     +			w(j+3-iw,iw),x1
		    s = z(j+iw-1,2) - alph(iw)
		    if ( abs(s) .lt. xloss*alph(iw) ) then
			ierw = ierw + 15
			goto 70
		    endif
		    w(j+3-iw,iw) = s/alph(4-iw)
		    if ( lwrite ) print *,'   w(',j+3-iw,iw,')+ = ',
     +			w(j+3-iw,iw),abs(alph(iw)/alph(4-iw))
		endif
		dwy(j,2,iw) = dyz(2,j,2)/alph(4-iw)
		do 15 i=1,2
		    dwz(j,i,iw) = w(j,iw) - z(i,iw)
		    if ( abs(dwz(j,i,iw)) .ge. xloss*abs(w(j,iw)) )
     +								goto 14
		    if ( lwrite ) print *,'  dwz(',j,i,iw,')  = ',
     +			dwz(j,i,iw),abs(w(j,iw))
		    dwz(j,i,iw) = z(i+2,iw) - w(j+2,iw)
		    if ( lwrite ) print *,'  dwz(',j,i,iw,')+ = ',
     +			dwz(j,i,iw),abs(w(j+2,iw))
		    if ( abs(dwz(j,i,iw)) .ge. xloss*abs(w(j+2,iw)) )
     +								goto 14
		    dwz(j,i,iw) = dwy(j,2,iw) + dyz(2,i,iw)
		    if ( lwrite ) print *,'  dwz(',j,i,iw,')++= ',
     +			dwz(j,i,iw),abs(dwy(j,2,iw))
		    if ( abs(dwz(j,i,iw)) .ge. xloss*abs(dwy(j,2,iw)) )
     +								goto 14
		    l4 = .TRUE.
		    call ffdwz(dwz(1,1,iw),w(1,iw),z(1,iw),j,i,iw,
     +			alph(1),alph(3),xpi,dpipj,piDpj,sdel2i,6,ierw)
   14		    continue
   15		continue
   20	    continue
	else
*	    convert to complex ...
	    jsoort(2*iw-1) = -10
	    jsoort(2*iw) = -10
	    if ( isoort(4).ge.0 .and. (iw.eq.1 .or. isoort(2).ge.0) )
     +								then
		cd2yzz(2) = d2yzz(2)
		do 21 i=1,4
		    cy(i,2) = y(i,2)
		    cz(i,2) = z(i,2)
   21		continue
		do 23 i=1,2
		    do 22 j=1,2
			cdyz(j,i,2) = dyz(j,i,2)
   22		    continue
   23		continue
	    endif
	    if ( isoort(2*iw) .ge. 0 ) then
		cd2yzz(iw) = d2yzz(iw)
		do 24 i=1,4
		    cy(i,iw) = y(i,iw)
		    cz(i,iw) = z(i,iw)
   24		continue
		do 26 i=1,2
		    do 25 j=1,2
			cdyz(j,i,iw) = dyz(j,i,iw)
   25		    continue
   26		continue
	    endif
	    cd2yww(iw) = -cd2yzz(2)/DBLE(alph(4-iw))
	    do 30 j=1,2
		cw(j+iw-1,iw) = cz(j+3-iw,2)/DBLE(alph(4-iw))
		cw(j+3-iw,iw) = 1 - cw(j+iw-1,iw)
		if ( absc(cw(j+3-iw,iw)) .lt. xloss ) then
		    if (lwrite) print *,'   cw(',j+3-iw,iw,')  = ',
     +			cw(j+3-iw,iw),x1
		    cs = cz(j+iw-1,2) - DBLE(alph(iw))
		    if ( absc(cs) .lt. xloss*alph(iw) ) ierw = ierw + 15
		    cw(j+3-iw,iw) = cs/DBLE(alph(4-iw))
		    if (lwrite) print *,'   cw(',j+3-iw,iw,')+ = ',
     +			cw(j+3-iw,iw),abs(alph(iw)/alph(4-iw))
		endif
		cdwy(j,2,iw) = cdyz(2,j,2)/DBLE(alph(4-iw))
		do 29 i=1,2
		    cdwz(j,i,iw) = cw(j,iw) - cz(i,iw)
		    if ( absc(cdwz(j,i,iw)) .ge. xloss*absc(cw(j,iw)) )
     +								goto 31
		    if ( lwrite ) print *,'  cdwz(',j,i,iw,')  = ',
     +			cdwz(j,i,iw),absc(cw(j,iw))
		    cdwz(j,i,iw) = cz(i+2,iw) - cw(j+2,iw)
		    if ( lwrite ) print *,'  cdwz(',j,i,iw,')+ = ',
     +			cdwz(j,i,iw),absc(cw(j+2,iw))
		    if ( absc(cdwz(j,i,iw)) .ge. xloss*absc(cw(j+2,iw)))
     +								goto 31
		    cdwz(j,i,iw) = cdwy(j,2,iw) + cdyz(2,i,iw)
		    if ( lwrite ) print *,'  cdwz(',j,i,iw,')++= ',
     +			cdwz(j,i,iw),absc(cdwy(j,2,iw))
		    if ( absc(cdwz(j,i,iw)).ge.xloss*absc(cdwy(j,2,iw)))
     +								goto 31
		    l4 = .TRUE.
		    if ( .not. lcpi ) then
			lcpi = .TRUE.
			calph(1) = alph(1)
			calph(3) = alph(3)
			csdel2 = sdel2
			cetami(1) = etami(1)
			cetami(3) = etami(3)
			do 28 k=1,6
			    cpi(k) = xpi(k)
			    do 27 m=1,6
				cdpipj(m,k) = dpipj(m,k)
				cpiDpj(m,k) = piDpj(m,k)
   27			    continue
   28			continue
		    endif
		    call ffcdwz(cdwz(1,1,iw),cw(1,iw),cz(1,iw),j,i,iw,
     +			calph(1),calph(3),cpi,cdpipj,cpiDpj,csdl2i,
     +			csdel2,6,ierw)
   31		    continue
   29		continue
   30	    continue
	endif
	endif
   70	continue
   90	continue
	ierw = ierw-ier
*  #]   get w-roots:
*  #[   write output:
	if ( lwrite ) then
	    print *,'ffxc0p: found roots:'
	    do 85 i=1,3
		print *,'  k = ',i
		if ( isoort(2*i) .gt. 0 ) then
		    print *,'  ym,ym1 = ',y(1,i),y(3,i),' (not used)'
		    print *,'  yp,yp1 = ',y(2,i),y(4,i)
		    print *,'  zm,zm1 = ',z(1,i),z(3,i)
		    print *,'  zp,zp1 = ',z(2,i),z(4,i)
		    if ( l4 .and. i.ne.2 .and. jsoort(2*i-1).ne.0 ) then
			if ( isoort(4) .gt. 0 ) then
			    print *,'  wm,wm1 = ',w(1,i),w(3,i)
			    print *,'  wp,wp1 = ',w(2,i),w(4,i)
			else
			    print *,'  cwm,cwm1 = ',cw(1,i),cw(3,i)
			    print *,'  cwp,cwp1 = ',cw(2,i),cw(4,i)
			endif
		    endif
		elseif ( isoort(2*i) .eq. 0 ) then
		    if ( isoort(2*i-1) .eq. 0 ) then
			print *,'  no roots, all is zero'
		    else
			print *,'  yp,yp1 = ',y(2,i),y(4,i)
			print *,'  zp,zp1 = ',z(2,i),z(4,i)
			if ( l4 .and. i.ne.2 .and. jsoort(2*i-1).ne.0 )
     +				then
			    if ( isoort(4) .gt. 0 ) then
				print *,'  wm,wm1 = ',w(1,i),w(3,i)
				print *,'  wp,wp1 = ',w(2,i),w(4,i)
			    else
				print *,'  cwm,cwm1 = ',cw(1,i),cw(3,i)
				print *,'  cwp,cwp1 = ',cw(2,i),cw(4,i)
			    endif
			endif
		    endif
		else
		    print *,'  cym,cym1 = ',cy(1,i),cy(3,i),'(not used)'
		    print *,'  cyp,cyp1 = ',cy(2,i),cy(4,i)
		    print *,'  czm,czm1 = ',cz(1,i),cz(3,i)
		    print *,'  czp,czp1 = ',cz(2,i),cz(4,i)
		    if ( i .ne. 2 .and. isoort(2*i-1) .ne. 0 ) then
			print *,'  cwm,cwm1 = ',cw(1,i),cw(3,i)
			print *,'  cwp,cwp1 = ',cw(2,i),cw(4,i)
		    endif
		endif
   85	    continue
	endif
	if ( lwrite ) print '(a)','  ##] get roots:'
*  #]   write output:
*  #[   which case:
	if ( l4 ) then
*	    21-aug-1995.  added check for isoort(2*i-1).eq.0 to avoid
*	    undefined variables etc in ffdcs, ffdcrr.  They should be
*	    able to handle this, but are not (yet?)
	    if ( ierw .ge. 1 .or. isoort(1).eq.0 .or. isoort(3).eq.0
     +			.or. isoort(5).eq.0 ) then
		l4pos = .FALSE.
	    else
		ier = ier + ierw
	    endif
	endif
*  #]   which case:
*  #] get roots etc:
*  #[ logarithms for 4point function:
	if ( npoin .eq. 4 ) then
	if ( lwrite ) print '(a)','  ##[ logarithms for Ai<0:'
	do 95 i = 1,3
	    if ( ilogi(i) .ne. -999 ) goto 95
	    if ( isoort(2*i) .gt. 0 .and.
     +				isoort(2*i-1) .ge. 0 ) then
		s1 = -dyz(2,1,i)/dyz(2,2,i)
		if ( lwrite ) then
*		    fantasize imag part, but suppress error message
		    ier0 = 0
		    clogi(i) = zxfflg(s1,1,x1,ier0)
		    print *,'clogi  = ',clogi(i)
		endif
		if ( abs(s1-1) .lt. xloss ) then
		    clogi(i) = dfflo1(d2yzz(i)/dyz(2,2,i),ier)
		    ilogi(i) = 0
		else
		    if ( abs(s1+1) .lt. xloss ) then
			clogi(i) = dfflo1(-2*sdel2i(i)/(xpi(i+3)*
     +				dyz(2,2,i)),ier)
		    else
			clogi(i) = zxfflg(abs(s1),0,x0,ier)
		    endif
		    if ( dyz(2,2,i).gt.0 .and. dyz(2,1,i).gt.0 ) then
			ilogi(i) = -1
		    elseif ( dyz(2,1,i).lt.0 .and. dyz(2,2,i).lt.0) then
			ilogi(i) = +1
		    else
			ilogi(i) = 0
		    endif
		endif
		if ( lwrite ) print *,'clogi+ = ',clogi(i)+
     +			DCMPLX(x0,pi)*ilogi(i)
	    elseif ( isoort(2*i-1) .lt. 0 ) then
*		for stability split the unit circle up in 4*pi/2
*		(this may have to be improved to 8*pi/4...)
		ier0 = 0
		if ( lwrite ) then
		    if ( abs(DBLE(cdyz(2,1,i))) .lt. xalog2 .or.
     +			abs(DIMAG(cdyz(2,2,i))) .lt. xalog2 ) then
			cs = -DCMPLX(DBLE(cdyz(2,1,i))/xalog2,DIMAG(cdyz
     +				(2,1,i))/xalog2) / DCMPLX(DBLE(cdyz(2,2,
     +				i))/xalog2,DIMAG(cdyz(2,2,i))/xalog2)
		    else
			cs = -cdyz(2,1,i)/cdyz(2,2,i)
		    endif
		    clogi(i)=zfflog(cs,0,c0,ier0)
		    print *,'isoort = ',isoort(2*i-1)
		    print *,'cdyz(2,1) = ',cdyz(2,1,i)
		    print *,'cdyz(2,2) = ',cdyz(2,2,i)
		    print *,'clogi = ',clogi(i)
		endif
		if ( DBLE(cdyz(2,1,i)) .gt. DIMAG(cdyz(2,1,i)) ) then
		    s = 2*atan2(DIMAG(cdyz(2,1,i)),DBLE(cdyz(2,1,i)))
		    clogi(i) = DCMPLX(x0,s)
		    ilogi(i) = -1
		elseif ( DBLE(cdyz(2,1,i)) .lt. -DIMAG(cdyz(2,1,i)))
     +								then
		    if ( DIMAG(cdyz(2,1,i)) .eq. 0 ) then
			call fferr(84,ier)
		    endif
		    s = 2*atan2(-DIMAG(cdyz(2,1,i)),-DBLE(cdyz(2,1,i)))
		    clogi(i) = DCMPLX(x0,s)
		    ilogi(i) = 1
		else
		    s1 = -DBLE(cdyz(2,1,i))
		    s2 = DIMAG(cdyz(2,1,i))
		    s = 2*atan2(s1,s2)
		    clogi(i) = DCMPLX(x0,s)
		    ilogi(i) = 0
		endif
		if ( lwrite ) print *,'clogi+= ',clogi(i)+
     +			DCMPLX(x0,pi)*ilogi(i)
	    endif
	    if ( lwrite ) then
		print *,'ffxc0p:',i,': ',clogi(i),' + ',ilogi(i),'*i*pi'
	    endif
   95	continue
*	An algorithm to obtain the sum of two small logarithms more
*	accurately has been put in ffcc0p, not yet here
	if ( lwrite ) print '(a)','  ##] logarithms for Ai<0:'
	endif
*  #] logarithms for 4point function:
*  #[ real case integrals:
	ier1 = ier
	if ( .not. lcompl ) then
	    if ( .not. l4 .or. .not. l4pos ) then
*		normal case
		do 100 i=1,3
		    if ( lwrite ) print '(a,i1,a)','  ##[ xs3 nr ',i,':'
		    j = 2*i-1
		    if ( isoort(j) .eq. 0 ) then
			if ( lwrite ) then
			    print *,'ffxc0p: xk=0, ma=mb/Si-0 -> S3 = 0'
			    print *,'isoort:',isoort(j),isoort(j+1)
			endif
		    else
			ier0 = ier
			call ffcxs3(cs3(20*i-19),ipi12(j),y(1,i),z(1,i),
     +				dyz(1,1,i),d2yzz(i),dy2z(1,i),xpi,piDpj,
     +				i,6,isoort(j),ier0)
			ier1 = max(ier1,ier0)
		    endif
		    if ( lwrite ) print '(a,i1,a)','  ##] xs3 nr ',i,':'
  100		continue
		isoort(7) = 0
		isoort(8) = 0
	    else
		do 110 i=1,3,2
		    j = 2*i-1
		    isoort(j+2) = jsoort(j)
		    isoort(j+3) = jsoort(j+1)
		    if ( lwrite ) print '(a,i1,a)','  ##[ xs4 nr ',i,':'
		    ier0 = ier
		    call ffcxs4(cs3(20*i-19),ipi12(j),w(1,i),y(1,i),
     +			z(1,i),dwy(1,1,i),dwz(1,1,i),dyz(1,1,i),
     +			d2yww(i),d2yzz(i),xpi,piDpj,i,6,isoort(j),ier0)
		    ier1 = max(ier1,ier0)
		    if ( lwrite ) print '(a,i1,a)','  ##] xs4 nr ',i,':'
  110		continue
	    endif
*  #] real case integrals:
*  #[ complex case integrals:
	else
*	    convert xpi
	    if ( .not.lcpi ) then
		cetami(1) = etami(1)
		cetami(3) = etami(3)
		do 190 i=1,6
		    cpi(i) = xpi(i)
  190		continue
	    endif
	    if ( .not. l4 .or. .not. l4pos ) then
*		normal case
		do 200 i=1,3
		    if ( lwrite ) print '(a,i1,a)','  ##[ xs3 nr ',i,':'
		    j = 2*i-1
		    ier0 = ier
		    if ( isoort(j) .eq. 0 ) then
			if ( lwrite ) then
			    print *,'ffxc0p: xk=0, ma=mb/Si-0 -> S3 = 0'
			    print *,'isoort:',isoort(j),isoort(j+1)
			endif
		    elseif ( isoort(j) .gt. 0 ) then
			call ffcxs3(cs3(20*i-19),ipi12(2*i-1),y(1,i),
     +				z(1,i),dyz(1,1,i),d2yzz(i),dy2z(1,i),
     +				xpi,piDpj,i,6,isoort(j),ier0)
		    else
			call ffcs3(cs3(20*i-19),ipi12(2*i-1),cy(1,i),
     +				cz(1,i),cdyz(1,1,i),cd2yzz(i),cpi,
     +				cpiDpj,i,6,isoort(j),ier0)
		    endif
		    ier1 = max(ier1,ier0)
		    if ( lwrite ) print '(a,i1,a)','  ##] xs3 nr ',i,':'
  200		continue
		isoort(7) = 0
		isoort(8) = 0
	    else
		isoort(3) = jsoort(1)
		isoort(4) = jsoort(2)
		if ( lwrite ) print '(a)','  ##[ xs4 nr 1:'
		ier0 = ier
		if ( isoort(1) .gt. 0 .and. isoort(3) .gt. 0 ) then
		    call ffcxs4(cs3(1),ipi12(1),w(1,1),y(1,1),
     +			z(1,1),dwy(1,1,1),dwz(1,1,1),dyz(1,1,1),
     +			d2yww(1),d2yzz(1),xpi,piDpj,1,6,isoort(1),ier0)
		else
		    call ffcs4(cs3(1),ipi12(1),cw(1,1),cy(1,1),
     +			cz(1,1),cdwy(1,1,1),cdwz(1,1,1),cdyz(1,1,1),
     +			cd2yww(1),cd2yzz(1),cpi,cpiDpj,
     +			DCMPLX(xpi(5)*alph(3)**2),cetami,1,6,isoort(1),
     +			ier0)
		endif
		ier1 = max(ier1,ier0)
		if ( lwrite ) print '(a)','  ##] xs4 nr 1:'
		if ( lwrite ) print '(a)','  ##[ xs4 nr 2:'
		isoort(7) = jsoort(5)
		isoort(8) = jsoort(6)
		ier0 = ier
		if ( isoort(5) .gt. 0 .and. isoort(7) .gt. 0 ) then
		    call ffcxs4(cs3(41),ipi12(5),w(1,3),y(1,3),
     +			z(1,3),dwy(1,1,3),dwz(1,1,3),dyz(1,1,3),
     +			d2yww(3),d2yzz(3),xpi,piDpj,3,6,isoort(5),ier0)
		else
		    call ffcs4(cs3(41),ipi12(5),cw(1,3),cy(1,3),
     +			cz(1,3),cdwy(1,1,3),cdwz(1,1,3),cdyz(1,1,3),
     +			cd2yww(3),cd2yzz(3),cpi,cpiDpj,
     +			DCMPLX(xpi(5)*alph(1)**2),cetami,3,6,isoort(5),
     +			ier0)
		endif
		ier1 = max(ier1,ier0)
		if ( lwrite ) print '(a)','  ##] xs4 nr 2:'
	    endif
	endif
	ier = ier1
*  #] complex case integrals:
*###] ffxc0p:
	end
*###[ ffrt3p:
	subroutine ffrt3p(clogip,ilogip,irota,clogi,ilogi,idir)
***#[*comment:***********************************************************
*									*
*	rotates the arrays clogi,ilogi also over irota (idir=+1) or	*
*	back (-1)							*
*									*
*	Input:	irota	  (integer)	index in rotation array		*
*		clogi(3)  (complex)	only if idir=-1			*
*		ilogi(3)  (integer)	indicates which clogi are needed*
*					(idir=+1), i*pi terms (idir=-1)	*
*		idir	  (integer)	direction: forward (+1) or	*
*					backward (-1)			*
*	Output:	clogip(3)  (integer)	clogi rotated			*
*		ilogip(3)  (integer)	ilogi rotated			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer irota,idir,ilogi(3),ilogip(3)
	DOUBLE COMPLEX clogi(3),clogip(3)
*
*	local variables
*
	integer i,inew(6,6)
	save inew
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data inew /1,2,3,4,5,6,
     +		   2,3,1,5,6,4,
     +		   3,1,2,6,4,5,
     +		   1,3,2,6,5,4,
     +		   3,2,1,5,4,6,
     +		   2,1,3,4,6,5/
*  #] declarations:
*  #[ rotate:
*
*	the clogi, ilogi are numbered according to the p_i
*
	if ( idir .eq. +1 ) then
	    do 10 i=1,3
		ilogip(inew(i+3,irota)-3) = ilogi(i)
		clogip(inew(i+3,irota)-3) = clogi(i)
   10	    continue
	else
	    do 20 i=1,3
		ilogip(i) = ilogi(inew(i+3,irota)-3)
		clogip(i) = clogi(inew(i+3,irota)-3)
   20	    continue
	endif
*
*  #] rotate:
*###] ffrt3p:
	end

