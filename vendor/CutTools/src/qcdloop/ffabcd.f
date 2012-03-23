*###[ ffabcd:
	subroutine ffabcd(aijkl,xpi,dpipj,piDpj,del2s,sdel2s,
     +		in,jn,jin,isji, kn,ln,lkn,islk, ns, ifirst, ier)
***#[*comment:***********************************************************
*									*
*	Calculate the a,b,c,d of the equation for qij.qkl		*
*									*
*	a      = s4.s4^2						*
*									*
*		  si sj  sk sl	/ sm sn	 sm sn	  sm sn	  mu   ro\	*
*	-b/2   = d	d	|d	d      - d	s4   s4	 |	*
*		  mu nu  nu ro	\ mu s4	 ro s4	  sm sn		 /	*
*									*
*	 _	  si sj	 sk sl	/ mu s4	  ro	  mu s4	  ro\		*
*	vD/2   = d	d	|d	s4   +	 d	s4  |		*
*		  mu nu	 nu ro	\ s3 s4		  s3 s4	    /		*
*									*
*	with	sm = s3, sn = s4					*
*		p(jin) = isji*(sj-si)					*
*		p(lkn) = islk*(sl-sk)					*
*									*
*	Input:	xpi(ns)			as usual			*
*		dpipj(ns,ns)		  -"-				*
*		piDpj(ns,ns)		  -"-				*
*		in,jn,jin,isjn		see above			*
*		kn,ln,lkn,islk		see above			*
*									*
*	Output:	del4d2			see above			*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer in,jn,jin,isji,kn,ln,lkn,islk,ns,ifirst,
     +		ier
	DOUBLE PRECISION aijkl,xpi(10),dpipj(10,10),piDpj(10,10),del2s
	DOUBLE PRECISION sdel2s
*
*	local variables:
*
	integer i,j,ji,k,l,lk,isii
	integer ii,ll
	integer iii(6,2)
	save iii
	logical ldet(4)
	DOUBLE PRECISION xa,xb,xc,xd,s(24),del3(4),som,somb,somd,xbp,
     +		xdp,smaxp,smax,save,xmax,rloss,del2d2,dum,del2i,del2j,
     +		del2ji,del2k,del2l,del2lk,d2d2i,d2d2j,d2d2ji,d2d2k,
     +		d2d2l,d2d2lk,d3d2m,d3d2n,d3d2nm
	save del3,ldet
*
*	common blocks:
*
	include 'ff.h'
*
*	data
*
	data iii / 0,3,4,0,7,0,
     +		   0,3,4,0,7,0/
*	data isign/1,1,1,0,1,0,
*     +		   1,1,1,0,1,0/
*  #] declarations: 
*  #[ initialisaties:
	if ( ifirst .eq. 0 ) then
	    ifirst = ifirst + 1
	    ldet(2) = .FALSE.
	    ldet(3) = .FALSE.
	    ldet(4) = .FALSE.
	endif
	xa = xpi(4)**2
*  #] initialisaties: 
*  #[ check input:
	if ( ltest ) then
	    if ( abs(isji) .ne. 1 ) print *,'ff2d22: error: abs(isji)',
     +		' /= 1',isji
	    if ( abs(islk) .ne. 1 ) print *,'ff2d22: error: abs(islk)',
     +		' /= 1',islk
	    if ( ns .ne. 10 ) print *,'ffabcd: only valid for ns=10!!'
	endif
*  #] check input: 
*  #[ prepare input:
	i = in
	j = jn
	ji = jin
	k = kn
	l = ln
	lk = lkn
*	sort it so that i<j, k<l, i<=k, and if i=k, j<=l
*	(I  think this is superfluous as the indices are sorted when
*	called)
*	if ( i .gt. j ) then
*	    ii = i
*	    i = j
*	    j = ii
*	    isji = -isji
*	endif
*	if ( k .gt. l ) then
*	    ii = k
*	    k = l
*	    l = ii
*	    islk = -islk
*	endif
*	if ( 16*i + j .gt. 16*k + l ) then
*	    ii = i
*	    i = k
*	    k = ii
*	    ii = j
*	    j = l
*	    l = ii
*	    ii = ji
*	    ji = lk
*	    lk = ii
*	    ii = isji
*	    isji = islk
*	    islk = ii
*	endif
*  #] prepare input: 
*  #[ special cases:
	if ( k .eq. 3 ) then
	    xb = 0
	    xc = 0
	    xd = 0
*	    print *,'  b,c,d = 0 (kl=34)'
	    goto 990
	elseif ( j .ge. 3 .and. l .ge. 3 ) then
*	    the whole thing collapses to factor*det3
*	    we have a good memory of things already calculated ...
	    if ( .not.ldet(i+k) ) then
		ldet(i+k) = .TRUE.
		iii(1,1) = i
		iii(4,1) = isgn(3,i)*inx(3,i)
		iii(6,1) = isgn(i,4)*inx(i,4)
		iii(1,2) = k
		iii(4,2) = isgn(3,k)*inx(3,k)
		iii(6,2) = isgn(k,4)*inx(k,4)
		call ffdl3s(del3(i+k),xpi,piDpj,iii,10,ier)
	    endif
	    if ( l .eq. 4 .and. j .eq. 4 ) then
		xb = xpi(4)**2*del3(i+k)/del2s
		xd = 0
		xc = xb**2/xa
	    elseif ( l .eq. 4 .or. j .eq. 4 ) then
		xb = piDpj(3,4)*xpi(4)*del3(i+k)/del2s
		xd = -xpi(4)*del3(i+k)/sdel2s
		xc = xpi(4)*xpi(3)*del3(i+k)**2/del2s**2
	    else
*		l .eq. 3 .and. j .eq. 3
		xd = -2*piDpj(3,4)*del3(i+k)/sdel2s
		s(1) = xpi(3)*xpi(4)
		s(2) = 2*piDpj(3,4)**2
		som = s(2) - s(1)
		if ( abs(som) .ge. xloss*abs(s(1)) ) goto 20
		call ffwarn(88,ier,som,s(1))
   20		continue
		xb = som*del3(i+k)/del2s
		xc = xpi(3)**2*del3(i+k)**2/del2s**2
	    endif
	    goto 900
	endif
	if ( j .eq. 2 .and. l .eq. 4 ) then
	    call ff3dl2(s(1),xpi,dpipj,piDpj, 4, 1,2,5,+1,
     +			k,3,inx(3,k),isgn(3,k), 4, 3,4,7,+1, 10,ier)
	    xb = -xpi(4)*s(1)/del2s
	    iii(1,1) = 1
	    iii(2,1) = 2
	    iii(4,1) = 5
	    iii(5,1) = 10
	    iii(6,1) = 8
	    iii(1,2) = k
	    iii(4,2) = isgn(3,k)*inx(3,k)
	    iii(6,2) = isgn(k,4)*inx(k,4)
	    call ffdl3s(s(1),xpi,piDpj,iii,10,ier)
*	    restore values for other users
	    iii(2,1) = 3
	    iii(5,1) = 7
	    xd = -xpi(4)*s(1)/sdel2s
	    goto 800
	endif
*  #] special cases: 
*  #[ normal case b:
*
*	First term:
*
	call ff2dl2(del2d2,dum,xpi,dpipj,piDpj, 4,
     +			i,j,ji,isji, 4, k,l,lk,islk, 10, ier)
	s(1) = -del2d2*del2s
*
*	Second and third term, split i,j
*
	if ( i .eq. 4 ) then
	    del2i = 0
	else
	    ii = inx(4,i)
	    isii = isgn(4,i)
	    call ffdl2s(del2i,xpi,piDpj,i,4,ii,isii,3,4,7,+1,10,ier)
	endif
	if ( j .eq. 4 ) then
	    del2j = 0
	else
	    ii = inx(4,j)
	    isii = isgn(4,j)
	    call ffdl2s(del2j,xpi,piDpj,j,4,ii,isii,3,4,7,+1,10,ier)
	endif
	call ff2dl2(d2d2i,dum,xpi,dpipj,piDpj, i, k,l,lk,islk, 4,
     +						3,4,7,+1, 10, ier)
	call ff2dl2(d2d2j,dum,xpi,dpipj,piDpj, j, k,l,lk,islk, 4,
     +						3,4,7,+1, 10, ier)
	s(2) = +del2i*d2d2j
	s(3) = -del2j*d2d2i
	somb = s(1) + s(2) + s(3)
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( abs(somb) .ge. xloss*smax ) goto 90
	xmax = smax
	save = somb

*	if the first term is wrong ... forget about it
	if ( abs(somb) .lt. xloss*abs(s(1)) ) then
	    if ( lwrite ) print *,'somb: s = ',s(1),s(2),s(3)
	    goto 80
	endif
	if ( lwrite ) print *,'  somb  = ',somb,s(1),s(2),s(3)

	call ffdl2t(del2ji,piDpj, ji,4, 3,4,7,+1,+1, 10,ier)
	call ff2dl2(d2d2ji,dum,xpi,dpipj,piDpj, ji, k,l,lk,islk, 4,
     +						3,4,7,+1, 10, ier)
	s(2) = +del2j*d2d2ji
	s(3) = -del2ji*d2d2j
	somb = s(1) + isji*(s(2) + s(3))
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( lwrite ) print *,'  somb+1= ',somb,s(1),s(2),s(3),isji
	if ( abs(somb) .ge. xloss*smax ) goto 90
	if ( smax .lt. xmax ) then
	    save = somb
	    xmax = smax
	endif

	s(2) = +del2i*d2d2ji
	s(3) = -del2ji*d2d2i
	somb = s(1) + isji*(s(2) + s(3))
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( lwrite ) print *,'  somb+2= ',somb,s(1),s(2),s(3),isji
	if ( abs(somb) .ge. xloss*max(abs(s(1)),abs(s(2)),abs(s(3))) )
     +		goto 90
	if ( smax .lt. xmax ) then
	    save = somb
	    xmax = smax
	endif
*
*	Second and third term, split k,l
*
*	more of the same ...
*	if ( k .eq. 4 ) then
*	    del2k = 0
*	else
*	    ii = inx(4,k)
*	    isii = isgn(4,k)
*	    call ffdl2s(del2k,xpi,piDpj,k,4,ii,isii,3,4,7,+1,10,ier)
*	endif
*	if ( l .eq. 4 ) then
*	    del2l = 0
*	else
*	    ii = inx(4,l)
*	    isii = isgn(4,l)
*	    call ffdl2s(del2l,xpi,piDpj,l,4,ii,isii,3,4,7,+1,10,ier)
*	endif
*	call ff2dl2(d2d2k,dum,xpi,dpipj,piDpj, k, i,j,ji,isji, 4,
*     +						3,4,7,+1, 10, ier)
*	call ff2dl2(d2d2l,dum,xpi,dpipj,piDpj, l, i,j,ji,isji, 4,
*     +						3,4,7,+1, 10, ier)
*	s(2) = +del2k*d2d2l
*	s(3) = -del2l*d2d2k
*	somb = s(1) + s(2) + s(3)
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( lwrite ) print *,'  somb+3= ',somb,s(1),s(2),s(3)
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
*
*	call ffdl2t(del2lk,piDpj, lk,4, 3,4,7,+1,+1, 10,ier)
*	call ff2dl2(d2d2lk,dum,xpi,dpipj,piDpj, lk, i,j,ji,isji, 4,
*     +						3,4,7,+1, 10, ier)
*	s(2) = +del2l*d2d2lk
*	s(3) = -del2lk*d2d2l
*	somb = s(1) + islk*(s(2) + s(3))
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( lwrite ) print *,'  somb+4= ',somb,s(1),s(2),s(3),islk
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
*
*	s(2) = +del2k*d2d2lk
*	s(3) = -del2lk*d2d2k
*	somb = s(1) + islk*(s(2) + s(3))
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( lwrite ) print *,'  somb+5= ',somb,s(1),s(2),s(3),isji
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
**
*	Second and third term, split m,n
**
*	call ff3dl2(d3d2m,xpi,dpipj,piDpj, 3, i,j,ji,isji,
*     +				k,l,lk,islk, 4, 3,4,7,+1, 10,ier)
*	call ff3dl2(d3d2n,xpi,dpipj,piDpj, 4, i,j,ji,isji,
*     +				k,l,lk,islk, 4, 3,4,7,+1, 10,ier)
*	s(2) = +d3d2m*piDpj(4,4)
*	s(3) = -d3d2n*piDpj(3,4)
*	somb = s(1) + s(2) + s(3)
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( lwrite ) print *,'  somb+6= ',somb,s(1),s(2),s(3)
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
*
*	call ff3dl2(d3d2nm,xpi,dpipj,piDpj, 7, i,j,ji,isji,
*     +				k,l,lk,islk, 4, 3,4,7,+1, 10,ier)
*	s(2) = +d3d2n*piDpj(7,4)
*	s(3) = -d3d2nm*piDpj(4,4)
*	somb = s(1) + s(2) + s(3)
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( lwrite ) print *,'  somb+7= ',somb,s(1),s(2),s(3)
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
*
*	s(2) = +d3d2m*piDpj(7,4)
*	s(3) = -d3d2nm*piDpj(3,4)
*	somb = s(1) + s(2) + s(3)
*	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*	if ( lwrite ) print *,'  somb+8= ',somb,s(1),s(2),s(3)
*	if ( abs(somb) .ge. xloss*smax ) goto 90
*	if ( smax .lt. xmax ) then
*	    save = somb
*	    xmax = smax
*	endif
*
   80	continue
*
*	give up:
*
	somb = save
	call ffwarn(89,ier,somb,xmax)
	if ( lwrite ) then
	    print *,'ffabcd: giving up on somb'
	    print *,'    i,j,k,l = ',i,j,k,l
	    print *,'    xpi     = ',xpi
	endif
   90	continue
	xb = somb/del2s
*  #] normal case b: 
*  #[ normal case d:
	call ff3dl2(s(1),xpi,dpipj,piDpj, 4, i,j,ji,isji, k,l,lk,islk,
     +						4, 3,4,7,+1, 10, ier)
	if ( i .eq. k .and. j .eq. l ) then
	    somd = -2*s(1)
	    if ( lwrite ) s(2) = s(1)
	else
	    call ff3dl2(s(2),xpi,dpipj,piDpj, 4, k,l,lk,islk,
     +				i,j,ji,isji, 4, 3,4,7,+1, 10, ier)
	    somd = - s(1) - s(2)
	    if ( abs(somd) .lt. xloss*abs(s(1)) ) then
		call ffwarn(90,ier,somd,s(1))
	    endif
	endif
*	if ( lwrite ) print *,'  somd = ',somd,s(1),s(2)
	xd = -somd/sdel2s
*  #] normal case d: 
*  #[ normal case c:
  800	continue
	s(1) = xb - xd
	s(2) = xb + xd
	som = s(1)*s(2)
	if ( min(abs(s(1)),abs(s(2))) .ge. xloss*abs(xb) ) goto 220
*	take into account that we know that we only need x+
	if ( xb*xd .ge. 0 ) goto 220
	call ffwarn(91,ier,min(abs(s(1)),abs(s(2))),xb)
	if ( lwrite ) print *,'b-d,b+d,b,d: ',s(1),s(2),xb,xd
  220	continue
	xc = som/xa
*  #] normal case c: 
*  #[ check output:
  900	continue
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    s(1)  = -piDpj(in,kn)*piDpj(jn,3)*piDpj(ln,3)*piDpj(4,4)
     +							**2
	    s(2)  = +piDpj(in,kn)*piDpj(jn,3)*piDpj(ln,4)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(3)  = +piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,3)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(4)  = -piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,4)*piDpj(3,4)
     +							**2
	    s(5)  = +piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,4)*piDpj(3,3)
     +							*piDpj(4,4)
	    s(6)  = -piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,4)*piDpj(3,4)
     +							**2
	    s(7)  = +piDpj(in,ln)*piDpj(jn,3)*piDpj(kn,3)*piDpj(4,4)
     +							**2
	    s(8)  = -piDpj(in,ln)*piDpj(jn,3)*piDpj(kn,4)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(9)  = -piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,3)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(10) = +piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,4)*piDpj(3,4)
     +							**2
	    s(11) = -piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,4)*piDpj(3,3)
     +							*piDpj(4,4)
	    s(12) = +piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,4)*piDpj(3,4)
     +							**2
	    s(13) = +piDpj(in,3)*piDpj(jn,kn)*piDpj(ln,3)*piDpj(4,4)
     +							**2
	    s(14) = -piDpj(in,3)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(15) = -piDpj(in,3)*piDpj(jn,ln)*piDpj(kn,3)*piDpj(4,4)
     +							**2
	    s(16) = +piDpj(in,3)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(17) = -piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,3)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(18) = +piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,4)
     +							**2
	    s(19) = +piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,3)*piDpj(3,4)
     +							*piDpj(4,4)
	    s(20) = -piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,4)
     +							**2
	    s(21) = -piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,3)
     +							*piDpj(4,4)
	    s(22) = +piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,4)
     +							**2
	    s(23) = +piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,3)
     +							*piDpj(4,4)
	    s(24) = -piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,4)
     +							**2
	    xbp = s(1)
	    smaxp = abs(s(1))
	    do 910 ll = 2,24
		xbp = xbp + s(ll)
		smaxp = max(smaxp,abs(xbp))
  910	    continue
	    xbp = xbp/del2s
	    smaxp = abs(smaxp/del2s)
	    if ( rloss*abs(xb-xbp) .gt. precx*smaxp ) then
		print *,'ffabcd: error: xb does not agree with ',
     +			'normal case:'
		print *,'  xb:  ',xb
		print *,'  xbp: ',xbp,smaxp
		print *,'  diff:',xb-xbp
		xb = xbp
	    endif
	    s(1)  = + piDpj(in,kn)*piDpj(jn,3)*piDpj(ln,4)*piDpj(4,4)
	    s(2)  = - piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,4)*piDpj(3,4)
	    s(3)  = + piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,3)*piDpj(4,4)
	    s(4)  = - piDpj(in,kn)*piDpj(jn,4)*piDpj(ln,4)*piDpj(3,4)
	    s(5)  = - piDpj(in,ln)*piDpj(jn,3)*piDpj(kn,4)*piDpj(4,4)
	    s(6)  = + piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,4)*piDpj(3,4)
	    s(7)  = - piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,3)*piDpj(4,4)
	    s(8)  = + piDpj(in,ln)*piDpj(jn,4)*piDpj(kn,4)*piDpj(3,4)
	    s(9)  = - piDpj(in,3)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(4,4)
	    s(10) = + piDpj(in,3)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(4,4)
	    s(11) = + piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,4)
	    s(12) = - piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,4)
	    s(13) = - piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,3)*piDpj(4,4)
	    s(14) = + piDpj(in,4)*piDpj(jn,kn)*piDpj(ln,4)*piDpj(3,4)
	    s(15) = + piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,3)*piDpj(4,4)
	    s(16) = - piDpj(in,4)*piDpj(jn,ln)*piDpj(kn,4)*piDpj(3,4)
	    xdp = s(1)
	    smaxp = abs(s(1))
	    do 920 ll = 2,16
		xdp = xdp + s(ll)
		smaxp = max(smaxp,abs(xdp))
  920	    continue
	    xdp = -xdp/sdel2s
	    smaxp = abs(smaxp/sdel2s)
	    if ( rloss*abs(xd-xdp) .gt. precx*smaxp ) then
		print *,'ffabcd: error: xd does not agree with ',
     +			'normal case:'
		print *,'  xd:  ',xd
		print *,'  xdp: ',xdp,smaxp
		print *,'  diff:',xd-xdp
		xd = xdp
	    endif
	endif
*  #] check output: 
*  #[ and tne final answer:
  990	continue
	call ffroot(dum,aijkl,xa,xb,xc,xd,ier)
*  #] and tne final answer: 
*###] ffabcd: 
	end

