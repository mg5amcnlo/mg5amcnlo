*###[ ff2dl2:
	subroutine ff2dl2(del2d2,del2n,xpi,dpipj,piDpj, i,
     +			j,k,kj,iskj,l, m,n,nm,isnm, ns, ier)
***#[*comment:***********************************************************
*									*
*	Calculate							*
*									*
*	 si mu	 mu sl							*
*	d	d	= si.sj*sk.sm*sl.sn - si.sk*sj.sm*sl.sn		*
*	 sj sk	 sm sn		- si.sj*sk.sn*sl.sm + si.sk*sj.sn*sl.sm	*
*									*
*	with p(kj) = iskj*(sk-sj)					*
*	with p(nm) = isnm*(sn-sm)					*
*									*
*	Input:	xpi(ns)			as usual			*
*		dpipj(ns,ns)		  -"-				*
*		piDpj(ns,ns)		  -"-				*
*		i,j,k,kj,iskj		see above			*
*		l,m,n,nm,isnm		  -"-				*
*									*
*	Output:	del2d2			see above			*
*		del2n			it is needed in fftran anyway	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer i,j,k,kj,iskj,l,m,n,nm,isnm,ns,ier
	DOUBLE PRECISION del2d2,del2n,xpi(10),dpipj(10,10),piDpj(10,10)
*
*	local variables:
*
	integer isii,ii,ik,ij,im,in,ier0,ier1
	DOUBLE PRECISION s(5),del2m,del2nm,som,xmax,smax
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    if ( abs(iskj) .ne. 1 ) print *,'ff2dl2: error: abs(iskj) ',
     +		'<> 1 but ',iskj
	    if ( abs(isnm) .ne. 1 ) print *,'ff2dl2: error: abs(isnm) ',
     +		'<> 1 but ',isnm
	    if ( ns .ne. 10 ) print *,'ff2dl2: error: ns <> 10 !!'
	    if ( kj.eq.0 ) then
		print *,'ff2dl2: error: kj=0:j,k,id,idsub=',j,k,id,idsub
	    endif
	    if ( nm.eq.0 ) then
		print *,'ff2dl2: error: nm=0:m,n,id,idsub=',m,n,id,idsub
	    endif
	endif
*  #] check input: 
*  #[ get del2n:
*	we need this in any case !
	ier1 = ier
	if ( i .eq. n ) then
	    del2n = 0
	elseif ( i .le. 4 ) then
	    ii = inx(n,i)
	    isii = isgn(n,i)
	    call ffdl2s(del2n,xpi,piDpj,i,n,ii,isii,j,k,kj,iskj,10,ier1)
	else
	    call ffdl2t(del2n,piDpj,i,n,j,k,kj,iskj,+1,10,ier1)
	endif
*  #] get del2n:
*  #[ special cases:
	ier0 = ier
	if ( i .eq. l .and. j .eq. m .and. k .eq. n ) then
	    call ffdl3m(s,.FALSE.,x0,x0,xpi,dpipj,piDpj,ns,j,k,kj,
     +		i,1,ier0)
	    del2d2 = -s(1)
*	    if ( lwrite ) print *,'del2d2 = ',del2d2
	    ier = max(ier0,ier1)
	    return
	endif
	if ( k .eq. l .and. j .le. 4 ) then
	    call ffdl2s(del2m,xpi,piDpj, j,l,inx(l,j),isgn(l,j),
     +		m,n,nm,isnm, 10,ier0)
	    del2d2 = -piDpj(i,k)*del2m
*	    if ( lwrite ) print *,'del2d2 = ',del2d2
	    ier = max(ier0,ier1)
	    return
	endif
*	not yet tested:
*	if ( j .eq. l .and. k .le. 4 ) then
*	    call ffdl2s(del2m,xpi,piDpj, k,l,inx(k,j),isgn(k,j),
*     +		m,n,nm,isnm, 10,ier0)
*	    del2d2 = piDpj(i,j)*del2m
*	    ier = max(ier0,ier1)
*	    return
*	endif
*  #] special cases:
*  #[ calculations:
	ier0 = ier
	if ( i .eq. m ) then
	    del2m = 0
	elseif ( i .le. 4 ) then
	    ii = inx(m,i)
	    isii = isgn(m,i)
	    call ffdl2s(del2m,xpi,piDpj,i,m,ii,isii,j,k,kj,iskj,10,ier1)
	else
	    call ffdl2t(del2m,piDpj,i,m,j,k,kj,iskj,+1,10,ier1)
	endif
	s(1) = del2m*piDpj(n,l)
	s(2) = del2n*piDpj(m,l)
	smax = abs(s(1))*DBLE(10)**(ier0-ier)
	del2d2 = s(1) - s(2)
	if ( abs(del2d2) .ge. xloss*smax ) goto 60

	som = del2d2
	xmax = smax
	if ( lwrite ) print *,'  del2d2  = ',del2d2,xmax

	ier0 = ier
	call ffdl2t(del2nm,piDpj,i,nm,j,k,kj,iskj,+1,10,ier0)
	s(1) = del2n*piDpj(nm,l)
	s(2) = del2nm*piDpj(n,l)
	del2d2 = isnm*(s(1) - s(2))
	smax = abs(s(2))*DBLE(10)**(ier0-ier)
	if ( lwrite ) print *,'  del2d2+ = ',del2d2,smax
	if ( abs(del2d2) .ge. xloss*abs(s(1)) ) goto 60

	if ( smax .lt. xmax ) then
	    som = del2d2
	    xmax = smax
	endif

	s(1) = del2m*piDpj(nm,l)
	s(2) = del2nm*piDpj(m,l)
	del2d2 = isnm*(s(1) - s(2))
	smax = abs(s(2))*DBLE(10)**(ier0-ier)
	if ( lwrite ) print *,'  del2d2+ = ',del2d2,smax
	if ( abs(del2d2) .ge. xloss*abs(s(1)) ) goto 60

	if ( smax .lt. xmax ) then
	    som = del2d2
	    xmax = smax
	endif

*	One more special case:
	if ( k .eq. m ) then
	    isii = -1
	    ik = j
	    ij = k
	    im = m
	    in = n
	elseif ( j .eq. m ) then
	    isii = +1
	    ik = k
	    ij = j
	    im = m
	    in = n
	elseif ( j .eq. n ) then
	    isii = -1
	    ik = k
	    ij = j
	    im = n
	    in = m
	elseif ( k .eq. n ) then
	    isii = +1
	    ik = j
	    ij = k
	    im = n
	    in = m
	else
	    goto 50
	endif
	if ( ij .eq. im .and. i .le. 4 .and. ij .le. 4 .and. in .le. 4 )
     +								then
	    if ( inx(ij,i) .gt. 0 .and. inx(im,l) .gt. 0 ) then
		if (  abs(dpipj(i,inx(ij,i))) .lt. xloss*abs(xpi(ij))
     +		.and. abs(dpipj(l,inx(im,l))) .lt. xloss*abs(xpi(im)) )
     +								then
		    s(1) = piDpj(l,in)*piDpj(ik,ij)*dpipj(i,inx(ij,i))/2
		    s(2) = isgn(ij,i)*piDpj(l,in)*xpi(ij)*piDpj(ik,
     +			inx(ij,i))/2
		    s(3) = -piDpj(i,ij)*piDpj(ik,in)*piDpj(l,im)
		    s(4) = piDpj(i,ik)*piDpj(im,in)*dpipj(l,inx(im,l))/2
		    s(5) = isgn(im,l)*piDpj(i,ik)*xpi(im)*piDpj(in,
     +			inx(im,l))/2
		    del2d2 = s(1) + s(2) + s(3) + s(4) + s(5)
		    if ( isii .lt. 0 ) del2d2 = -del2d2
		    smax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +			abs(s(5)))
		    if ( lwrite ) print *,'  del2d2* = ',del2d2,s
		    if ( abs(del2d2) .ge. xloss**2*abs(smax) ) goto 60
		    if ( smax .lt. xmax ) then
			som = del2d2
			xmax = smax
		    endif
		endif
	    endif
	endif
   50	continue
*
*	give up
*
	del2d2 = som
	if ( lwarn ) call ffwarn(123,ier,del2d2,xmax)
	if ( lwrite ) then
	    print *,'ff2dl2: giving up on this case'
	    print *,'        indices: i=n:',i,j,k,l,m,n
	    print *,'        xpi: ',xpi
	endif

   60	continue
*  #] calculations:
*  #[ check:
	if ( ltest ) then
	    s(1) = + piDpj(i,j)*piDpj(k,m)*piDpj(l,n)
	    s(2) = - piDpj(i,k)*piDpj(j,m)*piDpj(l,n)
	    s(3) = - piDpj(i,j)*piDpj(k,n)*piDpj(l,m)
	    s(4) = + piDpj(i,k)*piDpj(j,n)*piDpj(l,m)
	    som = s(1) + s(2) + s(3) + s(4)
	    xmax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)))
	    if ( xloss*abs(som-del2d2) .gt. precx*xmax ) then
		print *,'ff2dl2: error: del2d2 not correct: ',del2d2,
     +		som,xmax,del2d2-som
	    endif
	endif
*  #] check: 
*###] ff2dl2:
	end
*###[ ff2d22:
	subroutine ff2d22(dl2d22,xpi,dpipj,piDpj, i, j,k,kj,iskj,
     +			m,n,nm,isnm, ns, ier)
***#[*comment:***********************************************************
*									*
*	Calculate							*
*									*
*	/ si mu	 mu nu \2						*
*	|d	d      |						*
*	\ sj sk	 sm sn /						*
*									*
*	=   si.sj^2*sk.sm^2*sn.sn					*
*	- 2*si.sj^2*sk.sm*sk.sn*sm.sn					*
*	+   si.sj^2*sk.sn^2*sm.sm					*
*	- 2*si.sj*si.sk*sj.sm*sk.sm*sn.sn				*
*	+ 2*si.sj*si.sk*sj.sm*sk.sn*sm.sn				*
*	+ 2*si.sj*si.sk*sj.sn*sk.sm*sm.sn				*
*	- 2*si.sj*si.sk*sj.sn*sk.sn*sm.sm				*
*	+   si.sk^2*sj.sm^2*sn.sn					*
*	- 2*si.sk^2*sj.sm*sj.sn*sm.sn					*
*	+   si.sk^2*sj.sn^2*sm.sm					*
*									*
*	Input:	xpi(ns)			as usual			*
*		dpipj(ns,ns)		  -"-				*
*		piDpj(ns,ns)		  -"-				*
*		i,j,k,kj,iskj		see above			*
*		m,n,nm,isnm		 -"-				*
*									*
*	Output:	dl2d22			see above			*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer i,j,k,kj,iskj,m,n,nm,isnm,ns,ier
	DOUBLE PRECISION dl2d22,xpi(10),dpipj(10,10),piDpj(10,10)
*
*	local variables:
*
	integer ii,isii
	DOUBLE PRECISION s(10),del2s,del23,del24,del27,som,smax,xmax
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations: 
*  #[ check input:
	if ( ltest ) then
	    if ( abs(iskj) .ne. 1 ) print *,'ff2d22: error: abs(iskj) ',
     +		'<> 1 but ',iskj
	    if ( abs(isnm) .ne. 1 ) print *,'ff2d22: error: abs(isnm) ',
     +		'<> 1 but ',isnm
	    if ( ns .ne. 10 ) print *,'ff2d22: error: ns <> 10 !!'
	    if ( m .ne. 3 .or. n .ne. 4 ) print *,'ff2d22: error ',
     +		'only for m=3,n=4 !!'
	endif
*  #] check input: 
*  #[ special cases:
	if ( i .eq. n .or. i .eq. m ) then
	    call ffdl2s(del2s,xpi,piDpj, j,k,kj,iskj, m,n,nm,isnm,
     +							 10,ier)
	    dl2d22 = xpi(i)*del2s**2
*	    if ( lwrite ) print *,'  dl2d22  = ',dl2d22
	    return
	endif
*  #] special cases: 
*  #[ calculations:
*	We use the product form
	if ( i .eq. 3 ) then
	    del23 = 0
	elseif ( i .le. 4 ) then
	    ii = inx(3,i)
	    isii = isgn(3,i)
	    call ffdl2s(del23,xpi,piDpj,i,3,ii,isii,j,k,kj,iskj,10,ier)
	else
	    call ffdl2t(del23,piDpj,i,3,j,k,kj,iskj,+1,10,ier)
	endif
	if ( i .eq. 4 ) then
	    del24 = 0
	elseif ( i .le. 4 ) then
	    ii = inx(n,i)
	    isii = isgn(n,i)
	    call ffdl2s(del24,xpi,piDpj,i,4,ii,isii,j,k,kj,iskj,10,ier)
	else
	    call ffdl2t(del24,piDpj,i,4,j,k,kj,iskj,+1,10,ier)
	endif

	s(1) = xpi(4)*del23**2
	s(2) = -2*piDpj(3,4)*del23*del24
	s(3) = xpi(3)*del24**2
	dl2d22 = s(1) + s(2) + s(3)
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( abs(dl2d22) .ge. xloss*smax ) goto 110

	som = dl2d22
	xmax = smax
	if ( lwrite ) print *,'  dl2d22  = ',dl2d22,s(1),s(2),s(3)

*	try the special case k=4 (for use in ee->mumu among others)
	if ( i .lt. 4 .and. k .eq. 4 .and. abs(s(3)) .lt. xloss*smax
     +		.and. ( abs(dpipj(i,inx(4,i))) .lt. xloss*xpi(i) .or.
     +		abs(piDpj(j,inx(4,i))) .lt. xloss*abs(piDpj(j,4)) ) )
     +								then
	    s(1) = -del23*piDpj(i,4)*piDpj(j,3)*xpi(4)
	    s(2) =  del23*dpipj(i,inx(4,i))*piDpj(j,4)*piDpj(3,4)
	    s(4) =  del23*piDpj(3,4)*xpi(4)*piDpj(j,inx(4,i))*isgn(4,i)
	    dl2d22 = s(1) + s(2) + s(3) + s(4)
	    smax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)))
	    if ( lwrite ) print *,'  dl2d22* = ',dl2d22,s(1),s(2),s(3),
     +		s(4)
	    if ( abs(dl2d22) .ge. xloss*smax ) goto 110

	    if ( smax .lt. xmax ) then
		som = dl2d22
		xmax = smax
	    endif
	endif

	call ffdl2t(del27,piDpj,i,7,j,k,kj,iskj,+1,10,ier)
	s(1) = xpi(7)*del24**2
	s(2) = -2*piDpj(4,7)*del24*del27
	s(3) = xpi(4)*del27**2
	dl2d22 = s(1) + s(2) + s(3)
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( lwrite ) print *,'  dl2d22+ = ',dl2d22,s(1),s(2),s(3)
	if ( abs(dl2d22) .ge. xloss*smax ) goto 110

	if ( smax .lt. xmax ) then
	    som = dl2d22
	    xmax = smax
	endif

	s(1) = xpi(7)*del23**2
	s(2) = -2*piDpj(3,7)*del23*del27
	s(3) = xpi(3)*del27**2
	dl2d22 = s(1) + s(2) + s(3)
	smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
	if ( lwrite ) print *,'  dl2d22+ = ',dl2d22,s(1),s(2),s(3)
	if ( abs(dl2d22) .ge. xloss*smax ) goto 110
*
*	We'll have to think of something more intelligent ...
*
	if ( smax .lt. xmax ) then
	    som = dl2d22
	    xmax = smax
	endif

	dl2d22 = som
	if ( lwarn ) call ffwarn(122,ier,dl2d22,xmax)
	if ( lwrite ) then
	    print *,'ff2d22: give up on this case ...'
	    print *,'        indices: ijkmn:',i,j,k,m,n
	    print *,'        xpi:',xpi
	endif

  110	continue
*  #] calculations: 
*  #[ check:
	if ( ltest ) then
	    s(1) = +   piDpj(i,j)**2*piDpj(k,m)**2*piDpj(n,n)
	    s(2) = - 2*piDpj(i,j)**2*piDpj(k,m)*piDpj(k,n)*piDpj(m,n)
	    s(3) = +   piDpj(i,j)**2*piDpj(k,n)**2*piDpj(m,m)
	    s(4) = - 2*piDpj(i,j)*piDpj(i,k)*piDpj(j,m)*piDpj(k,m)*
     +		piDpj(n,n)
	    s(5) = + 2*piDpj(i,j)*piDpj(i,k)*piDpj(j,m)*piDpj(k,n)*
     +		piDpj(m,n)
	    s(6) = + 2*piDpj(i,j)*piDpj(i,k)*piDpj(j,n)*piDpj(k,m)*
     +		piDpj(m,n)
	    s(7) = - 2*piDpj(i,j)*piDpj(i,k)*piDpj(j,n)*piDpj(k,n)*
     +		piDpj(m,m)
	    s(8) = +   piDpj(i,k)**2*piDpj(j,m)**2*piDpj(n,n)
	    s(9) = - 2*piDpj(i,k)**2*piDpj(j,m)*piDpj(j,n)*piDpj(m,n)
	    s(10)= +   piDpj(i,k)**2*piDpj(j,n)**2*piDpj(m,m)
	    som = 0
	    xmax = 0
	    do 900 ii=1,10
		som = som + s(ii)
		xmax = max(xmax,abs(s(ii)))
  900	    continue
	    if ( xloss*abs(som-dl2d22) .gt. precx*xmax ) then
		print *,'ff2c22: error: dl2d22 not correct: ',dl2d22,
     +		som,xmax
	    endif
	endif
*  #] check: 
*###] ff2d22: 
	end
*###[ ff3dl2:
	subroutine ff3dl2(del3d2,xpi,dpipj,piDpj, i,
     +		j,k,kj,iskj, l,m,ml,isml, n, o,p,po,ispo, ns, ier)
***#[*comment:***********************************************************
*									*
*	Calculate							*
*									*
*	 si mu	 mu nu	 mu sn						*
*	d	d	d	= ...					*
*	 sj sk	 sl sm	 so sp						*
*									*
*	with p(kj) = iskj*(sk-sj)					*
*	     p(ml) = isml*(sm-sl)					*
*	     p(po) = ispo*(sp-so)					*
*									*
*	Input:	xpi(ns)			as usual			*
*		dpipj(ns,ns)		  -"-				*
*		piDpj(ns,ns)		  -"-				*
*		i,j,k,kj,iskj		see above			*
*		l,m,ml,isml		  -"-				*
*		n,o,p,po,ispo		  -"-				*
*									*
*	Output:	del3d2			see above			*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer i,j,k,kj,iskj,l,m,ml,isml,n,o,p,po,ispo,ns,ier
	DOUBLE PRECISION del3d2,xpi(10),dpipj(10,10),piDpj(10,10)
*
*	local variables:
*
	integer isii,ii
	DOUBLE PRECISION s(2),dl2il,dl2im,dl2ln,dl2mn,dl2iml,dl2mln
	DOUBLE PRECISION d2d2j,d2d2k,d2d2kj,dum,d2d2o,d2d2p,d2d2po
	DOUBLE PRECISION som,xmax
*
*	common blocks:
*
	include 'ff.h'
*  #] declarations: 
*  #[ check input:
	if ( ltest ) then
	    if ( abs(iskj) .ne. 1 ) print *,'ff3dl2: error: abs(iskj) ',
     +		'<> 1 but ',iskj
	    if ( abs(isml) .ne. 1 ) print *,'ff3dl2: error: abs(isml) ',
     +		'<> 1 but ',isml
	    if ( abs(ispo) .ne. 1 ) print *,'ff3dl2: error: abs(ispo) ',
     +		'<> 1 but ',ispo
	    if ( ns .ne. 10 ) print *,'ff3dl2: error: ns <> 10 !!'
	endif
*  #] check input: 
*  #[ split up l,m:
	if ( i .eq. l ) then
	    dl2il = 0
	elseif ( i .le. 4 ) then
	    ii = inx(l,i)
	    isii = isgn(l,i)
	    call ffdl2s(dl2il,xpi,piDpj,i,l,ii,isii,j,k,kj,iskj,10,ier)
	else
	    call ffdl2t(dl2il,piDpj,i,l,j,k,kj,iskj,+1,10,ier)
	endif
	if ( m .eq. n ) then
	    dl2mn = 0
	elseif ( i .le. 4 ) then
	    ii = inx(n,m)
	    isii = isgn(n,m)
	    call ffdl2s(dl2mn,xpi,piDpj,m,n,ii,isii,o,p,po,ispo,10,ier)
	else
	    call ffdl2t(dl2mn,piDpj,m,n,o,p,po,ispo,+1,10,ier)
	endif
	s(1) = dl2il*dl2mn
	if ( i .eq. m ) then
	    dl2im = 0
	elseif ( i .le. 4 ) then
	    ii = inx(m,i)
	    isii = isgn(m,i)
	    call ffdl2s(dl2im,xpi,piDpj,i,m,ii,isii,j,k,kj,iskj,10,ier)
	else
	    call ffdl2t(dl2im,piDpj,i,m,j,k,kj,iskj,+1,10,ier)
	endif
	if ( l .eq. n ) then
	    dl2ln = 0
	elseif ( i .le. 4 ) then
	    ii = inx(n,l)
	    isii = isgn(n,l)
	    call ffdl2s(dl2ln,xpi,piDpj,l,n,ii,isii,o,p,po,ispo,10,ier)
	else
	    call ffdl2t(dl2ln,piDpj,l,n,o,p,po,ispo,+1,10,ier)
	endif
	s(2) = dl2im*dl2ln
	del3d2 = s(1) - s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( lwrite ) print *,'  del3d2  = ',del3d2,s(1),-s(2)
	som = del3d2
	xmax = abs(s(1))
*
*	rotate l,m
*
	call ffdl2t(dl2mln,piDpj,ml,n,o,p,po,ispo,+1,10,ier)
	call ffdl2t(dl2iml,piDpj,i,ml,j,k,kj,iskj,+1,10,ier)
	s(1) = dl2im*dl2mln
	s(2) = dl2iml*dl2mn
	del3d2 = isml*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

	s(1) = dl2il*dl2mln
	s(2) = dl2iml*dl2ln
	del3d2 = isml*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

*  #] split up l,m: 
*  #[ split up j,k:
	call ff2dl2(d2d2k,dum,xpi,dpipj,piDpj, k, l,m,ml,isml, n,
     +						o,p,po,ispo, 10, ier)
	call ff2dl2(d2d2j,dum,xpi,dpipj,piDpj, j, l,m,ml,isml, n,
     +						o,p,po,ispo, 10, ier)
	s(1) = piDpj(i,j)*d2d2k
	s(2) = piDpj(i,k)*d2d2j
	del3d2 = s(1) - s(2)
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

	call ff2dl2(d2d2kj,dum,xpi,dpipj,piDpj, kj, l,m,ml,isml, n,
     +						o,p,po,ispo, 10, ier)
	s(1) = piDpj(i,k)*d2d2kj
	s(2) = piDpj(i,kj)*d2d2k
	del3d2 = iskj*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

	s(1) = piDpj(i,j)*d2d2kj
	s(2) = piDpj(i,kj)*d2d2j
	del3d2 = iskj*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

*  #] split up j,k: 
*  #[ split up o,p:
	call ff2dl2(d2d2o,dum,xpi,dpipj,piDpj, i, j,k,kj,iskj, o,
     +						l,m,ml,isml, 10, ier)
	call ff2dl2(d2d2p,dum,xpi,dpipj,piDpj, i, j,k,kj,iskj, p,
     +						l,m,ml,isml, 10, ier)
	s(1) = piDpj(p,n)*d2d2o
	s(2) = piDpj(o,n)*d2d2p
	del3d2 = s(1) - s(2)
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

	call ff2dl2(d2d2po,dum,xpi,dpipj,piDpj, i, j,k,kj,iskj, po,
     +						l,m,ml,isml, 10, ier)
	s(1) = piDpj(po,n)*d2d2p
	s(2) = piDpj(p,n)*d2d2po
	del3d2 = ispo*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

	s(1) = piDpj(po,n)*d2d2o
	s(2) = piDpj(o,n)*d2d2po
	del3d2 = ispo*(s(1) - s(2))
	if ( lwrite ) print *,'  del3d2+ = ',del3d2,s(1),-s(2)
	if ( abs(del3d2) .ge. xloss*abs(s(1)) ) return

	if ( abs(s(1)) .lt. xmax ) then
	    som = del3d2
	    xmax = abs(s(1))
	endif

*  #] split up o,p: 
*  #[ give up:
	del3d2 = som
	if ( lwarn ) call ffwarn(124,ier,del3d2,xmax)
*  #] give up: 
*###] ff3dl2: 
	end
