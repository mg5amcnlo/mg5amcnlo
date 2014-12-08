*###[ fftest:
	subroutine fftest
***#[*comment:***************************************************
*	test the two-, three- and fourpoint functions		*
*	with values in the file input, in the			*
*	following order:					*
*	@   4*true/false (lwrite,ltest,l4also,ldc3c4,lmem)	*
*	@   1/2/3/4	two-, three- or fourpoints		*
*	@   one line with arbitrary comment			*
*	@   2/2/0/0 renormalization constants (delta,mu)	*
*	@   1/2/3/4 complex (internal) masses			*
*			    without brackets			*
*	@   0/1/3/6 external momenta (+s,t) (real)		*
*	@   # of times the function has to be			*
*		 evaluated (for timing purposes)		*
*	@   1/3/6 other momenta or 12345 to enter		*
*		 new comment, masses and momenta		*
*	@   # of times						*
*	etc							*
***#]*comment:***************************************************
*  #[ declarations:
	implicit none
	integer i,k,npoint,nmom,nren,ier,icon,ifile,ial,ntime,ialsav,
     +		iersav,ier0,ier1,ier2,is1,is2,ip,iinx(6,4),imiss
	parameter(icon=3)
	parameter(ifile=1)
	DOUBLE COMPLEX cpi(20),cqi(20),ca0,cb0,cb1,cb0i(3),cc0,cc1i(2),
     +		cc0i(4),cc0r,cd0,cd1i(3),ce0,cd0i(5),ca0i(2),cdb0,cdb0p,
     +		cb2p,cb2d,cb2i(2),cc,cpp,cb01,cb02
	DOUBLE COMPLEX cai(5),cbij(4,5,5),ccij(13,5,5),cdi(33,5),ce(126)
	DOUBLE PRECISION xpi(20),xqi(20),delta,a(2),xma,xmb,t1,t2,fftyd,
     +		xnul,xmax,rloss,absc,eps,del2i(4),odel2i(4)
	character*79 text
	logical lsquar,lcon,ldotsa
	integer inew(10,6),inew5(20,12),irota,ntens,ndiv
	save inew,inew5,iinx
	common /ffcut/ delta
	include 'ff.h'
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data iinx /2,3,4,6,7,10,
     +		   1,3,4,9,7,8,
     +		   1,2,4,5,10,8,
     +		   1,2,3,5,6,9/
	data inew /1,2,3,4,5,6,7,8,9,10,
     +		   4,1,2,3,8,5,6,7,10,9,
     +		   3,4,1,2,7,8,5,6,9,10,
     +		   2,3,4,1,6,7,8,5,10,9,
     +		   4,2,3,1,10,6,9,8,7,5,
     +		   1,3,2,4,9,6,10,8,5,7/
	data inew5
     +	       /1,2,3,4,5, 6,7,8,9,10,11,12,13,14,15, 16,17,18,19,20,
     +		2,1,3,4,5, 6,11,8,9,15,7,14,13,12,10, 16,18,17,19,-20,
     +		1,3,2,4,5, 11,7,12,9,10,6,8,15,14,13, -16,17,19,18,20,
     +		1,2,4,3,5, 6,12,8,13,10,14,7,9,11,15, 16,-17,18,20,19,
     +		1,2,3,5,4, 6,7,13,9,14,11,15,8,10,12, 20,17,-18,19,16,
     +		5,2,3,4,1, 15,7,8,14,10,13,12,11,9,6, 17,16,18,-19,20,
     +		2,1,4,3,5, 6,14,8,13,15,12,11,9,7,10, 16,-18,17,20,-19,
     +		1,3,2,5,4, 11,7,15,9,14,6,13,12,10,8, -20,17,-19,18,16,
     +		5,2,4,3,1, 15,12,8,11,10,9,7,14,13,6, 17,-16,18,-20,19,
     +		2,1,3,5,4, 6,11,13,9,12,7,10,8,15,14, 20,18,-17,19,-16,
     +		5,3,2,4,1, 13,7,12,14,10,15,8,6,9,11, -17,16,19,-18,20,
     +	      1,3,5,2,4, 11,13,15,12,14,10,7,9,6,8,-20,-17,-19,-16,-18/
	nevent = 0
*  #] declarations:
*  #[ read input:
	call ffini
	open(ifile,file='ffinput',status='old')
	rewind(ifile)
	read(ifile,*,end=999,err=999)lwrite
	read(ifile,*,end=999,err=999)ltest
	read(ifile,*,end=999,err=999)l4also
	read(ifile,*,end=999,err=999)ldc3c4
	read(ifile,*,end=999,err=999)lmem
	read(ifile,*,end=999,err=999)ldot
	read(ifile,*,end=999,err=999)lwarn
	read(ifile,*,end=999,err=999)ial
	read(ifile,*,end=999,err=999)nschem
	read(ifile,*,end=999,err=999)nwidth
	read(ifile,*,end=999,err=999)ndiv
	if ( nwidth.le.0 ) then
	    onshel = .TRUE.
	else
	    onshel = .FALSE.
	endif
	if (lwrite) print *,'fftest: give debug output'
	if (ltest) print *,'fftest: test consistency'
	if (l4also) print *,'fftest: consider 4*4 dilogarithms as well'
	if (ldc3c4) print *,'fftest: consider the difference of the ',
     +		'3point functions as well'
	if (lmem) print *,'fftest: use memory'
	if (ldot) print *,'fftest: calculate dotproducts for tensor ',
     +		'integrals'
	if ( lwarn ) print *,'fftest: give warning messages'
	print *,'sign of root in shift, ial = ',ial
	print *,'fftest: requested scheme = ',nschem
	print *,'fftest: requested ndiv   = ',ndiv
	if ( nschem .eq. 1 ) then
	print *,'(use the real masses everywhere)'
	elseif ( nschem .eq. 2 ) then
	print *,'(use the complex mass only in poles)'
	elseif ( nschem .eq. 3 ) then
	print *,'(use the complex mass in poles and divergent logs)'
	elseif ( nschem .eq. 4 ) then
	print *,'(use the complex mass when there are poles or ',
     +		'divergent logs)'
	elseif ( nschem .eq. 5 ) then
	print *,'(use the complex mass when there are poles, ',
     +		'divergent logs or (0,m,m) thresholds)'
	elseif ( nschem .eq. 6 ) then
	print *,'(use the complex mass when there are poles, ',
     +		'divergent logs or thresholds)'
	elseif ( nschem .eq. 7 ) then
	print *,'(use the complex mass everywhere)'
	endif
	if ( onshel ) then
	    print *,'using onshell scheme'
	else
	    print *,'nwidth = ',nwidth
	endif
	if ( lwrite ) then
	    open(icon,file='CON:',status='old',err=11)
	    lcon = .TRUE.
	    goto 13
	endif
   11	continue
	lcon = .FALSE.
   13	continue
	read(ifile,*,end=999,err=999)npoint
	print *,'fftest:',npoint,' punts functie gevraagd'
	if ( npoint .lt. 0 ) then
	    lsquar = .TRUE.
	    npoint = -npoint
	else
	    lsquar = .FALSE.
	endif
	if ( npoint .gt. 10 ) then
	    ntens = npoint/10
	    if ( ntens.gt.0 ) ldot = .TRUE.
	    npoint = mod(npoint,10)
	endif
	if (npoint.eq.1) then
	    nmom = 0
	    nren = 2
	elseif (npoint.eq.2) then
	    nmom = 1
	    nren = 2
	elseif (npoint.eq.3) then
	    nmom = 3
	    nren = 1
	elseif (npoint.eq.4) then
	    nmom = 6
	    nren = 1
	elseif (npoint.eq.5) then
	    nmom = 10
	    nren = 1
	else
	    print*,'error: npoint=',npoint,' not yet implemented'
	    stop
	endif
    1	continue
	nevent = nevent + 1
	isgnal = sign(1,ial)
	read(ifile,'(a)',end=999,err=999)text
	if ( lcon ) write(icon,'(a)')text
	if ( text .eq. 'stop' ) stop
	print '(3a)','####[ ',text,':'
	print '(a)','--##[ input: '
	print *,nren,' renormalisatie variabelen'
	do 14 i=1,nren
	    read(ifile,*,end=999,err=999)a(i)
	    if (i.eq.1)then
		if ( lsquar ) then
		    delta = a(1)
		else
		    delta = a(1)**2
		endif
		print *,'delta=',delta
	    endif
	    if (i.eq.2)print *,'mu   =',a(2)
   14	continue
    5	do 2 i=1,npoint
	    if ( lsquar ) then
		read(ifile,*,end=999,err=999) xma,xmb
		cpi(i) = DCMPLX(xma,xmb)
		print *,'massa',i,'^2 =',cpi(i)
	    else
		read(ifile,*,end=999,err=999) xma,xmb
		cpi(i) = DCMPLX(xma**2,xma*xmb)
		print *,'massa',i,'^2 =',cpi(i)
	    endif
    2	continue
	do 12 i=1,nmom
	    if ( lsquar ) then
		read(ifile,*,end=999,err=999) xma
		cpi(i+npoint) = DCMPLX(DBLE(xma))
		print *,'pi(',i,')^2 (B&D)=',cpi(i+npoint)
	    else
		read(ifile,*,end=999,err=999) xma,xmb
		if ( xmb.eq.0 ) then
		    cpi(i+npoint) = xma**2
		elseif ( xma.eq.0 ) then
		    cpi(i+npoint) = -xmb**2
		else
		    print *,'fftest: error: complex p^2????'
		    cpi(i+npoint) = DCMPLX(xma,xmb)**2
		endif
		print *,'cpi(',i,')^2 (B&D)=',cpi(i+npoint)
	    endif
   12	continue
	do 16 i=nmom+npoint+1,20
	    cpi(i) = 0
   16	continue
*  #] read input:
*  #[ compute onshell quantities:
	do 17 i=1,nmom+npoint
	    xpi(i) = DBLE(cpi(i))
   17	continue
	do 19 is1=1,npoint
	    do 18 is2=is1+1,npoint
	    	if ( npoint.eq.2 ) then
	    	    ip = 3
	    	elseif ( npoint.eq.3 ) then
	    	    if ( abs(is1-is2).eq.1 ) then
	    	    	ip = is1+3
	    	    else
	    	    	ip = is2+3
	    	    endif
	    	elseif ( npoint.eq.4 ) then
	    	    ip = inx(is1,is2)
	    	elseif ( npoint.eq.5 ) then
	    	    ip = inx5(is1,is2)
	    	elseif ( onshel ) then
	    	    print *,'fftest: cannot yet compute onshell ',
     +	    	    	'momenta for npoint>5'
	    	    stop
	    	endif
	    	if ( abs(xpi(is1)-xpi(ip)).lt.-5*DIMAG(cpi(is1)) 
     +	    		.and. xpi(is2).lt.-DIMAG(cpi(is1)) ) then
		    xpi(ip) = xpi(is1)
*why?		    xpi(is2) = 0
	    	elseif ( abs(xpi(is2)-xpi(ip)).lt.-5*DIMAG(cpi(is2)) 
     +	    		.and. xpi(is1).lt.-DIMAG(cpi(is2)) ) then
		    xpi(ip) = xpi(is2)
*why?		    xpi(is1) = 0
	    	endif
   18	    continue
   19	continue
	if ( onshel .and. nschem.lt.7 ) then
	    print *,'onshell:'
	    do 22 i=1,npoint
		print *,'massa',i,'^2 =',xpi(i)
   22	    continue
	    do 32 i=1,nmom
		print *,'pi(',i,')^2 (B&D)=',xpi(i+npoint)
   32	    continue
	endif
	read(ifile,*,end=999,err=999) k
	print *,'aantal keer (voor timing): ',k
	print '(a)','--##] input: '
	id = 1
*  #] compute onshell quantities:
*  #[ one point function:
**************the one point function******************
	if (npoint .eq. 1 ) then
	if ( k.gt.1 ) t1 = fftyd(1)
	do 100 i=1,k
	ier = 0
	call ffza0(ca0,a(1),a(2),cpi(1),xpi(1),ndiv,ier)
  100	continue
	if ( k.gt.1 ) t2 = fftyd(2)
	print '(a,2g23.13,i6)','a0 = ',ca0,ier
	if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
*  #] one point function:
*  #[ two point function:
**************the two point function******************
	elseif (npoint .eq. 2 ) then
	if ( k.gt.1 ) t1 = fftyd(1)
	do 200 i=1,k
	ier = 0
	call ffzb0(cb0,a(1),a(2),cpi(3),cpi(1),cpi(2),
     +		xpi(3),xpi(1),xpi(2),ndiv,ier)
  200	continue
	if ( k.gt.1 ) t2 = fftyd(2)
	print '(a,2g23.13,i6)','b0 = ',cb0,ier
	if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	if ( lcon ) write(icon,'(a,2g23.13,i6)')'b0 = ',cdb0,ier
	call ffzdb0(cdb0,cdb0p,cpi(3),cpi(1),cpi(2),
     +		xpi(3),xpi(1),xpi(2),ndiv,ier)
	print '(a,2g23.13,i6)','   b0''= ',cdb0,ier
	print '(a,2g23.13,i6)','xp*b0''= ',cdb0p,ier
	if ( lcon ) write(icon,'(a,2g23.13,i6)')'b0''= ',cdb0,ier
	if ( .not.onshel .or. nschem.ge.7 ) then
	    xnul = absc(cdb0*cpi(3)-cdb0p)
	else
	    xnul = absc(cdb0*DBLE(xpi(3))-cdb0p)
	endif
	rloss = xloss**3*DBLE(10)**(-mod(ier,50))
	if ( rloss*xnul .gt. precc*absc(cdb0p) ) print *,
     +		'fftest: error: p^2*B0'' != p^2*B0'':',cpi(3)*cdb0,
     +		cdb0p,xnul
	if ( .not.onshel .or. nschem.ge.7 ) then
*
*	    check B0'
*
	    ldotsa = ldot
	    ldot = .FALSE.
	    eps = 1.e-3
	    cpp = cpi(3)*(1-eps/2)
	    ier = 0
	    call ffzb0(cb01,a(1),a(2),cpp,cpi(1),cpi(2),
     +		xpi(3),xpi(1),xpi(2),ndiv,ier)
	    cpp = cpi(3)*(1+eps/2)
	    ier = 0
	    call ffzb0(cb02,a(1),a(2),cpp,cpi(1),cpi(2),
     +		xpi(3),xpi(1),xpi(2),ndiv,ier)
	    cdb0p = (cb02-cb01)/eps
	    print '(a,2g23.13,i6)','xp*b0''~ ',cdb0p,ier
	    ldot = ldotsa
	endif
	if ( ntens .ge. 1 ) then
	    call ffza0(ca0i(1),a(1),a(2),cpi(1),xpi(1),ndiv,ier)
	    call ffza0(ca0i(2),a(1),a(2),cpi(2),xpi(2),ndiv,ier)
	    iersav = ier
	    if ( k.gt.1 ) t1 = fftyd(1)
	    do 210 i=1,k
		ier = iersav
		call ffzb1(cb1,cb0,ca0i,cpi(3),cpi(1),cpi(2),cfpij2,
     +			xpi(3),xpi(1),xpi(2),fpij2,ier)
  210	    continue
	    if ( k.gt.1 ) t2 = fftyd(2)
	    print '(a,2g23.13,i6)','b1 = ',cb1,ier
	    if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'b1 = ',cb1,ier
	    if ( ntens.gt.1 ) then
		if ( k.gt.1 ) t1 = fftyd(1)
		iersav = ier
		do 220 i=1,k
		    ier = iersav
		    call ffzb2(cb2p,cb2d,cb1,cb0,ca0i,
     +			cpi(3),cpi(1),cpi(2),cfpij2,
     +			xpi(3),xpi(1),xpi(2),fpij2,ier)
  220		continue
		if ( k.gt.1 ) t2 = fftyd(2)
		ier1 = ier
		print '(a,2g23.13,i6)','b2p= ',cb2p,ier
		print '(a,2g23.13,i6)','b2d= ',cb2d,ier
		if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'b2p= ',cb2p,
     +			ier
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'b2d= ',cb2d,
     +			ier
		if ( k.gt.1 ) t1 = fftyd(1)
		do 230 i=1,k
		    ier = iersav
		    call ffzb2p(cb2i,cb1,cb0,ca0i,
     +			cpi(3),cpi(1),cpi(2),cfpij2,
     +			xpi(3),xpi(1),xpi(2),fpij2,ier)
  230		continue
		if ( k.gt.1 ) t2 = fftyd(2)
		ier = max(ier,ier1)
		print '(a,2g23.13,i6)','b21= ',cb2i(1),ier
		print '(a,2g23.13,i6)','b22= ',cb2i(2),ier
		if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'b21= ',cb2i(1),
     +			ier
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'b22= ',cb2i(2),
     +			ier
		rloss = xloss**2*10.d0**(-mod(ier,50))
		if ( cpi(3).ne.0 ) then
		    if ( .not.onshel .or. nschem.ge.7 ) then
			xnul = absc(cb2p-cb2d*(1/DBLE(cpi(3)))-cb2i(1))
			xmax = max(absc(cb2p),absc(cb2i(1)))
			if ( rloss*xnul .gt. precc*xmax ) print *,
     +			'fftest: error: B21 != Bp - Bd/xp: ',cb2i(1),
     +			cb2p,cb2d*(1/DBLE(cpi(3))),xnul
		    else
			xnul = absc(cb2p-cb2d*(1/DBLE(xpi(3)))-cb2i(1))
			xmax = max(absc(cb2p),absc(cb2i(1)))
			if ( rloss*xnul .gt. precc*xmax ) print *,
     +			'fftest: error: B21 != Bp - Bd/xp: ',cb2i(1),
     +			cb2p,cb2d*(1/DBLE(xpi(3))),xnul
		    endif
		endif
		xnul = absc(cb2d-cb2i(2))
		if ( rloss*xnul .gt. precc*absc(cb2i(2)) ) print *,
     +			'fftest: error: B22 != Bd: ',cb2i(2),cb2d,xnul
	    endif
	endif
	call ffwarn(998,ier,x0,x0)
*  #] two point function:
*  #[ three point function:
**************the three point function****************
	elseif (npoint .eq. 3 ) then
	if ( k.gt.1 ) t1 = fftyd(1)
	if ( lwrite ) write(*,'(a)')'   #[ C0:'
	do 300 i=1,k
	ier = 0
	call ffzc0(cc0,cpi,xpi,ndiv,ier)
  300	continue
	if ( k.gt.1 ) t2 = fftyd(2)
	if ( lwrite ) write(*,'(a)')'   #] C0:'
	print '(a,2g23.13,i6)','c0 = ',cc0,ier
	if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	if ( lcon ) write(icon,'(a,2g23.13,i6)')'c0 = ',cc0,ier
	if ( ntens.ge.1 ) then
	    if ( lwrite ) write(*,'(a)')'   #[ B0(1):'
	    ier0 = 0
	    call ffzb0(cb0i(1),a(1),a(2),cpi(4),cpi(1),cpi(2),
     +		xpi(4),xpi(1),xpi(2),ndiv,ier0)
	    if ( lwrite ) write(*,'(a)')'   #] B0(1): '
	    if ( lwrite ) write(*,'(a)')'   #[ B0(2):'
	    ier1 = 0
	    call ffzb0(cb0i(2),a(1),a(2),cpi(5),cpi(2),cpi(3),
     +		xpi(4),xpi(2),xpi(3),ndiv,ier1)
	    ier2 = 0
	    if ( lwrite ) write(*,'(a)')'   #] B0(2): '
	    if ( lwrite ) write(*,'(a)')'   #[ B0(3):'
	    call ffzb0(cb0i(3),a(1),a(2),cpi(6),cpi(3),cpi(1),
     +		xpi(4),xpi(3),xpi(1),ndiv,ier2)
	    if ( lwrite ) write(*,'(a)')'   #] B0(3): '
	    if ( lwrite ) write(*,'(a)')'   #[ C1:'
	    do 310 i=1,k
	    	ier = max(ier0,ier1,ier2)
	    	call ffzc1(cc1i,cc0,cb0i,cpi,cfpij3,fodel2,xpi,fpij3,
     +	    		fdel2,ier)
  310	    continue
	    if ( k.gt.1 ) t2 = fftyd(2)
	    if ( lwrite ) write(*,'(a)')'   #] C1:'
	    print '(a,2g23.13,i6)','c11= ',cc1i(1),ier
	    print '(a,2g23.13,i6)','c12= ',cc1i(2),ier
	    if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'c11= ',cc1i(1),ier
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'c12= ',cc1i(2),ier
	endif
	call ffwarn(998,ier,x0,x0)
	if ( lcon ) call flush(icon)
*  #] three point function:
*  #[ four point function:
**************the four point function*****************
	elseif (npoint .eq. 4 ) then
	ntime = 1
	print '(a,i2)','---#[ rotation 1: isgnal = ',isgnal
	if ( lcon ) write(icon,'(a,i2,a)')'rotation 1, isgnal = ',
     +		isgnal,':'
	if ( k.gt.1 ) t1 = fftyd(1)
	do 400 i=1,k
	    ier = 0
	    ialsav = isgnal
	    call ffzd0(cd0,cpi,xpi,ndiv,ier)
  400	continue
	if ( k.gt.1 ) t2 = fftyd(2)
	call ffwarn(998,ier,x0,x0)
	print '(a,i2)','---#] rotation 1: isgnal = ',isgnal
	print '(a,2g23.13,i6)','d0 = ',cd0,ier
	if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	if ( lcon ) write(icon,'(a,2g23.13,i6)')'d0 = ',cd0,ier
	if ( ntens.ge.1 ) then
	    do 404 imiss=1,4
	       	if ( lwrite ) write(*,'(a,i1,a)')'   #[ C0(',imiss,'):'
	    	do 401 i=1,6
	    	    xqi(i) = xpi(iinx(i,imiss))
  401	    	continue
	    	ier0 = 0
	    	call ffzc0(cc0i(imiss),cpi(i),xpi(i),ndiv,ier0)
	    	del2i(imiss) = fdel2
	    	odel2i(imiss) = fodel2
	    	ier = max(ier,ier0)
	    	if ( lwrite ) write(*,'(a,i1,a)')'   #] C0(',imiss,'): '
  404	    continue
	    if ( lwrite ) write(*,'(a)')'   #[ D1:'
	    do 405 i=1,k
	    	call ffzd1(cd1i,cd0,cc0i,cpi,cfpij4,fodel3,odel2i,
     +	    		xpi,fpij4,fdel3,del2i,ier)
  405	    continue
	    if ( k.gt.1 ) t2 = fftyd(2)
	    if ( lwrite ) write(*,'(a)')'   #] D1:'
	    print '(a,2g23.13,i6)','d11= ',cd1i(1),ier
	    print '(a,2g23.13,i6)','d12= ',cd1i(2),ier
	    print '(a,2g23.13,i6)','d13= ',cd1i(3),ier
	    if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'cd1= ',cd1i(1),ier
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'cd2= ',cd1i(2),ier
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'cd3= ',cd1i(3),ier
	endif
	if ( lcon ) call flush(icon)
	do 410 i=1,10
	    cqi(i) = cpi(i)
	    xqi(i) = xpi(i)
  410	continue
  420	continue
	do 440 irota=2,6
	    do 430 i=1,10
		cpi(inew(i,irota)) = cqi(i)
		xpi(inew(i,irota)) = xqi(i)
  430	    continue
	    print '(a,i1,a,i2)','---#[ rotation ',irota,': isgnal = ',
     +		isgnal
	    if ( lcon ) write(icon,'(a,i1,a,i2,a)')'rotation ',
     +		irota,', isgnal = ',isgnal,':'
	    ier = 0
	    id = id + 1
	    isgnal = ialsav
	    call ffzd0(cd0,cpi,xpi,ndiv,ier)
	    call ffwarn(998,ier,x0,x0)
	    print '(a,i1,a,i2)','---#] rotation ',irota,': isgnal = ',
     +		isgnal
	    print '(a,2g23.13,i6)','d0 = ',cd0,ier
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'d0 = ',cd0,ier
	    if ( lcon ) call flush(icon)
  440	continue
	k = 1
*	Check independence of root chosen
	isgnal = -sign(1,ial)
	ialsav = isgnal
	if ( ntime .eq. 1 ) then
	    ntime = 2
	    print '(a,i2)','---#[ rotation 1: isgnal = ',isgnal
	    if ( lcon ) write(icon,'(a,i2,a)')'rotation 1, isgnal = ',
     +		isgnal,':'
	    ier = 0
	    id = id + 1
	    call ffzd0(cd0,cqi,xqi,ndiv,ier)
	    call ffwarn(998,ier,x0,x0)
	    print '(a,i2)','---#] rotation 1: isgnal = ',isgnal
	    print '(a,2g23.13,i6)','d0 = ',cd0,ier
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'d0 = ',cd0,ier
	    if ( lcon ) call flush(icon)
	    goto 420
	endif
*  #] four point function:
*  #[ five point function:
**************the five point function*****************
	elseif ( npoint .eq. 5 ) then
	    ntime = 1
	    print '(a,i2)','---#[ rotation 1: isgnal = ',isgnal
	    if ( lcon ) write(icon,'(a,i2,a)')'rotation 1, isgnal = ',
     +		isgnal,':'
	    if ( k.gt.1 ) t1 = fftyd(1)
	    do 500 i=1,k
		ier = 0
		ialsav = isgnal
		call ffzei(cai,cbij,ccij,cdi,ce, a(1),a(2), cpi,xpi, 
     +			ndiv,ntens,ier)
		ce0 = ce(1)
  500	    continue
	    if ( k.gt.1 ) t2 = fftyd(2)
	    call ffwarn(998,ier,x0,x0)
	    print '(a,i2)','---#] rotation 1: isgnal = ',isgnal
	    print '(a,2g23.13,i6)','e0 = ',ce0,ier
	    if ( k.gt.1 ) print *,' in ',(t2-t1)/k,'sec'
	    if ( lcon ) write(icon,'(a,2g23.13,i6)')'e0 = ',
     +			ce0,ier
	    if ( lcon ) call flush(icon)
	    do 510 i=1,20
		xqi(i) = xpi(i)
		cqi(i) = cpi(i)
  510	    continue
  520	    continue
	    do 540 irota=2,12
		do 530 i=1,20
		    if ( inew5(i,irota) .le. 0 ) then
			xpi(-inew5(i,irota)) = 0
			cpi(-inew5(i,irota)) = 0
		    else
			xpi(inew5(i,irota)) = xqi(i)
			cpi(inew5(i,irota)) = cqi(i)
		    endif
  530		continue
		print '(a,i2,a,i2)','---#[ rotation ',irota,
     +			': isgnal = ',isgnal
		if ( lcon ) write(icon,'(a,i2,a,i2,a)')'rotation ',
     +			irota,', isgnal = ',isgnal,':'
		ier = 0
		id = id + 1
		isgnal = ialsav
		call ffze0(ce0,cd0i,cpi,xpi,ndiv,ier)
		call ffwarn(998,ier,x0,x0)
		print '(a,i2,a,i2)','---#] rotation ',irota,
     +			': isgnal = ',isgnal
		print '(a,2g23.13,i6)','e0 = ',ce0,ier
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'e0 = ',
     +			ce0,ier
		if ( lcon ) call flush(icon)
  540	    continue
	    k = 1
*	    Check independence of root chosen
	    isgnal = -sign(1,ial)
	    ialsav = isgnal
	    if ( ntime .eq. 1 ) then
		ntime = 2
		print '(a,i2)','---#[ rotation 1: isgnal = ',isgnal
		if ( lcon ) write(icon,'(a,i2,a)')
     +			'rotation 1, isgnal = ',isgnal,':'
		ier = 0
		id = id + 1
		call ffze0(ce0,cd0i,cqi,xqi,ndiv,ier)
		call ffwarn(998,ier,x0,x0)
		print '(a,i2)','---#] rotation 1: isgnal = ',isgnal
		print '(a,2g23.13,i6)','e0 = ',ce0,ier
		if ( lcon ) write(icon,'(a,2g23.13,i6)')'e0 = ',
     +			ce0,ier
		if ( lcon ) call flush(icon)
		goto 520
	    endif
*  #] five point function:
*  #[ exit:
	else
	    print*,'error: npoint=',npoint,' not yet implemented'
	    return
	endif
***************************************************
	call ffexi
	if ( npoint .eq. 1 ) goto 5
	print '(3a)','####] ',text,':'
	goto 1
  999	close(ifile)
*  #] exit:
*###] fftest:
	end
