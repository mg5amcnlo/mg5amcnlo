*	$Id: ffxf0.f,v 1.5 1996/02/07 09:28:49 gj Exp $
*###[ ffxf0:
	subroutine ffxf0(cf0,ce0i,cd0ij,xpi,ier)
***#[*comment:***********************************************************
*									*
*	calculate							*
*									*
*	      1  /   /						     \-1*
*	f0= -----\dq |(q^2-m_1^2)((q+p_1)^2-m_2^2)...((q-p_6)^2-m_6^2|	*
*	    ipi^2/   \						     /	*
*									*
*	following the six five-point-function method in ....		*
*	As an extra the ten fourpoint functions Dij are also returned	*
*	plus the six fivepoint functions Ei.				*
*									*
*	Input:	xpi = m_i^2	(real)	  i=1,6				*
*		xpi = p_i.p_i	(real)	  i=7,12 (note: B&D metric)	*
*		xpi = (p_i+p_{i+1})^2 (r) i=13,18			*
*		xpi = (p_i+p_{i+1}+p_{i+3})^2 (r)  i=19,21		*
*									*
*	Output:	cf0		(complex) F0				*
*		ce0i(6)		(complex) E0 with s_i missing		*
*		cd0ij(6,6)	(complex) D0 with s_i and s_j missing	*
*		ier		(integer) no of digits lost, >50 error	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	DOUBLE PRECISION xpi(21)
	DOUBLE COMPLEX cf0,ce0i(6),cd0ij(6,6)
	integer ier
*
*	local variables
*
	integer i,j,ier0
	DOUBLE PRECISION dpipj(21,21)
*
*	common blocks:
*
	include 'ff.h'
*
*  #] declarations: 
*  #[ get differences:
*
*	simulate the differences in the masses etc..
*
	if ( lwrite ) then
	    print *,'ffxf0: input xpi: '
	    print '(i3,e24.16)',(i,xpi(i),i=1,21)
	endif
*
*	no redundant input yet (may be necessary)
*
*
*	the differences
*
	ier0 = 0
	if ( lwarn ) then
	    do 20 i=1,21
		dpipj(i,i) = 0
		do 10 j=1,i-1
		    dpipj(j,i) = xpi(j) - xpi(i)
		    dpipj(i,j) = -dpipj(j,i)
		    if ( abs(dpipj(j,i)) .lt. xloss*abs(xpi(i))
     +			.and. xpi(i) .ne. xpi(j) ) then
			call ffwarn(193,ier0,dpipj(j,i),xpi(i))
			if ( lwrite ) print *,'between xpi(',i,
     +				') and xpi(',j,')'
		    endif
   10		continue
   20	    continue
	else
	    do 40 i=1,21
		do 30 j=1,21
		    dpipj(j,i) = xpi(j) - xpi(i)
   30		continue
   40	    continue
	endif
*  #] get differences: 
*  #[ call ffxf0a:
	call ffxf0a(cf0,ce0i,cd0ij,xpi,dpipj,ier)
*  #] call ffxf0a: 
*###] ffxf0: 
	end
*###[ ffxf0a:
	subroutine ffxf0a(cf0,ce0i,cd0ij,xpi,dpipj,ier)
***#[*comment:***********************************************************
*									*
*	calculate							*
*									*
*	      1  /   /						     \-1*
*	f0= -----\dq |(q^2-m_1^2)((q+p_1)^2-m_2^2)...((q-p_5)^2-m_5^2|	*
*	    ipi^2/   \						     /	*
*									*
*	following the five four-point-function method in ....		*
*	As an extra the five fourpoint function Di are also reurned	*
*									*
*	Input:	xpi = m_i^2	   (real)  i=1,6			*
*		xpi = p_i.p_i	   (real)  i=7,12 (note: B&D metric)	*
*		xpi = (p_i+p_{i+1})^2 (r)  i=13,18			*
*		xpi = (p_i+p_{i+1}+p_{i+2})^2 (r)  i=19,21		*
*		dpipj(21,21)	   (real)  = pi(i) - pi(j)		*
*									*
*	Output:	cf0		  (complex)				*
*		ce0i(6)		  (complex) E0 with s_i missing		*
*		cd0ij(6,6)	  (complex) D0 with s_i,s_j missing	*
*		ier		  (integer) <50:lost # digits 100=error	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX cf0,ce0i(6),cd0ij(6,6)
	DOUBLE PRECISION xpi(21),dpipj(21,21)
*
*	local variables
*
	integer i,j,k,l,m,ii(10),ier2,ier1,ier0,irota,itype,ndiv,idum,
     +		idone,ii4(6),is
	logical lwsav,ldel2s,lwhich
	DOUBLE COMPLEX c,cfac,cs,cd0i(5),csum,csi(7)
	DOUBLE PRECISION del6,xpi4(13),dpipj4(10,13),piDpj4(10,10),
     +		absc,xmax,piDpj(21,21),xqi4(13),dqiqj4(10,13),
     +		qiDqj4(10,10),del2s,xmx4(6,6),dl4rij(6,6),xpi5(20),
     +		dpipj5(15,20),piDpj5(15,15),dl4ri(5),dl5ri(6),xlosn, 
     +		d5sp,dl4q(6),psum
	save ii4
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data ii4 /5,6,7,8,9,10/
*
*  #] declarations: 
*  #[ initialisations:
*
	ndiv = 0
	idsub = 0
	cf0 = 0
	do 2 i=1,6
	    ce0i(i) = 0
	    do 1 j=1,6
		cd0ij(i,j) = 0
    1	    continue
    2	continue
*
*  #] initialisations: 
*  #[ get dot products:
*
	idsub = idsub + 1
	call ffdot6(piDpj,xpi,dpipj,ier)
	if ( ldot ) then
	    do 6 i=1,21
		do 5 j=1,21
		    fpij6(j,i) = piDpj(j,i)
    5		continue
    6	    continue
	continue
	endif
	if ( ltest ) then
	    ii(1) = 7
	    ii(2) = 8
	    ii(3) = 9
	    ii(4) = 10
	    ii(5) = 11
	    call ffdl5p(xpi,piDpj,21,ii,ier)
	endif
	if ( lwrite ) print *,'After dotproducts ier = ',ier
*
*  #] get dot products: 
*  #[ five and four point stuff:
*
	ier2 = ier
	do 100 i=1,6
*
*	    get the five-point momenta
*
	    ier1 = ier
	    call ffpi65(xpi5,dpipj5,piDpj5,xpi,dpipj,piDpj,i,ier1)
*
*	    get fourpoint functions
*
	    do 90 k=1,5
		j=k
		if ( lwrite ) print '(a,2i2,a)',
     +		    '####[ ffxf0a: fourpoint function nr ',i,j+1,': '
		if ( k.lt.i ) then
*		    we already have it
		else
		    j = j+1
		    ier0 = ier
*
*		    get four-point momenta
*
		    call ffpi54(xpi4,dpipj4,piDpj4,xpi5,dpipj5,piDpj5,k,
     +			ier0)
		    if ( ltest ) then
			idum = ier
			call ffpi64(xqi4,dqiqj4,qiDqj4,xpi,dpipj,piDpj,
     +				i,j,idum)
			xlosn = xloss*DBLE(10)**(-mod(ier0,50))
			do 12 l=1,13
			    if ( xlosn*abs(xpi4(l)-xqi4(l)).gt.precx*abs
     +			      (xpi4(l)) ) print*,'ffxf0a: error: xpi4(',
     +			      l,') != xqi4(',l,'): ',xpi4(l),xqi4(l)
			    do 11 m=1,10
				if ( xlosn*abs(dpipj4(m,l)-dqiqj4(m,l))
     +				    .gt.precx*abs(xpi4(l)) ) print *,
     +				    'ffxf0a: error: dpipj4(',m,l,') !=',
     +				    ' dqiqj4(',m,l,'): ',dpipj4(m,l),
     +				    dqiqj4(m,l),dpipj4(m,l)-dqiqj4(m,l)
   11			    continue
   12			continue
			do 14 l=1,10
			    do 13 m=1,10
				if ( piDpj4(m,l).ne.qiDqj4(m,l) ) print
     +				    *,'ffxf0a: error: piDpj4(',m,l,
     +				    ') != qiDqj4(',m,l,'): ',piDpj4(m,
     +				    l),qiDqj4(m,l)
   13			    continue
   14			continue
		    endif
		    ier1 = ier0
		    call ffxdir(cs,cfac,idone,xpi4,dpipj4,6,ndiv,ier1)
		    if ( idone .gt. 0 ) then
*			done
			xmax = abs(cs)*10d0**(-mod((ier1-ier0),50))
		    else
			ier1 = ier0
*
*			rotate to calculable posistion
*
			call ffrot4(irota,del2s,xqi4,dqiqj4,qiDqj4,xpi4,
     +				dpipj4,piDpj4,5,itype,ier0)
			if ( itype .lt. 0 ) then
			    print *,'ffxf0:  error:  Cannot handle '//
     +				'this 4point masscombination yet:'
			    print *,(xpi(j),j=1,20)
			    return
			endif
			if ( itype .eq. 1 ) then
			    ldel2s = .TRUE.
			    isgnal = +1
			    print *,'ffxf0a: Cannot handle del2s=0 yet'
			    stop
			else
			    ldel2s = .FALSE.
			endif
			if ( itype .eq. 2 ) then
			    print *,'ffxf0a: Cannot handle doubly IR ',
     +				'divergent function yet'
			    stop
			endif
*
*			get fourpoint function
*
			if ( lwrite ) then
			    print *,'xpi for ffxd0e: '
			    print '(i3,e24.16)',(m,xqi4(m),m=1,13)
			endif
			lwsav = lwrite
			lwrite = .FALSE.
			idsub = idsub + 1
			call ffxd0e(cs,cfac,xmax, .FALSE.,ndiv,xqi4,
     +				dqiqj4,qiDqj4,del2s,ldel2s,ier0)
			lwrite = lwsav
			ier1 = max(ier1,ier0)
		    endif
		    cd0ij(i,j) = cs*cfac
		    cd0ij(j,i) = cd0ij(i,j)
		    xmx4(i,j) = xmax*absc(cfac)
		    xmx4(j,i) = xmx4(i,j)
		    if ( ldot ) then
			call ffdl3p(fdl3ij(i,j),qiDqj4,10,ii4,ii4,
     +				ier0)
			fdl3ij(j,i) = fdl3ij(i,j)
			fd4sij(i,j) = fdel4s
			fd4sij(j,i) = fdel4s
*			let's check that these have been set by ffxd0e...
			if ( ltest ) then
			    ier0 = 0
			    call ffdel4(fdel4s,xpi4,piDpj4,10,ier0)
			    if ( xloss*10d0**(-ier0-1)*abs(fd4sij(i,j)-
     +			    	    fdel4s).gt.precx*abs(fdel4s) ) then
			    	print *,'ffxf0a: error: Del4s was not'//
     +			    		' correct',fd4sij(i,j),fdel4s,
     +			    		fd4sij(i,j)-fdel4s,ier0
			    endif
			endif
		    endif
		endif
*
*		get the coefficient determinant (not symmetric!)
*
		idsub = idsub + 1
		ier0 = ier
		call ffdl4r(dl4rij(i,j),xpi5,piDpj5,15,k,ier0)
		ier1 = max(ier1,ier0)
*
*		and fill the five-point linear arrays
*
		cd0i(k) = cd0ij(i,j)
		dl4ri(k) = dl4rij(i,j)
		if ( lwrite ) then
		    print '(a,2i2,a)',
     +			'####] ffxf0a: fourpoint function nr ',i,j,': '
		    print *,'dl4rij(',i,j,') = ',dl4rij(i,j)
		    print *,'cd0ij(',i,j,')  = ',cd0ij(i,j),xmx4(i,j),
     +			ier0
		endif
   90	    continue
*
*	    call ffxe00
*
	    if ( lwrite ) print '(a,i2,a)',
     +		'####[ ffxf0a: fivepoint function nr ',i,': '
	    call ffxe00(ce0i(i),cd0i,dl4ri,xpi5,piDpj5,ier1)
	    if ( lwrite ) print '(a,i2,a)',
     +		'####] ffxf0a: fivepoint function nr ',i,': '
	    if ( lwrite ) print *,'ce0i(',i,') = ',ce0i(i),ier1
	    if ( ldot ) fdl4i(i) = fdel4
	    ier2 = max(ier2,ier1)
  100	continue
	ier = ier2
	if ( lwrite ) print *,'after E0s ier = ',ier
*
*  #] five and four point stuff: 
*  #[ six point stuff:
*
	ier1 = 0
	call ffdel6(del6,xpi,piDpj,21,ier1)
	csum = 0
	xmax = 0
	do i=1,6
	    if ( ce0i(i) .ne. 0 ) then
		ier0 = 0
		call ffdl5r(dl5ri(i),xpi,piDpj,21,i,ier0)
		csum = csum + DBLE(dl5ri(i))*ce0i(i)
		xmax = max(xmax,absc(csum))
		ier1 = max(ier1,ier0)
	    endif
	enddo
	ier = max(ier,ier1)
*
*	Check for cancellations in the final adding up
*
	if ( lwarn .and. 2*absc(csum) .lt. xloss*xmax )
     +		call ffwarn(191,ier,absc(csum),xmax)
*
*	Check for a sum close to the minimum of the range (underflow
*	problems)
*
	if ( lwarn .and. absc(csum).lt.xalogm/precc .and. csum.ne.0 )
     +		call ffwarn(192,ier,absc(csum),xalogm/precc)
*
*	If the imaginary part is very small it most likely is zero
*	(can be removed, just esthetically more pleasing)
*
	if ( abs(DIMAG(csum)) .lt. precc*abs(DBLE(csum)) )
     +		csum = DCMPLX(DBLE(csum))
*
*	Finally ...
*
	cf0 = csum*DBLE(-1/(2*del6))
*
*  #] six point stuff:
*  #[ print output:
	if ( lwrite ) then
	    print *,'ffxf0a: cf0 = ',cf0,ier
	endif
	if ( .FALSE. ) then
	    do i=1,6
		if ( xpi(i).eq.0 ) then
*		    assume it's a photon
		    psum = 0
		    do j=1,5
			k = i+j-1
			if ( k.gt.6 ) k = k-6
			psum = psum + xpi(k+6)
			do l=1,j-1
			    m = i+l-1
			    if ( m.gt.6 ) m = m-6
			    psum = psum + 2*piDpj(k+6,m+6)
			enddo
			k = i+j
			if ( k.gt.6 ) k = k-6
			if ( abs(piDpj(i,k)).gt.xloss*abs(xpi(k)) ) then
			    print *,'ratio coeffs ',k,' is ',dl5ri(k)/
     +				(-2*del6)
			    print *,'propagator   ',k,' is ',1/(psum-xpi
     +				(k))
			endif
		    enddo
		endif
	    enddo
	endif
*  #] print output:
*###] ffxf0a:
	end
*###[ ffxf0r:
	subroutine ffxf0r(cf0,ce0i,cd0ij,xpi,ier)
***#[*comment:***********************************************************
*									*
*	Tries all 12 easy permutations of the 6pointfunction		*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
	integer ier,nrot
	parameter(nrot=12)
	DOUBLE PRECISION xpi(21),xqi(21)
	DOUBLE COMPLEX cf0,ce0i(6),cd0ij(6,6),cf0p,ce0ip(6),cd0ijp(6,6),
     +		ce0iq(6),cd0ijq(6,6)
	integer inew(21,nrot),irota,ier1,i,j,k,l,icon,ialsav,init
	parameter(icon=3)
	logical lcon
	save inew,init,lcon
	include 'ff.h'
	data inew
     +	/1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18, 19,20,21,
     +	 2,3,4,5,6,1, 8,9,10,11,12,7, 14,15,16,17,18,13, 20,21,19,
     +	 3,4,5,6,1,2, 9,10,11,12,7,8, 15,16,17,18,13,14, 21,19,20,
     +	 4,5,6,1,2,3, 10,11,12,7,8,9, 16,17,18,13,14,15, 19,20,21,
     +	 5,6,1,2,3,4, 11,12,7,8,9,10, 17,18,13,14,15,16, 20,21,19,
     +	 6,1,2,3,4,5, 12,7,8,9,10,11, 18,13,14,15,16,17, 21,19,20,
     +	 6,5,4,3,2,1, 11,10,9,8,7,12, 16,15,14,13,18,17, 21,20,19,
     +	 5,4,3,2,1,6, 10,9,8,7,12,11, 15,14,13,18,17,16, 20,19,21,
     +	 4,3,2,1,6,5, 9,8,7,12,11,10, 14,13,18,17,16,15, 19,21,20,
     +	 3,2,1,6,5,4, 8,7,12,11,10,9, 13,18,17,16,15,14, 21,20,19,
     +	 2,1,6,5,4,3, 7,12,11,10,9,8, 18,17,16,15,14,13, 20,19,21,
     +	 1,6,5,4,3,2, 12,11,10,9,8,7, 17,16,15,14,13,18, 19,21,20/
	data init /0/
*  #] declarations: 
*  #[ open console for some activity on screen:
	if ( init .eq. 0 ) then
	    init = 1
	    if ( lwrite ) then
		open(icon,file='CON:',status='old',err=11)
		lcon = .TRUE.
		goto 13
	    endif
   11	    continue
	    lcon = .FALSE.
   13	    continue
	endif
*  #] open console for some activity on screen: 
*  #[ calculations:
	cf0 = 0
	ier = 999
	ialsav = isgnal
	    do 20 irota=1,nrot
		do 10 i=1,21
		    xqi(inew(i,irota)) = xpi(i)
   10		continue
		print '(a,i2,a,i2)','---#[ rotation ',irota,': isgnal ',
     +			isgnal
		if (lcon) write(icon,'(a,i2,a,i2)')'rotation ',irota,',
     +			isgnal ',isgnal
		ier1 = 0
		ner = 0
		id = id + 1
		isgnal = ialsav
		call ffxf0(cf0p,ce0ip,cd0ijp,xqi,ier1)
		ier1 = ier1 + ner
		if ( ier.gt.5 ) call ffwarn(998,ier,x0,x0)
		print '(a,i2,a,i2)','---#] rotation ',irota,': isgnal ',
     +			isgnal
		print '(a,2g28.16,i3)','f0 = ',cf0p,ier1
		do 15 k=1,6
		    ce0iq(k) = ce0ip(inew(k,irota))
		    print '(a,2g28.16,i3)','e0 = ',ce0iq(k),k
   15		continue
		do 17 k=1,6
		    do 16 l=k+1,6
			cd0ijq(l,k)=cd0ijp(inew(l,irota),inew(k,irota))
			print '(a,2g28.16,2i3)','d0 = ',cd0ijq(l,k),l,k
   16		    continue
   17		continue
		if (lcon) write(icon,'(a,2g28.16,i3)')'f0 = ',cf0p,ier1
		if ( ier1 .lt. ier ) then
		    cf0 = cf0p
		    do 19 k=1,6
			ce0i(k) = ce0iq(k)
			do 18 l=k+1,6
			    cd0ij(l,k) =
     +				cd0ijp(inew(l,irota),inew(k,irota))
			    cd0ij(k,l) = cd0ij(l,k)
   18			continue
   19		    continue
		    ier = ier1
		endif
   20	    continue
*  #] calculations: 
*###] ffxf0r: 
	end
