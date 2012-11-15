*###[ ffxdi:
	subroutine ffxdi(cd4pppp,cd4ppdel,cd4deldel, cd3ppp,cd3pdel,
     +		cd2pp,cd2del, cd1p, dl2pij, cd0,cc0i,cb0ij,ca0i,
     +		del4s,del3p,del2pi, xpi,piDpj, d0,xmu, degree, ier)
***#[*comment:***********************************************************
*									*
*	Compute the tensor functions D1-D(degree) in the determinant	*
*	scheme, i.e. with basis p1-p3 and (instead of d_(mu,nu))	*
*	\delta_{p1 p2 p3 mu}^{p1 p2 p3 nu}.				*
*									*
*	Input:	cd0	   (complex)	D0				*
*		cc0i(4)	   (complex)	C0 with Ni=(Q+..)^2-mi^2 missing*
*		cb0ij(4,4) (complex)	B0 _with_ Ni,Nj	(only for	*
*					degree>1)			*
*		ca0i(4)	   (complex)	A0 with Ni (only for degree>2)	*
*		del4s	   (real)	delta(s1,s2,s3,s4)(s1,s2,s3,s4)	*
*					(only needed when degree>1)	*
*		del3p	   (real)	delta(p1,p2,p3,p1,p2,p3)	*
*		del2pi(4)  (real)	delta(pipj)(pi,pj) belonging to	*
*					cc0i(i)				*
*		xpi(13)	   (real)	1-4: mi^2, 5-10: p(i-4)^2	*
*		piDpj(10,10) (re)	pi.pj				*
*		d0	   (real)	\ renormalization constants	*
*		xmu	   (real)       / used in B0, A0		*
*		degree	   (integer)	1-4				*
*		ier	   (integer)	number of unreliable digits in	*
*					input				*
*									*
*	Output:	ier			number of digits lost in the	*
*					least stable result		*
*		dl2pij(6,6)(real)	determinants delta(pi,pj,pk,pl)	*
*		cd1p(3)	   (complex)	coeffs of p1,p2,p3		*
*	   only when degree>1:						*
*		cd2pp(3,3) (complex)	coeffs of p1p1,(p1p2+p2p1),...	*
*		cd2del	   (complex)	coeff of delta(p1,p2,p3,mu,..)	*
*	   only when degree>2:						*
*		cd3ppp(3,3,3)(compl)	coeffs of p1p1p1,p1(p1p2+p2p1),	*
*					(p1p2p3+p1p3p2+p2p1p3+p2p3p1+..)*
*		cd3pdel(3) (complex)	coeffs of pidel (symmetrized)	*
*	   only when degree>3:						*
*		cd4pppp(3,3,3,3)(co)	you guessed it!			*
*		cd4ppdel(3,3)(compl)					*
*		cd4deldel  (complex)					*
*									*
*	Note: at this moment (28-feb-1993) only D1 and D2 are coded.	*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer degree,ier
	DOUBLE PRECISION dl2pij(6,6),del4s,del3p,del2pi(4),xpi(13),
     +		piDpj(10,10),d0,xmu
	DOUBLE COMPLEX cd4pppp(3,3,3,3),cd4ppdel(3,3),cd4deldel,
     +		cd3ppp(3,3,3),cd3pdel(3),cd2pp(3,3),cd2del,
     +		cd1p(3),cd0,cc0i(4),cb0ij(4,4),ca0i(4)
*
*	local variables
*
	integer i,j,k,ier0,ier1,ier2,inx43(6,4),sgn43(6,4),i2p(5:8,5:8),
     +		isgnsa,ii4(6)
	logical lsave1,lsave2
	DOUBLE PRECISION a,xpi3(6),xlosn,dl3qi(7),xmax,vgl,xnul
	DOUBLE COMPLEX cc,cs(25),cnul
	save inx43,sgn43,i2p,ii4
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data inx43 /2,3,4,6,7,10,
     +		    1,3,4,9,7,8,
     +		    1,2,4,5,10,8,
     +		    1,2,3,5,6,9/
	data sgn43 /+1,+1,+1,+1,+1,-1,
     +		    +1,+1,+1,-1,+1,+1,
     +		    +1,+1,+1,+1,+1,+1,
     +		    +1,+1,+1,+1,+1,+1/
	data i2p /0,0,0,0,
     +		  1,0,0,0,
     +		  2,4,0,0,
     +		  3,5,6,0/
	data ii4 /5,6,7,8,9,10/
*
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffxdi: input:'
	    print *,'  degree ',degree
	    print *,'  xpi  = ',xpi
	    print *,'  ier  = ',ier
	endif
	if ( degree .gt. 2 ) then
	    print *,'ffxdi: degree > 2 not yet supported: ',degree
	    stop
	endif
	if ( del2pi(1).eq.0 .or. del2pi(2).eq.0 .or. del2pi(3).eq.0
     +		.or. del2pi(4).eq.0 ) then
	    call fferr(87,ier)
	    return
	endif
	if ( ltest ) then
*
*	    the D0
*
	    ier0 = ier
	    lsave1 = ldot
	    lsave2 = lwrite
	    ldot = .TRUE.
	    lwrite = .FALSE.
	    isgnsa = isgnal
	    call ffxd0(cc,xpi,ier0)
	    isgnal = isgnsa
	    ldot = lsave1
	    lwrite = lsave2
	    xlosn = xloss*DBLE(10)**(-mod(ier0,50))
	    if ( xlosn*abs(cc-cd0) .gt. precc*abs(cd0) ) print *,
     +		'ffxdi: error: input D0 disagrees with recomputed: ',
     +		cd0,cc,cd0-cc,ier,ier0
	    if ( xlosn*abs(del3p-fdel3) .gt. precx*abs(del3p) ) print *,
     +		'ffxdi: error: input del3p disagrees with recomputed: ',
     +		del3p,fdel3,del3p-fdel3,ier,ier0
	    if ( xlosn*abs(del4s-fdel4s) .gt. precx*abs(del4s) ) print*,
     +		'ffxdi: error: input del4s disagrees with recomputed: ',
     +		del4s,fdel4s,del4s-fdel4s,ier,ier0
	    do 20 i=1,10
		do 10 j=1,10
		    if ( xlosn*abs(piDpj(j,i)-fpij4(j,i)) .gt. precx*
     +			abs(piDpj(j,i)) ) print *,'ffxdi: error: input '
     +			,'piDpj(',j,i,') disagrees with recomputed: ',
     +			piDpj(j,i),fpij4(j,i),piDpj(j,i)-fpij4(j,i)
   10		continue
   20	    continue
*
*	    the C0s
*
	    do 40 i=1,4
		do 30 j=1,6
		    xpi3(j) = xpi(inx43(j,i))
   30		continue
		if ( idot.gt.0 ) then
		    do 36 j=1,6
*			distribute dotproducts
			do 35 k=1,6
			    fpij3(k,j) = fpij4(inx43(k,i),inx43(j,i))*
     +				sgn43(k,i)*sgn43(j,i)
   35			continue
   36		    continue
		endif
		ier0 = ier
		lsave1 = ldot
		lsave2 = lwrite
		ldot = .TRUE.
		lwrite = .FALSE.
		call ffxc0(cc,xpi3,ier0)
		isgnal = isgnsa
		ldot = lsave1
		lwrite = lsave2
		xlosn = xloss*DBLE(10)**(-mod(ier0,50))
		if ( xlosn*abs(cc-cc0i(i)) .gt. precc*abs(cc0i(i)) )
     +		    print *,'ffxdi: error: input C0(',i,') disagrees ',
     +		    'with recomputed: ',cc0i(i),cc,cc0i(i)-cc,ier,ier0
		if ( xlosn*abs(del2pi(i)-fdel2) .gt. precx*abs(del2pi(i)
     +		    ) ) print *,'ffxdi: error: input del2pi(',i,
     +		    ') disagrees with recomputed: ',del2pi(i),fdel2,
     +		    del2pi(i)-fdel2
   40	    continue
*
*	    the B0s
*
	    if ( degree .lt. 2 ) goto 80
	    do 60 i=1,3
		do 50 j=i+1,4
		    ier0 = ier
		    lsave2 = lwrite
		    lwrite = .FALSE.
		    call ffxb0(cc,d0,xmu,xpi(inx(i,j)),xpi(i),xpi(j),
     +			ier0)
		    lwrite = lsave2
		    xlosn = xloss*DBLE(10)**(-mod(ier0,50))
		    if ( cb0ij(i,j) .ne. cb0ij(j,i) ) print *,
     +			'ffxdi: error: cb0ij(',i,j,') != cb0ij(',j,i,
     +			') : ',cb0ij(i,j),cb0ij(j,i)
		    if ( xlosn*abs(cc-cb0ij(i,j)) .gt. precc*abs(cb0ij(i
     +			,j)) ) print *,'ffxdi: error: input B0(',i,j,
     +			') disagrees with recomputed: ',cb0ij(i,j),cc,
     +			cb0ij(i,j)-cc,ier,ier0
   50		continue
   60	    continue
*
*	    the A0s
*
	    if ( degree .lt. 3 ) goto 80
	    do 70 i=1,4
		ier0 = ier
		lsave2 = lwrite
		lwrite = .FALSE.
		call ffxa0(cc,d0,xmu,xpi(i),ier0)
		lwrite = lsave2
		xlosn = xloss*DBLE(10)**(-mod(ier0,50))
		if ( xlosn*abs(cc-ca0i(i)) .gt. precc*abs(ca0i(i)) )
     +		    print *,'ffxdi: error: input A0(',i,') disagrees ',
     +		    'with recomputed: ',ca0i(i),cc,ca0i(i)-cc,ier,ier0
   70	    continue
   80	    continue
	endif
	if ( .not.ltest ) then
*	    to check when called from ffzfi, ffzei
	    do i=1,10
	    	xnul = piDpj(i,5) + piDpj(i,6) + piDpj(i,9)
	    	xmax = max(abs(piDpj(i,6)),abs(piDpj(i,9)))
	    	if ( xloss*abs(xnul).gt.precx*xmax ) then
	    	    print *,'ffxdi: error: i569 does not add up to 0: ',
     +	    	    	i,piDpj(i,5),piDpj(i,6),piDpj(i,9),xnul,ier
	    	endif
	    	xnul = piDpj(i,6) + piDpj(i,7) - piDpj(i,10)
	    	xmax = max(abs(piDpj(i,7)),abs(piDpj(i,10)))
	    	if ( xloss*abs(xnul).gt.precx*xmax ) then
	    	    print *,'ffxdi: error: i670 does not add up to 0: ',
     +	    	    	i,piDpj(i,6),piDpj(i,7),piDpj(i,10),xnul,ier
	    	endif
	    	xnul = piDpj(i,7) + piDpj(i,8) - piDpj(i,9)
	    	xmax = max(abs(piDpj(i,8)),abs(piDpj(i,9)))
	    	if ( xloss*abs(xnul).gt.precx*xmax ) then
	    	    print *,'ffxdi: error: i789 does not add up to 0: ',
     +	    	    	i,piDpj(i,7),piDpj(i,8),piDpj(i,9),xnul,ier
	    	endif
	    	xnul = piDpj(i,8) + piDpj(i,5) + piDpj(i,10)
	    	xmax = max(abs(piDpj(i,5)),abs(piDpj(i,10)))
	    	if ( xloss*abs(xnul).gt.precx*xmax ) then
	    	    print *,'ffxdi: error: i850 does not add up to 0: ',
     +	    	    	i,piDpj(i,8),piDpj(i,5),piDpj(i,10),xnul,ier
	    	endif
	    enddo
	    ier0 = ier
	    call ffdl3p(vgl,piDpj,10,ii4,ii4,ier0)
	    xlosn = xloss*DBLE(10)**(-mod(ier0,50))
	    if ( xlosn*abs(del3p-vgl).gt.precx*abs(vgl) ) then
		print *,'ffxdi: error: input del3p disagrees with '//
     +		'recomputed: ',del3p,vgl,del3p-vgl,ier,ier0
	    endif
	    do i=1,4
		ier0 = ier
		call ffdel2(vgl,piDpj,10,inx43(4,i),inx43(5,i),
     +			inx43(6,i),0,ier0)
		xlosn = xloss*DBLE(10)**(-mod(ier0,50))
		if ( xlosn*abs(del2pi(i)-vgl).gt.precx*abs(vgl) ) then
		    print *,'ffxdi: error: input del2pi(',i,
     +			') disagrees with recomputed: ',del2pi(i),vgl,
     +			del2pi(i)-vgl,ier,ier0
		endif
	    enddo
	endif
	if ( degree .le. 0 ) then
	    if ( ltest ) print *,'ffxdi: rather useless call to ffxdi'
	    return
	endif
*  #] check input:
*  #[ preliminaries:
*	not needed?  security first!
	if ( lwrite ) then
	    print *,'i2p(5,6) = ',i2p(5,6)
	    print *,'i2p(6,7) = ',i2p(6,7)
	    print *,'i2p(7,8) = ',i2p(7,8)
	    print *,'i2p(5,8) = ',i2p(5,8)
	endif
	dl2pij(i2p(5,6),i2p(5,6)) = del2pi(4)
	dl2pij(i2p(6,7),i2p(6,7)) = del2pi(1)
	dl2pij(i2p(7,8),i2p(7,8)) = del2pi(2)
	dl2pij(i2p(5,8),i2p(5,8)) = del2pi(3)
*  #] preliminaries:
*  #[ get determinants:
*
	ier1 = ier
	call ffdl2i(dl2pij(i2p(6,7),i2p(7,8)),piDpj,10,
     +		6,7,10,+1,7,8,9,+1,ier1)
	dl2pij(i2p(7,8),i2p(6,7)) = dl2pij(i2p(6,7),i2p(7,8))
*
	ier0 = ier
	call ffdl2i(dl2pij(i2p(5,8),i2p(6,7)),piDpj,10,
     +		6,7,10,+1,5,8,10,-1,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(6,7),i2p(5,8)) = dl2pij(i2p(5,8),i2p(6,7))
*
	ier0 = ier
	call ffdl2i(dl2pij(i2p(5,6),i2p(6,7)),piDpj,10,
     +		6,7,10,+1,5,6,9,-1,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(6,7),i2p(5,6)) = dl2pij(i2p(5,6),i2p(6,7))
*
	ier0 = ier
	call ffdl2t(dl2pij(i2p(5,7),i2p(6,7)),piDpj,5,7,
     +		6,7,10,-1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(6,7),i2p(5,7)) = dl2pij(i2p(5,7),i2p(6,7))
*
	ier0 = ier
	call ffdl2t(dl2pij(i2p(5,7),i2p(7,8)),piDpj,5,7,
     +		7,8,9,-1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(7,8),i2p(5,7)) = dl2pij(i2p(5,7),i2p(7,8))
*
	ier0 = ier
	call ffdl2t(dl2pij(i2p(5,7),i2p(5,8)),piDpj,5,7,
     +		5,8,10,+1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(5,8),i2p(5,7)) = dl2pij(i2p(5,7),i2p(5,8))
*
	ier0 = ier
	call ffdl2t(dl2pij(i2p(5,6),i2p(5,7)),piDpj,5,7,
     +		5,6,9,+1,-1, 10,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(5,7),i2p(5,6)) = dl2pij(i2p(5,6),i2p(5,7))
*
	ier0 = ier
	call ffdl2i(dl2pij(i2p(5,6),i2p(7,8)),piDpj,10,
     +		5,6,9,-1,7,8,9,+1,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(7,8),i2p(5,6)) = dl2pij(i2p(5,6),i2p(7,8))
*
	ier0 = ier
	call ffdl2i(dl2pij(i2p(5,6),i2p(5,8)),piDpj,10,
     +		5,6,9,-1,5,8,10,-1,ier0)
	ier1 = max(ier1,ier0)
	dl2pij(i2p(5,8),i2p(5,6)) = dl2pij(i2p(5,6),i2p(5,8))
*
	ier0 = ier
	call ffdl3q(dl3qi(i2p(6,7)),piDpj, 1,6,7, 0,10,0, 0,-1,0,
     +		0,+1,0, ier0)
	ier1 = max(ier1,ier0)
*
	ier0 = ier
	call ffdl3q(dl3qi(i2p(5,7)),piDpj, 1,5,7, 2,0,0, -1,0,0,
     +		+1,0,0, ier0)
	ier1 = max(ier1,ier0)
*
	ier0 = ier
	call ffdl3q(dl3qi(i2p(5,6)),piDpj, 1,2,3, 5,6,9, +1,+1,+1,
     +		-1,-1,-1, ier0)
	ier1 = max(ier1,ier0)
*
	if ( degree.gt.1 ) then
*
	ier0 = ier
	call ffdl3q(dl3qi(i2p(5,8)),piDpj, 1,5,8, 2,10,4, -1,-1,+1,
     +		+1,-1,+1, ier0)
	ier1 = max(ier1,ier0)
*
	ier0 = ier
	call ffdl3q(dl3qi(i2p(7,8)),piDpj, 3,4,1, 7,8,10, +1,+1,+1,
     +		-1,-1,-1, ier0)
	ier1 = max(ier1,ier0)
*
	ier0 = ier
	call ffdl3q(dl3qi(7),piDpj, 2,3,4, 6,7,10, +1,+1,+1,
     +		-1,-1,+1, ier0)
	ier1 = max(ier1,ier0)
*
	endif
	ier = ier1
	if ( lwrite ) print *,'ier after determinants = ',ier
*
*  #] get determinants:
*  #[ D1:
*-	#[ D11:
*
*	see the Form job D1.frm
*
	if ( lwrite ) print *,'ffxdi: D11'
	cs(1) = - cc0i(1)*DBLE(del2pi(1))
	cs(2) = + cc0i(2)*DBLE(dl2pij(i2p(6,7),i2p(7,8)))
	cs(3) = + cc0i(3)*DBLE(dl2pij(i2p(5,8),i2p(6,7)))
	cs(4) = + cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(6,7)))
	cs(5) = + 2*cd0*DBLE(dl3qi(i2p(6,7)))
*
	cd1p(1) = 0
	xmax = 0
	do 110 i=1,5
	    cd1p(1) = cd1p(1) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  110	continue
	if ( lwarn .and. abs(cd1p(1)) .lt. xloss*xmax ) then
	    a = abs(cd1p(1))
	    call ffwarn(164,ier1,a,xmax)
	    if ( lwrite ) print *,'cs,cd1p(1) = ',(cs(i),i=1,5),cd1p(1)
	endif
	cd1p(1) = cd1p(1)*(1/DBLE(2*del3p))
*
*-	#] D11:
*-	#[ D12:
*
	if ( lwrite ) print *,'ffxdi: D12'
	cs(1) = + cc0i(1)*DBLE(dl2pij(i2p(5,7),i2p(6,7)))
	cs(2) = - cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8)))
	cs(3) = - cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8)))
	cs(4) = - cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(5,7)))
	cs(5) = - 2*cd0*DBLE(dl3qi(i2p(5,7)))
*
	cd1p(2) = 0
	xmax = 0
	do 120 i=1,5
	    cd1p(2) = cd1p(2) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  120	continue
	if ( lwarn .and. abs(cd1p(2)) .lt. xloss*xmax ) then
	    a = abs(cd1p(2))
	    ier0 = ier
	    call ffwarn(164,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd1p(2) = ',(cs(i),i=1,5),cd1p(2)
	endif
	cd1p(2) = cd1p(2)*(1/DBLE(2*del3p))
*
*-	#] D12:
*-	#[ D13:
*
	if ( lwrite ) print *,'ffxdi: D13'
	cs(1) = - cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7)))
	cs(2) = + cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8)))
	cs(3) = + cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8)))
	cs(4) = + cc0i(4)*DBLE(del2pi(4))
	cs(5) = + 2*cd0*DBLE(dl3qi(i2p(5,6)))
*
	cd1p(3) = 0
	xmax = 0
	do 130 i=1,5
	    cd1p(3) = cd1p(3) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  130	continue
	if ( lwarn .and. abs(cd1p(3)) .lt. xloss*xmax ) then
	    a = abs(cd1p(3))
	    ier0 = ier
	    call ffwarn(164,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd1p(3) = ',(cs(i),i=1,5),cd1p(3)
	endif
	cd1p(3) = cd1p(3)*(1/DBLE(2*del3p))
*
*-	#] D13:
*-	#[ print output:
	if ( lwrite ) then
	    print *,'ffxdi: D1:'
	    print *,'cd1p = '
	    print '(6e20.13)',cd1p
	    print *,'ier  = ',ier1
	endif
*-	#] print output:
	if ( degree .eq. 1 ) then
	    ier = ier1
	    return
	endif
*  #] D1:
*  #[ D2:
*
*	see the form job d2.frm
*
*-	#[ D2del:
*
	if ( lwrite ) print *,'ffxdi: D2del'
	cs(1) = -2*DBLE(del4s)*cd0
	cs(2) = +DBLE(dl3qi(i2p(5,6)))*cc0i(4)
	cs(3) = +DBLE(dl3qi(i2p(5,8)))*cc0i(3)
	cs(4) = +DBLE(dl3qi(i2p(7,8)))*cc0i(2)
	cs(5) = -DBLE(dl3qi(7))*cc0i(1)
*
	cd2del = 0
	xmax = 0
	do 210 i=1,5
	    cd2del = cd2del + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  210	continue
	if ( lwarn .and. abs(cd2del) .lt. xloss*xmax ) then
	    a = abs(cd2del)
	    ier0 = ier
	    call ffwarn(189,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2del = ',(cs(i),i=1,5),cd2del
	endif
	cd2del = cd2del*DBLE(1/(-2*Del3p**2))
*
*-	#] D2del:
*-	#[ D2pp(1,1):
*
	if ( lwrite ) print *,'D2pp(1,1)'
	cs(1) = -cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(5,6)*
     +		del3p/del2pi(4))
	cs(2) = -cb0ij(1,2)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*piDpj(5,10)*
     +		del3p/del2pi(3))
	cs(3) = -cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,9)*
     +		del3p/del2pi(4))
	cs(4) = +cb0ij(1,3)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*piDpj(7,9)*
     +		del3p/del2pi(2))
	cs(5) = -cb0ij(1,4)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*piDpj(8,10)*
     +		del3p/del2pi(3))
	cs(6) = -cb0ij(1,4)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*piDpj(7,8)*
     +		del3p/del2pi(2))
	cs(7) = -cb0ij(2,3)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,6)*
     +		del3p/del2pi(4))
	cs(8) = -cb0ij(2,4)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*piDpj(10,10)*
     +		del3p/del2pi(3))
	cs(9) = -cb0ij(3,4)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*piDpj(7,7)*
     +		del3p/del2pi(2))
	cs(10) = -4*cc0i(1)*DBLE(dl3qi(i2p(6,7))*del2pi(1))
	cs(11) = +2*cc0i(1)*DBLE(dl3qi(7)*del2pi(1))
	cs(12) = -2*cc0i(2)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*
     +		dl2pij(i2p(6,7),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(13) = +4*cc0i(2)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*
     +		dl3qi(i2p(6,7)))
	cs(14) = -2*cc0i(3)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*
     +		dl2pij(i2p(5,8),i2p(6,7))*dl3qi(i2p(5,8))/del2pi(3))
	cs(15) = +4*cc0i(3)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*
     +		dl3qi(i2p(6,7)))
	cs(16) = -2*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*
     +		dl2pij(i2p(5,6),i2p(6,7))*dl3qi(i2p(5,6))/del2pi(4))
	cs(17) = +4*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*
     +		dl3qi(i2p(6,7)))
	cs(18) = +4*cd0*DBLE(dl3qi(i2p(6,7))*dl3qi(i2p(6,7)))
*
	cd2pp(1,1) = 0
	xmax = 0
	do 220 i=1,18
	    cd2pp(1,1) = cd2pp(1,1) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  220	continue
	if ( lwarn .and. abs(cd2pp(1,1)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(1,1))
	    ier0 = ier
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(1,1) = ',(cs(i),i=1,18),
     +		cd2pp(1,1)
	endif
	cd2pp(1,1) = cd2pp(1,1)*DBLE(1/(4*Del3p**2))
*
*-	#] D2pp(1,1):
*-	#[ D2pp(1,2):
*
	if ( lwrite ) print *,'D2pp(1,2)'
	cs(1)=+cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(5,
     +	6)*del3p/del2pi(4))
	cs(2)=+cb0ij(1,2)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(3)=+cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(6,
     +	9)*del3p/del2pi(4))
	cs(4)=-cb0ij(1,3)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	9)*del3p/del2pi(2))
	cs(5)=+cb0ij(1,4)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*piDpj(8,
     +	10)*del3p/del2pi(3))
	cs(6)=+cb0ij(1,4)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	8)*del3p/del2pi(2))
	cs(7)=+cb0ij(2,3)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(6,
     +	6)*del3p/del2pi(4))
	cs(8)=+cb0ij(2,4)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(9)=-cb0ij(2,4)*DBLE(piDpj(7,10)*del3p)
	cs(10)=+cb0ij(3,4)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	7)*del3p/del2pi(2))
	cs(11)=-2*cc0i(1)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*del3p)
	cs(12)=+2*cc0i(1)*DBLE(dl3qi(i2p(5,7))*del2pi(1))
	cs(13)=+2*cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*dl2pij(i2p(6,
     +	7),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(14)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*dl3qi(i2p(6,
     +	7)))
	cs(15)=-2*cc0i(2)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*dl3qi(i2p(5,
     +	7)))
	cs(16)=+2*cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*dl2pij(i2p(5,
     +	8),i2p(6,7))*dl3qi(i2p(5,8))/del2pi(3))
	cs(17)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*dl3qi(i2p(6,
     +	7)))
	cs(18)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*dl3qi(i2p(5,
     +	7)))
	cs(19)=+2*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*dl2pij(i2p(5,
     +	6),i2p(6,7))*dl3qi(i2p(5,6))/del2pi(4))
	cs(20)=-2*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*dl3qi(i2p(6,
     +	7)))
	cs(21)=-2*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*dl3qi(i2p(5,
     +	7)))
	cs(22)=-4*cd0*DBLE(dl3qi(i2p(5,7))*dl3qi(i2p(6,7)))
*
	cd2pp(1,2) = 0
	xmax = 0
	do 230 i=1,22
	    cd2pp(1,2) = cd2pp(1,2) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  230	continue
	if ( lwarn .and. abs(cd2pp(1,2)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(1,2))
	    ier0 = ier
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(1,2) = ',(cs(i),i=1,22),
     +		cd2pp(1,2)
	endif
	cd2pp(1,2) = cd2pp(1,2)*DBLE(1/(4*Del3p**2))
	cd2pp(2,1) = cd2pp(1,2)
*
*-	#] D2pp(1,2):
*-	#[ D2pp(1,3):
*
	if ( lwrite ) print *,'D2pp(1,3)'
	cs(1)=-cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(2)=-cb0ij(1,2)*DBLE(piDpj(5,6)*del3p)
	cs(3)=+cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	9)*del3p/del2pi(2))
	cs(4)=-cb0ij(1,3)*DBLE(piDpj(6,9)*del3p)
	cs(5)=-cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(8,
     +	10)*del3p/del2pi(3))
	cs(6)=-cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	8)*del3p/del2pi(2))
	cs(7)=-cb0ij(2,3)*DBLE(piDpj(6,6)*del3p)
	cs(8)=-cb0ij(2,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(10,
     +	10)*del3p/del2pi(3))
	cs(9)=-cb0ij(3,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	7)*del3p/del2pi(2))
	cs(10)=+2*cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*del3p)
	cs(11)=-2*cc0i(1)*DBLE(dl3qi(i2p(5,6))*del2pi(1))
	cs(12)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl2pij(i2p(6,
     +	7),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(13)=+2*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl3qi(i2p(6,
     +	7)))
	cs(14)=+2*cc0i(2)*DBLE(dl2pij(i2p(6,7),i2p(7,8))*dl3qi(i2p(5,
     +	6)))
	cs(15)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl2pij(i2p(5,
     +	8),i2p(6,7))*dl3qi(i2p(5,8))/del2pi(3))
	cs(16)=+2*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl3qi(i2p(6,
     +	7)))
	cs(17)=+2*cc0i(3)*DBLE(dl2pij(i2p(5,8),i2p(6,7))*dl3qi(i2p(5,
     +	6)))
	cs(18)=+2*cc0i(4)*DBLE(dl3qi(i2p(6,7))*del2pi(4))
	cs(19)=+4*cd0*DBLE(dl3qi(i2p(5,6))*dl3qi(i2p(6,7)))
*
	cd2pp(1,3) = 0
	xmax = 0
	do 240 i=1,19
	    cd2pp(1,3) = cd2pp(1,3) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  240	continue
	if ( lwarn .and. abs(cd2pp(1,3)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(1,3))
	    ier0 = ier
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(1,3) = ',(cs(i),i=1,19),
     +		cd2pp(1,3)
	endif
	cd2pp(1,3) = cd2pp(1,3)*DBLE(1/(4*Del3p**2))
	cd2pp(3,1) = cd2pp(1,3)
*
*-	#] D2pp(1,3):
*-	#[ D2pp(2,2):
*
	if ( lwrite ) print *,'D2pp(2,2)'
	cs(1)=-cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(5,
     +	5)*del3p/del2pi(4))
	cs(2)=-cb0ij(1,2)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*piDpj(5,
     +	5)*del3p/del2pi(3))
	cs(3)=-cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(5,
     +	9)*del3p/del2pi(4))
	cs(4)=-cb0ij(1,3)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	9)*del3p/del2pi(2))
	cs(5)=-cb0ij(1,4)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*piDpj(5,
     +	8)*del3p/del2pi(3))
	cs(6)=+cb0ij(1,4)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	8)*del3p/del2pi(2))
	cs(7)=-cb0ij(2,3)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*piDpj(5,
     +	6)*del3p/del2pi(4))
	cs(8)=-cb0ij(2,3)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*piDpj(6,
     +	7)*del3p/del2pi(1))
	cs(9)=-cb0ij(2,4)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(10)=+cb0ij(2,4)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*piDpj(7,
     +	10)*del3p/del2pi(1))
	cs(11)=-cb0ij(3,4)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*piDpj(7,
     +	7)*del3p/del2pi(1))
	cs(12)=+cb0ij(3,4)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*piDpj(7,
     +	7)*del3p/del2pi(2))
	cs(13)=+2*cc0i(1)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*dl2pij(i2p(5,
     +	7),i2p(6,7))*dl3qi(7)/del2pi(1))
	cs(14)=-4*cc0i(1)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*dl3qi(i2p(5,
     +	7)))
	cs(15)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*dl2pij(i2p(5,
     +	7),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(16)=+4*cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*dl3qi(i2p(5,
     +	7)))
	cs(17)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*dl2pij(i2p(5,
     +	7),i2p(5,8))*dl3qi(i2p(5,8))/del2pi(3))
	cs(18)=+4*cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*dl3qi(i2p(5,
     +	7)))
	cs(19)=-2*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*dl2pij(i2p(5,
     +	6),i2p(5,7))*dl3qi(i2p(5,6))/del2pi(4))
	cs(20)=+4*cc0i(4)*DBLE(dl2pij(i2p(5,6),i2p(5,7))*dl3qi(i2p(5,
     +	7)))
	cs(21)=+4*cd0*DBLE(dl3qi(i2p(5,7))*dl3qi(i2p(5,7)))
*
	cd2pp(2,2) = 0
	xmax = 0
	do 250 i=1,21
	    cd2pp(2,2) = cd2pp(2,2) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  250	continue
	if ( lwarn .and. abs(cd2pp(2,2)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(2,2))
	    ier0 = ier
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(2,2) = ',(cs(i),i=1,21),
     +		cd2pp(2,2)
	endif
	cd2pp(2,2) = cd2pp(2,2)*DBLE(1/(4*Del3p**2))
*
*-	#] D2pp(2,2):
*-	#[ D2pp(2,3):
*
	if ( lwrite ) print *,'D2pp(2,3)'
*
	cs(1)=+cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	5)*del3p/del2pi(3))
	cs(2)=+cb0ij(1,2)*DBLE(piDpj(5,5)*del3p)
	cs(3)=+cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	9)*del3p/del2pi(2))
	cs(4)=+cb0ij(1,3)*DBLE(piDpj(5,9)*del3p)
	cs(5)=+cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	8)*del3p/del2pi(3))
	cs(6)=-cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	8)*del3p/del2pi(2))
	cs(7)=+cb0ij(2,3)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,
     +	7)*del3p/del2pi(1))
	cs(8)=+cb0ij(2,3)*DBLE(piDpj(5,6)*del3p)
	cs(9)=+cb0ij(2,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(10)=-cb0ij(2,4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(7,
     +	10)*del3p/del2pi(1))
	cs(11)=+cb0ij(3,4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(7,
     +	7)*del3p/del2pi(1))
	cs(12)=-cb0ij(3,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	7)*del3p/del2pi(2))
	cs(13)=-2*cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*dl2pij(i2p(5,
     +	7),i2p(6,7))*dl3qi(7)/del2pi(1))
	cs(14)=+2*cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*dl3qi(i2p(5,
     +	7)))
	cs(15)=+2*cc0i(1)*DBLE(dl2pij(i2p(5,7),i2p(6,7))*dl3qi(i2p(5,
     +	6)))
	cs(16)=+2*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl2pij(i2p(5,
     +	7),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(17)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl3qi(i2p(5,
     +	7)))
	cs(18)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,7),i2p(7,8))*dl3qi(i2p(5,
     +	6)))
	cs(19)=+2*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl2pij(i2p(5,
     +	7),i2p(5,8))*dl3qi(i2p(5,8))/del2pi(3))
	cs(20)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl3qi(i2p(5,
     +	7)))
	cs(21)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,7),i2p(5,8))*dl3qi(i2p(5,
     +	6)))
	cs(22)=-2*cc0i(4)*DBLE(dl3qi(i2p(5,7))*del2pi(4))
	cs(23)=-4*cd0*DBLE(dl3qi(i2p(5,6))*dl3qi(i2p(5,7)))
*
	cd2pp(2,3) = 0
	xmax = 0
	do 260 i=1,23
	    cd2pp(2,3) = cd2pp(2,3) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  260	continue
	if ( lwarn .and. abs(cd2pp(2,3)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(2,3))
	    ier = ier0
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(2,3) = ',(cs(i),i=1,23),
     +		cd2pp(2,3)
	endif
	cd2pp(2,3) = cd2pp(2,3)*DBLE(1/(4*Del3p**2))
	cd2pp(3,2) = cd2pp(2,3)
*
*-	#] D2pp(2,3):
*-	#[ D2pp(3,3):
*
	if ( lwrite ) print *,'D2pp(3,3)'
	cs(1)=+cb0ij(1,2)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	5)*del3p/del2pi(3))
	cs(2)=+cb0ij(1,3)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(9,
     +	9)*del3p/del2pi(2))
	cs(3)=+cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	8)*del3p/del2pi(3))
	cs(4)=-cb0ij(1,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(8,
     +	9)*del3p/del2pi(2))
	cs(5)=-cb0ij(2,3)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,
     +	6)*del3p/del2pi(1))
	cs(6)=+cb0ij(2,4)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*piDpj(5,
     +	10)*del3p/del2pi(3))
	cs(7)=+cb0ij(2,4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,
     +	10)*del3p/del2pi(1))
	cs(8)=-cb0ij(3,4)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*piDpj(6,
     +	7)*del3p/del2pi(1))
	cs(9)=-cb0ij(3,4)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*piDpj(7,
     +	9)*del3p/del2pi(2))
	cs(10)=+2*cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*dl2pij(i2p(5,
     +	6),i2p(6,7))*dl3qi(7)/del2pi(1))
	cs(11)=-4*cc0i(1)*DBLE(dl2pij(i2p(5,6),i2p(6,7))*dl3qi(i2p(5,
     +	6)))
	cs(12)=-2*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl2pij(i2p(5,
     +	6),i2p(7,8))*dl3qi(i2p(7,8))/del2pi(2))
	cs(13)=+4*cc0i(2)*DBLE(dl2pij(i2p(5,6),i2p(7,8))*dl3qi(i2p(5,
     +	6)))
	cs(14)=-2*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl2pij(i2p(5,
     +	6),i2p(5,8))*dl3qi(i2p(5,8))/del2pi(3))
	cs(15)=+4*cc0i(3)*DBLE(dl2pij(i2p(5,6),i2p(5,8))*dl3qi(i2p(5,
     +	6)))
	cs(16)=+2*cc0i(4)*DBLE(dl3qi(i2p(5,6))*del2pi(4))
	cs(17)=+4*cd0*DBLE(dl3qi(i2p(5,6))*dl3qi(i2p(5,6)))
*
	cd2pp(3,3) = 0
	xmax = 0
	do 270 i=1,17
	    cd2pp(3,3) = cd2pp(3,3) + cs(i)
	    a = abs(cs(i))
	    xmax = max(xmax,a)
  270	continue
	if ( lwarn .and. abs(cd2pp(3,3)) .lt. xloss*xmax ) then
	    a = abs(cd2pp(3,3))
	    ier0 = ier
	    call ffwarn(190,ier0,a,xmax)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'cs,cd2pp(3,3) = ',(cs(i),i=1,17),
     +		cd2pp(3,3)
	endif
	cd2pp(3,3) = cd2pp(3,3)*DBLE(1/(4*Del3p**2))
*
*-	#] D2pp(3,3):
*-	#[ print output:
	if ( lwrite ) then
	    print '(a,2e20.13)','cd2del = ',cd2del
	    print '(a)','cd2pp  = '
	    print '(6e20.13)',cd2pp
	    print *,'ier    = ',ier1
	endif
	if ( ltest ) then
	    xlosn = xloss*DBLE(10)**(-2-mod(ier1,50))
	    cs(1) =   DBLE(piDpj(5,5))*cd2pp(1,1)
	    cs(2) = 2*DBLE(piDpj(5,6))*cd2pp(1,2)
	    cs(3) = 2*DBLE(piDpj(5,7))*cd2pp(1,3)
	    cs(4) =   DBLE(piDpj(6,6))*cd2pp(2,2)
	    cs(5) = 2*DBLE(piDpj(6,7))*cd2pp(2,3)
	    cs(6) =   DBLE(piDpj(7,7))*cd2pp(3,3)
	    cs(7) =   DBLE(del3p)*cd2del
	    cs(8) = - cc0i(1)
	    cs(9) = - DBLE(piDpj(1,1))*cd0
	    cnul = 0
	    xmax = 0
	    do 910 i=1,9
		cnul = cnul + cs(i)
		a = abs(cs(i))
		xmax = max(xmax,a)
  910	    continue
	    if ( lwrite ) print *,'ffxdi: checking D2.gmumu= ',cnul,xmax
	    if ( xlosn*abs(cnul) .gt. precc*xmax ) print *,'ffxdi: ',
     +		'error: D2(mu,mu) not correct ',cnul,xmax,ier1
	    cs(1) = 4*DBLE(piDpj(5,5)*piDpj(7,5))*cd2pp(1,1)
	    cs(2) = 4*DBLE(piDpj(5,5)*piDpj(7,6))*cd2pp(1,2)
	    cs(3) = 4*DBLE(piDpj(5,6)*piDpj(7,5))*cd2pp(1,2)
	    cs(4) = 4*DBLE(piDpj(5,5)*piDpj(7,7))*cd2pp(1,3)
	    cs(5) = 4*DBLE(piDpj(5,7)*piDpj(7,5))*cd2pp(1,3)
	    cs(6) = 4*DBLE(piDpj(5,6)*piDpj(7,6))*cd2pp(2,2)
	    cs(7) = 4*DBLE(piDpj(5,6)*piDpj(7,7))*cd2pp(2,3)
	    cs(8) = 4*DBLE(piDpj(5,7)*piDpj(7,6))*cd2pp(2,3)
	    cs(9) = 4*DBLE(piDpj(5,7)*piDpj(7,7))*cd2pp(3,3)
	    cs(10)= - cb0ij(1,3)
	    cs(11)= + cb0ij(1,4)
	    cs(12)= + cb0ij(2,3)
	    cs(13)= - cb0ij(2,4)
	    cs(14)= - 2*DBLE(piDpj(1,7))*cc0i(2)
	    cs(15)= + 2*DBLE(piDpj(1,7))*cc0i(1)
	    cs(16)= - 2*DBLE(piDpj(1,5))*cc0i(4)
	    cs(17)= + 2*DBLE(piDpj(1,5))*cc0i(3)
	    cs(18)= - 4*DBLE(piDpj(1,5)*piDpj(1,7))*cd0
	    cnul = 0
	    xmax = 0
	    do 920 i=1,18
		cnul = cnul + cs(i)
		a = abs(cs(i))
		xmax = max(xmax,a)
  920	    continue
	    if ( lwrite ) print *,'ffxdi: checking D2.p1p3 = ',cnul,xmax
	    if ( xlosn*abs(cnul) .gt. precc*xmax ) print *,'ffxdi :',
     +		'error: D2(p1,p3) not correct ',cnul,xmax,ier1
	    cs(1) = 4*DBLE(piDpj(6,5)*piDpj(8,5))*cd2pp(1,1)
	    cs(2) = 4*DBLE(piDpj(6,5)*piDpj(8,6))*cd2pp(1,2)
	    cs(3) = 4*DBLE(piDpj(6,6)*piDpj(8,5))*cd2pp(1,2)
	    cs(4) = 4*DBLE(piDpj(6,5)*piDpj(8,7))*cd2pp(1,3)
	    cs(5) = 4*DBLE(piDpj(6,7)*piDpj(8,5))*cd2pp(1,3)
	    cs(6) = 4*DBLE(piDpj(6,6)*piDpj(8,6))*cd2pp(2,2)
	    cs(7) = 4*DBLE(piDpj(6,6)*piDpj(8,7))*cd2pp(2,3)
	    cs(8) = 4*DBLE(piDpj(6,7)*piDpj(8,6))*cd2pp(2,3)
	    cs(9) = 4*DBLE(piDpj(6,7)*piDpj(8,7))*cd2pp(3,3)
	    cs(10)= - cb0ij(2,4)
	    cs(11)= + cb0ij(1,2)
	    cs(12)= + cb0ij(3,4)
	    cs(13)= - cb0ij(1,3)
	    cs(14)= - 2*DBLE(piDpj(1,8))*cc0i(3)
	    cs(15)= + 2*DBLE(piDpj(1,8))*cc0i(2)
	    cs(16)= - 2*DBLE(piDpj(1,6))*cc0i(1)
	    cs(17)= + 2*DBLE(piDpj(1,6))*cc0i(4)
	    cs(18)= - 4*DBLE(piDpj(1,6)*piDpj(1,8))*cd0
	    cnul = 0
	    xmax = 0
	    do 930 i=1,18
		cnul = cnul + cs(i)
		a = abs(cs(i))
		xmax = max(xmax,a)
  930	    continue
	    if ( lwrite ) print *,'ffxdi: checking D2.p2p4 = ',cnul,xmax
	    if ( xlosn*abs(cnul) .gt. precc*xmax ) print *,'ffxdi :',
     +		'error: D2(p2,p4) not correct ',cnul,xmax,ier1
	endif
*-	#] print output:
	if ( degree .eq. 2 ) then
	    ier = ier1
	    return
	endif
*  #] D2:
	print *,'ffxdi: error: D3 not ready'
	stop
*###] ffxdi:
	end
