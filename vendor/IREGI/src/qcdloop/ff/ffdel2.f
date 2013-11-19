*###[ ffdel2:
	subroutine ffdel2(del2,piDpj,ns,i1,i2,i3,lerr,ier)
*************************************************************************
*	calculate in a numerically stable way				*
*	del2(piDpj(i1,i1),piDpj(i2,i2),piDpj(i3,i3)) =			*
*		= piDpj(i1,i1)*piDpj(i2,i2) - piDpj(i1,i2)^2		*
*		= piDpj(i1,i1)*piDpj(i3,i3) - piDpj(i1,i3)^2		*
*		= piDpj(i2,i2)*piDpj(i3,i3) - piDpj(i2,i3)^2		*
*	ier is the usual error flag.					*
*************************************************************************
	implicit none
*
*	arguments:
*
	integer ns,i1,i2,i3,lerr,ier
	DOUBLE PRECISION del2,piDpj(ns,ns)
*
*	local variables
*
	DOUBLE PRECISION s1,s2
*
*	common blocks
*
	include 'ff.h'
*
*	calculations
*
	idsub = idsub + 1
	if ( abs(piDpj(i1,i2)) .lt. abs(piDpj(i1,i3)) .and.
     +	     abs(piDpj(i1,i2)) .lt. abs(piDpj(i2,i3)) ) then
	    s1 = piDpj(i1,i1)*piDpj(i2,i2)
	    s2 = piDpj(i1,i2)**2
	elseif ( abs(piDpj(i1,i3)) .lt. abs(piDpj(i2,i3)) ) then
	    s1 = piDpj(i1,i1)*piDpj(i3,i3)
	    s2 = piDpj(i1,i3)**2
	else
	    s1 = piDpj(i2,i2)*piDpj(i3,i3)
	    s2 = piDpj(i2,i3)**2
	endif
	del2 = s1 - s2
	if ( abs(del2) .lt. xloss*s2 ) then
	    if ( lerr .eq. 0 ) then
*		we know we have another chance
		if ( del2.ne.0 ) then
		    ier = ier + int(log10(xloss*abs(s2/del2)))
		else
		    ier = ier + int(log10(xloss*abs(s2)/xclogm))
		endif
	    else
		if ( lwarn ) call ffwarn(71,ier,del2,s1)
	    endif
	endif
*###] ffdel2:
	end
*###[ ffdl2p:
	subroutine ffdl2p(delps1,xpi,dpipj,piDpj,
     +		ip1,ip2,ip3,is1,is2,is3,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*	delta_{ip1,is2}^{ip1,ip2}					*
*	ier is the usual error flag.					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ip1,ip2,ip3,is1,is2,is3,ier
	DOUBLE PRECISION delps1,xpi(ns),dpipj(ns,ns),piDpj(ns,ns)
*
*	local variables
*
	DOUBLE PRECISION s1,s2,s3,xmax,som
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ stupid tree:
*	1
	s1 = xpi(ip1)*piDpj(ip2,is2)
	s2 = piDpj(ip1,ip2)*piDpj(ip1,is2)
	delps1 = s1 - s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( lwrite ) print *,'  delps1   = ',delps1,s1,s2
	som = delps1
	xmax = abs(s1)
*	2
	s1 = piDpj(ip1,ip2)*piDpj(ip3,is2)
	s2 = piDpj(ip1,ip3)*piDpj(ip2,is2)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+1 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	3
	s1 = piDpj(ip1,ip3)*piDpj(ip1,is2)
	s2 = xpi(ip1)*piDpj(ip3,is2)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+2 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	4
	s1 = xpi(ip1)*piDpj(ip2,is1)
	s2 = piDpj(ip1,is1)*piDpj(ip1,ip2)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+3 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	5
	s1 = piDpj(ip1,is2)*piDpj(ip2,is1)
	s2 = piDpj(ip1,is1)*piDpj(ip2,is2)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+4 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	6
	s1 = piDpj(ip1,ip2)*piDpj(ip3,is1)
	s2 = piDpj(ip1,ip3)*piDpj(ip2,is1)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+5 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	7
	s1 = piDpj(ip2,is2)*piDpj(ip3,is1)
	s2 = piDpj(ip2,is1)*piDpj(ip3,is2)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+6 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	8
	s1 = piDpj(ip1,ip3)*piDpj(ip1,is1)
	s2 = xpi(ip1)*piDpj(ip3,is1)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+7 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*	9
	s1 = piDpj(ip1,is1)*piDpj(ip3,is2)
	s2 = piDpj(ip1,is2)*piDpj(ip3,is1)
	delps1 = s1 - s2
	if ( lwrite ) print *,'  delps1+8 = ',delps1,s1,s2
	if ( abs(delps1) .ge. xloss*abs(s1) ) goto 100
	if ( abs(s1) .lt. xmax ) then
	    som = delps1
	    xmax = abs(s1)
	endif
*10	22-nov-1993 yet another one
	if ( dpipj(1,1).eq.0 ) then
	    s1 = +xpi(ip1)*dpipj(is3,is2)/2
	    s2 = -piDpj(ip1,ip2)*dpipj(is2,is1)/2
	    s3 = +xpi(ip1)*piDpj(ip2,ip3)/2
	    delps1 = s1+s2+s3
	    if ( lwrite ) print *,'  delps1+9 = ',delps1,s1,s2,s3
	    if ( abs(delps1) .ge. xloss*max(abs(s1),abs(s2)) ) goto 100
	    if ( max(abs(s1),abs(s2)) .lt. xmax ) then
		som = delps1
		xmax = abs(s1)
	    endif
	endif
*	NO possibility
	delps1 = som
	if ( lwarn ) call ffwarn(92,ier,delps1,xmax)
	if ( lwrite ) then
	    print *,'xpi = ',xpi
	    print *,'ip1,ip2,ip3,is1,is2,is3 = ',ip1,ip2,ip3,is1,is2,is3
	endif
  100	continue
*  #] stupid tree:
*###] ffdl2p:
	end
*###[ ffdl2s:
	subroutine ffdl2s(delps1,xpi,piDpj,in,jn,jin,isji,
     +					kn,ln,lkn,islk,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*		\delta_{si,sj}^{sk,sl}					*
*									*
*	with p(ji) = isji*(sj-si)					*
*	     p(lk) = islk*(sl-sk)					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer in,jn,jin,isji,kn,ln,lkn,islk,ns,ier
	DOUBLE PRECISION delps1,xpi(ns),piDpj(ns,ns)
*
*	local variables
*
	integer ii,jj,i,j,ji,k,l,lk,ihlp
	DOUBLE PRECISION s1,s2,som,smax
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    if ( abs(isji) .ne. 1 ) print *,'ffdl2s: error: abs(isji) ',
     +		' <> 1 but ',isji
	    if ( abs(islk) .ne. 1 ) print *,'ffdl2s: error: abs(islk) ',
     +		' <> 1 but ',islk
	endif
*  #] check input:
*  #[ stupid tree:
	idsub = idsub + 1
	som = 0
	smax = 0
	i = in
	j = jn
	ji = jin
	k = kn
	l = ln
	lk = lkn
	do 20 ii=1,3
	    do 10 jj=1,3
		s1 = piDpj(i,k)*piDpj(j,l)
		s2 = piDpj(i,l)*piDpj(j,k)
		delps1 = s1 - s2
		if ( ii .gt. 1 ) delps1 = isji*delps1
		if ( jj .gt. 1 ) delps1 = islk*delps1
		if ( ii .eq. 3 .neqv. jj .eq. 3 ) delps1 = -delps1
		if ( abs(delps1) .ge. xloss*abs(s1) ) goto 30

		if ( lwrite ) print *,'  delps1+',3*ii+jj-3,'=',delps1,
     +			abs(s1)
*
*		Save the most accurate estimate so far:
		if ( ii .eq. 1 .and. jj .eq. 1 .or. abs(s1) .lt. smax
     +			) then
		    som = delps1
		    smax = abs(s1)
		endif
*
*		rotate the jj's
		if ( lk .eq. 0 ) goto 20
		ihlp = k
		k = l
		l = lk
		lk = ihlp
   10	    continue
*
*	    and the ii's
	    if ( ji .eq. 0 ) goto 25
	    ihlp = i
	    i = j
	    j = ji
	    ji = ihlp
   20	continue
   25	continue
	delps1 = som
	if ( lwarn ) call ffwarn(83,ier,delps1,smax)
   30	continue
	if ( lwrite .and. 3*ii+jj-3.ne.1 .and. 3*ii+jj-3.ne.13 ) 
     +		print *,'  delps1+',3*ii+jj-3,'=', delps1,s1,s2
*  #] stupid tree:
*###] ffdl2s:
	end
*###[ ffdl2t:
	subroutine ffdl2t(delps,piDpj,in,jn,kn,ln,lkn,islk,iss,ns,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*		\delta_{si,sj}^{sk,sl}					*
*									*
*	with p(lk) = islk*(iss*sl - sk)	(islk,iss = +/-1)		*
*	and NO relationship between s1,s2 assumed (so 1/2 the		*
*	possibilities of ffdl2s).					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer in,jn,ip1,kn,ln,lkn,islk,iss,ns,ier
	DOUBLE PRECISION delps,piDpj(ns,ns)
*
*	local variables
*
	integer i
	DOUBLE PRECISION s1,s2,som,smax,xnul,xlosn
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    if ( abs(islk) .ne. 1 )
     +		print *,'ffdl2i: error: |islk| != 1 ',islk
	    if ( abs(iss) .ne. 1 )
     +		print *,'ffdl2i: error: |iss| != 1 ',iss
	    xlosn = xloss*DBLE(10)**(-1-mod(ier,50))
	    do 10 i=1,ns
		xnul = islk*iss*piDpj(ln,i) - islk*piDpj(kn,i) -
     +			piDpj(lkn,i)
		smax = max(abs(piDpj(ln,i)),abs(piDpj(kn,i)))
		if ( xlosn*abs(xnul) .gt. precx*smax ) then
		    print *,'ffdl2t: error: dotproducts ',islk*iss*ln,
     +		    -islk*kn,-lkn,' with ',i,' do not add to 0:',
     +		    islk*iss*piDpj(ln,i),-iss*piDpj(kn,i),-piDpj(lkn,i),
     +		    xnul,ier
		endif
   10	    continue
	endif
*  #] check input:
*  #[ calculations:
	if ( in .eq. jn ) then
	    delps = 0
	    return
	endif
	s1 = piDpj(kn,in)*piDpj(ln,jn)
	s2 = piDpj(ln,in)*piDpj(kn,jn)
	delps = s1 - s2
	if ( abs(delps) .ge. xloss*abs(s1) ) goto 20
	if ( lwrite ) print *,'  delps  = ',delps,s1,-s2
	som = delps
	smax = abs(s1)

	s1 = piDpj(kn,in)*piDpj(lkn,jn)
	s2 = piDpj(lkn,in)*piDpj(kn,jn)
	delps = iss*islk*(s1 - s2)
	if ( lwrite ) print *,'  delps+ = ',delps,islk,s1,-s2
	if ( abs(delps) .ge. xloss*abs(s1) ) goto 20
	if ( abs(s1) .lt. smax ) then
	    som = delps
	    smax = abs(s1)
	endif

	s1 = piDpj(lkn,in)*piDpj(ln,jn)
	s2 = piDpj(ln,in)*piDpj(lkn,jn)
	delps = islk*(- s1 + s2)
	if ( lwrite ) print *,'  delps++= ',delps,islk,-s1,s2
	if ( abs(delps) .ge. xloss*abs(s1) ) goto 20
	if ( abs(s1) .lt. smax ) then
	    som = delps
	    smax = abs(s1)
	endif
*
*	give up
*
	delps = som
	if ( lwarn ) call ffwarn(93,ier,delps,smax)

   20	continue
*  #] calculations:
*###] ffdl2t:
	end
*###[ ffdl3m:
	subroutine ffdl3m(del3mi,ldel,del3,del2,xpi,dpipj,piDpj,ns,ip1n,
     +		ip2n,ip3n,is,itime,ier)
***#[*comment:***********************************************************
*									*
*	Calculate xpi(i)*del2 - del3(piDpj)				*
*									*
*	  /  si	mu \2		(This appears to be one of the harder	*
*	= | d	   |		 determinants to calculate accurately.	*
*	  \  p1	p2 /		 Note that we allow a loss of xloss^2)	*
*									*
*	Input:	ldel		iff .true. del2 and del3 exist		*
*		del3		\delta^{s(1),p1,p2}_{s(1),p1,p2}	*
*		del2		\delta^{p1,p2}_{p1,p2}			*
*		xpi(ns)		standard				*
*		dpipj(ns,ns)	standard				*
*		piDpj(ns,ns)	standard				*
*		ipi		pi = xpi(abs(ipi)) [p3=-p1 +/-p2]	*
*		is		si = xpi(is,is+1,..,is+itime-1)		*
*		itime		number of functions to calculate	*
*									*
*	Output:	del3mi(3)	(\delta^{s_i \mu}_{p_1 p_2})^2		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ip1n,ip2n,ip3n,is,itime,ier
	logical ldel
	DOUBLE PRECISION del3mi(itime),del3,del2,xpi(ns),dpipj(ns,ns),
     +		piDpj(ns,ns)
*
*	local variables:
*
	DOUBLE PRECISION s(7),som,smax,del2s,delps,xsom,xmax
	integer i,j,k,ip1,ip2,ip3,ipn,is1,is2,isi,is3,ihlp,iqn,jsgnq,
     +		jsgn1,jsgn2,jsgn3,jsgnn,iadj(10,10,3:4),init,nm
	save iadj,init
	logical lsign,lmax,ltwist
*
*	common blocks:
*
	include 'ff.h'
*
*	data
*
	data iadj /200*0/
	data init /0/
*  #] declarations:
*  #[ initialisations:
	if ( init .eq. 0 ) then
	    init = 1
*
*	    Fill the array with adjacent values: if
*		x = iadj(i,j)
*		k = abs(mod(k,100))
*		jsgnk = sign(x)
*		jsgnj = 1-2*theta(x-100)  (ie -1 iff |x|>100)
*	    then
*		pi(k) = jsgnk*( p(i) - jsgnj*pi(j) )
*
	    do 5 nm=3,4
		do 4 i=1,nm
		    is1 = i
		    is2 = i+1
		    if ( is2 .gt. nm ) is2 = 1
		    is3 = i-1
		    if ( is3 .eq. 0 ) is3 = nm
		    ip1 = is1 + nm
		    iadj(is1,is2,nm) = -ip1
		    iadj(is2,is1,nm) = ip1
		    iadj(ip1,is2,nm) = -is1
		    iadj(is2,ip1,nm) = is1
		    iadj(is1,ip1,nm) = 100+is2
		    iadj(ip1,is1,nm) = 100+is2
		    if ( nm .eq. 3 ) then
			iadj(ip1,is2+3,3) = -100-is3-3
			iadj(is2+3,ip1,3) = -100-is3-3
		    endif
    4		continue
    5	    continue

	    iadj(3,1,4) = -9
	    iadj(1,3,4) = 9
	    iadj(9,1,4) = -3
	    iadj(1,9,4) = 3
	    iadj(3,9,4) = 100+1
	    iadj(9,3,4) = 100+1

	    iadj(2,4,4) = -10
	    iadj(4,2,4) = 10
	    iadj(10,4,4) = -2
	    iadj(4,10,4) = 2
	    iadj(2,10,4) = 100+4
	    iadj(10,2,4) = 100+4

	endif
	if ( ns .eq. 6 ) then
	    nm = 3
	else
	    nm = 4
	endif
*  #] initialisations:
*  #[ superfluous code:
*	if ( ns .ne. 6 ) print *,'ffdl3m: called with ns <> 6 !!'
*	if ( ip1n .lt. 4 ) then
*	    lsign = .TRUE.
*	else
*	    lsign = .FALSE.
*	endif
*	if ( ltest .and. lsign ) then
*	    if ( ip3n .eq. 4 ) then
*		if ( ip1n .ne. 1 .or. ip2n .ne. 2 ) goto 2
*	    elseif ( ip3n .eq. 5 ) then
*		if ( ip1n .ne. 2 .or. ip2n .ne. 3 ) goto 2
*	    elseif ( ip3n .eq. 6 ) then
*		if ( ip1n .ne. 3 .or. ip2n .ne. 1 ) goto 2
*	    else
*		goto 2
*	    endif
*	    goto 3
*    2	    continue
*	    print *,'ffdl3m: unexpected combination of indices',ip1,ip2,
*     +					ip3
*    3	    continue
*	endif
*	this went at he end:
*  #[ special case 4,5,6:
*	    Next try - I don't give up easily
*	    if ( nm .eq. 6 .and. ip1n .eq. 4 .and. ip2n .eq. 5 .and.
*     +			ip3n .eq. 6 .and. is .eq. 1 ) then
*		is3 = isi + 1
*		if ( is3 .eq. 4 ) is3 = 1
*		is1 = is3 + 1
*		if ( is1 .eq. 4 ) is1 = 1
*		ip1 = is1 + 3
*		ip2 = isi + 3
*		ip3 = is3 + 3
*		This is an algorithm of last resort.  Add special
*		cases at will.
*		s(1) = xpi(ip1)*xpi(ip2)*xpi(ip3)
*		s(2) = dpipj(is1,isi)*dpipj(ip1,ip2)**2
*		s(3) = -dpipj(is1,isi)*xpi(ip3)*(xpi(ip1)+xpi(ip2))
*		s(4) = 2*dpipj(is1,isi)*dpipj(is1,is3)*
*     +			piDpj(ip1,ip3)
*		s(5) = -2*dpipj(is1,is3)*xpi(ip1)*piDpj(ip2,ip3)
*		s(6) = dpipj(is1,isi)**2*xpi(ip3)
*		s(7) = dpipj(is1,is3)**2*xpi(ip1)
*		som = s(1)
*		smax = abs(s(1))
*		do 31 j=2,7
*		    som = som + s(j)
*		    smax = max(smax,abs(som))
*   31		continue
*		som = som/4
*		smax = smax/4
*		if (lwrite) print *,'  del3mi(',isi,')++= ',som,smax
*		if ( abs(som) .ge. xloss*smax ) goto 35
*		if ( smax .lt. xmax ) then
*		    xsom = som
*		    xmax = smax
*		endif
*	    endif
*  #] special case 4,5,6:
*  #] superfluous code:
*  #[ easy tries:
	do 40 i=1,itime
	    isi = i+is-1
	    lmax = .FALSE.
*
*	    get xpi(isi)*del2 - del3 ... if del3 and del2 are defined
*
	    if ( ldel ) then
		s(1) = xpi(isi)*del2
		som = s(1) - del3
		smax = abs(s(1))
		if ( abs(som) .ge. xloss**2*smax ) goto 35
		if ( lwrite ) print *,'  del3mi(',isi,')  =',som,s(1),
     +			del3
		xsom = som
		xmax = smax
		lmax = .TRUE.
	    endif
	    ip1 = ip1n
	    ip2 = ip2n
	    ip3 = ip3n
	    do 20 j=1,3
*
*		otherwise use the simple threeterm formula
*
		s(1) = xpi(ip2)*piDpj(ip1,isi)**2
		s(2) = xpi(ip1)*piDpj(ip2,isi)*piDpj(ip2,isi)
		s(3) = -2*piDpj(ip2,isi)*piDpj(ip2,ip1)*piDpj(ip1,isi)
		som = s(1) + s(2) + s(3)
		smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
		if ( lwrite .and. (ldel.or.j.ne.1) ) print *,
     +			'  del3mi(',isi,')+ =',som,(s(k),k=1,3)
		if ( abs(som) .ge. xloss**2*smax ) goto 35
		if ( lwrite .and. .not.(ldel.or.j.ne.1) ) print *,
     +			'  del3mi(',isi,')  =',som,(s(k),k=1,3)
		if ( .not. lmax .or. smax .lt. xmax ) then
		    xsom = som
		    xmax = smax
		    lmax = .TRUE.
		endif
*
*		if there are cancellations between two of the terms:
*		we try mixing with isi.
*
*		First map cancellation to s(2)+s(3) (do not mess up
*		rotations...)
*
		if ( abs(s(1)+s(3)) .lt. abs(s(3))/2 ) then
		    ihlp = ip1
		    ip1 = ip2
		    ip2 = ihlp
		    som = s(1)
		    s(1) = s(2)
		    s(2) = som
		    ltwist = .TRUE.
		else
		    ltwist = .FALSE.
		endif
		if ( abs(s(2)+s(3)) .lt. abs(s(3))/2 ) then
*
*		switch to the vector pn so that si = jsgn1*p1 + jsgnn*pn
*
		k = iadj(isi,ip1,nm)
		if ( k .ne. 0 ) then
		    ipn = abs(k)
		    jsgnn = isign(1,k)
		    if ( ipn .gt. 100 ) then
			ipn = ipn - 100
			jsgn1 = -1
		    else
			jsgn1 = +1
		    endif
		    if (abs(dpipj(ipn,isi)).lt.xloss*abs(piDpj(ip1,isi))
     +		     .and.
     +			abs(piDpj(ipn,ip2)).lt.xloss*abs(piDpj(ip2,isi))
     +			   ) then
*		same:	s(1) = xpi(ip2)*piDpj(ip1,isi)**2
			s(2) = jsgnn*piDpj(isi,ip2)*piDpj(ipn,ip2)*
     +								xpi(ip1)
			s(3) = jsgn1*piDpj(isi,ip2)*piDpj(ip1,ip2)*
     +							dpipj(ipn,isi)
			som = s(1) + s(2) + s(3)
			smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
			if ( lwrite ) print *,
     +			    '  del3mi(',isi,')++=',som,(s(k),k=1,3)
*			print *,'    (isi+ip1) with isi,ip1,ip2,ipn: ',
*     +				isi,ip1,ip2,ipn
*			print *,'xpi(ip2),piDpj(ip1,isi)',xpi(ip2),
*     +				piDpj(ip1,isi)
*			print *,'piDpj(isi,ip2),piDpj(ipn,ip2),xpi(ip1)'
*     +				,piDpj(isi,ip2),piDpj(ipn,ip2),xpi(ip1)
			if ( abs(som) .ge. xloss**2*smax ) goto 35
			if ( smax .lt. xmax ) then
			    xsom = som
			    xmax = smax
			endif
*
*			there may be a cancellation between s(1) and
*			s(2) left.  Introduce a vector q such that
*			pn = jsgnq*q + jsgn2*p2.  We also need the sign
*			jsgn3 in p3 = -p1 - jsgn3*p2
*
			k = iadj(ipn,ip2,nm)
			if ( k .ne. 0 ) then
			    iqn = abs(k)
*not used		    jsgnq = isign(1,k)
			    if ( iqn .gt. 100 ) then
				iqn = iqn - 100
				jsgn2 = -1
			    else
				jsgn2 = +1
			    endif
			    k = iadj(ip1,ip2,nm)
			    if ( k .eq. 0 .or. k .lt. 100 ) then
*				we have p1,p2,p3 all p's
				jsgn3 = +1
			    elseif ( k .lt. 0 ) then
*				ip1,ip2 are 2*s,1*p such that p2-p1=ip3
				jsgn3 = -1
			    else
				jsgn3 = 0
			    endif
*			    we need one condition on the signs for this
*			    to work
			    if ( ip3.ne.0 .and. jsgn1*jsgn2.eq.jsgnn*
     +			      jsgn3 .and. abs(s(3)).lt.xloss*smax ) then
				s(1) = piDpj(ip1,isi)**2*dpipj(iqn,ipn)
				s(2) = -jsgn2*jsgn1*piDpj(ipn,ip2)*
     +					piDpj(ip1,isi)*dpipj(ipn,isi)
*				s(3) stays the same
				s(4) = -jsgn2*jsgn1*piDpj(ipn,ip2)*
     +					xpi(ip1)*piDpj(isi,ip3)
				som = s(1) + s(2) + s(3) + s(4)
				smax =max(abs(s(1)),abs(s(2)),abs(s(3)),
     +					abs(s(4)))
				if ( lwrite ) print *,
     +				'  del3mi(',isi,')+2=',som,(s(k),k=1,4)
				if ( abs(som).ge.xloss**2*smax ) goto 35
				if ( smax .lt. xmax ) then
				    xsom = som
				    xmax = smax
				endif
			    endif
			endif
		    endif
		endif
		k = iadj(isi,ip2,nm)
		if ( k .ne. 0 ) then
		    ipn = abs(k)
		    jsgnn = isign(1,k)
		    if ( ipn .gt. 100 ) then
			jsgn1 = -1
			ipn = ipn - 100
		    else
			jsgn1 = +1
		    endif
		    if (abs(dpipj(ipn,isi)).lt.xloss*abs(piDpj(ip2,isi))
     +		     .and.
     +			abs(piDpj(ipn,ip1)).lt.xloss*abs(piDpj(ip1,isi))
     +			   ) then
			s(1) = jsgnn*piDpj(isi,ip1)*piDpj(ipn,ip1)*
     +								xpi(ip2)
			s(2) = xpi(ip1)*piDpj(ip2,isi)**2
			s(3) = jsgn1*piDpj(isi,ip1)*piDpj(ip2,ip1)*
     +							dpipj(ipn,isi)
			som = s(1) + s(2) + s(3)
			smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
			if ( lwrite ) print *,
     +			    '  del3mi(',isi,')++=',som,(s(k),k=1,3)
			print *,'    (isi+ip2) with isi,ip1,ip2,ipn: ',
     +				isi,ip1,ip2,ipn
			if ( abs(som) .ge. xloss**2*smax ) goto 35
			if ( smax .lt. xmax ) then
			    xsom = som
			    xmax = smax
			endif
		    endif
		endif
		endif
*this does not suffice
*		if ( lsign ) then
*		    if ( abs(s(1)) .lt. abs(s(2)) ) then
*			s(2) = piDpj(isi,ip2)*piDpj(isi,ip3)*xpi(ip1)
*			if ( j .eq. 2 ) s(2) = -s(2)
*			s(3) = piDpj(isi,ip1)*piDpj(isi,ip2)*
*     +				dpipj(ip3,ip2)
*		    else
*			s(1) = piDpj(isi,ip1)*piDpj(isi,ip3)*xpi(ip2)
*			if ( j .eq. 1 ) s(1) = -s(1)
*			s(3) = piDpj(isi,ip1)*piDpj(isi,ip2)*
*     +				dpipj(ip3,ip1)
*		    endif
*		    if ( j .eq. 3 ) s(3) = -s(3)
**
*		    som = s(1) + s(2) + s(3)
*		    smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
*		    if ( lwrite ) print *,
*     +			'  del3mi(',isi,')++=',som,(s(k),k=1,3)
*		    if ( abs(som) .ge. xloss**2*smax ) goto 35
*		    if ( smax .lt. xmax ) then
*			xmax = smax
*			xsom = som
*		    endif
*		endif
*nor does this
*		if ( j .eq. 1 )
*     +			call ffdel2(del2s,piDpj,6,ip1,ip2,ip3,1,ier)
*		call ffdl2t(delps,piDpj,isi,ip2,ip1,ip2,ip3,+1,+1,6,ier)
*		s(1) = piDpj(isi,ip2)**2*del2s/xpi(ip2)
*		s(2) = delps**2/xpi(ip2)
*		som = s(1) + s(2)
*		smax = abs(s(1))
*		if ( lwrite ) print *,
*     +			'  del3mi(',isi,')++=',del3mi(i),(s(k),k=1,2)
*		if ( abs(som) .ge. xloss*smax ) goto 35
*		if ( smax .lt. xmax ) then
*		    xmax = smax
*		    xsom = som
*		endif
*
*		rotate the ipi
*
		if ( ip3 .eq. 0 ) goto 30
		if ( j .ne. 3 ) then
		    if ( .not. ltwist ) then
			ihlp = ip1
			ip1 = ip2
			ip2 = ip3
			ip3 = ihlp
		    else
			ihlp = ip2
			ip2 = ip3
			ip3 = ihlp
		    endif
		endif
   20	    continue
   30	    continue
*  #] easy tries:
*  #[ choose the best value:
*
*	    These values are the best found:
*
	    som = xsom
	    smax = xmax
	    if ( lwarn ) call ffwarn(75,ier,som,smax)
	    if ( lwrite ) then
		print *,'ffdl3m: giving up:'
		print *,'ip1,ip2,ip3,is,itime =',ip1,ip2,ip3,is,itime
		print *,'xpi = ',xpi
	    endif

   35	    continue
	    del3mi(i) = som
   40	continue
*  #] choose the best value:
*###] ffdl3m:
	end
