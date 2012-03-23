*###[ ffxdir:
	subroutine ffxdir(cs,cfac,idone,xpi,dpipj,ipoin,ndiv,ier)
***#[*comment:***********************************************************
*									*
*	Check if this 4point function is IRdivergent and if so, get it	*
*	using ffxdbd and set idone to 1 (or 2 if 2 IR poles)		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ipoin,idone,ndiv,ier
	DOUBLE COMPLEX cs,cfac
	DOUBLE PRECISION xpi(13),dpipj(10,13)
*
*	local variables
*
	integer i,j,k,l,ier0,ii(6),notijk(4,4,4)
	DOUBLE PRECISION del4s,rloss
	save notijk
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data notijk/
     +	0,0,0,0,0,0,4,3,0,4,0,2,0,3,2,0,0,0,4,3,0,0,0,0,4,0,0,1,3,0,1,0,
     +	0,4,0,2,4,0,0,1,0,0,0,0,2,1,0,0,0,3,2,0,3,0,1,0,2,1,0,0,0,0,0,0/
*
*  #] declarations:
*  #[ work:
*
	idone = 0
	do 25 i=1,4
	    if ( xpi(i) .ne. 0 ) goto 25
	    do 24 j=1,3
		if ( j .eq. i ) goto 24
		if ( dpipj(j,inx(j,i)) .ne. 0 ) goto 24
		do 23 k=j+1,4
		    if ( k .eq. i ) goto 23
		    if ( dpipj(k,inx(k,i)) .ne. 0 ) goto 23
*
*		    we found an IR divergent function;
*		    first check whether it is linearly divergent
*
		    l = notijk(k,j,i)
		    if ( ltest ) then
			if ( l.eq.0 .or. l.eq.i .or. l.eq.j .or. l.eq.k
     +				) print *,'ffxkbd: error, l wrong: ',l
		    endif
*
*		    do we have a linear divergence on our hands?
*
		    if ( dpipj(l,inx(l,i)) .eq. 0 ) then
			if ( lwrite ) print *,'ffxdir: found ',
     +				'linearly divergent combo'
			if ( ndiv.eq.-1 ) ndiv = 1
		    elseif ( ndiv.gt.0 ) then
			if ( lwrite ) print *,'Not enough singularities'
			cs = 0
			cfac = 1
			idone = 1
			return
		    endif
*
*		    the complex case
*
		    if ( lsmug ) then
*
*			use Wim & Ansgard's formulae whenever possible
*
			if ( c2sisj(i,j).eq.0 .and. c2sisj(i,k).eq.0 )
     +				then
			    call ffxdbd(cs,cfac,xpi,dpipj,i,j,k,l,ier)
			    goto 98
			endif
			if ( c2sisj(i,j).eq.0 .and. dpipj(i,inx(i,l))
     +				.eq.0 .and. c2sisj(i,l).eq.0 ) then
			    call ffxdbd(cs,cfac,xpi,dpipj,i,j,l,k,ier)
			    goto 98
			endif
			if ( c2sisj(i,k).eq.0 .and. dpipj(i,inx(i,l))
     +				.eq.0 .and. c2sisj(i,l).eq.0 ) then
			    call ffxdbd(cs,cfac,xpi,dpipj,i,k,l,j,ier)
			    goto 98
			endif
*
*			is it nasty?
*
			if ( dpipj(i,inx(i,l)).eq.0 ) then
			    if ( c2sisj(j,i).eq.0 ) then
				goto 99
			    elseif ( c2sisj(k,i).eq.0 ) then
				goto 99
			    elseif ( c2sisj(l,i).eq.0 ) then
				goto 99
			    else
				call fferr(71,ier)
				print *,'xpi = ',xpi
				print *,'id,idsub = ',id,idsub
				return
			    endif
			endif
*
*			then it just is logarithmiocally divergent
*			let the ffxc0i handle this
*
		    else
*
*			the real case
*
			if ( dpipj(i,inx(i,l)).eq.0 ) then
			    call fferr(73,ier)
			    print *,'xpi = ',xpi
			    idone = 1
			    return
			endif
			call ffxdbd(cs,cfac,xpi,dpipj,i,j,k,l,ier)
			goto 98
		    endif
   23		continue
   24	    continue
   25	continue
	idone = 0
	lnasty = .FALSE.
	if ( ndiv.eq.-1 ) ndiv = 0
	return
*
*	clean up
*
   98	continue
	if ( ldot .and. ipoin.eq.4 ) then
	    ier0 = 0
	    if ( idot.lt.1 ) then
	    	call ffdot4(fpij4,xpi,dpipj,10,ier0)
	    endif
	    ii(1)= 5
	    ii(2)= 6
	    ii(3)= 7
	    ii(4)= 8
	    ii(5)= 9
	    ii(6)= 10
	    if ( abs(idot).lt.2 ) then
		fidel3 = ier0
		call ffdl3p(fdel3,fpij4,10,ii,ii,fidel3)
	    endif
	    if ( ltest ) then
	    	if ( lwrite ) print *,'ffxdir: checking fdel4s'
		call ffdel4(del4s,xpi,fpij4,10,ier0)
		rloss = xloss*DBLE(10)**(-mod(ier0,50))
		if ( rloss*abs(del4s-fdel4s) .gt. precx*abs(del4s) )
     +			print *,'ffxdir: error: del4s wrong: ',fdel4s,
     +			del4s,fdel4s-del4s,ier0
	    endif
	endif
*
*	and finito
*
	if ( ndiv.eq.-1 ) ndiv = 0
	idone = 1
	if ( xpi(j) .eq. 0 .or. xpi(k) .eq. 0 ) idone = 2
	if ( xpi(j) .eq. 0 .and. xpi(k) .eq. 0 ) idone = 3
	return
*
*	nasty - set some flags
*
   99	continue
	if ( lwrite ) print *,'ffxdir: nasty D0'
	lnasty = .TRUE.
	return
*
*  #] work:
*###] ffxdir:
	end
*###[ ffxdbd:
	subroutine ffxdbd(csom,cfac,xpi,dpipj,ilam,i1,i4,ic,ier)
***#[*comment:***********************************************************
*									*
*	The IR divergent fourpoint function with real masses		*
*	according to Beenakker & Denner, Nucl.Phys.B338(1990)349.	*
*									*
*	Input:	xpi(13)	      real	momenta^2			*
*		dpipj(10,13)  real	xpi(i)-xpi(j)			*
*		ilam	      integer	position of m=0			*
*		i1,i4	      integer	position of other 2 IR masses	*
*		ic	      integer	position of complex mass	*
*	/ffcut/	delta	      real	cutoff to use instead of lam^2	*
*									*
*	Output:	csom,cfac	      complex	D0  = csom*cfac		*
*		ier	      integer	number of digits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ilam,i1,i4,ic,ier
	DOUBLE COMPLEX csom,cfac
	DOUBLE PRECISION xpi(13),dpipj(10,13)
*
*	local variables
*
	integer ier0,ier1,ipi12,ip,init,is,i2,i3,i,iepst,iepss,ieps2,
     +		ieps3
	DOUBLE PRECISION absc,xmax
	DOUBLE PRECISION xxs(3),xxt(1),xx2(3),xx3(3),xm0,xm1,xm4,xlam,
     +		d,dfflo1,fac
	DOUBLE COMPLEX c,cs(21),z,zlg,som,cxt
	DOUBLE COMPLEX zxfflg,zfflog
	save init
*
*	common blocks
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data init /0/
*
*  #] declarations:
*  #[ check input:
*
	if ( init .eq. 0 ) then
	    init = 1
	    print *,'ffxdbd: using IR cutoff delta = lam^2 = ',delta
	endif
	if ( lwrite ) then
	    print *,'ffxdbd: input: ilam,i1,i4,ic = ',ilam,i1,i4,ic
	endif
	if ( ltest ) then
	    if ( delta .eq. 0 ) print *,'ffxdbd: error: (IR)delta = 0!'
	    if ( xpi(ilam) .ne. 0 ) print *,'ffxdbd: error: lam != 0 ',
     +		ilam,xpi(ilam)
	    if ( dpipj(i1,inx(ilam,i1)) .ne. 0 ) print *,
     +		'ffxdbd: error: m1^2 != p1^2 ',i1,inx(ilam,i1),xpi(i1),
     +		xpi(inx(ilam,i1)),dpipj(i1,inx(ilam,i1))
	    if ( dpipj(i4,inx(ilam,i4)) .ne. 0 ) print *,
     +		'ffxdbd: error: m4^2 != p4^2 ',i4,inx(ilam,i4),xpi(i4),
     +		xpi(inx(ilam,i4)),dpipj(i4,inx(ilam,i4))
	    if ( lsmug ) then
	    if ( c2sisj(i1,ilam).ne.0 ) print *,'ffxdbd: error: m(',i1,
     +		') not onshell, c2sisj(',i1,ilam,') = ',c2sisj(i1,ilam)
	    if ( c2sisj(i4,ilam).ne.0 ) print *,'ffxdbd: error: m(',i4,
     +		') not onshell, c2sisj(',i4,ilam,') = ',c2sisj(i4,ilam)
	    endif
	endif
	if ( xpi(i1).eq.0 .or. xpi(i4).eq.0 ) then
	    call fferr(98,ier)
	    return
	endif
*
*  #] check input:
*  #[ preliminaries:
*
	csom = 0
	cfac = 1
	xm0 = sqrt(xpi(ic))
	xm1 = sqrt(xpi(i1))
	xm4 = sqrt(xpi(i4))
	xlam = sqrt(delta)
*
*  #] preliminaries:
*  #[ special case m0=0, m1=m2, m3=m4:
	if ( xpi(ic) .eq. 0 ) then
*
*	    even more special case: 2 points of IR divergence:
*
	    if ( dpipj(i1,inx(ic,i1)).eq.0 .and.
     +		 dpipj(i4,inx(ic,i4)).eq.0 ) then
		if ( lwrite ) print *,'ffxdbd: doubly IR case'
		ier0 = 0
		call ffxkfn(xxs,iepss,xpi(inx(i1,i4)),xm1,xm4,ier0)
		if ( ier0.ge.100 ) then
		    call fferr(44,ier)
		    return
		endif
		ier = ier + ier0
		if ( abs(xxs(2)).gt.xloss ) then
		    zlg = zxfflg(xxs(1),iepss,x0,ier)
		else
		    zlg = DBLE(dfflo1(xxs(2),ier))
		endif
		csom = -2*zlg*zxfflg(-delta/xpi(inx(ilam,ic)),-2,x0,ier)
		fac = xxs(1)/(xm1*xm4*xpi(inx(ilam,ic))*xxs(2)*xxs(3))
		cfac = fac
		if ( ldot .and. abs(idot).lt.4 ) then
		    fdel4s = 1/(16*fac**2)
		    if ( lwrite ) print *,'del4s = ',fdel4s
		endif
		return
	    endif
*  #] special case m0=0, m1=m2, m3=m4:
*  #[ special case m0=0, m1=m2, m3!=m4:
	    if ( dpipj(i1,inx(ic,i1)).eq.0 .or.
     +		 dpipj(i4,inx(ic,i4)).eq.0 ) then
		if ( dpipj(i1,inx(ic,i1)).ne.0 ) then
		    i = i4
		    i4 = i1
		    i1 = i
		endif
		if ( lwrite ) print *,'ffxdbd: special case m0=0, ',
     +			'm1=m2 but m3!=m4'
*
*		From Wim Beenakker, Priv.Comm.
*
		ier0 = 0
		call ffxkfn(xxs,iepss,xpi(inx(i1,i4)),xm1,xm4,ier0)
		if ( ier0.ge.100 ) then
		    call fferr(44,ier)
		    return
		endif
		ier = ier + ier0
		ier0 = ier
		ier1 = ier
		if ( abs(xxs(2)).gt.xloss ) then
		    zlg = zxfflg(xxs(1),iepss,x0,ier0)
		else
		    zlg = DBLE(dfflo1(xxs(2),ier0))
		endif
		cs(1) = zlg**2
		ier1 = max(ier0,ier1)
		ier0 = ier
		if ( xxs(1)**2.lt.xloss ) then
		    cs(2) = -2*DBLE(dfflo1(xxs(1)**2,ier0))*zlg
		else
		    cs(2) = -2*zxfflg(xxs(2)*xxs(3),0,x0,ier0)*zlg
		endif
		ier1 = max(ier0,ier1)
		ier0 = ier
		cs(3) = zxfflg(delta/xpi(i4),0,x0,ier0)*zlg
		ier1 = max(ier0,ier1)
		ier0 = ier
		cs(4) = 2*zxfflg(dpipj(inx(ic,i4),i4)/xpi(inx(ilam,ic)),
     +			-1,dpipj(inx(ic,i4),i4),ier0)*zlg
		ier1 = max(ier0,ier1)
		ier0 = ier
		call ffzxdl(cs(5),ip,zlg,xxs(1)**2,iepss,ier0)
		cs(5) = -cs(5)
		ipi12 = -ip + 2
		ier1 = max(ier0,ier1)
		ier = ier1
		som = cs(1) + cs(2) + cs(3) + cs(4) + cs(5) +
     +			ipi12*DBLE(pi12)
		xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)),
     +			absc(cs(4)),absc(cs(5)))
		if ( lwarn .and. absc(som) .lt. xloss*xmax )
     +			call ffwarn(194,ier,absc(som),xmax)
*
		if ( lwrite ) then
		    print *,'cs  = '
		    print '(i5,2e16.8)',(i,cs(i),i=1,5),6,ipi12*pi12
		    print '(a,2e16.8,i4)','som = ',som,ier
		endif
		csom = som
		fac = -xxs(1)/(xm1*xm4*xpi(inx(ilam,ic))*xxs(2)*xxs(3))
		cfac = fac
		if ( ldot .and. abs(idot).lt.4 ) then
		    fdel4s = 1/(16*fac**2)
		    if ( lwrite ) print *,'del4s = ',fdel4s
		endif
		return
	    endif
*  #] special case m0=0, m1=m2, m3!=m4:
*  #[ special case m0=0, m1!=m2, m3!=m4:
*
*	    This also crashes...
*
	    xm0 = precx*max(xm1,xm4)
	    if ( lwrite ) print *,'ffxdir: dirty hack, put m0 != 0',xm0
	endif
*  #] special case m0=0, m1!=m2, m3!=m4:
*  #[ get dimensionless vars:
*
*	we follow the notation of Wim & Ansgar closely
*	remember that for -pi we have ieps=+2 and v.v.
*
	if ( lsmug ) then
*	    all is not what it seems
	    if ( nschem .ge. 3 ) then
		cxt = DBLE(xm0*xlam)/c2sisj(ic,ilam)
	    else
		cxt = DBLE(xm0*xlam)/DBLE(c2sisj(ic,ilam))
	    endif
	else
	    if ( dpipj(ic,inx(ilam,ic)) .eq. 0 ) then
		call fferr(73,ier)
		print *,'xpi = ',xpi
		return
	    endif
	    xxt(1) = xm0*xlam/dpipj(ic,inx(ilam,ic))
	endif
	iepst = -2
	ier1 = 0
	ier0 = 0
	call ffxkfn(xxs,iepss,xpi(inx(i1,i4)),xm1,xm4,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffxkfn(xx2,ieps2,xpi(inx(i1,ic)),xm1,xm0,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffxkfn(xx3,ieps3,xpi(inx(i4,ic)),xm4,xm0,ier0)
	ier1 = max(ier0,ier1)
	if ( ier1 .ge. 100 ) then
	    call ffzdbd(csom,cfac,xpi,dpipj,ilam,i1,i4,ic,ier)
	    return
	endif
	ier = ier + ier1
*
	if ( lwrite ) then
	    print *,'IR divergent fourpoint function according to ',
     +		'Beenakker and Denner'
	    if ( lsmug ) then
		print *,'cxt = ',cxt
	    else
		print *,'xxt = ',xxt,iepst
	    endif
	    print *,'xxs = ',xxs,iepss
	    print *,'xx2 = ',xx2,ieps2
	    print *,'xx3 = ',xx3,ieps3
	endif
*  #] get dimensionless vars:
*  #[ fill array:
*
	ier1 = 0
	ier0 = 0
	zlg = zxfflg(xxs(1),iepss,x0,ier)
	d = xxs(1)**2
	if ( abs(d) .lt. xloss ) then
	    cs(1) = 2*zlg*DBLE(dfflo1(d,ier0))
	else
	    cs(1) = 2*zlg*zxfflg(xxs(2)*xxs(3),-iepss,x0,ier0)
	endif
	ier1 = max(ier0,ier1)
	ier0 = 0
	if ( lsmug ) then
	    cs(2) = -2*zlg*zfflog(cxt,iepst,c0,ier0)
	else
	    cs(2) = -2*zlg*zxfflg(xxt(1),iepst,x0,ier0)
	endif
	ier1 = max(ier0,ier1)
*
	ipi12 = 6
*
	ier0 = 0
	call ffzxdl(cs(3),ip,zlg,xxs(1)**2,iepss,ier0)
	ipi12 = ipi12 + ip
	ier1 = max(ier0,ier1)
	ier0 = 0
	if ( abs(xx2(2)) .gt. xloss ) then
	    z = zxfflg(xx2(1),ieps2,x0,ier0)
	else
	    z = dfflo1(xx2(2),ier0)
	endif
	cs(4) = z**2
	ier1 = max(ier0,ier1)
	ier0 = 0
	if ( abs(xx3(2)) .gt. xloss ) then
	    z = zxfflg(xx3(1),ieps3,x0,ier0)
	else
	    z = dfflo1(xx3(2),ier0)
	endif
	cs(5) = z**2
	ier1 = max(ier0,ier1)
*
	is = 6
	do 110 i2=-1,+1,2
	    do 100 i3=-1,+1,2
*
		ier0 = 0
		call ffzxdl(cs(is),ip,zlg,xxs(1)*xx2(1)**i2*xx3(1)**i3,
     +			0,ier0)
		cs(is) = -cs(is)
		ipi12 = ipi12 - ip
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		if ( abs(xxs(2)) .gt. xloss ) then
		     cs(is) = -zlg*zxfflg(xxs(1),iepss,x0,ier0)
		else
		     cs(is) = -zlg*DBLE(dfflo1(xxs(2),ier0))
		endif
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		if ( abs(xx2(2)) .gt. xloss ) then
		    cs(is) = -zlg*zxfflg(xx2(1)**i2,i2*ieps2,x0,ier0)
		elseif ( i2.eq.1 ) then
		    cs(is) = -zlg*DBLE(dfflo1(xx2(2),ier0))
		else
		    cs(is) = -zlg*DBLE(dfflo1(-xx2(2)/xx2(1),ier0))
		endif
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		if ( abs(xx3(2)) .gt. xloss ) then
		    cs(is) = -zlg*zxfflg(xx3(1)**i3,i3*ieps3,x0,ier0)
		elseif ( i3.eq.1 ) then
		    cs(is) = -zlg*DBLE(dfflo1(xx3(2),ier0))
		else
		    cs(is) = -zlg*DBLE(dfflo1(-xx3(2)/xx3(1),ier0))
		endif
		is = is + 1
		ier1 = max(ier0,ier1)
*
  100	    continue
  110	continue
	ier = ier + ier1
*
*  #] fill array:
*  #[ sum:
*
	som = 0
	xmax = 0
	is = is - 1
	do 200 i=1,is
	    som = som + cs(i)
	    xmax = max(xmax,absc(cs(i)))
  200	continue
	som = som + ipi12*DBLE(pi12)
	if ( lwarn .and. absc(som) .lt. xloss*xmax )
     +		call ffwarn(194,ier,absc(som),xmax)
*
*  #] sum:
*  #[ overall factors:
*
	csom = som
	if ( lsmug ) then
	    if ( nschem .ge. 2 ) then
		cfac = -DBLE(xxs(1)/((xm1*xm4*xxs(2)*xxs(3))))/
     +			c2sisj(ilam,ic)
	    else
		cfac = -DBLE(xxs(1))/(DBLE(xm1*xm4*xxs(2)*xxs(3))*
     +			DBLE(c2sisj(ilam,ic)))
	    endif
	    if ( ldot .and. abs(idot).lt.4 ) then
		fdel4s = 16*(xm1*xm4*dpipj(inx(ilam,ic),ic)*xxs(2)*
     +			xxs(3)/xxs(1))**2
	    endif
	else
	    fac = xxs(1)/(xm1*xm4*dpipj(inx(ilam,ic),ic)*xxs(2)*xxs(3))
	    cfac = fac
	    if ( ldot .and. abs(idot).lt.4 ) then
		fdel4s = 1/(16*fac**2)
		if ( lwrite ) print *,'del4s = ',fdel4s
	    endif
	endif
*
*  #] overall factors:
*  #[ print debug info:
	if ( lwrite ) then
	    print *,'cs = '
	    do 910 i=1,is
		print *,i,cs(i)
  910	    continue
	    print *,'som = ',som,ipi12
	    print *,'cd0 = ',csom*cfac
	endif
*  #] print debug info:
*###] ffxdbd:
	end
*###[ ffxkfn:
	subroutine ffxkfn(x,ieps,xpi,xm,xmp,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the K-function in this paper:				*
*									*
*			      1-sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*		K(p^2,m,mp) = -----------------------------		*
*			      1+sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*									*
*	and fill x(1) = -K, x(2) = 1+K, x(3) = 1-K			*
*	ieps gives the sign of the imaginary part: -2 -> +ieps and v.v. *
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ieps,ier
	DOUBLE PRECISION x(3),xpi,xm,xmp
*
*	local variables
*
	DOUBLE PRECISION wortel,xx1,xx2,xx3
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ work:
*
*	special case
*
	if ( xpi.eq.0 .and. xm.eq.xmp ) then
	    x(1) = 1
	    x(2) = 0
	    x(3) = 2
	    return
	endif
*
*	normal case
*
	xx1 = xpi - (xm-xmp)**2
	if ( lwarn .and. abs(xx1) .lt. xloss*max(abs(xpi),xm**2)
     +		) then
	    call ffwarn(178,ier,xx1,max(xpi,xm**2))
	    if ( lwrite ) print *,'need extra input'
	endif
	xx2 = 1 - 4*xm*xmp/xx1
	if ( lwarn .and. abs(xx2) .lt. xloss )
     +	    call ffwarn(179,ier,xx2,x1)
	if ( xx2 .lt. 0 ) then
	    if ( lwrite ) then
		print *,'ffxkfn: cannot handle s < 4*m*mp, to ffzdbd'
		print *,'  s,m,mp = ',xpi,xm,xmp
	    endif
	    ier = ier + 100
	    return
	endif
	wortel = sqrt(xx2)
	xx3 = 1/(1+wortel)
	x(1) = -4*xm*xmp*xx3**2/xx1
	x(2) = 2*xx3
	x(3) = 2*wortel*xx3
*
	ieps = -2
*
*  #] work:
*  #[ print output:
	if ( lwrite ) then
	    print *,'ffxkfn: input: xpi,xm,xmp = ',xpi,xm,xmp
	    print *,'        output: x,ier = ',x,ier
	endif
*  #] print output:
*###] ffxkfn:
	end
*###[ ffzdbd:
	subroutine ffzdbd(csom,cfac,xpi,dpipj,ilam,i1,i4,ic,ier)
***#[*comment:***********************************************************
*									*
*	The IR divergent fourpoint function with real masses		*
*	according to Beenakker & Denner, Nucl.Phys.B338(1990)349.	*
*	but in the case at least one of the roots is complex		*
*									*
*	Input:	xpi(13)	      real	momenta^2			*
*		dpipj(10,13)  real	xpi(i)-xpi(j)			*
*		ilam	      integer	position of m=0			*
*		i1,i4	      integer	position of other 2 IR masses	*
*		ic	      integer	position of complex mass	*
*	/ffcut/	delta	      real	cutoff to use instead of lam^2	*
*									*
*	Output:	csom,cfac	      complex	D0  = csom*cfac		*
*		ier	      integer	number of digits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ilam,i1,i4,ic,ier
	DOUBLE COMPLEX csom,cfac
	DOUBLE PRECISION xpi(13),dpipj(10,13)
*
*	local variables
*
	integer ier0,ier1,ipi12,ip,init,is,i2,i3,i,iepst,iepss,ieps2,
     +		ieps3
	DOUBLE PRECISION absc,xmax
	DOUBLE PRECISION xm0,xm1,xm4,xlam,xxt(1)
	DOUBLE COMPLEX c,cs(21),z,zlg,som,cxt,cxs(3),cx2(3),cx3(3)
	DOUBLE COMPLEX zxfflg,zfflog,zfflo1
	save init
*
*	common blocks
*
	include 'ff.h'
	DOUBLE PRECISION delta
	common /ffcut/ delta
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*	data
*
	data init /0/
*
*  #] declarations:
*  #[ check input:
*
	if ( init .eq. 0 ) then
	    init = 1
	    print *,'ffzdbd: using IR cutoff delta = lam^2 = ',delta
	endif
	if ( lwrite ) then
	    print *,'ffzdbd: input: ilam,i1,i4,ic = ',ilam,i1,i4,ic
	endif
	if ( ltest ) then
	    if ( delta .eq. 0 ) print *,'ffzdbd: error: (IR)delta = 0!'
	    if ( xpi(ilam) .ne. 0 ) print *,'ffzdbd: error: lam != 0 ',
     +		ilam,xpi(ilam)
	    if ( dpipj(i1,inx(ilam,i1)) .ne. 0 ) print *,
     +		'ffzdbd: error: m1^2 != p1^2 ',i1,inx(ilam,i1),xpi(i1),
     +		xpi(inx(ilam,i1)),dpipj(i1,inx(ilam,i1))
	    if ( dpipj(i4,inx(ilam,i4)) .ne. 0 ) print *,
     +		'ffzdbd: error: m4^2 != p4^2 ',i4,inx(ilam,i4),xpi(i4),
     +		xpi(inx(ilam,i4)),dpipj(i4,inx(ilam,i4))
	endif
*
*  #] check input:
*  #[ preliminaries:
*
	xm0 = sqrt(xpi(ic))
	xm1 = sqrt(xpi(i1))
	xm4 = sqrt(xpi(i4))
	xlam = sqrt(delta)
*
*  #] preliminaries:
*  #[ special case m0=0, m1=m2, m3!=m4:
*	UNPHYSICAL!
*	if ( xpi(ic) .eq. 0 ) then
*	    if ( dpipj(i1,inx(ic,i1)).eq.0 .or.
*     +		 dpipj(i4,inx(ic,i4)).eq.0 ) then
*		if ( dpipj(i1,inx(ic,i1)).ne.0 ) then
*		    i = i4
*		    i4 = i1
*		    i1 = i
*		endif
*		if ( lwrite ) print *,'ffzdbd: special case m0=0, ',
*     +			'm1=m2 but m3!=m4'
**
*		From Wim Beenakker, Priv.Comm.
**
*		call ffzkfn(cxs,iepss,xpi(inx(i1,i4)),xm1,xm4,ier)
*		ier0 = ier
*		ier1 = ier
*		if ( absc(cxs(2)).gt.xloss ) then
*		    zlg = zfflog(cxs(1),iepss,c0,ier0)
*		else
*		    zlg = zfflo1(cxs(2),ier0)
*		endif
*		cs(1) = zlg**2
*		ier1 = max(ier0,ier1)
*		ier0 = ier
*		if ( absc(cxs(1))**2.lt.xloss ) then
*		    cs(2) = -2*zfflo1(cxs(1)**2,ier0)*zlg
*		else
*		    cs(2) = -2*zfflog(cxs(2)*cxs(3),0,c0,ier0)*zlg
*		endif
*		ier1 = max(ier0,ier1)
*		ier0 = ier
*		cs(3) = zxfflg(delta/xpi(i4),0,x0,ier0)*zlg
*		ier1 = max(ier0,ier1)
*		ier0 = ier
*		cs(4) = 2*zxfflg(dpipj(inx(ic,i4),i4)/xpi(inx(ilam,ic)),
*     +			-1,dpipj(inx(ic,i4),i4),ier0)*zlg
*		ier1 = max(ier0,ier1)
*		ier0 = ier
*		call ffzzdl(cs(5),ip,zlg,cxs(1)**2,ier0)
*		cs(5) = -cs(5)
*		ipi12 = -ip + 2
*		ier1 = max(ier0,ier1)
*		ier = ier1
*		som = cs(1) + cs(2) + cs(3) + cs(4) + cs(5) +
*     +			ipi12*DBLE(pi12)
*		xmax = max(absc(cs(1)),absc(cs(2)),absc(cs(3)),
*     +			absc(cs(4)),absc(cs(5)))
*		if ( lwarn .and. absc(som) .lt. xloss*xmax )
*     +			call ffwarn(194,ier,absc(som),xmax)
**
*		if ( lwrite ) then
*		    print *,'cs  = '
*		    print '(i5,2e16.8)',(i,cs(i),i=1,5),6,ipi12*pi12
*		    print '(a,2e16.8,i4)','som = ',som,ier
*		endif
*		csom = som
*		cfac = -cxs(1)/(xm1*xm4*xpi(inx(ilam,ic))*cxs(2)*cxs(3))
*		if ( ldot .and. abs(idot).lt.4 ) then
*		    fdel4s = 1/(16*DBLE(cfac)**2)
*		    if ( xloss*abs(DIMAG(cfac)) .gt. precc*abs(DBLE(cfac
*     +			)) ) then
*			print *,'ffzdbd: error: fac is not real: ',cfac
*		    endif
*		    if ( lwrite ) print *,'del4s = ',fdel4s
*		endif
*		return
*	    endif
**
*	    otherwise the normal case is OK
**
*	endif
*  #] special case m0=0, m1=m2, m3!=m4:
*  #[ get dimensionless vars:
*
*	we follow the notation of Wim & Ansgar closely
*	remember that for -pi we have ieps=+2 and v.v.
*
	if ( lsmug ) then
*	    all is not what it seems
	    if ( nschem .ge. 3 ) then
		cxt = DBLE(xm0*xlam)/c2sisj(ic,ilam)
	    else
		cxt = DBLE(xm0*xlam)/DBLE(c2sisj(ic,ilam))
	    endif
	else
	    xxt(1) = xm0*xlam/dpipj(ic,inx(ilam,ic))
	endif
	iepst = -2
	ier1 = 0
	ier0 = 0
	call ffzkfn(cxs,iepss,xpi(inx(i1,i4)),xm1,xm4,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffzkfn(cx2,ieps2,xpi(inx(i1,ic)),xm1,xm0,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffzkfn(cx3,ieps3,xpi(inx(i4,ic)),xm4,xm0,ier0)
	ier1 = max(ier0,ier1)
	ier = ier + ier1
*
	if ( lwrite ) then
	    print *,'IR divergent fourpoint function according to ',
     +		'Beenakker and Denner'
	    if ( lsmug ) then
		print *,'cxt = ',cxt
	    else
		print *,'xxt = ',xxt,iepst
	    endif
	    print *,'cxs = ',cxs,iepss
	    print *,'cx2 = ',cx2,ieps2
	    print *,'cx3 = ',cx3,ieps3
	endif
*  #] get dimensionless vars:
*  #[ fill array:
*
	ier1 = 0
	ier0 = 0
	zlg = zfflog(cxs(1),iepss,c0,ier)
	c = cxs(1)**2
	if ( absc(c) .lt. xloss ) then
	    cs(1) = 2*zlg*zfflo1(c,ier0)
	else
	    cs(1) = 2*zlg*zfflog(cxs(2)*cxs(3),-iepss,c0,ier0)
	endif
	ier1 = max(ier0,ier1)
	ier0 = 0
	if ( lsmug ) then
	    cs(2) = -2*zlg*zfflog(cxt,iepst,c0,ier0)
	else
	    cs(2) = -2*zlg*zxfflg(xxt(1),iepst,x0,ier0)
	endif
	ier1 = max(ier0,ier1)
*
	ipi12 = 6
*
	ier0 = 0
	call ffzzdl(cs(3),ip,zlg,cxs(1)**2,ier0)
	ipi12 = ipi12 + ip
	ier1 = max(ier0,ier1)
	ier0 = 0
	z = zfflog(cx2(1),ieps2,c0,ier0)
	cs(4) = z**2
	ier1 = max(ier0,ier1)
	ier0 = 0
	z = zfflog(cx3(1),ieps3,c0,ier0)
	cs(5) = z**2
	ier1 = max(ier0,ier1)
*
	is = 6
	do 110 i2=-1,+1,2
	    do 100 i3=-1,+1,2
*
		ier0 = 0
		call ffzzdl(cs(is),ip,zlg,cxs(1)*cx2(1)**i2*cx3(1)**i3,
     +			ier0)
		cs(is) = -cs(is)
		ipi12 = ipi12 - ip
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		cs(is) = -zlg*zfflog(cxs(1),iepss,c0,ier0)
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		cs(is) = -zlg*zfflog(cx2(1)**i2,i2*ieps2,c0,ier0)
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		cs(is) = -zlg*zfflog(cx3(1)**i3,i3*ieps3,c0,ier0)
		is = is + 1
		ier1 = max(ier0,ier1)
*
  100	    continue
  110	continue
	ier = ier + ier1
*
*  #] fill array:
*  #[ sum:
*
	som = 0
	xmax = 0
	is = is - 1
	do 200 i=1,is
	    som = som + cs(i)
	    xmax = max(xmax,absc(cs(i)))
  200	continue
	som = som + ipi12*DBLE(pi12)
	if ( lwarn .and. absc(som) .lt. xloss*xmax )
     +		call ffwarn(194,ier,absc(som),xmax)
*
*  #] sum:
*  #[ overall factors:
*
	csom = som
	if ( lsmug ) then
	    if ( nschem .ge. 2 ) then
		cfac = -cxs(1)/(DBLE(xm1*xm4)*cxs(2)*cxs(3)*
     +			c2sisj(ilam,ic))
	    else
		cfac = -cxs(1)/(DBLE(xm1*xm4)*cxs(2)*cxs(3)*
     +			DBLE(c2sisj(ilam,ic)))
	    endif
	    if ( ldot .and. abs(idot).lt.4 ) then
		c = 16*(DBLE(xm1*xm4*dpipj(inx(ilam,ic),ic))*
     +			cxs(2)*cxs(3)/cxs(1))**2
		fdel4s = DBLE(c)
		if ( xloss*DIMAG(c) .gt. precc*DBLE(c) ) then
		    print *,'ffzdbd: error: Del4s is not real ',c
		endif
	    endif
	else
	    cfac = cxs(1)/(DBLE(xm1*xm4*dpipj(inx(ilam,ic),ic))*
     +		cxs(2)*cxs(3))
	    if ( ldot .and. abs(idot).lt.4 ) then
		fdel4s = 1/(16*DBLE(cfac)**2)
		if ( xloss*abs(DIMAG(cfac)) .gt. precc*abs(DBLE(cfac)) )
     +			then
		    print *,'ffzdbd: error: fac is not real: ',cfac
		endif
		if ( lwrite ) print *,'del4s = ',fdel4s
	    endif
	endif
*
*  #] overall factors:
*  #[ print debug info:
	if ( lwrite ) then
	    print *,'cs = '
	    do 910 i=1,is
		print *,i,cs(i)
  910	    continue
	    print *,'som = ',som,ipi12
	    print *,'cd0 = ',csom*cfac
	endif
*  #] print debug info:
*###] ffzdbd:
	end
*###[ ffzkfn:
	subroutine ffzkfn(cx,ieps,xpi,xm,xmp,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the K-function in this paper:				*
*									*
*			      1-sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*		K(p^2,m,mp) = -----------------------------		*
*			      1+sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*									*
*	and fill x(1) = -K, x(2) = 1+K, x(3) = 1-K			*
*	the roots are allowed to be imaginary				*
*	ieps gives the sign of the imaginary part: -2 -> +ieps and v.v. *
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ieps,ier
	DOUBLE PRECISION xpi,xm,xmp
	DOUBLE COMPLEX cx(3)
*
*	local variables
*
	DOUBLE PRECISION xx1,xx2
	DOUBLE COMPLEX wortel,cx3
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ work:
*
	xx1 = xpi - (xm-xmp)**2
	if ( lwarn .and. abs(xx1) .lt. xloss*max(abs(xpi),xm**2)
     +		) then
	    call ffwarn(178,ier,xx1,max(xpi,xm**2))
	    if ( lwrite ) print *,'need extra input'
	endif
	xx2 = 1 - 4*xm*xmp/xx1
	if ( lwarn .and. abs(xx2) .lt. xloss )
     +	    call ffwarn(179,ier,xx2,x1)
	if ( xx2 .ge. 0 ) then
	    wortel = sqrt(xx2)
	else
	    wortel = DCMPLX(DBLE(0),DBLE(sqrt(-xx2)))
	endif
	cx3 = 1/(1+wortel)
	if ( xx1.eq.0 ) then
	    print *,'ffzkfn: error: xx1=0, contact author'
	    cx(1) = 1/xclogm
	else
	    cx(1) = DBLE(-4*xm*xmp/xx1)*cx3**2
	endif
	cx(2) = 2*cx3
	cx(3) = 2*wortel*cx3
*
	ieps = -2
*
*  #] work:
*  #[ print output:
	if ( lwrite ) then
	    print *,'ffzkfn: input: xpi,xm,xmp = ',xpi,xm,xmp
	    print *,'        output: cx,ier = ',cx,ier
	endif
*  #] print output:
*###] ffzkfn:
	end
