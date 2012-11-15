*--#[ log:
*	$Id: ffcdbd.f,v 1.4 1997/03/27 21:28:07 gj Exp $
*	$Log: ffcdbd.f,v $
*	Revision 1.4  1997/03/27 21:28:07  gj
*	Added explicit check for mass-divergent boxes
*
c Revision 1.3  1995/12/12  12:48:13  gj
c When ndiv=-1 the D0 returns how divergent it was; E0 and F0 use this info to
c set the non-divergent ones to zero on output when ndiv>0.  Same for E0 in F0.
c
c Revision 1.2  1995/11/10  18:53:55  gj
c Added nasty D0 call
c
*--#] log:
*###[ ffcdir:
	subroutine ffcdir(cs,cfac,ldone,iir,cpi,cdpipj,ipoin,ndiv,ier)
***#[*comment:***********************************************************
*									*
*	Check if this 4point function is IRdivergent and if so, get it	*
*	using ffcdbd and set ldone to .TRUE., otherwise .FALSE.		*
*	the place of the IR divergences is returned in iir:		*
*	when iir(i,0) != 0  then iir(i,1) is the photon, iir(i,2-3) the	*
*	IR particles and iir(i,4) the other one				*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer iir(2,4),ipoin,ier,ndiv
	logical ldone
	DOUBLE COMPLEX cs,cfac,cpi(13),cdpipj(10,13)
*
*	local variables
*
	integer i,j,k,l,m,ier0,ii(6),notijk(4,4,4),ir
	DOUBLE COMPLEX dl3p
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
	ir = 1
	iir(ir,1) = 0
*
	do 25 i=1,4
	    do 24 j=1,4
		if ( j .eq. i ) goto 24
		if ( abs(DBLE(cdpipj(j,inx(j,i)))) .gt. -nwidth*
     +			DIMAG(cpi(j)) ) goto 24
		do 23 k=j+1,4
		    if ( k .eq. i ) goto 23
		    if ( abs(DBLE(cdpipj(k,inx(k,i)))) .gt. -nwidth*
     +			DIMAG(cpi(k)) ) goto 23
		    l = notijk(k,j,i)
		    if ( abs(DBLE(cpi(i))) .gt. -xloss*(DIMAG(cpi(k)) +
     +			DIMAG(cpi(j))+DIMAG(cpi(l))) ) goto 25
*
		    if ( abs(DBLE(cdpipj(l,inx(l,i)))) .le. -nwidth*
     +			DIMAG(cpi(l)) ) then
			if ( lwrite ) print *,'ffcdir: linearly IR ',
     +				'divergent ',i,j,k,l,cpi
			if ( ndiv.eq.-1 ) ndiv = 1
*
*			if possible use Wim & Ansgard's formulae
*
			if ( cpi(i).eq.0 ) then
			    if ( cdpipj(inx(i,j),j).eq.0 .and.
     +				 cdpipj(inx(i,k),k).eq.0 ) then
				call ffcdbd(cs,cfac,cpi,cdpipj,i,j,k,l,
     +					ier)
				goto 99
			    endif
			    if ( cdpipj(inx(i,j),j).eq.0 .and.
     +				 cdpipj(inx(i,l),l).eq.0 ) then
				call ffcdbd(cs,cfac,cpi,cdpipj,i,j,l,k,
     +					ier)
				goto 99
			    endif
			    if ( cdpipj(inx(i,k),k).eq.0 .and.
     +				 cdpipj(inx(i,l),l).eq.0 ) then
				call ffcdbd(cs,cfac,cpi,cdpipj,i,k,l,j,
     +					ier)
				goto 99
			    endif
			else
			    print *,'ffcdir: error: cannot handle ',
     +				'finite photon mass yet'
			    ier = ier + 100
			    cs = 0
			    cfac = 1
			    goto 99
			endif
*
*			it is thus nasty...
*
			call ffcdna(cs,cfac,cpi,cdpipj,ier)
			goto 99
		    elseif ( onshel .and. ndiv.ge.1 ) then
			if ( lwrite ) print *,'ffcdir: not divergent ',
     +				'enough, ndiv = ',ndiv
			cs = 0
			cfac = 1
			goto 99
		    endif
		    if ( lwrite ) print *,'ffcdir: IR divergent ',i,j,k,
     +			l,cpi
		    if ( ndiv.eq.-1 ) ndiv = 0
*
*		    it may be doable by W&A algorithm
*
		    if ( cdpipj(inx(i,j),j).eq.0 .and.
     +			 cdpipj(inx(i,k),k).eq.0 ) then
			call ffcdbd(cs,cfac,cpi,cdpipj,i,j,k,l,ier)
			goto 99
		    endif
*
*		    it is just a normal logarithmically divergent D0
*
		    if ( ir.gt.2 ) then
			call fferr(70,ier)
			ir = ir-1
		    endif
		    if ( DIMAG(cpi(j)).ne.0 .or. DIMAG(cpi(k)).ne.0 )
     +				then
			iir(ir,1) = i
			iir(ir,2) = j
			iir(ir,3) = k
			iir(ir,4) = l
			ir = ir + 1
			if ( ir.le.2 ) iir(ir,1) = 0
		    endif
   23		continue
   24	    continue
   25	continue
	ldone = .FALSE.
	if ( ndiv.eq.-1 ) ndiv = 0
	return
   99	continue
	if ( ldot .and. ipoin .eq. 4 ) then
	    ier0 = 0
	    call ffcot4(cfpij4,cpi,cdpipj,10,ier0)
	    do 122 l=1,10
		do 121 m=1,10
		    fpij4(m,l) = DBLE(cfpij4(m,l))
  121		continue
  122	    continue
	    ii(1)= 5
	    ii(2)= 6
	    ii(3)= 7
	    ii(4)= 8
	    ii(5)= 9
	    ii(6)= 10
	    call ffcl3p(dl3p,cfpij4,10,ii,ii,ier0)
	    fodel3 = DBLE(dl3p)
	    fdel3 = fodel3
	endif
*
*	and finito
*
	ldone = .TRUE.
	return
*  #] work:
*###] ffcdir:
	end
*###[ ffcdbd:
	subroutine ffcdbd(cs,cfac,cpi,cpipj,ilam,i1,i4,ic,ier)
***#[*comment:***********************************************************
*									*
*	The IR divergent fourpoint function with one complex mass	*
*	according to Beenakker & Denner, Nucl.Phys.B338(1990)349.	*
*									*
*	Input:	cpi(13)	      complex	momenta^2			*
*		cpipj(10,13)  complex	cpi(i)-cpi(j)			*
*		ilam	      integer	position of m=0			*
*		i1,i4	      integer	position of other 2 IR masses	*
*		ic	      integer	position of complex mass	*
*	/ffcut/	delta	      real	cutoff to use instead of lam^2	*
*									*
*	Output:	cs,cfac	      complex	D0 = cs*cfac			*
*		ier	      integer	number of digits lost		*
*									*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ilam,i1,i4,ic,ier
	DOUBLE COMPLEX cs,cfac,cpi(13),cpipj(10,13)
*
*	local variables
*
	integer ier0,ier1,ipi12,ip,init,is,i2,i3,i,j,iepss
	DOUBLE PRECISION absc,xmax,xpi(13),dpipj(10,13),xxs(3),xp,xma,
     +		xmb,d,dfflo1
	DOUBLE COMPLEX c,xxt(3),xx2(3),xx3(3),xm0,xm1,xm4,xlam,
     +		csi(21),z,zlg,zfflog,zfflo1,zxfflg
	save init
*for Absoft
*	DOUBLE COMPLEX csqrt
*	external csqrt
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
	    print *,'ffcdbd: using IR cutoff delta = lam^2 = ',delta
	endif
	if ( ltest ) then
	    if ( delta .eq. 0 ) print *,'ffcdbd: error: (IR)delta = 0!'
	    if ( max(ilam,i1,i4,ic) .gt. 4 .or. min(ilam,i1,i4,ic) .lt.
     +		1 ) print *,'ffcdbd: error: ilam,i1,i4,ic not correct ',
     +		ilam,i1,i4,ic
	    if ( cpi(ilam) .ne. 0 ) print *,'ffcdbd: error: lam != 0 ',
     +		ilam,cpi(ilam)
	    if ( cpipj(i1,inx(ilam,i1)) .ne. 0 ) print *,
     +		'ffcdbd: error: m1^2 != p1^2 ',i1,inx(ilam,i1),cpi(i1),
     +		cpi(inx(ilam,i1)),cpipj(i1,inx(ilam,i1))
	    if ( cpipj(i4,inx(ilam,i4)) .ne. 0 ) print *,
     +		'ffcdbd: error: m4^2 != p4^2 ',i4,inx(ilam,i4),cpi(i4),
     +		cpi(inx(ilam,i4)),cpipj(i4,inx(ilam,i4))
	endif
	if ( cpi(i1).eq.0 .or. cpi(i4).eq.0 ) then
	    call fferr(98,ier)
	    return
	endif
*  #] check input:
*  #[ real case:
	if ( nschem.le.3 ) then
	    do 20 i=1,13
		xpi(i) = DBLE(cpi(i))
		do 10 j=1,10
		    dpipj(j,i) = DBLE(cpipj(j,i))
   10		continue
   20	    continue
	    lsmug = .TRUE.
	    c2sisj(ilam,ic) = cpipj(ic,inx(ilam,ic))
	    c2sisj(ic,ilam) = cpipj(ic,inx(ilam,ic))
	    c2sisj(i1,ilam) = 0
	    c2sisj(ilam,i1) = 0
	    c2sisj(i4,ilam) = 0
	    c2sisj(ilam,i4) = 0
	    call ffxdbd(cs,cfac,xpi,dpipj,ilam,i1,i4,ic,ier)
	    lsmug = .FALSE.
	    return
	endif
*  #] real case:
*  #[ get dimensionless vars:
*
	xm0 = sqrt(cpi(ic))
	xm1 = sqrt(cpi(i1))
	xm4 = sqrt(cpi(i4))
	xlam = sqrt(delta)
*
*	we follow the notation of Wim & Ansgar closely
*
	xxt(1) = xm0*xlam/cpipj(ic,inx(ilam,ic))
	xxt(2) = 1-xxt(1)
	xxt(3) = 1+xxt(1)
	ier1 = 0
	ier0 = 0
*	this one is real!
	xp = DBLE(cpi(inx(i1,i4)))
	xma = DBLE(xm1)
	xmb = DBLE(xm4)
	call ffxkfn(xxs,iepss,xp,xma,xmb,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffckfn(xx2,cpi(inx(i1,ic)),xm1,xm0,ier0)
	ier1 = max(ier0,ier1)
	ier0 = 0
	call ffckfn(xx3,cpi(inx(i4,ic)),xm4,xm0,ier0)
	ier1 = max(ier0,ier1)
	ier = ier + ier1
*
	if ( lwrite ) then
	    print *,'IR divergent fourpoint function according to ',
     +		'Beenakker and Denner'
	    print *,'xxt = ',xxt
	    print *,'xxs = ',xxs
	    print *,'xx2 = ',xx2
	    print *,'xx3 = ',xx3
	endif
*
*  #] get dimensionless vars:
*  #[ fill array:
*
	ipi12 = 0
	ier1 = 0
	ier0 = 0
	zlg = zxfflg(xxs(1),iepss,x0,ier)
	d = xxs(1)**2
	if ( abs(d) .lt. xloss ) then
	    csi(1) = 2*zlg*DBLE(dfflo1(d,ier0))
	else
	    csi(1) = 2*zlg*zxfflg(xxs(2)*xxs(3),-iepss,x0,ier0)
	endif
	ier1 = max(ier0,ier1)
	ier0 = 0
	csi(2) = -2*zlg*log(xxt(1))
	ier1 = max(ier0,ier1)
*
	ipi12 = ipi12 + 6
*
	ier0 = 0
	call ffzxdl(csi(3),ip,zlg,xxs(1)**2,iepss,ier0)
	ipi12 = ipi12 + ip
	ier1 = max(ier0,ier1)
*
	ier0 = 0
	z = zfflog(xx2(1),0,c0,ier0)
	csi(4) = z**2
	ier1 = max(ier0,ier1)
*
	ier0 = 0
	z = zfflog(xx3(1),0,c0,ier0)
	csi(5) = z**2
	ier1 = max(ier0,ier1)
*
	is = 6
	do 110 i2=-1,+1,2
	    do 100 i3=-1,+1,2
*
		ier0 = 0
		call ffzzdl(csi(is),ip,zlg,DBLE(xxs(1))*xx2(1)**i2*
     +			xx3(1)**i3,ier0)
		csi(is) = -csi(is)
		ipi12 = ipi12 - ip
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		if ( abs(xxs(2)) .gt. xloss ) then
		     csi(is) = -zlg*zxfflg(xxs(1),iepss,x0,ier0)
		else
		     csi(is) = -zlg*DBLE(dfflo1(xxs(2),ier0))
		endif
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		csi(is) = -zlg*zfflog(xx2(1)**i2,0,c0,ier0)
		is = is + 1
		ier1 = max(ier0,ier1)
*
		ier0 = 0
		csi(is) = -zlg*zfflog(xx3(1)**i3,0,c0,ier0)
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
	cs = 0
	xmax = 0
	is = is - 1
	do 200 i=1,is
	    cs = cs + csi(i)
	    xmax = max(xmax,absc(csi(i)))
  200	continue
	cs = cs + ipi12*DBLE(pi12)
	if ( lwarn .and. absc(cs) .lt. xloss*xmax )
     +		call ffwarn(177,ier,absc(cs),xmax)
*
*  #] sum:
*  #[ overall factors:
*
	cfac = DBLE(xxs(1))/(xm1*xm4*cpipj(inx(ilam,ic),ic)*
     +		DBLE(xxs(2)*xxs(3)))
*
*  #] overall factors:
*  #[ print debug info:
	if ( lwrite ) then
	    print *,'csi = '
	    do 910 i=1,is
		print *,i,csi(i)
  910	    continue
	    print *,'cs  = ',cs,ipi12
	    print *,'overall factor = ',z
	    print *,'cd0 = ',cs*cfac
	endif
*  #] print debug info:
*###] ffcdbd:
	end
*###[ ffckfn:
	subroutine ffckfn(x,cpi,xm,xmp,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the K-function in this paper:				*
*									*
*			      1-sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*		K(p^2,m,mp) = -----------------------------		*
*			      1+sqrt(1-4*m*mp/(z-(m-mp)^2))		*
*									*
*	and fill x(1) = -K, x(2) = 1+K, x(3) = 1-K			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE COMPLEX x(3),cpi,xm,xmp
*
*	local variables
*
	DOUBLE PRECISION absc
	DOUBLE COMPLEX c,wortel,cc1,cc2,cc3
*for Absoft
*	DOUBLE COMPLEX csqrt
*	external csqrt
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(c) = abs(DBLE(c)) + abs(DIMAG(c))
*
*  #] declarations:
*  #[ work:
*
	cc1 = cpi - (xm-xmp)**2
	if ( lwarn .and. absc(cc1) .lt. xloss*max(absc(cpi),absc(xm)**2)
     +		) then
	    call ffwarn(178,ier,absc(cc1),max(absc(cpi),absc(xm)**2))
	    if ( lwrite ) print *,'need extra input'
	endif
	cc2 = 1 - 4*xm*xmp/cc1
	if ( lwarn .and. absc(cc2) .lt. xloss )
     +	    call ffwarn(179,ier,absc(cc1),x1)
	wortel = sqrt(cc2)
	cc3 = 1/(1+wortel)
	x(1) = -4*xm*xmp*cc3**2/cc1
	x(2) = 2*cc3
	x(3) = 2*wortel*cc3
*
*  #] work:
*###] ffckfn:
	end
