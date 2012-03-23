*	$Id: ffcel5.f,v 1.2 1995/12/08 10:37:10 gj Exp $
*###[ ffcel5:
	subroutine ffcel5(del5,xpi,pDp,ns,iquad,ier)
***#[*comment:***********************************************************
*									*
*	Calculate del5(pDp) = det(si.sj)	with			*
*	the momenta as follows:						*
*	p(1-5) = s(i)							*
*	p(5-10) = p(i)							*
*	p(11-15) = p(i)+p(i+1)						*
*									*
*	Input:	xpi(ns)		(complex)	the usual 5-pt momenta	*
*		pDp(ns,ns)	(complex)	their dot products	*
*		ns		(integer)	should be 15		*
*		iquad		(integer)	0:normal, 1:no checking	*
*						for canc., only 1 perm.	*
*		ier		(integer)	usual error flag	*
*									*
*	Output:	del5		(complex)	det(si.sj)		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,iquad,ier
	DOUBLE COMPLEX del5,xpi(15),pDp(15,15)
*
*	local variables:
*
	integer mem,nperm,nsi
	parameter(mem=10,nperm=1296,nsi=73)
	integer i,j1,j2,j3,j4,j5,iperm(5,nperm),
     +		imem,memarr(mem,3),memind,inow,init,ifile,ier0
	DOUBLE COMPLEX s(nsi),del5p,cc
	DOUBLE PRECISION xmax,xmaxp,absc
	save iperm,memind,memarr,inow,init
*
*	common blocks:
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ data:
	data memind /0/
	data memarr /mem*0,mem*0,mem*1/
	data inow /1/
	data init /0/
*
*	read permutations from file ffperm5.dat.  Included as DATA
*	statements they generated too much code in Absoft (54K)
*
	if ( init .eq. 0 ) then
	    init = 1
	    call ffopen(ifile,'ffperm5.dat',ier0)
	    if ( ier0 .ne. 0 ) goto 910
	    read(ifile,*)
	    read(ifile,*)
	    do 1 i=1,nperm,4
		read(ifile,*,err=920,end=920)
     +			((iperm(j1,j2),j1=1,5),j2=i,i+3)
    1	    continue
	    close(ifile)
	endif
*  #] data:
*  #[ check input:
	if ( ltest .and. ns .ne. 15 ) then
	    print *,'ffcel5: error: ns <> 15!'
	    stop
	endif
	if ( lwrite ) then
	    print *,'ffcel5: xpi = ',xpi
	endif
*  #] check input:
*  #[ out of memory:
*
*	see if we know were to start, if not: go on as last time
*
	if ( iquad.ne.1 ) then
	    do 5 i=1,mem
		if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) )
     +			then
		    inow = memarr(i,3)
		    if ( lwrite ) print *,'ffcel5: found in memory'
		    goto 6
		endif
    5	    continue
    6	    continue
	else
	    inow = 1
	endif
*  #] out of memory:
*  #[ calculations:
	imem = inow
	del5 = 0
	xmax = 0

   10	continue
*
*	we only try the diagonal elements: top==bottom
*
	j1 = iperm(1,inow)
	j2 = iperm(2,inow)
	j3 = iperm(3,inow)
	j4 = iperm(4,inow)
	j5 = iperm(5,inow)
*
*	The following was generated with the Form program
*	V	p1,p2,p3,p4,p5;
*	L	f = (e_(p1,p2,p3,p4,p5))**2;
*	Contract;
*	print +s;
*	.end
*	plus the substituion //p#@1\./p#@2/=/pDp(j@1,j@2)/
*
*  #[ terms:
	s(1)=+  xpi(j1)*xpi(j2)*xpi(j3)*xpi(j4)*xpi(j5)
	s(2)=-  xpi(j1)*xpi(j2)*xpi(j3)*pDp(j4,j5)**2
	s(3)=-  xpi(j1)*xpi(j2)*pDp(j3,j4)**2*xpi(j5)
	s(4)=+2*xpi(j1)*xpi(j2)*pDp(j3,j4)*pDp(j3,j5)*pDp(j4,j5)
	s(5)=-  xpi(j1)*xpi(j2)*pDp(j3,j5)**2*xpi(j4)
	s(6)=-  xpi(j1)*pDp(j2,j3)**2*xpi(j4)*xpi(j5)
	s(7)=+  xpi(j1)*pDp(j2,j3)**2*pDp(j4,j5)**2
	s(8)=+2*xpi(j1)*pDp(j2,j3)*pDp(j2,j4)*pDp(j3,j4)*xpi(j5)
	s(9)=-2*xpi(j1)*pDp(j2,j3)*pDp(j2,j4)*pDp(j3,j5)*pDp(j4,j5)
	s(10)=-2*xpi(j1)*pDp(j2,j3)*pDp(j2,j5)*pDp(j3,j4)*pDp(j4,j5)
	s(11)=+2*xpi(j1)*pDp(j2,j3)*pDp(j2,j5)*pDp(j3,j5)*xpi(j4)
	s(12)=-  xpi(j1)*pDp(j2,j4)**2*xpi(j3)*xpi(j5)
	s(13)=+  xpi(j1)*pDp(j2,j4)**2*pDp(j3,j5)**2
	s(14)=+2*xpi(j1)*pDp(j2,j4)*pDp(j2,j5)*xpi(j3)*pDp(j4,j5)
	s(15)=-2*xpi(j1)*pDp(j2,j4)*pDp(j2,j5)*pDp(j3,j4)*pDp(j3,j5)
	s(16)=-  xpi(j1)*pDp(j2,j5)**2*xpi(j3)*xpi(j4)
	s(17)=+  xpi(j1)*pDp(j2,j5)**2*pDp(j3,j4)**2
	s(18)=-  pDp(j1,j2)**2*xpi(j3)*xpi(j4)*xpi(j5)
	s(19)=+  pDp(j1,j2)**2*xpi(j3)*pDp(j4,j5)**2
	s(20)=+  pDp(j1,j2)**2*pDp(j3,j4)**2*xpi(j5)
	s(21)=-2*pDp(j1,j2)**2*pDp(j3,j4)*pDp(j3,j5)*pDp(j4,j5)
	s(22)=+  pDp(j1,j2)**2*pDp(j3,j5)**2*xpi(j4)
	s(23)=+2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j3)*xpi(j4)*xpi(j5)
	s(24)=-2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j3)*pDp(j4,j5)**2
	s(25)=-2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j4)*pDp(j3,j4)*xpi(j5)
	s(26)=+2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j4)*pDp(j3,j5)*pDp(j4,j5)
	s(27)=+2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j5)*pDp(j3,j4)*pDp(j4,j5)
	s(28)=-2*pDp(j1,j2)*pDp(j1,j3)*pDp(j2,j5)*pDp(j3,j5)*xpi(j4)
	s(29)=-2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j3)*pDp(j3,j4)*xpi(j5)
	s(30)=+2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j3)*pDp(j3,j5)*pDp(j4,j5)
	s(31)=+2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j4)*xpi(j3)*xpi(j5)
	s(32)=-2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j4)*pDp(j3,j5)**2
	s(33)=-2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j5)*xpi(j3)*pDp(j4,j5)
	s(34)=+2*pDp(j1,j2)*pDp(j1,j4)*pDp(j2,j5)*pDp(j3,j4)*pDp(j3,j5)
	s(35)=+2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j3)*pDp(j3,j4)*pDp(j4,j5)
	s(36)=-2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j3)*pDp(j3,j5)*xpi(j4)
	s(37)=-2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j4)*xpi(j3)*pDp(j4,j5)
	s(38)=+2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j4)*pDp(j3,j4)*pDp(j3,j5)
	s(39)=+2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j5)*xpi(j3)*xpi(j4)
	s(40)=-2*pDp(j1,j2)*pDp(j1,j5)*pDp(j2,j5)*pDp(j3,j4)**2
	s(41)=-  pDp(j1,j3)**2*xpi(j2)*xpi(j4)*xpi(j5)
	s(42)=+  pDp(j1,j3)**2*xpi(j2)*pDp(j4,j5)**2
	s(43)=+  pDp(j1,j3)**2*pDp(j2,j4)**2*xpi(j5)
	s(44)=-2*pDp(j1,j3)**2*pDp(j2,j4)*pDp(j2,j5)*pDp(j4,j5)
	s(45)=+  pDp(j1,j3)**2*pDp(j2,j5)**2*xpi(j4)
	s(46)=+2*pDp(j1,j3)*pDp(j1,j4)*xpi(j2)*pDp(j3,j4)*xpi(j5)
	s(47)=-2*pDp(j1,j3)*pDp(j1,j4)*xpi(j2)*pDp(j3,j5)*pDp(j4,j5)
	s(48)=-2*pDp(j1,j3)*pDp(j1,j4)*pDp(j2,j3)*pDp(j2,j4)*xpi(j5)
	s(49)=+2*pDp(j1,j3)*pDp(j1,j4)*pDp(j2,j3)*pDp(j2,j5)*pDp(j4,j5)
	s(50)=+2*pDp(j1,j3)*pDp(j1,j4)*pDp(j2,j4)*pDp(j2,j5)*pDp(j3,j5)
	s(51)=-2*pDp(j1,j3)*pDp(j1,j4)*pDp(j2,j5)**2*pDp(j3,j4)
	s(52)=-2*pDp(j1,j3)*pDp(j1,j5)*xpi(j2)*pDp(j3,j4)*pDp(j4,j5)
	s(53)=+2*pDp(j1,j3)*pDp(j1,j5)*xpi(j2)*pDp(j3,j5)*xpi(j4)
	s(54)=+2*pDp(j1,j3)*pDp(j1,j5)*pDp(j2,j3)*pDp(j2,j4)*pDp(j4,j5)
	s(55)=-2*pDp(j1,j3)*pDp(j1,j5)*pDp(j2,j3)*pDp(j2,j5)*xpi(j4)
	s(56)=-2*pDp(j1,j3)*pDp(j1,j5)*pDp(j2,j4)**2*pDp(j3,j5)
	s(57)=+2*pDp(j1,j3)*pDp(j1,j5)*pDp(j2,j4)*pDp(j2,j5)*pDp(j3,j4)
	s(58)=-  pDp(j1,j4)**2*xpi(j2)*xpi(j3)*xpi(j5)
	s(59)=+  pDp(j1,j4)**2*xpi(j2)*pDp(j3,j5)**2
	s(60)=+  pDp(j1,j4)**2*pDp(j2,j3)**2*xpi(j5)
	s(61)=-2*pDp(j1,j4)**2*pDp(j2,j3)*pDp(j2,j5)*pDp(j3,j5)
	s(62)=+  pDp(j1,j4)**2*pDp(j2,j5)**2*xpi(j3)
	s(63)=+2*pDp(j1,j4)*pDp(j1,j5)*xpi(j2)*xpi(j3)*pDp(j4,j5)
	s(64)=-2*pDp(j1,j4)*pDp(j1,j5)*xpi(j2)*pDp(j3,j4)*pDp(j3,j5)
	s(65)=-2*pDp(j1,j4)*pDp(j1,j5)*pDp(j2,j3)**2*pDp(j4,j5)
	s(66)=+2*pDp(j1,j4)*pDp(j1,j5)*pDp(j2,j3)*pDp(j2,j4)*pDp(j3,j5)
	s(67)=+2*pDp(j1,j4)*pDp(j1,j5)*pDp(j2,j3)*pDp(j2,j5)*pDp(j3,j4)
	s(68)=-2*pDp(j1,j4)*pDp(j1,j5)*pDp(j2,j4)*pDp(j2,j5)*xpi(j3)
	s(69)=-  pDp(j1,j5)**2*xpi(j2)*xpi(j3)*xpi(j4)
	s(70)=+  pDp(j1,j5)**2*xpi(j2)*pDp(j3,j4)**2
	s(71)=+  pDp(j1,j5)**2*pDp(j2,j3)**2*xpi(j4)
	s(72)=-2*pDp(j1,j5)**2*pDp(j2,j3)*pDp(j2,j4)*pDp(j3,j4)
	s(73)=+  pDp(j1,j5)**2*pDp(j2,j4)**2*xpi(j3)
*  #] terms:
*
	del5p = 0
	xmaxp = 0
	do 20 i=1,nsi
	    del5p = del5p + s(i)
	    xmaxp = max(xmaxp,absc(s(i)))
   20	continue
	if ( iquad.ne.1 .and. absc(del5p) .lt. xloss**2*xmaxp ) then
	    if ( lwrite ) print *,'del5+',inow,' = ',del5p,xmaxp,
     +		j1,j2,j3,j4,j5
	    if ( inow .eq. imem .or. xmaxp .lt. xmax ) then
		del5 = del5p
		xmax = xmaxp
	    endif
	    inow = inow + 1
	    if ( inow .gt. nperm ) inow = 1
	    if ( inow .eq. imem ) then
		if ( lwarn ) call ffwarn(160,ier,absc(del5),xmax)
		goto 800
	    endif
	    goto 10
	endif
	if ( inow .ne. imem ) then
	    if ( lwrite ) print *,'del5+',inow,' = ',del5p,xmaxp,
     +		j1,j2,j3,j4,j5
	endif
	del5 = del5p
	xmax = xmaxp
*  #] calculations:
*  #[ into memory:
  800	continue
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
*  #] into memory:
*  #[ error messages:
	return
  910	print *,'ffcel5: error: cannot open file ffperm5.dat with data'
	stop
  920	print *,'ffcel5: error: error reading from ffperm5.dat'
	stop
*  #] error messages:
*###] ffcel5:
	end
*###[ ffcl4p:
	subroutine ffcl4p(cl4p,cpi,cpiDpj,ns,ii,ier)
***#[*comment:***********************************************************
*	calculate in a numerically stable way				*
*									*
*	     p1 p2 p3 p4						*
*	delta								*
*	     p1 p2 p3 p4						*
*									*
*	with pn = xpi(ii(n)), n=1,4					*
*	     p5 = -p1-p2-p3-p4						*
*	     xpi(ii(n+5)) = pn+p(n+1), n=1,5				*
*									*
*	ier is the usual error flag.					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ii(10),ier
	DOUBLE COMPLEX cl4p,cpi(ns),cpiDpj(ns,ns)
*
*	local variables
*
	integer i,j,jj(10)
	DOUBLE PRECISION dl4p,xpi(10),piDpj(10,10),sprecx
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ calculations:
	do 20 i=1,10
	    jj(i) = i
	    xpi(i) = DBLE(cpi(ii(i)))
	    do 10 j=1,10
		piDpj(j,i) = DBLE(cpiDpj(ii(j),ii(i)))
   10	    continue
   20	continue
	sprecx = precx
	precx = precc
	call ffdl4p(dl4p,xpi,piDpj,10,jj,ier)
	cl4p = dl4p
	precx = sprecx
*  #] calculations:
*  #[ debug output:
	if ( lwrite ) then
	    print *,'ffcl4p: input'
	    print *,'ii  = ',ii
	    print *,'cpi = ',cpi
	    print *,'xpi = ',xpi
	    print *,'ffdl4s: output ',dl4p
	endif
*  #] debug output:
*###] ffcl4p:
	end
*###[ ffcl4r:
	subroutine ffcl4r(dl4r,xpi,piDpj,ns,miss,ier)
***#[*comment:***********************************************************
*	calculate in a numerically stable way				*
*									*
*	     s1 s2 s3 s4						*
*	delta								*
*	     p1 p2 p3 p4						*
*									*
*	with s(miss) NOT included					*
*									*
*	ier is the usual error flag.					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,miss,ier
	DOUBLE COMPLEX dl4r,xpi(ns),piDpj(ns,ns)
*
*	local variables
*
	integer i,j,k,ii(4),jj(4),ipermp(4,125),iperms(4,125),
     +		iplace(11,5),minus(125),mem,msign
	parameter(mem=10)
	integer memarr(mem,4),inow,jnow,imem,jmem,memind
	DOUBLE COMPLEX s(24),som,cc,cnul
	DOUBLE PRECISION xmax,smax,absc
	save ipermp,iperms,iplace,minus,memarr,inow,jnow,memind
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*  #] declarations:
*  #[ data:
	data memind /0/
	data memarr /mem*0,mem*0,mem*1,mem*1/
	data inow,jnow /1,1/
*
*	data	(see getpermp.for)
*
	data ipermp/
     +	1,2,3,4,1,2,5,3,1,2,3,8,1,2,10,3,1,2,4,5,1,2,7,4,1,2,8,4,1,2,4,
     +	9,1,2,4,10,1,2,5,7,1,2,9,5,1,2,7,8,1,2,10,7,1,2,8,9,1,2,9,10,1,
     +	3,5,4,1,3,4,6,1,3,4,7,1,3,9,4,1,3,10,4,1,3,6,5,1,3,7,5,1,3,5,8,
     +	1,3,5,9,1,3,8,6,1,3,6,10,1,3,8,7,1,3,7,10,1,3,9,8,1,3,10,8,1,3,
     +	10,9,1,4,5,6,1,4,8,5,1,4,6,7,1,4,6,8,1,4,9,6,1,4,10,6,1,4,7,8,1,
     +	4,8,9,1,4,8,10,1,5,7,6,1,5,6,9,1,5,8,7,1,5,9,8,1,6,7,8,1,6,10,7,
     +	1,6,8,9,1,6,9,10,1,7,10,8,1,8,10,9,2,3,4,5,2,3,6,4,2,3,4,9,2,3,
     +	5,6,2,3,8,5,2,3,9,5,2,3,5,10,2,3,6,8,2,3,10,6,2,3,8,9,2,3,9,10,
     +	2,4,6,5,2,4,5,7,2,4,5,8,2,4,10,5,2,4,7,6,2,4,8,6,2,4,6,9,2,4,6,
     +	10,2,4,9,7,2,4,9,8,2,4,10,9,2,5,6,7,2,5,9,6,2,5,7,8,2,5,7,9,2,5,
     +	10,7,2,5,8,9,2,5,9,10,2,6,8,7,2,6,7,10,2,6,9,8,2,6,10,9,2,7,8,9,
     +	2,7,9,10,3,4,7,5,3,4,5,10,3,4,6,7,3,4,10,6,3,4,7,9,3,4,9,10,3,5,
     +	7,6,3,5,6,10,3,5,8,7,3,5,9,7,3,5,7,10,3,5,10,8,3,5,10,9,3,6,7,8,
     +	3,6,10,7,3,6,8,10,3,7,9,8,3,7,10,9,3,8,9,10,4,5,6,7,4,5,10,6,4,
     +	5,7,8,4,5,8,10,4,6,8,7,4,6,7,9,4,6,10,8,4,6,9,10,4,7,8,9,4,8,10,
     +	9,5,6,9,7,5,6,7,10,5,6,10,9,5,7,9,8,5,7,8,10,5,8,9,10,6,7,8,9,6,
     +	7,10,8,6,7,9,10,6,8,10,9,7,8,9,10/
	data iperms/
     +	1,2,3,4,1,2,3,7,1,2,8,3,1,2,3,10,1,2,6,4,1,2,4,7,1,2,4,9,1,2,6,7
     +	,1,2,8,6,1,2,6,10,1,2,7,8,1,2,7,9,1,2,10,7,1,2,9,8,1,2,10,9,1,3,
     +	4,5,1,3,6,4,1,3,10,4,1,3,7,5,1,3,5,8,1,3,10,5,1,3,6,7,1,3,8,6,1,
     +	3,6,10,1,3,10,7,1,3,8,10,1,4,5,6,1,4,7,5,1,4,9,5,1,4,6,7,1,4,6,9
     +	,1,4,6,10,1,4,10,7,1,4,10,9,1,5,6,7,1,5,8,6,1,5,6,10,1,5,7,8,1,5
     +	,7,9,1,5,10,7,1,5,9,8,1,5,10,9,1,6,8,7,1,6,9,7,1,6,8,9,1,6,8,10,
     +	1,6,9,10,1,7,10,8,1,7,10,9,1,8,9,10,2,3,4,5,2,3,8,4,2,3,9,4,2,3,
     +	7,5,2,3,5,8,2,3,10,5,2,3,8,7,2,3,9,7,2,3,8,9,2,3,8,10,2,3,9,10,2
     +	,4,5,6,2,4,7,5,2,4,9,5,2,4,6,8,2,4,6,9,2,4,8,7,2,4,9,7,2,4,8,9,2
     +	,5,6,7,2,5,8,6,2,5,6,10,2,5,7,8,2,5,7,9,2,5,10,7,2,5,9,8,2,5,10,
     +	9,2,6,8,7,2,6,9,7,2,6,8,9,2,6,8,10,2,6,9,10,2,7,10,8,2,7,10,9,2,
     +	8,9,10,3,4,5,6,3,4,8,5,3,4,9,5,3,4,5,10,3,4,6,8,3,4,6,9,3,4,10,8
     +	,3,4,10,9,3,5,6,7,3,5,8,6,3,5,6,10,3,5,7,8,3,5,7,9,3,5,10,7,3,5,
     +	9,8,3,5,10,9,3,6,8,7,3,6,9,7,3,6,8,9,3,6,8,10,3,6,9,10,3,7,10,8,
     +	3,7,10,9,3,8,9,10,4,5,6,7,4,5,8,6,4,5,6,10,4,5,7,8,4,5,7,9,4,5,1
     +	0,7,4,5,9,8,4,5,10,9,4,6,8,7,4,6,9,7,4,6,8,9,4,6,8,10,4,6,9,10,4
     +	,7,10,8,4,7,10,9,4,8,9,10/
	data iplace /
     +		2,3,4,5, 07,08,09,15, +12,+13, 17,
     +		1,3,4,5, 11,08,09,10, -14,+13, 18,
     +		1,2,4,5, 06,12,09,10, -14,-15, 19,
     +		1,2,3,5, 06,07,13,10, +11,-15, 20,
     +		1,2,3,4, 06,07,08,14, +11,+12, 16/
	data minus /
     +		+1,+1,+1,+1,+1,+1,-1,+1,+1,+1,+1,-1,+1,-1,-1,+1,
     +		+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,+1,-1,+1,-1,+1,
     +		+1,-1,+1,+1,+1,+1,-1,+1,-1,-1,+1,-1,-1,+1,-1,+1,
     +		-1,-1,+1,+1,-1,+1,+1,+1,+1,-1,-1,+1,-1,+1,+1,-1,
     +		+1,-1,+1,-1,-1,+1,+1,+1,+1,-1,+1,-1,-1,+1,-1,-1,
     +		+1,-1,+1,-1,-1,+1,+1,-1,+1,+1,-1,+1,-1,+1,+1,+1,
     +		+1,-1,+1,-1,-1,+1,-1,-1,+1,-1,+1,-1,-1,+1,+1,+1,
     +		+1,-1,+1,-1,-1,+1,-1,-1,+1,-1,+1,-1,-1/
*  #] data:
*  #[ check input:
	if ( ltest ) then
	    if ( miss.gt.5 .or. miss.lt.1 ) then
		print *,'ffcl4r: error: miss < 1 or > 5: ',miss
		stop
	    endif
	    do 4 i=1,15
		cnul = 0
		xmax = 0
		do 1 j=6,10
		    cnul = cnul + piDpj(j,i)
		    xmax = max(xmax,absc(piDpj(j,i)))
    1		continue
		if ( xloss*absc(cnul) .gt. precx*xmax ) print *,
     +			'ffcl4r: error: sum p',i,'.p6-10 do not add ',
     +			'up to 0: ',cnul,xmax
		cnul = 0
		xmax = 0
		do 2 j=11,15
		    cnul = cnul + piDpj(j,i)
		    xmax = max(xmax,absc(piDpj(j,i)))
    2		continue
		if ( xloss*absc(cnul) .gt. precx*xmax ) print *,
     +			'ffcl4r: error: sum p',i,'.p11-15 do not add ',
     +			'up to 0: ',cnul,xmax
		do 3 j=6,10
		    k = j+1
		    if ( k.eq.11 ) k=6
		    cnul = piDpj(i,j) + piDpj(i,k) - piDpj(i,j+5)
		    xmax = max(abs(piDpj(i,j)),abs(piDpj(i,k)))
		    if ( xloss*absc(cnul) .gt. precx*xmax ) print *,
     +			'ffcl4r: error: sum p',i,'.p',j,k,j+5,' do ',
     +			'not add up to 0: ',cnul,xmax
    3		continue
    4	    continue
	endif
*  #] check input:
*  #[ out of memory:
*
*	see if we know were to start, if not: go on as last time
*
	do 5 i=1,mem
	    if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) )
     +			then
		inow = memarr(i,3)
		jnow = memarr(i,4)
		if ( lwrite ) then
		    print *,'ffcl4r: found in memory'
		    print *,'  inow, jnow = ',inow,jnow
		endif
		goto 6
	    endif
    5	continue
    6	continue
*  #] out of memory:
*  #[ calculations:
*
*	loop over all permutations of the si and the pi -
*	we have 125*125 = 15625 possibilities before we give up ....
*	15-feb-1993: well, let's only consider 25 at a time, otherwise
*		the time spent here becomes ludicrous
*
	imem = inow
	jmem = jnow
	dl4r = 0
	xmax = -1
*
	do 110 i=1,5
	ii(1) = abs(iplace((iperms(1,inow)),miss))
	ii(2) = abs(iplace((iperms(2,inow)),miss))
	ii(3) = abs(iplace((iperms(3,inow)),miss))
	ii(4) = abs(iplace((iperms(4,inow)),miss))
	msign = sign(1,iplace((iperms(1,inow)),miss))*
     +		sign(1,iplace((iperms(2,inow)),miss))*
     +		sign(1,iplace((iperms(3,inow)),miss))*
     +		sign(1,iplace((iperms(4,inow)),miss))
	do 100 j=1,5
	    jj(1) = ipermp(1,jnow) + 5
	    jj(2) = ipermp(2,jnow) + 5
	    jj(3) = ipermp(3,jnow) + 5
	    jj(4) = ipermp(4,jnow) + 5
*
	    s( 1) = +piDpj(ii(1),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(4),jj(4))
	    s( 2) = +piDpj(ii(2),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(4),jj(4))
	    s( 3) = +piDpj(ii(3),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(4),jj(4))
	    s( 4) = -piDpj(ii(1),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(4),jj(4))
	    s( 5) = -piDpj(ii(3),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(4),jj(4))
	    s( 6) = -piDpj(ii(2),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(4),jj(4))
*
	    s( 7) = -piDpj(ii(1),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(3),jj(4))
	    s( 8) = -piDpj(ii(2),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(3),jj(4))
	    s( 9) = -piDpj(ii(4),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(3),jj(4))
	    s(10) = +piDpj(ii(1),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(3),jj(4))
	    s(11) = +piDpj(ii(4),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(3),jj(4))
	    s(12) = +piDpj(ii(2),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(3),jj(4))
*
	    s(13) = -piDpj(ii(1),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(2),jj(4))
	    s(14) = -piDpj(ii(4),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(2),jj(4))
	    s(15) = -piDpj(ii(3),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(2),jj(4))
	    s(16) = +piDpj(ii(1),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(2),jj(4))
	    s(17) = +piDpj(ii(3),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(1),jj(3))*piDpj(ii(2),jj(4))
	    s(18) = +piDpj(ii(4),jj(1))*piDpj(ii(1),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(2),jj(4))
*
	    s(19) = -piDpj(ii(4),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(1),jj(4))
	    s(20) = -piDpj(ii(2),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(1),jj(4))
	    s(21) = -piDpj(ii(3),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(1),jj(4))
	    s(22) = +piDpj(ii(4),jj(1))*piDpj(ii(3),jj(2))*
     +			piDpj(ii(2),jj(3))*piDpj(ii(1),jj(4))
	    s(23) = +piDpj(ii(3),jj(1))*piDpj(ii(2),jj(2))*
     +			piDpj(ii(4),jj(3))*piDpj(ii(1),jj(4))
	    s(24) = +piDpj(ii(2),jj(1))*piDpj(ii(4),jj(2))*
     +			piDpj(ii(3),jj(3))*piDpj(ii(1),jj(4))
*
	    som = 0
	    smax = 0
	    do 80 k=1,24
		som = som + s(k)
		smax = max(smax,absc(som))
   80	    continue
	    if ( smax .lt. xmax .or. xmax .lt. 0 ) then
		dl4r = msign*minus(inow)*som
		xmax = smax
	    endif
	    if ( lwrite ) then
		print *,'dl4r+',i-1,j-1,' = ',msign*minus(inow)*som,smax
		print *,'      inow,ii = ',inow,ii
		print *,'      jnow,jj = ',jnow,jj
	    endif
	    if ( absc(dl4r) .ge. xloss**2*xmax ) goto 120
   99	    continue
*	    increase with something that is relative prime to 125 so that
*	    eventually we cover all possibilities, but with a good
*	    scatter.
	    jnow = jnow + 49
	    if ( jnow .gt. 125 ) jnow = jnow - 125
  100	continue
  109	continue
*	again, a number relative prime to 125 and a few times smaller
	inow = inow + 49
	if ( inow .gt. 125 ) inow = inow - 125
  110	continue
	if ( lwarn ) call ffwarn(169,ier,absc(dl4r),xmax)
  120	continue
*  #] calculations:
*  #[ into memory:
  800	continue
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
	memarr(memind,4) = jnow
*  #] into memory:
*###] ffcl4r:
	end

