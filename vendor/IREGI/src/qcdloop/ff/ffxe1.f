*	$Id: ffxe1.f,v 1.6 1997/04/07 19:10:57 gj Exp $
*###[ ffxe1:
	subroutine ffxe1(ce1i,ce0,del3ij,del4i,cd0i,xpi,piDpj,del4,ier)
***#[*comment:***********************************************************
*									*
*	calculate the							*
*	E1(mu) = E11*p1(mu) + E12*p2(mu) + E13*p3(mu) + E14*p4(mu)	*
*	numerically							*
*									*
*	Input:	ce0	     complex	scalar fivepoint function	*
*		cd0i(5)	     complex	scalar fourpoint functions	*
*						without s1,s2,s3,s4,s5	*
*		xpi(20)	     real	masses (1-5), momenta^2 (6-20)	*
*		piDpj(15,15) real	dotproducts as in E0		*
*		del4	     real	delta_(p1p2p3p4)^(p1p2p3p4)	*
*		ier	     integer	digits lost so far		*
*	Output:	ce1i(4)	     complex	E11,E12,E13,E14			*
*		del3ij(5,5)  real	delta(p(i+1),p(i+2),p(i+3);	*
*					      p(j+1),p(j+2),p(j+3))	*
*		del4i(4)     real	delta(s1,(p1,p2,p3,p4)-pi;	*
*					      p1,p2,p3,p4)		*
*		ier	     integer	number of dgits lost		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION del3ij(5,5),del4i(4),xpi(20),piDpj(15,15),del4
	DOUBLE COMPLEX ce1i(4),ce0,cd0i(5)
*
*	local variables
*
	integer i,j,ii(6,5),ier0,ier1,jj(10),init
	DOUBLE PRECISION xmax,absc,xheck,del4p,xlosn
	DOUBLE COMPLEX cs(11,4),cc,cnul
	save ii,init
*
*	common blocks
*
	include 'ff.h'
*
*	statement function
*
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*
*	data
*
	data init /0/
*
*  #] declarations:
*  #[ check input:
        if ( ltest ) then
	    do 1 i=1,10
		jj(i) = i+5
    1	    continue
	    ier0 = ier
	    call ffdl4p(del4p,xpi,piDpj,15,jj,ier0)
	    xheck = del4 - del4p
	    if ( xloss*abs(xheck) .gt. precx*abs(del4) ) print *,
     +		'ffxe1: error: del4 wrong ',del4,del4p,xheck
	endif
*  #] check input:
*  #[ work:
*
*	See Form job e1.frm
*  #[ e1.log:
*   E1 =
*	+ D(1)*p1(mu)*Del4^-1 * (  - 1/2*delta(p2,p3,p4,p2,p3,p4) )
*	+ D(1)*p2(mu)*Del4^-1 * ( 1/2*delta(p2,p3,p4,p1,p3,p4) )
*	+ D(1)*p3(mu)*Del4^-1 * (  - 1/2*delta(p2,p3,p4,p1,p2,p4) )
*	+ D(1)*p4(mu)*Del4^-1 * ( 1/2*delta(p2,p3,p4,p1,p2,p3) )
*	+ D(2)*p1(mu)*Del4^-1 * ( 1/2*delta(p1,p3,p4,p2,p3,p4) + 1/2*delta(p2,p3,p4,p2,p3,p4) )
*	+ D(2)*p2(mu)*Del4^-1 * (  - 1/2*delta(p1,p3,p4,p1,p3,p4) - 1/2*delta(p2,p3,p4,p1,p3,p4) )
*	+ D(2)*p3(mu)*Del4^-1 * ( 1/2*delta(p1,p3,p4,p1,p2,p4) + 1/2*delta(p2,p3,p4,p1,p2,p4) )
*	+ D(2)*p4(mu)*Del4^-1 * (  - 1/2*delta(p1,p3,p4,p1,p2,p3) - 1/2*delta(p2,p3,p4,p1,p2,p3) )
*	+ D(3)*p1(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p4,p2,p3,p4) - 1/2*delta(p1,p3,p4,p2,p3,p4) )
*	+ D(3)*p2(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p4,p1,p3,p4) + 1/2*delta(p1,p3,p4,p1,p3,p4) )
*	+ D(3)*p3(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p4,p1,p2,p4) - 1/2*delta(p1,p3,p4,p1,p2,p4) )
*	+ D(3)*p4(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p4,p1,p2,p3) + 1/2*delta(p1,p3,p4,p1,p2,p3) )
*	+ D(4)*p1(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p3,p2,p3,p4) + 1/2*delta(p1,p2,p4,p2,p3,p4) )
*	+ D(4)*p2(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p3,p1,p3,p4) - 1/2*delta(p1,p2,p4,p1,p3,p4) )
*	+ D(4)*p3(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p3,p1,p2,p4) + 1/2*delta(p1,p2,p4,p1,p2,p4) )
*	+ D(4)*p4(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p3,p1,p2,p3) - 1/2*delta(p1,p2,p4,p1,p2,p3) )
*	+ D(5)*p1(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p3,p2,p3,p4) )
*	+ D(5)*p2(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p3,p1,p3,p4) )
*	+ D(5)*p3(mu)*Del4^-1 * (  - 1/2*delta(p1,p2,p3,p1,p2,p4) )
*	+ D(5)*p4(mu)*Del4^-1 * ( 1/2*delta(p1,p2,p3,p1,p2,p3) )
*	+ E*p1(mu)*Del4^-1 * (  - delta(p1,p2,p3,p2,p3,p4)*p4.s1 + delta(p1,p2,
*	  p4,p2,p3,p4)*p3.s1 - delta(p1,p3,p4,p2,p3,p4)*p2.s1 + delta(p2,p3,p4,
*	  p2,p3,p4)*p1.s1 )
*	+ E*p2(mu)*Del4^-1 * ( delta(p1,p2,p3,p1,p3,p4)*p4.s1 - delta(p1,p2,p4,
*	  p1,p3,p4)*p3.s1 + delta(p1,p3,p4,p1,p3,p4)*p2.s1 - delta(p2,p3,p4,p1,
*	  p3,p4)*p1.s1 )
*	+ E*p3(mu)*Del4^-1 * (  - delta(p1,p2,p3,p1,p2,p4)*p4.s1 + delta(p1,p2,
*	  p4,p1,p2,p4)*p3.s1 - delta(p1,p3,p4,p1,p2,p4)*p2.s1 + delta(p2,p3,p4,
*	  p1,p2,p4)*p1.s1 )
*	+ E*p4(mu)*Del4^-1 * ( delta(p1,p2,p3,p1,p2,p3)*p4.s1 - delta(p1,p2,p4,
*	  p1,p2,p3)*p3.s1 + delta(p1,p3,p4,p1,p2,p3)*p2.s1 - delta(p2,p3,p4,p1,
*	  p2,p3)*p1.s1 );
*  #] e1.log:
*	All the contributions are quite similar.  Note that we split
*	the non-four point determinants in 2 4point determinants, this
*	is not quick but easy.
*
*	first the indices
*
	if ( init.eq.0 ) then
	    init = 1
	    do 10 i=1,5
		ii(1,i) = i+6
		if ( ii(1,i) .gt. 10 ) ii(1,i) = 6
		ii(2,i) = ii(1,i) + 1
		if ( ii(2,i) .gt. 10 ) ii(2,i) = 6
		ii(3,i) = ii(2,i) + 1
		if ( ii(3,i) .gt. 10 ) ii(3,i) = 6
		ii(4,i) = ii(3,i) + 6
		if ( ii(4,i) .gt. 15 ) ii(4,i) = 11
		ii(5,i) = ii(1,i) + 5
		ii(6,i) = ii(2,i) + 5
   10	    continue
	endif
*
*	the determinants
*
	ier1 = ier
	do 30 i=1,5
	    do 20 j=i,5
*		we do not need (3,3), but compute it anyway for export
		ier0 = ier
		idsub = idsub + 1
		call ffdl3p(del3ij(i,j),piDpj,15,ii(1,i),ii(1,j),ier0)
		del3ij(j,i) = del3ij(i,j)
		ier1 = max(ier1,ier0)
   20	    continue
   30	continue
	do 40 i=1,4
	    ier0 = ier
	    idsub = idsub + 1
	    call ffdl4s(del4i(i),xpi,piDpj,15,1,i+5,10,ier0)
	    ier1 = max(ier1,ier0)
   40	continue
	ier = ier1
*
*	the terms with D0
*
	do 100 i=1,5
	    cs(i+5,1) = 0
	    cs(i  ,1) = -cd0i(i)*DBLE(del3ij(i,1))
	    cs(i+5,2) = cs(i,1)
	    cs(i  ,2) = -cd0i(i)*DBLE(del3ij(i,2))
	    cs(i+5,3) = +cd0i(i)*DBLE(del3ij(i,5))
	    cs(i  ,3) = +cd0i(i)*DBLE(del3ij(i,4))
	    cs(i+5,4) = 0
	    cs(i  ,4) = cs(i+5,3)
  100	continue
*
*	the terms with E0
*
	do 110 i=1,4
	    cs(11,i) = 2*DBLE(del4i(i))*ce0
	    if ( mod(i,2) .eq. 0 ) cs(11,i) = -cs(11,i)
  110	continue
*
*	sum
*
	ier1 = ier
	do 130 i=1,4
	    ce1i(i) = 0
	    xmax = 0
	    do 120 j=1,11
		ce1i(i) = ce1i(i) + cs(j,i)
		xmax = max(xmax,absc(cs(j,i)))
  120	    continue
	    if ( absc(ce1i(i)) .lt. xloss*xmax ) then
		ier0 = ier
		call ffwarn(171,ier0,absc(ce1i(i)),xmax)
		ier1 = max(ier1,ier0)
	    endif
	    if ( lwrite ) then
		do 125 j=1,11
		    print *,cs(j,i)
  125		continue
		print *,'---------------- +'
		print *,ce1i(i)
	    endif
	    ce1i(i) = ce1i(i)/DBLE(2*del4)
  130	continue
	ier = ier1
*  #] work:
*  #[ test output:
	if ( ltest ) then
*	    test a few identities: 2pi.Q = N(i+1)-Ni+2s1.pi
	    xlosn = xloss*DBLE(10)**(-2-mod(ier,50))
	    do i=1,4
		do j=1,4
		    cs(j,i) = 2*piDpj(j+5,i+5)*ce1i(j)
		enddo
		cs(5,i) = -cd0i(i+1)
		cs(6,i) = +cd0i(i)
		cs(7,i) = -2*piDpj(1,i+5)*ce0
		cnul = 0
		xmax = 0
		do j=1,7
		    cnul = cnul + cs(j,i)
		    xmax = max(xmax,absc(cnul))
		enddo
	    	if ( lwrite ) print *,'ffxe1: checking E1.p(',i,'): ',
     +	    		cnul,xmax,ier
		if ( xlosn*absc(cnul).gt.precc*xmax ) then
		    print *,'ffxe1: error: E1 fails consistency check',i
		    print '(i3,2g20.12)',(j,cs(j,i),j=1,7)
		    print *,'---- +'
		    print '(a3,2g20.12,i4)','som',cnul,ier
		endif
	    enddo
	endif
*  #] test output:
*###] ffxe1:
	end
*###[ ffdl4s:
	subroutine ffdl4s(del4,xpi,piDpj,ns,is,miss1,miss2,ier)
***#[*comment:***********************************************************
*									*
*	Calculate the 4x4 determinant					*
*									*
*		p1 p2 p3 p4						*
*	\delta								*
*		si pi pj pk						*
*									*
*	with pi pj pk given by p1,p2,p3,p4,p5 with miss1,miss2 missing. *
*									*
*	Input:	xpi(ns)		real	diagonal dotproducts		*
*		piDpj(ns,ns)	real	dotproducts			*
*		ns		integer					*
*		is		integer	si=xpi(is)			*
*		miss1,miss2	integer	see above			*
*	Output:	del4		real					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ns,is,miss1,miss2,ier
	DOUBLE PRECISION del4,piDpj(ns,ns),xpi(ns)
*
*	local variables
*
	integer i,j,k,ii(4),jj(4),ipermp(4,60),mem
	parameter(mem=10)
	integer memarr(mem,4),inow,jnow,imem,jmem,memind
	DOUBLE PRECISION s(24),som,xmax,smax
	save ipermp,memarr,inow,jnow,memind
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data memind /0/
	data memarr /mem*0,mem*0,mem*1,mem*1/
	data inow,jnow /1,1/
*
*	(the permutations with 2 from each (1-5) and (6-10) are
*	still lacking)
*
	data ((ipermp(j,i),j=1,4),i=1,35)
     +		   /1,2,3,4,  2,3,4,5,  3,4,5,1,  4,5,1,2,  5,1,2,3,
     +		    6,2,3,4,                      4,5,6,2,  5,6,2,3,
     +		    1,6,3,4,                      4,5,1,6,  5,1,6,3,
     +		    1,7,3,4,  7,3,4,5,                      5,1,7,3,
     +		    1,2,7,4,  2,7,4,5,                      5,1,2,7,
     +		    1,2,8,4,  2,8,4,5,  8,4,5,1,
     +		    1,2,3,8,  2,3,8,5,  3,8,5,1,
     +		              2,3,9,5,  3,9,5,1,  9,5,1,2,
     +		              2,3,4,9,  3,4,9,1,  4,9,1,2,
     +		                        3,4,10,1, 4,10,1,2, 10,1,2,3,
     +		                        3,4,5,10, 4,5,10,2, 5,10,2,3/

	data ((ipermp(j,i),j=1,4),i=36,60)
     +		   /                    8,9,1,6,            1,6,7,8,
     +		                        8,9,10,1,           10,1,7,8,
     +		    2,7,8,9,                      9,10,2,7,
     +		    6,2,8,9,                      9,10,6,2,
     +		              3,8,9,10,                     10,6,3,8,
     +		              7,3,9,10,                     10,6,7,3,
     +		    6,7,4,9,            4,9,10,6,
     +		    6,7,8,4,            8,4,10,6,
     +		              7,8,5,10,           5,10,6,7,
     +		              7,8,9,5,            9,5,6,7,
     +		    6,7,8,9,  7,8,9,10, 8,9,10,6, 9,10,6,7, 10,6,7,8/
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffdl4s: is,miss1,miss2 = ',is,miss1,miss2
	endif
	if ( ns.ne.15 ) then
	    print *,'ffdl4s: only for ns=15, not ',ns
	    stop
	endif
*  #] check input:
*  #[ special case:
*
*	the special case (miss1,miss2 adjacent, is not between them)
*	goes to ffdl4r
*
	i = abs(miss1-miss2)
	if ( i.eq.1 .or. i.eq.4 ) then
	    if ( miss1+miss2 .ne. 16 ) then
		j = min(miss1,miss2) - 4
	    else
		j = 1
	    endif
	    if ( .not.( is .eq. j ) ) then
		if ( lwrite ) print *,'ffdl4s: using ffdl4r'
		call ffdl4r(del4,xpi,piDpj,ns,j,ier)
		return
	    endif
	endif
*  #] special case:
*  #[ out of memory:
*
*	see if we know were to start, if not: go on as last time
*
	do 5 i=1,mem
	    if ( id .eq. memarr(i,1) .and. idsub .eq. memarr(i,2) ) then
		inow = memarr(i,3)
		jnow = memarr(i,4)
		if ( lwrite ) print *,'ffdel5: found in memory'
		goto 6
	    endif
    5	continue
    6	continue
*  #] out of memory:
*  #[ big loop:
*
*	loop over all permutations of the 1,2,3,4; leave the lower side
*	for the time being
*
	imem = inow
	jmem = jnow
	del4 = 0
	xmax = 0
*
	do 110 i=1,1
	ii(1) = is
	j = 2
	do 90 k=6,10
	    if ( k .ne. miss1 .and. k .ne. miss2 ) then
		ii(j) = k
		j = j+1
	    endif
   90	continue
	if ( lwrite ) print *,'    ii= ',ii
	do 100 j=1,60
	    jj(1) = ipermp(1,jnow) + 5
	    jj(2) = ipermp(2,jnow) + 5
	    jj(3) = ipermp(3,jnow) + 5
	    jj(4) = ipermp(4,jnow) + 5
	    if ( lwrite ) print *,'    jj= ',jj
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
		smax = max(smax,abs(som))
   80	    continue
	    if ( ( inow .eq. imem .and. jnow .eq. jmem ) .or.
     +			smax .lt. xmax ) then
		del4 = som
		xmax = smax
	    endif
	    if ( lwrite ) then
		print *,'del4+',i-1,j-1,' = ',som,smax,ii,jj
	    endif
	    if ( abs(del4) .ge. xloss**2*smax ) goto 120
	    jnow = jnow + 1
	    if ( jnow .gt. 60 ) jnow = 1
  100	continue
  110	continue
	if ( lwarn ) call ffwarn(169,ier,del4,xmax)
  120	continue
*  #[ into memory:
  800	continue
	memind = memind + 1
	if ( memind .gt. mem ) memind = 1
	memarr(memind,1) = id
	memarr(memind,2) = idsub
	memarr(memind,3) = inow
	memarr(memind,4) = jnow
*  #] into memory:
*  #] big loop:
*###] ffdl4s:
	end

