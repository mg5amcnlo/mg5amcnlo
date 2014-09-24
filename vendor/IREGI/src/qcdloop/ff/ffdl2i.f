*	$Id: ffdl2i.f,v 1.4 1996/01/10 15:36:43 gj Exp $
*###[ ffdl2i:
	subroutine ffdl2i(dl2i,piDpj,ns,i1,i2,i3,isn,j1,j2,j3,jsn,ier)
***#[*comment:***********************************************************
*									*
*	calculate in a numerically stable way				*
*									*
*	   p(i1) p(i2)		with	p(i3) = isn*(p(i1)+p(i2)	*
*	del				p(j3) = jsn*(p(j1)+p(j2)	*
*	   p(j1) p(j2)							*
*									*
*	ier is the usual error flag.					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,i1,i2,i3,isn,j1,j2,j3,jsn,ier
	DOUBLE PRECISION dl2i,piDpj(ns,ns)
*
*	local variables
*
	integer i
	DOUBLE PRECISION s1,s2,del2,xmax,xnul,xlosn
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ check input:
	if ( lwrite ) then
	    print *,'ffdl2i: arbitrary 2x2 p-like determinant, ier ',ier
	    print *,'i1,i2,i3,isn = ',i1,i2,i3,isn
	    print *,'j1,j2,j3,jsn = ',j1,j2,j3,jsn
	endif
	if ( .TRUE. .or. ltest ) then
	    xlosn = max(sqrt(precx),xloss*DBLE(10)**(-2-mod(ier,50)))
	    if ( abs(isn) .ne. 1 )
     +		print *,'ffdl2i: error: |isn| != 1 ',isn
	    if ( abs(jsn) .ne. 1 )
     +		print *,'ffdl2i: error: |jsn| != 1 ',jsn
	    do 10 i=1,ns
		xnul = piDpj(i1,i) + piDpj(i2,i) - isn*piDpj(i3,i)
		xmax = max(abs(piDpj(i1,i)),abs(piDpj(i2,i)))
		if ( xlosn*abs(xnul) .gt. precx*xmax ) print *,
     +		    'ffdl2i: error: dotproducts ',i1,i2,i3,' with ',i,
     +		    ' do not add to 0',piDpj(i1,i),piDpj(i2,i),
     +		    isn*piDpj(i3,i),xnul,ier
		xnul = piDpj(j1,i) + piDpj(j2,i) - jsn*piDpj(j3,i)
		xmax = max(abs(piDpj(j1,i)),abs(piDpj(j2,i)))
		if ( xlosn*abs(xnul) .gt. precx*xmax ) print *,
     +		    'ffdl2i: error: dotproducts ',j1,j2,j3,' with ',i,
     +		    ' do not add to 0',piDpj(j1,i),piDpj(j2,i),
     +		    jsn*piDpj(j3,i),xnul,ier
   10	    continue
	endif
*  #] check input:
*  #[ stupid tree:
*
*	calculations
*
	idsub = idsub + 1
*
*	stupid tree
*
	s1 = +piDpj(i1,j1)*piDpj(i2,j2)
	s2 = -piDpj(i1,j2)*piDpj(i2,j1)
	dl2i = s1 + s2
	xmax = abs(s1)
	if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	if ( lwrite ) print *,'dl2i+1= ',dl2i,xmax
*
	s1 = +piDpj(i1,j1)*piDpj(i3,j2)
	s2 = -piDpj(i1,j2)*piDpj(i3,j1)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+2= ',del2*isn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i3,j1)*piDpj(i2,j2)
	s2 = -piDpj(i3,j2)*piDpj(i2,j1)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+3= ',del2*isn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i1,j1)*piDpj(i2,j3)
	s2 = -piDpj(i1,j3)*piDpj(i2,j1)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+4= ',del2*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i1,j1)*piDpj(i3,j3)
	s2 = -piDpj(i1,j3)*piDpj(i3,j1)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+5= ',del2*isn*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i3,j1)*piDpj(i2,j3)
	s2 = -piDpj(i3,j3)*piDpj(i2,j1)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+6= ',del2*isn*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i1,j3)*piDpj(i2,j2)
	s2 = -piDpj(i1,j2)*piDpj(i2,j3)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+7= ',del2*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i1,j3)*piDpj(i3,j2)
	s2 = -piDpj(i1,j2)*piDpj(i3,j3)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+8= ',del2*isn*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	s1 = +piDpj(i3,j3)*piDpj(i2,j2)
	s2 = -piDpj(i3,j2)*piDpj(i2,j3)
	del2 = s1 + s2
	if ( lwrite ) print *,'dl2i+9= ',del2*isn*jsn,abs(s1)
	if ( abs(s1) .lt. xmax ) then
	    dl2i = del2*isn*jsn
	    xmax = abs(s1)
	    if ( abs(dl2i) .ge. xloss*xmax ) goto 100
	endif
*
	if ( lwarn ) call ffwarn(165,ier,dl2i,xmax)
*
  100	continue
*  #] stupid tree:
*###] ffdl2i: 
	end
*###[ ffdl3q:
	subroutine ffdl3q(dl3q,piDpj,i1,i2,i3,j1,j2,j3,
     +		isn1,isn2,isn3,jsn1,jsn2,jsn3,ier)
***#[*comment:***********************************************************
*									*
*	calculate the 3x3 determinant					*
*									*
*	      p(i1) p(i2) p(i3)	      /	p(j1) = jsn1*(p(i1)-isn1*p(i2))	*
*	delta			 with |	p(j2) = jsn2*(p(i2)-isn2*p(i3))	*
*	      p5    p6    p7	      \	p(j3) = jsn3*(p(i3)-isn3*p(i1))	*
*									*
*	and piDpj(10,10) in standard four-point notation.		*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer i1,i2,i3,j1,j2,j3,isn1,isn2,isn3,jsn1,jsn2,jsn3,ier
	DOUBLE PRECISION dl3q,piDpj(10,10)
*
*	local variables
*
	logical lset
	integer ier0,ier1,i
	DOUBLE PRECISION del2i(3),s(23),xmax,xmaxp,som
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ debug input:
	if ( lwrite ) then
	    print *,'ffdl3q: determinant delta(',i1,i2,i3,';5,6,7)'
	    print *,'input: i1,i2,i3 = ',i1,i2,i3
	    print *,'input: j1,j2,j3 = ',j1,j2,j3
	    print *,'input: isigns   = ',isn1,isn2,isn3
	    print *,'input: jsigns   = ',jsn1,jsn2,jsn3
	    print *,'(p(j1) = jsn1*(p(i1)-isn1*p(i2) etc.)'
	endif
*  #] debug input:
*  #[ first try:
*
	lset = .FALSE.
	if ( isn1 .eq. -1 ) then
	    ier1 = ier
	    if ( lwrite ) print *,'ffdl2i #1'
	    call ffdl2i(del2i(1),piDpj,10, i1,i2,j1,jsn1,6,7,10,+1,ier1)
	    if ( lwrite ) print *,'ffdl2t #2'
	    ier0 = ier
	    call ffdl2t(del2i(2),piDpj,7,5, i1,i2,j1,-jsn1,-1, 10,ier0)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'ffdl2i #3'
	    ier0 = ier
	    call ffdl2i(del2i(3),piDpj,10, i1,i2,j1,jsn1,5,6,9,-1,ier0)
	    ier1 = max(ier1,ier0)
	    s(1) = piDpj(i3,5)*del2i(1)
	    s(2) = piDpj(i3,6)*del2i(2)
	    s(3) = piDpj(i3,7)*del2i(3)
	    som = s(1) + s(2) + s(3)
	    xmax = DBLE(10)**(ier1-ier)*max(abs(s(1)),abs(s(2)),
     +	    	abs(s(3)))
	    dl3q = som
	    xmaxp = xmax
	    lset = .TRUE.
	    if ( lwrite ) then
		print *,'dl3q 1 = ',dl3q,xmax
		print *,'(s     = ',s(1),s(2),s(3),')'
	    endif
	    if ( abs(dl3q) .ge. xloss*xmax ) goto 900
	endif
	if ( isn2 .eq. -1 ) then
	    ier1 = ier
	    if ( lwrite ) print *,'ffdl2i #1'
	    call ffdl2i(del2i(1),piDpj,10, i2,i3,j2,jsn2,6,7,10,+1,ier1)
	    if ( lwrite ) print *,'ffdl2t #2'
	    ier0 = ier
	    call ffdl2t(del2i(2),piDpj,7,5, i2,i3,j2,-jsn2,-1, 10,ier0)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'ffdl2i #3'
	    ier0 = ier
	    call ffdl2i(del2i(3),piDpj,10, i2,i3,j2,jsn2,5,6,9,-1,ier0)
	    ier1 = max(ier1,ier0)
	    s(1) = piDpj(i1,5)*del2i(1)
	    s(2) = piDpj(i1,6)*del2i(2)
	    s(3) = piDpj(i1,7)*del2i(3)
	    som = s(1) + s(2) + s(3)
	    xmax = DBLE(10)**(ier1-ier)*max(abs(s(1)),abs(s(2)),
     +	    	abs(s(3)))
	    if ( .not.lset ) then
		dl3q = som
		xmaxp = xmax
		lset = .TRUE.
	    elseif ( xmax .lt. xmaxp ) then
		dl3q = som
		xmaxp = xmax
	    endif
	    if ( lwrite ) then
		print *,'dl3q 2 = ',som,xmax
		print *,'(s     = ',s(1),s(2),s(3),')'
	    endif
	    if ( abs(dl3q) .ge. xloss*xmax ) goto 900
	endif
	if ( isn3 .eq. -1 ) then
	    if ( lwrite ) print *,'ffdl2i #1'
	    ier1 = ier
	    call ffdl2i(del2i(1),piDpj,10, i3,i1,j3,jsn3,6,7,10,+1,ier1)
	    if ( lwrite ) print *,'ffdl2t #2'
	    ier0 = ier
	    call ffdl2t(del2i(2),piDpj,7,5, i3,i1,j3,-jsn3,-1, 10,ier0)
	    ier1 = max(ier1,ier0)
	    if ( lwrite ) print *,'ffdl2i #3'
	    ier0 = ier
	    call ffdl2i(del2i(3),piDpj,10, i3,i1,j3,jsn3,5,6,9,-1,ier0)
	    ier1 = max(ier1,ier0)
	    s(1) = piDpj(i2,5)*del2i(1)
	    s(2) = piDpj(i2,6)*del2i(2)
	    s(3) = piDpj(i2,7)*del2i(3)
	    som = s(1) + s(2) + s(3)
	    xmax = DBLE(10)**(ier1-ier)*max(abs(s(1)),abs(s(2)),
     +	    	abs(s(3)))
	    if ( .not.lset ) then
		dl3q = som
		xmaxp = xmax
		lset = .TRUE.
	    elseif ( xmax .lt. xmaxp ) then
		dl3q = som
		xmaxp = xmax
	    endif
	    if ( lwrite ) then
		print *,'dl3q 3 = ',som,xmax
		print *,'(s     = ',s(1),s(2),s(3),')'
	    endif
	    if ( abs(dl3q) .ge. xloss*xmax ) goto 900
	endif
*  #] first try:
*  #[ last try:
	if ( .not. lset ) then
	    s(1) = + piDpj(i1,5)*piDpj(i2,6)*piDpj(i3,7)
	    s(2) = - piDpj(i1,5)*piDpj(i2,7)*piDpj(i3,6)
	    s(3) = - piDpj(i1,6)*piDpj(i2,5)*piDpj(i3,7)
	    s(4) = + piDpj(i1,6)*piDpj(i2,7)*piDpj(i3,5)
	    s(5) = + piDpj(i1,7)*piDpj(i2,5)*piDpj(i3,6)
	    s(6) = - piDpj(i1,7)*piDpj(i2,6)*piDpj(i3,5)
	    dl3q = s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
	    xmax = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +		abs(s(5)),abs(s(6)))
	    if ( lwrite ) then
		print *,'dl3q 0 = ',dl3q,xmax
		print *,'(s     = ',s(1),s(2),s(3),s(4),s(5),s(6),')'
	    endif
	    if ( abs(dl3q) .ge. xloss*xmax ) goto 900
	endif
*  #] last try:
*  #[ final:
	if ( lwarn ) call ffwarn(166,ier,dl3q,xmax)
  900	continue
*  #] final:
*  #[ check output:
	if ( ltest ) then
	    s(1) = + piDpj(i1,5)*piDpj(i2,6)*piDpj(i3,7)
	    s(2) = - piDpj(i1,5)*piDpj(i2,7)*piDpj(i3,6)
	    s(3) = - piDpj(i1,6)*piDpj(i2,5)*piDpj(i3,7)
	    s(4) = + piDpj(i1,6)*piDpj(i2,7)*piDpj(i3,5)
	    s(5) = + piDpj(i1,7)*piDpj(i2,5)*piDpj(i3,6)
	    s(6) = - piDpj(i1,7)*piDpj(i2,6)*piDpj(i3,5)
	    som = s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
	    xmaxp = max(abs(s(1)),abs(s(2)),abs(s(3)),abs(s(4)),
     +		abs(s(5)),abs(s(6)))
	    if ( lwrite ) then
		print *,'dl3q = ',som,xmaxp
	    endif
	    if ( xloss*abs(som-dl3q) .gt. precx*max(xmax,xmaxp) ) then
		print *,'ffdl3q: error: answer does not agree with ',
     +		    'normal case: ',dl3q,som,max(xmax,xmaxp),dl3q-som
	    endif
	endif
*  #] check output:
*###] ffdl3q:
	end

