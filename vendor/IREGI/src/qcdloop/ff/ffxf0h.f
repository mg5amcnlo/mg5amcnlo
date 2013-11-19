*	$Id: ffxf0h.f,v 1.2 1995/12/08 10:47:53 gj Exp $
*###[ ffdot6:
	subroutine ffdot6(piDpj,xpi,dpipj,ier)
***#[*comment:***********************************************************
*									*
*	calculate the dotproducts pi.pj with				*
*									*
*		xpi(i) = s_i			i=1,6			*
*		xpi(i) = p_i			i=7,12			*
*		xpi(i) = p_i+p_{i+1}		i=13,18			*
*		xpi(i) = p_i+p_{i+1}+p_{i+2	i=19,21			*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	DOUBLE PRECISION xpi(21),dpipj(21,21),piDpj(21,21)
*
*	local variables
*
	integer is1,is2,is3,is4,is5,ip7,ip8,ip9,ip10,ip13,ip14,ip15,
     +		ip17,ip19,ip20,i,j,igehad(21,21),itel,i1,i2,i3,i4,i5,i6,
     +		i8,i9,i13,i14,i19,n,jtel,ii1,ii2,ii3,ii4,sgn19,sgn20,
     +		sgn21,s4,s5,s13,s14,s19,ss2,ier0,ier1
*	werkt niet bij Absoft
*	parameter (locwrt=.FALSE.)
	logical locwrt
	DOUBLE PRECISION xheck,xmax
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data locwrt /.FALSE./
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    ier0 = 0
	    call ffxhck(xpi,dpipj,21,ier0)
	    if ( ier0 .ne. 0 ) print *,'Checked by ffdot6'
	endif
	if ( locwrt ) then
	    do 2 i=1,21
		do 1 j=1,21
		    igehad(j,i) = 0
    1		continue
    2	    continue
	endif
*  #] check input:
*  #[ indices:
	ier1 = ier
	do 10 is1=1,6
	    is2 = is1 + 1
	    if ( is2 .eq. 7 ) is2 = 1
	    is3 = is2 + 1
	    if ( is3 .eq. 7 ) is3 = 1
	    ip7 = is1 + 6
	    ip8 = is2 + 6
	    ip13= ip7 + 6
*
*	    we have now defined a 3point function
*
*	          | -p13
*	          |
*	         / \
*	      s1/   \s3
*	    ___/_____\___
*	    p7   s2    p8
*
*  #] indices:
*  #[ all in one vertex:
*
*	    pi.pi, si.si
*
	    piDpj(is1,is1) = xpi(is1)
	    piDpj(ip7,ip7) = xpi(ip7)
	    piDpj(ip13,ip13) = xpi(ip13)
	    if ( locwrt ) then
		igehad(is1,is1) = igehad(is1,is1) + 1
		igehad(ip7,ip7) = igehad(ip7,ip7) + 1
		igehad(ip13,ip13) = igehad(ip13,ip13) + 1
	    endif
*
*	    si.s(i+1)
*
	    if ( xpi(is2) .le. xpi(is1) ) then
		piDpj(is1,is2) = (dpipj(is1,ip7) + xpi(is2))/2
	    else
		piDpj(is1,is2) = (dpipj(is2,ip7) + xpi(is1))/2
	    endif
	    piDpj(is2,is1) = piDpj(is1,is2)
	    if ( locwrt ) then
		igehad(is1,is2) = igehad(is1,is2) + 1
		igehad(is2,is1) = igehad(is2,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(is1,is2)) .lt.
     +			xloss*min(xpi(is1),xpi(is2)) ) then
		ier0 = ier
		call ffwarn(195,ier0,piDpj(is1,is2),min(xpi(is1),
     +			xpi(is2)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is2),xpi(ip7)
	    endif
*
*	    si.s(i+2)
*
	    if ( xpi(is1) .le. xpi(is3) ) then
		piDpj(is3,is1) = (dpipj(is3,ip13) + xpi(is1))/2
	    else
		piDpj(is3,is1) = (dpipj(is1,ip13) + xpi(is3))/2
	    endif
	    piDpj(is1,is3) = piDpj(is3,is1)
	    if ( locwrt ) then
		igehad(is1,is3) = igehad(is1,is3) + 1
		igehad(is3,is1) = igehad(is3,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(is1,is3)) .lt.
     +			xloss*min(xpi(is1),xpi(is3)) ) then
		ier0 = ier
		call ffwarn(196,ier0,piDpj(is1,is3),min(xpi(is1),
     +			xpi(is3)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is3),xpi(ip13)
	    endif
*
*	    pi.si
*
	    if ( abs(xpi(ip7)) .le. xpi(is1) ) then
		piDpj(ip7,is1) = (dpipj(is2,is1) - xpi(ip7))/2
	    else
		piDpj(ip7,is1) = (dpipj(is2,ip7) - xpi(is1))/2
	    endif
	    piDpj(is1,ip7) = piDpj(ip7,is1)
	    if ( locwrt ) then
		igehad(is1,ip7) = igehad(is1,ip7) + 1
		igehad(ip7,is1) = igehad(ip7,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip7,is1)) .lt.
     +			xloss*min(abs(xpi(ip7)),xpi(is1))) then
		ier0 = ier
		call ffwarn(197,ier0,piDpj(ip7,is1),min( abs(xpi(ip7)),
     +			xpi(is1)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is2),xpi(ip7)
	    endif
*
*	    pi.s(i+1)
*
	    if ( abs(xpi(ip7)) .le. xpi(is2) ) then
		piDpj(ip7,is2) = (dpipj(is2,is1) + xpi(ip7))/2
	    else
		piDpj(ip7,is2) = (dpipj(ip7,is1) + xpi(is2))/2
	    endif
	    if ( locwrt ) then
		igehad(is2,ip7) = igehad(is2,ip7) + 1
		igehad(ip7,is2) = igehad(ip7,is2) + 1
	    endif
	    piDpj(is2,ip7) = piDpj(ip7,is2)
	    if ( lwarn .and. abs(piDpj(ip7,is2)) .lt.
     +			xloss*min(abs(xpi(ip7)),xpi(is2))) then
		ier0 = ier
		call ffwarn(198,ier0,piDpj(ip7,is2),min(abs(xpi(ip7)),
     +			xpi(is2)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is2),xpi(ip7)
	    endif
*
*	    p(i+2).s(i)
*
	    if ( abs(xpi(ip13)) .le. xpi(is1) ) then
		piDpj(ip13,is1) = -(dpipj(is1,is3) + xpi(ip13))/2
	    else
		piDpj(ip13,is1) = -(dpipj(ip13,is3) + xpi(is1))/2
	    endif
	    piDpj(is1,ip13) = piDpj(ip13,is1)
	    if ( locwrt ) then
		igehad(is1,ip13) = igehad(is1,ip13) + 1
		igehad(ip13,is1) = igehad(ip13,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip13,is1)) .lt.
     +			xloss*min(abs(xpi(ip13)),xpi(is1))) then
		ier0 = ier
		call ffwarn(199,ier0,piDpj(ip13,is1),min(abs(xpi(ip13)),
     +			xpi(is1)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is3),xpi(ip13)
	    endif
*
*	    p(i+2).s(i+2)
*
	    if ( abs(xpi(ip13)) .le. xpi(is3) ) then
		piDpj(ip13,is3) = -(dpipj(is1,is3) - xpi(ip13))/2
	    else
		piDpj(ip13,is3) = -(dpipj(is1,ip13) - xpi(is3))/2
	    endif
	    piDpj(is3,ip13) = piDpj(ip13,is3)
	    if ( locwrt ) then
		igehad(is3,ip13) = igehad(is3,ip13) + 1
		igehad(ip13,is3) = igehad(ip13,is3) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip13,is3)) .lt.
     +			xloss*min(abs(xpi(ip13)),xpi(is3))) then
		ier0 = ier
		call ffwarn(206,ier0,piDpj(ip13,is3),min(abs(xpi(ip13)),
     +			xpi(is3)))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is3),xpi(ip13)
	    endif
*  #] all in one vertex:
*  #[ all in one 3point:
*
*	    pi.s(i+2)
*
	    if ( min(abs(dpipj(is2,is1)),abs(dpipj(ip13,ip8))) .le.
     +		 min(abs(dpipj(ip13,is1)),abs(dpipj(is2,ip8))) ) then
		piDpj(ip7,is3) = (dpipj(ip13,ip8) + dpipj(is2,is1))/2
	    else
		piDpj(ip7,is3) = (dpipj(ip13,is1) + dpipj(is2,ip8))/2
	    endif
	    piDpj(is3,ip7) = piDpj(ip7,is3)
	    if ( locwrt ) then
		igehad(is3,ip7) = igehad(is3,ip7) + 1
		igehad(ip7,is3) = igehad(ip7,is3) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip7,is3)) .lt. xloss*
     +		    min(abs(dpipj(ip13,ip8)),abs(dpipj(ip13,is1)))) then
		ier0 = ier
		call ffwarn(200,ier0,piDpj(ip7,is3),
     +			min(abs(dpipj(ip13,ip8)),abs(dpipj(ip13,is1))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is2),xpi(ip8),
     +			xpi(ip13)
	    endif
*
*	    p(i+1).s(i)
*
	    if ( min(abs(dpipj(is3,is2)),abs(dpipj(ip7,ip13))) .le.
     +		 min(abs(dpipj(ip7,is2)),abs(dpipj(is3,ip13))) ) then
		piDpj(ip8,is1) = (dpipj(ip7,ip13) + dpipj(is3,is2))/2
	    else
		piDpj(ip8,is1) = (dpipj(ip7,is2) + dpipj(is3,ip13))/2
	    endif
	    piDpj(is1,ip8) = piDpj(ip8,is1)
	    if ( locwrt ) then
		igehad(is1,ip8) = igehad(is1,ip8) + 1
		igehad(ip8,is1) = igehad(ip8,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip8,is1)) .lt. xloss*
     +		    min(abs(dpipj(ip7,ip13)),abs(dpipj(ip7,is2))) ) then
		ier0 = ier
		call ffwarn(201,ier0,piDpj(ip8,is1),
     +			min(abs(dpipj(ip7,ip13)),abs(dpipj(ip7,is2))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is2),xpi(is3),xpi(ip7),
     +			xpi(ip13)
	    endif
*
*	    p(i+2).s(i+1)
*
	    if ( min(abs(dpipj(is1,is3)),abs(dpipj(ip8,ip7))) .le.
     +		 min(abs(dpipj(ip8,is3)),abs(dpipj(is1,ip7))) ) then
		piDpj(ip13,is2) = -(dpipj(ip8,ip7) + dpipj(is1,is3))/2
	    else
		piDpj(ip13,is2) = -(dpipj(ip8,is3) + dpipj(is1,ip7))/2
	    endif
	    piDpj(is2,ip13) = piDpj(ip13,is2)
	    if ( locwrt ) then
		igehad(is2,ip13) = igehad(is2,ip13) + 1
		igehad(ip13,is2) = igehad(ip13,is2) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip13,is2)) .lt. xloss*
     +		    min(abs(dpipj(ip8,ip7)),abs(dpipj(ip8,is3))) ) then
		ier0 = ier
		call ffwarn(202,ier0,piDpj(ip13,is2),
     +			min(abs(dpipj(ip8,ip7)),abs(dpipj(ip8,is3))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is2),xpi(is3),xpi(ip7),
     +			xpi(ip8)
	    endif
*  #] all in one 3point:
*  #[ all external 3point:
*
*	    pi.p(i+1)
*
	    if ( abs(xpi(ip8)) .le. abs(xpi(ip7)) ) then
		piDpj(ip7,ip8) = (dpipj(ip13,ip7) - xpi(ip8))/2
	    else
		piDpj(ip7,ip8) = (dpipj(ip13,ip8) - xpi(ip7))/2
	    endif
	    piDpj(ip8,ip7) = piDpj(ip7,ip8)
	    if ( locwrt ) then
		igehad(ip8,ip7) = igehad(ip8,ip7) + 1
		igehad(ip7,ip8) = igehad(ip7,ip8) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip7,ip8)) .lt.
     +			xloss*min(abs(xpi(ip7)),abs(xpi(ip8))) ) then
		ier0 = ier
		call ffwarn(203,ier0,piDpj(ip7,ip8),min(abs(xpi(ip7)),
     +			abs(xpi(ip8))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(ip7),xpi(ip8),xpi(ip13)
	    endif
*
*	    p(i+1).p(i+2)
*
	    if ( abs(xpi(ip13)) .le. abs(xpi(ip8)) ) then
		piDpj(ip8,ip13) = -(dpipj(ip7,ip8) - xpi(ip13))/2
	    else
		piDpj(ip8,ip13) = -(dpipj(ip7,ip13) - xpi(ip8))/2
	    endif
	    piDpj(ip13,ip8) = piDpj(ip8,ip13)
	    if ( locwrt ) then
		igehad(ip13,ip8) = igehad(ip13,ip8) + 1
		igehad(ip8,ip13) = igehad(ip8,ip13) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip8,ip13)) .lt.
     +			xloss*min(abs(xpi(ip8)),abs(xpi(ip13))) ) then
		ier0 = ier
		call ffwarn(204,ier0,piDpj(ip8,ip13),min(abs(xpi(ip8)),
     +			abs(xpi(ip13))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(ip7),xpi(ip8),xpi(ip13)
	    endif
*
*	    p(i+2).p(i)
*
	    if ( abs(xpi(ip7)) .le. abs(xpi(ip13)) ) then
		piDpj(ip13,ip7) = -(dpipj(ip8,ip13) - xpi(ip7))/2
	    else
		piDpj(ip13,ip7) = -(dpipj(ip8,ip7) - xpi(ip13))/2
	    endif
	    piDpj(ip7,ip13) = piDpj(ip13,ip7)
	    if ( locwrt ) then
		igehad(ip7,ip13) = igehad(ip7,ip13) + 1
		igehad(ip13,ip7) = igehad(ip13,ip7) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip13,ip7)) .lt.
     +			xloss*min(abs(xpi(ip13)),abs(xpi(ip7))) ) then
		ier0 = ier
		call ffwarn(205,ier0,piDpj(ip13,ip7),min(abs(xpi(ip13)),
     +			abs(xpi(ip7))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(ip7),xpi(ip8),xpi(ip13)
	    endif
*  #] all external 3point:
*  #[ the other 3point:
	    is4 = is3 + 1
	    if ( is4 .eq. 7 ) is4 = 1
	    ip9 = is3 + 6
	    ip19 = is1 + 18
	    if ( ip19.gt.21 ) then
		ip19 = ip19 - 3
		sgn19 = -1
	    else
		sgn19 = +1
	    endif
*
*	    we now work with the threepoint configuration
*
*	          |p19
*	          |
*	         / \
*	      s1/   \s4
*	    ___/_____\___
*	    p13  s3    p8
*
	    is5 = is4 + 1
	    if ( is5.gt.6 ) is5 = 1
	    ip14 = is2 + 12
	    ip15 = is3 + 12
	    ip17 = is5 + 12
*
*	    and the threepoint configuration
*
*	          |p19
*	          |
*	         / \
*	      s1/   \s4
*	    ___/_____\___
*	    p7   s2   p14
*
*
*	    and the threepoint configuration (only twice!)
*
*	          |p17
*	          |
*	         / \
*	      s1/   \s5
*	    ___/_____\___
*	    p13  s3   p15
*
*	    we forgot one s1.s4, but not too often!
*
	    if ( is1.le.3 ) then
		piDpj(ip19,ip19) = xpi(ip19)
		if ( xpi(is1).lt.xpi(is4) ) then
		    piDpj(is1,is4) = (xpi(is1) + dpipj(is4,ip19))/2
		else
		    piDpj(is1,is4) = (xpi(is4) + dpipj(is1,ip19))/2
		endif
		piDpj(is4,is1) = piDpj(is1,is4)
		if ( locwrt ) then
		    igehad(ip19,ip19) = igehad(ip19,ip19) + 1
		    igehad(is1,is4) = igehad(is1,is4) + 1
		    igehad(is4,is1) = igehad(is4,is1) + 1
		endif
		if ( lwarn .and. abs(piDpj(is4,is1)) .lt.
     +			xloss*min(abs(xpi(is4)),abs(xpi(is1))) ) then
		    ier0 = ier
		    call ffwarn(207,ier0,piDpj(is4,is1),
     +			min(abs(xpi(is4)),abs(xpi(is1))))
		    ier1 = max(ier1,ier0)
		    if (lwrite) print *,'among ',xpi(is1),xpi(is4),
     +			xpi(ip19)
		endif
	    endif
*
*	    another missing simple one
*
	    if ( xpi(is1).lt.abs(xpi(ip19)) ) then
		piDpj(is1,ip19) = (xpi(is1) + dpipj(ip19,is4))/2
	    else
		piDpj(is1,ip19) = (xpi(ip19) + dpipj(is1,is4))/2
	    endif
	    if ( sgn19.eq.+1 ) piDpj(is1,ip19) = -piDpj(is1,ip19)
	    piDpj(ip19,is1) = piDpj(is1,ip19)
	    if ( locwrt ) then
		igehad(is1,ip19) = igehad(is1,ip19) + 1
		igehad(ip19,is1) = igehad(ip19,is1) + 1
	    endif
	    if ( lwarn .and. abs(piDpj(ip19,is1)) .lt.
     +			xloss*min(abs(xpi(is1)),abs(xpi(ip19))) ) then
		ier0 = ier
		call ffwarn(207,ier0,piDpj(is1,ip19),
     +			min(abs(xpi(is1)),abs(xpi(ip19))))
		ier1 = max(ier1,ier0)
		if (lwrite) print *,'among ',xpi(is1),xpi(is4),xpi(ip19)
	    endif
*
*	    and a series in one go
*
	    do 11 itel = 1,7
		if ( itel .eq. 1 ) then
		    i1 = is1
		    i2 = is3
		    i3 = is4
		    i4 = ip13
		    s4 = 1
		    i5 = ip9
		    s5 = 1
		    i6 = ip19
		elseif ( itel .eq. 2 ) then
		    i1 = is3
		    i2 = is4
		    i3 = is1
		    i4 = ip9
		    s4 = 1
		    i5 = ip19
		    s5 = -sgn19
		    i6 = ip13
		elseif ( itel .eq. 3 ) then
		    i1 = is4
		    i2 = is1
		    i3 = is3
		    i4 = ip19
		    s4 = -sgn19
		    i5 = ip13
		    s5 = 1
		    i6 = ip9
		elseif ( itel .eq. 4 ) then
		    i1 = is1
		    i2 = is2
		    i3 = is4
		    i4 = ip7
		    s4 = 1
		    i5 = ip14
		    s5 = 1
		    i6 = ip19
		elseif ( itel .eq. 5 ) then
		    i1 = is2
		    i2 = is4
		    i3 = is1
		    i4 = ip14
		    s4 = 1
		    i5 = ip19
		    s5 = -sgn19
		    i6 = ip7
		elseif ( itel .eq. 6 ) then
		    i1 = is4
		    i2 = is1
		    i3 = is2
		    i4 = ip19
		    s4 = -sgn19
		    i5 = ip7
		    s5 = 1
		    i6 = ip14
		else
		    i1 = is1
		    i2 = is3
		    i3 = is5
		    i4 = ip13
		    s4 = 1
		    i5 = ip15
		    s5 = 1
		    i6 = ip17
		endif
*
*		in one go: the opposite sides
*
		if ( min(abs(dpipj(i3,i2)),abs(dpipj(i4,i6))) .le.
     +		     min(abs(dpipj(i4,i2)),abs(dpipj(i3,i6))) ) then
		    piDpj(i5,i1) = (dpipj(i3,i2) + dpipj(i4,i6))/2
		else
		    piDpj(i5,i1) = (dpipj(i4,i2) + dpipj(i3,i6))/2
		endif
		if ( s5.eq.-1 ) piDpj(i5,i1) = -piDpj(i5,i1)
		piDpj(i1,i5) = piDpj(i5,i1)
		if ( locwrt ) then
		    igehad(i1,i5) = igehad(i1,i5) + 1
		    igehad(i5,i1) = igehad(i5,i1) + 1
		endif
		if ( lwarn .and. abs(piDpj(i5,i1)) .lt. xloss*
     +			min(abs(dpipj(i4,i6)),abs(dpipj(i4,i2)))) then
		    ier0 = ier
		    call ffwarn(201,ier0,piDpj(i5,i1),min(abs(dpipj(i4,
     +			i6)),abs(dpipj(i4,i2))))
		    ier1 = max(ier1,ier0)
		    if (lwrite) print *,'among ',xpi(i3),xpi(i2),
     +			xpi(i4),xpi(i6)
		endif
*
*		and the remaining external ones
*
		if ( abs(xpi(i5)) .le. abs(xpi(i4)) ) then
		    piDpj(i4,i5) = (dpipj(i6,i4) - xpi(i5))/2
		else
		    piDpj(i4,i5) = (dpipj(i6,i5) - xpi(i4))/2
		endif
		if ( s4.ne.s5 ) piDpj(i4,i5) = -piDpj(i4,i5)
		piDpj(i5,i4) = piDpj(i4,i5)
		if ( locwrt ) then
		    igehad(i5,i4) = igehad(i5,i4) + 1
		    igehad(i4,i5) = igehad(i4,i5) + 1
		endif
		if ( lwarn .and. abs(piDpj(i4,i5)) .lt. xloss*
     +			min(abs(xpi(i4)),abs(xpi(i5))) ) then
		    ier0 = ier
		    call ffwarn(203,ier0,piDpj(i4,i5),
     +			min(abs(xpi(i4)),abs(xpi(i5))))
		    ier1 = max(ier1,ier0)
		    if (lwrite) print *,'among ',xpi(i4),xpi(i5),xpi(i6)
		endif
   11	    continue
*  #] the other 3point:
*  #[ 4point indices:
	    ip10 = is4+6
	    ip20 = is2+18
	    if ( ip20.gt.21 ) then
		ip20 = ip20 - 3
		sgn20 = -1
	    else
		sgn20 = +1
	    endif
	    if ( is1.le.3 ) then
		n = 3
	    else
		n = 2
	    endif
	    do 13 jtel=1,n
		if ( jtel.eq.1 ) then
		    i3 = is3
		    i4 = is4
		    i8 = ip8
		    i9 = ip9
		    i13 = ip13
		    s13 = 1
		    i14 = ip14
		    s14 = 1
		    i19 = ip19
		    s19 = -sgn19
		elseif ( jtel.eq.2 ) then
		    i3 = is3
		    i4 = is5
		    i8 = ip8
		    i9 = ip15
		    i13 = ip13
		    s13 = 1
		    i14 = ip20
		    s14 = sgn20
		    i19 = ip17
		    s19 = 1
		else
		    i3 = is4
		    i4 = is5
		    i8 = ip14
		    i9 = ip10
		    i13 = ip19
		    s13 = sgn19
		    i14 = ip20
		    s14 = sgn20
		    i19 = ip17
		    s19 = 1
		endif
*
*	        we now have the fourpoint configuration
*
*		    \i19   /i9
*		     \____/
*		     | i4 |   \
*		   s1|    |i3 |i14
*		     |____|   /
*		   p7/ s2 \i8
*		    / \__/ \
*                     i13
*
		do 12 itel = 1,2
		    if ( itel .eq. 1 ) then
			ii1 = ip7
			ii2 = i9
			ss2 = 1
			ii3 = i8
			ii4 = i19
		    else
			ii1 = i8
			ii2 = i19
			ss2 = s19
			ii3 = ip7
			ii4 = i9
		    endif
		    if ( min(abs(dpipj(ii3,i13)),abs(dpipj(ii4,i14)))
     +		    .le. min(abs(dpipj(ii4,i13)),abs(dpipj(ii3,i14))) )
     +				then
			piDpj(ii1,ii2)=(dpipj(ii3,i13)+dpipj(ii4,i14))/2
		    else
			piDpj(ii1,ii2)=(dpipj(ii4,i13)+dpipj(ii3,i14))/2
		    endif
		    if ( ss2.eq.-1 ) piDpj(ii1,ii2) = -piDpj(ii1,ii2)
		    piDpj(ii2,ii1) = piDpj(ii1,ii2)
		    if ( locwrt ) then
			igehad(ii1,ii2) = igehad(ii1,ii2) + 1
			igehad(ii2,ii1) = igehad(ii2,ii1) + 1
		    endif
		    if ( lwarn .and. abs(piDpj(ii2,ii1)) .lt. xloss*min(
     +			  abs(dpipj(ii4,i14)),abs(dpipj(ii4,i13)))) then
			ier0 = ier
			call ffwarn(208,ier0,piDpj(ii2,ii1),min(abs(
     +				dpipj(ii4,i14)),abs(dpipj(ii4,i13))))
			ier1 = max(ier1,ier0)
			if (lwrite) print *,'among ',xpi(ii3),xpi(i13),
     +				xpi(ii4),xpi(i14)
		    endif
   12		continue
*
*		we are only left with p11.p12 etc.
*
		if ( min(abs(dpipj(i19,i9)),abs(dpipj(i8,ip7))) .le.
     +		     min(abs(dpipj(i8,i9)),abs(dpipj(i19,ip7))) ) then
		    piDpj(i13,i14) = (dpipj(i8,ip7) + dpipj(i19,i9))/2
		else
		    piDpj(i13,i14) = (dpipj(i8,i9) + dpipj(i19,ip7))/2
		endif
		if ( s13.ne.s14 ) piDpj(i13,i14) = -piDpj(i13,i14)
		piDpj(i14,i13) = piDpj(i13,i14)
		if ( locwrt ) then
		    igehad(i14,i13) = igehad(i14,i13) + 1
		    igehad(i13,i14) = igehad(i13,i14) + 1
		endif
		if ( lwarn .and. abs(piDpj(i13,i14)) .lt. xloss*min(
     +			abs(dpipj(i8,ip7)),abs(dpipj(i8,i9))) ) then
		    ier0 = ier
		    call ffwarn(202,ier0,piDpj(i13,i14),
     +			min(abs(dpipj(i8,ip7)),abs(dpipj(i8,i9))))
		    if (lwrite) print *,'among ',xpi(i8),xpi(ip7),
     +			xpi(i19),xpi(i9)
		    ier1 = max(ier1,ier0)
		endif
   13	    continue
   10	continue
	ier = ier1
*  #] 4point indices:
*  #[ check:
	if ( locwrt ) then
	    print *,'We hebben gehad:'
	    print '(21i2)',igehad
	endif
	if ( ltest ) then
	    do 40 i = 1,21
*
*		sum over all (incoming) momenta => 0
*
		xheck = 0
		xmax = 0
		do 20 j=7,12
		    xheck = xheck + piDpj(j,i)
		    xmax = max(abs(piDpj(j,i)),xmax)
   20		continue
		if ( xloss*abs(xheck) .gt. precx*xmax ) print *,
     +			'ffdot6: error: dotproducts with p(',i,
     +			') wrong: (som(.p(i))<>0) ',
     +			(piDpj(i,j),j=6,10),xheck
*
*		sum over all (incoming) momentum pairs => 0
*
		xheck = 0
		xmax = 0
		do 25 j=13,18
		    xheck = xheck + piDpj(j,i)
		    xmax = max(abs(piDpj(j,i)),xmax)
   25		continue
		if ( xloss*abs(xheck) .gt. precx*xmax ) print *,
     +			'ffdot6: error: dotproducts with p(',i,
     +			') wrong: (som(.(p(i)+p(i+1)))<>0) ',
     +			(piDpj(i,j),j=11,15),xheck
*
*		check for symmetry
*
		do 30 j=1,21
		    if ( piDpj(i,j) .ne. piDpj(j,i) ) print *,
     +			'ffdot6: error: piDpj(',i,j,') <> piDpj',j,i,')'
   30		continue
*
*		check the diagonal
*
		if ( piDpj(i,i) .ne. xpi(i) ) print *,'ffdot6: error: ',
     +			'piDpj(',i,i,') <> xpi(',i,')'
		do 35 j=7,12
		    do 34 i6=1,2
			if ( i6.eq.1 ) then
*
*			    see if indeed pi+p(i+1) = p(i+5)
*
			    i2 = j+6
			    i1 = j+1
			    if ( i1 .eq. 13 ) i1 = 7
			else
*
*			    check that si+p(i+5) = s(i+1)
*
			    i2 = i1-6
			    i1 = j-6
			endif
			xheck = piDpj(j,i)+piDpj(i1,i)-piDpj(i2,i)
			xmax = max(abs(piDpj(j,i)),abs(piDpj(i2,i)),
     +				abs(piDpj(i1,i)))
			if ( xloss*abs(xheck) .gt. precx*xmax ) print *,
     +				'ffdot6: error: piDpj(',j,i,')+piDpj(',
     +				i2,i,')-piDpj(',i1,i,') <> 0',xmax,xheck
   34		    continue
   35		continue
   40	    continue
	endif
*  #] check:
*###] ffdot6:
	end
*###[ ffpi65:
	subroutine ffpi65(xpi5,dpipj5,piDpj5,xpi,dpipj,piDpj,inum,ier)
***#[*comment:***********************************************************
*									*
*	Gets the dotproducts pertaining to the five point function with	*
*	s_i missing out of the six point function dotproduct array.	*
*									*
*	Input:	xpi	real(21)	si.si,pi.pi			*
*		dpipj	real(21,21)	xpi(i) - xpi(j)			*
*		piDpj	real(21,21)	pi(i).pi(j)			*
*		inum	integer		1--6				*
*									*
*	Output:	xpi5	real(20)	five-point momenta		*
*		dpipj5	real(15,20)					*
*		piDpj5	real(15,15)					*
*		ier	integer						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer inum,ier
	DOUBLE PRECISION xpi(21),dpipj(21,21),piDpj(21,21),xpi5(20),
     +		dpipj5(15,20),piDpj5(15,15),qDq(15,15)
*
*	local variables
*
	integer i,j,iplace(15,6),isigns(15,6),ier0,i6,i7,i8,i9
	save iplace,isigns
	DOUBLE PRECISION xmax
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data iplace /
     +		2,3,4,5,6, 08,09,10,11,18, 14,15,16,20,21,
     +		1,3,4,5,6, 13,09,10,11,12, 19,15,16,17,21,
     +		1,2,4,5,6, 07,14,10,11,12, 19,20,16,17,18,
     +		1,2,3,5,6, 07,08,15,11,12, 13,20,21,17,18,
     +		1,2,3,4,6, 07,08,09,16,12, 13,14,21,19,18,
     +		1,2,3,4,5, 07,08,09,10,17, 13,14,15,19,20/
*
	data isigns /
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,-1,-1,
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,+1,-1,
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,+1,+1,
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,+1,+1,
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,-1,+1,
     +		+1,+1,+1,+1,+1, +1,+1,+1,+1,+1, +1,+1,+1,-1,-1/
*
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    ier0 = 0
	    call ffxhck(xpi,dpipj,21,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffpi65: input corrupted'
	endif
*  #] check input:
*  #[ distribute:
*
*	copy xpi(1-15)
*
	do 20 i=1,15
	    xpi5(i) = xpi(iplace(i,inum))
	    do 10 j=1,15
		dpipj5(j,i) = dpipj(iplace(j,inum),iplace(i,inum))
   10	    continue
   20	continue
*
*	these cannot be simply right now (maybe later when I add the
*	redundant pi to F0 as well)
*
	do 15 i=1,5
	    i6 = i+5
	    i7 = i6+1
	    if ( i7 .ge. 11 ) i7 = 6
	    i8 = i7+1
	    if ( i8 .ge. 11 ) i8 = 6
	    i9 = i8+1
	    if ( i9 .ge. 11 ) i9 = 6
	    xpi5(i+15) = xpi5(i6)+xpi5(i7)+xpi5(i8)-xpi5(i6+5)-
     +		xpi5(i7+5)+xpi5(i9+5)
	    xmax = max(abs(xpi5(i6)),abs(xpi5(i7)),abs(xpi5(i8)),abs(
     +			xpi5(i6+5)),abs(xpi5(i7+5)),abs(xpi5(i9+5)))
	    if ( abs(xpi5(i+15)) .lt. xloss*xmax )
     +			call ffwarn(168,ier,xpi5(i+15),xmax)
   15	continue
*
*	and the differences
*
	do 40 i=16,20
	    do 30 j=1,15
		dpipj5(j,i) = xpi5(j) - xpi5(i)
   30	    continue
   40	continue
*
*	copy the dotproducts (watch the signs of p10-p15!)
*
	do 60 i=1,15
	    do 50 j=1,15
		piDpj5(j,i) = isigns(j,inum)*isigns(i,inum)*
     +			piDpj(iplace(j,inum),iplace(i,inum))
   50	    continue
   60	continue
*  #] distribute:
*  #[ check:
	if ( lwrite ) then
	    print *,'ffpi65: xpi5 = ',xpi5
	endif
	if ( ltest ) then
	    ier0 = 0
	    call ffxhck(xpi5,dpipj5,15,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffpi65: error detected'
*
*	    check piDpj
*
	    ier0 = 0
	    call ffdot5(qDq,xpi5,dpipj5,ier0)
	    do 190 i=1,15
		do 180 j=1,15
		    if ( xloss*abs(qDq(j,i)-piDpj5(j,i)) .gt. precx*
     +			abs(qDq(j,i)) ) print *,'ffpi65: error: ',
     +			'piDpj5(',j,i,') not correct: ',piDpj5(j,i),
     +			qDq(j,i),piDpj5(j,i)-qDq(j,i)
  180		continue
  190	    continue
	endif
*  #] check:
*###] ffpi65:
	end
*###[ ffpi64:
	subroutine ffpi64(xpi4,dpipj4,piDpj4,xpi,dpipj,piDpj,inum,jnum,
     +		ier)
***#[*comment:***********************************************************
*									*
*	Gets the dotproducts pertaining to the fourpoint function with  *
*	s_i,s_j missing out of the six point function dotproduct array.	*
*									*
*	Input:	xpi	real(21)	si.si,pi.pi			*
*		dpipj	real(21,21)	xpi(i) - xpi(j)			*
*		piDpj	real(21,21)	pi(i).pi(j)			*
*		inum,jnum integer	1--6, unequal			*
*									*
*	Output:	xpi4	real(13)					*
*		dpipj4	real(10,13)					*
*		piDpj4	real(10,10)					*
*		ier	integer						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer inum,jnum,ier
	DOUBLE PRECISION xpi(21),dpipj(21,21),piDpj(21,21),xpi4(13),
     +		dpipj4(10,13),piDpj4(10,10)
*
*	local variables
*
	integer i,j,knum,iplace(11,15),isigns(11,15),ij2k(6,6),ier0
	save iplace,isigns,ij2k
	DOUBLE PRECISION xmax,qDq(10,10),xlosn
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
	data iplace /
     +		3,4,5,6, 09,10,11,21, 15,16, 00,
     +		2,4,5,6, 14,10,11,18, 20,16, 00,
     +		2,3,5,6, 08,15,11,18, 20,21, 00,
     +		2,3,4,6, 08,09,16,18, 14,21, 00,
     +		2,3,4,5, 08,09,10,20, 14,15, 00,
     +		1,4,5,6, 19,10,11,12, 17,16, 00,
     +		1,3,5,6, 13,15,11,12, 17,21, 00,
     +		1,3,4,6, 13,09,16,12, 19,21, 00,
     +		1,3,4,5, 13,09,10,17, 19,15, 00,
     +		1,2,5,6, 07,20,11,12, 17,18, 00,
     +		1,2,4,6, 07,14,16,12, 19,18, 00,
     +		1,2,4,5, 07,14,10,17, 19,20, 00,
     +		1,2,3,6, 07,08,21,12, 13,18, 00,
     +		1,2,3,5, 07,08,15,17, 13,20, 00,
     +		1,2,3,4, 07,08,09,19, 13,14, 00/
*
	data isigns /
     +		+1,+1,+1,+1, +1,+1,+1,-1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,-1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, +1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, +1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, +1,-1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,-1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,-1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,+1, -1,+1, +0,
     +		+1,+1,+1,+1, +1,+1,+1,-1, -1,+1, +0/
*
	data ij2k /
     +		 0, 1, 2, 3, 4, 5,
     +		 1, 0, 6, 7, 8, 9,
     +		 2, 6, 0,10,11,12,
     +		 3, 7,10, 0,13,14,
     +		 4, 8,11,13, 0,15,
     +		 5, 9,12,14,15, 0/
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    if ( inum.eq.jnum ) print *,'ffpi64: undefined for i=j ',
     +		inum,jnum
	    if ( inum.lt.1 .or. inum.gt.6 .or. jnum.lt.1 .or. jnum.gt.6
     +		) print *,'ffpi84: i or j out of range ',inum,jnum
	    ier0 = 0
	    call ffxhck(xpi,dpipj,21,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffpi64: dpipj corrupted'
	endif
*  #] check input:
*  #[ distribute:
	knum = ij2k(inum,jnum)
*
*	copy p5-p11
*
	do 20 i=1,10
	    xpi4(i) = xpi(iplace(i,knum))
	    do 10 j=1,10
		dpipj4(j,i) = dpipj(iplace(j,knum),iplace(i,knum))
   10	    continue
   20	continue
*
*	these cannot be simply copied I think
*
	xpi4(11) = xpi4(5)+xpi4(6)+xpi4(7)+xpi4(8)-xpi4(9)-xpi4(10)
	if ( lwarn ) then
	    xmax = max(abs(xpi4(5)),abs(xpi4(6)),abs(xpi4(7)),
     +		       abs(xpi4(8)),abs(xpi4(9)),abs(xpi4(10)))
	    if ( abs(xpi4(11)) .lt. xloss*xmax )
     +		    call ffwarn(153,ier,xpi4(11),xmax)
	endif
	xpi4(12) = -xpi4(5)+xpi4(6)-xpi4(7)+xpi4(8)+xpi4(9)+xpi4(10)
	if ( lwarn ) then
	    xmax = max(abs(xpi4(5)),abs(xpi4(6)),abs(xpi4(7)),
     +		       abs(xpi4(8)),abs(xpi4(9)),abs(xpi4(10)))
	    if ( abs(xpi4(12)) .lt. xloss*xmax )
     +		    call ffwarn(154,ier,xpi4(12),xmax)
	endif
	xpi4(13) = xpi4(5)-xpi4(6)+xpi4(7)-xpi4(8)+xpi4(9)+xpi4(10)
	if ( lwarn ) then
	    xmax = max(abs(xpi4(5)),abs(xpi4(6)),abs(xpi4(7)),
     +		       abs(xpi4(8)),abs(xpi4(9)),abs(xpi4(10)))
	    if ( abs(xpi4(13)) .lt. xloss*xmax )
     +		    call ffwarn(155,ier,xpi4(13),xmax)
	endif
*
*	and the differences
*
	do 40 i=11,13
	    do 30 j=1,10
		dpipj4(j,i) = xpi4(j) - xpi4(i)
   30	    continue
   40	continue
*
*	copy the dotproducts (watch the signs of p9,p10!)
*
	do 60 i=1,10
	    do 50 j=1,10
		piDpj4(j,i) = isigns(j,knum)*isigns(i,knum)*
     +			piDpj(iplace(j,knum),iplace(i,knum))
   50	    continue
   60	continue
*  #] distribute:
*  #[ check:
	if ( lwrite ) then
	    print *,'ffpi64: '
	    print *,'  knum   = ',knum
	    print *,'  iplace = ',(iplace(i,knum),i=1,10)
	    print *,'  isigns = ',(isigns(i,knum),i=1,10)
	    print *,'  xpi4   = ',xpi4
	endif
	if ( ltest ) then
	    ier0 = 0
	    call ffxhck(xpi4,dpipj4,10,ier0)
	    call ffxuvw(xpi4,dpipj4,ier0)
	    if ( ier0 .ne. 0 ) print *,'ffpi64: error detected'
*
*	    check piDpj
*
	    ier0 = 0
	    call ffdot4(qDq,xpi4,dpipj4,10,ier0)
	    xlosn = xloss**2*DBLE(10)**(-mod(ier0,50))
	    do 190 i=1,10
		do 180 j=1,10
		    if ( xlosn*abs(qDq(j,i)-piDpj4(j,i)) .gt. precx*
     +			abs(qDq(j,i)) ) print *,'ffpi64: error: ',
     +			'piDpj4(',j,i,') not correct: ',piDpj4(j,i),
     +			qDq(j,i),piDpj4(j,i)-qDq(j,i)
  180		continue
  190	    continue
	endif
*  #] check:
*###] ffpi64:
	end
