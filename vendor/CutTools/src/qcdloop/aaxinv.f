
* file aaxinv	4-oct-1990

*###[ : aaxi3 :
	subroutine aaxi3(xi3,xpi,ier)
*###[ : comment:*******************************************************
*###] : comment:**********************************************************
*###[ : declarations :
	implicit none
* arguments
	DOUBLE PRECISION xi3(3),xpi(6)
	integer ier
* local variables
	integer i
	DOUBLE PRECISION e3(3),s1,s2,s3,xnul,xmax
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*###] : declarations :
*###[ : kinematical matrix x3 and inverse xi3:
*	the dotproducts are imported via ff.h
*	definition see ffxc0.ffdot3:comment
	s1=fpij3(4,4)
	s2=fpij3(5,5)
	s3=fpij3(4,5)
*	       inverse kinematical matrix xi3
*	       the determinant is also provided by ff
	if ( fdel2.eq.0 ) then
	    call fferr(89,ier)
	    return
	endif
	if ( atest ) then
*	    make sure that they are correct.
	    do i=4,5
		xnul = fpij3(i,i) - xpi(i)
		if ( xloss*abs(xnul).gt.precx*abs(xpi(i)) ) then
		    print *,'aaxi3: error: saved fpij3(',i,i,
     +			') does not agree with recomputed: ',
     +			fpij3(4,4),xpi(4),xnul
		endif
	    enddo
	    xnul = 2*fpij3(4,5) + xpi(4) + xpi(5) - xpi(6)
	    xmax = max(abs(xpi(4)),abs(xpi(5)),abs(xpi(6)))
	    if ( xloss*abs(xnul).gt.precx*xmax ) then
	        print *,'aaxi3: error: saved fpij3(4,5) does not ',
     +			'agree with recomputed: ',2*fpij3(4,5),
     +			xpi(6)-xpi(4)-xpi(5),xnul,xmax
	    endif
	    xnul = fdel2 - xpi(4)*xpi(5) + fpij3(4,5)**2
	    xmax = max(abs(fdel2),fpij3(4,5)**2)
	    if ( xloss*abs(xnul).gt.precx*xmax ) then
	        print *,'aaxi3: error: saved fdel2 does not ',
     +			'agree with recomputed: ',fdel2,
     +			xpi(4)*xpi(5) - fpij3(4,5)**2,xnul,xmax
	    endif
	endif
	xi3(1)= s2/fdel2
	xi3(3)=-s3/fdel2
	xi3(2)= s1/fdel2
*###] : kinematical matrix x3 and inverse xi3:
*###[ : check: on accuracy
	if ( atest ) then
	    e3(1)= s1*xi3(1)+s3*xi3(3)
	    e3(2)= s3*xi3(3)+s2*xi3(2)
	    e3(3)= s1*xi3(3)+s3*xi3(2)
	    if ( abs(e3(1)-1) .gt. .1d-4 ) then
		print *,'aaxi3: error in xi3(1) or xi3(3): ',e3(1)-1,xi3
	    endif
	    if ( abs(e3(2)-1) .gt. .1d-4 ) then
		print *,'aaxi3: error in xi3(2) or xi3(3): ',e3(2)-1,xi3
	    endif
	    if ( abs(e3(3)) .gt. .1d-4 ) then
		print *,'aaxi3: error in xi3(2) or xi3(3): ',e3(3),xi3
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'aaxi3:imported dots and inv:'
	    print *,'s..xi3 ',s1,xi3(1)
	    print *,'       ',s2,xi3(2)
	    print *,'       ',s3,xi3(3)
	    print *,' '
	endif
*###] : check:
*###] : aaxi3 :
	end

*###[ : aaxi4 :
	subroutine aaxi4(xi4,ier)
*###[ : comment:*******************************************************
*###] : comment:**********************************************************
*###[ : declarations :
	implicit none
* arguments
	DOUBLE PRECISION xi4(6)
	integer ier
* local variables
	integer i,ier0,ier1
	DOUBLE PRECISION e4(6),s1,s2,s3,s4,s5,s6,del2
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*###] : declarations :
*###[ : kinematical matrix x4 and inverse xi4:
	if ( fdel3.eq.0 ) then
	    call fferr(90,ier)
	    return
	endif
*	the dotproducts are imported via ff.h
*	definition see ffxd0.ffdot4:comment
*	       inverse kinematical matrix xi4
*	       the determinant is also provided by ff
*	xi4(1)=( +s2*s3-s6**2 )/fdel3
*	xi4(4)=( -s3*s4+s5*s6 )/fdel3
*	xi4(5)=( -s2*s5+s4*s6 )/fdel3
*	xi4(2)=( +s1*s3-s5**2 )/fdel3
*	xi4(6)=( -s1*s6+s4*s5 )/fdel3
*	xi4(3)=( +s1*s2-s4**2 )/fdel3
	ier1 = ier
*
	ier0 = ier
	call ffdel2(del2,fpij4,10,6,7,10,1,ier0)
	ier1 = max(ier0,ier1)
	xi4(1) = +del2/fdel3
*
	del2 = fpij4(5,5)*fpij4(7,7) - fpij4(5,7)**2
	if ( lwarn .and. abs(del2).lt.xloss*fpij4(5,7)**2 ) then
	    ier0 = ier
	    call ffwarn(263,ier0,del2,fpij4(5,7)**2)
	    ier1 = max(ier0,ier1)
	endif
	xi4(2) = +del2/fdel3
*
	ier0 = ier
	call ffdel2(del2,fpij4,10,5,6,9,1,ier0)
	ier1 = max(ier0,ier1)
	xi4(3) = +del2/fdel3
*
	ier0 = ier
	call ffdl2t(del2,fpij4,5,7,6,7,10,-1,-1,10,ier0)
	ier1 = max(ier1,ier0)
	xi4(4) = -del2/fdel3
*
	ier0 = ier
	call ffdl2i(del2,fpij4,10,5,6,9,-1,6,7,10,+1,ier0)
	ier1 = max(ier1,ier0)
	xi4(5) = +del2/fdel3
*
	ier0 = ier
	call ffdl2t(del2,fpij4,5,7,5,6,9,+1,-1,10,ier0)
	ier1 = max(ier1,ier0)
	xi4(6) = -del2/fdel3
*
*###] : kinematical matrix x4 and inverse xi4:
*###[ : check:
	if ( atest ) then
	    s1=fpij4(5,5)
	    s2=fpij4(6,6)
	    s3=fpij4(7,7)
	    s4=fpij4(5,6)
	    s5=fpij4(5,7)
	    s6=fpij4(6,7)
	    e4(1) = ( s1*xi4(1)+s4*xi4(4)+s5*xi4(5) )
	    e4(2) = ( s4*xi4(4)+s2*xi4(2)+s6*xi4(6) )
	    e4(3) = ( s5*xi4(5)+s6*xi4(6)+s3*xi4(3) )
	    e4(4) = ( s1*xi4(4)+s4*xi4(2)+s5*xi4(6) )
	    e4(5) = ( s1*xi4(5)+s4*xi4(6)+s5*xi4(3) )
	    e4(6) = ( s4*xi4(5)+s2*xi4(6)+s6*xi4(3) )
	    do 12 i=1,3
		if (  abs(e4(i)-1.d0) .gt. .1d-5  .or.
     +		      abs(e4(i+3) ) .gt. .1d-5 ) then
		    print *,'aaxi4: error in xi4'
		    return
		endif
 12	    continue
	endif
*###] : check:
*###] : aaxi4 :
	end

*###[ : aaxi5 :
	subroutine aaxi5(xi5,ier)
*###[ : comment:*******************************************************
*###] : comment:**********************************************************
*###[ : declarations :
	implicit none
* arguments
	DOUBLE PRECISION xi5(10)
	integer ier
* local variables
	DOUBLE PRECISION e5(10),s1,s2,s3,s4,s5,s6,s7,s8,s9,s10
	integer i,j
*
*	common blocks
*
	include 'ff.h'
	include 'aa.h'
*###] : declarations :
*###[ : kinematical matrix x5 and inverse xi5:
	if ( fdel4.eq.0 ) then
	    call fferr(91,ier)
	    return
	endif
*	the dotproducts are imported via ff.h
*	definition see ffex0.ffdot5:comment
	s1  = fpij5(6,6)
	s2  = fpij5(7,7)
	s3  = fpij5(8,8)
	s4  = fpij5(9,9)
	s5  = fpij5(6,7)
	s6  = fpij5(6,8)
	s7  = fpij5(6,9)
	s8  = fpij5(7,8)
	s9  = fpij5(7,9)
	s10 = fpij5(8,9)
*
* inverse kinematical matrix xi5
	xi5(1)=
     +	  s2*s3*s4-s2*s10**2-s3*s9**2-s4*s8**2+2*s8*s9*s10
	xi5(5)=
     +	  -s3*s4*s5+s3*s7*s9+s4*s6*s8+s5*s10**2-s6*s9*s10-s7*s8*s10
	xi5(6)=
     +	  -s2*s4*s6+s2*s7*s10+s4*s5*s8-s5*s9*s10+s6*s9**2-s7*s8*s9
	xi5(7)=
     +	  -s2*s3*s7+s2*s6*s10+s3*s5*s9-s5*s8*s10-s6*s8*s9+s7*s8**2
	xi5(2)=
     +	  +s1*s3*s4-s1*s10**2-s3*s7**2-s4*s6**2+2*s6*s7*s10
	xi5(8)=
     +	  -s1*s4*s8+s1*s9*s10+s4*s5*s6-s5*s7*s10-s6*s7*s9+s7**2*s8
	xi5(9)=
     +	  -s1*s3*s9+s1*s8*s10+s3*s5*s7-s5*s6*s10-s6*s7*s8+s6**2*s9
	xi5(3)=
     +	  +s1*s2*s4-s1*s9**2-s2*s7**2-s4*s5**2+2*s5*s7*s9
	xi5(10)=
     +	  -s1*s2*s10+s1*s8*s9+s2*s6*s7-s5*s6*s9-s5*s7*s8+s5**2*s10
	xi5(4)=
     +	  +s1*s2*s3-s1*s8**2-s2*s6**2-s3*s5**2+2*s5*s6*s8

* the determinant is also provided by ff
	do 20 i=1,10
 20	xi5(i) = xi5(i) / fdel4
*###] : kinematical matrix x5 and inverse xi5:
*###[ : check:
	if ( atest ) then
	  e5(1)=( s1*xi5(1)+s5*xi5(5)+s6*xi5(6)+s7*xi5(7) )
	  e5(2)=( s5*xi5(5)+s2*xi5(2)+s8*xi5(8)+s9*xi5(9) )
	  e5(3)=( s6*xi5(6)+s8*xi5(8)+s3*xi5(3)+s10*xi5(10) )
	  e5(4)=( s7*xi5(7)+s9*xi5(9)+s10*xi5(10)+s4*xi5(4) )
	  e5(5)=( s1*xi5(5)+s5*xi5(2)+s6*xi5(8)+s7*xi5(9) )
	  e5(6)=( s1*xi5(6)+s5*xi5(8)+s6*xi5(3)+s7*xi5(10) )
	  e5(7)=( s1*xi5(7)+s5*xi5(9)+s6*xi5(10)+s7*xi5(4) )
	  e5(8)=( s5*xi5(6)+s2*xi5(8)+s8*xi5(3)+s9*xi5(10) )
	  e5(9)=( s5*xi5(7)+s2*xi5(9)+s8*xi5(10)+s9*xi5(4) )
	  e5(10)=( s6*xi5(7)+s8*xi5(9)+s3*xi5(10)+s10*xi5(4) )
	     do 12 i=1,4
	     if (  abs(e5(i)-1.d0) .gt. .1d-5  .or.
     +		   abs(e5(i+6)	 ) .gt. .1d-5 ) then
	       print *,'aaxi5: error in xi5'
	       return
	     endif
 12	     continue
	endif
*###] : check:
*###] : aaxi5 :
	end




