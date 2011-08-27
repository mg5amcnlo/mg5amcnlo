
* file aaxinv	4-oct-1990

*###[ : aaci3 :
	subroutine aaci3(ci3,cpi,ier)
*###[ : comment:*******************************************************
*###] : comment:**********************************************************
*###[ : declarations :
	implicit none
* arguments
	DOUBLE COMPLEX ci3(3),cpi(6)
	integer ier
* local variables
	integer i
	DOUBLE PRECISION xmax,absc
	DOUBLE COMPLEX e3(3),s1,s2,s3,cnul,cc
* common blocks
	include 'ff.h'
	include 'aa.h'
* statement function
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*###] : declarations :
*###[ : kinematical matrix x3 and inverse ci3:
*	the dotproducts are imported via ff.h
*	definition see ffxc0.ffdot3:comment
	s1=cfpij3(4,4)
	s2=cfpij3(5,5)
	s3=cfpij3(4,5)
*	       inverse kinematical matrix ci3
*	       the determinant is also provided by ff
	if ( fdel2.eq.0 ) then
	    call fferr(89,ier)
	    return
	endif
	if ( atest ) then
*	    make sure that they are correct.
	    do i=4,5
		cnul = cfpij3(i,i) - cpi(i)
		if ( xloss*absc(cnul).gt.precc*absc(cpi(i)) ) then
		    print *,'aaci3: error: saved cfpij3(',i,i,
     +			') does not agree with recomputed: ',
     +			cfpij3(4,4),cpi(4),cnul
		endif
	    enddo
	    cnul = 2*cfpij3(4,5) + cpi(4) + cpi(5) - cpi(6)
	    xmax = max(absc(cpi(4)),absc(cpi(5)),absc(cpi(6)))
	    if ( xloss*absc(cnul).gt.precc*xmax ) then
	        print *,'aaci3: error: saved cfpij3(4,5) does not ',
     +			'agree with recomputed: ',2*cfpij3(4,5),
     +			cpi(6)-cpi(4)-cpi(5),cnul,xmax
	    endif
	    cnul = fdel2 - cpi(4)*cpi(5) + cfpij3(4,5)**2
	    xmax = max(abs(fdel2),absc(cfpij3(4,5)**2))
	    if ( xloss*absc(cnul).gt.precc*xmax ) then
	        print *,'aaci3: error: saved fdel2 does not ',
     +			'agree with recomputed: ',fdel2,
     +			cpi(4)*cpi(5) - cfpij3(4,5)**2,cnul,xmax
	    endif
	endif
	ci3(1)= s2*DBLE(1/fdel2)
	ci3(3)=-s3*DBLE(1/fdel2)
	ci3(2)= s1*DBLE(1/fdel2)
*###] : kinematical matrix x3 and inverse ci3:
*###[ : check: on accuracy
	if ( atest ) then
	    e3(1)= s1*ci3(1)+s3*ci3(3)
	    e3(2)= s3*ci3(3)+s2*ci3(2)
	    e3(3)= s1*ci3(3)+s3*ci3(2)
	    if ( absc(e3(1)-1) .gt. .1d-4 ) then
		print *,'aaci3: error in ci3(1) or ci3(3): ',e3(1)-1,ci3
	    endif
	    if ( absc(e3(2)-1) .gt. .1d-4 ) then
		print *,'aaci3: error in ci3(2) or ci3(3): ',e3(2)-1,ci3
	    endif
	    if ( absc(e3(3)) .gt. .1d-4 ) then
		print *,'aaci3: error in ci3(2) or ci3(3): ',e3(3),ci3
	    endif
	endif
	if ( awrite ) then
	    print *,' '
	    print *,'aaci3:imported dots and inv:'
	    print *,'s..ci3 ',s1,ci3(1)
	    print *,'       ',s2,ci3(2)
	    print *,'       ',s3,ci3(3)
	    print *,' '
	endif
*###] : check:
*###] : aaci3 :
	end

*###[ : aaci4 :
	subroutine aaci4(ci4,ier)
*###[ : comment:*******************************************************
*###] : comment:**********************************************************
*###[ : declarations :
	implicit none
* arguments
	DOUBLE COMPLEX ci4(6)
	integer ier
* local variables
	integer i,ier0,ier1
	DOUBLE COMPLEX e4(6),s1,s2,s3,s4,s5,s6,cdel2,cc
	DOUBLE PRECISION absc
* common blocks
	include 'ff.h'
	include 'aa.h'
* statement function
	absc(cc) = abs(DBLE(cc)) + abs(DIMAG(cc))
*###] : declarations :
*###[ : kinematical matrix x4 and inverse ci4:
	if ( fdel3.eq.0 ) then
	    call fferr(90,ier)
	    return
	endif
*	the dotproducts are imported via ff.h
*	definition see ffxd0.ffdot4:comment
*	       inverse kinematical matrix ci4
*	       the determinant is also provided by ff
*	ci4(1)=( +s2*s3-s6**2 )/fdel3
*	ci4(4)=( -s3*s4+s5*s6 )/fdel3
*	ci4(5)=( -s2*s5+s4*s6 )/fdel3
*	ci4(2)=( +s1*s3-s5**2 )/fdel3
*	ci4(6)=( -s1*s6+s4*s5 )/fdel3
*	ci4(3)=( +s1*s2-s4**2 )/fdel3
	ier1 = ier
*
	ier0 = ier
	call ffcel2(cdel2,cfpij4,10,6,7,10,1,ier0)
	ier1 = max(ier0,ier1)
	ci4(1) = +cdel2*(1/fdel3)
*
	cdel2 = cfpij4(5,5)*cfpij4(7,7) - cfpij4(5,7)**2
	if ( lwarn .and. absc(cdel2).lt.xloss*absc(cfpij4(5,7)**2) ) 
     +		then
	    ier0 = ier
	    call ffwarn(263,ier0,cdel2,absc(cfpij4(5,7)**2))
	    ier1 = max(ier0,ier1)
	endif
	ci4(2) = +cdel2*(1/fdel3)
*
	ier0 = ier
	call ffdel2(cdel2,cfpij4,10,5,6,9,1,ier0)
	ier1 = max(ier0,ier1)
	ci4(3) = +cdel2*(1/fdel3)
*
	ier0 = ier
	call ffdl2t(cdel2,cfpij4,5,7,6,7,10,-1,-1,10,ier0)
	ier1 = max(ier1,ier0)
	ci4(4) = -cdel2*(1/fdel3)
*
	ier0 = ier
	call ffdl2i(cdel2,cfpij4,10,5,6,9,-1,6,7,10,+1,ier0)
	ier1 = max(ier1,ier0)
	ci4(5) = +cdel2*(1/fdel3)
*
	ier0 = ier
	call ffdl2t(cdel2,cfpij4,5,7,5,6,9,+1,-1,10,ier0)
	ier1 = max(ier1,ier0)
	ci4(6) = -cdel2*(1/fdel3)
*
*###] : kinematical matrix x4 and inverse ci4:
*###[ : check:
	if ( atest ) then
	    s1=cfpij4(5,5)
	    s2=cfpij4(6,6)
	    s3=cfpij4(7,7)
	    s4=cfpij4(5,6)
	    s5=cfpij4(5,7)
	    s6=cfpij4(6,7)
	    e4(1) = ( s1*ci4(1)+s4*ci4(4)+s5*ci4(5) )
	    e4(2) = ( s4*ci4(4)+s2*ci4(2)+s6*ci4(6) )
	    e4(3) = ( s5*ci4(5)+s6*ci4(6)+s3*ci4(3) )
	    e4(4) = ( s1*ci4(4)+s4*ci4(2)+s5*ci4(6) )
	    e4(5) = ( s1*ci4(5)+s4*ci4(6)+s5*ci4(3) )
	    e4(6) = ( s4*ci4(5)+s2*ci4(6)+s6*ci4(3) )
	    do 12 i=1,3
		if (  absc(e4(i)-1.d0) .gt. .1d-5  .or.
     +		      absc(e4(i+3) ) .gt. .1d-5 ) then
		    print *,'aaci4: error in ci4'
		    return
		endif
 12	    continue
	endif
*###] : check:
*###] : aaci4 :
	end
