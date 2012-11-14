*###[ ffai:
	subroutine ffai(ai,daiaj,aai,laai,del2s,sdel2s,xpi,dpipj,piDpj,
     +		ier)
***#[*comment:***********************************************************
*									*
*	calculates the coefficients of the projective transformation	*
*									*
*	    xi = ai*ui / (som aj*uj )					*
*									*
*	such that the coefficients of z^2, z*x and z*y vanish:		*
*									*
*	    a2/a1 = ( lij +/- lam1/2(xp1,xm1,xm2) ) / (2*xm2)		*
*	    a3 = ( xm2*a2 - xm1*a1 ) / ( xl23*a2 - xl13*a1 )		*
*	    a4 = ( xm2*a2 - xm1*a1 ) / ( xl24*a2 - xl14*a1 )		*
*									*
*	the differences ai-aj = daiaj(i,j) are also evaluated.		*
*									*
*	Input:	del2s		real	delta(s3,s4,s3,s4)		*
*		sdel2s		real	sqrt(-del2s)			*
*		xpi(10)		real	masses, momenta^2		*
*		dpipj(10,10	real	xpi(i) - xpi(j)			*
*		piDpj(10,10)	real	dotproducts			*
*									*
*	Output:	ai(4)		real	Ai of the transformation	*
*		daiaj(4,4)	real	Ai-Aj				*
*		aai(4)		real	the other roots			*
*		laai		logical	if .TRUE. aai are defined	*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	logical laai
	DOUBLE PRECISION ai(4),daiaj(4,4),aai(4),del2s,sdel2s,xpi(10),
     +		dpipj(10,10),piDpj(10,10)
*
*	local variables
*
	integer i,j,ier0,ier1,ier2
	DOUBLE PRECISION del2sa,del2sb,del3mi(2),aim(4),aaim(4),delps,
     +		del3m(1),dum,da2a1m,da1a3m,da1a4m,da2a3m,da2a4m,da3a4m
*	for debugging purposes
	DOUBLE COMPLEX ca1m
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations: 
*  #[ get ai:
	if ( lwrite ) print *,'ffai: xpi = ',(xpi(i),i=1,10),ier
*
*	A4: some arbitrary normalisation ...
*
	ai(4) = 1
	aai(4) = 1
	ier2 = ier
	if ( del2s .ne. 0 ) then
*
*	    A3: simple solution of quadratic equation
*
	    ier0 = ier
	    call ffroot(aaim(3),aim(3),xpi(4),piDpj(4,3),xpi(3),
     +							sdel2s,ier0)
	    ier2 = max(ier2,ier0)
	    if ( aim(3) .eq. 0 ) then
*		choose the other root
		if ( lwrite ) print *,'ffai: 1/A_3 = 0'
		ier = ier + 100
		return
	    endif
	    ai(3) = ai(4)/aim(3)
	    if ( aaim(3) .ne. 0 ) then
		laai = .TRUE.
		aai(3) = aai(4)/aaim(3)
	    else
		laai = .FALSE.
	    endif
*
*	    A2: a bit more complicated quadratic equation
*
	    ier1 = ier
	    ier0 = ier
	    call ffdl2s(del2sa,xpi,piDpj, 2,4,10,1, 3,4,7,1, 10,ier0)
	    ier1 = max(ier1,ier0)
	    ier0 = ier
	    call ffdl3m(del3mi(2),.FALSE.,x0,x0,xpi,dpipj,piDpj,10,
     +						3,4,7, 2,1,ier0)
	    ier1 = max(ier1,ier0)
	    call ffroot(aim(2),aaim(2),xpi(4),piDpj(4,2),del3mi(2)/del2s
     +		,del2sa/sdel2s,ier1)
	    ier2 = max(ier2,ier1)
	    if ( aim(2) .eq. 0 ) then
		if ( lwrite ) print *,'ffai: 1/A_2 = 0'
		ier = ier + 100
		return
	    endif
	    ai(2) = ai(4)/aim(2)
	    if ( laai ) then
		if ( aaim(2) .eq. 0 ) then
		    laai = .FALSE.
		else
		    aai(2) = aai(4)/aaim(2)
		endif
	    endif
*
*	    A1: same as A2, except for the special nasty case.
*
	    if ( .not.lnasty ) then
	    ier0 = ier
	    ier1 = ier
	    call ffdl2s(del2sb,xpi,piDpj, 1,4,8,-1, 3,4,7,1, 10,ier0)
	    ier1 = max(ier1,ier0)
	    ier0 = ier
	    call ffdl3m(del3mi(1),.FALSE.,x0,x0,xpi,dpipj,piDpj,10,
     +						3,4,7, 1,1,ier0)
	    ier1 = max(ier1,ier0)
	    call ffroot(aim(1),aaim(1),xpi(4),piDpj(4,1),del3mi(1)/del2s
     +		,del2sb/sdel2s,ier1)
	    ier2 = max(ier2,ier1)
	    if ( aim(1) .eq. 0 ) then
		if ( lwrite ) print *,'ffai: 1/A_1 = 0'
		ier = ier + 100
		return
	    endif
	    ai(1) = ai(4)/aim(1)
	    if ( laai ) then
		if ( aaim(1) .eq. 0 ) then
		    laai = .FALSE.
		else
		    aai(1) = aai(4)/aaim(1)
		endif
	    endif
	    else
		laai = .FALSE.
		ca1m = (c2sisj(1,4) - (c2sisj(1,3)*DBLE(xpi(4)) -
     +			c2sisj(1,4)*DBLE(piDpj(3,4)))/DBLE(sdel2s))/
     +			DBLE(2*xpi(4))
		ca1 = DBLE(ai(4))/ca1m
		if ( lwrite ) print *,'ffai: A1 = ',ca1
		ai(1) = ai(4)/DBLE(ca1m)
	    endif
	else
*
*	    the special case del2s=0 with xpi(3)=xpi(4),xpi(7)=0
*
	    laai = .FALSE.
	    ai(3) = ai(4)
	    if ( piDpj(7,2) .eq. 0 .or. piDpj(7,1) .eq. 0 ) then
		call fferr(55,ier)
		return
	    endif
	    ai(2) = ai(4)*xpi(3)/piDpj(7,2)
	    ai(1) = ai(4)*xpi(3)/piDpj(7,1)
	endif
	ier = ier2
*  #] get ai: 
*  #[ get daiaj:
	ier2 = ier
	do 120 i=1,4
	    daiaj(i,i) = 0
	    do 110 j=i+1,4
		daiaj(j,i) = ai(j) - ai(i)
		if ( abs(daiaj(j,i)) .ge. xloss*abs(ai(i)) ) goto 105
		if ( lwrite ) print *,'daiaj(',j,i,') = ',daiaj(j,i),
     +			ai(j),-ai(i),ier
		if ( del2s .eq. 0 ) then
*  #[		    del2s=0:
		    if ( i .eq. 1 .and. j .eq. 2 ) then
			daiaj(2,1) = -ai(1)*ai(2)*piDpj(5,7)/xpi(3)
			goto 104
		    elseif ( i .eq. 3 .and. j .eq. 4 ) then
			daiaj(4,3) = 0
			goto 104
		    endif
		    ier1 = ier
		    call ffwarn(146,ier1,daiaj(j,i),ai(i))
		    goto 105
*  #]		    del2s=0: 
		elseif ( lnasty .and. i.eq.1 ) then
		    ier1 = ier
		    call ffwarn(146,ier1,daiaj(j,i),ai(i))
		    goto 105
		endif
		ier0 = ier
		if ( i .eq. 1 .and. j .eq. 2 ) then
*  #[		    daiaj(2,1):
*
*		    some determinants (as usual)
*
*		    as the vertex p1,s4,? does not exist we use ffdl2t
*
		    call ffdl2t(delps,piDpj, 5,4, 3,4,7,1,+1, 10,ier0)
		    ier1 = max(ier1,ier0)
		    ier0 = ier
		    call ffdl3m(del3m,.FALSE.,x0,x0,xpi,dpipj,piDpj,
     +			10, 3,4,7, 5,1, ier0)
		    ier1 = max(ier1,ier0)
		    call ffroot(dum,da2a1m,xpi(4),piDpj(4,5),
     +			del3m(1)/del2s,-delps/sdel2s,ier1)
		    daiaj(2,1) = -ai(1)*ai(2)*da2a1m
		    goto 104
*  #]		    daiaj(2,1): 
		elseif ( i .eq. 1 .and. j .eq. 3 ) then
*  #[		    daiaj(3,1):
*
*		    Again, the solution of a simple quadratic equation
*
		    call ffdl2t(delps,piDpj, 9,4, 3,4,7,1,+1, 10,ier0)
		    ier1 = ier0
		    ier0 = ier
		    call ffdl3m(del3m,.FALSE.,x0,x0,xpi,dpipj,piDpj,
     +			10, 3,4,7, 9,1, ier0)
		    ier1 = max(ier1,ier0)
		    call ffroot(dum,da1a3m,xpi(4),-piDpj(4,9),
     +			del3m(1)/del2s,delps/sdel2s,ier1)
		    daiaj(3,1) = -ai(1)*ai(3)*da1a3m
		    goto 104
*  #]		    daiaj(3,1): 
		elseif ( i .eq. 1 .and. j .eq. 4 ) then
*  #[		    daiaj(4,1):
*
*		    Again, the solution of a simple quadratic equation
*
		    call ffdl2s(delps,xpi,piDpj,4,1,8,1,3,4,7,1,10,ier0)
		    ier1 = ier0
		    ier0 = ier
		    call ffdl3m(del3m,.FALSE.,x0,x0,xpi,dpipj,piDpj,
     +			10, 3,4,7, 8,1, ier0)
		    ier1 = max(ier0,ier1)
		    call ffroot(dum,da1a4m,xpi(4),piDpj(4,8),del3m(1)/
     +			del2s,delps/sdel2s,ier1)
		    daiaj(4,1) = ai(1)*ai(4)*da1a4m
		    goto 104
*  #]		    daiaj(4,1): 
		elseif ( i .eq. 2 .and. j .eq. 3 ) then
*  #[		    daiaj(3,2):
*
*		    Again, the solution of a simple quadratic equation
*
		    call ffdl2t(delps,piDpj, 6,4, 3,4,7,1,+1, 10,ier0)
		    ier1 = ier0
		    ier0 = ier
		    call ffdl3m(del3m,.FALSE.,x0,x0,xpi,dpipj,piDpj,
     +			10, 3,4,7, 6,1, ier0)
		    ier1 = max(ier1,ier0)
		    call ffroot(dum,da2a3m,xpi(4),-piDpj(4,6),
     +			del3m(1)/del2s,delps/sdel2s,ier1)
		    daiaj(3,2) = ai(2)*ai(3)*da2a3m
		    goto 104
*  #]		    daiaj(3,2): 
		elseif ( i .eq. 2 .and. j .eq. 4 ) then
*  #[		    daiaj(4,2):
*
*		    Again, the solution of a simple quadratic equation
*
		    call ffdl2s(delps,xpi,piDpj,2,4,10,1,3,4,7,1,10,
     +								ier0)
		    ier1 = ier0
		    ier0 = ier
		    call ffdl3m(del3m,.FALSE.,x0,x0,xpi,dpipj,piDpj,
     +			10, 3,4,7, 10,1, ier0)
		    ier1 = max(ier0,ier1)
		    call ffroot(dum,da2a4m,xpi(4),piDpj(4,10),del3m(1)/
     +			del2s,delps/sdel2s,ier1)
		    daiaj(4,2) = -ai(2)*ai(4)*da2a4m
		    goto 104
*  #]		    daiaj(4,2): 
		elseif ( i .eq. 3 .and. j .eq. 4 ) then
*  #[		    daiaj(4,3):
*
*		    Again, the solution of a very simple quadratic equation
*
		    ier1 = ier
		    call ffroot(dum,da3a4m,xpi(4),-piDpj(4,7),
     +			xpi(7),sdel2s,ier1)
		    daiaj(4,3) = ai(3)*ai(4)*da3a4m
		    goto 104
*  #]		    daiaj(4,3): 
		endif
  104		continue
		if ( lwrite ) print *,'daiaj(',j,i,')+= ',daiaj(j,i),ier
  105		continue
		daiaj(i,j) = -daiaj(j,i)
		ier2 = max(ier2,ier1)
  110	    continue
  120	continue
	ier = ier2
*  #] get daiaj: 
*  #[ debug output:
	if ( lwrite ) then
	    print *,'ffai: Found Ai: ',ai
	    print *,'      Ai-Aj:    ',daiaj
	    print *,'      ier       ',ier
	endif
*  #] debug output: 
*###] ffai:
	end
*###[ fftran:
	subroutine fftran(ai,daiaj,aai,laai,xqi,dqiqj,qiDqj,
     +		del2s,sdel2s,xpi,dpipj,piDpj,ier)
***#[*comment:***********************************************************
*									*
*	Transform the impulses according to				*
*									*
*	    ti  = Ai*si							*
*	    qij = (Ai*si - Aj*sj)					*
*									*
*	In case del2s=0 it calculates the same coefficients but for	*
*	for A1,A2 leave out the delta with 2*delta = 1-xpi(4)/xpi(3)	*
*	infinitesimal.							*
*									*
*	Input:	ai(4)		ai					*
*		daiaj(4,4)	ai-aj					*
*		del2s		\delta^{s(3) s4}_{s(3) s4}		*
*		sdel2s		sqrt(del2s)				*
*		xpi(10)		masses = s1-s2-s(3)-s4			*
*		dpipj(10,10)	differences				*
*		piDpj(10,10)	dotproducts				*
*									*
*	Output:	xqi(10)		transformed momenta			*
*		dqiqj(10,10)	differences				*
*		qiDqj(10,10)	dotproducts				*
*		ier		(integer) 0=ok,1=inaccurate,2=error	*
*									*
*	Calls:	ffxlmb,...						*
*									*
***#]*comment:*********************************************************** 
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ier
	logical laai
	DOUBLE PRECISION ai(4),daiaj(4,4),aai(4),xqi(10),dqiqj(10,10),
     +		qiDqj(10,10),del2s,sdel2s,xpi(10),dpipj(10,10),
     +		piDpj(10,10)
*
*	local variables
*
	integer i,j,ji,k,kj,l,lk,is,isgnji,isgnlk,
     +		ifirst,i1,j1,k1,j2,kk,kkj,ier0,ier1,ier2
	logical lgo
	DOUBLE PRECISION xmax,dum,delps,del2d2,dl2d22,aijk,aijkl,
     +		xheck,smax,s(3),rloss,som
*
*	common blocks
*
	include 'ff.h'
*
	ifirst = 0
*  #] declarations: 
*  #[ si.sj -> ti.tj:
*
*	calculate the dotproducts of ti(i) = ai*si(i): no problems.
*
	do 20 i=1,4
	    xqi(i) = ai(i)**2 * xpi(i)
	    qiDqj(i,i) = xqi(i)
	    do 10 j=i+1,4
		qiDqj(j,i) = ai(j)*ai(i)*piDpj(j,i)
		qiDqj(i,j) = qiDqj(j,i)
   10	    continue
   20	continue
*
*	and the smuggled ones for the onshell complex D0
*
	if ( lsmug ) then
	    do 40 j=1,3
		do 30 i=i+1,4
		    c2sisj(i,j) = DBLE(ai(j)*ai(i))*c2sisj(i,j)
		    c2sisj(j,i) = c2sisj(i,j)
   30		continue
   40	    continue
	endif
	if ( lnasty ) then
	    do 60 j=3,4
*
*		we also hide in this array the corresponding real value
*		in (j,2) and (2,j), and the untransformed in (j,j).
*		Not beuatiful, but we need these to get the correct
*		Riemann sheets.
*
		c2sisj(j,j) = c2sisj(j,1)
		c2sisj(j,2) = ai(j)*ai(1)*DBLE(c2sisj(j,1))
		c2sisj(2,j) = c2sisj(j,2)
		c2sisj(j,1) = DBLE(ai(j))*ca1*c2sisj(j,1)
		c2sisj(1,j) = c2sisj(j,1)
*
   60	    continue
	    if ( lwrite ) then
		print *,'c2sisj(1,3-4) = ',c2sisj(1,3),c2sisj(1,4)
		print *,'c2sisj(2,3-4) = ',c2sisj(2,3),c2sisj(2,4)
	    endif
	endif
*
*  #] si.sj -> ti.tj: 
*  #[ si.pj -> ti.qj:
*
*	The dotproducts ti.qjk are still not too bad
*	Notice that t3.p = t4.p, so qiDqj(3,5-10) = qiDqj(4,5-10)
*
	ier2 = ier
	do 90 i=1,4
	    do 80 j=1,3
		do 70 k=j+1,4
		    ier1 = ier
		    kj = inx(k,j)
		    is = isgn(k,j)
		    if ( .not. ltest .and. i.eq.4 .and.
     +			(del2s.ne.0 .or. kj.eq.5 .or. kj.eq.7 )) then
			qiDqj(kj,4) = qiDqj(kj,3)
			goto 65
		    endif
		    s(1) = qiDqj(k,i)
		    s(2) = qiDqj(j,i)
		    qiDqj(kj,i) = is*(s(1) - s(2))
		    if ( abs(qiDqj(kj,i)).ge.xloss*abs(s(1)) ) goto 65
		    if ( lwrite ) print *,'qiDqj(',kj,i,')  =',
     +			qiDqj(kj,i),is,s(1),s(2),ier
		    ier0 = ier
		    if ( del2s .eq. 0 ) then
*
*			the special cases for del2s-0
*
			if ( kj .eq. 5 ) then
			    call ffdl2t(delps,piDpj, 7,i, 1,2,5,
     +							1,1,10,ier0)
			    qiDqj(5,i) = ai(1)*ai(2)*ai(i)*delps/xpi(3)
			elseif ( kj .eq. 7 ) then
			    qiDqj(kj,i) = ai(i)*ai(4)**2*piDpj(kj,i)
			else
*
*			    the pi has a mixed delta/no delta behaviour
*
			    call ffwarn(144,ier1,qiDqj(kj,i),s(1))
			    if ( lwrite ) print *,'in qiDqj(',kj,i,')'
			    goto 65
			endif
			if ( lwrite ) print *,'qiDqj(',kj,i,')+ =',
     +				qiDqj(kj,i),max(ier2,ier1)
			goto 65
		    endif
*
*		    Normal case, from the quadratic equation ...
*
		    ier1 = ier0
		    ier0 = ier
		    call ff2dl2(del2d2,delps,xpi,dpipj,piDpj, i,
     +			j,k,kj,is, 4, 3,4,7,+1, 10, ier0)
		    ier1 = max(ier1,ier0)
		    ier0 = ier
		    call ff2d22(dl2d22,xpi,dpipj,piDpj, i, j,k,kj,is,
     +			3,4,7,+1, 10,ier0)
		    ier1 = max(ier1,ier0)
		    call ffroot(dum,aijk,xpi(4),delps,dl2d22/del2s,
     +			-del2d2/sdel2s,ier1)
*		    the minus sign is because we have aijk, not aikj.
		    qiDqj(kj,i) = -is*aijk*ai(i)*ai(j)*ai(k)
		    if ( lwrite ) print *,'qiDqj(',kj,i,')+ =',
     +			qiDqj(kj,i),max(ier2,ier1)
   65		    continue
		    qiDqj(i,kj) = qiDqj(kj,i)
		    ier2 = max(ier2,ier1)
   70		continue
   80	    continue
   90	continue
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    do 100 i=5,10
		if ( del2s.eq.0 .and. (i.ne.5 .and. i.ne.7) ) goto 100
		if ( lnasty .and. (i.eq.5.or.i.eq.8.or.i.eq.9)) goto 100
		if ( rloss*abs(qiDqj(i,3)-qiDqj(i,4)) .gt. precx*
     +		    abs(qiDqj(i,3)))print *,'fftran: error: t3.q',i,
     +		    ' /= t4.q',i,': ',qiDqj(i,3),qiDqj(i,4),
     +		    qiDqj(i,3)-qiDqj(i,4),ier
  100	    continue
	endif
*  #] si.pj -> ti.qj: 
*  #[ pi.pj -> qi.qj:
	do 180 i=1,3
	    do 170 j=i+1,4
		ji = inx(j,i)
		isgnji = isgn(j,i)
		do 160 k=i,3
		    do 150 l=k+1,4
			if ( k .eq. i .and. l .lt. j ) goto 150
			ier1 = ier
			lk = inx(l,k)
			isgnlk = isgn(l,k)
*
*			Some are zero by definition, or equal to others
*
			if ( del2s .ne. 0 .and. (ji.eq.7 .or. lk.eq.7)
     +				.or.
     +			     del2s .eq. 0 .and. (ji.eq.7 .and. (lk.eq.7
     +				.or. lk.eq.5) .or. ji.eq.5 .and. lk.eq.7
     +			    ) ) then
			    qiDqj(lk,ji) = 0
			    goto 145
			endif
			if ( j.eq.4 .and. (del2s.ne.0 .or. lk.eq.5) )
     +								then
			    qiDqj(lk,ji) = isgnji*isgn(3,i)*
     +				qiDqj(lk,inx(3,i))
			    goto 145
			endif
			if ( l.eq.4 .and. (del2s.ne.0 .or. ji.eq.5) )
     +								then
			    qiDqj(lk,ji) = isgnlk*isgn(3,k)*
     +				qiDqj(inx(3,k),ji)
			    goto 145
			endif
*
*			First normal try
*
			if ( abs(qiDqj(k,ji)).le.abs(qiDqj(i,lk)) ) then
			    s(1) = qiDqj(k,ji)
			    s(2) = qiDqj(l,ji)
			    is = isgnlk
			else
			    s(1) = qiDqj(i,lk)
			    s(2) = qiDqj(j,lk)
			    is = isgnji
			endif
			qiDqj(lk,ji) = is*(s(2) - s(1))
			if ( abs(qiDqj(lk,ji)) .ge. xloss**2*abs(s(1)) )
     +				goto 145
			if ( lwrite ) print *,'qiDqj(',lk,ji,')  = ',
     +			    qiDqj(lk,ji),isgnji,isgnlk,s(1),s(2),ier2
*
*			First the special case del2s=0
*
			if ( del2s .eq. 0 ) then
			    if ( ji .eq. 5 .and. lk .eq. 5 ) then
				call ffdl3m(s(1),.FALSE.,x0,x0,xpi,dpipj
     +					,piDpj, 10, 1,2,5, 7, 1,ier1)
				qiDqj(5,5) =ai(1)**2*ai(2)**2*s(1)/xpi(3
     +					)**2
				if ( lwrite ) print *,'qiDqj(',lk,ji,
     +				  ')+ =',qiDqj(lk,ji),max(ier2,ier1)
			    else
				call ffwarn(145,ier1,qiDqj(lk,ji),s(1))
			    endif
			    goto 145
			endif
*
*			Otherwise use determinants
*
			call ffabcd(aijkl,xpi,dpipj,piDpj,del2s,
     +			    sdel2s, i,j,ji,isgnji, k,l,lk,isgnlk, 10,
     +			    ifirst, ier1)
			qiDqj(lk,ji) = (isgnji*isgnlk)*
     +				aijkl*ai(i)*ai(j)*ai(k)*ai(l)
			if ( lwrite ) print *,'qiDqj(',lk,ji,')+ = ',
     +				qiDqj(lk,ji),max(ier2,ier1)
			goto 145
*			print *,'fftran: warning: numerical problems ',
*     +				'in qiDqj(',lk,ji,')'
  145			continue
			if ( lk .ne. ji ) then
			    qiDqj(ji,lk) = qiDqj(lk,ji)
			else
			    xqi(ji) = qiDqj(lk,ji)
			endif
			ier2 = max(ier2,ier1)
  150		    continue
  160		continue
  170	    continue
  180	continue
	ier = ier2
	if ( ltest ) then
	    rloss = xloss**2*DBLE(10)**(-mod(ier,50))
	    if ( del2s .ne. 0 ) then
	    do 810 i=1,2
		do 800 j=i,2
		   s(1) = isgn(i,3)*isgn(j,3)*qiDqj(inx(i,3),inx(j,3))
		   s(2) = isgn(i,4)*isgn(j,4)*qiDqj(inx(i,4),inx(j,4))
		   if ( rloss*abs(s(1)-s(2)).gt.precx*max(abs(
     +			s(1)),abs(s(2))) ) print *,'fftran: error: q',i,
     +			'3.q',j,'3 /= q',i,'4.q',j,'4 : ',s(1),s(2),
     +			s(1)-s(2),ier
  800		continue
  810	    continue
	    endif
	    do 830 i=1,10
		do 820 j=i+1,10
		    if ( qiDqj(i,j) .ne. qiDqj(j,i) ) print *,
     +			'fftran: error: qiDqj(',i,j,')/= qiDqj(',j,i,')'
  820		continue
  830	    continue
	    do 840 i=1,10
		xheck = qiDqj(i,5)+qiDqj(i,6)+qiDqj(i,7)+qiDqj(i,8)
		smax = max(abs(qiDqj(i,5)),abs(qiDqj(i,6)),
     +			abs(qiDqj(i,7)),abs(qiDqj(i,8)))
		if ( rloss*abs(xheck) .gt. precx*smax ) print *,
     +			'fftran: error: No momentum conservation in ',
     +			'qiDqj, i=',i,' j=5678 ',xheck,smax,ier
		xheck = qiDqj(i,5)+qiDqj(i,6)+qiDqj(i,9)
		smax = max(abs(qiDqj(i,5)),abs(qiDqj(i,6)),
     +			abs(qiDqj(i,9)))
		if ( rloss*abs(xheck) .gt. precx*smax ) print *,
     +			'fftran: error: No momentum conservation in ',
     +			'qiDqj, i=',i,' j=569 ',xheck,smax,ier
		xheck = qiDqj(i,5)+qiDqj(i,10)+qiDqj(i,8)
		smax = max(abs(qiDqj(i,5)),abs(qiDqj(i,10)),
     +			abs(qiDqj(i,8)))
		if ( rloss*abs(xheck) .gt. precx*smax ) print *,
     +			'fftran: error: No momentum conservation in ',
     +			'qiDqj, i=',i,' j=5810 ',xheck,smax,ier
  840	    continue
	endif
*  #] pi.pj -> qi.qj: 
*  #[ si^2 - sj^2:
*
*	the differences may be awkward
*
	ier2 = ier
	do 140 i=1,4
	    dqiqj(i,i) = 0
	    do 130 j=i+1,4
		ier0 = ier
		dqiqj(j,i) = xqi(j) - xqi(i)
		smax = abs(xqi(i))
		if ( abs(dqiqj(j,i)) .ge. xloss*smax ) goto 125
		if ( lwrite ) print *,'dqiqj(',j,i,')  = ',
     +			dqiqj(j,i),xqi(j),-xqi(i),ier2
		if ( abs(daiaj(j,i)) .le. xloss*abs(ai(i)) )
     +								then
		    s(1) = daiaj(j,i)*(ai(i)+ai(j))*xpi(j)
		    s(2) = ai(i)**2*dpipj(j,i)
		    som = s(1) + s(2)
		    xmax = abs(s(1))
		    if ( lwrite ) print *,'dqiqj(',j,i,')+ = ',
     +				som,s(1),s(2),ier2
		    if ( xmax.lt.smax ) then
			dqiqj(j,i) = som
			smax = xmax
		    endif
		    if ( abs(dqiqj(j,i)) .ge. xloss*smax ) goto 125
		endif
*
*		give up
*
		if ( lwarn ) call ffwarn(125,ier0,dqiqj(j,i),smax)
		if ( lwrite ) print *,' (between qi(',i,') and qi(',j,
     +			'))'
  125		continue
		dqiqj(i,j) = -dqiqj(j,i)
		ier2 = max(ier2,ier0)
  130	    continue
  140	continue
*  #] si^2 - sj^2: 
*  #[ si^2 - pj^2:
	do 210 i=1,4
	    do 200 j=1,4
		do 190 kk=j+1,4
		    ier0 = ier
		    k = kk
		    kj = inx(k,j)
		    kkj = kj
*
*		    Use that q_(i4)^2 = q_(i3)^2
*
		    if ( del2s.ne.0 .and. k.eq.4 ) then
			if ( j .eq. 3 ) then
			    dqiqj(7,i) = -xqi(i)
			else
			    dqiqj(kj,i) = dqiqj(inx(j,3),i)
			endif
			goto 185
		    elseif ( kj .eq. 7 ) then
			dqiqj(7,i) = -xqi(i)
			goto 185
		    endif
		    xmax = 0
  181		    continue
		    som = xqi(kj) - xqi(i)
		    if ( lwrite .and. kk .ne. k ) print *,'dqiqj(',kj,i,
     +			')4+= ',som,xqi(kj),xqi(i),ier2
		    if ( k.eq.kk .or. abs(xqi(i)).lt.xmax ) then
			dqiqj(kj,i) = som
			xmax = abs(xqi(i))
			if ( abs(dqiqj(kj,i)) .ge. xloss*xmax ) goto 185
		    endif
		    if ( lwrite .and. kk .eq. k ) print *,'dqiqj(',kj,i,
     +			')  = ',dqiqj(kj,i),xqi(kj),xqi(i),ier2
*
*		    second try
*		    we assume that qi.qj, i,j<=3 are known
*
		    if ( abs(dqiqj(k,i)) .lt. abs(dqiqj(j,i)) ) then
			j1 = k
			j2 = j
		    else
			j2 = k
			j1 = j
		    endif
		    s(1) = dqiqj(j1,i)
		    s(2) = xqi(j2)
		    s(3) = -2*qiDqj(j1,j2)
		    som = s(1) + s(2) + s(3)
		    smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
		    if ( lwrite ) print *,'dqiqj(',kj,i,')+ = ',
     +			som,s(1),s(2),s(3),ier2
		    if ( smax.lt.xmax ) then
			dqiqj(kj,i) = som
			xmax = smax
			if ( abs(dqiqj(kj,i)) .ge. xloss*xmax ) goto 185
		    endif
*
*		    third try: rearrange s(2),s(3)
*		    this works if ai(j1)~ai(j2)
*
		    if ( abs(daiaj(j2,j1)) .lt. xloss*abs(ai(j1)) ) then
			s(2) = ai(j2)*daiaj(j2,j1)*xpi(j2)
			s(3) = ai(j2)*ai(j1)*dpipj(kj,j1)
			som = s(1) + s(2) + s(3)
			smax = max(abs(s(1)),abs(s(2)),abs(s(3)))
			if ( lwrite ) print *,'dqiqj(',kj,i,')++= ',
     +				som,s(1),s(2),s(3),ier2
			if ( smax.lt.xmax ) then
			    dqiqj(kj,i) = som
			    xmax = smax
			    if ( abs(dqiqj(kj,i)) .ge. xloss*xmax )
     +				goto 185
			endif
		    endif
*
*		    There is a trick involving the other root for j2=4
*		    Of course it also works for j2=3.
*
		    if ( laai .and. j2 .ge. 3 ) then
			s(2) = -ai(4)**2*(ai(j1)/aai(j1))*xpi(4)
			som = s(1) + s(2)
			smax = abs(s(1))
			if ( lwrite ) print *,'dqiqj(',kj,i,')3+= ',
     +				som,s(1),s(2),ier2
			if ( smax.lt.xmax ) then
			    dqiqj(kj,i) = som
			    xmax = smax
			    if ( abs(dqiqj(kj,i)) .ge. xloss*xmax )
     +				goto 185
			endif
		    endif
*
*		    If k = 3 we can also try with k = 4 -- should give
*		    the same
*
		    if ( del2s.ne.0 .and. kk.eq.3 .and. k.eq.3 ) then
			k = 4
			kj = inx(k,j)
			dqiqj(kj,i) = dqiqj(kkj,i)
			if ( lwrite ) print *,'trying with ',kj,
     +				' instead of ',kkj
			goto 181
		    endif
		    if ( del2s.ne.0 .and. kk.eq.4 .and. k.eq.4 ) then
			k = 3
			kj = inx(k,j)
			dqiqj(kj,i) = dqiqj(kkj,i)
			if ( lwrite ) print *,'trying with ',kj,
     +				' instead of ',kkj
			goto 181
		    endif
*
*		    give up
*
		    if ( lwarn ) call ffwarn(126,ier0,dqiqj(kj,i),xmax)
		    if ( lwrite ) print *,' (between qi(',kj,') and qi('
     +			,i,'))'

  185		    continue
		    if ( k .ne. kk ) then
			dqiqj(kkj,i) = dqiqj(kj,i)
			dqiqj(i,kkj) = -dqiqj(kj,i)
		    else
			dqiqj(i,kj) = -dqiqj(kj,i)
		    endif
		    ier2 = max(ier2,ier0)
  190		continue
  200	    continue
  210	continue
*  #] si^2 - pj^2: 
*  #[ pi^2 - pj^2:
	do 280 i=1,4
	    do 270 j=i+1,4
		ji = inx(j,i)
		dqiqj(ji,ji) = 0
		do 260 k=i,4
		    do 250 l=k+1,4
			ier0 = ier
			if ( k .eq. i .and. l .le. j ) goto 250
			lk = inx(l,k)
			if ( del2s .eq. 0 ) then
*
*			    special case:
*
			    if ( j.eq.4 .and. i.eq.3 ) then
				dqiqj(lk,7) = xqi(lk)
				goto 245
			    endif
			    if ( l.eq.4 .and. k.eq.3 ) then
				dqiqj(7,ji) = -xqi(ji)
				goto 245
			    endif
			else
*
*			Use that t_3.p_i = t_4.p_i
*
			    if ( k.eq.i .and. j.eq.3 .and. l.eq.4 ) then
				dqiqj(lk,ji) = 0
				goto 245
			    endif
			    if ( j.eq.4 ) then
				if ( i .eq. 3 ) then
				    dqiqj(lk,7) = xqi(lk)
				else
				    dqiqj(lk,ji) = dqiqj(lk,inx(i,3))
				endif
				goto 245
			    endif
			    if ( l.eq.4 ) then
				if ( k .eq. 3 ) then
				    dqiqj(7,ji) = -xqi(ji)
				else
				    dqiqj(lk,ji) = dqiqj(inx(k,3),ji)
				endif
				goto 245
			    endif
			endif
*
*			We really have to calculate something
*
			dqiqj(lk,ji) = xqi(lk) - xqi(ji)
			smax = abs(xqi(lk))
			if ( abs(dqiqj(lk,ji)).ge.xloss*smax ) goto 245
			if ( lwrite ) print *,'dqiqj(',lk,ji,')  =',
     +				dqiqj(lk,ji),xqi(lk),xqi(ji),ier2
*
*			First the special case j=k,l
*
			i1 = i
			j1 = j
			k1 = k
			lgo = .FALSE.
			if ( j .eq. k ) then
			    k1 = l
			    lgo = .TRUE.
			elseif ( j .eq. l ) then
			    lgo = .TRUE.
			elseif ( i .eq. k ) then
			    i1 = j
			    j1 = i
			    k1 = l
			    lgo = .TRUE.
			endif
			if ( lgo ) then
			    s(1) = dqiqj(k1,i1)
			    s(2) = 2*isgn(i1,k1)*qiDqj(j1,inx(i1,k1))
			    xmax = abs(s(1))
			    if ( xmax .lt. smax ) then
				smax = xmax
				dqiqj(lk,ji) = s(1) + s(2)
				if ( lwrite ) print *,'dqiqj(',lk,ji,
     +				  ')+ =',dqiqj(lk,ji),s(1),s(2),ier2
				if ( abs(dqiqj(lk,ji)).ge.xloss*smax )
     +						goto 245
			    endif
			endif
*
*			Just some recombinations
*
			if ( abs(dqiqj(l,ji)).lt.abs(dqiqj(k,ji)) ) then
			    j1 = l
			    j2 = k
			else
			    j2 = l
			    j1 = k
			endif
			s(1) = dqiqj(j1,ji)
			s(2) = xqi(j2)
			s(3) = -2*qiDqj(j1,j2)
*			only if this is an improvement
			xmax = max(abs(s(1)),abs(s(2)),abs(s(3)))
			if (  xmax .lt. smax ) then
			    smax = xmax
			    dqiqj(lk,ji) = s(1) + s(2) + s(3)
			    if ( lwrite ) print *,'dqiqj(',lk,ji,')+1=',
     +				dqiqj(lk,ji),s(1),s(2),s(3),ier2
			    if ( abs(dqiqj(lk,ji)) .ge. xloss*smax )
     +				goto 245
			endif
			if ( abs(dqiqj(j,lk)).lt.abs(dqiqj(i,lk)) ) then
			    j1 = j
			    j2 = i
			else
			    j2 = j
			    j1 = i
			endif
			s(1) = -dqiqj(j1,lk)
			s(2) = -xqi(j2)
			s(3) = 2*qiDqj(j1,j2)
*			only if this is an improvement
			xmax = max(abs(s(1)),abs(s(2)),abs(s(3)))
			if (  xmax .lt. smax ) then
			    dqiqj(lk,ji) = s(1) + s(2) + s(3)
			    smax = xmax
			    if ( lwrite ) print *,'dqiqj(',lk,ji,')+2=',
     +				dqiqj(lk,ji),s(1),s(2),s(3),ier2
			    if ( abs(dqiqj(lk,ji)) .ge. xloss*smax )
     +				goto 245
			endif
*
*			give up
*
			if ( lwarn ) call ffwarn(127,ier0,dqiqj(lk,ji),
     +				smax)
			if ( lwrite ) print *,' (between qi(',lk,
     +				') and qi(',ji,'))'

  245			continue
			dqiqj(ji,lk) = -dqiqj(lk,ji)
			ier2 = max(ier2,ier0)
  250		    continue
  260		continue
  270	    continue
  280	continue
	ier = ier2
*  #] pi^2 - pj^2: 
*  #[ debug:
	if ( lwrite ) then
	    print *,'fftran: transformed momenta'
	    print *,xqi
	    print '(10e16.8)',qiDqj
	    print *,'ier = ',ier
	endif
*  #] debug: 
*###] fftran:
	end
