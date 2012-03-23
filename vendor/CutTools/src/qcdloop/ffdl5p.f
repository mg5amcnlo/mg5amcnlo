*--#[ log:
*	$Id: ffdl5p.f,v 1.3 1996/02/12 21:06:19 gj Exp $
*	$Log: ffdl5p.f,v $
c Revision 1.3  1996/02/12  21:06:19  gj
c Added safety check on ns in ffdl5r, updated comment
c
c Revision 1.2  1995/12/08  10:44:14  gj
c Added forgotten 'abs' in error calculation.
c
*--#] log:
*###[ ffdl5p:
	subroutine ffdl5p(xpi,pDp,ns,ii,ier)
***#[*comment:***********************************************************
*	check that							*
*									*
*	     p1 p2 p3 p4 p5		     s1 p1 p2 p3 p4		*
*	delta		    = 0,	delta		    = 0		*
*	     p1 p2 p3 p4 p5		     p1 p2 p3 p4 p5		*
*									*
*	with pn = xpi(ii(n)), n=1,5					*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,ii(5),ier
	DOUBLE PRECISION xpi(ns),pDp(ns,ns)
*
*	local variables
*
	integer i,j1,j2,j3,j4,j5
	DOUBLE PRECISION s(109),som,xmax,xlosn
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ del5(p):
	j1 = ii(1)
	j2 = ii(2)
	j3 = ii(3)
	j4 = ii(4)
	j5 = ii(5)
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
*
	som = 0
	xmax = 0
	do 80 i=1,73
	    som = som + s(i)
	    xmax = max(xmax,abs(som))
   80	continue
	xlosn = xloss*DBLE(10)**(-1-mod(ier,50))
	if ( xlosn*abs(som) .gt. precx*xmax )
     +	    print *,'ffdl5p: error: dl5p != 0: ',som,xmax
	if ( lwrite ) print *,'ffdl5p: dl5p = ',som,xmax
*
*  #] del5(p):
*###] ffdl5p:
	end
*###[ ffdl5r:
	subroutine ffdl5r(dl5r,xpi,piDpj,ns,inum,ier)
***#[*comment:***********************************************************
*	calculate in a numerically stable way				*
*									*
*	     s1 pi+1 pi+2 pi+3 pi+4					*
*	delta								*
*	     pi pi+1 pi+2 pi+3 pi+4					*
*									*
*	ier is the usual error flag.					*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments:
*
	integer ns,inum,ier
	DOUBLE PRECISION dl5r,xpi(ns),piDpj(ns,ns)
*
*	local variables
*
	integer i,j,k,is,ip(5),ii(10),jj(15),i54(10,6)
	logical lagain
	DOUBLE PRECISION s(109),som,xmax,smax
	DOUBLE PRECISION t10,t101,t104,t105,t108,t109,t112,t116,
     +		t120,t121,t128,t129,t13,t132,t135,t139,t14,t143,t146,
     +		t147,t148,t15,t16,t182,t185,t19,t190,t194,t2,t20,t202,
     +		t203,t206,t21,t210,t214,t218,t22,t222,t230,t234,t235,
     +		t25,t26,t27,t275,t28,t282,t285,t289,t29,t295,t298,t30,
     +		t302,t33,t367,t37,t42,t49,t5,t53,t54,t58,t6,t68,t69,t74,
     +		t75,t79,t80,t81,t85,t86,t89,t9,t92,t97
	save i54
*
*	common blocks
*
	include 'ff.h'
*  #] declarations:
*  #[ data:
*
*	data
*
	data i54/
     +		 8, 9,10,11, 18, 14,15,16, 20,21,
     +		 9,10,11,12, 13, 15,16,17, 19,21,
     +		10,11,12, 7, 14, 16,17,18, 19,20,
     +		11,12, 7, 8, 15, 17,18,13, 20,21,
     +		12, 7, 8, 9, 16, 18,13,14, 19,21,
     +		 7, 8, 9,10, 17, 13,14,15, 19,20/
*
*  #] data:
*  #[ check input:
	if ( ltest ) then
	    if ( inum.gt.6 .or. inum.lt.1 ) then
		print *,'ffdl5r: error: inum < 1 or > 6: ',inum
		stop
	    endif
	endif
	if ( ns.ne.21 ) then
	    print *,'ffdl5r: only for 6point pi, ns should be 21, not ',
     +		ns
	    stop
	endif
*  #] check input:
*  #[ calculations:
*
	is = 1
	do 10 i=1,10
	    s(i) = abs(piDpj(i54(i,inum),is))
   10	continue
	call ffsort(s,ii,10)
	do 20 i=1,10
	    jj(i) = i54(ii(i),inum)
   20	continue
*	just for safety...
	jj(11) = -99999
	ip(1) = inum + 6
	call ff5ind(ip,jj,1,ier)
	lagain = .FALSE.
*
*	we compute  \delta^{pi pa pb pc pd pe}_{s1 pa pb pc pd pe}
*	with {pi,pa-pe} lin.independent, pa-pe arbitrary.  This way we
*	never need to determine the sign of (pa-pe) w.r.t. (pi+1 - pi+4)
*	see dl5r.frm -> dl5r.map
*
   30	continue
	if ( lwrite ) print *,'ip = ',ip
*
* 	#[ define t's:
	t2 = piDpj(is,ip(1))
	t5 = piDpj(ip(2),ip(5))
	t6 = t5**2
	t9 = piDpj(ip(3),ip(4))
	t10 = t9**2
	t13 = piDpj(ip(2),ip(3))
	t14 = t13**2
	t15 = piDpj(ip(4),ip(5))
	t16 = t15**2
	t19 = piDpj(ip(2),ip(4))
	t20 = t19**2
	t21 = piDpj(ip(3),ip(5))
	t22 = t21**2
	t25 = piDpj(ip(2),ip(2))
	t26 = piDpj(ip(3),ip(3))
	t27 = piDpj(ip(4),ip(4))
	t28 = piDpj(ip(5),ip(5))
	t29 = t27*t28
	t30 = t26*t29
	t33 = t26*t16
	t37 = t10*t28
	t42 = t22*t27
	t49 = t26*t27
	t53 = piDpj(is,ip(2))
	t54 = piDpj(ip(1),ip(2))
	t58 = t26*t28
	t68 = piDpj(ip(1),ip(3))
	t69 = t13*t29
	t74 = t21*t27
	t75 = t5*t74
	t79 = piDpj(ip(1),ip(4))
	t80 = t9*t28
	t81 = t13*t80
	t85 = t21*t15
	t86 = t13*t85
	t89 = t19*t58
	t92 = t13*t16
	t97 = t19*t80
	t101 = t19*t85
	t104 = t9*t15
	t105 = t5*t104
	t108 = piDpj(ip(1),ip(5))
	t109 = t5*t49
	t112 = t5*t10
	t116 = t19*t22
	t120 = t26*t15
	t121 = t5*t120
	t128 = t9*t21
	t129 = t5*t128
	t132 = t13*t104
	t135 = t13*t74
	t139 = t19*t120
	t143 = t19*t128
	t146 = piDpj(is,ip(3))
	t147 = t5*t15
	t148 = t13*t147
	t182 = t25*t80
	t185 = t25*t85
	t190 = t13*t19*t28
	t194 = piDpj(is,ip(4))
	t202 = t5*t21
	t203 = t19*t202
	t206 = t6*t9
	t210 = t25*t104
	t214 = t25*t74
	t218 = t13*t19*t15
	t222 = t13*t5*t27
	t230 = t20*t21
	t234 = t5*t9
	t235 = t19*t234
	t275 = piDpj(is,ip(5))
	t282 = t25*t120
	t285 = t25*t128
	t289 = t14*t15
	t295 = t13*t19*t21
	t298 = t13*t234
	t302 = t19*t5*t26
	t367 = t9*t85
* 	#] define t's:
* 	#[ fill s-array:
	s(1) = +t2*t20*t22
	s(2) = -t146*t79*t206
	s(3) = +t275*t108*t20*t26
	s(4) = +t146*t108*t235
	s(5) = -t146*t54*t75
	s(6) = -t146*t79*t190
	s(7) = -2*t53*t54*t367
	s(8) = -2*t2*t19*t129
	s(9) = +2*t2*t19*t121
	s(10) = -2*t2*t13*t105
	s(11) = -2*t2*t13*t101
	s(12) = +2*t2*t13*t97
	s(13) = +2*t2*t25*t367
	s(14) = -t2*t25*t33
	s(15) = -t275*t79*t302
	s(16) = +t275*t79*t298
	s(17) = +t275*t79*t295
	s(18) = -t275*t79*t289
	s(19) = -t275*t79*t285
	s(20) = +t275*t79*t282
	s(21) = -t194*t79*t25*t58
	s(22) = +t275*t68*t235
	s(23) = -t275*t68*t222
	s(24) = +t275*t68*t218
	s(25) = +t275*t68*t214
	s(26) = -t275*t68*t210
	s(27) = -t275*t54*t112
	s(28) = +t146*t79*t148
	s(29) = +t275*t54*t109
	s(30) = +t275*t54*t143
	s(31) = -t275*t54*t139
	s(32) = -t275*t54*t135
	s(33) = -t194*t108*t302
	s(34) = -t2*t14*t29
	s(35) = +t194*t108*t298
	s(36) = -t2*t20*t58
	s(37) = +t194*t108*t295
	s(38) = +t146*t54*t105
	s(39) = -t194*t108*t289
	s(40) = +t53*t68*t101
	s(41) = -t194*t68*t206
	s(42) = +t2*t25*t30
	s(43) = +t194*t68*t203
	s(44) = +t2*t6*t10
	s(45) = +t53*t54*t33
	s(46) = +t194*t68*t148
	s(47) = +t53*t79*t86
	s(48) = -t194*t68*t185
	s(49) = +t194*t68*t182
	s(50) = +t194*t54*t129
	s(51) = -t194*t54*t121
	s(52) = -t194*t54*t116
	s(53) = +t194*t54*t89
	s(54) = -t53*t108*t139
	s(55) = -t194*t54*t81
	s(56) = -t53*t79*t116
	s(57) = -t194*t108*t285
	s(58) = +t146*t54*t69
	s(59) = -t146*t108*t222
	s(60) = -t53*t68*t92
	s(61) = -t146*t108*t230
	s(62) = -t2*t25*t42
	s(63) = +t53*t54*t37
	s(64) = +t275*t54*t132
	s(65) = +t194*t54*t86
	s(66) = +t53*t108*t109
	s(67) = +t2*t14*t16
	s(68) = +t146*t108*t218
	s(69) = -t2*t25*t37
	s(70) = -t53*t68*t75
	s(71) = +t53*t54*t42
	s(72) = -t2*t6*t49
	s(73) = +t53*t68*t105
	s(74) = +2*t2*t13*t75
	s(75) = -t194*t68*t190
	s(76) = +t146*t54*t101
	s(77) = +t53*t108*t132
	s(78) = -t53*t108*t135
	s(79) = -t53*t68*t97
	s(80) = -t53*t54*t30
	s(81) = -t146*t54*t92
	s(82) = -t146*t79*t185
	s(83) = +t146*t79*t203
	s(84) = -t146*t54*t97
	s(85) = -t275*t68*t230
	s(86) = -t146*t108*t210
	s(87) = +t53*t79*t129
	s(88) = +t53*t108*t143
	s(89) = -t53*t108*t112
	s(90) = +t53*t79*t89
	s(91) = +t194*t108*t282
	s(92) = -2*t275*t108*t13*t19*t9
	s(93) = +t146*t79*t182
	s(94) = -2*t194*t79*t13*t202
	s(95) = -2*t146*t68*t19*t147
	s(96) = +t146*t108*t214
	s(97) = -t53*t79*t81
	s(98) = -t53*t79*t121
	s(99) = +t275*t108*t14*t27
	s(100) = +t275*t108*t25*t10
	s(101) = -t275*t108*t25*t49
	s(102) = +t194*t79*t25*t22
	s(103) = +t146*t68*t20*t28
	s(104) = +t194*t79*t14*t28
	s(105) = -t146*t68*t25*t29
	s(106) = +t146*t68*t25*t16
	s(107) = +t53*t68*t69
	s(108) = +t194*t79*t6*t26
	s(109) = +t146*t68*t6*t27
* 	#] fill s-array:
*
	som = 0
	xmax = 0
	do 100 i=1,109
	    som = som + s(i)
	    xmax = max(xmax,abs(s(i)))
  100	continue
*
	if ( .not.lagain ) then
	    dl5r = som
	    smax = xmax
	    if ( lwrite ) print *,'dl5r = ',dl5r,xmax
	    if ( lwarn ) call ffwarn(188,ier,dl5r,xmax)
	    if ( ltest ) then
		do 900 i=2,5
		    k = inum + i - 1 + 6
		    if ( k.gt.12 ) k = k-6
		    ip(i) = k
  900		continue
		lagain = .TRUE.
		goto 30
	    endif
	else
	    if ( xloss*abs(som-dl5r) .gt. precx*max(smax,xmax) ) then
		print *,'ffdl5r: error: is not what it should be: ',
     +			dl5r,som,dl5r-som,max(smax,xmax)
	    endif
	endif
*
*  #] calculations:
*###] ffdl5r:
	end
