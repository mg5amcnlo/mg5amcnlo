*	$Id: ffdel6.f,v 1.4 1996/03/14 15:53:15 gj Exp $
*###[ ffdel6:
	subroutine ffdel6(del6s,xpi,piDpj,ns,ier)
***#[*comment:***********************************************************
*									*
*	compute the coefficient of the F0 in the decomposition in 5 E0s	*
*	note that this is not a proper determinant as the s_i do not	*
*	exist when the p_i live in 4-space.				*
*									*
*		     s1 p1 p2 p3 p4 p5					*
*	del6 = delta							*
*		     s1 p1 p2 p3 p4 p5					*
*									*
*	Input:	xpi	real(ns)	1-6: mi_2, 7-21: p_i^2		*
*		piDpj	real(ns,ns)	pi.pj				*
*		ns	integer		assumed 21 for the time being	*
*		ier	integer		usual error flag		*
*	Output	del6s	real						*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit logical (a-r,u-z)
	implicit DOUBLE PRECISION (s,t)
*
*	arguments
*
	integer ns,ier
	DOUBLE PRECISION del6s,xpi(21),piDpj(21,21)
*
*	local vars
*
	integer i,is,ip(5),ii(15)
	DOUBLE PRECISION som(315),xmx,sum,xmax
*
*	common blocks
*
	include 'ff.h'
*
*  #] declarations:
*  #[ check input:
	if ( ltest ) then
	    if ( ns.ne.21 ) then
		print *,'ffdel6: only for ns=21 for the time being'
		stop
	    endif
	endif
	if ( lwrite ) then
	    print *,'ffdel6: input '
	    print *,'xpi = ',xpi
	endif
*  #] check input:
*  #[ work:
	do 100 is=1,6
*
*	find a linearly independent set ipi such that s.pi minimal
*
	do 5 i=1,15
	    som(i) = abs(piDpj(6+i,is))
    5	continue
	call ffsort(som,ii,15)
	do 6 i=1,15
	    ii(i) = ii(i)+6
    6	continue
	call ff5ind(ip,ii,0,ier)
*
*	not so straight from Maple
*
	t1 = piDpj(is,ip(3))
	t2 = t1**2
	t3 = piDpj(ip(1),ip(4))
	t4 = t3**2
	t5 = piDpj(ip(2),ip(5))
	t6 = t5**2
	t10 = piDpj(is,ip(2))
	t11 = t10**2
	t12 = piDpj(ip(1),ip(5))
	t13 = t12**2
	t14 = piDpj(ip(3),ip(4))
	t15 = t14**2
	t19 = piDpj(is,ip(5))
	t20 = t19**2
	t21 = piDpj(ip(2),ip(3))
	t22 = t21**2
	t26 = piDpj(ip(1),ip(3))
	t27 = t26**2
	t28 = piDpj(ip(2),ip(4))
	t29 = t28**2
	t33 = piDpj(ip(4),ip(5))
	t34 = t33**2
	t38 = piDpj(ip(1),ip(2))
	t39 = t38**2
	t43 = xpi(ip(1))
	t44 = xpi(ip(2))
	t45 = xpi(ip(3))
	t46 = xpi(ip(4))
	t52 = xpi(ip(5))
	t53 = piDpj(is,ip(4))
	t54 = t53**2
	t66 = piDpj(ip(3),ip(5))
	t67 = t66**2
	t77 = piDpj(is,ip(1))
	t78 = t77**2
	t222 = t66*t33
	t228 = t14*t33
	t234 = t14*t66
	t254 = t5*t33
	t260 = t28*t33
	t266 = t28*t5
	t278 = t5*t66
	t284 = t21*t33
	t285 = t12*t284
	t290 = t21*t66
	t296 = t21*t5
	t302 = t28*t14
	t308 = t21*t14
	t315 = t21*t28
	t321 = t14*t222
	t325 = t21*t34
	t330 = t28*t222
	t335 = t5*t228
	t340 = t21*t222
	t345 = t28*t67
	t350 = t5*t234
	t355 = t21*t228
	t360 = t28*t234
	t369 = t5*t15
	t374 = t28*t254
	t378 = t21*t254
	t383 = t28*t278
	t388 = t6*t14
	t393 = t21*t260
	t398 = t29*t66
	t403 = t5*t14
	t404 = t28*t403
	t409 = t21*t278
	t414 = t22*t33
	t419 = t28*t66
	t420 = t21*t419
	t425 = t21*t403
	t430 = t21*t302
	t446 = t12*t33
	t452 = t3*t33
	t472 = t12*t66
	t478 = t26*t66
	t494 = t3*t14
	t500 = t26*t14
	t515 = t26*t34
	t520 = t3*t222
	t526 = t12*t228
	t531 = t26*t222
	t536 = t3*t67
	t541 = t12*t234
	t546 = t26*t228
	t551 = t3*t234
	t556 = t12*t15
	t561 = t3*t446
	t593 = t12*t14
	t599 = t26*t472
	t619 = t26*t494
	t630 = t12*t5
	t648 = t3*t28
	t674 = t3*t254
	t679 = t12*t260
	t691 = t3*t6
	t696 = t12*t266
	t706 = t3*t266
	t711 = t12*t29
	t745 = t12*t28
	t751 = t38*t630
	t771 = t38*t648
	t775 = t26*t21
	t806 = t26*t278
	t811 = t12*t290
	t826 = t12*t296
	t842 = t12*t22
	t879 = t12*t21
	t906 = t38*t775
	t918 = t26*t302
	t923 = t3*t308
	t939 = t3*t315
	t990 = t3*t21
	t1231 = t12*t419
	t1236 = t12*t403
	t1261 = t3*t278
	t1303 = t3*t284
	t1308 = t3*t419
	t1313 = t3*t403
	t1321 = t12*t302
	t1330 = t12*t308
	t1417 = t12*t315
	som(1) = +t45*t52*t78*t29
	som(2) = +t44*t52*t2*t4
	som(3) = +t44*t52*t78*t15
	som(4) = +t44*t46*t2*t13
	som(5) = +t44*t46*t20*t27
	som(6) = +t43*t46*t20*t22
	som(7) = -2*t45*t46*t77*t10*t630
	som(8) = -t2*t39*t34
	som(9) = +t44*t46*t78*t67
	som(10) = -2*t45*t54*t751
	som(11) = +t44*t45*t20*t4
	som(12) = +t44*t45*t54*t13
	som(13) = +t44*t45*t78*t34
	som(14) = -t44*t45*t46*t52*t78
	som(15) = +t44*t52*t54*t27
	som(16) = +t43*t45*t20*t29
	som(17) = +t43*t52*t11*t15
	som(18) = +t43*t45*t54*t6
	som(19) = +2*t52*t77*t53*t26*t315
	som(20) = -t20*t4*t22
	som(21) = +2*t45*t46*t52*t77*t10*t38
	som(22) = -t11*t4*t67
	som(23) = -t78*t22*t34
	som(24) = -t2*t13*t29
	som(25) = +2*t78*t28*t350
	som(26) = +2*t78*t21*t335
	som(27) = +2*t78*t21*t330
	som(28) = -2*t52*t54*t906
	som(29) = -2*t52*t2*t771
	som(30) = -t11*t13*t15
	som(31) = +2*t52*t77*t53*t38*t308
	som(32) = +2*t44*t53*t19*t26*t593
	som(33) = +2*t52*t77*t1*t939
	som(34) = -2*t52*t11*t619
	som(35) = -2*t52*t77*t1*t26*t29
	som(36) = +2*t44*t53*t19*t26*t3*t66
	som(37) = -t20*t39*t15
	som(38) = +2*t52*t77*t1*t38*t302
	som(39) = -2*t52*t78*t430
	som(40) = -2*t46*t20*t906
	som(41) = -2*t44*t53*t19*t27*t33
	som(42) = -2*t46*t2*t751
	som(43) = +2*t52*t77*t10*t923
	som(44) = +2*t44*t1*t19*t3*t593
	som(45) = +2*t52*t77*t10*t918
	som(46) = -2*t52*t77*t10*t38*t15
	som(47) = -2*t44*t1*t19*t4*t66
	som(48) = +t43*t52*t54*t22
	som(49) = +2*t44*t1*t19*t26*t452
	som(50) = -2*t44*t1*t53*t13*t14
	som(51) = -2*t46*t11*t599
	som(52) = +2*t44*t1*t53*t3*t472
	som(53) = +2*t46*t1*t19*t38*t879
	som(54) = +2*t44*t1*t53*t26*t446
	som(55) = +2*t46*t1*t19*t38*t26*t5
	som(56) = -2*t44*t77*t19*t556
	som(57) = +2*t44*t77*t19*t551
	som(58) = -2*t46*t1*t19*t39*t66
	som(59) = +t43*t46*t11*t67
	som(60) = +2*t44*t77*t19*t546
	som(61) = +2*t46*t10*t19*t26*t879
	som(62) = +2*t44*t77*t53*t541
	som(63) = -2*t44*t77*t53*t536
	som(64) = +2*t44*t77*t53*t531
	som(65) = -2*t46*t10*t19*t27*t5
	som(66) = +2*t44*t77*t1*t526
	som(67) = -2*t46*t78*t409
	som(68) = +2*t46*t10*t19*t38*t478
	som(69) = +2*t44*t77*t1*t520
	som(70) = -2*t46*t10*t1*t13*t21
	som(71) = -t54*t27*t6
	som(72) = -2*t44*t77*t1*t515
	som(73) = -2*t45*t20*t771
	som(74) = -2*t44*t52*t1*t53*t26*t3
	som(75) = +2*t46*t10*t1*t26*t630
	som(76) = +2*t46*t10*t1*t38*t472
	som(77) = -2*t44*t52*t77*t53*t500
	som(78) = -2*t44*t46*t1*t19*t26*t12
	som(79) = -2*t44*t52*t77*t1*t494
	som(80) = -2*t1*t19*t26*t706
	som(81) = -2*t45*t11*t561
	som(82) = -2*t46*t77*t19*t842
	som(83) = +2*t46*t77*t19*t26*t296
	som(84) = +4*t77*t10*t38*t321
	som(85) = -2*t44*t46*t77*t19*t478
	som(86) = +2*t46*t77*t19*t38*t290
	som(87) = -2*t44*t46*t77*t1*t472
	som(88) = +2*t46*t77*t1*t826
	som(89) = -t54*t13*t22
	som(90) = -2*t45*t78*t374
	som(91) = +2*t44*t46*t52*t77*t1*t26
	som(92) = -2*t46*t77*t1*t26*t6
	som(93) = -2*t44*t20*t619
	som(94) = -2*t44*t45*t53*t19*t3*t12
	som(95) = -t54*t39*t67
	som(96) = -2*t44*t54*t599
	som(97) = +2*t46*t77*t1*t38*t278
	som(98) = -t2*t4*t6
	som(99) = +2*t20*t26*t939
	som(100) = -2*t44*t45*t77*t19*t452
	som(101) = -2*t44*t78*t321
	som(102) = -2*t44*t2*t561
	som(103) = +2*t46*t77*t10*t811
	som(104) = -2*t44*t45*t77*t53*t446
	som(105) = +2*t46*t77*t10*t806
	som(106) = +2*t53*t19*t3*t842
	som(107) = +2*t44*t45*t52*t77*t53*t3
	som(108) = +2*t44*t45*t46*t77*t19*t12
	som(109) = -2*t43*t20*t430
	som(110) = -t78*t6*t15
	som(111) = -2*t46*t77*t10*t38*t67
	som(112) = +2*t43*t53*t19*t425
	som(113) = -2*t53*t19*t26*t1417
	som(114) = -2*t46*t52*t10*t1*t38*t26
	som(115) = -2*t53*t19*t26*t3*t296
	som(116) = +2*t53*t19*t27*t266
	som(117) = +2*t43*t53*t19*t420
	som(118) = -2*t77*t10*t26*t335
	som(119) = -2*t53*t19*t38*t1330
	som(120) = -2*t43*t53*t19*t414
	som(121) = -2*t46*t52*t77*t1*t38*t21
	som(122) = -2*t43*t2*t374
	som(123) = +2*t43*t1*t19*t404
	som(124) = -2*t43*t54*t409
	som(125) = -2*t53*t19*t38*t3*t290
	som(126) = -2*t43*t1*t19*t398
	som(127) = +2*t43*t1*t19*t393
	som(128) = -2*t43*t1*t53*t388
	som(129) = +2*t43*t1*t53*t383
	som(130) = -2*t46*t52*t77*t10*t775
	som(131) = +2*t43*t1*t53*t378
	som(132) = +2*t20*t38*t918
	som(133) = +2*t20*t38*t923
	som(134) = -2*t53*t19*t38*t26*t403
	som(135) = -2*t43*t10*t19*t369
	som(136) = -2*t53*t19*t38*t26*t419
	som(137) = +2*t43*t10*t19*t360
	som(138) = +2*t45*t53*t19*t38*t745
	som(139) = +4*t53*t19*t38*t26*t284
	som(140) = -2*t43*t11*t321
	som(141) = +2*t53*t19*t39*t234
	som(142) = +2*t43*t10*t19*t355
	som(143) = +2*t45*t53*t19*t38*t3*t5
	som(144) = +2*t43*t10*t53*t350
	som(145) = -2*t43*t10*t53*t345
	som(146) = +2*t43*t10*t53*t340
	som(147) = +2*t43*t10*t1*t335
	som(148) = +2*t54*t38*t806
	som(149) = +2*t54*t38*t811
	som(150) = +2*t54*t26*t826
	som(151) = -t20*t27*t29
	som(152) = -2*t43*t10*t1*t325
	som(153) = +2*t43*t10*t1*t330
	som(154) = -2*t1*t19*t3*t1417
	som(155) = -t78*t29*t67
	som(156) = +2*t1*t19*t4*t296
	som(157) = -2*t45*t53*t19*t39*t33
	som(158) = -2*t43*t52*t1*t53*t315
	som(159) = +2*t1*t19*t26*t711
	som(160) = -2*t43*t52*t10*t53*t308
	som(161) = +2*t45*t10*t19*t3*t745
	som(162) = -2*t43*t52*t10*t1*t302
	som(163) = -2*t43*t46*t1*t19*t296
	som(164) = -2*t1*t19*t38*t1321
	som(165) = -2*t45*t10*t19*t4*t5
	som(166) = -2*t43*t46*t10*t19*t290
	som(167) = +4*t1*t19*t38*t1308
	som(168) = -2*t1*t19*t38*t1313
	som(169) = +2*t45*t10*t19*t38*t452
	som(170) = -2*t1*t53*t38*t285
	som(171) = -2*t1*t19*t38*t1303
	som(172) = -2*t1*t19*t38*t26*t260
	som(173) = -2*t43*t46*t10*t1*t278
	som(174) = +2*t43*t46*t52*t10*t1*t21
	som(175) = -2*t45*t10*t53*t13*t28
	som(176) = +2*t1*t19*t39*t228
	som(177) = +2*t1*t53*t13*t315
	som(178) = +2*t45*t10*t53*t3*t630
	som(179) = -2*t43*t45*t53*t19*t266
	som(180) = -2*t1*t53*t3*t826
	som(181) = -2*t1*t53*t26*t696
	som(182) = +2*t45*t10*t53*t38*t446
	som(183) = +2*t1*t53*t26*t691
	som(184) = -2*t43*t45*t10*t19*t260
	som(185) = +4*t1*t53*t38*t1236
	som(186) = -2*t1*t53*t38*t1231
	som(187) = +2*t43*t45*t52*t10*t53*t28
	som(188) = -2*t45*t77*t19*t711
	som(189) = -2*t1*t53*t38*t1261
	som(190) = +2*t43*t45*t46*t10*t19*t5
	som(191) = -2*t1*t53*t38*t26*t254
	som(192) = +2*t45*t77*t19*t706
	som(193) = +2*t1*t53*t39*t222
	som(194) = -2*t43*t44*t53*t19*t234
	som(195) = -2*t43*t44*t1*t19*t228
	som(196) = +2*t2*t38*t674
	som(197) = +2*t2*t38*t679
	som(198) = +2*t2*t3*t696
	som(199) = -2*t10*t19*t3*t1330
	som(200) = +2*t10*t19*t4*t290
	som(201) = +2*t45*t77*t19*t38*t260
	som(202) = +2*t43*t44*t52*t1*t53*t14
	som(203) = -2*t10*t19*t26*t1321
	som(204) = -2*t43*t44*t1*t53*t222
	som(205) = -t11*t27*t34
	som(206) = +t43*t52*t2*t29
	som(207) = +4*t10*t19*t26*t1313
	som(208) = +2*t43*t44*t46*t1*t19*t66
	som(209) = +2*t43*t44*t45*t53*t19*t33
	som(210) = +2*t45*t77*t53*t696
	som(211) = -2*t10*t19*t26*t1308
	som(212) = -t43*t44*t45*t52*t54
	som(213) = -2*t45*t77*t53*t691
	som(214) = -2*t10*t19*t26*t1303
	som(215) = -2*t43*t45*t10*t53*t254
	som(216) = +t45*t46*t78*t6
	som(217) = +2*t45*t77*t53*t38*t254
	som(218) = +2*t10*t19*t27*t260
	som(219) = +2*t10*t19*t38*t556
	som(220) = -2*t10*t19*t38*t551
	som(221) = +t43*t46*t2*t6
	som(222) = -2*t10*t19*t38*t546
	som(223) = +2*t10*t53*t13*t308
	som(224) = -2*t10*t53*t3*t811
	som(225) = -2*t10*t53*t26*t1236
	som(226) = +4*t10*t53*t26*t1231
	som(227) = +t43*t45*t11*t34
	som(228) = +2*t45*t77*t10*t679
	som(229) = -2*t10*t53*t26*t285
	som(230) = -2*t10*t53*t26*t1261
	som(231) = +2*t10*t53*t27*t254
	som(232) = +2*t45*t77*t10*t674
	som(233) = -2*t10*t53*t38*t541
	som(234) = +2*t10*t53*t38*t536
	som(235) = -2*t10*t53*t38*t531
	som(236) = +2*t10*t1*t13*t302
	som(237) = +t46*t52*t2*t39
	som(238) = -2*t45*t77*t10*t38*t34
	som(239) = -2*t10*t1*t3*t1231
	som(240) = -2*t10*t1*t3*t1236
	som(241) = +4*t10*t1*t3*t285
	som(242) = +t46*t52*t11*t27
	som(243) = +2*t10*t1*t4*t278
	som(244) = -2*t45*t52*t10*t53*t38*t3
	som(245) = -2*t10*t1*t26*t679
	som(246) = -2*t10*t1*t26*t674
	som(247) = +t46*t52*t78*t22
	som(248) = -2*t10*t1*t38*t526
	som(249) = -2*t10*t1*t38*t520
	som(250) = +2*t10*t1*t38*t515
	som(251) = +4*t77*t19*t12*t430
	som(252) = -2*t77*t19*t3*t425
	som(253) = -2*t77*t19*t3*t420
	som(254) = +2*t77*t19*t3*t414
	som(255) = -2*t77*t19*t26*t404
	som(256) = +t45*t52*t54*t39
	som(257) = +2*t77*t19*t26*t398
	som(258) = -2*t45*t52*t77*t53*t38*t28
	som(259) = -2*t77*t19*t26*t393
	som(260) = +2*t77*t19*t38*t369
	som(261) = -2*t77*t19*t38*t360
	som(262) = -2*t45*t52*t77*t10*t648
	som(263) = -2*t77*t19*t38*t355
	som(264) = -2*t77*t53*t12*t425
	som(265) = -2*t77*t53*t12*t420
	som(266) = +2*t11*t26*t520
	som(267) = +2*t11*t26*t526
	som(268) = +2*t11*t3*t541
	som(269) = +t45*t46*t20*t39
	som(270) = +2*t77*t53*t12*t414
	som(271) = +4*t77*t53*t3*t409
	som(272) = +2*t77*t53*t26*t388
	som(273) = -2*t77*t53*t26*t383
	som(274) = -2*t77*t53*t26*t378
	som(275) = -2*t77*t53*t38*t350
	som(276) = +2*t77*t53*t38*t345
	som(277) = -2*t77*t53*t38*t340
	som(278) = -2*t77*t1*t12*t404
	som(279) = +2*t77*t1*t12*t398
	som(280) = -2*t77*t1*t12*t393
	som(281) = +2*t77*t1*t3*t388
	som(282) = -2*t45*t46*t10*t19*t38*t12
	som(283) = -2*t77*t1*t3*t383
	som(284) = -2*t77*t1*t3*t378
	som(285) = +4*t77*t1*t26*t374
	som(286) = -2*t77*t1*t38*t335
	som(287) = -2*t77*t1*t38*t330
	som(288) = +t43*t44*t2*t34
	som(289) = +2*t77*t1*t38*t325
	som(290) = +2*t77*t10*t12*t369
	som(291) = -2*t77*t10*t12*t360
	som(292) = -2*t77*t10*t12*t355
	som(293) = -2*t77*t10*t3*t350
	som(294) = +2*t77*t10*t3*t345
	som(295) = -2*t77*t10*t3*t340
	som(296) = +t45*t46*t11*t13
	som(297) = -t43*t45*t46*t52*t11
	som(298) = -2*t77*t10*t26*t330
	som(299) = +2*t77*t10*t26*t325
	som(300) = +2*t52*t1*t53*t38*t990
	som(301) = +2*t52*t1*t53*t38*t26*t28
	som(302) = -2*t52*t1*t53*t39*t14
	som(303) = +t43*t44*t54*t67
	som(304) = +2*t52*t10*t53*t26*t990
	som(305) = +t45*t52*t11*t4
	som(306) = -2*t52*t10*t53*t27*t28
	som(307) = +2*t52*t10*t53*t38*t500
	som(308) = -2*t52*t10*t1*t4*t21
	som(309) = -2*t45*t46*t77*t19*t38*t5
	som(310) = +2*t52*t10*t1*t26*t648
	som(311) = +2*t52*t10*t1*t38*t494
	som(312) = -t43*t44*t45*t46*t20
	som(313) = -2*t52*t77*t53*t3*t22
	som(314) = -t43*t44*t46*t52*t2
	som(315) = +t43*t44*t20*t15

	sum = 0
	xmx = 0
	do 10 i=1,315
	    sum = sum + som(i)
	    xmx = max(xmx,abs(som(i)))
   10	continue
	if ( lwrite ) then
	    print *,'ffdel6s: del6s',is,' = ',sum,xmx
	endif
	if ( is.eq.1 ) then
	    del6s = sum
	    xmax = xmx
	endif
	if ( xmx.lt.xmax ) then
	    del6s = sum
	    xmax = xmx
	endif
	if ( abs(del6s) .gt. xloss**2*xmax ) goto 110
  100	continue
	if ( lwarn ) call ffwarn(187,ier,sum,xmx)
  110	continue
*
*  #] work:
*###] ffdel6:
	end
*###[ ffsort:
	subroutine ffsort(a,ii,nn)
***#[*comment:***********************************************************
*									*
*	Sort the array a(nn): give the position of the smallest element	*
*	in ii(1), ..., largest in ii(nn).  I use a fancy merge-sort	*
*	algorithm which is probably not the samrtest thing to do with	*
*	the small arrays for which it is used, but it was fun to program*
*	To extend to larger arrays: just change 1024 to some power of 2	*
*									*
*	Input:	a	real(nn)		array			*
*		nn	integer						*
*	Output:	ii	integer(nn)	a(ii(1))<=a(ii(2))<=.<=a(ii(nn))*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer nn,ii(nn)
	DOUBLE PRECISION a(nn)
*
*	local variables
*
	integer i,j,k,jj(1024,2),h,j12,j21,l,m,n,o
*
*	common
*
	include 'ff.h'
*
*  #] declarations:
*  #[ work:
	if ( nn.gt.1024 ) then
	    print  *,'ffsort: can only sort up to 1024 elments, not ',nn
	    stop
	endif
	do 10 i=1,nn
	    jj(i,1) = i
   10	continue
	j12 = 1
	j21 = 2
*
*	do the first sweep faster
*
	do 15 i=1,nn-1,2
	    if ( a(jj(i,j12)) .le. a(jj(i+1,j12)) ) then
		jj(i,j21) = jj(i,j12)
		jj(i+1,j21) = jj(i+1,j12)
	    else
		jj(i,j21) = jj(i+1,j12)
		jj(i+1,j21) = jj(i,j12)
	    endif
   15	continue
	if ( mod(nn,2).ne.0 ) jj(nn,j21) = jj(nn,j12)
	o   = j12
	j12 = j21
	j21 = o
*
*	and do the other sweeps (works also for k=1,10)
*
	do 100 k=2,nint(log(dble(1024))/log(dble(2)))
	    h = 2**k
	    do 90 j=1,nn,h
		l = j
		n = j
		m = j+h/2
		if ( m.gt.nn ) then
		    do 17 o=j,nn
			jj(o,j21) = jj(o,j12)
   17		    continue
		    goto 90
		endif
		do 20 i=1,2*1024
		    if ( a(jj(l,j12)) .le. a(jj(m,j12)) ) then
			jj(n,j21) = jj(l,j12)
			l = l+1
			n = n+1
			if ( l.ge.j+h/2 ) then
			    do 18 o=m,min(j+h-1,nn)
				jj(n,j21) = jj(o,j12)
				n = n+1
   18			    continue
			    goto 21
			endif
		    else
			jj(n,j21) = jj(m,j12)
			m = m+1
			n = n+1
			if ( m.ge.j+h .or. m.gt.nn ) then
			    do 19 o=l,j+h/2-1
				jj(n,j21) = jj(o,j12)
				n = n+1
   19			    continue
			    goto 21
			endif
		    endif
   20		continue
   21		continue
		if ( n.ne.j+h .and. n.ne.nn+1 ) print *,'n wrong: ',n
   90	    continue
	    o   = j12
	    j12 = j21
	    j21 = o
	    if ( h.ge.nn ) goto 900
  100	continue
  900	continue
	do 901 i=1,nn
	    ii(i) = jj(i,j12)
  901	continue
*  #] work:
*  #[ debug output:
*	if ( lwrite ) then
*	    print *,'This should be sorted:'
*	    do 910 i=1,nn
*		print '(i5,f20.8)',ii(i),a(ii(i))
*  910	    continue
*	endif
*  #] debug output:
*###] ffsort:
	end
*###[ ff5ind:
	subroutine ff5ind(ip,ii,ngiven,ier)
***#[*comment:***********************************************************
*									*
*	Find a set of 5 independent external momenta (disregarding the	*
*	fact that we live in 4-dim space), preferring low indices in ii	*
*	the first ngiven are already given in ip.			*
*									*
*	Input:	ii	integer(15)	some ordered set of 7-21	*
*		ngiven	integer		the first ngiven ip(i) are input*
*	Output:	ip	integer(5)	p(ip(i)) are independent momenta*
*									*
***#]*comment:***********************************************************
*  #[ declarations:
	implicit none
*
*	arguments
*
	integer ii(15),ip(5),ngiven,ier
*
*	local variables
*
	integer i,j,k,oldk,t,in,third(7:21,7:21),idep(7:21),depi(15),i1
	save third
*
*	common blocks
*
	include 'ff.h'
*
*	data
*
*	the array which gives the third vector which forms a dependent
*	set of 3
*		 7  8  9 10 11 12  13 14 15 16 17 18  19 20 21
	data third/
     +		 0,13, 0, 0, 0,18,  8,19, 0, 0,20,12, 14,17, 0,
     +		13, 0,14, 0, 0, 0,  7, 9,20, 0, 0,21,  0,15,18,
     +		 0,14, 0,15, 0, 0, 19, 8,10,21, 0, 0, 13, 0,16,
     +		 0, 0,15, 0,16, 0,  0,20, 9,11,19, 0, 17,14, 0,
     +		 0, 0, 0,16, 0,17,  0, 0,21,10,12,20,  0,18,15,
     +		18, 0, 0, 0,17, 0, 21, 0, 0,19,11, 7, 16, 0,13,

     +		 8, 7,19, 0, 0,21,  0, 0,17, 0,15, 0,  9, 0,12,
     +		19, 9, 8,20, 0, 0,  0, 0, 0,18, 0,16,  7,10, 0,
     +		 0,20,10, 9,21, 0, 17, 0, 0, 0,13, 0,  0, 8,11,
     +		 0, 0,21,11,10,19,  0,18, 0, 0, 0,14, 12, 0, 9,
     +		20, 0, 0,19,12,11, 15, 0,13, 0, 0, 0, 10, 7, 0,
     +		12,21, 0, 0,20, 7,  0,16, 0,14, 0, 0,  0,11, 8,

     +		14, 0,13,17, 0,16,  9, 7, 0,12,10, 0,  0, 0, 0,
     +		17,15, 0,14,18, 0,  0,10, 8, 0, 7,11,  0, 0, 0,
     +		 0,18,16, 0,15,13, 12, 0,11, 9, 0, 8,  0, 0, 0/
*
*  #] declarations:
*  #[ work:
	if ( lwrite ) then
	    print *,'ff5ind: input:  ',ii
	    print *,'        ngiven: ',ngiven,': ',(ip(i),i=1,ngiven)
	endif
*
	do 15 i=7,21
	    idep(i) = 0
   15	continue
*
	in = 1
	k = 0
	i = 1
	do 100 i1=1,1024
*
*	    dependent?
*
	    if ( in.gt.1 ) then
		if ( in.le.ngiven ) then
		    if ( idep(ip(in)) .ne. 0 ) then
			print *,'ff5ind: error: given vectors already ',
     +				'dependent ',(ip(j),j=1,ngiven)
			goto 101
		    endif
		else
		    if ( idep(ii(i)) .ne. 0 ) then
**			if ( lwrite ) print *,'Rejected: ',ii(i)
			i = i+1
			if ( i.gt. 15 ) goto 101
			goto 100
		    endif
		endif
	    endif
*
*	    Found one!
*
	    if ( in.gt.ngiven ) then
		ip(in) = ii(i)
		i = i+1
	    endif
**	    if ( lwrite ) print *,'Found: ',ip(in)
	    if ( in.eq.5 ) goto 120
*
*	    paint this one and all other dependent vectors black
*	    (recursively)
*
	    idep(ip(in)) = 1
	    k = k+1
	    depi(k) = ip(in)
	    in = in+1
	    oldk = k
   80	    continue
	    do 90 j=1,oldk-1
		t = third(depi(j),depi(oldk))
		if ( t.ne.0 ) then
		    if ( idep(t).eq.0 ) then
**			if ( lwrite ) print *,'Vectors ',depi(j),
**     +				depi(oldk),' give ',t
			idep(t) = 1
			k = k+1
			depi(k) = t
		    endif
		endif
   90	    continue
	    if ( k.gt.oldk ) then
		oldk = oldk+1
		goto 80
	    endif
  100	continue
  101	continue
	call fferr(69,ier)
	do 110 i=1,5
	    ip(i) = i+6
  110	continue
  120	continue
	if ( lwrite ) then
	    print *,'ff5ind: found lin. independent combination ',ip
	endif
*  #] work:
*###] ff5ind:
	end
