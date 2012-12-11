      integer function pdgtonew(id,mconv)
C...convert (mconv=1) from PDG96 numbering scheme to 2004 numbering scheme
C...     or (mconv=2) from 2004 numbering scheme to PDG96 numbering scheme
C
C           id = particle identification number
C
#include "stdlun.inc"

      integer id,mconv

C... itabold(I) converts all defined PDG96 particle ID's to a standard scheme
      parameter (mxlist = 257)
      integer itabold(mxlist), itabnew(mxlist), noant(6,2)

      data itabold(1)  /      1/	! d
      data itabold(2)  /      2/	! u
      data itabold(3)  /      3/	! s
      data itabold(4)  /      4/	! c
      data itabold(5)  /      5/	! b
      data itabold(6)  /      6/	! t
      data itabold(7)  /      9/	! g
      data itabold(8)  /     22/	! gamma               0
      data itabold(9)  /     24/	! W                   +
      data itabold(10) /     23/	! Z                   0
      data itabold(11) /     21/	! g
      data itabold(12) /     25/	! h0
      data itabold(13) /     35/	! H0
      data itabold(14) /     36/	! A0
      data itabold(15) /     37/	! H+
      data itabold(16) /     12/	! nu(e)               0
      data itabold(17) /     14/	! nu(mu)              0
      data itabold(18) /     16/	! nu(tau)             0
      data itabold(19) /     11/	! e                   -
      data itabold(20) /     13/	! mu                  -
      data itabold(21) /     15/	! tau                 -
      data itabold(22) /   1103/	!
      data itabold(23) /   2101/	!
      data itabold(24) /   2103/	!
      data itabold(25) /   2203/	!
      data itabold(26) /   3101/	!
      data itabold(27) /   3103/	!
      data itabold(28) /   3201/	!
      data itabold(29) /   3203/	!
      data itabold(30) /    211/	! pi                  +
      data itabold(31) /    111/	! pi                  0
      data itabold(32) /    221/	! eta                 0
      data itabold(33) /  60221/	! f(0)(400-1200)      0
      data itabold(34) /    113/	! rho(770)          0,+
      data itabold(35) /    213/	! rho(770)          0,+
      data itabold(36) /    223/	! omega(782)          0
      data itabold(37) /    331/	! eta'(958)           0
      data itabold(38) /  10221/	! f(0)(980)           0
      data itabold(39) /  10111/	! a(0)(980)         0,+
      data itabold(40) /  10211/	! a(0)(980)         0,+
      data itabold(41) /    333/	! phi(1020)           0
      data itabold(42) /  10223/	! h(1)(1170)          0
      data itabold(43) /  10113/	! b(1)(1235)        0,+
      data itabold(44) /  10213/	! b(1)(1235)        0,+
      data itabold(45) /  20113/	! a(1)(1260)        0,+
      data itabold(46) /  20213/	! a(1)(1260)        0,+
      data itabold(47) /    225/	! f(2)(1270)          0
      data itabold(48) /  20223/	! f(1)(1285)          0
      data itabold(49) /  20221/	! eta(1295)           0
      data itabold(50) /  20111/	! pi(1300)          0,+
      data itabold(51) /  20211/	! pi(1300)          0,+
      data itabold(52) /    115/	! a(2)(1320)        0,+
      data itabold(53) /    215/	! a(2)(1320)        0,+
      data itabold(54) /  30221/	! f(0)(1370)          0
      data itabold(55) /  30223/	! f(1)(1420)          0
      data itabold(56) /  50223/	! omega(1420)         0
      data itabold(57) /  40221/	! eta(1440)           0
      data itabold(58) /  40113/	! rho(1450)         0,+
      data itabold(59) /  40213/	! rho(1450)         0,+
      data itabold(60) /  50221/	! f(0)(1590)          0
      data itabold(61) /  40223/	! f(1)(1510)          0
      data itabold(62) /    335/	! f(2)'(1525)         0
      data itabold(63) /  60223/	! omega(1600)         0
      data itabold(64) /    227/	! omega(3)(1670)      0
      data itabold(65) /  10115/	! pi(2)(1670)       0,+
      data itabold(66) /  10215/	! pi(2)(1670)       0,+
      data itabold(67) /  10333/	! phi(1680)           0
      data itabold(68) /    117/	! rho(3)(1690)      0,+
      data itabold(69) /    217/	! rho(3)(1690)      0,+
      data itabold(70) /  30113/	! rho(1700)         0,+
      data itabold(71) /  30213/	! rho(1700)         0,+
      data itabold(72) /    337/	! phi(3)(1850)        0
      data itabold(73) /  20225/	! f(2)(2010)          0
      data itabold(74) /    229/	! f(4)(2050)          0
      data itabold(75) /  30225/	! f(2)(2300)          0
      data itabold(76) /  40225/	! f(2)(2340)          0
      data itabold(77) /    321/	! K                   +
      data itabold(78) /    311/	! K                   0
      data itabold(79) /    310/	! K(S)                0
      data itabold(80) /    130/	! K(L)                0
      data itabold(81) /    313/	! K*(892)             0
      data itabold(82) /    323/	! K*(892)             +
      data itabold(83) /  10313/	! K(1)(1270)        0,+
      data itabold(84) /  10323/	! K(1)(1270)        0,+
      data itabold(85) /  20313/	! K(1)(1400)        0,+
      data itabold(86) /  20323/	! K(1)(1400)        0,+
      data itabold(87) /  30313/	! K*(1410)          0,+
      data itabold(88) /  30323/	! K*(1410)          0,+
      data itabold(89) /  10311/	! K(0)*(1430)       0,+
      data itabold(90) /  10321/	! K(0)*(1430)       0,+
      data itabold(91) /    315/	! K(2)*(1430)         0
      data itabold(92) /    325/	! K(2)*(1430)         +
      data itabold(93) /  40313/	! K*(1680)          0,+
      data itabold(94) /  40323/	! K*(1680)          0,+
      data itabold(95) /  10315/	! K(2)(1770)        0,+
      data itabold(96) /  10325/	! K(2)(1770)        0,+
      data itabold(97) /    317/	! K(3)*(1780)       0,+
      data itabold(98) /    327/	! K(3)*(1780)       0,+
      data itabold(99) /  20315/	! K(2)(1820)        0,+
      data itabold(100)/  20325/	! K(2)(1820)        0,+
      data itabold(101)/    319/	! K(4)*(2045)       0,+
      data itabold(102)/    329/	! K(4)*(2045)       0,+
      data itabold(103)/    411/	! D                   +
      data itabold(104)/    421/	! D                   0
      data itabold(105)/    423/	! D*(2007)            0
      data itabold(106)/    413/	! D*(2010)            +
      data itabold(107)/  10423/	! D(1)(2420)          0
      data itabold(108)/    425/	! D(2)*(2460)         0
      data itabold(109)/    415/	! D(2)*(2460)         +
      data itabold(110)/    431/	! D(s)                +
      data itabold(111)/    433/	! D(s)*               +
      data itabold(112)/  10433/	! D(s1)(2536)         +
      data itabold(113)/    521/	! B                   +
      data itabold(114)/    511/	! B                   0
      data itabold(115)/    513/	! B*                0,+
      data itabold(116)/    523/	! B*                0,+
      data itabold(117)/    531/	! B(s)                0
      data itabold(118)/    441/	! eta(c)(1S)          0
      data itabold(119)/    443/	! J/psi(1S)           0
      data itabold(120)/  10441/	! chi(c0)(1P)         0
      data itabold(121)/  10443/	! chi(c1)(1P)         0
      data itabold(122)/    445/	! chi(c2)(1P)         0
      data itabold(123)/  20443/	! psi(2S)             0
      data itabold(124)/  30443/	! psi(3770)           0
      data itabold(125)/  40443/	! psi(4040)           0
      data itabold(126)/  50443/	! psi(4160)           0
      data itabold(127)/  60443/	! psi(4415)           0
      data itabold(128)/    553/	! Upsilon(1S)         0
      data itabold(129)/    551/	! chi(b0)(1P)         0
      data itabold(130)/  10553/	! chi(b1)(1P)         0
      data itabold(131)/    555/	! chi(b2)(1P)         0
      data itabold(132)/  20553/	! Upsilon(2S)         0
      data itabold(133)/  10551/	! chi(b0)(2P)         0
      data itabold(134)/  70553/	! chi(b1)(2P)         0
      data itabold(135)/  10555/	! chi(b2)(2P)         0
      data itabold(136)/  30553/	! Upsilon(3S)         0
      data itabold(137)/  40553/	! Upsilon(4S)         0
      data itabold(138)/  50553/	! Upsilon(10860)      0
      data itabold(139)/  60553/	! Upsilon(11020)      0
      data itabold(140)/   2212/	! p                   +
      data itabold(141)/   2112/	! n                   0
      data itabold(142)/  12112/	! N(1440)           0,+
      data itabold(143)/  12212/	! N(1440)           0,+
      data itabold(144)/   1214/	! N(1520)           0,+
      data itabold(145)/   2124/	! N(1520)           0,+
      data itabold(146)/  22112/	! N(1535)           0,+
      data itabold(147)/  22212/	! N(1535)           0,+
      data itabold(148)/  32112/	! N(1650)           0,+
      data itabold(149)/  32212/	! N(1650)           0,+
      data itabold(150)/   2116/	! N(1675)           0,+
      data itabold(151)/   2216/	! N(1675)           0,+
      data itabold(152)/  12116/	! N(1680)           0,+
      data itabold(153)/  12216/	! N(1680)           0,+
      data itabold(154)/  21214/	! N(1700)           0,+
      data itabold(155)/  22124/	! N(1700)           0,+
      data itabold(156)/  42112/	! N(1710)           0,+
      data itabold(157)/  42212/	! N(1710)           0,+
      data itabold(158)/  31214/	! N(1720)           0,+
      data itabold(159)/  32124/	! N(1720)           0,+
      data itabold(160)/   1218/	! N(2190)           0,+
      data itabold(161)/   2128/	! N(2190)           0,+
      data itabold(162)/   1114/	! Delta(1232)  -,0,+,++
      data itabold(163)/   2114/	! Delta(1232)  -,0,+,++
      data itabold(164)/   2214/	! Delta(1232)  -,0,+,++
      data itabold(165)/   2224/	! Delta(1232)  -,0,+,++
      data itabold(166)/  31114/	! Delta(1600)  -,0,+,++
      data itabold(167)/  32114/	! Delta(1600)  -,0,+,++
      data itabold(168)/  32214/	! Delta(1600)  -,0,+,++
      data itabold(169)/  32224/	! Delta(1600)  -,0,+,++
      data itabold(170)/   1112/	! Delta(1620)  -,0,+,++
      data itabold(171)/   1212/	! Delta(1620)  -,0,+,++
      data itabold(172)/   2122/	! Delta(1620)  -,0,+,++
      data itabold(173)/   2222/	! Delta(1620)  -,0,+,++
      data itabold(174)/  11114/	! Delta(1700)  -,0,+,++
      data itabold(175)/  12114/	! Delta(1700)  -,0,+,++
      data itabold(176)/  12214/	! Delta(1700)  -,0,+,++
      data itabold(177)/  12224/	! Delta(1700)  -,0,+,++
      data itabold(178)/  11112/	! Delta(1900)  -,0,+,++
      data itabold(179)/  11212/	! Delta(1900)  -,0,+,++
      data itabold(180)/  12122/	! Delta(1900)  -,0,+,++
      data itabold(181)/  12222/	! Delta(1900)  -,0,+,++
      data itabold(182)/   1116/	! Delta(1905)  -,0,+,++
      data itabold(183)/   1216/	! Delta(1905)  -,0,+,++
      data itabold(184)/   2126/	! Delta(1905)  -,0,+,++
      data itabold(185)/   2226/	! Delta(1905)  -,0,+,++
      data itabold(186)/  21112/	! Delta(1910)  -,0,+,++
      data itabold(187)/  21212/	! Delta(1910)  -,0,+,++
      data itabold(188)/  22122/	! Delta(1910)  -,0,+,++
      data itabold(189)/  22222/	! Delta(1910)  -,0,+,++
      data itabold(190)/  21114/	! Delta(1920)  -,0,+,++
      data itabold(191)/  22114/	! Delta(1920)  -,0,+,++
      data itabold(192)/  22214/	! Delta(1920)  -,0,+,++
      data itabold(193)/  22224/	! Delta(1920)  -,0,+,++
      data itabold(194)/  11116/	! Delta(1930)  -,0,+,++
      data itabold(195)/  11216/	! Delta(1930)  -,0,+,++
      data itabold(196)/  12126/	! Delta(1930)  -,0,+,++
      data itabold(197)/  12226/	! Delta(1930)  -,0,+,++
      data itabold(198)/   1118/	! Delta(1950)  -,0,+,++
      data itabold(199)/   2118/	! Delta(1950)  -,0,+,++
      data itabold(200)/   2218/	! Delta(1950)  -,0,+,++
      data itabold(201)/   2228/	! Delta(1950)  -,0,+,++
      data itabold(202)/   3122/	! Lambda              0
      data itabold(203)/  13122/	! Lambda(1405)        0
      data itabold(204)/   3124/	! Lambda(1520)        0
      data itabold(205)/  23122/	! Lambda(1600)        0
      data itabold(206)/  33122/	! Lambda(1670)        0
      data itabold(207)/  13124/	! Lambda(1690)        0
      data itabold(208)/  43122/	! Lambda(1800)        0
      data itabold(209)/  53122/	! Lambda(1810)        0
      data itabold(210)/   3126/	! Lambda(1820)        0
      data itabold(211)/  13126/	! Lambda(1830)        0
      data itabold(212)/  23124/	! Lambda(1890)        0
      data itabold(213)/   3128/	! Lambda(2100)        0
      data itabold(214)/  23126/	! Lambda(2110)        0
      data itabold(215)/   3222/	! Sigma               +
      data itabold(216)/   3212/	! Sigma               0
      data itabold(217)/   3112/	! Sigma               -
      data itabold(218)/   3114/	! Sigma(1385)         -
      data itabold(219)/   3214/	! Sigma(1385)         0
      data itabold(220)/   3224/	! Sigma(1385)         +
      data itabold(221)/  13112/	! Sigma(1660)     -,0,+
      data itabold(222)/  13212/	! Sigma(1660)     -,0,+
      data itabold(223)/  13222/	! Sigma(1660)     -,0,+
      data itabold(224)/  13114/	! Sigma(1670)     -,0,+
      data itabold(225)/  13214/	! Sigma(1670)     -,0,+
      data itabold(226)/  13224/	! Sigma(1670)     -,0,+
      data itabold(227)/  23112/	! Sigma(1750)     -,0,+
      data itabold(228)/  23212/	! Sigma(1750)     -,0,+
      data itabold(229)/  23222/	! Sigma(1750)     -,0,+
      data itabold(230)/   3116/	! Sigma(1775)     -,0,+
      data itabold(231)/   3216/	! Sigma(1775)     -,0,+
      data itabold(232)/   3226/	! Sigma(1775)     -,0,+
      data itabold(233)/  13116/	! Sigma(1915)     -,0,+
      data itabold(234)/  13216/	! Sigma(1915)     -,0,+
      data itabold(235)/  13226/	! Sigma(1915)     -,0,+
      data itabold(236)/  23114/	! Sigma(1940)     -,0,+
      data itabold(237)/  23214/	! Sigma(1940)     -,0,+
      data itabold(238)/  23224/	! Sigma(1940)     -,0,+
      data itabold(239)/   3118/	! Sigma(2030)     -,0,+
      data itabold(240)/   3218/	! Sigma(2030)     -,0,+
      data itabold(241)/   3228/	! Sigma(2030)     -,0,+
      data itabold(242)/   3322/	! Xi                  0
      data itabold(243)/   3312/	! Xi                  -
      data itabold(244)/   3314/	! Xi(1530)            -
      data itabold(245)/   3324/	! Xi(1530)            0
      data itabold(246)/  13314/	! Xi(1820)          -,0
      data itabold(247)/  13324/	! Xi(1820)          -,0
      data itabold(248)/   3334/	! Omega               -
      data itabold(249)/   4122/	! Lambda(c)           +
      data itabold(250)/  14122/	! Lambda(c)(2593)     +
      data itabold(251)/   4112/	! Sigma(c)(2455)      0
      data itabold(252)/   4212/	! Sigma(c)(2455)      +
      data itabold(253)/   4222/	! Sigma(c)(2455)     ++
      data itabold(254)/   4322/	! Xi(c)               +
      data itabold(255)/   4312/	! Xi(c)               0
      data itabold(256)/   4332/	! Omega(c)            0
      data itabold(257)/   5122/	! Lambda(b)           0

      data itabnew(1)  /      1/	! d
      data itabnew(2)  /      2/	! u
      data itabnew(3)  /      3/	! s
      data itabnew(4)  /      4/	! c
      data itabnew(5)  /      5/	! b
      data itabnew(6)  /      6/	! t
      data itabnew(7)  /      9/	! g
      data itabnew(8)  /     22/	! gamma               0
      data itabnew(9)  /     24/	! W                   +
      data itabnew(10) /     23/	! Z                   0
      data itabnew(11) /     21/	! g
      data itabnew(12) /     25/	! h0
      data itabnew(13) /     35/	! H0
      data itabnew(14) /     36/	! A0
      data itabnew(15) /     37/	! H+
      data itabnew(16) /     12/	! nu(e)               0
      data itabnew(17) /     14/	! nu(mu)              0
      data itabnew(18) /     16/	! nu(tau)             0
      data itabnew(19) /     11/	! e                   -
      data itabnew(20) /     13/	! mu                  -
      data itabnew(21) /     15/	! tau                 -
      data itabnew(22) /   1103/	!
      data itabnew(23) /   2101/	!
      data itabnew(24) /   2103/	!
      data itabnew(25) /   2203/	!
      data itabnew(26) /   3101/	!
      data itabnew(27) /   3103/	!
      data itabnew(28) /   3201/	!
      data itabnew(29) /   3203/	!
      data itabnew(30) /    211/	! pi                  +
      data itabnew(31) /    111/	! pi                  0
      data itabnew(32) /    221/	! eta                 0
      data itabnew(33) /9000221/	! f(0)(400-1200)      0
      data itabnew(34) /    113/	! rho(770)          0,+
      data itabnew(35) /    213/	! rho(770)          0,+
      data itabnew(36) /    223/	! omega(782)          0
      data itabnew(37) /    331/	! eta'(958)           0
      data itabnew(38) /9010221/	! f(0)(980)           0
      data itabnew(39) /9000111/	! a(0)(980)         0,+
      data itabnew(40) /9000211/	! a(0)(980)         0,+
      data itabnew(41) /    333/	! phi(1020)           0
      data itabnew(42) /  10223/	! h(1)(1170)          0
      data itabnew(43) /  10113/	! b(1)(1235)        0,+
      data itabnew(44) /  10213/	! b(1)(1235)        0,+
      data itabnew(45) /  20113/	! a(1)(1260)        0,+
      data itabnew(46) /  20213/	! a(1)(1260)        0,+
      data itabnew(47) /    225/	! f(2)(1270)          0
      data itabnew(48) /  20223/	! f(1)(1285)          0
      data itabnew(49) / 100221/	! eta(1295)           0
      data itabnew(50) / 100111/	! pi(1300)          0,+
      data itabnew(51) / 100211/	! pi(1300)          0,+
      data itabnew(52) /    115/	! a(2)(1320)        0,+
      data itabnew(53) /    215/	! a(2)(1320)        0,+
      data itabnew(54) /  10221/	! f(0)(1370)          0
      data itabnew(55) /  20333/	! f(1)(1420)          0
      data itabnew(56) / 100223/	! omega(1420)         0
      data itabnew(57) / 100331/	! eta(1440)           0
      data itabnew(58) / 100113/	! rho(1450)         0,+
      data itabnew(59) / 100213/	! rho(1450)         0,+
      data itabnew(60) /9030221/	! f(0)(1590)          0
      data itabnew(61) /9000223/	! f(1)(1510)          0
      data itabnew(62) /    335/	! f(2)'(1525)         0
      data itabnew(63) /  30223/	! omega(1600)         0
      data itabnew(64) /    227/	! omega(3)(1670)      0
      data itabnew(65) /  10115/	! pi(2)(1670)       0,+
      data itabnew(66) /  10215/	! pi(2)(1670)       0,+
      data itabnew(67) / 100333/	! phi(1680)           0
      data itabnew(68) /    117/	! rho(3)(1690)      0,+
      data itabnew(69) /    217/	! rho(3)(1690)      0,+
      data itabnew(70) /  30113/	! rho(1700)         0,+
      data itabnew(71) /  30213/	! rho(1700)         0,+
      data itabnew(72) /    337/	! phi(3)(1850)        0
      data itabnew(73) /9060225/	! f(2)(2010)          0
      data itabnew(74) /    229/	! f(4)(2050)          0
      data itabnew(75) /9080225/	! f(2)(2300)          0
      data itabnew(76) /9090225/	! f(2)(2340)          0
      data itabnew(77) /    321/	! K                   +
      data itabnew(78) /    311/	! K                   0
      data itabnew(79) /    310/	! K(S)                0
      data itabnew(80) /    130/	! K(L)                0
      data itabnew(81) /    313/	! K*(892)             0
      data itabnew(82) /    323/	! K*(892)             +
      data itabnew(83) /  10313/	! K(1)(1270)        0,+
      data itabnew(84) /  10323/	! K(1)(1270)        0,+
      data itabnew(85) /  20313/	! K(1)(1400)        0,+
      data itabnew(86) /  20323/	! K(1)(1400)        0,+
      data itabnew(87) / 100313/	! K*(1410)          0,+
      data itabnew(88) / 100323/	! K*(1410)          0,+
      data itabnew(89) /  10311/	! K(0)*(1430)       0,+
      data itabnew(90) /  10321/	! K(0)*(1430)       0,+
      data itabnew(91) /    315/	! K(2)*(1430)         0
      data itabnew(92) /    325/	! K(2)*(1430)         +
      data itabnew(93) /  30313/	! K*(1680)          0,+
      data itabnew(94) /  30323/	! K*(1680)          0,+
      data itabnew(95) /  10315/	! K(2)(1770)        0,+
      data itabnew(96) /  10325/	! K(2)(1770)        0,+
      data itabnew(97) /    317/	! K(3)*(1780)       0,+
      data itabnew(98) /    327/	! K(3)*(1780)       0,+
      data itabnew(99) /  20315/	! K(2)(1820)        0,+
      data itabnew(100)/  20325/	! K(2)(1820)        0,+
      data itabnew(101)/    319/	! K(4)*(2045)       0,+
      data itabnew(102)/    329/	! K(4)*(2045)       0,+
      data itabnew(103)/    411/	! D                   +
      data itabnew(104)/    421/	! D                   0
      data itabnew(105)/    423/	! D*(2007)            0
      data itabnew(106)/    413/	! D*(2010)            +
      data itabnew(107)/  10423/	! D(1)(2420)          0
      data itabnew(108)/    425/	! D(2)*(2460)         0
      data itabnew(109)/    415/	! D(2)*(2460)         +
      data itabnew(110)/    431/	! D(s)                +
      data itabnew(111)/    433/	! D(s)*               +
      data itabnew(112)/  10433/	! D(s1)(2536)         +
      data itabnew(113)/    521/	! B                   +
      data itabnew(114)/    511/	! B                   0
      data itabnew(115)/    513/	! B*                0,+
      data itabnew(116)/    523/	! B*                0,+
      data itabnew(117)/    531/	! B(s)                0
      data itabnew(118)/    441/	! eta(c)(1S)          0
      data itabnew(119)/    443/	! J/psi(1S)           0
      data itabnew(120)/  10441/	! chi(c0)(1P)         0
      data itabnew(121)/  20443/	! chi(c1)(1P)         0
      data itabnew(122)/    445/	! chi(c2)(1P)         0
      data itabnew(123)/ 100443/	! psi(2S)             0
      data itabnew(124)/  30443/	! psi(3770)           0
      data itabnew(125)/9000443/	! psi(4040)           0
      data itabnew(126)/9010443/	! psi(4160)           0
      data itabnew(127)/9020443/	! psi(4415)           0
      data itabnew(128)/    553/	! Upsilon(1S)         0
      data itabnew(129)/  10551/	! chi(b0)(1P)         0
      data itabnew(130)/  20553/	! chi(b1)(1P)         0
      data itabnew(131)/    555/	! chi(b2)(1P)         0
      data itabnew(132)/ 100553/	! Upsilon(2S)         0
      data itabnew(133)/ 110551/	! chi(b0)(2P)         0
      data itabnew(134)/ 120553/	! chi(b1)(2P)         0
      data itabnew(135)/ 100555/	! chi(b2)(2P)         0
      data itabnew(136)/ 200553/	! Upsilon(3S)         0
      data itabnew(137)/ 300553/	! Upsilon(4S)         0
      data itabnew(138)/9000553/	! Upsilon(10860)      0
      data itabnew(139)/9010553/	! Upsilon(11020)      0
      data itabnew(140)/   2212/	! p                   +
      data itabnew(141)/   2112/	! n                   0
      data itabnew(142)/  12112/	! N(1440)           0,+
      data itabnew(143)/  12212/	! N(1440)           0,+
      data itabnew(144)/   1214/	! N(1520)           0,+
      data itabnew(145)/   2124/	! N(1520)           0,+
      data itabnew(146)/  22112/	! N(1535)           0,+
      data itabnew(147)/  22212/	! N(1535)           0,+
      data itabnew(148)/  32112/	! N(1650)           0,+
      data itabnew(149)/  32212/	! N(1650)           0,+
      data itabnew(150)/   2116/	! N(1675)           0,+
      data itabnew(151)/   2216/	! N(1675)           0,+
      data itabnew(152)/  12116/	! N(1680)           0,+
      data itabnew(153)/  12216/	! N(1680)           0,+
      data itabnew(154)/  21214/	! N(1700)           0,+
      data itabnew(155)/  22124/	! N(1700)           0,+
      data itabnew(156)/  42112/	! N(1710)           0,+
      data itabnew(157)/  42212/	! N(1710)           0,+
      data itabnew(158)/  31214/	! N(1720)           0,+
      data itabnew(159)/  32124/	! N(1720)           0,+
      data itabnew(160)/   1218/	! N(2190)           0,+
      data itabnew(161)/   2128/	! N(2190)           0,+
      data itabnew(162)/   1114/	! Delta(1232)  -,0,+,++
      data itabnew(163)/   2114/	! Delta(1232)  -,0,+,++
      data itabnew(164)/   2214/	! Delta(1232)  -,0,+,++
      data itabnew(165)/   2224/	! Delta(1232)  -,0,+,++
      data itabnew(166)/  31114/	! Delta(1600)  -,0,+,++
      data itabnew(167)/  32114/	! Delta(1600)  -,0,+,++
      data itabnew(168)/  32214/	! Delta(1600)  -,0,+,++
      data itabnew(169)/  32224/	! Delta(1600)  -,0,+,++
      data itabnew(170)/   1112/	! Delta(1620)  -,0,+,++
      data itabnew(171)/   1212/	! Delta(1620)  -,0,+,++
      data itabnew(172)/   2122/	! Delta(1620)  -,0,+,++
      data itabnew(173)/   2222/	! Delta(1620)  -,0,+,++
      data itabnew(174)/  11114/	! Delta(1700)  -,0,+,++
      data itabnew(175)/  12114/	! Delta(1700)  -,0,+,++
      data itabnew(176)/  12214/	! Delta(1700)  -,0,+,++
      data itabnew(177)/  12224/	! Delta(1700)  -,0,+,++
      data itabnew(178)/  11112/	! Delta(1900)  -,0,+,++
      data itabnew(179)/  11212/	! Delta(1900)  -,0,+,++
      data itabnew(180)/  12122/	! Delta(1900)  -,0,+,++
      data itabnew(181)/  12222/	! Delta(1900)  -,0,+,++
      data itabnew(182)/   1116/	! Delta(1905)  -,0,+,++
      data itabnew(183)/   1216/	! Delta(1905)  -,0,+,++
      data itabnew(184)/   2126/	! Delta(1905)  -,0,+,++
      data itabnew(185)/   2226/	! Delta(1905)  -,0,+,++
      data itabnew(186)/  21112/	! Delta(1910)  -,0,+,++
      data itabnew(187)/  21212/	! Delta(1910)  -,0,+,++
      data itabnew(188)/  22122/	! Delta(1910)  -,0,+,++
      data itabnew(189)/  22222/	! Delta(1910)  -,0,+,++
      data itabnew(190)/  21114/	! Delta(1920)  -,0,+,++
      data itabnew(191)/  22114/	! Delta(1920)  -,0,+,++
      data itabnew(192)/  22214/	! Delta(1920)  -,0,+,++
      data itabnew(193)/  22224/	! Delta(1920)  -,0,+,++
      data itabnew(194)/  11116/	! Delta(1930)  -,0,+,++
      data itabnew(195)/  11216/	! Delta(1930)  -,0,+,++
      data itabnew(196)/  12126/	! Delta(1930)  -,0,+,++
      data itabnew(197)/  12226/	! Delta(1930)  -,0,+,++
      data itabnew(198)/   1118/	! Delta(1950)  -,0,+,++
      data itabnew(199)/   2118/	! Delta(1950)  -,0,+,++
      data itabnew(200)/   2218/	! Delta(1950)  -,0,+,++
      data itabnew(201)/   2228/	! Delta(1950)  -,0,+,++
      data itabnew(202)/   3122/	! Lambda              0
      data itabnew(203)/  13122/	! Lambda(1405)        0
      data itabnew(204)/   3124/	! Lambda(1520)        0
      data itabnew(205)/  23122/	! Lambda(1600)        0
      data itabnew(206)/  33122/	! Lambda(1670)        0
      data itabnew(207)/  13124/	! Lambda(1690)        0
      data itabnew(208)/  43122/	! Lambda(1800)        0
      data itabnew(209)/  53122/	! Lambda(1810)        0
      data itabnew(210)/   3126/	! Lambda(1820)        0
      data itabnew(211)/  13126/	! Lambda(1830)        0
      data itabnew(212)/  23124/	! Lambda(1890)        0
      data itabnew(213)/   3128/	! Lambda(2100)        0
      data itabnew(214)/  23126/	! Lambda(2110)        0
      data itabnew(215)/   3222/	! Sigma               +
      data itabnew(216)/   3212/	! Sigma               0
      data itabnew(217)/   3112/	! Sigma               -
      data itabnew(218)/   3114/	! Sigma(1385)         -
      data itabnew(219)/   3214/	! Sigma(1385)         0
      data itabnew(220)/   3224/	! Sigma(1385)         +
      data itabnew(221)/  13112/	! Sigma(1660)     -,0,+
      data itabnew(222)/  13212/	! Sigma(1660)     -,0,+
      data itabnew(223)/  13222/	! Sigma(1660)     -,0,+
      data itabnew(224)/  13114/	! Sigma(1670)     -,0,+
      data itabnew(225)/  13214/	! Sigma(1670)     -,0,+
      data itabnew(226)/  13224/	! Sigma(1670)     -,0,+
      data itabnew(227)/  23112/	! Sigma(1750)     -,0,+
      data itabnew(228)/  23212/	! Sigma(1750)     -,0,+
      data itabnew(229)/  23222/	! Sigma(1750)     -,0,+
      data itabnew(230)/   3116/	! Sigma(1775)     -,0,+
      data itabnew(231)/   3216/	! Sigma(1775)     -,0,+
      data itabnew(232)/   3226/	! Sigma(1775)     -,0,+
      data itabnew(233)/  13116/	! Sigma(1915)     -,0,+
      data itabnew(234)/  13216/	! Sigma(1915)     -,0,+
      data itabnew(235)/  13226/	! Sigma(1915)     -,0,+
      data itabnew(236)/  23114/	! Sigma(1940)     -,0,+
      data itabnew(237)/  23214/	! Sigma(1940)     -,0,+
      data itabnew(238)/  23224/	! Sigma(1940)     -,0,+
      data itabnew(239)/   3118/	! Sigma(2030)     -,0,+
      data itabnew(240)/   3218/	! Sigma(2030)     -,0,+
      data itabnew(241)/   3228/	! Sigma(2030)     -,0,+
      data itabnew(242)/   3322/	! Xi                  0
      data itabnew(243)/   3312/	! Xi                  -
      data itabnew(244)/   3314/	! Xi(1530)            -
      data itabnew(245)/   3324/	! Xi(1530)            0
      data itabnew(246)/  13314/	! Xi(1820)          -,0
      data itabnew(247)/  13324/	! Xi(1820)          -,0
      data itabnew(248)/   3334/	! Omega               -
      data itabnew(249)/   4122/	! Lambda(c)           +
      data itabnew(250)/  14122/	! Lambda(c)(2593)     +
      data itabnew(251)/   4112/	! Sigma(c)(2455)      0
      data itabnew(252)/   4212/	! Sigma(c)(2455)      +
      data itabnew(253)/   4222/	! Sigma(c)(2455)     ++
      data itabnew(254)/   4232/	! Xi(c)               +
      data itabnew(255)/   4132/	! Xi(c)               0
      data itabnew(256)/   4332/	! Omega(c)            0
      data itabnew(257)/   5122/	! Lambda(b)           0

      data noant/-21,-22,-23,-25,-35,-36,-21,-22,-23,-25,-35,-36/
      save itabold,itabnew,noant

      pdgtonew=0
      ida=iabs(id)

      if(ida.eq.0)then
        write(lnhout,*) ' PDGTONEW 1: particle id is zero'
      elseif(ida.ge.81 .and. ida.le.99)then
        pdgtonew=id
      elseif(mconv.eq.1)then
        do i = 1,mxlist
           if(ida.eq.itabold(i))then
              pdgtonew=isign(itabnew(i),id)
           endif
        enddo
      elseif(mconv.eq.2)then
        do i = 1,mxlist
           if(ida.eq.itabnew(i))then
              pdgtonew=isign(itabold(i),id)
           endif
        enddo
      else
        write(lnhout,*) ' PDGTONEW 2: unallowed conversion option'
        go to 100
      endif
c...check for illegal anti-particles
      do j=1,6
         if(pdgtonew.eq.noant(j,mconv)) pdgtonew = 0
      enddo
C...check for illegal anti KS, KL
      if(id.eq.-130 .or. id.eq.-310) pdgtonew=0
c...print warning for 60221
      if(ida.eq.60221 .and. mconv.eq.1) write(lnhout,1003)
C...there are too many warnings - use only for debugging purposes
c      if(pdgtonew.eq.0 .and. ida.ne.0)then
c        if(mconv.eq.1) write(lnhout,1001) id
c        if(mconv.eq.2) write(lnhout,1002) id
c      endif
  100 continue
      return
 1001 format(' PDGTONEW 3: ID ',i7,' is not a defined 1996 PDG state')
 1002 format(' PDGTONEW 4: ID ',i7,' is not a defined new state')
 1003 format(' PDGTONEW: 60221 has been used for both f0(400-1200)',
     1       ' and fJ(1710)')
      end
