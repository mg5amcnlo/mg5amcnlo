      integer function hepcmp(id)

C...Purpose: to compress the standard ID codes for use in mass and decay
C...arrays; also to check whether a given code actually is defined.

C      ID     = particle ID
C      hepcmp = compressed index

      integer id
      integer cmplist(2210)

      integer hepchg
      external hepchg

      data cmplist(   1) /         1/
      data cmplist(   2) /         2/
      data cmplist(   3) /         3/
      data cmplist(   4) /         4/
      data cmplist(   5) /         5/
      data cmplist(   6) /         6/
      data cmplist(   7) /         7/
      data cmplist(   8) /         8/
      data cmplist(   9) /         9/
      data cmplist(  10) /        10/
      data cmplist(  11) /        11/
      data cmplist(  12) /        12/
      data cmplist(  13) /        13/
      data cmplist(  14) /        14/
      data cmplist(  15) /        15/
      data cmplist(  16) /        16/
      data cmplist(  17) /        17/
      data cmplist(  18) /        18/
      data cmplist(  19) /        19/
      data cmplist(  20) /        20/
      data cmplist(  21) /        21/
      data cmplist(  22) /        22/
      data cmplist(  23) /        23/
      data cmplist(  24) /        24/
      data cmplist(  25) /        25/
      data cmplist(  26) /        26/
      data cmplist(  27) /        27/
      data cmplist(  28) /        28/
      data cmplist(  29) /        29/
      data cmplist(  30) /        30/
      data cmplist(  31) /        31/
      data cmplist(  32) /        32/
      data cmplist(  33) /        33/
      data cmplist(  34) /        34/
      data cmplist(  35) /        35/
      data cmplist(  36) /        36/
      data cmplist(  37) /        37/
      data cmplist(  38) /        38/
      data cmplist(  39) /        39/
      data cmplist(  40) /        40/
      data cmplist(  41) /        41/
      data cmplist(  42) /        42/
      data cmplist(  43) /        43/
      data cmplist(  44) /        44/
      data cmplist(  45) /        45/
      data cmplist(  46) /        46/
      data cmplist(  47) /        47/
      data cmplist(  48) /        48/
      data cmplist(  49) /        49/
      data cmplist(  50) /        50/
      data cmplist(  51) /        51/
      data cmplist(  52) /        52/
      data cmplist(  53) /        53/
      data cmplist(  54) /        54/
      data cmplist(  55) /        55/
      data cmplist(  56) /        56/
      data cmplist(  57) /        57/
      data cmplist(  58) /        58/
      data cmplist(  59) /        59/
      data cmplist(  60) /        60/
      data cmplist(  61) /        61/
      data cmplist(  62) /        62/
      data cmplist(  63) /        63/
      data cmplist(  64) /        64/
      data cmplist(  65) /        65/
      data cmplist(  66) /        66/
      data cmplist(  67) /        67/
      data cmplist(  68) /        68/
      data cmplist(  69) /        69/
      data cmplist(  70) /        70/
      data cmplist(  71) /        71/
      data cmplist(  72) /        72/
      data cmplist(  73) /        73/
      data cmplist(  74) /        74/
      data cmplist(  75) /        75/
      data cmplist(  76) /        76/
      data cmplist(  77) /        77/
      data cmplist(  78) /        78/
      data cmplist(  79) /        79/
      data cmplist(  80) /        80/
      data cmplist(  81) /        81/
      data cmplist(  82) /        82/
      data cmplist(  83) /        83/
      data cmplist(  84) /        84/
      data cmplist(  85) /        85/
      data cmplist(  86) /        86/
      data cmplist(  87) /        87/
      data cmplist(  88) /        88/
      data cmplist(  89) /        89/
      data cmplist(  90) /        90/
      data cmplist(  91) /        91/
      data cmplist(  92) /        92/
      data cmplist(  93) /        93/
      data cmplist(  94) /        94/
      data cmplist(  95) /        95/
      data cmplist(  96) /        96/
      data cmplist(  97) /        97/
      data cmplist(  98) /        98/
      data cmplist(  99) /        99/
      data cmplist( 100) /       100/
      data cmplist( 101) /       310/
      data cmplist( 102) /       130/
      data cmplist( 103) /         0/
      data cmplist( 104) /         0/
      data cmplist( 105) /         0/
      data cmplist( 106) /         0/
      data cmplist( 107) /       110/
      data cmplist( 108) /       990/
      data cmplist( 109) /      9990/
      data cmplist( 110) /         0/
      data cmplist( 111) /   1000001/
      data cmplist( 112) /   1000002/
      data cmplist( 113) /   1000003/
      data cmplist( 114) /   1000004/
      data cmplist( 115) /   1000005/
      data cmplist( 116) /   1000006/
      data cmplist( 117) /   1000007/
      data cmplist( 118) /   1000008/
      data cmplist( 119) /   1000009/
      data cmplist( 120) /         0/
      data cmplist( 121) /   1000011/
      data cmplist( 122) /   1000012/
      data cmplist( 123) /   1000013/
      data cmplist( 124) /   1000014/
      data cmplist( 125) /   1000015/
      data cmplist( 126) /   1000016/
      data cmplist( 127) /   1000017/
      data cmplist( 128) /   1000018/
      data cmplist( 129) /   1000019/
      data cmplist( 130) /         0/
      data cmplist( 131) /   1000021/
      data cmplist( 132) /   1000022/
      data cmplist( 133) /   1000023/
      data cmplist( 134) /   1000024/
      data cmplist( 135) /   1000025/
      data cmplist( 136) /   1000026/
      data cmplist( 137) /   1000027/
      data cmplist( 138) /   1000028/
      data cmplist( 139) /   1000029/
      data cmplist( 140) /         0/
      data cmplist( 141) /   1000031/
      data cmplist( 142) /   1000032/
      data cmplist( 143) /   1000033/
      data cmplist( 144) /   1000034/
      data cmplist( 145) /   1000035/
      data cmplist( 146) /   1000036/
      data cmplist( 147) /   1000037/
      data cmplist( 148) /   1000038/
      data cmplist( 149) /   1000039/
      data cmplist( 150) /         0/
      data cmplist( 151) /   1000041/
      data cmplist( 152) /   1000042/
      data cmplist( 153) /   1000043/
      data cmplist( 154) /   1000044/
      data cmplist( 155) /   1000045/
      data cmplist( 156) /   1000046/
      data cmplist( 157) /   1000047/
      data cmplist( 158) /   1000048/
      data cmplist( 159) /   1000049/
      data cmplist( 160) /         0/
      data cmplist( 161) /   1000051/
      data cmplist( 162) /   1000052/
      data cmplist( 163) /   1000053/
      data cmplist( 164) /   1000054/
      data cmplist( 165) /   1000055/
      data cmplist( 166) /   1000056/
      data cmplist( 167) /   1000057/
      data cmplist( 168) /   1000058/
      data cmplist( 169) /   1000059/
      data cmplist( 170) /         0/
      data cmplist( 171) /   2000001/
      data cmplist( 172) /   2000002/
      data cmplist( 173) /   2000003/
      data cmplist( 174) /   2000004/
      data cmplist( 175) /   2000005/
      data cmplist( 176) /   2000006/
      data cmplist( 177) /   2000007/
      data cmplist( 178) /   2000008/
      data cmplist( 179) /   2000009/
      data cmplist( 180) /         0/
      data cmplist( 181) /   2000011/
      data cmplist( 182) /   2000012/
      data cmplist( 183) /   2000013/
      data cmplist( 184) /   2000014/
      data cmplist( 185) /   2000015/
      data cmplist( 186) /   2000016/
      data cmplist( 187) /   2000017/
      data cmplist( 188) /   2000018/
      data cmplist( 189) /   2000019/
      data cmplist( 190) /         0/
      data cmplist( 191) /   3000111/
      data cmplist( 192) /   3000211/
      data cmplist( 193) /   3000221/
      data cmplist( 194) /   3000113/
      data cmplist( 195) /   3000213/
      data cmplist( 196) /   3100221/
      data cmplist( 197) /   3000223/
      data cmplist( 198) /         0/
      data cmplist( 199) /         0/
      data cmplist( 200) /         0/
      data cmplist( 201) /   3100021/
      data cmplist( 202) /   3060111/
      data cmplist( 203) /   3160111/
      data cmplist( 204) /   3130113/
      data cmplist( 205) /   3140113/
      data cmplist( 206) /   3150113/
      data cmplist( 207) /   3160113/
      data cmplist( 208) /         0/
      data cmplist( 209) /         0/
      data cmplist( 210) /         0/
      data cmplist( 211) /   4000001/
      data cmplist( 212) /   4000002/
      data cmplist( 213) /   4000003/
      data cmplist( 214) /   4000004/
      data cmplist( 215) /   4000005/
      data cmplist( 216) /   4000006/
      data cmplist( 217) /   4000007/
      data cmplist( 218) /   4000008/
      data cmplist( 219) /   4000009/
      data cmplist( 220) /         0/
      data cmplist( 221) /   4000011/
      data cmplist( 222) /   4000012/
      data cmplist( 223) /   4000013/
      data cmplist( 224) /   4000014/
      data cmplist( 225) /   4000015/
      data cmplist( 226) /   4000016/
      data cmplist( 227) /   4000017/
      data cmplist( 228) /   4000018/
      data cmplist( 229) /   4000019/
      data cmplist( 230) /   4000039/
      data cmplist( 231) /   9900061/
      data cmplist( 232) /   9900062/
      data cmplist( 233) /   9900024/
      data cmplist( 234) /   9900012/
      data cmplist( 235) /   9900014/
      data cmplist( 236) /   9900016/
      data cmplist( 237) /   9900023/
      data cmplist( 238) /   9221132/
      data cmplist( 239) /   9331122/
      data cmplist( 240) /         0/
      data cmplist( 241) /   9910113/
      data cmplist( 242) /   9910211/
      data cmplist( 243) /   9910223/
      data cmplist( 244) /   9910333/
      data cmplist( 245) /   9910443/
      data cmplist( 246) /   9912112/
      data cmplist( 247) /   9912212/
      data cmplist( 248) /   9900443/
      data cmplist( 249) /   9900441/
      data cmplist( 250) /   9910441/
      data cmplist( 251) /   9920022/
      data cmplist( 252) /   9922212/
      data cmplist( 253) /     10022/
      data cmplist( 254) /     20022/
      data cmplist( 255) /   9900553/
      data cmplist( 256) /   9900551/
      data cmplist( 257) /   9910551/
      data cmplist( 258) /         0/
      data cmplist( 259) /         0/
      data cmplist( 260) /         0/
      data cmplist( 261) /         0/
      data cmplist( 262) /         0/
      data cmplist( 263) /         0/
      data cmplist( 264) /         0/
      data cmplist( 265) /         0/
      data cmplist( 266) /         0/
      data cmplist( 267) /         0/
      data cmplist( 268) /         0/
      data cmplist( 269) /         0/
      data cmplist( 270) /         0/
      data cmplist( 271) /         0/
      data cmplist( 272) /         0/
      data cmplist( 273) /         0/
      data cmplist( 274) /         0/
      data cmplist( 275) /         0/
      data cmplist( 276) /         0/
      data cmplist( 277) /         0/
      data cmplist( 278) /         0/
      data cmplist( 279) /         0/
      data cmplist( 280) /         0/
      data cmplist( 281) /         0/
      data cmplist( 282) /         0/
      data cmplist( 283) /         0/
      data cmplist( 284) /         0/
      data cmplist( 285) /         0/
      data cmplist( 286) /         0/
      data cmplist( 287) /         0/
      data cmplist( 288) /         0/
      data cmplist( 289) /         0/
      data cmplist( 290) /         0/
      data cmplist( 291) /         0/
      data cmplist( 292) /         0/
      data cmplist( 293) /         0/
      data cmplist( 294) /         0/
      data cmplist( 295) /         0/
      data cmplist( 296) /         0/
      data cmplist( 297) /         0/
      data cmplist( 298) /         0/
      data cmplist( 299) /         0/
      data cmplist( 300) /         0/
      data cmplist( 301) /         0/
      data cmplist( 302) /         0/
      data cmplist( 303) /         0/
      data cmplist( 304) /         0/
      data cmplist( 305) /         0/
      data cmplist( 306) /         0/
      data cmplist( 307) /         0/
      data cmplist( 308) /         0/
      data cmplist( 309) /         0/
      data cmplist( 310) /         0/
      data cmplist( 311) /         0/
      data cmplist( 312) /         0/
      data cmplist( 313) /         0/
      data cmplist( 314) /         0/
      data cmplist( 315) /         0/
      data cmplist( 316) /         0/
      data cmplist( 317) /         0/
      data cmplist( 318) /         0/
      data cmplist( 319) /         0/
      data cmplist( 320) /         0/
      data cmplist( 321) /         0/
      data cmplist( 322) /         0/
      data cmplist( 323) /         0/
      data cmplist( 324) /         0/
      data cmplist( 325) /         0/
      data cmplist( 326) /         0/
      data cmplist( 327) /         0/
      data cmplist( 328) /         0/
      data cmplist( 329) /         0/
      data cmplist( 330) /         0/
      data cmplist( 331) /         0/
      data cmplist( 332) /         0/
      data cmplist( 333) /         0/
      data cmplist( 334) /         0/
      data cmplist( 335) /         0/
      data cmplist( 336) /         0/
      data cmplist( 337) /         0/
      data cmplist( 338) /         0/
      data cmplist( 339) /         0/
      data cmplist( 340) /         0/
      data cmplist( 341) /         0/
      data cmplist( 342) /         0/
      data cmplist( 343) /         0/
      data cmplist( 344) /         0/
      data cmplist( 345) /         0/
      data cmplist( 346) /         0/
      data cmplist( 347) /         0/
      data cmplist( 348) /         0/
      data cmplist( 349) /         0/
      data cmplist( 350) /         0/
      data cmplist( 351) /         0/
      data cmplist( 352) /         0/
      data cmplist( 353) /         0/
      data cmplist( 354) /         0/
      data cmplist( 355) /         0/
      data cmplist( 356) /         0/
      data cmplist( 357) /         0/
      data cmplist( 358) /         0/
      data cmplist( 359) /         0/
      data cmplist( 360) /         0/
      data cmplist( 361) /         0/
      data cmplist( 362) /         0/
      data cmplist( 363) /         0/
      data cmplist( 364) /         0/
      data cmplist( 365) /         0/
      data cmplist( 366) /         0/
      data cmplist( 367) /         0/
      data cmplist( 368) /         0/
      data cmplist( 369) /         0/
      data cmplist( 370) /         0/
      data cmplist( 371) /         0/
      data cmplist( 372) /         0/
      data cmplist( 373) /         0/
      data cmplist( 374) /         0/
      data cmplist( 375) /         0/
      data cmplist( 376) /         0/
      data cmplist( 377) /         0/
      data cmplist( 378) /         0/
      data cmplist( 379) /         0/
      data cmplist( 380) /         0/
      data cmplist( 381) /         0/
      data cmplist( 382) /         0/
      data cmplist( 383) /         0/
      data cmplist( 384) /         0/
      data cmplist( 385) /         0/
      data cmplist( 386) /         0/
      data cmplist( 387) /         0/
      data cmplist( 388) /         0/
      data cmplist( 389) /         0/
      data cmplist( 390) /         0/
      data cmplist( 391) /         0/
      data cmplist( 392) /         0/
      data cmplist( 393) /         0/
      data cmplist( 394) /         0/
      data cmplist( 395) /         0/
      data cmplist( 396) /         0/
      data cmplist( 397) /         0/
      data cmplist( 398) /         0/
      data cmplist( 399) /         0/
      data cmplist( 400) /         0/
      data cmplist( 401) /         0/
      data cmplist( 402) /         0/
      data cmplist( 403) /         0/
      data cmplist( 404) /         0/
      data cmplist( 405) /         0/
      data cmplist( 406) /         0/
      data cmplist( 407) /         0/
      data cmplist( 408) /         0/
      data cmplist( 409) /         0/
      data cmplist( 410) /         0/
      data cmplist( 411) /         0/
      data cmplist( 412) /         0/
      data cmplist( 413) /         0/
      data cmplist( 414) /         0/
      data cmplist( 415) /         0/
      data cmplist( 416) /         0/
      data cmplist( 417) /         0/
      data cmplist( 418) /         0/
      data cmplist( 419) /         0/
      data cmplist( 420) /         0/
      data cmplist( 421) /         0/
      data cmplist( 422) /         0/
      data cmplist( 423) /         0/
      data cmplist( 424) /         0/
      data cmplist( 425) /         0/
      data cmplist( 426) /         0/
      data cmplist( 427) /         0/
      data cmplist( 428) /         0/
      data cmplist( 429) /         0/
      data cmplist( 430) /         0/
      data cmplist( 431) /         0/
      data cmplist( 432) /         0/
      data cmplist( 433) /         0/
      data cmplist( 434) /         0/
      data cmplist( 435) /         0/
      data cmplist( 436) /         0/
      data cmplist( 437) /         0/
      data cmplist( 438) /         0/
      data cmplist( 439) /         0/
      data cmplist( 440) /         0/
      data cmplist( 441) /         0/
      data cmplist( 442) /         0/
      data cmplist( 443) /         0/
      data cmplist( 444) /         0/
      data cmplist( 445) /         0/
      data cmplist( 446) /         0/
      data cmplist( 447) /         0/
      data cmplist( 448) /         0/
      data cmplist( 449) /         0/
      data cmplist( 450) /         0/
      data cmplist( 451) /         0/
      data cmplist( 452) /         0/
      data cmplist( 453) /         0/
      data cmplist( 454) /         0/
      data cmplist( 455) /         0/
      data cmplist( 456) /         0/
      data cmplist( 457) /         0/
      data cmplist( 458) /         0/
      data cmplist( 459) /         0/
      data cmplist( 460) /         0/
      data cmplist( 461) /         0/
      data cmplist( 462) /         0/
      data cmplist( 463) /         0/
      data cmplist( 464) /         0/
      data cmplist( 465) /         0/
      data cmplist( 466) /         0/
      data cmplist( 467) /         0/
      data cmplist( 468) /         0/
      data cmplist( 469) /         0/
      data cmplist( 470) /         0/
      data cmplist( 471) /         0/
      data cmplist( 472) /         0/
      data cmplist( 473) /         0/
      data cmplist( 474) /         0/
      data cmplist( 475) /         0/
      data cmplist( 476) /         0/
      data cmplist( 477) /         0/
      data cmplist( 478) /         0/
      data cmplist( 479) /         0/
      data cmplist( 480) /         0/
      data cmplist( 481) /         0/
      data cmplist( 482) /         0/
      data cmplist( 483) /         0/
      data cmplist( 484) /         0/
      data cmplist( 485) /         0/
      data cmplist( 486) /         0/
      data cmplist( 487) /         0/
      data cmplist( 488) /         0/
      data cmplist( 489) /         0/
      data cmplist( 490) /         0/
      data cmplist( 491) /         0/
      data cmplist( 492) /         0/
      data cmplist( 493) /         0/
      data cmplist( 494) /         0/
      data cmplist( 495) /         0/
      data cmplist( 496) /         0/
      data cmplist( 497) /         0/
      data cmplist( 498) /         0/
      data cmplist( 499) /         0/
      data cmplist( 500) /         0/
      data cmplist( 501) /         0/
      data cmplist( 502) /         0/
      data cmplist( 503) /         0/
      data cmplist( 504) /         0/
      data cmplist( 505) /         0/
      data cmplist( 506) /         0/
      data cmplist( 507) /         0/
      data cmplist( 508) /         0/
      data cmplist( 509) /         0/
      data cmplist( 510) /         0/
      data cmplist( 511) /         0/
      data cmplist( 512) /         0/
      data cmplist( 513) /         0/
      data cmplist( 514) /         0/
      data cmplist( 515) /         0/
      data cmplist( 516) /         0/
      data cmplist( 517) /         0/
      data cmplist( 518) /         0/
      data cmplist( 519) /         0/
      data cmplist( 520) /         0/
      data cmplist( 521) /         0/
      data cmplist( 522) /         0/
      data cmplist( 523) /         0/
      data cmplist( 524) /         0/
      data cmplist( 525) /         0/
      data cmplist( 526) /         0/
      data cmplist( 527) /         0/
      data cmplist( 528) /         0/
      data cmplist( 529) /         0/
      data cmplist( 530) /         0/
      data cmplist( 531) /         0/
      data cmplist( 532) /         0/
      data cmplist( 533) /         0/
      data cmplist( 534) /         0/
      data cmplist( 535) /         0/
      data cmplist( 536) /         0/
      data cmplist( 537) /         0/
      data cmplist( 538) /         0/
      data cmplist( 539) /         0/
      data cmplist( 540) /         0/
      data cmplist( 541) /         0/
      data cmplist( 542) /         0/
      data cmplist( 543) /         0/
      data cmplist( 544) /         0/
      data cmplist( 545) /         0/
      data cmplist( 546) /         0/
      data cmplist( 547) /         0/
      data cmplist( 548) /         0/
      data cmplist( 549) /         0/
      data cmplist( 550) /         0/
      data cmplist( 551) /         0/
      data cmplist( 552) /         0/
      data cmplist( 553) /         0/
      data cmplist( 554) /         0/
      data cmplist( 555) /         0/
      data cmplist( 556) /         0/
      data cmplist( 557) /         0/
      data cmplist( 558) /         0/
      data cmplist( 559) /         0/
      data cmplist( 560) /         0/
      data cmplist( 561) /         0/
      data cmplist( 562) /         0/
      data cmplist( 563) /         0/
      data cmplist( 564) /         0/
      data cmplist( 565) /         0/
      data cmplist( 566) /         0/
      data cmplist( 567) /         0/
      data cmplist( 568) /         0/
      data cmplist( 569) /         0/
      data cmplist( 570) /         0/
      data cmplist( 571) /         0/
      data cmplist( 572) /         0/
      data cmplist( 573) /         0/
      data cmplist( 574) /         0/
      data cmplist( 575) /         0/
      data cmplist( 576) /         0/
      data cmplist( 577) /         0/
      data cmplist( 578) /         0/
      data cmplist( 579) /         0/
      data cmplist( 580) /         0/
      data cmplist( 581) /         0/
      data cmplist( 582) /         0/
      data cmplist( 583) /         0/
      data cmplist( 584) /         0/
      data cmplist( 585) /         0/
      data cmplist( 586) /         0/
      data cmplist( 587) /         0/
      data cmplist( 588) /         0/
      data cmplist( 589) /         0/
      data cmplist( 590) /         0/
      data cmplist( 591) /         0/
      data cmplist( 592) /         0/
      data cmplist( 593) /         0/
      data cmplist( 594) /         0/
      data cmplist( 595) /         0/
      data cmplist( 596) /         0/
      data cmplist( 597) /         0/
      data cmplist( 598) /         0/
      data cmplist( 599) /         0/
      data cmplist( 600) /         0/
      data cmplist( 601) /         0/
      data cmplist( 602) /         0/
      data cmplist( 603) /         0/
      data cmplist( 604) /         0/
      data cmplist( 605) /         0/
      data cmplist( 606) /         0/
      data cmplist( 607) /         0/
      data cmplist( 608) /         0/
      data cmplist( 609) /         0/
      data cmplist( 610) /         0/
      data cmplist( 611) /         0/
      data cmplist( 612) /         0/
      data cmplist( 613) /         0/
      data cmplist( 614) /         0/
      data cmplist( 615) /         0/
      data cmplist( 616) /         0/
      data cmplist( 617) /         0/
      data cmplist( 618) /         0/
      data cmplist( 619) /         0/
      data cmplist( 620) /         0/
      data cmplist( 621) /         0/
      data cmplist( 622) /         0/
      data cmplist( 623) /         0/
      data cmplist( 624) /         0/
      data cmplist( 625) /         0/
      data cmplist( 626) /         0/
      data cmplist( 627) /         0/
      data cmplist( 628) /         0/
      data cmplist( 629) /         0/
      data cmplist( 630) /         0/
      data cmplist( 631) /         0/
      data cmplist( 632) /         0/
      data cmplist( 633) /         0/
      data cmplist( 634) /         0/
      data cmplist( 635) /         0/
      data cmplist( 636) /         0/
      data cmplist( 637) /         0/
      data cmplist( 638) /         0/
      data cmplist( 639) /         0/
      data cmplist( 640) /         0/
      data cmplist( 641) /         0/
      data cmplist( 642) /         0/
      data cmplist( 643) /         0/
      data cmplist( 644) /         0/
      data cmplist( 645) /         0/
      data cmplist( 646) /         0/
      data cmplist( 647) /         0/
      data cmplist( 648) /         0/
      data cmplist( 649) /         0/
      data cmplist( 650) /         0/
      data cmplist( 651) /         0/
      data cmplist( 652) /         0/
      data cmplist( 653) /         0/
      data cmplist( 654) /         0/
      data cmplist( 655) /         0/
      data cmplist( 656) /         0/
      data cmplist( 657) /         0/
      data cmplist( 658) /         0/
      data cmplist( 659) /         0/
      data cmplist( 660) /         0/
      data cmplist( 661) /         0/
      data cmplist( 662) /         0/
      data cmplist( 663) /         0/
      data cmplist( 664) /         0/
      data cmplist( 665) /         0/
      data cmplist( 666) /         0/
      data cmplist( 667) /         0/
      data cmplist( 668) /         0/
      data cmplist( 669) /         0/
      data cmplist( 670) /         0/
      data cmplist( 671) /         0/
      data cmplist( 672) /         0/
      data cmplist( 673) /         0/
      data cmplist( 674) /         0/
      data cmplist( 675) /         0/
      data cmplist( 676) /         0/
      data cmplist( 677) /         0/
      data cmplist( 678) /         0/
      data cmplist( 679) /         0/
      data cmplist( 680) /         0/
      data cmplist( 681) /         0/
      data cmplist( 682) /         0/
      data cmplist( 683) /         0/
      data cmplist( 684) /         0/
      data cmplist( 685) /         0/
      data cmplist( 686) /         0/
      data cmplist( 687) /         0/
      data cmplist( 688) /         0/
      data cmplist( 689) /         0/
      data cmplist( 690) /         0/
      data cmplist( 691) /         0/
      data cmplist( 692) /         0/
      data cmplist( 693) /         0/
      data cmplist( 694) /         0/
      data cmplist( 695) /         0/
      data cmplist( 696) /         0/
      data cmplist( 697) /         0/
      data cmplist( 698) /         0/
      data cmplist( 699) /         0/
      data cmplist( 700) /         0/
      data cmplist( 701) /         0/
      data cmplist( 702) /         0/
      data cmplist( 703) /         0/
      data cmplist( 704) /         0/
      data cmplist( 705) /         0/
      data cmplist( 706) /         0/
      data cmplist( 707) /         0/
      data cmplist( 708) /         0/
      data cmplist( 709) /         0/
      data cmplist( 710) /         0/
      data cmplist( 711) /         0/
      data cmplist( 712) /         0/
      data cmplist( 713) /         0/
      data cmplist( 714) /         0/
      data cmplist( 715) /         0/
      data cmplist( 716) /         0/
      data cmplist( 717) /         0/
      data cmplist( 718) /         0/
      data cmplist( 719) /         0/
      data cmplist( 720) /         0/
      data cmplist( 721) /         0/
      data cmplist( 722) /         0/
      data cmplist( 723) /         0/
      data cmplist( 724) /         0/
      data cmplist( 725) /         0/
      data cmplist( 726) /         0/
      data cmplist( 727) /         0/
      data cmplist( 728) /         0/
      data cmplist( 729) /         0/
      data cmplist( 730) /         0/
      data cmplist( 731) /         0/
      data cmplist( 732) /         0/
      data cmplist( 733) /         0/
      data cmplist( 734) /         0/
      data cmplist( 735) /         0/
      data cmplist( 736) /         0/
      data cmplist( 737) /         0/
      data cmplist( 738) /         0/
      data cmplist( 739) /         0/
      data cmplist( 740) /         0/
      data cmplist( 741) /         0/
      data cmplist( 742) /         0/
      data cmplist( 743) /         0/
      data cmplist( 744) /         0/
      data cmplist( 745) /         0/
      data cmplist( 746) /         0/
      data cmplist( 747) /         0/
      data cmplist( 748) /         0/
      data cmplist( 749) /         0/
      data cmplist( 750) /         0/
      data cmplist( 751) /         0/
      data cmplist( 752) /         0/
      data cmplist( 753) /         0/
      data cmplist( 754) /         0/
      data cmplist( 755) /         0/
      data cmplist( 756) /         0/
      data cmplist( 757) /         0/
      data cmplist( 758) /         0/
      data cmplist( 759) /         0/
      data cmplist( 760) /         0/
      data cmplist( 761) /         0/
      data cmplist( 762) /         0/
      data cmplist( 763) /         0/
      data cmplist( 764) /         0/
      data cmplist( 765) /         0/
      data cmplist( 766) /         0/
      data cmplist( 767) /         0/
      data cmplist( 768) /         0/
      data cmplist( 769) /         0/
      data cmplist( 770) /         0/
      data cmplist( 771) /         0/
      data cmplist( 772) /         0/
      data cmplist( 773) /         0/
      data cmplist( 774) /         0/
      data cmplist( 775) /         0/
      data cmplist( 776) /         0/
      data cmplist( 777) /         0/
      data cmplist( 778) /         0/
      data cmplist( 779) /         0/
      data cmplist( 780) /         0/
      data cmplist( 781) /         0/
      data cmplist( 782) /         0/
      data cmplist( 783) /         0/
      data cmplist( 784) /         0/
      data cmplist( 785) /         0/
      data cmplist( 786) /         0/
      data cmplist( 787) /         0/
      data cmplist( 788) /         0/
      data cmplist( 789) /         0/
      data cmplist( 790) /         0/
      data cmplist( 791) /         0/
      data cmplist( 792) /         0/
      data cmplist( 793) /         0/
      data cmplist( 794) /         0/
      data cmplist( 795) /         0/
      data cmplist( 796) /         0/
      data cmplist( 797) /         0/
      data cmplist( 798) /         0/
      data cmplist( 799) /         0/
      data cmplist( 800) /         0/
      data cmplist( 801) /         0/
      data cmplist( 802) /         0/
      data cmplist( 803) /         0/
      data cmplist( 804) /         0/
      data cmplist( 805) /         0/
      data cmplist( 806) /         0/
      data cmplist( 807) /         0/
      data cmplist( 808) /         0/
      data cmplist( 809) /         0/
      data cmplist( 810) /         0/
      data cmplist( 811) /         0/
      data cmplist( 812) /         0/
      data cmplist( 813) /         0/
      data cmplist( 814) /         0/
      data cmplist( 815) /         0/
      data cmplist( 816) /         0/
      data cmplist( 817) /         0/
      data cmplist( 818) /         0/
      data cmplist( 819) /         0/
      data cmplist( 820) /         0/
      data cmplist( 821) /         0/
      data cmplist( 822) /         0/
      data cmplist( 823) /         0/
      data cmplist( 824) /         0/
      data cmplist( 825) /         0/
      data cmplist( 826) /         0/
      data cmplist( 827) /         0/
      data cmplist( 828) /         0/
      data cmplist( 829) /         0/
      data cmplist( 830) /         0/
      data cmplist( 831) /         0/
      data cmplist( 832) /         0/
      data cmplist( 833) /         0/
      data cmplist( 834) /         0/
      data cmplist( 835) /         0/
      data cmplist( 836) /         0/
      data cmplist( 837) /         0/
      data cmplist( 838) /         0/
      data cmplist( 839) /         0/
      data cmplist( 840) /         0/
      data cmplist( 841) /         0/
      data cmplist( 842) /         0/
      data cmplist( 843) /         0/
      data cmplist( 844) /         0/
      data cmplist( 845) /         0/
      data cmplist( 846) /         0/
      data cmplist( 847) /         0/
      data cmplist( 848) /         0/
      data cmplist( 849) /         0/
      data cmplist( 850) /         0/
      data cmplist( 851) /         0/
      data cmplist( 852) /         0/
      data cmplist( 853) /         0/
      data cmplist( 854) /         0/
      data cmplist( 855) /         0/
      data cmplist( 856) /         0/
      data cmplist( 857) /         0/
      data cmplist( 858) /         0/
      data cmplist( 859) /         0/
      data cmplist( 860) /         0/
      data cmplist( 861) /         0/
      data cmplist( 862) /         0/
      data cmplist( 863) /         0/
      data cmplist( 864) /         0/
      data cmplist( 865) /         0/
      data cmplist( 866) /         0/
      data cmplist( 867) /         0/
      data cmplist( 868) /         0/
      data cmplist( 869) /         0/
      data cmplist( 870) /         0/
      data cmplist( 871) /         0/
      data cmplist( 872) /         0/
      data cmplist( 873) /         0/
      data cmplist( 874) /         0/
      data cmplist( 875) /         0/
      data cmplist( 876) /         0/
      data cmplist( 877) /         0/
      data cmplist( 878) /         0/
      data cmplist( 879) /         0/
      data cmplist( 880) /         0/
      data cmplist( 881) /         0/
      data cmplist( 882) /         0/
      data cmplist( 883) /         0/
      data cmplist( 884) /         0/
      data cmplist( 885) /         0/
      data cmplist( 886) /         0/
      data cmplist( 887) /         0/
      data cmplist( 888) /         0/
      data cmplist( 889) /         0/
      data cmplist( 890) /         0/
      data cmplist( 891) /         0/
      data cmplist( 892) /         0/
      data cmplist( 893) /         0/
      data cmplist( 894) /         0/
      data cmplist( 895) /         0/
      data cmplist( 896) /         0/
      data cmplist( 897) /         0/
      data cmplist( 898) /         0/
      data cmplist( 899) /         0/
      data cmplist( 900) /         0/
      data cmplist( 901) /         0/
      data cmplist( 902) /         0/
      data cmplist( 903) /         0/
      data cmplist( 904) /         0/
      data cmplist( 905) /         0/
      data cmplist( 906) /         0/
      data cmplist( 907) /         0/
      data cmplist( 908) /         0/
      data cmplist( 909) /         0/
      data cmplist( 910) /         0/
      data cmplist( 911) /         0/
      data cmplist( 912) /         0/
      data cmplist( 913) /         0/
      data cmplist( 914) /         0/
      data cmplist( 915) /         0/
      data cmplist( 916) /         0/
      data cmplist( 917) /         0/
      data cmplist( 918) /         0/
      data cmplist( 919) /         0/
      data cmplist( 920) /         0/
      data cmplist( 921) /         0/
      data cmplist( 922) /         0/
      data cmplist( 923) /         0/
      data cmplist( 924) /         0/
      data cmplist( 925) /         0/
      data cmplist( 926) /         0/
      data cmplist( 927) /         0/
      data cmplist( 928) /         0/
      data cmplist( 929) /         0/
      data cmplist( 930) /         0/
      data cmplist( 931) /         0/
      data cmplist( 932) /         0/
      data cmplist( 933) /         0/
      data cmplist( 934) /         0/
      data cmplist( 935) /         0/
      data cmplist( 936) /         0/
      data cmplist( 937) /         0/
      data cmplist( 938) /         0/
      data cmplist( 939) /         0/
      data cmplist( 940) /         0/
      data cmplist( 941) /         0/
      data cmplist( 942) /         0/
      data cmplist( 943) /         0/
      data cmplist( 944) /         0/
      data cmplist( 945) /         0/
      data cmplist( 946) /         0/
      data cmplist( 947) /         0/
      data cmplist( 948) /         0/
      data cmplist( 949) /         0/
      data cmplist( 950) /         0/
      data cmplist( 951) /         0/
      data cmplist( 952) /         0/
      data cmplist( 953) /         0/
      data cmplist( 954) /         0/
      data cmplist( 955) /         0/
      data cmplist( 956) /         0/
      data cmplist( 957) /         0/
      data cmplist( 958) /         0/
      data cmplist( 959) /         0/
      data cmplist( 960) /         0/
      data cmplist( 961) /         0/
      data cmplist( 962) /         0/
      data cmplist( 963) /         0/
      data cmplist( 964) /         0/
      data cmplist( 965) /         0/
      data cmplist( 966) /         0/
      data cmplist( 967) /         0/
      data cmplist( 968) /         0/
      data cmplist( 969) /         0/
      data cmplist( 970) /         0/
      data cmplist( 971) /         0/
      data cmplist( 972) /         0/
      data cmplist( 973) /         0/
      data cmplist( 974) /         0/
      data cmplist( 975) /         0/
      data cmplist( 976) /         0/
      data cmplist( 977) /         0/
      data cmplist( 978) /         0/
      data cmplist( 979) /         0/
      data cmplist( 980) /         0/
      data cmplist( 981) /         0/
      data cmplist( 982) /         0/
      data cmplist( 983) /         0/
      data cmplist( 984) /         0/
      data cmplist( 985) /         0/
      data cmplist( 986) /         0/
      data cmplist( 987) /         0/
      data cmplist( 988) /         0/
      data cmplist( 989) /         0/
      data cmplist( 990) /         0/
      data cmplist( 991) /         0/
      data cmplist( 992) /         0/
      data cmplist( 993) /         0/
      data cmplist( 994) /         0/
      data cmplist( 995) /         0/
      data cmplist( 996) /         0/
      data cmplist( 997) /         0/
      data cmplist( 998) /         0/
      data cmplist( 999) /         0/
      data cmplist(1000) /         0/
      data cmplist(1001) /         0/
      data cmplist(1002) /         0/
      data cmplist(1003) /         0/
      data cmplist(1004) /         0/
      data cmplist(1005) /         0/
      data cmplist(1006) /         0/
      data cmplist(1007) /         0/
      data cmplist(1008) /         0/
      data cmplist(1009) /         0/
      data cmplist(1010) /         0/
      data cmplist(1011) /         0/
      data cmplist(1012) /         0/
      data cmplist(1013) /         0/
      data cmplist(1014) /         0/
      data cmplist(1015) /         0/
      data cmplist(1016) /         0/
      data cmplist(1017) /         0/
      data cmplist(1018) /         0/
      data cmplist(1019) /         0/
      data cmplist(1020) /         0/
      data cmplist(1021) /         0/
      data cmplist(1022) /         0/
      data cmplist(1023) /         0/
      data cmplist(1024) /         0/
      data cmplist(1025) /         0/
      data cmplist(1026) /         0/
      data cmplist(1027) /         0/
      data cmplist(1028) /         0/
      data cmplist(1029) /         0/
      data cmplist(1030) /         0/
      data cmplist(1031) /         0/
      data cmplist(1032) /         0/
      data cmplist(1033) /         0/
      data cmplist(1034) /         0/
      data cmplist(1035) /         0/
      data cmplist(1036) /         0/
      data cmplist(1037) /         0/
      data cmplist(1038) /         0/
      data cmplist(1039) /         0/
      data cmplist(1040) /         0/
      data cmplist(1041) /         0/
      data cmplist(1042) /         0/
      data cmplist(1043) /         0/
      data cmplist(1044) /         0/
      data cmplist(1045) /         0/
      data cmplist(1046) /         0/
      data cmplist(1047) /         0/
      data cmplist(1048) /         0/
      data cmplist(1049) /         0/
      data cmplist(1050) /         0/
      data cmplist(1051) /         0/
      data cmplist(1052) /         0/
      data cmplist(1053) /         0/
      data cmplist(1054) /         0/
      data cmplist(1055) /         0/
      data cmplist(1056) /         0/
      data cmplist(1057) /         0/
      data cmplist(1058) /         0/
      data cmplist(1059) /         0/
      data cmplist(1060) /         0/
      data cmplist(1061) /         0/
      data cmplist(1062) /         0/
      data cmplist(1063) /         0/
      data cmplist(1064) /         0/
      data cmplist(1065) /         0/
      data cmplist(1066) /         0/
      data cmplist(1067) /         0/
      data cmplist(1068) /         0/
      data cmplist(1069) /         0/
      data cmplist(1070) /         0/
      data cmplist(1071) /         0/
      data cmplist(1072) /         0/
      data cmplist(1073) /         0/
      data cmplist(1074) /         0/
      data cmplist(1075) /         0/
      data cmplist(1076) /         0/
      data cmplist(1077) /         0/
      data cmplist(1078) /         0/
      data cmplist(1079) /         0/
      data cmplist(1080) /         0/
      data cmplist(1081) /         0/
      data cmplist(1082) /         0/
      data cmplist(1083) /         0/
      data cmplist(1084) /         0/
      data cmplist(1085) /         0/
      data cmplist(1086) /         0/
      data cmplist(1087) /         0/
      data cmplist(1088) /         0/
      data cmplist(1089) /         0/
      data cmplist(1090) /         0/
      data cmplist(1091) /         0/
      data cmplist(1092) /         0/
      data cmplist(1093) /         0/
      data cmplist(1094) /         0/
      data cmplist(1095) /         0/
      data cmplist(1096) /         0/
      data cmplist(1097) /         0/
      data cmplist(1098) /         0/
      data cmplist(1099) /         0/
      data cmplist(1100) /         0/
      data cmplist(1101) /         0/
      data cmplist(1102) /         0/
      data cmplist(1103) /         0/
      data cmplist(1104) /         0/
      data cmplist(1105) /         0/
      data cmplist(1106) /         0/
      data cmplist(1107) /         0/
      data cmplist(1108) /         0/
      data cmplist(1109) /         0/
      data cmplist(1110) /         0/
      data cmplist(1111) /         0/
      data cmplist(1112) /         0/
      data cmplist(1113) /         0/
      data cmplist(1114) /         0/
      data cmplist(1115) /         0/
      data cmplist(1116) /         0/
      data cmplist(1117) /         0/
      data cmplist(1118) /         0/
      data cmplist(1119) /         0/
      data cmplist(1120) /         0/
      data cmplist(1121) /         0/
      data cmplist(1122) /         0/
      data cmplist(1123) /         0/
      data cmplist(1124) /         0/
      data cmplist(1125) /         0/
      data cmplist(1126) /         0/
      data cmplist(1127) /         0/
      data cmplist(1128) /         0/
      data cmplist(1129) /         0/
      data cmplist(1130) /         0/
      data cmplist(1131) /         0/
      data cmplist(1132) /         0/
      data cmplist(1133) /         0/
      data cmplist(1134) /         0/
      data cmplist(1135) /         0/
      data cmplist(1136) /         0/
      data cmplist(1137) /         0/
      data cmplist(1138) /         0/
      data cmplist(1139) /         0/
      data cmplist(1140) /         0/
      data cmplist(1141) /         0/
      data cmplist(1142) /         0/
      data cmplist(1143) /         0/
      data cmplist(1144) /         0/
      data cmplist(1145) /         0/
      data cmplist(1146) /         0/
      data cmplist(1147) /         0/
      data cmplist(1148) /         0/
      data cmplist(1149) /         0/
      data cmplist(1150) /         0/
      data cmplist(1151) /         0/
      data cmplist(1152) /         0/
      data cmplist(1153) /         0/
      data cmplist(1154) /         0/
      data cmplist(1155) /         0/
      data cmplist(1156) /         0/
      data cmplist(1157) /         0/
      data cmplist(1158) /         0/
      data cmplist(1159) /         0/
      data cmplist(1160) /         0/
      data cmplist(1161) /         0/
      data cmplist(1162) /         0/
      data cmplist(1163) /         0/
      data cmplist(1164) /         0/
      data cmplist(1165) /         0/
      data cmplist(1166) /         0/
      data cmplist(1167) /         0/
      data cmplist(1168) /         0/
      data cmplist(1169) /         0/
      data cmplist(1170) /         0/
      data cmplist(1171) /         0/
      data cmplist(1172) /         0/
      data cmplist(1173) /         0/
      data cmplist(1174) /         0/
      data cmplist(1175) /         0/
      data cmplist(1176) /         0/
      data cmplist(1177) /         0/
      data cmplist(1178) /         0/
      data cmplist(1179) /         0/
      data cmplist(1180) /         0/
      data cmplist(1181) /         0/
      data cmplist(1182) /         0/
      data cmplist(1183) /         0/
      data cmplist(1184) /         0/
      data cmplist(1185) /         0/
      data cmplist(1186) /         0/
      data cmplist(1187) /         0/
      data cmplist(1188) /         0/
      data cmplist(1189) /         0/
      data cmplist(1190) /         0/
      data cmplist(1191) /         0/
      data cmplist(1192) /         0/
      data cmplist(1193) /         0/
      data cmplist(1194) /         0/
      data cmplist(1195) /         0/
      data cmplist(1196) /         0/
      data cmplist(1197) /         0/
      data cmplist(1198) /         0/
      data cmplist(1199) /         0/
      data cmplist(1200) /         0/
      data cmplist(1201) /         0/
      data cmplist(1202) /         0/
      data cmplist(1203) /         0/
      data cmplist(1204) /         0/
      data cmplist(1205) /         0/
      data cmplist(1206) /         0/
      data cmplist(1207) /         0/
      data cmplist(1208) /         0/
      data cmplist(1209) /         0/
      data cmplist(1210) /         0/
      data cmplist(1211) /         0/
      data cmplist(1212) /         0/
      data cmplist(1213) /         0/
      data cmplist(1214) /         0/
      data cmplist(1215) /         0/
      data cmplist(1216) /         0/
      data cmplist(1217) /         0/
      data cmplist(1218) /         0/
      data cmplist(1219) /         0/
      data cmplist(1220) /         0/
      data cmplist(1221) /         0/
      data cmplist(1222) /         0/
      data cmplist(1223) /         0/
      data cmplist(1224) /         0/
      data cmplist(1225) /         0/
      data cmplist(1226) /         0/
      data cmplist(1227) /         0/
      data cmplist(1228) /         0/
      data cmplist(1229) /         0/
      data cmplist(1230) /         0/
      data cmplist(1231) /         0/
      data cmplist(1232) /         0/
      data cmplist(1233) /         0/
      data cmplist(1234) /         0/
      data cmplist(1235) /         0/
      data cmplist(1236) /         0/
      data cmplist(1237) /         0/
      data cmplist(1238) /         0/
      data cmplist(1239) /         0/
      data cmplist(1240) /         0/
      data cmplist(1241) /         0/
      data cmplist(1242) /         0/
      data cmplist(1243) /         0/
      data cmplist(1244) /         0/
      data cmplist(1245) /         0/
      data cmplist(1246) /         0/
      data cmplist(1247) /         0/
      data cmplist(1248) /         0/
      data cmplist(1249) /         0/
      data cmplist(1250) /         0/
      data cmplist(1251) /         0/
      data cmplist(1252) /         0/
      data cmplist(1253) /         0/
      data cmplist(1254) /         0/
      data cmplist(1255) /         0/
      data cmplist(1256) /         0/
      data cmplist(1257) /         0/
      data cmplist(1258) /         0/
      data cmplist(1259) /         0/
      data cmplist(1260) /         0/
      data cmplist(1261) /         0/
      data cmplist(1262) /         0/
      data cmplist(1263) /         0/
      data cmplist(1264) /         0/
      data cmplist(1265) /         0/
      data cmplist(1266) /         0/
      data cmplist(1267) /         0/
      data cmplist(1268) /         0/
      data cmplist(1269) /         0/
      data cmplist(1270) /         0/
      data cmplist(1271) /         0/
      data cmplist(1272) /         0/
      data cmplist(1273) /         0/
      data cmplist(1274) /         0/
      data cmplist(1275) /         0/
      data cmplist(1276) /         0/
      data cmplist(1277) /         0/
      data cmplist(1278) /         0/
      data cmplist(1279) /         0/
      data cmplist(1280) /         0/
      data cmplist(1281) /         0/
      data cmplist(1282) /         0/
      data cmplist(1283) /         0/
      data cmplist(1284) /         0/
      data cmplist(1285) /         0/
      data cmplist(1286) /         0/
      data cmplist(1287) /         0/
      data cmplist(1288) /         0/
      data cmplist(1289) /         0/
      data cmplist(1290) /         0/
      data cmplist(1291) /         0/
      data cmplist(1292) /         0/
      data cmplist(1293) /         0/
      data cmplist(1294) /         0/
      data cmplist(1295) /         0/
      data cmplist(1296) /         0/
      data cmplist(1297) /         0/
      data cmplist(1298) /         0/
      data cmplist(1299) /         0/
      data cmplist(1300) /         0/
      data cmplist(1301) /         0/
      data cmplist(1302) /         0/
      data cmplist(1303) /         0/
      data cmplist(1304) /         0/
      data cmplist(1305) /         0/
      data cmplist(1306) /         0/
      data cmplist(1307) /         0/
      data cmplist(1308) /         0/
      data cmplist(1309) /         0/
      data cmplist(1310) /         0/
      data cmplist(1311) /         0/
      data cmplist(1312) /         0/
      data cmplist(1313) /         0/
      data cmplist(1314) /         0/
      data cmplist(1315) /         0/
      data cmplist(1316) /         0/
      data cmplist(1317) /         0/
      data cmplist(1318) /         0/
      data cmplist(1319) /         0/
      data cmplist(1320) /         0/
      data cmplist(1321) /         0/
      data cmplist(1322) /         0/
      data cmplist(1323) /         0/
      data cmplist(1324) /         0/
      data cmplist(1325) /         0/
      data cmplist(1326) /         0/
      data cmplist(1327) /         0/
      data cmplist(1328) /         0/
      data cmplist(1329) /         0/
      data cmplist(1330) /         0/
      data cmplist(1331) /         0/
      data cmplist(1332) /         0/
      data cmplist(1333) /         0/
      data cmplist(1334) /         0/
      data cmplist(1335) /         0/
      data cmplist(1336) /         0/
      data cmplist(1337) /         0/
      data cmplist(1338) /         0/
      data cmplist(1339) /         0/
      data cmplist(1340) /         0/
      data cmplist(1341) /         0/
      data cmplist(1342) /         0/
      data cmplist(1343) /         0/
      data cmplist(1344) /         0/
      data cmplist(1345) /         0/
      data cmplist(1346) /         0/
      data cmplist(1347) /         0/
      data cmplist(1348) /         0/
      data cmplist(1349) /         0/
      data cmplist(1350) /         0/
      data cmplist(1351) /         0/
      data cmplist(1352) /         0/
      data cmplist(1353) /         0/
      data cmplist(1354) /         0/
      data cmplist(1355) /         0/
      data cmplist(1356) /         0/
      data cmplist(1357) /         0/
      data cmplist(1358) /         0/
      data cmplist(1359) /         0/
      data cmplist(1360) /         0/
      data cmplist(1361) /         0/
      data cmplist(1362) /         0/
      data cmplist(1363) /         0/
      data cmplist(1364) /         0/
      data cmplist(1365) /         0/
      data cmplist(1366) /         0/
      data cmplist(1367) /         0/
      data cmplist(1368) /         0/
      data cmplist(1369) /         0/
      data cmplist(1370) /         0/
      data cmplist(1371) /         0/
      data cmplist(1372) /         0/
      data cmplist(1373) /         0/
      data cmplist(1374) /         0/
      data cmplist(1375) /         0/
      data cmplist(1376) /         0/
      data cmplist(1377) /         0/
      data cmplist(1378) /         0/
      data cmplist(1379) /         0/
      data cmplist(1380) /         0/
      data cmplist(1381) /         0/
      data cmplist(1382) /         0/
      data cmplist(1383) /         0/
      data cmplist(1384) /         0/
      data cmplist(1385) /         0/
      data cmplist(1386) /         0/
      data cmplist(1387) /         0/
      data cmplist(1388) /         0/
      data cmplist(1389) /         0/
      data cmplist(1390) /         0/
      data cmplist(1391) /         0/
      data cmplist(1392) /         0/
      data cmplist(1393) /         0/
      data cmplist(1394) /         0/
      data cmplist(1395) /         0/
      data cmplist(1396) /         0/
      data cmplist(1397) /         0/
      data cmplist(1398) /         0/
      data cmplist(1399) /         0/
      data cmplist(1400) /         0/
      data cmplist(1401) /         0/
      data cmplist(1402) /         0/
      data cmplist(1403) /         0/
      data cmplist(1404) /         0/
      data cmplist(1405) /         0/
      data cmplist(1406) /         0/
      data cmplist(1407) /         0/
      data cmplist(1408) /         0/
      data cmplist(1409) /         0/
      data cmplist(1410) /         0/
      data cmplist(1411) /         0/
      data cmplist(1412) /         0/
      data cmplist(1413) /         0/
      data cmplist(1414) /         0/
      data cmplist(1415) /         0/
      data cmplist(1416) /         0/
      data cmplist(1417) /         0/
      data cmplist(1418) /         0/
      data cmplist(1419) /         0/
      data cmplist(1420) /         0/
      data cmplist(1421) /         0/
      data cmplist(1422) /         0/
      data cmplist(1423) /         0/
      data cmplist(1424) /         0/
      data cmplist(1425) /         0/
      data cmplist(1426) /         0/
      data cmplist(1427) /         0/
      data cmplist(1428) /         0/
      data cmplist(1429) /         0/
      data cmplist(1430) /         0/
      data cmplist(1431) /         0/
      data cmplist(1432) /         0/
      data cmplist(1433) /         0/
      data cmplist(1434) /         0/
      data cmplist(1435) /         0/
      data cmplist(1436) /         0/
      data cmplist(1437) /         0/
      data cmplist(1438) /         0/
      data cmplist(1439) /         0/
      data cmplist(1440) /         0/
      data cmplist(1441) /         0/
      data cmplist(1442) /         0/
      data cmplist(1443) /         0/
      data cmplist(1444) /         0/
      data cmplist(1445) /         0/
      data cmplist(1446) /         0/
      data cmplist(1447) /         0/
      data cmplist(1448) /         0/
      data cmplist(1449) /         0/
      data cmplist(1450) /         0/
      data cmplist(1451) /         0/
      data cmplist(1452) /         0/
      data cmplist(1453) /         0/
      data cmplist(1454) /         0/
      data cmplist(1455) /         0/
      data cmplist(1456) /         0/
      data cmplist(1457) /         0/
      data cmplist(1458) /         0/
      data cmplist(1459) /         0/
      data cmplist(1460) /         0/
      data cmplist(1461) /         0/
      data cmplist(1462) /         0/
      data cmplist(1463) /         0/
      data cmplist(1464) /         0/
      data cmplist(1465) /         0/
      data cmplist(1466) /         0/
      data cmplist(1467) /         0/
      data cmplist(1468) /         0/
      data cmplist(1469) /         0/
      data cmplist(1470) /         0/
      data cmplist(1471) /         0/
      data cmplist(1472) /         0/
      data cmplist(1473) /         0/
      data cmplist(1474) /         0/
      data cmplist(1475) /         0/
      data cmplist(1476) /         0/
      data cmplist(1477) /         0/
      data cmplist(1478) /         0/
      data cmplist(1479) /         0/
      data cmplist(1480) /         0/
      data cmplist(1481) /         0/
      data cmplist(1482) /         0/
      data cmplist(1483) /         0/
      data cmplist(1484) /         0/
      data cmplist(1485) /         0/
      data cmplist(1486) /         0/
      data cmplist(1487) /         0/
      data cmplist(1488) /         0/
      data cmplist(1489) /         0/
      data cmplist(1490) /         0/
      data cmplist(1491) /         0/
      data cmplist(1492) /         0/
      data cmplist(1493) /         0/
      data cmplist(1494) /         0/
      data cmplist(1495) /         0/
      data cmplist(1496) /         0/
      data cmplist(1497) /         0/
      data cmplist(1498) /         0/
      data cmplist(1499) /         0/
      data cmplist(1500) /         0/
      data cmplist(1501) /         0/
      data cmplist(1502) /         0/
      data cmplist(1503) /         0/
      data cmplist(1504) /         0/
      data cmplist(1505) /         0/
      data cmplist(1506) /         0/
      data cmplist(1507) /         0/
      data cmplist(1508) /         0/
      data cmplist(1509) /         0/
      data cmplist(1510) /         0/
      data cmplist(1511) /         0/
      data cmplist(1512) /         0/
      data cmplist(1513) /         0/
      data cmplist(1514) /         0/
      data cmplist(1515) /         0/
      data cmplist(1516) /         0/
      data cmplist(1517) /         0/
      data cmplist(1518) /         0/
      data cmplist(1519) /         0/
      data cmplist(1520) /         0/
      data cmplist(1521) /         0/
      data cmplist(1522) /         0/
      data cmplist(1523) /         0/
      data cmplist(1524) /         0/
      data cmplist(1525) /         0/
      data cmplist(1526) /         0/
      data cmplist(1527) /         0/
      data cmplist(1528) /         0/
      data cmplist(1529) /         0/
      data cmplist(1530) /         0/
      data cmplist(1531) /         0/
      data cmplist(1532) /         0/
      data cmplist(1533) /         0/
      data cmplist(1534) /         0/
      data cmplist(1535) /         0/
      data cmplist(1536) /         0/
      data cmplist(1537) /         0/
      data cmplist(1538) /         0/
      data cmplist(1539) /         0/
      data cmplist(1540) /         0/
      data cmplist(1541) /         0/
      data cmplist(1542) /         0/
      data cmplist(1543) /         0/
      data cmplist(1544) /         0/
      data cmplist(1545) /         0/
      data cmplist(1546) /         0/
      data cmplist(1547) /         0/
      data cmplist(1548) /         0/
      data cmplist(1549) /         0/
      data cmplist(1550) /         0/
      data cmplist(1551) /         0/
      data cmplist(1552) /         0/
      data cmplist(1553) /         0/
      data cmplist(1554) /         0/
      data cmplist(1555) /         0/
      data cmplist(1556) /         0/
      data cmplist(1557) /         0/
      data cmplist(1558) /         0/
      data cmplist(1559) /         0/
      data cmplist(1560) /         0/
      data cmplist(1561) /         0/
      data cmplist(1562) /         0/
      data cmplist(1563) /         0/
      data cmplist(1564) /         0/
      data cmplist(1565) /         0/
      data cmplist(1566) /         0/
      data cmplist(1567) /         0/
      data cmplist(1568) /         0/
      data cmplist(1569) /         0/
      data cmplist(1570) /         0/
      data cmplist(1571) /         0/
      data cmplist(1572) /         0/
      data cmplist(1573) /         0/
      data cmplist(1574) /         0/
      data cmplist(1575) /         0/
      data cmplist(1576) /         0/
      data cmplist(1577) /         0/
      data cmplist(1578) /         0/
      data cmplist(1579) /         0/
      data cmplist(1580) /         0/
      data cmplist(1581) /         0/
      data cmplist(1582) /         0/
      data cmplist(1583) /         0/
      data cmplist(1584) /         0/
      data cmplist(1585) /         0/
      data cmplist(1586) /         0/
      data cmplist(1587) /         0/
      data cmplist(1588) /         0/
      data cmplist(1589) /         0/
      data cmplist(1590) /         0/
      data cmplist(1591) /         0/
      data cmplist(1592) /         0/
      data cmplist(1593) /         0/
      data cmplist(1594) /         0/
      data cmplist(1595) /         0/
      data cmplist(1596) /         0/
      data cmplist(1597) /         0/
      data cmplist(1598) /         0/
      data cmplist(1599) /         0/
      data cmplist(1600) /         0/
      data cmplist(1601) /         0/
      data cmplist(1602) /         0/
      data cmplist(1603) /         0/
      data cmplist(1604) /         0/
      data cmplist(1605) /         0/
      data cmplist(1606) /         0/
      data cmplist(1607) /         0/
      data cmplist(1608) /         0/
      data cmplist(1609) /         0/
      data cmplist(1610) /         0/
      data cmplist(1611) /         0/
      data cmplist(1612) /         0/
      data cmplist(1613) /         0/
      data cmplist(1614) /         0/
      data cmplist(1615) /         0/
      data cmplist(1616) /         0/
      data cmplist(1617) /         0/
      data cmplist(1618) /         0/
      data cmplist(1619) /         0/
      data cmplist(1620) /         0/
      data cmplist(1621) /         0/
      data cmplist(1622) /         0/
      data cmplist(1623) /         0/
      data cmplist(1624) /         0/
      data cmplist(1625) /         0/
      data cmplist(1626) /         0/
      data cmplist(1627) /         0/
      data cmplist(1628) /         0/
      data cmplist(1629) /         0/
      data cmplist(1630) /         0/
      data cmplist(1631) /         0/
      data cmplist(1632) /         0/
      data cmplist(1633) /         0/
      data cmplist(1634) /         0/
      data cmplist(1635) /         0/
      data cmplist(1636) /         0/
      data cmplist(1637) /         0/
      data cmplist(1638) /         0/
      data cmplist(1639) /         0/
      data cmplist(1640) /         0/
      data cmplist(1641) /         0/
      data cmplist(1642) /         0/
      data cmplist(1643) /         0/
      data cmplist(1644) /         0/
      data cmplist(1645) /         0/
      data cmplist(1646) /         0/
      data cmplist(1647) /         0/
      data cmplist(1648) /         0/
      data cmplist(1649) /         0/
      data cmplist(1650) /         0/
      data cmplist(1651) /         0/
      data cmplist(1652) /         0/
      data cmplist(1653) /         0/
      data cmplist(1654) /         0/
      data cmplist(1655) /         0/
      data cmplist(1656) /         0/
      data cmplist(1657) /         0/
      data cmplist(1658) /         0/
      data cmplist(1659) /         0/
      data cmplist(1660) /         0/
      data cmplist(1661) /         0/
      data cmplist(1662) /         0/
      data cmplist(1663) /         0/
      data cmplist(1664) /         0/
      data cmplist(1665) /         0/
      data cmplist(1666) /         0/
      data cmplist(1667) /         0/
      data cmplist(1668) /         0/
      data cmplist(1669) /         0/
      data cmplist(1670) /         0/
      data cmplist(1671) /         0/
      data cmplist(1672) /         0/
      data cmplist(1673) /         0/
      data cmplist(1674) /         0/
      data cmplist(1675) /         0/
      data cmplist(1676) /         0/
      data cmplist(1677) /         0/
      data cmplist(1678) /         0/
      data cmplist(1679) /         0/
      data cmplist(1680) /         0/
      data cmplist(1681) /         0/
      data cmplist(1682) /         0/
      data cmplist(1683) /         0/
      data cmplist(1684) /         0/
      data cmplist(1685) /         0/
      data cmplist(1686) /         0/
      data cmplist(1687) /         0/
      data cmplist(1688) /         0/
      data cmplist(1689) /         0/
      data cmplist(1690) /         0/
      data cmplist(1691) /         0/
      data cmplist(1692) /         0/
      data cmplist(1693) /         0/
      data cmplist(1694) /         0/
      data cmplist(1695) /         0/
      data cmplist(1696) /         0/
      data cmplist(1697) /         0/
      data cmplist(1698) /         0/
      data cmplist(1699) /         0/
      data cmplist(1700) /         0/
      data cmplist(1701) /         0/
      data cmplist(1702) /         0/
      data cmplist(1703) /         0/
      data cmplist(1704) /         0/
      data cmplist(1705) /         0/
      data cmplist(1706) /         0/
      data cmplist(1707) /         0/
      data cmplist(1708) /         0/
      data cmplist(1709) /         0/
      data cmplist(1710) /         0/
      data cmplist(1711) /         0/
      data cmplist(1712) /         0/
      data cmplist(1713) /         0/
      data cmplist(1714) /         0/
      data cmplist(1715) /         0/
      data cmplist(1716) /         0/
      data cmplist(1717) /         0/
      data cmplist(1718) /         0/
      data cmplist(1719) /         0/
      data cmplist(1720) /         0/
      data cmplist(1721) /         0/
      data cmplist(1722) /         0/
      data cmplist(1723) /         0/
      data cmplist(1724) /         0/
      data cmplist(1725) /         0/
      data cmplist(1726) /         0/
      data cmplist(1727) /         0/
      data cmplist(1728) /         0/
      data cmplist(1729) /         0/
      data cmplist(1730) /         0/
      data cmplist(1731) /         0/
      data cmplist(1732) /         0/
      data cmplist(1733) /         0/
      data cmplist(1734) /         0/
      data cmplist(1735) /         0/
      data cmplist(1736) /         0/
      data cmplist(1737) /         0/
      data cmplist(1738) /         0/
      data cmplist(1739) /         0/
      data cmplist(1740) /         0/
      data cmplist(1741) /         0/
      data cmplist(1742) /         0/
      data cmplist(1743) /         0/
      data cmplist(1744) /         0/
      data cmplist(1745) /         0/
      data cmplist(1746) /         0/
      data cmplist(1747) /         0/
      data cmplist(1748) /         0/
      data cmplist(1749) /         0/
      data cmplist(1750) /         0/
      data cmplist(1751) /         0/
      data cmplist(1752) /         0/
      data cmplist(1753) /         0/
      data cmplist(1754) /         0/
      data cmplist(1755) /         0/
      data cmplist(1756) /         0/
      data cmplist(1757) /         0/
      data cmplist(1758) /         0/
      data cmplist(1759) /         0/
      data cmplist(1760) /         0/
      data cmplist(1761) /         0/
      data cmplist(1762) /         0/
      data cmplist(1763) /         0/
      data cmplist(1764) /         0/
      data cmplist(1765) /         0/
      data cmplist(1766) /         0/
      data cmplist(1767) /         0/
      data cmplist(1768) /         0/
      data cmplist(1769) /         0/
      data cmplist(1770) /         0/
      data cmplist(1771) /         0/
      data cmplist(1772) /         0/
      data cmplist(1773) /         0/
      data cmplist(1774) /         0/
      data cmplist(1775) /         0/
      data cmplist(1776) /         0/
      data cmplist(1777) /         0/
      data cmplist(1778) /         0/
      data cmplist(1779) /         0/
      data cmplist(1780) /         0/
      data cmplist(1781) /         0/
      data cmplist(1782) /         0/
      data cmplist(1783) /         0/
      data cmplist(1784) /         0/
      data cmplist(1785) /         0/
      data cmplist(1786) /         0/
      data cmplist(1787) /         0/
      data cmplist(1788) /         0/
      data cmplist(1789) /         0/
      data cmplist(1790) /         0/
      data cmplist(1791) /         0/
      data cmplist(1792) /         0/
      data cmplist(1793) /         0/
      data cmplist(1794) /         0/
      data cmplist(1795) /         0/
      data cmplist(1796) /         0/
      data cmplist(1797) /         0/
      data cmplist(1798) /         0/
      data cmplist(1799) /         0/
      data cmplist(1800) /         0/
      data cmplist(1801) /         0/
      data cmplist(1802) /         0/
      data cmplist(1803) /         0/
      data cmplist(1804) /         0/
      data cmplist(1805) /         0/
      data cmplist(1806) /         0/
      data cmplist(1807) /         0/
      data cmplist(1808) /         0/
      data cmplist(1809) /         0/
      data cmplist(1810) /         0/
      data cmplist(1811) /         0/
      data cmplist(1812) /         0/
      data cmplist(1813) /         0/
      data cmplist(1814) /         0/
      data cmplist(1815) /         0/
      data cmplist(1816) /         0/
      data cmplist(1817) /         0/
      data cmplist(1818) /         0/
      data cmplist(1819) /         0/
      data cmplist(1820) /         0/
      data cmplist(1821) /         0/
      data cmplist(1822) /         0/
      data cmplist(1823) /         0/
      data cmplist(1824) /         0/
      data cmplist(1825) /         0/
      data cmplist(1826) /         0/
      data cmplist(1827) /         0/
      data cmplist(1828) /         0/
      data cmplist(1829) /         0/
      data cmplist(1830) /         0/
      data cmplist(1831) /         0/
      data cmplist(1832) /         0/
      data cmplist(1833) /         0/
      data cmplist(1834) /         0/
      data cmplist(1835) /         0/
      data cmplist(1836) /         0/
      data cmplist(1837) /         0/
      data cmplist(1838) /         0/
      data cmplist(1839) /         0/
      data cmplist(1840) /         0/
      data cmplist(1841) /         0/
      data cmplist(1842) /         0/
      data cmplist(1843) /         0/
      data cmplist(1844) /         0/
      data cmplist(1845) /         0/
      data cmplist(1846) /         0/
      data cmplist(1847) /         0/
      data cmplist(1848) /         0/
      data cmplist(1849) /         0/
      data cmplist(1850) /         0/
      data cmplist(1851) /         0/
      data cmplist(1852) /         0/
      data cmplist(1853) /         0/
      data cmplist(1854) /         0/
      data cmplist(1855) /         0/
      data cmplist(1856) /         0/
      data cmplist(1857) /         0/
      data cmplist(1858) /         0/
      data cmplist(1859) /         0/
      data cmplist(1860) /         0/
      data cmplist(1861) /         0/
      data cmplist(1862) /         0/
      data cmplist(1863) /         0/
      data cmplist(1864) /         0/
      data cmplist(1865) /         0/
      data cmplist(1866) /         0/
      data cmplist(1867) /         0/
      data cmplist(1868) /         0/
      data cmplist(1869) /         0/
      data cmplist(1870) /         0/
      data cmplist(1871) /         0/
      data cmplist(1872) /         0/
      data cmplist(1873) /         0/
      data cmplist(1874) /         0/
      data cmplist(1875) /         0/
      data cmplist(1876) /         0/
      data cmplist(1877) /         0/
      data cmplist(1878) /         0/
      data cmplist(1879) /         0/
      data cmplist(1880) /         0/
      data cmplist(1881) /         0/
      data cmplist(1882) /         0/
      data cmplist(1883) /         0/
      data cmplist(1884) /         0/
      data cmplist(1885) /         0/
      data cmplist(1886) /         0/
      data cmplist(1887) /         0/
      data cmplist(1888) /         0/
      data cmplist(1889) /         0/
      data cmplist(1890) /         0/
      data cmplist(1891) /         0/
      data cmplist(1892) /         0/
      data cmplist(1893) /         0/
      data cmplist(1894) /         0/
      data cmplist(1895) /         0/
      data cmplist(1896) /         0/
      data cmplist(1897) /         0/
      data cmplist(1898) /         0/
      data cmplist(1899) /         0/
      data cmplist(1900) /         0/
      data cmplist(1901) /         0/
      data cmplist(1902) /         0/
      data cmplist(1903) /         0/
      data cmplist(1904) /         0/
      data cmplist(1905) /         0/
      data cmplist(1906) /         0/
      data cmplist(1907) /         0/
      data cmplist(1908) /         0/
      data cmplist(1909) /         0/
      data cmplist(1910) /         0/
      data cmplist(1911) /         0/
      data cmplist(1912) /         0/
      data cmplist(1913) /         0/
      data cmplist(1914) /         0/
      data cmplist(1915) /         0/
      data cmplist(1916) /         0/
      data cmplist(1917) /         0/
      data cmplist(1918) /         0/
      data cmplist(1919) /         0/
      data cmplist(1920) /         0/
      data cmplist(1921) /         0/
      data cmplist(1922) /         0/
      data cmplist(1923) /         0/
      data cmplist(1924) /         0/
      data cmplist(1925) /         0/
      data cmplist(1926) /         0/
      data cmplist(1927) /         0/
      data cmplist(1928) /         0/
      data cmplist(1929) /         0/
      data cmplist(1930) /         0/
      data cmplist(1931) /         0/
      data cmplist(1932) /         0/
      data cmplist(1933) /         0/
      data cmplist(1934) /         0/
      data cmplist(1935) /         0/
      data cmplist(1936) /         0/
      data cmplist(1937) /         0/
      data cmplist(1938) /         0/
      data cmplist(1939) /         0/
      data cmplist(1940) /         0/
      data cmplist(1941) /         0/
      data cmplist(1942) /         0/
      data cmplist(1943) /         0/
      data cmplist(1944) /         0/
      data cmplist(1945) /         0/
      data cmplist(1946) /         0/
      data cmplist(1947) /         0/
      data cmplist(1948) /         0/
      data cmplist(1949) /         0/
      data cmplist(1950) /         0/
      data cmplist(1951) /         0/
      data cmplist(1952) /         0/
      data cmplist(1953) /         0/
      data cmplist(1954) /         0/
      data cmplist(1955) /         0/
      data cmplist(1956) /         0/
      data cmplist(1957) /         0/
      data cmplist(1958) /         0/
      data cmplist(1959) /         0/
      data cmplist(1960) /         0/
      data cmplist(1961) /         0/
      data cmplist(1962) /         0/
      data cmplist(1963) /         0/
      data cmplist(1964) /         0/
      data cmplist(1965) /         0/
      data cmplist(1966) /         0/
      data cmplist(1967) /         0/
      data cmplist(1968) /         0/
      data cmplist(1969) /         0/
      data cmplist(1970) /         0/
      data cmplist(1971) /         0/
      data cmplist(1972) /         0/
      data cmplist(1973) /         0/
      data cmplist(1974) /         0/
      data cmplist(1975) /         0/
      data cmplist(1976) /         0/
      data cmplist(1977) /         0/
      data cmplist(1978) /         0/
      data cmplist(1979) /         0/
      data cmplist(1980) /         0/
      data cmplist(1981) /         0/
      data cmplist(1982) /         0/
      data cmplist(1983) /         0/
      data cmplist(1984) /         0/
      data cmplist(1985) /         0/
      data cmplist(1986) /         0/
      data cmplist(1987) /         0/
      data cmplist(1988) /         0/
      data cmplist(1989) /         0/
      data cmplist(1990) /         0/
      data cmplist(1991) /         0/
      data cmplist(1992) /         0/
      data cmplist(1993) /         0/
      data cmplist(1994) /         0/
      data cmplist(1995) /         0/
      data cmplist(1996) /         0/
      data cmplist(1997) /         0/
      data cmplist(1998) /         0/
      data cmplist(1999) /         0/
      data cmplist(2000) /         0/
      data cmplist(2001) /         0/
      data cmplist(2002) /         0/
      data cmplist(2003) /         0/
      data cmplist(2004) /         0/
      data cmplist(2005) /         0/
      data cmplist(2006) /         0/
      data cmplist(2007) /         0/
      data cmplist(2008) /         0/
      data cmplist(2009) /         0/
      data cmplist(2010) /         0/
      data cmplist(2011) /         0/
      data cmplist(2012) /         0/
      data cmplist(2013) /         0/
      data cmplist(2014) /         0/
      data cmplist(2015) /         0/
      data cmplist(2016) /         0/
      data cmplist(2017) /         0/
      data cmplist(2018) /         0/
      data cmplist(2019) /         0/
      data cmplist(2020) /         0/
      data cmplist(2021) /         0/
      data cmplist(2022) /         0/
      data cmplist(2023) /         0/
      data cmplist(2024) /         0/
      data cmplist(2025) /         0/
      data cmplist(2026) /         0/
      data cmplist(2027) /         0/
      data cmplist(2028) /         0/
      data cmplist(2029) /         0/
      data cmplist(2030) /         0/
      data cmplist(2031) /         0/
      data cmplist(2032) /         0/
      data cmplist(2033) /         0/
      data cmplist(2034) /         0/
      data cmplist(2035) /         0/
      data cmplist(2036) /         0/
      data cmplist(2037) /         0/
      data cmplist(2038) /         0/
      data cmplist(2039) /         0/
      data cmplist(2040) /         0/
      data cmplist(2041) /         0/
      data cmplist(2042) /         0/
      data cmplist(2043) /         0/
      data cmplist(2044) /         0/
      data cmplist(2045) /         0/
      data cmplist(2046) /         0/
      data cmplist(2047) /         0/
      data cmplist(2048) /         0/
      data cmplist(2049) /         0/
      data cmplist(2050) /         0/
      data cmplist(2051) /         0/
      data cmplist(2052) /         0/
      data cmplist(2053) /         0/
      data cmplist(2054) /         0/
      data cmplist(2055) /         0/
      data cmplist(2056) /         0/
      data cmplist(2057) /         0/
      data cmplist(2058) /         0/
      data cmplist(2059) /         0/
      data cmplist(2060) /         0/
      data cmplist(2061) /         0/
      data cmplist(2062) /         0/
      data cmplist(2063) /         0/
      data cmplist(2064) /         0/
      data cmplist(2065) /         0/
      data cmplist(2066) /         0/
      data cmplist(2067) /         0/
      data cmplist(2068) /         0/
      data cmplist(2069) /         0/
      data cmplist(2070) /         0/
      data cmplist(2071) /         0/
      data cmplist(2072) /         0/
      data cmplist(2073) /         0/
      data cmplist(2074) /         0/
      data cmplist(2075) /         0/
      data cmplist(2076) /         0/
      data cmplist(2077) /         0/
      data cmplist(2078) /         0/
      data cmplist(2079) /         0/
      data cmplist(2080) /         0/
      data cmplist(2081) /         0/
      data cmplist(2082) /         0/
      data cmplist(2083) /         0/
      data cmplist(2084) /         0/
      data cmplist(2085) /         0/
      data cmplist(2086) /         0/
      data cmplist(2087) /         0/
      data cmplist(2088) /         0/
      data cmplist(2089) /         0/
      data cmplist(2090) /         0/
      data cmplist(2091) /         0/
      data cmplist(2092) /         0/
      data cmplist(2093) /         0/
      data cmplist(2094) /         0/
      data cmplist(2095) /         0/
      data cmplist(2096) /         0/
      data cmplist(2097) /         0/
      data cmplist(2098) /         0/
      data cmplist(2099) /         0/
      data cmplist(2100) /         0/
      data cmplist(2101) /         0/
      data cmplist(2102) /         0/
      data cmplist(2103) /         0/
      data cmplist(2104) /         0/
      data cmplist(2105) /         0/
      data cmplist(2106) /         0/
      data cmplist(2107) /         0/
      data cmplist(2108) /         0/
      data cmplist(2109) /         0/
      data cmplist(2110) /         0/
      data cmplist(2111) /         0/
      data cmplist(2112) /         0/
      data cmplist(2113) /         0/
      data cmplist(2114) /         0/
      data cmplist(2115) /         0/
      data cmplist(2116) /         0/
      data cmplist(2117) /         0/
      data cmplist(2118) /         0/
      data cmplist(2119) /         0/
      data cmplist(2120) /         0/
      data cmplist(2121) /         0/
      data cmplist(2122) /         0/
      data cmplist(2123) /         0/
      data cmplist(2124) /         0/
      data cmplist(2125) /         0/
      data cmplist(2126) /         0/
      data cmplist(2127) /         0/
      data cmplist(2128) /         0/
      data cmplist(2129) /         0/
      data cmplist(2130) /         0/
      data cmplist(2131) /         0/
      data cmplist(2132) /         0/
      data cmplist(2133) /         0/
      data cmplist(2134) /         0/
      data cmplist(2135) /         0/
      data cmplist(2136) /         0/
      data cmplist(2137) /         0/
      data cmplist(2138) /         0/
      data cmplist(2139) /         0/
      data cmplist(2140) /         0/
      data cmplist(2141) /         0/
      data cmplist(2142) /         0/
      data cmplist(2143) /         0/
      data cmplist(2144) /         0/
      data cmplist(2145) /         0/
      data cmplist(2146) /         0/
      data cmplist(2147) /         0/
      data cmplist(2148) /         0/
      data cmplist(2149) /         0/
      data cmplist(2150) /         0/
      data cmplist(2151) /         0/
      data cmplist(2152) /         0/
      data cmplist(2153) /         0/
      data cmplist(2154) /         0/
      data cmplist(2155) /         0/
      data cmplist(2156) /         0/
      data cmplist(2157) /         0/
      data cmplist(2158) /         0/
      data cmplist(2159) /         0/
      data cmplist(2160) /         0/
      data cmplist(2161) /         0/
      data cmplist(2162) /         0/
      data cmplist(2163) /         0/
      data cmplist(2164) /         0/
      data cmplist(2165) /         0/
      data cmplist(2166) /         0/
      data cmplist(2167) /         0/
      data cmplist(2168) /         0/
      data cmplist(2169) /         0/
      data cmplist(2170) /         0/
      data cmplist(2171) /         0/
      data cmplist(2172) /         0/
      data cmplist(2173) /         0/
      data cmplist(2174) /         0/
      data cmplist(2175) /         0/
      data cmplist(2176) /         0/
      data cmplist(2177) /         0/
      data cmplist(2178) /         0/
      data cmplist(2179) /         0/
      data cmplist(2180) /         0/
      data cmplist(2181) /         0/
      data cmplist(2182) /         0/
      data cmplist(2183) /         0/
      data cmplist(2184) /         0/
      data cmplist(2185) /         0/
      data cmplist(2186) /         0/
      data cmplist(2187) /         0/
      data cmplist(2188) /         0/
      data cmplist(2189) /         0/
      data cmplist(2190) /         0/
      data cmplist(2191) /         0/
      data cmplist(2192) /         0/
      data cmplist(2193) /         0/
      data cmplist(2194) /         0/
      data cmplist(2195) /         0/
      data cmplist(2196) /         0/
      data cmplist(2197) /         0/
      data cmplist(2198) /         0/
      data cmplist(2199) /         0/
      data cmplist(2200) /         0/
      data cmplist(2201) /1001001002/
      data cmplist(2202) /1002001002/
      data cmplist(2203) /1003001002/
      data cmplist(2204) /1003002001/
      data cmplist(2205) /1004002001/
      data cmplist(2206) /         0/
      data cmplist(2207) /         0/
      data cmplist(2208) /         0/
      data cmplist(2209) /         0/
      data cmplist(2210) /         0/

      save cmplist

C...Subdivide standard ID code into constituent pieces.
      hepcmp=0
      kqa=iabs(id)
      kqn=mod(kqa/1000000000,10)
      kqna=mod(kqa/1000000,1000)
      kqnz=mod(kqa/1000,1000)
      kqx=mod(kqa/1000000,10)
      kqr=mod(kqa/100000,10)
      kql=mod(kqa/10000,10)
      kq3=mod(kqa/1000,10)
      kq2=mod(kqa/100,10)
      kq1=mod(kqa/10,10)
      kqj=mod(kqa,10)
      kshort=mod(kqa,100)
      kqsm=(kqj-1)/2
      irt = mod(kqa,10000)
      k99=mod(kqa/100000,100)
      is = kqsm
      if(kqx.eq.9) is = kqsm + 5

      if(kqa.eq.0)  then
      elseif(kqa.ge.10000000) then
C... ions 
	do i=2201,2210
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
        if(id.lt.0) hepcmp=0
      elseif(kqa.le.100) then
C... need to check antiparticle legality
        hepcmp=kqa
        ich=hepchg(id)
        if(id.lt.0.and.ich.eq.0)then
          if(kqa.ge.1.and.kqa.le.8)then
C... quarks
          elseif(kqa.ge.11.and.kqa.le.18)then
C... leptons
          elseif(kqa.eq.41 .or. kqa.eq.42)then
C... R0 or LQ
          elseif(kqa.ge.81)then
C... internal definitions
          else
            hepcmp=0
          endif
        endif
      elseif(kqj.eq.0) then
	do i=101,110
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
        if(id.lt.0) hepcmp=0
      elseif(kshort.eq.22 .and. kql.gt.0 .and. kq2.eq.0) then
C virtual gamma and Cerenkov
	do i=241,260
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
        if(id.lt.0) hepcmp=0
      elseif(kqx.eq.1 .or. kqx.eq.2)then
C...SUSY
	do i=111,190
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
	if(hepcmp.gt.0)then
          ich=hepchg(id)
          if(id.lt.0.and.ich.eq.0)then
            if(irt.ge.1.and.irt.le.8)then
C... quarks
            elseif(irt.ge.11.and.irt.le.18)then
C... leptons
            else
              hepcmp=0
            endif
          endif
        else
C... now what about R-hadrons?
C... 2211 - 6791 [entire meson space]
        if(kqx.eq.1 .and. kqr.eq.0 .and. kqj.gt.0 
     1	   .and. kq1.gt.0 .and. kq2.gt.0 )then
	  if( kql.gt.0 .and. (kq2.eq.0 .or. kq3.eq.0))then
	  elseif(kq1.eq.7 .or. kq1.eq.8)then
	  elseif(kq2.eq.7 .or. kq2.eq.8)then
	  elseif(kq3.eq.7 .or. kq3.eq.8)then
	  elseif(kql.eq.7 .or. kql.eq.8)then
	  elseif(kq1.gt.kq2)then
	  elseif(kq3.gt.0 .and. kq1.gt.kq3)then
	  elseif(kq3.gt.0 .and. kq2.gt.kq3)then
	  elseif(kql.gt.0 .and. kq1.gt.kql)then
	  elseif(kql.gt.0 .and. kq2.gt.kql)then
	  elseif(kql.gt.0 .and. kq3.gt.kql)then
	  else
            ich=hepchg(id)
C... must have integer charge
	    if(ich.eq.6 .or. ich.eq.-6 .or. ich.eq.3 .or. ich.eq.-3
     1         .or. ich.eq.0) then
              hepcmp=2210 + kql*250 + kq3*170 + (kq2-1)*90
     1			+ (kq1-1)*9 + kqj
	    endif
	  endif
	endif
	endif
      elseif(kqx.ge.3 .and. kqx.le.4)then
	do i=191,240
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
        if(kq3.eq.0 .and. id.lt.0) then
          if(kq1.gt.0 .and. kq2.eq.kq1) hepcmp=0
	endif
      elseif(k99.eq.99)then
	do i=231,260
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
        if(kq3.eq.0 .and. id.lt.0) then
          if(kq1.gt.0 .and. kq2.eq.kq1) hepcmp=0
	endif
	if(id.lt.0 .and. kshort.eq.22)  hepcmp=0
      elseif(kqx.eq.9 .and. kqr.ge.1)then
C...possible pentaquark
	do i=231,240
            if(cmplist(i).eq.kqa) hepcmp = i
	enddo
      elseif(kqx.gt.4 .and. kqx.lt.9)then
C...undefined

      elseif(kq1.eq.0) then
C...Diquarks.
C... 103 
        if(kqx.ne.0 .and. kqr.ne.0 .and. kql.ne.0)then
C... no excited states
        elseif(kqj.ne.1.and.kqj.ne.3) then
C... only 2 spin states
        elseif(kq3.eq.0.or.kq3.eq.9.or.kq2.eq.0.or.kq2.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq3.lt.kq2) then
C... illegal quark order
        elseif(kq3.eq.kq2 .and. kqj.eq.1) then
C... illegal spin
        else
C... all diquarks are the same...
          hepcmp=103
        endif

      elseif(kq3.eq.0) then
C...Mesons.
C... 260 - ...
        if(kq2.eq.0.or.kq2.eq.9.or.kq1.eq.0.or.kq1.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq2.lt.kq1) then
C... illegal quark order
        elseif(id.lt.0.and.(kq2.eq.kq1)) then
C... illegal antiparticle
        elseif(mod(kqj,2).eq.0)then
C... illegal spin
        elseif(kq2.le.2 .and. kq1.eq.1) then
C... 260 - 559 [ kqr = 0,1,2  kql = 0-4 ]
	  hepcmp = 260 + (kq2-1)*150 + kqr*50 + kql*10 + is
	  if(kql.gt.4 .or. kqr.gt.2) hepcmp = 0
	elseif(kq2.eq.2 .and. kq1.eq.2) then
C... 560 - 799 [ kqr = 0,1,2  kql = 0-9 ]
	  hepcmp = 560 + kqr*100 + kql*10 + is
	  if(kqr.gt.2) hepcmp = 0
	elseif(kq2.eq.3 .and. kq1.eq.3) then
C... 800 - 859 [ kqr = 0,1  kql = 0-2 ]
	  hepcmp = 860 + kqr*30 + kql*10 + is
	  if(kql.gt.2 .or. kqr.gt.1) hepcmp = 0
	elseif(kq2.eq.3 .and. kq1.le.2) then
C... 860 - 1099 [ kqr = 0,1,2  kql = 0-3 ]
	  hepcmp = 920 + (kq1-1)*120 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.2) hepcmp = 0
	elseif(kq2.eq.4 .and. kq1.le.3) then
C... 1100 - 1189 [ kqr = 0  kql = 0-2 ]
	  hepcmp = 1160 + kqr*90 + (kq1-1)*30 + kql*10 + is
cc	  if(kql.gt.2 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp = 0
	  if(kql.gt.2 .or. kqr.gt.2 .or. kqx.gt.0) hepcmp = 0
	elseif(kq2.eq.4 .and. kq1.eq.4) then
C... 1190 - 1269 [ kqr = 0,1  kql = 0-3 ]
	  hepcmp = 1430 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.1) hepcmp = 0
	elseif(kq2.eq.5 .and. kq1.le.4) then
C... 1270 - 1389 [ kqr = 0  kql = 0-2 ]
	  hepcmp = 1510 + (kq1-1)*30 + kql*10 + is
	  if(kql.gt.2 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp = 0
	elseif(kq2.eq.5 .and. kq1.eq.5) then
C... 1390 - 1549 [ kqr = 0,1,2,3  kql = 0-3 ]
	  hepcmp = 1630 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.3) hepcmp = 0
        elseif(kq2.eq.6)then
C... 1550 - 1609 [ kqr = 0  kql = 0 ]
	  hepcmp = 1790 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp = 0
        elseif(kq2.eq.7)then
C... 1610 - 1679 [ kqr = 0  kql = 0 ]
	  hepcmp = 1850 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp = 0
        elseif(kq2.eq.8)then
C... 1680 - 1759 [ kqr = 0  kql = 0 ]
	  hepcmp = 1920 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp = 0
        endif

      elseif(kqx.eq.0.and.(kqj.eq.2.or.kqj.eq.4)) then
C...Baryons.
C... no excited states and only spins 1/2 and 3/2
        if(kq3.eq.0.or.kq3.eq.9.or.kq2.eq.0.or.kq2.eq.9.or.
     1        kq1.eq.0.or.kq1.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq3.lt.kq1.or.kq3.lt.kq2) then
C... illegal quark order
        elseif(kq3.ge.7) then
C... l and h baryons are all the same
C... 104, 105
          hepcmp=97+kq3
          if(kq3.eq.kq1.and.kq3.eq.kq2.and.kqj.eq.2) hepcmp=0
          if(kq2.lt.kq1 .and. kq3.eq.kq1) hepcmp=0
          if(kq2.lt.kq1 .and. kqj.eq.4) hepcmp=0
          if(kq3.gt.8) hepcmp=0
        elseif(kqj.eq.2) then
C...Spin 1/2 baryons.
          if(kq3.eq.kq1 .and. kq3.eq.kq2)then
          elseif(kq2.lt.kq1 .and. kq3.eq.kq1)then
          elseif(kq2.lt.kq1) then
C...antisymmtric (Lambda, etc.)
C... 1760 - 1780
            hepcmp=2000+((kq3-1)*(kq3-2)*(kq3-3))/6+
     1                  ((kq1-1)*(kq1-2))/2+kq2
          else
C...symmtric
C... 1780 - 1791
            hepcmp=2020+((kq3+1)*kq3*(kq3-1))/6+(kq2*(kq2-1))/2+kq1
          endif
        elseif(kqj.eq.4) then
C...Spin 3/2 baryons.
          if(kq2.lt.kq1 .and. kq3.eq.kq1)then
          elseif(kq2.lt.kq1) then
C...no excited states for antisymmetric baryons
          else
C...symmetric
C... 1840 - 1896
            hepcmp=2080+((kq3+1)*kq3*(kq3-1))/6+(kq2*(kq2-1))/2+kq1
          endif
        endif

      endif

      return
      end
