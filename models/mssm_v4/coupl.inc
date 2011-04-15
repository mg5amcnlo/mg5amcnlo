c====================================================================
c
c  Define common block containing all coupling constants and masses.
c
c====================================================================
c
      double complex       gg(2)
      double precision     g
      common /COUPL_BASIC/ g,gg   
c
c      double precision     hmass, wmass, zmass, amass,
      double precision     wmass, zmass, amass,
     &                     tmass, bmass, lmass
      common /COUPL_MASS/  wmass, zmass, amass,
     &                     tmass, bmass, lmass

      double precision     wwidth, zwidth, 
     &                     twidth, lwidth, awidth
      common /COUPL_WIDTH/ wwidth, zwidth, 
     &                     twidth, lwidth, awidth

      double complex       gal(2), gad(2), gau(2), gwf(2),
     &                     gzn(2), gzl(2), gzd(2), gzu(2)
      double precision     gw, gwwa, gwwz
      common /COUPL_GAUGE/ gal   , gad   , gau   , gwf   ,
     &                     gzn   , gzl   , gzd   , gzu   ,
     &                     gw, gwwa, gwwz

c      double complex       gwfc(2),  gwfs(2), gwfm(2)
c      common /coupl_ckm/   gwfc,     gwfs   , gwfm	

c      double complex       gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh
c      common /COUPL_SCAL/  gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh

c      double complex       ghtop(2), ghbot(2), ghtau(2), ghcha(2)
c      common /COUPL_YUK/   ghtop   , ghbot   , ghtau   , ghcha

c      double precision     xzmass, xwmass
c      common /COUPL_XMASS/ xzmass, xwmass

c      double complex       xzl(2) , xzb(2) , xzt(2) ,
c     &                     xwpq(2), xwmq(2), xwpl(2), xwml(2)
c      common /COUPL_XFFV/  xzl    , xzb    , xzt    ,
c     &                     xwpq   , xwmq   , xwpl   , xwml

c      double complex       xzhz, xwhwp, xwhwm
c      common /COUPL_XVSS/  xzhz, xwhwp, xwhwm

c      double complex       xwzwp, xwzwm, xwawp, xwawm
c      common /COUPL_XVVS/  xwzwp, xwzwm, xwawp, xwawm

c      double complex       xwzhwp, xwzhwm, xwahwp, xwahwm
c      common /COUPL_XVVSS/ xwzhwp, xwzhwm, xwahwp, xwahwm

c======================================================================
c
c  Define common block containing MSSM coupling constants and masses.
c
c======================================================================
c

      double precision    mdl, mdr, mul, mur, msl, msr,
     &			  mcl, mcr, mb1, mb2, mt1, mt2,
     &			  mel, mer, mml, mmr, ml1, ml2,
     &			  mve, mvm, mvt, 
     &			  mn1, mn2, mn3, mn4, mx1, mx2,
     &			  mgo,
     &                    mh1, mh2, mh3, mhc
      common /MSSM_MASS/  mdl, mdr, mul, mur, msl, msr,
     &			  mcl, mcr, mb1, mb2, mt1, mt2,
     &			  mel, mer, mml, mmr, ml1, ml2,
     &			  mve, mvm, mvt, 
     &			  mn1, mn2, mn3, mn4, mx1, mx2,
     &			  mgo,
     &                    mh1, mh2, mh3, mhc

      double precision    wdl, wdr, wul, wur, wsl, wsr,
     &			  wcl, wcr, wb1, wb2, wt1, wt2,
     &			  wel, wer, wml, wmr, wl1, wl2,
     &			  wve, wvm, wvt, 
     &			  wn1, wn2, wn3, wn4, wx1, wx2,
     &			  wgo,
     &                    wh1, wh2, wh3, whc
      common /MSSM_WIDTH/ wdl, wdr, wul, wur, wsl, wsr,
     &			  wcl, wcr, wb1, wb2, wt1, wt2,
     &			  wel, wer, wml, wmr, wl1, wl2,
     &			  wve, wvm, wvt, 
     &			  wn1, wn2, wn3, wn4, wx1, wx2,
     &			  wgo,
     &                    wh1, wh2, wh3, whc

      double complex    gc, g2c, ggi(2)
      common /MSSM_QCD/ gc, g2c, ggi

      double complex    gzn11(2), gzn12(2), gzn13(2), gzn14(2), gzn22(2),
     &			gzn23(2), gzn24(2), gzn33(2), gzn34(2), gzn44(2),
     &			gax(2)  , gzx11(2), gzx12(2), gzx22(2),
     &			gwn1x1(2), gwx1n1(2), gwn1x2(2), gwx2n1(2),
     &			gwn2x1(2), gwx1n2(2), gwn2x2(2), gwx2n2(2),
     &			gwn3x1(2), gwx1n3(2), gwn3x2(2), gwx2n3(2),
     &			gwn4x1(2), gwx1n4(2), gwn4x2(2), gwx2n4(2)
      common /MSSM_FFV/ gzn11, gzn12, gzn13, gzn14, gzn22,
     &			gzn23, gzn24, gzn33, gzn34, gzn44,
     &			gax  , gzx11, gzx12, gzx22,
     &			gwn1x1, gwx1n1, gwn1x2, gwx2n1,
     &			gwn2x1, gwx1n2, gwn2x2, gwx2n2,
     &			gwn3x1, gwx1n3, gwn3x2, gwx2n3,
     &			gwn4x1, gwx1n4, gwn4x2, gwx2n4

      double complex	gh1ll(2), gh2ll(2), gh3ll(2),
     &			gh1bb(2), gh2bb(2), gh3bb(2),
     &			gh1tt(2), gh2tt(2), gh3tt(2),
     &			ghmq(2), ghpq(2), ghml(2), ghpl(2)
      common /YUK_FFS/	gh1ll, gh2ll, gh3ll,
     &			gh1bb, gh2bb, gh3bb,
     &			gh1tt, gh2tt, gh3tt,
     &			ghmq, ghpq, ghml, ghpl

      double complex	gqlgom(2), gqlgop(2), gqrgom(2), gqrgop(2),
     &			gb1gom(2), gb1gop(2), gb2gom(2), gb2gop(2),
     &                  gt1gom(2), gt1gop(2), gt2gom(2), gt2gop(2),
     &			gdln1m(2), gdln1p(2), gdln2m(2), gdln2p(2),
     &			gdln3m(2), gdln3p(2), gdln4m(2), gdln4p(2),
     &			gdrn1m(2), gdrn1p(2), gdrn2m(2), gdrn2p(2),
     &			gdrn3m(2), gdrn3p(2), gdrn4m(2), gdrn4p(2),
     &			guln1m(2), guln1p(2), guln2m(2), guln2p(2),
     &			guln3m(2), guln3p(2), guln4m(2), guln4p(2),
     &			gurn1m(2), gurn1p(2), gurn2m(2), gurn2p(2),
     &			gurn3m(2), gurn3p(2), gurn4m(2), gurn4p(2),
     &			gb1n1m(2), gb1n1p(2), gb2n1m(2), gb2n1p(2),
     &			gb1n2m(2), gb1n2p(2), gb2n2m(2), gb2n2p(2),
     &			gb1n3m(2), gb1n3p(2), gb2n3m(2), gb2n3p(2),
     &			gb1n4m(2), gb1n4p(2), gb2n4m(2), gb2n4p(2),
     &			gt1n1m(2), gt1n1p(2), gt2n1m(2), gt2n1p(2),
     &			gt1n2m(2), gt1n2p(2), gt2n2m(2), gt2n2p(2),
     &			gt1n3m(2), gt1n3p(2), gt2n3m(2), gt2n3p(2),
     &			gt1n4m(2), gt1n4p(2), gt2n4m(2), gt2n4p(2),
     &			geln1m(2), geln1p(2), geln2m(2), geln2p(2),
     &			geln3m(2), geln3p(2), geln4m(2), geln4p(2),
     &			gern1m(2), gern1p(2), gern2m(2), gern2p(2),
     &			gern3m(2), gern3p(2), gern4m(2), gern4p(2),
     &			gl1n1m(2), gl1n1p(2), gl1n2m(2), gl1n2p(2),
     &			gl1n3m(2), gl1n3p(2), gl1n4m(2), gl1n4p(2),
     &			gl2n1m(2), gl2n1p(2), gl2n2m(2), gl2n2p(2),
     &			gl2n3m(2), gl2n3p(2), gl2n4m(2), gl2n4p(2),
     &			gsvn1m(2), gsvn1p(2), gsvn2m(2), gsvn2p(2),
     &			gsvn3m(2), gsvn3p(2), gsvn4m(2), gsvn4p(2),
     &			gb1x1m(2), gb1x1p(2), gb1x2m(2), gb1x2p(2),
     &			gb2x1m(2), gb2x1p(2), gb2x2m(2), gb2x2p(2),
     &			gt1x1m(2), gt1x1p(2), gt1x2m(2), gt1x2p(2),
     &			gt2x1m(2), gt2x1p(2), gt2x2m(2), gt2x2p(2),
     &			gdlx1m(2), gdlx1p(2), gdlx2m(2), gdlx2p(2),
     &			gulx1m(2), gulx1p(2), gulx2m(2), gulx2p(2),
     &			gelx1m(2), gelx1p(2), gelx2m(2), gelx2p(2),
     &			gvex1m(2), gvex1p(2), gvex2m(2), gvex2p(2),
     &			gl1x1m(2), gl1x1p(2), gl1x2m(2), gl1x2p(2),
     &			gl2x1m(2), gl2x1p(2), gl2x2m(2), gl2x2p(2),
     &			gvtx1m(2), gvtx1p(2), gvtx2m(2), gvtx2p(2),
     &                  gh1n11(2), gh1n12(2), gh1n13(2), gh1n14(2),
     &                  gh1n21(2), gh1n22(2), gh1n23(2), gh1n24(2),
     &                  gh1n31(2), gh1n32(2), gh1n33(2), gh1n34(2),
     &                  gh1n41(2), gh1n42(2), gh1n43(2), gh1n44(2),
     &                  gh2n11(2), gh2n12(2), gh2n13(2), gh2n14(2),
     &                  gh2n21(2), gh2n22(2), gh2n23(2), gh2n24(2),
     &                  gh2n31(2), gh2n32(2), gh2n33(2), gh2n34(2),
     &                  gh2n41(2), gh2n42(2), gh2n43(2), gh2n44(2),
     &                  gh3n11(2), gh3n12(2), gh3n13(2), gh3n14(2),
     &                  gh3n21(2), gh3n22(2), gh3n23(2), gh3n24(2),
     &                  gh3n31(2), gh3n32(2), gh3n33(2), gh3n34(2),
     &                  gh3n41(2), gh3n42(2), gh3n43(2), gh3n44(2),
     &                  gh1x11(2), gh1x12(2), gh1x13(2), gh1x14(2),
     &                  gh1x21(2), gh1x22(2), gh1x23(2), gh1x24(2),
     &                  gh2x11(2), gh2x12(2), gh2x13(2), gh2x14(2),
     &                  gh2x21(2), gh2x22(2), gh2x23(2), gh2x24(2),
     &                  gh3x11(2), gh3x12(2), gh3x13(2), gh3x14(2),
     &                  gh3x21(2), gh3x22(2), gh3x23(2), gh3x24(2),
     &                  ghn1x1(2), ghn2x1(2), ghn3x1(2), ghn4x1(2),
     &                  ghn1x2(2), ghn2x2(2), ghn3x2(2), ghn4x2(2),
     &                  ghx1n1(2), ghx1n2(2), ghx1n3(2), ghx1n4(2),
     &                  ghx2n1(2), ghx2n2(2), ghx2n3(2), ghx2n4(2)
      common /MSSM_FFS/ gqlgom, gqlgop, gqrgom, gqrgop,
     &                  gb1gom, gb1gop, gb2gom, gb2gop,
     &                  gt1gom, gt1gop, gt2gom, gt2gop,
     &			gdln1m, gdln1p, gdln2m, gdln2p,
     &			gdln3m, gdln3p, gdln4m, gdln4p,
     &			gdrn1m, gdrn1p, gdrn2m, gdrn2p,
     &			gdrn3m, gdrn3p, gdrn4m, gdrn4p,
     &			guln1m, guln1p, guln2m, guln2p,
     &			guln3m, guln3p, guln4m, guln4p,
     &			gurn1m, gurn1p, gurn2m, gurn2p,
     &			gurn3m, gurn3p, gurn4m, gurn4p,
     &			gb1n1m, gb1n1p, gb2n1m, gb2n1p,
     &			gb1n2m, gb1n2p, gb2n2m, gb2n2p,
     &			gb1n3m, gb1n3p, gb2n3m, gb2n3p,
     &			gb1n4m, gb1n4p, gb2n4m, gb2n4p,
     &			gt1n1m, gt1n1p, gt2n1m, gt2n1p,
     &			gt1n2m, gt1n2p, gt2n2m, gt2n2p,
     &			gt1n3m, gt1n3p, gt2n3m, gt2n3p,
     &			gt1n4m, gt1n4p, gt2n4m, gt2n4p,
     &			geln1m, geln1p, geln2m, geln2p,
     &			geln3m, geln3p, geln4m, geln4p,
     &			gern1m, gern1p, gern2m, gern2p,
     &			gern3m, gern3p, gern4m, gern4p,
     &			gl1n1m, gl1n1p, gl1n2m, gl1n2p,
     &			gl1n3m, gl1n3p, gl1n4m, gl1n4p,
     &			gl2n1m, gl2n1p, gl2n2m, gl2n2p,
     &			gl2n3m, gl2n3p, gl2n4m, gl2n4p,
     &			gsvn1m, gsvn1p, gsvn2m, gsvn2p,
     &			gsvn3m, gsvn3p, gsvn4m, gsvn4p,
     &			gb1x1m, gb1x1p, gb1x2m, gb1x2p,
     &			gb2x1m, gb2x1p, gb2x2m, gb2x2p,
     &			gt1x1m, gt1x1p, gt1x2m, gt1x2p,
     &			gt2x1m, gt2x1p, gt2x2m, gt2x2p,
     &			gdlx1m, gdlx1p, gdlx2m, gdlx2p,
     &			gulx1m, gulx1p, gulx2m, gulx2p,
     &			gelx1m, gelx1p, gelx2m, gelx2p,
     &			gvex1m, gvex1p, gvex2m, gvex2p,
     &			gl1x1m, gl1x1p, gl1x2m, gl1x2p,
     &			gl2x1m, gl2x1p, gl2x2m, gl2x2p,
     &			gvtx1m, gvtx1p, gvtx2m, gvtx2p,
     &                  gh1n11, gh1n12, gh1n13, gh1n14,
     &                  gh1n21, gh1n22, gh1n23, gh1n24,
     &                  gh1n31, gh1n32, gh1n33, gh1n34,
     &                  gh1n41, gh1n42, gh1n43, gh1n44,
     &                  gh2n11, gh2n12, gh2n13, gh2n14,
     &                  gh2n21, gh2n22, gh2n23, gh2n24,
     &                  gh2n31, gh2n32, gh2n33, gh2n34,
     &                  gh2n41, gh2n42, gh2n43, gh2n44,
     &                  gh3n11, gh3n12, gh3n13, gh3n14,
     &                  gh3n21, gh3n22, gh3n23, gh3n24,
     &                  gh3n31, gh3n32, gh3n33, gh3n34,
     &                  gh3n41, gh3n42, gh3n43, gh3n44,
     &                  gh1x11, gh1x12, gh1x13, gh1x14,
     &                  gh1x21, gh1x22, gh1x23, gh1x24,
     &                  gh2x11, gh2x12, gh2x13, gh2x14,
     &                  gh2x21, gh2x22, gh2x23, gh2x24,
     &                  gh3x11, gh3x12, gh3x13, gh3x14,
     &                  gh3x21, gh3x22, gh3x23, gh3x24,
     &                  ghn1x1, ghn2x1, ghn3x1, ghn4x1,
     &                  ghn1x2, ghn2x2, ghn3x2, ghn4x2,
     &                  ghx1n1, ghx1n2, ghx1n3, ghx1n4,
     &                  ghx2n1, ghx2n2, ghx2n3, ghx2n4

      double complex	gwwh1, gwwh2, gzzh1, gzzh2
      common /MSSM_VVS/	gwwh1, gwwh2, gzzh1, gzzh2

      double complex    gzh1h3, gzh2h3, gahchc, gzhchc, 
     &                  gwh1hc, gwhch1, gwh2hc, gwhch2,
     &                  gwh3hc, gwhch3,
     &			gadldl, gaulul,
     &                  gzdldl, gzulul, gzdrdr, gzurur,
     &                  gzb1b1, gzb1b2, gzb2b1, gzb2b2,
     &                  gzt1t1, gzt1t2, gzt2t1, gzt2t2,
     &			gwqlql,
     &                  gwb1t1, gwb1t2, gwb2t1, gwb2t2,
     &                  gwt1b1, gwt2b1, gwt1b2, gwt2b2,
     &			gaelel, gzelel, gzerer,
     &                  gzl1l1, gzl1l2, gzl2l1, gzl2l2,
     &			gzsvsv, gwelve, gwl1vt, gwl2vt
      common /MSSM_VSS/ gzh1h3, gzh2h3, gahchc, gzhchc,
     &                  gwh1hc, gwhch1, gwh2hc, gwhch2,
     &                  gwh3hc, gwhch3,
     &			gadldl, gaulul,
     &                  gzdldl, gzulul, gzdrdr, gzurur,
     &                  gzb1b1, gzb1b2, gzb2b1, gzb2b2,
     &                  gzt1t1, gzt1t2, gzt2t1, gzt2t2,
     &			gwqlql,
     &                  gwb1t1, gwb1t2, gwb2t1, gwb2t2,
     &                  gwt1b1, gwt2b1, gwt1b2, gwt2b2,
     &			gaelel, gzelel, gzerer,
     &                  gzl1l1, gzl1l2, gzl2l1, gzl2l2,
     &			gzsvsv, gwelve, gwl1vt, gwl2vt

      double complex     gaadldl, gaaulul, gaaelel,
     &                   gazdldl, gazdrdr, gazulul, gazurur,
     &                   gazelel, gazerer,
     &                   gazb1b1, gazb1b2, gazb2b1, gazb2b2,
     &                   gazt1t1, gazt1t2, gazt2t1, gazt2t2,
     &                   gazl1l1, gazl1l2, gazl2l1, gazl2l2,
     &                   gzzdldl, gzzdrdr, gzzulul, gzzurur,
     &                   gzzb1b1, gzzb1b2, gzzb2b1, gzzb2b2,
     &                   gzzt1t1, gzzt1t2, gzzt2t1, gzzt2t2,
     &                   gzzelel, gzzerer, gzzveve,
     &                   gzzl1l1, gzzl1l2, gzzl2l1, gzzl2l2,
     &                   gwauldl, gwadlul,
     &                   gwab1t1, gwab1t2, gwab2t1, gwab2t2,
     &                   gwat1b1, gwat1b2, gwat2b1, gwat2b2,
     &                   gwaveel, gwaelve,
     &                   gwavtl1, gwal1vt, gwavtl2, gwal2vt,
     &                   gwzuldl, gwzdlul,
     &                   gwzb1t1, gwzb1t2, gwzb2t1, gwzb2t2,
     &                   gwzt1b1, gwzt1b2, gwzt2b1, gwzt2b2,
     &                   gwzveel, gwzelve,
     &                   gwzvtl1, gwzl1vt, gwzvtl2, gwzl2vt,
     &                   gwwflfl,
     &                   gwwb1b1, gwwb1b2, gwwb2b1, gwwb2b2,
     &                   gwwt1t1, gwwt1t2, gwwt2t1, gwwt2t2,
     &                   gwwl1l1, gwwl1l2, gwwl2l1, gwwl2l2,
     &                   gwwh1h1, gwwh2h2, gwwh3h3,
     &                   gzzh1h1, gzzh2h2, gzzh3h3,
     &                   gaahchc, gazhchc, gzzhchc, gwwhchc,
     &                   gwah1hc, gwah2hc, gwah3hc,
     &                   gwahch1, gwahch2, gwahch3,
     &                   gwzh1hc, gwzh2hc, gwzh3hc,
     &                   gwzhch1, gwzhch2, gwzhch3,
     &                   ggadldl, ggaulul,
     &                   ggzdldl, ggzdrdr, ggzulul, ggzurur,
     &                   ggzb1b1, ggzb1b2, ggzb2b1, ggzb2b2,
     &                   ggzt1t1, ggzt1t2, ggzt2t1, ggzt2t2,
     &                   ggwuldl, ggwdlul,
     &                   ggwt1b1, ggwb1t1, ggwt1b2, ggwb2t1,
     &                   ggwt2b1, ggwb1t2, ggwt2b2, ggwb2t2
      common /MSSM_VVSS/ gaadldl, gaaulul, gaaelel,
     &                   gazdldl, gazdrdr, gazulul, gazurur,
     &                   gazelel, gazerer,
     &                   gazb1b1, gazb1b2, gazb2b1, gazb2b2,
     &                   gazt1t1, gazt1t2, gazt2t1, gazt2t2,
     &                   gazl1l1, gazl1l2, gazl2l1, gazl2l2,
     &                   gzzdldl, gzzdrdr, gzzulul, gzzurur,
     &                   gzzb1b1, gzzb1b2, gzzb2b1, gzzb2b2,
     &                   gzzt1t1, gzzt1t2, gzzt2t1, gzzt2t2,
     &                   gzzelel, gzzerer, gzzveve,
     &                   gzzl1l1, gzzl1l2, gzzl2l1, gzzl2l2,
     &                   gwauldl, gwadlul,
     &                   gwab1t1, gwab1t2, gwab2t1, gwab2t2,
     &                   gwat1b1, gwat1b2, gwat2b1, gwat2b2,
     &                   gwaveel, gwaelve,
     &                   gwavtl1, gwal1vt, gwavtl2, gwal2vt,
     &                   gwzuldl, gwzdlul,
     &                   gwzb1t1, gwzb1t2, gwzb2t1, gwzb2t2,
     &                   gwzt1b1, gwzt1b2, gwzt2b1, gwzt2b2,
     &                   gwzveel, gwzelve,
     &                   gwzvtl1, gwzl1vt, gwzvtl2, gwzl2vt,
     &                   gwwflfl,
     &                   gwwb1b1, gwwb1b2, gwwb2b1, gwwb2b2,
     &                   gwwt1t1, gwwt1t2, gwwt2t1, gwwt2t2,
     &                   gwwl1l1, gwwl1l2, gwwl2l1, gwwl2l2,
     &                   gwwh1h1, gwwh2h2, gwwh3h3,
     &                   gzzh1h1, gzzh2h2, gzzh3h3,
     &                   gaahchc, gazhchc, gzzhchc, gwwhchc,
     &                   gwah1hc, gwah2hc, gwah3hc,
     &                   gwahch1, gwahch2, gwahch3,
     &                   gwzh1hc, gwzh2hc, gwzh3hc,
     &                   gwzhch1, gwzhch2, gwzhch3,
     &                   ggadldl, ggaulul,
     &                   ggzdldl, ggzdrdr, ggzulul, ggzurur,
     &                   ggzb1b1, ggzb1b2, ggzb2b1, ggzb2b2,
     &                   ggzt1t1, ggzt1t2, ggzt2t1, ggzt2t2,
     &                   ggwuldl, ggwdlul,
     &                   ggwt1b1, ggwb1t1, ggwt1b2, ggwb2t1,
     &                   ggwt2b1, ggwb1t2, ggwt2b2, ggwb2t2

      double complex    gh111, gh112, gh122, gh222,
     &                  gh133, gh233, gh1cc, gh2cc,
     &                  gh1ulul, gh1urur, gh1dldl, gh1drdr,
     &                  gh1b1b1, gh1b1b2, gh1b2b1, gh1b2b2,
     &                  gh1t1t1, gh1t1t2, gh1t2t1, gh1t2t2,
     &                  gh1elel, gh1erer, gh1veve,
     &                  gh1l1l1, gh1l1l2, gh1l2l1, gh1l2l2,
     &                  gh2ulul, gh2urur, gh2dldl, gh2drdr,
     &                  gh2b1b1, gh2b1b2, gh2b2b1, gh2b2b2,
     &                  gh2t1t1, gh2t1t2, gh2t2t1, gh2t2t2,
     &                  gh2elel, gh2erer, gh2veve,
     &                  gh2l1l1, gh2l1l2, gh2l2l1, gh2l2l2,
     &                  gh3b1b1, gh3b1b2, gh3b2b1, gh3b2b2,
     &                  gh3t1t1, gh3t1t2, gh3t2t1, gh3t2t2,
     &                  gh3l1l1, gh3l1l2, gh3l2l1, gh3l2l2,
     &                  ghculdl, ghcdlul,
     &                  ghct1b1, ghcb1t1, ghct1b2, ghcb2t1,
     &                  ghct2b1, ghcb1t2, ghct2b2, ghcb2t2,
     &                  ghcveel, ghcelve,
     &                  ghcl1vt, ghcvtl1, ghcl2vt, ghcvtl2
      common /MSSM_SSS/ gh111, gh112, gh122, gh222,
     &                  gh133, gh233, gh1cc, gh2cc,
     &                  gh1ulul, gh1urur, gh1dldl, gh1drdr,
     &                  gh1b1b1, gh1b1b2, gh1b2b1, gh1b2b2,
     &                  gh1t1t1, gh1t1t2, gh1t2t1, gh1t2t2,
     &                  gh1elel, gh1erer, gh1veve,
     &                  gh1l1l1, gh1l1l2, gh1l2l1, gh1l2l2,
     &                  gh2ulul, gh2urur, gh2dldl, gh2drdr,
     &                  gh2b1b1, gh2b1b2, gh2b2b1, gh2b2b2,
     &                  gh2t1t1, gh2t1t2, gh2t2t1, gh2t2t2,
     &                  gh2elel, gh2erer, gh2veve,
     &                  gh2l1l1, gh2l1l2, gh2l2l1, gh2l2l2,
     &                  gh3b1b1, gh3b1b2, gh3b2b1, gh3b2b2,
     &                  gh3t1t1, gh3t1t2, gh3t2t1, gh3t2t2,
     &                  gh3l1l1, gh3l1l2, gh3l2l1, gh3l2l2,
     &                  ghculdl, ghcdlul,
     &                  ghct1b1, ghcb1t1, ghct1b2, ghcb2t1,
     &                  ghct2b1, ghcb1t2, ghct2b2, ghcb2t2,
     &                  ghcveel, ghcelve,
     &                  ghcl1vt, ghcvtl1, ghcl2vt, ghcvtl2
