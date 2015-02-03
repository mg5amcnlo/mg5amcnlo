       SUBROUTINE FEYNHIGGS(AMA,TGBET,AMT,AMST1,AMST2,STHT,AMSB1,
     .                      AMSB2,STHB,AMU,AMGLU,AM2,AML,AMH,SA,CA)

c     program FeynHiggsFast

c --------------------------------------------------------------
c
c     FeynHiggsFast
c     =============
c      
c       Calculation of the masses of the neutral CP-even
c       Higgs bosons in the MSSM
c       
c       Authors: Sven Heinemeyer (one-, two-loop part, new renormalization)
c                Andreas Dabelstein (one-loop part)
c                Markus Frank (new renormalization)
c       
c       Based on hep-ph/9803277, hep-ph/9807423, hep-ph/9812472,
c                hep-ph/9903404, hep-ph/9910283
c       by S. Heinemeyer, W. Hollik, G. Weiglein
c       and on hep-ph/0001002
c       by M. Carena, H. Haber, S. Heinemeyer, W. Hollik,
c          C. Wagner and G. Weiglein
c       new non-log O(alpha_t^2) corrections taken from hep-ph/0112177
c       by A. Brignole, G. Degrassi, P. Slavich and F. Zwirner
c
c       new renormalization implemented based on hep-ph/0202166
c       by M. Frank, S. Heinemeyer, W. Hollik and G. Weiglein
c      
c       In case of problems or questions,
c       contact Sven Heinemeyer
c       email: Sven.Heinemeyer@physik.uni-muenchen.de
c       
c       FeynHiggs homepage:
c       http://www.feynhiggs.de
c
c --------------------------------------------------------------


      implicit real*8(a-z)
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      real*8 mma,ttb,mmt,mmm,mmue,mmsusy,au,ad,mh1,mh2,mh12,mh22,l1,l2
      real*8 mtlr,mblr
      double precision mlhlle1, mlhlle2, 
     $       mlhllediag1, mlhllediag2, mhhllediag1, mhhllediag2
      double precision allle1, allle2
      integer ii,j,k,ic,pri,naeh,selec,selec2,selec3,selec4,
     $        selec5,selec6, sps, lhbms
      integer lleselec1,msbarselec,mtmsbarselec, msbarselecsave, mpgut
      double precision delrholimit, delrholimitnew, delrhores,
     $                 delrho1loop, delrho2loopgluon, delrho2loopgluino,
     $                 prefdelrhogluino,
     $                 delrho2loopmt2yuk, delrho2loopmt2yuksm
      double precision msusytrnew
      double precision yepsilon,ymuee,ypi,ymz,ymw,mgf,yas,yalphasmz,
     $                 ycf,yeps,yeins,ymbb
      complex*16 yi
      double precision mst1msbar, mst2msbar, sttmsbar, qt1, qt2, qts,
     $                 mst1os, mst2os, sttos,
     $                 msb1msbar, msb2msbar, stbmsbar, qb1, qb2, qbs,
     $                 mbs1os, msb2os, stbos
      double precision xmsusytl, xmsusytr, xmsusybl, xmsusybr, 
     $                 xmtlr, xmblr
      double precision lhseol, hhseol, xhseol, p1seol, p2seol, p1p2seol,
     $                 mlheff1, mhheff1, aleff1, alefff1,
     $                 mlheff2, mhheff2, aleff2, alefff2
      double precision mhptree, mhp1l
      double precision msusytlmsbar, msusytrmsbar, msusybrmsbar, 
     $                 atmsbar, abmsbar, xtmsbar, xbmsbar, qmsbar

c      external fapr,gundu,gundutl,ftest

      common /smpara1/ ymw,ymz,mmt
      common /smpara2/ yepsilon,ymuee,ypi,ygf,ycf,yeps,yi,yeins,
     $                 yas,yalphasmz,ymbb
      common /susypara/ ttb,mma,mmm,mmue,au,ad
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common /chargedhiggs/ mhptree, mhp1l
      common / err/ ic
      common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      common /lle/ mtlr, mblr, mlhlle1, mlhlle2,
     $             mlhllediag1, mlhllediag2, mhhllediag1, mhhllediag2
      common /lle2/ allle1, allle2
      common /mhapp/ mlhapp, mhhapp, mh2app, mh1app
      common /zeromom/ mlheff1, mhheff1, alefff1, 
     $                 mlheff2, mhheff2, alefff2
      common /lleselec/ lleselec1
      common /msbarselec/ msbarselec, mtmsbarselec
      common /gutrel/ mpgut
      integer msbariter
      common /msbarselec2/ msbariter
      double precision mtpole, mbpole
      common /polemasses/ mtpole, mbpole
      integer alphatsq
      common /alphat2/alphatsq

      double precision mudim
      common /msbar/ mudim


      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      integer error

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C--COMMON Blocks from HDECAY
      COMMON/MASSES_HDEC/AMS,AMC,AMB,AMT0
      COMMON/PARAM_HDEC/GF0,ALPH,AMTAU,AMMUON,AMZ,AMW
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


c ----------------------------------------------------------------
c setting the SM parameters 
c ----------------------------------------------------------------
      epsilon = 1.01d0
      muee     = 1.d0      

      pi = 3.14159265897d0

c     mz  = 91.187d0
c     mw =  80.451d0
c     gf = 1.16639d-5
c     alphasmz = 0.118d0
      cf = 4d0/3d0
      eps = 1d-10
      i = (0d0, 1d0)
      eins = 1d0

c     delrholimit = 2d-3

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C--Translation from HDECAY
      mz = AMZ
      mw = AMW
      gf = GF0
      alphasmz = ALPHAS_HDEC(AMZ,3)
      as = ALPHAS_HDEC(AMT,3)
      sps = 0
      lhbms = 0
      selec5 = 1
      selec6 = 1
      lleselec1 = 1
      mpgut = 1
      mtmsbarselec = 1
      msbarselec = 2
      msbarselecsave = msbarselec
      selec2 = 4
      selec3 = 3
      selec4 = 1
      delmbresum = 2
      delrholimitnew = 1.d0
      delrholimit = 1.d0
      print1 = 1
      print2 = 1
      print3 = 1
      msbariter = 1
      mtselec = 0
      alphatsq = 2
      ttb = TGBET
      mst1 = amst1
      qt1 = mst1
      mst2 = amst2
      qt2 = mst2
      stt = stht
      qts = dsqrt(mst1*mst2)
      msb1 = amsb1
      qb1 = msb1
      msb2 = amsb2
      qb2 = msb2
      stb = sthb
      qbs = dsqrt(msb1*msb2)
      mmt = AMT
      mbpole = AMB
      mgl = amglu
      mmue = AMU
      mmm = am2
      mma = AMA
      mudim = mmt
      selec = 3
      mbb = runm_hdec(amt,5)
c     mbb = amb
      mtpole = mmt
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c passing over the SM parameters
      yepsilon = epsilon
      ymuee = muee
      ypi = pi
      ymz = mz
      ymw = mw
      ygf = gf
      yas = as
      yalphasmz = alphasmz
      ycf = cf
      yeps = eps
      yi = i
      yeins = eins
      ymbb = mbb

c -----------------------------------------------------------------
c
c  switches
c  ========
c
c  selec2/ii = 1: full 1-loop + 2-loop QCD
c              2: same as 1, but in addition with
c                           mt = 166.5 at 2-loop
c              3: same as 2, but in addition with
c                           Yukawa term added for light Higgs
c
c  selec4 = 1: mt = pole mass at 2-loop in the stop mass matrix
c           2: mt = running mass
c           this this an option only if selec2/ii > 1,
c           otherwise selec4 = 1 is automatically chosen.
c
c  selec = 1: top/stop sector only in 1-loop calculation
c          2: top/stop + bottom/sbottom sector only
c          3: full 1-loop calculation
c
c  selec5 = 1: full result for the two-loop SEs is used
c           2: LL-Expansion for the two-loop SEs is used
c
c  selec6 = 2: the short formula from the LLE is also used in order to
c              to derive the Higgs masses 
c              You get two values for the Higgs masses then: 
c              the 'complete' one and the 'short' one
c         = 1: the short formula is not used, you get only one result
c
c  lleselec1 = 1: the LLE contribution is evaluated with the short
c                 (and published) formula
c            = 2: the LLE contribution is evaluated with the long
c                 (and unpublished) formula -> LLExpansionP2.f
c
c  msbarselec = 1: input parameters are on-shell parameter
c             = 2: input parameters are MSbar parameters
c                  (they are then transformed into the on-shell scheme
c                   in order to use them for FeynHiggs)
c
c  alphatsq = 1: alpha_t^2 corrections in LL approximation (RGiEP)
c           = 2:           full calculation (EP 2-loop)
c
c  delmbresum = 1: Delta mb corrections are not included in mh calculation
c             = 2: Delta mb corrections are included   
c
c -----------------------------------------------------------------

c     single point calculation with online parameter input

      goto 1234

      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*) "---------------------------------------------------"
      write(*,*)
      write(*,*) "Welcome to FeynHiggsFast"
      write(*,*) "========================"
      write(*,*)
      write(*,*) "Calculation of the masses of the neutral CP-even"
      write(*,*) "Higgs bosons in the MSSM (fast and approximate)"
      write(*,*) 
      write(*,*) "Based on hep-ph/9803277, hep-ph/9807423, " //
     $           "hep-ph/9812472, hep-ph/9903404"
      write(*,*) "by S. Heinemeyer, W. Hollik, G. Weiglein"
      write(*,*) "and on hep-ph/0001002"
      write(*,*) "by M. Carena, H. Haber, S. Heinemeyer, "//
     $           "W. Hollik, C. Wagner and G. Weiglein"
      write(*,*) "New renormalization based on hep-ph/0202166"
      write(*,*) "by M. Frank, S. Heinemeyer, W. Hollik " //
     $           "and G. Weiglein"
      write(*,*) "new non-log O(alpha_t^2) corrections taken " //
     $           "from hep-ph/0112177"
      write(*,*) "by A. Brignole, G. Degrassi, P. Slavich " //
     $           "and F. Zwirner"
      write(*,*)
      write(*,*) "In case of problems or questions,"
      write(*,*) "contact Sven Heinemeyer"
      write(*,*) "email: Sven.Heinemeyer@physik.uni-muenchen.de"
      write(*,*) 
      write(*,*) "FeynHiggs home page: www.feynhiggs.de"
      write(*,*)
      write(*,*) "---------------------------------------------------"
      write(*,*) "enter 0 to start"
      read(*,*) selec
      

 90   continue

      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*)
      write(*,*) "SETUP:      ( * indicates one input)"
      write(*,*) "======"
      write(*,*)
      write(*,*) "* depth of calculation ?"
      write(*,*) "   1: full 1-loop + 2-loop QCD"
      write(*,*) "   2: same as 1, but in addition with"
      write(*,*) "                 running mt at 2-loop"
      write(*,*) "   3: same as 2, but in addition with"
      write(*,*) "                 O(alpha_t^2)-log term included"
      write(*,*) "   4: same as 3, but in addition with"
      write(*,*) "                 O(alpha_t^2)-non-log term included"
      write(*,*) "      (4 gives most accurate result)"
      write(*,*) "* Select input:"
      write(*,*) "   1: On-Shell: Msusy_top_L,R, Xt, Msusy_bot_L,R, Xb"
      write(*,*) "   2: On-Shell: MSt1, MSt2, stt, MSb1, MSb2, stb"
      write(*,*) "   3: MSbar   : MSt1, MSt2, stt, MSb1, MSb2, stb"
      write(*,*) "                (no iteration)"
      write(*,*) "   4: MSbar   : MSt1, MSt2, stt, MSb1, MSb2, stb"
      write(*,*) "                (with iteration)"
      write(*,*) "   5: SPS (selection possible)"
      write(*,*) "   6: Les Houches BMS (input: MA, tb)"
      write(*,*) 
      write(*,*) "* mt in the OS stop mass matrix at 2-loop ?"
      write(*,*) "   --> irrelevant if input = 2, 3, 4 has been chosen"
      write(*,*) "   1: mt = pole mass"
      write(*,*) "   2: mt = running mass"
      write(*,*) "* Delta mb resummation for Higgs masses ?"
      write(*,*) "   1: not included"
      write(*,*) "   2: included"
      write(*,*) "* Limit for Delta rho = 2 * 10^-3 ? (0 = ok)"
      write(*,*)
      write(*,*) "What additional information do you want?"
      write(*,*) "*    Test output of your input parameters? " //
     $           "(0: yes // 1: no)"
      write(*,*) "*    Sfermion masses etc? (0: yes // 1: no)"
      write(*,*) "*    Neutralino/Chargino masses etc? " //
     $                 "(0: yes // 1: no)"
      write(*,*)
      sps = 0
      lhbms = 0
      selec5 = 1
      selec6 = 1
      lleselec1 = 1
      mpgut = 1
      mtmsbarselec = 1
      msbarselec = 2
      msbarselecsave = msbarselec
      read(*,*) ii, selec3, selec4, delmbresum, 
     $          delrholimit, 
     $          print1, print2, print3
      write(*,*)
      if ((ii.ne.1).and.(ii.ne.2).and.(ii.ne.3).and.(ii.ne.4)) then
         write(*,*) "WRONG INPUT FOR DEPTH OF CALCULATION"
         goto 90
      endif
      if ((selec3.ne.1).and.(selec3.ne.2).and.
     $    (selec3.ne.3).and.(selec3.ne.4).and.
     $    (selec3.ne.5).and.(selec3.ne.6)) then 
         write(*,*) "WRONG INPUT FOR INPUT SELECTION"
         goto 90
      endif
      msbariter = 0
      if (selec3.eq.4) then
         msbariter = 1
         selec3 = 3
      endif
      if (selec3.ne.1) selec4 = 1
      if ((selec4.ne.1).and.(selec4.ne.2)) then 
         write(*,*) "WRONG INPUT FOR MT IN STOP MASS MATRIX"
         goto 90
      endif
      if ((delmbresum.ne.1).and.(delmbresum.ne.2)) then 
         write(*,*) "WRONG INPUT FOR DELTA MB RESUMMATION"
         goto 90
      endif
      if (delrholimit.eq.0d0) delrholimit = 2d-3
      if ((print1.ne.0).and.(print1.ne.1))  then
         write(*,*) "WRONG INPUT FOR PARAMTER TEST OUTPUT"
         goto 90
      endif
      if ((print2.ne.0).and.(print2.ne.1)) then 
         write(*,*) "WRONG INPUT FOR SFERMION OUTPUT"
         goto 90
      endif
      if ((print3.ne.0).and.(print3.ne.1)) then
         write(*,*) "WRONG INPUT FOR CHARGINO/NEUTRALINO OUTPUT"
         goto 90
      endif

      if (ii.le.3) alphatsq = 1
      if (ii.eq.4) alphatsq = 2
      mtselec = 0
      goto 110

 100  continue
      write(*,*)
      write(*,*) "What next? (0: new calculation // 1: setup)"
      read(*,*) selec
      if (selec.eq.1) goto 90
      if (sps.ne.0) selec3 = 5
      if (lhbms.ne.0) selec3 = 6

 110  continue


c --> Input of parameters

      if (selec3.eq.1) then
c --> Input: OS, Msusy, Xt, Xb

      write(*,*)
      write(*,*)
      write(*,*) "INPUT:"
      write(*,*) "======"
      write(*,*)
      write(*,*) "tan(beta)"
      write(*,*) "Msusy_top_L"
      write(*,*) "Msusy_top_R (0: Msusy_top_R = Msusy_top_L)"
      write(*,*) "Msusy_bot_R (0: Msusy_bot_R = Msusy_top_L)"
      write(*,*) "Xt == MtLR"
      write(*,*) "Xb == MbLR (1: Ab = At)"
      write(*,*) "Mtop(pole) (0: Mtop = 175 // 1: Mtop = 174.3)"
      write(*,*) "Mbottom(pole) (0: Mbot = 4.5 // 1: Mbot = 4.25)"
      write(*,*) "Mgluino (0: Mgl = 500 // 1: Mgl = Msusy_top_L)"
      write(*,*) "mu (1: mu = 200 // 2: mu = Msusy_top_L)"
      write(*,*) "M2 (1: M2 = 400 // 2: M2 = Msusy_top_L)"
      write(*,*) "MA"
      write(*,*) "renormalization scale "
     $           // "(0: Mtop/2 // 1: Mtop // 2: 2 Mtop)"
      write(*,*) 'Accuracy for the one-loop calculation:'
      write(*,*) "           1 = top only, 2 = top/bottom only, 3 = all"
      write(*,*)
      read(*,*) ttb, msusytl, msusytr, msusybr, mtlr, mblr, 
     $          mmt, mbpole, mgl, mmue, mmm, mma, mudim, selec
      if (ttb.eq.0d0) goto 90
      mmsusy = msusytl
      msusybl = msusytl
      if (msusytr.eq.0d0) msusytr = mmsusy
      if (msusybr.eq.0d0) msusybr = mmsusy
      if (mmt.eq.0d0) mmt = 175d0
      if (mmt.eq.1d0) mmt = 174.3d0
      if (mbpole.eq.0d0) mbpole = 4.5d0
      if (mbpole.eq.1d0) mbpole = 4.25d0
      mbb = 2.97d0
      mtpole = mmt
      if (mgl.eq.0d0) mgl = 500d0
      if (mgl.eq.1d0) mgl = mmsusy
      if (mmue.eq.1d0) mmue = 200d0
      if (mmue.eq.2d0) mmue = mmsusy
      if (mmm.eq.1d0) mmm = 400d0
      if (mmm.eq.2d0) mmm = mmsusy
      if (mudim.eq.0d0) mudim = mmt/2d0
      if (mudim.eq.1d0) mudim = mmt
      if (mudim.eq.2d0) mudim = mmt*2d0

      elseif (selec3.eq.2) then
c --> Input: OS, MSt1, MSt2, stt, MSb1, MSb2, stb

      write(*,*)
      write(*,*)
      write(*,*) "INPUT:"
      write(*,*) "tan(beta)"
      write(*,*) "MSt1"
      write(*,*) "MSt2"
      write(*,*) "sin(theta_stop) (1: stt = sin(-Pi/4))"
      write(*,*) "MSb1"
      write(*,*) "MSb2"
      write(*,*) "sin(theta_sbot) (1: stb = sin(+Pi/4))"
      write(*,*) "Mtop(pole) (0: Mtop = 175 // 1: Mtop = 174.3)"
      write(*,*) "Mbottom(pole) (0: Mbot = 4.5 // 1: Mbot = 4.25)"
      write(*,*) "Mgluino (0: Mgl = 500 // 1: Mgl = 1000)"
      write(*,*) "mu (1: mu = 200 // 2: mu = 1000)"
      write(*,*) "M2 (1: M2 = 400 // 2: M2 = 1000)"
      write(*,*) "MA"
      write(*,*) "renormalization scale "
     $           // "(0: Mtop/2 // 1: Mtop // 2: 2 Mtop)"
      write(*,*) 'Accuracy for the one-loop calculation:'
      write(*,*) "           1 = top only, 2 = top/bottom only, 3 = all"
      write(*,*)
      read(*,*) ttb, mst1, mst2, stt, msb1, msb2, stb,
     $          mmt, mbpole, mgl, mmue, 
     $          mmm, mma, mudim, selec
      if (ttb.eq.0d0) goto 90
      if (stt.eq.1d0) stt = -0.70710678d0
      if (stb.eq.1d0) stb = +0.70710678d0
      if (mmt.eq.0d0) mmt = 175d0
      if (mmt.eq.1d0) mmt = 174.3d0
      if (mbpole.eq.0d0) mbpole = 4.5d0
      if (mbpole.eq.1d0) mbpole = 4.25d0
      mbb = 2.97d0
      mtpole = mmt
      if (mmue.eq.1d0) mmue = 200d0
      if (mmue.eq.2d0) mmue = 1000d0
      if (mgl.eq.0d0) mgl = 500d0
      if (mgl.eq.1d0) mgl = 1000d0
      if (mmm.eq.1d0) mmm = 400d0
      if (mmm.eq.2d0) mmm = 1000d0
      if (mudim.eq.0d0) mudim = mmt/2d0
      if (mudim.eq.1d0) mudim = mmt
      if (mudim.eq.2d0) mudim = mmt*2d0

      elseif (selec3.eq.3) then
c --> Input: MSbar, MSt1, MSt2, stt, MSb1, MSb2, stb

      write(*,*)
      write(*,*)
      write(*,*) "INPUT:"
      write(*,*) "tan(beta)"
      write(*,*) "MSt1, scale of MSt1 (1: q = MSt1)"
      write(*,*) "MSt2, scale of MSt2 (1: q = MSt2)"
      write(*,*) "sin(theta_stop) (1: stt = sin(-Pi/4))"
      write(*,*) "scale of sine " //
     $           "(1: MSt1 // 2: MSt2 // 3: Sqrt[MSt1 MSt2] )"
      write(*,*) "MSb1, scale of MSb1 (1: q = MSb1)"
      write(*,*) "MSb2, scale of MSb2 (1: q = MSb2)"
      write(*,*) "sin(theta_sbot) (1: stb = sin(+Pi/4))"
      write(*,*) "scale of sine " //
     $           "(1: MStb // 2: MSb2 // 3: Sqrt[MSb1 MSb2] )"
      write(*,*) "Mtop(pole) (0: Mtop = 175 // 1: Mtop = 174.3)"
      write(*,*) "Mbottom(pole) (0: Mbot = 4.5 // 1: Mbot = 4.25)"
      write(*,*) "Mgluino (0: Mgl = 500 // 1: Mgl = 1000)"
      write(*,*) "mu (1: mu = 200 // 2: mu = 1000)"
      write(*,*) "M2 (1: M2 = 400 // 2: M2 = 1000)"
      write(*,*) "MA"
      write(*,*) "renormalization scale "
     $           // "(0: Mtop/2 // 1: Mtop // 2: 2 Mtop)"
      write(*,*) 'Accuracy for the one-loop calculation:'
      write(*,*) "           1 = top only, 2 = top/bottom only, 3 = all"
      write(*,*)
      read(*,*) ttb, mst1, qt1, mst2, qt2, stt, qts, 
     $               msb1, qb1, msb2, qb2, stb, qbs,
     $          mmt, mbpole, mgl, mmue, 
     $          mmm, mma, mudim, selec
      if (ttb.eq.0d0) goto 90
      if (qt1.eq.1d0) qt1 = mst1
      if (qt2.eq.1d0) qt2 = mst2
      if (qts.eq.1d0) then
         qts = mst1
      elseif (qts.eq.2d0) then
         qts = mst2
      elseif (qts.eq.3d0) then
         qts = dsqrt(mst1 * mst2)
      endif
      if (stt.eq.1d0) stt = -0.70710678d0
      if (qb1.eq.1d0) qb1 = msb1
      if (qb2.eq.1d0) qb2 = msb2
      if (qbs.eq.1d0) then
         qbs = msb1
      elseif (qbs.eq.2d0) then
         qbs = msb2
      elseif (qbs.eq.3d0) then
         qbs = dsqrt(msb1 * msb2)
      endif
      if (stb.eq.1d0) stb = +0.70710678d0
      if (mmt.eq.0d0) mmt = 175d0
      if (mmt.eq.1d0) mmt = 174.3d0
      if (mbpole.eq.0d0) mbpole = 4.5d0
      if (mbpole.eq.1d0) mbpole = 4.25d0
      mbb = 2.97d0
      mtpole = mmt
      if (mmue.eq.1d0) mmue = 200d0
      if (mmue.eq.2d0) mmue = 1000d0
      if (mgl.eq.0d0) mgl = 500d0
      if (mgl.eq.1d0) mgl = 1000d0
      if (mmm.eq.1d0) mmm = 400d0
      if (mmm.eq.2d0) mmm = 1000d0
      if (mudim.eq.0d0) mudim = mmt/2d0
      if (mudim.eq.1d0) mudim = mmt
      if (mudim.eq.2d0) mudim = mmt*2d0

      elseif (selec3.eq.4) then
c --> Input: MSbar, MSt1, MSt2, stt, Msusy_bot_R, Xb

      write(*,*)
      write(*,*)
      write(*,*) "INPUT:"
      write(*,*) "tan(beta)"
      write(*,*) "MSt1, scale of MSt1 (1: q = MSt1)"
      write(*,*) "MSt2, scale of MSt2 (1: q = MSt2)"
      write(*,*) "sin(theta_stop) (0: stt = 0 // 1: stt = sin(-Pi/4))"
      write(*,*) "scale of sine " //
     $           "(1: MSt1 // 2: MSt2 // 3: Sqrt[MSt1 MSt2] )"
      write(*,*) "NOTE: Msusy_bot_L = Msusy_top_L"
      write(*,*) "Msusy_bot_R (1: Msusy_bot_R = Msusy_bot_L)"
      write(*,*) "Xb (1: Ab = At)"
      write(*,*) "Mtop(pole) (0: Mtop = 175 // 1: Mtop = 174.3)"
      write(*,*) "Mbottom(pole) (0: Mbot = 4.5 // 1: Mbot = 4.25)"
      write(*,*) "Mgluino (0: Mgl = 500 // 1: Mgl = 1000)"
      write(*,*) "mu (1: mu = 200 // 2: mu = 1000)"
      write(*,*) "M2 (1: M2 = 400 // 2: M2 = 1000)"
      write(*,*) "MA"
      write(*,*) "renormalization scale "
     $           // "(0: Mtop/2 // 1: Mtop // 2: 2 Mtop)"
      write(*,*) 'Accuracy for the one-loop calculation:'
      write(*,*) "           1 = top only, 2 = top/bottom only, 3 = all"
      write(*,*)
      read(*,*) ttb, mst1, qt1, mst2, qt2, stt, qts, msusybr, mblr,
     $          mmt, mbpole, mgl, mmue, 
     $          mmm, mma, mudim, selec
      if (ttb.eq.0d0) goto 90
      if (qt1.eq.1d0) qt1 = mst1
      if (qt2.eq.1d0) qt2 = mst2
      if (qts.eq.1d0) then
         qts = mst1
      elseif (qts.eq.2d0) then
         qts = mst2
      elseif (qts.eq.3d0) then
         qts = dsqrt(mst1 * mst2)
      endif
      if (stt.eq.1d0) stt = -0.70710678d0
      if (mmt.eq.0d0) mmt = 175d0
      if (mmt.eq.1d0) mmt = 174.3d0
      if (mbpole.eq.0d0) mbpole = 4.5d0
      if (mbpole.eq.1d0) mbpole = 4.25d0
      mbb = 2.97d0
      mtpole = mmt
      if (mmue.eq.1d0) mmue = 200d0
      if (mmue.eq.2d0) mmue = 1000d0
      if (mgl.eq.0d0) mgl = 500d0
      if (mgl.eq.1d0) mgl = 1000d0
      if (mmm.eq.1d0) mmm = 400d0
      if (mmm.eq.2d0) mmm = 1000d0
      if (mudim.eq.0d0) mudim = mmt/2d0
      if (mudim.eq.1d0) mudim = mmt
      if (mudim.eq.2d0) mudim = mmt*2d0

      elseif (selec3.eq.5) then
c----------------------------------------------------------------
c
c --> sps.h
c
c----------------------------------------------------------------
 15      continue
         write(*,*)
         write(*,*)
         write(*,*) "SPS: Snowmass Points and Slopes"
         write(*,*) "==============================="
         write(*,*)
         write(*,*) "see hep-ph/0202233, hep-ph/0201233"
         write(*,*) "SPS home page: www.ippp.dur.ac.uk/~georg/sps/"
         write(*,*)
         write(*,*)
      write(*,*) "  SPS             Point            Characteristic"
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) "mSUGRA:  m0  m1/2     A0  tb"
      write(*,*) "----------------------------------------------------"
      write(*,*) " 1a(1)  100   250   -100  10       typical"
      write(*,*) " 1b(11) 200   400      0  30       typical"
      write(*,*) " 2     1450   300      0  10       focus point"
      write(*,*) " 3       90   400      0  10       coannihilation"
      write(*,*) " 4      400   300      0  50       large tan beta"
      write(*,*) " 5      150   300  -1000   5       light stop"
      write(*,*) "----------------------------------------------------"
      write(*,*) " 6      150   300      0  10       M1=480, M2=M3=300"
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) " GMSB: Lambda  Mmes  Nmes tb"
      write(*,*) "----------------------------------------------------"
      write(*,*) " 7     40000  80000   3   15       stau NLSP"
      write(*,*) " 8    100000 200000   1   15       neutralino NLSP"
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) " AMSB:   m0  maux         tb"
      write(*,*) "----------------------------------------------------"
      write(*,*) " 9      450  60000        10"
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) 
      write(*,*) "SPS number ?"
      read(*,*) sps
      write(*,*)
      if ((sps.ne.1).and.(sps.ne.2).and.(sps.ne.3).and.(sps.ne.4).and.
     $    (sps.ne.5).and.(sps.ne.6).and.(sps.ne.7).and.(sps.ne.8).and.
     $    (sps.ne.9).and.(sps.ne.11)) then
         write(*,*) "WRONG INPUT -- TRY AGAIN!!"
         goto 15
      endif

      if (sps.eq.1) then
         msusytlmsbar = 496d0
         msusytrmsbar = 425d0
         msusybrmsbar = 517d0
         atmsbar = -510d0
         abmsbar = -773d0
         qmsbar = 455d0
         mmm = 193d0
         mgl = 595d0
         mmue = 352d0
         mma = 394d0
         ttb = 10d0
      elseif (sps.eq.11) then
         msusytlmsbar = 762d0
         msusytrmsbar = 670d0
         msusybrmsbar = 780d0
         atmsbar = -729d0
         abmsbar = -987d0
         qmsbar = 454d0
         mmm = 311d0
         mgl = 916d0
         mmue = 496d0
         mma = 526d0
         ttb = 30d0
         write(*,*) "WARNING: SPS1b implemented only as approximation"
      elseif (sps.eq.2) then
         msusytlmsbar = 1295d0
         msusytrmsbar = 998d0
         msusybrmsbar = 1520d0
         atmsbar = -564d0
         abmsbar = -797d0
         qmsbar = 1077d0
         mmm = 234d0
         mgl = 784d0
         mmue = 125d0
         mma = 1443d0
         ttb = 10d0
      elseif (sps.eq.3) then
         msusytlmsbar = 761d0
         msusytrmsbar = 661d0
         msusybrmsbar = 786d0
         atmsbar = -734d0
         abmsbar = -1042d0
         qmsbar = 704d0
         mmm = 311d0
         mgl = 914d0
         mmue = 509d0
         mma = 572d0
         ttb = 10d0
      elseif (sps.eq.4) then
         msusytlmsbar = 640d0
         msusytrmsbar = 557d0
         msusybrmsbar = 673d0
         atmsbar = -552d0
         abmsbar = -730d0
         qmsbar = 571d0
         mmm = 233d0
         mgl = 721d0
         mmue = 377d0
         mma = 404d0
         ttb = 50d0
      elseif (sps.eq.5) then
         msusytlmsbar = 535d0
         msusytrmsbar = 361d0
         msusybrmsbar = 621d0
         atmsbar = -906d0
         abmsbar = -1671d0
         qmsbar = 450d0
         mmm = 235d0
         mgl = 710d0
         mmue = 640d0
         mma = 694d0
         ttb = 5d0
      elseif (sps.eq.6) then
         msusytlmsbar = 591d0
         msusytrmsbar = 517d0
         msusybrmsbar = 619d0
         atmsbar = -570d0
         abmsbar = -811d0
         qmsbar = 548d0
         mmm = 232d0
         mgl = 708d0
         mmue = 394d0
         mma = 463d0
         ttb = 10d0
      elseif (sps.eq.7) then
         msusytlmsbar = 836d0
         msusytrmsbar = 780d0
         msusybrmsbar = 827d0
         atmsbar = -319d0
         abmsbar = -350d0
         qmsbar = 840d0
         mmm = 327d0
         mgl = 926d0
         mmue = 300d0
         mma = 378d0
         ttb = 15d0
      elseif (sps.eq.8) then
         msusytlmsbar = 1043d0
         msusytrmsbar = 953d0
         msusybrmsbar = 1026d0
         atmsbar = -297d0
         abmsbar = -330d0
         qmsbar = 988d0
         mmm = 272d0
         mgl = 821d0
         mmue = 398d0
         mma = 514d0
         ttb = 15d0
      elseif (sps.eq.9) then
         msusytlmsbar = 1112d0
         msusytrmsbar = 1003d0
         msusybrmsbar = 1232d0
         atmsbar = -350d0
         abmsbar = 216d0
         qmsbar = 1076d0
         mmm = -176d0
         mgl = 1275d0
         mmue = 870d0
         mma = 911d0
         ttb = 10d0
      endif

      mmt = 175d0
      mbpole = 4.25d0
      mbb = 2.97d0
      xtmsbar = atmsbar - mmue/ttb
      xbmsbar = abmsbar - mmue/ttb
      mudim = mmt
      selec = 3
      selec3 = 3
      atau = 2
      qt1 = qmsbar
      qt2 = qmsbar
      qts = qmsbar
      qb1 = qmsbar
      qb2 = qmsbar
      qbs = qmsbar

      call def4(msusytlmsbar, msusytrmsbar, xtmsbar, mmt, ttb,
     $          mst1, mst2, stt, 1)
      call def4(msusytlmsbar, msusybrmsbar, xbmsbar, mbb, ttb,
     $          msb1, msb2, stb, 2)

c----------------------------------------------------------------
c
c --> end of sps.h
c
c----------------------------------------------------------------
      

      elseif (selec3.eq.6) then
c----------------------------------------------------------------
c
c --> lhbms.h
c
c----------------------------------------------------------------
      
 16   continue
      write(*,*)
      write(*,*)
      write(*,*) "LH BMS (Les Houches Benchmarsk)"
      write(*,*) "==============================="
      write(*,*)
      write(*,*) "see hep-ph/0202167 or www.feynhiggs.de"
      write(*,*)
      write(*,*)
      write(*,*) "   #    name/characteristic"
      write(*,*) "-----------------------------------------------------"
      write(*,*) "   1    mh-max"
      write(*,*) " (11)   contrained mh-max (better for b -> s gamma)"
      write(*,*) "   2    no-mixing"
      write(*,*) "   3    gluophobic Higgs (gg -> h suppressed)"
      write(*,*) "   4    small alpha_eff (h -> bb, tau tau suppressed)"
      write(*,*) "-----------------------------------------------------"
      write(*,*) 
      write(*,*) "LH BMS number ?  // MA ?  // tan beta ?"
      read(*,*) lhbms, mma, ttb
      write(*,*)
      if ((lhbms.ne.1).and.(lhbms.ne.2).and.(lhbms.ne.3).and.
     $    (lhbms.ne.4).and.(lhbms.ne.11)) then
         write(*,*) "WRONG INPUT -- TRY AGAIN!!"
         goto 16
      endif

      if (lhbms.eq.1) then
         mmsusy = 1000d0
         mmue = 200d0
         mmm = 200d0
         mtlr = 2000d0
         mgl = 800d0
      elseif (lhbms.eq.11) then
         mmsusy = 1000d0
         mmue = 200d0
         mmm = 200d0
         mtlr = -2000d0
         mgl = 800d0
      elseif (lhbms.eq.2) then
         mmsusy = 2000d0
         mmue = 200d0
         mmm = 200d0
         mtlr = 0d0
         mgl = 800d0
      elseif (lhbms.eq.3) then
         mmsusy = 350d0
         mmue = 300d0
         mmm = 300d0
         mtlr = -750d0
         mgl = 500d0
      elseif (lhbms.eq.4) then
         mmsusy = 800d0
         mmue = 2000d0
         mmm = 500d0
         mtlr = -1100d0
         mgl = 500d0
      endif

      mmt = 174.3d0
      mbpole = 4.25d0
      mbb = 2.97d0
      mudim = mmt
      mblr = 1
      atau = 2
      msusytl = mmsusy
      msusytr = mmsusy
      msusybl = mmsusy
      msusybr = mmsusy

      selec = 3
      selec3 = 1


c----------------------------------------------------------------
c
c --> end of lhbms.h
c
c----------------------------------------------------------------
      

      else
         write(*,*) 'internal error for selec3... stopping'
         stop
      endif


1234  CONTINUE


c --> Evaluating the OS parametes (if necessary...)

      if (selec3.eq.1) then
c --> Input: OS, Msusy, Xt, Xb

      au = mtlr + mmue/ttb
      if (mblr.eq.1d0) then
         ad = au
         mblr = ad - mmue * ttb
      endif
      ad = mblr + mmue * ttb
      mlrt = mtlr
      mlrb = mblr

      elseif (selec3.eq.2) then
c --> Input: OS, MSt1, MSt2, stt, MSb1, MSb2, stb

      call def4b(mst1,mst2,stt,mmt,2d0/3d0,
     $           ttb,msusytl,msusytr,mtlr,1)
      call def4b(msb1,msb2,stb,mbb,-1d0/3d0,
     $           ttb,msusybl,msusybr,mblr,2)
      if ((msusytl.eq.0d0).or.(msusybl.eq.0d0)) then
         write(*,*) 'inconsistency in the sfermion sector'
         write(*,*) '... skipping calculation'
         mh12 = 119.9999d0
         goto 100
      endif
      au = mtlr + mmue/ttb
      ad = mblr + mmue*ttb
      mlrt = mtlr
      mlrb = mblr
      write(*,*)
      write(*,*) 'Transition to unphysical parameters completed'

      elseif (selec3.eq.3) then
c --> Input: MSbar, MSt1, MSt2, stt, MSb1, MSb2, stb

      mst1msbar = mst1
      mst2msbar = mst2
      sttmsbar = stt
      mtos = mmt
      msb1msbar = msb1
      msb2msbar = msb2
      stbmsbar = stb
      mbos = mbb
c      write(*,*) 'vor MSbartoOnShellPPtopbot'
      call MSbartoOnShellPPtopbot(mst1msbar,mst2msbar,sttmsbar,
     $     qt1, qt2, qts,
     $     msb1msbar, msb2msbar, stbmsbar,
     $     qb1, qb2, qbs,
     $     mtos, mbos, mgl, yalphasmz, ymz,
     $     mst1os, mst2os, sttos,
     $     msb1os, msb2os, stbos)
      mst1 = mst1os
      mst2 = mst2os
      stt = sttos
      msb1 = msb1os
      msb2 = msb2os
      stb = stbos

      call def4b(mst1,mst2,stt,mmt,2d0/3d0,
     $           ttb,msusytl,msusytr,mtlr,1)
      call def4b(msb1,msb2,stb,mbb,-1d0/3d0,
     $           ttb,msusybl,msusybr,mblr,2)
      if ((msusytl.eq.0d0).or.(msusybl.eq.0d0)) then
         write(*,*) 'inconsistency in the sfermion sector'
         write(*,*) '... skipping calculation'
         goto 100
      endif

      au = mtlr + mmue/ttb
      ad = mblr + mmue*ttb
c      al = mllr + mmue*ttb
      mlrt = mtlr
      mlrb = mblr
c     write(*,*) 'At, Ab:', real(au), real(ad)
c     write(*,*) 'Transition to unphysical OS parameters completed'


      elseif (selec3.eq.4) then
c --> Input: MSbar, MSt1, MSt2, stt, Msusy_bot_R, Xb

      mst1msbar = mst1
      mst2msbar = mst2
      sttmsbar = stt
      mtos = mmt
c      write(*,*) 'vor MSbartoOnShellPP'
      call MSbartoOnShellPP(mst1msbar,mst2msbar,sttmsbar,
     $     qt1, qt2, qts,
     $     mtos, mgl, alphasmz, ymz,
     $     mst1os, mst2os, sttos)
      mst1 = mst1os
      mst2 = mst2os
      stt = sttos

      call def5(ttb,mmt,ymw,ymz,mst1,mst2,stt, msusybr, mblr,
     $          xmsusytl,xmsusytr,xmsusybl,xmsusybr,xmtlr,xmblr)
c      write(*,*) 'Results of def5:'
c      write(*,*) real(xmsusytl), real(xmsusytr), real(xmsusybl),
c     $           real(xmsusybr), real(xmtlr), real(xmblr)
      msusytl = xmsusytl
      msusytr = xmsusytr
      msusybl = xmsusybl
      msusybr = xmsusybr
      if (msusybr.eq.1d0) msusybr = msusybl
      mtlr = xmtlr
      mblr = xmblr
      if (msusytl.eq.0d0) then 
         write(*,*) 'Stop masses and the mixing angle do not fit'
     $              // ' together'
         goto 100
      endif
      au = mtlr + mmue/ttb
      ad = mblr + mmue*ttb
      if (mblr.eq.1d0) ad = au
      mblr = ad - mmue*ttb
      mlrt = mtlr
      mlrb = mblr
      write(*,*) 'Transition to unphysical OS parameters completed'


      else
         write(*,*) 'internal error for selec3... stopping'
         stop
      endif
            



      ymbb = mbb

      if (print1.eq.0) then
      write(*,*)
      write(*,*) "Your OS parameters:"
      write(*,*) "MT, Msusy(top-left), Msusy(top-right), Xt, At"
      write(*,*) "MB, Msusy(bot-left), Msusy(bot-rigth), Xb, Ab"
      write(*,*) "tb, Mgl, Mu, M2, MA"
      write(*,*) real(mmt), real(msusytl), real(msusytr), real(mtlr),
     $                                                    real(au)
      write(*,*) real(mbb), real(msusybl), real(msusybr), real(mblr),
     $                                                    real(ad)
      write(*,*) real(ttb), real(mgl), real(mmue), real(mmm), real(mma)
      write(*,*)
      endif


c     selec2 = ii
      dr1l = 0
c     write(*,*) "-------------------------------------------------"
c     write(*,*) "-------------------------------------------------"
c     write(*,*) "Performing the calculation..."

      call feynhiggssub(mh1,mh2,mh12,mh22)

      GOTO 2345

      write(*,*) "-------------------------------------------------"
      write(*,*) "-------------------------------------------------"
c      write(*,*) mst1, mst2
c      write(*,*) msb1, msb2
      if (   (real(mst1).le.0d0).or.(real(mst2).le.0d0).
     $    or.(real(msb1).le.0d0).or.(real(msb2).le.0d0)) then
         write(*,*) "negative entry in sfermion mass matrix"
         write(*,*) "MSt1, MSt2: ", real(mst1), real(mst2)
         write(*,*) "MSb1, MSb2: ", real(msb1), real(msb2)
         goto 100
      endif
      if (   (real(mst1).le.65d0).or.(real(mst2).le.65d0).
     $    or.(real(msb1).le.65d0).or.(real(msb2).le.65d0)) then
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
         write(*,*) "experimental excluded sfermion masses:"
         write(*,*) "MSt1, MSt2: ", real(mst1), real(mst2)
         write(*,*) "MSb1, MSb2: ", real(msb1), real(msb2)
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
      else
         if (print2.eq.0) then
         write(*,*) "The Sfermion masses:"
         write(*,*) "MSt1, MSt2: ", real(mst1), real(mst2)
         write(*,*) "MSb1, MSb2: ", real(msb1), real(msb2)
         ctt = dsqrt(1d0 - stt**2)
         ctb = dsqrt(1d0 - stb**2)
         write(*,*) "cos(mixing angles): ", real(ctt), real(ctb)
         write(*,*) "sin(mixing angles): ", real(stt), real(stb)
         endif
      endif
      if (print3.eq.0) then
         write(*,*) "Chargino Masses:", real(mcha(1)), real(mcha(2))
         write(*,*) "The mixing matrices:"
         write(*,*) "             U                             V"
         write(*,*) real(umix(1,1)), real(umix(1,2)),
     $              real(vmix(1,1)), real(vmix(1,2))
         write(*,*) real(umix(2,1)), real(umix(2,2)),
     $              real(vmix(2,1)), real(vmix(2,2))
         write(*,*) "Neutralino Masses:"
         write(*,*) real(mne(1)), real(mne(2)), 
     $              real(mne(3)), real(mne(4))
         write(*,*) "The mixing matrix:"
         write(*,*) real(nmix(1,1)), real(nmix(1,2)),
     $              real(nmix(1,3)), real(nmix(1,4))
         write(*,*) real(nmix(2,1)), real(nmix(2,2)),
     $              real(nmix(2,3)), real(nmix(2,4))
         write(*,*) real(nmix(3,1)), real(nmix(3,2)),
     $              real(nmix(3,3)), real(nmix(3,4))
         write(*,*) real(nmix(4,1)), real(nmix(4,2)),
     $              real(nmix(4,3)), real(nmix(4,4))
      endif

      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) "The results:  light Higgs     heavy Higgs     alpha"
      write(*,*) "----------------------------------------------------"
      write(*,*) "mh-tree :   ", real(mlh), real(mhh), real(alpha)
      write(*,*) "----------------------------------------------------"
      write(*,*) "mh-1loop:"
      if ((mh1.ne.119.9999d0).and.(mh2.ne.119.9999d0)) then
         write(*,*) " --> BEST  :", real(mh1), real(mh2), real(alefff1)
      else 
         write(*,*) " --> BEST  : Higgs sector not ok at 1-loop"
      endif
      write(*,*) "Yuk-approx :", real(mlhapp), real(mhhapp)
      if (((dabs(mlhapp - mh1).ge.10d0).or.
     $     (dabs(mhhapp - mh2).ge.10d0)).and.
     $    (mh1.ne.119.9999d0).and.(mh2.ne.119.9999d0)) then
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
         write(*,*) "possible numerical instability detected"
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
      endif
      write(*,*) "----------------------------------------------------"
      write(*,*) "mh-2loop:"
      if ((mh12.ne.119.9999d0).and.(mh22.ne.119.9999d0)) then
         write(*,*) " --> BEST  :", real(mh12),real(mh22), real(alefff2)
      else 
         write(*,*) " --> BEST  : Higgs sector not ok at 2-loop"
      endif
      write(*,*) "Yuk-approx :", real(mh1app), real(mh2app)
      if (((dabs(mh1app - mh12).ge.10d0).or.
     $     (dabs(mh2app - mh22).ge.10d0)).and.
     $    (mh1.ne.119.9999d0).and.(mh2.ne.119.9999d0)) then
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
         write(*,*) "possible numerical instability detected"
         write(*,*) "WARNING - WARNING - WARNING - WARNING"
      endif
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*) "charged Higgs, tree:", real(mhptree)
      write(*,*) "             1-loop:", real(mhp1l)


 99   continue

      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      call delrho(mst1, mst2, stt, msb1, msb2, stb, mgl, mmt, mbb, 
     $            gf, as, cf, gs, el, mz, mw,
     $            mlheff2, mhheff2, mma, alefff2, datan(ttb), 
     $            delrho1loop, delrho2loopgluon, delrho2loopgluino,
     $            delrho2loopmt2yuk, delrho2loopmt2yuksm)
      delrhores = delrho1loop + delrho2loopgluon + delrho2loopgluino +
     $            (delrho2loopmt2yuk - delrho2loopmt2yuksm)
      if (dabs(delrhores).gt.delrholimit) then
         write(*,*) 'WARNING: Delta rho > experimental limit'
      endif
      write(*,*) 'Delta rho 1-loop         : ', delrho1loop
      write(*,*) 'Delta rho 2-loop (gluon) : ', delrho2loopgluon
      write(*,*) 'Delta rho 2-loop (gluino): ', delrho2loopgluino
      write(*,*) 'Delta rho 2-loop (MT4eff): ', 
     $               (delrho2loopmt2yuk - delrho2loopmt2yuksm)
      write(*,*) 'Delta rho total          : ', delrhores
      write(*,*) "----------------------------------------------------"


      write(*,*)
      write(*,*) "collected WARNINGS:"
      if (   (real(mst1).le.65d0).or.(real(mst2).le.65d0).
     $    or.(real(msb1).le.65d0).or.(real(msb2).le.65d0)) then
         write(*,*) "experimental excluded sfermion masses"
      endif
      if ((mh1.eq.119.9999d0).or.(mh2.eq.119.9999d0)) then
         write(*,*) " --> BEST  : Higgs sector not ok at 1-loop"
      endif
c      if (mlheff1.eq.0d0) then
c         write(*,*) "zero mom.  : Higgs sector not ok at 1-loop"
c      endif
      if (((dabs(mlhapp - mh1).ge.10d0).or.
     $     (dabs(mhhapp - mh2).ge.10d0)).and.
     $    (mh1.ne.119.9999d0).and.(mh2.ne.119.9999d0)) then
         write(*,*) "possible numerical instability detected"
      endif
      if ((mh12.eq.119.9999d0).or.(mh22.eq.119.9999d0)) then
         write(*,*) " --> BEST  : Higgs sector not ok at 2-loop"
      endif
c      if (mlheff2.eq.0d0) then
c         write(*,*) "zero mom.  : Higgs sector not ok at 2-loop"
c      endif
      if (((dabs(mh1app - mh12).ge.10d0).or.
     $     (dabs(mh2app - mh22).ge.10d0)).and.
     $    (mh1.ne.119.9999d0).and.(mh2.ne.119.9999d0)) then
         write(*,*) "possible numerical instability detected"
      endif
      if (dabs(delrhores).gt.delrholimit) then
         write(*,*) 'Delta rho > experimental limit'
      endif
      write(*,*)
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      error = 0
      write(*,*) "General WARNINGS:"
      write(*,*)
      if (dmb.gt.0.8d0) then
         write(*,*) "Delta mb > 0.8"
         error = 1
      endif
      if (dmb.lt.-0.8d0) then
         write(*,*) "Delta mb < -0.8"
         error = 1
      endif
      if (dabs(mtlr)/dsqrt(msusytl*msusytr).ge.2.3d0) then
         write(*,*) "|Xt|/Msusy > 2.3"
         error = 1
      endif
      if (error.eq.1) then 
         write(*,*) "possible implication: numerical instability"
      endif
      if (sps.eq.11) then
         write(*,*) "WARNING: SPS1b implemented only as approximation"
      endif
      write(*,*) "----------------------------------------------------"
      write(*,*) "----------------------------------------------------"
      write(*,*)



      goto 100
2345  CONTINUE

      AML = real(mh12)
      AMH = real(mh22)
      ALPHA0 = real(alefff2)
      SA = DSIN(ALPHA0)
      CA = DCOS(ALPHA0)

      return
      end

c --------------------------------------------------------------
c --------------------------------------------------------------
c --------------------------------------------------------------


      subroutine feynhiggssub(mh1,mh2,mh12,mh22)

      implicit real*8(a-z)
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------
      complex*16 P1se, P1se1, P2se, P2se1, P1P2se, P1P2se1
      complex*16 p1setl, p2setl, p1p2setl
      complex*16 LLExpansionP2MTrun, LLExpansionP2MTrun1 
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      real*8 mma,ttb,mmt,mmm,mmue,mmsusy,au,ad,mh1,mh2,mh12,mh22
      real*8 l1, l2, l3, l4
c      double precision mlhlle1, mlhlle2, 
c     $       mlhllediag1, mlhllediag2, mhhllediag1, mhhllediag2
      integer ii,j,k,ic,pri,naeh,selec,selec2,selec4,selec5,selec6
      integer lleselec1,mpgut
      double precision m1zl, m1ol, m1tl, m2zl, m2ol, m2tl, 
     $                 mlhapp, mhhapp, mh2app, mh1app, 
     $                 azl, aol, atl,
     $                 mp12, mp22, mp1p22, mp12ol, mp22ol, mp1p22ol,
     $                 mp12tl, mp22tl, mp1p22tl,
     $                 mdiag1, mdiag2, mixang,
     $                 renlh, renhh, renxh, renlhsub, renhhsub, renxhsub
c      double precision allle1, allle2
      double precision alphasmz, alphas, mtrun, xttilde, vvv, ttt, 
     $                 delmlhsq, ms1, ms2, fac, alem
      double precision yepsilon,ymuee,ypi,ymz,ymw,mgf,yas,yalphasmz,
     $                 ycf,yeps,yeins,ymbb
      double precision lhseol, hhseol, xhseol, p1seol, p2seol, p1p2seol,
     $                 mlheff1sq, mhheff1sq, mlheff2sq, mhheff2sq,
     $                 mlheff1, mhheff1, aleff1, alefff1,
     $                 mlheff2, mhheff2, aleff2, alefff2
      complex*16 yi
      double precision mhptree, mhp1l

      common /smpara1/ ymw,ymz,mmt
      common /smpara2/ yepsilon,ymuee,ypi,ygf,ycf,yeps,yi,yeins,
     $                 yas,yalphasmz,ymbb
      common /susypara/ ttb,mma,mmm,mmue,au,ad
      common /gutrel/ mpgut
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common / err/ ic
      common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      common /selftl/p1setl,p2setl,p1p2setl
c      common /lle/ mtlr, mblr, mlhlle1, mlhlle2,
c     $             mlhllediag1, mlhllediag2, mhhllediag1, mhhllediag2
c      common /lle2/ allle1, allle2
      common /lleselec/ lleselec1
      common /mhapp/ mlhapp, mhhapp, mh2app, mh1app
      common /zeromom/ mlheff1, mhheff1, alefff1, 
     $                 mlheff2, mhheff2, alefff2
      common /chargedhiggs/ mhptree, mhp1l
      double precision mtpole, mbpole
      common /polemasses/ mtpole, mbpole
      integer alphatsq
      common /alphat2/alphatsq
      double precision delp1, delp2, delp1p2

      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum

c passing over the SM parameters
      epsilon = yepsilon
      muee = ymuee
      pi = ypi
      mz = ymz
      mw = ymw
      gf = ygf
      alphasmz = yalphasmz
      cf = ycf
      eps = yeps
      i = yi
      eins = yeins
      mbb = ymbb

      ppi = pi

      mmz  = mz
      mz2 = mmz**2
      mmz2 = mmz**2
      mmw =  mw
      mw2 = mmw**2
      mmw2 = mmw**2
      cw2 = (mmw/mmz)**2
      ccw2 = (mmw/mmz)**2
      cw = dsqrt(ccw2)
      ccw = dsqrt(ccw2)
      sw2 = 1.d0 - ccw2
      ssw2 = 1.d0 - ccw2
      sw = dsqrt(ssw2)
      ssw = dsqrt(ssw2)
      el = dsqrt(8d0*gf*MMW**2*(1-MMW**2/MMZ**2)/dsqrt(2d0))
      elec = el
      elec2 = el**2
      alem = el**2/(4d0 * pi) 
      gmue = gf

      mtt=mmt
      mtt2 = mtt**2
      mt=mmt
      mb  = mbb
      mbb2 = mbb**2

      alphas = alphasmz/(1d0 + (11d0 - 10d0/3d0)/(4d0*ppi) * alphasmz
     $                       * dlog(mmt**2/mmz**2))
      as = alphas
      gs = dsqrt(4*ppi*as)
      call mtrunning(mmt, mtrun, alphas)

      maa=mma
      ma = mma

      mlheff1 = 0d0
      mlheff2 = 0d0


c softbreaking parameter

c-sh  Uebergabe
      mq = msusytl
      mq2 = msusytl**2
      mu2 = msusytr**2
      mb2 = msusybl**2  ! should be equal to mq2
c      mb2 = msusytl**2  ! should be equal to mq2
      md2 = msusybr**2
      mf2  = msusytl**2
      mfd2 = mf2
c      write(*,*) 'Msusy_top/bot_L:', real(msusytl), real(msusybl)

c --> passing over the parameters to Dabelstein's program
      mm=mmm
c --> mue convention:
c     Here the mue convention is fixed. Internally FeynHiggs works with
c     the MtLR = At + mue CTb. Due to the '-' sign below, externally
c     i.e. in the front-end one can work with the MtLR = At - mue CTb
c     convention.
      mu = -mmue
      mue = -mmue
      if (mpgut.eq.1) then
         mp = 5.d0/3.d0 * mm * (ssw/ccw)**2
c         write(*,*) 'M1:', real(mp)
      elseif (mpgut.ne.2) then
         write(*,*) 'M1 is not defined !!!'
         stop
      endif
      mgll = mgl

      mel = 0.51d-3
      mmu = 0.1057d0
      mta = 1.777d0
      mup = 0.0415d0
      mdn = 0.04151d0
      mch = 1.50d0
      mst = 0.150d0

c-sh  Uebergabe
      beta = datan(ttb)
      c2b = dcos(2d0*datan(ttb))
      cb = dcos(datan(ttb))
      sb = dsin(datan(ttb))

      mlh2 = 0.5d0*(maa**2+mmz**2 - dsqrt((maa**2+mmz**2)**2 - 4.d0*
     &              mmz**2*maa**2*dcos(2.d0*beta)**2))
      mlh = dsqrt(mlh2)
      Mhh2 = 0.5d0*(maa**2+mmz**2 + dsqrt((maa**2+mmz**2)**2 -
     &              4.d0*mmz**2*maa**2*dcos(2.d0*beta)**2))
      mhh = dsqrt(mhh2)
      mhp = dsqrt(maa**2 + mmw**2)
      mhptree = mhp

csh   new formula: eq (7.5) from S.H.'s PhD thesis
      alpha = datan((-(mma**2 + mmz**2) * sb * cb)/
     $              (mmz**2 * cb**2 + mma**2 * sb**2 - mlh**2))

c-sh  Uebergabe
      mssupq=au
c-sh  Uebergabe
      mssdnq=ad
      mssdnl =   mssdnq


      call mix
      call def2
      if (delmbresum.eq.2) then
         dmb = deltambnoew(alphas, mst1, mst2, msb1, msb2,
     $                         au, ad, mgl, mmt, mbb, mmue, ttb)
c        write(*,*) 'evaluating Delta mb for Higgs masses:', real(dmb)
         call def4(msusybl, msusybr, ad - mmue * ttb, mbb/(1d0+dmb), 
     $             ttb, msb1dmb, msb2dmb, stbdmb, 2)
         tsbdmb = dasin(stbdmb)
c        write(*,*) 'MSb1, MSb2, stb:'
c        write(*,*) real(msb1dmb), real(msb2dmb), real(stbdmb)
         if ((msb1dmb.le.0d0).or.(msb2dmb.le.0d0)) then
            msb1dmb = msb1
            msb2dmb = msb2
            stbdmb = stb
            tsbdmb = dasin(stbdmb)
            write(*,*) 'WARNING: unphysical sbottom masses'
            write(*,*) ' --> using sbottom masses without Delta mb'//
     $                 'corrections'
            write(*,*) ' ==> results inconsistent!'
         endif
      endif

      if ((mst1.le.0d0).or.(mst2.le.0d0).or.
     $    (msb2.le.0d0).or.(msb2.le.0d0)) then
         mh1 = 119.9999d0
         mh2 = 119.9999d0
         mh12 = 119.9999d0
         mh22 = 119.9999d0
         goto 999
      endif


c --> find approximation for light Higgs mass
      p1se = p1se1()
      p2se = p2se1()
      p1p2se = p1p2se1()
c      write(*,*) 'self-energies at one loop:'
c      write(*,*) real(p1se), real(p2se), real(p1p2se)
      mlhapp = (.5d0 * (ma**2 + mz**2 - p2se - p1se) -
     $     (.25d0 * ((ma**2 + mz**2)**2 + (p2se - p1se)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1se - p2se) * dcos(2d0*beta) * (ma**2 - mz**2) +
     $     p1p2se * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2se**2 )**.5d0 )**.5d0
      mhhapp = (.5d0 * (ma**2 + mz**2 - p2se - p1se) +
     $     (.25d0 * ((ma**2 + mz**2)**2 + (p2se - p1se)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1se - p2se) * dcos(2d0*beta) * (ma**2 - mz**2) +
     $     p1p2se * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2se**2 )**.5d0 )**.5d0
c      write(*,*) "light higgs approximation: ", real(mlhapp)



c      write(*,*) 'vor mt-Aenderung'
      mtold = mt
c --> new top mass for two-loop contribution
      if (selec2.ge.2) then
c      write(*,*) 'mt = ', real(mt), real(mtt), real(mmt)
c     write(*,*) "using running mt for two-loop contribution:", mtrun
      mt = mtrun
      if (selec4.eq.2) then
         call def2
c        write(*,*) '... also for mt in Stop mass matrix'
      endif
      endif
      if ((mst1.le.0d0).or.(mst2.le.0d0).or.
     $    (msb2.le.0d0).or.(msb2.le.0d0)) then
c         write(*,*) 'Problem in Sfermion sector:'
c         write(*,*) 'MSt1, MSt2:', real(mst1), real(mst2)
c         write(*,*) 'MSb1, MSb2:', real(msb1), real(msb2)
         mh1 = 119.9999d0
         mh2 = 119.9999d0
         mh12 = 119.9999d0
         mh22 = 119.9999d0
         goto 999
      endif
c --> end of new parameter definition

c      write(*,*) 'vor 2-loop Berechnung'
c     calculation of the 2-loop contribution
      if (selec5.eq.1) then
         p1setl = 0d0
         p1p2setl = 0d0
      if (msusytl.eq.msusytr) then
         ms2 = dsqrt(msusytl**2 + mtrun**2) 
      else
         ms2 = msfkt(msusytl, msusytr, mtrun)
      endif                                                                   
      LLExpansionP2MTrun = LLExpansionP2MTrun1(mtrun,ms2)      
           p2setl = cf*el**2*as/
     $            (3d0 * 2d0**8 * sw**2 * pi**3 * sb**2)*
     $            mtrun**2/mw**2 * dreal(llexpansionp2mtrun)

c$$$         write(*,*) 'self-energies at two loop:'
c$$$         write(*,*) p1setl, p2setl, p1p2setl
c$$$         write(*,*) real(p1setl), real(p2setl), real(p1p2setl)
c$$$         write(*,*) real(mt), real(mst1), real(mst2), real(stt)
c$$$         write(*,*) real(cf), real(el), real(gs), real(sw), real(pi),
c$$$     $              real(sb), real(mw), real(mgl), real(eps)
c$$$         write(*,*) sb**2, (real(p2setl) * sb**2 *
c$$$     $   (1d0 + (4d0*MZ**2*(1d0 - 2d0*sb**2)*(1d0 - sb**2))/MA**2))
      else
         p1setl = 0d0
         p1p2setl = 0d0
         p2setl = 0d0
      endif


c----------------------------------------------------------------

      delmlhsq = 0d0
      if (selec2.ge.3) then
c     including the leading Yukawa term for the light Higgs mass
c     this term is taken from Carena, Espinoza, Quiros, Wagner
c     Nucl. Phys. B461 (1996) 407

      if (mh12.ne.119.9999d0) then

c     write(*,*) 'including two-loop Yukawa term'

      if (alphatsq.eq.1) then
         write(*,*) "... using LL approximation (RGiEP)"
      if (selec4.eq.2) then
         write(*,*) "... also with running mt in stop mass matrix"
      endif

      xttilde = (((MSt2**2 - MSt1**2)/(4d0*mtrun**2) * 
     $            (2d0 * stt * dsqrt(1d0 - stt**2))**2 )**2 *
     $           (2d0 - (MSt2**2 + MSt1**2)/(MSt2**2 - MSt1**2) *
     $                  dlog(MSt2**2/MSt1**2)) +
     $           (MSt2**2 - MSt1**2)/(2d0 * mtrun**2) *
     $            (2d0 * stt * dsqrt(1d0 - stt**2))**2 *
     $                  dlog(MSt2**2/MSt1**2) )
      ttt = .5d0 * dlog(MSt2**2 * MSt1**2/mtrun**4)
      vvv = 174.1d0
      delmlhsq = ( 3d0/(4d0 * ppi**2) * mtrun**4/vvv**2 *
     $            ( 1d0/(16d0 * ppi**2) * 3d0/2d0 * mtrun**2/vvv**2 *
     $             ( xttilde * ttt + ttt**2)))

      delm22 = delmlhsq/dsin(beta)**2

      p2setl = p2setl -1d0/sb**2 * delmlhsq

c --> new routine from hep-ph/0112177
      else
c        write(*,*) "... using full calculation (EP 2-loop)"
         call BDSZHiggs(mtrun**2, mma**2, msusybl**2,
     $                  mst1**2, mst2**2, stt, dsqrt(1d0-stt**2),
     $                  mmt**2, mue, ttb, 246.218d0**2, 1, 
     $                  delp1, delp2, delp1p2)
         p1setl = p1setl - delp1
         p2setl = p2setl - delp2
         p1p2setl = p1p2setl - delp1p2
         delmlhsq = sb**2 * delp2

         delm22 = delmlhsq/dsin(beta)**2

      endif
c --> end of new routine from hep-ph/0112177

      endif
      endif

c----------------------------------------------------------------
      

c --> two-loop approximations for the Higgs masses
c      write(*,*) 'pure top:', real(p1se), real(p2se), real(p1p2se)
      mh1app = (.5d0 * (ma**2 + mz**2 
     $     - (p2se + p2setl) - (p1se + p1setl)) -
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $     + ((p2se + p2setl) - (p1se + p1setl))**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * ((p1se + p1setl) 
     $     - (p2se + p2setl)) * dcos(2d0*beta) * (ma**2 - mz**2) +
     $     (p1p2se + p1p2setl) * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     (p1p2se + p1p2setl)**2 )**.5d0 )**.5d0
      mh2app = (.5d0 * (ma**2 + mz**2 
     $     - (p2se + p2setl) - (p1se + p1setl)) +
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $     + ((p2se + p2setl) - (p1se + p1setl))**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * ((p1se + p1setl) 
     $     - (p2se + p2setl)) * dcos(2d0*beta) * (ma**2 - mz**2) +
     $     (p1p2se + p1p2setl) * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     (p1p2se + p1p2setl)**2 )**.5d0 )**.5d0

  




c --> setting back the top mass to its original value for next calculation
      mt = mtold
c      write(*,*) 'mt = ', real(mt), real(mtt), real(mmt)
      if (selec4.eq.2) then
c --> setting back the sfermion masses
         call def2
      endif
c --> end of setting back the parameters

      

c --> charged Higgs at one loop
      mhp1l = dsqrt(ma**2 + mw**2 + 3d0/(4d0 * pi) * alem/(sw2 * mw2) *
     $            (  2d0 * mt**2 * mb**2/(sb**2 * cb**2) 
     $             - mw2 * (mt**2/sb**2 + mb**2/cb**2) 
     $            + 2d0/3d0 * mw2**2) * dlog(msusytl/mt)
     $            + mw2/(6d0 * pi) * 15d0 * alem/cw2 * dlog(msusytl/mw))


c --> neutral Higgs masses at two loop
      call mlhren(0.01d0, lhseol)
      lhseol = lhseol + mlh**2
      call mhhren(0.01d0, hhseol)
      hhseol = hhseol + mhh**2
      call mxhren(0.01d0, xhseol)
      xhseol = xhseol 
      p1seol =   dcos(alpha)**2 * hhseol 
     $         + dsin(alpha)**2 * lhseol
     $         - 2d0 * dsin(alpha) * dcos(alpha) * xhseol
      p2seol =   dsin(alpha)**2 * hhseol
     $         + dcos(alpha)**2 * lhseol
     $         + 2d0 * dsin(alpha) * dcos(alpha) * xhseol
      p1p2seol =   dsin(alpha) * dcos(alpha) * (hhseol - lhseol)
     $           + (dcos(alpha)**2 - dsin(alpha)**2) * xhseol

      mlheff1sq = (.5d0 * (ma**2 + mz**2 
     $                   - p2seol - p1seol) -
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + (p2seol - p1seol)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1seol - p2seol) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     p1p2seol * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2seol**2 )**.5d0 )
      if (mlheff1sq.gt.0d0) then
         mlheff1 = (.5d0 * (ma**2 + mz**2 
     $                   - p2seol - p1seol) -
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + (p2seol - p1seol)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1seol - p2seol) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     p1p2seol * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2seol**2 )**.5d0 )**.5d0
      else
c        write(*,*) 'WARNING: error at effective light Higgs' //
c    $              ' mass at 1 loop'
         mlheff1 = 0d0
      endif
      mhheff1sq = (.5d0 * (ma**2 + mz**2 
     $                   - p2seol - p1seol) +
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + (p2seol - p1seol)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1seol - p2seol) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     p1p2seol * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2seol**2 )**.5d0 )
      if (mhheff1sq.gt.0d0) then
         mhheff1 = (.5d0 * (ma**2 + mz**2 
     $                   - p2seol - p1seol) +
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + (p2seol - p1seol)**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * (p1seol - p2seol) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     p1p2seol * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     p1p2seol**2 )**.5d0 )**.5d0
      else
c        write(*,*) 'WARNING: error at effective heavy Higgs' //
c    $              ' mass at 1 loop'
         mhheff1 = 0d0
      endif
c      write(*,*) 'after mhheff1'
      if ((mlheff1.ne.0d0).and.(mhheff1.ne.0d0)) then
         aleff1 = datan((-(ma**2 + mz**2) * sb * cb - p1p2seol)/
     $            (mz**2 * cb**2 + ma**2 * sb**2 - p1seol - mlheff1**2))
      else
         aleff1 = 0d0
      endif
      alefff1 = aleff1


      mlheff2 = (.5d0 * (ma**2 + mz**2 
     $                   - (p2seol+p2setl) - (p1seol+p1setl)) -
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + ((p2seol+p2setl) - (p1seol+p1setl))**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * ((p1seol+p1setl) - (p2seol+p2setl)) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     (p1p2seol+p1p2setl) * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     (p1p2seol+p1p2setl)**2 )**.5d0 )**.5d0
      mhheff2 = (.5d0 * (ma**2 + mz**2 
     $                   - (p2seol+p2setl) - (p1seol+p1setl)) +
     $     (.25d0 * ((ma**2 + mz**2)**2 
     $               + ((p2seol+p2setl) - (p1seol+p1setl))**2) -
     $      ma**2 * mz**2 * (dcos(2d0*beta))**2 +
     $      .5d0 * ((p1seol+p1setl) - (p2seol+p2setl)) * dcos(2d0*beta) 
     $                           * (ma**2 - mz**2) +
     $     (p1p2seol+p1p2setl) * dsin(2d0*beta) * (ma**2 + mz**2) +
     $     (p1p2seol+p1p2setl)**2 )**.5d0 )**.5d0
      aleff2 = datan((-(ma**2 + mz**2) * sb * cb 
     $                 - (p1p2seol+dreal(p1p2setl)))/
     $               (mz**2 * cb**2 + ma**2 * sb**2 
     $                 - (p1seol+dreal(p1setl)) - mlheff2**2))
      alefff2 = aleff2

c     write(*,*) 'effective result:', real(ma), real(mz), real(beta)
c     write(*,*) real(mlheff1), real(mhheff1), real(aleff1)
c     write(*,*) real(mlheff2), real(mhheff2), real(aleff2)
      mh1 = mlheff1
      mh2 = mhheff1
      mh12 = mlheff2
      mh22 = mhheff2

c ----------------------------------------------------------------



 999  continue





      end

c ===========================================================

      subroutine mlhren (x, mout)
c
      implicit double precision (a-z)
c
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      common /sigmaasave/siasave, dsiasave
c
      q2 = x**2
c      write(*,*)'q2=',q2
c
c renorm.light scalar higgs mass
c
      call sigmaa  (maa**2 ,sab,sas,saf,sac,sat)
      if (selec.eq.4) then
         selfa = (sab + sas + saf      )
      elseif (selec.eq.3) then
         selfa = (sab + sas + saf + sac)
      elseif (selec.eq.2) then
         selfa = (      sas + saf      )
      elseif (selec.eq.1) then
         selfa = (      sas + saf      )
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif
c      siasave = selfa
c MF 200801: selfa is now really the A-Selfenergy and not
c Dabelstein's combination of sigmaa and dsigmaa
c
c
      call tadlh (0.d0, sltb,slts,sltf,sltc,sltt)
c      write(*,*)'tadlh:',sltb,slts,sltf,sltc,sltt
      if (selec.eq.4) then
         tadlt = sltb + slts + sltf
      elseif (selec.eq.3) then
         tadlt = sltb + slts + sltf + sltc
      elseif (selec.eq.2) then
         tadlt =        slts + sltf         
      elseif (selec.eq.1) then
         tadlt =        slts + sltf         
c         write(*,*) 'tadlt:', slts, sltf
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif
c     tadlt = slts + sltf
c     tadlt = sltt
c
      call tadhh (0.d0, shtb,shts,shtf,shtc,shtt)
c      write(*,*)'tadhh:',shtb,shts,shtf,shtc,shtt
      if (selec.eq.4) then
         tadht = shtb + shts + shtf
      elseif (selec.eq.3) then
         tadht = shtb + shts + shtf + shtc
      elseif (selec.eq.2) then
         tadht =        shts + shtf         
      elseif (selec.eq.1) then
         tadht =        shts + shtf         
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif
c     tadht = shts + shtf
c     tadht = shtt
c
      call sigmaz (mmz**2 ,szb,szs,szf,szc,szt)
c      write(*,*)'sigmaz:',szb,szs,szf,szc,szt
      if (selec.eq.4) then
         selfz = szb + szs + szf
      elseif (selec.eq.3) then
         selfz = szb + szs + szf + szc
      elseif (selec.eq.2) then
         selfz =       szs + szf
      elseif (selec.eq.1) then
         selfz =       szs + szf
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif


      dtb = (dzh0() - dzhh())/(2.0D0*dcos(2.0D0*alpha))
c MF 200801: tan(beta) Counterterm dtb contains the missing mudim-
c terms necessary for MSbar.

      deltamh = dcos(beta-alpha)**2 * selfa +
     &          elec/(2.d0*ssw*mmw) * dsin(beta-alpha)**2*dcos(beta-
     &          alpha) * tadht -
     &          elec/(2.d0*ssw*mmw) * dsin(beta-alpha) * (1.d0 +
     &          dcos(beta-alpha)**2) * tadlt +
     &          dsin(alpha+beta)**2 * selfz -
     &          dtb * dsin(2*beta) *
     &          (maa**2 * dsin(beta-alpha) * dcos(beta-alpha) -
     &           mmz**2 * dsin(alpha+beta) * dcos(alpha+beta))
c MF 200801: deltamh is now written in a general form including dtb
c      write(*,*) "nach deltamh",deltamh
c

      call sigmalh (q2, slhb,slhs,slhf,slhc,slht)
c      write(*,*)'sigmalh:',slhb,slhs,slhf,slhc,slht
      if (selec.eq.4) then
         selfh =  slhb + slhs + slhf
      elseif (selec.eq.3) then
         selfh =  slhb + slhs + slhf + slhc
      elseif (selec.eq.2) then
         selfh =         slhs + slhf 
      elseif (selec.eq.1) then
         selfh =         slhs + slhf 
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif

      xxlh = selfh + dzh0()*(q2 - mlh**2) - deltamh
c MF 210801: Counterterm dZHH contains the missing mudim-terms
c necessary for MSbar.

c
      mout = xxlh + q2 - mlh**2
c
      
      
      return
      end
c
c ============================================================
c
      subroutine mhhren (x, mout)
c
      implicit double precision (a-z)
c
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
c
      q2 = x**2
c
c renorm. heavy scalar higgs mass
c
      call sigmaa  (maa**2 ,sab,sas,saf,sac,sat)
      if (selec.eq.4) then
         selfa = (sab + sas + saf      )
      elseif (selec.eq.3) then
         selfa = (sab + sas + saf + sac)
      elseif (selec.eq.2) then
         selfa = (      sas + saf      )
      elseif (selec.eq.1) then
         selfa = (      sas + saf      )
      else
         write(*,*) "Error in mhhren: selec out or range"
      endif
c MF 200801: selfa is now really the A-Selfenergy and not
c Dabelstein's combination of sigmaa and dsigmaa
c
      call tadlh (0.d0, sltb,slts,sltf,sltc,sltt)
      if (selec.eq.4) then
         tadlt = sltb + slts + sltf
      elseif (selec.eq.3) then
         tadlt = sltb + slts + sltf + sltc
      elseif (selec.eq.2) then
         tadlt =        slts + sltf         
      elseif (selec.eq.1) then
         tadlt =        slts + sltf         
      else
         write(*,*) "Error in mhhren: selec out or range"
      endif

      call tadhh (0.d0, shtb,shts,shtf,shtc,shtt)
      if (selec.eq.4) then
         tadht = shtb + shts + shtf
      elseif (selec.eq.3) then
         tadht = shtb + shts + shtf + shtc
      elseif (selec.eq.2) then
         tadht =        shts + shtf         
      elseif (selec.eq.1) then
         tadht =        shts + shtf         
      else
         write(*,*) "Error in mhhren: selec out or range"
      endif

      call sigmaz (mmz**2 ,szb,szs,szf,szc,szt)
      if (selec.eq.4) then
         selfz = szb + szs + szf
      elseif (selec.eq.3) then
         selfz = szb + szs + szf + szc
      elseif (selec.eq.2) then
         selfz =       szs + szf
      elseif (selec.eq.1) then
         selfz =       szs + szf
      else
         write(*,*) "Error in mhhren: selec out or range"
      endif


      dtb = (dzh0() - dzhh())/(2.0D0*dcos(2.0D0*alpha))
c MF 200801: tan(beta) Counterterm dtb contains the missing mudim-
c terms necessary for MSbar.


      deltamh = dsin(beta-alpha)**2 * selfa -
     &          elec/(2.d0*ssw*mmw) * dcos(beta-alpha) * (1.d0 +
     &          dsin(beta-alpha)**2 ) * tadht +
     &          elec/(2.d0*ssw*mmw) * dcos(beta-alpha)**2 *
     &          dsin(beta-alpha) * tadlt +
     &          dcos(beta+alpha)**2 * selfz +
     &          dtb * dsin(2*beta) *
     &          (maa**2 * dsin(beta-alpha) * dcos(beta-alpha) -
     &           mmz**2 * dsin(alpha+beta) * dcos(alpha+beta))
c MF 200801: deltamh is now written in a general form including dtb
c
      call sigmahh (q2, slhb,slhs,slhf,slhc,slht)
      if (selec.eq.4) then
         selfH =  slhb + slhs + slhf
      elseif (selec.eq.3) then
         selfH =  slhb + slhs + slhf + slhc
      elseif (selec.eq.2) then
         selfH =         slhs + slhf 
      elseif (selec.eq.1) then
         selfH =         slhs + slhf 
c         write(*,*) 'subroutine mhhren:', real(slhs), real(slhf) 
      else
         write(*,*) "Error in mhhren: selec out or range"
      endif

      xxlh = selfH + dzhh()*(q2 - mhh**2) - deltamh
c MF 210801: Counterterm dZHH contains the missing mudim-terms
c necessary for MSbar.

c
      mout = xxlh + q2 - mhh**2
c      write(*,*) 'mhhren:', real(dsqrt(q2)), real(xxlh), real(mhh**2)

      return
      end
c
c ====================================================================
c
      subroutine mxhren (x, mout)
c
      implicit double precision (a-z)
c
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      q2 = x**2
c
c renorm. scalar higgs mixing
c
      call sigmaa  (maa**2 ,sab,sas,saf,sac,sat)
      if (selec.eq.4) then
         selfa = (sab + sas + saf      )
      elseif (selec.eq.3) then
         selfa = (sab + sas + saf + sac)
      elseif (selec.eq.2) then
         selfa = (      sas + saf      )
      elseif (selec.eq.1) then
         selfa = (      sas + saf      )
      else
         write(*,*) "Error in mxhren: selec out or range"
      endif
c MF 200801: selfa is now really the A0-Selfenergy and not
c Dabelstein's combination of sigmaa and dsigmaa
c
      call tadlh (0.d0, sltb,slts,sltf,sltc,sltt)
      if (selec.eq.4) then
         tadlt = sltb + slts + sltf
      elseif (selec.eq.3) then
         tadlt = sltb + slts + sltf + sltc
      elseif (selec.eq.2) then
         tadlt =        slts + sltf         
      elseif (selec.eq.1) then
         tadlt =        slts + sltf         
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif

      call tadhh (0.d0, shtb,shts,shtf,shtc,shtt)
      if (selec.eq.4) then
         tadht = shtb + shts + shtf
      elseif (selec.eq.3) then
         tadht = shtb + shts + shtf + shtc
      elseif (selec.eq.2) then
         tadht =        shts + shtf         
      elseif (selec.eq.1) then
         tadht =        shts + shtf         
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif

      call sigmaz (mmz**2 ,szb,szs,szf,szc,szt)
      if (selec.eq.4) then
         selfz = szb + szs + szf
      elseif (selec.eq.3) then
         selfz = szb + szs + szf + szc
      elseif (selec.eq.2) then
         selfz =       szs + szf
      elseif (selec.eq.1) then
         selfz =       szs + szf
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif

      dtb = (dzh0() - dzhh())/(2.0D0*dcos(2.0D0*alpha))
c MF 200801: tan(beta) Counterterm dtb contains the missing mudim-
c terms necessary for MSbar.


      deltamh = dsin(alpha-beta)*dcos(alpha-beta) * selfa -
     &          elec/(2.d0*ssw*mmw) * dsin(beta-alpha)**3 * tadht -
     &          elec/(2.d0*ssw*mmw) * dcos(beta-alpha)**3 * tadlt -
     &          dsin(beta+alpha)*dcos(beta+alpha) * selfz -
     &          dtb * dsin(beta) * dcos(beta) *
     &          (maa**2 * (dcos(beta-alpha)**2-dsin(beta-alpha)**2) +
     &           mmz**2 * (dcos(alpha+beta)**2-dsin(alpha+beta)**2))
c MF 200801: deltamh is now written in a general form including dtb

      call sigmaxh (q2, slhb,slhs,slhf,slhc,slht)
      if (selec.eq.4) then
         selfx =  slhb + slhs + slhf
      elseif (selec.eq.3) then
         selfx =  slhb + slhs + slhf + slhc
      elseif (selec.eq.2) then
         selfx =         slhs + slhf 
      elseif (selec.eq.1) then
         selfx =         slhs + slhf 
      else
         write(*,*) "Error in mlhren: selec out or range"
      endif

      xxlh = selfx + dsin(2*alpha)*dtb*(q2 - (mhh**2 + mlh**2)/2.0D0) -
     &   deltamh
c MF 210801: the field renormalization (the term prop. to q2) contains
c the missing mudim-terms necessary for MSbar. Note: dtb is used for
c convenience only, it has simply the same analytic form as dZHh.

c
      mout = xxlh
c
      return
      end

c----------------------------------------------------------------
      double precision function msfkt(msusytl, msusytr, mtrun)
 
      double precision msusytl, msusytr, mtrun
 
c$$$       msfkt = dsqrt(dsqrt(msusytl**2 * msusytr**2
c$$$     $                    + mt**2 * (msusytl**2 + msusytr**2) + mt**4
c$$$     $                    + mt**2 * mtlr**2) - mt**2)
      msfkt = dsqrt( dsqrt(  msusytl**2 * msusytr**2
     $         + mtrun**2 * (msusytl**2 + msusytr**2) + mtrun**4))
 
      end
 
 
c----------------------------------------------------------------
c
c --> MSbartoOnShellPPtopbot.f
c
c----------------------------------------------------------------
      
      subroutine MSbartoOnShellPPtopbot(mst1msbar,mst2msbar,sttmsbar,
     $     qt1, qt2, qts,
     $     msb1msbar, msb2msbar, stbmsbar,
     $     qb1, qb2, qbs,
     $     mtos, mbos, xmgl, asmz, mz,
     $     mst1os, mst2os, sttos,
     $     msb1os, msb2os, stbos)

      complex*16 ShiftMSbarOSMSt1sq, ShiftMSbarOSMSt1sq1
      complex*16 ShiftMSbarOSMSt2sq, ShiftMSbarOSMSt2sq1
      complex*16 ShiftMSbarOSstt, ShiftMSbarOSstt1
      complex*16 ShiftMSbarOSMSb1sq, ShiftMSbarOSMSb1sq1
      complex*16 ShiftMSbarOSMSb2sq, ShiftMSbarOSMSb2sq1
      complex*16 ShiftMSbarOSstb, ShiftMSbarOSstb1

c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c varcom3.h
c
      double precision msb1, msb2, stb, mb

      common /msbartoos3/ msb1, msb2, stb, mb
c -------------------------------------------------------------------

      double precision mst1msbar, mst2msbar, sttmsbar, qt1, qt2, qts,
     $                 mst1os,    mst2os,    sttos
      double precision msb1msbar, msb2msbar, stbmsbar, qb1, qb2, qbs,
     $                 msb1os,    msb2os,    stbos
      double precision mst1org, mst2org, sttorg,
     $                 msb1org, msb2org, stborg
      double precision mtos, mbos, asmz, as, mz, xmgl
      double precision shiftt1, shiftt2, shifttx
      double precision shiftb1, shiftb2, shiftbx
      integer itert, iterb
      integer msbariter
      common /msbarselec2/ msbariter
 
c$$$      write(*,*) 'variables in MSbartoOnShellPP:'
c$$$      write(*,*) real(mst1msbar), real(mst2msbar), real(sttmsbar),
c$$$     $           real(qt1), real(qt2), real(qts), real(mtos), 
c$$$     $           real(msb1msbar), real(msb2msbar), real(stbmsbar),
c$$$     $           real(qb1), real(qb2), real(qbs), real(mbos), 
c$$$     $           real(xmgl), real(asmz), real(mz)
      eps = 1d-6
      cf = 4d0/3d0
      pi = 3.14159265358979d0
      i = (0d0, 1d0)

      mst1 = mst1msbar
      mst2 = mst2msbar
      stt = sttmsbar
      mt = mtos
      msb1 = msb1msbar
      msb2 = msb2msbar
      stb = stbmsbar
      mb = mbos
      call alphasmt(asmz,mt,mz,as)
      gs = dsqrt(4d0 * pi * as)
      mgl = xmgl
      itert = 0
      iterb = 0

      mst1org = mst1
      mst2org = mst2
      sttorg = stt
      msb1org = msb1
      msb2org = msb2
      stborg = stb

c     write(*,*) 'MSbar to OS: Stop sector evaluation'
 80   continue

      mue = qt1
      ShiftMSbarOSMSt1sq = ShiftMSbarOSMSt1sq1()
      shiftt1 = dreal(ShiftMSbarOSMSt1sq)
c      write(*,*) 'shiftt1:', real(shiftt1)
      mue = qt2
      ShiftMSbarOSMSt2sq = ShiftMSbarOSMSt2sq1()
      shiftt2 = dreal(ShiftMSbarOSMSt2sq)
c      write(*,*) 'shiftt2:', real(shiftt2)
      mue = qts
      ShiftMSbarOSstt = ShiftMSbarOSstt1()
      shifttx = dreal(ShiftMSbarOSstt)
c      write(*,*) 'shifttx:', real(shifttx)

      if ((mst1org**2-shiftt1).ge.0d0) then
         mst1os = dsqrt(mst1org**2 - shiftt1)
      else
         mst1os = 0d0
         itert = 51
         write(*,*) 'no MSt1 OS'
      endif
      if ((mst2org**2-shiftt2).ge.0d0) then
         mst2os = dsqrt(mst2org**2 - shiftt2)
      else
         mst2os = 0d0
         itert = 51
         write(*,*) 'no MSt2 OS'
      endif
      sttos = sttorg - shifttx

      if (dabs(sttos).gt.1d0) then
      write(*,*) "MSbar to OnShell: |stt| > 1"
         if ((sttos-1d0).le.1d-2) then
            sttos = 1d0-1d-5
         elseif (dabs((-sttos-1d0)).le.1d-2) then
            sttos = -1d0+1d-5
         else
            sttos = 2d0
         endif
      endif

      if ((itert.le.50).and.(msbariter.eq.1)) then
c$$$         if ((dabs((mst1os - mst1)/mst1os).ge.1d-2).or.
c$$$     $       (dabs((mst2os - mst2)/mst2os).ge.1d-2).or.
c$$$     $       (dabs((sttos - stt)/sttos).ge.1d-2)) then
         if ((dabs((mst1os - mst1)).ge.1d0).or.
     $       (dabs((mst2os - mst2)).ge.1d0).or.
     $       (dabs((sttos - stt)).ge.1d-3)) then
            itert = itert + 1
            mst1 = mst1os
            mst2 = mst2os
            stt = sttos
c           write(*,*) 
c           write(*,*) "MSbar to OnShell: one more iteration:", itert
c           write(*,*)
c           write(*,*) "MSt1, MSt2, stt"
c           write(*,*) real(mst1msbar), real(mst2msbar), real(sttmsbar)
c           write(*,*) real(mst1os), real(mst2os), real(sttos)
c           write(*,*) real(shiftt1), real(shiftt2), real(shifttx)
            goto 80
         endif
         elseif ((mst1os.ne.0d0).and.(mst2os.ne.0d0).and.
     $           (msbariter.eq.1)) then
         write(*,*) 'WARNING: More than 50 iterations in OS Stop calc.'
      endif


c     write(*,*) 'MSbar to OS: Sbottom sector evaluation'
 85   continue

      mue = qb1
      ShiftMSbarOSMSb1sq = ShiftMSbarOSMSb1sq1()
      shiftb1 = dreal(ShiftMSbarOSMSb1sq)
      mue = qb2
      ShiftMSbarOSMSb2sq = ShiftMSbarOSMSb2sq1()
      shiftb2 = dreal(ShiftMSbarOSMSb2sq)
      mue = qbs
      ShiftMSbarOSstb = ShiftMSbarOSstb1()
      shiftbx = dreal(ShiftMSbarOSstb)

      msb1os = dsqrt(msb1org**2 - shiftb1)
      msb2os = dsqrt(msb2org**2 - shiftb2)
      stbos = stborg - shiftbx
c$$$      if (dabs(stbos).gt.1d0) then
c$$$         if ((stbos-1d0).le.1d-5) then
c$$$            stbos = 1d0-1d-5
c$$$         elseif (dabs((-stbos-1d0)).le.1d-5) then
c$$$            stbos = -1d0+1d-5
c$$$         else
c$$$            stbos = 2d0
c$$$         endif
c$$$      endif

      if ((iterb.le.50).and.(msbariter.eq.1)) then
         if ((dabs((msb1os - msb1)/msb1os).ge.1d-2).or.
     $       (dabs((msb2os - msb2)/msb2os).ge.1d-2).or.
     $       (dabs((stbos - stb)/stbos).ge.1d-2)) then
            iterb = iterb + 1
            msb1 = msb1os
            msb2 = msb2os
            stb = stbos
c           write(*,*) 
c           write(*,*) "MSbar to OnShell: one more iteration:", iterb
c           write(*,*)
c           write(*,*) "MSb1, MSb2, stb"
c           write(*,*) real(msb1msbar), real(msb2msbar), real(stbmsbar)
c           write(*,*) real(msb1os), real(msb2os), real(stbos)
c           write(*,*) real(shiftb1), real(shiftb2), real(shiftbx)
            goto 85
         endif
         elseif ((msb1os.ne.0d0).and.(msb2os.ne.0d0).and.
     $           (msbariter.eq.1)) then
         write(*,*) 'WARNING: More than 20 iterations in OS Sbot calc.'
      endif



c     write(*,*) 
c     write(*,*) "MSbar to OnShell"
c     write(*,*) "================"
c     write(*,*)
c     write(*,*) "MSt1, MSt2, stt"
c     write(*,*) real(mst1msbar), real(mst2msbar), real(sttmsbar)
c     write(*,*) real(mst1os), real(mst2os), real(sttos)
c     write(*,*) real(shiftt1), real(shiftt2), real(shifttx)
c     write(*,*) 
c     write(*,*) "MSb1, MSb2, stb"
c     write(*,*) real(msb1msbar), real(msb2msbar), real(stbmsbar)
c     write(*,*) real(msb1os), real(msb2os), real(stbos)
c     write(*,*) real(shiftb1), real(shiftb2), real(shiftbx)
c     write(*,*) 

 100  continue

      end

c----------------------------------------------------------------
c
c --> MSbartoOnShellPP.f
c
c----------------------------------------------------------------
      
      subroutine MSbartoOnShellPP(mst1msbar,mst2msbar,sttmsbar,
     $     q1, q2, qs,
     $     mtos, xmgl, asmz, mz,
     $     mst1os, mst2os, sttos)

      complex*16 ShiftMSbarOSMSt1sq, ShiftMSbarOSMSt1sq1
      complex*16 ShiftMSbarOSMSt2sq, ShiftMSbarOSMSt2sq1
      complex*16 ShiftMSbarOSstt, ShiftMSbarOSstt1

c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------

      double precision mst1msbar, mst2msbar, sttmsbar, q1, q2, qs,
     $                 mst1os,    mst2os,    sttos
      double precision mtos, asmz, as, mz, xmgl
      double precision shift1, shift2, shiftx
 
c      write(*,*) 'variables in MSbartoOnShellPP:'
c      write(*,*) real(mst1msbar), real(mst2msbar), real(sttmsbar),
c     $           real(q1), real(q2), real(qs), real(mtos), real(xmgl),
c     $           real(asmz), real(mz)
      eps = 1d-6
      cf = 4d0/3d0
      pi = 3.14159265358979d0
      i = (0d0, 1d0)

      mst1 = mst1msbar
      mst2 = mst2msbar
      stt = sttmsbar
      mt = mtos
      call alphasmt(asmz,mt,mz,as)
      gs = dsqrt(4d0 * pi * as)
      mgl = xmgl

      mue = q1
      ShiftMSbarOSMSt1sq = ShiftMSbarOSMSt1sq1()
      shift1 = dreal(ShiftMSbarOSMSt1sq)
      mue = q2
      ShiftMSbarOSMSt2sq = ShiftMSbarOSMSt2sq1()
      shift2 = dreal(ShiftMSbarOSMSt2sq)
      mue = qs
      ShiftMSbarOSstt = ShiftMSbarOSstt1()
      shiftx = dreal(ShiftMSbarOSstt)

      mst1os = dsqrt(mst1**2 - shift1)
      mst2os = dsqrt(mst2**2 - shift2)
      sttos = stt - shiftx



      write(*,*) 
      write(*,*) "MSbar to OnShell"
      write(*,*) "================"
      write(*,*)
      write(*,*) "MSt1, MSt2, stt"
      write(*,*) real(mst1msbar), real(mst2msbar), real(sttmsbar)
      write(*,*) real(mst1os), real(mst2os), real(sttos)
      write(*,*) real(shift1), real(shift2), real(shiftx)
      write(*,*) 

 100  continue

      end




	double precision function dzhh()
c The part of the field renormalization dZHH containing the divergence
c Delta and the dimensional regularization scale mudim is unambiguously
c defined: C * (Delta + Log[mudim^2]). The function dzhh returns
c -C * Log[mudim^2], which is exactly the mudim-dependence that remains
c in the HH-SE when an MSbar field renormalization is used.
c This mudim-dependence is missing in the original FH code, where mudim=1
c is used throughout.
c MF 200801

	implicit double precision (a-z)

	real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
        real*8 mcha(1:2),mne(1:4)
	integer pri,naeh,selec,selec2,selec4,selec5,selec6
	double precision mudim

        common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,
     &	            mmw2,mmw,beta,alpha
        common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
        common /mixing/ umix,vmix,nmix
        common /print/ pri,naeh,selec,selec2,selec4,selec5,selec6
	common /msbar/ mudim

      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum

        if (mudim.le.1d0) then
           write(*,*) 'WARNING: renormalization scale mu is not set'
           mudim = mtt
        endif

	dzhh = 0.0D0
        if (selec.ge.1) dzhh = dzhh +
     -  (3*elec2*mtt**2*dsin(alpha)**2)/
     -   (32.D0*mmw2*ppi**2*ssw2*dsin(beta)**2)
        if (selec.ge.2) then 
        if (delmbresum.eq.1) then
           dzhh = dzhh +
     -  (3*elec2*mbb**2*dcos(alpha)**2)/
     -   (32.D0*mmw2*ppi**2*ssw2*dcos(beta)**2)
        else
           dzhh = dzhh +
     -  (3*elec2)/
     -   (32.D0*mmw2*ppi**2*ssw2) *
     $   (-mbb/(1d0 + dmb) * (dcos(alpha)/dcos(beta) +
     $                        dmb * dsin(alpha)/dsin(beta)))**2
        endif
        endif
        if (selec.ge.3) dzhh = dzhh +
     -  (elec2*((3*mdn**2*dcos(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mel**2*dcos(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mmu**2*dcos(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (3*mst**2*dcos(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mta**2*dcos(alpha)**2)/(mmw2*dcos(beta)**2) - 
     -       2*dcos(-alpha + beta)**2 - 
     -       dcos(-alpha + beta)**2/ccw2 + 
     -       (3*mch**2*dsin(alpha)**2)/(mmw2*dsin(beta)**2) + 
     -       (3*mup**2*dsin(alpha)**2)/(mmw2*dsin(beta)**2) - 
     -       2*dsin(-alpha + beta)**2 - 
     -       dsin(-alpha + beta)**2/ccw2 + 
     -       (2*(ssw*nmix(1,1) - ccw*nmix(1,2))**2*
     -          (dcos(alpha)*nmix(1,3) - dsin(alpha)*nmix(1,4))**2)
     -         /ccw2 + (2*(ssw*nmix(2,1) - ccw*nmix(2,2))**2*
     -          (dcos(alpha)*nmix(2,3) - dsin(alpha)*nmix(2,4))**2)
     -         /ccw2 + (dcos(alpha)*
     -            (ssw*(nmix(1,3)*nmix(2,1) + 
     -                 nmix(1,1)*nmix(2,3)) - 
     -              ccw*(nmix(1,3)*nmix(2,2) + nmix(1,2)*nmix(2,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(1,4)*nmix(2,1) + 
     -                   nmix(1,1)*nmix(2,4))) + 
     -              ccw*(nmix(1,4)*nmix(2,2) + nmix(1,2)*nmix(2,4))
     -              ))**2/ccw2 + 
     -       (2*(ssw*nmix(3,1) - ccw*nmix(3,2))**2*
     -          (dcos(alpha)*nmix(3,3) - dsin(alpha)*nmix(3,4))**2)
     -         /ccw2 + (dcos(alpha)*
     -            (ssw*(nmix(1,3)*nmix(3,1) + 
     -                 nmix(1,1)*nmix(3,3)) - 
     -              ccw*(nmix(1,3)*nmix(3,2) + nmix(1,2)*nmix(3,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(1,4)*nmix(3,1) + 
     -                   nmix(1,1)*nmix(3,4))) + 
     -              ccw*(nmix(1,4)*nmix(3,2) + nmix(1,2)*nmix(3,4))
     -              ))**2/ccw2 + 
     -       (dcos(alpha)*(ssw*
     -               (nmix(2,3)*nmix(3,1) + nmix(2,1)*nmix(3,3)) - 
     -              ccw*(nmix(2,3)*nmix(3,2) + nmix(2,2)*nmix(3,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(2,4)*nmix(3,1) + 
     -                   nmix(2,1)*nmix(3,4))) + 
     -              ccw*(nmix(2,4)*nmix(3,2) + nmix(2,2)*nmix(3,4))
     -              ))**2/ccw2 + 
     -       (2*(ssw*nmix(4,1) - ccw*nmix(4,2))**2*
     -          (dcos(alpha)*nmix(4,3) - dsin(alpha)*nmix(4,4))**2)
     -         /ccw2 + (dcos(alpha)*
     -            (ssw*(nmix(1,3)*nmix(4,1) + 
     -                 nmix(1,1)*nmix(4,3)) - 
     -              ccw*(nmix(1,3)*nmix(4,2) + nmix(1,2)*nmix(4,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(1,4)*nmix(4,1) + 
     -                   nmix(1,1)*nmix(4,4))) + 
     -              ccw*(nmix(1,4)*nmix(4,2) + nmix(1,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       (dcos(alpha)*(ssw*
     -               (nmix(2,3)*nmix(4,1) + nmix(2,1)*nmix(4,3)) - 
     -              ccw*(nmix(2,3)*nmix(4,2) + nmix(2,2)*nmix(4,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(2,4)*nmix(4,1) + 
     -                   nmix(2,1)*nmix(4,4))) + 
     -              ccw*(nmix(2,4)*nmix(4,2) + nmix(2,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       (dcos(alpha)*(ssw*
     -               (nmix(3,3)*nmix(4,1) + nmix(3,1)*nmix(4,3)) - 
     -              ccw*(nmix(3,3)*nmix(4,2) + nmix(3,2)*nmix(4,3))
     -              ) + dsin(alpha)*
     -            (-(ssw*(nmix(3,4)*nmix(4,1) + 
     -                   nmix(3,1)*nmix(4,4))) + 
     -              ccw*(nmix(3,4)*nmix(4,2) + nmix(3,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       2*(dcos(alpha)*umix(1,2)*vmix(1,1) + 
     -           dsin(alpha)*umix(1,1)*vmix(1,2))**2 + 
     -       2*(dcos(alpha)*umix(2,2)*vmix(2,1) + 
     -           dsin(alpha)*umix(2,1)*vmix(2,2))**2 + 
     -       2*((dcos(alpha)*umix(2,2)*vmix(1,1) + 
     -             dsin(alpha)*umix(2,1)*vmix(1,2))**2 + 
     -          (dcos(alpha)*umix(1,2)*vmix(2,1) + 
     -             dsin(alpha)*umix(1,1)*vmix(2,2))**2)))/
     -   (32.D0*ppi**2*ssw2)
	dzhh = dzhh*2.0D0*dlog(mudim)

	end


c----------------------------------------------------------------

      double precision function dzh0()
c The part of the field renormalization dZh0 containing the divergence
c Delta and the dimensional regularization scale mudim is unambiguously
c defined: C * (Delta + Log[mudim^2]). The function dzh0 returns
c -C * Log[mudim^2], which is exactly the mudim-dependence that remains
c in the h0-SE when an MSbar field renormalization is used.
c This mudim-dependence is missing in the original FH code, where mudim=1
c is used throughout.
c MF 200801

	implicit double precision (a-z)

	real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
        real*8 mcha(1:2),mne(1:4)
	integer pri,naeh,selec,selec2,selec4,selec5,selec6
        double precision mudim

        common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,
     &	            mmw2,mmw,beta,alpha
        common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
        common /mixing/ umix,vmix,nmix
        common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
	common /msbar/ mudim

      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum

        if (mudim.le.1d0) then
           write(*,*) 'WARNING: renormalization scale mu is not set'
           mudim = mtt
        endif

	dzh0 = 0.0D0
        if (selec.ge.1) dzh0 = dzh0 +
     -  (3*elec2*mtt**2*dcos(alpha)**2)/
     -   (32.D0*mmw2*ppi**2*ssw2*dsin(beta)**2)
        if (selec.ge.2) then
        if (delmbresum.eq.1) then
           dzh0 = dzh0 +
     -  (3*elec2*mbb**2*dsin(alpha)**2)/
     -   (32.D0*mmw2*ppi**2*ssw2*dcos(beta)**2)
        else
           dzh0 = dzh0 +
     -  (3*elec2)/
     -   (32.D0*mmw2*ppi**2*ssw2) *
     $   (mbb/(1d0 + dmb) * (dsin(alpha)/dcos(beta) -
     $                       dmb * dcos(alpha)/dsin(beta)))**2
        endif
        endif
        if (selec.ge.3) dzh0 = dzh0 +
     -  (elec2*(-2*dcos(-alpha + beta)**2 - 
     -       dcos(-alpha + beta)**2/ccw2 + 
     -       (3*mdn**2*dsin(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mel**2*dsin(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mmu**2*dsin(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (3*mst**2*dsin(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (mta**2*dsin(alpha)**2)/(mmw2*dcos(beta)**2) + 
     -       (3*mch**2*dcos(alpha)**2)/(mmw2*dsin(beta)**2) + 
     -       (3*mup**2*dcos(alpha)**2)/(mmw2*dsin(beta)**2) - 
     -       2*dsin(-alpha + beta)**2 - 
     -       dsin(-alpha + beta)**2/ccw2 + 
     -       (2*(ssw*nmix(1,1) - ccw*nmix(1,2))**2*
     -          (dsin(alpha)*nmix(1,3) + dcos(alpha)*nmix(1,4))**2)
     -         /ccw2 + (2*(ssw*nmix(2,1) - ccw*nmix(2,2))**2*
     -          (dsin(alpha)*nmix(2,3) + dcos(alpha)*nmix(2,4))**2)
     -         /ccw2 + (dsin(alpha)*
     -            (ssw*(nmix(1,3)*nmix(2,1) + 
     -                 nmix(1,1)*nmix(2,3)) - 
     -              ccw*(nmix(1,3)*nmix(2,2) + nmix(1,2)*nmix(2,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(1,4)*nmix(2,1) + 
     -                 nmix(1,1)*nmix(2,4)) - 
     -              ccw*(nmix(1,4)*nmix(2,2) + nmix(1,2)*nmix(2,4))
     -              ))**2/ccw2 + 
     -       (2*(ssw*nmix(3,1) - ccw*nmix(3,2))**2*
     -          (dsin(alpha)*nmix(3,3) + dcos(alpha)*nmix(3,4))**2)
     -         /ccw2 + (dsin(alpha)*
     -            (ssw*(nmix(1,3)*nmix(3,1) + 
     -                 nmix(1,1)*nmix(3,3)) - 
     -              ccw*(nmix(1,3)*nmix(3,2) + nmix(1,2)*nmix(3,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(1,4)*nmix(3,1) + 
     -                 nmix(1,1)*nmix(3,4)) - 
     -              ccw*(nmix(1,4)*nmix(3,2) + nmix(1,2)*nmix(3,4))
     -              ))**2/ccw2 + 
     -       (dsin(alpha)*(ssw*
     -               (nmix(2,3)*nmix(3,1) + nmix(2,1)*nmix(3,3)) - 
     -              ccw*(nmix(2,3)*nmix(3,2) + nmix(2,2)*nmix(3,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(2,4)*nmix(3,1) + 
     -                 nmix(2,1)*nmix(3,4)) - 
     -              ccw*(nmix(2,4)*nmix(3,2) + nmix(2,2)*nmix(3,4))
     -              ))**2/ccw2 + 
     -       (2*(ssw*nmix(4,1) - ccw*nmix(4,2))**2*
     -          (dsin(alpha)*nmix(4,3) + dcos(alpha)*nmix(4,4))**2)
     -         /ccw2 + (dsin(alpha)*
     -            (ssw*(nmix(1,3)*nmix(4,1) + 
     -                 nmix(1,1)*nmix(4,3)) - 
     -              ccw*(nmix(1,3)*nmix(4,2) + nmix(1,2)*nmix(4,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(1,4)*nmix(4,1) + 
     -                 nmix(1,1)*nmix(4,4)) - 
     -              ccw*(nmix(1,4)*nmix(4,2) + nmix(1,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       (dsin(alpha)*(ssw*
     -               (nmix(2,3)*nmix(4,1) + nmix(2,1)*nmix(4,3)) - 
     -              ccw*(nmix(2,3)*nmix(4,2) + nmix(2,2)*nmix(4,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(2,4)*nmix(4,1) + 
     -                 nmix(2,1)*nmix(4,4)) - 
     -              ccw*(nmix(2,4)*nmix(4,2) + nmix(2,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       (dsin(alpha)*(ssw*
     -               (nmix(3,3)*nmix(4,1) + nmix(3,1)*nmix(4,3)) - 
     -              ccw*(nmix(3,3)*nmix(4,2) + nmix(3,2)*nmix(4,3))
     -              ) + dcos(alpha)*
     -            (ssw*(nmix(3,4)*nmix(4,1) + 
     -                 nmix(3,1)*nmix(4,4)) - 
     -              ccw*(nmix(3,4)*nmix(4,2) + nmix(3,2)*nmix(4,4))
     -              ))**2/ccw2 + 
     -       2*(dsin(alpha)*umix(1,2)*vmix(1,1) - 
     -           dcos(alpha)*umix(1,1)*vmix(1,2))**2 + 
     -       2*(dsin(alpha)*umix(2,2)*vmix(2,1) - 
     -           dcos(alpha)*umix(2,1)*vmix(2,2))**2 + 
     -       2*((dsin(alpha)*umix(2,2)*vmix(1,1) - 
     -             dcos(alpha)*umix(2,1)*vmix(1,2))**2 + 
     -          (dsin(alpha)*umix(1,2)*vmix(2,1) - 
     -             dcos(alpha)*umix(1,1)*vmix(2,2))**2)))/
     -   (32.D0*ppi**2*ssw2)
        dzh0 = dzh0*2.0D0*dlog(mudim)

	end

c----------------------------------------------------------------
      
      complex*16 function P1se1()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      P1se1= 

     &  (3D0*EL**2*MT**2*MW**(-2)*PI**(-2)*SB**(-2)*SW**(-2)*(MUE**
     &  2-MUE**2*SB**4-4D0*MUE**2*STT**2+4D0*MUE**2*STT**4+2D0*MUE*
     &  SB**3*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2)**(1D0/
     &  2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/
     &  MT)-SB**2*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-
     &  MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**2+SB**4*(-
     &  ((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*
     &  STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**2+((MST1**2-MST2**2)*SB*
     &  STT*(1D0-STT**2)**(1D0/2D0)*(-(MUE*(1D0-SB**2)**(1D0/2D0))-
     &  MUE*SB**2*(1D0-SB**2)**(1D0/2D0)+SB*(-((MUE*(1D0-SB**2)**
     &  (1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/
     &  2D0))/MT)-SB**3*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**
     &  2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)))/MT+((MST1**2-
     &  MST2**2)*SB*STT*(1D0-STT**2)**(1D0/2D0)*(MUE*(1D0-SB**2)**
     &  (1D0/2D0)+MUE*SB**2*(1D0-SB**2)**(1D0/2D0)-SB*(-((MUE*(1D0-
     &  SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**
     &  (1D0/2D0))/MT)+SB**3*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT))*
     &  LOG(MST1**2))/MT+(-(MUE**2)+MUE**2*SB**4-2D0*MUE*SB**3*(1D0-
     &  SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+SB**2*(-
     &  ((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*
     &  STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**2-SB**4*(-((MUE*(1D0-
     &  SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-
     &  STT**2D0)**(1D0/2D0))/MT)**2)*LOG(MST1**2)+(MST2**2*SB*STT*
     &  (1D0-STT**2)**(1D0/2D0)*(MUE*(1D0-SB**2)**(1D0/2D0)+MUE*SB**
     &  2*(1D0-SB**2)**(1D0/2D0)-SB*(-((MUE*(1D0-SB**2)**(1D0/2D0))/
     &  SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+SB**
     &  3*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*
     &  STT*(1D0-STT**2)**(1D0/2D0))/MT))*LOG(MST1**2*MST2**(-2)))/
     &  MT+((MST2**2*MUE**2-MST2**2*MUE**2*SB**4-2D0*MST1**2*MUE**2*
     &  STT**2-2D0*MST2**2*MUE**2*STT**2+2D0*MST1**2*MUE**2*STT**4+
     &  2D0*MST2**2*MUE**2*STT**4+2D0*MST2**2*MUE*SB**3*(1D0-SB**
     &  2)**(1D0/2D0)*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-
     &  MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-MST2**2*SB**2*(-
     &  ((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*
     &  STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**2+MST2**2*SB**4*(-((MUE*
     &  (1D0-SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*
     &  (1D0-STT**2D0)**(1D0/2D0))/MT)**2)*LOG(MST1**2*MST2**(-2)))/
     &  (-(MST1**2)+MST2**2)))/32D0
      end


c----------------------------------------------------------------
      
      complex*16 function P2se1()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      P2se1=

     &  (3D0*EL**2*MT**2*MW**(-2)*PI**(-2)*SB**(-2)*SW**(-2)*(-
     &  (MUE**2*SB**2)+MUE**2*SB**4+2D0*MUE*SB*(1D0-SB**2)**(3D0/
     &  2D0)*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*
     &  STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*SB**2*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2-SB**4*(-((MUE*(1D0-SB**2D0)**(1D0/
     &  2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/
     &  2D0))/MT)**2-4D0*STT**2*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/
     &  SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/
     &  MT)**2+4D0*STT**4*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+
     &  ((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**
     &  2+((-(MST1**2)+MST2**2)*SB**2*STT*(1D0-STT**2)**(1D0/2D0)*(-
     &  (MUE*SB*(1D0-SB**2)**(1D0/2D0))+2D0*(-((MUE*(1D0-SB**2)**
     &  (1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/
     &  2D0))/MT)-SB**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**
     &  2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)))/MT-2D0*MT**2*
     &  LOG(MST1**2)-4D0*MT*STT*(1D0-STT**2)**(1D0/2D0)*(-((MUE*
     &  (1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-
     &  STT**2)**(1D0/2D0))/MT)*LOG(MST1**2)+(MST1**2*SB**2*STT*
     &  (1D0-STT**2)**(1D0/2D0)*(-(MUE*SB*(1D0-SB**2)**(1D0/2D0))+
     &  2D0*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*
     &  STT*(1D0-STT**2)**(1D0/2D0))/MT)-SB**2*(-((MUE*(1D0-SB**2)**
     &  (1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/
     &  2D0))/MT))*LOG(MST1**2))/MT+((-(MST1**2*MUE**2*SB**2)+MST1**
     &  2*MUE**2*SB**4+2D0*MST1**2*MUE*SB*(1D0-SB**2)**(3D0/2D0)*(-
     &  ((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*
     &  (1D0-STT**2)**(1D0/2D0))/MT)+2D0*MST1**2*SB**2*(-((MUE*(1D0-
     &  SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-
     &  STT**2D0)**(1D0/2D0))/MT)**2-MST1**2*SB**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2-2D0*MST1**2*STT**2*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2-2D0*MST2**2*STT**2*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2+2D0*MST1**2*STT**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2+2D0*MST2**2*STT**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2)*LOG(MST1**2))/(-(MST1**2)+MST2**2)-
     &  2D0*MT**2*LOG(MST2**2)+4D0*MT*STT*(1D0-STT**2)**(1D0/2D0)*(-
     &  ((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*
     &  (1D0-STT**2)**(1D0/2D0))/MT)*LOG(MST2**2)+(MST2**2*SB**2*
     &  STT*(1D0-STT**2)**(1D0/2D0)*(MUE*SB*(1D0-SB**2)**(1D0/2D0)-
     &  2D0*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*
     &  STT*(1D0-STT**2)**(1D0/2D0))/MT)+SB**2*(-((MUE*(1D0-SB**2)**
     &  (1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/
     &  2D0))/MT))*LOG(MST2**2))/MT+((-(MST2**2*MUE**2*SB**2)+MST2**
     &  2*MUE**2*SB**4+2D0*MST2**2*MUE*SB*(1D0-SB**2)**(3D0/2D0)*(-
     &  ((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*
     &  (1D0-STT**2)**(1D0/2D0))/MT)+2D0*MST2**2*SB**2*(-((MUE*(1D0-
     &  SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-
     &  STT**2D0)**(1D0/2D0))/MT)**2-MST2**2*SB**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2-2D0*MST1**2*STT**2*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2-2D0*MST2**2*STT**2*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2+2D0*MST1**2*STT**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2+2D0*MST2**2*STT**4*(-((MUE*(1D0-SB**
     &  2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**
     &  2D0)**(1D0/2D0))/MT)**2)*LOG(MST2**2))/(MST1**2-MST2**2)+
     &  4D0*MT**2*LOG(MT**2)))/32D0
      end


c----------------------------------------------------------------
      
      complex*16 function P1P2se1()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      P1P2se1=

     &  (3D0*EL**2*MT**2*MW**(-2)*PI**(-2)*SB**(-2)*SW**(-2)*(MST1**
     &  2*SB*(1D0-SB**2)**(1D0/2D0)+MST2**2*SB*(1D0-SB**2)**(1D0/
     &  2D0)-MST1**2*SB**3*(1D0-SB**2)**(1D0/2D0)-MST2**2*SB**3*
     &  (1D0-SB**2)**(1D0/2D0)+MUE**2*SB**3*(1D0-SB**2)**(1D0/2D0)-
     &  MST1**2*SB*(1D0-SB**2)**(3D0/2D0)-MST2**2*SB*(1D0-SB**2)**
     &  (3D0/2D0)+MUE*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-
     &  MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*MUE*SB**2*(-
     &  ((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*
     &  (1D0-STT**2)**(1D0/2D0))/MT)+2D0*MUE*SB**4*(-((MUE*(1D0-SB**
     &  2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**
     &  (1D0/2D0))/MT)-4D0*MUE*STT**2*(-((MUE*(1D0-SB**2)**(1D0/
     &  2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/
     &  MT)+4D0*MUE*STT**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+SB*(1D0-
     &  SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/SB)+
     &  ((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/MT)**
     &  2-SB**3*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2D0)**(1D0/
     &  2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/
     &  2D0))/MT)**2+((-(MST1**2)+MST2**2)*SB*STT*(1D0-STT**2)**
     &  (1D0/2D0)*(MUE*SB**3+(1D0-SB**2)**(3D0/2D0)*(-((MUE*(1D0-
     &  SB**2)**(1D0/2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**
     &  (1D0/2D0))/MT)))/MT-2D0*MT*MUE*STT*(1D0-STT**2)**(1D0/2D0)*
     &  LOG(MST1**2)+(MST1**2*SB*STT*(1D0-STT**2)**(1D0/2D0)*(MUE*
     &  SB**3+(1D0-SB**2)**(3D0/2D0)*(-((MUE*(1D0-SB**2)**(1D0/
     &  2D0))/SB)+((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/
     &  MT))*LOG(MST1**2))/MT+((MST1**2*MUE**2*SB**3*(1D0-SB**2)**
     &  (1D0/2D0)+MST1**2*MUE*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST1**2*MUE*SB**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST1**2*MUE*SB**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST1**2*MUE*STT**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST2**2*MUE*STT**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST1**2*MUE*STT**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST2**2*MUE*STT**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+MST1**2*
     &  SB*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/
     &  SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/
     &  MT)**2-MST1**2*SB**3*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-
     &  SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-
     &  STT**2D0)**(1D0/2D0))/MT)**2)*LOG(MST1**2))/(-(MST1**2)+
     &  MST2**2)+2D0*MT*MUE*STT*(1D0-STT**2)**(1D0/2D0)*LOG(MST2**
     &  2)-(MST2**2*SB*STT*(1D0-STT**2)**(1D0/2D0)*(MUE*SB**3+(1D0-
     &  SB**2)**(3D0/2D0)*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT))*
     &  LOG(MST2**2))/MT+((MST2**2*MUE**2*SB**3*(1D0-SB**2)**(1D0/
     &  2D0)+MST2**2*MUE*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST2**2*MUE*SB**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST2**2*MUE*SB**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST1**2*MUE*STT**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)-2D0*
     &  MST2**2*MUE*STT**2*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST1**2*MUE*STT**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+2D0*
     &  MST2**2*MUE*STT**4*(-((MUE*(1D0-SB**2)**(1D0/2D0))/SB)+
     &  ((MST1**2-MST2**2)*STT*(1D0-STT**2)**(1D0/2D0))/MT)+MST2**2*
     &  SB*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-SB**2D0)**(1D0/2D0))/
     &  SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-STT**2D0)**(1D0/2D0))/
     &  MT)**2-MST2**2*SB**3*(1D0-SB**2)**(1D0/2D0)*(-((MUE*(1D0-
     &  SB**2D0)**(1D0/2D0))/SB)+((MST1**2D0-MST2**2D0)*STT*(1D0-
     &  STT**2D0)**(1D0/2D0))/MT)**2)*LOG(MST2**2))/(MST1**2-MST2**
     &  2)))/32D0
      end


c----------------------------------------------------------------
c
c --> LLExpansionP2MTruncode.f
c
c----------------------------------------------------------------
      
      complex*16 function LLExpansionP2MTrun1(mtrun,ms)
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      double precision mtrun,ms
      LLExpansionP2MTrun1=

     &  -(144D0*MLRT**2*MS**(-4)*MTRUN**2*(MS**2-2D0*MTRUN**2))+
     &  12D0*MLRT**4*MS**(-8)*MTRUN**2*(15D0*MS**4-68D0*MS**2*
     &  MTRUN**2+55D0*MTRUN**4)+(4D0*MLRT**6*MS**(-12)*MTRUN**4*
     &  (271D0*MS**6-1655D0*MS**4*MTRUN**2+2576D0*MS**2*MTRUN**4-
     &  1189D0*MTRUN**6))/(5D0*(MS**2-MTRUN**2))+(MLRT**8*MS**(-16)*
     &  MTRUN**4*(MS**2D0-MTRUN**2D0)**(-2)*(-(90D0*MS**10)+43179D0*
     &  MTRUN**10+10522D0*MS**8*MTRUN**2-69410D0*MS**6*MTRUN**4+
     &  150943D0*MS**4*MTRUN**6-135132D0*MS**2*MTRUN**8))/35D0+
     &  576D0*MLRT*MS**(-6)*MTRUN**3*(MS**2D0-MTRUN**2D0)**2*PI+
     &  24D0*MLRT**3*MS**(-10)*MTRUN**3*(-(MS**2)+MTRUN**2)*(7D0*
     &  MS**4-52D0*MS**2*MTRUN**2+48D0*MTRUN**4)*PI+(3D0*MLRT**5*
     &  MS**(-14)*MTRUN**3*(25D0*MS**8-1288D0*MS**6*MTRUN**2+7520D0*
     &  MS**4*MTRUN**4-12032D0*MS**2*MTRUN**6+5760D0*MTRUN**8)*PI)/
     &  10D0+(3D0*MLRT**7*MS**(-18)*MTRUN**3*(35D0*MS**12+1326592D0*
     &  MS**2*MTRUN**10-430080D0*MTRUN**12+3556D0*MS**10*MTRUN**2-
     &  116720D0*MS**8*MTRUN**4+688896D0*MS**6*MTRUN**6-1472384D0*
     &  MS**4*MTRUN**8)*PI)/(560D0*(MS**2-MTRUN**2))+(24D0*MLRT**5*
     &  MS**(-12)*MTRUN**4*(62D0*MS**6-465D0*MS**4*MTRUN**2+764D0*
     &  MS**2*MTRUN**4-360D0*MTRUN**6))/(5D0*DCMPLX(DCMPLX(MS**2D0-
     &  MTRUN**2D0))**(1D0/2D0))+(3D0*MLRT**6*MS**(-14)*MTRUN**3*
     &  (25D0*MS**8-1288D0*MS**6*MTRUN**2+7520D0*MS**4*MTRUN**4-
     &  12032D0*MS**2*MTRUN**6+5760D0*MTRUN**8)*PI)/(20D0*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0))+288D0*MLRT*
     &  MS**(-4)*MTRUN**2*(-(3D0*MS**2)+2D0*MTRUN**2)*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0)+48D0*MLRT**3*
     &  MS**(-8)*MTRUN**2*(3D0*MS**4-28D0*MS**2*MTRUN**2+24D0*
     &  MTRUN**4)*DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0)-
     &  12D0*MLRT**4*MS**(-10)*MTRUN**3*(7D0*MS**4-52D0*MS**2*
     &  MTRUN**2+48D0*MTRUN**4)*PI*DCMPLX(DCMPLX(MS**2D0-MTRUN**
     &  2D0))**(1D0/2D0)+(12D0*MLRT**7*MS**(-16)*MTRUN**4*(-(15D0*
     &  MS**10)+6720D0*MTRUN**10+1455D0*MS**8*MTRUN**2-10250D0*MS**
     &  6*MTRUN**4+22940D0*MS**4*MTRUN**6-20848D0*MS**2*MTRUN**8))/
     &  (35D0*DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(3D0/2D0))+(3D0*
     &  MLRT**8*MS**(-18)*MTRUN**3*(35D0*MS**12+1326592D0*MS**2*
     &  MTRUN**10-430080D0*MTRUN**12+3556D0*MS**10*MTRUN**2-
     &  116720D0*MS**8*MTRUN**4+688896D0*MS**6*MTRUN**6-1472384D0*
     &  MS**4*MTRUN**8)*PI)/(1120D0*DCMPLX(DCMPLX(MS**2D0-MTRUN**
     &  2D0))**(3D0/2D0))+288D0*MLRT**2*MS**(-6)*MTRUN**3*PI*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(3D0/2D0)+(24D0*MLRT**4*
     &  MS**(-8)*(3D0*MS**8-8D0*MS**6*MTRUN**2+6D0*MS**4*MTRUN**4+
     &  3D0*MS**2*MTRUN**6-3D0*MTRUN**8)*DLOG(MS**(-2)*MTRUN**2))/
     &  (MS**2-MTRUN**2)+(12D0*MLRT**6*MS**(-12)*MTRUN**2*(MS**2D0-
     &  MTRUN**2D0)**(-2)*(10D0*MS**10+20D0*MTRUN**10-47D0*MS**8*
     &  MTRUN**2+72D0*MS**6*MTRUN**4-18D0*MS**4*MTRUN**6-36D0*MS**2*
     &  MTRUN**8)*DLOG(MS**(-2)*MTRUN**2))/5D0+(12D0*MLRT**8*MS**(-
     &  16)*MTRUN**4*(MS**2D0-MTRUN**2D0)**(-3)*(42D0*MS**12+285D0*
     &  MS**2*MTRUN**10-105D0*MTRUN**12-256D0*MS**10*MTRUN**2+561D0*
     &  MS**8*MTRUN**4-441D0*MS**6*MTRUN**6-85D0*MS**4*MTRUN**8)*
     &  DLOG(MS**(-2)*MTRUN**2))/35D0+(48D0*MLRT**3*(3D0*MS**4-6D0*
     &  MS**2*MTRUN**2+4D0*MTRUN**4)*DCMPLX(DCMPLX(MS**2D0-MTRUN**
     &  2D0))**(1D0/2D0)*DLOG(MS**(-2)*MTRUN**2))/(MS**6-MS**4*
     &  MTRUN**2)+(24D0*MLRT**5*MS**(-8)*MTRUN**2*(10D0*MS**6-35D0*
     &  MS**4*MTRUN**2+44D0*MS**2*MTRUN**4-18D0*MTRUN**6)*DLOG(MS**
     &  (-2)*MTRUN**2))/(5D0*DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**
     &  (3D0/2D0))+(24D0*MLRT**7*MS**(-12)*MTRUN**4*(42D0*MS**8-
     &  196D0*MS**6*MTRUN**2+351D0*MS**4*MTRUN**4-276D0*MS**2*
     &  MTRUN**6+80D0*MTRUN**8)*DLOG(MS**(-2)*MTRUN**2))/(35D0*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(5D0/2D0))-144D0*MLRT**
     &  2*DLOG(MTRUN**2D0/MS**2D0)**2+144D0*(2D0*MS**2+MTRUN**2)*
     &  DLOG(MTRUN**2D0/MS**2D0)**2-288D0*MLRT*DCMPLX(DCMPLX(MS**
     &  2D0-MTRUN**2D0))**(1D0/2D0)*DLOG(MTRUN**2D0/MS**2D0)**2-
     &  144D0*MLRT**2*MS**(-6)*(MS**6-3D0*MS**4*MTRUN**2+4D0*MS**2*
     &  MTRUN**4-2D0*MTRUN**6)*DLOG(-1D0+MS**2*MTRUN**(-2))+(24D0*
     &  MLRT**6*MS**(-14)*MTRUN**2*(5D0*MS**8-21D0*MS**6*MTRUN**2+
     &  132D0*MS**4*MTRUN**4-296D0*MS**2*MTRUN**6+180D0*MTRUN**8)*
     &  DLOG(-1D0+MS**2*MTRUN**(-2)))/5D0+24D0*MLRT**4*MS**(-10)*
     &  (3D0*MS**8-5D0*MS**6*MTRUN**2+19D0*MS**4*MTRUN**4-41D0*MS**
     &  2*MTRUN**6+24D0*MTRUN**8)*DLOG(-1D0+MS**2*MTRUN**(-2))+
     &  (12D0*MLRT**8*MS**(-18)*MTRUN**4*(42D0*MS**8-340D0*MS**6*
     &  MTRUN**2+2437D0*MS**4*MTRUN**4-5429D0*MS**2*MTRUN**6+3360D0*
     &  MTRUN**8)*DLOG(-1D0+MS**2*MTRUN**(-2)))/35D0+288D0*MLRT*MS**
     &  (-6)*MTRUN**2*(MS**4-3D0*MS**2*MTRUN**2+2D0*MTRUN**4)*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0)*DLOG(-1D0+MS**
     &  2*MTRUN**(-2))+(48D0*MLRT**7*MS**(-18)*MTRUN**4*(21D0*MS**8-
     &  140D0*MS**6*MTRUN**2+1136D0*MS**4*MTRUN**4-2662D0*MS**2*
     &  MTRUN**6+1680D0*MTRUN**8)*DCMPLX(DCMPLX(MS**2D0-MTRUN**
     &  2D0))**(1D0/2D0)*DLOG(-1D0+MS**2*MTRUN**(-2)))/35D0+(48D0*
     &  MLRT**5*MS**(-14)*MTRUN**2*(5D0*MS**8-15D0*MS**6*MTRUN**2+
     &  116D0*MS**4*MTRUN**4-286D0*MS**2*MTRUN**6+180D0*MTRUN**8)*
     &  DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0)*DLOG(-1D0+MS**
     &  2*MTRUN**(-2)))/5D0+48D0*MLRT**3*MS**(-10)*(3D0*MS**8-3D0*
     &  MS**6*MTRUN**2+14D0*MS**4*MTRUN**4-38D0*MS**2*MTRUN**6+24D0*
     &  MTRUN**8)*DCMPLX(DCMPLX(MS**2D0-MTRUN**2D0))**(1D0/2D0)*
     &  DLOG(-1D0+MS**2*MTRUN**(-2))-288D0*MLRT*DCMPLX(DCMPLX(MS**
     &  2D0-MTRUN**2D0))**(1D0/2D0)*DLOG(MS**(-2)*MTRUN**2)*DLOG(-
     &  1D0+MS**2*MTRUN**(-2))-144D0*MLRT**2*MS**(-4)*DLOG(MS**(-2)*
     &  MTRUN**2)*(MS**4+MS**2*MTRUN**2-MTRUN**4+MS**4*DLOG(-1D0+
     &  MS**2*MTRUN**(-2)))+144D0*DLOG(MS**(-2)*MTRUN**2)*(-(4D0*
     &  MTRUN**2)+2D0*(MS**2-MTRUN**2)*DLOG(-1D0+MS**2*MTRUN**(-2)))
      end





c----------------------------------------------------------------
c
c --> Hhmasssr.f
c
c----------------------------------------------------------------
      
c     %%%%%%%%%%%%%%%%%%%%%% geaendert! %%%%%%%%%%%%%%%%%%%%%%%%%
      DOUBLE PRECISION FUNCTION DELTA (EPSILON,MUEE,MASS)
C
      IMPLICIT REAL*8(A-Z)
C
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
C
      EULERG = 0.57721566D0
C
c     DELTA = 2.D0/EPSILON
c     DELTA = 2.D0/EPSILON - EULERG + DLOG(4.D0 * PPI) -
c    &        DLOG(MASS**2/MUEE**2)
c     DELTA = -EULERG + DLOG(4.D0 * PPI) -
c    &         DLOG(MASS**2/MUEE**2)
c       DELTA = 2.D0/EPSILON - DLOG(MASS**2/MUEE**2)
c       DELTA = -DLOG(MASS**2)
      DELTA = -DLOG(MASS**2/muee**2)
      delta = 0d0
C
      RETURN
C
      END
C
      DOUBLE PRECISION FUNCTION FF(S,MASS1,MASS2)
C
      IMPLICIT REAL*8(A-Z)
      REAL*8 B0,B1
C
      IF (MASS1.LE.(1.D-8)) THEN
         XHELP = (S - MASS2**2)**2
         IF (XHELP.LE.(1.D-8)) THEN
            FF = 1.D0
         ELSE
         FF = 1.D0 + (MASS2**2/S - 1.D0) * DLOG(DABS(1.D0 -
     &        S/MASS2**2))
      ENDIF
      ELSE
C
	CALL BQUER2 (S,MASS1,MASS2,B0,B1,PB0,PB1)
C
        IF ((MASS1 - MASS2)**2.LE.1.D-10) THEN
	   FF =  B0
        ELSE
           FF = - 1.D0 + (MASS1**2 + MASS2**2)/(MASS1**2 - MASS2**2) *
     &          DLOG(MASS1/MASS2) + B0
        ENDIF
C
      ENDIF
      RETURN
C
      END
c
C =================================================================
C
      DOUBLE PRECISION FUNCTION P(S,M)
C
***************************************************************
* Real part of the 1-loop QED vacuumpolarisation contribution *
* from a fermion with mass m                                  *
*                                                             *
*   Relation with the function F:                             *
*     P(s,m) = 1/3 - (1 + 2m**2/s)*F(s,m,m)                   *
***************************************************************
      IMPLICIT DOUBLE PRECISION (A-Z)
      IF(S.EQ.0) THEN
	P = 0.D0
      ELSE IF(S.LT.0) THEN
	X = SQRT(1.D0-4.D0*M**2/S)
	P = -8.D0/3.D0+X**2-
     .          X*(3.D0-X**2)*LOG(-4.D0*M**2/(S*(1.D0+X)**2))/2.D0
      ELSE IF(S.LT.4.D0*M**2) THEN
	X = SQRT(4.D0*M**2/S-1.D0)
	P = -8.D0/3.D0-X**2+X*(3.D0+X**2)*ATAN(1.D0/X)
      ELSE
	X = SQRT(1.D0-4.D0*M**2/S)
	P = -8.D0/3.D0+X**2-
     .          X*(3.D0-X**2)*LOG(4.D0*M**2/(S*(1.D0+X)**2))/2.D0
      ENDIF
      END
c ===============================================================
      COMPLEX*16 FUNCTION L2(X)
      IMPLICIT DOUBLE PRECISION (A-Z)
      PARAMETER(PPI=3.1415926535897932D0,Z2=1.64493406684823D0)
* Statement function (Sp(u)=Li2(x), u=-log(1-x) ):
      SP(U)=((((((-1.D0/10886400.D0)*U**2+1.D0/211680.D0)*U**2-
     .         1.D0/3600)*U**2+1.D0/36.D0)*U-1.D0/4.D0)*U+1.D0)*U
*
      REL2 = -7.D0/2.D0-2.D0*X-(X+3.D0/2.D0)*LOG(X**2)
      IF(X.GT.1) THEN
	U    = -LOG(1.D0+1.D0/X)
	IML2 = -PPI*(2.D0*(1.D0+X)**2*U+3.D0+2.D0*X)
	REL2 =  REL2+2.D0*(1.D0+X)**2*(-SP(U)-U*LOG(X))
      ELSEIF(X.GT.0.D0) THEN
	U    = -LOG(1.D0+X)
	U1   =  LOG(X)
	IML2 = -PPI*(2.D0*(1.D0+X)**2*(U+LOG(X))+3.D0+2.D0*X)
	REL2 =  REL2+2.D0*(1.D0+X)**2*(SP(U)+Z2-U1**2/2.D0-U*U1)
      ELSEIF(X.GT.-1.D0/2.D0) THEN
	U    = -LOG(1.D0+X)
	U1   =  LOG(-X)
	IML2 =  0.D0
	REL2 =  REL2+2.D0*(1.D0+X)**2*(SP(U)-2.D0*Z2-U1**2/2.D0-U*U1)
      ELSEIF(X.GT.-2.D0) THEN
	U    = -LOG(-X)
	IML2 =  0.D0
	REL2 =  REL2+2.D0*(1.D0+X)**2*(-SP(U)-Z2-U**2/2.D0)
      ELSE
	U    = -LOG(1.D0+1.D0/X)
	IML2 =  0.D0
	REL2 =  REL2+2.D0*(1.D0+X)**2*(-SP(U)-U*LOG(-X))
      ENDIF
      L2=DCMPLX(REL2,IML2)
      END
c ===============================================================
      COMPLEX*16 FUNCTION L3(X)
      IMPLICIT DOUBLE PRECISION (A-Z)
      PARAMETER(PPI=3.1415926535897932D0)
      IF(X.GT.1.D0/4.D0) THEN
	SQ  =SQRT(4.D0*X-1.D0)
	F   =ATAN(1.D0/SQ)
	IML3=0.D0
	REL3=(2.5D0-2.D0*X+(4.D0*X+2.D0)*SQ*F-8.D0*X*(X+2.D0)*F**2)/3.D0
      ELSEIF(X.GT.0.D0) THEN
	SQ  =SQRT(1.D0-4.D0*X)
	F   =LOG((1.D0+SQ)/(1.D0-SQ))
	IML3=-PPI*((2.D0*X+1.D0)*SQ+2.D0*X*(X+2.D0)*F)/3.D0
	REL3=(2.5D0-2.D0*X+(2.D0*X+1.D0)*SQ*F+
     .                              2.D0*X*(X+2.D0)*(F**2-PPI**2))/3.D0
      ELSE
	SQ  =SQRT(1.D0-4.D0*X)
	F   =LOG((SQ+1.D0)/(SQ-1.D0))
	IML3=0.D0
	REL3=(2.5D0-2.D0*X+(2.D0*X+1.D0)*SQ*F+2.D0*X*(X+2.D0)*F**2)/3.D0
      ENDIF
      L3=DCMPLX(REL3,IML3)
      END
C ==========================================================
C
      SUBROUTINE CFUNC(S,MF,M1,M2,M3,C0,C1P,C1M,C20,C2P,C2M)
C
C  DEFINITION OF THE INVARIANT FUNCTIONS IN THE 3-POINT INTEGRALS
C  WITH EQUAL EXTERNAL MASSES MF.
C  S = MOMENTUM TRANSFER; M1,M2,M3 ARE THE INTERNAL MASSES
C
C                                   P2 ( = PF)
C                        M2   .
C
C               S .     .       M3
C
C                        M1   .
C                                  P1 ( = -PF)
C
C
C  C0 = SCALAR INTEGRAL, C1P/M = C1+/-,  C2P/M= C2+/-
C
      IMPLICIT REAL*8(A-Z)
      COMPLEX*16 C0, C1P,C1M,C20,C2P,C2M,CSCAL,B0S12,B031,B032,
     &           B132,B131,B1S12,C1,C2
      XMF=MF*MF
      C0=CSCAL(S,MF,M1,M2,M3)
      CALL BQUER(S,M1,M2,B0S12,B1S12)
      CALL BQUER(XMF,M3,M1,B031,B131)
      CALL BQUER(XMF,M3,M2,B032,B132)
C
C  B0JK := B0(XMF,MJ,MK),  B1JK:= B1(XMF,MF,MK)
C  B0S12 := B0(S,M1,M2)
C
C  THE C1 FUNCTIONS:
C
      MS=4.D0*XMF-S
      XM1=M1**2
      XM2=M2**2
      XM3=M3**2
      C1=.5D0*DLOG(XM3/M1/M2)+B0S12-.5D0*(B031+B032)
     &   +(XMF+XM3-XM1/2.D0-XM2/2.D0) *C0
      C1P=C1/MS
      C1M=(DLOG(M2/M1)+B031-B032+(XM2-XM1)*C0)/2.D0/S
C
C  THE C2 FUNCTIONS:
C
      C2=1.D0+B0S12+(XM1+XM2-2.D0*XM3-2.D0*XMF)*C1P+(XM1-XM2)*C1M
     &   +2.D0*XM3*C0
      C20=C2/4.D0
      C2=                     (B131+B132+2.D0*B0S12-.5D0)/4.D0
     &   +(2.D0*XM3-XM1-XM2+2.D0*XMF)/2.D0*C1P-C20
      C2P=C2/MS
      C2=                    -(B131+B132-.5D0)/4.D0
     &   -(XM1-XM2)/2.D0*C1M-C20
      C2M=C2/S
C
      RETURN
      END
C
C ===========================================================
C
      SUBROUTINE BQUER2(X,M1,M2,B0,B1,P0,P1)
C
C  B0 AND B1 ARE THE (FINITE) INARIANT FUNCTIONS IN THE
C  2-POINT INTEGRALS, P0 AND P1 THEIR DERIVATIVES.
C  REAL PARTS ONLY, NEEDED FOR FERMION RENORMALIZATION.
C  X = Q**2;  M1,M2 ARE THE INTERNAL MASSES
C
      IMPLICIT REAL*8(A-Z)
      EXTERNAL F
c old version removed 15Jun00
c      LM=DLOG(M2/M1)
c      if ((dabs(m2).le.1d-3).or.(dabs(m1).le.1d-3))
c     $     write(*,*) 'masses in log:', m1, m2
      IF (M1.EQ.M2) THEN
        LM=0.
      ELSE
        LM=DLOG(M2/M1)
      ENDIF
      CF=F(X,M1,M2)
      XM1=M1**2
      XM2=M2**2
      IF (M1.EQ.M2) GOTO 10
      B0=1.D0-(XM2+XM1)/(XM2-XM1)*LM+CF
      B1=-.25D0+XM1/(XM2-XM1)*LM+(XM2-XM1-X)/2.D0/X*CF
      GOTO 20
10    B0=CF
      B1=-B0/2.D0+0.25d0
20    CONTINUE
C
C   CALCULATION OF THE DERIVATIVES:
C
      SM=XM1+XM2
      DM=XM2-XM1
      SM12=(M1+M2)**2
      DM12=(M1-M2)**2
      S=DSQRT(DABS(SM12-X))
      D=DSQRT(DABS(DM12-X))
      if (dabs(s).lt.1.d-8) then
       s = 1.d-10     
      endif
      if (dabs(d).lt.1.d-8) then
       d = 1.d-10
      endif
      KLAM=(DM*DM/(X*X)-SM/X)/S/D
      ANF=-1.D0/X+DM/(X*X)*LM
      IF (X.LT.DM12) GOTO 30
      IF (X.GT.SM12) GOTO 40
      FACT=2.D0*DATAN(D/S)
      GOTO 41
30    EPS=1.D0
      FACT=DLOG(DABS((S+D)/(S-D)))
      GOTO 41
40    EPS=-1.D0
      FACT=-DLOG(DABS((S+D)/(S-D)))
41    CONTINUE
      DERIV=ANF-KLAM*FACT
      P0=DERIV
      B1P=.5D0-LM-2.D0*B1-B0+(XM2-XM1-X)*DERIV
      P1=B1P/2.D0/X
      RETURN
      END
C
C ================================================================
C
C**************************************************************
C                                                             *
C  THE SCALAR VERTEX INTEGRAL WITH EQUAL EXTERNAL MASSES MF   *
C                                                             *
C**************************************************************
C
      COMPLEX*16 FUNCTION CSCAL(S,MF,M1,M2,M3)
C
C  S = MOMENTUM TRANSFER; M1,M2,M3  ARE THE INTERNAL MASSES
C
      IMPLICIT REAL*8 (A-Y)
      COMPLEX*16 Z1,Z2,Z11,Z12,Z21 ,Z22,CL1,CL2,CL3,CSPENU,SPENCE,
     &           INT,DCMPLX
      XMF=MF*MF
C.........XMF ETC.   ARE FERMION AND BOSON MASSES SQUARED
      XM1=M1*M1
      XM2=M2*M2
      XM3=M3*M3
C
C..T'HOOFT-VELTMAN PARAMETERS
      A=1.D0
      B=XMF/S
      C=-1.D0
      D=XM1-XM2-S
      E=XM3-XM1-XMF+S
      F=XM2/S
      D=D/S
      E=E/S
C..DISCRIMINANTE FOR ALPHA-EQUATION
      DISC=C*C-4.D0*A*B
      IF (DISC .LT. 0.D0) GOTO 500
      AL=(-C-DSQRT(DISC))/2.D0/B
      NENNER=C+2.D0*AL*B
C..THE FIRST INTEGRAL.............................................
      Y0=-(D+E*AL+2.D0*A+C*AL)/NENNER
      Y01=Y0-1.D0
      D1=(C+E)**2-4.D0*B*(A+D+F)
      X1=-(C+E)/2.D0/B
      IF (D1.GT.0.D0) GOTO 10
C.......COMPLEX ZEROES OF LOGARITHMS
      SQ1=DSQRT(-D1)
      X2=SQ1/2.D0/B
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=Y01/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=Y01/(Y0-Z2)
      CL1=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 15
10    CONTINUE
C........REAL ZEROES
      SQ1=DSQRT(D1)
      X2=SQ1/2.D0/B
      Y1=X1+X2
      Y2=X1-X2
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
      Y11=Y0/(Y0-Y1)
      Y12=Y01/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=Y01/(Y0-Y2)
      CL1=CSPENU(Y11,SIG1)-CSPENU(Y12,SIG2)+CSPENU(Y21,-SIG1)
     &   -CSPENU(Y22,-SIG2)
15    CONTINUE
C..THE SECOND INTEGRAL............................................
      Y0=-(D+E*AL)/NENNER/(1.D0-AL)
      Y01=Y0-1.D0
      D2=(E+D)**2-4.D0*F*(A+B+C)
      X1=-(E+D)/2.D0/(A+B+C)
      IF(D2.GT.0.D0) GOTO 20
C.......COMPLEX ZEROES OF LOGARITHMS
      SQ2=DSQRT(-D2)
      X2=SQ2/2.D0/(A+B+C)
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=Y01/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=Y01/(Y0-Z2)
      CL2=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 25
20    CONTINUE
C........REAL ZEROES
      X2=DSQRT(D2)/2.D0/(A+B+C)
      Y1=X1+X2
      Y2=X1-X2
      Y11=Y0/(Y0-Y1)
      Y12=Y01/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=Y01/(Y0-Y2)
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
      CL2=CSPENU(Y11,SIG1)-CSPENU(Y12,SIG2)+CSPENU(Y21,-SIG1)
     &   -CSPENU(Y22,-SIG2)
25    CONTINUE
C..THE THIRD INTEGRAL............................................
      Y0=(D+E*AL)/NENNER/AL
      Y01=Y0-1.D0
      D3=D*D-4.D0*A*F
      X1=-D/2.D0/A
      IF (D3.GT.0.D0) GOTO 30
C........COMPLEX ZEROES OF LOGARITHMS
      SQ3=DSQRT(-D3)
      X2=SQ3/2.D0/A
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=Y01/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=Y01/(Y0-Z2)
      CL3=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 35
30    CONTINUE
C........REAL ZEROES
      X2=DSQRT(D3)/2.D0/A
      Y1=X1+X2
      Y2=X1-X2
 31   FORMAT(1H ,3E12.4)
      Y11=Y0 /(Y0-Y1)
      Y12=Y01/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=Y01/(Y0-Y2)
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
      CL3=CSPENU(Y11,SIG1)-CSPENU(Y12,SIG2)+CSPENU(Y21,-SIG1)
     &   -CSPENU(Y22,-SIG2)
35    CONTINUE
C..SUMMATION OF THE 3 INTEGRALS ....................................
      INT=-CL1+CL2-CL3
      CSCAL=INT/NENNER/S
      GOTO 501
500   CONTINUE
C..ERROR MESSAGE FOR COMPLEX ALPHA................................
      WRITE(6,21)
21    FORMAT(1H ,'  I CANNOT HANDLE A COMPLEX ALPHA (DS)')
501   RETURN
      END
C
C ================================================================
C
      COMPLEX*16 FUNCTION CSPENU(X,SIG)
C
      IMPLICIT REAL*8(A-Y)
      COMPLEX*16 Z,CPPI,SPENCE,ZX
      PPI=3.1415926536D0
      PPI6=PPI*PPI/6.D0
      CPPI=DCMPLX(0.D0,PPI)
      IF (X.LT.1.D0) GOTO 10
      IF(X.EQ.1.D0) GOTO 11
      LX=DLOG(X)
      X1=1.D0-X
      LX1=DLOG(-X1)
      Z=DCMPLX(X1,0.D0)
      IF (SIG.GT.0.D0) GOTO 5
      CSPENU=-SPENCE(Z)+PPI6-LX*(LX1+CPPI)
      GOTO 20
5     CSPENU=-SPENCE(Z)+PPI6-LX*(LX1-CPPI)
      GOTO 20
10    ZX=DCMPLX(X,0.D0)
      CSPENU=SPENCE(ZX)
      GOTO 20
11    CSPENU=DCMPLX(PPI6,0.D0)
20    RETURN
      END
C
C ==============================================================

      double precision function dcot(x)
c
      implicit double precision (a-z)
c
      dcot = 1.d0/dtan(x)
c
      return
      end



c================================================================

      subroutine mix
c
      implicit double precision (a-z)
      real*8 ar(1:4,1:4),eig(1:4),ev(1:4,1:4),work(1:4)
      real*8 nmix(1:4,1:4)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),vmixtrans(1:2,1:2)
      real*8 mcha(1:2),mne(1:4)
      integer ierr,nrot,ic,flag
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
      double precision mdiag(1:2,1:2), mnon(1:2,1:2), mzwi(1:2,1:2)
c
c the chargino and neutralino masseigenvalues are
c calculated by their respective eigenvalues.
c Also the rotation matrixelements are given.
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /susyset/ mu,mm,mp
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
      common /err/ ic
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      double precision  MSmuLtotsq, MSmuRtotsq, MSmuneutsq
      common /smuon/  MSmuLtotsq, MSmuRtotsq, MSmuneutsq

c
c charginos ------------------------------------------------------------
c

c$$$      vdet = -mm*mu - mmw**2*2.d0*dsin(beta)*dcos(beta)
c$$$      vd = vdet/dabs(vdet)
c$$$c
c$$$      mchaa = (mm**2+mu**2+2.d0*mmw**2 + dsqrt( (mm**2 - mu**2)**2 +
c$$$     &   4.d0*mmw**4*dcos(2.d0*beta)**2 + 4.d0*mmw**2*(mm**2 + mu**2 -
c$$$     &   2.d0*mm*mu*dsin(2.d0*beta) )) )/2.d0
c$$$      mchab = (mm**2+mu**2+2.d0*mmw**2 - dsqrt( (mm**2 - mu**2)**2 +
c$$$     &   4.d0*mmw**4*dcos(2.d0*beta)**2 + 4.d0*mmw**2*(mm**2 + mu**2 -
c$$$     &   2.d0*mm*mu*dsin(2.d0*beta) )) )/2.d0
c$$$      mcha(1) =  dsqrt(dabs(mchaa))
c$$$      mcha(2) =  dsqrt(dabs(mchab))
c$$$csh --> commented out: 27.10.99
c$$$      if (vdet.lt.0.d0) then
c$$$      mcha(2) =  -mcha(2)
c$$$      endif
c$$$c
c$$$
c$$$csh --> old version, commented out: 27.10.99
c$$$      canga = datan( (-mu*mcha(1) - mm*mcha(2)) / ( dsqrt(2.d0)*mmw*
c$$$     &        dsin(beta)*mcha(1) + dsqrt(2.d0)*mmw*dcos(beta)*
c$$$     &        mcha(2) )  )
c$$$      cangb = datan( (dsqrt(2.d0)*mmw*dsin(beta)* mcha(1) + dsqrt(2.d0)
c$$$     &        *mmw*dcos(beta)*mcha(2) ) / ( mm*mcha(1) +
c$$$     &        mu*mcha(2) )  )
c$$$      if (cangb.lt.0.d0) then
c$$$       cangb = cangb + ppi
c$$$      endif
c$$$c
c$$$      write(*,*) 'Charginos:'
c$$$      write(*,*) 'Cha 1: ', mcha(1)
c$$$      write(*,*) 'Cha 2: ', mcha(2)
c$$$      write (6,*) ' det X = ',vdet
c$$$      write (6,*) ' angle = ',canga,cangb
c$$$c      write(*,*) real(dsin(canga)), real(dcos(canga))
c$$$c     $         , real(dsin(cangb)), real(dcos(cangb))
c$$$c
c$$$	umix(1,1) =  dcos(canga)
c$$$	umix(1,2) =  dsin(canga)
c$$$	umix(2,1) = -dsin(canga)
c$$$	umix(2,2) =  dcos(canga)
c$$$c
c$$$	vmix(1,1) =  dcos(cangb)
c$$$	vmix(1,2) =  dsin(cangb) ! * (-1d0) * vd
c$$$	vmix(2,1) = -dsin(cangb)
c$$$	vmix(2,2) =  dcos(cangb) ! * (-1d0) * vd


c$$$
c$$$*     Chargino-Massen:
c$$$*     ~~~~~~~~~~~~~~~~
c$$$
c$$$      sb = dsin(beta)
c$$$      cb = dcos(beta)
c$$$      sb2 = sb**2
c$$$      cb2 = cb**2
c$$$      wz = dsqrt(2d0)
c$$$      mm1=sqrt((mm**2-mu**2)**2+4d0*mmw**4*(sb2-cb2)**2+4d0*mmw**2
c$$$     &         * (mm**2+mu**2-2d0*mm*mu*2d0*sb*cb))
c$$$      mm2=mm**2+mu**2+2d0*mmw**2
c$$$      mpl=sqrt(0.5d0*(mm2+mm1))
c$$$      mmi=sqrt(0.5d0*(mm2-mm1))
c$$$      det=(-mm*mu-2d0*mmw**2*sb*cb)
c$$$      det=det/dabs(det)
c$$$      tpp=wz*mmw*(sb*mpl+det*cb*mmi)/(mm*mpl+det*mu*mmi)
c$$$      tpm=(-mu*mpl-det*mm*mmi)/(wz*mmw*(sb*mpl+det*cb*mmi))
c$$$      pp=datan(tpp)
c$$$      if (pp.le.0d0) pp=pp+ppi
c$$$      pm=datan(tpm)
c$$$      cpp=dcos(pp)
c$$$      spp=dsin(pp)
c$$$      cpm=dcos(pm)
c$$$      spm=dsin(pm)
c$$$
c$$$      umix(2,1)=cpm
c$$$      umix(2,2)=spm
c$$$      umix(1,1)=-spm
c$$$      umix(1,2)=cpm
c$$$      vmix(2,1)=cpp
c$$$      vmix(2,2)=spp
c$$$      vmix(1,1)=-spp
c$$$      vmix(1,2)=cpp
c$$$
c$$$      if (det.lt.0d0) then
c$$$         vmix(1,2)=-vmix(1,2)
c$$$         vmix(1,1)=-vmix(1,1)
c$$$      endif
c$$$
c$$$      mcha(1)=mmi
c$$$      mcha(2)=mpl
c$$$
c$$$c      umix(1,1) = umix(1,1) * (-1d0)
c$$$c      umix(1,2) = umix(1,2) * (-1d0)
c$$$c      umix(2,1) = umix(2,1) * (-1d0)
c$$$c      umix(2,2) = umix(2,2) * (-1d0)
c$$$c      vmix(1,1) = vmix(1,1) * (-1d0)
c$$$c      vmix(1,2) = vmix(1,2) * (-1d0)
c$$$c      vmix(2,1) = vmix(2,1) * (-1d0)
c$$$c      vmix(2,2) = vmix(2,2) * (-1d0)

      sb = dsin(beta)
      cb = dcos(beta)
      sb2 = sb**2
      cb2 = cb**2
      wz = dsqrt(2d0)
      mm1=sqrt((mm**2-mu**2)**2+4d0*mmw**4*(sb2-cb2)**2+4d0*mmw**2
     &         * (mm**2+mu**2-2d0*mm*mu*2d0*sb*cb))
      mm2=mm**2+mu**2+2d0*mmw**2
      mpl=sqrt(0.5d0*(mm2+mm1))
      mmi=sqrt(0.5d0*(mm2-mm1))

      aa = mm
      bb = wz * mmw * sb
      cc = wz * mmw * cb
      dd = -mu

      nn = dsqrt((cc**2 + dd**2 - mmi**2)**2 + (aa * cc + bb * dd)**2)

      umix(1,1) = 1d0/nn * (cc**2 + dd**2 - mmi**2)
      umix(1,2) = 1d0/nn * (- aa * cc - bb * dd)
      umix(2,1) = 1d0/nn * (  aa * cc + bb * dd)
      umix(2,2) = 1d0/nn * (cc**2 + dd**2 - mmi**2)
      vmix(1,1) = 1d0/nn * ((aa * dd**2 - aa * mmi**2 - bb*cc*dd)/mmi)
      vmix(1,2) = 1d0/nn * ((bb * cc**2 - bb * mmi**2 - aa*cc*dd)/mmi)
      vmix(2,1) = 1d0/nn * ((aa*bb*dd + cc * mpl**2 - bb**2 * cc)/mpl)
      vmix(2,2) = 1d0/nn * ((aa*bb*cc + dd * mpl**2 - aa**2 * dd)/mpl)

      mcha(1) = mmi
      mcha(2) = mpl


c
c check determinante,orthog.
c
c       vdet = vmix(1,1)*vmix(2,2)-vmix(1,2)*vmix(2,1)
c       write(6,*) ' det v = ',vdet
c       vdet = umix(1,1)*umix(2,2)-umix(1,2)*umix(2,1)
c       write(6,*) ' det u = ',vdet
c$$$      or1 = vmix(1,1)*vmix(1,1) + vmix(2,1)*vmix(2,1)
c$$$      or2 = vmix(1,1)*vmix(1,2) + vmix(2,1)*vmix(2,2)
c$$$      or3 = vmix(1,2)*vmix(1,1) + vmix(2,2)*vmix(2,1)
c$$$      or4 = vmix(1,2)*vmix(1,2) + vmix(2,2)*vmix(2,2)
c$$$      write(6,*) 'orth v > ',or1,or2,or3,or4
c$$$      or1 = umix(1,1)*umix(1,1) + umix(2,1)*umix(2,1)
c$$$      or2 = umix(1,1)*umix(1,2) + umix(2,1)*umix(2,2)
c$$$      or3 = umix(1,2)*umix(1,1) + umix(2,2)*umix(2,1)
c$$$      or4 = umix(1,2)*umix(1,2) + umix(2,2)*umix(2,2)
c$$$      write(6,*) 'orth u > ',or1,or2,or3,or4
c$$$ 
c$$$      msum1 = mcha(1)*vmix(1,1)*umix(1,2) + mcha(2)*vmix(2,1)*
c$$$     &        umix(2,2)
c$$$      msum2 = mcha(1)*vmix(1,2)*umix(1,1) + mcha(2)*vmix(2,2)*
c$$$     &        umix(2,1)
c$$$      msum3 = mcha(1)*vmix(1,1)*umix(1,1) + mcha(2)*vmix(2,1)*
c$$$     &        umix(2,1)
c$$$      msum4 = mcha(1)*vmix(1,2)*umix(1,2) + mcha(2)*vmix(2,2)*
c$$$     &        umix(2,2)
c$$$      write (6,*) ' mm mu  ',msum3,msum4
c$$$      write(*,*) 'correct values:', real(mm), real(-mu)
c$$$      write (6,*) ' mmw     ',msum1,msum2
c$$$      write(*,*) 'correct values:',real(mmw * dsqrt(2d0) * dcos(beta)), 
c$$$     $                             real(mmw * dsqrt(2d0) * dsin(beta)) 
c$$$
c$$$c --> check: is diagonalization really performed correctly?
c$$$       vmixtrans(1,1) = vmix(1,1)
c$$$       vmixtrans(1,2) = vmix(2,1)
c$$$       vmixtrans(2,1) = vmix(1,2)
c$$$       vmixtrans(2,2) = vmix(2,2)
c$$$       mnon(1,1) = mm
c$$$       mnon(1,2) = dsqrt(2d0) * mmw * dsin(beta)
c$$$       mnon(2,1) = dsqrt(2d0) * mmw * dcos(beta)
c$$$       mnon(2,2) = -mu
c$$$       call matmult(umix, mnon, mzwi)
c$$$       call matmult(mzwi, vmixtrans, mdiag)
c$$$       write(*,*) 'diagonalized chargino mass matrix:'
c$$$       write(*,*) real(mdiag(1,1)), real(mdiag(1,2))
c$$$       write(*,*) real(mdiag(2,1)), real(mdiag(2,2))
c$$$

c
c  neutralinos  --------------------------------------------------------
c
      ar(1,1) =  (mp)
      ar(1,2) =  0.d0
      ar(1,3) =  (-mmz * ssw * dcos(beta))
      ar(1,4) =  (mmz * ssw * dsin(beta))
      ar(2,1) =  0.d0
      ar(2,2) =  (mm)
      ar(2,3) =  (mmz * ccw * dcos(beta))
      ar(2,4) =  (-mmz * ccw * dsin(beta))
      ar(3,1) =  ar(1,3)
      ar(3,2) =  ar(2,3)
      ar(3,3) =  0.d0
      ar(3,4) =  (mu)
      ar(4,1) =  ar(1,4)
      ar(4,2) =  ar(2,4)
      ar(4,3) =  ar(3,4)
      ar(4,4) =  0.d0
c
      call jacobi2 (ar,4,4,eig,ev,NROT,flag)
c
c      write (6,*) ' Neutr. Eigenvalues : '
c      write (6,*) ' M neut. 1 = ',eig(1)
c      write (6,*) ' M neut. 2 = ',eig(2)
c      write (6,*) ' M neut. 3 = ',eig(3)
c      write (6,*) ' M neut. 4 = ',eig(4)
c      write (6,*) '   '
c
      mne(1) = eig(1)
      mne(2) = eig(2)
      mne(3) = eig(3)
      mne(4) = eig(4)
c
       ic = 0
      if (flag.eq.1) then
       ic = 1
      endif
c
c      write (6,*) ' Neutr. Eigenvectors : '
c      write (6,*) ' N_i1 = ', real(ev(1,1)),real(ev(1,2)),
c     $                        real(ev(1,3)),real(ev(1,4))
c      write (6,*) ' N_i2 = ', real(ev(2,1)),real(ev(2,2)),
c     $                        real(ev(2,3)),real(ev(2,4))
c      write (6,*) ' N_i3 = ', real(ev(3,1)),real(ev(3,2)),
c     $                        real(ev(3,3)),real(ev(3,4))
c      write (6,*) ' N_i4 = ', real(ev(4,1)),real(ev(4,2)),
c     $                        real(ev(4,3)),real(ev(4,4))
c      write (6,*)
c
      nmix(1,1) = ev(1,1)
      nmix(2,1) = ev(1,2)
      nmix(3,1) = ev(1,3)
      nmix(4,1) = ev(1,4)
c
      nmix(1,2) = ev(2,1)
      nmix(2,2) = ev(2,2)
      nmix(3,2) = ev(2,3)
      nmix(4,2) = ev(2,4)
c
      nmix(1,3) =  ev(3,1)
      nmix(2,3) =  ev(3,2)
      nmix(3,3) =  ev(3,3)
      nmix(4,3) =  ev(3,4)
c
      nmix(1,4) = ev(4,1)
      nmix(2,4) = ev(4,2)
      nmix(3,4) = ev(4,3)
      nmix(4,4) = ev(4,4)
c

c$$$c
c$$$c check ortho
c$$$c
c$$$      or1 = nmix(1,1)*nmix(1,1) + nmix(2,1)*nmix(2,1) +
c$$$     &      nmix(3,1)*nmix(3,1) + nmix(4,1)*nmix(4,1)
c$$$      or2 = nmix(1,2)*nmix(1,2) + nmix(2,2)*nmix(2,2) +
c$$$     &      nmix(3,2)*nmix(3,2) + nmix(4,2)*nmix(4,2)
c$$$      or3 = nmix(1,3)*nmix(1,3) + nmix(2,3)*nmix(2,3) +
c$$$     &      nmix(3,3)*nmix(3,3) + nmix(4,3)*nmix(4,3)
c$$$      or4 = nmix(1,4)*nmix(1,4) + nmix(2,4)*nmix(2,4) +
c$$$     &      nmix(3,4)*nmix(3,4) + nmix(4,4)*nmix(4,4)
c$$$      or5 = nmix(1,1)*nmix(1,2) + nmix(2,1)*nmix(2,2) +
c$$$     &      nmix(3,1)*nmix(3,2) + nmix(4,1)*nmix(4,2)
c$$$      or6 = nmix(1,1)*nmix(1,3) + nmix(2,1)*nmix(2,3) +
c$$$     &      nmix(3,1)*nmix(3,3) + nmix(4,1)*nmix(4,3)
c$$$      or7 = nmix(1,2)*nmix(1,3) + nmix(2,2)*nmix(2,3) +
c$$$     &      nmix(3,2)*nmix(3,3) + nmix(4,2)*nmix(4,3)
c$$$      or8 = nmix(1,2)*nmix(1,4) + nmix(2,2)*nmix(2,4) +
c$$$     &      nmix(3,2)*nmix(3,4) + nmix(4,2)*nmix(4,4)
c$$$      or9 = nmix(1,3)*nmix(1,4) + nmix(2,3)*nmix(2,4) +
c$$$     &      nmix(3,3)*nmix(3,4) + nmix(4,3)*nmix(4,4)
c$$$      write(*,*) 'Neutralino Ortho-Check:'
c$$$      write (6,*) or1,or2,or3
c$$$      write (6,*) or4,or5,or6
c$$$      write (6,*) or7,or8,or9
c$$$c
c$$$c     ar(1,1) =  (mp)
c$$$c     ar(1,2) =  0.d0
c$$$c     ar(1,3) =  (-mmz * ssw * dcos(beta))
c$$$c     ar(1,4) =  (mmz * ssw * dsin(beta))
c$$$c     ar(2,1) =  0.d0
c$$$c     ar(2,2) =  (mm)
c$$$c     ar(2,3) =  (mmz * ccw * dcos(beta))
c$$$c     ar(2,4) =  (-mmz * ccw * dsin(beta))
c$$$c     ar(3,1) =  ar(1,3)
c$$$c     ar(3,2) =  ar(2,3)
c$$$c     ar(3,3) =  0.d0
c$$$c     ar(3,4) =  (-mu)
c$$$c     ar(4,1) =  ar(1,4)
c$$$c     ar(4,2) =  ar(2,4)
c$$$c     ar(4,3) =  ar(3,4)
c$$$c     ar(4,4) =  0.d0
c$$$c
c$$$c     do 327 l = 1,4
c$$$c      do 328 m = 1,4
c$$$c     xx = 0.d0
c$$$c     do 325 ii = 1,4
c$$$c      do 326 k = 1,4
c$$$c       xx =  nmix(l,ii) * ar(ii,k) * nmix(m,k) + xx
c$$$c 326    continue
c$$$c 325   continue
c$$$c      write (6,*) xx
c$$$c 328    continue
c$$$c 327    continue
c
c sfermionmassen  -----------------------------------------------------
c
c      ic = 0
c
      a = mq2 + mmz**2*dcos(2.d0*beta)*(0.5-ssw2*2.d0/3.d0) + mup**2
      d = mu2 + mmz**2*dcos(2.d0*beta)*ssw2*2.d0/3.d0 + mup**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 500.d0**2
c      d = 501.d0**2
      b = mup*(mssupq + mu*dcot(beta))
      mupsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mupsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
c     if ((a+d-dsqrt((a-d)**2+4.d0*b**2)).le.4.d4) then
c      write (6,*) ' M_sup < 100 GeV '
c      ic = 1
c     endif
      ang7 = datan(2.d0*b/(a-d))/2.d0
c
      a = mb2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2/3.d0) + mdn**2
      d = md2 - mmz**2*dcos(2.d0*beta)*ssw2/3.d0 + mdn**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 500.d0**2
c      d = 501.d0**2
      b = mdn*(mssdnq + mu*dtan(beta))
      mdnsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mdnsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
c     if ((a+d-dsqrt((a-d)**2+4.d0*b**2)).le.4.d4) then
c      write (6,*) ' M_sdn < 100 GeV '
c      ic = 1
c     endif
      ang8 = datan(2.d0*b/(a-d))/2.d0
c
      a = mq2 + mmz**2*dcos(2.d0*beta)*(0.5-ssw2*2.d0/3.d0) + mch**2
      d = mu2 + mmz**2*dcos(2.d0*beta)*ssw2*2.d0/3.d0 + mch**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 800.d0**2
c      d = 801.d0**2
      b = mch*(mssupq + mu*dcot(beta))
      mchsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mchsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang9 = datan(2.d0*b/(a-d))/2.d0
c
      a = mb2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2/3.d0) + mst**2
      d = md2 - mmz**2*dcos(2.d0*beta)*ssw2/3.d0 + mst**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 800.d0**2
c      d = 801.d0**2
      b = mst*(mssdnq + mu*dtan(beta))
      mstsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mstsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang10 = datan(2.d0*b/(a-d))/2.d0
c
      a = mq2 + mmz**2*dcos(2.d0*beta)*(0.5-ssw2*2.d0/3.d0) + mtt**2
      d = mu2 + mmz**2*dcos(2.d0*beta)*ssw2*2.d0/3.d0 + mtt**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 80.d0**2
c      d = 81.d0**2
      b = mtt*(mssupq + mu*dcot(beta))
      mtsr =  dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mtsl =  dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang11 = datan(2.d0*b/(a-d))/2.d0

c --> note:
c     The bug for the top-type squarks has been fixed for the top only.
c     Concerning the bottom-type squarks, mq2 === md2 (= msusytl**2)
c     so the problem should not arise when squark masses are calculated
c     with the help of these unphysical parameters. 
c     The bottom sector agrees with the sbottom routines in def2.f

c$$$      write(*,*) '-------------------------------------------'
c$$$      write(*,*) 'subroutine mix: stop sector'
c$$$      write(*,*) 'LL, LR, RL, RR:'
c$$$      write(*,*) real(a), real(b), real(b), real(d)
c$$$      write(*,*) 'MSt1, MSt2, ctt, stt'
c$$$      write(*,*) real(mtsl), real(mtsr), 
c$$$     $           real(dcos(ang11)), real(dsin(ang11))
c$$$      write(*,*) '-------------------------------------------'
c$$$      mnon(1,1) = a
c$$$      mnon(1,2) = b
c$$$      mnon(2,1) = b
c$$$      mnon(2,2) = d
c$$$      call diagonalization2(dcos(ang11), dsin(ang11), mnon, mdiag)
c$$$      write(*,*) "stop: diagonalization check:"
c$$$      write(*,*) real(dsqrt(mdiag(1,1))), real(mdiag(1,2))
c$$$      write(*,*) real(mdiag(2,1)), real(dsqrt(mdiag(2,2)))
c$$$      write(*,*) '-------------------------------------------'
c
      a = mb2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2/3.d0) + mbb**2
      d = md2 - mmz**2*dcos(2.d0*beta)*ssw2/3.d0 + mbb**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
c      a = 250.d0**2
c      d = 251.d0**2
      b = mbb*(mssdnq + mu*dtan(beta))
c --> correction factor ffak implemented at 06/29/99
c     ffak should correct for problem arising for tan beta < 1
      mbsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mbsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      if ((dabs(mbsl).le.100.d0).or.(dabs(mbsr).le.100.d0)) then
c       write (6,*) ' M_sb < 100 GeV '
c       ic = 1
      endif
      ang12 = datan(2.d0*b/(a-d))/2.d0
c$$$      write(*,*) '-------------------------------------------'
c$$$      write(*,*) 'subroutine mix: sbottom sector'
c$$$      write(*,*) 'LL, LR, RL, RR:'
c$$$      write(*,*) real(a), real(b), real(b), real(d)
c$$$      write(*,*) 'MSb1, MSb2, ctb, stb'
c$$$      write(*,*) real(mbsl), real(mbsr), 
c$$$     $           real(dcos(ang12)), real(dsin(ang12))
c$$$      write(*,*) '-------------------------------------------'
c$$$      mnon(1,1) = a
c$$$      mnon(1,2) = b
c$$$      mnon(2,1) = b
c$$$      mnon(2,2) = d
c$$$      call diagonalization2(dcos(ang12), dsin(ang12), mnon, mdiag)
c$$$      write(*,*) "sbottom: diagonalization check:"
c$$$      write(*,*) real(dsqrt(mdiag(1,1))), real(mdiag(1,2))
c$$$      write(*,*) real(mdiag(2,1)), real(dsqrt(mdiag(2,2)))
c$$$      write(*,*) '-------------------------------------------'

c
      a = mf2 + mmz**2*dcos(2.d0*beta)*0.5
      d = 0.d0
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = 0.d0
      mvesl = dsqrt(dabs((a+d+dsqrt((a-d)**2+4.d0*b**2))/2.d0))
c      write(*,*) 'mvesl:', mvesl, a, mf2, mmz, beta
      if (mvesl.le.1d-2) mvesl = 0.1d0
      mvesr = 1.d-5
      ang1 = datan(2.d0*b/(a-d))/2.d0
c
      a = mf2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2) + mel**2
      d = mfd2 - mmz**2*dcos(2.d0*beta)*ssw2 + mel**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = mel*(mssdnl + mu*dtan(beta))
      melsl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      melsr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang2 = datan(2.d0*b/(a-d))/2.d0
c
      a = mf2 + mmz**2*dcos(2.d0*beta)*0.5
      MSmuneutsq = a
      d = 0.d0
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = 0.d0
      mvmsl = dsqrt(dabs((a+d+dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      if (mvmsl.le.1d-2) mvmsl = 0.1d0
      mvmsr = 1.d-5
      ang3 = datan(2.d0*b/(a-d))/2.d0
c
      a = mf2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2) + mmu**2
      d = mfd2 - mmz**2*dcos(2.d0*beta)*ssw2 + mmu**2
      MSmuLtotsq = a
      MSmuRtotsq = d
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = mmu*(mssdnl + mu*dtan(beta))
c      b = 0.d0
      mmusl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mmusr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang4 = datan(2.d0*b/(a-d))/2.d0
c
      a = mf2 + mmz**2*dcos(2.d0*beta)*0.5
      d = 0.d0
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = 0.d0
      mvtsl = dsqrt(dabs((a+d+dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      if (mvtsl.le.1d-2) mvtsl = 0.1d0
      mvtsr = 1.d-5
      ang5 = datan(2.d0*b/(a-d))/2.d0
c
      a = mf2 + mmz**2*dcos(2.d0*beta)*(-0.5d0+ssw2) + mta**2
      d = mfd2 - mmz**2*dcos(2.d0*beta)*ssw2 + mta**2
      ffak = 1d0
      if (a.ge.d) ffak = -1d0
      b = mta*(mssdnl + mu*dtan(beta))
      mtasl = dsqrt(dabs((a+d-ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      mtasr = dsqrt(dabs((a+d+ffak*dsqrt((a-d)**2+4.d0*b**2))/2.d0))
      ang6 = datan(2.d0*b/(a-d))/2.d0
c
      return
      end

c==================================================================
      SUBROUTINE RFIND (FUNC,Xa,Xb,XACC, RTXH)
      INTEGER MAXIT,J
      DOUBLE PRECISION FUNC,X1,X2,XACC,RTXH,
     &       xa,xb,xmid,DX,F,FL,fmid,SSWAP,XL
      EXTERNAL FUNC
      PARAMETER (MAXIT=50)
      x1=xa
      x2=xb
      F     = FUNC(X1)
      Fmid  = FUNC(X2)
c      write (6,*) 'X:',x1,x2
c      write (6,*) 'F:',f,fmid
      IF (f*fmid.gt.0.d0) THEN
c        write (6,*) ' ERROR : can not find root '
        rtxh=119.9999d0
       else
       DO 11 J=1,MAXIT
       xmid = (x2+x1)/2.d0 
       fl = FUNC(xmid)
       if (f*fl.lt.0.d0) then
         x2 = xmid
       else
         x1 = xmid
         f  = fl      
       endif
       dx = x2-x1
       rtxh = (x1+x2)/2.d0
c       write (6,*) x1,x2
c
       IF (DABS(DX).LT.XACC.OR.F.EQ.0.D0) RETURN
11     CONTINUE
       PAUSE 'RTSEC EXCEED MAXIMUM ITERATIONS'
       endif
       END
c=====================================================================
c
      subroutine genzquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
      implicit double precision (a-z)
      real*8 b0,b1,pb0,pb1
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c          ......../          \........                 .     .
c        q         \          /        q                  .  .
c                   \________/                q ...................... q
c                        m2
c
c
      pb0 = 0.d0
      pb1 = 0.d0
      pb22 = 0.d0
c
      if(dabs(m2).le.1d-7) then
       if(dabs(m1).le.1d-7) then
	 if (dabs(q2).le.1d-7) then
	 b0  = delta(epsilon,muee,1.d0) - dlog(m1*m2)
	 b1  = -0.5d0 * (delta(epsilon,muee,1.d0)- dlog(m1*m2))
	 b22 = 0.d0
	 else
	 b0 = delta(epsilon,muee,1.d0) + 2.d0 - dlog(q2)
	 b1 = -0.5d0 * (delta(epsilon,muee,1.d0) + 2.d0) +
     &         0.5d0 * dlog(q2)
	 b22 = (q2 * b1/2.d0 - q2/6.d0) / 3.d0
	 endif
       else
	if (q2.le.1.d-8) then
       b0 = delta(epsilon,muee,1.d0) - dlog(m1**2) + 1.d0
       b1 = -0.5d0 * ( delta(epsilon,muee,1.d0) - dlog(m1**2) )
     &      - 0.25d0
       b22  = (aa(dabs(m1))/2.d0 -
     &         m1**2 * b1/2.d0 + m1**2/2.d0 )/3.d0
	else
       b0 = delta(epsilon,muee,1.d0) + 1.d0 - 2.d0 *
     &      dlog(dabs(m1)) + ff(q2,0.d0,dabs(m1))
       b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &     +1.d0) + (m1**2 - q2)/(2.d0 * q2) * ff(q2,0.d0,dabs(m1))
       b22  = (aa(dabs(m1))/2.d0 +  (q2 -
     &        m1**2) * b1/2.d0 + (m1**2)/2.d0
     &        - q2/6.d0)/3.d0
       pb22 = ( (b1 + (q2 - m1**2)
     &         * pb1)/2.d0 - 1.d0/6.d0) / 3.d0
	endif
       endif
      else
      if (q2.le.1.d-8) then
      b0 = delta(epsilon,muee,1.d0) - dlog(dabs(m1*m2)) + 1.d0
      b1 = -0.5d0 - 0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2))
       if (dabs(dabs(m1)-dabs(m2)).le.1.d-6) then
       b0 = b0 - 1.d0
       b1 = b1 + 0.5d0
       else
       b0 = b0 - (m1**2+m2**2)/(m1**2-m2**2)*dlog(dabs(m1/m2))
       b1 = b1 + m2**2/(m2**2-m1**2)*dlog(dabs(m2/m1)) +
     &      1.d0/2.d0/(m1**2-m2**2) * ( (m2**2+m1**2)/2.d0
     &      - m2**2*m1**2/(m2**2-m1**2)*dlog(m2**2/m1**2) )
       endif
      else
      call bquer2(q2,dabs(m2),dabs(m1),
     &             b0,b1,pb0,pb1)
       b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
       b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
      endif
      b22  = (aa(dabs(m1))/2.d0 + m2**2 * b0 + (q2 + m2**2 -
     &        m1**2) * b1/2.d0 + (m1**2 + m2**2)/2.d0
     &        - q2/6.d0)/3.d0
       pb22 = (m2**2 * pb0 + (b1 + (q2 + m2**2 - m1**2)
     &         * pb1)/2.d0 - 1.d0/6.d0) / 3.d0
      endif
c
      if (typ.eq.1) then
	siga  = 4.d0 * b22
	dsiga = 4.d0 * pb22
c       siga = 2.d0*(m1**2+m2**2-q2/3.d0)/epsilon
      else
      if (typ.eq.2) then
	siga  = -b0
	dsiga = -pb0
c       siga = -2.d0/epsilon
      else
      if (typ.eq.3) then
	siga  = 2.d0 * (-2.d0 * b22 + aa(m1) + m2**2 * b0
     &          + q2 * b1)
	dsiga = 2.d0 * (-2.d0 * pb22 + m2**2 * pb0 + b1 + q2 * pb1)
c       siga  = 2.d0 * ( m2**2 * 2.d0 /epsilon - q2/epsilon
c    &          -( m1**2 + m2**2 - q2/3.d0 ) /epsilon  + 2.d0 *
c    &           m1**2/epsilon  )
      else
      if (typ.eq.4) then
	siga  = 8.d0 * ((at * a + bt * b) * (-2.d0 * b22 +
     &         aa(dabs(m1)) + m2**2 * b0 + q2 * b1) -
     &         (bt * a + at * b) * (m1 * m2) * b0 )
	dsiga = 8.d0 * ((at * a + bt * b) * (-2.d0 * pb22 +
     &         m2**2 * pb0 + q2 * pb1 + b1) -
     &         (bt * a + at * b) * (m1 * m2) * pb0 )
c       siga = 8.d0 * ((at * a + bt * b) * (m1**2 + m2**2 -
c    &         2.d0*q2/3.d0) - 2.d0 * (bt * a + at * b) * m1 * m2)
c    &         / epsilon
      else
      if (typ.eq.5) then
	siga  = (10.d0 * b22 + (4.d0 * q2 + m1**2 + m2**2) *
     &       b0 + aa(m1) + aa(m2) - 2.d0 * ( m1**2 + m2**2
     &       - q2/3.d0 ) )
	dsiga = (10.d0 * pb22 + 4.d0 * b0 + (4.d0 * q2 +
     &           m1**2 + m2**2) * pb0 - 2.d0/3.d0)
c       siga = (9.d0*(m1**2+m2**2)+19.d0/3.d0*q2)/epsilon
      else
      if (typ.eq.6) then
	siga  = -b22
	dsiga = -pb22
c       siga = -(m1**2+m2**2-q2/3.d0)/2.d0/epsilon
      else
      if (typ.eq.7) then
	siga  = -aa(m1)
	dsiga = 0.d0
c       siga = -2.d0*m1**2/epsilon
      else
      if (typ.eq.8) then
	siga  =  6.d0 * aa(m1) - 4.d0 * m1**2
	dsiga =  0.d0
c       siga = 12.d0*m1**2/epsilon
      else
       write (6,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
      endif
      endif
      endif
      endif
c
      return
      end
c
c ----------------------------------------------------------
c
      subroutine sigmaz (s ,sigb,sigs,sigf,sigc,sigt)
c
c     selfenergy of z boson
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)


c      write(*,*) 'Z-SE:', real(maa), real(mhh)
c     
c boson loops
c
c  notation :
c     genzquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c o.k :
      call genzquad (1,s,maa,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2/
     &       (4.d0*ssw2*ccw2)*siga
c o.k :
      call genzquad (1,s,maa,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2/(4.d0*ppi)**2 * dcos(beta-alpha)**2/
     &       (4.d0*ssw2*ccw2)*siga
c o.k :
      call genzquad (1,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = -elec2/(4.d0*ppi)**2 * (ccw2-ssw2)**2/(4.d0*ccw2*ssw2)*siga
c o.k :
      call genzquad (1,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = -elec2/(4.d0*ppi)**2 * dcos(beta-alpha)**2/
     &       (4.d0*ssw2*ccw2)*siga
c o.k :
      call genzquad (1,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2/
     &       (4.d0*ccw2*ssw2)*siga
c o.k :
      call genzquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*(ccw2-ssw2)**2/(4.d0*ccw2*ssw2)*siga
c o.k :
      call genzquad (2,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2*mmz**2*dcos(beta-alpha)**2/
     &       (ssw2*ccw2)*siga
c o.k :
      call genzquad (2,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2*mmz**2*dsin(beta-alpha)**2/
     &       (ssw2*ccw2)*siga
c o.k :
      call genzquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * 2.d0*mmz**2*ssw2 * siga
c
      v3 = (-0.5d0 + 2.d0 * ssw2 )/(2.d0 * ssw * ccw)
      a3 =  -0.5d0 / (2.d0 * ssw * ccw)
      a = (v3 + a3) / 2.d0
      b = (v3 - a3) / 2.d0
c o.k :
c      write(*,*) 'Z-SE: vor top10'
      call genzquad (4,s,mel,mel,a,a,b,b, siga,dsiga)
      top10a = -elec2 / (4.d0 * ppi)**2 * siga
c
      call genzquad (4,s,mmu,mmu,a,a,b,b, siga,dsiga)
      top10b = -elec2 / (4.d0 * ppi)**2 * siga
c
      call genzquad (4,s,mta,mta,a,a,b,b, siga,dsiga)
      top10c = -elec2 / (4.d0 * ppi)**2 * siga
c
      v3 =  0.5d0 / (2.d0 * ssw * ccw)
      a3 =  0.5d0 / (2.d0 * ssw * ccw)
      a = (v3 + a3) / 2.d0
      b = (v3 - a3) / 2.d0
c
      if (s.lt.1.d-7) then
      call genzquad (4,s,1.d-5,1.d-5,a,a,b,b, siga,dsiga)
      else
      call genzquad (4,s,0.d0,0.d0,a,a,b,b, siga,dsiga)
      endif
      top10n = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      v3 = (0.5d0 - 4.d0/3.d0 * ssw2 )/(2.d0 * ssw * ccw)
      a3 =  0.5d0 / (2.d0 * ssw * ccw)
      a = (v3 + a3) / 2.d0
      b = (v3 - a3) / 2.d0
c
      call genzquad (4,s,mup,mup,a,a,b,b, siga,dsiga)
      top10d = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      call genzquad (4,s,mch,mch,a,a,b,b, siga,dsiga)
      top10e = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      call genzquad (4,s,mtt,mtt,a,a,b,b, siga,dsiga)
      top10f = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      v3 = (-0.5d0 + 2.d0/3.d0 * ssw2 )/(2.d0 * ssw * ccw)
      a3 =  -0.5d0 / (2.d0 * ssw * ccw)
      a = (v3 + a3) / 2.d0
      b = (v3 - a3) / 2.d0
c
      call genzquad (4,s,mdn,mdn,a,a,b,b, siga,dsiga)
      top10g = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      call genzquad (4,s,mst,mst,a,a,b,b, siga,dsiga)
      top10h = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      call genzquad (4,s,mbb,mbb,a,a,b,b, siga,dsiga)
      top10i = -elec2 / (4.d0 * ppi)**2 * siga * 3.d0
c
      if (selec.ge.3) then
         top10 = top10a + top10b + top10c + top10d + top10e +
     &           top10f + top10g + top10h + top10i + top10n
      elseif (selec.eq.2) then
         top10 = top10f + top10i
      elseif (selec.eq.1) then
         top10 = top10f
      else
         write(*,*) "Error in Sigma-Z: selec out or range"
      endif
      a = 1.d0
c
c      write(*,*) 'Z-SE: vor top11'
      call genzquad (1,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genzquad (1,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genzquad (1,s,melsr,melsl,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genzquad (1,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top11a = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)*((dcos(ang2)**2
     &  -2.d0*ssw2)**2*siga1 + (dsin(ang2)*dcos(ang2))**2*(siga2+siga3)
     &  + (dsin(ang2)**2-2.d0*ssw2)**2*siga4 )
c
      call genzquad (1,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genzquad (1,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genzquad (1,s,mmusr,mmusl,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genzquad (1,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top11b = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)*((dcos(ang4)**2
     &  -2.d0*ssw2)**2*siga1 + (dsin(ang4)*dcos(ang4))**2*(siga2+siga3)
     &  + (dsin(ang4)**2-2.d0*ssw2)**2*siga4 )
c
      call genzquad (1,s,mtasl,mtasl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mtasl,mtasr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mtasr,mtasl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mtasr,mtasr,a,a,a,a, siga4,dsiga4)
      top11c = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)*((dcos(ang6)**2
     &  -2.d0*ssw2)**2*siga1 + (dsin(ang6)*dcos(ang6))**2*(siga2+siga3)
     &  + (dsin(ang6)**2-2.d0*ssw2)**2*siga4 )
c
      call genzquad (1,s,mvesl,mvesl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mvesl,mvesr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mvesl,mvesr,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mvesr,mvesr,a,a,a,a, siga4,dsiga4)
      top11d = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)* ( dcos(ang1)**4
     & * siga1 + (dsin(ang1)*dcos(ang1))**2*(siga2+siga3) +
     & dsin(ang1)**4 * siga4 )
c
      call genzquad (1,s,mvmsl,mvmsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mvmsl,mvmsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mvmsl,mvmsr,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mvmsr,mvmsr,a,a,a,a, siga4,dsiga4)
      top11e = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)* ( dcos(ang3)**4
     & * siga1 + (dsin(ang3)*dcos(ang3))**2*(siga2+siga3) +
     & dsin(ang3)**4 * siga4 )
c
      call genzquad (1,s,mvtsl,mvtsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mvtsl,mvtsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mvtsl,mvtsr,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mvtsr,mvtsr,a,a,a,a, siga4,dsiga4)
      top11f = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2)* ( dcos(ang5)**4
     & * siga1 + (dsin(ang5)*dcos(ang5))**2*(siga2+siga3) +
     & dsin(ang5)**4 * siga4 )
c
      call genzquad (1,s,mupsl,mupsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mupsl,mupsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mupsr,mupsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mupsr,mupsr,a,a,a,a, siga4,dsiga4)
      top11g = -elec2/(4.d0*ppi)**2*3.d0/(4.d0*ssw2*ccw2) * (
     & (-dcos(ang7)**2+4.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang7)*
     & dcos(ang7))**2*(siga2+siga3) + (-dsin(ang7)**2+4.d0*ssw2/3.d0)**2
     & * siga4 )
c
      call genzquad (1,s,mchsl,mchsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mchsl,mchsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mchsr,mchsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mchsr,mchsr,a,a,a,a, siga4,dsiga4)
      top11h = -elec2/(4.d0*ppi)**2*3.d0/(4.d0*ssw2*ccw2) * (
     & (-dcos(ang9)**2+4.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang9)*
     & dcos(ang9))**2*(siga2+siga3) + (-dsin(ang9)**2+4.d0*ssw2/3.d0)**2
     & * siga4 )
c
      call genzquad (1,s,mtsl,mtsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mtsl,mtsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mtsr,mtsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mtsr,mtsr,a,a,a,a, siga4,dsiga4)
      top11i = -elec2/(4.d0*ppi)**2*3.d0/(4.d0*ssw2*ccw2) * (
     & (-dcos(ang11)**2+4.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang11)*
     & dcos(ang11))**2*(siga2+siga3)+(-dsin(ang11)**2+4.d0*ssw2/3.d0)**2
     & * siga4 )
c
      call genzquad (1,s,mdnsl,mdnsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mdnsl,mdnsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mdnsr,mdnsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mdnsr,mdnsr,a,a,a,a, siga4,dsiga4)
      top11j = -elec2/(4.d0*ppi)**2 * 3.d0 /(4.d0*ssw2*ccw2) * (
     & (dcos(ang8)**2-2.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang8)*
     & dcos(ang8))**2*(siga2+siga3) + (dsin(ang8)**2-2.d0*ssw2/3.d0)**2
     & * siga4 )
c
      call genzquad (1,s,mstsl,mstsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mstsl,mstsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mstsr,mstsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mstsr,mstsr,a,a,a,a, siga4,dsiga4)
      top11k = -elec2/(4.d0*ppi)**2 * 3.d0 /(4.d0*ssw2*ccw2) * (
     & (dcos(ang10)**2-2.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang10)*
     & dcos(ang10))**2*(siga2+siga3) + 
     $     (dsin(ang10)**2-2.d0*ssw2/3.d0)**2
     & * siga4 )
c
      if (delmbresum.eq.1) then
      call genzquad (1,s,mbsl,mbsl,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,mbsl,mbsr,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,mbsr,mbsl,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,mbsr,mbsr,a,a,a,a, siga4,dsiga4)
      top11l = -elec2/(4.d0*ppi)**2 * 3.d0 /(4.d0*ssw2*ccw2) * (
     & (dcos(ang12)**2-2.d0*ssw2/3.d0)**2 * siga1 + (dsin(ang12)*
     & dcos(ang12))**2*(siga2+siga3) 
     $     + (dsin(ang12)**2-2.d0*ssw2/3.d0)**2
     & * siga4 )
      else
      call genzquad (1,s,msb1dmb,msb1dmb,a,a,a,a, siga1,dsiga1)
      call genzquad (1,s,msb1dmb,msb2dmb,a,a,a,a, siga2,dsiga2)
      call genzquad (1,s,msb2dmb,msb1dmb,a,a,a,a, siga3,dsiga3)
      call genzquad (1,s,msb2dmb,msb2dmb,a,a,a,a, siga4,dsiga4)
      top11l = -elec2/(4.d0*ppi)**2 * 3.d0 /(4.d0*ssw2*ccw2) * (
     & (dcos(tsbdmb)**2-2.d0*ssw2/3.d0)**2 * siga1 + (dsin(tsbdmb)*
     & dcos(tsbdmb))**2*(siga2+siga3) 
     $     + (dsin(tsbdmb)**2-2.d0*ssw2/3.d0)**2
     & * siga4 )
      endif
c
      if (selec.ge.3) then
      top11 = top11a + top11b + top11c + top11d + top11e +
     &        top11f + top11g + top11h + top11i + top11j +
     &        top11k + top11l
      elseif (selec.eq.2) then
         top11 = top11i + top11l
      elseif (selec.eq.1) then
         top11 = top11i
      else
         write(*,*) "Error in Sigma-Z: selec out or range"
      endif
c o.k :
c      write(*,*) 'Z-SE: vor top15'
      call genzquad (5,s,mmw,mmw,a,a,a,a, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2 * ccw2/ssw2 * siga
c o.k :
      call genzquad (6,s,mmw,mmw,a,a,a,a, siga,dsiga)
      top16 = -elec2/(4.d0*ppi)**2 * 2.d0*ccw2/ssw2 * siga
c o.k :
      call genzquad (7,s,mhh,a,a,a,a,a, siga,dsiga)
      top17a = -elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga
c o.k :
      call genzquad (7,s,mlh,a,a,a,a,a, siga,dsiga)
      top17b = -elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga
c o.k :
      call genzquad (7,s,maa,a,a,a,a,a, siga,dsiga)
      top17c = -elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga
c
      top17 = top17a + top17b + top17c
c o.k :
      call genzquad (7,s,mhp,a,a,a,a,a, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2*(ccw2-ssw2)**2/(2.d0*ssw2*ccw2)*siga
c o.k :
      call genzquad (7,s,mmz,a,a,a,a,a, siga,dsiga)
      top19 = -elec2/(4.d0*ppi)**2/(4.d0*ssw2*ccw2) * siga
c o.k :
      call genzquad (7,s,mmw,a,a,a,a,a, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2*(ccw2-ssw2)**2/(2.d0*ssw2*ccw2)*siga
c
c      write(*,*) 'Z-SE: vor top21'
c      write(*,*) 'Z-SE: top21a'
      call genzquad (7,s,melsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,melsr,a,a,a,a,a, siga2,dsiga2)
      top21a = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * ((
     & (-1.d0+2.d0*ssw2)**2*dcos(ang2)**2+4.d0*ssw2**2*dsin(ang2)**2)
     & * siga1 + ((-1.d0+2.d0*ssw2)**2*dsin(ang2)**2+4.d0*ssw2**2*
     & dcos(ang2)**2 )* siga2 )
c
c      write(*,*) 'Z-SE: top21b'
      call genzquad (7,s,mmusl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mmusr,a,a,a,a,a, siga2,dsiga2)
      top21b = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * ((
     & (-1.d0+2.d0*ssw2)**2*dcos(ang4)**2+4.d0*ssw2**2*dsin(ang4)**2)
     & * siga1 + ((-1.d0+2.d0*ssw2)**2*dsin(ang4)**2+4.d0*ssw2**2*
     & dcos(ang4)**2 )* siga2 )
c                              
c      write(*,*) 'Z-SE: top21c'
      call genzquad (7,s,mtasl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mtasr,a,a,a,a,a, siga2,dsiga2)
      top21c = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * ((
     & (-1.d0+2.d0*ssw2)**2*dcos(ang6)**2+4.d0*ssw2**2*dsin(ang6)**2)
     & * siga1 + ((-1.d0+2.d0*ssw2)**2*dsin(ang6)**2+4.d0*ssw2**2*
     & dcos(ang6)**2 )* siga2 )
c
c      write(*,*) 'Z-SE: top21d'
      call genzquad (7,s,mvesl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mvesr,a,a,a,a,a, siga2,dsiga2)
      top21d = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2*ccw2) 
     $     * ( dcos(ang1)**2 *
     & siga1 + dsin(ang1)**2 * siga2 )
c
c      write(*,*) 'Z-SE: top21e'
      call genzquad (7,s,mvmsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mvmsr,a,a,a,a,a, siga2,dsiga2)
      top21e = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2*ccw2) 
     $     * ( dcos(ang3)**2 *
     & siga1 + dsin(ang3)**2 * siga2 )
c
c      write(*,*) 'Z-SE: top21f'
      call genzquad (7,s,mvtsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mvtsr,a,a,a,a,a, siga2,dsiga2)
      top21f = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2*ccw2) 
     $     * ( dcos(ang5)**2 *
     & siga1 + dsin(ang5)**2 * siga2 )
c
c      write(*,*) 'Z-SE: top21g'
      call genzquad (7,s,mupsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mupsr,a,a,a,a,a, siga2,dsiga2)
      top21g = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((1.d0-4.d0*ssw2/3.d0)**2*dcos(ang7)**2+(4.d0*ssw2/3.d0)**2*
     & dsin(ang7)**2)*siga1 
     $     + ((1.d0-4.d0*ssw2/3.d0)**2*dsin(ang7)**2+
     & (4.d0*ssw2/3.d0)**2*dcos(ang7)**2)*siga2 )
c
c      write(*,*) 'Z-SE: top21h'
      call genzquad (7,s,mchsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mchsr,a,a,a,a,a, siga2,dsiga2)
      top21h = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((1.d0-4.d0*ssw2/3.d0)**2*dcos(ang9)**2+(4.d0*ssw2/3.d0)**2*
     & dsin(ang9)**2)*siga1 + ((1.d0-4.d0*ssw2/3.d0)**2*dsin(ang9)**2+
     & (4.d0*ssw2/3.d0)**2*dcos(ang9)**2)*siga2 )
c
c      write(*,*) 'Z-SE: top21i'
      call genzquad (7,s,mtsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mtsr,a,a,a,a,a, siga2,dsiga2)
      top21i = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((1.d0-4.d0*ssw2/3.d0)**2*dcos(ang11)**2+(4.d0*ssw2/3.d0)**2*
     & dsin(ang11)**2)*siga1 + ((1.d0-4.d0*ssw2/3.d0)**2*dsin(ang11)**2+
     & (4.d0*ssw2/3.d0)**2*dcos(ang11)**2)*siga2 )
c
c      write(*,*) 'Z-SE: top21j'
      call genzquad (7,s,mdnsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mdnsr,a,a,a,a,a, siga2,dsiga2)
      top21j = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((-1.d0+2.d0*ssw2/3.d0)**2*dcos(ang8)**2+(2.d0*ssw2/3.d0)**2*
     & dsin(ang8)**2)*siga1 + ((-1.d0+2.d0*ssw2/3.d0)**2*dsin(ang8)**2+
     & (2.d0*ssw2/3.d0)**2*dcos(ang8)**2)*siga2 )
c
c      write(*,*) 'Z-SE: top21k'
      call genzquad (7,s,mstsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mstsr,a,a,a,a,a, siga2,dsiga2)
      top21k = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((-1.d0+2.d0*ssw2/3.d0)**2*dcos(ang10)**2+(2.d0*ssw2/3.d0)**2*
     & dsin(ang10)**2)*siga1 
     $     + ((-1.d0+2.d0*ssw2/3.d0)**2*dsin(ang10)**2+
     & (2.d0*ssw2/3.d0)**2*dcos(ang10)**2)*siga2 )
c
c      write(*,*) 'Z-SE: top21l'
      if (delmbresum.eq.1) then
      call genzquad (7,s,mbsl,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,mbsr,a,a,a,a,a, siga2,dsiga2)
      top21l = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((-1.d0+2.d0*ssw2/3.d0)**2*dcos(ang12)**2+(2.d0*ssw2/3.d0)**2*
     & dsin(ang12)**2)*siga1 
     $     + ((-1.d0+2.d0*ssw2/3.d0)**2*dsin(ang12)**2+
     & (2.d0*ssw2/3.d0)**2*dcos(ang12)**2)*siga2 )
      else
      call genzquad (7,s,msb1dmb,a,a,a,a,a, siga1,dsiga1)
      call genzquad (7,s,msb2dmb,a,a,a,a,a, siga2,dsiga2)
      top21l = -elec2/(4.d0*ppi)**2/(2.d0*ssw2*ccw2) * 3.d0 * (
     & ((-1.d0+2.d0*ssw2/3.d0)**2*dcos(tsbdmb)**2+(2.d0*ssw2/3.d0)**2*
     & dsin(tsbdmb)**2)*siga1 
     $     + ((-1.d0+2.d0*ssw2/3.d0)**2*dsin(tsbdmb)**2+
     & (2.d0*ssw2/3.d0)**2*dcos(tsbdmb)**2)*siga2 )
      endif
c
      if (selec.ge.3) then
      top21 = top21a + top21b + top21c + top21d + top21e +
     &        top21f + top21g + top21h + top21i + top21j +
     &        top21k + top21l
      elseif (selec.eq.2) then
         top21 = top21i + top21l
      elseif (selec.eq.1) then
         top21 = top21i
      else
         write(*,*) "Error in Sigma-Z: selec out or range"
      endif
c o.k :
      call genzquad (8,s,mmw,a,a,a,a,a, siga,dsiga)
      top23 = elec2/(4.d0*ppi)**2 * ccw2/ssw2 * siga
c
c chargino, neutralino loops
c
c      write(*,*) 'Z-SE: vor top13'
      top13 = 0.d0
      do 540 ii = 1,4
       do 541 j = 1,4
      a = (-nmix(ii,3)*nmix(j,3) + nmix(ii,4)*nmix(j,4)) / 2.d0
      b = -a
      call genzquad (4,s,mne(ii),mne(j),a,a,b,b, siga,dsiga)
      top13h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2*ccw2)* siga / 2.d0
      top13 = top13 + top13h
541   continue
540   continue
c
c      write(*,*) 'Z-SE: vor top14'
      top14 = 0.d0
      do 542 ii = 1,2
       do 543 j = 1,2
      a = -vmix(ii,1)*vmix(j,1) - vmix(ii,2)*vmix(j,2)/2.d0
      b = -umix(ii,1)*umix(j,1) - umix(ii,2)*umix(j,2)/2.d0
      if (ii.eq.j) then
      a = a + ssw2
      b = b + ssw2
      endif
      call genzquad (4,s,mcha(ii),mcha(j),a,a,b,b, siga,dsiga)
      top14a = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2*ccw2) * siga
      top14 = top14 + top14a
543   continue
542   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (6,*) ' Z - selfenergy : ', real(dsqrt(s))
       write (6,*) ' A0 H0       = ', top1
       write (6,*) ' A0 h0       = ', top2
       write (6,*) ' H+ H-       = ', top3
       write (6,*) ' G0 H0       = ', top4
       write (6,*) ' G0 h0       = ', top5
       write (6,*) ' G+ G-       = ', top6
       write (6,*) ' Z0 H0       = ', top7
       write (6,*) ' Z0 h0       = ', top8
       write (6,*) ' W+ G-       = ', top9
       write (6,*) ' fer fer     = ', top10
       write (6,*) ' sfer sfer   = ', top11
       write (6,*) ' neu  neu    = ', top13
       write (6,*) ' cha  cha    = ', top14
       write (6,*) ' W+ W-       = ', top15
       write (6,*) ' gh+ gh-     = ', top16
       write (6,*) '   four point interactions : '
       write (6,*) ' H0          = ', top17a
       write (6,*) ' h0          = ', top17b
       write (6,*) ' A           = ', top17c
       write (6,*) ' H+          = ', top18
       write (6,*) ' G0          = ', top19
       write (6,*) ' G+          = ', top20
       write (6,*) ' sfer        = ', top21
       write (6,*) ' W+          = ', top23
       write (6,*) '               '
      endif
c
      sigb = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &       top9 + top15 + top16 + top17 + top18 + top19 + top20 +
     &       top23
      sigf = top10 
      sigs = top11 + top21 
      sigc = top13 + top14
      sigt = top10f + top11i + top21i
c
      return
      end


c=====================================================================
      subroutine genaquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
      implicit double precision (a-z)
      complex*16 aa
      real*8 mcha(1:2),mne(1:4)
      integer typ

c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c           _______/          \_______                  .     .
c        q         \          /        q                  .  .
c                   \________/                q ____________.__________q
c                       m2
c

c      if (dabs(dabs(m1) + dabs(m2) - dsqrt(q2)).le.1d0)
c     $     write(*,*) 'dSigmaA:', real(m1), real(m2),
c     $     real(dsqrt(q2)), real(dabs(m1) + dabs(m2) - dsqrt(q2))
c      write(3,*) 'SigmaA:', real(dsqrt(q2)), real(m1), real(m2)
      q20 = 0.1d0
      q21 = (dsqrt(q20)+1d0)**2
      q2d = ((dsqrt(q2)+1d0)**2)
      call bquer2(q2d,dabs(m2),dabs(m1),
     &             b0d,b1d,pb0d,pb1d)
      call bquer2(q2,dabs(m2),dabs(m1),
     &             b0,b1,pb0,pb1)
      call bquer2(q20,dabs(m2),dabs(m1),
     &             b00,b10,pb00,pb10)
      call bquer2(q21,dabs(m2),dabs(m1),
     &             b01,b11,pb01,pb11)

      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b0d = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0d
      b00 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b00
      b01 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b01
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
      b1d = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1d
      b10 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b10
      b11 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b11
c
      if (typ.eq.1) then
       siga  = -(q2 * (b0 - 2.d0 * b1) + m2**2 * b0 + aa(dabs(m1)))
       dsiga = - ( (b0 - 2.d0 * b1) + q2 *
     &    (pb0 - 2.d0 * pb1) + m2**2 * pb0 )
       dsigad = - ( (b0d - 2.d0 * b1d) + q2d *
     &    (pb0d - 2.d0 * pb1d) + m2**2 * pb0d )
       dsiga0 = - ( (b00 - 2.d0 * b10) + q20 *
     &    (pb00 - 2.d0 * pb10) + m2**2 * pb00 )
       dsiga1 = - ( (b01 - 2.d0 * b11) + q21 *
     &    (pb01 - 2.d0 * pb11) + m2**2 * pb01 )
c      siga = -(q2*4.d0/epsilon + m2**2*2.d0/epsilon+2.d0*m1**2/epsilon)
c      dsiga = - ( 4.d0/epsilon )
      else
      if (typ.eq.2) then
       siga  = -b0
       dsiga = -pb0
       dsigad = -pb0d
       dsiga0 = -pb00
       dsiga1 = -pb01
c      siga  = -2.d0/epsilon
c      dsiga = 0.d0
      else
      if (typ.eq.3) then
c         if ((m1.eq.175d0).or.(m1.eq.4.5d0)) then
c            write(*,*) "genaquad: ", m1, aa(dabs(m1)), q2 * b1
c         endif
       siga  = -4.d0 * ( aa(dabs(m1)) + q2 * b1 )
       dsiga = -4.d0 * ( b1 + q2 * pb1 )
       dsigad = -4.d0 * ( b1d + q2d * pb1d )
       dsiga0 = -4.d0 * ( b10 + q20 * pb10 )
       dsiga1 = -4.d0 * ( b11 + q21 * pb11 )
c      siga  = -4.d0 * ( 2.d0*m1**2/epsilon - q2/epsilon )
c      dsiga =  4.d0/epsilon
      else
      if (typ.eq.4) then
       siga  = 8.d0 * ((at * a + bt * b) * m1 * m2 * b0
     &     + (at * b + a * bt) * ( q2 *
     &     b1 + aa(dabs(m1)) + m2**2 * b0 ))
       dsiga = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb0
     &     + (at * b + a * bt) * ( b1 + q2 * pb1
     &        + m2**2 * pb0))
       dsigad = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb0d
     &     + (at * b + a * bt) * ( b1d + q2d * pb1d
     &        + m2**2 * pb0d))
       dsiga0 = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb00
     &     + (at * b + a * bt) * ( b10 + q20 * pb10
     &        + m2**2 * pb00))
       dsiga1 = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb01
     &     + (at * b + a * bt) * ( b11 + q21 * pb11
     &        + m2**2 * pb01))
c      siga  =  8.d0 * ((at * b + bt * a) * ( 2.d0 * m1**2 + 2.d0 *
c    &    m2**2 - q2 ) + (a * at + b * bt) * 2.d0 * m1 * m2  ) / epsilon
c      dsiga = -8.d0 * ((at * b + a * bt))/epsilon
      else
      if (typ.eq.5) then
       siga  = - (4.d0 * aa(dabs(m1)) - 2.d0 * m1**2)
       dsiga =  0.d0
       dsigad =  0.d0
       dsiga0 =  0.d0
       dsiga1 =  0.d0
c      siga  = -8.d0 * m1**2/epsilon
      else
      if (typ.eq.6) then
       siga  = aa(dabs(m1))
       dsiga = 0.d0
       dsigad = 0.d0
       dsiga0 = 0.d0
       dsiga1 = 0.d0
c      siga  = 2.d0 * m1**2 / epsilon
      else
       write (6,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
      endif
      endif

c      if (dabs(dsigad-dsiga).gt.0.015d0) then
c      if ((dabs(dsigad-dsiga).gt.0.015d0).and.
      if ((dabs(dsigad-dsiga).gt.2d0*dabs(dsiga1-dsiga0)).and.
     $    ((dabs(m1)+dabs(m2))/1.8d0.lt.dsqrt(q2)).and.
     $    ((dabs(m1)+dabs(m2)+1d0).gt.dsqrt(q2)).and.
     $     (m1.ne.maa).and.(m1.ne.mhp).and.(m1.ne.mhh).and.
     $     (m2.ne.maa).and.(m2.ne.mhp).and.(m2.ne.mhh)) then
c         if (typ.eq.3) then
c         write(*,*) '------------------------------'
c         write(*,*) real(dabs(dsigad-dsiga)), real(dsiga1-dsiga0),
c     $              real(m1), real(m2)
c         write(*,*) real(dsigad),real(dsiga),real(dsiga1),real(dsiga0)
c         write(*,*) typ, real(b1), real(pb1), real(dsqrt(q2))
c         write(*,*) real(b1d), real(pb1d), real(q2d)
c         write(*,*) '------------------------------'
c         endif
         dsiga = dsiga0
      endif
c
      return
      end
c
c$$$c=====================================================================
c$$$      subroutine genaquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c$$$c
c$$$      implicit double precision (a-z)
c$$$      complex*16 aa
c$$$      integer typ
c$$$c
c$$$      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
c$$$     &               beta,alpha
c$$$      common /singl/ epsilon,muee,lambda
c$$$c     common /renpara/xo,zo,mgll
c$$$c
c$$$c                    ________                              .
c$$$c                   /   m1   \                          .     .  m1
c$$$c           _______/          \_______                  .     .
c$$$c        q         \          /        q                  .  .
c$$$c                   \________/                q ____________.__________q
c$$$c                       m2
c$$$c
c$$$
c$$$c      if (dabs(dabs(m1) + dabs(m2) - dsqrt(q2)).le.1d0)
c$$$c     $     write(*,*) 'dSigmaA:', real(m1), real(m2),
c$$$c     $     real(dsqrt(q2)), real(dabs(m1) + dabs(m2) - dsqrt(q2))
c$$$c      write(3,*) 'SigmaA:', real(dsqrt(q2)), real(m1), real(m2)
c$$$      call bquer2(q2,dabs(m2),dabs(m1),
c$$$     &             b0,b1,pb0,pb1)
c$$$      call bquer2(0.1d0,dabs(m2),dabs(m1),
c$$$     &             b00,b10,pb00,pb10)
c$$$      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
c$$$      b00 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b00
c$$$      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
c$$$     &      + 0.5d0) + b1
c$$$      b10 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
c$$$     &      + 0.5d0) + b10
c$$$c
c$$$      if (typ.eq.1) then
c$$$       siga  = -(q2 * (b0 - 2.d0 * b1) + m2**2 * b0 + aa(dabs(m1)))
c$$$       dsigaq = - ( (b0 - 2.d0 * b1) + q2 *
c$$$     &    (pb0 - 2.d0 * pb1) + m2**2 * pb0 )
c$$$       dsiga0 = - ( (b00 - 2.d0 * b10) + 0.1d0 *
c$$$     &    (pb00 - 2.d0 * pb10) + m2**2 * pb00 )
c$$$c      siga = -(q2*4.d0/epsilon + m2**2*2.d0/epsilon+2.d0*m1**2/epsilon)
c$$$c      dsiga = - ( 4.d0/epsilon )
c$$$      else
c$$$      if (typ.eq.2) then
c$$$       siga  = -b0
c$$$       dsigaq = -pb0
c$$$       dsiga0 = -pb00
c$$$c      siga  = -2.d0/epsilon
c$$$c      dsiga = 0.d0
c$$$      else
c$$$      if (typ.eq.3) then
c$$$c         if ((m1.eq.175d0).or.(m1.eq.4.5d0)) then
c$$$c            write(*,*) "genaquad: ", m1, aa(dabs(m1)), q2 * b1
c$$$c         endif
c$$$       siga  = -4.d0 * ( aa(dabs(m1)) + q2 * b1 )
c$$$       dsigaq = -4.d0 * ( b1 + q2 * pb1 )
c$$$       dsiga0 = -4.d0 * ( b10 + 0.1d0 * pb10 )
c$$$c      siga  = -4.d0 * ( 2.d0*m1**2/epsilon - q2/epsilon )
c$$$c      dsiga =  4.d0/epsilon
c$$$      else
c$$$      if (typ.eq.4) then
c$$$       siga  = 8.d0 * ((at * a + bt * b) * m1 * m2 * b0
c$$$     &     + (at * b + a * bt) * ( q2 *
c$$$     &     b1 + aa(dabs(m1)) + m2**2 * b0 ))
c$$$       dsigaq = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb0
c$$$     &     + (at * b + a * bt) * ( b1 + q2 * pb1
c$$$     &        + m2**2 * pb0))
c$$$       dsiga0 = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb00
c$$$     &     + (at * b + a * bt) * ( b10 + 0.1d0 * pb10
c$$$     &        + m2**2 * pb00))
c$$$c      siga  =  8.d0 * ((at * b + bt * a) * ( 2.d0 * m1**2 + 2.d0 *
c$$$c    &    m2**2 - q2 ) + (a * at + b * bt) * 2.d0 * m1 * m2  ) / epsilon
c$$$c      dsiga = -8.d0 * ((at * b + a * bt))/epsilon
c$$$      else
c$$$      if (typ.eq.5) then
c$$$       siga  = - (4.d0 * aa(dabs(m1)) - 2.d0 * m1**2)
c$$$       dsigaq =  0.d0
c$$$       dsiga0 =  0.d0
c$$$c      siga  = -8.d0 * m1**2/epsilon
c$$$      else
c$$$      if (typ.eq.6) then
c$$$       siga  = aa(dabs(m1))
c$$$       dsigaq = 0.d0
c$$$       dsiga0 = 0.d0
c$$$c      siga  = 2.d0 * m1**2 / epsilon
c$$$      else
c$$$       write (6,*) ' typpindent wrong '
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$c
c$$$      if (dsigaq.eq.0d0) then
c$$$         dsiga = 0d0
c$$$      elseif (dabs((dsigaq-dsiga0)/dsiga0).gt.0.15d0) then
c$$$         dsiga = dsiga0
c$$$      else
c$$$         dsiga = dsigaq
c$$$      endif
c$$$
c$$$      return
c$$$      end
c$$$c
c$$$c=====================================================================
c$$$      subroutine genaquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c$$$c
c$$$      implicit double precision (a-z)
c$$$      complex*16 aa
c$$$      integer typ
c$$$c
c$$$      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
c$$$     &               beta,alpha
c$$$      common /singl/ epsilon,muee,lambda
c$$$c     common /renpara/xo,zo,mgll
c$$$c
c$$$c                    ________                              .
c$$$c                   /   m1   \                          .     .  m1
c$$$c           _______/          \_______                  .     .
c$$$c        q         \          /        q                  .  .
c$$$c                   \________/                q ____________.__________q
c$$$c                       m2
c$$$c
c$$$
c$$$c      if (dabs(dabs(m1) + dabs(m2) - dsqrt(q2)).le.1d0)
c$$$c     $     write(*,*) 'dSigmaA:', real(m1), real(m2),
c$$$c     $     real(dsqrt(q2)), real(dabs(m1) + dabs(m2) - dsqrt(q2))
c$$$c      write(3,*) 'SigmaA:', real(dsqrt(q2)), real(m1), real(m2)
c$$$      call bquer2(q2,dabs(m2),dabs(m1),
c$$$     &             b0,b1,pb0,pb1)
c$$$      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
c$$$      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
c$$$     &      + 0.5d0) + b1
c$$$c
c$$$      if (typ.eq.1) then
c$$$       siga  = -(q2 * (b0 - 2.d0 * b1) + m2**2 * b0 + aa(dabs(m1)))
c$$$       dsiga = - ( (b0 - 2.d0 * b1) + q2 *
c$$$     &    (pb0 - 2.d0 * pb1) + m2**2 * pb0 )
c$$$c      siga = -(q2*4.d0/epsilon + m2**2*2.d0/epsilon+2.d0*m1**2/epsilon)
c$$$c      dsiga = - ( 4.d0/epsilon )
c$$$      else
c$$$      if (typ.eq.2) then
c$$$       siga  = -b0
c$$$       dsiga = -pb0
c$$$c      siga  = -2.d0/epsilon
c$$$c      dsiga = 0.d0
c$$$      else
c$$$      if (typ.eq.3) then
c$$$c         if ((m1.eq.175d0).or.(m1.eq.4.5d0)) then
c$$$c            write(*,*) "genaquad: ", m1, aa(dabs(m1)), q2 * b1
c$$$c         endif
c$$$       siga  = -4.d0 * ( aa(dabs(m1)) + q2 * b1 )
c$$$       dsiga = -4.d0 * ( b1 + q2 * pb1 )
c$$$c      siga  = -4.d0 * ( 2.d0*m1**2/epsilon - q2/epsilon )
c$$$c      dsiga =  4.d0/epsilon
c$$$      else
c$$$      if (typ.eq.4) then
c$$$       siga  = 8.d0 * ((at * a + bt * b) * m1 * m2 * b0
c$$$     &     + (at * b + a * bt) * ( q2 *
c$$$     &     b1 + aa(dabs(m1)) + m2**2 * b0 ))
c$$$       dsiga = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb0
c$$$     &     + (at * b + a * bt) * ( b1 + q2 * pb1
c$$$     &        + m2**2 * pb0))
c$$$c      siga  =  8.d0 * ((at * b + bt * a) * ( 2.d0 * m1**2 + 2.d0 *
c$$$c    &    m2**2 - q2 ) + (a * at + b * bt) * 2.d0 * m1 * m2  ) / epsilon
c$$$c      dsiga = -8.d0 * ((at * b + a * bt))/epsilon
c$$$      else
c$$$      if (typ.eq.5) then
c$$$       siga  = - (4.d0 * aa(dabs(m1)) - 2.d0 * m1**2)
c$$$       dsiga =  0.d0
c$$$c      siga  = -8.d0 * m1**2/epsilon
c$$$      else
c$$$      if (typ.eq.6) then
c$$$       siga  = aa(dabs(m1))
c$$$       dsiga = 0.d0
c$$$c      siga  = 2.d0 * m1**2 / epsilon
c$$$      else
c$$$       write (6,*) ' typpindent wrong '
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$      endif
c$$$c
c$$$      return
c$$$      end
c$$$c
c ----------------------------------------------------------
c
c      double precision function aa (m)
c
c      implicit double precision (a-z)
c
c      common /singl/ epsilon,muee,lambda
c
c      if (m.le.1.d-8) then
c       aa = 0.d0
c      else
c       aa = m**2 * ( delta(epsilon,muee,1.d0) - dlog(m**2) + 1.d0 )
c      endif
c
c      return
c      end
c
c -------------------------------------------------------------
c
c     double precision function dcot(x)
c
c     implicit double precision (a-z)
c
c     dcot = 1.d0/dtan(x)
c
c     return
c     end
c
c ---------------------------------------------------------
c
      subroutine sigmaa (s ,sigmaab,sigmaas,sigmaaf,sigmaac,
     &                      sigmaat)
c
c     selfenergy of pseudoscalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)


c      write(*,*) "sigmaa: i = ", i
c      write(*,*) "mhp:", real(mhp)
c
c boson loops
c
c  notation :
c     genwquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genaquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2 / (2.d0*ssw2) * siga
c
      call genaquad (1,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2/(4.d0*ccw2*ssw2)
     &       * siga
c
      call genaquad (1,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2/(4.d0*ppi)**2 * dcos(beta-alpha)**2/(4.d0*ccw2*ssw2)
     &       * siga
c
      call genaquad (2,s,maa,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = -elec2/(4.d0*ppi)**2 * mmz**2*dcos(2.d0*beta)**2
     &       * dcos(beta+alpha)**2/(4.d0*ccw2*ssw2) * siga
c
      call genaquad (2,s,maa,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2 * mmz**2*dcos(2.d0*beta)**2
     &      * dsin(beta+alpha)**2/(4.d0*ccw2*ssw2) * siga
c      write(*,*) 'dSigmaA:', real(maa), real(mlh), real(mhh)
c
      call genaquad (2,s,mhh,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2 * mmz**2*dsin(2.d0*beta)**2
     &      * dcos(beta+alpha)**2/(4.d0*ccw2*ssw2) * siga
c
      call genaquad (2,s,mlh,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2 * mmz**2*dsin(2.d0*beta)**2
     &      * dsin(beta+alpha)**2/(4.d0*ccw2*ssw2) * siga
c
      call genaquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * mmw**2/(2.d0*ssw2) * siga
c
      call genaquad (2,s,mupsr,mupsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12a = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * siga * mup**2 * 3.d0
c      write(*,*) '-O-', real(mupsr), real(mupsl), real(siga),
c     $           real(top12a)
c
      call genaquad (2,s,mchsr,mchsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12b = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * siga * mch**2 * 3.d0
c
      call genaquad (2,s,mtsr,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12c = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * siga * mtt**2 * 3.d0
c      write(*,*) real(siga), real(mtt),
c     $           real(top12c), real(mu), real(mssupq), real(beta)
c
      call genaquad (2,s,mdnsr,mdnsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12d = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * siga * mdn**2 * 3.d0
c
      call genaquad (2,s,mstsr,mstsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12e = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * siga * mst**2 * 3.d0
c
      if (delmbresum.eq.1) then
      call genaquad (2,s,mbsr,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * siga * mbb**2 * 3.d0
      else
      call genaquad (2,s,msb2dmb,msb1dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga,dsiga)
      top12f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * siga * 
     $     mbbdmb**2 * 3.d0
      endif
c
      call genaquad (2,s,melsr,melsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12g = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * siga * mel**2
c
      call genaquad (2,s,mmusr,mmusl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12h = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * siga * mmu**2
c
      call genaquad (2,s,mtasr,mtasl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12i = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * siga * mta**2
c
      if (selec.ge.3) then
      top12 = top12a + top12b + top12c + top12d + top12e +
     &        top12f + top12g + top12h + top12i
      elseif (selec.eq.2) then
         top12 = top12c + top12f
      elseif (selec.eq.1) then
         top12 = top12c
      else
         write(*,*) "Error in Sigma-A: selec out or range"
      endif



c      write(*,*) 'vor top13'
c
      call genaquad (5,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2) * siga
c
      call genaquad (5,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ccw2 * ssw2)
     &        * siga
c
      call genaquad (6,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = elec2 / (4.d0 * ppi)**2 * dcos(2.d0 * beta)**2 /
     &        (4.d0 * ccw2 * ssw2) * siga
c
      call genaquad (6,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top16 = elec2 / (4.d0 * ppi)**2 * (1.d0 + dsin(2.d0 *
     &   beta)**2 - ssw2/ccw2 * dcos(2.d0 * beta)**2) / (4.d0 *
     &   ssw2) * siga
c
      call genaquad (6,s,maa,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17 = elec2 / (4.d0 * ppi)**2 * 3.d0 * dcos(2.d0 *
     &       beta)**2 /(8.d0 * ssw2 * ccw2) * siga
c
      call genaquad (6,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = elec2 / (4.d0 * ppi)**2 * (3.d0 * dsin(2.d0 *
     &       beta)**2 - 1.d0) /(8.d0 * ssw2 * ccw2) * siga
c
      call genaquad (6,s,mlh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top19 = elec2 / (4.d0 * ppi)**2 * dcos(2.d0 * beta) *
     &        dcos(2.d0 * alpha) /(8.d0 * ssw2 * ccw2) * siga
c
      call genaquad (6,s,mhh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2 / (4.d0 * ppi)**2 * dcos(2.d0 * beta) *
     &        dcos(2.d0 * alpha) /(8.d0 * ssw2 * ccw2) * siga
c
c      write(*,*) 'vor top21'
      call genaquad (6,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21a = -elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*3.d0 * ( siga1 * (
     &       ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &       beta) - mup**2 / mmw2 * dcot(beta)**2 ) * dcos(ang7)**2 +
     &       ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mup**2 / mmw2 * dcot(beta)**2 ) * dsin(ang7)**2)
     &   +   siga2 * ( ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &       beta) - mup**2 / mmw2 * dcot(beta)**2 ) * dsin(ang7)**2 +
     &        ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mup**2 / mmw2 * dcot(beta)**2 ) * dcos(ang7)**2))
c      write(*,*) '_O_', real(mupsl), real(mupsr), 
c     $           real(siga1), real(siga2), real(top21a)
c
      call genaquad (6,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21b = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*3.d0 * ( siga1 * (
     &         ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &         beta) - mch**2 / mmw2 * dcot(beta)**2 ) * dcos(ang9)**2+
     &         ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         beta) - mch**2 / mmw2 * dcot(beta)**2 ) * dsin(ang9)**2)
     &   +    siga2 * ( ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0*
     &        beta) - mch**2 / mmw2 * dcot(beta)**2 ) * dsin(ang9)**2+
     &        ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mch**2 / mmw2 * dcot(beta)**2 ) * dcos(ang9)**2))
c
      call genaquad (6,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21c = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (siga1*
     &       ( ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &        beta) - mtt**2 / mmw2 * dcot(beta)**2 ) * dcos(ang11)**2+
     &       ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mtt**2 / mmw2 * dcot(beta)**2 ) * dsin(ang11)**2)
     &  + siga2 * ( ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &        beta) - mtt**2 / mmw2 * dcot(beta)**2 ) * dsin(ang11)**2+
     &        ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mtt**2 / mmw2 * dcot(beta)**2 ) * dcos(ang11)**2))
c
      call genaquad (6,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21d = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (siga1*
     &       ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &        beta) - mdn**2 / mmw2 * dtan(beta)**2 ) * dcos(ang8)**2+
     &       ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mdn**2 / mmw2 * dtan(beta)**2 ) * dsin(ang8)**2)+
     &  siga2 * ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &        beta) - mdn**2 / mmw2 * dtan(beta)**2 ) * dsin(ang8)**2
     &      + ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mdn**2 / mmw2 * dtan(beta)**2 ) * dcos(ang8)**2))
c
      call genaquad (6,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21e = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (siga1*
     &  ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &    beta) - mst**2 / mmw2 * dtan(beta)**2 ) * dcos(ang10)**2 +
     &    ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &     beta) - mst**2 / mmw2 * dtan(beta)**2 ) * dsin(ang10)**2 )+
     &  siga2 * ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &     beta) - mst**2 / mmw2 * dtan(beta)**2 ) * dsin(ang10)**2+
     &     ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &     beta) - mst**2 / mmw2 * dtan(beta)**2 ) * dcos(ang10)**2))
c
      if (delmbresum.eq.1) then
      call genaquad (6,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (siga1*
     &   ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &     beta) - mbb**2 / mmw2 * dtan(beta)**2 ) * dcos(ang12)**2 +
     &     ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &      beta) - mbb**2 / mmw2 * dtan(beta)**2 ) * dsin(ang12)**2)+
     &  siga2 * ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &      beta) - mbb**2 / mmw2 * dtan(beta)**2 ) * dsin(ang12)**2+
     &      ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mbb**2 / mmw2 * dtan(beta)**2 ) * dcos(ang12)**2))
      else
      call genaquad (6,s,msb1dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genaquad (6,s,msb2dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      top21f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 *(siga1*
     &   ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &     beta) - mbbdmb**2 / mmw2 * dtan(beta)**2 ) 
     $      * dcos(tsbdmb)**2 +
     &     ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &      beta) - mbbdmb**2 / mmw2 * dtan(beta)**2 ) 
     $     *dsin(tsbdmb)**2)+
     &  siga2 * ( ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &      beta) - mbbdmb**2 / mmw2 * dtan(beta)**2 ) 
     $     * dsin(tsbdmb)**2+
     &      ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &        beta) - mbbdmb**2 / mmw2 * dtan(beta)**2 ) 
     $        * dcos(tsbdmb)**2))
      endif
c
      call genaquad (6,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21g = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * ( siga1 *
     &    ( ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &      beta) - mel**2 / mmw2 * dtan(beta)**2 ) * dcos(ang2)**2+
     &      ( -ssw2/ccw2 * dcos(2.d0 *
     &      beta) - mel**2 / mmw2 * dtan(beta)**2 ) * dsin(ang2)**2)+
     &  siga2 * ( ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &       beta) - mel**2 / mmw2 * dtan(beta)**2 ) * dsin(ang2)**2+
     &       ( -ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mel**2 / mmw2 * dtan(beta)**2 ) * dcos(ang2)**2))
c
      call genaquad (6,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21h = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * ( siga1 * (
     &     ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &        beta) - mmu**2 / mmw2 * dtan(beta)**2 ) * dcos(ang4)**2+
     &      ( -ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mmu**2 / mmw2 * dtan(beta)**2 ) * dsin(ang4)**2)+
     &  siga2 * ( ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &       beta) - mmu**2 / mmw2 * dtan(beta)**2 ) * dsin(ang4)**2+
     &       ( -ssw2/ccw2 * dcos(2.d0 *
     &       beta) - mmu**2 / mmw2 * dtan(beta)**2 ) * dcos(ang4)**2))
c
      call genaquad (6,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genaquad (6,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top21i = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * ( siga1 *
     &   ( ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &    beta) - mta**2 / mmw2 * dtan(beta)**2 ) * dcos(ang6)**2 +
     &   ( -ssw2/ccw2 * dcos(2.d0 *
     &     beta) - mta**2 / mmw2 * dtan(beta)**2 ) * dsin(ang6)**2 )+
     &  siga2 * ( ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &     beta) - mta**2 / mmw2 * dtan(beta)**2 ) * dsin(ang6)**2 +
     &    ( -ssw2/ccw2 * dcos(2.d0 *
     &     beta) - mta**2 / mmw2 * dtan(beta)**2 ) * dcos(ang6)**2 ))
c
c      write(*,*) 'mvesl:', mvesl
      call genaquad (6,s,mvesl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top21j = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*beta)
c
      call genaquad (6,s,mvmsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top21k = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*beta)
c
      call genaquad (6,s,mvtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top21l = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*beta)
c
      if (selec.ge.3) then
      top21 = top21a + top21b + top21c + top21d + top21e + top21l +
     &        top21f + top21g + top21h + top21i + top21j + top21k
c      write(*,*) real(top21a), real(top21b), real(top21c), real(top21d),
c     $           real(top21e), real(top21f), real(top21g), real(top21h),
c     $           real(top21i), real(top21j), real(top21k), real(top21l)
      elseif (selec.eq.2) then
         top21 = top21c + top21f
      elseif (selec.eq.1) then
         top21 = top21c
      else
         write(*,*) "Error in Sigma-A: selec out or range"
      endif

c      write(*,*) 'vor top9'
c
c fermion loops
c
      call genaquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9a = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mup**2 * 3.d0
c
      call genaquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9b = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mch**2 * 3.d0
c
c--------------- top loop -----------------------------------------
csh 20.01.00
csh old version with no threshold correction
c      call genaquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)

csh new version with simple threshold correction
      if ((((mtt + mtt - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mtt + mtt - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
c     $    (((mtt + mtt - dsqrt(s))/dsqrt(s)).le.0.006d0)) then
         call genaquad(3, ((mtt + mtt) * 0.975d0)**2,
c         call genaquad(3, ((mtt + mtt) * 0.994d0)**2,
     $                 mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      else 
         call genaquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      endif

      top9c = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mtt**2 * 3.d0
c--------------- top loop -----------------------------------------
c      write(*,*) "A-SE: top:", s
c      write(*,*) dcot(beta), siga, mtt, top9c
c      write(*,*) mtt**2 * dcot(beta)**2
c
      call genaquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9d = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mdn**2 * 3.d0
c
      call genaquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9e = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mst**2 * 3.d0
c
      if (delmbresum.eq.1) then
      call genaquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9f = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mbb**2 * 3.d0
      else
      call genaquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9f = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mbbdmb**2 * 3.d0
      endif
c      write(*,*) "A-SE: bottom:", s
c      write(*,*) dtan(beta), siga, mbb, top9f
c      write(*,*) mbb**2 * dtan(beta)**2
c
      call genaquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9g = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mel**2
c
      call genaquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9h = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mmu**2
c
      call genaquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9i = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * siga * mta**2
c
      if (selec.ge.3) then
      top9 = top9a + top9b + top9c + top9d + top9e + top9f +
     &       top9g + top9h + top9i
c      write(*,*) 'top9:', real(top9)
c      write(*,*) real(top9a), real(top9b), real(top9c), real(top9d),
c     $           real(top9e), real(top9f), real(top9g), real(top9h),
c     $           real(top9i)
      elseif (selec.eq.2) then
         top9 = top9c + top9f
      elseif (selec.eq.1) then
         top9 = top9c
      else
         write(*,*) "Error in Sigma-A: selec out or range"
      endif
c      write(*,*) real(top9a), real(top9b), real(top9c), real(top9d),
c     $           real(top9e), real(top9f), real(top9g), real(top9h),
c     $           real(top9i)


c      write(*,*) 'vor top10'
c o.k :
      top10 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dsin(beta) +
     &      vmix(j,2)*umix(ii,1)*dcos(beta) )/dsqrt(2.d0)
      b = -(vmix(ii,1)*umix(j,2)*dsin(beta) +
     &      vmix(ii,2)*umix(j,1)*dcos(beta) )/dsqrt(2.d0)
      call genaquad (4,s,mcha(ii),mcha(j),a,-b,b,-a, siga,dsiga)
      top10h = elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c      write(*,*) ii, j, real(top10h)
c      write(*,*) ii, j, real(vmix(ii,j)), real(umix(ii,j)), 
c     $           real(mcha(ii)) 
      top10 = top10 + top10h
 621   continue
620   continue
c o.k :
c      write(*,*) 'vor top11'
      top11 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dsin(beta) - sm * dcos(beta)
      b = -a
      call genaquad (4,s,mne(j),mne(ii),a,a,b,b, siga,dsiga)
      top11h = elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * siga
c      write(*,*) ii, j, real(top11h)
c      write(*,*) ii, j, real(nmix(ii,j)), real(mne(ii))
      top11 = top11 + top11h
 623   continue
622   continue
c
      sigmaab = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &          top13 + top14 + top15 + top16 + top17 + top18 + top19 +
     &          top20
      sigmaaf = top9
      sigmaas = top12  + top21 
      sigmaac = top10 + top11
c      sigmaac = top11
      sigmaat = top12c + top21c + top9c
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (6,*) ' A - selfenergy : ', real(dsqrt(s))
       write (6,*) ' W+ H-       = ', top1
       write (6,*) ' Z0 H0       = ', top2
       write (6,*) ' Z0 h0       = ', top3
       write (6,*) ' A0 H0       = ', top4
       write (6,*) ' A0 h0       = ', top5
       write (6,*) ' G0 H0       = ', top6
       write (6,*) ' G0 h0       = ', top7
       write (6,*) ' G+ H-       = ', top8
       write (6,*) ' fer fer     = ', top9
       write (6,*) ' sfer sfer   = ', top12
       write (6,*) ' cha         = ', top10
       write (6,*) ' neu         = ', top11
       write (6,*) '   four point interactions : '
       write (6,*) ' W+          = ', top13
       write (6,*) ' Z0          = ', top14
       write (6,*) ' H+          = ', top15
       write (6,*) ' G+          = ', top16
       write (6,*) ' A           = ', top17
       write (6,*) ' G0          = ', top18
       write (6,*) ' h0          = ', top19
       write (6,*) ' H0          = ', top20
       write (6,*) ' sfer        = ', top21
       write (6,*) '   '
      endif
c
      return
      end
c
c -------------------------------------------------------------
c
c
      subroutine dsigmaa (s ,dsigmaab,dsigmaas,dsigmaaf,dsigmaac,
     &                       dsigmaat)
c
c     derivative of pseudoscalar higgsparticle selfenergy
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
      double precision tfak, dfak
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

      tfak = 0.1d0/(dtan(beta)**2)
      dfak = 1d0 - tfak
c
c boson loops
c
c  notation :
c     genwquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genaquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2 / (2.d0*ssw2) * dsiga
c
      call genaquad (1,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2/(4.d0*ccw2*ssw2)
     &       * dsiga
c
      call genaquad (1,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2/(4.d0*ppi)**2 * dcos(beta-alpha)**2/(4.d0*ccw2*ssw2)
     &       * dsiga
c
      call genaquad (2,s,maa,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = -elec2/(4.d0*ppi)**2 * mmz**2*dcos(2.d0*beta)**2
     &       * dcos(beta+alpha)**2/(4.d0*ccw2*ssw2) * dsiga
c
      call genaquad (2,s,maa,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2 * mmz**2*dcos(2.d0*beta)**2
     &      * dsin(beta+alpha)**2/(4.d0*ccw2*ssw2) * dsiga
c
      call genaquad (2,s,mhh,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2 * mmz**2*dsin(2.d0*beta)**2
     &      * dcos(beta+alpha)**2/(4.d0*ccw2*ssw2) * dsiga
c
      call genaquad (2,s,mlh,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2 * mmz**2*dsin(2.d0*beta)**2
     &      * dsin(beta+alpha)**2/(4.d0*ccw2*ssw2) * dsiga
c
      call genaquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * mmw**2/(2.d0*ssw2) * dsiga
c
      call genaquad (2,s,mupsr,mupsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12a = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * dsiga * mup**2 * 3.d0
c
      call genaquad (2,s,mchsr,mchsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12b = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * dsiga * mch**2 * 3.d0
c
c--------------- Stop loop -----------------------------------------
csh old version with no threshold correction
c      call genaquad (2,s,mtsr,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)

csh new version with simple threshold correction
      if ((((mtsl + mtsr - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mtsl + mtsr - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
         call genaquad(2, ((mtsl + mtsr) * 0.975d0)**2,
     $                 mtsr,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      else 
         call genaquad (2,s,mtsr,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      endif


      top12c = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssupq * dcot(beta))**2 * dsiga * mtt**2 * 3.d0
c--------------- Stop loop -----------------------------------------
c
      call genaquad (2,s,mdnsr,mdnsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12d = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * dsiga * mdn**2 * 3.d0
c
      call genaquad (2,s,mstsr,mstsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12e = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * dsiga * mst**2 * 3.d0
c
c--------------- Sbottom loop --------------------------------------
csh old version with no threshold correction
c      call genaquad (2,s,mbsr,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)

csh new version with simple threshold correction
      write(*,*) 'dA-SE should not be used any more'
      write(*,*) 'Delta mb corrections not yet implemented here'
      if ((((mbsl + mbsr - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mbsl + mbsr - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
         call genaquad(2, ((mbsl + mbsr) * 0.975d0)**2,
     $                 mbsr,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      else 
         call genaquad (2,s,mbsr,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      endif

      top12f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnq * dtan(beta))**2 * dsiga * mbb**2 * 3.d0

c$$$      write(*,*) 'dA:', real(mbsr), real(mbsl), real(dsiga), 
c$$$     $     real(mu), real(mssdnq), real(mbb), real(beta)
c$$$      top12c = 0d0
c--------------- Sbottom loop --------------------------------------
c
      call genaquad (2,s,melsr,melsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12g = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * dsiga * mel**2
c
      call genaquad (2,s,mmusr,mmusl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12h = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * dsiga * mmu**2
c
      call genaquad (2,s,mtasr,mtasl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12i = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) *
     &  2.d0*(mu - mssdnl * dtan(beta))**2 * dsiga * mta**2
c
c$$$      write(*,*) 'dA:', real(top12a), real(top12b), real(top12c),
c$$$     $     real(top12d), real(top12e), real(top12f), real(top12g),
c$$$     $     real(top12h), real(top12i)
      if (selec.ge.3) then
      top12 = top12a + top12b + top12c + top12d + top12e +
     &        top12f + top12g + top12h + top12i
      elseif (selec.eq.2) then
         top12 = top12c + top12f
      elseif (selec.eq.1) then
         top12 = top12c
      else
         write(*,*) "Error in Sigma-dA: selec out or range"
      endif


c
c fermion loops
c
      call genaquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9a = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mup**2 * 3.d0
c
      call genaquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9b = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mch**2 * 3.d0
c
c--------------- top loop -----------------------------------------
csh old version with no threshold correction
c      call genaquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)

csh new version with simple threshold correction
      if ((((mtt + mtt - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mtt + mtt - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
c     $    (((mtt + mtt - dsqrt(s))/dsqrt(s)).le.0.006d0)) then
         call genaquad(3, ((mtt + mtt) * 0.975d0)**2,
c         call genaquad(3, ((mtt + mtt) * 0.994d0)**2,
     $                 mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      else 
         call genaquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      endif

      top9c = elec2/(4.d0*ppi)**2 * dcot(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mtt**2 * 3.d0
c--------------- top loop -----------------------------------------
c
      call genaquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9d = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mdn**2 * 3.d0
c
      call genaquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9e = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mst**2 * 3.d0
c
      write(*,*) 'dA-SE should not be used any more'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call genaquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9f = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mbb**2 * 3.d0
c
      call genaquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9g = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mel**2
c
      call genaquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9h = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mmu**2
c
      call genaquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9i = elec2/(4.d0*ppi)**2 * dtan(beta)**2/(4.d0*ssw2*mmw**2)
     &        * dsiga * mta**2
c
      if (selec.ge.3) then
      top9 = top9a + top9b + top9c + top9d + top9e + top9f +
     &       top9g + top9h + top9i
      elseif (selec.eq.2) then
         top9 = top9c + top9f
      elseif (selec.eq.1) then
         top9 = top9c
      else
         write(*,*) "Error in Sigma-dA: selec out or range"
      endif


c --> chargino loops
      top10 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dsin(beta) +
     &      vmix(j,2)*umix(ii,1)*dcos(beta) )/dsqrt(2.d0)
      b = -(vmix(ii,1)*umix(j,2)*dsin(beta) +
     &      vmix(ii,2)*umix(j,1)*dcos(beta) )/dsqrt(2.d0)
csh old version with no threshold correction
c      call genaquad (4,s,mcha(ii),mcha(j),a,-b,b,-a, siga,dsiga)
csh new version with simple threshold correction
      if ((((mcha(ii) + mcha(j) - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mcha(ii) + mcha(j) - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
         call genaquad(4, ((mcha(ii) + mcha(j)) * 0.975d0)**2,
     $                 mcha(ii),mcha(j),a,-b,b,-a, siga,dsiga)
      else 
         call genaquad (4,s,mcha(ii),mcha(j),a,-b,b,-a, siga,dsiga)
      endif
      top10h = elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * dsiga
      top10 = top10 + top10h
 621   continue
620   continue


c --> neutralino loops
      top11 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dsin(beta) - sm * dcos(beta)
      b = -a
csh old version with no threshold correction
c      call genaquad (4,s,mne(j),mne(ii),a,a,b,b, siga,dsiga)
csh new version with simple threshold correction
      if ((((mne(ii) + mne(j) - dsqrt(s))/dsqrt(s)).ge.0d0).and.
     $    (((mne(ii) + mne(j) - dsqrt(s))/dsqrt(s)).le.0.025d0)) then
         call genaquad(4, ((mne(ii) + mne(j)) * 0.975d0)**2,
     $                 mne(j),mne(ii),a,a,b,b, siga,dsiga)
      else 
         call genaquad (4,s,mne(j),mne(ii),a,a,b,b, siga,dsiga)
      endif
      top11h = elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * dsiga
      top11 = top11 + top11h
 623   continue
622   continue
c
      dsigmaab = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8
      dsigmaaf = top9  
      dsigmaas = top12  
      dsigmaac = top10 + top11
      dsigmaat = top12c + top9c
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (6,*) ' derivative of A - selfenergy : ', real(dsqrt(s))
       write (6,*) ' W+ H-       = ', top1
       write (6,*) ' Z0 H0       = ', top2
       write (6,*) ' Z0 h0       = ', top3
       write (6,*) ' A0 H0       = ', top4
       write (6,*) ' A0 h0       = ', top5
       write (6,*) ' G0 H0       = ', top6
       write (6,*) ' G0 h0       = ', top7
       write (6,*) ' G+ H-       = ', top8
       write (6,*) ' fer fer     = ', top9
       write (6,*) ' sfer sfer   = ', top12
       write (6,*) ' cha         = ', top10
       write (6,*) ' neu         = ', top11
       write (6,*) '   '
      endif
c
      return
      end
c










c=====================================================================
c
      subroutine genazquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c           _______/          \_______                  .     .
c        q         \          /        q                  .  .
c                   \________/                q ____________.__________q
c                       m2
c
      call bquer2(q2,dabs(m2),dabs(m1),
     &             b0,b1,pb0,pb1)
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
c
c$$$       if ((m1.eq.mcha(1)).or.(m1.eq.mcha(2))) then
c$$$          write(*,*) real(m1), real(m2), real(b0), real(b1)
c$$$          write(*,*) real((a * at + b * bt) * m2 * (b0 + b1)),
c$$$     $               real((a * bt + b * at) * m1 * b1), 
c$$$     $               real(b0 + b1), real(b1)
c$$$       endif
      if (typ.eq.1) then
       siga = -(b0 - b1)
c      siga = -1.d0/epsilon
      else
      if (typ.eq.2) then
       siga = b0 + 2.d0 * b1
c      siga = 0.d0
      else
      if (typ.eq.3) then
       siga  = -8.d0 * ( (a * at + b * bt) * m2 * (b0 + b1) +
     &                   (a * bt + b * at) * m1 * b1 )
c      siga = -8.d0*( (a * at + b * bt) * m2 -
c    &               (a * bt + b * at) * m1 )/epsilon
      else
      if (typ.eq.4) then
       siga  = 8.d0 * ( (a * at + b * bt) * m2 * (b0 + b1) +
csh test       siga  = 8.d0 * ( (a * at + b * bt) * m2 * (b1) +
     &                  (a * bt + b * at) * m1 * b1 )
c      siga  = 8.d0 * ( (a * at + b * bt) * m2 -
c    &          (a * bt + b * at) * m1  ) /epsilon
      else
       write (6,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
c
      return
      end
c
c ----------------------------------------------------------
c
      subroutine sigmaaz (s, sigmaazb,sigmaazs,sigmaazf,sigmaazc,
     &                       sigmaazt)
c
c     selfenergy of pseudoscalar - Z0 mixing
c     here Sigma_vu = q_vu * Sigma.
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
      integer print5
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      common /secheck/print5

c      write(*,*) 'AZ:', real(mmz), real(mhh)
c
c boson loops
c
c  notation :
c     genazquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genazquad (1,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2 / (2.d0*ssw2*ccw2) * siga *
     &       mmz * dcos(beta-alpha)*dsin(beta-alpha)
c
      call genazquad (1,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2/(4.d0*ppi)**2 / (2.d0*ssw2*ccw2) * siga *
     &       mmz * dcos(beta-alpha)*dsin(beta-alpha)
c
      call genazquad (2,s,maa,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga * mmz *
     &       dcos(2.d0*beta)*dsin(beta-alpha)*dcos(beta+alpha)
c
      call genazquad (2,s,maa,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga * mmz *
     &       dcos(2.d0*beta)*dcos(beta-alpha)*dsin(beta+alpha)
c
      call genazquad (2,s,mmz,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga * mmz *
     &       dsin(2.d0*beta)*dcos(beta-alpha)*dcos(beta+alpha)
c
      call genazquad (2,s,mmz,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = elec2/(4.d0*ppi)**2 / (4.d0*ssw2*ccw2) * siga * mmz *
     &       dsin(2.d0*beta)*dsin(beta-alpha)*dsin(beta+alpha)
c
c fermion loops :
c
      a =  ( -0.5d0 + 2.d0 * ssw2 - 0.5d0) / (4.d0 * ssw * ccw)
      b =  ( -0.5d0 + 2.d0 * ssw2 + 0.5d0) / (4.d0 * ssw * ccw)
      at =  -0.5d0
      bt =   0.5d0
      call genazquad (3,s,mel,mel,a,at,b,bt, siga,dsiga)
      top7a = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mel*dtan(beta) * siga
c
      call genazquad (3,s,mmu,mmu,a,at,b,bt, siga,dsiga)
      top7b = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mmu*dtan(beta) * siga
c
      call genazquad (3,s,mta,mta,a,at,b,bt, siga,dsiga)
      top7c = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mta*dtan(beta) * siga
c
      a = ( 0.5d0 - 4.d0 * ssw2/3.d0 + 0.5d0) / (4.d0 * ssw * ccw)
      b = ( 0.5d0 - 4.d0 * ssw2/3.d0 - 0.5d0) / (4.d0 * ssw * ccw)
      call genazquad (3,s,mup,mup,a,at,b,bt, siga,dsiga)
      top7d = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mup*dcot(beta)*3.d0 *
     &         siga
c      write(*,*) 'AZ-SE (up):', top7d
c
      call genazquad (3,s,mch,mch,a,at,b,bt, siga,dsiga)
      top7e = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mch*dcot(beta)*3.d0 *
     &         siga
c
      call genazquad (3,s,mtt,mtt,a,at,b,bt, siga,dsiga)
      top7f = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mtt*dcot(beta)*3.d0 *
     &         siga
      if (print5.eq.1) write(*,*) 'AZ-SE, top :', top7f
c      write(*,*) 'AZ-SE (top):', top7f
c
      a =  ( -0.5d0 + 2.d0 * ssw2/3.d0 - 0.5d0) / (4.d0 * ssw * ccw)
      b =  ( -0.5d0 + 2.d0 * ssw2/3.d0 + 0.5d0) / (4.d0 * ssw * ccw)
      call genazquad (3,s,mdn,mdn,a,at,b,bt, siga,dsiga)
      top7g = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mdn*dtan(beta)*3.d0 *
     &         siga
c      write(*,*) 'AZ-SE (dn):', top7g
c
      call genazquad (3,s,mst,mst,a,at,b,bt, siga,dsiga)
      top7h = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mst*dtan(beta)*3.d0 *
     &         siga
c
      call genazquad (3,s,mbb,mbb,a,at,b,bt, siga,dsiga)
c$$$      write(*,*) 'AZ: bottom contribution:'
c$$$      write(*,*) real(mbb), real(a), real(at), real(b), real(bt), 
c$$$     $           real(siga), real(dsiga)
      write(*,*) 'AZ-SE should not be used any more'
      write(*,*) 'Delta mb corrections not yet implemented here'
      top7i = -elec2/(4.d0*ppi)**2 /(2.d0*ssw*mmw)*mbb*dtan(beta)*3.d0 *
     &         siga
      if (print5.eq.1) write(*,*) 'AZ-SE, bot :', top7i
c      write(*,*) 'AZ-SE (bot):', top7i
c
c$$$      write(*,*) 'AZ: fermion contributions:'
c$$$      write(*,*) real(top7a), real(top7b), real(top7c), real(top7d),
c$$$     $           real(top7e), real(top7f), real(top7g), real(top7h),
c$$$     $           real(top7i)
      if (selec.ge.3) then
      top7 = top7a + top7b + top7c + top7d + top7e + top7f + top7g +
     &       top7h + top7i
      elseif (selec.eq.2) then
         top7 = top7f + top7i
      elseif (selec.eq.1) then
         top7 = top7f
      else
         write(*,*) "Error in Sigma-AZ: selec out or range"
      endif


c
      top8 = 0.d0
      do 760 ii = 1,2
       do 761 j = 1,2
c --> Haber/Kabe couplings for Z Chi Chi          
       a = -vmix(ii,1)*vmix(j,1)-vmix(ii,2)*vmix(j,2)/2.d0
       b = -umix(ii,1)*umix(j,1)-umix(ii,2)*umix(j,2)/2.d0
c --> Rosieks couplings for Z Chi Chi (wrong implementation)
c        a = -vmix(ii,1)*vmix(j,1)
c        b = -umix(ii,1)*umix(j,1)
       at =  (vmix(j,1)*umix(ii,2) * dsin(beta) +
     &        vmix(j,2)*umix(ii,1) * dcos(beta))/dsqrt(2.d0)
c --> old version (wrong!!)
c       bt =  (vmix(ii,1)*umix(j,2) * dsin(beta) +
c     &        vmix(ii,2)*umix(j,1) * dcos(beta))/dsqrt(2.d0)
corrected by sh, 20Apr99
c     check performed by Torsten Blank
       bt = - (vmix(ii,1)*umix(j,2) * dsin(beta) +
     &         vmix(ii,2)*umix(j,1) * dcos(beta))/dsqrt(2.d0)
       if (ii.eq.j) then
c --> Haber/Kabe couplings for Z Chi Chi          
	a = a + ssw2
	b = b + ssw2
c --> Rosieks couplings for Z Chi Chi (wrong implementation)
c        a = a + ccw2 - ssw2
c        b = b + ccw2 - ssw2
       endif

c       write(*,*) ii, j, 
c     $  real(elec2/(ssw2*ccw) * mcha(j) * (a * at + b * bt)),
c     $  real(elec2/(ssw2*ccw) * mcha(ii)* (a * bt + b * at))
c     $  real(mcha(j) * (a * at + b * bt)),
c     $  real(mcha(ii)* (a * bt + b * at))
c --> old version
      call genazquad (4,s,mcha(j),mcha(ii),a,at,b,bt, siga,dsiga)
c --> old version
c      top8h =  elec2/(4.d0*ppi)**2 /(4.d0*ssw2*ccw) * siga
corrected by sh, 20Apr99
c     check performed by Torsten Blank
      top8h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2*ccw) * siga
c      write(*,*) 'AZ-SE: top8h:', real(top8h)
      top8 = top8 + top8h
761   continue
760   continue
c      top8 = -top8 ! no reason for change of sign
c      write(*,*) 'AZ-SE: Charginos:', real(top8)
c
      top9 = 0.d0
      do 762 ii = 1,4
       do 763 j = 1,4
      a = (-nmix(ii,3)*nmix(j,3) + nmix(ii,4)*nmix(j,4))/2.d0
      b = -a
      qm = ( nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &       nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) ) /2.d0
      sm = ( nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &       nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) ) /2.d0
      at = qm * dsin(beta) - sm * dcos(beta)
c --> old version
c      bt = at
corrected by sh, 20Apr99
c     check performed by Torsten Blank
      bt = -at
      call genazquad (3,s,mne(j),mne(ii),a,at,b,bt, siga,dsiga)
c      call genazquad (3,s,mne(ii),mne(j),a,at,b,bt, siga,dsiga)
c                          ^^^^^^^^^^^^^^ change irrelevant
c --> old version
c      top9h = -elec2/(4.d0*ppi)**2 /(8.d0*ssw2*ccw) * siga
corrected by sh, 20Apr99
c     check performed by Torsten Blank
      top9h = +elec2/(4.d0*ppi)**2 /(8.d0*ssw2*ccw) * siga
      top9 = top9 + top9h
763   continue
762   continue
c      top9 = -top9 ! no reason for change of sign
c
c printroutine
c
      pr = 0
c
      if(print5.eq.1) then
      write(*,*) "----------------------------------------------------"
       write (*,*) ' A - Z0 - mixing : ', real(dsqrt(s))
       write (*,*) ' Z0 H0       = ', top1
       write (*,*) ' Z0 h0       = ', top2
       write (*,*) ' A  H0       = ', top3
       write (*,*) ' A  h0       = ', top4
       write (*,*) ' G0 H0       = ', top5
       write (*,*) ' G0 h0       = ', top6
       write (*,*) ' fer fer     = ', top7
       write (*,*) ' cha cha     = ', top8
       write (*,*) ' neu neu     = ', top9
       write (*,*) '   '
      write(*,*) "----------------------------------------------------"
      endif
c
      sigmaazb = top1 + top2 + top3 + top4 + top5 + top6
      sigmaazf = top7
      sigmaazc = top8 + top9
      sigmaazs = 0.d0
      sigmaazt = top7f
c      write(*,*) 'SigmaAZ:', real(top8), real(top9)
c
      return
      end
c

c=====================================================================
c
      subroutine genwquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
      implicit double precision (a-z)
      real*8 b0,b1,pb0,pb1                        
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c          ......../          \........                 .     .
c        q         \          /        q                  .  .
c                   \________/                q ...................... q
c                        m2
c
c
      pb22 = 0.d0
      pb0  = 0.d0
      pb1 = 0.d0
c
      if(dabs(m2).le.1d-7) then
       if(dabs(m1).le.1d-7) then
	 b0 = delta(epsilon,muee,1.d0) + 2.d0 - dlog(q2)
	 b1 = -0.5d0 * (delta(epsilon,muee,1.d0) + 2.d0) +
     &         0.5d0 * dlog(q2)
	 b22 = (q2 * b1/2.d0 - q2/6.d0) / 3.d0
       else
	if (q2.le.1.d-8) then
       b0 = delta(epsilon,muee,1.d0) - dlog(m1**2) + 1.d0
       b1 = -0.5d0 * ( delta(epsilon,muee,1.d0) - dlog(m1**2) )
     &      - 0.25d0
       b22  = (aa(dabs(m1))/2.d0 -
     &         m1**2 * b1/2.d0 + m1**2/2.d0 )/3.d0
	else
       b0 = delta(epsilon,muee,1.d0) + 1.d0 - 2.d0 *
     &      dlog(dabs(m1)) + f(q2,0.d0,dabs(m1))
       b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &     +1.d0) + (m1**2 - q2)/(2.d0 * q2) * f(q2,0.d0,dabs(m1))
       b22  = (aa(dabs(m1))/2.d0 +  (q2 -
     &        m1**2) * b1/2.d0 + (m1**2)/2.d0
     &        - q2/6.d0)/3.d0
	endif
       endif
      else
      if (q2.le.1.d-8) then
      b0 = delta(epsilon,muee,1.d0) - dlog(dabs(m1*m2)) + 1.d0
      b1 = -0.5d0 - 0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2))
       if (dabs(dabs(m1)-dabs(m2)).le.1.d-6) then
       b0 = b0 - 1.d0
       b1 = b1 + 0.5d0
       else
       b0 = b0 - (m1**2+m2**2)/(m1**2-m2**2)*dlog(dabs(m1/m2))
       b1 = b1 + m2**2/(m2**2-m1**2)*dlog(dabs(m2/m1)) +
     &      1.d0/2.d0/(m1**2-m2**2) * ( (m2**2+m1**2)/2.d0
     &      - m2**2*m1**2/(m2**2-m1**2)*dlog(m2**2/m1**2) )
       endif
      else
       call bquer2(q2,dabs(m2),dabs(m1),
     &             b0,b1,pb0,pb1)
       b0 =  delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
       b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
      endif
      b22  = (aa(dabs(m1))/2.d0 + m2**2 * b0 + (q2 + m2**2 -
     &        m1**2) * b1/2.d0 + (m1**2 + m2**2)/2.d0
     &        - q2/6.d0)/3.d0
      endif
c
      if (typ.eq.1) then
	siga  = 4.d0 * b22
c       siga = 2.d0*(m1**2+m2**2-q2/3.d0)/epsilon
	dsiga = 4.d0 * pb22
      else
      if (typ.eq.2) then
	siga  = -b0
c       siga = -2.d0/epsilon
	dsiga = -pb0
      else
      if (typ.eq.3) then
       siga  = 2.d0 * (-2.d0 * b22 + aa(m1) + m2**2 * b0
     &         + q2 * b1)
c      siga  = 2.d0 * (m1**2 + m2**2 - 2.d0*q2/3.d0)/epsilon
       dsiga = 2.d0 * (-2.d0 * pb22 + m2**2 * pb0 + b1 +
     &                   q2 * pb1)
      else
c
      if (typ.eq.4) then
	siga  = 8.d0 * ((at * a + bt * b) * (-2.d0 * b22 +
     &         aa(dabs(m1)) + m2**2 * b0 + q2 * b1) -
     &         (bt * a + at * b) * m1 * m2 * b0 )
c       siga = 8.d0 * ((at * a + bt * b) * (m1**2 + m2**2 -
c    &    2.d0*q2/3.d0) - 2.d0 * (bt * a + at * b) * m1 * m2
c    &    ) / epsilon
	dsiga = 8.d0 * ((at * a + bt * b) * (-2.d0 * pb22 +
     &         m2**2 * pb0 + q2 * pb1 + b1) -
     &         (bt * a + at * b) * m1 * m2 * pb0 )
      else
      if (typ.eq.5) then
	siga  = (10.d0 * b22 + (4.d0 * q2 + m1**2 + m2**2) *
     &          b0 + aa(m1) + aa(m2) - 2.d0 * ( m1**2 + m2**2
     &           - q2/3.d0 ) )
c       siga = (9.d0*(m1**2+m2**2) + 19.d0/3.d0*q2)/epsilon
	dsiga = (10.d0 * pb22 + 4.d0 * b0 + (4.d0 * q2 +
     &           m1**2 + m2**2) * pb0 - 2.d0/3.d0)
      else
      if (typ.eq.6) then
	siga  = -b22
c       siga = -(m1**2+m2**2-q2/3.d0)/2.d0/epsilon
	dsiga = -pb22
      else
      if (typ.eq.7) then
	siga  = -aa(m1)
c       siga = -2.d0*m1**2/epsilon
	dsiga = 0.d0
      else
      if (typ.eq.8) then
	siga  =  6.d0 * aa(m1) - 4.d0 * m1**2
c       siga = 12.d0*m1**2/epsilon
	dsiga =  0.d0
      else
       write (*,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
      endif
      endif
      endif
      endif
c
      return
      end
c
c ----------------------------------------------------------
c
      subroutine sigmaw (s, sswb,ssws,sswf,sswc)
c
c     selfenergy of w boson
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "Sigma-W should not be used for Higgs masses!!"
c
c boson loops
c
c  notation :
c     genwquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c o.k :
      call genwquad (1,s,mhp,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2*dsin(beta-alpha)**2/(4.d0*ssw2)*siga
c o.k :
      call genwquad (1,s,mhp,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2/(4.d0*ppi)**2*dcos(beta-alpha)**2/(4.d0*ssw2)*siga
c o.k :
      call genwquad (1,s,mhp,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = -elec2/(4.d0*ppi)**2/(4.d0*ssw2)*siga
c o.k :
      call genwquad (1,s,mmw,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = -elec2/(4.d0*ppi)**2*dcos(beta-alpha)**2/(4.d0*ssw2)*siga
c o.k :
      call genwquad (1,s,mmw,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*dsin(beta-alpha)**2/(4.d0*ssw2)*siga
c o.k :
      call genwquad (1,s,mmw,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c o.k :
      call genwquad (2,s,mmw,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2*mmw**2*dcos(beta-alpha)**2/ssw2*siga
c o.k :
      call genwquad (2,s,mmw,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2*mmw**2*dsin(beta-alpha)**2/ssw2*siga
c o.k :
      call genwquad (2,s,mmw,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2*mmw**2*siga
c o.k :
      call genwquad (2,s,mmz,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2*mmz**2*ssw2*siga
c o.k :
      call genwquad (1,s,mvesl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mvesl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,melsl,mvesr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,melsr,mvesr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12a = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang1)*
     & dcos(ang2))**2+siga2*(dcos(ang1)*dsin(ang2))**2+siga3*(dsin(ang1)
     & *dcos(ang2))**2+siga4*(dsin(ang1)*dsin(ang2))**2 )
c
      call genwquad (1,s,mvmsl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mvmsl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,mmusl,mvmsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,mmusr,mvmsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12b = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang3)*
     & dcos(ang4))**2+siga2*(dcos(ang3)*dsin(ang4))**2+siga3*(dsin(ang3)
     & *dcos(ang4))**2+siga4*(dsin(ang3)*dsin(ang4))**2 )
c
      call genwquad (1,s,mvtsl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mvtsl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,mtasl,mvtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,mtasr,mvtsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12c = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang5)*
     & dcos(ang6))**2+siga2*(dcos(ang5)*dsin(ang6))**2+siga3*(dsin(ang5)
     & *dcos(ang6))**2+siga4*(dsin(ang5)*dsin(ang6))**2 )
c
      call genwquad (1,s,mupsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mupsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,mupsr,mdnsl,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,mupsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12d = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang7)*
     & dcos(ang8))**2+siga2*(dcos(ang7)*dsin(ang8))**2+siga3*(dsin(ang7)
     & *dcos(ang8))**2+siga4*(dsin(ang7)*dsin(ang8))**2 )
c
      call genwquad (1,s,mchsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mchsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,mchsr,mstsl,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,mchsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12e = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang9)*
     & dcos(ang10))**2+siga2*(dcos(ang9)*dsin(ang10))**2+siga3*(dsin
     & (ang9)*dcos(ang10))**2+siga4*(dsin(ang9)*dsin(ang10))**2 )
c
      write(*,*) 'W-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call genwquad (1,s,mtsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (1,s,mtsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genwquad (1,s,mtsr,mbsl,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      call genwquad (1,s,mtsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga4,dsiga4)
      top12f = -elec2/(4.d0*ppi)**2/(2.d0*ssw2)*(siga1*(dcos(ang11)*
     & dcos(ang12))**2+siga2*(dcos(ang11)*dsin(ang12))**2+siga3*
     & (dsin(ang11)*dcos(ang12))**2+siga4*(dsin(ang11)*dsin(ang12))**2 )
c
      top12 = top12a+top12b+top12c + (top12d + top12e + top12f)*3.d0
c o.k :
      call genwquad (5,s,mmw,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2/(4.d0*ppi)**2*siga
c o.k :
      call genwquad (5,s,mmz,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2*ccw2/ssw2*siga
c o.k :
      call genwquad (6,s,mmw,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top16 = -2.d0*elec2/(4.d0*ppi)**2*ccw2/ssw2*siga
c o.k :
      call genwquad (6,s,mmw,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17 = -2.d0*elec2/(4.d0*ppi)**2*siga
c o.k :
      call genwquad (7,s,mhh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18a = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c o.k :
      call genwquad (7,s,mlh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18b = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c o.k :
      call genwquad (7,s,maa,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18c = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c
      top18 = top18a + top18b + top18c
c o.k :
      call genwquad (7,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top19 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga
c o.k :
      call genwquad (7,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20a = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
c o.k :
      call genwquad (7,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20b = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2) * siga
c
      top20 = top20a + top20b
c o.k :
      call genwquad (8,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top21 = elec2/(4.d0*ppi)**2 /(2.d0*ssw2) * siga
c o.k :
      call genwquad (8,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top22 = elec2/(4.d0*ppi)**2 * ccw2/(2.d0*ssw2) * siga
c
      call genwquad (7,s,mvesl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mvesr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23a =-elec2/(4.d0*ppi)**2/(2.d0*ssw2) * ( siga1 * dcos(ang1)**2
     &         + siga2 * dsin(ang1)**2 )
c
      call genwquad (7,s,mvmsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mvmsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23b =-elec2/(4.d0*ppi)**2/(2.d0*ssw2) * ( siga1 * dcos(ang3)**2
     &         + siga2 * dsin(ang3)**2 )
c
      call genwquad (7,s,mvtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mvtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23c =-elec2/(4.d0*ppi)**2/(2.d0*ssw2) * ( siga1 * dcos(ang5)**2
     &         + siga2 * dsin(ang5)**2 )
c
      call genwquad (7,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23d =-elec2/(4.d0*ppi)**2/(2.d0*ssw2) * ( siga1 * dcos(ang2)**2
     &         + siga2 * dsin(ang2)**2 )
c
      call genwquad (7,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23e =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*( siga1 * dcos(ang4)**2
     &         + siga2 * dsin(ang4)**2 )
c
      call genwquad (7,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23f =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*( siga1 * dcos(ang6)**2
     &         + siga2 * dsin(ang6)**2 )
c
      call genwquad (7,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23g =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*( siga1 * dcos(ang7)**2
     &         + siga2 * dsin(ang7)**2 )
c
      call genwquad (7,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23h =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*( siga1 * dcos(ang8)**2
     &         + siga2 * dsin(ang8)**2 )
c
      call genwquad (7,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23i =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*(siga1 * dcos(ang10)**2
     &         + siga2 * dsin(ang10)**2 )
c
      call genwquad (7,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23j =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*( siga1 * dcos(ang9)**2
     &         + siga2 * dsin(ang9)**2 )
c
      write(*,*) 'W-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call genwquad (7,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23k =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*(siga1 * dcos(ang12)**2
     &         + siga2 * dsin(ang12)**2 )
c
      call genwquad (7,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genwquad (7,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top23l =-elec2/(4.d0 * ppi)**2/(2.d0*ssw2)*(siga1 * dcos(ang11)**2
     &         + siga2 * dsin(ang11)**2 )
c
      top23 =   top23a + top23b + top23c + top23d + top23e + top23f
     &   +3.d0*(top23g + top23h + top23i + top23j + top23k + top23l)
c
c fermion loops
c o.k
      call genwquad (3,s,mel,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11a = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga
c
      call genwquad (3,s,mmu,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11b = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga
c
      call genwquad (3,s,mta,0.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11c = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga
c
      call genwquad (3,s,mdn,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11d = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga * 3.d0
c
      call genwquad (3,s,mst,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11e = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga * 3.d0
c
      call genwquad (3,s,mbb,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11f = -elec2 / (4.d0 * ppi)**2 /(2.d0*ssw2) * siga * 3.d0
c
      top11 = top11a + top11b + top11c + top11d + top11e + top11f
c
      top13 = 0.d0
      do 640 ii = 1,2
       do 641 j = 1,4
      at =  -nmix(j,4)*vmix(ii,2)/dsqrt(2.d0) + nmix(j,2)*vmix(ii,1)
      bt =   nmix(j,3)*umix(ii,2)/dsqrt(2.d0) + nmix(j,2)*umix(ii,1)
      call genwquad (4,s,mcha(ii),mne(j),at,at,bt,bt, siga,dsiga)
      top13h = -elec2 / (4.d0 * ppi)**2 /(4.d0*ssw2) * siga
      top13 = top13 + top13h
641   continue
640   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' W - selfenergy : ', real(dsqrt(s))
       write (*,*) ' H+ H0       = ', top1
       write (*,*) ' H+ h0       = ', top2
       write (*,*) ' H+ A0       = ', top3
       write (*,*) ' G+ H0       = ', top4
       write (*,*) ' G+ h0       = ', top5
       write (*,*) ' G+ G0       = ', top6
       write (*,*) ' W+ H0       = ', top7
       write (*,*) ' W+ h0       = ', top8
       write (*,*) ' G+ gamma    = ', top9
       write (*,*) ' G+ Z0       = ', top10
       write (*,*) ' fer fer     = ', top11
       write (*,*) ' sfer sfer   = ', top12
       write (*,*) ' cha neut    = ', top13
       write (*,*) ' W+ gamma    = ', top14
       write (*,*) ' W+ Z0       = ', top15
       write (*,*) ' gh+ ghZ     = ', top16
       write (*,*) ' gh+ ghgam   = ', top17
       write (*,*) '   four point interactions : '
       write (*,*) ' H0          = ', top18a
       write (*,*) ' h0          = ', top18b
       write (*,*) ' A           = ', top18c
       write (*,*) ' H+          = ', top19
       write (*,*) ' G0 + G+     = ', top20
       write (*,*) ' W+          = ', top21
       write (*,*) ' Z0          = ', top22
       write (*,*) ' sfer        = ', top23
       write (*,*) '               '
      endif
c
      sswb = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &      top9 + top10 + top14 + top15 + top16 + top17 + top18 +
     &      top19 + top20 + top21 + top22
      sswf = top11
      ssws = top12 + top23
      sswc = top13
c
      return
      end
c=====================================================================
c
      subroutine gengquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
c     m1 = m2 !
c
      implicit double precision (a-z)
      real*8 b0,b1,pb0,pb1                        
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c          ......../          \........                 .     .
c        q         \          /        q                  .  .
c                   \________/                q ...................... q
c                        m2
c
c
      if((dabs(m2).le.1d-7).or.(dabs(m1).le.1d-7)) then
       write (*,*) ' !!! mi = 0.0 !!!  '
      else
       if (dabs(q2).gt.1d-7) then
       call bquer2 (q2,dabs(m2),dabs(m1),
     &             b0,b1,pb0,pb1)
       b0 =  delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
       b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       b22  = (aa(dabs(m1))/2.d0 + m2**2 * b0 + (q2 + m2**2 -
     &        m1**2) * b1/2.d0 + (m1**2 + m2**2)/2.d0
     &        - q2/6.d0)/3.d0
       pb22 = (m2**2 * pb0 + (b1 + (q2 + m2**2 - m1**2)
     &         * pb1)/2.d0 - 1.d0/6.d0) / 3.d0
	else
       b0 = delta(epsilon,muee,1.d0) - dlog(dabs(m1*m2))
       b1 = -b0/2.d0
       pb0 = 1.d0/(6.d0*m1**2)
       pb1 = -pb0/2.d0
       b22  = (aa(dabs(m1))/2.d0 + m2**2 * b0 + (q2 + m2**2 -
     &        m1**2) * b1/2.d0 + (m1**2 + m2**2)/2.d0
     &        - q2/6.d0)/3.d0
       pb22 = (m2**2 * pb0 + (b1 + (q2 + m2**2 - m1**2)
     &         * pb1)/2.d0 - 1.d0/6.d0) / 3.d0
       endif
      endif
c
      if (typ.eq.1) then
	dsiga = 4.d0 * b22
	siga  = 4.d0 * pb22
c       siga = -1.d0/epsilon/3.d0 * 4.d0 / 2.d0
      else
      if (typ.eq.2) then
	dsiga = -b0
	siga  = -pb0
c       siga = 0.d0
      else
      if (typ.eq.3) then
	siga  = 2.d0 * (-2.d0 * pb22 + m2**2 * pb0 + b1 +
     &                   q2 * pb1)
	dsiga = 2.d0 * (-2.d0 * b22 + aa(m1) + m2**2 * b0
     &          + q2 * b1)
      else
      if (typ.eq.4) then
	siga  = 8.d0 * ((at * a + bt * b) * (-2.d0 * pb22 +
     &          m2**2 * pb0 + q2 * pb1 + b1) -
     &          (bt * a + at * b) * m1 * m2 * pb0 )
c       siga = -8.d0 * (at * a + bt * b) * 2.d0 /epsilon/3.d0
	dsiga = 8.d0 * ((at * a + bt * b) * (-2.d0 * b22 +
     &         aa(dabs(m1)) + m2**2 * b0 + q2 * b1) -
     &         (bt * a + at * b) * (m1 * m2) * b0 )
      else
      if (typ.eq.5) then
       dsiga = (10.d0 * b22 + (4.d0 * q2 + m1**2 + m2**2) *
     &       b0 + aa(m1) + aa(m2) - 2.d0 * ( m1**2 + m2**2
     &       - q2/3.d0 ) )
       siga  = (10.d0 * pb22 + 4.d0 * b0 + (4.d0 * q2 +
     &           m1**2 + m2**2) * pb0 + 2.d0/3.d0)
c      siga = 19.d0/3.d0/epsilon
      else
      if (typ.eq.6) then
	dsiga = -b22
	siga  = -pb22
c       siga =  1.d0/6.d0/epsilon
      else
      if (typ.eq.7) then
	dsiga  = -aa(m1)
	siga = 0.d0
      else
      if (typ.eq.8) then
	dsiga  =  6.d0 * aa(m1) - 4.d0 * m1**2
	siga =  0.d0
      else
       write (*,*) ' typpindent wrong '
c
      endif
      endif
      endif
      endif
      endif
      endif
      endif
      endif
c
      return
      end
c
c ----------------------------------------------------------
c
      subroutine dsigmag (s, sgb,sgs,sgf,sgc)
c
c     photon - vacuumpolarization
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "dSigma-g should not be used for Higgs masses!!"
c
c  notation :
c     gengquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
c boson loops
c
      call gengquad (1,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = -elec2/(4.d0*ppi)**2 * 2.d0 * mmw**2 * siga
c
      call gengquad (1,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5a = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5b = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5c = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5d = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5e = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5f = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5g = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      call gengquad (1,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5h = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      write(*,*) 'dgamma-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call gengquad (1,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5i = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      if (selec.ge.3) then
      top5 = top5a + top5b + top5c + (top5d + top5e + top5f + top5g +
     &       top5h + top5i)*3.d0
      elseif (selec.eq.2) then
         top5 = top5i + top5f
      elseif (selec.eq.1) then
         top5 = top5f
      else
         write(*,*) "Error in Sigma-dg: selec out or range"
      endif


c
      call gengquad (1,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13a = -elec2/(4.d0 * ppi)**2 * siga
c
      call gengquad (1,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13b = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13c = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (1,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13d = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13e = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13f = -elec2/(4.d0*ppi)**2 * siga * 4.d0/9.d0
c
      call gengquad (1,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13g = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      call gengquad (1,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13h = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      call gengquad (1,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13i = -elec2/(4.d0*ppi)**2 * siga * 1.d0/9.d0
c
      if (selec.ge.3) then
      top13 = top13a + top13b + top13c + (top13d + top13e + top13f +
     &        top13g + top13h + top13i)*3.d0
      elseif (selec.eq.2) then
         top13 = top13f + top13i
      elseif (selec.eq.1) then
         top13 = top13f
      else
         write(*,*) "Error in Sigma-Z: selec out or range"
      endif

c
      call gengquad (5,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (6,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2 * siga * 2.d0
c
      aa = 1.d0/2.d0
      call gengquad (4,s,mel,mel,aa,aa,aa,aa, siga,dsiga)
      top4a = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (4,s,mmu,mmu,aa,aa,aa,aa, siga,dsiga)
      top4b = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (4,s,mta,mta,aa,aa,aa,aa, siga,dsiga)
      top4c = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (4,s,mup,mup,aa,aa,aa,aa, siga,dsiga)
      top4d = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call gengquad (4,s,mdn,mdn,aa,aa,aa,aa, siga,dsiga)
      top4e = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      call gengquad (4,s,mch,mch,aa,aa,aa,aa, siga,dsiga)
      top4f = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call gengquad (4,s,mst,mst,aa,aa,aa,aa, siga,dsiga)
      top4g = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      call gengquad (4,s,mtt,mtt,aa,aa,aa,aa, siga,dsiga)
      top4h = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call gengquad (4,s,mbb,mbb,aa,aa,aa,aa, siga,dsiga)
      top4i = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      if (selec.ge.3) then
      top4 = top4a + top4b + top4c + top4d + top4e + top4f +
     &       top4g + top4h + top4i
      elseif (selec.eq.2) then
         top4 = top4h + top4i
      elseif (selec.eq.1) then
         top4 = top4h
      else
         write(*,*) "Error in Sigma-dg: selec out or range"
      endif
c
      a = 0.5d0
      call gengquad (4,s,mcha(1),mcha(1),a,a,a,a, siga,dsiga)
      top8a = -elec2/(4.d0*ppi)**2 * siga
c
      call gengquad (4,s,mcha(2),mcha(2),a,a,a,a, siga,dsiga)
      top8b = -elec2/(4.d0*ppi)**2 * siga
c
      top8 = top8a + top8b
c
      sgb = top1 + top2 + top3 + top6 + top7
      sgs = top5 + top13
      sgc = top8
      sgf = top4
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' photon - vac.polarization : ', real(dsqrt(s))
       write (*,*) ' H+ H-       = ', top1
       write (*,*) ' G+ G-       = ', top2
       write (*,*) ' W+ G-       = ', top3
       write (*,*) ' W+ W-       = ', top6
       write (*,*) ' gh+ gh-     = ', top7
       write (*,*) ' fer fer     = ', top4
       write (*,*) ' sfer sfer l = ', top5
       write (*,*) ' sfer sfer r = ', top13
       write (*,*) ' charg i,i   = ', top8
       write (*,*) '               '
      endif
c
      return
      end
c
c ---------------------------------------------------------------------
c
      subroutine sigmag (s, sigmagb,sigmags,sigmagf,sigmagc)
c
c     photon - selfenergy
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "Sigma-g should not be used for Higgs masses!!"

c
c  notation :
c     gengquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
c boson loops
c
      call gengquad (1,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = -elec2/(4.d0*ppi)**2 * 2.d0 * mmw**2 * dsiga
c
      call gengquad (1,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5a = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5b = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5c = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5d = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5e = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5f = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5g = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      call gengquad (1,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5h = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      write(*,*) 'gamma-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call gengquad (1,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5i = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      top5 = top5a + top5b + top5c + (top5d + top5e + top5f + top5g +
     &       top5h + top5i)*3.d0
c
      call gengquad (1,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13a = -elec2/(4.d0 * ppi)**2 * dsiga
c
      call gengquad (1,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13b = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13c = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (1,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13d = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13e = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13f = -elec2/(4.d0*ppi)**2 * dsiga * 4.d0/9.d0
c
      call gengquad (1,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13g = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      call gengquad (1,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13h = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      call gengquad (1,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13i = -elec2/(4.d0*ppi)**2 * dsiga * 1.d0/9.d0
c
      top13 = top13a + top13b + top13c + (top13d + top13e + top13f +
     &        top13g + top13h + top13i)*3.d0
c
      call gengquad (5,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (6,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2 * dsiga * 2.d0
c
      aa = 1.d0/2.d0
      call genzquad (4,s,mel,mel,aa,aa,aa,aa, siga,dsiga)
      top4a = -elec2/(4.d0*ppi)**2 * siga
c
      call genzquad (4,s,mmu,mmu,aa,aa,aa,aa, siga,dsiga)
      top4b = -elec2/(4.d0*ppi)**2 * siga
c
      call genzquad (4,s,mta,mta,aa,aa,aa,aa, siga,dsiga)
      top4c = -elec2/(4.d0*ppi)**2 * siga
c
      call genzquad (4,s,mup,mup,aa,aa,aa,aa, siga,dsiga)
      top4d = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call genzquad (4,s,mdn,mdn,aa,aa,aa,aa, siga,dsiga)
      top4e = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      call genzquad (4,s,mch,mch,aa,aa,aa,aa, siga,dsiga)
      top4f = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call genzquad (4,s,mst,mst,aa,aa,aa,aa, siga,dsiga)
      top4g = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      call genzquad (4,s,mtt,mtt,aa,aa,aa,aa, siga,dsiga)
      top4h = -elec2/(4.d0*ppi)**2 * siga * 4.d0/3.d0
c
      call genzquad (4,s,mbb,mbb,aa,aa,aa,aa, siga,dsiga)
      top4i = -elec2/(4.d0*ppi)**2 * siga /3.d0
c
      top4 = top4a + top4b + top4c + top4d + top4e + top4f +
     &       top4g + top4h + top4i
c
      a = 0.5d0
      call gengquad (4,s,mcha(1),mcha(1),a,a,a,a, siga,dsiga)
      top8a = -elec2/(4.d0*ppi)**2 * dsiga
c
      call gengquad (4,s,mcha(2),mcha(2),a,a,a,a, siga,dsiga)
      top8b = -elec2/(4.d0*ppi)**2 * dsiga
c
      top8 = top8a + top8b
c
      call gengquad (7,s,mhp,a,a,a,a,a, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * 2.d0 * dsiga
c
      call gengquad (7,s,mmw,a,a,a,a,a, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2 * 2.d0 * dsiga
c
      call gengquad (7,s,melsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,melsr,a,a,a,a,a, siga2,dsiga2)
      top11a = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2)
c
      call gengquad (7,s,mmusl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mmusr,a,a,a,a,a, siga2,dsiga2)
      top11b = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2)
c
      call gengquad (7,s,mtasl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mtasr,a,a,a,a,a, siga2,dsiga2)
      top11c = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2)
c
      call gengquad (7,s,mupsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mupsr,a,a,a,a,a, siga2,dsiga2)
      top11d = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 4.d0/3.d0
c
      call gengquad (7,s,mdnsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mdnsr,a,a,a,a,a, siga2,dsiga2)
      top11e = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 1.d0/3.d0
c
      call gengquad (7,s,mchsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mchsr,a,a,a,a,a, siga2,dsiga2)
      top11f = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 4.d0/3.d0
c
      call gengquad (7,s,mstsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mstsr,a,a,a,a,a, siga2,dsiga2)
      top11g = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 1.d0/3.d0
c
      call gengquad (7,s,mtsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mtsr,a,a,a,a,a, siga2,dsiga2)
      top11h = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 4.d0/3.d0
c
      write(*,*) 'gamma-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call gengquad (7,s,mbsl,a,a,a,a,a, siga1,dsiga1)
      call gengquad (7,s,mbsr,a,a,a,a,a, siga2,dsiga2)
      top11i = -elec2/(4.d0*ppi)**2 * 2.d0 * (dsiga1+dsiga2) * 1.d0/3.d0
c
      top11 = top11a + top11b + top11c + top11d + top11e + top11f +
     &        top11g + top11h + top11i
c
      call gengquad (8,s,mmw,a,a,a,a,a, siga,dsiga)
      top12 = elec2/(4.d0*ppi)**2 * dsiga
c
      sigmagb = top1 + top2 + top3 + top6 + top7 + top9 + top10 + top12
      sigmags = top5 + top13 + top11
      sigmagc = top8
      sigmagf = top4
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' photon - self energie '
       write (*,*) ' H+ H-       = ', top1
       write (*,*) ' G+ G-       = ', top2
       write (*,*) ' W+ G-       = ', top3
       write (*,*) ' W+ W-       = ', top6
       write (*,*) ' gh+ gh-     = ', top7
       write (*,*) ' fer fer     = ', top4
       write (*,*) ' sfer sfer l = ', top5
       write (*,*) ' sfer sfer r = ', top13
       write (*,*) ' charg i,i   = ', top8
       write (*,*) ' H+          = ', top9
       write (*,*) ' G+          = ', top10
       write (*,*) ' sfer        = ', top11
       write (*,*) ' W+          = ', top12
       write (*,*) '               '
      endif
c
      return
      end


c=====================================================================
c
      subroutine sigmagz (s ,sgzb,sgzs,sgzf,sgzc)
c
c     selfenergy of gamma - z - mixing
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /susyset/ mu,mm,mp
      common /singl/ epsilon,muee,lambda
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2

      write(*,*) "Sigma-gZ should not be used for Higgs masses!!"
c
c  notation :
c     genzquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
c  boson loops
c
c o.k :
      call genzquad (1,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 =  elec2/(4.d0*ppi)**2 * (ccw2-ssw2)/(2.d0*ssw*ccw) * siga
c o.k :
      call genzquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 =  elec2/(4.d0*ppi)**2 * (ccw2-ssw2)/(2.d0*ssw*ccw) * siga
c o.k :
      call genzquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = -elec2/(4.d0*ppi)**2 * 2.d0 * ssw * mmw * mmz * siga
c o.k :
      call genzquad (5,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 =  elec2/(4.d0*ppi)**2 * ccw / ssw * siga
c o.k :
      call genzquad (6,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 =  2.d0*elec2/(4.d0*ppi)**2 * ccw / ssw * siga
c o.k :
      call genzquad (7,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 =  elec2 / (4.d0 * ppi)**2 * (ccw2 - ssw2)/(ssw * ccw) *siga
c o.k :
      call genzquad (7,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 =  elec2/(4.d0*ppi)**2 * (ccw2-ssw2)/(ssw*ccw) * siga
c o.k :
      call genzquad (8,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = -elec2/(4.d0*ppi)**2 * ccw/ssw * siga
c
      call genzquad (1,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5a = -elec2/(4.d0*ppi)**2*(-dcos(ang2)**2+2.d0*ssw2)
     $     /(2.d0*ssw*ccw)
     & * siga
c
      call genzquad (1,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5b = -elec2/(4.d0*ppi)**2*(-dcos(ang4)**2+2.d0*ssw2)
     $     /(2.d0*ssw*ccw)
     & * siga
c
      call genzquad (1,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5c = -elec2/(4.d0*ppi)**2*(-dcos(ang6)**2+2.d0*ssw2)
     $     /(2.d0*ssw*ccw)
     & * siga
c
      call genzquad (1,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5d = elec2/(4.d0*ppi)**2*2.d0/3.d0*(dcos(ang7)**2-4.d0/3.d0*
     & ssw2 )/        (2.d0*ssw*ccw) * siga
c
      call genzquad (1,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5e = -elec2/(4.d0*ppi)**2 /3.d0 * (-dcos(ang8)**2+2.d0/3.d0*
     & ssw2 )/   (2.d0*ssw*ccw) * siga
c
      call genzquad (1,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5f = elec2/(4.d0*ppi)**2*2.d0/3.d0*(dcos(ang9)**2-4.d0/3.d0*
     & ssw2 )/   (2.d0*ssw*ccw) * siga
c
      call genzquad (1,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5g = -elec2/(4.d0*ppi)**2 /3.d0 * (-dcos(ang10)**2+2.d0/3.d0*
     & ssw2 )/   (2.d0*ssw*ccw) * siga
c
      call genzquad (1,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5h = elec2/(4.d0*ppi)**2*2.d0/3.d0*(dcos(ang11)**2-4.d0/3.d0*
     & ssw2 )/   (2.d0*ssw*ccw) * siga
c
      write(*,*) 'gamma-Z-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call genzquad (1,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5i = -elec2/(4.d0*ppi)**2 /3.d0 * (-dcos(ang12)**2+2.d0/3.d0*
     & ssw2 )/   (2.d0*ssw*ccw) * siga
c
      top5 =  top5a + top5b + top5c + (top5d + top5e + top5f + top5g +
     &        top5h + top5i)*3.d0
c
      call genzquad (1,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6a =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang2)**2+2.d0*ssw2)
c
      call genzquad (1,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6b =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang4)**2+2.d0*ssw2)
c
      call genzquad (1,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6c =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw)  * siga *
     &         (-dsin(ang6)**2+2.d0*ssw2)
c
      call genzquad (1,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6d =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang7)**2+4.d0/3.d0*ssw2) *2.d0/3.d0
c
      call genzquad (1,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6e =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang8)**2+2.d0/3.d0*ssw2) *1.d0/3.d0
c
      call genzquad (1,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6f = elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &        (-dsin(ang9)**2+4.d0/3.d0*ssw2) *2.d0/3.d0
c
      call genzquad (1,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6g =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang10)**2+2.d0/3.d0*ssw2) *1.d0/3.d0
c
      call genzquad (1,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6h =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw) * siga *
     &         (-dsin(ang11)**2+4.d0/3.d0*ssw2) *2.d0/3.d0
c
      call genzquad (1,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6i =  elec2/(4.d0*ppi)**2 /(2.d0*ssw*ccw)  * siga *
     &         (-dsin(ang12)**2+2.d0/3.d0*ssw2) *1.d0/3.d0
c
      top6 = -(top6a + top6b + top6c + (top6d + top6e + top6f + top6g +
     &        top6h + top6i)*3.d0 )
c
      call genzquad (7,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11a =  elec2/(4.d0*ppi)**2 * 2.d0*(-dcos(ang2)**2/2.d0+ssw2)
     & /(ssw*ccw)* siga
c
      call genzquad (7,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11b = elec2/(4.d0*ppi)**2 * 2.d0*(-dcos(ang4)**2/2.d0+ssw2)
     & /(ssw*ccw)* siga
c
      call genzquad (7,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11c =  elec2/(4.d0*ppi)**2 * 2.d0*(-dcos(ang6)**2/2.d0+ssw2)
     & /(ssw*ccw)* siga
c
      call genzquad (7,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11d = -elec2/(4.d0*ppi)**2 * 4.d0/3.d0 * (dcos(ang7)**2/2.d0-
     &   2.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      call genzquad (7,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11e = -elec2/(4.d0*ppi)**2 * 4.d0/3.d0 * (dcos(ang9)**2/2.d0-
     &   2.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      call genzquad (7,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11f = -elec2/(4.d0*ppi)**2 * 4.d0/3.d0 * (dcos(ang11)**2/2.d0-
     &   2.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      call genzquad (7,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11g =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0 * (-dcos(ang8)**2/2.d0
     & +1.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      call genzquad (7,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11h =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0 * (-dcos(ang10)**2/2.d0
     & +1.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      write(*,*) 'gamma-Z-SE should not be used'
      write(*,*) 'Delta mb corrections not yet implemented here'
      call genzquad (7,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11i =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0 * (-dcos(ang12)**2/2.d0
     & +1.d0/3.d0*ssw2)/(ssw*ccw) * siga
c
      top11 = -(top11a + top11b + top11c + ( top11d + top11e + top11f +
     &         top11g + top11h + top11i)*3.d0)
c
      call genzquad (7,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14a =  elec2/(4.d0*ppi)**2 * 2.d0/(ssw*ccw)* siga *
     &          (-dsin(ang2)**2/2.d0+ssw2)
c
      call genzquad (7,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14b =  elec2/(4.d0*ppi)**2 * 2.d0/(ssw*ccw)* siga *
     &          (-dsin(ang4)**2/2.d0+ssw2)
c
      call genzquad (7,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14c =  elec2/(4.d0*ppi)**2 * 2.d0/(ssw*ccw)*  siga *
     &          (-dsin(ang6)**2/2.d0+ssw2)
c
      call genzquad (7,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14d = elec2/(4.d0*ppi)**2 * 4.d0/3.d0/(ssw*ccw) * siga *
     &         (-dsin(ang7)**2/2.d0+2.d0/3.d0*ssw2)
c
      call genzquad (7,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14e =  elec2/(4.d0*ppi)**2 * 4.d0/3.d0/(ssw*ccw) * siga *
     &         (-dsin(ang9)**2/2.d0+2.d0/3.d0*ssw2)
c
      call genzquad (7,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14f =  elec2/(4.d0*ppi)**2 * 4.d0/3.d0/(ssw*ccw)* siga *
     &         (-dsin(ang11)**2/2.d0+2.d0/3.d0*ssw2)
c
      call genzquad (7,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14g =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0/(ssw*ccw)* siga *
     &          (-dsin(ang8)**2/2.d0+1.d0/3.d0*ssw2)
c
      call genzquad (7,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14h =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0/(ssw*ccw)* siga *
     &          (-dsin(ang10)**2/2.d0+1.d0/3.d0*ssw2)
c
      call genzquad (7,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14i =  elec2/(4.d0*ppi)**2 * 2.d0/3.d0/(ssw*ccw)* siga *
     &          (-dsin(ang12)**2/2.d0+1.d0/3.d0*ssw2)
c
      top14 = -(top14a + top14b + top14c + (top14d + top14e + top14f +
     &         top14g + top14h + top14i) * 3.d0)
c
c fermion loops :
c
      a = 0.5d0
      b = 0.5d0
      at = (-0.5d0 + 2.d0 * ssw2 + 0.5d0) / (2.d0 * ssw * ccw)
      bt = (-0.5d0 + 2.d0 * ssw2 - 0.5d0) / (2.d0 * ssw * ccw)
c top4 o.k :
      call genzquad (4,s,mel,mel,a,at,b,bt, siga,dsiga)
      top4a = elec2 / (4.d0 * ppi)**2 * ( -siga )
c
      call genzquad (4,s,mmu,mmu,a,at,b,bt, siga,dsiga)
      top4b = elec2 / (4.d0 * ppi)**2 * ( -siga )
c
      call genzquad (4,s,mta,mta,a,at,b,bt, siga,dsiga)
      top4c = elec2 / (4.d0 * ppi)**2 * ( -siga )
c
      at = (0.5d0 - 4.d0/3.d0 * ssw2 - 0.5d0) / (2.d0 * ssw * ccw)
      bt = (0.5d0 - 4.d0/3.d0 * ssw2 + 0.5d0) / (2.d0 * ssw * ccw)
c
      call genzquad (4,s,mup,mup,a,at,b,bt, siga,dsiga)
      top4d = elec2 / (4.d0 * ppi)**2 * ( 2.d0/3.d0 * siga ) * 3.d0
c
      call genzquad (4,s,mch,mch,a,at,b,bt, siga,dsiga)
      top4e = elec2 / (4.d0 * ppi)**2 * ( 2.d0/3.d0 * siga ) * 3.d0
c
      call genzquad (4,s,mtt,mtt,a,at,b,bt, siga,dsiga)
      top4f = elec2 / (4.d0 * ppi)**2 * ( 2.d0/3.d0 * siga ) * 3.d0
c
      at = (-0.5d0 + 2.d0/3.d0 * ssw2 + 0.5d0) / (2.d0 * ssw * ccw)
      bt = (-0.5d0 + 2.d0/3.d0 * ssw2 - 0.5d0) / (2.d0 * ssw * ccw)
c
      call genzquad (4,s,mdn,mdn,a,at,b,bt, siga,dsiga)
      top4g = elec2 / (4.d0 * ppi)**2 * ( -1.d0/3.d0 * siga ) * 3.d0
c
      call genzquad (4,s,mst,mst,a,at,b,bt, siga,dsiga)
      top4h = elec2 / (4.d0 * ppi)**2 * ( -1.d0/3.d0 * siga ) * 3.d0
c
      call genzquad (4,s,mbb,mbb,a,at,b,bt, siga,dsiga)
      top4i = elec2 / (4.d0 * ppi)**2 * ( -1.d0/3.d0 * siga ) * 3.d0
c
      top4 = (top4a + top4b + top4c + top4d + top4e + top4f + top4g +
     &       top4h + top4i)/2.d0
c
      c = 0.5d0
      a = -vmix(1,1)*vmix(1,1) - vmix(1,2)*vmix(1,2)/2.d0 + ssw2
      b = -umix(1,1)*umix(1,1) - umix(1,2)*umix(1,2)/2.d0 + ssw2
      call genzquad (4,s,mcha(1),mcha(1),c,a,c,b, siga,dsiga)
      top13a = -elec2/(4.d0*ppi)**2 * siga / (2.d0*ssw*ccw)
c
      a = -vmix(2,1)*vmix(2,1) - vmix(2,2)*vmix(2,2)/2.d0 + ssw2
      b = -umix(2,1)*umix(2,1) - umix(2,2)*umix(2,2)/2.d0 + ssw2
      call genzquad (4,s,mcha(2),mcha(2),c,a,c,b, siga,dsiga)
      top13b = -elec2/(4.d0*ppi)**2 * siga  / (2.d0*ssw*ccw)
c
      top13 = top13a + top13b
c
      sgzb = top1 + top2 + top3 + top7 + top8 +
     &           top9 + top10 + top12
      sgzf = top4
      sgzs = top5 + top6 + top11 + top14
      sgzc = top13
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' Gamma - Z - mixing : ', real(dsqrt(s))
       write (*,*) ' H+ H-       = ', top1
       write (*,*) ' G+ G-       = ', top2
       write (*,*) ' W+ G-       = ', top3
       write (*,*) ' W+ W-       = ', top7
       write (*,*) ' gh+ gh-     = ', top8
       write (*,*) ' fer fer     = ', top4
       write (*,*) ' sfer sfer l = ', top5
       write (*,*) ' sfer sfer r = ', top6
       write (*,*) ' cha  i,i    = ', top13
       write (*,*) '   four point interactions : '
       write (*,*) ' H+          = ', top9
       write (*,*) ' G+          = ', top10
       write (*,*) ' W+          = ', top12
       write (*,*) ' sfer    ,l  = ', top11
       write (*,*) ' sfer    ,r  = ', top14
       write (*,*) '    '
      endif
c
      return
      end
c=====================================================================
 
      subroutine sigmahh (s, sigmahhb,sigmahhs,sigmahhf,sigmahhc,
     &                       sigmahht)
c
c     selfenergy of heavy scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2/(2.d0*ssw2)*siga*dsin(beta-alpha)**2
c
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  siga
c
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  siga
c
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga 
     $     * dcos(beta-alpha)**2
c
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 * siga 
     $     * dcos(beta-alpha)**2
c
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * siga
     &      * dcos(beta-alpha)**2 / 2.d0
c
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dcos(beta-alpha) -
     &      mmz/(2.d0*ccw) * dcos(2.d0*beta) * dcos(beta+alpha))**2
     &      * siga
c
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2 * 9.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        siga * dcos(2.d0*alpha)**2 * dcos(beta+alpha)**2
c
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta))**2
c
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &       (dcos(2.d0*alpha) * dcos(beta+alpha) - 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta))**2
c
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &         dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * siga *
     &         dsin(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga * (mmw *
     &         dsin(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dcos
     &         (alpha+beta))**2
c
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmw**2 *
     &         dcos(beta-alpha)**2 / 2.d0
c
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmz**2 *
     &         dcos(beta-alpha)**2 / (4.d0*ccw2)
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mup/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11 * dcos(ang7)
     &    **2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*ang7))**2 * siga1 +  
     &   2.d0*( a12*dcos(2.d0*ang7)+(a22-a11)*dsin(ang7)*dcos(ang7) )
     &   **2 * siga2 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2 - a12*
     &   dsin(2.d0*ang7))**2 * siga3 )
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mch/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11 * dcos(ang9)
     &    **2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*ang9))**2 * siga1 +  
     &   2.d0*( a12*dcos(2.d0*ang9)+(a22-a11)*dsin(ang9)*dcos(ang9) )
     &   **2 * siga2 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2 - a12*
     &   dsin(2.d0*ang9))**2 * siga3 )
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &      (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
c      write(*,*) 'top23c: ', real(siga1), real(siga2), real(siga3)
c xxxyyyzzz
c      siga3 = 0d0

c      top23c = real(-elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0 * 
c     $              (a11*dsin(ang11)**2 + a22*dcos(ang11)**2 
c     $               - a12 * dsin(2.d0*ang11))**2 * siga3)
c      write(*,*) 'top23c: ', top23c 
      top23c = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11 * dcos(ang11)
     &   **2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*ang11))**2 * siga1 +
     &  2.d0 *(a12*dcos(2.d0*ang11)+(a22-a11)*dsin(ang11)*dcos(ang11))
     &   **2 * siga2 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2 - a12*
     &   dsin(2.d0*ang11))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mdn/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23d = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11 * dcos(ang8)
     &    **2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*ang8))**2 * siga1 +  
     &   2.d0*( a12*dcos(2.d0*ang8)+(a22-a11)*dsin(ang8)*dcos(ang8) )
     &   **2 * siga2 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2 - a12*
     &   dsin(2.d0*ang8))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mst/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang10)
     &    **2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*ang10))**2 * siga1 +
     &   2.d0*( a12*dcos(2.d0*ang10)+(a22-a11)*dsin(ang10)*dcos(ang10))
     &   **2 * siga2 + (a11*dsin(ang10)**2 + a22*dcos(ang10)**2 - a12*
     &   dsin(2.d0*ang10))**2 * siga3 ) 
c
      if (delmbresum.eq.1) then
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mbb/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0*((a11*dcos(ang12)
     &    **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2 * siga1 +
     &   2.d0*( a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
     &   **2 * siga2 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
     &   dsin(2.d0*ang12))**2 * siga3 )
      else
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mbbdmb**2 
     $        * dcos(alpha)/(mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mbbdmb**2 
     $        * dcos(alpha)/(mmw*dcos(beta)) )
      a12 =  -mbbdmb/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,msb1dmb,msb1dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (2,s,msb1dmb,msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      call genhquad (2,s,msb2dmb,msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 
     $     * 3.d0*((a11*dcos(tsbdmb)**2 + a22*dsin(tsbdmb)**2 
     $     + a12*dsin(2.d0*tsbdmb))**2 * siga1 +
     &   2.d0*( a12*dcos(2.d0*tsbdmb)+(a22-a11)*dsin(tsbdmb)
     $     *dcos(tsbdmb))**2 * siga2 + (a11*dsin(tsbdmb)**2 + 
     $     a22*dcos(tsbdmb)**2 - a12*
     &   dsin(2.d0*tsbdmb))**2 * siga3 )
      endif
c      write(*,*) 'HH-SE: Sbottom loops:'
c      write(*,*) a11, a22, a12
c      write(*,*) (a11*dcos(ang12)
c     &    **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2,
c     $    2.d0*( a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
c     &    **2, 
c     $    (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
c     &    dsin(2.d0*ang12))**2
c      write(*,*) siga1, siga2, siga3
c      write(*,*) (a11*dcos(ang12)
c     &    **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2 * siga1,
c     $    2.d0*( a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
c     &    **2 * siga2,
c     $    (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
c     &    dsin(2.d0*ang12))**2 * siga3
cc
      a11 =  (mmz/ccw* (0.5d0
     &      -ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mel/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang2)
     &    **2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*ang2))**2 * siga1 +     
     &   2.d0*( a12*dcos(2.d0*ang2) + (a22-a11)*dsin(ang2)*dcos(ang2) )
     &   **2 * siga2 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2 - a12*
     &   dsin(2.d0*ang2))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mmu/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang4)
     &    **2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*ang4))**2 * siga1 +     
     &   2.d0*( a12*dcos(2.d0*ang4) + (a22-a11)*dsin(ang4)*dcos(ang4) )
     &   **2 * siga2 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2 - a12*
     &   dsin(2.d0*ang4))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mta/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang6)
     &    **2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*ang6))**2 * siga1 +     
     &   2.d0*( a12*dcos(2.d0*ang6)+ (a22-a11)*dsin(ang6)*dcos(ang6) )
     &   **2 * siga2 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2 - a12*
     &   dsin(2.d0*ang6))**2 * siga3 )
c
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta)  )**2
c
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta)  )**2
c
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta) )**2
c
      if (selec.ge.3) then
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
      elseif (selec.eq.2) then
         top23 = top23c + top23f
      elseif (selec.eq.1) then
         top23 = top23c
      else
         write(*,*) "Error in Sigma-H: selec out or range"
      endif
c      write(*,*) 'HH-SE: Sfermion contributions:'
c      write(*,*) top23a, top23b, top23c, top23d, top23e, top23f,
c     $           top23g, top23h, top23i, top23j, top23k, top23l


c
      call genhquad (5,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top29 = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2) * siga
c
      call genhquad (5,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top30 = -elec2/(4.d0*ppi)**2 /(4.d0*ccw2*ssw2) * siga
c
      call genhquad (6,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top31 = elec2/(4.d0*ppi)**2 * (1.d0 - dsin(2.d0*beta) *
     &        dsin(2.d0*alpha) + ssw2/ccw2 * dcos(2.d0*beta)*dcos(
     &        2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top32 = elec2/(4.d0*ppi)**2 * (1.d0 + dsin(2.d0 *
     &   beta)* dsin(2.d0*alpha) - ssw2/ccw2 * dcos(2.d0*beta)*dcos(
     &   2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mhh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top34 = elec2/(4.d0*ppi)**2 * 3.d0 * dcos(2.d0 *
     &       alpha)**2/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mlh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top33 = elec2/(4.d0*ppi)**2 * (3.d0 * dsin(2.d0 *
     &       alpha)**2 - 1.d0)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top35 = elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &        dcos(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,maa,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top36 = -elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &        dcos(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      a1 =  (-(0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &      alpha) - mup**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
      a2 =  ( -2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &      alpha) - mup**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37a = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang7)**2 + a2*dsin(ang7)**2 ) * siga1 +
     &  ( a1*dsin(ang7)**2 + a2*dcos(ang7)**2 ) * siga2  )
c
      a1 =  (-(0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &      alpha) - mch**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
      a2 =  ( -2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &      alpha) - mch**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37b = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang9)**2 + a2*dsin(ang9)**2 ) * siga1 +
     &  ( a1*dsin(ang9)**2 + a2*dcos(ang9)**2 ) * siga2  )
c
      a1 =  (-(0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &      alpha) - mtt**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
      a2 =  ( -2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &      alpha) - mtt**2 / mmw2 * dsin(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37c = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang11)**2 + a2*dsin(ang11)**2 ) * siga1 +
     &  ( a1*dsin(ang11)**2 + a2*dcos(ang11)**2 ) * siga2  )
c      if (dabs(s-100.001**2).le.1) then
c         write(*,*) "H-SE: top: ", siga1, siga2
c         write(*,*) dsin(alpha)**2, dsin(beta)**2
c         write(*,*) top37c
c      endif
c
      a1 =  (-(-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mdn**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mdn**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37d = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang8)**2 + a2*dsin(ang8)**2 ) * siga1 +
     &  ( a1*dsin(ang8)**2 + a2*dcos(ang8)**2 ) * siga2  )
c
      a1 =  (-(-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mst**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mst**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37e = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang10)**2 + a2*dsin(ang10)**2 ) * siga1 +
     &  ( a1*dsin(ang10)**2 + a2*dcos(ang10)**2 ) * siga2  )
c
      if (delmbresum.eq.1) then
      a1 =  (-(-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mbb**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mbb**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang12)**2 + a2*dsin(ang12)**2 ) * siga1 +
     &  ( a1*dsin(ang12)**2 + a2*dcos(ang12)**2 ) * siga2  )
      else
      a1 =  (-(-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mbbdmb**2 / mmw2 
     $        * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mbbdmb**2 / mmw2 
     $        * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,msb1dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (6,s,msb2dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(tsbdmb)**2 + a2*dsin(tsbdmb)**2 ) * siga1 +
     &  ( a1*dsin(tsbdmb)**2 + a2*dcos(tsbdmb)**2 ) * siga2  )
      endif
c      if (dabs(s-100.001**2).le.1d0) then
c         write(*,*) "H-SE: bottom: ", siga1, siga2
c         write(*,*) dcos(alpha)**2, dcos(beta)**2
c         write(*,*) top37f
c      endif
c
      a1 =  (-(-0.5d0 + 1.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mel**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mel**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37g = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang2)**2 + a2*dsin(ang2)**2 ) * siga1 +
     &  ( a1*dsin(ang2)**2 + a2*dcos(ang2)**2 ) * siga2  )
c
      a1 =  (-(-0.5d0 + 1.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mmu**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mmu**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37h = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang4)**2 + a2*dsin(ang4)**2 ) * siga1 +
     &  ( a1*dsin(ang4)**2 + a2*dcos(ang4)**2 ) * siga2  )
c
      a1 =  (-(-0.5d0 + 1.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mta**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2)
      a2 =  ( 1.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mta**2 / mmw2 * dcos(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37i = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang6)**2 + a2*dsin(ang6)**2 ) * siga1 +
     &  ( a1*dsin(ang6)**2 + a2*dcos(ang6)**2 ) * siga2  )
c
      call genhquad (6,s,mvesl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37j = elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      call genhquad (6,s,mvmsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37k = elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      call genhquad (6,s,mvtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37l = elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      if (selec.ge.3) then
      top37 = top37a + top37b + top37c + top37d + top37e + top37l +
     &        top37f + top37g + top37h + top37i + top37j + top37k
      elseif (selec.eq.2) then
         top37 = top37c + top37f
      elseif (selec.eq.1) then
         top37 = top37c
      else
         write(*,*) "Error in Sigma-H: selec out or range"
      endif
c
c fermion loops
c
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mtt**2 * 3.d0
c      write(*,*) "H-SE: top: ", s
c      write(*,*) dsin(alpha)**2, dsin(beta)**2, siga, mtt
c      write(*,*) top17c
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mst**2 * 3.d0
c
      if (delmbresum.eq.1) then
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mbb**2 * 3.d0
      else
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) 
     $     * siga * 3.d0 * (mbbdmb * (dcos(alpha)/dcos(beta) 
     $                      + dmb * dsin(alpha)/dsin(beta)))**2
      endif
c      write(*,*) "H-SE: bottom: ", s
c      write(*,*) dcos(alpha)**2, dcos(beta)**2, siga, mbb
c      write(*,*) top17f
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mta**2
c
      if (selec.ge.3) then
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
      elseif (selec.eq.2) then
         top17 = top17c + top17f
      elseif (selec.eq.1) then
         top17 = top17c
      else
         write(*,*) "Error in Sigma-H: selec out or range"
      endif
c
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dcos(alpha) +
     &      vmix(j,2)*umix(ii,1)*dsin(alpha) )/dsqrt(2.d0)
      b =  (vmix(ii,1)*umix(j,2)*dcos(alpha) +
     &      vmix(ii,2)*umix(j,1)*dsin(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,b,b,a, siga,dsiga)
      top21h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
      top21 = top21 + top21h
621   continue
620   continue
c
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dcos(alpha) - sm * dsin(alpha)
      b = a
      call genhquad (4,s,mne(ii),mne(j),a,a,b,b, siga,dsiga)
      top22h = -elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * siga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' H0 - selfenergy : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   four point interactions : '
       write (*,*) ' W+      = ', top29
       write (*,*) ' Z0      = ', top30
       write (*,*) ' G+      = ', top31
       write (*,*) ' H+      = ', top32
       write (*,*) ' h0      = ', top33
       write (*,*) ' H0      = ', top34
       write (*,*) ' G0      = ', top35
       write (*,*) ' A       = ', top36
       write (*,*) ' sfer    = ', top37
       write (*,*) '   '
      endif
c
      sigmahhb = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &           top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &           top18 + top20 + top29 + top30 + top31 + top32 + top33 +
     &           top34 + top35 + top36
      sigmahhs = top23 + top37
c      write(*,*) 'subroutine sigmahh: ', top23, top37
      sigmahhf = top17
      sigmahhc = top21 + top22
      sigmahht = top17c + top23c + top37c
c
      return
      end
c
c -------------------------------------------------------------------
c
      subroutine dsigmahh (s, dsigmahhb,dsigmahhs,dsigmahhf,dsigmahhc,
     &                        dsigmahht)
c
c     derivative of selfenergy of heavy scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "dSigma-H should not be used for Higgs masses!!"
      write(*,*) 'Delta mb corrections not yet implemented here'

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) * dsiga 
     $     * dsin(beta-alpha)**2
c
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2/(4.d0*ppi)**2 * dsin(beta-alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  dsiga
c
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  dsiga
c
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) * dsiga 
     $     * dcos(beta-alpha)**2
c
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 * dsiga * dcos(beta-alpha)**2
c
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * dsiga
     &      * dcos(beta-alpha)**2 / 2.d0
c
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dcos(beta-alpha) -
     &      mmz/(2.d0*ccw) * dcos(2.d0*beta) * dcos(beta+alpha))**2
     &      * dsiga
c
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2 * 9.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        dsiga * dcos(2.d0*alpha)**2 * dcos(beta+alpha)**2
c
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta))**2
c
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &       (dcos(2.d0*alpha) * dcos(beta+alpha) - 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta))**2
c
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &         dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &         dsin(2.d0*beta)**2 * dcos(alpha+beta)**2
c
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * dsiga * (mmw *
     &         dsin(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dcos
     &         (alpha+beta))**2
c
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmw**2 *
     &         dcos(beta-alpha)**2 / 2.d0
c
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmz**2 *
     &         dcos(beta-alpha)**2 / (4.d0*ccw2)
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mup/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang7)
     &    **2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*ang7))**2 * dsiga1 +
     &   2.d0*( a12*dcos(2.d0*ang7)+(a22-a11)*dsin(ang7)*dcos(ang7) )
     &   **2 * dsiga2 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2 - a12*
     &   dsin(2.d0*ang7))**2 * dsiga3 )
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mch/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang9)
     &    **2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*ang9))**2 * dsiga1 +
     &   2.d0*( a12*dcos(2.d0*ang9)+(a22-a11)*dsin(ang9)*dcos(ang9) )
     &   **2 * dsiga2 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2 - a12*
     &   dsin(2.d0*ang9))**2 * dsiga3 )
c
      a11 =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22 =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) + mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12 =  mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &      (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23c = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang11)
     &   **2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*ang11))**2 * dsiga1 +
     &  2.d0 *(a12*dcos(2.d0*ang11)+(a22-a11)*dsin(ang11)*dcos(ang11))
     &   **2 * dsiga2 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2-a12*
     &   dsin(2.d0*ang11))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mdn/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23d = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11 * dcos(ang8)
     &    **2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*ang8))**2 * dsiga1 +     
     &   2.d0*( a12*dcos(2.d0*ang8)+(a22-a11)*dsin(ang8)*dcos(ang8) )
     &   **2 * dsiga2 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2 - a12*
     &   dsin(2.d0*ang8))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mst/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang10)
     &    **2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*ang10))**2 * dsiga1 +
     &   2.d0*( a12*dcos(2.d0*ang10)+(a22-a11)*dsin(ang10)*dcos(ang10))
     &   **2 * dsiga2 + (a11*dsin(ang10)**2 + a22*dcos(ang10)**2 - a12*
     &   dsin(2.d0*ang10))**2 * dsiga3 ) 
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dcos(alpha+beta) + mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mbb/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0*((a11 * dcos(ang12)
     &    **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2 * dsiga1 +     
     &   2.d0*( a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
     &   **2 * dsiga2 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
     &   dsin(2.d0*ang12))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mel/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang2)
     &    **2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*ang2))**2 * dsiga1 +     
     &   2.d0*( a12*dcos(2.d0*ang2) + (a22-a11)*dsin(ang2)*dcos(ang2) )
     &   **2 * dsiga2 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2 - a12*
     &   dsin(2.d0*ang2))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mmu/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang4)
     &    **2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*ang4))**2 * dsiga1 +     
     &   2.d0*( a12*dcos(2.d0*ang4) + (a22-a11)*dsin(ang4)*dcos(ang4) )
     &   **2 * dsiga2 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2 - a12*
     &   dsin(2.d0*ang4))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22 =  -(mmz/ccw *
     &      (-1.d0) * ssw2*dcos(alpha+beta) + mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  -mta/(2.d0*
     &       mmw * dcos(beta)) * 
     &       (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2 / (4.d0 * ppi)**2 /ssw2 *  ((a11 * dcos(ang6)
     &    **2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*ang6))**2 * dsiga1 +     
     &   2.d0*( a12*dcos(2.d0*ang6)+ (a22-a11)*dsin(ang6)*dcos(ang6) )
     &   **2 * dsiga2 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2 - a12*
     &   dsin(2.d0*ang6))**2 * dsiga3 )
c
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta)  )**2
c
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta)  )**2
c
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (-mmz/ccw*(0.5d0
     &         )*dcos(alpha+beta) )**2
c
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
c
c fermion loops
c
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mtt**2 * 3.d0
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mst**2 * 3.d0
c
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mbb**2 * 3.d0
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mta**2
c
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
c
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dcos(alpha) +
     &      vmix(j,2)*umix(ii,1)*dsin(alpha) )/dsqrt(2.d0)
      b =  (vmix(ii,1)*umix(j,2)*dcos(alpha) +
     &      vmix(ii,2)*umix(j,1)*dsin(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,b,b,a, siga,dsiga)
      top21h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * dsiga
      top21 = top21 + top21h
621   continue
620   continue
c
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dcos(alpha) - sm * dsin(alpha)
      b = a
      call genhquad (4,s,mne(ii),mne(j),a,a,b,b, siga,dsiga)
      top22h = -elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * dsiga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' H0 - derivative - selfenergy : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   '
      endif
c
      dsigmahhb =top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &            top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &            top18 + top20
      dsigmahhs = top23
      dsigmahhf = top17
      dsigmahhc = top21 + top22
      dsigmahht = top17c + top23c
c
      return
      end
c=====================================================================
      subroutine genhquad (typ,q2,m1,m2,a,at,b,bt, siga,dsiga)
c
      implicit double precision (a-z)
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
c     common /renpara/xo,zo,mgll
c
c                    ________                              .
c                   /   m1   \                          .     .  m1
c           _______/          \_______                  .     .
c        q         \          /        q                  .  .
c                   \________/                q ____________.__________q
c                       m2
c
c      if (q2.le.1.d-3) then
c       call bquer2(1.d-3,dabs(m2),dabs(m1),
c     &            b0,b1,pb0,pb1)
c       else
c       call bquer2(q2,dabs(m2),dabs(m1),
c     &            b0,b1,pb0,pb1)
c      endif
c      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
c      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
c     &      + 0.5d0) + b1
c
      if (typ.eq.1) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
c
       siga  = -(q2 * (b0 - 2.d0 * b1) + m2**2 * b0 + aa(dabs(m1)))
       dsiga = - ( (b0 - 2.d0 * b1) + q2 *
     &            (pb0 - 2.d0 * pb1) + m2**2 * pb0 )
c      siga  = -(4.d0*q2/epsilon + 2.d0*m2**2/epsilon + 2.d0*
c    &           m1**2/epsilon)
c      dsiga = - 4.d0/epsilon
      else
      if (typ.eq.2) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
c      if ((q2.gt.42d0**2).and.(q2.le.46d0**2).and.(m1.eq.m2).and.
c     $    (m1.gt.20d0).and.(m1.le.22d0)) then
c         write(*,*) 'b0: ', b0
c      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       siga  = -b0
       dsiga = -pb0
c      siga = -2.d0/epsilon
c      dsiga = 0.d0
      else
      if (typ.eq.3) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       siga  =  4.d0 * ( 2.d0 * m1**2 * b0 + aa(dabs(m1)) + q2 * b1 )
       dsiga =  4.d0 * ( 2.d0 * m1**2 * pb0 + b1 + q2 * pb1 )
c      siga  =  4.d0 * ( 4.d0 * m1**2 - q2 + 2.d0 * m1**2 )/epsilon
c      dsiga = 0.d0
      else
      if (typ.eq.4) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       siga  = 8.d0 * ((at * a + bt * b) * m1 * m2 * b0
     &     + (at * b + a * bt) * ( q2 *
     &     b1 + aa(dabs(m1)) + m2**2 * b0 ))
       dsiga = 8.d0 * ((at * a + bt * b) * m1 * m2 * pb0
     &     + (at * b + a * bt) * ( b1 + q2 * pb1
     &        + m2**2 * pb0))
c      siga  =  8.d0 * ((at * b + bt * a) * ( 2.d0 * m1**2 + 2.d0 *
c    &   m2**2 - q2 ) + (a * at + b * bt) * 2.d0 * m1 * m2  ) /epsilon
c      dsiga = -8.d0 * ((at * b + a * bt))/epsilon
      else
      if (typ.eq.5) then
       siga  = - (4.d0 * aa(dabs(m1)) - 2.d0 * m1**2)
       dsiga =  0.d0
c      siga  = - (8.d0 * m1**2 )/epsilon
      else
      if (typ.eq.6) then
       siga  = aa(dabs(m1))
       dsiga = 0.d0
c      siga  = 2.d0*m1**2/epsilon
      else
      if (typ.eq.8) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       siga  = - (4.d0 * b0 - 2.d0)
       dsiga = - 4.d0 * pb0
c      siga  = - 8.d0/epsilon
c      dsiga = 0.d0
      else
      if (typ.eq.9) then
c
      if (q2.le.1.d-3) then
       call bquer2(1.d-3,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
       else
       call bquer2(q2,dabs(m2),dabs(m1),
     &            b0,b1,pb0,pb1)
      endif
      b0 = delta(epsilon,muee,1.d0) - dlog (dabs(m1*m2)) + b0
      b1 = -0.5d0 * (delta(epsilon,muee,1.d0) - dlog(m1**2)
     &      + 0.5d0) + b1
       siga  = b0
       dsiga = pb0
c      siga  = 2.d0/epsilon
c      dsiga = 0.d0
      else
       write (*,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
      endif
      endif
      endif
      endif
c
      return
      end
c
c ---------------------------------------------------------
c
      subroutine sigmalh (s, sigmalhb,sigmalhs,sigmalhf,sigmalhc,
     &                       sigmalht)
c
c     selfenergy of light scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
c      write(*,*) "vor top1"
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2/(2.d0*ssw2)*siga * dcos(beta-alpha)**2
c
c      write(*,*) "vor top2"
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  siga
c
c      write(*,*) "vor top3"
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dsin(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  siga
c
c      write(*,*) "vor top4"
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) 
     $     * siga * dsin(beta-alpha)**2
c
c      write(*,*) "vor top5"
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 
     $     * siga * dsin(beta-alpha)**2
c
c      write(*,*) "vor top6"
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * siga
     &      * dsin(beta-alpha)**2 / 2.d0
c
c      write(*,*) "vor top7"
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dsin(beta-alpha) +
     &      mmz/(2.d0*ccw) * dcos(2.d0*beta) * dsin(beta+alpha))**2
     &      * siga
c
c      write(*,*) "vor top8"
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * 9.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        siga * dcos(2.d0*alpha)**2 * dsin(beta+alpha)**2
c
c      write(*,*) "vor top9"
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &       (dcos(2.d0*alpha) * dcos(beta+alpha) - 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta))**2
c
c      write(*,*) "vor top10"
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta))**2
c
c      write(*,*) "vor top11"
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
c      write(*,*) "vor top12"
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &         dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
c      write(*,*) "vor top13"
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
c      write(*,*) "vor top14"
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * siga *
     &         dsin(2.d0*beta)**2 * dsin(alpha+beta)**2
c
c      write(*,*) "vor top15"
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga * (mmw *
     &         dcos(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dsin
     &         (alpha+beta))**2
c
c      write(*,*) "vor top18"
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmw**2 *
     &         dsin(beta-alpha)**2 / 2.d0
c
c      write(*,*) "vor top20"
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmz**2 *
     &         dsin(beta-alpha)**2 / (4.d0*ccw2)
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mup/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang7)
     &    **2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*ang7))**2 * siga1 +
     &   2.d0*( a12*dcos(2.d0*ang7)+(a22-a11)*dsin(ang7)*dcos(ang7))
     &   **2 * siga2 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2 - a12*
     &   dsin(2.d0*ang7))**2 * siga3 )
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mch/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang9)
     &    **2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*ang9))**2 * siga1 +
     &   2.d0*(a12*dcos(2.d0*ang9)+(a22-a11)*dsin(ang9)*dcos(ang9))
     &   **2 * siga2 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2 - a12*
     &   dsin(2.d0*ang9))**2 * siga3 )
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mtt/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23c = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang11)
     &  **2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*ang11))**2 * siga1 +     
     &  2.d0*(a12*dcos(2.d0*ang11) + (a22-a11)*dsin(ang11)*dcos(ang11))
     &   **2 * siga2 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2 - a12*
     &   dsin(2.d0*ang11))**2 * siga3 )
c
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mdn/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 = -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1) 
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2) 
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3) 
      top23d = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang8)
     &  **2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*ang8))**2 * siga1 +     
     &  2.d0*( a12*dcos(2.d0*ang8)+(a22-a11)*dsin(ang8)*dcos(ang8))
     &   **2 * siga2 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2 - a12*
     &   dsin(2.d0*ang8))**2 * siga3 ) 
c
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mst/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang10)
     &  **2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*ang10))**2 * siga1 +
     & 2.d0*(a12*dcos(2.d0*ang10)+(a22-a11)*dsin(ang10)*dcos(ang10))
     &   **2 * siga2 + (a11*dsin(ang10)**2 + a22*dcos(ang10)**2 - a12*
     &   dsin(2.d0*ang10))**2 * siga3 )
c
      if (delmbresum.eq.1) then
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mbb/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang12)
     &  **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2 * siga1 + 
     &  2.d0*(a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
     &   **2 * siga2 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
     &   dsin(2.d0*ang12))**2 * siga3 )
      else
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) 
     $        - mbbdmb**2 * dsin(alpha)/(mmw*dcos(beta)) )
      a12 = mbbdmb/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) 
     $     + mbbdmb**2 * dsin(alpha)/(mmw*dcos(beta)) )
c
      call genhquad (2,s,msb1dmb, msb1dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (2,s,msb1dmb, msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      call genhquad (2,s,msb2dmb, msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 
     $     *((a11*dcos(tsbdmb)**2 + a22*dsin(tsbdmb)**2 
     $     + a12*dsin(2.d0*tsbdmb))**2 * siga1 + 
     &  2.d0*(a12*dcos(2.d0*tsbdmb)+(a22-a11)*dsin(tsbdmb)
     $     *dcos(tsbdmb))**2 * siga2 + (a11*dsin(tsbdmb)**2 
     $     + a22*dcos(tsbdmb)**2 - a12*
     &   dsin(2.d0*tsbdmb))**2 * siga3 )
      endif
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mel/(2.d0*
     &         mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang2)
     &  **2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*ang2))**2 * siga1 +     
     &  2.d0*(a12*dcos(2.d0*ang2)+(a22-a11)*dsin(ang2)*dcos(ang2))
     &  **2 * siga2 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2 - a12*
     &  dsin(2.d0*ang2))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mmu/(2.d0*
     &         mmw * dcos(beta)) *
     &      (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang4)
     &  **2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*ang4))**2 * siga1 +     
     &  2.d0*( a12*dcos(2.d0*ang4)+(a22-a11)*dsin(ang4)*dcos(ang4))
     &  **2 * siga2 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2 - a12*
     &  dsin(2.d0*ang4))**2 * siga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mta/(2.d0*
     &         mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang6)
     &  **2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*ang6))**2 * siga1 +     
     &  2.d0*( a12*dcos(2.d0*ang6)+(a22-a11)*dsin(ang6)*dcos(ang6))
     &  **2 * siga2 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2 - a12*
     &  dsin(2.d0*ang6))**2 * siga3 )
c
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )**2
c
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )**2
c
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta)  )**2
c
      if (selec.ge.3) then
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
      elseif (selec.eq.2) then
         top23 = top23c + top23f
      elseif (selec.eq.1) then
         top23 = top23c
      else
         write(*,*) "Error in Sigma-A: selec out or range"
      endif

c
      call genhquad (5,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top29 = -elec2/(4.d0*ppi)**2 /(2.d0*ssw2) * siga
c
      call genhquad (5,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top30 = -elec2/(4.d0*ppi)**2 /(4.d0*ccw2*ssw2) * siga
c
      call genhquad (6,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top31 = elec2/(4.d0*ppi)**2 * (1.d0 + dsin(2.d0*beta) *
     &        dsin(2.d0*alpha) - ssw2/ccw2 * dcos(2.d0*beta)*dcos(
     &        2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top32 = elec2/(4.d0*ppi)**2 * (1.d0 - dsin(2.d0 *
     &   beta)* dsin(2.d0*alpha) + ssw2/ccw2 * dcos(2.d0*beta)*dcos(
     &   2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mlh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top33 = elec2/(4.d0*ppi)**2 * 3.d0 * dcos(2.d0 *
     &       alpha)**2/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mhh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top34 = elec2/(4.d0*ppi)**2 * (3.d0 * dsin(2.d0 *
     &       alpha)**2 - 1.d0)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top35 = -elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &        dcos(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,maa,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top36 = elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &        dcos(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      a1 =  ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mup**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
      a2 =  ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mup**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37a = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang7)**2 + a2*dsin(ang7)**2 ) * siga1 +
     &  ( a1*dsin(ang7)**2 + a2*dcos(ang7)**2 ) * siga2  )
c
      a1 =  ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mch**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
      a2 =  ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mch**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37b = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang9)**2 + a2*dsin(ang9)**2 ) * siga1 +
     &  ( a1*dsin(ang9)**2 + a2*dcos(ang9)**2 ) * siga2  )
c
      a1 =  ((0.5d0 - 2.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mtt**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
      a2 =  ( 2.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mtt**2 / mmw2 * dcos(alpha)**2/dsin(beta)**2 )
c
      call genhquad (6,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37c = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang11)**2 + a2*dsin(ang11)**2 ) * siga1 +
     &  ( a1*dsin(ang11)**2 + a2*dcos(ang11)**2 ) * siga2  )
c
      a1 =  ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mdn**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mdn**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37d = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0*(  
     &  ( a1*dcos(ang8)**2 + a2*dsin(ang8)**2 ) * siga1 +
     &  ( a1*dsin(ang8)**2 + a2*dcos(ang8)**2 ) * siga2  )
c
      a1 =  ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mst**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mst**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37e = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang10)**2 + a2*dsin(ang10)**2 ) * siga1 +
     &  ( a1*dsin(ang10)**2 + a2*dcos(ang10)**2 ) * siga2  )
c
      if (delmbresum.eq.1) then
      a1 =  ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mbb**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mbb**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0*(
     &  ( a1*dcos(ang12)**2 + a2*dsin(ang12)**2 ) * siga1 +
     &  ( a1*dsin(ang12)**2 + a2*dcos(ang12)**2 ) * siga2  )
      else
      a1 =  ((-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2 * dcos(2.d0 *
     &       alpha) - mbbdmb**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -1.d0/3.d0 * ssw2/ccw2 * dcos(2.d0 *
     &       alpha) - mbbdmb**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
c
      call genhquad (6,s,msb1dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (6,s,msb2dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * 3.d0*(
     &  ( a1*dcos(tsbdmb)**2 + a2*dsin(tsbdmb)**2 ) * siga1 +
     &  ( a1*dsin(tsbdmb)**2 + a2*dcos(tsbdmb)**2 ) * siga2  )
      endif
c
      a1 =  ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mel**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mel**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37g = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang2)**2 + a2*dsin(ang2)**2 ) * siga1 +
     &  ( a1*dsin(ang2)**2 + a2*dcos(ang2)**2 ) * siga2  )
c
      a1 =  ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mmu**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mmu**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37h = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang4)**2 + a2*dsin(ang4)**2 ) * siga1 +
     &  ( a1*dsin(ang4)**2 + a2*dcos(ang4)**2 ) * siga2  )
c
      a1 =  ((-0.5d0 + ssw2)/ccw2 * dcos(2.d0 *
     &          alpha) - mta**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2)
      a2 =  ( -ssw2/ccw2 * dcos(2.d0 *
     &         alpha) - mta**2 / mmw2 * dsin(alpha)**2/dcos(beta)**2 )
c
      call genhquad (6,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37i = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * (
     &  ( a1*dcos(ang6)**2 + a2*dsin(ang6)**2 ) * siga1 +
     &  ( a1*dsin(ang6)**2 + a2*dcos(ang6)**2 ) * siga2  )
c
      call genhquad (6,s,mvesl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37j = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      call genhquad (6,s,mvmsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37k = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      call genhquad (6,s,mvtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37l = -elec2 / (4.d0 * ppi)**2 /(2.d0 * ssw2) * siga *
     &         0.5d0 / ccw2 * dcos(2.d0*alpha)
c
      if (selec.ge.3) then
      top37 = top37a + top37b + top37c + top37d + top37e + top37l +
     &        top37f + top37g + top37h + top37i + top37j + top37k
      elseif (selec.eq.2) then
         top37 = top37c + top37f
      elseif (selec.eq.1) then
         top37 = top37c
      else
         write(*,*) "Error in Sigma-h: selec out or range"
      endif

c
c fermion loops
c
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mtt**2 * 3.d0
c      write(*,*) 'top17c:', top17c
c      write(*,*) dsqrt(s), siga, dsiga
c      write(*,*) mtt, elec2, ppi, alpha, ssw2, mmw, beta
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mst**2 * 3.d0
c
      if (delmbresum.eq.1) then
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mbb**2 * 3.d0
      else
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2 * mmw**2) 
     $     * siga * 3.d0 * (mbbdmb * (dsin(alpha)/dcos(beta)
     $                      - dmb * dcos(alpha)/dsin(beta)))**2
      endif
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mta**2
c
      if (selec.ge.3) then
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
      elseif (selec.eq.2) then
         top17 = top17c + top17f
      elseif (selec.eq.1) then
         top17 = top17c
c         write(*,*) 'h-SE: pure top:', real(top17c)
      else
         write(*,*) "Error in Sigma-h: selec out or range"
      endif

c
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dsin(alpha) -
     &      vmix(j,2)*umix(ii,1)*dcos(alpha) )/dsqrt(2.d0)
      b =  (vmix(ii,1)*umix(j,2)*dsin(alpha) -
     &      vmix(ii,2)*umix(j,1)*dcos(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,b,b,a, siga,dsiga)
      top21h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
      top21 = top21 + top21h
621   continue
620   continue
c
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dsin(alpha) + sm * dcos(alpha)
      b = a
      call genhquad (4,s,mne(ii),mne(j),a,a,b,b, siga,dsiga)
      top22h = -elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * siga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' h0 - selfenergy : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   four point interactions : '
       write (*,*) ' W+      = ', top29
       write (*,*) ' Z0      = ', top30
       write (*,*) ' G+      = ', top31
       write (*,*) ' H+      = ', top32
       write (*,*) ' h0      = ', top33
       write (*,*) ' H0      = ', top34
       write (*,*) ' G0      = ', top35
       write (*,*) ' A       = ', top36
       write (*,*) ' sfer    = ', top37
       write (*,*) '   '
      endif
c
      sigmalhb = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &           top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &           top18 + top20 + top29 + top30 + top31 + top32 + top33 +
     &           top34 + top35 + top36
      sigmalhs = top23 + top37
      sigmalhf = top17
      sigmalhc = top21 + top22
      sigmalht = top17c + top23c + top37c 
c      write(*,*) 'sigmalhf:', sigmalhf
c
      return
      end
c
c -------------------------------------------------------------------
c
      subroutine dsigmalh (s, dsigmalhb,dsigmalhs,dsigmalhf,dsigmalhc,
     &                        dsigmalht)
c
c     derivative of selfenergy of light scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "dSigma-h should not be used for Higgs masses!!"
      write(*,*) 'Delta mb corrections not yet implemented here'

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      if (pri.eq.1) write(*,*) 'top1'
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = elec2/(4.d0*ppi)**2/(2.d0*ssw2)*dsiga * dcos(beta-alpha)**2
c
      if (pri.eq.1) write(*,*) 'top2'
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  dsiga
c
      if (pri.eq.1) write(*,*) 'top3'
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dsin(beta - alpha)**2 /(
     &        4.d0 * ccw2 * ssw2) *  dsiga
c
      if (pri.eq.1) write(*,*) 'top4'
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) 
     $     * dsiga * dsin(beta-alpha)**2
c
      if (pri.eq.1) write(*,*) 'top5'
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 * dsiga * dsin(beta-alpha)**2
c
      if (pri.eq.1) write(*,*) 'top6'
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * dsiga
     &      * dsin(beta-alpha)**2 / 2.d0
c
      if (pri.eq.1) write(*,*) 'top7'
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dsin(beta-alpha) +
     &      mmz/(2.d0*ccw) * dcos(2.d0*beta) * dsin(beta+alpha))**2
     &      * dsiga
c
      if (pri.eq.1) write(*,*) 'top8'
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * 9.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        dsiga * dcos(2.d0*alpha)**2 * dsin(beta+alpha)**2
c
      if (pri.eq.1) write(*,*) 'top9'
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &       (dcos(2.d0*alpha) * dcos(beta+alpha) - 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta))**2
c
      if (pri.eq.1) write(*,*) 'top10'
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta))**2
c
      if (pri.eq.1) write(*,*) 'top11'
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
      if (pri.eq.1) write(*,*) 'top12'
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = -elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &         dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
      if (pri.eq.1) write(*,*) 'top13'
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = -elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta)**2
c
      if (pri.eq.1) write(*,*) 'top14'
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = -elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &         dsin(2.d0*beta)**2 * dsin(alpha+beta)**2
c
      if (pri.eq.1) write(*,*) 'top15'
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * dsiga * (mmw *
     &         dcos(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dsin
     &         (alpha+beta))**2
c
      if (pri.eq.1) write(*,*) 'top18'
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmw**2 *
     &         dsin(beta-alpha)**2 / 2.d0
c
      if (pri.eq.1) write(*,*) 'top20'
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmz**2 *
     &         dsin(beta-alpha)**2 / (4.d0*ccw2)
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mup/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      if (pri.eq.1) write(*,*) 'top23a'
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang7)
     &    **2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*ang7))**2 * dsiga1 +
     &   2.d0*( a12*dcos(2.d0*ang7)+(a22-a11)*dsin(ang7)*dcos(ang7))
     &   **2 * dsiga2 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2 - a12*
     &   dsin(2.d0*ang7))**2 * dsiga3 )
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mch/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      if (pri.eq.1) write(*,*) 'top23b'
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang9)
     &    **2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*ang9))**2 * dsiga1 +
     &   2.d0*(a12*dcos(2.d0*ang9)+(a22-a11)*dsin(ang9)*dcos(ang9))
     &   **2 * dsiga2 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2 - a12*
     &   dsin(2.d0*ang9))**2 * dsiga3 )
c
      a11 = (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
      a12 = mtt/(2.d0*
     &      mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
      a22 = (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)))
c
      if (pri.eq.1) write(*,*) 'top23c'
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23c = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang11)
     &  **2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*ang11))**2 * dsiga1 +     
     &  2.d0*(a12*dcos(2.d0*ang11) + (a22-a11)*dsin(ang11)*dcos(ang11))
     &   **2 * dsiga2 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2 - a12*
     &   dsin(2.d0*ang11))**2 * dsiga3 )
c
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mdn/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 = -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23d'
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1) 
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2) 
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3) 
      top23d = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 * ((a11*dcos(ang8)
     &  **2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*ang8))**2 * dsiga1 +     
     &  2.d0*( a12*dcos(2.d0*ang8)+(a22-a11)*dsin(ang8)*dcos(ang8))
     &   **2 * dsiga2 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2 - a12*
     &   dsin(2.d0*ang8))**2 * dsiga3 ) 
c
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mst/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23e'
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang10)
     &  **2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*ang10))**2 * dsiga1 +
     & 2.d0*(a12*dcos(2.d0*ang10)+(a22-a11)*dsin(ang10)*dcos(ang10))
     &   **2 * dsiga2+(a11*dsin(ang10)**2 + a22*dcos(ang10)**2 - a12*
     &   dsin(2.d0*ang10))**2 * dsiga3 )
c
      a11 = (mmz/ccw* (0.5d0
     &      -1.d0/3.d0*ssw2)*dsin(alpha+beta) - mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 = mbb/(2.d0*
     &      mmw * dcos(beta)) * 
     &      (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a22 =  -(mmz/ccw *
     &      (-1.d0/3.d0) * ssw2*dsin(alpha+beta) + mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23f'
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2 / (4.d0 * ppi)**2 /ssw2 * 3.d0 *((a11*dcos(ang12)
     &  **2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*ang12))**2 * dsiga1 +
     &  2.d0*(a12*dcos(2.d0*ang12)+(a22-a11)*dsin(ang12)*dcos(ang12))
     &   **2 * dsiga2 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2 - a12*
     &   dsin(2.d0*ang12))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mel/(2.d0*
     &         mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23g'
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang2)
     &  **2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*ang2))**2 * dsiga1 +     
     &  2.d0*(a12*dcos(2.d0*ang2)+(a22-a11)*dsin(ang2)*dcos(ang2))
     &  **2 * dsiga2 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2 - a12*
     &  dsin(2.d0*ang2))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mmu/(2.d0*
     &         mmw * dcos(beta)) *
     &      (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23h'
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang4)
     &  **2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*ang4))**2 * dsiga1 +     
     &  2.d0*( a12*dcos(2.d0*ang4)+(a22-a11)*dsin(ang4)*dcos(ang4))
     &  **2 * dsiga2 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2 - a12*
     &  dsin(2.d0*ang4))**2 * dsiga3 )
c
      a11 =  (mmz/ccw* (0.5d0
     &      -1.d0*ssw2)*dsin(alpha+beta) - mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a12 =  mta/(2.d0*
     &         mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a22 =  -(mmz/ccw* (
     &      -1.d0)*ssw2*dsin(alpha+beta) + mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
c
      if (pri.eq.1) write(*,*) 'top23i'
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2 / (4.d0 * ppi)**2 /ssw2 *   ((a11*dcos(ang6)
     &  **2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*ang6))**2 * dsiga1 +     
     &  2.d0*( a12*dcos(2.d0*ang6)+(a22-a11)*dsin(ang6)*dcos(ang6))
     &  **2 * dsiga2 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2 - a12*
     &  dsin(2.d0*ang6))**2 * dsiga3 )
c
      if (pri.eq.1) write(*,*) 'top23j'
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )**2
c
      if (pri.eq.1) write(*,*) 'top23k'
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )**2
c
      if (pri.eq.1) write(*,*) 'top23l'
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta)  )**2
c
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
c
c fermion loops
c
      if (pri.eq.1) write(*,*) 'top17'
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2 / (4.d0 * ppi)**2 * dcos(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mtt**2 * 3.d0
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mst**2 * 3.d0
c
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mbb**2 * 3.d0
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = -elec2 / (4.d0 * ppi)**2 * dsin(alpha)**2 /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mta**2
c
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
c
      if (pri.eq.1) write(*,*) 'top21'
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a =  (vmix(j,1)*umix(ii,2)*dsin(alpha) -
     &      vmix(j,2)*umix(ii,1)*dcos(alpha) )/dsqrt(2.d0)
      b =  (vmix(ii,1)*umix(j,2)*dsin(alpha) -
     &      vmix(ii,2)*umix(j,1)*dcos(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,b,b,a, siga,dsiga)
      top21h = -elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * dsiga
      top21 = top21 + top21h
621   continue
620   continue
c
      if (pri.eq.1) write(*,*) 'top22'
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      if (pri.eq.1) write(*,*) 'top22:',ii,j
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      a =  qm * dsin(alpha) + sm * dcos(alpha)
      b = a
      call genhquad (4,s,mne(ii),mne(j),a,a,b,b, siga,dsiga)
      top22h = -elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * dsiga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' h0 - derivative of selfenergy : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   '
      endif
c
      if (pri.eq.1) write(*,*) 'Letzte Summation'
      if (pri.eq.1) write(*,*) 'dsigmalhb'
      dsigmalhb =top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &           top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &           top18 + top20
      if (pri.eq.1) write(*,*) 'dsigmalhs'
      dsigmalhs = top23 
      if (pri.eq.1) write(*,*) 'dsigmalhf'
      dsigmalhf = top17
      if (pri.eq.1) write(*,*) 'dsigmalhc'
      dsigmalhc = top21 + top22
      if (pri.eq.1) write(*,*) 'dsigmalht'
      dsigmalht = top17c + top23c 
c
      if (pri.eq.1) write(*,*) 'dsigmalh-Ende'
      if (pri.eq.1) write(*,*) dsigmalhb,dsigmalhs,dsigmalhf,
     &                         dsigmalhc,dsigmalht
      return
      end
c=====================================================================
c
      subroutine sigmaxh (s, sigmahhb,sigmahhs,sigmahhf,sigmahhc,
     &                       sigmahht)
c
c     mixing of heavy - light scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga * dcos(beta-alpha)*
     &        dsin(beta-alpha)
c
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha) *
     &        dsin(beta - alpha) / (4.d0 * ccw2 * ssw2) *  siga
c
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dsin(beta - alpha) *
     &       dcos(beta - alpha) / (4.d0 * ccw2 * ssw2) *  siga
c
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga * dsin(beta-alpha) *
     &       dcos(beta-alpha)
c
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 
     $     * siga * dsin(beta-alpha) *
     &       dcos(beta-alpha)
c
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * siga
     &      * dsin(beta-alpha) * dcos(beta-alpha) / 2.d0
c
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dsin(beta-alpha) +
     &       mmz/(2.d0*ccw) * dcos(2.d0*beta)*dsin(beta+alpha) ) *
     &     ( mmw*dcos(beta-alpha) - mmz/(2.d0*ccw) * dcos(2.d0*beta)*
     &       dcos(beta+alpha) ) * siga
c
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * 3.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        siga * dcos(2.d0*alpha) * dsin(beta+alpha) *
     &     (2.d0*dsin(2.d0*alpha)*dsin(alpha+beta)-dcos(2.d0*alpha) *
     &          dcos(alpha+beta) )
c
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &       (-dcos(2.d0*alpha) * dcos(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta)) * ( 2.d0 *
     &       dsin(2.d0*alpha) * dcos(beta+alpha) + dcos(2.d0*alpha) *
     &       dsin(beta+alpha) )
c
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 =  elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta)) * dcos(2.d0*alpha) *
     &        dcos(alpha+beta) * 3.d0
c
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * siga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * siga *
     &        dsin(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 =  elec2/(4.d0*ppi)**2/(2.d0*ssw2) * siga * (mmw *
     &         dcos(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dsin
     &         (alpha+beta)) * (mmw * dsin(beta-alpha) -
     &         mmz/ccw * dsin(2.d0*beta) * dcos(alpha+beta))
c
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmw**2 *
     &         dsin(beta-alpha) * dcos(beta-alpha) / 2.d0
c
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * siga * mmz**2 *
     &         dsin(beta-alpha) * dcos(beta-alpha)  / (4.d0*ccw2)
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mup/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mup/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang7)
     &    **2 + a22a*dsin(ang7)**2 + a12a*dsin(2.d0*ang7))*(a11b *
     &    dcos(ang7)**2 + a22b*dsin(ang7)**2 + a12b*dsin(2.d0*ang7))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang7)+(a22a-a11a)*dsin(ang7)*dcos(ang7)) *
     &   (a12b*dcos(2.d0*ang7)+(a22b-a11b)*dsin(ang7)*dcos(ang7)) *
     &   siga2 + (a11a*dsin(ang7)**2 + a22a*dcos(ang7)**2- a12a*
     &   dsin(2.d0*ang7)) * (a11b*dsin(ang7)**2 + a22b*dcos(ang7)**2
     &   - a12b * dsin(2.d0*ang7))* siga3 )
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mch/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mch/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang9)
     &    **2 + a22a*dsin(ang9)**2 + a12a*dsin(2.d0*ang9))*(a11b *
     &    dcos(ang9)**2 + a22b*dsin(ang9)**2 + a12b*dsin(2.d0*ang9))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang9)+(a22a-a11a)*dsin(ang9)*dcos(ang9)) *
     &   (a12b*dcos(2.d0*ang9)+(a22b-a11b)*dsin(ang9)*dcos(ang9)) *
     &   siga2 + (a11a*dsin(ang9)**2 + a22a*dcos(ang9)**2- a12a*
     &   dsin(2.d0*ang9)) * (a11b*dsin(ang9)**2 + a22b*dcos(ang9)**2
     &   - a12b * dsin(2.d0*ang9))* siga3 )
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mtt/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c      write(*,*) 'top23c:', real(mssupq)
c
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23c = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang11)
     &    **2 + a22a*dsin(ang11)**2 + a12a*dsin(2.d0*ang11))*(a11b *
     &    dcos(ang11)**2 + a22b*dsin(ang11)**2 + a12b*dsin(2.d0*ang11))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang11)+(a22a-a11a)*dsin(ang11)*dcos(ang11)) *
     &   (a12b*dcos(2.d0*ang11)+(a22b-a11b)*dsin(ang11)*dcos(ang11)) *
     &   siga2 + (a11a*dsin(ang11)**2 + a22a*dcos(ang11)**2- a12a*
     &   dsin(2.d0*ang11)) * (a11b*dsin(ang11)**2 + a22b*dcos(ang11)**2
     &   - a12b * dsin(2.d0*ang11))* siga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mdn/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mdn/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23d = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang8)
     &    **2 + a22a*dsin(ang8)**2 + a12a*dsin(2.d0*ang8))*(a11b *
     &    dcos(ang8)**2 + a22b*dsin(ang8)**2 + a12b*dsin(2.d0*ang8))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang8)+(a22a-a11a)*dsin(ang8)*dcos(ang8)) *
     &   (a12b*dcos(2.d0*ang8)+(a22b-a11b)*dsin(ang8)*dcos(ang8)) *
     &   siga2 + (a11a*dsin(ang8)**2 + a22a*dcos(ang8)**2- a12a*
     &   dsin(2.d0*ang8)) * (a11b*dsin(ang8)**2 + a22b*dcos(ang8)**2
     &   - a12b * dsin(2.d0*ang8))* siga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mst/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mst/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang10)
     &    **2 + a22a*dsin(ang10)**2 + a12a*dsin(2.d0*ang10))*(a11b *
     &    dcos(ang10)**2 + a22b*dsin(ang10)**2+ a12b*dsin(2.d0*ang10))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang10)+(a22a-a11a)*dsin(ang10)*dcos(ang10)) *
     &   (a12b*dcos(2.d0*ang10)+(a22b-a11b)*dsin(ang10)*dcos(ang10)) *
     &   siga2 + (a11a*dsin(ang10)**2 + a22a*dcos(ang10)**2- a12a*
     &   dsin(2.d0*ang10)) * (a11b*dsin(ang10)**2+a22b*dcos(ang10)**2
     &   - a12b * dsin(2.d0*ang10))* siga3 )
c
      if (delmbresum.eq.1) then
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mbb/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mbb/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a*dcos(ang12)
     &    **2 + a22a*dsin(ang12)**2 + a12a*dsin(2.d0*ang12))*(a11b *
     &    dcos(ang12)**2 + a22b*dsin(ang12)**2+a12b*dsin(2.d0*ang12))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang12)+(a22a-a11a)*dsin(ang12)*dcos(ang12))*
     &   (a12b*dcos(2.d0*ang12)+(a22b-a11b)*dsin(ang12)*dcos(ang12))*
     &   siga2 + (a11a*dsin(ang12)**2 + a22a*dcos(ang12)**2- a12a*
     &   dsin(2.d0*ang12)) * (a11b*dsin(ang12)**2+a22b*dcos(ang12)**2
     &   - a12b * dsin(2.d0*ang12))* siga3 )
      else
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) 
     $        - mbbdmb**2 * dsin(alpha)/(mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) 
     $     - mbbdmb**2 * dcos(alpha)/(mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) 
     $     + mbbdmb**2 * dsin(alpha)/(mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) 
     $     - mbbdmb**2 * dcos(alpha)/(mmw*dcos(beta)) )
      a12a = -mbbdmb/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mbbdmb/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,msb1dmb,msb1dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (2,s,msb1dmb,msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      call genhquad (2,s,msb2dmb,msb2dmb,1.d0,1.d0,1.d0,1.d0, 
     $               siga3,dsiga3)
      top23f = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0
     $     *( (a11a*dcos(tsbdmb)
     &    **2 + a22a*dsin(tsbdmb)**2 
     $     + a12a*dsin(2.d0*tsbdmb))*(a11b *
     &    dcos(tsbdmb)**2 + a22b*dsin(tsbdmb)**2
     $     +a12b*dsin(2.d0*tsbdmb))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*tsbdmb)+(a22a-a11a)
     $     *dsin(tsbdmb)*dcos(tsbdmb))*
     &   (a12b*dcos(2.d0*tsbdmb)+(a22b-a11b)
     $     *dsin(tsbdmb)*dcos(tsbdmb))*
     &   siga2 + (a11a*dsin(tsbdmb)**2 
     $     + a22a*dcos(tsbdmb)**2- a12a*
     &   dsin(2.d0*tsbdmb)) * (a11b*dsin(tsbdmb)**2
     $     +a22b*dcos(tsbdmb)**2
     &   - a12b * dsin(2.d0*tsbdmb))* siga3 )
      endif
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mel/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mel/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang2)
     &    **2 + a22a*dsin(ang2)**2 + a12a*dsin(2.d0*ang2))*(a11b *
     &    dcos(ang2)**2 + a22b*dsin(ang2)**2+a12b*dsin(2.d0*ang2))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang2)+(a22a-a11a)*dsin(ang2)*dcos(ang2))*
     &   (a12b*dcos(2.d0*ang2)+(a22b-a11b)*dsin(ang2)*dcos(ang2))*
     &   siga2 + (a11a*dsin(ang2)**2 + a22a*dcos(ang2)**2- a12a*
     &   dsin(2.d0*ang2)) * (a11b*dsin(ang2)**2+a22b*dcos(ang2)**2
     &   - a12b * dsin(2.d0*ang2))* siga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mmu/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a12b = -mmu/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang4)
     &    **2 + a22a*dsin(ang4)**2 + a12a*dsin(2.d0*ang4))*(a11b *
     &    dcos(ang4)**2 + a22b*dsin(ang4)**2+a12b*dsin(2.d0*ang4))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang4)+(a22a-a11a)*dsin(ang4)*dcos(ang4))*
     &   (a12b*dcos(2.d0*ang4)+(a22b-a11b)*dsin(ang4)*dcos(ang4))*
     &   siga2 + (a11a*dsin(ang4)**2 + a22a*dcos(ang4)**2- a12a*
     &   dsin(2.d0*ang4)) * (a11b*dsin(ang4)**2+a22b*dcos(ang4)**2
     &   - a12b * dsin(2.d0*ang4))* siga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mta/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a12b = -mta/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang6)
     &    **2 + a22a*dsin(ang6)**2 + a12a*dsin(2.d0*ang6))*(a11b *
     &    dcos(ang6)**2 + a22b*dsin(ang6)**2+a12b*dsin(2.d0*ang6))
     &    *siga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang6)+(a22a-a11a)*dsin(ang6)*dcos(ang6))*
     &   (a12b*dcos(2.d0*ang6)+(a22b-a11b)*dsin(ang6)*dcos(ang6))*
     &   siga2 + (a11a*dsin(ang6)**2 + a22a*dcos(ang6)**2- a12a*
     &   dsin(2.d0*ang6)) * (a11b*dsin(ang6)**2+a22b*dcos(ang6)**2
     &   - a12b * dsin(2.d0*ang6))* siga3 )
c
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * siga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      if (selec.ge.3) then
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
      elseif (selec.eq.2) then
         top23 = top23c + top23f
      elseif (selec.eq.1) then
         top23 = top23c
      else
         write(*,*) "Error in Sigma-hH: selec out or range"
      endif

c
      call genhquad (6,s,mmw,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top31 = -elec2/(4.d0*ppi)**2 * (dsin(2.d0*beta) *
     &        dcos(2.d0*alpha) + ssw2/ccw2 * dcos(2.d0*beta)*dsin(
     &        2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mhp,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top32 =  elec2/(4.d0*ppi)**2 * (dsin(2.d0 * beta)
     &        *dcos(2.d0*alpha) + ssw2/ccw2 * dcos(2.d0*beta)*dsin(
     &        2.d0*alpha) ) * siga / (4.d0*ssw2)
c
      call genhquad (6,s,mlh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top33 = elec2/(4.d0*ppi)**2 * 3.d0 * dcos(2.d0 *
     &        alpha)*dsin(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mhh,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top34 = -elec2/(4.d0*ppi)**2 * 3.d0 * dsin(2.d0 *
     &         alpha)*dcos(2.d0 * alpha)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,mmz,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top35 = -elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &         dsin(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      call genhquad (6,s,maa,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top36 = elec2/(4.d0*ppi)**2 * dcos(2.d0*beta) *
     &        dsin(2.d0*alpha)/(8.d0*ssw2*ccw2) * siga
c
      a1 = (2.d0 * (0.5d0 - 2.d0/3.d0 * ssw2)/ccw2
     &      - mup**2/mmw2 / dsin(beta)**2 ) * dsin(2.d0*alpha)
      a2 = ( 2.d0 * 2.d0/3.d0 * ssw2/ccw2
     &      - mup**2/mmw2 /dsin(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mupsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mupsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37a = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(ang7)**2 + a2*dsin(ang7)**2 ) * siga1 +
     &  ( a1*dsin(ang7)**2 + a2*dcos(ang7)**2 ) * siga2  )
c
      a1 = (2.d0 * (0.5d0 - 2.d0/3.d0 * ssw2)/ccw2
     &      - mch**2/mmw2 / dsin(beta)**2 ) * dsin(2.d0*alpha)
      a2 = ( 2.d0 * 2.d0/3.d0 * ssw2/ccw2
     &      - mch**2/mmw2 /dsin(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mchsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mchsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37b = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(ang9)**2 + a2*dsin(ang9)**2 ) * siga1 +
     &  ( a1*dsin(ang9)**2 + a2*dcos(ang9)**2 ) * siga2  )
c
      a1 = (2.d0 * (0.5d0 - 2.d0/3.d0 * ssw2)/ccw2
     &      - mtt**2/mmw2 / dsin(beta)**2 ) * dsin(2.d0*alpha)
      a2 = ( 2.d0 * 2.d0/3.d0 * ssw2/ccw2
     &      - mtt**2/mmw2 /dsin(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37c = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 * (
     &  ( a1*dcos(ang11)**2 + a2*dsin(ang11)**2 ) * siga1 +
     &  ( a1*dsin(ang11)**2 + a2*dcos(ang11)**2 ) * siga2  )
c      write(*,*) 'top37c:'
c      write(*,*) ( a1*dcos(ang11)**2 + a2*dsin(ang11)**2 ), siga1
c      write(*,*) ( a1*dsin(ang11)**2 + a2*dcos(ang11)**2 ), siga2
c
      a1 =  (2.d0 * (-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2
     &       + mdn**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0/3.d0 * ssw2/ccw2
     &         + mdn**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mdnsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mdnsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37d = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(ang8)**2 + a2*dsin(ang8)**2 ) * siga1 +
     &  ( a1*dsin(ang8)**2 + a2*dcos(ang8)**2 ) * siga2  )
c
      a1 =  (2.d0 * (-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2
     &       + mst**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0/3.d0 * ssw2/ccw2
     &         + mst**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mstsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mstsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37e = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(ang10)**2 + a2*dsin(ang10)**2 ) * siga1 +
     &  ( a1*dsin(ang10)**2 + a2*dcos(ang10)**2 ) * siga2  )
c
      if (delmbresum.eq.1) then
      a1 =  (2.d0 * (-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2
     &       + mbb**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0/3.d0 * ssw2/ccw2
     &         + mbb**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mbsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mbsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(ang12)**2 + a2*dsin(ang12)**2 ) * siga1 +
     &  ( a1*dsin(ang12)**2 + a2*dcos(ang12)**2 ) * siga2  )
      else
      a1 =  (2.d0 * (-0.5d0 + 1.d0/3.d0 * ssw2)/ccw2
     &       + mbbdmb**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0/3.d0 * ssw2/ccw2
     &         + mbbdmb**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,msb1dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga1,dsiga1)
      call genhquad (6,s,msb2dmb,1.d0,1.d0,1.d0,1.d0,1.d0, 
     $               siga2,dsiga2)
      top37f = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * 3.d0 *  (
     &  ( a1*dcos(tsbdmb)**2 + a2*dsin(tsbdmb)**2 ) * siga1 +
     &  ( a1*dsin(tsbdmb)**2 + a2*dcos(tsbdmb)**2 ) * siga2  )
      endif
c
      a1 =  (2.d0 * (-0.5d0 + 1.d0 * ssw2)/ccw2
     &       + mel**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0 * ssw2/ccw2
     &         + mel**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,melsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,melsr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37g = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) *  (
     &  ( a1*dcos(ang2)**2 + a2*dsin(ang2)**2 ) * siga1 +
     &  ( a1*dsin(ang2)**2 + a2*dcos(ang2)**2 ) * siga2  )
c
      a1 =  (2.d0 * (-0.5d0 + 1.d0 * ssw2)/ccw2
     &       + mmu**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0 * ssw2/ccw2
     &         + mmu**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mmusl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mmusr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37h = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) *  (
     &  ( a1*dcos(ang4)**2 + a2*dsin(ang4)**2 ) * siga1 +
     &  ( a1*dsin(ang4)**2 + a2*dcos(ang4)**2 ) * siga2  )
c
      a1 =  (2.d0 * (-0.5d0 + 1.d0 * ssw2)/ccw2
     &       + mta**2/mmw2 / dcos(beta)**2 ) * dsin(2.d0*alpha)
      a2 =  ( -2.d0 * 1.d0 * ssw2/ccw2
     &         + mta**2/mmw2 /dcos(beta)**2 ) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mtasl,1.d0,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (6,s,mtasr,1.d0,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      top37i = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) *  (
     &  ( a1*dcos(ang6)**2 + a2*dsin(ang6)**2 ) * siga1 +
     &  ( a1*dsin(ang6)**2 + a2*dcos(ang6)**2 ) * siga2  )
c
      call genhquad (6,s,mvesl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37j = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * siga *
     &         (1.d0 / ccw2) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mvmsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37k = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * siga *
     &         (1.d0 / ccw2) * dsin(2.d0*alpha)
c
      call genhquad (6,s,mvtsl,1.d0,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top37l = -elec2 / (4.d0 * ppi)**2 /(4.d0 * ssw2) * siga *
     &         (1.d0 / ccw2) * dsin(2.d0*alpha)
c
      if (selec.ge.3) then
      top37 = top37a + top37b + top37c + top37d + top37e + top37l +
     &        top37f + top37g + top37h + top37i + top37j + top37k
      elseif (selec.eq.2) then
         top37 = top37c + top37f
      elseif (selec.eq.1) then
         top37 = top37c
      else
         write(*,*) "Error in Sigma-hH: selec out or range"
      endif

c
c fermion loops
c
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &     ssw2 * mmw**2*dsin(beta)**2) * siga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * siga * mtt**2 * 3.d0
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mst**2 * 3.d0
c
      if (delmbresum.eq.1) then
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mbb**2 * 3.d0
      else
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = elec2/(4.d0*ppi)**2 /(4.d0 * ssw2 * mmw**2) * siga 
     $     * mbbdmb**2 * 3.d0 * 
     $     (dsin(alpha)/dcos(beta) - dmb * dcos(alpha)/dsin(beta)) *
     $     (dcos(alpha)/dcos(beta) + dmb * dsin(alpha)/dsin(beta))
      endif
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * siga * mta**2
c
      if (selec.ge.3) then
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
      elseif (selec.eq.2) then
         top17 = top17c + top17f
      elseif (selec.eq.1) then
         top17 = top17c
      else
         write(*,*) "Error in Sigma-hH: selec out or range"
      endif

c
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a = (vmix(j,1)*umix(ii,2)*dsin(alpha) -
     &     vmix(j,2)*umix(ii,1)*dcos(alpha) )/dsqrt(2.d0)
      b = (vmix(ii,1)*umix(j,2)*dsin(alpha) -
     &     vmix(ii,2)*umix(j,1)*dcos(alpha) )/dsqrt(2.d0)
      at = (vmix(j,1)*umix(ii,2)*dcos(alpha) +
     &      vmix(j,2)*umix(ii,1)*dsin(alpha) )/dsqrt(2.d0)
      bt = (vmix(ii,1)*umix(j,2)*dcos(alpha) +
     &      vmix(ii,2)*umix(j,1)*dsin(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,bt,b,at, siga,dsiga)
      top21h =  elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * siga
      top21 = top21 + top21h
621   continue
620   continue
c
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      at = qm * dsin(alpha) + sm * dcos(alpha)
      a =  qm * dcos(alpha) - sm * dsin(alpha)
      call genhquad (4,s,mne(ii),mne(j),a,at,a,at, siga,dsiga)
      top22h = elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * siga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' h0-H0  - mixing : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   four point interactions : '
       write (*,*) ' G+      = ', top31
       write (*,*) ' H+      = ', top32
       write (*,*) ' h0      = ', top33
       write (*,*) ' H0      = ', top34
       write (*,*) ' G0      = ', top35
       write (*,*) ' A       = ', top36
       write (*,*) ' sfer    = ', top37
       write (*,*) '   '
      endif
c
      sigmahhb = top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &           top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &           top18 + top20 + top31 + top32 + top33 +
     &           top34 + top35 + top36
      sigmahhs = top23 + top37
      sigmahhf = top17
      sigmahhc = top21 + top22
      sigmahht = top17c + top23c + top37c 
c
      return
      end
c
c -------------------------------------------------------------------
c
      subroutine dsigmaxh (s, dsigmahhb,dsigmahhs,dsigmahhf,dsigmahhc,
     &                        dsigmahht)
c
c     deriv. of mixing of heavy - light scalar higgsparticle
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer pr,ii,j,selec,selec2,selec4,selec5,selec6,pri,naeh
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6

      write(*,*) "dSigma-hH should not be used for Higgs masses!!"
      write(*,*) 'Delta mb corrections not yet implemented here'

c
c boson loops
c
c  notation :
c      genhquad (typ,s,mupper,mlower,a,at,b,bt, siga,dsiga)
c
      call genhquad (1,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top1 = -elec2/(4.d0*ppi)**2/(2.d0*ssw2) 
     $     * dsiga * dcos(beta-alpha)*
     &        dsin(beta-alpha)
c
      call genhquad (1,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top2 = -elec2 / (4.d0 * ppi)**2 * dcos(beta - alpha) *
     &        dsin(beta - alpha) / (4.d0 * ccw2 * ssw2) *  dsiga
c
      call genhquad (1,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top3 = elec2 / (4.d0 * ppi)**2 * dsin(beta - alpha) *
     &       dcos(beta - alpha) / (4.d0 * ccw2 * ssw2) *  dsiga
c
      call genhquad (1,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top4 = elec2/(4.d0*ppi)**2/(2.d0*ssw2) 
     $     * dsiga * dsin(beta-alpha) *
     &       dcos(beta-alpha)
c
      call genhquad (8,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top5 = -elec2/(4.d0*ppi)**2*mmw**2/ssw2 
     $     * dsiga * dsin(beta-alpha) *
     &       dcos(beta-alpha)
c
      call genhquad (8,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top6 = -elec2/(4.d0*ppi)**2*mmz**2/(ssw2*ccw2) * dsiga
     &      * dsin(beta-alpha) * dcos(beta-alpha) / 2.d0
c
      call genhquad (2,s,mhp,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top7 = -elec2/(4.d0*ppi)**2/ssw2 * ( mmw*dsin(beta-alpha) +
     &       mmz/(2.d0*ccw) * dcos(2.d0*beta)*dsin(beta+alpha) ) *
     &     ( mmw*dcos(beta-alpha) - mmz/(2.d0*ccw) * dcos(2.d0*beta)*
     &       dcos(beta+alpha) ) * dsiga
c
      call genhquad (2,s,mlh,mlh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top8 = -elec2/(4.d0*ppi)**2 * 3.d0*mmz**2/(8.d0*ccw2*ssw2) *
     &        dsiga * dcos(2.d0*alpha) * dsin(beta+alpha) *
     &     (2.d0*dsin(2.d0*alpha)*dsin(alpha+beta)-dcos(2.d0*alpha) *
     &          dcos(alpha+beta) )
c
      call genhquad (2,s,mlh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top9 = elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &       (-dcos(2.d0*alpha) * dcos(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dsin(alpha+beta)) * ( 2.d0 *
     &       dsin(2.d0*alpha) * dcos(beta+alpha) + dcos(2.d0*alpha) *
     &       dsin(beta+alpha) )
c
      call genhquad (2,s,mhh,mhh,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top10 =  elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &       (dcos(2.d0*alpha) * dsin(beta+alpha) + 2.d0 * dsin
     &       (2.d0*alpha) * dcos(alpha+beta)) * dcos(2.d0*alpha) *
     &        dcos(alpha+beta) * 3.d0
c
      call genhquad (2,s,maa,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top11 = elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top12 = elec2/(4.d0*ppi)**2 * mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top13 = elec2/(4.d0*ppi)**2 * mmz**2/(8.d0*ccw2*ssw2) * dsiga *
     &        dcos(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmz,maa,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top14 = elec2/(4.d0*ppi)**2*mmz**2/(4.d0*ccw2*ssw2) * dsiga *
     &        dsin(2.d0*beta)**2 * dsin(alpha+beta) *
     &        dcos(alpha+beta)
c
      call genhquad (2,s,mmw,mhp,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top15 =  elec2/(4.d0*ppi)**2/(2.d0*ssw2) * dsiga * (mmw *
     &         dcos(beta-alpha) - mmz/ccw * dsin(2.d0*beta) * dsin
     &         (alpha+beta)) * (mmw * dsin(beta-alpha) -
     &         mmz/ccw * dsin(2.d0*beta) * dcos(alpha+beta))
c
      call genhquad (9,s,mmw,mmw,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top18 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmw**2 *
     &         dsin(beta-alpha) * dcos(beta-alpha) / 2.d0
c
      call genhquad (9,s,mmz,mmz,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top20 = -elec2/(4.d0*ppi)**2/ssw2 * dsiga * mmz**2 *
     &         dsin(beta-alpha) * dcos(beta-alpha)  / (4.d0*ccw2)
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mup**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mup**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mup/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mup/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mupsl,mupsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mupsl,mupsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mupsr,mupsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23a = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang7)
     &    **2 + a22a*dsin(ang7)**2 + a12a*dsin(2.d0*ang7))*(a11b *
     &    dcos(ang7)**2 + a22b*dsin(ang7)**2 + a12b*dsin(2.d0*ang7))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang7)+(a22a-a11a)*dsin(ang7)*dcos(ang7)) *
     &   (a12b*dcos(2.d0*ang7)+(a22b-a11b)*dsin(ang7)*dcos(ang7)) *
     &   dsiga2 + (a11a*dsin(ang7)**2 + a22a*dcos(ang7)**2- a12a*
     &   dsin(2.d0*ang7)) * (a11b*dsin(ang7)**2 + a22b*dcos(ang7)**2
     &   - a12b * dsin(2.d0*ang7))* dsiga3 )
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mch**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mch**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mch/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mch/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mchsl,mchsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mchsl,mchsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mchsr,mchsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23b = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang9)
     &    **2 + a22a*dsin(ang9)**2 + a12a*dsin(2.d0*ang9))*(a11b *
     &    dcos(ang9)**2 + a22b*dsin(ang9)**2 + a12b*dsin(2.d0*ang9))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang9)+(a22a-a11a)*dsin(ang9)*dcos(ang9)) *
     &   (a12b*dcos(2.d0*ang9)+(a22b-a11b)*dsin(ang9)*dcos(ang9)) *
     &   dsiga2 + (a11a*dsin(ang9)**2 + a22a*dcos(ang9)**2- a12a*
     &   dsin(2.d0*ang9)) * (a11b*dsin(ang9)**2 + a22b*dcos(ang9)**2
     &   - a12b * dsin(2.d0*ang9))* dsiga3 )
c
      a11a =  (mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a11b = (-mmz/ccw*(0.5d0 -
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a22a =  (mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dsin(alpha+beta) - mtt**2 * dcos(alpha)/
     &      (mmw*dsin(beta)) )
      a22b =  (-mmz/ccw*(
     &      2.d0/3.d0 *ssw2)*dcos(alpha+beta) - mtt**2 * dsin(alpha)/
     &      (mmw*dsin(beta)) )
      a12a = mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dsin(alpha) - mssupq * dcos(alpha))
      a12b = -mtt/(2.d0*
     &        mmw * dsin(beta)) *
     &        (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call genhquad (2,s,mtsl,mtsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtsl,mtsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtsr,mtsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23c = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang11)
     &    **2 + a22a*dsin(ang11)**2 + a12a*dsin(2.d0*ang11))*(a11b *
     &    dcos(ang11)**2 + a22b*dsin(ang11)**2 + a12b*dsin(2.d0*ang11))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang11)+(a22a-a11a)*dsin(ang11)*dcos(ang11)) *
     &   (a12b*dcos(2.d0*ang11)+(a22b-a11b)*dsin(ang11)*dcos(ang11)) *
     &   dsiga2 + (a11a*dsin(ang11)**2 + a22a*dcos(ang11)**2- a12a*
     &   dsin(2.d0*ang11)) * (a11b*dsin(ang11)**2 + a22b*dcos(ang11)**2
     &   - a12b * dsin(2.d0*ang11))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mdn**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mdn**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mdn/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mdn/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mdnsl,mdnsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mdnsl,mdnsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mdnsr,mdnsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23d = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang8)
     &    **2 + a22a*dsin(ang8)**2 + a12a*dsin(2.d0*ang8))*(a11b *
     &    dcos(ang8)**2 + a22b*dsin(ang8)**2 + a12b*dsin(2.d0*ang8))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang8)+(a22a-a11a)*dsin(ang8)*dcos(ang8)) *
     &   (a12b*dcos(2.d0*ang8)+(a22b-a11b)*dsin(ang8)*dcos(ang8)) *
     &   dsiga2 + (a11a*dsin(ang8)**2 + a22a*dcos(ang8)**2- a12a*
     &   dsin(2.d0*ang8)) * (a11b*dsin(ang8)**2 + a22b*dcos(ang8)**2
     &   - a12b * dsin(2.d0*ang8))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mst**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mst**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mst/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mst/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mstsl,mstsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mstsl,mstsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mstsr,mstsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23e = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a * dcos(ang10)
     &    **2 + a22a*dsin(ang10)**2 + a12a*dsin(2.d0*ang10))*(a11b *
     &    dcos(ang10)**2 + a22b*dsin(ang10)**2+ a12b*dsin(2.d0*ang10))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang10)+(a22a-a11a)*dsin(ang10)*dcos(ang10)) *
     &   (a12b*dcos(2.d0*ang10)+(a22b-a11b)*dsin(ang10)*dcos(ang10)) *
     &   dsiga2 + (a11a*dsin(ang10)**2 + a22a*dcos(ang10)**2- a12a*
     &   dsin(2.d0*ang10)) * (a11b*dsin(ang10)**2+a22b*dcos(ang10)**2
     &   - a12b * dsin(2.d0*ang10))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dsin(alpha+beta) - mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dsin(alpha+beta) + mbb**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0/3.d0 *ssw2)*dcos(alpha+beta) - mbb**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mbb/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mbb/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call genhquad (2,s,mbsl,mbsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mbsl,mbsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mbsr,mbsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23f = -elec2/(4.d0 * ppi)**2 /ssw2 * 3.d0*( (a11a*dcos(ang12)
     &    **2 + a22a*dsin(ang12)**2 + a12a*dsin(2.d0*ang12))*(a11b *
     &    dcos(ang12)**2 + a22b*dsin(ang12)**2+a12b*dsin(2.d0*ang12))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang12)+(a22a-a11a)*dsin(ang12)*dcos(ang12))*
     &   (a12b*dcos(2.d0*ang12)+(a22b-a11b)*dsin(ang12)*dcos(ang12))*
     &   dsiga2 + (a11a*dsin(ang12)**2 + a22a*dcos(ang12)**2- a12a*
     &   dsin(2.d0*ang12)) * (a11b*dsin(ang12)**2+a22b*dcos(ang12)**2
     &   - a12b * dsin(2.d0*ang12))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mel**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mel**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mel/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnq * dsin(alpha))
      a12b = -mel/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,melsl,melsl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,melsl,melsr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,melsr,melsr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23g = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang2)
     &    **2 + a22a*dsin(ang2)**2 + a12a*dsin(2.d0*ang2))*(a11b *
     &    dcos(ang2)**2 + a22b*dsin(ang2)**2+a12b*dsin(2.d0*ang2))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang2)+(a22a-a11a)*dsin(ang2)*dcos(ang2))*
     &   (a12b*dcos(2.d0*ang2)+(a22b-a11b)*dsin(ang2)*dcos(ang2))*
     &   dsiga2 + (a11a*dsin(ang2)**2 + a22a*dcos(ang2)**2- a12a*
     &   dsin(2.d0*ang2)) * (a11b*dsin(ang2)**2+a22b*dcos(ang2)**2
     &   - a12b * dsin(2.d0*ang2))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mmu**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mmu**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mmu/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a12b = -mmu/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mmusl,mmusl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mmusl,mmusr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mmusr,mmusr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23h = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang4)
     &    **2 + a22a*dsin(ang4)**2 + a12a*dsin(2.d0*ang4))*(a11b *
     &    dcos(ang4)**2 + a22b*dsin(ang4)**2+a12b*dsin(2.d0*ang4))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang4)+(a22a-a11a)*dsin(ang4)*dcos(ang4))*
     &   (a12b*dcos(2.d0*ang4)+(a22b-a11b)*dsin(ang4)*dcos(ang4))*
     &   dsiga2 + (a11a*dsin(ang4)**2 + a22a*dcos(ang4)**2- a12a*
     &   dsin(2.d0*ang4)) * (a11b*dsin(ang4)**2+a22b*dcos(ang4)**2
     &   - a12b * dsin(2.d0*ang4))* dsiga3 )
c
      a11a =  -(mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dsin(alpha+beta) - mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a11b = ( mmz/ccw*(0.5d0 -
     &      1.d0 *ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a22a =  (mmz/ccw*(
     &      -1.d0 *ssw2)*dsin(alpha+beta) + mta**2 * dsin(alpha)/
     &      (mmw*dcos(beta)) )
      a22b =  (-mmz/ccw*(
     &      -1.d0 *ssw2)*dcos(alpha+beta) - mta**2 * dcos(alpha)/
     &      (mmw*dcos(beta)) )
      a12a = -mta/(2.d0*
     &       mmw * dcos(beta)) *
     &       (mu*dcos(alpha) - mssdnl * dsin(alpha))
      a12b = -mta/(2.d0*
     &        mmw * dcos(beta)) *
     &        (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call genhquad (2,s,mtasl,mtasl,1.d0,1.d0,1.d0,1.d0, siga1,dsiga1)
      call genhquad (2,s,mtasl,mtasr,1.d0,1.d0,1.d0,1.d0, siga2,dsiga2)
      call genhquad (2,s,mtasr,mtasr,1.d0,1.d0,1.d0,1.d0, siga3,dsiga3)
      top23i = -elec2/(4.d0 * ppi)**2 /ssw2 * ( (a11a*dcos(ang6)
     &    **2 + a22a*dsin(ang6)**2 + a12a*dsin(2.d0*ang6))*(a11b *
     &    dcos(ang6)**2 + a22b*dsin(ang6)**2+a12b*dsin(2.d0*ang6))
     &    *dsiga1+ 2.d0*
     &   (a12a*dcos(2.d0*ang6)+(a22a-a11a)*dsin(ang6)*dcos(ang6))*
     &   (a12b*dcos(2.d0*ang6)+(a22b-a11b)*dsin(ang6)*dcos(ang6))*
     &   dsiga2 + (a11a*dsin(ang6)**2 + a22a*dcos(ang6)**2- a12a*
     &   dsin(2.d0*ang6)) * (a11b*dsin(ang6)**2+a22b*dcos(ang6)**2
     &   - a12b * dsin(2.d0*ang6))* dsiga3 )
c
      call genhquad (2,s,mvesl,mvesl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23j = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      call genhquad (2,s,mvmsl,mvmsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23k = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      call genhquad (2,s,mvtsl,mvtsl,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top23l = -elec2 / (4.d0 * ppi)**2 /ssw2 * dsiga * (mmz/ccw*(0.5d0
     &         )*dsin(alpha+beta) )*(-mmz/ccw*(0.5d0)*dcos(alpha+beta))
c
      top23 = top23a + top23b + top23c + top23d + top23e + top23f +
     &        top23g + top23h + top23i + top23j + top23k + top23l
c
c fermion loops
c
      call genhquad (3,s,mup,mup,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17a = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &     ssw2 * mmw**2*dsin(beta)**2) * dsiga * mup**2 * 3.d0
c
      call genhquad (3,s,mch,mch,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17b = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mch**2 * 3.d0
c
      call genhquad (3,s,mtt,mtt,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17c = -elec2/(4.d0*ppi)**2 * dcos(alpha)*dsin(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dsin(beta)**2) * dsiga * mtt**2 * 3.d0
c
      call genhquad (3,s,mdn,mdn,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17d = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mdn**2 * 3.d0
c
      call genhquad (3,s,mst,mst,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17e = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mst**2 * 3.d0
c
      call genhquad (3,s,mbb,mbb,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17f = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mbb**2 * 3.d0
c
      call genhquad (3,s,mel,mel,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17g = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mel**2
c
      call genhquad (3,s,mmu,mmu,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17h = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mmu**2
c
      call genhquad (3,s,mta,mta,1.d0,1.d0,1.d0,1.d0, siga,dsiga)
      top17i = elec2/(4.d0*ppi)**2 * dsin(alpha)*dcos(alpha) /(4.d0 *
     &       ssw2 * mmw**2*dcos(beta)**2) * dsiga * mta**2
c
      top17 = top17a + top17b + top17c + top17d + top17e + top17f +
     &        top17g + top17h + top17i
c
      top21 = 0.d0
      do 620 ii = 1,2
       do 621 j = 1,2
      a = (vmix(j,1)*umix(ii,2)*dsin(alpha) -
     &     vmix(j,2)*umix(ii,1)*dcos(alpha) )/dsqrt(2.d0)
      b = (vmix(ii,1)*umix(j,2)*dsin(alpha) -
     &     vmix(ii,2)*umix(j,1)*dcos(alpha) )/dsqrt(2.d0)
      at = (vmix(j,1)*umix(ii,2)*dcos(alpha) +
     &      vmix(j,2)*umix(ii,1)*dsin(alpha) )/dsqrt(2.d0)
      bt = (vmix(ii,1)*umix(j,2)*dcos(alpha) +
     &      vmix(ii,2)*umix(j,1)*dsin(alpha) )/dsqrt(2.d0)
      call genhquad (4,s,mcha(ii),mcha(j),a,bt,b,at, siga,dsiga)
      top21h =  elec2/(4.d0*ppi)**2 /(4.d0*ssw2) * dsiga
      top21 = top21 + top21h
621   continue
620   continue
c
      top22 = 0.d0
      do 622 ii = 1,4
       do 623 j = 1,4
      qm = (nmix(ii,3)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,3)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      sm = (nmix(ii,4)*(nmix(j,2)-nmix(j,1)*ssw/ccw) +
     &      nmix(j,4)*(nmix(ii,2)-nmix(ii,1)*ssw/ccw) )/2.d0
      at = qm * dsin(alpha) + sm * dcos(alpha)
      a =  qm * dcos(alpha) - sm * dsin(alpha)
      call genhquad (4,s,mne(ii),mne(j),a,at,a,at, siga,dsiga)
      top22h = elec2/(4.d0*ppi)**2 /(8.d0*ssw2) * dsiga
      top22 = top22 + top22h
623   continue
622   continue
c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' h0-H0 derivative - mixing : ', real(dsqrt(s))
       write (*,*) ' W+ H-       = ', top1
       write (*,*) ' Z0 A        = ', top2
       write (*,*) ' Z0 G0       = ', top3
       write (*,*) ' W+ G-       = ', top4
       write (*,*) ' W+ W-       = ', top5
       write (*,*) ' Z0 Z0       = ', top6
       write (*,*) ' H+ H-       = ', top7
       write (*,*) ' h0 h0       = ', top8
       write (*,*) ' H0 h0       = ', top9
       write (*,*) ' H0 H0       = ', top10
       write (*,*) ' A  A        = ', top11
       write (*,*) ' G+ G-       = ', top12
       write (*,*) ' G0 G0       = ', top13
       write (*,*) ' G0 A0       = ', top14
       write (*,*) ' G+ H-       = ', top15
       write (*,*) ' Gh+ Gh-     = ', top18
       write (*,*) ' Ghz Ghz     = ', top20
       write (*,*) ' fer fer     = ', top17
       write (*,*) ' sfer sfer   = ', top23
       write (*,*) ' cha cha     = ', top21
       write (*,*) ' neu neu     = ', top22
       write (*,*) '   '
      endif
c
      dsigmahhb =top1 + top2 + top3 + top4 + top5 + top6 + top7 + top8 +
     &            top9 + top10 + top11 + top12 + top13 + top14 + top15 +
     &            top18 + top20
      dsigmahhs = top23
      dsigmahhf = top17
      dsigmahhc = top21 + top22
      dsigmahht = top17c + top23c 
c
      return
      end
c=====================================================================

      subroutine gentadH (typ,m1,a,b, siga)
c
      implicit double precision (a-z)
      complex*16 aa
      integer typ
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
c     common /renpara/xo,zo,mgll

c
c                               .
c                            .     .  m1
c                           .       .
c                            .     .
c                               .
c                               .
c                               .  h, H
c                               .
c
      if (typ.eq.1) then
	siga  = 4.d0 * aa(dabs(m1)) - 2.d0 * m1**2
c       siga  = 8.d0 * m1**2 / epsilon
      else
      if (typ.eq.2) then
	siga  = - aa(dabs(m1))
c       siga  = -2.d0 * m1**2 / epsilon
      else
      if (typ.eq.3) then
	siga  = 4.d0 * m1 * aa(dabs(m1))
c       siga  = 8.d0 * m1**3 / epsilon
      else
      if (typ.eq.4) then
	siga  = aa(dabs(m1))
c       siga  = 2.d0 * m1**2 / epsilon
      else
      if (typ.eq.5) then
	siga  = 4.d0 * (a + b) * m1 * aa(dabs(m1))
c       siga  = 8.d0 * (a + b) * m1 * m1**2 / epsilon
      else
       write (*,*) ' typpindent wrong '
      endif
      endif
      endif
      endif
      endif
c
      return
      end
c
c ----------------------------------------------------------
c
      subroutine tadhh (s, sigmahtb,sigmahts,sigmahtf,sigmahtc,sigmahtt)
c
c     tadpol of the heavy higgs boson
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer ii,pr
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

c
c boson loops
c
c  notation :
c     gentadH (typ,m1,a,b, siga)
c
      call gentadH (1,mmw,1.d0,1.d0, siga)
      top1 = -elec/(4.d0*ppi)**2/ssw * mmw * dcos(beta-alpha) * siga
c
      call gentadH (1,mmz,1.d0,1.d0, siga)
      top2 = -elec/(4.d0*ppi)**2/ssw 
     $     * mmz/ccw*dcos(beta-alpha)*siga/2.d0
c
      call gentadH (2,mhp,1.d0,1.d0, siga)
      top3 = (mmw*dcos(beta-alpha) - mmz /(2.d0*ccw) * dcos(2.d0*beta) *
     &    dcos(beta+alpha)) * siga * elec/(4.d0*ppi)**2/ssw
c
      call gentadH (2,mhh,1.d0,1.d0, siga)
      top4a = elec/(4.d0*ppi)**2/ssw * 3.d0 * mmz * dcos(2.d0*alpha) *
     &    dcos(beta+alpha) /(4.d0*ccw) * siga
c
      call gentadH (2,mlh,1.d0,1.d0, siga)
      top4b = elec/(4.d0*ppi)**2/ssw * mmz /(4.d0*ccw) * ( 2.d0 *
     &        dsin(2.d0*alpha) * dsin(alpha+beta) - dcos(beta+alpha) *
     &        dcos(2.d0*alpha) ) * siga
c
      call gentadH (2,maa,1.d0,1.d0, siga)
      top4c = -elec/(4.d0*ppi)**2/ssw * mmz * dcos(2.d0*beta) *
     &    dcos(beta+alpha)/(4.d0*ccw) * siga
c
      top4 = top4a + top4b + top4c
c
      call gentadH (2,mmw,1.d0,1.d0, siga)
      top5 = elec/(4.d0*ppi)**2/ssw * mmz * dcos(2.d0*beta) * dcos(
     &       beta + alpha) /(2.d0 * ccw) * siga
c
      call gentadH (2,mmz,1.d0,1.d0, siga)
      top6 = elec/(4.d0*ppi)**2/ssw  * mmz * dcos(2.d0*beta) * dcos(
     &       beta + alpha) /(4.d0 * ccw) * siga
c
      call gentadH (3,mup,1.d0,1.d0, siga)
      top7a = elec/(4.d0*ppi)**2/ssw  * mup * dsin(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mch,1.d0,1.d0, siga)
      top7b = elec/(4.d0*ppi)**2/ssw  * mch * dsin(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mtt,1.d0,1.d0, siga)
      top7c = elec/(4.d0*ppi)**2/ssw  * mtt * dsin(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mdn,1.d0,1.d0, siga)
      top7d = elec/(4.d0*ppi)**2/ssw  * mdn * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
c
      call gentadH (3,mst,1.d0,1.d0, siga)
      top7e = elec/(4.d0*ppi)**2/ssw  * mst * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
c
      if (delmbresum.eq.1) then
      call gentadH (3,mbb,1.d0,1.d0, siga)
      top7f = elec/(4.d0*ppi)**2/ssw  * mbb * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
      else
      call gentadH (3,mbb,1.d0,1.d0, siga)
      top7f = elec/(4.d0*ppi)**2/ssw/(2.d0 * mmw) * siga * 3.d0 *
     $        mbbdmb * (dcos(alpha)/dcos(beta) 
     $                  + dmb * dsin(alpha)/dsin(beta)) 
      endif
c
      call gentadH (3,mel,1.d0,1.d0, siga)
      top7g = elec/(4.d0*ppi)**2/ssw  * mel * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      call gentadH (3,mmu,1.d0,1.d0, siga)
      top7h = elec/(4.d0*ppi)**2/ssw  * mmu * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      call gentadH (3,mta,1.d0,1.d0, siga)
      top7i = elec/(4.d0*ppi)**2/ssw  * mta * dcos(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      if (selec.ge.3) then
      top7 = top7a + top7b + top7c + top7d + top7e + top7f +
     &       top7g + top7h + top7i
      elseif (selec.eq.2) then
         top7 = top7c + top7f
      elseif (selec.eq.1) then
         top7 = top7c
      else
         write(*,*) "Error in Tadpole-H: selec out or range"
      endif

c
      call gentadH (4,mmw,1.d0,1.d0, siga)
      top9 = elec/(4.d0*ppi)**2/ssw  * mmw * dcos(beta-alpha) *
     &       siga
c
      call gentadH (4,mmz,1.d0,1.d0, siga)
      top10 = elec/(4.d0*ppi)**2/ssw  * mmz * dcos(beta-alpha)/2.d0 *
     &        siga / ccw
c
      top11 = 0.d0
      do 580 ii = 1,2
      q11 = vmix(ii,1) * umix(ii,2)/ dsqrt(2.d0)
      s11 = vmix(ii,2) * umix(ii,1)/ dsqrt(2.d0)
      a = q11 * dcos(alpha) + s11 * dsin(alpha)
      b = a
      call gentadH (5,mcha(ii),a,b, siga)
      top11h = elec/(4.d0*ppi)**2/ssw  * siga / 2.d0
      top11 = top11 + top11h
580   continue
c
      top12 = 0.d0
      do 590 ii = 1,4
      q11 = nmix(ii,3) * (nmix(ii,2)-nmix(ii,1) * ssw/ccw)
      s11 = nmix(ii,4) * (nmix(ii,2)-nmix(ii,1) * ssw/ccw)
      a = q11 * dcos(alpha) - s11 * dsin(alpha)
      b = a
      call gentadH (5,mne(ii),a,b, siga)
      top12h = elec/(4.d0*ppi)**2/ssw  * siga / 4.d0
      top12 = top12 + top12h
590   continue
c
      a11 =  (mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &         ssw2) * dcos(alpha+beta) + mup**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 2.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mup**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a12 =  mup/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call gentadH (2,mupsl,1.d0,1.d0, siga1)
      call gentadH (2,mupsr,1.d0,1.d0, siga2)
      top13a = elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang7)**2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*
     &      ang7))* siga1 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2
     &      - a12*dsin(2.d0*ang7))* siga2 )
c
      a11 =  (mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &         ssw2) * dcos(alpha+beta) + mch**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 2.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mch**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a12 =  mch/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call gentadH (2,mchsl,1.d0,1.d0, siga1)
      call gentadH (2,mchsr,1.d0,1.d0, siga2)
      top13b = elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang9)**2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*
     &      ang9))* siga1 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2
     &      - a12*dsin(2.d0*ang9))* siga2 ) 
c
      a11 =  (mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &         ssw2) * dcos(alpha+beta) + mtt**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 2.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mtt**2/(mmw*dsin(beta)) *
     &         dsin(alpha) )
      a12 =  mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &       (mu*dcos(alpha) + mssupq * dsin(alpha))
c
      call gentadH (2,mtsl,1.d0,1.d0, siga1)
      call gentadH (2,mtsr,1.d0,1.d0, siga2)
      top13c = elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang11)**2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*
     &      ang11))* siga1 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2
     &      - a12*dsin(2.d0*ang11))* siga2 )
c      write(*,*) 'top13c:', top13c
c      write(*,*) real(a11), real(a22), real(a12)
c      write(*,*) real(dcos(ang11)), real(dsin(ang11)), 
c     $           real(alpha), real(beta)
c      write(*,*) siga1, siga2
c      write(*,*) real(mtt), real(mmz), real(ccw), real(ssw), 
c     $           real(mu), real(mssupq), real(elec), real(ppi)
c      write(*,*) real(elec/(4.d0*ppi)**2/ssw * 3.d0),
c     $           real(a11*dcos(ang11)**2 + a22*dsin(ang11)**2 + 
c     $                a12*dsin(2.d0*ang11)), 
c     $           real(a11*dsin(ang11)**2 + a22*dcos(ang11)**2
c     &              - a12*dsin(2.d0*ang11))
c      write(*,*) real((a11*dcos(ang11)**2 + a22*dsin(ang11)**2 + 
c     $                a12*dsin(2.d0*ang11)) * siga1),
c     $           real((a11*dsin(ang11)**2 + a22*dcos(ang11)**2
c     &              - a12*dsin(2.d0*ang11)) * siga2)

      a11 = (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dcos(alpha+beta) - mdn**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mdn**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 =  -mdn/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call gentadH (2,mdnsl,1.d0,1.d0, siga1)
      call gentadH (2,mdnsr,1.d0,1.d0, siga2)
      top13d = -elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang8)**2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*
     &      ang8))* siga1 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2
     &      - a12*dsin(2.d0*ang8))* siga2 )
c
      a11 = (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dcos(alpha+beta) - mst**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mst**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = -mst/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call gentadH (2,mstsl,1.d0,1.d0, siga1)
      call gentadH (2,mstsr,1.d0,1.d0, siga2)
      top13e = -elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang10)**2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*
     &      ang10))* siga1 + (a11*dsin(ang10)**2 + a22*dcos(ang10)**2
     &      - a12*dsin(2.d0*ang10))* siga2 )
c
      if (delmbresum.eq.1) then
      a11 = (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dcos(alpha+beta) - mbb**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mbb**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = - mbb/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call gentadH (2,mbsl,1.d0,1.d0, siga1)
      call gentadH (2,mbsr,1.d0,1.d0, siga2)
      top13f = -elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang12)**2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*
     &      ang12))* siga1 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2
     &      - a12*dsin(2.d0*ang12))* siga2 )
      else
      a11 = (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dcos(alpha+beta) - mbbdmb**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dcos(alpha + beta) + mbbdmb**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = - mbbdmb/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnq * dcos(alpha))
c
      call gentadH (2,msb1dmb,1.d0,1.d0, siga1)
      call gentadH (2,msb2dmb,1.d0,1.d0, siga2)
      top13f = -elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(tsbdmb)**2 + a22*dsin(tsbdmb)**2 + a12*dsin(2.d0*
     &      tsbdmb))* siga1 + (a11*dsin(tsbdmb)**2 
     $     + a22*dcos(tsbdmb)**2
     &      - a12*dsin(2.d0*tsbdmb))* siga2 )
      endif
c
      a11 = (mmz/ccw * (0.5d0 - 1.d0*
     &         ssw2) * dcos(alpha+beta) - mel**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0 * ssw2 *
     &         dcos(alpha + beta) + mel**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = - mel/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call gentadH (2,melsl,1.d0,1.d0, siga1)
      call gentadH (2,melsr,1.d0,1.d0, siga2)
      top13g = -elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang2)**2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*
     &      ang2))* siga1 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2
     &      - a12*dsin(2.d0*ang2))* siga2 )
c
      a11 = (mmz/ccw * (0.5d0 - 1.d0*
     &         ssw2) * dcos(alpha+beta) - mmu**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0 * ssw2 *
     &         dcos(alpha + beta) + mmu**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = - mmu/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call gentadH (2,mmusl,1.d0,1.d0, siga1)
      call gentadH (2,mmusr,1.d0,1.d0, siga2)
      top13h = -elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang4)**2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*
     &      ang4))* siga1 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2
     &      - a12*dsin(2.d0*ang4))* siga2 )
c
      a11 = (mmz/ccw * (0.5d0 - 1.d0*
     &         ssw2) * dcos(alpha+beta) - mta**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a22 = -(-mmz/ccw * 1.d0 * ssw2 *
     &         dcos(alpha + beta) + mta**2/(mmw*dcos(beta)) *
     &         dcos(alpha) )
      a12 = - mta/(2.d0*
     &       mmw * dcos(beta)) * 
     &      (mu*dsin(alpha) + mssdnl * dcos(alpha))
c
      call gentadH (2,mtasl,1.d0,1.d0, siga1)
      call gentadH (2,mtasr,1.d0,1.d0, siga2)
      top13i = -elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang6)**2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*
     &      ang6))* siga1 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2
     &      - a12*dsin(2.d0*ang6))* siga2 )
c
      call gentadH (2,mvesl,1.d0,1.d0, siga)
      top13j = elec/(4.d0*ppi)**2/ssw  * (mmz/ccw * 0.5d0 *
     &         dcos(alpha+beta) ) * siga
c
      call gentadH (2,mvmsl,1.d0,1.d0, siga)
      top13k = elec/(4.d0*ppi)**2/ssw  * (mmz/ccw * 0.5d0 *
     &         dcos(alpha+beta) ) * siga
c
      call gentadH (2,mvtsl,1.d0,1.d0, siga)
      top13l = elec/(4.d0*ppi)**2/ssw  * (mmz/ccw * 0.5d0 *
     &         dcos(alpha+beta) ) * siga
c
      if (selec.ge.3) then
      top13 = top13a + top13b + top13c + top13d + top13e + top13f +
     &        top13g + top13h + top13i + top13j + top13k + top13l
      elseif (selec.eq.2) then
         top13 = top13c + top13f
      elseif (selec.eq.1) then
         top13 = top13c
c         write(*,*) 'tadhh top13:', real(top13c)
      else
         write(*,*) "Error in Tadpole-H: selec out or range"
      endif

c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' H0 - Tadpole : '
       write (*,*) ' W+        = ', top1
       write (*,*) ' Z0        = ', top2
       write (*,*) ' H+        = ', top3
       write (*,*) ' h0        = ', top4a
       write (*,*) ' H0        = ', top4b
       write (*,*) ' A         = ', top4c
       write (*,*) ' G+        = ', top5
       write (*,*) ' G0        = ', top6
       write (*,*) ' fer       = ', top7
       write (*,*) ' Gh+       = ', top9
       write (*,*) ' Ghz       = ', top10
       write (*,*) ' sfer      = ', top13
       write (*,*) ' ch        = ', top11
       write (*,*) ' ne        = ', top12
      endif
c
      sigmahtb = top1 + top2 + top3 + top4 + top5 + top6 + top9 +
     &           top10
      sigmahts = top13
      sigmahtf = top7
      sigmahtc = top11 + top12
      sigmahtt = top7c + top13c
c      write(*,*) 'tadH:', sigmahts, sigmahtf
c
      return
      end


c=====================================================================
c
      subroutine tadlh (s, sigmaltb,sigmalts,sigmaltf,sigmaltc,sigmaltt)
c
c     tadpol of the light higgs boson
c
      implicit double precision (a-z)
      real*8 umix(1:2,1:2),vmix(1:2,1:2),nmix(1:4,1:4)
      real*8 mcha(1:2),mne(1:4)
      complex*16 aa
      integer ii,pr
      integer pri,naeh,selec,selec2,selec4,selec5,selec6
c
      common /param/ ssw2,ssw,ccw2,ccw,ppi,elec2,elec,mmz,mmz2,mmw2,mmw,
     &               beta,alpha
      common /singl/ epsilon,muee,lambda
      common /susyset/ mu,mm,mp
      common /mass/ mel,mmu,mta,mup,mdn,mch,mst,mbb,mbb2,mtt,mtt2,
     &              melsl,mmusl,mtasl,mupsl,mvesl,mvmsl,mvtsl,
     &              mdnsl,mstsl,mchsl,mtsl,mbsl,mhh,mlh,maa,mhp,
     &              melsr,mmusr,mtasr,mupsr,mvesr,mvmsr,mvtsr,
     &              mdnsr,mstsr,mchsr,mtsr,mbsr, mcha,mne
      common /mixing/ umix,vmix,nmix
      common /fangle/ ang1,ang2,ang3,ang4,ang5,ang6,ang7,ang8,ang9,
     &                ang10,ang11,ang12
      common /abreak/mssupq,mssdnq,mssdnl
      common /break/ mq2,mu2,mb2,md2,mf2,mfd2
c     common /renpara/xo,zo,mgll
      common /print/pri,naeh,selec,selec2,selec4,selec5,selec6
      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

c
c  notation :
c     gentadH (typ,m1,a,b, siga)
c
      call gentadH (1,mmw,1.d0,1.d0, siga)
      top1 = -elec/(4.d0*ppi)**2/ssw * mmw * dsin(beta-alpha) * siga
c
      call gentadH (1,mmz,1.d0,1.d0, siga)
      top2 = -elec/(4.d0*ppi)**2/ssw/2.d0/ccw*mmz*dsin(beta-alpha)*siga
c
      call gentadH (2,mhp,1.d0,1.d0, siga)
      top3 = (mmw*dsin(beta-alpha) + mmz/(2.d0*ccw) * dcos(2.d0*beta) *
     &        dsin(beta+alpha)) * siga * elec/(4.d0*ppi)**2/ssw
c
      call gentadH (2,mlh,1.d0,1.d0, siga)
      top4a = elec/(4.d0*ppi)**2/ssw * 3.d0 * mmz * dcos(2.d0*alpha) *
     &        dsin(beta+alpha) /(4.d0*ccw) * siga
c
      call gentadH (2,mhh,1.d0,1.d0, siga)
      top4b = -elec/(4.d0*ppi)**2/ssw * mmz /(4.d0 * ccw) * ( 2.d0 *
     &        dsin(2.d0*alpha) * dcos(alpha+beta) + dsin(beta+alpha) *
     &        dcos(2.d0*alpha) ) * siga
c
      call gentadH (2,maa,1.d0,1.d0, siga)
      top4c = elec/(4.d0*ppi)**2/ssw * mmz * dcos(2.d0*beta) *
     &        dsin(beta+alpha)/(4.d0*ccw) * siga
c
      top4 = top4a + top4b + top4c
c
      call gentadH (2,mmw,1.d0,1.d0, siga)
      top5 = -elec/(4.d0*ppi)**2/ssw * mmz * dcos(2.d0*beta) * dsin
     &       (beta+alpha) /(2.d0*ccw) * siga
c
      call gentadH (2,mmz,1.d0,1.d0, siga)
      top6 = -elec/(4.d0*ppi)**2/ssw * mmz * dcos(2.d0*beta) * dsin
     &        (beta + alpha) /(4.d0*ccw) * siga
c
      call gentadH (3,mup,1.d0,1.d0, siga)
      top7a = elec/(4.d0*ppi)**2/ssw * mup * dcos(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mch,1.d0,1.d0, siga)
      top7b = elec/(4.d0*ppi)**2/ssw * mch * dcos(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mtt,1.d0,1.d0, siga)
      top7c = elec/(4.d0*ppi)**2/ssw * mtt * dcos(alpha) /(2.d0 *
     &        mmw * dsin(beta)) * siga * 3.d0
c
      call gentadH (3,mdn,1.d0,1.d0, siga)
      top7d = -elec/(4.d0*ppi)**2/ssw * mdn * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
c
      call gentadH (3,mst,1.d0,1.d0, siga)
      top7e = -elec/(4.d0*ppi)**2/ssw * mst * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
c
      if (delmbresum.eq.1) then
      call gentadH (3,mbb,1.d0,1.d0, siga)
      top7f = -elec/(4.d0*ppi)**2/ssw * mbb * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga * 3.d0
      else
      call gentadH (3,mbb,1.d0,1.d0, siga)
      top7f = -elec/(4.d0*ppi)**2/ssw/(2.d0 * mmw) * siga * 3.d0 *
     $        mbbdmb * (dsin(alpha)/dcos(beta) 
     $                  - dmb * dcos(alpha)/dsin(beta))
      endif
c
      call gentadH (3,mel,1.d0,1.d0, siga)
      top7g = -elec/(4.d0*ppi)**2/ssw * mel * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      call gentadH (3,mmu,1.d0,1.d0, siga)
      top7h = -elec/(4.d0*ppi)**2/ssw * mmu * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      call gentadH (3,mta,1.d0,1.d0, siga)
      top7i = -elec/(4.d0*ppi)**2/ssw * mta * dsin(alpha) /(2.d0 *
     &        mmw * dcos(beta)) * siga
c
      if (selec.ge.3) then
      top7 = top7a + top7b + top7c + top7d + top7e + top7f +
     &       top7g + top7h + top7i
      elseif (selec.eq.2) then
         top7 = top7c + top7f
      elseif (selec.eq.1) then
         top7 = top7c
      else
         write(*,*) "Error in Tadpole-h: selec out or range"
      endif

c
      call gentadH (4,mmw,1.d0,1.d0, siga)
      top9 = elec/(4.d0*ppi)**2/ssw * mmw * dsin(beta-alpha) * siga
c
      call gentadH (4,mmz,1.d0,1.d0, siga)
      top10 = elec/(4.d0*ppi)**2/ssw * mmz * dsin(beta-alpha)/2.d0 *
     &        siga / ccw
c
      top11 = 0.d0
      do 760 ii = 1,2
      qx = vmix(ii,1) * umix(ii,2) / dsqrt(2.d0)
      sx = vmix(ii,2) * umix(ii,1) / dsqrt(2.d0)
      a = qx * dsin(alpha) - sx * dcos(alpha)
      b = a
      call gentadH (5,mcha(ii),a,b, siga)
      top11h = -elec/(4.d0*ppi)**2/ssw * siga / 2.d0
      top11 = top11 + top11h
760   continue
c
      top12 = 0.d0
      do 770 ii = 1,4
      qx = nmix(ii,3) * ( nmix(ii,2) - nmix(ii,1) * ssw/ccw )
      sx = nmix(ii,4) * ( nmix(ii,2) - nmix(ii,1) * ssw/ccw )
      a = qx * dsin(alpha) + sx * dcos(alpha)
      b = a
      call gentadH (5,mne(ii),a,b, siga)
      top12h = -elec/(4.d0*ppi)**2/ssw * siga / 4.d0
      top12 = top12 + top12h
770   continue
c
      a11 = (-mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &       ssw2) * dsin(alpha+beta) + mup**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a22 = (-mmz/ccw * 2.d0/3.d0 * ssw2 *
     &       dsin(alpha + beta) + mup**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a12 =  -mup/(2.d0*
     &       mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
c
      call gentadH (2,mupsl,1.d0,1.d0, siga1)
      call gentadH (2,mupsr,1.d0,1.d0, siga2)
      top13a = elec/(4.d0*ppi)**2/ssw * 3.d0 * ((
     &      a11*dcos(ang7)**2 + a22*dsin(ang7)**2 + a12*dsin(2.d0*
     &      ang7))* siga1 + (a11*dsin(ang7)**2 + a22*dcos(ang7)**2
     &      - a12*dsin(2.d0*ang7))* siga2 )
c
      a11 = (-mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &       ssw2) * dsin(alpha+beta) + mch**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a22 = (-mmz/ccw * 2.d0/3.d0 * ssw2 *
     &       dsin(alpha + beta) + mch**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a12 =  -mch/(2.d0*
     &       mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
c
      call gentadH (2,mchsl,1.d0,1.d0, siga1)
      call gentadH (2,mchsr,1.d0,1.d0, siga2)
      top13b = elec/(4.d0*ppi)**2/ssw  * 3.d0 *  ((
     &      a11*dcos(ang9)**2 + a22*dsin(ang9)**2 + a12*dsin(2.d0*
     &      ang9))* siga1 + (a11*dsin(ang9)**2 + a22*dcos(ang9)**2
     &      - a12*dsin(2.d0*ang9))* siga2 )
c
      a11 = (-mmz/ccw * (0.5d0 - 2.d0/3.d0 *
     &       ssw2) * dsin(alpha+beta) + mtt**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a22 = (-mmz/ccw * 2.d0/3.d0 * ssw2 *
     &       dsin(alpha + beta) + mtt**2/(mmw*dsin(beta)) *
     &       dcos(alpha) )
      a12 =  -mtt/(2.d0*
     &       mmw * dsin(beta)) *
     &      (mu*dsin(alpha) - mssupq * dcos(alpha))
c
      call gentadH (2,mtsl,1.d0,1.d0, siga1)
      call gentadH (2,mtsr,1.d0,1.d0, siga2)
      top13c = elec/(4.d0*ppi)**2/ssw  * 3.d0 *  ((
     &      a11*dcos(ang11)**2 + a22*dsin(ang11)**2 + a12*dsin(2.d0*
     &      ang11))* siga1 + (a11*dsin(ang11)**2 + a22*dcos(ang11)**2
     &      - a12*dsin(2.d0*ang11))* siga2 )
c      write(*,*) 'top13c:', top13c
c      write(*,*) a11, a22, a12
c      write(*,*) dcos(ang11), dsin(ang11), siga1, siga2
c
      a11 =  (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dsin(alpha+beta) - mdn**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dsin(alpha + beta) - mdn**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mdn/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnq*
     &       dsin(alpha))
c
      call gentadH (2,mdnsl,1.d0,1.d0, siga1)
      call gentadH (2,mdnsr,1.d0,1.d0, siga2)
      top13d = elec/(4.d0*ppi)**2/ssw * 3.d0 *  ((
     &      a11*dcos(ang8)**2 + a22*dsin(ang8)**2 + a12*dsin(2.d0*
     &      ang8))* siga1 + (a11*dsin(ang8)**2 + a22*dcos(ang8)**2
     &      - a12*dsin(2.d0*ang8))* siga2 )
c
      a11 =  (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dsin(alpha+beta) - mst**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dsin(alpha + beta) - mst**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mst/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnq*
     &       dsin(alpha))
c
      call gentadH (2,mstsl,1.d0,1.d0, siga1)
      call gentadH (2,mstsr,1.d0,1.d0, siga2)
      top13e = elec/(4.d0*ppi)**2/ssw * 3.d0 *  ((
     &    a11*dcos(ang10)**2 + a22*dsin(ang10)**2 + a12*dsin(2.d0*
     &    ang10))* siga1 + (a11*dsin(ang10)**2 + a22*dcos(ang10)**2
     &    - a12*dsin(2.d0*ang10))* siga2 )
c
      if (delmbresum.eq.1) then
      a11 =  (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dsin(alpha+beta) - mbb**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dsin(alpha + beta) - mbb**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mbb/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnq*
     &       dsin(alpha))
c
      call gentadH (2,mbsl,1.d0,1.d0, siga1)
      call gentadH (2,mbsr,1.d0,1.d0, siga2)
      top13f = elec/(4.d0*ppi)**2/ssw  * 3.d0 *  ((
     &      a11*dcos(ang12)**2 + a22*dsin(ang12)**2 + a12*dsin(2.d0*
     &      ang12))* siga1 + (a11*dsin(ang12)**2 + a22*dcos(ang12)**2
     &      - a12*dsin(2.d0*ang12))* siga2 )
      else
      a11 =  (mmz/ccw * (0.5d0 - 1.d0/3.d0*
     &         ssw2) * dsin(alpha+beta) - mbbdmb**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * 1.d0/3.d0 * ssw2 *
     &         dsin(alpha + beta) - mbbdmb**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mbbdmb/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnq*
     &       dsin(alpha))
c
      call gentadH (2,msb1dmb,1.d0,1.d0, siga1)
      call gentadH (2,msb2dmb,1.d0,1.d0, siga2)
      top13f = elec/(4.d0*ppi)**2/ssw  * 3.d0 *  ((
     &      a11*dcos(tsbdmb)**2 + a22*dsin(tsbdmb)**2 + a12*dsin(2.d0*
     &      tsbdmb))* siga1 + (a11*dsin(tsbdmb)**2 
     $     + a22*dcos(tsbdmb)**2
     &      - a12*dsin(2.d0*tsbdmb))* siga2 )
      endif
c
      a11 =  (mmz/ccw * (0.5d0 - 
     &         ssw2) * dsin(alpha+beta) - mel**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw *  ssw2 *
     &         dsin(alpha + beta) - mel**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 =  mel/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnl*
     &       dsin(alpha))
c
      call gentadH (2,melsl,1.d0,1.d0, siga1)
      call gentadH (2,melsr,1.d0,1.d0, siga2)
      top13g = elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang2)**2 + a22*dsin(ang2)**2 + a12*dsin(2.d0*
     &      ang2))* siga1 + (a11*dsin(ang2)**2 + a22*dcos(ang2)**2
     &      - a12*dsin(2.d0*ang2))* siga2 )
c
      a11 =  (mmz/ccw * (0.5d0 - 
     &         ssw2) * dsin(alpha+beta) - mmu**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * ssw2 *
     &         dsin(alpha + beta) - mmu**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mmu/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnl*
     &       dsin(alpha))
c
      call gentadH (2,mmusl,1.d0,1.d0, siga1)
      call gentadH (2,mmusr,1.d0,1.d0, siga2)
      top13h = elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang4)**2 + a22*dsin(ang4)**2 + a12*dsin(2.d0*
     &      ang4))* siga1 + (a11*dsin(ang4)**2 + a22*dcos(ang4)**2
     &      - a12*dsin(2.d0*ang4))* siga2 )
c
      a11 =  (mmz/ccw * (0.5d0 - 
     &         ssw2) * dsin(alpha+beta) - mta**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a22 =  (mmz/ccw * ssw2 *
     &         dsin(alpha + beta) - mta**2/(mmw*dcos(beta)) *
     &         dsin(alpha) )
      a12 = mta/(2.d0*mmw*dcos(beta)) * (mu*dcos(alpha) - mssdnl*
     &       dsin(alpha))
c
      call gentadH (2,mtasl,1.d0,1.d0, siga1)
      call gentadH (2,mtasr,1.d0,1.d0, siga2)
      top13i = elec/(4.d0*ppi)**2/ssw *  ((
     &      a11*dcos(ang6)**2 + a22*dsin(ang6)**2 + a12*dsin(2.d0*
     &      ang6))* siga1 + (a11*dsin(ang6)**2 + a22*dcos(ang6)**2
     &      - a12*dsin(2.d0*ang6))* siga2 )
c
      call gentadH (2,mvesl,1.d0,1.d0, siga)
      top13j = elec/(4.d0*ppi)**2/ssw  * (-mmz/ccw * 0.5d0 *
     &         dsin(alpha+beta) ) * siga
c
      call gentadH (2,mvmsl,1.d0,1.d0, siga)
      top13k = elec/(4.d0*ppi)**2/ssw  * (-mmz/ccw * 0.5d0 *
     &         dsin(alpha+beta) ) * siga
c
      call gentadH (2,mvtsl,1.d0,1.d0, siga)
      top13l = elec/(4.d0*ppi)**2/ssw  * (-mmz/ccw * 0.5d0 *
     &         dsin(alpha+beta) ) * siga
c
      if (selec.ge.3) then
      top13 = top13a + top13b + top13c + top13d + top13e + top13f +
     &        top13g + top13h + top13i + top13j + top13k + top13l
      elseif (selec.eq.2) then
         top13 = top13c + top13f
      elseif (selec.eq.1) then
         top13 = top13c
      else
         write(*,*) "Error in Tadpole-h: selec out or range"
      endif

c
c printroutine
c
      pr = 0
c
      if(pr.eq.1) then
       write (*,*) ' h0 - Tadpole : '
       write (*,*) ' W+        = ', top1
       write (*,*) ' Z0        = ', top2
       write (*,*) ' H+        = ', top3
       write (*,*) ' h0        = ', top4b
       write (*,*) ' H0        = ', top4a
       write (*,*) ' A         = ', top4c
       write (*,*) ' G+        = ', top5
       write (*,*) ' G0        = ', top6
       write (*,*) ' fer       = ', top7
       write (*,*) ' Gh+       = ', top9
       write (*,*) ' Ghz       = ', top10
       write (*,*) ' sfer      = ', top13
       write (*,*) ' cha       = ', top11
       write (*,*) ' neu       = ', top12
      endif
c
      sigmaltb = top1 + top2 + top3 + top4 + top5 + top6 + top9 +
     &           top10
      sigmalts = top13
      sigmaltf = top7
      sigmaltc = top11 + top12
      sigmaltt = top7c + top13c 
c
      return
      end


c=====================================================================
      SUBROUTINE JACOBI2(A,N,NP,D,V,NROT,flag)
      implicit double precision (a-h,o-z)
      PARAMETER (NMAX=100)
c      DIMENSION A(NP,NP),D(NP),V(NP,NP),B(NMAX),Z(NMAX)
      real*8 A(NP,NP),D(NP),V(NP,NP),B(NMAX),Z(NMAX)
      integer flag

      flag = 0

      DO 12 IP=1,N
        DO 11 IQ=1,N
	  V(IP,IQ)=0.d0
11      CONTINUE
	V(IP,IP)=1.d0
12    CONTINUE

      DO 13 IP=1,N
        B(IP)=A(IP,IP)
        D(IP)=B(IP)
	Z(IP)=0.d0
13    CONTINUE

      NROT=0
      DO 24 I=1,50
	SM=0.d0
        DO 15 IP=1,N-1
          DO 14 IQ=IP+1,N
            SM=SM+ABS(A(IP,IQ))
14        CONTINUE
15      CONTINUE

	IF(SM.EQ.0.d0)RETURN
        IF(I.LT.4)THEN
	  TRESH=0.2d0*SM/N**2
        ELSE
	  TRESH=0.d0
        ENDIF

        DO 22 IP=1,N-1
          DO 21 IQ=IP+1,N
	    G=100.d0*dABS(A(IP,IQ))
	    IF((I.GT.4).AND.(dABS(D(IP))+G.EQ.dABS(D(IP)))
     *         .AND.(dABS(D(IQ))+G.EQ.dABS(D(IQ))))THEN
	      A(IP,IQ)=0.d0
	    ELSE IF(dABS(A(IP,IQ)).GT.TRESH)THEN
              H=D(IQ)-D(IP)
	      IF(dABS(H)+G.EQ.dABS(H))THEN
                T=A(IP,IQ)/H
              ELSE
		THETA=0.5d0*H/A(IP,IQ)
		T=1.d0/(dABS(THETA)+dSQRT(1.d0+THETA**2))
		IF(THETA.LT.0.d0)T=-T
              ENDIF
	      C=(1.d0+T**2)**(-0.5d0)
              S=T*C
	      TAU=S/(1.d0+C)
              H=T*A(IP,IQ)
              Z(IP)=Z(IP)-H
              Z(IQ)=Z(IQ)+H
              D(IP)=D(IP)-H
              D(IQ)=D(IQ)+H
	      A(IP,IQ)=0.d0

              DO 16 J=1,IP-1
                G=A(J,IP)
                H=A(J,IQ)
                A(J,IP)=G-S*(H+G*TAU)
                A(J,IQ)=H+S*(G-H*TAU)
16            CONTINUE
              DO 17 J=IP+1,IQ-1
                G=A(IP,J)
                H=A(J,IQ)
                A(IP,J)=G-S*(H+G*TAU)
                A(J,IQ)=H+S*(G-H*TAU)
17            CONTINUE
              DO 18 J=IQ+1,N
                G=A(IP,J)
                H=A(IQ,J)
                A(IP,J)=G-S*(H+G*TAU)
                A(IQ,J)=H+S*(G-H*TAU)
18            CONTINUE
              DO 19 J=1,N
                G=V(J,IP)
                H=V(J,IQ)
                V(J,IP)=G-S*(H+G*TAU)
                V(J,IQ)=H+S*(G-H*TAU)
19            CONTINUE
              NROT=NROT+1
            ENDIF
21        CONTINUE
22      CONTINUE

        DO 23 IP=1,N
          B(IP)=B(IP)+Z(IP)
          D(IP)=B(IP)
	  Z(IP)=0.d0
23      CONTINUE
24    CONTINUE

      write (*,*) '50 iterations should never happen'
      flag = 1
      RETURN
      END


c----------------------------------------------------------------
	
	subroutine mtrunning(mt, mtrun, alpha3)
	
	double precision mt, mtrun, alpha3, pi, fac
	integer mtselec
	common /otherselec/ mtselec

	pi = 3.14159265358979d0

c	write(*,*) 'mtselec:', mtselec
	if (mtselec.eq.0) then
	   mtrun = mt/(1d0 + 4d0 *alpha3/(3d0*pi))
	elseif (mtselec.eq.1) then
c --> inconsistent routine from HZHA
	   fac = 16.11d0 - 1.04d0 * (4d0 - 6.5/mt)
	mtrun =  mt/(1d0 + 4d0 *alpha3/(3d0*pi) + fac * (alpha3/pi)**2)
	else
	   write(*,*) 'WARNING: mtselec out of range!!'
	   stop
	endif

c	write(*,*) 'mtrun:', real(mtrun)

	end

c----------------------------------------------------------------
      
      subroutine alphasmt(asmz,mt,mz,asmt)

      double precision asmz, asmt, mt, mz, pi

      pi = 3.14159265358979d0

      asmt = asmz/(1d0 + (11d0 - 10d0/3d0)/(4d0*pi) * asmz
     $                       * dlog(mt**2/mz**2))

      end

c----------------------------------------------------------------
      
      
      double precision function deltambnoew(als, 
     $                           mst1, mst2, msb1, msb2, 
     $                           at, ab, mgl, mt, mb, mu, tb)

      double precision als, mst1, mst2, msb1, msb2, mgl, mt, mb,
     $                 at, ab, mu, tb, ht, pi, t
      double precision Tdmb

      pi = 3.14159265358979d0

      ht = mt/(1d0 + 4d0 * als/(3d0 * pi)) / 174.1d0

c      write(*,*) 'deltamb:'
c      write(*,*) als, mst1, mst2, msb1, msb2, at, ab, mgl, 
c     $           mt, mb, mu, tb, ht

      deltambnoew = -2d0 * als/(3d0 * pi) * mgl * (ab - mu * tb) *
     $     Tdmb(msb1, msb2, mgl)
c     $     + ht**2/(4d0 * pi)**2 * (at - mu/tb) * mu * tb *
c     $     Tdmb(mst1,mst2,mu)

c      write(*,*) -2d0 * als/(3d0 * pi) * mgl * (ab - mu * tb),
c     $     T(msb1, msb2, mgl),
c     $     + ht**2/(4d0 * pi)**2 * (at - mu/tb) * mu * tb,
c     $     T(mst1, mst2, mu)

      end

c----------------------------------------------------------------
      
      double precision function Tdmb(x, y , z) 

      double precision x, y, z

      if(x.eq.y) x = x - 0.00001
      if(x.eq.z) x = x - 0.00002
      if(y.eq.z) y = y - 0.00003                

      Tdmb = (x**2*y**2*dlog(x**2/y**2) + x**2*z**2*dlog(z**2/x**2) 
     $     + y**2*z**2*dlog(y**2/z**2))
     $     /((x**2-y**2)*(y**2-z**2)*(x**2-z**2))

c       write(*,*) 'xyz', x, y, z, t

      end       



c----------------------------------------------------------------
c
c --> xybc.f
c
c----------------------------------------------------------------
      
C-----XYB22(K^2,M1,M2)--------------------------------------

      COMPLEX*16 FUNCTION XYB22(K2,M1,M2)

      IMPLICIT NONE
      COMPLEX*16 AA,xyB0
      DOUBLE PRECISION K2,M1,M2,M12,M22
      integer pri,naeh,mtn,dr,b22on, selec5, selec6

c      common/print/pri,naeh,mtn,dr,b22on

      M12=M1*M1
      M22=M2*M2

      if (b22on.eq.1) dr=0
      if (dr.eq.0) then
         IF (K2.GT.0D0) THEN
            XYB22=(.5D0*(AA(M1)+AA(M2))+(M12+M22-K2/2D0)
     $           *xyB0(K2,M1,M2)
     &           +(M12-M22)/(2.D0*K2)*(AA(M1)-AA(M2))
     &           -(M12-M22)**2/(2.D0*K2)*xyB0(K2,M1,M2)
     $           +M12+M22-K2/3.D0)/6.D0
         ELSE IF (M1.NE.M2) THEN
            XYB22=1/6.D0*(AA(M2)+2.D0*M12*xyB0(K2,M1,M2)+M12+M22
     &           -(M12-M22)/2.D0*xyB0(K2,M1,M2)-(M1**2+M2**2)/4.D0
     &           -M12*M22/(2.D0*(M12-M22))*(xyB0(K2,M1,M1)
     $           -xyB0(K2,M2,M2)))
         ELSE
            XYB22=(AA(M1)+2.D0*M12*xyB0(K2,M1,M2)+2.D0*M12)/6.D0
         ENDIF
      elseif (dr.eq.1) then
         IF (K2.GT.0D0) THEN
            XYB22=(.5D0*(AA(M1)+AA(M2))+(M12+M22-K2/2D0)
     $           *xyB0(K2,M1,M2)
     &           +(M12-M22)/(2.D0*K2)*(AA(M1)-AA(M2))
     &           -(M12-M22)**2/(2.D0*K2)*xyB0(K2,M1,M2))/6.D0
         ELSE IF (M1.NE.M2) THEN
            XYB22=1/6.D0*(AA(M2)+2.D0*M12*xyB0(K2,M1,M2)
     &           -(M12-M22)/2.D0*xyB0(K2,M1,M2)-(M1**2+M2**2)/4.D0
     &           -M12*M22/(2.D0*(M12-M22))*(xyB0(K2,M1,M1)
     $           -xyB0(K2,M2,M2)))
         ELSE
            XYB22=(AA(M1)+2.D0*M12*xyB0(K2,M1,M2)+2.D0*M12)/6.D0
         ENDIF
      else
         write (*,*) 'Fehler in XYB22'
      endif

      if (b22on.eq.1) dr=1
      END

C-----DELB22(K^2,M1^2,M2^2)--------------------------------

      COMPLEX*16 FUNCTION DELB22(K2,M12,M22)

      IMPLICIT NONE
      DOUBLE PRECISION K2,M12,M22

      DELB22=(3.D0*(M12+M22)-K2)/12.D0

      END



C-----B22S(K^2,M1,M2)--------------------------------------

      COMPLEX*16 FUNCTION B22S(K2,M1,M2)

      IMPLICIT NONE
      COMPLEX*16 xyB0,B0S,B0SS
      DOUBLE PRECISION K2,M1,M2,M12,M22
      integer pri,naeh,mtn,dr,b22on,selec5,selec6

c      common/print/pri,naeh,mtn,dr,b22on

      M12=M1*M1
      M22=M2*M2
      k2=0.d0

c      IF (K2.NE.0D0) THEN
c        WRITE(*,*) 'B22`(q^2 <> 0) KANN NICHT BERECHNET WERDEN'
c      ENDIF
      IF (K2.NE.0D0) GOTO 1850

      if (b22on.eq.1) dr=0
      if (dr.eq.0) then
         IF (M1.EQ.M2) GOTO 1830
         B22S=(2D0*M12*B0S(0D0,M1,M2)
     &        -xyB0(0D0,M1,M2)/2D0-(M12-M22)/2D0*B0S(0D0,M1,M2)
     &        -(M12-M22)/2D0*(B0S(0D0,M1,M2)+(M12-M22)*B0SS(0D0,M1,M2))
     &        -1D0/3D0)/6D0
         GOTO 1850
         
 1830    B22S=(2D0*M12*B0S(0D0,M1,M1)-xyB0(0D0,M1,M1)/2D0-1D0/3D0)/6D0
      elseif (dr.eq.1) then
         IF (M1.EQ.M2) GOTO 1840
         B22S=(2D0*M12*B0S(0D0,M1,M2)
     &        -xyB0(0D0,M1,M2)/2D0-(M12-M22)/2D0*B0S(0D0,M1,M2)
     &        -(M12-M22)/2D0*(B0S(0D0,M1,M2)+(M12-M22)*B0SS(0D0,M1,M2))
     &         )/6D0
         GOTO 1850

 1840    B22S=(2D0*M12*B0S(0D0,M1,M1)-xyB0(0D0,M1,M1)/2D0)/6D0
      endif

      if (b22on.eq.1) dr=1
1850  END


C-----xyB1(K^2,M1,M2)----------------------------------------
C     ACHTUNG: NUR SPEZIELLER FALL: K^2<>0, 0<>M1<>M2<>0

      COMPLEX*16 FUNCTION xyB1(K2,M1,M2)

      IMPLICIT NONE
      DOUBLE PRECISION K2,M1,M12,M2,M22
      COMPLEX*16 xyB0,AA

      M12=M1**2
      M22=M2**2

      XYB1=(AA(M1)-AA(M2)+(M22-M12-K2)*xyB0(K2,M1,M2))/(2D0*K2)

      END


C-----B0'(K^2,M1,M2)--------------------------------------

      COMPLEX*16 FUNCTION B0S(K2,M1,M2)

      IMPLICIT NONE
      DOUBLE PRECISION K2,M1,M12,M2,M22,MM,MM2
      COMPLEX*16 xyB0

      M12=M1*M1
      M22=M2*M2

      IF (K2.EQ.0D0) GOTO 1850

      IF (M12+M22.NE.0D0) GOTO 1810
      B0S=-1D0/K2
      GOTO 1899

1810  IF (M1.NE.M2) GOTO 1820
      B0S=(2D0*M12/K2*(xyB0(K2,M1,M1)-xyB0(0D0,M1,M1))-1D0)/(K2-4D0*M12)
      GOTO 1899

1820  IF (M1.EQ.0D0) GOTO 1830
      IF (M2.EQ.0D0) GOTO 1832
      B0S=((M12+M22)*xyB0(K2,M1,M2)-M12*xyB0(0D0,M1,M1)-M22
     $     *xyB0(0D0,M2,M2)
     &     -K2-(M12-M22)**2/K2*(xyB0(K2,M1,M2)-xyB0(0D0,M1,M2)))/
     &     ((K2-(M1-M2)**2)*(K2-(M1+M2)**2))
      GOTO 1899

1830  MM=M1
      GOTO 1835
1832  MM=M2
1835  MM2=MM*MM
      B0S=(MM2*xyB0(K2,0D0,MM)-M2*xyB0(0D0,MM,MM)-K2-MM**4/K2
     $     *(xyB0(K2,0D0,MM)
     &     -xyB0(0D0,0D0,MM)))/(K2-MM2)**2
      GOTO 1899

1850  IF (M1.NE.M2) GOTO 1860
      B0S=1/(6D0*M12)
      GOTO 1899

1860  IF (M1.EQ.0D0) GOTO 1870
      IF (M2.EQ.0D0) GOTO 1872
      B0S=(M12+M22)/(2D0*(M12-M22)**2)+M12*M22/(M12-M22)**3*
     &    (xyB0(0D0,M1,M1)-xyB0(0D0,M2,M2))
      GOTO 1899

1870  B0S=1D0/(2D0*M22)
      GOTO 1899
1872  B0S=1D0/(2D0*M12)


1899  END


C-----B0''(K2,M1,M2)--------------------------------------

      COMPLEX*16 FUNCTION B0SS(K2,M1,M2)

      IMPLICIT NONE
      DOUBLE PRECISION K2,M1,M12,M2,M22,MM,MM2

      M12=M1*M1
      M22=M2*M2

      IF (K2.NE.0D0) THEN
        WRITE(*,*) 'B0``(q^2 <> 0) KANN NICHT BERECHNET WERDEN'
      ENDIF
      IF (K2.NE.0D0) GOTO 1940

      IF (M1.NE.M2) GOTO 1900
      B0SS=1D0/(30D0*M12)
      GOTO 1940

1900  IF (M1.EQ.0D0) GOTO 1930
      IF (M2.EQ.0D0) GOTO 1932
      B0SS=(M1**4+10D0*M12*M22+M2**4)/(3D0*(M12-M22)**4)
     &     +(2D0*M12*M22*(M12+M22))/(M12-M22)**5*LOG(M12/M22)
      GOTO 1940

1930  B0SS=1D0/(3D0*M2**4)
      GOTO 1940
1932  B0SS=1D0/(3D0*M1**4)


1940  END


C-----A(M)---IM PROGRAMM:AA(M)----------------------------

      COMPLEX*16 FUNCTION AA(M)

      IMPLICIT NONE
      DOUBLE PRECISION M

      IF (M.EQ.0D0) THEN
        AA=0.D0
      ELSE
        AA=M**2*(-LOG(M**2)+1.D0)
      ENDIF

      END

C-----xyB0(K^2,M1,M2)---------------------------------------

      COMPLEX*16 FUNCTION xyB0(K2,mM1,mM2)

      IMPLICIT NONE
      COMPLEX*16 B0B0,B1B1,THETHA
      DOUBLE PRECISION K2,M1,M2,M12,M22,EE,mm1,mm2

      EE=1D-200
      m1=mm1
      m2=mm2
      THETHA=(0.D0,0.D0)
      IF (K2.GE.0.D0) THETHA=(0.D0,3.14159265358979D0)
      IF (M1.EQ.0.D0) M1=1.D-50
      M12=M1*M1
      IF (M2.EQ.0.D0) M2=1.D-50
      M22=M2*M2

      IF (K2.EQ.0D0) GOTO 1980
      IF (M12+M22.LT.1D-50) THEN
        XYB0=2-LOG(K2)+THETHA
      ELSE
        CALL BQUER(K2,M1,M2,B0B0,B1B1)
        XYB0=B0B0-LOG(M1*M2)
      ENDIF
      GOTO 2000

1980  IF (M12.NE.M22) GOTO 1990
      XYB0=-LOG(M12)
      GOTO 2000

1990  XYB0=1D0-(M12*LOG(M12)-(M22)*LOG(M22))/(M12-M22)

2000  END

C-----HOLLIKS PROGRAMM--------------------------------------------------
C********************************************************
C
      SUBROUTINE BQUER(S,M1,M2,B0,B1)
C
C  B0 AND B1 ARE THE (FINITE) INARIANT FUNCTIONS IN THE
C  2-POINT INTEGRALS.
C  S = Q**2;  M1,M2 ARE THE INTERNAL MASSES
C
      IMPLICIT REAL*8(A-Z)
      DOUBLE PRECISION DREAL,S,M1,M2,EE
      EXTERNAL F,xyG
      COMPLEX*16 xyB0,xyB1,CF
      EE=1.D-100
      XM1=M1**2
      XM2=M2**2
      IF (M1.EQ.M2) THEN
        LM=0.
      ELSE
        LM=DLOG(M2/M1)
      ENDIF
      CF=DCMPLX( F(S,M1,M2),xyG(S,M1,M2))
      IF (M1.EQ.M2) GOTO 10
      XYB0=1.D0-(XM2+XM1)/(XM2-XM1)*LM+CF
      XYB1=-.25D0+XM1/(XM2-XM1)*LM+(XM2-XM1-S)/2.D0/S*CF
      GOTO 20
10    CONTINUE
      XYB0=CF
      XYB1=-.5D0*CF
20    RETURN
      END
C********************************************************
C  !! WIRD NICHT BEN™TIGT !!
C     SUBROUTINE BQUER1(X,M1,M2,B0,B1,P0,P1)
C**************************************************************
C  THE SCALAR VERTEX INTEGRAL WITH EQUAL EXTERNAL MASSES MF   *
C**************************************************************
C
      COMPLEX*16 FUNCTION xyC0(WUS,Q,M2,M3,M1)
C
C  S = MOMENTUM TRANSFER; M1,M2,M3  ARE THE INTERNAL MASSES
C
      IMPLICIT REAL*8 (A-Y)
      COMPLEX*16 Z1,Z2,Z11,Z12,Z21 ,Z22,CL1,CL2,CL3,CSPEN,SPENCE,
     &           INT,DCMPLX,CSCAL
      DOUBLE PRECISION DREAL,Q,S,WUS,EE
      EE=1.D-100
      S=WUS*WUS
      MF=Q
      XMF=MF*MF
C.........XM1 ETC.   ARE FERMION AND BOSON MASSES SQUARED
      XM1=M1*M1
      XM2=M2*M2
      XM3=M3*M3
C     WRITE(*,*)REAL(M1),REAL(M2),REAL(M3)
C
C..T'HOOFT-VELTMAN PARAMETERS
      A=1.D0
      B=XMF/S
      C=-1.D0
      D=XM1-XM2-S
      E=XM3-XM1-XMF+S
      F=XM2/S
      D=D/S
      E=E/S
C..DISCRIMINANTE FOR ALPHA-EQUATION
      DISC=C*C-4.D0*A*B
      IF (DISC .LT. 0.D0) GOTO 500
      AL=(-C-DSQRT(DISC))/2.D0/B
      NENNER=C+2.D0*AL*B
C..THE FIRST INTEGRAL.............................................
C     Y0=-(D+E*AL+2.D0*A+C*AL)/NENNER+EE
C     Y01=Y0-1.D0+EE
      Y0=-(D+E*AL+2.D0*A+C*AL)/NENNER
      Y01=Y0-1.D0
      D1=(C+E)**2-4.D0*B*(A+D+F)
      X1=-(C+E)/2.D0/B
      IF (D1.GT.0.D0) GOTO 10
C.......COMPLEX ZEROES OF LOGARITHMS
      SQ1=DSQRT(-D1)
      X2=SQ1/2.D0/B
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=(Y01+EE)/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=(Y01+EE)/(Y0-Z2)
      CL1=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 15
10    CONTINUE
C........REAL ZEROES
      SQ1=DSQRT(D1)
      X2=SQ1/2.D0/B
      Y1=X1+X2
      Y2=X1-X2
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
      Y11=Y0/(Y0-Y1)
      Y12=(Y01+EE)/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=(Y01+EE)/(Y0-Y2)
c      CL1=CSPEN(Y11,SIG1)-CSPEN(Y12,SIG2)+CSPEN(Y21,-SIG1)
c     &   -CSPEN(Y22,-SIG2)
      CL1=CSPEN(dcmplx(Y11,SIG1))-CSPEN(dcmplx(Y12,SIG2))
     $   +CSPEN(dcmplx(Y21,-SIG1))
     &   -CSPEN(dcmplx(Y22,-SIG2))
15    CONTINUE
C..THE SECOND INTEGRAL............................................
C     Y0=-(D+E*AL)/NENNER/(1.D0-AL)+EE
C     Y01=Y0-1.D0+EE
      Y0=-(D+E*AL)/NENNER/(1.D0-AL)
      Y01=Y0-1.D0
      D2=(E+D)**2-4.D0*F*(A+B+C)
      X1=-(E+D)/2.D0/(A+B+C)
      IF(D2.GT.0.D0) GOTO 20
C.......COMPLEX ZEROES OF LOGARITHMS
      SQ2=DSQRT(-D2)
      X2=SQ2/2.D0/(A+B+C)
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=(Y01+EE)/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=(Y01+EE)/(Y0-Z2)
      CL2=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 25
20    CONTINUE
C........REAL ZEROES
      X2=DSQRT(D2)/2.D0/(A+B+C)
      Y1=X1+X2
      Y2=X1-X2
      Y11=Y0/(Y0-Y1)
      Y12=(Y01+EE)/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=(Y01+EE)/(Y0-Y2)
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
c      CL2=CSPEN(Y11,SIG1)-CSPEN(Y12,SIG2)+CSPEN(Y21,-SIG1)
c     &   -CSPEN(Y22,-SIG2)
      CL2=CSPEN(dcmplx(Y11,SIG1))-CSPEN(dcmplx(Y12,SIG2))
     $   +CSPEN(dcmplx(Y21,-SIG1))
     &   -CSPEN(dcmplx(Y22,-SIG2))
25    CONTINUE
C..THE THIRD INTEGRAL............................................
C     Y0=(D+E*AL)/NENNER/AL+EE
C     Y01=Y0-1.D0+EE
      Y0=(D+E*AL)/NENNER/AL
      Y01=Y0-1.D0
      D3=D*D-4.D0*A*F
      X1=-D/2.D0/A
      IF (D3.GT.0.D0) GOTO 30
C........COMPLEX ZEROES OF LOGARITHMS
      SQ3=DSQRT(-D3)
      X2=SQ3/2.D0/A
      Z1=DCMPLX(X1,X2)
      Z2=DCMPLX(X1,-X2)
      Z11=Y0/(Y0-Z1)
      Z12=(Y01+EE)/(Y0-Z1)
      Z21=Y0/(Y0-Z2)
      Z22=(Y01+EE)/(Y0-Z2)
      CL3=SPENCE(Z11)-SPENCE(Z12)+SPENCE(Z21)-SPENCE(Z22)
      GOTO 35
30    CONTINUE
C........REAL ZEROES
      X2=DSQRT(D3)/2.D0/A
      Y1=X1+X2
      Y2=X1-X2
 31   FORMAT(1H ,3E12.4)
      Y11=Y0 /(Y0-Y1)
      Y12=(Y01+EE)/(Y0-Y1)
      Y21=Y0/(Y0-Y2)
      Y22=(Y01+EE)/(Y0-Y2)
      SIG1= Y0/DABS(Y0)
      SIG2= Y01/DABS(Y01)
c      CL3=CSPEN(Y11,SIG1)-CSPEN(Y12,SIG2)+CSPEN(Y21,-SIG1)
c     &   -CSPEN(Y22,-SIG2)
      CL3=CSPEN(dcmplx(Y11,SIG1))-CSPEN(dcmplx(Y12,SIG2))
     $   +CSPEN(dcmplx(Y21,-SIG1))
     &   -CSPEN(dcmplx(Y22,-SIG2))
35    CONTINUE
C..SUMMATION OF THE 3 INTEGRALS ....................................
      INT=-CL1+CL2-CL3
      CSCAL=INT/NENNER/S
      GOTO 501
500   CONTINUE
C..ERROR MESSAGE FOR COMPLEX ALPHA................................
      WRITE(6,21)
21    FORMAT(1H ,'  I CANNOT HANDLE A COMPLEX ALPHA (SH)')
      write(*,*) 'mf:',mf
      write(*,*) 'wus:',wus
      write(*,*) 'Int. Massen:',m1,m2,m3
      GOTO 502
501   xyC0=CSCAL
502   RETURN
      END
C
C*******************************************************************
C
c$$$      COMPLEX*16 FUNCTION CSPEN(X,SIG)
c$$$      IMPLICIT REAL*8(A-Y)
c$$$      COMPLEX*16 Z,CPI,SPENCE,ZX
c$$$      DOUBLE PRECISION DREAL
c$$$      PI=3.1415926536D0
c$$$      PI6=PI*PI/6.D0
c$$$      CPI=DCMPLX(0.D0,PI)
c$$$      IF (X.LT.1.D0) GOTO 10
c$$$      IF(X.EQ.1.D0) GOTO 11
c$$$      LX=DLOG(X)
c$$$      X1=1.D0-X
c$$$      LX1=DLOG(-X1)
c$$$      Z=DCMPLX(X1,0.D0)
c$$$      IF (SIG.GT.0.D0) GOTO 5
c$$$      CSPEN=-SPENCE(Z)+PI6-LX*(LX1+CPI)
c$$$      GOTO 20
c$$$5     CSPEN=-SPENCE(Z)+PI6-LX*(LX1-CPI)
c$$$      GOTO 20
c$$$10    ZX=DCMPLX(X,0.D0)
c$$$      CSPEN=SPENCE(ZX)
c$$$      GOTO 20
c$$$11    CSPEN=DCMPLX(PI6,0.D0)
c$$$20    RETURN
c$$$      END
C
C**************************************************************
C
      COMPLEX*16 FUNCTION CLN(X,SIG)
      IMPLICIT REAL*8(A-Z)
      COMPLEX*16 Z,CPI,CDLOG
      DOUBLE PRECISION DREAL
      PI=3.1415926536D0
      CPI=DCMPLX(0.D0,PI)
      IF (X.GT.0.D0) GOTO 10
      X1=-X
      IF (SIG.GT.0.D0) GOTO 5
      CLN=DLOG(X1)-CPI
      GOTO 20
5     CLN=DLOG(X1)+CPI
      GOTO 20
10    Y=DLOG(X)
      CLN=DCMPLX(Y,0.D0)
20    RETURN
      END
      COMPLEX*16 FUNCTION SPENCE(XX)
C  THE DILOGARITHM FOR GENERAL COMPLEX ARGUMENT.
C  NOT ALLOWED: REAL(XX) GT 1 WITH IMAG(XX)=0.
      IMPLICIT REAL*8(A-Z)
      INTEGER N
      COMPLEX*16 XX,X,Z,D,P,       CDLOG
      DOUBLE PRECISION DREAL
      DIMENSION A(19)
      PI=3.1415926536D0
      X=XX
      XR=DREAL(X)
      XI=DIMAG(X)
      IF(XR.NE.1.) GOTO 111
      IF(XI.EQ.0.) GOTO 20
111   CONTINUE
C    PROJECTION INTO THE CONVERGENCE RADIUS
      VOR=1.D0
      P=DCMPLX(0.D0,0.D0)
      R=DREAL(X)
      IF (R .LE. 0.5D0) GOTO 1
      P=PI*PI/6.D0- CDLOG(X)*CDLOG(1.D0-X)
      VOR=-1.D0
      X=1.D0-X
    1 CONTINUE
      B=CDABS(X)
      IF (B .LT. 1.D0) GOTO 2
      P=P - (PI*PI/6.D0+ CDLOG(-X)*CDLOG(-X)/2.D0)*VOR
      VOR=VOR*(-1.D0)
      X=1.D0/X
    2 CONTINUE
C    CALCULATION OF THE SPENCE FUNCTION
      A(1)=1.D0
      A(2)=-0.5D0
      A(3)=1.D0/6.D0
      A(5)=-1.D0/30.D0
      A(7)=1.D0/42.D0
      A(9)=-1.D0/30.D0
      A(11)=5.D0/66.D0
      A(13)=-691.D0/2730.D0
      A(15)=7.D0/6.D0
      A(17)=-3617.D0/510.D0
      A(19)=43867.D0/798.D0
      DO 5 N=2,9,1
      A(2*N)=0.D0
    5 CONTINUE
      Z=(-1.D0)*CDLOG(1.D0-X)
      D=DCMPLX(A(19),0.D0)
      DO 10 N=1,18,1
      D=D*Z/(20.D0-N) + A(19-N)
   10 CONTINUE
      D=D*Z
      SPENCE=D*VOR + P
      GOTO 30
   20 CONTINUE
      SPENCE=PI*PI/6.D0
   30 CONTINUE
      RETURN
      END
***************************************************
      DOUBLE PRECISION FUNCTION F(S,MA,MB)
* Real part of the function F(s,ma,mb)
      IMPLICIT REAL*8(A-Z)
      DOUBLE PRECISION DREAL
      PARAMETER(EPS=1.D-6)
      MA2=MA**2
      MB2=MB**2
      IF(ABS(S).LT.EPS) THEN
             F=0.0D0
      ELSEIF(MA.LT.EPS) THEN
           IF(S.GT.MB2+EPS) THEN
             F=1.D0+(1.D0-MB2/S)*LOG(1./(S/MB2-1.D0))
           ELSEIF(S.LT.MB2-EPS) THEN
             F=1.D0+(1.D0-MB2/S)*LOG(1./(1.D0-S/MB2))
           ELSE
             F=1.D0
           ENDIF
      ELSEIF(MB.LT.EPS) THEN
           IF(S.GT.MA2+EPS) THEN
             F=1.D0+(1.D0-MA2/S)*LOG(1./(S/MA2-1.D0))
           ELSEIF(S.LT.MA2-EPS) THEN
             F=1.D0+(1.D0-MA2/S)*LOG(1./(1.D0-S/MA2))
           ELSE
             F=1.D0
           ENDIF
      ELSE
          IF(ABS(MB-MA).LT.EPS) THEN
           F=2.D0
          ELSE
           F=1.D0+((MA2-MB2)/S-(MA2+MB2)/(MA2-MB2))*LOG(MB/MA)
          ENDIF
          IF(S.GE.(MA+MB)**2) THEN
            RPLUS=SQRT(S-(MA+MB)**2)
            RMIN =SQRT(S-(MA-MB)**2)
            F=F- RPLUS*RMIN*LOG((RPLUS+RMIN)**2/(4.D0*MA*MB))/S
          ELSEIF(S.LT.(MA-MB)**2) THEN
            RPLUS=SQRT((MA+MB)**2-S)
            RMIN =SQRT((MA-MB)**2-S)
            F=F+ RPLUS*RMIN*LOG((RPLUS+RMIN)**2/(4.D0*MA*MB))/S
          ELSE
            RPLUS=SQRT((MA+MB)**2-S)
            RMIN =SQRT(S-(MA-MB)**2)
            F=F- 2.D0*RPLUS*RMIN*ATAN(RMIN/RPLUS)/S
          ENDIF
        ENDIF
        END
      DOUBLE PRECISION FUNCTION DF(S,MA,MB)
*****   Derivative of the real part of the function F(s,ma,mb).    *****
      IMPLICIT REAL*8(A-Z)
      DOUBLE PRECISION DREAL
      PARAMETER(EPS=1.D-6)
      IF(S.LT.(MA-MB)**2) THEN
         RPLUS=SQRT((MA+MB)**2-S)
         RMIN =SQRT((MA-MB)**2-S)
         DF=( (MB**2-MA**2)*LOG(MB/MA)/S-
     .   ((RMIN**2+2.D0*RMIN**2*RPLUS**2/S+RPLUS**2)/(2.D0*RMIN*RPLUS))*
     .           LOG((RMIN+RPLUS)**2/(4.D0*MA*MB))  -1.D0 )/S
      ELSEIF(S.LT.(MA+MB)**2) THEN
         RPLUS=SQRT((MA+MB)**2/S-1.D0)
         RMIN =SQRT(1.D0-(MA-MB)**2/S)
         DF=( (MB**2-MA**2)*LOG(MB/MA)/S+
     .        ((RMIN**2+2.D0*RMIN**2*RPLUS**2-RPLUS**2)/(RMIN*RPLUS))*
     .           ATAN(RMIN/RPLUS)  -1.D0 )/S
      ELSE
         RPLUS=SQRT(S-(MA+MB)**2)
         RMIN =SQRT(S-(MA-MB)**2)
         DF=( (MB**2-MA**2)*LOG(MB/MA)/S-
     .   ((RMIN**2-2.D0*RMIN**2*RPLUS**2/S+RPLUS**2)/(2.D0*RMIN*RPLUS))*
     .           LOG((RMIN+RPLUS)**2/(4.D0*MA*MB))  -1.D0 )/S
*        Y=SQRT(S-4.D0*MA**2)
*        Z=SQRT(S)
*        DF=( (Y**2-S)*LOG((Z+Y)/(Z-Y))/(2.D0*Z*Y)-1.D0)/S
      ENDIF
      END
      DOUBLE PRECISION FUNCTION xyG(S,MA,MB)
* Imaginary part of the function F(s,ma,mb)
      IMPLICIT DOUBLE PRECISION (A-Z)
      PARAMETER (PI=3.1415926535897932D0)
      xyG = 0.D0
      IF(S.GT.(MA+MB)**2)  xyG=PI*SQRT((S-(MA+MB)**2)*(S-(MA-MB)**2))/S
      END



c----------------------------------------------------------------
c
c --> lamspen.f
c
c----------------------------------------------------------------
      
      double precision function l(m2)

      implicit none

      double precision m2,mue,pi,g

      pi=3.14159265358979d0
      g=.577215664901532d0
      mue = 1d0

      l=g+log(m2/(4d0*pi*mue**2))
c      l=0d0

      end

c-------------------------------------------------------------------
      complex*16 function cl(m2)

      implicit none

      double precision mue,pi,g
      complex*16 m2

      pi=3.14159265358979d0
      g=.577215664901532d0
      mue = 1d0

      cl=g+cdlog(m2/(4d0*pi*mue**2))
c      l=0d0

      end

c-------------------------------------------------------------------

      complex*16 function lam(xx,yy)
c --> Berechnung der Lambda-Funktion

      implicit none

      complex*16 x,y
      double precision xx,yy

      x=xx*(1d0,0d0)
      y=yy*(1d0,0d0)

      if ((cdabs(y).lt.1d-8).and.(cdabs(x-1d0).lt.1d-8)) then
         lam=(1d0,0d0)-x
      elseif ((cdabs(x).lt.1d-8).and.(cdabs(y-1d0).lt.1d-8)) then 
         lam=(1d0,0d0)-y
      else
         lam=cdsqrt((1d0,0d0)+x**2+y**2-2d0*x-2d0*y-2d0*x*y)
      endif

c      write(*,*) 'lam:',dreal(x),dreal(y),lam

      end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        FUNCTION CSPEN(Z)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C       SPENCE-FUNKTION KOMPLEX, FREI NACH HOLLIK                     C
C---------------------------------------------------------------------C
C       20.07.83    LAST CHANGED 10.05.89        ANSGAR DENNER        C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
        COMPLEX*16 CSPEN,W,SUM,Z,U
        REAL*8 RZ,AZ,A1
        REAL*8 B(9)/
     1   0.1666666666666666666666666667D0,
     2  -0.0333333333333333333333333333D0,
     3   0.0238095238095238095238095238D0,
     4  -0.0333333333333333333333333333D0,
     5   0.0757575757575757575757575758D0,
     6  -0.2531135531135531135531135531D0,
     7   1.1666666666666666666666666667D0,
     8  -7.09215686274509804D0         ,
     9  54.97117794486215539D0         /
C     BEACHTE:                 B(N)=B2N
C     B(1)=1./6.
C     B(2)=-1./30.
C     B(3)=1./42.
C     B(4)=-1./30.
C     B(5)=5./66.
C     B(6)=-691./2730.
C     B(7)=7./6.
C     B(8)=-3617./510.
C     B(9)=43867./798.
C     B(10)=-174611./330.
C     B(11)=854513./138.
C     PI=3.1415926535897932384
C     PI*PI/6.=1.6449..., PI*PI/3=3.28986...
C
c      write(*,*) 'z:',z
      Z =Z*DCMPLX(1D0)
      RZ=DREAL(Z)
      AZ=CDABS(Z)
      A1=CDABS(1D0-Z)
c      write(*,*)'z, rz, az, a1:',z,rz,az,a1
C     IF((SNGL(RZ) .EQ. 0.0) .AND. (SNGL(DIMAG(Z)) .EQ. 0.0)) THEN
C ---> CHANGED  10.5.89
      IF(AZ .LT. 1D-20) THEN
        CSPEN=-CDLOG(1D0-Z)
c        write(*,*) 'cspen:', cspen
        RETURN
      END IF
      IF((SNGL(RZ) .EQ. 1.0) .AND. (SNGL(DIMAG(Z)) .EQ. 0.0)) THEN
        CSPEN=1.64493406684822643D0
c        write(*,*) 'cspen:', cspen
        RETURN
      END IF
      IF(RZ.GT.5D-1) GOTO 20
      IF(AZ.GT.1D0) GOTO 10
      W=-CDLOG(1D0-Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 2
c      write(*,*) 'u:',u
c      write(*,*) 'sum:',sum
      DO 1 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 2
      SUM=SUM+U*B(K)
 1    CONTINUE
 2    CSPEN=SUM
c        write(*,*) 'cspen:', cspen
      RETURN
10    W=-CDLOG(1D0-1D0/Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 12

      DO 11 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(B(K)*U/SUM).LT.1D-20) GOTO 12
      SUM=SUM+U*B(K)
11    CONTINUE
12    CSPEN=-SUM-1.64493406684822643D0-.5D0*CDLOG(-Z)**2
c        write(*,*) 'cspen:', cspen
      RETURN
20    IF(A1.GT.1D0) GOTO 30
      W=-CDLOG(Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 22
      DO 21 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 22
      SUM=SUM+U*B(K)
21    CONTINUE
22    CSPEN=-SUM+1.64493406684822643D0-CDLOG(Z)*CDLOG(1D0-Z)
c        write(*,*) 'cspen:', cspen
      RETURN
30    W=CDLOG(1D0-1D0/Z)
      SUM=W-0.25D0*W*W
      U=W
      IF(CDABS(U).LT.1D-10) GOTO 32
      DO 31 K=1,9
      U=U*W*W/DFLOAT(2*K*(2*K+1))
      IF(CDABS(U*B(K)/SUM).LT.1D-20) GOTO 32
      SUM=SUM+U*B(K)
31    CONTINUE
32    CSPEN=SUM+3.28986813369645287D0
     *               +.5D0*CDLOG(Z-1D0)**2-CDLOG(Z)*CDLOG(1D0-Z)
50    CONTINUE
c        write(*,*) 'cspen:', cspen
      END


c----------------------------------------------------------------
      

      FUNCTION TRILOG(X)
C     *******************
C CALCULATES THE TRILOGARITHM FOR REAL ARGUMENTS X </= 1.
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMPLEX*16 WGPLG
c      COMPLEX*16 WGPLG, trilog
      TRILOG=DREAL(WGPLG(2,1,X))
c      TRILOG=WGPLG(2,1,X)
      RETURN

      end

C***********************************************************************


      COMPLEX*16 FUNCTION WGPLG (N,P,X)
 
      INTEGER P,P1,NC(10),INDEX(31)
      DOUBLE PRECISION FCT(0:4),SGN(0:4),U(0:4),S1(4,4),C(4,4)
      DOUBLE PRECISION A(0:30,10)
      DOUBLE PRECISION X,X1,H,ALFA,R,Q,C1,C2,B0,B1,B2,ZERO,HALF
 
      COMPLEX*16 V(0:5),SK,SM
 
      DATA FCT /1.0D0,1.0D0,2.0D0,6.0D0,24.0D0/
      DATA SGN /1.0D0,-1.0D0,1.0D0,-1.0D0,1.0D0/
      DATA ZERO /0.0D0/, HALF /0.5D0/
      DATA C1 /1.33333 33333 333D0/, C2 /0.33333 33333 3333D0/
 
      DATA S1(1,1) /1.64493 40668 482D0/
      DATA S1(1,2) /1.20205 69031 596D0/
      DATA S1(1,3) /1.08232 32337 111D0/
      DATA S1(1,4) /1.03692 77551 434D0/
      DATA S1(2,1) /1.20205 69031 596D0/
      DATA S1(2,2) /2.70580 80842 778D-1/
      DATA S1(2,3) /9.65511 59989 444D-2/
      DATA S1(3,1) /1.08232 32337 111D0/
      DATA S1(3,2) /9.65511 59989 444D-2/
      DATA S1(4,1) /1.03692 77551 434D0/
 
      DATA C(1,1) / 1.64493 40668 482D0/
      DATA C(1,2) / 1.20205 69031 596D0/
      DATA C(1,3) / 1.08232 32337 111D0/
      DATA C(1,4) / 1.03692 77551 434D0/
      DATA C(2,1) / 0.00000 00000 000D0/
      DATA C(2,2) /-1.89406 56589 945D0/
      DATA C(2,3) /-3.01423 21054 407D0/
      DATA C(3,1) / 1.89406 56589 945D0/
      DATA C(3,2) / 3.01423 21054 407D0/
      DATA C(4,1) / 0.00000 00000 000D0/
 
      DATA INDEX /1,2,3,4,6*0,5,6,7,7*0,8,9,8*0,10/
 
      DATA NC /24,26,28,30,22,24,26,19,22,17/
 
      DATA A( 0,1) / .96753 21504 3498D0/
      DATA A( 1,1) / .16607 30329 2785D0/
      DATA A( 2,1) / .02487 93229 2423D0/
      DATA A( 3,1) / .00468 63619 5945D0/
      DATA A( 4,1) / .00100 16274 9616D0/
      DATA A( 5,1) / .00023 20021 9609D0/
      DATA A( 6,1) / .00005 68178 2272D0/
      DATA A( 7,1) / .00001 44963 0056D0/
      DATA A( 8,1) / .00000 38163 2946D0/
      DATA A( 9,1) / .00000 10299 0426D0/
      DATA A(10,1) / .00000 02835 7538D0/
      DATA A(11,1) / .00000 00793 8705D0/
      DATA A(12,1) / .00000 00225 3670D0/
      DATA A(13,1) / .00000 00064 7434D0/
      DATA A(14,1) / .00000 00018 7912D0/
      DATA A(15,1) / .00000 00005 5029D0/
      DATA A(16,1) / .00000 00001 6242D0/
      DATA A(17,1) / .00000 00000 4827D0/
      DATA A(18,1) / .00000 00000 1444D0/
      DATA A(19,1) / .00000 00000 0434D0/
      DATA A(20,1) / .00000 00000 0131D0/
      DATA A(21,1) / .00000 00000 0040D0/
      DATA A(22,1) / .00000 00000 0012D0/
      DATA A(23,1) / .00000 00000 0004D0/
      DATA A(24,1) / .00000 00000 0001D0/
 
      DATA A( 0,2) / .95180 88912 7832D0/
      DATA A( 1,2) / .43131 13184 6532D0/
      DATA A( 2,2) / .10002 25071 4905D0/
      DATA A( 3,2) / .02442 41559 5220D0/
      DATA A( 4,2) / .00622 51246 3724D0/
      DATA A( 5,2) / .00164 07883 1235D0/
      DATA A( 6,2) / .00044 40792 0265D0/
      DATA A( 7,2) / .00012 27749 4168D0/
      DATA A( 8,2) / .00003 45398 1284D0/
      DATA A( 9,2) / .00000 98586 9565D0/
      DATA A(10,2) / .00000 28485 6995D0/
      DATA A(11,2) / .00000 08317 0847D0/
      DATA A(12,2) / .00000 02450 3950D0/
      DATA A(13,2) / .00000 00727 6496D0/
      DATA A(14,2) / .00000 00217 5802D0/
      DATA A(15,2) / .00000 00065 4616D0/
      DATA A(16,2) / .00000 00019 8033D0/
      DATA A(17,2) / .00000 00006 0204D0/
      DATA A(18,2) / .00000 00001 8385D0/
      DATA A(19,2) / .00000 00000 5637D0/
      DATA A(20,2) / .00000 00000 1735D0/
      DATA A(21,2) / .00000 00000 0536D0/
      DATA A(22,2) / .00000 00000 0166D0/
      DATA A(23,2) / .00000 00000 0052D0/
      DATA A(24,2) / .00000 00000 0016D0/
      DATA A(25,2) / .00000 00000 0005D0/
      DATA A(26,2) / .00000 00000 0002D0/
 
      DATA A( 0,3) / .98161 02799 1365D0/
      DATA A( 1,3) / .72926 80632 0726D0/
      DATA A( 2,3) / .22774 71490 9321D0/
      DATA A( 3,3) / .06809 08329 6197D0/
      DATA A( 4,3) / .02013 70118 3064D0/
      DATA A( 5,3) / .00595 47848 0197D0/
      DATA A( 6,3) / .00176 76901 3959D0/
      DATA A( 7,3) / .00052 74821 8502D0/
      DATA A( 8,3) / .00015 82746 1460D0/
      DATA A( 9,3) / .00004 77492 2076D0/
      DATA A(10,3) / .00001 44792 0408D0/
      DATA A(11,3) / .00000 44115 4886D0/
      DATA A(12,3) / .00000 13500 3870D0/
      DATA A(13,3) / .00000 04148 1779D0/
      DATA A(14,3) / .00000 01279 3307D0/
      DATA A(15,3) / .00000 00395 9070D0/
      DATA A(16,3) / .00000 00122 9055D0/
      DATA A(17,3) / .00000 00038 2658D0/
      DATA A(18,3) / .00000 00011 9459D0/
      DATA A(19,3) / .00000 00003 7386D0/
      DATA A(20,3) / .00000 00001 1727D0/
      DATA A(21,3) / .00000 00000 3687D0/
      DATA A(22,3) / .00000 00000 1161D0/
      DATA A(23,3) / .00000 00000 0366D0/
      DATA A(24,3) / .00000 00000 0116D0/
      DATA A(25,3) / .00000 00000 0037D0/
      DATA A(26,3) / .00000 00000 0012D0/
      DATA A(27,3) / .00000 00000 0004D0/
      DATA A(28,3) / .00000 00000 0001D0/
 
      DATA A( 0,4) /1.06405 21184 614 D0/
      DATA A( 1,4) /1.06917 20744 981 D0/
      DATA A( 2,4) / .41527 19325 1768D0/
      DATA A( 3,4) / .14610 33293 6222D0/
      DATA A( 4,4) / .04904 73264 8784D0/
      DATA A( 5,4) / .01606 34086 0396D0/
      DATA A( 6,4) / .00518 88935 0790D0/
      DATA A( 7,4) / .00166 29871 7324D0/
      DATA A( 8,4) / .00053 05827 9969D0/
      DATA A( 9,4) / .00016 88702 9251D0/
      DATA A(10,4) / .00005 36832 8059D0/
      DATA A(11,4) / .00001 70592 3313D0/
      DATA A(12,4) / .00000 54217 4374D0/
      DATA A(13,4) / .00000 17239 4082D0/
      DATA A(14,4) / .00000 05485 3275D0/
      DATA A(15,4) / .00000 01746 7795D0/
      DATA A(16,4) / .00000 00556 7550D0/
      DATA A(17,4) / .00000 00177 6234D0/
      DATA A(18,4) / .00000 00056 7224D0/
      DATA A(19,4) / .00000 00018 1313D0/
      DATA A(20,4) / .00000 00005 8012D0/
      DATA A(21,4) / .00000 00001 8579D0/
      DATA A(22,4) / .00000 00000 5955D0/
      DATA A(23,4) / .00000 00000 1911D0/
      DATA A(24,4) / .00000 00000 0614D0/
      DATA A(25,4) / .00000 00000 0197D0/
      DATA A(26,4) / .00000 00000 0063D0/
      DATA A(27,4) / .00000 00000 0020D0/
      DATA A(28,4) / .00000 00000 0007D0/
      DATA A(29,4) / .00000 00000 0002D0/
      DATA A(30,4) / .00000 00000 0001D0/
 
      DATA A( 0,5) / .97920 86066 9175D0/
      DATA A( 1,5) / .08518 81314 8683D0/
      DATA A( 2,5) / .00855 98522 2013D0/
      DATA A( 3,5) / .00121 17721 4413D0/
      DATA A( 4,5) / .00020 72276 8531D0/
      DATA A( 5,5) / .00003 99695 8691D0/
      DATA A( 6,5) / .00000 83806 4065D0/
      DATA A( 7,5) / .00000 18684 8945D0/
      DATA A( 8,5) / .00000 04366 6087D0/
      DATA A( 9,5) / .00000 01059 1733D0/
      DATA A(10,5) / .00000 00264 7892D0/
      DATA A(11,5) / .00000 00067 8700D0/
      DATA A(12,5) / .00000 00017 7654D0/
      DATA A(13,5) / .00000 00004 7342D0/
      DATA A(14,5) / .00000 00001 2812D0/
      DATA A(15,5) / .00000 00000 3514D0/
      DATA A(16,5) / .00000 00000 0975D0/
      DATA A(17,5) / .00000 00000 0274D0/
      DATA A(18,5) / .00000 00000 0077D0/
      DATA A(19,5) / .00000 00000 0022D0/
      DATA A(20,5) / .00000 00000 0006D0/
      DATA A(21,5) / .00000 00000 0002D0/
      DATA A(22,5) / .00000 00000 0001D0/
 
      DATA A( 0,6) / .95021 85196 3952D0/
      DATA A( 1,6) / .29052 52916 1433D0/
      DATA A( 2,6) / .05081 77406 1716D0/
      DATA A( 3,6) / .00995 54376 7280D0/
      DATA A( 4,6) / .00211 73389 5031D0/
      DATA A( 5,6) / .00047 85947 0550D0/
      DATA A( 6,6) / .00011 33432 1308D0/
      DATA A( 7,6) / .00002 78473 3104D0/
      DATA A( 8,6) / .00000 70478 8108D0/
      DATA A( 9,6) / .00000 18278 8740D0/
      DATA A(10,6) / .00000 04838 7492D0/
      DATA A(11,6) / .00000 01303 3842D0/
      DATA A(12,6) / .00000 00356 3769D0/
      DATA A(13,6) / .00000 00098 7174D0/
      DATA A(14,6) / .00000 00027 6586D0/
      DATA A(15,6) / .00000 00007 8279D0/
      DATA A(16,6) / .00000 00002 2354D0/
      DATA A(17,6) / .00000 00000 6435D0/
      DATA A(18,6) / .00000 00000 1866D0/
      DATA A(19,6) / .00000 00000 0545D0/
      DATA A(20,6) / .00000 00000 0160D0/
      DATA A(21,6) / .00000 00000 0047D0/
      DATA A(22,6) / .00000 00000 0014D0/
      DATA A(23,6) / .00000 00000 0004D0/
      DATA A(24,6) / .00000 00000 0001D0/
 
      DATA A( 0,7) / .95064 03218 6777D0/
      DATA A( 1,7) / .54138 28546 5171D0/
      DATA A( 2,7) / .13649 97959 0321D0/
      DATA A( 3,7) / .03417 94232 8207D0/
      DATA A( 4,7) / .00869 02788 3583D0/
      DATA A( 5,7) / .00225 28408 4155D0/
      DATA A( 6,7) / .00059 51608 9806D0/
      DATA A( 7,7) / .00015 99561 7766D0/
      DATA A( 8,7) / .00004 36521 3096D0/
      DATA A( 9,7) / .00001 20747 4688D0/
      DATA A(10,7) / .00000 33801 8176D0/
      DATA A(11,7) / .00000 09563 2476D0/
      DATA A(12,7) / .00000 02731 3129D0/
      DATA A(13,7) / .00000 00786 6968D0/
      DATA A(14,7) / .00000 00228 3195D0/
      DATA A(15,7) / .00000 00066 7205D0/
      DATA A(16,7) / .00000 00019 6191D0/
      DATA A(17,7) / .00000 00005 8018D0/
      DATA A(18,7) / .00000 00001 7246D0/
      DATA A(19,7) / .00000 00000 5151D0/
      DATA A(20,7) / .00000 00000 1545D0/
      DATA A(21,7) / .00000 00000 0465D0/
      DATA A(22,7) / .00000 00000 0141D0/
      DATA A(23,7) / .00000 00000 0043D0/
      DATA A(24,7) / .00000 00000 0013D0/
      DATA A(25,7) / .00000 00000 0004D0/
      DATA A(26,7) / .00000 00000 0001D0/
 
      DATA A( 0,8) / .98800 01167 2229D0/
      DATA A( 1,8) / .04364 06760 9601D0/
      DATA A( 2,8) / .00295 09117 8278D0/
      DATA A( 3,8) / .00031 47780 9720D0/
      DATA A( 4,8) / .00004 31484 6029D0/
      DATA A( 5,8) / .00000 69381 8230D0/
      DATA A( 6,8) / .00000 12464 0350D0/
      DATA A( 7,8) / .00000 02429 3628D0/
      DATA A( 8,8) / .00000 00504 0827D0/
      DATA A( 9,8) / .00000 00109 9075D0/
      DATA A(10,8) / .00000 00024 9467D0/
      DATA A(11,8) / .00000 00005 8540D0/
      DATA A(12,8) / .00000 00001 4127D0/
      DATA A(13,8) / .00000 00000 3492D0/
      DATA A(14,8) / .00000 00000 0881D0/
      DATA A(15,8) / .00000 00000 0226D0/
      DATA A(16,8) / .00000 00000 0059D0/
      DATA A(17,8) / .00000 00000 0016D0/
      DATA A(18,8) / .00000 00000 0004D0/
      DATA A(19,8) / .00000 00000 0001D0/
 
      DATA A( 0,9) / .95768 50654 6350D0/
      DATA A( 1,9) / .19725 24967 9534D0/
      DATA A( 2,9) / .02603 37031 3918D0/
      DATA A( 3,9) / .00409 38216 8261D0/
      DATA A( 4,9) / .00072 68170 7110D0/
      DATA A( 5,9) / .00014 09187 9261D0/
      DATA A( 6,9) / .00002 92045 8914D0/
      DATA A( 7,9) / .00000 63763 1144D0/
      DATA A( 8,9) / .00000 14516 7850D0/
      DATA A( 9,9) / .00000 03420 5281D0/
      DATA A(10,9) / .00000 00829 4302D0/
      DATA A(11,9) / .00000 00206 0784D0/
      DATA A(12,9) / .00000 00052 2823D0/
      DATA A(13,9) / .00000 00013 5066D0/
      DATA A(14,9) / .00000 00003 5451D0/
      DATA A(15,9) / .00000 00000 9436D0/
      DATA A(16,9) / .00000 00000 2543D0/
      DATA A(17,9) / .00000 00000 0693D0/
      DATA A(18,9) / .00000 00000 0191D0/
      DATA A(19,9) / .00000 00000 0053D0/
      DATA A(20,9) / .00000 00000 0015D0/
      DATA A(21,9) / .00000 00000 0004D0/
      DATA A(22,9) / .00000 00000 0001D0/
 
      DATA A( 0,10) / .99343 65167 1347D0/
      DATA A( 1,10) / .02225 77012 6826D0/
      DATA A( 2,10) / .00101 47557 4703D0/
      DATA A( 3,10) / .00008 17515 6250D0/
      DATA A( 4,10) / .00000 89997 3547D0/
      DATA A( 5,10) / .00000 12082 3987D0/
      DATA A( 6,10) / .00000 01861 6913D0/
      DATA A( 7,10) / .00000 00317 4723D0/
      DATA A( 8,10) / .00000 00058 5215D0/
      DATA A( 9,10) / .00000 00011 4739D0/
      DATA A(10,10) / .00000 00002 3652D0/
      DATA A(11,10) / .00000 00000 5082D0/
      DATA A(12,10) / .00000 00000 1131D0/
      DATA A(13,10) / .00000 00000 0259D0/
      DATA A(14,10) / .00000 00000 0061D0/
      DATA A(15,10) / .00000 00000 0015D0/
      DATA A(16,10) / .00000 00000 0004D0/
      DATA A(17,10) / .00000 00000 0001D0/
 
      IF(N .LT. 1 .OR. N .GT. 4 .OR. P .LT. 1 .OR. P .GT. 4 .OR.
     1   N+P .GT. 5) THEN
       WGPLG=ZERO
       PRINT 1000, N,P
       RETURN
      END IF
      IF(X .EQ. SGN(0)) THEN
       WGPLG=S1(N,P)
       RETURN
      END IF
 
      IF(X .GT. FCT(2) .OR. X .LT. SGN(1)) THEN
       X1=SGN(0)/X
       H=C1*X1+C2
       ALFA=H+H
       V(0)=SGN(0)
       V(1)=LOG(DCMPLX(-X,ZERO))
       DO 33 L = 2,N+P
   33  V(L)=V(1)*V(L-1)/L
       SK=ZERO
       DO 34 K = 0,P-1
       P1=P-K
       R=X1**P1/(FCT(P1)*FCT(N-1))
       SM=ZERO
       DO 35 M = 0,K
       N1=N+K-M
       L=INDEX(10*N1+P1-10)
       B1=ZERO
       B2=ZERO
       DO 31 I = NC(L),0,-1
       B0=A(I,L)+ALFA*B1-B2
       B2=B1
   31  B1=B0
       Q=(FCT(N1-1)/FCT(K-M))*(B0-H*B2)*R/P1**N1
   35  SM=SM+V(M)*Q
   34  SK=SK+SGN(K)*SM
       SM=ZERO
       DO 36 M = 0,N-1
   36  SM=SM+V(M)*C(N-M,P)
       WGPLG=SGN(N)*SK+SGN(P)*(SM+V(N+P))
       RETURN
      END IF
 
      IF(X .GT. HALF) THEN
       X1=SGN(0)-X
       H=C1*X1+C2
       ALFA=H+H
       V(0)=SGN(0)
       U(0)=SGN(0)
       V(1)=LOG(DCMPLX(X1,ZERO))
       U(1)=LOG(X)
       DO 23 L = 2,P
   23  V(L)=V(1)*V(L-1)/L
       DO 26 L = 2,N
   26  U(L)=U(1)*U(L-1)/L
       SK=ZERO
       DO 24 K = 0,N-1
       P1=N-K
       R=X1**P1/FCT(P1)
       SM=ZERO
       DO 25 M = 0,P-1
       N1=P-M
       L=INDEX(10*N1+P1-10)
       B1=ZERO
       B2=ZERO
       DO 12 I = NC(L),0,-1
       B0=A(I,L)+ALFA*B1-B2
       B2=B1
   12  B1=B0
       Q=SGN(M)*(B0-H*B2)*R/P1**N1
   25  SM=SM+V(M)*Q
   24  SK=SK+U(K)*(S1(P1,P)-SM)
       WGPLG=SK+SGN(P)*U(N)*V(P)
       RETURN
      END IF
 
      L=INDEX(10*N+P-10)
      H=C1*X+C2
      ALFA=H+H
      B1=ZERO
      B2=ZERO
      DO 11 I = NC(L),0,-1
      B0=A(I,L)+ALFA*B1-B2
      B2=B1
   11 B1=B0
      WGPLG=(B0-H*B2)*X**P/(FCT(P)*P**N)
      RETURN
 1000 FORMAT(/' ***** CERN SUBROUTINE WGPLG ... ILLEGAL VALUES',
     1        '   N = ',I3,'   P = ',I3)
      END
 
 

c----------------------------------------------------------------
c
c --> def2.f
c
c----------------------------------------------------------------

c --> calculation of physical parameters from unphysical stop/sbot parameters
      subroutine def2

c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      complex*16 MSb1shiftDRED
     &  , MSb1shiftDRED1 , MSb1shiftDRED2
     &  , MSb1shiftDRED3 , MSb1shiftDRED4
      complex*16 MSb1shiftDREG
     &  , MSb1shiftDREG1 , MSb1shiftDREG2
     &  , MSb1shiftDREG3 , MSb1shiftDREG4
      double precision ml2t, mr2t, ml2b, mr2b, sgnt, sgnb, tt, bb
     $               , m12, m22, delmsb1

      ml2t = msusytl**2 + mz**2*c2b*( .5d0 - 2d0/3d0*sw2) + mt**2
      mr2t = msusytr**2 + mz**2*c2b*(        2d0/3d0*sw2) + mt**2
      ml2b = msusybl**2 + mz**2*c2b*(-.5d0 + 1d0/3d0*sw2) + mb**2
      mr2b = msusybr**2 + mz**2*c2b*(      - 1d0/3d0*sw2) + mb**2
c      write(*,*) 'diagonal entries:'
c      write(*,*) ml2t, mr2t
      sgnt = (ml2t - mr2t)/dabs(ml2t - mr2t)
      sgnb = (ml2b - mr2b)/dabs(ml2b - mr2b)

      tt = datan(-sgnt*2d0*mt*mlrt/
     $           (-sgnt*(ml2t - mr2t) 
     $            -dsqrt((ml2t - mr2t)**2 + 4d0 * mt**2 * mlrt**2)))

      stt = dsin(tt)
      ctt = dcos(tt)

c      write(*,*) 'stop variables:'
c      write(*,*) real(ml2t), real(mr2t), real(sgnt), real(tt), real(stt)


      bb = datan(-sgnb*2d0*mb*mlrb/
     $           (-sgnb*(ml2b - mr2b) 
     $            -dsqrt((ml2b - mr2b)**2 + 4d0 * mb**2 * mlrb**2)))

      stb = dsin(bb)
      ctb = dcos(bb)

      m12 = .5d0*(ml2t + mr2t 
     $            + sgnt * dsqrt((ml2t - mr2t)**2 + 4d0*mt**2*mlrt**2))
      m22 = .5d0*(ml2t + mr2t 
     $            - sgnt * dsqrt((ml2t - mr2t)**2 + 4d0*mt**2*mlrt**2))

      
      if ((m12.lt.0d0) .or. (m22.lt.0d0))  then
         MSt1 = 0d0
         MSt2 = 0d0
         goto 100
      endif
      MSt1 = dsqrt(m12)
      MSt2 = dsqrt(m22)

 100  continue

      m12 = .5d0*(ml2b + mr2b 
     $            + sgnb * dsqrt((ml2b - mr2b)**2 + 4d0*mb**2*mlrb**2))
      m22 = .5d0*(ml2b + mr2b 
     $            - sgnb * dsqrt((ml2b - mr2b)**2 + 4d0*mb**2*mlrb**2))

      
      if ((m12.lt.0d0) .or. (m22.lt.0d0))  then
         MSb1 = 0d0
         MSb2 = 0d0
         goto 200
      endif
      MSb1 = dsqrt(m12)
      MSb2 = dsqrt(m22)

      if ((MSt1.eq.0d0).or.(MSt2.eq.0d0)) goto 200
      
c      write(*,*) real(msusytl), real(msusytr), 
c     $           real(msusybl), real(msusybr)
c      write(*,*) real(mlrt), real(mlrb)
c      write(*,*) 'Squark masses:'
c      write(*,*) real(MSt1), real(MSt2), real(stt)
c      write(*,*) real(MSb1), real(MSb2), real(stb)

c --> the following constitutes a finite shift in MSb1 in O(alpha_s)
c     which results in a contribution in O(alpha alpha_s) when inserted
c     in the one-loop calculation
      b = datan(tb)
      if (dr1l.eq.0) then
         delmsb1 = 0d0
      elseif (dr1l.eq.1) then
         MSb1shiftDREG = 
     &        + MSb1shiftDREG1() + MSb1shiftDREG2()
     &        + MSb1shiftDREG3() + MSb1shiftDREG4()
         delmsb1 = MSb1shiftDREG
         write(*,*) 'mass shift (DREG) applied for MSb1',
     $              real(msb1), real(dsqrt(msb1**2 + delmsb1))
      elseif (dr1l.eq.2) then
         MSb1shiftDRED = 
     &        + MSb1shiftDRED1() + MSb1shiftDRED2()
     &        + MSb1shiftDRED3() + MSb1shiftDRED4()
         delmsb1 = MSb1shiftDRED
         write(*,*) 'mass shift (DRED) applied for MSb1',
     $              real(msb1), real(dsqrt(msb1**2 + delmsb1))
      else
         write(*,*) 'WARNING: dr1l has forbidden value!!!'
         delmsb1 = 0d0
      endif
      msb1 = dsqrt(msb1**2 + delmsb1)
c      write(*,*) 'def2: MSb1:', real(msb1)

 200  continue


      end

c-------------------------------------------------------------------

c --> calculation of unphysical parameters from physical stop parameters
      subroutine def3(xtb,mst2,delmst,stt,
     $                xmsusytl,xmsusytr,xmsusybl,xmsusybr,xmtlr)

      double precision mst2,delmst,stt,ctt
      double precision xmsusytl,xmsusytr,xmsusybl,xmsusybr,xmtlr,
     $                 xmsusytl2, xmsusytr2
      double precision xmw,xmz,xmt,xtb,xb,xc2b,xsw2,dt1,dt2
      double precision rot(1:2,1:2), rottrans(1:2,1:2), 
     $                 mdiag(1:2,1:2), mnondiag(1:2,1:2), z(1:2,1:2)

      common /smpara1/ xmw,xmz,xmt

c      write(*,*) 'Parameters in def3:'
c      write(*,*) xtb, mst2, stt, delmst

      xb = datan(xtb)
      xc2b = dcos(2d0*xb)
      xsw2 = 1d0 - xmw**2/xmz**2
      dt1 = xmz**2 * xc2b * (.5d0 - 2d0/3d0 * xsw2)
      dt2 = xmz**2 * xc2b * (       2d0/3d0 * xsw2)

      mdiag(1,1) = (mst2 - delmst)**2
      mdiag(1,2) = 0d0
      mdiag(2,1) = 0d0
      mdiag(2,2) = mst2**2
      ctt = dsqrt(1d0 - stt**2)
      rot(1,1) = ctt
      rot(1,2) = stt
      rot(2,1) = -stt
      rot(2,2) = ctt
      rottrans(1,1) = ctt
      rottrans(1,2) = -stt
      rottrans(2,1) = stt
      rottrans(2,2) = ctt

      call matmult(rottrans, mdiag, z)
      call matmult(z, rot, mnondiag)

      xmsusytl2 = mnondiag(1,1) - xmt**2 - dt1
      xmsusytr2 = mnondiag(2,2) - xmt**2 - dt2
      if ((xmsusytl2.ge.0d0).and.(xmsusytr2.ge.0d0)) then
         xmsusytl = dsqrt(mnondiag(1,1) - xmt**2 - dt1)
         xmsusytr = dsqrt(mnondiag(2,2) - xmt**2 - dt2)
         xmsusybl = xmsusytl
         xmsusybr = xmsusybl
         xmtlr = mnondiag(1,2)/xmt
      else
         xmsusytl = 0d0
         xmsusytr = 0d0
         xmsusybl = 0d0
         xmsusybr = 0d0
         xmtlr = 0d0
      endif

c      write(*,*) 'parameters from def3:'
c      write(*,*) xmsusytl, xmsusytr, xmsusybl, xmtlr

      end


c-------------------------------------------------------------------

c --> calculation of physical parameters for one fermion sector
      subroutine def4(msusyl,msusyr,xf,mf,tb,msf1,msf2,stf,f)

c     f = 1: top-type fermions
c     f = 2: bottom type fermions

      double precision msusyl, msusyr, xf, mf, tb, msf1, msf2, stf, ctf
      double precision mw, mz, sw2, c2b, dum
      integer f
      double precision ml2t, mr2t, ml2b, mr2b, sgnt, sgnb, tt, bb
     $               , m12, m22
      common /smpara1/ mw, mz, dum

      sw2 = 1d0 - mw**2/mz**2
      c2b = dcos(2d0 * datan(tb))

      if (f.eq.1) then
         ml2t = msusyl**2 + mz**2*c2b*( .5d0 - 2d0/3d0*sw2) + mf**2
         mr2t = msusyr**2 + mz**2*c2b*(        2d0/3d0*sw2) + mf**2
      elseif (f.eq.2) then
         ml2t = msusyl**2 + mz**2*c2b*(-.5d0 + 1d0/3d0*sw2) + mf**2
         mr2t = msusyr**2 + mz**2*c2b*(      - 1d0/3d0*sw2) + mf**2
      else
         write(*,*) 'fermion selection in def4 out or range'
         stop
      endif
c      write(*,*) 'diagonal entries:'
c      write(*,*) ml2t, mr2t
      sgnt = (ml2t - mr2t)/dabs(ml2t - mr2t)

      tt = datan(-sgnt*2d0*mf*xf/
     $           (-sgnt*(ml2t - mr2t) 
     $            -dsqrt((ml2t - mr2t)**2 + 4d0 * mf**2 * xf**2)))

      stf = dsin(tt)
      ctf = dcos(tt)

c      write(*,*) 'stop variables:'
c      write(*,*) real(ml2t), real(mr2t), real(sgnt), real(tt), real(stt)


      m12 = .5d0*(ml2t + mr2t 
     $            + sgnt * dsqrt((ml2t - mr2t)**2 + 4d0*mf**2*xf**2))
      m22 = .5d0*(ml2t + mr2t 
     $            - sgnt * dsqrt((ml2t - mr2t)**2 + 4d0*mf**2*xf**2))

      
      if ((m12.lt.0d0) .or. (m22.lt.0d0))  then
         MSf1 = 0d0
         MSf2 = 0d0
         goto 100
      endif
      MSf1 = dsqrt(m12)
      MSf2 = dsqrt(m22)

 100  continue

c$$$      write(*,*) "Msusy    :",msusy
c$$$      write(*,*) "Mlr      :",mlr
c$$$      write(*,*) "MT       :", MT
c$$$      write(*,*) "theta_top:",tt
c$$$      write(*,*) "-Pi/4    :",-Pi/4d0
      
c      write(*,*) real(msusytl), real(msusytr), 
c     $           real(msusybl), real(msusybr)
c      write(*,*) real(xf), real(mlrb)
c      write(*,*) 'Squark masses:'
c      write(*,*) real(MSt1), real(MSt2), real(stt)
c      write(*,*) real(MSb1), real(MSb2), real(stb)


      end

c-------------------------------------------------------------------

c --> calculation of unphysical parameters from physical parameters
c     for one sfermion type
      subroutine def4b(msf1, msf2, stf, mf, qf, xtb, msusyl,msusyr,xf,f)

c     f = 1: top-type fermions
c     f = 2: bottom type fermions

      double precision msf1, msf2, stf, mf, qf,
     $                 msusyl, msusyr, xf, msusyl2, msusyr2
      double precision xmw,xmz,xmt,xtb,xb,xc2b,xsw2,dt1,dt2
      double precision rot(1:2,1:2), rottrans(1:2,1:2), 
     $                 mdiag(1:2,1:2), mnondiag(1:2,1:2), z(1:2,1:2)
      integer f

      common /smpara1/ xmw,xmz,xmt

c      write(*,*) 'Parameters in def4b:',
c     $     real(msf1), real(msf2), real(stf), real(mf), real(qf), 
c     $     real(xtb)

      xb = datan(xtb)
      xc2b = dcos(2d0*xb)
      xsw2 = 1d0 - xmw**2/xmz**2
      if (f.eq.1) then
         dt1 = xmz**2 * xc2b * (.5d0 - qf * xsw2)
         dt2 = xmz**2 * xc2b * (       qf * xsw2)
      elseif (f.eq.2) then
         dt1 = xmz**2 * xc2b * (-.5d0 - qf * xsw2)
         dt2 = xmz**2 * xc2b * (        qf * xsw2)
      else
         write(*,*) 'Error in def4b: sfermion selection'
      endif

      mdiag(1,1) = msf1**2
      mdiag(1,2) = 0d0
      mdiag(2,1) = 0d0
      mdiag(2,2) = msf2**2
      ctf = dsqrt(1d0 - stf**2)
      rot(1,1) = ctf
      rot(1,2) = stf
      rot(2,1) = -stf
      rot(2,2) = ctf
      rottrans(1,1) = ctf
      rottrans(1,2) = -stf
      rottrans(2,1) = stf
      rottrans(2,2) = ctf

      call diagonalization(rottrans, mdiag, rot, mnondiag)

      msusyl2 = mnondiag(1,1) - mf**2 - dt1
      msusyr2 = mnondiag(2,2) - mf**2 - dt2
      if ((msusyl2.ge.0d0).and.(msusyr2.ge.0d0)) then
         msusyl = dsqrt(mnondiag(1,1) - mf**2 - dt1)
         msusyr = dsqrt(mnondiag(2,2) - mf**2 - dt2)
         xf = mnondiag(1,2)/mf
      else
         msusyl = 0d0
         msusyr = 0d0
         xf = 0d0
      endif

c      write(*,*) 'parameters from def4b:',
c     $           real(msusyl), real(msusyr), real(xf)
c      write(*,*) real(mnondiag(1,1)), real(mnondiag(1,2)),
c     $           real(mnondiag(2,1)), real(mnondiag(2,2))

      end


c-------------------------------------------------------------------

c --> calculation of unphysical parameters from physical stop/sbot parameters
      subroutine def5(tb,mt,mw,mz,mst1,mst2,stt,msbotr, xb,
     $                msusytl,msusytr,msusybl,msusybr,mtlr,mblr)

      double precision mst1,mst2,stt,ctt
      double precision msusytl,msusytr,msusybl,msusybr,mtlr,mblr,
     $                 msusytl2, msusytr2, msbotr, xb
      double precision mw,mz,mt,tb,b,c2b,sw2,dt1,dt2
      double precision rot(1:2,1:2), rottrans(1:2,1:2), 
     $                 mdiag(1:2,1:2), mnondiag(1:2,1:2), z(1:2,1:2)

c      write(*,*) 'Parameters in def5:'
c      write(*,*) real(tb), real(mt), real(mw), real(mz), real(mst1),
c     $           real(mst2), real(stt), real(msbotr), real(xb)

      b = datan(tb)
      c2b = dcos(2d0*b)
      sw2 = 1d0 - mw**2/mz**2
      dt1 = mz**2 * c2b * (.5d0 - 2d0/3d0 * sw2)
      dt2 = mz**2 * c2b * (       2d0/3d0 * sw2)

      mdiag(1,1) = mst1**2
      mdiag(1,2) = 0d0
      mdiag(2,1) = 0d0
      mdiag(2,2) = mst2**2
      ctt = dsqrt(1d0 - stt**2)
      rot(1,1) = ctt
      rot(1,2) = stt
      rot(2,1) = -stt
      rot(2,2) = ctt
      rottrans(1,1) = ctt
      rottrans(1,2) = -stt
      rottrans(2,1) = stt
      rottrans(2,2) = ctt

      call diagonalization(rottrans, mdiag, rot, mnondiag)
c      write(*,*) 'after diagonalization'
 
      msusytl2 = mnondiag(1,1) - mt**2 - dt1
      msusytr2 = mnondiag(2,2) - mt**2 - dt2
      if ((msusytl2.ge.0d0).and.(msusytr2.ge.0d0)) then
         msusytl = dsqrt(mnondiag(1,1) - mt**2 - dt1)
         msusytr = dsqrt(mnondiag(2,2) - mt**2 - dt2)
         msusybl = msusytl
         msusybr = msbotr
         mtlr = mnondiag(1,2)/mt
         mblr = xb
      else
         msusytl = 0d0
         msusytr = 0d0
         msusybl = 0d0
         msusybr = 0d0
         mtlr = 0d0
         mblr = 0d0
      endif
c      write(*,*) 'def5 completed'

      end


c-------------------------------------------------------------------

      subroutine matmult(a, b, c)

      double precision a(1:2,1:2), b(1:2,1:2), c(1:2,1:2)

      c(1,1) = a(1,1)*b(1,1) + a(1,2)*b(2,1)
      c(1,2) = a(1,1)*b(1,2) + a(1,2)*b(2,2)
      c(2,1) = a(2,1)*b(1,1) + a(2,2)*b(2,1)
      c(2,2) = a(2,1)*b(1,2) + a(2,2)*b(2,2)

      end

c-------------------------------------------------------------------

      subroutine diagonalization(a, b, c, d)

      double precision a(1:2,1:2), b(1:2,1:2), c(1:2,1:2), d(1:2,1:2),
     $                 e(1:2,1:2)

      call matmult(a, b, e)
      call matmult(e, c, d)

      end

c-------------------------------------------------------------------

      subroutine diagonalization2(cos11, sin12, b, d)

      double precision a(1:2,1:2), b(1:2,1:2), c(1:2,1:2), d(1:2,1:2),
     $                 e(1:2,1:2), cos11, sin12

      a(1,1) = cos11
      a(1,2) = sin12
      a(2,1) = -a(1,2)
      a(2,2) = a(1,1)
      c(1,1) = a(1,1)
      c(1,2) = a(2,1)
      c(2,1) = a(1,2)
      c(2,2) = a(2,2)


      call matmult(a, b, e)
      call matmult(e, c, d)

      end


c----------------------------------------------------------------
c
c --> mhalphatsq.f
c
c----------------------------------------------------------------
      

      subroutine BDSZHiggs(t,A0,BL,T1,T2,st,ct,q,mu,tanb,v2,OS,
     $     S11,S22,S12)

c     Two-loop O(a_t^2) corrections to the CP-even Higgs mass matrix. 
c     Routine written by P. Slavich (e-mail: slavich@pd.infn.it).
c     Based on A. Brignole, G. Degrassi, P. Slavich and F. Zwirner, 
c     hep-ph/0112177.
c
c     Last update:  13/12/2001.
c
c
c     I/O PARAMETERS:
c     t = m_top^2, A0 = m_A^2, BL = m_sbotL^2, T1 = m_stop1^2, T2 = m_stop2^2,
c     st = sin(theta_stop), ct = cos(theta_stop), q = Q^2 (ren. scale),
c     mu = Higgs mixing parameter, tanb = tan(beta), v2 = v^2, 
c     OS = renormalization scheme for 1-loop (0 = DRbar, 1 = On-Shell),
c     Sij = 2-loop corrections to the CP-even Higgs mass matrix elements.

      implicit none

      integer OS
      real*8 ht,k,mt,pi,v2
      real*8 t,mu2,A0,BL,T1,T2,st,ct,q,A,X,mu,tanb,sb,cb,s2t,c2t
      real*8 F1,F2,F3,dmuF2,dmuF3,dAtF2,dAtF3,DM12,DM22
      real*8 DF1,DF2,DF3,DdmuF2,DdmuF3,DdAtF2,DdAtF3,F2_s
      real*8 S11,S22,S12,osdr,DMom,ShiftB,ShiftB2,ShiftB3

      pi = 3.14159265897d0

      mt = dsqrt(t)

      s2t = 2d0*ct*st
      c2t = ct**2 - st**2

      X = (T1-T2)*s2t/2d0/mt    
      A = X - mu/tanb           

      sb = dsin(datan(tanb))
      cb = dcos(datan(tanb))
      
      ht = dsqrt(2d0/v2)*mt/sb

      k = 3d0*ht**2/(16d0*Pi**2)**2 
      
      call funcs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,F1,F2,F3)
      call sfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,A,ht,
     $     dmuF2,dmuF3,dAtF2,dAtF3,DM12,DM22)
      call dfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,A,v2,
     $     DF1,DF2,DF3,DdmuF2,DdmuF3,DdAtF2,DdAtF3)

      osdr = 1d0*OS
 
      if(s2t.ne.0d0.and.A.ne.0d0) then

         S11 = .5d0 * ht**2 * mu**2 * s2t**2 * (F3 + 2d0*dmuF3 +
     $        osdr*(DF3 + 2d0*DdmuF3))
         
         S12 = .5d0 * ht**2 * mu * A  * s2t**2 * (F3 + dmuF3 + dAtF3 +
     $        osdr*(DF3 + DdmuF3 + DdAtF3)) + 
     $        ht**2 * mt * mu * s2t * (F2 + dmuF2 +
     $        osdr*(DF2 + DdmuF2))
         
         S22 = .5d0 * ht**2 * A**2 * s2t**2 * (F3 + 2d0*dAtF3 + 
     $        osdr*(DF3 + 2d0*DdAtF3)) + 
     $        2d0 * ht**2 * mt * A * s2t * (F2 + dAtF2 + 
     $        osdr*(DF2 + DdAtF2)) + 
     $        2d0 * ht**2 * mt**2 * (F1 + osdr*DF1)
         
c     some of the functions have poles in s2t=0 or in A=0. 
c     when necessary we consider the residues:
         
      elseif(s2t.eq.0d0.and.A.eq.0d0) then
         
         S11 = 0d0
         S12 = 0d0
         S22 = 2 * ht**2 * mt**2 * (F1 + osdr*DF1)

      elseif(s2t.eq.0d0.and.A.ne.0d0) then 

         call resfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,F2_s)

         S11 = 0d0
         S12 = ht**2 * mt * mu * (F2_s + osdr*DF2)
         S22 = 2d0 * ht**2 * mt**2 * (F1 + osdr*DF1) +
     $        2d0 * ht**2 * mt * A * (F2_s + osdr*(DF2 + DdAtF2))

      elseif(s2t.ne.0d0.and.A.eq.0d0) then

         S11 = .5d0 * ht**2 * mu**2 * s2t**2 * (F3 + 2d0*dmuF3 +
     $        osdr*(DF3 + 2d0*DdmuF3))
         S12 = .5d0 * ht**2 * mu * s2t**2 * osdr*DdAtF3 +
     $        ht**2 * mt * mu * s2t * (F2 + dmuF2 +
     $        osdr*(DF2 + DdmuF2))
         S22 = 2d0 * ht**2 * mt**2 * (F1 + osdr*DF1) +
     $        2d0 * ht**2 * mt * s2t * osdr*DdAtF2
 
      endif
         
      S11 = k*S11
      S12 = k*(S12 + DM12)
      S22 = k*(S22 + DM22)
      
      return
      end

*
***********************************************************************
*

      subroutine funcs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,F1,F2,F3)

      implicit none
      real*8 t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,F1,F2,F3
      real*8 F1ab,F1c,F2ab,F2c,F3ab,F3c

      F1 = F1ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu) 
     $     + F1c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)
     $     + F1c(t,A0,BL,T2,T1,-s2t,-c2t,cb,sb,q,mu)
      
      F2 = F2ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu) 
     $     + F2c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)
     $     - F2c(t,A0,BL,T2,T1,-s2t,-c2t,cb,sb,q,mu)

      F3 = F3ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu) 
     $     + F3c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)
     $     + F3c(t,A0,BL,T2,T1,-s2t,-c2t,cb,sb,q,mu)

      return
      end

*
*********************************************************************
*
            
      function F1ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi
      real*8 F1ab

      mu2 = mu**2
      if(mu2.eq.0d0) mu2 = 1d-10

      F1ab = 
     $     (2.*BL*mu2/t*(mu2-BL+t)/delt(BL,mu2,t)
     $     +(BL+mu2-t)/t)*phi(BL,mu2,t)
     $     +2.*A0*cb**2*(A0-6.*t)/(A0-4.*t)/t*phi(A0,t,t)
     $     -2.*cb**2*bdszLi2(1.-A0/t)
     $     -2. -(2.+sb**2)/3.*Pi**2
     $     + Log(t/q)*(
     $     (4.*(BL-mu2-10.*t)*t+A0*
     $     (mu2-BL+(6.+4.*sb**2)*t))/(A0-4.*t)/t
     $     +1./delt(BL,mu2,t)*((BL-mu2)**3/t
     $     +(2.*mu2**2+2.*BL*mu2+5.*BL*t+mu2*t-4.*BL**2-2.*t**2)))
     $     +Log(A0/q)*(4.*A0*cb**2/(A0-4.*t))
     $     +Log(BL/q)*(-BL/t+BL*(-BL+mu2+t)**2/t/delt(BL,mu2,t))
     $     +Log(mu2/q)*(mu2/t+mu2*(t**2-(BL-mu2)**2)/t/delt(BL,mu2,t))
     $     +Nc*(Log(t/q)**2-Log(T1/q)**2/2.-Log(T2/q)**2/2.)
     $     +Log(BL/q)*Log(mu2/q)-Log(BL/q)*Log(t/q)-Log(mu2/q)*Log(t/q)
     $     -3.*Log(t/q)**2-2.*Log(T1/q)**2-2.*Log(T2/q)**2

      return
      end
      
*
*********************************************************************
*

      function F1c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi,Xt,Yt,st2,ct2
      real*8 F1c

      mu2 = mu**2
      if(mu2.eq.0d0) mu2 = 1d-10

      Xt  = s2t*(T1-T2)/2d0/Sqrt(t)
      Yt  = s2t*(T1-T2)/2d0/Sqrt(t) - mu/sb/cb
      ct2 = (1d0+c2t)/2d0
      st2 = (1d0-c2t)/2d0
 
      F1c =
     $     2.*mu2**2*(mu2+t-T1)/T1/delt(T1,mu2,t)*phi(mu2,t,T1)
     $     -A0**2*(1+c2t**2)*cb**2*Yt**2
     $     /2./T2/delt(A0,T1,T2)*phi(A0,T1,T2)
     $     -cb**2*A0/T1*((2.*Sqrt(t)+s2t*Yt)/Sqrt(t)
     $     +(2.*Sqrt(t)+s2t*Yt)**2/2./(A0-4.*T1))*phi(A0,T1,T1)
     $     -cb**2/T1*phi(A0,BL,T1)*(
     $     2.*A0*BL*(ct2*t+s2t*Yt*Sqrt(t)+st2*Yt**2)/delt(A0,T1,BL)
     $     +(A0+BL-T1)*((1.+c2t)*Sqrt(t)+s2t*Yt)/2./Sqrt(t))
     $     +sb**2*(1+c2t+s2t*Xt/Sqrt(t))*bdszLi2(1-BL/T1)
     $     +(1-c2t)*bdszLi2(1-mu2/T1)
     $     +s2t**2*(T1-T2)**2/4./T1/T2*Nc
     $     +(1./3.+s2t*(sb**2*(Xt-Yt)+Yt)/4./Sqrt(t))*Pi**2
     $     -3.*s2t*(sb**2*Xt+cb**2*Yt)*(2.*t-T1)/2./Sqrt(t)/T1
     $     -(sb**2*Xt**2+cb**2*Yt**2)/4./T1/T2*
     $     ((1.+c2t**2)*T1+(5.-2.*c2t-c2t**2)*T2)
     $     +(3.-c2t)*mu2/T1-(3.-c2t)*cb**2*A0/2./T1
     $     -(1.+c2t)*t/2./T1-(1.-c2t)*BL/2./T1
     $     -(1.+c2t**2)*(T1**2+T2**2)/4./T1/T2
     $     +5./2.+c2t/2.-s2t**2/2.
     $     +Log(t/q)*(1.+mu2/t-t/T1-T1/t
     $     +(-(mu2-T1)**3/t + 4.*mu2**2 + 5.*mu2*t + 2.*t**2
     $     +mu2**2*t/T1-t**3/T1-2.*mu2*T1-2.*T1**2)/delt(T1,mu2,t))
     $     +Log(mu2/q)*(mu2*((-2.+c2t)*t+T1)/t/T1
     $     -mu2*((T1-t)**3+2.*mu2*(t-T1)*T1+mu2**2*(t+T1))
     $     /t/T1/delt(mu2,t,T1))
     $     +Log(BL/q)*((1-c2t)/2.*BL/T1
     $     -cb**2*BL*(A0-BL+T1)/T1/delt(A0,T1,BL)*
     $     (ct2*t+s2t*Yt*Sqrt(t)+st2*Yt**2)
     $     +sb**2*BL*(ct2*t+s2t*Xt*Sqrt(t)+st2*Xt**2)/(BL-T1)/T1)
     $     +Log(A0/q)*((3.-c2t)*A0*cb**2/2./T1
     $     +A0*cb**2*(2.*Sqrt(t)+s2t*Yt)**2/2./(A0-4.*T1)/T1
     $     +A0*(1+c2t**2)*cb**2*(A0*(T1+T2)-(T1-T2)**2)*Yt**2
     $     /4./T1/T2/delt(A0,T1,T2)
     $     +A0*cb**2*(A0-BL-T1)/T1/delt(A0,BL,T1)*
     $     (ct2*t+s2t*Yt*Sqrt(t)+st2*Yt**2))
     $     +Log(T2/q)*((1+c2t**2)*T2/4./T1-Nc*s2t**2*T2/4./T1
     $     -cb**2*(1+c2t**2)*Yt**2/4./T2
     $     +sb**2*(1+c2t**2)*Xt**2/4./T1
     $     +cb**2*(1+c2t**2)*Yt**2/4./T1/T2/delt(A0,T1,T2)*
     $     (A0**2*T1-A0*(2.*T1**2+5*T1*T2+T2**2)+(T1-T2)**2*(T1+T2)))
     $     +Log(T1/q)*(cb**2*(ct2*t+s2t*Yt*Sqrt(t)+st2*Yt**2)*
     $     (1./T1-(A0+BL-T1)/delt(A0,T1,BL))
     $     +cb**2*(1+c2t**2)*Yt**2/4./T1/T2/delt(A0,T1,T2)*
     $     (A0**2*T2-A0*(T1**2+5*T1*T2+2.*T2**2)+(T1-T2)**2*(T1+T2))
     $     +1./delt(T1,mu2,t)*((mu2-T1)**2*T1/t-(mu2-t)**2*(mu2+t)/T1
     $     +6.*mu2**2+4.*mu2*t+2.*t**2-3.*mu2*T1-2.*T1**2)
     $     +cb**2*(A0-8.*T1)/2./T1/(A0-4.*T1)*(2*Sqrt(t)+s2t*Yt)**2
     $     +sb**2*(BL-2.*T1)/T1/(BL-T1)*
     $     (ct2*t+s2t*Xt*Sqrt(t)+st2*Xt**2)
     $     -(1-c2t)*(mu2-2.*T1)/2./T1-s2t**2*Nc*(T1-2.*T2)/4./T2
     $     +sb**2*c2t*(1-c2t)*Xt**2/2./T1
     $     +sb**2*(1+c2t**2)*Xt**2/4./T2
     $     +sb**2*s2t*(t-3.*T1)/Sqrt(t)/T1*Xt
     $     -3.*cb**2*s2t*(t+T1)/Sqrt(t)/T1*Yt
     $     -cb**2*(5.-2.*c2t-c2t**2)*Yt**2/4./T1
     $     -(3.+c2t-8.*sb**2)*t/2./T1
     $     +(3.-c2t)/2./T1*mu2+(1+c2t**2)*T1/4./T2
     $     -T1/t+(-14+c2t-c2t**2)/2.)
     $     +(Nc+1.)*s2t**2/4.*
     $     (3.*Log(T1/q)**2-2.*Log(T1/q)*Log(T2/q)-Log(T2/q)**2)
     $     +s2t*(6.*sb**2*Xt+5*cb**2*Yt)/2./Sqrt(t)*Log(T1/q)**2
     $     +(9.+sb**2-cb**2*c2t)/2.*Log(T1/q)**2
     $     +Log(T1/q)*Log(T2/q)-2.*Log(t/q)*Log(T1/q)
     $     +cb**2*(1.+c2t+s2t*Yt/Sqrt(t))/2.*(Log(A0/q)*Log(T1/q)
     $     +Log(BL/q)*Log(T1/q)-Log(A0/q)*Log(BL/q))

      return
      end

*
*********************************************************************
*

      function F2ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)
      
      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi
      real*8 F2ab
      
      F2ab = -(3.+Nc)/2.*(Log(T1/q)**2-Log(T2/q)**2)

      return
      end

*
*********************************************************************
*

      function F2c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi,ct2,st2,Xt,Yt,At
      real*8 F2c

      mu2 = mu**2
      if(mu2.eq.0d0) mu2 = 1d-10

      Xt = s2t*(T1-T2)/2d0/Sqrt(t)
      Yt = Xt - mu/cb/sb
      At = sb**2*Xt+cb**2*Yt

      ct2 = (1d0+c2t)/2d0
      st2 = (1d0-c2t)/2d0

      F2c = 4*mu2**2*t/T1/delt(mu2,t,T1)*phi(mu2,t,T1)
     $ +(A0*c2t**2*Yt**2/(T1-T2)/T2 
     $ +(1+c2t**2)/2.*(T1/T2-1)*Yt**2*A0/delt(A0,T1,T2))
     $ *cb**2*phi(A0,T1,T2)
     $ -(2*A0*c2t**2*Sqrt(t)*Yt/s2t/T1/(T1-T2)
     $ +s2t*(A0*Yt/2./Sqrt(t)/T1+Yt*A0*(A0-4*T1)/2./Sqrt(t)/T1/(T1-T2))
     $ + A0*(2*Sqrt(t)+s2t*Yt)**2/2./T1/(A0-4*T1)
     $ + A0/T1+A0*c2t**2*Yt**2/T1/(T1-T2))*cb**2*phi(A0,T1,T1)
     $ -(2*A0*BL/T1*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)/delt(A0,BL,T1)
     $ +c2t**2*(A0+BL-T1)*Yt*Sqrt(t)/s2t/T1/(T1-T2)
     $ +s2t*((A0+BL-T1)*Yt/4./Sqrt(t)/T1
     $ +delt(A0,BL,T1)*Yt/2./Sqrt(t)/T1/(T1-T2))
     $ +ct2*(A0+BL-T1)/2./T1+c2t*Yt**2*(A0+BL-T1)/2./T1/(T1-T2)
     $ +c2t*delt(A0,BL,T1)/2./T1/(T1-T2)
     $ -c2t*t*(A0+BL-T1)/2./T1/(T1-T2))*cb**2*phi(A0,BL,T1)
     $ +(s2t*(BL-T1)*Xt/Sqrt(t)/(T1-T2)+s2t*Xt/2./Sqrt(t)+ct2
     $ -c2t*(t+T1-BL)/(T1-T2)+c2t*Xt*(2*c2t*Sqrt(t)+s2t*Xt)
     $ /s2t/(T1-T2))*sb**2*bdszLi2(1-BL/T1)
     $ +(1-c2t-2*c2t*(mu2-T1)/(T1-T2))*bdszLi2(1-mu2/T1)
     $ -21*s2t*T1*At/2./(T1-T2)/Sqrt(t)+c2t*T1/2./(T1-T2)
     $ +3*s2t*At/4./Sqrt(t)-3*s2t*Sqrt(t)*At/T1
     $ +c2t*(2*BL+2*A0*cb**2-4*mu2-2*t+T1
     $ +2*(sb**2*Xt**2+cb**2*Yt**2))/4./T1
     $ +(-5+c2t**2)/4./T1*(sb**2*Xt**2+cb**2*Yt**2)
     $ -(2*BL+6*A0*cb**2-12*mu2+2*t+(1+c2t**2-Nc*s2t**2)*T2)/4./T1
     $ +((1+c2t**2-Nc*s2t**2)*T1
     $ +(1+c2t**2)*(sb**2*Xt**2+cb**2*Yt**2))/4./T2
     $ -t/T1*Log(t/q)+(2*mu2*(t-T1)*T1+mu2**2*(t+T1)-(t-T1)**3)
     $ /T1/delt(T1,mu2,t)*Log(t*T1/q**2)-Log(mu2/q)*(2-c2t)*mu2/T1
     $ +mu2*((t-T1)**2-mu2**2)/T1/delt(T1,mu2,t)*Log(mu2*T1/q**2)
     $ +Log(BL/q)*(st2*BL/T1+sb**2*BL/T1/(BL-T1)*
     $ (ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2))
     $ -cb**2*BL*(A0-BL+T1)*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)
     $ /T1/delt(A0,T1,BL)*Log(BL*T1/q**2)
     $ +Log(A0/q)*((3-c2t)*cb**2*A0/2./T1
     $ +A0*cb**2*(2*Sqrt(t)+s2t*Yt)**2/2./(A0-4*T1)/T1)
     $ -cb**2*A0*(1+c2t**2)*(A0-T1-T2)*(T1-T2)*Yt**2
     $ /4./T1/T2/delt(A0,T1,T2)*Log(A0*T1/q**2)
     $ +cb**2*A0*(A0-BL-T1)*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)
     $ /T1/delt(A0,T1,BL)*Log(A0*T1/q**2)
     $ +Log(T2/q)*(c2t**2*(1+Nc)*(T1+T2)/(T1-T2)
     $ +c2t**2*(sb**2*Xt**2+cb**2*Yt**2)/(T1-T2)
     $ -(1+c2t**2)*sb**2*(T1-T2)*Xt**2/4./T1/T2
     $ +(1+c2t**2-s2t**2*Nc)*T2/4./T1
     $ +(1+c2t**2)*(sb**2*Xt**2+cb**2*Yt**2)/4./T2)
     $ -(1+c2t**2)*cb**2*Yt**2*(A0**2*T1+(T1-T2)**3
     $ -A0*(2*T1**2+3*T1*T2-T2**2))
     $ /4./T1/T2/delt(A0,T1,T2)*Log(T2*T1/q**2)
     $ +Log(T1/q)*(4*mu2*(mu2+t-T1)/delt(T1,mu2,t)
     $ +2*(1+c2t**2)*cb**2*Yt**2*A0*(A0-T1-3*T2)/4./T2/delt(A0,T1,T2)
     $ -2*cb**2*(A0+BL-T1)*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)
     $ /delt(A0,T1,BL)+cb**2*(2*Sqrt(t)+s2t*Yt)**2/2./T1
     $ -2*cb**2*(2*Sqrt(t)+s2t*Yt)**2/(A0-4*T1)
     $ +sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/T1
     $ -sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/(BL-T1)
     $ -st2*mu2/T1+1-c2t-6*c2t**2*Sqrt(t)*At/s2t/(T1-T2)
     $ +9*s2t*T1*At/Sqrt(t)/(T1-T2)
     $ -c2t*(1+c2t)*(sb**2*Xt**2+cb**2*Yt**2)/(T1-T2)
     $ -(1+c2t**2)*sb**2*(T1-T2)*Xt**2/4./T1/T2
     $ -c2t*(BL+A0*cb**2-2*mu2-t+(1+3*c2t*(1+Nc))*T1
     $ -c2t*(1+Nc)*T2)/(T1-T2)
     $ -3*s2t*At/2./Sqrt(t)+s2t*Sqrt(t)*(sb**2*Xt-3*cb**2*Yt)/T1
     $ -(3-2*c2t+c2t**2)/4./T1*(sb**2*Xt**2+cb**2*Yt**2)
     $ +s2t**2/2./T1*(sb**2*Xt**2-cb**2*Yt**2)
     $ -6+c2t+(Nc+1)*s2t**2/2.+((3-c2t)*mu2-(c2t+8*cb**2-5)*t)/2./T1
     $ -(2*c2t**2+(1-Nc)*s2t**2)*T1/4./T2)
     $ +Log(T2/q)**2*(3*c2t**2*Sqrt(t)*At/2./s2t/(T1-T2)
     $ +c2t*(1-2*c2t)*(sb**2*Xt**2+cb**2*Yt**2)/4./(T1-T2)+3/8.
     $ +c2t*(2*BL+2*A0*cb**2-4*mu2-2*t+(1-2*c2t*(1+Nc))*T1
     $ +(1-6*c2t*(1+Nc))*T2)/8./(T1-T2))
     $ +Log(T1/q)**2*(3*s2t/2./Sqrt(t)*At
     $ +s2t*cb**2*A0*Yt/(T1-T2)/Sqrt(t)
     $ +s2t*(BL-6*T1)*At/2./(T1-T2)/Sqrt(t)
     $ +9*c2t**2*Sqrt(t)/2./s2t*At/(T1-T2)
     $ +3*c2t*(1+2*c2t)*(sb**2*Xt**2+cb**2*Yt**2)/4./(T1-T2)
     $ +c2t*(6*BL+6*A0*cb**2-12*mu2-6*t+(7+26*c2t*(1+Nc))*T1
     $ -(1+2*c2t*(1+Nc))*T2)/8./(T1-T2)+25/8.-c2t/2.+s2t**2*(1+Nc))
     $ -(s2t*(2*A0+2*BL-T1-T2)*Yt/4./Sqrt(t)/(T1-T2)
     $ +(1+c2t)/4.+c2t**2*Sqrt(t)*Yt/s2t/(T1-T2)
     $ +c2t*(A0+BL-t-T1+Yt**2)/2./(T1-T2))*cb**2*Log(A0/T1)*Log(BL/T1)
     $ -cb**2*s2t*2*A0*Yt/(T1-T2)/Sqrt(t)*Log(A0/q)*Log(T1/q)
     $ -s2t*BL*At/(T1-T2)/Sqrt(t)*Log(BL/q)*Log(T1/q)
     $ -(c2t**2*(1+Nc)*(T1+T2)/(T1-T2)
     $ +c2t**2*(sb**2*Xt**2+cb**2*Yt**2)/(T1-T2))*Log(T1/q)*Log(T2/q)


      return
      end

*
*********************************************************************
*
      
      function F3ab(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi
      real*8 F3ab

      F3ab = (2.+Nc)/2.*(2.-Log(T1/q)-Log(T2/q))
     $     *(2.-(T1+T2)/(T1-T2)*Log(T1/T2))

      return
      end
      
*
*********************************************************************
*

      function F3c(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi,ct2,st2,Xt,Yt,At
      real*8 F3c

      mu2 = mu**2
      if(mu2.eq.0d0) mu2 = 1d-10

      Xt = s2t*(T1-T2)/2d0/Sqrt(t)
      Yt = Xt - mu/cb/sb
      At = sb**2*Xt+cb**2*Yt

      ct2 = (1d0+c2t)/2d0
      st2 = (1d0-c2t)/2d0

      F3c =
     $ (2*mu2*t*(mu2+t-T1)/T1/delt(T1,mu2,t)
     $ -(4*mu2*t+2*delt(T1,mu2,t))/T1/(T1-T2)
     $ -(mu2+t-T1)/T1)*phi(mu2,t,T1)
     $ +(A0*(1+c2t**2)*(A0-2*(T1+T2))*Yt**2/2./T2/delt(A0,T1,T2)
     $ +4*A0*c2t**2*(A0-2*(T1+T2))*Yt**2/T2/(T1-T2)**2
     $ -(1-3*c2t**2)*Yt**2/2./T2)*cb**2*phi(A0,T1,T2)
     $ +(-A0*(2*Sqrt(t)+s2t*Yt)**2/2./T1/(A0-4*T1)
     $ +A0*(2*Sqrt(t)+s2t*Yt)**2/2./T1/(T1-T2)
     $ -2*A0*c2t**2*Yt**2*(2*A0-7*T1-T2)/T1/(T1-T2)**2
     $ +2*A0*(1-3*c2t**2)*Sqrt(t)*(A0-4*T1)*Yt/s2t/T1/(T1-T2)**2
     $ -4*A0*c2t**2*Sqrt(t)*Yt/s2t/T1/(T1-T2))*cb**2*phi(A0,T1,T1)
     $ +(-2*A0*BL*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)/T1/delt(A0,T1,BL)
     $ +2*(1-3*c2t**2)*Sqrt(t)*Yt*delt(A0,T1,BL)/s2t/T1/(T1-T2)**2
     $ -2*c2t**2*Sqrt(t)*Yt*(A0+BL-T1)/s2t/T1/(T1-T2)
     $ +3*c2t*delt(A0,T1,BL)*(t-Yt**2)/T1/(T1-T2)**2
     $ +(A0+BL-T1)*(c2t*(t-Yt**2)+(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2))
     $ /T1/(T1-T2))*cb**2*phi(A0,BL,T1)
     $ -(1-3*c2t**2)*sb**2*Xt**2/(T1-T2)*bdszLi2(1-T2/T1)
     $ +(1-c2t+2*(1-3*c2t)*(mu2-T1)/(T1-T2)
     $ -6*c2t*(mu2-T1)**2/(T1-T2)**2)*bdszLi2(1-mu2/T1)
     $ +(-4*(1-3*c2t**2)*sb**2*Sqrt(t)*(BL-T1)*Xt/s2t/(T1-T2)**2
     $ +4*c2t**2*sb**2*Sqrt(t)*Xt/s2t/(T1-T2)
     $ -2*sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/(T1-T2)
     $ -2*sb**2*c2t*(3*BL-2*T1-T2)*(t-Xt**2)/(T1-T2)**2)
     $  *bdszLi2(1-BL/T1)
     $ +2*Sqrt(t)*At*(3*(7-18*c2t**2)*T1-9*c2t**2*T2)/s2t/(T1-T2)**2
     $ -3*s2t*Sqrt(t)*At*(4*T1-T2)/T1/(T1-T2)+2*c2t*At**2/(T1-T2)
     $ -21*c2t*T1*(sb**2*Xt**2+cb**2*Yt**2)/(T1-T2)**2
     $ +5*c2t/2./(T1-T2)*(sb**2*Xt**2+cb**2*Yt**2)
     $ -(6-2*c2t)/4./T1*(sb**2*Xt**2+cb**2*Yt**2)
     $ +3*mu2/T1+c2t*mu2*(T1+T2)/T1/(T1-T2)+18*c2t*mu2*T1/(T1-T2)**2
     $ +3*cb**2*A0*c2t/(T1-T2)-cb**2*A0*(3-c2t)/2./T1
     $ -12*(cb**2*A0+BL)*T1*c2t/(T1-T2)**2
     $ +3*c2t*BL/(T1-T2)-(1-c2t)*BL/2./T1
     $ +15*c2t*T1*t/(T1-T2)**2-3*c2t*t/2./(T1-T2)-(1+c2t)*t/2./T1
     $ -s2t**2/2.*(1+Nc)-c2t**2*(1+Nc)*(T1-T2)**2/4./T1/T2
     $ -9*(1+Nc)*T1*c2t**2/(T1-T2)-3*T1*(2*T1+3*T2)/(T1-T2)**2*c2t
     $ -(1-Nc)*(T1-T2)**2/4./T1/T2-(28+6*Nc)*T1/2./(T1-T2)
     $ +Log(t/q)*(-t/T1)
     $ -Log(t*T1/q**2)*t*((t-T1)**2-mu2**2)/T1/delt(T1,mu2,t)
     $ +Log(mu2/q)*(-6*mu2*(3*mu2-T2)*c2t/(T1-T2)**2-(2-c2t)*mu2/T1)
     $ -Log(mu2*T1/q**2)*mu2*(T1**2+mu2**2-t**2-2*mu2*T1)
     $ /T1/delt(T1,mu2,t)+Log(BL/q)*(12*BL*c2t*T1/(T1-T2)**2
     $ +sb**2*BL*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/T1/(BL-T1)
     $ +(1-c2t)*BL/2./T1-3*c2t*BL/(T1-T2))
     $ -Log(BL*T1/q**2)*cb**2*BL*(A0-BL+T1)/T1/delt(A0,T1,BL)
     $ *(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)
     $ +Log(A0/q)*A0*cb**2*((2*Sqrt(t)+s2t*Yt)**2/2./(A0-4*T1)/T1
     $ +12*c2t*T1/(T1-T2)**2+(3-c2t)/2./T1-3*c2t/(T1-T2))
     $ -Log(A0*T1/q**2)*A0*cb**2*
     $ (Yt**2*(1+c2t**2)/4./T1/T2/delt(A0,T1,T2)
     $ *((T1+T2)**2-A0*(T1+T2)+4*T1*T2)-(A0-BL-T1)/T1/delt(A0,T1,BL)
     $ *(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2))
     $ +Log(T2/q)*(12*c2t**2*T2*Sqrt(t)*At/s2t/(T1-T2)**2
     $ -6*s2t*Sqrt(t)*T2*At/(T1-T2)**2
     $ -3*(1-c2t)*cb**2*A0*T2/(T1-T2)**2-(1-3*c2t)*T2*BL/(T1-T2)**2
     $ +6*(1-c2t)*T2*mu2/(T1-T2)**2-(1+3*c2t)*T2*t/(T1-T2)**2
     $ +(sb**2*Xt**2+cb**2*Yt**2)*(-(2-3*c2t+23*c2t**2)*T2/(T1-T2)**2
     $ -(1+5*c2t**2)/2./(T1-T2)-(1+c2t**2)/4./T2)
     $ +(1+c2t**2)*sb**2*Xt**2*(T1**2-4*T1*T2-T2**2)/4./T1/T2/(T1-T2)
     $ -(1+Nc)*c2t**2/4./(T1-T2)**2*(9*T1**2+32*T1*T2+19*T2**2)
     $ -(1+Nc)*(T1-T2)/4./T1*c2t**2+T2*(2*T1+T2)/(T1-T2)**2*c2t
     $ -T2*((7+3*Nc)*T1+(1-Nc)*T2)/2./(T1-T2)**2
     $ -((1-Nc)*T1-(3-Nc)*T2)/2./(T1-T2)+(1-Nc)*T2/4./T1)
     $ +Log(T2*T1/q**2)*(1+c2t**2)*cb**2*Yt**2/4./T1/T2/delt(A0,T1,T2)
     $ *(A0**2*T1+(T1-T2)**3-2*T2*(T1**2-T2**2)
     $ -A0*(2*T1**2+T1*T2+T2**2))
     $ +Log(T1/q)*(2*(mu2+t-T1)**2/delt(T1,mu2,t)
     $ -cb**2*(1+c2t**2)*Yt**2/2./T2/delt(A0,T1,T2)
     $ *(A0**2-4*T2*(T1-T2)-A0*(T1+3*T2))
     $ -2*cb**2*(A0+BL-T1)*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)
     $ /delt(A0,T1,BL)
     $ +cb**2*(2*Sqrt(t)+s2t*Yt)**2/2./T1
     $ -2*cb**2*(2*Sqrt(t)+s2t*Yt)**2/(A0-4*T1)
     $ +sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/T1
     $ -sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)/(BL-T1)
     $ -(1-c2t)/2.*mu2/T1+1-c2t
     $ +12*Sqrt(t)*At*((-3+7*c2t**2)*T1+c2t**2*T2)/s2t/(T1-T2)**2
     $ +6*Sqrt(t)*(2*T1-T2)*s2t*At/(T1-T2)**2+sb**2*s2t**2*Xt**2/T1
     $ +s2t*Sqrt(t)/T1*(sb**2*Xt-3*cb**2*Yt)
     $ +(1+c2t**2)*sb**2*Xt**2*(T1**2+4*T1*T2-T2**2)/4./T1/T2/(T1-T2)
     $ +(sb**2*Xt**2+cb**2*Yt**2)*((2+15*c2t+23*c2t**2)*T1/(T1-T2)**2
     $ +(5-6*c2t-5*c2t**2)/2./(T1-T2)-(5-2*c2t-c2t**2)/4./T1)
     $ +3*A0*cb**2*(2*(1+c2t)*T1-(1-c2t)*T2)/(T1-T2)**2
     $ +BL*(2*(1+3*c2t)*T1-(1-3*c2t)*T2)/(T1-T2)**2
     $ +t*(2*(1-6*c2t)*T1-(1+3*c2t)*T2)/(T1-T2)**2
     $ +(5-c2t-8*cb**2)*t/2./T1+(3-c2t)*mu2/2./T1
     $ -2*(3+12*c2t)*mu2*T1/(T1-T2)**2-6*(1-c2t)*mu2/(T1-T2)
     $ -9/2.+s2t**2/2.*(1+Nc)+3*c2t/2.+c2t*T1*(4*T1+11*T2)/(T1-T2)**2
     $ +(1+Nc)*(65*T1**2+4*T1*T2-9*T2**2)*c2t**2/4./(T1-T2)**2
     $ +(1+Nc)*(T1-T2)*c2t**2/4./T2+(1-Nc)*T1/4./T2
     $ +T1*((7+3*Nc)*T1+(1-Nc)*T2)/2./(T1-T2)**2
     $ +(5*(5+Nc)*T1+(1-Nc)*T2)/2./(T1-T2))
     $ +(2*Log(mu2/q)**2-Log(mu2/T1)**2)*3*c2t*mu2**2/(T1-T2)**2
     $ +(2*Log(BL/q)**2-Log(BL/T1)**2)*
     $ (2*BL*(1-3*c2t**2)*Sqrt(t)*At/s2t/(T1-T2)**2-BL/2./(T1-T2)
     $ -3*BL*c2t*(T1+T2-2*t+2*sb**2*Xt**2+2*cb**2*Yt**2)/2./(T1-T2)**2)
     $ +(2*Log(A0/q)**2-Log(A0/T1)**2)*cb**2*
     $ (4*A0*(1-3*c2t**2)*Sqrt(t)*Yt/s2t/(T1-T2)**2-3*A0/2./(T1-T2)
     $ -3*A0*c2t*(T1+T2-2*t+2*Yt**2)/2./(T1-T2)**2)
     $ +Log(t/T1)*Log(mu2/T1)*(T1+T2-2*t-2*mu2)/(T1-T2)
     $ +Log(T2/T1)*Log(A0/T1)*Yt**2*cb**2*
     $ (8*A0*c2t**2+(1-3*c2t**2)*(T1-T2))/2./(T1-T2)**2
     $ +Log(BL/T1)*Log(A0/T1)*cb**2*(2*Sqrt(t)*Yt/(T1-T2)**2/s2t
     $ *((A0+BL)*(1-3*c2t**2)-(1-2*c2t**2)*T1+c2t**2*T2)
     $ +s2t*Sqrt(t)*Yt/(T1-T2)+(t+Yt**2)/2./(T1-T2)
     $ +3*c2t*(2*A0+2*BL-T1-T2)*(t-Yt**2)/2./(T1-T2)**2)
     $ +Log(T1/q)*Log(T2/q)*(2*c2t**2*(1+Nc)*(T1+T2)**2/(T1-T2)**2
     $ +((1+5*c2t**2)*T1
     $ -(1-11*c2t**2)*T2)/2./(T1-T2)**2*(sb**2*Xt**2+cb**2*Yt**2))
     $ +Log(T1/q)**2*(9*c2t**2*Sqrt(t)*At/s2t/(T1-T2)
     $ +6*Sqrt(t)*T1*(2-5*c2t**2)*At/s2t/(T1-T2)**2
     $ -3*Sqrt(t)*s2t*(2*T1-T2)*At/(T1-T2)**2
     $ -((5+5*c2t+13*c2t**2)*T1-(3-4*c2t-10*c2t**2)*T2)
     $ /2./(T1-T2)**2*(sb**2*Xt**2+cb**2*Yt**2)
     $ +mu2*((6+5*c2t)*T1-(3-4*c2t)*T2)/(T1-T2)**2
     $ +9/4.-c2t+5/4.*s2t**2*(1+Nc)
     $ +c2t*(t-BL-A0*cb**2)*(5*T1+4*T2)/2./(T1-T2)**2
     $ -(t+BL+3*A0*cb**2)*(2*T1-T2)/2./(T1-T2)**2
     $ +c2t/4.-9*T1*T2*c2t/2./(T1-T2)**2
     $ -c2t**2*(1+Nc)*(8*T1**2+16*T1*T2-T2**2)/2./(T1-T2)**2
     $ -T1*((14+5*Nc)*T1-(10+4*Nc)*T2)/2./(T1-T2)**2)
     $ +Log(T2/q)**2*(-3*c2t**2*Sqrt(t)*(T1+T2)*At/s2t/(T1-T2)**2
     $ +3*s2t*Sqrt(t)*T2*At/(T1-T2)**2
     $ +(2*T2-c2t*(T1+2*T2)+c2t**2*(2*T1+5*T2))
     $ /2./(T1-T2)**2*(sb**2*Xt**2+cb**2*Yt**2)
     $ +T2*(3*cb**2*A0+BL-6*mu2+t)/2./(T1-T2)**2
     $ -c2t*(cb**2*A0+BL-mu2-t)*(T1+2*T2)/2./(T1-T2)**2
     $ +c2t*mu2*(T1+2*T2)/2./(T1-T2)**2-3/4.+s2t**2/4.*(1+Nc)
     $ +c2t**2*((T1+T2)**2+3*T2**2)/2./(T1-T2)**2*(1+Nc)
     $ -c2t*((T1+T2)**2+2*T1*T2)/4./(T1-T2)**2
     $ +T2*(2*(1+Nc)*T1+(2-Nc)*T2)/2./(T1-T2)**2)

      return
      end

*
*********************************************************************
*

      subroutine resfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,F2_s)

      implicit none
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 delt,bdszLi2,phi,ct2,st2,Xt,Yt,At
      real*8 F2_s

      mu2 = mu**2

      Xt = s2t*(T1-T2)/2d0/Sqrt(t)
      Yt = Xt - mu/cb/sb
      At = sb**2*Xt+cb**2*Yt

      ct2 = (1d0+c2t)/2d0
      st2 = (1d0-c2t)/2d0
      
      F2_s =
     $     2*c2t**2*sb**2*Sqrt(t)*Xt/(T1-T2)*bdszLi2(1-BL/T1)
     $     -2*c2t**2*sb**2*Sqrt(t)*Xt/(T1-T2)*bdszLi2(1-BL/T2)
     $     -c2t**2*cb**2*Sqrt(t)*Yt/(T1-T2)*Log(A0/T1)*Log(BL/T1)
     $     +c2t**2*cb**2*Sqrt(t)*Yt/(T1-T2)*Log(A0/T2)*Log(BL/T2)
     $     -6*At*c2t**2*Sqrt(t)/(T1-T2)*Log(T1/T2)
     $     +3*c2t**2*At*Sqrt(t)/(T1-T2)*Log(T1/q)**2
     $     -3*c2t**2*At*Sqrt(t)/(T1-T2)*Log(T2/q)**2
     $     -c2t**2*cb**2*Yt*Sqrt(t)*
     $     (A0+BL-T1)/T1/(T1-T2)*phi(A0,BL,T1)
     $     +c2t**2*cb**2*Yt*Sqrt(t)*
     $     (A0+BL-T2)/T2/(T1-T2)*phi(A0,BL,T2)
     $     -2*A0*c2t**2*cb**2*Yt*Sqrt(t)
     $     /T1/(T1-T2)*phi(A0,T1,T1)
     $     +2*A0*c2t**2*cb**2*Yt*Sqrt(t)
     $     /T2/(T1-T2)*phi(A0,T2,T2)

      return
      end

*
*********************************************************************
*

      subroutine sfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,At,ht,
     $     dmuF2,dmuF3,dAtF2,dAtF3,DM12,DM22)
      
      implicit none
      real*8 t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,At,ht
      real*8 Pi/3.141592654/,Nc/3d0/
      real*8 dmuF2,dmuF3,dAtF2,dAtF3,DM12,DM22


      dmuF2 = -Nc/4.*(Log(T1/q)**2-Log(T2/q)**2)

      dmuF3 = -Nc/4.*(Log(T1/q)+Log(T2/q)-2.)
     $     *(2.-(T1+T2)/(T1-T2)*Log(T1/T2))

      dAtF2 = -(3.+Nc)/2.*(Log(T1/q)**2-Log(T2/q)**2)

      dAtF3 = -(3.+Nc)/2.*(Log(T1/q)+Log(T2/q)-2.)
     $     *(2.-(T1+T2)/(T1-T2)*Log(T1/T2))

      DM12 = ht**2*Nc*s2t*mu*Sqrt(t)/4.
     $     *(Log(T1/q)**2-Log(T2/q)**2)
     $     +ht**2*Nc*s2t**2*mu*At/8.*(Log(T1/q)+Log(T2/q)-2.)
     $     *(2.-(T1+T2)/(T1-T2)*Log(T1/T2))
      
      DM22 = ht**2*Nc*t*(Log(T1/q)**2+Log(T2/q)**2-2.*Log(t/q)**2)
     $     +ht**2*Nc*s2t*At*Sqrt(t)*(Log(T1/q)**2-Log(T2/q)**2)
     $     +ht**2*Nc*s2t**2*At**2/4.*(Log(T1/q)+Log(T2/q)-2.)
     $     *(2.-(T1+T2)/(T1-T2)*Log(T1/T2))
      
      return
      end

*     
***********************************************************************
*


      subroutine dfuncs(t,A0,BL,T1,T2,s2t,c2t,cb,sb,q,mu,A,v2,
     $     DF1,DF2,DF3,DdmuF2,DdmuF3,DdAtF2,DdAtF3)
      
c     shift of the parameters from DRbar to On-Shell scheme
 
      implicit none      
      real*8 t,mu2,A0,BL,T1,T2,s2t,c2t,cb,sb,q,A,mu,Xt,Yt
      real*8 myB0,myAA
      real*8 DF1,DF2,DF3,DdmuF2,DdmuF3,DdAtF2,DdAtF3
      real*8 Pi/3.141592654/,mt,Nc/3d0/,ct2,st2,v2,v22
      real*8 F1o,F2o,F3o,dm1,dm2,dmt,dAt,dth,ds2t,dv2,dv22,dmu,dcotb
      real*8 pi12_1,pi12_2
      
      mu2 = mu**2

      Xt = A + mu*cb/sb
      Yt = A - mu*sb/cb

      ct2 = (1d0+c2t)/2d0
      st2 = (1d0-c2t)/2d0

      v22 = v2*sb**2
      
      mt = Sqrt(t)
      
      F1o = Log(T1/q) + Log(T2/q) - 2d0*Log(t/q)
      F2o = Log(T1/q) - Log(T2/q) 
      F3o = 2d0 - (T1+T2)/(T1-T2)*(Log(T1/q) - Log(T2/q))

c     counterterms:

      dv2 = v22 *Nc/2d0* (2d0 *Log(t/q) - 1d0 - BL/t + 
     $     T1/t * ct2 * (2d0* BL/(BL-T1) * Log(BL/T1) - 1d0)+
     $     T2/t * st2 * (2d0* BL/(BL-T2) * Log(BL/T2) - 1d0))

      dcotb = 0d0               ! beta DRbar
      dv22 = v22/v2*dv2

      dmt = Sqrt(t)/2d0*((1d0-5d0/2d0*sb**2)*Log(t/q)
     $     +3d0*cb**2/2d0*A0/t*(1d0-Log(A0/q))
     $     -3d0/2d0*mu2/t*(1d0-Log(mu2/q))
     $     +5d0*sb**2-1d0+cb**2/2d0*(1d0-A0/t)*myB0(t,0d0,A0,q)
     $     +cb**2*(2d0-A0/t)*myB0(t,t,A0,q)
     $     +(T1/t*(1d0-Log(T1/q))+(t-T1+mu2)/t*myB0(t,mu2,T1,q)
     $     +T2/t*(1d0-Log(T2/q))+(t-T2+mu2)/t*myB0(t,mu2,T2,q)
     $     +BL/t*(1d0-Log(BL/q))+(t-BL+mu2)/t*myB0(t,mu2,BL,q))/2d0)

      dmu = 0d0                 ! mu DRbar
      
      dm1 = ((T1-t-mu2)*myB0(T1,t,mu2,q) - myAA(t,q)
     $     + st2*(T1-mu2)*myB0(T1,0d0,mu2,q) - (1d0+st2)*myAA(mu2,q)
     $     + cb**2*(1d0+st2)* myAA(A0,q) + st2* myAA(BL,q) +
     $     (c2t**2-(Nc-1d0)/2d0*s2t**2)*myAA(T2,q)
     $     + (Nc+1d0)/2d0* s2t**2* myAA(T1,q) + 1d0/2d0*  
     $     (sb**2* (2d0 *Sqrt(t)+ s2t* Xt)**2* myB0(T1,T1,0d0,q) +
     $     cb**2*(2d0* Sqrt(t)+ s2t* Yt)**2* myB0(T1,T1,A0,q) + 
     $     sb**2*(1d0+c2t**2)*Xt**2*myB0(T1,T2,0d0,q) + 
     $     cb**2*(1d0+c2t**2)*Yt**2*myB0(T1,T2,A0,q)) +
     $     sb**2*(ct2*t+s2t*Sqrt(t)*Xt+st2*Xt**2)*myB0(T1,BL,0d0,q) +
     $     cb**2*(ct2*t+s2t*Sqrt(t)*Yt+st2*Yt**2)*myB0(T1,BL,A0,q))
      
      dm2 = ((T2-t-mu2)*myB0(T2,t,mu2,q) - myAA(t,q)
     $     + ct2*(T2-mu2)*myB0(T2,0d0,mu2,q) - (1d0+ct2)*myAA(mu2,q)
     $     + cb**2* (1d0+ct2)*myAA(A0,q) + ct2* myAA(BL,q) +
     $     (c2t**2 - (Nc-1d0)/2d0* s2t**2)*myAA(T1,q)
     $     +(Nc+1d0)/2d0* s2t**2* myAA(T2,q) + 1d0/2d0* 
     $     (sb**2* (2d0* Sqrt(t)- s2t* Xt)**2* myB0(T2,T2,0d0,q) +
     $     cb**2*(2d0*Sqrt(t)- s2t*Yt)**2*myB0(T2,T2,A0,q) + 
     $     sb**2*(1d0+c2t**2)* Xt**2* myB0(T2,T1,0d0,q) + 
     $     cb**2*(1d0+c2t**2)* Yt**2* myB0(T2,T1,A0,q)) +
     $     sb**2*(st2*t-s2t*Sqrt(t)*Xt+ct2*Xt**2)*myB0(T2,BL,0d0,q) +
     $     cb**2*(st2*t-s2t*Sqrt(t)*Yt+ct2*Yt**2)*myB0(T2,BL,A0,q))

      pi12_1 =1d0/2d0*(s2t*((T1-mu2)*myB0(T1,0d0,mu2,q)-myAA(mu2,q))
     $     + s2t* cb**2* myAA(A0,q) + s2t* myAA(BL,q) +
     $     (Nc+1d0)* c2t* s2t* (myAA(T1,q) - myAA(T2,q)) +
     $     sb**2*c2t*Xt*(2d0* Sqrt(t) + s2t* Xt)* myB0(T1,T1,0d0,q) +
     $     cb**2*c2t*Yt*(2d0* Sqrt(t) + s2t* Yt)* myB0(T1,T1,A0,q) +
     $     sb**2*c2t*Xt*(2d0* Sqrt(t) - s2t* Xt)* myB0(T1,T2,0d0,q) +
     $     cb**2*c2t*Yt*(2d0* Sqrt(t) - s2t* Yt)* myB0(T1,T2,A0,q) -
     $     sb**2*(s2t*t-2d0*c2t*Sqrt(t)*Xt-s2t*Xt**2)
     $     *myB0(T1,BL,0d0,q) -
     $     cb**2*(s2t*t-2d0*c2t*Sqrt(t)*Yt-s2t*Yt**2)
     $     *myB0(T1,BL,A0,q))

      pi12_2 =1d0/2d0*(s2t*((T2-mu2)*myB0(T2,0d0,mu2,q)-myAA(mu2,q))
     $     + s2t* cb**2* myAA(A0,q) + s2t* myAA(BL,q) +
     $     (Nc+1d0)* c2t* s2t* (myAA(T1,q) - myAA(T2,q)) +
     $     sb**2*c2t*Xt*(2d0* Sqrt(t) + s2t* Xt)* myB0(T2,T1,0d0,q) +
     $     cb**2*c2t*Yt*(2d0* Sqrt(t) + s2t* Yt)* myB0(T2,T1,A0,q) +
     $     sb**2*c2t*Xt*(2d0* Sqrt(t) - s2t* Xt)* myB0(T2,T2,0d0,q) +
     $     cb**2*c2t*Yt*(2d0* Sqrt(t) - s2t* Yt)* myB0(T2,T2,A0,q) -
     $     sb**2*(s2t*t-2d0*c2t*Sqrt(t)*Xt-s2t*Xt**2)
     $     *myB0(T2,BL,0d0,q) -
     $     cb**2*(s2t*t-2d0*c2t*Sqrt(t)*Yt-s2t*Yt**2)
     $     *myB0(T2,BL,A0,q))

      dth = (pi12_1 + pi12_2)/2d0/(T1-T2)

      ds2t = 2d0*c2t*dth

      dAt = ((dm1-dm2)/(T1-T2) + ds2t/s2t - dmt/mt)*Xt
     $     - mu * dcotb - dmu * cb/sb

      DF1 = dm1/T1 + dm2/T2 - 4d0*dmt/mt + (4d0*dmt/mt - dv22/v22)*F1o
      DF2 = dm1/T1 - dm2/T2 + (3d0*dmt/mt - dv22/v22 + ds2t/s2t)*F2o
      DF3 = (2d0*T1*T2/(T1-T2)**2*Log(T1/T2) - (T1+T2)/(T1-T2)) 
     $     *(dm1/T1-dm2/T2) + (2d0*dmt/mt-dv22/v22+2d0*ds2t/s2t)*F3o

      DdmuF2 = dmu/mu * F2o       
      DdmuF3 = dmu/mu * F3o       

      DdAtF2 = dAt/A * F2o       
      DdAtF3 = dAt/A * F3o       

c     residues of some singular functions for s2t=0 and for A=0

      if(s2t.eq.0d0) then         
         DF2 = ds2t*F2o
         DdAtF2 = ds2t*Xt/A
      endif

      if(mu.eq.0d0) then
         DdmuF2 = dmu * F2o       
         DdmuF3 = dmu * F3o       
      endif

      if(A.eq.0d0) then
         DdAtF2 = dAt * F2o       
         DdAtF3 = dAt * F3o       
      endif
      
      return
      end

  

c----------------------------------------------------------------
c
c --> mhalphatsqfuncs.f
c
c----------------------------------------------------------------
      
*      
*     SOME AUXILIARY FUNCTIONS CALLED BY THE DSZ & BDSZ ROUTINES
*
*     Last update 13/12/2001
*
***********************************************************************
*

      real*8 function myAA(m,q)      
      real*8 m,q

      if(m.ne.0d0) then
         myAA = m*(1d0-Log(m/q))
      else
         myAA = 0d0
      endif

      return
      end

*
***********************************************************************
*


      real*8 function myB0(q,m1,m2,mu2) 

c     from Degrassi and Sirlin, Phys. Rev. D46 (1992) 3104.
      
      real*8 q,m1,m2,Omega,mu2
      
      if(m1.eq.0d0.and.m2.ne.0d0) then
         
         if(m2.ne.q) then
            myB0 = -(Log(m2/mu2)-2-(m2/q-1d0)*Log(dabs(1d0 - q/m2))) 
         else 
            myB0 = -(Log(m2/mu2) - 2)
         endif

      elseif(m2.eq.0d0.and.m1.ne.0d0) then

         if(m1.ne.q) then
            myB0 = -(Log(m1/mu2)-2-(m1/q-1d0)*Log(dabs(1d0 - q/m1))) 
         else
            myB0 = -(Log(m1/mu2) - 2)
         endif
         
      elseif(m2.eq.0d0.and.m1.eq.0d0) then

         myB0 = -(Log(q/mu2) - 2)  ! cut the imaginary part (I Pi)
         
      else
         
         myB0 = -( dlog(q/mu2)-2.d0 + 
     1        1.d0/2.d0*( 1.d0 + (m1/q-m2/q))*dlog(m1/q) +
     2        1.d0/2.d0*( 1.d0 - (m1/q-m2/q))*dlog(m2/q) +
     3        2.d0*Omega(m1/q,m2/q))
         
      endif

      return
      end
      
c     function Omega(a,b) contained in myB0
      real*8 function Omega(a,b)
      real*8 a,b,cbig
      Cbig = (a+b)/2.d0 - (a-b)**2.d0/4.d0 -1.d0/4.d0
      if(Cbig.gt.0.d0) then
         Omega = dsqrt(Cbig)*
     1        (datan((1.d0 + a - b)/(2.d0*dsqrt(Cbig))) +
     2        datan((1.d0 - a + b)/(2.d0*dsqrt(Cbig))) )
      elseif(Cbig.lt.0d0) then
         Cbig = - Cbig
         Omega = 1.d0/2.d0*dsqrt(Cbig)*
     1        dlog((a/2.d0 +b/2.d0 -1.d0/2.d0 -dsqrt(Cbig))/
     2        (a/2.d0 + b/2.d0 -1.d0/2.d0 + dsqrt(Cbig)))
      else
         Omega = 0         
      endif

      return
      end
      
*
**********************************************************************
*

      function phi(x,y,z)

c     from Davydychev and Tausk, Nucl. Phys. B397 (1993) 23

      implicit none
      real*8 x,y,z,phi,pphi,myphi
      
      if(x.le.z.and.y.le.z) then
         pphi = myphi(x,y,z)
      elseif(z.le.x.and.y.le.x) then
         pphi = z/x*myphi(z,y,x)
      elseif(z.le.y.and.x.le.y) then
         pphi = z/y*myphi(z,x,y)
      endif

      phi = pphi
      
      end
      
      function myphi(x,y,z)
      
      implicit none

      real*8 x,y,z,myphi
      real*8 u,v
      real*8 Pi/3.14159265358979/
      complex*16 clam,cxp,cxm,bdszCLi2,ccphi

c     auxiliary variables

      u = x/z
      v = y/z
      
      if((1d0-u-v)**2.ge.4d0*u*v) then         
         clam = dcmplx(dsqrt((1d0-u-v)**2 - 4d0*u*v),0d0)
      else
         clam = dcmplx(0d0,dsqrt(4d0*u*v - (1d0-u-v)**2))
      endif

      cxp = (1d0+(u-v)-clam)/2d0
      cxm = (1d0-(u-v)-clam)/2d0
      
c     phi function from eq. (A4)

      ccphi = (2d0*log(cxp)*log(cxm) - log(u)*log(v) - 
     &     2d0*(bdszCLi2(cxp) + bdszCLi2(cxm)) + Pi**2/3d0)/clam
      myphi = dreal(ccphi)

      return
      end


*
***********************************************************************
*

      function delt(x,y,z)

      implicit none

      real*8 x,y,z,delt

      delt = x**2 + y**2 + z**2 - 2d0*(x*y + x*z + y*z)

      return
      end

*
***********************************************************************
*

      function bdszLi2(x)

      implicit none

      complex*16 bdszCLi2,z
      real*8 x,bdszLi2

      z = DCMPLX(x,0d0)
      bdszLi2 = dreal(bdszCLi2(z))

      return
      end

*
***********************************************************************
*

      COMPLEX*16 FUNCTION bdszCLi2(Z)

c     routine for the complex dilogarithm function
      
      COMPLEX*16 Z,Z1,CLI20,MOD2
      REAL*8 MOD,PI3,PI6
      DATA PI6/1.64493406684823/,PI3/3.28986813369645/
      IF(Z.EQ.(1d0,0d0)) THEN
        bdszCLi2=DCMPLX(PI6,0d0)
        RETURN
      ENDIF
      LAB=0
      Z1=Z
      CLI20=(0d0,0d0)
      MOD=SQRT(DREAL(Z)**2+DIMAG(Z)**2)
      IF(MOD.GT.1d0)  Z1=1d0/Z   
      IF(DREAL(Z1).GE.0.5d0)  THEN
          LAB=1
          Z1=1d0-Z1
      ENDIF
      DO 20 ITE=1,100
      MOD2=Z1**ITE/(ITE**2)
      CLI20=MOD2+CLI20
      IF(SQRT(DREAL(MOD2)**2+DIMAG(MOD2)**2).LE.10E-10) GO TO 30
20    CONTINUE
30    IF(LAB.EQ.1)  CLI20=-CLI20+PI6-LOG(Z1)*LOG(1d0-Z1)
      IF(MOD.LE.1d0) THEN
        bdszCLi2=CLI20
        RETURN
      ELSE
        bdszCLi2=-PI6-CLI20-CDLOG(-Z)**2/2d0
      ENDIF
      RETURN
      END


c----------------------------------------------------------------
c
c --> ShiftMSbarOSMSt1sqcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSMSt1sq1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------

      ShiftMSbarOSMSt1sq1=
     &  -((GS**2*PI**(-2)*(3D0*CF*(2D0*MGL**2+7D0*MST1**2+2D0*MT**
     &  2)-4D0*(-(4D0*MST2**2*STT**2*(-1D0+STT**2))+MST1**2*(1D0-
     &  2D0*STT**2D0)**2)))/48D0)-(CF*GS**2*PI**(-2)*(MGL**2-MST1**
     &  2+MT**2-4D0*MGL*MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MT)**2*MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))))/8D0+(CF*GS**2*
     &  MGL**2*PI**(-2)*DLOG(MGL**2*MUE**(-2)))/8D0+(GS**2*MST1**2*
     &  PI**(-2)*(9D0*CF-4D0*(1D0-2D0*STT**2D0)**2)*DLOG(MST1**2*
     &  MUE**(-2)))/48D0+(GS**2*MST2**2*PI**(-2)*STT**2*(-1D0+STT**
     &  2)*DLOG(MST2**2*MUE**(-2)))/3D0+(CF*GS**2*MT**2*PI**(-2)*
     &  DLOG(MT**2*MUE**(-2)))/8D0+(CF*GS**2*PI**(-2)*(MGL**2-MST1**
     &  2+MT**2-4D0*MGL*MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MT)**2*MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))-DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*EPS+MT)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/
     &  (2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-
     &  (CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+(-I*EPS+
     &  MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+(-I*
     &  EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**
     &  2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0))))))/8D0
      end


c----------------------------------------------------------------
c
c --> ShiftMSbarOSMSt2sqcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSMSt2sq1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------

      ShiftMSbarOSMSt2sq1=
     &  -((GS**2*PI**(-2)*(3D0*CF*(2D0*MGL**2+7D0*MST2**2+2D0*MT**
     &  2)-4D0*(-(4D0*MST1**2*STT**2*(-1D0+STT**2))+MST2**2*(1D0-
     &  2D0*STT**2D0)**2)))/48D0)-(CF*GS**2*PI**(-2)*(MGL**2-MST2**
     &  2+MT**2+4D0*MGL*MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MT)**2*MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST2**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST2**2+EPS*DCMPLX(0D0,1D0)))))/8D0+(CF*GS**2*
     &  MGL**2*PI**(-2)*DLOG(MGL**2*MUE**(-2)))/8D0+(GS**2*MST1**2*
     &  PI**(-2)*STT**2*(-1D0+STT**2)*DLOG(MST1**2*MUE**(-2)))/3D0+
     &  (GS**2*MST2**2*PI**(-2)*(9D0*CF-4D0*(1D0-2D0*STT**2D0)**2)*
     &  DLOG(MST2**2*MUE**(-2)))/48D0+(CF*GS**2*MT**2*PI**(-2)*
     &  DLOG(MT**2*MUE**(-2)))/8D0+(CF*GS**2*PI**(-2)*(MGL**2-MST2**
     &  2+MT**2+4D0*MGL*MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MT)**2*MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST2**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST2**2+EPS*DCMPLX(0D0,1D0)))-DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*EPS+MT)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/
     &  (2D0*(MST2**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-
     &  (CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**2+(-I*EPS+
     &  MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST2**2+(-I*
     &  EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**
     &  2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST2**2+EPS*DCMPLX(0D0,1D0))))))/8D0
      end



c----------------------------------------------------------------
c
c --> ShiftMSbarOSsttcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSstt1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------

      ShiftMSbarOSstt1=
     &  DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0)*((GS**2*PI**(-2)*
     &  STT*(-1D0+2D0*STT**2)*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/
     &  2D0))/6D0+(CF*GS**2*MGL*MT*PI**(-2)*(1D0-2D0*STT**2)*(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-CDLOG((-I*EPS+MT)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/
     &  (2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-
     &  (CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+(-I*EPS+
     &  MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+(-I*
     &  EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**
     &  2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))))/(4D0*(MST1**2-
     &  MST2**2))-(GS**2*MST1**2*PI**(-2)*STT*(-1D0+2D0*STT**2)*
     &  DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0)*DLOG(MST1**2*MUE**(-
     &  2)))/(6D0*(MST1**2-MST2**2))-(GS**2*MST2**2*PI**(-2)*STT*(-
     &  1D0+2D0*STT**2)*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0)*
     &  DLOG(MST2**2*MUE**(-2)))/(6D0*(-(MST1**2)+MST2**2))+(CF*GS**
     &  2*MGL*MT*PI**(-2)*(-1D0+2D0*STT**2)*(2D0+((-(CDLOG((-I*EPS+
     &  MGL)**2*MUE**(-2)))-CDLOG((-I*EPS+MT)**2*MUE**(-2))))/2D0-
     &  ((-I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2))*
     &  CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/(2D0*(MST1**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-(CDLOG(((-I*EPS+MT)**(-
     &  2)*((-I*EPS+MGL)**2-MST1**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+
     &  MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**
     &  2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*
     &  (MST1**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*
     &  ((-I*EPS+MGL)**2-MST1**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*
     &  (-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**
     &  2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**
     &  (-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-
     &  I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**
     &  2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0)))-DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2*MUE**(-2)))-
     &  CDLOG((-I*EPS+MT)**2*MUE**(-2))))/2D0-((-I*EPS+MT)**2*(-1D0+
     &  (-I*EPS+MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-
     &  I*EPS+MT)**(-2)))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*
     &  EPS+MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-
     &  MST1**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-
     &  I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/
     &  2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+
     &  MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+
     &  MGL)**2-MST1**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**
     &  2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+
     &  MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))))/(4D0*(MST1**2-MST2**2)))
      end



c----------------------------------------------------------------
c
c --> ShiftMSbarOSMSb1sqcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSMSb1sq1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c varcom3.h
c
      double precision msb1, msb2, stb, mb

      common /msbartoos3/ msb1, msb2, stb, mb
c -------------------------------------------------------------------

      ShiftMSbarOSMSb1sq1=
     &  -((GS**2*PI**(-2)*(3D0*CF*(2D0*MB**2+2D0*MGL**2+7D0*MSB1**
     &  2)-4D0*(-(4D0*MSB2**2*STB**2*(-1D0+STB**2))+MSB1**2*(1D0-
     &  2D0*STB**2D0)**2)))/48D0)-(CF*GS**2*PI**(-2)*(MB**2+MGL**2-
     &  MSB1**2-4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+
     &  MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+
     &  MGL)**(-2)))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))))/8D0+(CF*
     &  GS**2*MB**2*PI**(-2)*DLOG(MB**2*MUE**(-2)))/8D0+(CF*GS**2*
     &  MGL**2*PI**(-2)*DLOG(MGL**2*MUE**(-2)))/8D0+(GS**2*MSB1**2*
     &  PI**(-2)*(9D0*CF-4D0*(1D0-2D0*STB**2D0)**2)*DLOG(MSB1**2*
     &  MUE**(-2)))/48D0+(GS**2*MSB2**2*PI**(-2)*STB**2*(-1D0+STB**
     &  2)*DLOG(MSB2**2*MUE**(-2)))/3D0+(CF*GS**2*PI**(-2)*(MB**2+
     &  MGL**2-MSB1**2-4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**
     &  (1D0/2D0))*(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-
     &  CDLOG((-I*EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-
     &  1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*
     &  (-I*EPS+MGL)**(-2)))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-
     &  I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+
     &  (-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-
     &  (-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*
     &  EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+
     &  EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*
     &  EPS+MB)**2+(-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*
     &  EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*
     &  EPS+MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+
     &  MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+
     &  MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0)))-DREAL(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-
     &  CDLOG((-I*EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-
     &  1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*
     &  (-I*EPS+MGL)**(-2)))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-
     &  I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+
     &  (-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-
     &  (-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*
     &  EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+
     &  EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*
     &  EPS+MB)**2+(-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*
     &  EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*
     &  EPS+MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+
     &  MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+
     &  MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))))/8D0
      end



c----------------------------------------------------------------
c
c --> ShiftMSbarOSMSb2sqcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSMSb2sq1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c varcom3.h
c
      double precision msb1, msb2, stb, mb

      common /msbartoos3/ msb1, msb2, stb, mb
c -------------------------------------------------------------------
      ShiftMSbarOSMSb2sq1=
     &  -((GS**2*PI**(-2)*(3D0*CF*(2D0*MB**2+2D0*MGL**2+7D0*MSB2**
     &  2)+4D0*(4D0*MSB1**2*STB**2*(-1D0+STB**2)-MSB2**2*(1D0-2D0*
     &  STB**2D0)**2)))/48D0)-(CF*GS**2*PI**(-2)*(MB**2+MGL**2-
     &  MSB2**2+4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/
     &  2D0))*(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-CDLOG((-I*
     &  EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+
     &  MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+
     &  MGL)**(-2)))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,1D0)))))/8D0+(CF*
     &  GS**2*MB**2*PI**(-2)*DLOG(MB**2*MUE**(-2)))/8D0+(CF*GS**2*
     &  MGL**2*PI**(-2)*DLOG(MGL**2*MUE**(-2)))/8D0+(GS**2*MSB1**2*
     &  PI**(-2)*STB**2*(-1D0+STB**2)*DLOG(MSB1**2*MUE**(-2)))/3D0+
     &  (GS**2*MSB2**2*PI**(-2)*(9D0*CF-4D0*(1D0-2D0*STB**2D0)**2)*
     &  DLOG(MSB2**2*MUE**(-2)))/48D0+(CF*GS**2*PI**(-2)*(MB**2+
     &  MGL**2-MSB2**2+4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**
     &  (1D0/2D0))*(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-
     &  CDLOG((-I*EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-
     &  1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*
     &  (-I*EPS+MGL)**(-2)))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,1D0)))+((-
     &  I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+
     &  (-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-
     &  (-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*
     &  EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+
     &  EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*
     &  EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*
     &  EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*
     &  EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+
     &  MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+
     &  MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,
     &  1D0)))-DREAL(2D0+((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-
     &  CDLOG((-I*EPS+MGL)**2*MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-
     &  1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*
     &  (-I*EPS+MGL)**(-2)))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,1D0)))+((-
     &  I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+
     &  (-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-
     &  (-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*
     &  EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+
     &  EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*
     &  EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-(4D0*(-I*
     &  EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*
     &  EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+
     &  MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+
     &  MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB2**2+EPS*DCMPLX(0D0,
     &  1D0))))))/8D0
      end




c----------------------------------------------------------------
c
c --> ShiftMSbarOSstbcode.f
c
c----------------------------------------------------------------
      
      complex*16 function ShiftMSbarOSstb1()
c -------------------------------------------------------------------
c varcom2.h
c
      double precision mst1, mst2, stt, mgl, mt, cf, gs, eps, pi, mue
      complex*16 i

      common /msbartoos/ mst1, mst2, stt, mgl, mt, cf, gs, eps, pi,
     $	                 mue, i
c -------------------------------------------------------------------
c -------------------------------------------------------------------
c varcom3.h
c
      double precision msb1, msb2, stb, mb

      common /msbartoos3/ msb1, msb2, stb, mb
c -------------------------------------------------------------------
      ShiftMSbarOSstb1=
     &  DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0)*((GS**2*PI**(-2)*
     &  STB*(-1D0+2D0*STB**2)*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/
     &  2D0))/6D0+(CF*GS**2*MB*MGL*PI**(-2)*(1D0-2D0*STB**2)*(2D0+
     &  ((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-CDLOG((-I*EPS+MGL)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*
     &  EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/
     &  (2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-
     &  (CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-
     &  MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))))/(4D0*
     &  (MSB1**2-MSB2**2))-(GS**2*MSB1**2*PI**(-2)*STB*(-1D0+2D0*
     &  STB**2)*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0)*DLOG(MSB1**
     &  2*MUE**(-2)))/(6D0*(MSB1**2-MSB2**2))-(GS**2*MSB2**2*PI**(-
     &  2)*STB*(-1D0+2D0*STB**2)*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/
     &  2D0)*DLOG(MSB2**2*MUE**(-2)))/(6D0*(-(MSB1**2)+MSB2**2))+
     &  (CF*GS**2*MB*MGL*PI**(-2)*(-1D0+2D0*STB**2)*(2D0+((-
     &  (CDLOG((-I*EPS+MB)**2*MUE**(-2)))-CDLOG((-I*EPS+MGL)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*
     &  EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/
     &  (2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-
     &  (CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-
     &  MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))-DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MB)**2*MUE**(-2)))-CDLOG((-I*EPS+MGL)**2*
     &  MUE**(-2))))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*
     &  EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/
     &  (2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-
     &  (CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-
     &  MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0))))))/(4D0*
     &  (MSB1**2-MSB2**2)))
      end




c----------------------------------------------------------------
c
c --> MSb1shiftDREDcode.f
c
c----------------------------------------------------------------
      
      complex*16 function MSb1shiftDRED1()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDRED1=
     &  EINS*((GS**2*PI**(-2)*(-(MST1**2)+MST2**2+2D0*MB**2*STB**2+
     &  2D0*MST1**2*STB**2-2D0*MT**2*STB**2-4D0*MB**2*STB**4-4D0*
     &  MST1**2*STB**4+4D0*MT**2*STB**4+MSB1**2*(1D0-4D0*STB**2+6D0*
     &  STB**4-4D0*STB**6)+MSB2**2*(-1D0+2D0*STB**2-2D0*STB**4+4D0*
     &  STB**6)+2D0*MST1**2*STT**2-2D0*MST2**2*STT**2-2D0*MST1**2*
     &  STB**2*STT**2+2D0*MST2**2*STB**2*STT**2+4D0*MST1**2*STB**4*
     &  STT**2-4D0*MST2**2*STB**4*STT**2-2D0*MZ**2*STB**2*DCOS(2D0*
     &  B)+2D0*(-(MW**2)+MZ**2)*STB**2*DCOS(2D0*B)+4D0*MZ**2*
     &  STB**4*DCOS(2D0*B)-4D0*(-(MW**2)+MZ**2)*STB**4*DCOS(2D0*
     &  B)))/(12D0*(-1D0+STB**2))+(GS**2*MB**2*PI**(-2)*
     &  DREAL(2D0-CDLOG((-I*EPS+MB)**2)+(CDLOG((-I*EPS+MB)**(-2)*(-
     &  (MB**2)+(-I*EPS+MB)**2+EPS*DCMPLX(0D0,-1D0)))*(-(MB**2)+(-I*
     &  EPS+MB)**2+EPS*DCMPLX(0D0,-1D0)))/(MB**2+EPS*DCMPLX(0D0,
     &  1D0))))/(3D0*(-1D0+STB**2))+(GS**2*MSB1**2*PI**(-2)*
     &  DREAL(2D0-CDLOG((-I*EPS+MSB1)**2)+(CDLOG((-I*EPS+MSB1)**(-
     &  2)*(-(MSB1**2)+(-I*EPS+MSB1)**2+EPS*DCMPLX(0D0,-1D0)))*(-
     &  (MSB1**2)+(-I*EPS+MSB1)**2+EPS*DCMPLX(0D0,-1D0)))/(MSB1**2+
     &  EPS*DCMPLX(0D0,1D0))))/3D0-(GS**2*MSB2**2*PI**(-2)*STB**2*
     &  DREAL(2D0-CDLOG((-I*EPS+MSB2)**2)+(CDLOG((-I*EPS+MSB2)**(-
     &  2)*(-(MSB2**2)+(-I*EPS+MSB2)**2+EPS*DCMPLX(0D0,-1D0)))*(-
     &  (MSB2**2)+(-I*EPS+MSB2)**2+EPS*DCMPLX(0D0,-1D0)))/(MSB2**2+
     &  EPS*DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))-(GS**2*PI**(-2)*
     &  (MB**2+MGL**2-MSB1**2-4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**
     &  2D0))**(1D0/2D0))*DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2))-
     &  CDLOG((-I*EPS+MSB1)**2)))/2D0-((-I*EPS+MSB1)**2*(-1D0+(-I*
     &  EPS+MGL)**2*(-I*EPS+MSB1)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*
     &  EPS+MSB1)**(-2)))/(2D0*(MB**2+EPS*DCMPLX(0D0,1D0)))+((-I*
     &  EPS+MSB1)**2*(-(CDLOG(((-I*EPS+MSB1)**(-2)*(-(MB**2)+(-I*
     &  EPS+MGL)**2+(-I*EPS+MSB1)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB1)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB1)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MSB1)**(-
     &  2)-(-I*EPS+MSB1)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MSB1)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-
     &  I*EPS+MSB1)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB1)**(-2)*
     &  (MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MSB1)**(-2)*
     &  (-(MB**2)+(-I*EPS+MGL)**2+(-I*EPS+MSB1)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MSB1)**2D0)+(I*EPS+MB**2D0-(-
     &  I*EPS+MGL)**2D0-(-I*EPS+MSB1)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+
     &  MSB1)**(-2)-(-I*EPS+MSB1)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB1)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB1)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB1)**(-
     &  2)*(MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MB**2+EPS*
     &  DCMPLX(0D0,1D0)))))/(12D0*(-1D0+STB**2))+(GS**2*PI**(-2)*
     &  (MB**2+MGL**2-MSB1**2-(4D0*MB*MGL*MSB2**2*STB*(-1D0+2D0*
     &  STB**2))/((-(MSB1**2)+MSB2**2)*DCMPLX(DCMPLX(1D0-STB**
     &  2D0))**(1D0/2D0))-4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**
     &  2D0))**(1D0/2D0)-(4D0*MB*MGL*STB*(1D0-2D0*STB**2)*(-(MB**2)+
     &  MT**2+MSB2**2*STB**2-MST2**2*STT**2+MST1**2*(-1D0+STT**2)+
     &  MZ**2*DCOS(2D0*B)-(-(MW**2)+MZ**2)*DCOS(2D0*B)))/
     &  ((MSB1**2-MSB2**2)*DCMPLX(DCMPLX(1D0-STB**2D0))**(3D0/
     &  2D0)))*DREAL(2D0+((-(CDLOG((-I*EPS+MB)**2))-CDLOG((-I*EPS+
     &  MGL)**2)))/2D0-((-I*EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*
     &  EPS+MGL)**(-2))*CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/
     &  (2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-
     &  (CDLOG(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-
     &  MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**
     &  2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MGL)**(-2)*((-I*EPS+MB)**2+(-I*
     &  EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-
     &  I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+
     &  MSB1**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)-(-I*EPS+MGL)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*
     &  EPS-(-I*EPS+MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MGL)**(-2)*(MSB1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MSB1**2+EPS*DCMPLX(0D0,1D0)))))/6D0)
      end


      complex*16 function MSb1shiftDRED2()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDRED2=
     &  EINS*(-((GS**2*MST1**2*PI**(-2)*(-1D0+STT**2)*DREAL(2D0-
     &  CDLOG((-I*EPS+MST1)**2)+(CDLOG((-I*EPS+MST1)**(-2)*(-(MST1**
     &  2)+(-I*EPS+MST1)**2+EPS*DCMPLX(0D0,-1D0)))*(-(MST1**2)+(-I*
     &  EPS+MST1)**2+EPS*DCMPLX(0D0,-1D0)))/(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/(3D0*(-1D0+STB**2)))+(GS**2*MST2**2*PI**(-2)*STT**2*
     &  DREAL(2D0-CDLOG((-I*EPS+MST2)**2)+(CDLOG((-I*EPS+MST2)**(-
     &  2)*(-(MST2**2)+(-I*EPS+MST2)**2+EPS*DCMPLX(0D0,-1D0)))*(-
     &  (MST2**2)+(-I*EPS+MST2)**2+EPS*DCMPLX(0D0,-1D0)))/(MST2**2+
     &  EPS*DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))-(GS**2*MT**2*PI**
     &  (-2)*DREAL(2D0-CDLOG((-I*EPS+MT)**2)+(CDLOG((-I*EPS+MT)**(-
     &  2)*(-(MT**2)+(-I*EPS+MT)**2+EPS*DCMPLX(0D0,-1D0)))*(-(MT**
     &  2)+(-I*EPS+MT)**2+EPS*DCMPLX(0D0,-1D0)))/(MT**2+EPS*
     &  DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))-(GS**2*PI**(-2)*(MB**
     &  2+MGL**2-MSB2**2+4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**
     &  2D0))**(1D0/2D0))*DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2))-
     &  CDLOG((-I*EPS+MSB2)**2)))/2D0-((-I*EPS+MSB2)**2*(-1D0+(-I*
     &  EPS+MGL)**2*(-I*EPS+MSB2)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*
     &  EPS+MSB2)**(-2)))/(2D0*(MB**2+EPS*DCMPLX(0D0,1D0)))+((-I*
     &  EPS+MSB2)**2*(-(CDLOG(((-I*EPS+MSB2)**(-2)*(-(MB**2)+(-I*
     &  EPS+MGL)**2+(-I*EPS+MSB2)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MSB2)**(-
     &  2)-(-I*EPS+MSB2)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-
     &  I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB2)**(-2)*
     &  (MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MSB2)**(-2)*
     &  (-(MB**2)+(-I*EPS+MGL)**2+(-I*EPS+MSB2)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-
     &  I*EPS+MGL)**2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+
     &  MSB2)**(-2)-(-I*EPS+MSB2)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB2)**(-
     &  2)*(MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MB**2+EPS*
     &  DCMPLX(0D0,1D0)))))/(12D0*(-1D0+STB**2))-(GS**2*PI**(-2)*
     &  STB**2*(MB**2+MGL**2-MSB2**2+4D0*MB*MGL*STB*
     &  DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0))*DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MB)**2))-CDLOG((-I*EPS+MGL)**2)))/2D0-((-I*
     &  EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*
     &  CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/(2D0*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**
     &  (-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+
     &  MGL)**(-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB2**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-
     &  2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**
     &  (-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB2**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0)))))/(6D0*(-1D0+STB**2)))
      end


      complex*16 function MSb1shiftDRED3()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDRED3=
     &  EINS*((GS**2*PI**(-2)*(-(MGL**2*(-1D0+STT**2))+(MST1**2-MT**
     &  2)*(-1D0+STT**2)-4D0*MGL*MT*STT**3*DCMPLX(DCMPLX(1D0-STT**
     &  2D0))**(1D0/2D0))*DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2))-
     &  CDLOG((-I*EPS+MT)**2)))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))))/(6D0*(-1D0+STB**
     &  2))+(GS**2*PI**(-2)*(MGL**2-MST1**2+MT**2-4D0*MGL*MT*STT*
     &  DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MST1)**2)))/2D0-((-I*
     &  EPS+MST1)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2))*
     &  CDLOG((-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)))/(2D0*(MT**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MST1)**2*(-(CDLOG(((-I*EPS+
     &  MST1)**(-2)*((-I*EPS+MGL)**2+(-I*EPS+MST1)**2-MT**2+
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)-(-I*EPS+MST1)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST1)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MST1)**(-2)*((-I*EPS+MGL)**2+(-I*
     &  EPS+MST1)**2-MT**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-
     &  I*EPS+MST1)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**
     &  2D0+MT**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)-(-I*EPS+MST1)**(-
     &  2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST1)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MT**2+EPS*DCMPLX(0D0,1D0)))))/(12D0*(-
     &  1D0+STB**2))+(GS**2*PI**(-2)*(MGL**2-MST2**2+MT**2+4D0*MGL*
     &  MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MST2)**2)))/2D0-
     &  ((-I*EPS+MST2)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MST2)**(-
     &  2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)))/(2D0*(MT**2+
     &  EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MST2)**2*(-(CDLOG(((-I*EPS+
     &  MST2)**(-2)*((-I*EPS+MGL)**2+(-I*EPS+MST2)**2-MT**2+
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)-(-I*EPS+MST2)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST2)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MST2)**(-2)*((-I*EPS+MGL)**2+(-I*
     &  EPS+MST2)**2-MT**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-
     &  I*EPS+MST2)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**
     &  2D0+MT**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)-(-I*EPS+MST2)**(-
     &  2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST2)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MT**2+EPS*DCMPLX(0D0,1D0)))))/(12D0*(-
     &  1D0+STB**2)))
      end


      complex*16 function MSb1shiftDRED4()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDRED4=
     &  EINS*((GS**2*PI**(-2)*STT**2*(MGL**2-MST2**2+MT**2+4D0*MGL*
     &  MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MT)**2)))/2D0-((-
     &  I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2))*
     &  CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/(2D0*(MST2**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-(CDLOG(((-I*EPS+MT)**(-
     &  2)*((-I*EPS+MGL)**2-MST2**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+
     &  MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**
     &  2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*
     &  (MST2**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*
     &  ((-I*EPS+MGL)**2-MST2**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*
     &  (-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**
     &  2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**
     &  (-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-
     &  I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**
     &  2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MST2**2+EPS*DCMPLX(0D0,
     &  1D0)))))/(6D0*(-1D0+STB**2))+(GS**2*MSB1**2*PI**(-2)*(-1D0+
     &  2D0*STB**2)*(-(MSB2**2*(1D0+2D0*STB**4))+MSB1**2*(1D0-2D0*
     &  STB**2+2D0*STB**4)+2D0*STB**2*(MB**2+MST1**2-MT**2-MST1**2*
     &  STT**2+MST2**2*STT**2-MZ**2*DCOS(2D0*B)+(-(MW**2)+MZ**2)*
     &  DCOS(2D0*B)))*LOG(MSB1**2))/(12D0*(MSB1**2-MSB2**2)*(-
     &  1D0+STB**2))-(GS**2*MSB2**2*PI**(-2)*(-1D0+2D0*STB**2)*(-
     &  (MSB2**2*(1D0+2D0*STB**4))+MSB1**2*(1D0-2D0*STB**2+2D0*STB**
     &  4)+2D0*STB**2*(MB**2+MST1**2-MT**2-MST1**2*STT**2+MST2**2*
     &  STT**2-MZ**2*DCOS(2D0*B)+(-(MW**2)+MZ**2)*DCOS(2D0*
     &  B)))*LOG(MSB2**2))/(12D0*(MSB1**2-MSB2**2)*(-1D0+STB**
     &  2))-(GS**2*MST1**2*PI**(-2)*(-1D0+2D0*STT**2)*LOG(MST1**2))/
     &  (12D0*(-1D0+STB**2))+(GS**2*MST2**2*PI**(-2)*(-1D0+2D0*STT**
     &  2)*LOG(MST2**2))/(12D0*(-1D0+STB**2)))
      end




c----------------------------------------------------------------
c
c --> MSb1shiftDREGcode.f
c
c----------------------------------------------------------------
      
      complex*16 function MSb1shiftDREG1()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDREG1=
     &  EINS*((GS**2*PI**(-2)*(-(MSB2**2)-MST1**2+MST2**2+2D0*MT**2+
     &  2D0*MSB2**2*STB**2+2D0*MST1**2*STB**2-2D0*MT**2*STB**2-2D0*
     &  MSB2**2*STB**4-4D0*MST1**2*STB**4+4D0*MT**2*STB**4+4D0*
     &  MSB2**2*STB**6+MB**2*(-2D0+2D0*STB**2-4D0*STB**4)+MSB1**2*
     &  (1D0-4D0*STB**2+6D0*STB**4-4D0*STB**6)+2D0*MST1**2*STT**2-
     &  2D0*MST2**2*STT**2-2D0*MST1**2*STB**2*STT**2+2D0*MST2**2*
     &  STB**2*STT**2+4D0*MST1**2*STB**4*STT**2-4D0*MST2**2*STB**4*
     &  STT**2-2D0*MZ**2*STB**2*DCOS(2D0*B)+2D0*(-(MW**2)+MZ**2)*
     &  STB**2*DCOS(2D0*B)+4D0*MZ**2*STB**4*DCOS(2D0*B)-4D0*(-
     &  (MW**2)+MZ**2)*STB**4*DCOS(2D0*B)))/(12D0*(-1D0+STB**2))+
     &  (GS**2*MB**2*PI**(-2)*DREAL(2D0-CDLOG((-I*EPS+MB)**2)+
     &  (CDLOG((-I*EPS+MB)**(-2)*(-(MB**2)+(-I*EPS+MB)**2+EPS*
     &  DCMPLX(0D0,-1D0)))*(-(MB**2)+(-I*EPS+MB)**2+EPS*DCMPLX(0D0,-
     &  1D0)))/(MB**2+EPS*DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))+
     &  (GS**2*MSB1**2*PI**(-2)*DREAL(2D0-CDLOG((-I*EPS+MSB1)**2)+
     &  (CDLOG((-I*EPS+MSB1)**(-2)*(-(MSB1**2)+(-I*EPS+MSB1)**2+EPS*
     &  DCMPLX(0D0,-1D0)))*(-(MSB1**2)+(-I*EPS+MSB1)**2+EPS*
     &  DCMPLX(0D0,-1D0)))/(MSB1**2+EPS*DCMPLX(0D0,1D0))))/3D0-(GS**
     &  2*MSB2**2*PI**(-2)*STB**2*DREAL(2D0-CDLOG((-I*EPS+MSB2)**2)+
     &  (CDLOG((-I*EPS+MSB2)**(-2)*(-(MSB2**2)+(-I*EPS+MSB2)**2+EPS*
     &  DCMPLX(0D0,-1D0)))*(-(MSB2**2)+(-I*EPS+MSB2)**2+EPS*
     &  DCMPLX(0D0,-1D0)))/(MSB2**2+EPS*DCMPLX(0D0,1D0))))/(3D0*(-
     &  1D0+STB**2))-(GS**2*PI**(-2)*(MB**2+MGL**2-MSB1**2-4D0*MB*
     &  MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0))*DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MSB1)**2)))/2D0-
     &  ((-I*EPS+MSB1)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MSB1)**(-
     &  2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MSB1)**(-2)))/(2D0*(MB**2+
     &  EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MSB1)**2*(-(CDLOG(((-I*EPS+
     &  MSB1)**(-2)*(-(MB**2)+(-I*EPS+MGL)**2+(-I*EPS+MSB1)**2+
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MSB1)**2D0)+
     &  (I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-I*EPS+MSB1)**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MSB1)**(-2)-(-I*EPS+MSB1)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MSB1)**2D0)+
     &  (I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-I*EPS+MSB1)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MSB1)**(-2)*(MB**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MSB1)**(-2)*(-(MB**2)+(-I*EPS+
     &  MGL)**2+(-I*EPS+MSB1)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MSB1)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-
     &  I*EPS+MSB1)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/
     &  2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+MSB1)**(-2)-(-I*EPS+
     &  MSB1)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+
     &  MSB1)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-I*EPS+MSB1)**
     &  2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB1)**(-2)*(MB**2+EPS*
     &  DCMPLX(0D0,1D0))))/2D0))/(2D0*(MB**2+EPS*DCMPLX(0D0,
     &  1D0)))))/(12D0*(-1D0+STB**2))+(GS**2*PI**(-2)*(MB**2+MGL**2-
     &  MSB1**2-(4D0*MB*MGL*MSB2**2*STB*(-1D0+2D0*STB**2))/((-
     &  (MSB1**2)+MSB2**2)*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0))-
     &  4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0)-(4D0*
     &  MB*MGL*STB*(1D0-2D0*STB**2)*(-(MB**2)+MT**2+MSB2**2*STB**2-
     &  MST2**2*STT**2+MST1**2*(-1D0+STT**2)+MZ**2*DCOS(2D0*B)-(-
     &  (MW**2)+MZ**2)*DCOS(2D0*B)))/((MSB1**2-MSB2**2)*
     &  DCMPLX(DCMPLX(1D0-STB**2D0))**(3D0/2D0)))*DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MB)**2))-CDLOG((-I*EPS+MGL)**2)))/2D0-((-I*
     &  EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*
     &  CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/(2D0*(MSB1**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**
     &  (-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+
     &  MGL)**(-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB1**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-
     &  2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB1**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**
     &  (-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB1**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB1**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB1**2+EPS*
     &  DCMPLX(0D0,1D0)))))/6D0)
      end


      complex*16 function MSb1shiftDREG2()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDREG2=
     &  EINS*(-((GS**2*MST1**2*PI**(-2)*(-1D0+STT**2)*DREAL(2D0-
     &  CDLOG((-I*EPS+MST1)**2)+(CDLOG((-I*EPS+MST1)**(-2)*(-(MST1**
     &  2)+(-I*EPS+MST1)**2+EPS*DCMPLX(0D0,-1D0)))*(-(MST1**2)+(-I*
     &  EPS+MST1)**2+EPS*DCMPLX(0D0,-1D0)))/(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/(3D0*(-1D0+STB**2)))+(GS**2*MST2**2*PI**(-2)*STT**2*
     &  DREAL(2D0-CDLOG((-I*EPS+MST2)**2)+(CDLOG((-I*EPS+MST2)**(-
     &  2)*(-(MST2**2)+(-I*EPS+MST2)**2+EPS*DCMPLX(0D0,-1D0)))*(-
     &  (MST2**2)+(-I*EPS+MST2)**2+EPS*DCMPLX(0D0,-1D0)))/(MST2**2+
     &  EPS*DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))-(GS**2*MT**2*PI**
     &  (-2)*DREAL(2D0-CDLOG((-I*EPS+MT)**2)+(CDLOG((-I*EPS+MT)**(-
     &  2)*(-(MT**2)+(-I*EPS+MT)**2+EPS*DCMPLX(0D0,-1D0)))*(-(MT**
     &  2)+(-I*EPS+MT)**2+EPS*DCMPLX(0D0,-1D0)))/(MT**2+EPS*
     &  DCMPLX(0D0,1D0))))/(3D0*(-1D0+STB**2))-(GS**2*PI**(-2)*(MB**
     &  2+MGL**2-MSB2**2+4D0*MB*MGL*STB*DCMPLX(DCMPLX(1D0-STB**
     &  2D0))**(1D0/2D0))*DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2))-
     &  CDLOG((-I*EPS+MSB2)**2)))/2D0-((-I*EPS+MSB2)**2*(-1D0+(-I*
     &  EPS+MGL)**2*(-I*EPS+MSB2)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*
     &  EPS+MSB2)**(-2)))/(2D0*(MB**2+EPS*DCMPLX(0D0,1D0)))+((-I*
     &  EPS+MSB2)**2*(-(CDLOG(((-I*EPS+MSB2)**(-2)*(-(MB**2)+(-I*
     &  EPS+MGL)**2+(-I*EPS+MSB2)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-
     &  1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MSB2)**(-
     &  2)-(-I*EPS+MSB2)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**2D0-(-
     &  I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB2)**(-2)*
     &  (MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MSB2)**(-2)*
     &  (-(MB**2)+(-I*EPS+MGL)**2+(-I*EPS+MSB2)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-
     &  I*EPS+MGL)**2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+
     &  MSB2)**(-2)-(-I*EPS+MSB2)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MSB2)**2D0)+(I*EPS+MB**2D0-(-I*EPS+MGL)**
     &  2D0-(-I*EPS+MSB2)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MSB2)**(-
     &  2)*(MB**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MB**2+EPS*
     &  DCMPLX(0D0,1D0)))))/(12D0*(-1D0+STB**2))-(GS**2*PI**(-2)*
     &  STB**2*(MB**2+MGL**2-MSB2**2+4D0*MB*MGL*STB*
     &  DCMPLX(DCMPLX(1D0-STB**2D0))**(1D0/2D0))*DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MB)**2))-CDLOG((-I*EPS+MGL)**2)))/2D0-((-I*
     &  EPS+MGL)**2*(-1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**(-2))*
     &  CDLOG((-I*EPS+MB)**2*(-I*EPS+MGL)**(-2)))/(2D0*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MGL)**2*(-(CDLOG(((-I*EPS+MGL)**
     &  (-2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MB)**2*(-I*EPS+
     &  MGL)**(-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB2**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MGL)**(-
     &  2)*((-I*EPS+MB)**2+(-I*EPS+MGL)**2-MSB2**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MB)**2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+
     &  MB)**2D0-(-I*EPS+MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MB)**2*(-I*EPS+MGL)**
     &  (-2)-(-I*EPS+MGL)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MB)**
     &  2D0*(-I*EPS+MGL)**2D0)+(I*EPS-(-I*EPS+MB)**2D0-(-I*EPS+
     &  MGL)**2D0+MSB2**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MGL)**(-2)*
     &  (MSB2**2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MSB2**2+EPS*
     &  DCMPLX(0D0,1D0)))))/(6D0*(-1D0+STB**2)))
      end


      complex*16 function MSb1shiftDREG3()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDREG3=
     &  EINS*((GS**2*PI**(-2)*(-(MGL**2*(-1D0+STT**2))+(MST1**2-MT**
     &  2)*(-1D0+STT**2)-4D0*MGL*MT*STT**3*DCMPLX(DCMPLX(1D0-STT**
     &  2D0))**(1D0/2D0))*DREAL(2D0+((-(CDLOG((-I*EPS+MGL)**2))-
     &  CDLOG((-I*EPS+MT)**2)))/2D0-((-I*EPS+MT)**2*(-1D0+(-I*EPS+
     &  MGL)**2*(-I*EPS+MT)**(-2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+
     &  MT)**(-2)))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))+((-I*EPS+
     &  MT)**2*(-(CDLOG(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**2+
     &  (-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+
     &  CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**
     &  (-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*((-I*EPS+MGL)**2-MST1**
     &  2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*
     &  EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+
     &  MT)**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*
     &  EPS-(-I*EPS+MGL)**2D0+MST1**2D0-(-I*EPS+MT)**2D0)**2D0))**
     &  (1D0/2D0)-(-I*EPS+MT)**(-2)*(MST1**2+EPS*DCMPLX(0D0,1D0))))/
     &  2D0))/(2D0*(MST1**2+EPS*DCMPLX(0D0,1D0)))))/(6D0*(-1D0+STB**
     &  2))+(GS**2*PI**(-2)*(MGL**2-MST1**2+MT**2-4D0*MGL*MT*STT*
     &  DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+((-
     &  (CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MST1)**2)))/2D0-((-I*
     &  EPS+MST1)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2))*
     &  CDLOG((-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)))/(2D0*(MT**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MST1)**2*(-(CDLOG(((-I*EPS+
     &  MST1)**(-2)*((-I*EPS+MGL)**2+(-I*EPS+MST1)**2-MT**2+
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)-(-I*EPS+MST1)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST1)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MST1)**(-2)*((-I*EPS+MGL)**2+(-I*
     &  EPS+MST1)**2-MT**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-
     &  I*EPS+MST1)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**
     &  2D0+MT**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MST1)**(-2)-(-I*EPS+MST1)**(-
     &  2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST1)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST1)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST1)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MT**2+EPS*DCMPLX(0D0,1D0)))))/(12D0*(-
     &  1D0+STB**2))+(GS**2*PI**(-2)*(MGL**2-MST2**2+MT**2+4D0*MGL*
     &  MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MST2)**2)))/2D0-
     &  ((-I*EPS+MST2)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MST2)**(-
     &  2))*CDLOG((-I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)))/(2D0*(MT**2+
     &  EPS*DCMPLX(0D0,1D0)))+((-I*EPS+MST2)**2*(-(CDLOG(((-I*EPS+
     &  MST2)**(-2)*((-I*EPS+MGL)**2+(-I*EPS+MST2)**2-MT**2+
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-
     &  I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)-(-I*EPS+MST2)**(-2)*
     &  DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**2D0)+
     &  (I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST2)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))*(-(((-I*EPS+MST2)**(-2)*((-I*EPS+MGL)**2+(-I*
     &  EPS+MST2)**2-MT**2+DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-
     &  I*EPS+MST2)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**
     &  2D0+MT**2D0)**2D0))**(1D0/2D0)+EPS*DCMPLX(0D0,-1D0)))/2D0)+
     &  ((1D0+(-I*EPS+MGL)**2*(-I*EPS+MST2)**(-2)-(-I*EPS+MST2)**(-
     &  2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MST2)**
     &  2D0)+(I*EPS-(-I*EPS+MGL)**2D0-(-I*EPS+MST2)**2D0+MT**2D0)**
     &  2D0))**(1D0/2D0)-(-I*EPS+MST2)**(-2)*(MT**2+EPS*DCMPLX(0D0,
     &  1D0))))/2D0))/(2D0*(MT**2+EPS*DCMPLX(0D0,1D0)))))/(12D0*(-
     &  1D0+STB**2)))
      end


      complex*16 function MSb1shiftDREG4()
c -------------------------------------------------------------------
c varcom.h
c
      double precision MSt1, MSt2, Mgl, MT, MB, MW, MZ, MA
     $               , stt, ctt, stb, ctb  
     $               , MSb1, MSb2, Mue, PI, sw2, sw, cw
     $               , cf, el, gs, a, as, gf
     $               , tb, b, c2b, sb, cb, pref, eps, eins
     $               , msusytl, msusytr, msusybl, msusybr, mlrt, mlrb
     $               , x2, delmst, msusytaul, msusytaur
      complex*16 cspen, i, res, res1, res2, res3, res4, res5, res6
      integer r, s, t, dr1l
      double precision MSmuLtot, MSmuRtot, MSmuneut

      common/masses/MSt1, MSt2, MSb1, MSb2, Mgl, Mue, delmst
      common/input/msusytl, msusytr, msusybl, msusybr, mlrt, mlrb,
     $             msusytaul, msusytaur
      common/prec/tb, b, c2b, sb, cb, MZ, MW, MA, sw2, sw, cw, MT, MB, 
     $             gf, as, el, a, gs, stb, cf, stt, eps, i, eins, pi
      common /Sbottomshift/ dr1l
      common /SmuonSector/ MSmuLtot, MSmuRtot, MSmuneut

      double precision xmh12, xmh22, xma, xsa, xca
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca
c -------------------------------------------------------------------

      MSb1shiftDREG4=
     &  EINS*((GS**2*PI**(-2)*STT**2*(MGL**2-MST2**2+MT**2+4D0*MGL*
     &  MT*STT*DCMPLX(DCMPLX(1D0-STT**2D0))**(1D0/2D0))*DREAL(2D0+
     &  ((-(CDLOG((-I*EPS+MGL)**2))-CDLOG((-I*EPS+MT)**2)))/2D0-((-
     &  I*EPS+MT)**2*(-1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**(-2))*
     &  CDLOG((-I*EPS+MGL)**2*(-I*EPS+MT)**(-2)))/(2D0*(MST2**2+EPS*
     &  DCMPLX(0D0,1D0)))+((-I*EPS+MT)**2*(-(CDLOG(((-I*EPS+MT)**(-
     &  2)*((-I*EPS+MGL)**2-MST2**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-
     &  (4D0*(-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+
     &  MGL)**2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0))+CDLOG(((1D0+(-I*EPS+MGL)**2*(-I*
     &  EPS+MT)**(-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+
     &  MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**
     &  2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*
     &  (MST2**2+EPS*DCMPLX(0D0,1D0))))/2D0))*(-(((-I*EPS+MT)**(-2)*
     &  ((-I*EPS+MGL)**2-MST2**2+(-I*EPS+MT)**2+DCMPLX(DCMPLX(-(4D0*
     &  (-I*EPS+MGL)**2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**
     &  2D0+MST2**2D0-(-I*EPS+MT)**2D0)**2D0))**(1D0/2D0)+EPS*
     &  DCMPLX(0D0,-1D0)))/2D0)+((1D0+(-I*EPS+MGL)**2*(-I*EPS+MT)**
     &  (-2)-(-I*EPS+MT)**(-2)*DCMPLX(DCMPLX(-(4D0*(-I*EPS+MGL)**
     &  2D0*(-I*EPS+MT)**2D0)+(I*EPS-(-I*EPS+MGL)**2D0+MST2**2D0-(-
     &  I*EPS+MT)**2D0)**2D0))**(1D0/2D0)-(-I*EPS+MT)**(-2)*(MST2**
     &  2+EPS*DCMPLX(0D0,1D0))))/2D0))/(2D0*(MST2**2+EPS*DCMPLX(0D0,
     &  1D0)))))/(6D0*(-1D0+STB**2))+(GS**2*MSB1**2*PI**(-2)*(-1D0+
     &  2D0*STB**2)*(-(MSB2**2*(1D0+2D0*STB**4))+MSB1**2*(1D0-2D0*
     &  STB**2+2D0*STB**4)+2D0*STB**2*(MB**2+MST1**2-MT**2-MST1**2*
     &  STT**2+MST2**2*STT**2-MZ**2*DCOS(2D0*B)+(-(MW**2)+MZ**2)*
     &  DCOS(2D0*B)))*LOG(MSB1**2))/(12D0*(MSB1**2-MSB2**2)*(-
     &  1D0+STB**2))-(GS**2*MSB2**2*PI**(-2)*(-1D0+2D0*STB**2)*(-
     &  (MSB2**2*(1D0+2D0*STB**4))+MSB1**2*(1D0-2D0*STB**2+2D0*STB**
     &  4)+2D0*STB**2*(MB**2+MST1**2-MT**2-MST1**2*STT**2+MST2**2*
     &  STT**2-MZ**2*DCOS(2D0*B)+(-(MW**2)+MZ**2)*DCOS(2D0*
     &  B)))*LOG(MSB2**2))/(12D0*(MSB1**2-MSB2**2)*(-1D0+STB**
     &  2))-(GS**2*MST1**2*PI**(-2)*(-1D0+2D0*STT**2)*LOG(MST1**2))/
     &  (12D0*(-1D0+STB**2))+(GS**2*MST2**2*PI**(-2)*(-1D0+2D0*STT**
     &  2)*LOG(MST2**2))/(12D0*(-1D0+STB**2)))
      end



      subroutine delrho(mst1, mst2, stt, msb1, msb2, stb, mgl, 
     $           mt, mb, xgf, as, cf, gs, el, mz, mw, 
     $           mh12eff, mh22eff, ma, aleff, beta,
     $           res1loop, res2loopgluon, res2loopgluino,
     $           res2loopmt2yuk, res2loopmt2yuksm)

      double precision mst1, mst2, stt, ctt, mt, 
     $                 msb1, msb2, stb, ctb, mb, mgl, 
     $                 gf, as, pi, res, xgf, cf, gs, el, mw, mz, pref
      double precision oneloop, twoloop, xyf0, xyf1,
     $                 res1loop, res2loopgluon, res2loopgluino,
     $                 res2loopmt2yuk, res2loopmt2yuksm
      complex*16 delrhogluino

      double precision xmh12, xmh22, xma, xsa, xca,
     $                 mh12eff, mh22eff, ma, aleff, beta
      common/xhiggs/ xmh12, xmh22, xma, xsa, xca

      integer delmbresum
      double precision dmb, mbbdmb
      double precision msb1dmb, msb2dmb, stbdmb, tsbdmb
      double precision msb1save, msb2save, stbsave
      common /deltambresum/dmb, msb1dmb, msb2dmb, stbdmb, tsbdmb, 
     $                     delmbresum
      mbbdmb = mbb/(1d0 + dmb)

      if (delmbresum.eq.2) then
         msb1save = msb1
         msb2save = msb2
         stbsave = stb
         msb1 = msb1dmb
         msb2 = msb2dmb
         stb = stbdmb
      endif

c$$$      write(*,*) 'paramters in delrho:'
c$$$      write(*,*) 'mst1, mst2, stt, msb1, msb2, stb, xgf, as, mt, mb'
c$$$      write(*,*) 'mgl, cf, gs, el, mz, mw, mh, mH, MA, al, be'
c$$$      write(*,*) real(mst1), real(mst2), real(stt), real(msb1), 
c$$$     $           real(msb2), real(stb), real(xgf), real(as),
c$$$     $           real(mt), real(mb), real(mgl), real(cf), real(gs),
c$$$     $           real(el), real(mz), real(mw), real(mh12eff),
c$$$     $           real(mh22eff), real(ma), real(aleff), real(beta)

      pi = 3.14159265358979d0
      gf = xgf
      ctt = dsqrt(1d0 - stt**2)
      ctb = dsqrt(1d0 - stb**2)
      pref = (3d0 * cf * gs**2 * el**2 * mz**2)/
     $       (2**6 * mw**2 * (mw**2 - mz**2) * pi**4)
      prefmt2yuk = (3d0 * el**4 * mt**4 * mz**4)/
     $             (2**12 * mw**4 * (mw**2 - mz**2)**2 * pi**4)
      xmh12 = mh12eff
      xmh22 = mh22eff
      xsa = dsin(aleff)
      xca = dcos(aleff)
      xma = ma
 
      oneloop = 3d0 * gf/(8d0 * dsqrt(2d0) * pi**2) *
     $          ( - stt**2 * ctt**2 * xyf0(mst1**2, mst2**2)
     $            - stb**2 * ctb**2 * xyf0(msb1**2, msb2**2)
     $            + ctt**2 * ctb**2 * xyf0(mst1**2, msb1**2)
     $            + ctt**2 * stb**2 * xyf0(mst1**2, msb2**2)
     $            + stt**2 * ctb**2 * xyf0(mst2**2, msb1**2)
     $            + stt**2 * stb**2 * xyf0(mst2**2, msb2**2) )

      twoloop = gf * as/(4d0 * dsqrt(2d0) * pi**3) *
     $          ( - stt**2 * ctt**2 * xyf1(mst1**2, mst2**2)
     $            - stb**2 * ctb**2 * xyf1(msb1**2, msb2**2)
     $            + ctt**2 * ctb**2 * xyf1(mst1**2, msb1**2)
     $            + ctt**2 * stb**2 * xyf1(mst1**2, msb2**2)
     $            + stt**2 * ctb**2 * xyf1(mst2**2, msb1**2)
     $            + stt**2 * stb**2 * xyf1(mst2**2, msb2**2) )

      delrhogluino = 0d0

c      write(*,*) 'DelrhoSub:'
      res1loop = oneloop
      res2loopgluon = twoloop
      res2loopgluino = dreal(delrhoGluino) * pref


      res2loopmt2yuk = 0d0
      res2loopmt2yuksm = 0d0
c      write(*,*) 'MT2Yuk:', res2loopmt2yuk, res2loopmt2yuksm, prefmt2yuk




      if (delmbresum.eq.2) then
         msb1 = msb1save
         msb2 = msb2save
         stb = stbsave
         ctb = dsqrt(1d0 - stb**2)
      endif

      end


c-------------------------------------------------------------------

      double precision function xyf0(x,y)

      double precision x, y

      if (x.ne.y) then
c         write(*,*) 'xyf0:', x, y
      xyf0 = x + y - (2d0 * x * y)/(x - y) * dlog(x/y)
      else
      xyf0 = 0d0
      endif

      end


c-------------------------------------------------------------------

      double precision function xyf1(x2,y2)

      double precision x2, y2
      complex*16 ff, x, y, cspen
      
      x = (dsqrt(x2) - (0d0,1d0) * 10d-10)**2
      y = (dsqrt(y2) - (0d0,1d0) * 10d-10)**2

      if (x.ne.y) then
c         write(*,*) 'xyf1:', x, y
      ff = x + y 
     $   - (2d0 * x * y)/(x - y) * cdlog(x/y) * (2d0 + x/y * cdlog(x/y))
     $   + (x + y)*x**2/(x - y)**2 * (cdlog(x/y))**2 
     $   - 2d0 * (x - y) * cspen(1d0 - x/y)

      xyf1 = dreal(ff)
      else
      xyf1 = 0d0
      endif

      end


c-------------------------------------------------------------------

      double precision function switchoff(m)

      double precision m,so

      so = (m - 750d0)/125d0
      if (so.le.0d0) so = 0d0
      
      switchoff = so**2

      end

c-------------------------------------------------------------------
