c====================================================================
c
c  Define common block containing all coupling constants and masses
c  which are used in the HELAS routines.
c
c  These include masses, widths and real/complex couplings.
c
c  This file can be built automatically from particles.dat and
c  interactions.dat
c
c====================================================================
c
c     QCD 
c
      double complex       gg(2)
      double precision     g
      common /COUPL_QCD/   g,gg   
c
c    kinematical masses
c
      double precision     hmass, wmass, zmass, amass,
     &                     tmass, bmass, lmass, cmass
      common /COUPL_MASS/  hmass, wmass, zmass, amass,
     &                     tmass, bmass, lmass, cmass
c
c   widths
c
      double precision     hwidth, wwidth, zwidth, 
     &                     twidth, lwidth, awidth
      common /COUPL_WIDTH/ hwidth, wwidth, zwidth, 
     &                     twidth, lwidth, awidth
c
c  couplings in the feynman rules
c
      double complex       gal(2), gad(2), gau(2), gwf(2),
     &                     gzn(2), gzl(2), gzd(2), gzu(2)
      double precision     gw, gwwa, gwwz
      common /COUPL_GAUGE/ gal   , gad   , gau   , gwf   ,
     &                     gzn   , gzl   , gzd   , gzu   ,
     &                     gw, gwwa, gwwz
c
      double complex       gwfc(2),  gwfs(2), gwfm(2)
      common /coupl_ckm/   gwfc,     gwfs   , gwfm	
c
      double complex       gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh
      common /COUPL_SCAL/  gwwh, gzzh, gwwhh, gzzhh, ghhh, ghhhh
c
      double complex       ghtop(2), ghbot(2), ghtau(2), ghcha(2)
      common /COUPL_YUK/   ghtop   , ghbot   , ghtau   , ghcha
!
! couplings in the R2 Feynman rules:
!
      double complex       galr(2), gadr(2), gaur(2), gwfr(2),
     &                     gznr(2), gzlr(2), gzdr(2), gzur(2)
      double complex       pgal(4), pgad(4), pgau(4), pgwf(4),
     &                     pgzn(4), pgzl(4), pgzd(4), pgzu(4)

      common /COUPL_GAUGEr2/ galr   , gadr   , gaur   , gwfr   ,
     &                     gznr   , gzlr   , gzdr   , gzur   , 
     &                     pgal  , pgad   , pgau   , pgwf   ,
     &                     pgzn  , pgzl   , pgzd   , pgzu	   

     
      double complex       ghtopr(2), ghbotr(2), ghtaur(2), 
     &                     ghchar(2)
      common /COUPL_YUKr2/   ghtopr,ghbotr,ghtaur,ghchar

      double complex       ggr(2), pgg(4)
      common /COUPL_QCDr2/ ggr,  pgg
