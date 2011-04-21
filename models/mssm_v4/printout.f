      subroutine printout()
      implicit none
c
c     argument
c
c      integer detail_level
c
c     local
c
      integer i,iformat
      character*2 ab(2)
      real*8 ene
c
c     include
c
      include 'coupl.inc'
      include 'sm_read_values.inc'
c     include '../PDF/pdf.inc'
c      include '../alfas.inc'
c      include '../run.inc'
c
c     numbers
c
      double precision  Zero, One, Two, Three, Four, Half, Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
      parameter( Three = 3.0d0, Four = 4.0d0, Half = 0.5d0 )
      parameter( Rt2   = 1.414213562d0 )
      double precision  Pi, Fourpi
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Fourpi = Four * Pi )
c
c output all info
c
 10   format( 1x,a6,' mass = ',f8.3,' GeV ',3x,a6,' width = ',f8.4,' GeV')
c 11   format( 1x,a10,' = ',f10.7,2x,f10.7,a10,' = ',f10.7,2x,f10.7 )
c 12   format( 1x,a10,' = ',f6.2,a )
c 13   format( 1x,a5,' mass = ',f6.4,a5,' width = ' )
 14   format( 1x,2(a10,' = ',f12.7,1x,f12.7,3x) )
c 15   format( 1x,a10,' = ',f9.5,a )
c 16   format( 1x,a10,' = ',f7.5 )
c 17   format( 1x,a10,' = ',f8.4 )
c 18   format( 1x,a10,' = ',f8.4,' GeV' )
c 19   format( 1x,a10,' = ',f6.4,a10,' = ',f6.4 )
c 20   format( 1x,a10,' = ',f11.5,1x,f11.5 )
c 21   format( 1x,a10,' = ',f8.4,' GeV',1x,a10,' = ',f8.4,' GeV' )
c 22   format( 1x,a10,' = ',f10.8,a,f6.4 )
 22   format( 1x,a18,' = ',f8.4,3x,a,3x,i2 )
 23   format( 1x,a18,' = ',f8.4,3x,a,3x,f8.4 )
c 24   format( 1x,a,f7.3,' GeV        ',a,f7.4,' GeV  (calc @ LO)')
c 25   format( 1x,a,f8.3,' GeV',1x,a,f8.3,' GeV' )

      write(6,*)
      write(6,*)  'EW Params                     '
      write(6,*)  '-------------------------------'
      write(6,23) 'GF (10^-5*GeV^-2) ',gfermi*1d5
      write(6,23) '1/alpha           ',One/alpha
      write(6,23) 'M_Z   (GeV)       ',zmass
      write(6,*)

c      write(6,*)  'Strong coupling:      '
c      write(6,*)  '-------------------------------'
c      write(6,22) 'alpha_s (MZ)         ',asmz,'at loop',nloop
c      write(6,23) 'alpha_s (Q2)         ',g**2/Fourpi,'Q2 =',scale
c      write(6,23) 'g_s (Q2)             ',g,'Q2 =',scale      
c      write(6,*)

      write(6,*)  'Electroweak coupling:'
      write(6,*)  '---------------------'
      write(6,23) 'gw                    ',gw 
      write(6,*)
      write(6,*)  'Boson masses and widths:'
      write(6,*)  '------------------------'
      write(6,*)
c      if(lzdecay) then
c         write(6,24) 'Z         ',zmass, 'Z        ',zwidth
c      else
         write(6,10) 'Z         ',zmass, 'Z        ',zwidth
c      endif
c      if(lwdecay) then
c         write(6,24) 'W         ',wmass, 'W        ',wwidth
c      else
         write(6,10) 'W         ',wmass, 'W        ',wwidth
c      endif

c      write(6,10) 'H         ',hmass, 'H        ',hwidth
      write(6,*)
      write(6,*)  'Fermion masses and widths:'
      write(6,*)  '--------------------------'
      write(6,*)
c      if(ltdecay) then
c      write(6,24) 'top            ', tmass, 'top           ', twidth
c      else
      write(6,10) 'top            ', tmass, 'top           ', twidth
c      endif
      write(6,10) 'bottom         ', bmass, 'bottom        ', Zero
c      write(6,10) 'charm          ', cmass, 'charm         ', Zero
      write(6,10) 'tau            ', lmass, 'tau           ', Zero
      write(6,*)  'all other quark and lepton masses set to zero'
      write(6,*)
c      write(6,*)  'Quark running MSbar masses @         ',nloop,' loops:'
c      write(6,*)  '--------------------------------------'
c      write(6,*)
c      write(6,18) 
c     &   'm_top(hmass)_MSbar           ', mt_h
c      write(6,18) 
c     &   'm_bot(hmass)_MSbar           ', mb_h
c      write(6,18) 
c     &   'm_cha(hmass)_MSbar           ', mc_h

      write(6,*)
      write(6,*) 'MSSM particle masses and widths:'
      write(6,*)  '-------------------------------'
      write(6,*)
      write(6,10) 'dl         ',mdl,'dl        ',wdl
      write(6,10) 'dr         ',mdr,'dr        ',wdr
      write(6,10) 'ul         ',mul,'ul        ',wul
      write(6,10) 'ur         ',mur,'ur        ',wur
      write(6,10) 'sl         ',msl,'sl        ',wsl
      write(6,10) 'sr         ',msr,'sr        ',wsr
      write(6,10) 'cl         ',mcl,'cl        ',wcl
      write(6,10) 'cr         ',mcr,'cr        ',wcr
      write(6,10) 'b1         ',mb1,'b1        ',wb1
      write(6,10) 'b2         ',mb2,'b2        ',wb2
      write(6,10) 't1         ',mt1,'t1        ',wt1
      write(6,10) 't2         ',mt2,'t2        ',wt2
      write(6,10) 'el         ',mel,'el        ',wel
      write(6,10) 'er         ',mer,'er        ',wer
      write(6,10) 'ml         ',mml,'ml        ',wml
      write(6,10) 'mr         ',mmr,'mr        ',wmr
      write(6,10) 'l1         ',ml1,'l1        ',wl1
      write(6,10) 'l2         ',ml2,'l2        ',wl2
      write(6,10) 've         ',mve,'ve        ',wve
      write(6,10) 'vm         ',mvm,'vm        ',wvm
      write(6,10) 'vt         ',mvt,'vt        ',wvt
      write(6,10) 'n1         ',mn1,'n1        ',wn1
      write(6,10) 'n2         ',mn2,'n2        ',wn2
      write(6,10) 'n3         ',mn3,'n3        ',wn3
      write(6,10) 'n4         ',mn4,'n4        ',wn4
      write(6,10) 'x1         ',mx1,'x1        ',wx1
      write(6,10) 'x2         ',mx2,'x2        ',wx2
      write(6,10) 'go         ',mgo,'go        ',wgo
      write(6,10) 'h1         ',mh1,'h1        ',wh1
      write(6,10) 'h2         ',mh2,'h2        ',wh2
      write(6,10) 'h3         ',mh3,'h3        ',wh3
      write(6,10) 'hc         ',mhc,'hc        ',whc
      write(6,*)

c      if(detail_level.ge.1) then
c         write(6,*)
c         write(6,*) 'Z,W LO partial widths and effective BRs:'
c         write(6,*) '-------------------------------------'
c         write(6,*)
c         write(6,19) 'width Z->nn  =         ', w_z_nn,
c     &               ' GeV,   BR(Z->nn)  =         ', Three*w_z_nn/zwidth
c         write(6,19) 'width Z->ll  =         ', w_z_ll,
c     &               ' GeV,   BR(Z->ll)  =         ', Two*w_z_ll/zwidth
c         write(6,19) 'width Z->TT  =         ', w_z_tau,
c     &               ' GeV,   BR(Z->TT)  =         ', w_z_tau/zwidth
c         write(6,19) 'width Z->uu  =         ', w_z_uu,
c     &               ' GeV,   BR(Z->uu)  =         ', w_z_uu/zwidth
c         write(6,19) 'width Z->dd  =         ', w_z_dd,
c     &               ' GeV,   BR(Z->dd)  =         ', Two*w_z_dd/zwidth
c         write(6,19) 'width Z->cc  =         ', w_z_cc,
c     &               ' GeV,   BR(Z->cc)  =         ', w_z_cc/zwidth
c         write(6,19) 'width Z->bb  =         ', w_z_bb,
c     &               ' GeV,   BR(Z->bb)  =         ', w_z_bb/zwidth
c         write(6,*)
c         write(6,19) 'width W->nl  =         ', w_w_nl,
c     &               ' GeV,   BR(W->nl)  =         ', Two*w_w_nl/wwidth
c         write(6,19) 'width W->nT  =         ', w_w_tau,
c     &               ' GeV,   BR(W->nT)  =         ', w_w_tau/wwidth
c         write(6,19) 'width W->ud  =         ', w_w_ud, 
c     &               ' GeV,   BR(W->ud)  =         ', w_w_ud/wwidth
c         write(6,19) 'width W->cs  =         ', w_w_cs, 
c     &               ' GeV,   BR(W->cs)  =         ', w_w_cs/wwidth
c         write(6,*)
c      endif

c      if (.not. lhdecay) then
c       write(6,15) 'Higgs total         ',hwidth , ' GeV (set by hand)'
c         write(6,*)
c      else
c         write(6,*) 'HDECAY output for Higgs sector:'
c         write(6,*)  '------------------------------'
c         write(6,*)
c         write(6,15) 'Higgs tot         ', SMWDTH, ' GeV'
c         write(6,*)
c         if(detail_level.ge.1) then
c            write(6,16) 'BR (H -> bb~  ) =          ', SMBRB
c            write(6,16) 'BR (H -> taus ) =          ', SMBRL
c            write(6,16) 'BR (H -> muons) =          ', SMBRM
c            write(6,16) 'BR (H -> ss~  ) =          ', SMBRS
c            write(6,16) 'BR (H -> cc~  ) =          ', SMBRC
c            write(6,16) 'BR (H -> tt~  ) =          ', SMBRT
c            write(6,16) 'BR (H -> gg   ) =          ', SMBRG
c            write(6,16) 'BR (H -> AA   ) =          ', SMBRGA
c            write(6,16) 'BR (H -> ZA   ) =          ', SMBRZGA
c            write(6,16) 'BR (H -> W+W- ) =          ', SMBRW
c            write(6,16) 'BR (H -> ZZ   ) =          ', SMBRZ
c         endif
c         write(6,*)
c      endif
c         write(6,*) 'Higgs LO partial widths and BRs:'
c         write(6,*) '--------------------------------'
c         write(6,*)
c         write(6,22) 'width H->tt~    =         ', w_h_tt,
c     &               ' GeV,   BR_LO(H->tt~)     =         ',w_h_tt/hwidth
c         write(6,22) 'width H->bb~    =         ', w_h_bb,
c     &               ' GeV,   BR_LO(H->bb~)     =         ',w_h_bb/hwidth
c         write(6,22) 'width H->cc~    =         ', w_h_cc,
c     &               ' GeV,   BR_LO(H->cc~)     =         ',w_h_cc/hwidth
c         write(6,22) 'width H->ta ta~ =         ', w_h_tau,
c     &               ' GeV,   BR_LO(H->ta ta~)  =         ',w_h_tau/hwidth
c         write(6,22) 'width H->zz     =         ', w_h_zz,
c     &               ' GeV,   BR_LO(H->zz)      =         ',w_h_zz/hwidth
c         write(6,22) 'width H->w- w+  =         ', w_h_ww,
c     &               ' GeV,   BR_LO(H->w- w+)   =         ',w_h_ww/hwidth
c         write(6,*)

c         if(detail_level.ge.2) then

      write(6,*) 'Boson couplings:'
      write(6,*) '----------------'
      write(6,*)
      write(6,14) 'gwwa        ', gwwa
      write(6,14) 'gwwz        ', gwwz
c      write(6,14) 'gwwh        ', gwwh
c      write(6,14) 'gzzh        ', gzzh
c      write(6,14) 'ghhh        ', ghhh
c      write(6,*)
c      write(6,14) 'gwwhh =         ', gwwhh
c      write(6,14) 'gzzhh =         ', gzzhh
c      write(6,14) 'ghhhh =         ', ghhhh
      write(6,*)
      write(6,*) 'FFV couplings:'
      write(6,*) '--------------'
      write(6,*)
      write(6,14) 'gal(L)          ',gal(1), 'gal(R)          ',gal(2)
      write(6,14) 'gau(L)          ',gau(1), 'gau(R)          ',gau(2)
      write(6,14) 'gad(L)          ',gad(1), 'gad(R)          ',gad(2)
      write(6,*)
      write(6,14) 'gwf(L)          ',gwf(1), 'gwf(R)          ',gwf(2)
      write(6,*)
      write(6,14) 'gzn(L)          ',gzn(1), 'gzn(R)          ',gzn(2)
      write(6,14) 'gzl(L)          ',gzl(1), 'gzl(R)          ',gzl(2)
      write(6,14) 'gzu(L)          ',gzu(1), 'gzu(R)          ',gzu(2)
      write(6,14) 'gzd(L)          ',gzd(1), 'gzd(R)          ',gzd(2)
      write(6,*)
c      write(6,*) 'FFH couplings:'
c      write(6,*) '--------------'
c      write(6,*)
c      write(6,14) 'gHtop(L)         ',ghtop(1), 'gHtop(R)         ',ghtop(2)
c      write(6,14) 'gHbot(L)         ',ghbot(1), 'gHbot(R)         ',ghbot(2)
c      write(6,14) 'gHcha(L)         ',ghcha(1), 'gHcha(R)         ',ghcha(2)
c      write(6,14) 'gHtau(L)         ',ghtau(1), 'gHtau(R)         ',ghtau(2)
c      write(6,*)
      write(6,*) 'Strong couplings:'
      write(6,*) '-----------------'
      write(6,*)
      write(6,14) 'gg(1)           ',gg(1)   , 'gg(2)           ',gg(2)
      write(6,*)

      write(6,*) 'MSSM couplings:'
      write(6,*) '---------------'
      write(6,*)
      write(6,*) 'MSSM_QCD:'
      write(6,14) 'gc        ',gc
      write(6,14) 'g2c        ',g2c
      write(6,14) 'ggi(1)        ',ggi(1),'ggi(2)        ',ggi(2)
      write(6,*)
      write(6,*) 'MSSM_FFV:'
      write(6,14) 'gzn11(1)        ',gzn11(1),'gzn11(2)        ',gzn11(2)
      write(6,14) 'gzn12(1)        ',gzn12(1),'gzn12(2)        ',gzn12(2)
      write(6,14) 'gzn13(1)        ',gzn13(1),'gzn13(2)        ',gzn13(2)
      write(6,14) 'gzn14(1)        ',gzn14(1),'gzn14(2)        ',gzn14(2)
      write(6,14) 'gzn22(1)        ',gzn22(1),'gzn22(2)        ',gzn22(2)
      write(6,14) 'gzn23(1)        ',gzn23(1),'gzn23(2)        ',gzn23(2)
      write(6,14) 'gzn24(1)        ',gzn24(1),'gzn24(2)        ',gzn24(2)
      write(6,14) 'gzn33(1)        ',gzn33(1),'gzn33(2)        ',gzn33(2)
      write(6,14) 'gzn34(1)        ',gzn34(1),'gzn34(2)        ',gzn34(2)
      write(6,14) 'gzn44(1)        ',gzn44(1),'gzn44(2)        ',gzn44(2)
      write(6,14) 'gax(1)          ',gax(1),  'gax(2)          ',gax(2)
      write(6,14) 'gzx11(1)        ',gzx11(1),'gzx11(2)        ',gzx11(2)
      write(6,14) 'gzx12(1)        ',gzx12(1),'gzx12(2)        ',gzx12(2)
      write(6,14) 'gzx22(1)        ',gzx22(1),'gzx22(2)        ',gzx22(2)
      write(6,14) 'gwn1x1(1)        ',gwn1x1(1),'gwn1x1(2)        ',gwn1x1(2)
      write(6,14) 'gwx1n1(1)        ',gwx1n1(1),'gwx1n1(2)        ',gwx1n1(2)
      write(6,14) 'gwn1x2(1)        ',gwn1x2(1),'gwn1x2(2)        ',gwn1x2(2)
      write(6,14) 'gwx2n1(1)        ',gwx2n1(1),'gwx2n1(2)        ',gwx2n1(2)
      write(6,14) 'gwn2x1(1)        ',gwn2x1(1),'gwn2x1(2)        ',gwn2x1(2)
      write(6,14) 'gwx1n2(1)        ',gwx1n2(1),'gwx1n2(2)        ',gwx1n2(2)
      write(6,14) 'gwn2x2(1)        ',gwn2x2(1),'gwn2x2(2)        ',gwn2x2(2)
      write(6,14) 'gwx2n2(1)        ',gwx2n2(1),'gwx2n2(2)        ',gwx2n2(2)
      write(6,14) 'gwn3x1(1)        ',gwn3x1(1),'gwn3x1(2)        ',gwn3x1(2)
      write(6,14) 'gwx1n3(1)        ',gwx1n3(1),'gwx1n3(2)        ',gwx1n3(2)
      write(6,14) 'gwn3x2(1)        ',gwn3x2(1),'gwn3x2(2)        ',gwn3x2(2)
      write(6,14) 'gwx2n3(1)        ',gwx2n3(1),'gwx2n3(2)        ',gwx2n3(2)
      write(6,14) 'gwn4x1(1)        ',gwn4x1(1),'gwn4x1(2)        ',gwn4x1(2)
      write(6,14) 'gwx1n4(1)        ',gwx1n4(1),'gwx1n4(2)        ',gwx1n4(2)
      write(6,14) 'gwn4x2(1)        ',gwn4x2(1),'gwn4x2(2)        ',gwn4x2(2)
      write(6,14) 'gwx2n4(1)        ',gwx2n4(1),'gwx2n4(2)        ',gwx2n4(2)
      write(6,*)
      write(6,*) 'YUK_FFS:'
      write(6,14) 'gh1ll(1)        ',gh1ll(1),'gh1ll(2)        ',gh1ll(2)
      write(6,14) 'gh2ll(1)        ',gh2ll(1),'gh2ll(2)        ',gh2ll(2)
      write(6,14) 'gh3ll(1)        ',gh3ll(1),'gh3ll(2)        ',gh3ll(2)
      write(6,14) 'gh1bb(1)        ',gh1bb(1),'gh1bb(2)        ',gh1bb(2)
      write(6,14) 'gh2bb(1)        ',gh2bb(1),'gh2bb(2)        ',gh2bb(2)
      write(6,14) 'gh3bb(1)        ',gh3bb(1),'gh3bb(2)        ',gh3bb(2)
      write(6,14) 'gh1tt(1)        ',gh1tt(1),'gh1tt(2)        ',gh1tt(2)
      write(6,14) 'gh2tt(1)        ',gh2tt(1),'gh2tt(2)        ',gh2tt(2)
      write(6,14) 'gh3tt(1)        ',gh3tt(1),'gh3tt(2)        ',gh3tt(2)
      write(6,14) 'ghmq(1)        ',ghmq(1),'ghmq(2)        ',ghmq(2)
      write(6,14) 'ghpq(1)        ',ghpq(1),'ghpq(2)        ',ghpq(2)
      write(6,14) 'ghml(1)        ',ghml(1),'ghml(2)        ',ghml(2)
      write(6,14) 'ghpl(1)        ',ghpl(1),'ghpl(2)        ',ghpl(2)
      write(6,*)
      write(6,*) 'MSSM_FFS:'
      write(6,14) 'gqlgom(1)        ',gqlgom(1),'gqlgom(2)        ',gqlgom(2)
      write(6,14) 'gqlgop(1)        ',gqlgop(1),'gqlgop(2)        ',gqlgop(2)
      write(6,14) 'gqrgom(1)        ',gqrgom(1),'gqrgom(2)        ',gqrgom(2)
      write(6,14) 'gqrgop(1)        ',gqrgop(1),'gqrgop(2)        ',gqrgop(2)
      write(6,14) 'gb1gom(1)        ',gb1gom(1),'gb1gom(2)        ',gb1gom(2)
      write(6,14) 'gb1gop(1)        ',gb1gop(1),'gb1gop(2)        ',gb1gop(2)
      write(6,14) 'gb2gom(1)        ',gb2gom(1),'gb2gom(2)        ',gb2gom(2)
      write(6,14) 'gb2gop(1)        ',gb2gop(1),'gb2gop(2)        ',gb2gop(2)
      write(6,14) 'gt1gom(1)        ',gt1gom(1),'gt1gom(2)        ',gt1gom(2)
      write(6,14) 'gt1gop(1)        ',gt1gop(1),'gt1gop(2)        ',gt1gop(2)
      write(6,14) 'gt2gom(1)        ',gt2gom(1),'gt2gom(2)        ',gt2gom(2)
      write(6,14) 'gt2gop(1)        ',gt2gop(1),'gt2gop(2)        ',gt2gop(2)
      write(6,14) 'gdln1m(1)        ',gdln1m(1),'gdln1m(2)        ',gdln1m(2)
      write(6,14) 'gdln1p(1)        ',gdln1p(1),'gdln1p(2)        ',gdln1p(2)
      write(6,14) 'gdln2m(1)        ',gdln2m(1),'gdln2m(2)        ',gdln2m(2)
      write(6,14) 'gdln2p(1)        ',gdln2p(1),'gdln2p(2)        ',gdln2p(2)
      write(6,14) 'gdln3m(1)        ',gdln3m(1),'gdln3m(2)        ',gdln3m(2)
      write(6,14) 'gdln3p(1)        ',gdln3p(1),'gdln3p(2)        ',gdln3p(2)
      write(6,14) 'gdln4m(1)        ',gdln4m(1),'gdln4m(2)        ',gdln4m(2)
      write(6,14) 'gdln4p(1)        ',gdln4p(1),'gdln4p(2)        ',gdln4p(2)
      write(6,14) 'gdrn1m(1)        ',gdrn1m(1),'gdrn1m(2)        ',gdrn1m(2)
      write(6,14) 'gdrn1p(1)        ',gdrn1p(1),'gdrn1p(2)        ',gdrn1p(2)
      write(6,14) 'gdrn2m(1)        ',gdrn2m(1),'gdrn2m(2)        ',gdrn2m(2)
      write(6,14) 'gdrn2p(1)        ',gdrn2p(1),'gdrn2p(2)        ',gdrn2p(2)
      write(6,14) 'gdrn3m(1)        ',gdrn3m(1),'gdrn3m(2)        ',gdrn3m(2)
      write(6,14) 'gdrn3p(1)        ',gdrn3p(1),'gdrn3p(2)        ',gdrn3p(2)
      write(6,14) 'gdrn4m(1)        ',gdrn4m(1),'gdrn4m(2)        ',gdrn4m(2)
      write(6,14) 'gdrn4p(1)        ',gdrn4p(1),'gdrn4p(2)        ',gdrn4p(2)
      write(6,14) 'guln1m(1)        ',guln1m(1),'guln1m(2)        ',guln1m(2)
      write(6,14) 'guln1p(1)        ',guln1p(1),'guln1p(2)        ',guln1p(2)
      write(6,14) 'guln2m(1)        ',guln2m(1),'guln2m(2)        ',guln2m(2)
      write(6,14) 'guln2p(1)        ',guln2p(1),'guln2p(2)        ',guln2p(2)
      write(6,14) 'guln3m(1)        ',guln3m(1),'guln3m(2)        ',guln3m(2)
      write(6,14) 'guln3p(1)        ',guln3p(1),'guln3p(2)        ',guln3p(2)
      write(6,14) 'guln4m(1)        ',guln4m(1),'guln4m(2)        ',guln4m(2)
      write(6,14) 'guln4p(1)        ',guln4p(1),'guln4p(2)        ',guln4p(2)
      write(6,14) 'gurn1m(1)        ',gurn1m(1),'gurn1m(2)        ',gurn1m(2)
      write(6,14) 'gurn1p(1)        ',gurn1p(1),'gurn1p(2)        ',gurn1p(2)
      write(6,14) 'gurn2m(1)        ',gurn2m(1),'gurn2m(2)        ',gurn2m(2)
      write(6,14) 'gurn2p(1)        ',gurn2p(1),'gurn2p(2)        ',gurn2p(2)
      write(6,14) 'gurn3m(1)        ',gurn3m(1),'gurn3m(2)        ',gurn3m(2)
      write(6,14) 'gurn3p(1)        ',gurn3p(1),'gurn3p(2)        ',gurn3p(2)
      write(6,14) 'gurn4m(1)        ',gurn4m(1),'gurn4m(2)        ',gurn4m(2)
      write(6,14) 'gurn4p(1)        ',gurn4p(1),'gurn4p(2)        ',gurn4p(2)
      write(6,14) 'gb1n1m(1)        ',gb1n1m(1),'gb1n1m(2)        ',gb1n1m(2)
      write(6,14) 'gb1n1p(1)        ',gb1n1p(1),'gb1n1p(2)        ',gb1n1p(2)
      write(6,14) 'gb2n1m(1)        ',gb2n1m(1),'gb2n1m(2)        ',gb2n1m(2)
      write(6,14) 'gb2n1p(1)        ',gb2n1p(1),'gb2n1p(2)        ',gb2n1p(2)
      write(6,14) 'gb1n2m(1)        ',gb1n2m(1),'gb1n2m(2)        ',gb1n2m(2)
      write(6,14) 'gb1n2p(1)        ',gb1n2p(1),'gb1n2p(2)        ',gb1n2p(2)
      write(6,14) 'gb2n2m(1)        ',gb2n2m(1),'gb2n2m(2)        ',gb2n2m(2)
      write(6,14) 'gb2n2p(1)        ',gb2n2p(1),'gb2n2p(2)        ',gb2n2p(2)
      write(6,14) 'gb1n3m(1)        ',gb1n3m(1),'gb1n3m(2)        ',gb1n3m(2)
      write(6,14) 'gb1n3p(1)        ',gb1n3p(1),'gb1n3p(2)        ',gb1n3p(2)
      write(6,14) 'gb2n3m(1)        ',gb2n3m(1),'gb2n3m(2)        ',gb2n3m(2)
      write(6,14) 'gb2n3p(1)        ',gb2n3p(1),'gb2n3p(2)        ',gb2n3p(2)
      write(6,14) 'gb1n4m(1)        ',gb1n4m(1),'gb1n4m(2)        ',gb1n4m(2)
      write(6,14) 'gb1n4p(1)        ',gb1n4p(1),'gb1n4p(2)        ',gb1n4p(2)
      write(6,14) 'gb2n4m(1)        ',gb2n4m(1),'gb2n4m(2)        ',gb2n4m(2)
      write(6,14) 'gb2n4p(1)        ',gb2n4p(1),'gb2n4p(2)        ',gb2n4p(2)
      write(6,14) 'gt1n1m(1)        ',gt1n1m(1),'gt1n1m(2)        ',gt1n1m(2)
      write(6,14) 'gt1n1p(1)        ',gt1n1p(1),'gt1n1p(2)        ',gt1n1p(2)
      write(6,14) 'gt2n1m(1)        ',gt2n1m(1),'gt2n1m(2)        ',gt2n1m(2)
      write(6,14) 'gt2n1p(1)        ',gt2n1p(1),'gt2n1p(2)        ',gt2n1p(2)
      write(6,14) 'gt1n2m(1)        ',gt1n2m(1),'gt1n2m(2)        ',gt1n2m(2)
      write(6,14) 'gt1n2p(1)        ',gt1n2p(1),'gt1n2p(2)        ',gt1n2p(2)
      write(6,14) 'gt2n2m(1)        ',gt2n2m(1),'gt2n2m(2)        ',gt2n2m(2)
      write(6,14) 'gt2n2p(1)        ',gt2n2p(1),'gt2n2p(2)        ',gt2n2p(2)
      write(6,14) 'gt1n3m(1)        ',gt1n3m(1),'gt1n3m(2)        ',gt1n3m(2)
      write(6,14) 'gt1n3p(1)        ',gt1n3p(1),'gt1n3p(2)        ',gt1n3p(2)
      write(6,14) 'gt2n3m(1)        ',gt2n3m(1),'gt2n3m(2)        ',gt2n3m(2)
      write(6,14) 'gt2n3p(1)        ',gt2n3p(1),'gt2n3p(2)        ',gt2n3p(2)
      write(6,14) 'gt1n4m(1)        ',gt1n4m(1),'gt1n4m(2)        ',gt1n4m(2)
      write(6,14) 'gt1n4p(1)        ',gt1n4p(1),'gt1n4p(2)        ',gt1n4p(2)
      write(6,14) 'gt2n4m(1)        ',gt2n4m(1),'gt2n4m(2)        ',gt2n4m(2)
      write(6,14) 'gt2n4p(1)        ',gt2n4p(1),'gt2n4p(2)        ',gt2n4p(2)
      write(6,14) 'geln1m(1)        ',geln1m(1),'geln1m(2)        ',geln1m(2)
      write(6,14) 'geln1p(1)        ',geln1p(1),'geln1p(2)        ',geln1p(2)
      write(6,14) 'geln2m(1)        ',geln2m(1),'geln2m(2)        ',geln2m(2)
      write(6,14) 'geln2p(1)        ',geln2p(1),'geln2p(2)        ',geln2p(2)
      write(6,14) 'geln3m(1)        ',geln3m(1),'geln3m(2)        ',geln3m(2)
      write(6,14) 'geln3p(1)        ',geln3p(1),'geln3p(2)        ',geln3p(2)
      write(6,14) 'geln4m(1)        ',geln4m(1),'geln4m(2)        ',geln4m(2)
      write(6,14) 'geln4p(1)        ',geln4p(1),'geln4p(2)        ',geln4p(2)
      write(6,14) 'gern1m(1)        ',gern1m(1),'gern1m(2)        ',gern1m(2)
      write(6,14) 'gern1p(1)        ',gern1p(1),'gern1p(2)        ',gern1p(2)
      write(6,14) 'gern2m(1)        ',gern2m(1),'gern2m(2)        ',gern2m(2)
      write(6,14) 'gern2p(1)        ',gern2p(1),'gern2p(2)        ',gern2p(2)
      write(6,14) 'gern3m(1)        ',gern3m(1),'gern3m(2)        ',gern3m(2)
      write(6,14) 'gern3p(1)        ',gern3p(1),'gern3p(2)        ',gern3p(2)
      write(6,14) 'gern4m(1)        ',gern4m(1),'gern4m(2)        ',gern4m(2)
      write(6,14) 'gern4p(1)        ',gern4p(1),'gern4p(2)        ',gern4p(2)
      write(6,14) 'gl1n1m(1)        ',gl1n1m(1),'gl1n1m(2)        ',gl1n1m(2)
      write(6,14) 'gl1n1p(1)        ',gl1n1p(1),'gl1n1p(2)        ',gl1n1p(2)
      write(6,14) 'gl1n2m(1)        ',gl1n2m(1),'gl1n2m(2)        ',gl1n2m(2)
      write(6,14) 'gl1n2p(1)        ',gl1n2p(1),'gl1n2p(2)        ',gl1n2p(2)
      write(6,14) 'gl1n3m(1)        ',gl1n3m(1),'gl1n3m(2)        ',gl1n3m(2)
      write(6,14) 'gl1n3p(1)        ',gl1n3p(1),'gl1n3p(2)        ',gl1n3p(2)
      write(6,14) 'gl1n4m(1)        ',gl1n4m(1),'gl1n4m(2)        ',gl1n4m(2)
      write(6,14) 'gl1n4p(1)        ',gl1n4p(1),'gl1n4p(2)        ',gl1n4p(2)
      write(6,14) 'gl2n1m(1)        ',gl2n1m(1),'gl2n1m(2)        ',gl2n1m(2)
      write(6,14) 'gl2n1p(1)        ',gl2n1p(1),'gl2n1p(2)        ',gl2n1p(2)
      write(6,14) 'gl2n2m(1)        ',gl2n2m(1),'gl2n2m(2)        ',gl2n2m(2)
      write(6,14) 'gl2n2p(1)        ',gl2n2p(1),'gl2n2p(2)        ',gl2n2p(2)
      write(6,14) 'gl2n3m(1)        ',gl2n3m(1),'gl2n3m(2)        ',gl2n3m(2)
      write(6,14) 'gl2n3p(1)        ',gl2n3p(1),'gl2n3p(2)        ',gl2n3p(2)
      write(6,14) 'gl2n4m(1)        ',gl2n4m(1),'gl2n4m(2)        ',gl2n4m(2)
      write(6,14) 'gl2n4p(1)        ',gl2n4p(1),'gl2n4p(2)        ',gl2n4p(2)
      write(6,14) 'gsvn1m(1)        ',gsvn1m(1),'gsvn1m(2)        ',gsvn1m(2)
      write(6,14) 'gsvn1p(1)        ',gsvn1p(1),'gsvn1p(2)        ',gsvn1p(2)
      write(6,14) 'gsvn2m(1)        ',gsvn2m(1),'gsvn2m(2)        ',gsvn2m(2)
      write(6,14) 'gsvn2p(1)        ',gsvn2p(1),'gsvn2p(2)        ',gsvn2p(2)
      write(6,14) 'gsvn3m(1)        ',gsvn3m(1),'gsvn3m(2)        ',gsvn3m(2)
      write(6,14) 'gsvn3p(1)        ',gsvn3p(1),'gsvn3p(2)        ',gsvn3p(2)
      write(6,14) 'gsvn4m(1)        ',gsvn4m(1),'gsvn4m(2)        ',gsvn4m(2)
      write(6,14) 'gsvn4p(1)        ',gsvn4p(1),'gsvn4p(2)        ',gsvn4p(2)
      write(6,14) 'gb1x1m(1)        ',gb1x1m(1),'gb1x1m(2)        ',gb1x1m(2)
      write(6,14) 'gb1x1p(1)        ',gb1x1p(1),'gb1x1p(2)        ',gb1x1p(2)
      write(6,14) 'gb1x2m(1)        ',gb1x2m(1),'gb1x2m(2)        ',gb1x2m(2)
      write(6,14) 'gb1x2p(1)        ',gb1x2p(1),'gb1x2p(2)        ',gb1x2p(2)
      write(6,14) 'gb2x1m(1)        ',gb2x1m(1),'gb2x1m(2)        ',gb2x1m(2)
      write(6,14) 'gb2x1p(1)        ',gb2x1p(1),'gb2x1p(2)        ',gb2x1p(2)
      write(6,14) 'gb2x2m(1)        ',gb2x2m(1),'gb2x2m(2)        ',gb2x2m(2)
      write(6,14) 'gb2x2p(1)        ',gb2x2p(1),'gb2x2p(2)        ',gb2x2p(2)
      write(6,14) 'gt1x1m(1)        ',gt1x1m(1),'gt1x1m(2)        ',gt1x1m(2)
      write(6,14) 'gt1x1p(1)        ',gt1x1p(1),'gt1x1p(2)        ',gt1x1p(2)
      write(6,14) 'gt1x2m(1)        ',gt1x2m(1),'gt1x2m(2)        ',gt1x2m(2)
      write(6,14) 'gt1x2p(1)        ',gt1x2p(1),'gt1x2p(2)        ',gt1x2p(2)
      write(6,14) 'gt2x1m(1)        ',gt2x1m(1),'gt2x1m(2)        ',gt2x1m(2)
      write(6,14) 'gt2x1p(1)        ',gt2x1p(1),'gt2x1p(2)        ',gt2x1p(2)
      write(6,14) 'gt2x2m(1)        ',gt2x2m(1),'gt2x2m(2)        ',gt2x2m(2)
      write(6,14) 'gt2x2p(1)        ',gt2x2p(1),'gt2x2p(2)        ',gt2x2p(2)
      write(6,14) 'gdlx1m(1)        ',gdlx1m(1),'gdlx1m(2)        ',gdlx1m(2)
      write(6,14) 'gdlx1p(1)        ',gdlx1p(1),'gdlx1p(2)        ',gdlx1p(2)
      write(6,14) 'gdlx2m(1)        ',gdlx2m(1),'gdlx2m(2)        ',gdlx2m(2)
      write(6,14) 'gdlx2p(1)        ',gdlx2p(1),'gdlx2p(2)        ',gdlx2p(2)
      write(6,14) 'gulx1m(1)        ',gulx1m(1),'gulx1m(2)        ',gulx1m(2)
      write(6,14) 'gulx1p(1)        ',gulx1p(1),'gulx1p(2)        ',gulx1p(2)
      write(6,14) 'gulx2m(1)        ',gulx2m(1),'gulx2m(2)        ',gulx2m(2)
      write(6,14) 'gulx2p(1)        ',gulx2p(1),'gulx2p(2)        ',gulx2p(2)
      write(6,14) 'gelx1m(1)        ',gelx1m(1),'gelx1m(2)        ',gelx1m(2)
      write(6,14) 'gelx1p(1)        ',gelx1p(1),'gelx1p(2)        ',gelx1p(2)
      write(6,14) 'gelx2m(1)        ',gelx2m(1),'gelx2m(2)        ',gelx2m(2)
      write(6,14) 'gelx2p(1)        ',gelx2p(1),'gelx2p(2)        ',gelx2p(2)
      write(6,14) 'gvex1m(1)        ',gvex1m(1),'gvex1m(2)        ',gvex1m(2)
      write(6,14) 'gvex1p(1)        ',gvex1p(1),'gvex1p(2)        ',gvex1p(2)
      write(6,14) 'gvex2m(1)        ',gvex2m(1),'gvex2m(2)        ',gvex2m(2)
      write(6,14) 'gvex2p(1)        ',gvex2p(1),'gvex2p(2)        ',gvex2p(2)
      write(6,14) 'gl1x1m(1)        ',gl1x1m(1),'gl1x1m(2)        ',gl1x1m(2)
      write(6,14) 'gl1x1p(1)        ',gl1x1p(1),'gl1x1p(2)        ',gl1x1p(2)
      write(6,14) 'gl1x2m(1)        ',gl1x2m(1),'gl1x2m(2)        ',gl1x2m(2)
      write(6,14) 'gl1x2p(1)        ',gl1x2p(1),'gl1x2p(2)        ',gl1x2p(2)
      write(6,14) 'gl2x1m(1)        ',gl2x1m(1),'gl2x1m(2)        ',gl2x1m(2)
      write(6,14) 'gl2x1p(1)        ',gl2x1p(1),'gl2x1p(2)        ',gl2x1p(2)
      write(6,14) 'gl2x2m(1)        ',gl2x2m(1),'gl2x2m(2)        ',gl2x2m(2)
      write(6,14) 'gl2x2p(1)        ',gl2x2p(1),'gl2x2p(2)        ',gl2x2p(2)
      write(6,14) 'gvtx1m(1)        ',gvtx1m(1),'gvtx1m(2)        ',gvtx1m(2)
      write(6,14) 'gvtx1p(1)        ',gvtx1p(1),'gvtx1p(2)        ',gvtx1p(2)
      write(6,14) 'gvtx2m(1)        ',gvtx2m(1),'gvtx2m(2)        ',gvtx2m(2)
      write(6,14) 'gvtx2p(1)        ',gvtx2p(1),'gvtx2p(2)        ',gvtx2p(2)
      write(6,14) 'gh1n11(1)        ',gh1n11(1),'gh1n11(2)        ',gh1n11(2)
      write(6,14) 'gh1n12(1)        ',gh1n12(1),'gh1n12(2)        ',gh1n12(2)
      write(6,14) 'gh1n13(1)        ',gh1n13(1),'gh1n13(2)        ',gh1n13(2)
      write(6,14) 'gh1n14(1)        ',gh1n14(1),'gh1n14(2)        ',gh1n14(2)
      write(6,14) 'gh1n21(1)        ',gh1n21(1),'gh1n21(2)        ',gh1n21(2)
      write(6,14) 'gh1n22(1)        ',gh1n22(1),'gh1n22(2)        ',gh1n22(2)
      write(6,14) 'gh1n23(1)        ',gh1n23(1),'gh1n23(2)        ',gh1n23(2)
      write(6,14) 'gh1n24(1)        ',gh1n24(1),'gh1n24(2)        ',gh1n24(2)
      write(6,14) 'gh1n31(1)        ',gh1n31(1),'gh1n31(2)        ',gh1n31(2)
      write(6,14) 'gh1n32(1)        ',gh1n32(1),'gh1n32(2)        ',gh1n32(2)
      write(6,14) 'gh1n33(1)        ',gh1n33(1),'gh1n33(2)        ',gh1n33(2)
      write(6,14) 'gh1n34(1)        ',gh1n34(1),'gh1n34(2)        ',gh1n34(2)
      write(6,14) 'gh1n41(1)        ',gh1n41(1),'gh1n41(2)        ',gh1n41(2)
      write(6,14) 'gh1n42(1)        ',gh1n42(1),'gh1n42(2)        ',gh1n42(2)
      write(6,14) 'gh1n43(1)        ',gh1n43(1),'gh1n43(2)        ',gh1n43(2)
      write(6,14) 'gh1n44(1)        ',gh1n44(1),'gh1n44(2)        ',gh1n44(2)
      write(6,14) 'gh2n11(1)        ',gh2n11(1),'gh2n11(2)        ',gh2n11(2)
      write(6,14) 'gh2n12(1)        ',gh2n12(1),'gh2n12(2)        ',gh2n12(2)
      write(6,14) 'gh2n13(1)        ',gh2n13(1),'gh2n13(2)        ',gh2n13(2)
      write(6,14) 'gh2n14(1)        ',gh2n14(1),'gh2n14(2)        ',gh2n14(2)
      write(6,14) 'gh2n21(1)        ',gh2n21(1),'gh2n21(2)        ',gh2n21(2)
      write(6,14) 'gh2n22(1)        ',gh2n22(1),'gh2n22(2)        ',gh2n22(2)
      write(6,14) 'gh2n23(1)        ',gh2n23(1),'gh2n23(2)        ',gh2n23(2)
      write(6,14) 'gh2n24(1)        ',gh2n24(1),'gh2n24(2)        ',gh2n24(2)
      write(6,14) 'gh2n31(1)        ',gh2n31(1),'gh2n31(2)        ',gh2n31(2)
      write(6,14) 'gh2n32(1)        ',gh2n32(1),'gh2n32(2)        ',gh2n32(2)
      write(6,14) 'gh2n33(1)        ',gh2n33(1),'gh2n33(2)        ',gh2n33(2)
      write(6,14) 'gh2n34(1)        ',gh2n34(1),'gh2n34(2)        ',gh2n34(2)
      write(6,14) 'gh2n41(1)        ',gh2n41(1),'gh2n41(2)        ',gh2n41(2)
      write(6,14) 'gh2n42(1)        ',gh2n42(1),'gh2n42(2)        ',gh2n42(2)
      write(6,14) 'gh2n43(1)        ',gh2n43(1),'gh2n43(2)        ',gh2n43(2)
      write(6,14) 'gh2n44(1)        ',gh2n44(1),'gh2n44(2)        ',gh2n44(2)
      write(6,14) 'gh3n11(1)        ',gh3n11(1),'gh3n11(2)        ',gh3n11(2)
      write(6,14) 'gh3n12(1)        ',gh3n12(1),'gh3n12(2)        ',gh3n12(2)
      write(6,14) 'gh3n13(1)        ',gh3n13(1),'gh3n13(2)        ',gh3n13(2)
      write(6,14) 'gh3n14(1)        ',gh3n14(1),'gh3n14(2)        ',gh3n14(2)
      write(6,14) 'gh3n21(1)        ',gh3n21(1),'gh3n21(2)        ',gh3n21(2)
      write(6,14) 'gh3n22(1)        ',gh3n22(1),'gh3n22(2)        ',gh3n22(2)
      write(6,14) 'gh3n23(1)        ',gh3n23(1),'gh3n23(2)        ',gh3n23(2)
      write(6,14) 'gh3n24(1)        ',gh3n24(1),'gh3n24(2)        ',gh3n24(2)
      write(6,14) 'gh3n31(1)        ',gh3n31(1),'gh3n31(2)        ',gh3n31(2)
      write(6,14) 'gh3n32(1)        ',gh3n32(1),'gh3n32(2)        ',gh3n32(2)
      write(6,14) 'gh3n33(1)        ',gh3n33(1),'gh3n33(2)        ',gh3n33(2)
      write(6,14) 'gh3n34(1)        ',gh3n34(1),'gh3n34(2)        ',gh3n34(2)
      write(6,14) 'gh3n41(1)        ',gh3n41(1),'gh3n41(2)        ',gh3n41(2)
      write(6,14) 'gh3n42(1)        ',gh3n42(1),'gh3n42(2)        ',gh3n42(2)
      write(6,14) 'gh3n43(1)        ',gh3n43(1),'gh3n43(2)        ',gh3n43(2)
      write(6,14) 'gh3n44(1)        ',gh3n44(1),'gh3n44(2)        ',gh3n44(2)
      write(6,14) 'gh1x11(1)        ',gh1x11(1),'gh1x11(2)        ',gh1x11(2)
      write(6,14) 'gh1x12(1)        ',gh1x12(1),'gh1x12(2)        ',gh1x12(2)
      write(6,14) 'gh1x13(1)        ',gh1x13(1),'gh1x13(2)        ',gh1x13(2)
      write(6,14) 'gh1x14(1)        ',gh1x14(1),'gh1x14(2)        ',gh1x14(2)
      write(6,14) 'gh1x21(1)        ',gh1x21(1),'gh1x21(2)        ',gh1x21(2)
      write(6,14) 'gh1x22(1)        ',gh1x22(1),'gh1x22(2)        ',gh1x22(2)
      write(6,14) 'gh1x23(1)        ',gh1x23(1),'gh1x23(2)        ',gh1x23(2)
      write(6,14) 'gh1x24(1)        ',gh1x24(1),'gh1x24(2)        ',gh1x24(2)
      write(6,14) 'gh2x11(1)        ',gh2x11(1),'gh2x11(2)        ',gh2x11(2)
      write(6,14) 'gh2x12(1)        ',gh2x12(1),'gh2x12(2)        ',gh2x12(2)
      write(6,14) 'gh2x13(1)        ',gh2x13(1),'gh2x13(2)        ',gh2x13(2)
      write(6,14) 'gh2x14(1)        ',gh2x14(1),'gh2x14(2)        ',gh2x14(2)
      write(6,14) 'gh2x21(1)        ',gh2x21(1),'gh2x21(2)        ',gh2x21(2)
      write(6,14) 'gh2x22(1)        ',gh2x22(1),'gh2x22(2)        ',gh2x22(2)
      write(6,14) 'gh2x23(1)        ',gh2x23(1),'gh2x23(2)        ',gh2x23(2)
      write(6,14) 'gh2x24(1)        ',gh2x24(1),'gh2x24(2)        ',gh2x24(2)
      write(6,14) 'gh3x11(1)        ',gh3x11(1),'gh3x11(2)        ',gh3x11(2)
      write(6,14) 'gh3x12(1)        ',gh3x12(1),'gh3x12(2)        ',gh3x12(2)
      write(6,14) 'gh3x13(1)        ',gh3x13(1),'gh3x13(2)        ',gh3x13(2)
      write(6,14) 'gh3x14(1)        ',gh3x14(1),'gh3x14(2)        ',gh3x14(2)
      write(6,14) 'gh3x21(1)        ',gh3x21(1),'gh3x21(2)        ',gh3x21(2)
      write(6,14) 'gh3x22(1)        ',gh3x22(1),'gh3x22(2)        ',gh3x22(2)
      write(6,14) 'gh3x23(1)        ',gh3x23(1),'gh3x23(2)        ',gh3x23(2)
      write(6,14) 'gh3x24(1)        ',gh3x24(1),'gh3x24(2)        ',gh3x24(2)
      write(6,14) 'ghn1x1(1)        ',ghn1x1(1),'ghn1x1(2)        ',ghn1x1(2)
      write(6,14) 'ghn2x1(1)        ',ghn2x1(1),'ghn2x1(2)        ',ghn2x1(2)
      write(6,14) 'ghn3x1(1)        ',ghn3x1(1),'ghn3x1(2)        ',ghn3x1(2)
      write(6,14) 'ghn4x1(1)        ',ghn4x1(1),'ghn4x1(2)        ',ghn4x1(2)
      write(6,14) 'ghn1x2(1)        ',ghn1x2(1),'ghn1x2(2)        ',ghn1x2(2)
      write(6,14) 'ghn2x2(1)        ',ghn2x2(1),'ghn2x2(2)        ',ghn2x2(2)
      write(6,14) 'ghn3x2(1)        ',ghn3x2(1),'ghn3x2(2)        ',ghn3x2(2)
      write(6,14) 'ghn4x2(1)        ',ghn4x2(1),'ghn4x2(2)        ',ghn4x2(2)
      write(6,14) 'ghx1n1(1)        ',ghx1n1(1),'ghx1n1(2)        ',ghx1n1(2)
      write(6,14) 'ghx1n2(1)        ',ghx1n2(1),'ghx1n2(2)        ',ghx1n2(2)
      write(6,14) 'ghx1n3(1)        ',ghx1n3(1),'ghx1n3(2)        ',ghx1n3(2)
      write(6,14) 'ghx1n4(1)        ',ghx1n4(1),'ghx1n4(2)        ',ghx1n4(2)
      write(6,14) 'ghx2n1(1)        ',ghx2n1(1),'ghx2n1(2)        ',ghx2n1(2)
      write(6,14) 'ghx2n2(1)        ',ghx2n2(1),'ghx2n2(2)        ',ghx2n2(2)
      write(6,14) 'ghx2n3(1)        ',ghx2n3(1),'ghx2n3(2)        ',ghx2n3(2)
      write(6,14) 'ghx2n4(1)        ',ghx2n4(1),'ghx2n4(2)        ',ghx2n4(2)
      write(6,*)
      write(6,*) 'MSSM_VVS:'
      write(6,14) 'gwwh1        ',gwwh1
      write(6,14) 'gwwh2        ',gwwh2
      write(6,14) 'gzzh1        ',gzzh1
      write(6,14) 'gzzh2        ',gzzh2
      write(6,*)
      write(6,*) 'MSSM_VSS:'
      write(6,14) 'gzh1h3        ',gzh1h3
      write(6,14) 'gzh2h3        ',gzh2h3
      write(6,14) 'gahchc        ',gahchc
      write(6,14) 'gzhchc        ',gzhchc
      write(6,14) 'gwh1hc        ',gwh1hc
      write(6,14) 'gwhch1        ',gwhch1
      write(6,14) 'gwh2hc        ',gwh2hc
      write(6,14) 'gwhch2        ',gwhch2
      write(6,14) 'gwh3hc        ',gwh3hc
      write(6,14) 'gwhch3        ',gwhch3
      write(6,14) 'gadldl        ',gadldl
      write(6,14) 'gaulul        ',gaulul
      write(6,14) 'gzdldl        ',gzdldl
      write(6,14) 'gzulul        ',gzulul
      write(6,14) 'gzdrdr        ',gzdrdr
      write(6,14) 'gzurur        ',gzurur
      write(6,14) 'gzb1b1        ',gzb1b1
      write(6,14) 'gzb1b2        ',gzb1b2
      write(6,14) 'gzb2b1        ',gzb2b1
      write(6,14) 'gzb2b2        ',gzb2b2
      write(6,14) 'gzt1t1        ',gzt1t1
      write(6,14) 'gzt1t2        ',gzt1t2
      write(6,14) 'gzt2t1        ',gzt2t1
      write(6,14) 'gzt2t2        ',gzt2t2
      write(6,14) 'gwqlql        ',gwqlql
      write(6,14) 'gwb1t1        ',gwb1t1
      write(6,14) 'gwb1t2        ',gwb1t2
      write(6,14) 'gwb2t1        ',gwb2t1
      write(6,14) 'gwb2t2        ',gwb2t2
      write(6,14) 'gwt1b1        ',gwt1b1
      write(6,14) 'gwt2b1        ',gwt2b1
      write(6,14) 'gwt1b2        ',gwt1b2
      write(6,14) 'gwt2b2        ',gwt2b2
      write(6,14) 'gaelel        ',gaelel
      write(6,14) 'gzelel        ',gzelel
      write(6,14) 'gzerer        ',gzerer
      write(6,14) 'gzl1l1        ',gzl1l1
      write(6,14) 'gzl1l2        ',gzl1l2
      write(6,14) 'gzl2l1        ',gzl2l1
      write(6,14) 'gzl2l2        ',gzl2l2
      write(6,14) 'gzsvsv        ',gzsvsv
      write(6,14) 'gwelve        ',gwelve
      write(6,14) 'gwl1vt        ',gwl1vt
      write(6,*)
      write(6,*) 'MSSM_VVSS:'
      write(6,14) 'gaadldl        ',gaadldl
      write(6,14) 'gaaulul        ',gaaulul
      write(6,14) 'gaaelel        ',gaaelel
      write(6,14) 'gazdldl        ',gazdldl
      write(6,14) 'gazdrdr        ',gazdrdr
      write(6,14) 'gazulul        ',gazulul
      write(6,14) 'gazurur        ',gazurur
      write(6,14) 'gazelel        ',gazelel
      write(6,14) 'gazerer        ',gazerer
      write(6,14) 'gazb1b1        ',gazb1b1
      write(6,14) 'gazb1b2        ',gazb1b2
      write(6,14) 'gazb2b1        ',gazb2b1
      write(6,14) 'gazb2b2        ',gazb2b2
      write(6,14) 'gazt1t1        ',gazt1t1
      write(6,14) 'gazt1t2        ',gazt1t2
      write(6,14) 'gazt2t1        ',gazt2t1
      write(6,14) 'gazt2t2        ',gazt2t2
      write(6,14) 'gazl1l1        ',gazl1l1
      write(6,14) 'gazl1l2        ',gazl1l2
      write(6,14) 'gazl2l1          ',gazl2l1
      write(6,14) 'gazl2l2          ',gazl2l2
      write(6,14) 'gzzdldl          ',gzzdldl
      write(6,14) 'gzzdrdr          ',gzzdrdr
      write(6,14) 'gzzulul          ',gzzulul
      write(6,14) 'gzzurur          ',gzzurur
      write(6,14) 'gzzb1b1          ',gzzb1b1
      write(6,14) 'gzzb1b2          ',gzzb1b2
      write(6,14) 'gzzb2b1          ',gzzb2b1
      write(6,14) 'gzzb2b2          ',gzzb2b2
      write(6,14) 'gzzt1t1          ',gzzt1t1
      write(6,14) 'gzzt1t2          ',gzzt1t2
      write(6,14) 'gzzt2t1          ',gzzt2t1
      write(6,14) 'gzzt2t2          ',gzzt2t2
      write(6,14) 'gzzelel          ',gzzelel
      write(6,14) 'gzzerer          ',gzzerer
      write(6,14) 'gzzveve          ',gzzveve
      write(6,14) 'gzzl1l1          ',gzzl1l1
      write(6,14) 'gzzl1l2          ',gzzl1l2
      write(6,14) 'gzzl2l1          ',gzzl2l1
      write(6,14) 'gzzl2l2          ',gzzl2l2
      write(6,14) 'gwauldl          ',gwauldl
      write(6,14) 'gwadlul          ',gwadlul
      write(6,14) 'gwab1t1          ',gwab1t1
      write(6,14) 'gwab1t2          ',gwab1t2
      write(6,14) 'gwab2t1          ',gwab2t1
      write(6,14) 'gwab2t2          ',gwab2t2
      write(6,14) 'gwat1b1          ',gwat1b1
      write(6,14) 'gwat1b2          ',gwat1b2
      write(6,14) 'gwat2b1          ',gwat2b1
      write(6,14) 'gwat2b2          ',gwat2b2
      write(6,14) 'gwaveel          ',gwaveel
      write(6,14) 'gwaelve          ',gwaelve
      write(6,14) 'gwavtl1          ',gwavtl1
      write(6,14) 'gwal1vt          ',gwal1vt
      write(6,14) 'gwavtl2          ',gwavtl2
      write(6,14) 'gwal2vt          ',gwal2vt
      write(6,14) 'gwzuldl          ',gwzuldl
      write(6,14) 'gwzdlul          ',gwzdlul
      write(6,14) 'gwzb1t1          ',gwzb1t1
      write(6,14) 'gwzb1t2          ',gwzb1t2
      write(6,14) 'gwzb2t1          ',gwzb2t1
      write(6,14) 'gwzb2t2          ',gwzb2t2
      write(6,14) 'gwzt1b1          ',gwzt1b1
      write(6,14) 'gwzt1b2          ',gwzt1b2
      write(6,14) 'gwzt2b1          ',gwzt2b1
      write(6,14) 'gwzt2b2          ',gwzt2b2
      write(6,14) 'gwzveel          ',gwzveel
      write(6,14) 'gwzelve          ',gwzelve
      write(6,14) 'gwzvtl1          ',gwzvtl1
      write(6,14) 'gwzl1vt          ',gwzl1vt
      write(6,14) 'gwzvtl2          ',gwzvtl2
      write(6,14) 'gwzl2vt          ',gwzl2vt
      write(6,14) 'gwwflfl          ',gwwflfl
      write(6,14) 'gwwb1b1          ',gwwb1b1
      write(6,14) 'gwwb1b2          ',gwwb1b2
      write(6,14) 'gwwb2b1          ',gwwb2b1
      write(6,14) 'gwwb2b2          ',gwwb2b2
      write(6,14) 'gwwt1t1          ',gwwt1t1
      write(6,14) 'gwwt1t2          ',gwwt1t2
      write(6,14) 'gwwt2t1          ',gwwt2t1
      write(6,14) 'gwwt2t2          ',gwwt2t2
      write(6,14) 'gwwl1l1          ',gwwl1l1
      write(6,14) 'gwwl1l2          ',gwwl1l2
      write(6,14) 'gwwl2l1          ',gwwl2l1
      write(6,14) 'gwwl2l2          ',gwwl2l2
      write(6,14) 'gwwh1h1          ',gwwh1h1
      write(6,14) 'gwwh2h2          ',gwwh2h2
      write(6,14) 'gwwh3h3          ',gwwh3h3
      write(6,14) 'gzzh1h1          ',gzzh1h1
      write(6,14) 'gzzh2h2          ',gzzh2h2
      write(6,14) 'gzzh3h3          ',gzzh3h3
      write(6,14) 'gaahchc          ',gaahchc
      write(6,14) 'gazhchc          ',gazhchc
      write(6,14) 'gzzhchc          ',gzzhchc
      write(6,14) 'gwwhchc          ',gwwhchc
      write(6,14) 'gwah1hc          ',gwah1hc
      write(6,14) 'gwah2hc          ',gwah2hc
      write(6,14) 'gwah3hc          ',gwah3hc
      write(6,14) 'gwahch1          ',gwahch1
      write(6,14) 'gwahch2          ',gwahch2
      write(6,14) 'gwahch3          ',gwahch3
      write(6,14) 'gwzh1hc          ',gwzh1hc
      write(6,14) 'gwzh2hc          ',gwzh2hc
      write(6,14) 'gwzh3hc          ',gwzh3hc
      write(6,14) 'gwzhch1          ',gwzhch1
      write(6,14) 'gwzhch2          ',gwzhch2
      write(6,14) 'gwzhch3          ',gwzhch3
      write(6,14) 'ggadldl          ',ggadldl
      write(6,14) 'ggaulul          ',ggaulul
      write(6,14) 'ggzdldl          ',ggzdldl
      write(6,14) 'ggzdrdr          ',ggzdrdr
      write(6,14) 'ggzulul          ',ggzulul
      write(6,14) 'ggzurur          ',ggzurur
      write(6,14) 'ggzb1b1          ',ggzb1b1
      write(6,14) 'ggzb1b2          ',ggzb1b2
      write(6,14) 'ggzb2b1          ',ggzb2b1
      write(6,14) 'ggzb2b2          ',ggzb2b2
      write(6,14) 'ggzt1t1          ',ggzt1t1
      write(6,14) 'ggzt1t2          ',ggzt1t2
      write(6,14) 'ggzt2t1          ',ggzt2t1
      write(6,14) 'ggzt2t2          ',ggzt2t2
      write(6,14) 'ggwuldl          ',ggwuldl
      write(6,14) 'ggwdlul          ',ggwdlul
      write(6,14) 'ggwt1b1          ',ggwt1b1
      write(6,14) 'ggwb1t1          ',ggwb1t1
      write(6,14) 'ggwt1b2          ',ggwt1b2
      write(6,14) 'ggwb2t1          ',ggwb2t1
      write(6,14) 'ggwt2b1          ',ggwt2b1
      write(6,14) 'ggwb1t2          ',ggwb1t2
      write(6,14) 'ggwt2b2          ',ggwt2b2
      write(6,14) 'ggwb2t2          ',ggwb2t2
      write(6,*)
      write(6,*) 'MSSM_SSS:'
      write(6,14) 'gh111          ',gh111
      write(6,14) 'gh112          ',gh112
      write(6,14) 'gh122          ',gh122
      write(6,14) 'gh222          ',gh222
      write(6,14) 'gh133          ',gh133
      write(6,14) 'gh233          ',gh233
      write(6,14) 'gh1cc          ',gh1cc
      write(6,14) 'gh2cc          ',gh2cc
      write(6,14) 'gh1ulul          ',gh1ulul
      write(6,14) 'gh1urur          ',gh1urur
      write(6,14) 'gh1dldl          ',gh1dldl
      write(6,14) 'gh1drdr          ',gh1drdr
      write(6,14) 'gh1b1b1          ',gh1b1b1
      write(6,14) 'gh1b1b2          ',gh1b1b2
      write(6,14) 'gh1b2b1          ',gh1b2b1
      write(6,14) 'gh1b2b2          ',gh1b2b2
      write(6,14) 'gh1t1t1          ',gh1t1t1
      write(6,14) 'gh1t1t2          ',gh1t1t2
      write(6,14) 'gh1t2t1          ',gh1t2t1
      write(6,14) 'gh1t2t2          ',gh1t2t2
      write(6,14) 'gh1elel          ',gh1elel
      write(6,14) 'gh1erer          ',gh1erer
      write(6,14) 'gh1veve          ',gh1veve
      write(6,14) 'gh1l1l1          ',gh1l1l1
      write(6,14) 'gh1l1l2          ',gh1l1l2
      write(6,14) 'gh1l2l1          ',gh1l2l1
      write(6,14) 'gh1l2l2          ',gh1l2l2
      write(6,14) 'gh2ulul          ',gh2ulul
      write(6,14) 'gh2urur          ',gh2urur
      write(6,14) 'gh2dldl          ',gh2dldl
      write(6,14) 'gh2drdr          ',gh2drdr
      write(6,14) 'gh2b1b1          ',gh2b1b1
      write(6,14) 'gh2b1b2          ',gh2b1b2
      write(6,14) 'gh2b2b1          ',gh2b2b1
      write(6,14) 'gh2b2b2          ',gh2b2b2
      write(6,14) 'gh2t1t1          ',gh2t1t1
      write(6,14) 'gh2t1t2          ',gh2t1t2
      write(6,14) 'gh2t2t1          ',gh2t2t1
      write(6,14) 'gh2t2t2          ',gh2t2t2
      write(6,14) 'gh2elel          ',gh2elel
      write(6,14) 'gh2erer          ',gh2erer
      write(6,14) 'gh2veve          ',gh2veve
      write(6,14) 'gh2l1l1          ',gh2l1l1
      write(6,14) 'gh2l1l2          ',gh2l1l2
      write(6,14) 'gh2l2l1          ',gh2l2l1
      write(6,14) 'gh2l2l2          ',gh2l2l2
      write(6,14) 'gh3b1b1          ',gh3b1b1
      write(6,14) 'gh3b1b2          ',gh3b1b2
      write(6,14) 'gh3b2b1          ',gh3b2b1
      write(6,14) 'gh3b2b2          ',gh3b2b2
      write(6,14) 'gh3t1t1          ',gh3t1t1
      write(6,14) 'gh3t1t2          ',gh3t1t2
      write(6,14) 'gh3t2t1          ',gh3t2t1
      write(6,14) 'gh3t2t2          ',gh3t2t2
      write(6,14) 'gh3l1l1          ',gh3l1l1
      write(6,14) 'gh3l1l2          ',gh3l1l2
      write(6,14) 'gh3l2l1          ',gh3l2l1
      write(6,14) 'gh3l2l2          ',gh3l2l2
      write(6,14) 'ghculdl          ',ghculdl
      write(6,14) 'ghcdlul          ',ghcdlul
      write(6,14) 'ghct1b1          ',ghct1b1
      write(6,14) 'ghcb1t1          ',ghcb1t1
      write(6,14) 'ghct1b2          ',ghct1b2
      write(6,14) 'ghcb2t1          ',ghcb2t1
      write(6,14) 'ghct2b1          ',ghct2b1
      write(6,14) 'ghcb1t2          ',ghcb1t2
      write(6,14) 'ghct2b2          ',ghct2b2
      write(6,14) 'ghcb2t2          ',ghcb2t2
      write(6,14) 'ghcveel          ',ghcveel
      write(6,14) 'ghcelve          ',ghcelve
      write(6,14) 'ghcl1vt          ',ghcl1vt
      write(6,14) 'ghcvtl1          ',ghcvtl1
      write(6,14) 'ghcl2vt          ',ghcl2vt
      write(6,14) 'ghcvtl2          ',ghcvtl2
      write(6,*)
      write(6,*)      

c      endif
ccc
c      write(6,*)  'Collider parameters:'
c      write(6,*)  '--------------------'
c
c      do i=1,2
c         IF(LPP(i).EQ. 0) ab(i)='e'
c         IF(LPP(i).EQ. 1) ab(i)='P'
c         IF(LPP(i).EQ.-1) ab(i)='Pb'
c      enddo
c
c      ene=2d0*dsqrt(ebeam(1)*ebeam(2))
c
c      write(6,*)  
c      write(6,*) 'Running at ',ab(1),ab(2),'  machine @ ', ene, ' GeV'
c      write(6,*) 'PDF set          ',pdlabel
c      write(6,'(1x,a12,1x,f6.4,a12,i1,a7)') 
c     &     'alpha_s(Mz)=', asmz ,' running at ', nloop , ' loops.'
c            
c      if(fixed_ren_scale) then
c         write(6,*) 'Renormalization scale fixed @ ',scale 
c      else
c         write(6,*) 'Renormalization scale set on event-by-event basis'
c      endif
c      if(fixed_fac_scale) then
c         write(6,*) 'Factorization scales  fixed @ ',
c     &   dsqrt(q2fact(1)),dsqrt(q2fact(2)) 
c      else
c         write(6,*) 'Factorization   scale set on event-by-event basis'
c      endif
c   
c      write(6,*)  
      write(6,*)  
      
      return
      end

