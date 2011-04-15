      subroutine setpara(param_name,readlha)
c***********************************************************************
c This subroutine sets up the HELAS couplings of the STANDARD MODEL.
c***********************************************************************
      implicit none

c
c local
c
      character*(*) param_name
      logical readlha
      integer i
      real*8 dum

c
c calculated couplings
c
      include 'coupl.inc'
      include 'sm_read_values.inc'
      data yt,yb,yl/3*0d0/
      
c
c     local
c
      double precision  ee, ee2, ez, ey, sw, cw, sc2, sin2w, wm
c
c constants
c
      double complex  ci
      parameter( ci = ( 0.0d0, 1.0d0 ) )
      double precision  Zero, One, Two, Three, Four, Half, Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
      parameter( Three = 3.0d0, Four = 4.0d0, Half = 0.5d0 )
      parameter( Rt2   = 1.414213562d0 )
      double precision  Pi, Fourpi
      parameter( Pi = 3.14159265358979323846d0 )
      parameter( Fourpi = Four * Pi )

c
c     susy initialization parameters
c

      real*8    unimass(20),lowmass(0:99),bw(4,4),uu(2,2),vv(2,2)
      real*8    m_t(2,2),m_b(2,2),m_l(2,2)
      real*8    width(0:99)
      save unimass,lowmass,bw,uu,vv,
     $     m_t,m_b,m_l,width

c
c Read in parameters from the SLHA-file
c      
      if(readlha) then 
         call READ_SLHA_RIP(unimass,lowmass,width,
     &        bw,uu,vv,m_t,m_b,m_l,param_name)
         G = DSQRT(Fourpi*ALFAS) ! use setting of the param_card.dat @ NLO

c
c------------------------------------------
c Start calculating the couplings for HELAS
c------------------------------------------
c

c     
c     Strong coupling
c
c     As a rule we first check if a pdf has been chosen in the    
c     run_card.dat (which has been already read at this stage).
c     If there pdfs in the initial state, then the alpha_s(MZ) used
c     is set to the corresponding value.  
  
   
c
c useful values
c
      wm = sqrt(zmass**2/Two+
     $     sqrt(zmass**4/Four-Pi/Rt2*alpha/gfermi*zmass**2))
      sin2w  = One-(wm/zmass)**2
      cw  = sqrt( One - sin2w )
      ee2 = alpha * Fourpi
      sw  = sqrt( sin2w )
      ee  = sqrt( ee2 )
      ez  = ee/(sw*cw)
      ey  = ee*(sw/cw)
      sc2 = sin2w*( One - sin2w )

c
c vector boson couplings
c
      gw   = ee/sw
      gwwa = ee
      gwwz = ee*cw/sw

c
c fermion-fermion-vector couplings
c
      gal(1) = dcmplx(  ee          , Zero )
      gal(2) = dcmplx(  ee          , Zero )
      gau(1) = dcmplx( -ee*Two/Three, Zero )
      gau(2) = dcmplx( -ee*Two/Three, Zero )
      gad(1) = dcmplx(  ee/Three    , Zero )
      gad(2) = dcmplx(  ee/Three    , Zero )

      gwf(1) = dcmplx( -ee/sqrt(Two*sin2w), Zero )
      gwf(2) = dcmplx(  Zero              , Zero )

      gzn(1) = dcmplx( -ez*Half                     , Zero )
      gzn(2) = dcmplx(  Zero                        , Zero )
      gzl(1) = dcmplx( -ez*(-Half + sin2w)          , Zero )
      gzl(2) = dcmplx( -ey                          , Zero )
      gzu(1) = dcmplx( -ez*( Half - sin2w*Two/Three), Zero )
      gzu(2) = dcmplx(  ey*Two/Three                , Zero )
      gzd(1) = dcmplx( -ez*(-Half + sin2w/Three)    , Zero )
      gzd(2) = dcmplx( -ey/Three                    , Zero )

c---------------------------------------------------------
c Set Photon Width to Zero, used by symmetry optimization
c---------------------------------------------------------
      awidth = 0d0
            
c----------------------------
c Set MSSM couplings
c----------------------------

      call INIT_SUSY(zmass,wmass,
     $     unimass,lowmass,width,
     $     bw,uu,vv,m_l,m_b,m_t,.false.,.true.)
      endif
c
c set up SUSY-QCD couplings
c
      GG(1) = -G
      GG(2) = -G     
      call INIT_SUSY_QCD(g)

      return
      end

      
c=======================================================================
c
c subroutine INIT_SUSY
c subroutine INIT_SUSY_QCD (needs INIT_SUSY for initialization)
c
c   Written by Tilman Plehn
c
c   all mixing matrices assumed to be real
c       -> negative mass eigenvalues
c
c   particles are always those with negative charge (e.g. charginos)
c
c   ordering for FFV and FFS: F_in, F_out, V/S
c   where this ordering is also reflected in the name of the coupling
c   same thing in Kaoru's writeup: F_out, F_in, S/V
c
c   arrays for fermion couplings: 1=L and 2=R
c
c=======================================================================
c
      subroutine INIT_SUSY(mzx,mwx,unimass,
     $     lowmass,lowwidth,bw,uu,vv,r_l,r_b,r_t,lmcom,lwid)
      implicit none
c
c input/output variables
c
      double precision  mzx, mwx, rmt, rmb, rml, alphaem, gf,
     &                  unimass(1:20), lowmass(0:99), lowwidth(0:99),
     &                  bw(4,4), uu(2,2), vv(2,2),
     &                  r_l(2,2), r_b(2,2), r_t(2,2)
      data rmt,rmb,rml/3*0d0/
      logical  lmcom, lwid
c
c global variables (couplings)
c
      include 'sm_read_values.inc'
      include 'coupl.inc'

      double precision  sw, sw2, cw, tw, s2w, c2w, e, gx, gz,
     &                  qe, qu, qd, t3e, t3v, t3u, t3d, rt2, pi,
     &                  r_lc(2,2), r_bc(2,2), r_tc(2,2)
      common /ewparam/  sw, sw2, cw, tw, s2w, c2w, e, gx, gz,
     &                  qe, qu, qd, t3e, t3v, t3u, t3d, rt2, pi,
     &                  r_lc, r_bc, r_tc
c
c local variables
c
      double complex  c_u(3), c_d(3), c_u_cc(3), c_d_cc(3),
     &                dum_hnn(3,4,4,2), dum_hxx(3,2,2,2),
     &                dum_hnx(4,2,2), dum_hxn(2,4,2),
     &                dum_znn(4,4,2), dum_wnx(4,2,2), dum_wxn(2,4,2)

      double complex  gbln1m(2), gbln1p(2), gbrn1m(2), gbrn1p(2),
     &                gbln2m(2), gbln2p(2), gbrn2m(2), gbrn2p(2),
     &                gbln3m(2), gbln3p(2), gbrn3m(2), gbrn3p(2),
     &                gbln4m(2), gbln4p(2), gbrn4m(2), gbrn4p(2),
     &                gtln1m(2), gtln1p(2), gtrn1m(2), gtrn1p(2),
     &                gtln2m(2), gtln2p(2), gtrn2m(2), gtrn2p(2),
     &                gtln3m(2), gtln3p(2), gtrn3m(2), gtrn3p(2),
     &                gtln4m(2), gtln4p(2), gtrn4m(2), gtrn4p(2),
     &                glln1m(2), glln1p(2), glrn1m(2), glrn1p(2),
     &                glln2m(2), glln2p(2), glrn2m(2), glrn2p(2),
     &                glln3m(2), glln3p(2), glrn3m(2), glrn3p(2),
     &                glln4m(2), glln4p(2), glrn4m(2), glrn4p(2)

      double complex  gblx1m(2), gblx1p(2), gbrx1m(2), gbrx1p(2),
     &                gblx2m(2), gblx2p(2), gbrx2m(2), gbrx2p(2),
     &                gtlx1m(2), gtlx1p(2), gtrx1m(2), gtrx1p(2),
     &                gtlx2m(2), gtlx2p(2), gtrx2m(2), gtrx2p(2),
     &                gllx1m(2), gllx1p(2), glrx1m(2), glrx1p(2),
     &                gllx2m(2), gllx2p(2), glrx2m(2), glrx2p(2)

      double complex  gh1blbl, gh1brbr, gh1blbr, gh1brbl,
     &                gh2blbl, gh2brbr, gh2blbr, gh2brbl,
     &                gh3blbl, gh3brbr, gh3blbr, gh3brbl,
     &                gh1tltl, gh1trtr, gh1tltr, gh1trtl,
     &                gh2tltl, gh2trtr, gh2tltr, gh2trtl,
     &                gh3tltl, gh3trtr, gh3tltr, gh3trtl,
     &                gh1llll, gh1lrlr, gh1lllr, gh1lrll,
     &                gh2llll, gh2lrlr, gh2lllr, gh2lrll,
     &                gh3llll, gh3lrlr, gh3lllr, gh3lrll,
     &                ghctlbl, ghctlbr,
     &                ghctrbr, ghctrbl,
     &                ghcvtll, ghcvtlr

      double precision  pz(4,4), det_l, det_b, det_t,
     &                  tgb, sa, ca, sb, cb, cbma, sbma,
     &                  saa, caa, sbb, cbb, sbpa, cbpa, ghhh,
     &                  mu, a_top, a_bot, a_tau

      double precision wm

      integer  i1, i2, ih, ii, io, in, ic
c
c fixed parameters
c
      double complex  i_unit
      parameter ( i_unit = (0d0,1d0) )

      double precision  zero, one, two, three, four, half
      parameter ( zero = 0d0, one = 1d0, two = 2d0, three = 3d0 )
      parameter ( four = 4d0, half = 0.5d0 )

      logical     ldebug
      parameter ( ldebug = .false. )
cc
      rt2 = sqrt(two)
      pi  = four * datan(one)

c
c copy mixing angles into common block for SUSY-QCD
c
      do i1=1,2
         do i2=1,2
            r_lc(i1,i2) = r_l(i1,i2)
            r_bc(i1,i2) = r_b(i1,i2)
            r_tc(i1,i2) = r_t(i1,i2)
         end do
      end do

c
c useful EW parameters
c
      alphaem = alpha
      gf  = gfermi
      wm = sqrt(zmass**2/Two+
     $     sqrt(zmass**4/Four-Pi/Rt2*alpha/gfermi*zmass**2))
      sw2  = One-(wm/zmass)**2
      sw  = sqrt(sw2)
      cw  = sqrt(one - sw2)
      tw  = sw/cw
      s2w = two*sw*cw
      c2w = cw**2 - sw**2
      e   = sqrt( four*pi*alphaem )
      gx   = e/sw
      gz  = gx/cw

      if ( ldebug ) write(6,*) ' INIT_SUSY: weak ',cw,tw,s2w,e,gx,gz
c
c compute the photino-zino mixing matrix
c
      do i1 = 1,4
         pz(i1,1) =   cw*bw(i1,1) + sw*bw(i1,2)
         pz(i1,2) = - sw*bw(i1,1) + cw*bw(i1,2)
         pz(i1,3) = bw(i1,3)
         pz(i1,4) = bw(i1,4)
      end do

c
c lepton charges
c
      qe  = -one
      t3e = -half
      t3v =  half
c
c quark charges
c
      qu  =  two/three
      t3u =  half
      qd  = -one/three
      t3d = -half
c
c set all non-SM masses (LH input)
c
      mgo = lowmass(4)
      mn1 = lowmass(5)
      mn2 = lowmass(6)
      mn3 = lowmass(7)
      mn4 = lowmass(8)
      mx1 = lowmass(9)
      mx2 = lowmass(10)

      if ( lmcom ) then
         mdl = lowmass(15)
         mdr = lowmass(15)
         mul = lowmass(15)
         mur = lowmass(15)
         msl = lowmass(15)
         msr = lowmass(15)
         mcl = lowmass(15)
         mcr = lowmass(15)
         if(ldebug) write(6,*) 'using 8-flavor average squark mass:',mdl
      else
         mdl = lowmass(11)
         mdr = lowmass(12)
         mul = lowmass(13)
         mur = lowmass(14)
         msl = lowmass(46)
         msr = lowmass(47)
         mcl = lowmass(48)
         mcr = lowmass(49)
      end if

      mb1 = lowmass(17)
      mb2 = lowmass(18)
      mt1 = lowmass(19)
      mt2 = lowmass(20)

      mel = lowmass(30)
      mer = lowmass(31)
      mve = lowmass(32)
      mml = lowmass(50)
      mmr = lowmass(51)
      mvm = lowmass(52)
      ml1 = lowmass(33)
      ml2 = lowmass(34)
      mvt = lowmass(35)

      mh1 = lowmass(41)
      mh2 = lowmass(42)
      mh3 = lowmass(40)
      mhc = lowmass(43)
c
c Higgs sector parameters
c
      tgb  = unimass(10)
      sa   = lowmass(44)
      ca   = lowmass(45)
      cb   = one/sqrt(one+tgb**2)
      sb   = tgb*cb
      cbma = cb*ca + sb*sa
      sbma = sb*ca - cb*sa
      cbpa = ca*cb - sa*sb
      sbpa = sa*cb + ca*sb
      caa  = two*ca**2 - one
      saa  = two*sa*ca
      cbb  = two*cb**2 - one
      sbb  = two*sb*cb

      c_d(1) = -sa
      c_d(2) =  ca
      c_d(3) =  sb * i_unit
      c_u(1) =  ca
      c_u(2) =  sa
      c_u(3) =  cb * i_unit

      c_u_cc(1) =   c_u(1)
      c_d_cc(1) =   c_d(1)
      c_u_cc(2) =   c_u(2)
      c_d_cc(2) =   c_d(2)
      c_u_cc(3) = - c_u(3)
      c_d_cc(3) = - c_d(3)

c -- Here the running mt,mb,ml masses are calculated from yt,yb, --- c
c -- yl given at the scale Q                                     --- c
      if(gw.ne.0)then
        rmt   = dsqrt(2d0)*wmass*sb*yt/gw
        rmb   = dsqrt(2d0)*wmass*cb*yb/gw
        rml   = dsqrt(2d0)*wmass*cb*yl/gw
      endif
c      write(*,*)'Yukawa couplings and running masses:'
c      write(*,*)'yt,yb,yl: ',yt,yb,yl
c      write(*,*)'rmt,rmb,rml: ',rmt,rmb,rml

      if(rmt.eq.0) rmt=tmass
      if(rmb.eq.0) rmb=bmass
      if(rml.eq.0) rml=lmass

c
c set all non-SM widths (SLHA input)
c
      if ( lwid ) then

c     for Majorana fermions, need right sign of width for HELAS
         wgo = sign(lowwidth( 4),mgo)
         wn1 = sign(lowwidth( 5),mn1)
         wn2 = sign(lowwidth( 6),mn2)
         wn3 = sign(lowwidth( 7),mn3)
         wn4 = sign(lowwidth( 8),mn4)
         wx1 = lowwidth( 9)
         wx2 = lowwidth(10)

         if ( lmcom ) then
            wdl = lowwidth(15)
            wdr = lowwidth(15)
            wul = lowwidth(15)
            wur = lowwidth(15)
            wsl = lowwidth(15)
            wsr = lowwidth(15)
            wcl = lowwidth(15)
            wcr = lowwidth(15)
            if(ldebug) write(6,*) 'using 8-flavor average squark width:',wdl
         else
            wdl = lowwidth(11)
            wdr = lowwidth(12)
            wul = lowwidth(13)
            wur = lowwidth(14)
            wsl = lowwidth(11)
            wsr = lowwidth(12)
            wcl = lowwidth(13)
            wcr = lowwidth(14)
         end if

         wb1 = lowwidth(17)
         wb2 = lowwidth(18)
         wt1 = lowwidth(19)
         wt2 = lowwidth(20)

         wel = lowwidth(30)
         wer = lowwidth(31)
         wve = lowwidth(32)
         wml = lowwidth(30)
         wmr = lowwidth(31)
         wvm = lowwidth(32)
         wl1 = lowwidth(33)
         wl2 = lowwidth(34)
         wvt = lowwidth(35)

c   Higgs widths not in the standard version - should overwrite w/ Hdecay

         wh1 = lowwidth(41)
         wh2 = lowwidth(42)
         wh3 = lowwidth(40)
         whc = lowwidth(43)

      else

         do ii = 0,99
            lowwidth(ii) = zero
         end do

      end if
c
c off diagonal scalar mass matrix entries in SLHA conventions
c
      det_l = r_l(1,1)*r_l(2,2)-r_l(1,2)*r_l(2,1)
      det_b = r_b(1,1)*r_b(2,2)-r_b(1,2)*r_b(2,1)
      det_t = r_t(1,1)*r_t(2,2)-r_t(1,2)*r_t(2,1)

      mu = lowmass(0)

c   this is the (-) sign between SLHA and my internal conventions

      a_tau = -lowmass(36)
      a_bot = -lowmass(21)
      a_top = -lowmass(24)

c     
c FFS Higgs couplings
c

      gh1tt(1) = - gx/two/mwx * rmt/sb * ca
      gh2tt(1) = - gx/two/mwx * rmt/sb * sa
      gh3tt(1) = - gx/two/mwx * rmt/sb * cb*i_unit
      gh1tt(2) =   gh1tt(1)
      gh2tt(2) =   gh2tt(1)
      gh3tt(2) = - gh3tt(1)

      gh1bb(1) = + gx/two/mwx * rmb/cb * sa
c      write(*,*) 'check:->',e,' ',sw,' ',gx,' ',+ gx/two/mwx,' ', rmb/cb,'  ',sa
      gh2bb(1) = - gx/two/mwx * rmb/cb * ca
      gh3bb(1) = - gx/two/mwx * rmb/cb * sb*i_unit
      gh1bb(2) =   gh1bb(1)
      gh2bb(2) =   gh2bb(1)
      gh3bb(2) = - gh3bb(1)

      gh1ll(1) = + gx/two/mwx * rml/cb * sa
      gh2ll(1) = - gx/two/mwx * rml/cb * ca
      gh3ll(1) = - gx/two/mwx * rml/cb * sb*i_unit
      gh1ll(2) =   gh1ll(1)
      gh2ll(2) =   gh2ll(1)
      gh3ll(2) = - gh3ll(1)

      ghmq(1)  = gx/rt2/mwx * rmt  /tgb
      ghmq(2)  = gx/rt2/mwx * rmb  *tgb
      ghml(1)  = zero
      ghml(2)  = gx/rt2/mwx * rml  *tgb
      ghpq(1)  = ghmq(2)
      ghpl(1)  = ghml(2)
      ghpq(2)  = ghmq(1)
      ghpl(2)  = ghml(1)
c
c VVS Higgs couplings
c
      gwwh1  = gx*mwx * sbma
      gwwh2  = gx*mwx * cbma

      gzzh1  = gx*mzx/cw * sbma
c      write(*,*) 'check:->',gx,' mzx ',mzx,' cw  ',cw,' sbma ',sbma
      gzzh2  = gx*mzx/cw * cbma

      gahchc = e
      gzhchc = e*c2w/s2w

      gwhch1 = - gx/two * ( c_d(1)*sb - c_u_cc(1)*cb )
      gwh1hc =   gwhch1
      gwhch2 = - gx/two * ( c_d(2)*sb - c_u_cc(2)*cb )
      gwh2hc =   gwhch2
      gwhch3 = - gx/two * ( c_d(3)*sb - c_u_cc(3)*cb )
      gwh3hc = - gwhch3

      gzh1h3 = - gz/two * cbma * i_unit ! experimentally det'd
      gzh2h3 =   gz/two * sbma * i_unit ! experimentally det'd
c
c SSS 3-Higgs couplings
c
      ghhh  = -three/two * mzx*gz ! to set up others only

      gh111 =  ghhh * caa * sbpa
      gh112 =  ghhh * two/three * ( saa*sbpa - caa*cbpa/two )
      gh122 = -ghhh * two/three * ( saa*cbpa + caa*sbpa/two )
      gh222 =  ghhh * caa * cbpa
      gh133 =  ghhh * cbb * sbpa / three
      gh233 = -ghhh * cbb * cbpa / three

      gh1cc =  ghhh/three * (two*cw**2*sbma + cbb*sbpa)
      gh2cc =  ghhh/three * (two*cw**2*cbma - cbb*cbpa)
c
c SSS Higgs-sfermion couplings
c n.b. sign of A_f as in SLHA (unlike lowmass array)
c
      gh1ulul = - gz*mzx*(t3u-qu*sw2) * ( c_d(1)*cb - c_u(1)*sb )
      gh1dldl = - gz*mzx*(t3d-qd*sw2) * ( c_d(1)*cb - c_u(1)*sb )
      gh1elel = - gz*mzx*(t3e-qe*sw2) * ( c_d(1)*cb - c_u(1)*sb )
      gh1veve = - gz*mzx*(t3v       ) * ( c_d(1)*cb - c_u(1)*sb )
      gh1urur =   gz*mzx*(   -qu*sw2) * ( c_d(1)*cb - c_u(1)*sb )
      gh1drdr =   gz*mzx*(   -qd*sw2) * ( c_d(1)*cb - c_u(1)*sb )
      gh1erer =   gz*mzx*(   -qe*sw2) * ( c_d(1)*cb - c_u(1)*sb )

      gh2ulul = - gz*mzx*(t3u-qu*sw2) * ( c_d(2)*cb - c_u(2)*sb )
      gh2dldl = - gz*mzx*(t3d-qd*sw2) * ( c_d(2)*cb - c_u(2)*sb )
      gh2elel = - gz*mzx*(t3e-qe*sw2) * ( c_d(2)*cb - c_u(2)*sb )
      gh2veve = - gz*mzx*(t3v       ) * ( c_d(2)*cb - c_u(2)*sb )
      gh2urur =   gz*mzx*(   -qu*sw2) * ( c_d(2)*cb - c_u(2)*sb )
      gh2drdr =   gz*mzx*(   -qd*sw2) * ( c_d(2)*cb - c_u(2)*sb )
      gh2erer =   gz*mzx*(   -qe*sw2) * ( c_d(2)*cb - c_u(2)*sb )

      gh1llll = gh1elel - gx*rml**2/mwx/cb * c_d(1)
      gh1lrlr = gh1erer - gx*rml**2/mwx/cb * c_d(1)
      gh1lllr =         - gx*rml/two/mwx/cb *(a_tau*c_d_cc(1) - mu*c_u(1))
      gh1lrll = conjg(gh1lllr)

      gh2llll = gh2elel - gx*rml**2/mwx/cb * c_d(2)
      gh2lrlr = gh2erer - gx*rml**2/mwx/cb * c_d(2)
      gh2lllr =         - gx*rml/two/mwx/cb *(a_tau*c_d_cc(2) - mu*c_u(2))
      gh2lrll = conjg(gh2lllr)

      gh3llll = zero
      gh3lrlr = zero
      gh3lllr =         - gx*rml/two/mwx/cb *(a_tau*c_d_cc(3) - mu*c_u(3))
      gh3lrll = conjg(gh3lllr)

      gh1blbl = gh1dldl - gx*rmb**2/mwx/cb * c_d(1)
      gh1brbr = gh1drdr - gx*rmb**2/mwx/cb * c_d(1)
      gh1blbr =         - gx*rmb/two/mwx/cb *(a_bot*c_d_cc(1) - mu*c_u(1))
      gh1brbl = conjg(gh1blbr)

      gh2blbl = gh2dldl - gx*rmb**2/mwx/cb * c_d(2)
      gh2brbr = gh2drdr - gx*rmb**2/mwx/cb * c_d(2)
      gh2blbr =         - gx*rmb/two/mwx/cb *(a_bot*c_d_cc(2) - mu*c_u(2))
      gh2brbl = conjg(gh2blbr)

      gh3blbl = zero
      gh3brbr = zero
      gh3blbr =         - gx*rmb/two/mwx/cb *(a_bot*c_d_cc(3) - mu*c_u(3))
      gh3brbl = conjg(gh3blbr)

      gh1tltl = gh1ulul - gx*rmt**2/mwx/sb * c_u(1)
      gh1trtr = gh1urur - gx*rmt**2/mwx/sb * c_u(1)
      gh1tltr =         - gx*rmt/two/mwx/sb *(a_top*c_u_cc(1) - mu*c_d(1))
      gh1trtl = conjg(gh1tltr)

      gh2tltl = gh2ulul - gx*rmt**2/mwx/sb * c_u(2)
      gh2trtr = gh2urur - gx*rmt**2/mwx/sb * c_u(2)
      gh2tltr =         - gx*rmt/two/mwx/sb *(a_top*c_u_cc(2) - mu*c_d(2))
      gh2trtl = conjg(gh2tltr)

      gh3tltl = zero
      gh3trtr = zero
      gh3tltr =         - gx*rmt/two/mwx/sb *(a_top*c_u_cc(3) - mu*c_d(3))
      gh3trtl = conjg(gh3tltr)

      gh1l1l1 =   r_l(1,1)**2       * gh1llll
     &          + r_l(1,2)**2       * gh1lrlr 
     &          + r_l(1,1)*r_l(1,2) * gh1lllr
     &          + r_l(1,1)*r_l(1,2) * gh1lrll
      gh1l2l2 =   r_l(2,1)**2       * gh1llll
     &          + r_l(2,2)**2       * gh1lrlr 
     &          + r_l(2,1)*r_l(2,2) * gh1lllr
     &          + r_l(2,1)*r_l(2,2) * gh1lrll
      gh1l1l2 =   r_l(1,1)*r_l(2,1) * gh1llll
     &          + r_l(1,2)*r_l(2,2) * gh1lrlr 
     &          + r_l(1,1)*r_l(2,2) * gh1lllr
     &          + r_l(1,2)*r_l(2,1) * gh1lrll
      gh1l2l1 = conjg(gh1l1l2)

      gh2l1l1 =   r_l(1,1)**2       * gh2llll
     &          + r_l(1,2)**2       * gh2lrlr 
     &          + r_l(1,1)*r_l(1,2) * gh2lllr
     &          + r_l(1,1)*r_l(1,2) * gh2lrll
      gh2l2l2 =   r_l(2,1)**2       * gh2llll
     &          + r_l(2,2)**2       * gh2lrlr 
     &          + r_l(2,1)*r_l(2,2) * gh2lllr
     &          + r_l(2,1)*r_l(2,2) * gh2lrll
      gh2l1l2 =   r_l(1,1)*r_l(2,1) * gh2llll
     &          + r_l(1,2)*r_l(2,2) * gh2lrlr 
     &          + r_l(1,1)*r_l(2,2) * gh2lllr
     &          + r_l(1,2)*r_l(2,1) * gh2lrll
      gh2l2l1 = conjg(gh2l1l2)

      gh3l1l1 =   r_l(1,1)**2       * gh3llll
     &          + r_l(1,2)**2       * gh3lrlr 
     &          + r_l(1,1)*r_l(1,2) * gh3lllr
     &          + r_l(1,1)*r_l(1,2) * gh3lrll
      gh3l2l2 =   r_l(2,1)**2       * gh3llll
     &          + r_l(2,2)**2       * gh3lrlr 
     &          + r_l(2,1)*r_l(2,2) * gh3lllr
     &          + r_l(2,1)*r_l(2,2) * gh3lrll
      gh3l1l2 =   r_l(1,1)*r_l(2,1) * gh3llll
     &          + r_l(1,2)*r_l(2,2) * gh3lrlr 
     &          + r_l(1,1)*r_l(2,2) * gh3lllr
     &          + r_l(1,2)*r_l(2,1) * gh3lrll
      gh3l2l1 = conjg(gh3l1l2)

      gh1b1b1 =   r_b(1,1)**2       * gh1blbl
     &          + r_b(1,2)**2       * gh1brbr 
     &          + r_b(1,1)*r_b(1,2) * gh1blbr
     &          + r_b(1,1)*r_b(1,2) * gh1brbl
      gh1b2b2 =   r_b(2,1)**2       * gh1blbl
     &          + r_b(2,2)**2       * gh1brbr 
     &          + r_b(2,1)*r_b(2,2) * gh1blbr
     &          + r_b(2,1)*r_b(2,2) * gh1brbl
      gh1b1b2 =   r_b(1,1)*r_b(2,1) * gh1blbl
     &          + r_b(1,2)*r_b(2,2) * gh1brbr 
     &          + r_b(1,1)*r_b(2,2) * gh1blbr
     &          + r_b(1,2)*r_b(2,1) * gh1brbl
      gh1b2b1 = conjg(gh1b1b2)

      gh2b1b1 =   r_b(1,1)**2       * gh2blbl
     &          + r_b(1,2)**2       * gh2brbr 
     &          + r_b(1,1)*r_b(1,2) * gh2blbr
     &          + r_b(1,1)*r_b(1,2) * gh2brbl
      gh2b2b2 =   r_b(2,1)**2       * gh2blbl
     &          + r_b(2,2)**2       * gh2brbr 
     &          + r_b(2,1)*r_b(2,2) * gh2blbr
     &          + r_b(2,1)*r_b(2,2) * gh2brbl
      gh2b1b2 =   r_b(1,1)*r_b(2,1) * gh2blbl
     &          + r_b(1,2)*r_b(2,2) * gh2brbr 
     &          + r_b(1,1)*r_b(2,2) * gh2blbr
     &          + r_b(1,2)*r_b(2,1) * gh2brbl
      gh2b2b1 = conjg(gh2b1b2) 

      gh3b1b1 =   r_b(1,1)**2       * gh3blbl
     &          + r_b(1,2)**2       * gh3brbr 
     &          + r_b(1,1)*r_b(1,2) * gh3blbr
     &          + r_b(1,1)*r_b(1,2) * gh3brbl
      gh3b2b2 =   r_b(2,1)**2       * gh3blbl
     &          + r_b(2,2)**2       * gh3brbr 
     &          + r_b(2,1)*r_b(2,2) * gh3blbr
     &          + r_b(2,1)*r_b(2,2) * gh3brbl
      gh3b1b2 =   r_b(1,1)*r_b(2,1) * gh3blbl
     &          + r_b(1,2)*r_b(2,2) * gh3brbr 
     &          + r_b(1,1)*r_b(2,2) * gh3blbr
     &          + r_b(1,2)*r_b(2,1) * gh3brbl
      gh3b2b1 = conjg(gh3b1b2)

      gh1t1t1 =   r_t(1,1)**2       * gh1tltl
     &          + r_t(1,2)**2       * gh1trtr 
     &          + r_t(1,1)*r_t(1,2) * gh1tltr
     &          + r_t(1,1)*r_t(1,2) * gh1trtl
      gh1t2t2 =   r_t(2,1)**2       * gh1tltl
     &          + r_t(2,2)**2       * gh1trtr 
     &          + r_t(2,1)*r_t(2,2) * gh1tltr
     &          + r_t(2,1)*r_t(2,2) * gh1trtl
      gh1t1t2 =   r_t(1,1)*r_t(2,1) * gh1tltl
     &          + r_t(1,2)*r_t(2,2) * gh1trtr 
     &          + r_t(1,1)*r_t(2,2) * gh1tltr
     &          + r_t(1,2)*r_t(2,1) * gh1trtl
      gh1t2t1 = conjg(gh1t1t2) 

      gh2t1t1 =   r_t(1,1)**2       * gh2tltl
     &          + r_t(1,2)**2       * gh2trtr 
     &          + r_t(1,1)*r_t(1,2) * gh2tltr
     &          + r_t(1,1)*r_t(1,2) * gh2trtl
      gh2t2t2 =   r_t(2,1)**2       * gh2tltl
     &          + r_t(2,2)**2       * gh2trtr 
     &          + r_t(2,1)*r_t(2,2) * gh2tltr
     &          + r_t(2,1)*r_t(2,2) * gh2trtl
      gh2t1t2 =   r_t(1,1)*r_t(2,1) * gh2tltl
     &          + r_t(1,2)*r_t(2,2) * gh2trtr 
     &          + r_t(1,1)*r_t(2,2) * gh2tltr
     &          + r_t(1,2)*r_t(2,1) * gh2trtl
      gh2t2t1 = conjg(gh2t1t2) 

      gh3t1t1 =   r_t(1,1)**2       * gh3tltl
     &          + r_t(1,2)**2       * gh3trtr 
     &          + r_t(1,1)*r_t(1,2) * gh3tltr
     &          + r_t(1,1)*r_t(1,2) * gh3trtl
      gh3t2t2 =   r_t(2,1)**2       * gh3tltl
     &          + r_t(2,2)**2       * gh3trtr 
     &          + r_t(2,1)*r_t(2,2) * gh3tltr
     &          + r_t(2,1)*r_t(2,2) * gh3trtl
      gh3t1t2 =   r_t(1,1)*r_t(2,1) * gh3tltl
     &          + r_t(1,2)*r_t(2,2) * gh3trtr 
     &          + r_t(1,1)*r_t(2,2) * gh3tltr
     &          + r_t(1,2)*r_t(2,1) * gh3trtl
      gh3t2t1 = conjg(gh3t1t2)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: SSS Higgs couplings '
         write(6,*) '    ',gh1ulul,gh1dldl
         write(6,*) '    ',gh1elel,gh1veve
         write(6,*) '    ',gh1urur,gh1drdr
         write(6,*) '    ',gh1erer
         write(6,*) '    ',gh1llll,gh1lrlr
         write(6,*) '    ',gh1lllr,gh1lrll
         write(6,*) '    ',gh1blbl,gh1brbr
         write(6,*) '    ',gh1blbr,gh1brbl
         write(6,*) '    ',gh1tltl,gh1trtr
         write(6,*) '    ',gh1tltr,gh1trtl
         write(6,*) '    ',gh2ulul,gh2dldl
         write(6,*) '    ',gh2elel,gh2veve
         write(6,*) '    ',gh2urur,gh2drdr
         write(6,*) '    ',gh2erer
         write(6,*) '    ',gh2llll,gh2lrlr
         write(6,*) '    ',gh2lllr,gh2lrll
         write(6,*) '    ',gh2blbl,gh2brbr
         write(6,*) '    ',gh2blbr,gh2brbl
         write(6,*) '    ',gh2tltl,gh2trtr
         write(6,*) '    ',gh2tltr,gh2trtl
         write(6,*) '    ',gh3llll,gh3lrlr
         write(6,*) '    ',gh3lllr,gh3lrll
         write(6,*) '    ',gh3blbl,gh3brbr
         write(6,*) '    ',gh3blbr,gh3brbl
         write(6,*) '    ',gh3tltl,gh3trtr
         write(6,*) '    ',gh3tltr,gh3trtl
         write(6,*) ' INIT_SUSY: SSS Higgs couplings mixing '
         write(6,*) '    ',gh1l1l1,gh1l2l2
         write(6,*) '    ',gh1l1l2,gh1l2l1
         write(6,*) '    ',gh1b1b1,gh1b2b2
         write(6,*) '    ',gh1b1b2,gh1b2b1
         write(6,*) '    ',gh1t1t1,gh1t2t2
         write(6,*) '    ',gh1t1t2,gh1t2t1
         write(6,*) '    ',gh2l1l1,gh2l2l2
         write(6,*) '    ',gh2l1l2,gh2l2l1
         write(6,*) '    ',gh2b1b1,gh2b2b2
         write(6,*) '    ',gh2b1b2,gh2b2b1
         write(6,*) '    ',gh2t1t1,gh2t2t2
         write(6,*) '    ',gh2t1t2,gh2t2t1
         write(6,*) '    ',gh3l1l1,gh3l2l2
         write(6,*) '    ',gh3l1l2,gh3l2l1
         write(6,*) '    ',gh3b1b1,gh3b2b2
         write(6,*) '    ',gh3b1b2,gh3b2b1
         write(6,*) '    ',gh3t1t1,gh3t2t2
         write(6,*) '    ',gh3t1t2,gh3t2t1
      end if 

      ghculdl = - gx*mwx/rt2 * sbb
      ghcveel = - gx*mwx/rt2 * sbb
      ghcdlul =   ghculdl
      ghcelve =   ghcveel

      ghctlbl = ghculdl + gx/mwx/rt2 * (rmb**2*tgb + rmt**2/tgb)
      ghctrbr =           gx/mwx/rt2 * two*rmb*rmt/sbb
      ghctlbr =           gx/mwx/rt2 * rmb * ( a_bot*tgb + mu )
      ghctrbl =           gx/mwx/rt2 * rmt * ( a_top/tgb + mu )

      ghcvtll = ghcveel + gx/mwx/rt2 * rml**2 * tgb
      ghcvtlr =           gx/mwx/rt2 * rml * ( a_tau*tgb + mu )

      ghct1b1 =  r_t(1,1)*r_b(1,1) * ghctlbl 
     &         + r_t(1,1)*r_b(1,2) * ghctlbr
     &         + r_t(1,2)*r_b(1,1) * ghctrbl 
     &         + r_t(1,2)*r_b(1,2) * ghctrbr
      ghct2b1 =  r_t(2,1)*r_b(1,1) * ghctlbl 
     &         + r_t(2,1)*r_b(1,2) * ghctlbr
     &         + r_t(2,2)*r_b(1,1) * ghctrbl 
     &         + r_t(2,2)*r_b(1,2) * ghctrbr
      ghct1b2 =  r_t(1,1)*r_b(2,1) * ghctlbl 
     &         + r_t(1,1)*r_b(2,2) * ghctlbr
     &         + r_t(1,2)*r_b(2,1) * ghctrbl 
     &         + r_t(1,2)*r_b(2,2) * ghctrbr
      ghct2b2 =  r_t(2,1)*r_b(2,1) * ghctlbl 
     &         + r_t(2,1)*r_b(2,2) * ghctlbr
     &         + r_t(2,2)*r_b(2,1) * ghctrbl 
     &         + r_t(2,2)*r_b(2,2) * ghctrbr
      ghcb1t1 = conjg(ghct1b1)
      ghcb1t2 = conjg(ghct2b1)
      ghcb2t1 = conjg(ghct1b2)
      ghcb2t2 = conjg(ghct2b2)

      ghcvtl1 = r_l(1,1) * ghcvtll + r_l(1,2) * ghcvtlr
      ghcvtl2 = r_l(2,1) * ghcvtll + r_l(2,2) * ghcvtlr
      ghcl1vt = conjg(ghcvtl1)
      ghcl2vt = conjg(ghcvtl2)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: SSS Higgs couplings charged '
         write(6,*) '    ',ghculdl,ghcveel
         write(6,*) '    ',ghctlbl,ghctrbr
         write(6,*) '    ',ghctlbr,ghctrbl
         write(6,*) '    ',ghct1b1,ghct2b2
         write(6,*) '    ',ghct1b2,ghct2b1
         write(6,*) '    ',ghcvtl1,ghcvtl2
      end if
c
c FFS Higgs couplings to weak inos (cm_out-cm_in-higgs)
c
c   NOTE: Xi- is the particle, Xi+ is the anti-particle
c
      do ih = 1,3,1             ! Higgs

         do ii = 1,2,1          ! incoming C-
            do io = 1,2,1       ! outgoing C-

               dum_hxx(ih,ii,io,1) = zero
               dum_hxx(ih,ii,io,2) = zero

               dum_hxx(ih,ii,io,1) =  c_d_cc(ih)*uu(ii,2)*vv(io,1)
     &                              + c_u_cc(ih)*uu(ii,1)*vv(io,2)
               dum_hxx(ih,ii,io,1) = -gx/rt2 * dum_hxx(ih,ii,io,1)

            end do
         end do

         do ii = 1,2,1          ! incoming C-
            do io = 1,2,1       ! outgoing C-

               if (ih.lt.3) then
                  dum_hxx(ih,ii,io,2) =  dum_hxx(ih,io,ii,1)
               else if (ih.eq.3) then
                  dum_hxx(ih,ii,io,2) = -dum_hxx(ih,io,ii,1)
               end if

            end do
         end do

      end do

      do i1 = 1,2               ! left and right

         gh1x11(i1) = dum_hxx(1,1,1,i1)
         gh1x12(i1) = dum_hxx(1,1,2,i1)
         gh1x21(i1) = dum_hxx(1,2,1,i1)
         gh1x22(i1) = dum_hxx(1,2,2,i1)
         
         gh2x11(i1) = dum_hxx(2,1,1,i1)
         gh2x12(i1) = dum_hxx(2,1,2,i1)
         gh2x21(i1) = dum_hxx(2,2,1,i1)
         gh2x22(i1) = dum_hxx(2,2,2,i1)
         
         gh3x11(i1) = dum_hxx(3,1,1,i1)
         gh3x12(i1) = dum_hxx(3,1,2,i1)
         gh3x21(i1) = dum_hxx(3,2,1,i1)
         gh3x22(i1) = dum_hxx(3,2,2,i1)

      end do

      do ih = 1,3,1             ! Higgs

         do ii = 1,4,1          ! incoming N
            do io = 1,4,1       ! outgoing N

               dum_hnn(ih,ii,io,1) = zero
               dum_hnn(ih,ii,io,2) = zero

               dum_hnn(ih,ii,io,1) =
     &              ( c_d_cc(ih)*bw(ii,3)-c_u_cc(ih)*bw(ii,4) )
     &              * ( bw(io,2)-tw*bw(io,1) )
     &            + ( c_d_cc(ih)*bw(io,3)-c_u_cc(ih)*bw(io,4) )
     &              * ( bw(ii,2)-tw*bw(ii,1) )
               dum_hnn(ih,ii,io,1) = -gx/two * dum_hnn(ih,ii,io,1)

            end do
         end do

         do ii = 1,4,1          ! incoming N
            do io = 1,4,1       ! outgoing N

               if (ih.lt.3) then
                  dum_hnn(ih,ii,io,2) =  dum_hnn(ih,io,ii,1)
               else if (ih.eq.3) then
                  dum_hnn(ih,ii,io,2) = -dum_hnn(ih,io,ii,1)
               end if

            end do
         end do

      end do

      do i1 = 1,2               ! left and right for scalar/pseudoscalar

         gh1n11(i1) = dum_hnn(1,1,1,i1)
         gh1n12(i1) = dum_hnn(1,1,2,i1)
         gh1n13(i1) = dum_hnn(1,1,3,i1)
         gh1n14(i1) = dum_hnn(1,1,4,i1)
         gh1n21(i1) = dum_hnn(1,2,1,i1)
         gh1n22(i1) = dum_hnn(1,2,2,i1)
         gh1n23(i1) = dum_hnn(1,2,3,i1)
         gh1n24(i1) = dum_hnn(1,2,4,i1)
         gh1n31(i1) = dum_hnn(1,3,1,i1)
         gh1n32(i1) = dum_hnn(1,3,2,i1)
         gh1n33(i1) = dum_hnn(1,3,3,i1)
         gh1n34(i1) = dum_hnn(1,3,4,i1)
         gh1n41(i1) = dum_hnn(1,4,1,i1)
         gh1n42(i1) = dum_hnn(1,4,2,i1)
         gh1n43(i1) = dum_hnn(1,4,3,i1)
         gh1n44(i1) = dum_hnn(1,4,4,i1)

         gh2n11(i1) = dum_hnn(2,1,1,i1)
         gh2n12(i1) = dum_hnn(2,1,2,i1)
         gh2n13(i1) = dum_hnn(2,1,3,i1)
         gh2n14(i1) = dum_hnn(2,1,4,i1)
         gh2n21(i1) = dum_hnn(2,2,1,i1)
         gh2n22(i1) = dum_hnn(2,2,2,i1)
         gh2n23(i1) = dum_hnn(2,2,3,i1)
         gh2n24(i1) = dum_hnn(2,2,4,i1)
         gh2n31(i1) = dum_hnn(2,3,1,i1)
         gh2n32(i1) = dum_hnn(2,3,2,i1)
         gh2n33(i1) = dum_hnn(2,3,3,i1)
         gh2n34(i1) = dum_hnn(2,3,4,i1)
         gh2n41(i1) = dum_hnn(2,4,1,i1)
         gh2n42(i1) = dum_hnn(2,4,2,i1)
         gh2n43(i1) = dum_hnn(2,4,3,i1)
         gh2n44(i1) = dum_hnn(2,4,4,i1)

         gh3n11(i1) = dum_hnn(3,1,1,i1)
         gh3n12(i1) = dum_hnn(3,1,2,i1)
         gh3n13(i1) = dum_hnn(3,1,3,i1)
         gh3n14(i1) = dum_hnn(3,1,4,i1)
         gh3n21(i1) = dum_hnn(3,2,1,i1)
         gh3n22(i1) = dum_hnn(3,2,2,i1)
         gh3n23(i1) = dum_hnn(3,2,3,i1)
         gh3n24(i1) = dum_hnn(3,2,4,i1)
         gh3n31(i1) = dum_hnn(3,3,1,i1)
         gh3n32(i1) = dum_hnn(3,3,2,i1)
         gh3n33(i1) = dum_hnn(3,3,3,i1)
         gh3n34(i1) = dum_hnn(3,3,4,i1)
         gh3n41(i1) = dum_hnn(3,4,1,i1)
         gh3n42(i1) = dum_hnn(3,4,2,i1)
         gh3n43(i1) = dum_hnn(3,4,3,i1)
         gh3n44(i1) = dum_hnn(3,4,4,i1)

      end do

      do in = 1,4,1             ! N
         do ic = 1,2,1          ! C-

            dum_hnx(in,ic,1) = zero
            dum_hnx(in,ic,2) = zero

            dum_hnx(in,ic,2) =
     &        bw(in,3)*uu(ic,1) - ( bw(in,2)+tw*bw(in,1) )*uu(ic,2)/rt2
            dum_hnx(in,ic,1) =
     &      - bw(in,4)*vv(ic,1) - ( bw(in,2)+tw*bw(in,1) )*vv(ic,2)/rt2

            dum_hnx(in,ic,2) = -gx*sb * dum_hnx(in,ic,2)
            dum_hnx(in,ic,1) =  gx*cb * dum_hnx(in,ic,1)

            dum_hxn(ic,in,1) = dum_hnx(in,ic,2)
            dum_hxn(ic,in,2) = dum_hnx(in,ic,1)

         end do
      end do

      do i1 = 1,2               ! left and right for scalar/pseudoscalar

         ghn1x1(i1) = dum_hnx(1,1,i1)
         ghn2x1(i1) = dum_hnx(2,1,i1)
         ghn3x1(i1) = dum_hnx(3,1,i1)
         ghn4x1(i1) = dum_hnx(4,1,i1)
         ghn1x2(i1) = dum_hnx(1,2,i1)
         ghn2x2(i1) = dum_hnx(2,2,i1)
         ghn3x2(i1) = dum_hnx(3,2,i1)
         ghn4x2(i1) = dum_hnx(4,2,i1)

         ghx1n1(i1) = dum_hxn(1,1,i1)
         ghx1n2(i1) = dum_hxn(1,2,i1)
         ghx1n3(i1) = dum_hxn(1,3,i1)
         ghx1n4(i1) = dum_hxn(1,4,i1)
         ghx2n1(i1) = dum_hxn(2,1,i1)
         ghx2n2(i1) = dum_hxn(2,2,i1)
         ghx2n3(i1) = dum_hxn(2,3,i1)
         ghx2n4(i1) = dum_hxn(2,4,i1)

      end do

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFS Higgs couplings '
         write(6,*) '    ',gh1x11,gh1x12
         write(6,*) '    ',gh1x21,gh1x22
         write(6,*) '    ',gh2x11,gh2x12
         write(6,*) '    ',gh2x21,gh2x22
         write(6,*) '    ',gh3x11,gh3x12
         write(6,*) '    ',gh3x21,gh3x22
         write(6,*) '    ',gh1n11,gh1n12,gh1n13,gh1n14
         write(6,*) '    ',gh1n21,gh1n22,gh1n23,gh1n24
         write(6,*) '    ',gh1n31,gh1n32,gh1n33,gh1n34
         write(6,*) '    ',gh1n41,gh1n42,gh1n43,gh1n44
         write(6,*) '    ',gh2n11,gh2n12,gh2n13,gh2n14
         write(6,*) '    ',gh2n21,gh2n22,gh2n23,gh2n24
         write(6,*) '    ',gh2n31,gh2n32,gh2n33,gh2n34
         write(6,*) '    ',gh2n41,gh2n42,gh2n43,gh2n44
         write(6,*) '    ',gh3n11,gh3n12,gh3n13,gh3n14
         write(6,*) '    ',gh3n21,gh3n22,gh3n23,gh3n24
         write(6,*) '    ',gh3n31,gh3n32,gh3n33,gh3n34
         write(6,*) '    ',gh3n41,gh3n42,gh3n43,gh3n44
         write(6,*) '    ',ghn1x1,ghn2x1,ghn3x1,ghn4x1
         write(6,*) '    ',ghn1x2,ghn2x2,ghn3x2,ghn4x2
         write(6,*) '    ',ghx1n1,ghx1n2,ghx1n3,ghx1n4
         write(6,*) '    ',ghx2n1,ghx2n2,ghx2n3,ghx2n4
      end if
c
c VSS couplings - non-Higgs
c
      gaelel = - e * qe
      gaulul = - e * qu
      gadldl = - e * qd
      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: VSS photon '
         write(6,*) gaelel,gaulul,gadldl
      end if

      gzelel = -gz * (t3e-qe*sw2)
      gzerer = -gz * (   -qe*sw2)
      gzsvsv = -gz * (t3v       )
      gzl1l1 = r_l(1,1)**2      * gzelel + r_l(1,2)**2      * gzerer
      gzl2l2 = r_l(2,1)**2      * gzelel + r_l(2,2)**2      * gzerer
      gzl1l2 = r_l(2,1)*r_l(1,1)* gzelel + r_l(1,2)*r_l(2,2)* gzerer
      gzl2l1 = gzl1l2

      gzdldl = -gz * (t3d-qd*sw2)
      gzdrdr = -gz * (   -qd*sw2)
      gzb1b1 = r_b(1,1)**2      * gzdldl + r_b(1,2)**2      * gzdrdr
      gzb2b2 = r_b(2,1)**2      * gzdldl + r_b(2,2)**2      * gzdrdr
      gzb1b2 = r_b(2,1)*r_b(1,1)* gzdldl + r_b(1,2)*r_b(2,2)* gzdrdr
      gzb2b1 = gzb1b2

      gzulul = -gz * (t3u-qu*sw2)
      gzurur = -gz * (   -qu*sw2)
      gzt1t1 = r_t(1,1)**2      * gzulul + r_t(1,2)**2      * gzurur
      gzt2t2 = r_t(2,1)**2      * gzulul + r_t(2,2)**2      * gzurur
      gzt1t2 = r_t(2,1)*r_t(1,1)* gzulul + r_t(1,2)*r_t(2,2)* gzurur
      gzt2t1 = gzt1t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: VSS Z couplings '
         write(6,*) '    ',gzelel,gzerer,gzsvsv
         write(6,*) '    ',gzl1l1,gzl2l2
         write(6,*) '    ',gzl1l2,gzl2l1
         write(6,*) '    ',gzdldl,gzdrdr
         write(6,*) '    ',gzb1b1,gzb2b2
         write(6,*) '    ',gzb1b2,gzb2b1
         write(6,*) '    ',gzulul,gzurur
         write(6,*) '    ',gzt1t1,gzt2t2
         write(6,*) '    ',gzt1t2,gzt2t1
      end if

      gwelve = -gx/rt2
      gwl1vt =  r_l(1,1) * gwelve
      gwl2vt =  r_l(2,1) * gwelve

      gwqlql = -gx/rt2
      gwb1t1 =  r_b(1,1)*r_t(1,1) * gwqlql
      gwb1t2 =  r_b(1,1)*r_t(2,1) * gwqlql
      gwb2t1 =  r_b(2,1)*r_t(1,1) * gwqlql
      gwb2t2 =  r_b(2,1)*r_t(2,1) * gwqlql
      gwt1b1 = gwb1t1
      gwt1b2 = gwb2t1
      gwt2b1 = gwb1t2
      gwt2b2 = gwb2t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: VSS W couplings '
         write(6,*) '    ',gwelve
         write(6,*) '    ',gwl1vt,gwl2vt
         write(6,*) '    ',gwqlql
         write(6,*) '    ',gwb1t1,gwb1t2
         write(6,*) '    ',gwb2t1,gwb2t2
      end if
c
c VVSS couplings - non-Higgs QED/QFD
c
      gaaelel = two * e**2 * qe**2
      gaadldl = two * e**2 * qd**2
      gaaulul = two * e**2 * qu**2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: VVSS non-Higgs couplings'
         write(6,*) 'eL,eL',gaaelel
         write(6,*) 'qL,qL',gaadldl,gaaulul
      end if

      gazelel = two * e*qe * gz*(t3e-qe*sw2)
      gazdldl = two * e*qd * gz*(t3d-qd*sw2)
      gazulul = two * e*qu * gz*(t3u-qu*sw2)
      gazerer = two * e*qe * gz*(   -qe*sw2)
      gazdrdr = two * e*qd * gz*(   -qd*sw2)
      gazurur = two * e*qu * gz*(   -qu*sw2)

      gazl1l1 = r_l(1,1)**2      * gazelel + r_l(1,2)**2      * gazerer
      gazl2l2 = r_l(2,1)**2      * gazelel + r_l(2,2)**2      * gazerer
      gazl1l2 = r_l(2,1)*r_l(1,1)* gazelel + r_l(1,2)*r_l(2,2)* gazerer
      gazl2l1 = gazl1l2

      gazb1b1 = r_b(1,1)**2      * gazdldl + r_b(1,2)**2      * gazdrdr
      gazb2b2 = r_b(2,1)**2      * gazdldl + r_b(2,2)**2      * gazdrdr
      gazb1b2 = r_b(2,1)*r_b(1,1)* gazdldl + r_b(1,2)*r_b(2,2)* gazdrdr
      gazb2b1 = gazb1b2

      gazt1t1 = r_t(1,1)**2      * gazulul + r_t(1,2)**2      * gazurur
      gazt2t2 = r_t(2,1)**2      * gazulul + r_t(2,2)**2      * gazurur
      gazt1t2 = r_t(2,1)*r_t(1,1)* gazulul + r_t(1,2)*r_t(2,2)* gazurur
      gazt2t1 = gazt1t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: photon-Z-s-s couplings'
         write(6,*) 'fL,fL',gazdldl,gazulul,gazelel
         write(6,*) 'fR,fR',gazdrdr,gazurur,gazerer
         write(6,*) 'f1,f1',gazb1b1,gazt1t1,gazl1l1
         write(6,*) 'f2,f2',gazb2b2,gazt2t2,gazl2l2
         write(6,*) 'f1,f2',gazb1b2,gazt1t2,gazl1l2
         write(6,*) 'f2,f1',gazb2b1,gazt2t1,gazl2l1
      end if

      gzzveve = two * gz**2 *  t3v        **2
      gzzelel = two * gz**2 * (t3e-qe*sw2)**2
      gzzdldl = two * gz**2 * (t3d-qd*sw2)**2
      gzzulul = two * gz**2 * (t3u-qu*sw2)**2
      gzzerer = two * gz**2 * (   -qe*sw2)**2
      gzzdrdr = two * gz**2 * (   -qd*sw2)**2
      gzzurur = two * gz**2 * (   -qu*sw2)**2

      gzzl1l1 = r_l(1,1)**2      * gzzelel + r_l(1,2)**2      * gzzerer
      gzzl2l2 = r_l(2,1)**2      * gzzelel + r_l(2,2)**2      * gzzerer
      gzzl1l2 = r_l(2,1)*r_l(1,1)* gzzelel + r_l(1,2)*r_l(2,2)* gzzerer
      gzzl2l1 = gzzl1l2

      gzzb1b1 = r_b(1,1)**2      * gzzdldl + r_b(1,2)**2      * gzzdrdr
      gzzb2b2 = r_b(2,1)**2      * gzzdldl + r_b(2,2)**2      * gzzdrdr
      gzzb1b2 = r_b(2,1)*r_b(1,1)* gzzdldl + r_b(1,2)*r_b(2,2)* gzzdrdr
      gzzb2b1 = gzzb1b2

      gzzt1t1 = r_t(1,1)**2      * gzzulul + r_t(1,2)**2      * gzzurur
      gzzt2t2 = r_t(2,1)**2      * gzzulul + r_t(2,2)**2      * gzzurur
      gzzt1t2 = r_t(2,1)*r_t(1,1)* gzzulul + r_t(1,2)*r_t(2,2)* gzzurur
      gzzt2t1 = gzzt1t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: Z-Z-s-s couplings '
         write(6,*) 'fL,fL',gzzdldl,gzzulul
         write(6,*) 'fL,fL',gzzelel,gzzveve
         write(6,*) 'fR,fR',gzzdrdr,gzzurur,gzzerer
         write(6,*) 'f1,f1',gzzb1b1,gzzt1t1,gzzl1l1
         write(6,*) 'f2,f2',gzzb2b2,gzzt2t2,gzzl2l2
         write(6,*) 'f1,f2',gzzb1b2,gzzt1t2,gzzl1l2
         write(6,*) 'f1,f1',gzzb2b1,gzzt2t1,gzzl2l1
      end if

      gwwflfl =  gx**2/two

      gwwl1l1 = r_l(1,1)**2       * gwwflfl
      gwwl2l2 = r_l(2,1)**2       * gwwflfl
      gwwl1l2 = r_l(1,1)*r_l(2,1) * gwwflfl
      gwwl2l1 = gwwl1l2

      gwwb1b1 = r_b(1,1)**2       * gwwflfl
      gwwb2b2 = r_b(2,1)**2       * gwwflfl
      gwwb1b2 = r_b(1,1)*r_b(2,1) * gwwflfl
      gwwb2b1 = gwwb1b2

      gwwt1t1 = r_t(1,1)**2       * gwwflfl
      gwwt2t2 = r_t(2,1)**2       * gwwflfl
      gwwt1t2 = r_t(1,1)*r_t(2,1) * gwwflfl
      gwwt2t1 = gwwt1t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: W-W-s-s couplings '
         write(6,*) 'fL,fL',gwwflfl
         write(6,*) 'f1,f1',gwwb1b1,gwwt1t1,gwwl1l1
         write(6,*) 'f2,f2',gwwb2b2,gwwt2t2,gwwl2l2
         write(6,*) 'f1,f2',gwwb1b2,gwwt1t2,gwwl1l2
         write(6,*) 'f2,f1',gwwb2b1,gwwt2t1,gwwl2l1
      end if

      gwaveel =   e* qe     *gx     /rt2
      gwaelve =   gwaveel

      gwauldl =   e*(qd+qu) *gx     /rt2
      gwadlul =   gwauldl

      gwavtl1 = r_l(1,1) * gwaveel
      gwavtl2 = r_l(2,1) * gwaveel
      gwal1vt = gwavtl1
      gwal2vt = gwavtl2
      
      gwab1t1 = r_b(1,1)*r_t(1,1) * gwadlul
      gwab1t2 = r_b(1,1)*r_t(2,1) * gwadlul
      gwab2t1 = r_b(2,1)*r_t(1,1) * gwadlul
      gwab2t2 = r_b(2,1)*r_t(2,1) * gwadlul
      gwat1b1 = gwab1t1
      gwat2b1 = gwab1t2
      gwat1b2 = gwab2t1
      gwat2b2 = gwab2t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: photon-W-s-s couplings '
         write(6,*) 'fL   ',gwauldl,gwadlul
         write(6,*) 'fL   ',gwaveel,gwaelve
         write(6,*) 'f1,f1',gwab1t1,gwat1b1
         write(6,*) 'f1,f1',gwavtl1,gwal1vt
         write(6,*) 'f2,f2',gwab2t2,gwat2b2
         write(6,*) 'f2,f2',gwavtl2,gwal2vt
         write(6,*) 'f1,f2',gwab1t2,gwat1b2
         write(6,*) 'f2,f1',gwab2t1,gwat2b1
      end if

      gwzveel =  -e* qe     *gz*sw /rt2
      gwzelve =   gwzveel

      gwzuldl =  -e*(qd+qu) *gz*sw /rt2
      gwzdlul =   gwzuldl

      gwzvtl1 = r_l(1,1) * gwzveel
      gwzvtl2 = r_l(2,1) * gwzveel
      gwzl1vt = gwzvtl1
      gwzl2vt = gwzvtl2

      gwzb1t1 = r_b(1,1)*r_t(1,1) * gwzdlul
      gwzb1t2 = r_b(1,1)*r_t(2,1) * gwzdlul
      gwzb2t1 = r_b(2,1)*r_t(1,1) * gwzdlul
      gwzb2t2 = r_b(2,1)*r_t(2,1) * gwzdlul
      gwzt1b1 = gwzb1t1
      gwzt2b1 = gwzb1t2
      gwzt1b2 = gwzb2t1
      gwzt2b2 = gwzb2t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: Z-W-s-s couplings '
         write(6,*) 'fL   ',gwzuldl,gwzdlul
         write(6,*) 'fL   ',gwzveel,gwzelve
         write(6,*) 'f1,f1',gwzb1t1,gwzt1b1
         write(6,*) 'f1,f1',gwzvtl1,gwzl1vt
         write(6,*) 'f2,f2',gwzb2t2,gwzt2b2
         write(6,*) 'f2,f2',gwzvtl2,gwzl2vt
         write(6,*) 'f1,f2',gwzb1t2,gwzt1b2
         write(6,*) 'f2,f1',gwzb2t1,gwzt2b1
      end if
c
c VVSS couplings - Higgs
c
c   NOTE: HiHi coups receive factor 2 for identical particle combos
c         ZZ,AA receive a factor 2 as well (so ZZHiHi gets factor 4)
c         So these couplings differ by these factors relative to the
c         Hagiwara/Cho MSSM note.
c
      gwwh1h1 = gx**2/two * ( c_u_cc(1)*c_u(1) + c_d_cc(1)*c_d(1) )
      gwwh2h2 = gx**2/two * ( c_u_cc(2)*c_u(2) + c_d_cc(2)*c_d(2) )
      gwwh3h3 = gx**2/two * ( c_u_cc(3)*c_u(3) + c_d_cc(3)*c_d(3) )

      if ( ldebug ) then
         write(6,*)
         write(6,*) ' INIT_SUSY: WWHH couplings '
         write(6,*) '1,1',gwwh1h1
         write(6,*) '2,2',gwwh2h2
         write(6,*) '3,3',gwwh3h3
      end if

      gzzh1h1 = gz**2/two * ( c_u_cc(1)*c_u(1) + c_d_cc(1)*c_d(1) )
      gzzh2h2 = gz**2/two * ( c_u_cc(2)*c_u(2) + c_d_cc(2)*c_d(2) )
      gzzh3h3 = gz**2/two * ( c_u_cc(3)*c_u(3) + c_d_cc(3)*c_d(3) )

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: ZZHiHj couplings '
         write(6,*) '1,1',gzzh1h1
         write(6,*) '2,2',gzzh2h2
         write(6,*) '3,3',gzzh3h3
      end if

      gwwhchc = gx**2/two
      gaahchc = two * e**2
      gzzhchc = two * gz**2 * (one/two - sw2)**2
      gazhchc = two * e*gz  * (one/two - sw2)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: VVH+H- couplings '
         write(6,*) 'W+W-',gwwhchc
         write(6,*) 'AA  ',gaahchc
         write(6,*) 'AZ  ',gazhchc
         write(6,*) 'ZZ  ',gzzhchc
      end if

      gwah1hc = -e*gx*( c_d_cc(1)*sb - c_u(1)*cb )/two
      gwah2hc = -e*gx*( c_d_cc(2)*sb - c_u(2)*cb )/two
      gwah3hc = -e*gx*( c_d_cc(3)*sb - c_u(3)*cb )/two

      gwahch1 =  conjg(gwah1hc)
      gwahch2 =  conjg(gwah2hc)
      gwahch3 =  conjg(gwah3hc)

      gwzh1hc =  gz*gx*sw2*( c_d_cc(1)*sb - c_u(1)*cb )/two
      gwzh2hc =  gz*gx*sw2*( c_d_cc(2)*sb - c_u(2)*cb )/two
      gwzh3hc =  gz*gx*sw2*( c_d_cc(3)*sb - c_u(3)*cb )/two

      gwzhch1 =  conjg(gwzh1hc)
      gwzhch2 =  conjg(gwzh2hc)
      gwzhch3 =  conjg(gwzh3hc)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: WAHiHc couplings '
         write(6,*) 'WAH1Hc',gwah1hc
         write(6,*) 'WAHcH1',gwahch1
         write(6,*) 'WAH2Hc',gwah2hc
         write(6,*) 'WAHcH2',gwahch2
         write(6,*) 'WAH3Hc',gwah3hc
         write(6,*) 'WAHcH3',gwahch3
         write(6,*) 'WZH1Hc',gwzh1hc
         write(6,*) 'WZHcH1',gwzhch1
         write(6,*) 'WZH2Hc',gwzh2hc
         write(6,*) 'WZHcH2',gwzhch2
         write(6,*) 'WZH3Hc',gwzh3hc
         write(6,*) 'WZHcH3',gwzhch3
      end if
c
c FFV couplings
c
      gzx11(1) = gz*( uu(1,1)*uu(1,1)+uu(1,2)*uu(1,2)/two - sw2 )
      gzx11(2) = gz*( vv(1,1)*vv(1,1)+vv(1,2)*vv(1,2)/two - sw2 )
      gzx12(1) = gz*( uu(1,1)*uu(2,1)+uu(1,2)*uu(2,2)/two       )
      gzx12(2) = gz*( vv(1,1)*vv(2,1)+vv(1,2)*vv(2,2)/two       )
      gzx22(1) = gz*( uu(2,1)*uu(2,1)+uu(2,2)*uu(2,2)/two - sw2 )
      gzx22(2) = gz*( vv(2,1)*vv(2,1)+vv(2,2)*vv(2,2)/two - sw2 )

C      gzx21(1) = gzx12(1) ! handled by MG
C      gzx21(2) = gzx12(2) ! handled by MG

      gax(1)   = e
      gax(2)   = e

      do ii = 1,4,1             ! incoming N
         do io = 1,4,1          ! outgoing N

            dum_znn(ii,io,1) = zero
            dum_znn(ii,io,2) = zero

            dum_znn(ii,io,1) = pz(ii,3)*pz(io,3) - pz(ii,4)*pz(io,4)
            dum_znn(ii,io,1) = -gz * dum_znn(ii,io,1)/two

            dum_znn(ii,io,2) = -dum_znn(ii,io,1)

         end do
      end do

      do i1 = 1,2               ! left and right
         gzn11(i1) = dum_znn(1,1,i1)
         gzn12(i1) = dum_znn(1,2,i1)
         gzn13(i1) = dum_znn(1,3,i1)
         gzn14(i1) = dum_znn(1,4,i1)
C         gzn21(i1) = dum_znn(2,1,i1) ! handled by MG
         gzn22(i1) = dum_znn(2,2,i1)
         gzn23(i1) = dum_znn(2,3,i1)
         gzn24(i1) = dum_znn(2,4,i1)
C         gzn31(i1) = dum_znn(3,1,i1) ! handled by MG
C         gzn32(i1) = dum_znn(3,2,i1) ! handled by MG
         gzn33(i1) = dum_znn(3,3,i1)
         gzn34(i1) = dum_znn(3,4,i1)
C         gzn41(i1) = dum_znn(4,1,i1) ! handled by MG
C         gzn42(i1) = dum_znn(4,2,i1) ! handled by MG
C         gzn43(i1) = dum_znn(4,3,i1) ! handled by MG
         gzn44(i1) = dum_znn(4,4,i1)
      end do

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFV photon/Z couplings '
         write(6,*) '    ',gax,gzx11,gzx12,gzx22
         write(6,*) '    ',gzn11,gzn12,gzn13,gzn14
         write(6,*) '    ',gzn22,gzn23,gzn24
         write(6,*) '    ',gzn33,gzn34
         write(6,*) '    ',gzn44
      end if
c
c   assume conventions are in-out-V
c
      do in = 1,4,1             ! incoming N
         do ic = 1,2,1          ! outgoing C-

            dum_wnx(in,ic,1) = zero
            dum_wnx(in,ic,2) = zero

            dum_wnx(in,ic,1) = - bw(in,3)*uu(ic,2)-rt2*bw(in,2)*uu(ic,1)
            dum_wnx(in,ic,2) =   bw(in,4)*vv(ic,2)-rt2*bw(in,2)*vv(ic,1)

            dum_wnx(in,ic,1) = gx/rt2 * dum_wnx(in,ic,1)
            dum_wnx(in,ic,2) = gx/rt2 * dum_wnx(in,ic,2)

            dum_wxn(ic,in,1) = dum_wnx(in,ic,1)
            dum_wxn(ic,in,2) = dum_wnx(in,ic,2)

         end do
      end do

      do i1 = 1,2               ! left and right

         gwn1x1(i1) = dum_wnx(1,1,i1)
         gwn2x1(i1) = dum_wnx(2,1,i1)
         gwn3x1(i1) = dum_wnx(3,1,i1)
         gwn4x1(i1) = dum_wnx(4,1,i1)
         gwn1x2(i1) = dum_wnx(1,2,i1)
         gwn2x2(i1) = dum_wnx(2,2,i1)
         gwn3x2(i1) = dum_wnx(3,2,i1)
         gwn4x2(i1) = dum_wnx(4,2,i1)

         gwx1n1(i1) = dum_wxn(1,1,i1)
         gwx1n2(i1) = dum_wxn(1,2,i1)
         gwx1n3(i1) = dum_wxn(1,3,i1)
         gwx1n4(i1) = dum_wxn(1,4,i1)
         gwx2n1(i1) = dum_wxn(2,1,i1)
         gwx2n2(i1) = dum_wxn(2,2,i1)
         gwx2n3(i1) = dum_wxn(2,3,i1)
         gwx2n4(i1) = dum_wxn(2,4,i1)

      end do

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFV W couplings '
         write(6,*) '    ',gwn1x1,gwn1x2
         write(6,*) '    ',gwn2x1,gwn2x2
         write(6,*) '    ',gwn3x1,gwn3x2
         write(6,*) '    ',gwn4x1,gwn4x2
         write(6,*) '    ',gwx1n1,gwx2n1
         write(6,*) '    ',gwx1n2,gwx2n2
         write(6,*) '    ',gwx1n3,gwx2n3
         write(6,*) '    ',gwx1n4,gwx2n4
      end if
c
c FFS couplings - non-Higgs
c
C      geln1m(1) = rt2*( gx*sw*qe*pz(1,1) + gx*(t3e-qe*sw2)/cw*pz(1,2)) ! legacy code

      geln1m(1) = -rt2*gx*( t3e*bw(1,2)+bw(1,1)*(qe-t3e)*tw )
      geln1m(2) = zero
      gern1m(1) = zero
      gern1m(2) =  rt2*gx*(             bw(1,1)*(qe    )*tw )

      geln2m(1) = -rt2*gx*( t3e*bw(2,2)+bw(2,1)*(qe-t3e)*tw )
      geln2m(2) = zero
      gern2m(1) = zero
      gern2m(2) =  rt2*gx*(             bw(2,1)*(qe    )*tw )

      geln3m(1) = -rt2*gx*( t3e*bw(3,2)+bw(3,1)*(qe-t3e)*tw )
      geln3m(2) = zero
      gern3m(1) = zero
      gern3m(2) =  rt2*gx*(             bw(3,1)*(qe    )*tw )

      geln4m(1) = -rt2*gx*( t3e*bw(4,2)+bw(4,1)*(qe-t3e)*tw )
      geln4m(2) = zero
      gern4m(1) = zero
      gern4m(2) =  rt2*gx*(             bw(4,1)*(qe    )*tw )
c
c   use general symmetry for massless leptons
c
      geln1p(1) = geln1m(2)
      geln1p(2) = geln1m(1)
      gern1p(1) = gern1m(2)
      gern1p(2) = gern1m(1)

      geln2p(1) = geln2m(2)
      geln2p(2) = geln2m(1)
      gern2p(1) = gern2m(2)
      gern2p(2) = gern2m(1)

      geln3p(1) = geln3m(2)
      geln3p(2) = geln3m(1)
      gern3p(1) = gern3m(2)
      gern3p(2) = gern3m(1)

      geln4p(1) = geln4m(2)
      geln4p(2) = geln4m(1)
      gern4p(1) = gern4m(2)
      gern4p(2) = gern4m(1)

      glln1m(1) = -rt2*gx*( t3e*bw(1,2)+bw(1,1)*(qe-t3e)*tw )
      glln1m(2) = -rt2*gx*  rml/(two*mwx*cb)*bw(1,3)
      glrn1m(1) = -rt2*gx*  rml/(two*mwx*cb)*bw(1,3)
      glrn1m(2) =  rt2*gx*(             bw(1,1)*(qe    )*tw )

      glln2m(1) = -rt2*gx*( t3e*bw(2,2)+bw(2,1)*(qe-t3e)*tw )
      glln2m(2) = -rt2*gx*  rml/(two*mwx*cb)*bw(2,3)
      glrn2m(1) = -rt2*gx*  rml/(two*mwx*cb)*bw(2,3)
      glrn2m(2) =  rt2*gx*(             bw(2,1)*(qe    )*tw )

      glln3m(1) = -rt2*gx*( t3e*bw(3,2)+bw(3,1)*(qe-t3e)*tw )
      glln3m(2) = -rt2*gx*  rml/(two*mwx*cb)*bw(3,3)
      glrn3m(1) = -rt2*gx*  rml/(two*mwx*cb)*bw(3,3)
      glrn3m(2) =  rt2*gx*(             bw(3,1)*(qe    )*tw )

      glln4m(1) = -rt2*gx*( t3e*bw(4,2)+bw(4,1)*(qe-t3e)*tw )
      glln4m(2) = -rt2*gx*  rml/(two*mwx*cb)*bw(4,3)
      glrn4m(1) = -rt2*gx*  rml/(two*mwx*cb)*bw(4,3)
      glrn4m(2) =  rt2*gx*(             bw(4,1)*(qe    )*tw )

      gl1n1m(1) = r_l(1,1) * glln1m(1) + r_l(1,2) * glrn1m(1)
      gl2n1m(1) = r_l(2,1) * glln1m(1) + r_l(2,2) * glrn1m(1)
      gl1n1m(2) = r_l(1,1) * glln1m(2) + r_l(1,2) * glrn1m(2)
      gl2n1m(2) = r_l(2,1) * glln1m(2) + r_l(2,2) * glrn1m(2)

      gl1n2m(1) = r_l(1,1) * glln2m(1) + r_l(1,2) * glrn2m(1)
      gl2n2m(1) = r_l(2,1) * glln2m(1) + r_l(2,2) * glrn2m(1)
      gl1n2m(2) = r_l(1,1) * glln2m(2) + r_l(1,2) * glrn2m(2)
      gl2n2m(2) = r_l(2,1) * glln2m(2) + r_l(2,2) * glrn2m(2)

      gl1n3m(1) = r_l(1,1) * glln3m(1) + r_l(1,2) * glrn3m(1)
      gl2n3m(1) = r_l(2,1) * glln3m(1) + r_l(2,2) * glrn3m(1)
      gl1n3m(2) = r_l(1,1) * glln3m(2) + r_l(1,2) * glrn3m(2)
      gl2n3m(2) = r_l(2,1) * glln3m(2) + r_l(2,2) * glrn3m(2)

      gl1n4m(1) = r_l(1,1) * glln4m(1) + r_l(1,2) * glrn4m(1)
      gl2n4m(1) = r_l(2,1) * glln4m(1) + r_l(2,2) * glrn4m(1)
      gl1n4m(2) = r_l(1,1) * glln4m(2) + r_l(1,2) * glrn4m(2)
      gl2n4m(2) = r_l(2,1) * glln4m(2) + r_l(2,2) * glrn4m(2)

      glln1p(1) = glln1m(2)
      glln1p(2) = glln1m(1)
      glrn1p(1) = glrn1m(2)
      glrn1p(2) = glrn1m(1)

      glln2p(1) = glln2m(2)
      glln2p(2) = glln2m(1)
      glrn2p(1) = glrn2m(2)
      glrn2p(2) = glrn2m(1)

      glln3p(1) = glln3m(2)
      glln3p(2) = glln3m(1)
      glrn3p(1) = glrn3m(2)
      glrn3p(2) = glrn3m(1)

      glln4p(1) = glln4m(2)
      glln4p(2) = glln4m(1)
      glrn4p(1) = glrn4m(2)
      glrn4p(2) = glrn4m(1)

      gl1n1p(1) = gl1n1m(2)
      gl1n1p(2) = gl1n1m(1)
      gl2n1p(1) = gl2n1m(2)
      gl2n1p(2) = gl2n1m(1)

      gl1n2p(1) = gl1n2m(2)
      gl1n2p(2) = gl1n2m(1)
      gl2n2p(1) = gl2n2m(2)
      gl2n2p(2) = gl2n2m(1)

      gl1n3p(1) = gl1n3m(2)
      gl1n3p(2) = gl1n3m(1)
      gl2n3p(1) = gl2n3m(2)
      gl2n3p(2) = gl2n3m(1)

      gl1n4p(1) = gl1n4m(2)
      gl1n4p(2) = gl1n4m(1)
      gl2n4p(1) = gl2n4m(2)
      gl2n4p(2) = gl2n4m(1)

      gsvn1m(1) = -rt2*gx*( t3v*bw(1,2)+bw(1,1)*(  -t3v)*tw )
      gsvn1m(2) = zero

      gsvn2m(1) = -rt2*gx*( t3v*bw(2,2)+bw(2,1)*(  -t3v)*tw )
      gsvn2m(2) = zero

      gsvn3m(1) = -rt2*gx*( t3v*bw(3,2)+bw(3,1)*(  -t3v)*tw )
      gsvn3m(2) = zero

      gsvn4m(1) = -rt2*gx*( t3v*bw(4,2)+bw(4,1)*(  -t3v)*tw )
      gsvn4m(2) = zero

      gsvn1p(1) = gsvn1m(2)
      gsvn1p(2) = gsvn1m(1)

      gsvn2p(1) = gsvn2m(2)
      gsvn2p(2) = gsvn2m(1)

      gsvn3p(1) = gsvn3m(2)
      gsvn3p(2) = gsvn3m(1)

      gsvn4p(1) = gsvn4m(2)
      gsvn4p(2) = gsvn4m(1)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFS neutralino couplings 1'
         write(6,*) '    ',geln1p,gern1p,gsvn1p
         write(6,*) '    ',geln1m,gern1m,gsvn1m
         write(6,*) '    ',geln2p,gern2p,gsvn2p
         write(6,*) '    ',geln2m,gern2m,gsvn2m
         write(6,*) '    ',geln3p,gern3p,gsvn3p
         write(6,*) '    ',geln3m,gern3m,gsvn3m
         write(6,*) '    ',geln4p,gern4p,gsvn4p
         write(6,*) '    ',geln4m,gern4m,gsvn4m
         write(6,*) '    ',glln1p,glrn1p
         write(6,*) '    ',glln1m,glrn1m
         write(6,*) '    ',glln2p,glrn2p
         write(6,*) '    ',glln2m,glrn2m
         write(6,*) '    ',glln3p,glrn3p
         write(6,*) '    ',glln3m,glrn3m
         write(6,*) '    ',glln4p,glrn4p
         write(6,*) '    ',glln4m,glrn4m
         write(6,*) ' INIT_SUSY: FFS neutralino couplings 1 mixing'
         write(6,*) '    ',gl1n1p,gl2n1p
         write(6,*) '    ',gl1n1m,gl2n1m
         write(6,*) '    ',gl1n2p,gl2n2p
         write(6,*) '    ',gl1n2m,gl2n2m
         write(6,*) '    ',gl1n3p,gl2n3p
         write(6,*) '    ',gl1n3m,gl2n3m
         write(6,*) '    ',gl1n4p,gl2n4p
         write(6,*) '    ',gl1n4m,gl2n4m
      end if
c
c   the same for neutralino-quark-squark
c      relative sign (-rt2) between thesis and kaoru
c      note the additional (-) for gaugino part sqL->sqR
c
      gdln1m(1) = -rt2*gx*( t3d*bw(1,2)+bw(1,1)*(qd-t3d)*tw )
      gdln1m(2) = zero
      gdrn1m(1) = zero
      gdrn1m(2) =  rt2*gx*(             bw(1,1)*(qd    )*tw )

      gdln2m(1) = -rt2*gx*( t3d*bw(2,2)+bw(2,1)*(qd-t3d)*tw )
      gdln2m(2) = zero
      gdrn2m(1) = zero
      gdrn2m(2) =  rt2*gx*(             bw(2,1)*(qd    )*tw )

      gdln3m(1) = -rt2*gx*( t3d*bw(3,2)+bw(3,1)*(qd-t3d)*tw )
      gdln3m(2) = zero
      gdrn3m(1) = zero
      gdrn3m(2) =  rt2*gx*(             bw(3,1)*(qd    )*tw )

      gdln4m(1) = -rt2*gx*( t3d*bw(4,2)+bw(4,1)*(qd-t3d)*tw )
      gdln4m(2) = zero
      gdrn4m(1) = zero
      gdrn4m(2) =  rt2*gx*(             bw(4,1)*(qd    )*tw )

      gdln1p(1) = gdln1m(2)
      gdln1p(2) = gdln1m(1)
      gdrn1p(1) = gdrn1m(2)
      gdrn1p(2) = gdrn1m(1)

      gdln2p(1) = gdln2m(2)
      gdln2p(2) = gdln2m(1)
      gdrn2p(1) = gdrn2m(2)
      gdrn2p(2) = gdrn2m(1)

      gdln3p(1) = gdln3m(2)
      gdln3p(2) = gdln3m(1)
      gdrn3p(1) = gdrn3m(2)
      gdrn3p(2) = gdrn3m(1)

      gdln4p(1) = gdln4m(2)
      gdln4p(2) = gdln4m(1)
      gdrn4p(1) = gdrn4m(2)
      gdrn4p(2) = gdrn4m(1)

      gbln1m(1) = -rt2*gx*( t3d*bw(1,2)+bw(1,1)*(qd-t3d)*tw )
      gbln1m(2) = -rt2*gx*  rmb/(two*mwx*cb)*bw(1,3)
      gbrn1m(1) = -rt2*gx*  rmb/(two*mwx*cb)*bw(1,3)
      gbrn1m(2) =  rt2*gx*(             bw(1,1)*(qd    )*tw )

      gbln2m(1) = -rt2*gx*( t3d*bw(2,2)+bw(2,1)*(qd-t3d)*tw )
      gbln2m(2) = -rt2*gx*  rmb/(two*mwx*cb)*bw(2,3)
      gbrn2m(1) = -rt2*gx*  rmb/(two*mwx*cb)*bw(2,3)
      gbrn2m(2) =  rt2*gx*(             bw(2,1)*(qd    )*tw )

      gbln3m(1) = -rt2*gx*( t3d*bw(3,2)+bw(3,1)*(qd-t3d)*tw )
      gbln3m(2) = -rt2*gx*  rmb/(two*mwx*cb)*bw(3,3)
      gbrn3m(1) = -rt2*gx*  rmb/(two*mwx*cb)*bw(3,3)
      gbrn3m(2) =  rt2*gx*(             bw(3,1)*(qd    )*tw )

      gbln4m(1) = -rt2*gx*( t3d*bw(4,2)+bw(4,1)*(qd-t3d)*tw )
      gbln4m(2) = -rt2*gx*  rmb/(two*mwx*cb)*bw(4,3)
      gbrn4m(1) = -rt2*gx*  rmb/(two*mwx*cb)*bw(4,3)
      gbrn4m(2) =  rt2*gx*(             bw(4,1)*(qd    )*tw )

      gb1n1m(1) = r_b(1,1) * gbln1m(1) + r_b(1,2) * gbrn1m(1)
      gb2n1m(1) = r_b(2,1) * gbln1m(1) + r_b(2,2) * gbrn1m(1)
      gb1n1m(2) = r_b(1,1) * gbln1m(2) + r_b(1,2) * gbrn1m(2)
      gb2n1m(2) = r_b(2,1) * gbln1m(2) + r_b(2,2) * gbrn1m(2)

      gb1n2m(1) = r_b(1,1) * gbln2m(1) + r_b(1,2) * gbrn2m(1)
      gb2n2m(1) = r_b(2,1) * gbln2m(1) + r_b(2,2) * gbrn2m(1)
      gb1n2m(2) = r_b(1,1) * gbln2m(2) + r_b(1,2) * gbrn2m(2)
      gb2n2m(2) = r_b(2,1) * gbln2m(2) + r_b(2,2) * gbrn2m(2)

      gb1n3m(1) = r_b(1,1) * gbln3m(1) + r_b(1,2) * gbrn3m(1)
      gb2n3m(1) = r_b(2,1) * gbln3m(1) + r_b(2,2) * gbrn3m(1)
      gb1n3m(2) = r_b(1,1) * gbln3m(2) + r_b(1,2) * gbrn3m(2)
      gb2n3m(2) = r_b(2,1) * gbln3m(2) + r_b(2,2) * gbrn3m(2)

      gb1n4m(1) = r_b(1,1) * gbln4m(1) + r_b(1,2) * gbrn4m(1)
      gb2n4m(1) = r_b(2,1) * gbln4m(1) + r_b(2,2) * gbrn4m(1)
      gb1n4m(2) = r_b(1,1) * gbln4m(2) + r_b(1,2) * gbrn4m(2)
      gb2n4m(2) = r_b(2,1) * gbln4m(2) + r_b(2,2) * gbrn4m(2)

      gbln1p(1) = gbln1m(2)
      gbln1p(2) = gbln1m(1)
      gbrn1p(1) = gbrn1m(2)
      gbrn1p(2) = gbrn1m(1)

      gbln2p(1) = gbln2m(2)
      gbln2p(2) = gbln2m(1)
      gbrn2p(1) = gbrn2m(2)
      gbrn2p(2) = gbrn2m(1)

      gbln3p(1) = gbln3m(2)
      gbln3p(2) = gbln3m(1)
      gbrn3p(1) = gbrn3m(2)
      gbrn3p(2) = gbrn3m(1)

      gbln4p(1) = gbln4m(2)
      gbln4p(2) = gbln4m(1)
      gbrn4p(1) = gbrn4m(2)
      gbrn4p(2) = gbrn4m(1)

      gb1n1p(1) = gb1n1m(2)
      gb1n1p(2) = gb1n1m(1)
      gb2n1p(1) = gb2n1m(2)
      gb2n1p(2) = gb2n1m(1)

      gb1n2p(1) = gb1n2m(2)
      gb1n2p(2) = gb1n2m(1)
      gb2n2p(1) = gb2n2m(2)
      gb2n2p(2) = gb2n2m(1)

      gb1n3p(1) = gb1n3m(2)
      gb1n3p(2) = gb1n3m(1)
      gb2n3p(1) = gb2n3m(2)
      gb2n3p(2) = gb2n3m(1)

      gb1n4p(1) = gb1n4m(2)
      gb1n4p(2) = gb1n4m(1)
      gb2n4p(1) = gb2n4m(2)
      gb2n4p(2) = gb2n4m(1)

      guln1m(1) = -rt2*gx*( t3u*bw(1,2)+bw(1,1)*(qu-t3u)*tw )
      guln1m(2) = zero
      gurn1m(1) = zero
      gurn1m(2) =  rt2*gx*(             bw(1,1)*(qu    )*tw )

      guln2m(1) = -rt2*gx*( t3u*bw(2,2)+bw(2,1)*(qu-t3u)*tw )
      guln2m(2) = zero
      gurn2m(1) = zero
      gurn2m(2) =  rt2*gx*(             bw(2,1)*(qu    )*tw )

      guln3m(1) = -rt2*gx*( t3u*bw(3,2)+bw(3,1)*(qu-t3u)*tw )
      guln3m(2) = zero
      gurn3m(1) = zero
      gurn3m(2) =  rt2*gx*(             bw(3,1)*(qu    )*tw )

      guln4m(1) = -rt2*gx*( t3u*bw(4,2)+bw(4,1)*(qu-t3u)*tw )
      guln4m(2) = zero
      gurn4m(1) = zero
      gurn4m(2) =  rt2*gx*(             bw(4,1)*(qu    )*tw )

      guln1p(1) = guln1m(2)
      guln1p(2) = guln1m(1)
      gurn1p(1) = gurn1m(2)
      gurn1p(2) = gurn1m(1)

      guln2p(1) = guln2m(2)
      guln2p(2) = guln2m(1)
      gurn2p(1) = gurn2m(2)
      gurn2p(2) = gurn2m(1)

      guln3p(1) = guln3m(2)
      guln3p(2) = guln3m(1)
      gurn3p(1) = gurn3m(2)
      gurn3p(2) = gurn3m(1)

      guln4p(1) = guln4m(2)
      guln4p(2) = guln4m(1)
      gurn4p(1) = gurn4m(2)
      gurn4p(2) = gurn4m(1)

      gtln1m(1) = -rt2*gx*( t3u*bw(1,2)+bw(1,1)*(qu-t3u)*tw )
      gtln1m(2) = -rt2*gx*  rmt/(two*mwx*sb)*bw(1,4)
      gtrn1m(1) = -rt2*gx*  rmt/(two*mwx*sb)*bw(1,4)
      gtrn1m(2) =  rt2*gx*(             bw(1,1)*(qu    )*tw )

      gtln2m(1) = -rt2*gx*( t3u*bw(2,2)+bw(2,1)*(qu-t3u)*tw )
      gtln2m(2) = -rt2*gx*  rmt/(two*mwx*sb)*bw(2,4)
      gtrn2m(1) = -rt2*gx*  rmt/(two*mwx*sb)*bw(2,4)
      gtrn2m(2) =  rt2*gx*(             bw(2,1)*(qu    )*tw )

      gtln3m(1) = -rt2*gx*( t3u*bw(3,2)+bw(3,1)*(qu-t3u)*tw )
      gtln3m(2) = -rt2*gx*  rmt/(two*mwx*sb)*bw(3,4)
      gtrn3m(1) = -rt2*gx*  rmt/(two*mwx*sb)*bw(3,4)
      gtrn3m(2) =  rt2*gx*(             bw(3,1)*(qu    )*tw )

      gtln4m(1) = -rt2*gx*( t3u*bw(4,2)+bw(4,1)*(qu-t3u)*tw )
      gtln4m(2) = -rt2*gx*  rmt/(two*mwx*sb)*bw(4,4)
      gtrn4m(1) = -rt2*gx*  rmt/(two*mwx*sb)*bw(4,4)
      gtrn4m(2) =  rt2*gx*(             bw(4,1)*(qu    )*tw )

      gt1n1m(1) = r_t(1,1) * gtln1m(1) + r_t(1,2) * gtrn1m(1)
      gt2n1m(1) = r_t(2,1) * gtln1m(1) + r_t(2,2) * gtrn1m(1)
      gt1n1m(2) = r_t(1,1) * gtln1m(2) + r_t(1,2) * gtrn1m(2)
      gt2n1m(2) = r_t(2,1) * gtln1m(2) + r_t(2,2) * gtrn1m(2)

      gt1n2m(1) = r_t(1,1) * gtln2m(1) + r_t(1,2) * gtrn2m(1)
      gt2n2m(1) = r_t(2,1) * gtln2m(1) + r_t(2,2) * gtrn2m(1)
      gt1n2m(2) = r_t(1,1) * gtln2m(2) + r_t(1,2) * gtrn2m(2)
      gt2n2m(2) = r_t(2,1) * gtln2m(2) + r_t(2,2) * gtrn2m(2)

      gt1n3m(1) = r_t(1,1) * gtln3m(1) + r_t(1,2) * gtrn3m(1)
      gt2n3m(1) = r_t(2,1) * gtln3m(1) + r_t(2,2) * gtrn3m(1)
      gt1n3m(2) = r_t(1,1) * gtln3m(2) + r_t(1,2) * gtrn3m(2)
      gt2n3m(2) = r_t(2,1) * gtln3m(2) + r_t(2,2) * gtrn3m(2)

      gt1n4m(1) = r_t(1,1) * gtln4m(1) + r_t(1,2) * gtrn4m(1)
      gt2n4m(1) = r_t(2,1) * gtln4m(1) + r_t(2,2) * gtrn4m(1)
      gt1n4m(2) = r_t(1,1) * gtln4m(2) + r_t(1,2) * gtrn4m(2)
      gt2n4m(2) = r_t(2,1) * gtln4m(2) + r_t(2,2) * gtrn4m(2)

      gtln1p(1) = gtln1m(2)
      gtln1p(2) = gtln1m(1)
      gtrn1p(1) = gtrn1m(2)
      gtrn1p(2) = gtrn1m(1)

      gtln2p(1) = gtln2m(2)
      gtln2p(2) = gtln2m(1)
      gtrn2p(1) = gtrn2m(2)
      gtrn2p(2) = gtrn2m(1)

      gtln3p(1) = gtln3m(2)
      gtln3p(2) = gtln3m(1)
      gtrn3p(1) = gtrn3m(2)
      gtrn3p(2) = gtrn3m(1)

      gtln4p(1) = gtln4m(2)
      gtln4p(2) = gtln4m(1)
      gtrn4p(1) = gtrn4m(2)
      gtrn4p(2) = gtrn4m(1)

      gt1n1p(1) = gt1n1m(2)
      gt1n1p(2) = gt1n1m(1)
      gt2n1p(1) = gt2n1m(2)
      gt2n1p(2) = gt2n1m(1)

      gt1n2p(1) = gt1n2m(2)
      gt1n2p(2) = gt1n2m(1)
      gt2n2p(1) = gt2n2m(2)
      gt2n2p(2) = gt2n2m(1)

      gt1n3p(1) = gt1n3m(2)
      gt1n3p(2) = gt1n3m(1)
      gt2n3p(1) = gt2n3m(2)
      gt2n3p(2) = gt2n3m(1)

      gt1n4p(1) = gt1n4m(2)
      gt1n4p(2) = gt1n4m(1)
      gt2n4p(1) = gt2n4m(2)
      gt2n4p(2) = gt2n4m(1)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFS neutralino couplings 2'
         write(6,*) '    ',gdln1p,gdrn1p
         write(6,*) '    ',gdln1m,gdrn1m
         write(6,*) '    ',gdln2p,gdrn2p
         write(6,*) '    ',gdln2m,gdrn2m
         write(6,*) '    ',gdln3p,gdrn3p
         write(6,*) '    ',gdln3m,gdrn3m
         write(6,*) '    ',gdln4p,gdrn4p
         write(6,*) '    ',gdln4m,gdrn4m
         write(6,*) '    ',gbln1p,gbrn1p
         write(6,*) '    ',gbln1m,gbrn1m
         write(6,*) '    ',gbln2p,gbrn2p
         write(6,*) '    ',gbln2m,gbrn2m
         write(6,*) '    ',gbln3p,gbrn3p
         write(6,*) '    ',gbln3m,gbrn3m
         write(6,*) '    ',gbln4p,gbrn4p
         write(6,*) '    ',gbln4m,gbrn4m
         write(6,*) '    ',guln1p,gurn1p
         write(6,*) '    ',guln1m,gurn1m
         write(6,*) '    ',guln2p,gurn2p
         write(6,*) '    ',guln2m,gurn2m
         write(6,*) '    ',guln3p,gurn3p
         write(6,*) '    ',guln3m,gurn3m
         write(6,*) '    ',guln4p,gurn4p
         write(6,*) '    ',gtln4m,gtrn4m
         write(6,*) '    ',gtln1p,gtrn1p
         write(6,*) '    ',gtln1m,gtrn1m
         write(6,*) '    ',gtln2p,gtrn2p
         write(6,*) '    ',gtln2m,gtrn2m
         write(6,*) '    ',gtln3p,gtrn3p
         write(6,*) '    ',gtln3m,gtrn3m
         write(6,*) '    ',gtln4p,gtrn4p
         write(6,*) '    ',gtln4m,gtrn4m
         write(6,*) ' INIT_SUSY: FFS neutralino couplings 2 mixing'
         write(6,*) '    ',gb1n1p,gb2n1p
         write(6,*) '    ',gb1n1m,gb2n1m
         write(6,*) '    ',gb1n2p,gb2n2p
         write(6,*) '    ',gb1n2m,gb2n2m
         write(6,*) '    ',gb1n3p,gb2n3p
         write(6,*) '    ',gb1n3m,gb2n3m
         write(6,*) '    ',gb1n4p,gb2n4p
         write(6,*) '    ',gb1n4m,gb2n4m
         write(6,*) '    ',gt1n4m,gt2n4m
         write(6,*) '    ',gt1n1p,gt2n1p
         write(6,*) '    ',gt1n1m,gt2n1m
         write(6,*) '    ',gt1n2p,gt2n2p
         write(6,*) '    ',gt1n2m,gt2n2m
         write(6,*) '    ',gt1n3p,gt2n3p
         write(6,*) '    ',gt1n3m,gt2n3m
         write(6,*) '    ',gt1n4p,gt2n4p
         write(6,*) '    ',gt1n4m,gt2n4m
      end if
c
c   chargino-lepton-slepton
c
      gelx1m(1) = -rt2*gx*  uu(1,1)/rt2
      gelx1m(2) = zero
C      gerx1m(1) = zero ! doesn't exist
C      gerx1m(2) = zero

      gelx2m(1) = -rt2*gx*  uu(2,1)/rt2
      gelx2m(2) = zero
C      gerx2m(1) = zero ! doesn't exist
C      gerx2m(2) = zero

      gelx1p(1) = gelx1m(2)
      gelx1p(2) = gelx1m(1)
      gelx2p(1) = gelx2m(2)
      gelx2p(2) = gelx2m(1)

C      gerx1p(1) = gerx1m(2) ! doesn't exist
C      gerx1p(2) = gerx1m(1)
C      gerx2p(1) = gerx2m(2) ! doesn't exist
C      gerx2p(2) = gerx2m(1)

      gllx1m(1) = -rt2*gx*  uu(1,1)/rt2
      gllx1m(2) = zero
      glrx1m(1) = -rt2*gx*(-rml/(two*mwx*cb)*uu(1,2))
      glrx1m(2) = zero

      gllx2m(1) = -rt2*gx*  uu(2,1)/rt2
      gllx2m(2) = zero
      glrx2m(1) = -rt2*gx*(-rml/(two*mwx*cb)*uu(2,2))
      glrx2m(2) = zero

      gl1x1m(1) = r_l(1,1) * gllx1m(1) + r_l(1,2) * glrx1m(1)
      gl2x1m(1) = r_l(2,1) * gllx1m(1) + r_l(2,2) * glrx1m(1)
      gl1x1m(2) = r_l(1,1) * gllx1m(2) + r_l(1,2) * glrx1m(2)
      gl2x1m(2) = r_l(2,1) * gllx1m(2) + r_l(2,2) * glrx1m(2)

      gl1x2m(1) = r_l(1,1) * gllx2m(1) + r_l(1,2) * glrx2m(1)
      gl2x2m(1) = r_l(2,1) * gllx2m(1) + r_l(2,2) * glrx2m(1)
      gl1x2m(2) = r_l(1,1) * gllx2m(2) + r_l(1,2) * glrx2m(2)
      gl2x2m(2) = r_l(2,1) * gllx2m(2) + r_l(2,2) * glrx2m(2)

      gllx1p(1) = gllx1m(2)
      gllx1p(2) = gllx1m(1)
      gllx2p(1) = gllx2m(2)
      gllx2p(2) = gllx2m(1)

      glrx1p(1) = glrx1m(2)
      glrx1p(2) = glrx1m(1)
      glrx2p(1) = glrx2m(2)
      glrx2p(2) = glrx2m(1)

      gl1x1p(1) = gl1x1m(2)
      gl1x1p(2) = gl1x1m(1)
      gl2x1p(1) = gl2x1m(2)
      gl2x1p(2) = gl2x1m(1)

      gl1x2p(1) = gl1x2m(2)
      gl1x2p(2) = gl1x2m(1)
      gl2x2p(1) = gl2x2m(2)
      gl2x2p(2) = gl2x2m(1)

      gvex1m(1) = -rt2*gx * vv(1,1)/rt2
      gvex1m(2) = zero

      gvex2m(1) = -rt2*gx * vv(2,1)/rt2
      gvex2m(2) = zero

      gvex1p(1) = gvex1m(2)
      gvex1p(2) = gvex1m(1)

      gvex2p(1) = gvex2m(2)
      gvex2p(2) = gvex2m(1)

      gvtx1m(1) = -rt2*gx*  vv(1,1)/rt2
      gvtx1m(2) = -rt2*gx*(-rml/(two*mwx*cb)*uu(1,2))

      gvtx2m(1) = -rt2*gx*  vv(2,1)/rt2
      gvtx2m(2) = -rt2*gx*(-rml/(two*mwx*cb)*uu(2,2))

      gvtx1p(1) = gvtx1m(2)
      gvtx1p(2) = gvtx1m(1)

      gvtx2p(1) = gvtx2m(2)
      gvtx2p(2) = gvtx2m(1)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFS chargino couplings 1'
         write(6,*) '    ',gvex1m,gelx1m
         write(6,*) '    ',gvex1p,gelx1p
         write(6,*) '    ',gvex2m,gelx2m
         write(6,*) '    ',gvex2p,gelx2p
         write(6,*) ' INIT_SUSY: FFS chargino couplings 1 mixing'
         write(6,*) '    ',gvtx1m,gl1x1m,gl2x1m
         write(6,*) '    ',gvtx1p,gl1x1p,gl2x1p
         write(6,*) '    ',gvtx2m,gl1x2m,gl2x2m
         write(6,*) '    ',gvtx2p,gl1x2p,gl2x2p
      end if
c
c   chargino-quark-squark
c     no relative sign, because sqR coupling zero
c
      gdlx1m(1) = -rt2*gx*  uu(1,1)/rt2
      gdlx1m(2) = zero
C      gdrx1m(1) = zero
C      gdrx1m(2) = zero

      gdlx2m(1) = -rt2*gx*  uu(2,1)/rt2
      gdlx2m(2) = zero
C      gdrx2m(1) = zero
C      gdrx2m(2) = zero

      gdlx1p(1) = gdlx1m(2)
      gdlx1p(2) = gdlx1m(1)
      gdlx2p(1) = gdlx2m(2)
      gdlx2p(2) = gdlx2m(1)

C      gdrx1p(1) = gdrx1m(2)
C      gdrx1p(2) = gdrx1m(1)
C      gdrx2p(1) = gdrx2m(2)
C      gdrx2p(2) = gdrx2m(1)

      gblx1m(1) = -rt2*gx*  uu(1,1)/rt2
      gblx1m(2) = -rt2*gx*(-rmt/(two*mwx*sb)*vv(1,2))
      gbrx1m(1) = -rt2*gx*(-rmb/(two*mwx*cb)*uu(1,2))
      gbrx1m(2) = zero

      gblx2m(1) = -rt2*gx*  uu(2,1)/rt2
      gblx2m(2) = -rt2*gx*(-rmt/(two*mwx*sb)*vv(2,2))
      gbrx2m(1) = -rt2*gx*(-rmb/(two*mwx*cb)*uu(2,2))
      gbrx2m(2) = zero

      gb1x1m(1) = r_b(1,1) * gblx1m(1) + r_b(1,2) * gbrx1m(1)
      gb2x1m(1) = r_b(2,1) * gblx1m(1) + r_b(2,2) * gbrx1m(1)
      gb1x1m(2) = r_b(1,1) * gblx1m(2) + r_b(1,2) * gbrx1m(2)
      gb2x1m(2) = r_b(2,1) * gblx1m(2) + r_b(2,2) * gbrx1m(2)

      gb1x2m(1) = r_b(1,1) * gblx2m(1) + r_b(1,2) * gbrx2m(1)
      gb2x2m(1) = r_b(2,1) * gblx2m(1) + r_b(2,2) * gbrx2m(1)
      gb1x2m(2) = r_b(1,1) * gblx2m(2) + r_b(1,2) * gbrx2m(2)
      gb2x2m(2) = r_b(2,1) * gblx2m(2) + r_b(2,2) * gbrx2m(2)

      gblx1p(1) = gblx1m(2)
      gblx1p(2) = gblx1m(1)
      gblx2p(1) = gblx2m(2)
      gblx2p(2) = gblx2m(1)

      gbrx1p(1) = gbrx1m(2)
      gbrx1p(2) = gbrx1m(1)
      gbrx2p(1) = gbrx2m(2)
      gbrx2p(2) = gbrx2m(1)

      gb1x1p(1) = gb1x1m(2)
      gb1x1p(2) = gb1x1m(1)
      gb2x1p(1) = gb2x1m(2)
      gb2x1p(2) = gb2x1m(1)

      gb1x2p(1) = gb1x2m(2)
      gb1x2p(2) = gb1x2m(1)
      gb2x2p(1) = gb2x2m(2)
      gb2x2p(2) = gb2x2m(1)

      gulx1m(1) = -rt2*gx*  vv(1,1)/rt2
      gulx1m(2) = zero
C      gurx1m(1) = zero
C      gurx1m(2) = zero

      gulx2m(1) = -rt2*gx*  vv(2,1)/rt2
      gulx2m(2) = zero
C      gurx2m(1) = zero
C      gurx2m(2) = zero

      gulx1p(1) = gulx1m(2)
      gulx1p(2) = gulx1m(1)
      gulx2p(1) = gulx2m(2)
      gulx2p(2) = gulx2m(1)

C      gurx1p(1) = gurx1m(2)
C      gurx1p(2) = gurx1m(1)
C      gurx2p(1) = gurx2m(2)
C      gurx2p(2) = gurx2m(1)

      gtlx1m(1) = -rt2*gx*  vv(1,1)/rt2
      gtlx1m(2) = -rt2*gx*(-rmb/(two*mwx*cb)*uu(1,2))
      gtrx1m(1) = -rt2*gx*(-rmt/(two*mwx*sb)*vv(1,2))
      gtrx1m(2) = zero

      gtlx2m(1) = -rt2*gx*  vv(2,1)/rt2
      gtlx2m(2) = -rt2*gx*(-rmb/(two*mwx*cb)*uu(2,2))
      gtrx2m(1) = -rt2*gx*(-rmt/(two*mwx*sb)*vv(2,2))
      gtrx2m(2) = zero

      gt1x1m(1) = r_t(1,1) * gtlx1m(1) + r_t(1,2) * gtrx1m(1)
      gt2x1m(1) = r_t(2,1) * gtlx1m(1) + r_t(2,2) * gtrx1m(1)
      gt1x1m(2) = r_t(1,1) * gtlx1m(2) + r_t(1,2) * gtrx1m(2)
      gt2x1m(2) = r_t(2,1) * gtlx1m(2) + r_t(2,2) * gtrx1m(2)

      gt1x2m(1) = r_t(1,1) * gtlx2m(1) + r_t(1,2) * gtrx2m(1)
      gt2x2m(1) = r_t(2,1) * gtlx2m(1) + r_t(2,2) * gtrx2m(1)
      gt1x2m(2) = r_t(1,1) * gtlx2m(2) + r_t(1,2) * gtrx2m(2)
      gt2x2m(2) = r_t(2,1) * gtlx2m(2) + r_t(2,2) * gtrx2m(2)

      gtlx1p(1) = gtlx1m(2)
      gtlx1p(2) = gtlx1m(1)
      gtlx2p(1) = gtlx2m(2)
      gtlx2p(2) = gtlx2m(1)

      gtrx1p(1) = gtrx1m(2)
      gtrx1p(2) = gtrx1m(1)
      gtrx2p(1) = gtrx2m(2)
      gtrx2p(2) = gtrx2m(1)

      gt1x1p(1) = gt1x1m(2)
      gt1x1p(2) = gt1x1m(1)
      gt2x1p(1) = gt2x1m(2)
      gt2x1p(2) = gt2x1m(1)

      gt1x2p(1) = gt1x2m(2)
      gt1x2p(2) = gt1x2m(1)
      gt2x2p(1) = gt2x2m(2)
      gt2x2p(2) = gt2x2m(1)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: FFS chargino couplings 2'
         write(6,*) '    ',gulx1m,gdlx1m
         write(6,*) '    ',gulx1p,gdlx1p
         write(6,*) '    ',gulx2m,gdlx2m
         write(6,*) '    ',gulx2p,gdlx2p
         write(6,*) ' INIT_SUSY: FFS chargino couplings 2 mixing'
         write(6,*) '    ',gt1x1m,gb1x1m
         write(6,*) '    ',gt1x1p,gb1x1p
         write(6,*) '    ',gt1x2m,gb1x2m
         write(6,*) '    ',gt1x2p,gb1x2p
         write(6,*) '    ',gt2x1m,gb2x1m
         write(6,*) '    ',gt2x1p,gb2x1p
         write(6,*) '    ',gt2x2m,gb2x2m
         write(6,*) '    ',gt2x2p,gb2x2p
      end if
cc
C      call log_file(mzx,mwx,gf,unimass,lowmass,bw,pz,uu,vv)
ccc
      return
      end
c
c=======================================================================
c
c subroutine INIT_SUSY_QCD
c
c   all mixing matrices assumed to be real
c       -> negative mass eigenvalues
c
c   particles are always those with negative charge (e.g. charginos)
c   weak parameter are give through common block
c
c   ordering for FFV and FFS: F_in, F_out, V/S
c   where this ordering is also reflected in the name of the coupling
c   same thing in Kaoru's writeup: F_out, F_in, S/V
c
c   arrays for fermion couplings: 1=L and 2=R
c
c   Notes:
c     1. gc  used in just about all SUSY QCD processes
c     2. g2c used in g-g-sq-sq processes
c     3. ggi used in g-g-go-go, go-go-sq-sq and g-go-q-sq processes
c
c=======================================================================
c
      subroutine INIT_SUSY_QCD(gs)
      implicit none
c
c input/output variables
c
      double precision  gs
c
c global variables (couplings)
c
      include 'coupl.inc'

      double precision  sw, sw2, cw, tw, s2w, c2w, e, gx, gz,
     &                  qe, qu, qd, t3e, t3v, t3u, t3d, rt2, pi,
     &                  r_l(2,2), r_b(2,2), r_t(2,2)
      common /ewparam/  sw, sw2, cw, tw, s2w, c2w, e, gx, gz,
     &                  qe, qu, qd, t3e, t3v, t3u, t3d, rt2, pi,
     &                  r_l, r_b, r_t

c local variables
c
      double precision  zero, two
      parameter ( zero = 0d0, two = 2d0 )

      logical     ldebug
      parameter ( ldebug = .false. )
c
c basic SUSY-QCD couplings
c
c      PRINT *,'INIT_SUSY_QCD: alpha_s = ',gs**2/(4d0*pi)

      gc  = dcmplx( - gs , zero ) ! original; works for sq-sq~ production

      g2c = dcmplx( gs**2, zero ) ! original

      ggi(1) = dcmplx(-gs,zero) ! - (no i) fixed by qq~,gg -> gogo and 'in,out' MG rule
      ggi(2) = ggi(1)

      gqlgom(1) = - rt2*gs
      gqlgom(2) = zero

      gqrgom(1) = zero
      gqrgom(2) =   rt2*gs

      gqlgop(1) = gqlgom(2) ! s.b. NO relative sign for 'p' and 'm' types
      gqlgop(2) = gqlgom(1)
      gqrgop(1) = gqrgom(2)
      gqrgop(2) = gqrgom(1)

      gb1gom(1) = r_b(1,1) * gqlgom(1) + r_b(1,2) * gqrgom(1)
      gb2gom(1) = r_b(2,1) * gqlgom(1) + r_b(2,2) * gqrgom(1)
      gb1gom(2) = r_b(1,1) * gqlgom(2) + r_b(1,2) * gqrgom(2)
      gb2gom(2) = r_b(2,1) * gqlgom(2) + r_b(2,2) * gqrgom(2)

      gb1gop(1) = gb1gom(2)
      gb1gop(2) = gb1gom(1)
      gb2gop(1) = gb2gom(2)
      gb2gop(2) = gb2gom(1)

      gt1gom(1) = r_t(1,1) * gqlgom(1) + r_t(1,2) * gqrgom(1)
      gt2gom(1) = r_t(2,1) * gqlgom(1) + r_t(2,2) * gqrgom(1)
      gt1gom(2) = r_t(1,1) * gqlgom(2) + r_t(1,2) * gqrgom(2)
      gt2gom(2) = r_t(2,1) * gqlgom(2) + r_t(2,2) * gqrgom(2)

      gt1gop(1) = gt1gom(2)
      gt1gop(2) = gt1gom(1)
      gt2gop(1) = gt2gom(2)
      gt2gop(2) = gt2gom(1)

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: gluino couplings '
         write(6,*) ' gqlgom(1:2) ',dble(gqlgom(1)),dble(gqlgom(2))
         write(6,*) ' gqlgop(1:2) ',dble(gqlgop(1)),dble(gqlgop(2))
         write(6,*) ' gqrgom(1:2) ',dble(gqrgom(1)),dble(gqrgom(2))
         write(6,*) ' gqrgop(1:2) ',dble(gqrgop(1)),dble(gqrgop(2))
         write(6,*) ' gb1gom(1:2) ',dble(gb1gom(1)),dble(gb1gom(2))
         write(6,*) ' gb2gom(1:2) ',dble(gb2gom(1)),dble(gb2gom(2))
         write(6,*) ' gb1gop(1:2) ',dble(gb1gop(1)),dble(gb1gop(2))
         write(6,*) ' gb2gop(1:2) ',dble(gb2gop(1)),dble(gb2gop(2))
         write(6,*) ' gt1gom(1:2) ',dble(gt1gom(1)),dble(gt1gom(2))
         write(6,*) ' gt2gom(1:2) ',dble(gt2gom(1)),dble(gt2gom(2))
         write(6,*) ' gt1gop(1:2) ',dble(gt1gop(1)),dble(gt1gop(2))
         write(6,*) ' gt2gop(1:2) ',dble(gt2gop(1)),dble(gt2gop(2))
      end if 

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: SUSY-QCD couplings'
         write(6,*) ' GC ', gc
         write(6,*) ' G2C', g2c
         write(6,*) ' GGI', ggi
      end if
c
c VVSS couplings - mixed QCD-QED/QFD
c
      ggadldl = two * gs * e*qd
      ggaulul = two * gs * e*qu

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: g-photon-s-s couplings '
         write(6,*) 'qL,qL',ggadldl,ggaulul
      end if

      ggzdldl = two * gs * gz*(t3d-qd*sw2)
      ggzdrdr = two * gs * gz*(   -qd*sw2)
      ggzulul = two * gs * gz*(t3u-qu*sw2)
      ggzurur = two * gs * gz*(   -qu*sw2)

      ggzb1b1 = r_b(1,1)**2    * ggzdldl + r_b(1,2)**2    * ggzdrdr
      ggzb2b2 = r_b(2,1)**2    * ggzdldl + r_b(2,2)**2    * ggzdrdr
      ggzb1b2 = r_b(2,1)*r_b(1,1)* ggzdldl + r_b(1,2)*r_b(2,2)* ggzdrdr
      ggzb2b1 = r_b(2,1)*r_b(1,1)* ggzdldl + r_b(1,2)*r_b(2,2)* ggzdrdr

      ggzt1t1 = r_t(1,1)**2    * ggzulul + r_t(1,2)**2    * ggzurur
      ggzt2t2 = r_t(2,1)**2    * ggzulul + r_t(2,2)**2    * ggzurur
      ggzt1t2 = r_t(2,1)*r_t(1,1)* ggzulul + r_t(1,2)*r_t(2,2)* ggzurur
      ggzt2t1 = r_t(2,1)*r_t(1,1)* ggzulul + r_t(1,2)*r_t(2,2)* ggzurur

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: g-Z-s-s couplings '
         write(6,*) 'qL,qL',ggzdldl,ggzulul
         write(6,*) 'qR,qR',ggzdrdr,ggzurur
         write(6,*) 'q1,q1',ggzb1b1,ggzt1t1
         write(6,*) 'q2,q2',ggzb2b2,ggzt2t2
         write(6,*) 'q1,q2',ggzb1b2,ggzt1t2
         write(6,*) 'q2,q1',ggzb2b1,ggzt2t1
      end if

      ggwuldl = rt2*gx*gs
      ggwdlul = ggwuldl

      ggwb1t1 =  r_b(1,1)*r_t(1,1) * ggwdlul
      ggwb1t2 =  r_b(1,1)*r_t(2,1) * ggwdlul
      ggwb2t1 =  r_b(2,1)*r_t(1,1) * ggwdlul
      ggwb2t2 =  r_b(2,1)*r_t(2,1) * ggwdlul

      ggwt1b1 = ggwb1t1
      ggwt2b1 = ggwb1t2
      ggwt1b2 = ggwb2t1
      ggwt2b2 = ggwb2t2

      if ( ldebug ) then
         write(6,*) ' INIT_SUSY: g-W-s-s couplings '
         write(6,*) 'qL,qL',ggwuldl,ggwdlul
         write(6,*) 'q1,q1',ggwt1b1,ggwb1t1
         write(6,*) 'q1,q2',ggwt1b2,ggwb1t2
         write(6,*) 'q2,q1',ggwt1b1,ggwb2t1
         write(6,*) 'q2,q2',ggwt2b2,ggwb2t2
      end if
ccc
      return
      end
