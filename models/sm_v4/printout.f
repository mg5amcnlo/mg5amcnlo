      subroutine printout
      implicit none
c
c     local
c
      integer i,iformat
      character*2 ab(2)
      real*8 ene
      double precision  Zero, One, Two, Three, Four, Half, Rt2
      parameter( Zero = 0.0d0, One = 1.0d0, Two = 2.0d0 )
c
c     include
c
      include 'coupl.inc'
c
c     Common to lh_readin and printout
c
      double precision  alpha, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,gfermi,alfas,   
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
c
c output all info
c
 10   format( 1x,a,f7.3,' GeV        ',a,f7.4,' GeV' )
 11   format( 1x,a,f10.7,2x,f10.7,a,f10.7,2x,f10.7 )
 12   format( 1x,a,f6.2,a )
 13   format( 1x,a,f6.4,a )
 14   format( 1x,2(a,f10.7,', ',f10.7) )
 15   format( 1x,a,f9.5,a )
 16   format( 1x,a,f7.5 )
 17   format( 1x,a,f8.4 )
 18   format( 1x,a,f8.4,' GeV' )
 19   format( 1x,a,f6.4,a,f6.4 )
 20   format( 1x,a,f11.5,1x,f11.5 )
 21   format( 1x,a,f8.4,' GeV',1x,a,f8.4,' GeV' )
 22   format( 1x,a,f10.8,a,f6.4 )
 23   format( 1x,a,f8.4)
 24   format( 1x,a,f7.3,' GeV        ',a,f7.4,' GeV  (calc @ LO)')


      write(*,*) '*****************************************************'
      write(*,*) '*                    MadEvent                       *'
      write(*,*) '*        --------------------------------           *'
      write(*,*) '*          http://madgraph.hep.uiuc.edu             *'	
      write(*,*) '*          http://madgraph.phys.ucl.ac.be           *'
      write(*,*) '*          http://madgraph.roma2.infn.it            *'
      write(*,*) '*        --------------------------------           *'	
      write(*,*) '*                                                   *'
      write(*,*) '*         INTEGRATION CHANNEL LOG FILE              *'
      write(*,*) '*                                                   *'
      write(*,*) '*****************************************************'
      write(6,*)
      write(*,*) '*****************************************************'
      write(*,*) '*          SUMMARY OF THE SM PARAMETERS             *'
      write(*,*) '*****************************************************'
      write(6,*)
      write(6,*)  ' EW Params            '
      write(6,*)  '----------------------'
      write(6,23) ' GF (10^-5*GeV^-2) ',gfermi*1d5
      write(6,23) ' 1/alpha           ',1d0/alpha
      write(6,23) ' M_Z   (GeV)       ',zmass
      write(6,*)
      write(6,*)
      write(6,*)  'Boson masses and widths:'
      write(6,*)  '------------------------'
      write(6,*)
      write(6,24) 'Z mass  =  ',zmass, 'Z width  = ',zwidth
      write(6,24) 'W mass  =  ',wmass, 'W width  = ',wwidth
      write(6,24) 'H mass  =  ',hmass, 'H width  = ',hwidth
      write(6,*)
      write(6,*)  'Fermion masses and widths:'
      write(6,*)  '--------------------------'
      write(6,*)
      write(6,24) 'top    mass  =  ', tmass, 'top    width  = ', twidth
      write(6,10) 'bottom mass  =  ', bmass, 'bottom width  = ', Zero
      write(6,10) 'charm  mass  =  ', cmass, 'charm  width  = ', Zero
      write(6,10) 'tau    mass  =  ', lmass, 'tau    width  = ', Zero
      write(6,*)  'all other quark and lepton masses set to zero'
      write(6,*)
      write(6,*) 'Boson couplings:'
      write(6,*) '----------------'
      write(6,*)
      write(6,20) 'gwwa  = ', gwwa
      write(6,20) 'gwwz  = ', gwwz
      write(6,20) 'gwwh  = ', gwwh
      write(6,20) 'gzzh  = ', gzzh
      write(6,20) 'ghhh  = ', ghhh
      write(6,*)
      write(6,20) 'gwwhh = ', gwwhh
      write(6,20) 'gzzhh = ', gzzhh
      write(6,20) 'ghhhh = ', ghhhh
      write(6,*)
      write(6,*) 'FFV couplings:'
      write(6,*) '--------------'
      write(6,*)
      write(6,11) 'gal(L)   =  ',gal(1), '    gal(R)   =  ',gal(2)
      write(6,11) 'gau(L)   =  ',gau(1), '    gau(R)   =  ',gau(2)
      write(6,11) 'gad(L)   =  ',gad(1), '    gad(R)   =  ',gad(2)
      write(6,*)
      write(6,11) 'gwf(L)   =  ',gwf(1), '    gwf(R)   =  ',gwf(2)
      write(6,*)
      write(6,11) 'gzn(L)   =  ',gzn(1), '    gzn(R)   =  ',gzn(2)
      write(6,11) 'gzl(L)   =  ',gzl(1), '    gzl(R)   =  ',gzl(2)
      write(6,11) 'gzu(L)   =  ',gzu(1), '    gzu(R)   =  ',gzu(2)
      write(6,11) 'gzd(L)   =  ',gzd(1), '    gzd(R)   =  ',gzd(2)
      write(6,*)
      write(6,*) 'FFH couplings:'
      write(6,*) '--------------'
      write(6,*)
      write(6,14) 'gHtop(L) =  ',ghtop(1), '    gHtop(R) =  ',ghtop(2)
      write(6,14) 'gHbot(L) =  ',ghbot(1), '    gHbot(R) =  ',ghbot(2)
      write(6,14) 'gHcha(L) =  ',ghcha(1), '    gHcha(R) =  ',ghcha(2)
      write(6,14) 'gHtau(L) =  ',ghtau(1), '    gHtau(R) =  ',ghtau(2)
      write(6,*)
      write(6,*) 'Strong couplings:'
      write(6,*) '-----------------'
      write(6,*)
      write(6,14) 'gg(1)    =  ',gg(1)   , '    gg(2)    =  ',gg(2)
      write(6,*)

      return
      end

