      subroutine testcoupl
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
      include 'input.inc'
c
c     Common to lh_readin and printout
c
      double precision  alpha, sin2w, gfermi, alfas
      double precision  mtMS,mbMS,mcMS,mtaMS!MSbar masses
      double precision  Vud,Vus             !CKM matrix elements
      common/values/    alpha,sin2w,gfermi,alfas,
     &                  mtMS,mbMS,mcMS,mtaMS,
     &                  Vud
      open(unit=1,file="couplings_check.txt")
c
c output all info
c
 10   format( 1x,a10,2x,f7.3,' GeV        ',a16,2x,f7.4,' GeV' )
 11   format( 1x,a10,2x,f11.5,2x,f11.5,a3,f11.5,2x,f11.5 )
 12   format( 1x,a10,2x,f6.2,a )
 13   format( 1x,a10,2x,f6.4,a )
 14   format( 1x,2(a10,2x,f10.7,2x,f10.7) )
 15   format( 1x,a10,2x,f9.5,a )
 16   format( 1x,a10,2x,f7.5 )
 17   format( 1x,a10,2x,f8.4 )
 18   format( 1x,a10,2x,f8.4,' GeV' )
 19   format( 1x,a10,2x,f6.4,a13,2x,f6.4 )
 20   format( 1x,a10,2x,f11.5,1x,f11.5 )
 21   format( 1x,a10,2x,f8.4,' GeV',1x,a13,2x,f8.4,' GeV' )
 22   format( 1x,a10,2x,f10.8,a13,2x,f6.4 )
 23   format( 1x,a10,2x,f8.4)
 24   format( 1x,a10,2x,f7.3,' GeV        ',a16,2x,f7.4,' GeV  (calc @ LO)')
 25   format( 1x,a10,2x,f7.3,' GeV        ',a16,2x,f7.4,' GeV')


      write(1,20) 'gwwa ', gwwa
      write(1,20) 'gwwz ', gwwz
      write(1,20) 'gwwh ', gwwh
      write(1,20) 'gzzh ', gzzh
      write(1,20) 'ghhh ', ghhh
      write(1,20) 'gwwhh ', gwwhh
      write(1,20) 'gzzhh ', gzzhh
      write(1,20) 'ghhhh ', ghhhh
      write(1,11) 'gal ',gal(1),' ',gal(2)
      write(1,11) 'gau ',gau(1),' ',gau(2)
      write(1,11) 'gad ',gad(1),' ',gad(2)
      write(1,11) 'gwf ',gwf(1),' ',gwf(2)
      write(1,11) 'gzn ',gzn(1),' ',gzn(2)
      write(1,11) 'gzl ',gzl(1),' ',gzl(2)
      write(1,11) 'gzu ',gzu(1),' ',gzu(2)
      write(1,11) 'gzd ',gzd(1),' ',gzd(2)
      write(1,14) 'gHtop ',ghtop(1),' ',ghtop(2)
      write(1,14) 'gHbot ',ghbot(1),' ',ghbot(2)
      write(1,14) 'gHcha ',ghcha(1),' ',ghcha(2)
      write(1,14) 'gHtau ',ghtau(1),' ',ghtau(2)
      write(1,14) 'gg ',gg(1),' ',gg(2)

      return
      end
