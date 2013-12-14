c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
c
c begin reweight
      include 'reweight0.inc'
      include 'reweightNLO.inc'
      character*4 eesc(maxscales,maxscales)
      character*4 eepdf(0:maxPDFs)
      common/ceestrings/eesc,eepdf
c The following are useful, but not crucial. One has maxset replicas
c (corresponding eg to no-cuts, cuts1, cuts2,...) of up to maxplot plots
c (corresponding eg to pt1, eta1, pt2, ...), for each PDF and scale choices.
c The integers istartscales and istartPDFs are defined below
      integer maxset,maxplot,istartscales,istartPDFs
      common/cnumplots/maxset,maxplot,istartscales,istartPDFs
      integer isc,jsc,ipdf
c end reweight
      include 'run.inc'
      integer i,kk,kk1
      character*4 ddstr
      character*6 cc(2)
      data cc/' NLO  ',' Born '/
c
c resets histograms
      call inihist
c begin reweight
      call plotunc_setup()
      maxset=2
      maxplot=50
      istartscales=maxset*maxplot
      istartPDFs=(1+numscales*numscales)*maxset*maxplot
c end reweight
c
      do i=1,maxset
c default
        kk=(i-1)*maxplot
        ddstr='def'
        call bookup(kk+1,'total rate'//cc(i)//ddstr,
     #              1.0d0,0.5d0,5.5d0)
c begin reweight
        if(.not.doNLOscaleunc)goto 111
        do isc=1,numscales
          do jsc=1,numscales
            ddstr=eesc(isc,jsc)
            kk1=kk+istartscales+
     #          maxset*maxplot*((isc-1)*numscales+(jsc-1))
            call bookup(kk1+1,'total rate'//cc(i)//ddstr,
     #                  1.0d0,0.5d0,5.5d0)
          enddo
        enddo
 111    continue
        if(.not.doNLOPDFunc)goto 222
        do ipdf=0,numPDFs-1
          ddstr=eepdf(ipdf)
          kk1=kk+istartPDFs+maxset*maxplot*ipdf
          call bookup(kk1+1,'total rate'//cc(i)//ddstr,
     #                1.0d0,0.5d0,5.5d0)
        enddo
 222    continue
c end reweight
      enddo
      return
      end


      subroutine topout
      implicit none
      character*14 ytit
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      real*8 xnorm1,xnorm2
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      integer i,kk,kk1
      include 'dbook.inc'
c begin reweight
      include 'reweight0.inc'
      include 'reweightNLO.inc'
      character*4 eesc(maxscales,maxscales)
      character*4 eepdf(0:maxPDFs)
      common/ceestrings/eesc,eepdf
      integer isc,jsc,ipdf
c The following are defined in initplot
      integer maxset,maxplot,istartscales,istartPDFs
      common/cnumplots/maxset,maxplot,istartscales,istartPDFs
c end reweight
c
      if (unwgt) then
         ytit='events per bin'
      else
         ytit='sigma per bin '
      endif
      xnorm1=1.d0/float(itmax)
      xnorm2=1.d0/float(ncall*itmax)
      do i=1,NPLOTS
        if(usexinteg.and..not.mint) then
           call mopera(i,'+',i,i,xnorm1,0.d0)
        elseif(mint) then
           call mopera(i,'+',i,i,xnorm2,0.d0)
        endif
        call mfinal(i)
      enddo
c default
      do i=1,maxset
        kk=(i-1)*maxplot
        call multitop(kk+1,3,2,'total rate',ytit,'LIN')
      enddo
c begin reweight
      if(.not.doNLOscaleunc)goto 111
      do isc=1,numscales
        do jsc=1,numscales
c within these do loops, repeat what done for default
          do i=1,maxset
            kk=(i-1)*maxplot
            kk1=kk+istartscales+
     #          maxset*maxplot*((isc-1)*numscales+(jsc-1))
            call multitop(kk1+1,3,2,'total rate',ytit,'LIN')
          enddo
c end repeat
        enddo
      enddo
 111  continue
      if(.not.doNLOPDFunc)goto 222
      do ipdf=0,numPDFs-1
c within this do loop, repeat what done for default
        do i=1,maxset
          kk=(i-1)*maxplot
          kk1=kk+istartPDFs+maxset*maxplot*ipdf
          call multitop(kk1+1,3,2,'total rate',ytit,'LIN')
        enddo
c end repeat
      enddo
 222  continue
c end reweight
      return                
      end


      subroutine outfun(pp,ybst_til_tolab,www,itype)
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
C
C In MadFKS, the momenta PP given in input to this function are in the
C reduced parton c.m. frame. If need be, boost them to the lab frame.
C The rapidity of this boost is
C
C       YBST_TIL_TOLAB
C
C also given in input
C
C This is the rapidity that enters in the arguments of the sinh() and
C cosh() of the boost, in such a way that
C       ylab = ycm - ybst_til_tolab
C where ylab is the rapidity in the lab frame and ycm the rapidity
C in the center-of-momentum frame.
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
      implicit none
      include 'nexternal.inc'
      real*8 pp(0:3,nexternal),ybst_til_tolab,www
      integer itype
      real*8 var
      integer kk,kk1
c begin reweight
      include 'reweight.inc'
      include 'reweightNLO.inc'
      integer irwgt,isc,jsc,ipdf
      double precision wgtden,wwwrsc
c end reweight
c The following are defined in initplot
      integer maxset,maxplot,istartscales,istartPDFs
      common/cnumplots/maxset,maxplot,istartscales,istartPDFs
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
c
      kk=0
      if(itype.eq.20)kk=50
c begin reweight
      if(itype.eq.11)then
        irwgt=1
        wgtden=wgtrefNLO11
      elseif(itype.eq.12)then
        irwgt=2
        wgtden=wgtrefNLO12
      elseif(itype.eq.20)then
        irwgt=3
        wgtden=wgtrefNLO20
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif
c end reweight
c
      var=1.d0
      call mfill(kk+1,var,www)
c begin reweight
      if(.not.doNLOscaleunc.or.wgtden.eq.0.d0)goto 111
      do isc=1,numscales
        do jsc=1,numscales
          kk1=kk+istartscales+
     #        maxset*maxplot*((isc-1)*numscales+(jsc-1))
          wwwrsc=www * wgtNLOxsecmu(irwgt,isc,jsc)/wgtden
          call mfill(kk1+1,var,wwwrsc)
        enddo
      enddo
 111  continue
      if(.not.doNLOPDFunc.or.wgtden.eq.0.d0)goto 222
      do ipdf=0,numPDFs-1
        kk1=kk+istartPDFs+maxset*maxplot*ipdf
        wwwrsc=www * wgtNLOxsecPDF(irwgt,ipdf)/wgtden
        call mfill(kk1+1,var,wwwrsc)
      enddo
 222  continue
c end reweight
 999  return      
      end



c
c
c Utilities
c
c
      subroutine plotunc_setup()
      implicit none
      include 'reweight0.inc'
      include 'reweightNLO.inc'
      integer i,j
      character*4 eesc(maxscales,maxscales)
      character*4 eepdf(0:maxPDFs)
      common/ceestrings/eesc,eepdf
c
      if(.not.doNLOscaleunc)goto 111
c numscales*numscales tags sij
      if(numscales.gt.9)then
        write(*,*)'Number of scales too large: wrong sij tags',
     #            numscales
        stop
      endif
      do i=1,numscales
        do j=1,numscales
          eesc(i,j)='s'
          call fk88strnum(eesc(i,j),i)
          call fk88strnum(eesc(i,j),j)
        enddo
      enddo
 111  continue
      if(.not.doNLOPDFunc)goto 222
c numPDFs tags pxyz
      if(numPDFs.gt.999)then
        write(*,*)'Number of PDFs too large: wrong pxyz tags',
     #            numPDFs
        stop
      endif
      do i=0,numPDFs-1
        if(i.lt.10)then
          eepdf(i)='p00'
        elseif(i.lt.100)then
          eepdf(i)='p0'
        else
          eepdf(i)='p'
        endif
        call fk88strnum(eepdf(i),i)
      enddo
 222  continue
      return
      end


      subroutine fk88strnum(string,num)
c- writes the number num on the string string starting at the blank
c- following the last non-blank character
      character * (*) string
      character * 20 tmp
      l = len(string)
      write(tmp,'(i15)')num
      j=1
      dowhile(tmp(j:j).eq.' ')
        j=j+1
      enddo
      ipos = ifk88istrl(string)
      ito = ipos+1+(15-j)
      if(ito.gt.l) then
         write(*,*)'error, string too short'
         write(*,*) string
         stop
      endif


      string(ipos+1:ito)=tmp(j:)
      end
      function ifk88istrl(string)
c returns the position of the last non-blank character in string
      character * (*) string
      i = len(string)
      dowhile(i.gt.0.and.string(i:i).eq.' ')
         i=i-1
      enddo
      ifk88istrl = i
      end
