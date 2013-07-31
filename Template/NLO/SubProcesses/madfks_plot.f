c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
      double precision emax,ebin,etamin,etamax,etabin
      integer i,kk
      include 'run.inc'
      character*5 cc(2)
      data cc/' NLO ',' Born'/
c
      emax=dsqrt(ebeam(1)*ebeam(2))
      ebin=emax/50.d0
      etamin=-5.d0
      etamax=5.d0
      etabin=0.2d0
c resets histograms
      call inihist
c
      do i=1,2
        kk=(i-1)*50
        call bookup(kk+1,'total rate'//cc(i),1.0d0,0.5d0,5.5d0)
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
      integer i,kk
      include 'dbook.inc'
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
      do i=1,2
        kk=(i-1)*50
        call multitop(kk+1,3,2,'total rate',ytit,'LIN')
      enddo
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
      real*8 ppevs(0:3,nexternal)

      real*8 getrapidity,getpseudorap,chybst,shybst,chybstmo
      real*8 xd(1:3)
      data (xd(i),i=1,3)/0,0,1/
      real*8 pplab(0:3,nexternal)
      double precision ppcl(4,nexternal),y(nexternal)
      double precision pjet(0:3,nexternal)
      double precision cthjet(nexternal)
      integer nn,njet,nsub,jet(nexternal)
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat
      integer i,j,kk,imax

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
c masses
      double precision pmass(nexternal)
      double precision  pt1, eta1, y1, pt2, eta2, y2, pt3, eta3, y3, ht

      common/to_mass/pmass
      double precision ppkt(0:3,nexternal),ecut,djet
      double precision ycut, palg
      integer ipartjet(nexternal)

c
      if(itype.eq.11.or.itype.eq.12)then
        kk=0
      elseif(itype.eq.20)then
        kk=50
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif

      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo


        var=1.d0
        call mfill(kk+1,var,www)

 999  return      
      end


      function getrapidity(en,px,py,pl)
      implicit none
      real*8 getrapidity,en,px,py,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
      if (abs(en).lt.abs(pl)) en = dsqrt(px**2+py**2+pl**2)
c
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny )then
          y=0.5d0*log( xplus/xminus )
        else
          y=sign(1.d0,pl)*1.d8
        endif
      else
        y=sign(1.d0,pl)*1.d8
      endif
      getrapidity=y
      return
      end


      function getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      getpseudorap=eta
      return
      end

