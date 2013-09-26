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
      real * 8 xm0,gam,xmlow,xmupp,bin
      real * 8 xmi,xms,pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,k,jpr
      character*5 cc(5)
      data cc/'     ',' cut1',' cut2',' cut3',' cut4'/

      xmi=40.d0
      xms=120.d0
      bin=1.0d0

      call inihist

      do j=1,1
      k=30+(j-1)*5

      call bookup(k+ 1,'W pt'//cc(j),2.d0,0.d0,200.d0)
      call bookup(k+ 2,'W log pt'//cc(j),0.05d0,0.d0,5.d0)
      call bookup(k+ 3,'W y'//cc(j),0.25d0,-9.d0,9.d0)
      call bookup(k+ 4,'W eta'//cc(j),0.25d0,-9.d0,9.d0)
      call bookup(k+ 5,'mW'//cc(j),(bin),xmi,xms)

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
      integer i,kk,j,k
c
      if (unwgt) then
         ytit='events per bin'
      else
         ytit='sigma per bin '
      endif
      xnorm1=1.d0/float(itmax)
      xnorm2=1.d0/float(ncall*itmax)
      do i=1,500
        if(usexinteg.and..not.mint) then
           call mopera(i,'+',i,i,xnorm1,0.d0)
        elseif(mint) then
           call mopera(i,'+',i,i,xnorm2,0.d0)
        endif
        call mfinal(i)
      enddo

      do j=1,1
      k=30+(j-1)*5
      call multitop(k+ 1,3,2,'W pt',' ','LOG')
      call multitop(k+ 2,3,2,'W log pt',' ','LOG')
      call multitop(k+ 3,3,2,'W y',' ','LOG')
      call multitop(k+ 4,3,2,'W eta',' ','LOG')
      call multitop(k+ 5,3,2,'mW',' ','LOG')
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
      double precision djet,ecut,ycut
      double precision ppcl(4,nexternal),y(nexternal)
      double precision pjet(4,nexternal)
      double precision cthjet(nexternal)
      integer nn,njet,nsub,jet(nexternal)
      real*8 emax,getcth,cpar,dpar,thrust,dot,shat, getrapidity
      real*8 threedot,getpseudorap
      integer i,j,kk,imax

      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM
c masses
      double precision pmass(nexternal)
      common/to_mass/pmass
      real *8 pplab(0:3, nexternal)
      double precision shybst, chybst, chybstmo
      real*8 xd(1:3)
      data (xd(i), i=1,3) /0,0,1/
      real*8 ye,pte,etmiss,mtr,pw(0:3),yw,ptw,ptj,cphi,xmv,yv,etav,ptv
c
      kk=0
      if(itype.eq.11.or.itype.eq.12)then
        continue
      elseif(itype.eq.20)then
        return
      else
        write(*,*)'Error in outfun: unknown itype',itype
        stop
      endif
c
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo= chybst-1.d0
      do i=3,nexternal
      call boostwdir2(chybst, shybst, chybstmo,xd,pp(0,i), pplab(0,i))
      enddo
      do i=0,3
        pw(i)=pplab(i,3)
      enddo
      xmv=sqrt(dot(pw,pw))
      ptv=sqrt(pw(1)**2+pw(2)**2)
      yv=getrapidity(pw(0),pw(3))
      etav=getpseudorap(pw(0),pw(1),pw(2),pw(3))
C
      do j=1,1
        kk=30+(j-1)*5
          call mfill(kk+1,(ptv),(WWW))
          if(ptv.gt.0)call mfill(kk+2,(log10(ptv)),(WWW))
          call mfill(kk+3,(yv),(WWW))
          call mfill(kk+4,(etav),(WWW))
          call mfill(kk+5,(xmv),(WWW))
      enddo
C
 999  return      
      end



      function getrapidity(en,pl)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1.d-8)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if( (xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y=0.5d0*log( xplus/xminus  )
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
