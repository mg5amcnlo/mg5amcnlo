c
c
c Plotting routines
c
c
      subroutine initplot
      implicit none
c Book histograms in this routine. Use mbook or bookup. The entries
c of these routines are real*8
      include 'run.inc'
      REAL*8 pi
      PARAMETER (PI=3.14159265358979312D0)
      integer j,kk,jj

      call inihist
      kk=0
      call bookup(kk+ 1,'t pt',5d0,0d0,200d0)
      call bookup(kk+ 2,'t log pt',0.05d0,0d0,5d0)
      call bookup(kk+ 3,'t y',0.25d0,-6d0,6d0)
      call bookup(kk+ 4,'t eta',0.25d0,-6d0,6d0)
c
      call bookup(kk+ 5,'j1 pt',5d0,0d0,200d0)
      call bookup(kk+ 6,'j1 log pt',0.05d0,0d0,5d0)
      call bookup(kk+ 7,'j1 y',0.25d0,-6d0,6d0)
      call bookup(kk+ 8,'j1 eta',0.25d0,-6d0,6d0)
c
      call bookup(kk+ 9,'j2 pt',5d0,0d0,200d0)
      call bookup(kk+10,'j2 log pt',0.05d0,0d0,5d0)
      call bookup(kk+11,'j2 y',0.25d0,-6d0,6d0)
      call bookup(kk+12,'j2 eta',0.25d0,-6d0,6d0)
c
      call bookup(kk+13,'bj1 pt',5d0,0d0,200d0)
      call bookup(kk+14,'bj1 log pt',0.05d0,0d0,5d0)
      call bookup(kk+15,'bj1 y',0.25d0,-6d0,6d0)
      call bookup(kk+16,'bj1 eta',0.25d0,-6d0,6d0)
c
      call bookup(kk+17,'bj2 pt',5d0,0d0,200d0)
      call bookup(kk+18,'bj2 log pt',0.05d0,0d0,5d0)
      call bookup(kk+19,'bj2 y',0.25d0,-6d0,6d0)
      call bookup(kk+20,'bj2 eta',0.25d0,-6d0,6d0)
c
      call bookup(kk+21,'syst pt',5d0,0d0,200d0)
      call bookup(kk+22,'syst log pt',0.05d0,0d0,5d0)
      call bookup(kk+23,'syst y',0.25d0,-6d0,6d0)
      call bookup(kk+24,'syst eta',0.25d0,-6d0,6d0)
c

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
      integer i,kk,k
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
C
      k=0
      call multitop(k+ 1,2,3,'t pt',' ','LOG')
      call multitop(k+ 2,2,3,'t log pt',' ','LOG')
      call multitop(k+ 3,2,3,'t y',' ','LOG')
      call multitop(k+ 4,2,3,'t eta',' ','LOG')
c
      call multitop(k+ 5,2,3,'j1 pt',' ','LOG')
      call multitop(k+ 6,2,3,'j1 log pt',' ','LOG')
      call multitop(k+ 7,2,3,'j1 y',' ','LOG')
      call multitop(k+ 8,2,3,'j1 eta',' ','LOG')
c
      call multitop(k+ 9,2,3,'j2 pt',' ','LOG')
      call multitop(k+10,2,3,'j2 log pt',' ','LOG')
      call multitop(k+11,2,3,'j2 y',' ','LOG')
      call multitop(k+12,2,3,'j2 eta',' ','LOG')
c
      call multitop(k+13,2,3,'bj1 pt',' ','LOG')
      call multitop(k+14,2,3,'bj1 log pt',' ','LOG')
      call multitop(k+15,2,3,'bj1 y',' ','LOG')
      call multitop(k+16,2,3,'bj1 eta',' ','LOG')
c
      call multitop(k+17,2,3,'bj2 pt',' ','LOG')
      call multitop(k+18,2,3,'bj2 log pt',' ','LOG')
      call multitop(k+19,2,3,'bj2 y',' ','LOG')
      call multitop(k+20,2,3,'bj2 eta',' ','LOG')
c
      call multitop(k+21,2,3,'syst pt',' ','LOG')
      call multitop(k+22,2,3,'syst log pt',' ','LOG')
      call multitop(k+23,2,3,'syst y',' ','LOG')
      call multitop(k+24,2,3,'syst eta',' ','LOG')
c
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

      common/to_mass/pmass
      double precision ppkt(0:3,nexternal),ecut,djet
      double precision ycut, palg
      integer ipartjet(nexternal)
      double precision pttop,etatop,ytop,ptj1,etaj1,yj1,ptbj1,etabj1,
     &ybj1,ptbj2,etabj2,ybj2,ptmin,pttemp_spec,pttemp_bjet,pttemp,tmp,
     &getpt,p_top(0:3),p_b(0:3,nexternal),p_bjet(0:3,nexternal),
     &psyst(0:3),ptsyst,ysyst,etasyst,ptj2,yj2,
     &etaj2,pqcd(0:3,nexternal),ptjet(nexternal)

      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer j1,j2,jb,i1,i2,njets,mu
      logical is_b_jet(nexternal)
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

      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo

      do i=0,3
         p_top(i)=pplab(i,3)
      enddo

      i1=0
      i2=0
      ptmin=5d0
      if(getpt(pplab(0,4)).gt.ptmin.and.
     &   getpt(pplab(0,5)).gt.ptmin)then
         njets=2
         i1=4
         if(getpt(pplab(0,5)).gt.getpt(pplab(0,4)))i1=5
         i2=9-i1
      elseif(getpt(pplab(0,4)).gt.ptmin.or.
     &       getpt(pplab(0,5)).gt.ptmin)then
         njets=1
         i1=4
         if(getpt(pplab(0,5)).gt.getpt(pplab(0,4)))i1=5
      else
         njets=0
      endif

      j1=0
      j2=0
      jb=0
      if(njets.eq.2)then
         if(abs(pdg_type(i1)).eq.5)then
            jb=i1
            j1=i2
         elseif(abs(pdg_type(i2)).eq.5)then
            j1=i1
            jb=i2
         else
            j1=i1
            j2=i2
         endif
      elseif(njets.eq.1)then
         if(abs(pdg_type(i1)).eq.5)then
            jb=i1
         else
            j1=i1
         endif
      endif

      kk=0
      pttop = getpt(p_top)
      ytop  = getrapidity(p_top)
      etatop= getpseudorap(p_top)
      call mfill(kk+1,(pttop),(www))
      if(pttop.gt.0d0)
     &call mfill(kk+2,(log10(pttop)),(www))
      call mfill(kk+3,(ytop),(www))
      call mfill(kk+4,(etatop),(www))

      if(njets.ge.1)then
         if(j1.ne.0)then
            ptj1 = getpt(pplab(0,j1))
            yj1  = getrapidity(pplab(0,j1))
            etaj1= getpseudorap(pplab(0,j1))
            call mfill(kk+5,(ptj1),(www))
            call mfill(kk+6,(log10(ptj1)),(www))
            call mfill(kk+7,(yj1),(www))
            call mfill(kk+8,(etaj1),(www))
            do mu=0,3
               psyst(mu)=p_top(mu)+pplab(mu,j1)
            enddo
            ptsyst = getpt(psyst)
            ysyst  = getrapidity(psyst)
            etasyst= getpseudorap(psyst)
            call mfill(kk+21,(ptsyst),(www))
            if(ptsyst.gt.0d0)
     &           call mfill(kk+22,(log10(ptsyst)),(www))
            call mfill(kk+23,(ysyst),(www))
            call mfill(kk+24,(etasyst),(www))
         endif
         if(j2.ne.0)then
            ptj2 = getpt(pplab(0,j2))
            yj2  = getrapidity(pplab(0,j2))
            etaj2= getpseudorap(pplab(0,j2))
            call mfill(kk+9,(ptj2),(www))
            call mfill(kk+10,(log10(ptj2)),(www))
            call mfill(kk+11,(yj2),(www))
            call mfill(kk+12,(etaj2),(www))
         endif
         if(jb.ne.0)then
            ptbj1 = getpt(pplab(0,jb))
            ybj1  = getrapidity(pplab(0,jb))
            etabj1= getpseudorap(pplab(0,jb))
            call mfill(kk+13,(ptbj1),(www))
            call mfill(kk+14,(log10(ptbj1)),(www))
            call mfill(kk+15,(ybj1),(www))
            call mfill(kk+16,(etabj1),(www))
         endif
      endif
      
 999  return      
      end


      function getrapidity(p)
      implicit none
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y,p(0:3)
      parameter (tiny=1.d-5)
c
      en=p(0)
      pl=p(3)
      xplus=en+pl
      xminus=en-pl
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
        if( (xplus/xminus).gt.tiny )then
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


      function getpseudorap(p)
      implicit none
      real*8 getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th,p(0:3)
      parameter (tiny=1.d-5)
c
      en=p(0)
      ptx=p(1)
      pty=p(2)
      pl=p(3)
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

      function getpt(p)
      implicit none
      real*8 getpt,p(0:3)
      getpt=dsqrt(p(1)**2+p(2)**2)
      return
      end

      function getinvm(en,ptx,pty,pl)
      implicit none
      real*8 getinvm,en,ptx,pty,pl,tiny,tmp
      parameter (tiny=1.d-5)
c
      tmp=en**2-ptx**2-pty**2-pl**2
      if(tmp.gt.0.d0)then
        tmp=sqrt(tmp)
      elseif(tmp.gt.-tiny)then
        tmp=0.d0
      else
        write(*,*)'Attempt to compute a negative mass'
        stop
      endif
      getinvm=tmp
      return
      end


      function getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      getdelphi=tmp
      return
      end
