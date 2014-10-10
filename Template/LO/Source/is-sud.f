      function calcissud(kkfl,xx1,xx2,pt2min,pt2,nfnevl,relerr,ifail)

c...arguments
      integer kkfl
      double precision calcissud,xx1,xx2,pt2min,pt2
c...global variables
      double precision bthres,alam5,alam4,s,x1,x2
      integer kfl
      common/sudpars/bthres,alam5,alam4,s,x1,x2,kfl
      data alam5,alam4/2.11384241947499864d-2,3.6864E-002/
      data bthres,s/22.3109,196d6/
c...local variables
      integer n,minpts,maxpts,iwk,nfnevl,ifail
      parameter(n=2,minpts=5,maxpts=50000,iwk=10298) !iwk=7*(1+maxpts/34)
      double precision fsud,a(n),b(n),eps,wk(iwk),result,relerr
      parameter(eps=0.001)
      external fsud

      x1=xx1
      x2=xx2
      kfl=kkfl

c...limits for pt2
      a(1)=log(pt2min)
      if(abs(kfl).eq.5) a(1)=log(max(pt2min,bthres))
      b(1)=log(pt2)
c...limits for z
      a(2)=x1
      b(2)=1

      if(pt2.le.exp(a(1)))then
        calcissud=1
        nfnevl=0
        relerr=0
        ifail=0
        return
      endif

      if(abs(kfl).eq.5.and.x1.gt.0.6)then
        calcissud=0d0
        nfnevl=0
        relerr=0
        ifail=0
        return
      endif

c...perform integration
      call dadmul(fsud,n,a,b,minpts,maxpts,eps,wk,iwk,
     $   result,relerr,nfnevl,ifail)

      calcissud=exp(-result)
      return
      end


C...FSUD is the integrand called by dadmul 
      double precision function fsud(n,x)
      implicit none

c...arguments
      integer n
      double precision x(*)

c...global variables
      double precision bthres,alam5,alam4,s,x1,x2
      integer kfl
      common/sudpars/bthres,alam5,alam4,s,x1,x2,kfl

c...functions
      double precision pdg2pdf
      external pdg2pdf
      
c...local variables
      double precision b5,b4
      data b5,b4/3.8333333333333335,4.1666666666666/
      double precision pt2,pt,z,zmax,zint,s12
      DOUBLE PRECISION XPQ1(-7:7),XPQ2(-7:7)
      DATA XPQ1,XPQ2/30*0d0/
      double precision qm2(5),qmrel
      data qm2/3*0,2.2725,22.3109/
      double precision ptold,xiold,kflold,xpqold
      integer kfla,i

      pt2=exp(x(1))
      if(pt2.gt.s) then
        print *,'Error: Probably forgot log() for pt2 limits!'
        stop
      endif
      pt=dsqrt(pt2)
      z=x(2)
      s12=s*x1*x2
      zmax=1d0+pt2/(2d0*s12)-sqrt(pt2**2/(4d0*s12**2)+pt2/s12)
      kfla=iabs(kfl)

      fsud=0d0
      if(z.gt.zmax.or.z.le.x1) return

c   PT integral: 1/pt2*alpha_s(pt)/2pi
      if(pt2.gt.bthres) then
        fsud=1d0/(b5*log(pt2/alam5))
      else! if(pt2.gt.2.2725000000000000)then
        fsud=1d0/(b4*log(pt2/alam4))
      endif

c   Color coherence suppression
      if(.false.)then
        fsud=fsud*min(1d0,s12/4d0/pt2)
      endif
c   Z integral: need pdf ratio and splitting function
      zint=0
c      CALL PFTOPDG(1,x1/z,DSQRT(pt2),XPQ2)
      if(kfla.lt.6.and.kfl.ne.0)then
        XPQ1(kfl)=max(pdg2pdf(1,kfl,x1,pt),1d-20)
        XPQ2(kfl)=pdg2pdf(1,kfl,x1/z,pt)
        XPQ2(0)=pdg2pdf(1,21,x1/z,pt)
        qmrel=0
        if(qm2(kfla).gt.0)
     $     qmrel=2*z*(1-z)*qm2(kfla)/pt2
        zint=zint+XPQ2(kfl)/(z*XPQ1(kfl))*4d0/3d0*
     $     ((1+z**2)/(1-z)-qmrel)
        zint=zint+XPQ2(0)/(z*XPQ1(kfl))*0.5d0*(z**2+(1-z)**2+qmrel)
      elseif(kfl.eq.21.or.kfl.eq.0)then
        XPQ1(0)=pdg2pdf(1,0,x1,pt)
c        CALL PFTOPDG(1,x1/z,pt,XPQ2)
        do i=-2,3
          XPQ2(i)=pdg2pdf(1,i,x1/z,pt)
          if(i.ne.0)
     $       zint=zint+XPQ2(i)
        enddo
        zint=zint+XPQ2(3) ! for anti-s dist
        zint=zint/(z*XPQ1(0))*4d0/3d0*(1+(1-z)**2)/z
        zint=zint+XPQ2(0)/(z*XPQ1(0))*6*(1-z*(1-z))**2/(z*(1-z))
      endif
      fsud=fsud*zint

      return
      end


      double precision function pdg2pdf(ih,ipdg,x,xmu)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg,ipart
C
C     Include
C
      include 'PDF/pdf.inc'
C      
      double precision Ctq3df,Ctq4Fn,Ctq5Pdf,Ctq6Pdf,Ctq5L

c---------
      double precision xpdf_nnpdf(-6:6)
c---------

      integer mode,Irt

      ipart=ipdg
      if(ipart.eq.21) ipart=0

      if (pdlabel(1:5) .eq. 'cteq3') then
C     
         if (pdlabel .eq. 'cteq3_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq3_l') then
            mode=2
         elseif (pdlabel .eq. 'cteq3_d') then
            mode=3
         endif

         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq3df(mode,ipart,x,xmu,Irt)/x

         if(ipdg.ge.1.and.ipdg.le.2)
     $      pdg2pdf=pdg2pdf+Ctq3df(mode,-ipart,x,xmu,Irt)/x

C     
      elseif (pdlabel(1:5) .eq. 'cteq4') then
C     
         if (pdlabel .eq. 'cteq4_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq4_d') then
            mode=2
         elseif (pdlabel .eq. 'cteq4_l') then
            mode=3
         elseif (pdlabel .eq. 'cteq4a1') then
            mode=4
         elseif (pdlabel .eq. 'cteq4a2') then
            mode=5
         elseif (pdlabel .eq. 'cteq4a3') then
            mode=6
         elseif (pdlabel .eq. 'cteq4a4') then
            mode=7
         elseif (pdlabel .eq. 'cteq4a5') then
            mode=8
         elseif (pdlabel .eq. 'cteq4hj') then
            mode=9
         elseif (pdlabel .eq. 'cteq4lq') then
            mode=10
         endif
         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq4Fn(mode,ipart,x,xmu)
C
      elseif (pdlabel .eq. 'cteq5l1') then
C
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq5L(ipart,x,xmu)
C         
      elseif ((pdlabel(1:5) .eq. 'cteq5') .or. 
     .        (pdlabel(1:4) .eq. 'ctq5')) then
C         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq5Pdf(ipart,x,xmu)
C                  
      elseif (pdlabel(1:5) .eq. 'cteq6') then
C         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq6Pdf(ipart,x,xmu)

c----------------------------------------------
c     NNPDF ----------------------------------

      else if (pdlabel(1:7) .eq. 'nnpdf23') then
         call NNevolvePDF(x,mu,xpdf_nnpdf)
!     NNPDFevol retruns an array -6:6 as LHAPDF call
         pdg2pdf = xpdf_nnpdf(ipart) / x ! NNevolvePDF Returns xfx

         write(6,*)" in is-sud.f"
         write(6,*) " pdg2pdf = ", pdg2pdf

c----------------------------------------------
c---------------------------------------------

      endif      

      end

