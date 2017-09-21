c     This function defines the reweighting of the cross section to
c     include a muR-dependent pre-factor. Multiply by the muR-dependent
c     factor and devide by the muR-independent one.
c Note: This is implemented below for the Bottom Yukawa in the SM.
c       Change it to the factor you need to reweight.
      Double precision function rwgt_muR_dep_fac(scale,central)
      use extra_weights
      implicit none
      double precision scale,vev,mbmb,apimuR,apimZ,apimb,mbmuR,alphas,pi
      parameter (pi=3.14159265358979323846d0)
      include "nexternal.inc"
      include "genps.inc"
      include "coupl.inc"
      include "../../Source/MODEL/input.inc"
      integer i
      double precision central,tootiny,apicentral,mbcentral
      parameter (tootiny=1d-9)
      rwgt_muR_dep_fac = 1d0
c     This is relevant for a muR-dependent bottom-mass in Yukawa.
      IF(wgtcpower .ne. 0d0 .and. runfac .eq. 1) THEN
c$$$      vev    = 246.21845813429518469305d0 !vev in aMC@NLO from y_b->m_b
c$$$      mbmb = MDL_YB*vev/dsqrt(2d0)
c$$$com-- mbmb input for fixed Yukawa bmass in param_card.dat is used here
c$$$com-- as start value of running and to remove it from the cross section
c$$$c new settings NLO
c$$$      apimZ  = alphas(MDL_MZ)/pi
c$$$
c$$$      if(dabs(scale/central-1d0).lt.tootiny) then
c$$$c if scale muR is the same as the central scale of muR, get 
c$$$c "input value" mb(muR) with highest possible accuracy
c$$$         CALL runalpha(apimZ,MDL_MZ,central,4d0,4,0,apimuR)
c$$$         CALL runalpha(apimZ,MDL_MZ,mbmb,4d0,4,0,apimb)
c$$$         CALL runmass(mbmb,apimb,apimuR,4d0,4,mbmuR)
c$$$      else
c$$$c if scale and central are different (muR variations) do two steps:
c$$$c step 1: get "input value" mb(central scale) from most accurate running
c$$$         CALL runalpha(apimZ,MDL_MZ,central,4d0,4,0,apicentral)
c$$$         CALL runalpha(apimZ,MDL_MZ,mbmb,4d0,4,0,apimb)
c$$$         CALL runmass(mbmb,apimb,apicentral,4d0,4,mbcentral)
c$$$c step 2: get variation around central value, ie mb(muR), with loop 
c$$$c         order consistent with computation LO: 1-loop, NLO: 2-loop
c$$$         CALL runalpha(apicentral,central,scale,4d0,2,0,apimuR)
c$$$         CALL runmass(mbcentral,apicentral,apimuR,4d0,2,mbmuR)
c$$$      endif
c$$$      rwgt_muR_dep_fac = (mbmuR/mbmb)**wgtcpower
      ELSE
         return
      ENDIF
      END

C-{{{ routines for running of alphas:

C-{{{ subroutine odeint:

      SUBROUTINE odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,
     *rkqs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER nbad,nok,nvar,KMAXX,MAXSTP,NMAX
      REAL*8 eps,h1,hmin,x1,x2,ystart(nvar),TINY
      EXTERNAL derivs,rkqs
      PARAMETER (MAXSTP=10000,NMAX=50,KMAXX=200,TINY=1.e-30)
      INTEGER i,kmax,kount,nstp
      REAL*8 dxsav,h,hdid,hnext,x,xsav,dydx(NMAX),xp(KMAXX),y(NMAX),
     *yp(NMAX,KMAXX),yscal(NMAX)
      COMMON /path/ kmax,kount,dxsav,xp,yp
      x=x1
      h=sign(h1,x2-x1)
      nok=0
      nbad=0
      kount=0
      do 11 i=1,nvar
        y(i)=ystart(i)
11    continue
      if (kmax.gt.0) xsav=x-2.*dxsav
      do 16 nstp=1,MAXSTP
        call derivs(x,y,dydx)
        do 12 i=1,nvar
          yscal(i)=dabs(y(i))+dabs(h*dydx(i))+TINY
12      continue
        if(kmax.gt.0)then
          if(dabs(x-xsav).gt.dabs(dxsav)) then
            if(kount.lt.kmax-1)then
              kount=kount+1
              xp(kount)=x
              do 13 i=1,nvar
                yp(i,kount)=y(i)
13            continue
              xsav=x
            endif
          endif
        endif
        if((x+h-x2)*(x+h-x1).gt.0.) h=x2-x
        call rkqs(y,dydx,nvar,x,h,eps,yscal,hdid,hnext,derivs)
        if(hdid.eq.h)then
          nok=nok+1
        else
          nbad=nbad+1
        endif
        if((x-x2)*(x2-x1).ge.0.)then
          do 14 i=1,nvar
            ystart(i)=y(i)
14        continue
          if(kmax.ne.0)then
            kount=kount+1
            xp(kount)=x
            do 15 i=1,nvar
              yp(i,kount)=y(i)
15          continue
          endif
          return
        endif
        if(dabs(hnext).lt.hmin) write(6,*)
     *       'stepsize smaller than minimum in odeint'
        h=hnext
16    continue
      write(6,*) 'too many steps in odeint'
      stop
      return
      END

C-}}}
C-{{{ subroutine rkck:

      SUBROUTINE rkck(y,dydx,n,x,h,yout,yerr,derivs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER n,NMAX
      REAL*8 h,x,dydx(n),y(n),yerr(n),yout(n)
      EXTERNAL derivs
      PARAMETER (NMAX=50)
CU    USES derivs
      INTEGER i
      REAL*8 ak2(NMAX),ak3(NMAX),ak4(NMAX),ak5(NMAX),ak6(NMAX),
     *ytemp(NMAX),A2,A3,A4,A5,A6,B21,B31,B32,B41,B42,B43,B51,B52,B53,
     *B54,B61,B62,B63,B64,B65,C1,C3,C4,C6,DC1,DC3,DC4,DC5,DC6
      PARAMETER (A2=.2,A3=.3,A4=.6,A5=1.,A6=.875,B21=.2,B31=3./40.,
     *B32=9./40.,B41=.3,B42=-.9,B43=1.2,B51=-11./54.,B52=2.5,
     *B53=-70./27.,B54=35./27.,B61=1631./55296.,B62=175./512.,
     *B63=575./13824.,B64=44275./110592.,B65=253./4096.,C1=37./378.,
     *C3=250./621.,C4=125./594.,C6=512./1771.,DC1=C1-2825./27648.,
     *DC3=C3-18575./48384.,DC4=C4-13525./55296.,DC5=-277./14336.,
     *DC6=C6-.25)
      do 11 i=1,n
        ytemp(i)=y(i)+B21*h*dydx(i)
11    continue
      call derivs(x+A2*h,ytemp,ak2)
      do 12 i=1,n
        ytemp(i)=y(i)+h*(B31*dydx(i)+B32*ak2(i))
12    continue
      call derivs(x+A3*h,ytemp,ak3)
      do 13 i=1,n
        ytemp(i)=y(i)+h*(B41*dydx(i)+B42*ak2(i)+B43*ak3(i))
13    continue
      call derivs(x+A4*h,ytemp,ak4)
      do 14 i=1,n
        ytemp(i)=y(i)+h*(B51*dydx(i)+B52*ak2(i)+B53*ak3(i)+B54*ak4(i))
14    continue
      call derivs(x+A5*h,ytemp,ak5)
      do 15 i=1,n
        ytemp(i)=y(i)+h*(B61*dydx(i)+B62*ak2(i)+B63*ak3(i)+B64*ak4(i)+
     *B65*ak5(i))
15    continue
      call derivs(x+A6*h,ytemp,ak6)
      do 16 i=1,n
        yout(i)=y(i)+h*(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i))
16    continue
      do 17 i=1,n
        yerr(i)=h*(DC1*dydx(i)+DC3*ak3(i)+DC4*ak4(i)+DC5*ak5(i)+DC6*
     *ak6(i))
17    continue
      return
      END

C-}}}
C-{{{ subroutine rkqs:

      SUBROUTINE rkqs(y,dydx,n,x,htry,eps,yscal,hdid,hnext,derivs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER n,NMAX
      REAL*8 eps,hdid,hnext,htry,x,dydx(n),y(n),yscal(n)
      EXTERNAL derivs
      PARAMETER (NMAX=50)
CU    USES derivs,rkck
      INTEGER i
      REAL*8 errmax,h,htemp,xnew,yerr(NMAX),ytemp(NMAX),SAFETY,PGROW,
     *PSHRNK,ERRCON
      PARAMETER (SAFETY=0.9,PGROW=-.2,PSHRNK=-.25,ERRCON=1.89d-4)
      h=htry
1     call rkck(y,dydx,n,x,h,ytemp,yerr,derivs)
      errmax=0.d0
      do 11 i=1,n
        errmax=max(errmax,dabs(yerr(i)/yscal(i)))
11    continue
      errmax=errmax/eps
      if(errmax.gt.1.)then
        htemp=SAFETY*h*(errmax**PSHRNK)
        h=sign(max(dabs(htemp),0.1*dabs(h)),h)
        xnew=x+h
        if(xnew.eq.x) write(6,*) 'stepsize underflow in rkqs'
        goto 1
      else
        if(errmax.gt.ERRCON)then
          hnext=SAFETY*h*(errmax**PGROW)
        else
          hnext=5.*h
        endif
        hdid=h
        x=x+h
        do 12 i=1,n
          y(i)=ytemp(i)
12      continue
        return
      endif
      END

C-}}}
C-{{{ subroutine runalpha:

      subroutine runalpha(api0,mu0,mu,nf,nloop,verb,apiout)
C..
c..   NEEDS:  rkck.f rkqs.f odeint.f  (from Numerical Recipes)
c..   
c..   Note:  api = {\alpha_s \over \pi}
C..
c..   purpose : computes the value of api(mu) from api(mu0)
c..   method  : solving RG-equation by adaptive Runge-Kutta method
c..   uses    : odeint.for  from Numerical Recipes
C..
c..   api0  :  api(mu0)
c..   nf    :  number of flavors
c..   nloop :  number of loops
c..   verb  :  0=quiet,  1=verbose
c..   apiout:  api(mu)    
C..
      implicit real*8 (a-h,o-z)
      INTEGER KMAXX,NMAX,NVAR
      PARAMETER (KMAXX=200,NMAX=50,NVAR=1)
      INTEGER kmax,kount,nbad,nok,nrhs,verb
      REAL*8 dxsav,eps,h1,hmin,x,y,apif(NVAR),api0,apiout,pi
      real*8 mu,mu0,l0,lf,nf
c..   /path/  is for odeint.for:
      COMMON /path/ kmax,kount,dxsav,x(KMAXX),y(NMAX,KMAXX)
      common /bfunc/ beta0,beta1,beta2,beta3
      COMMON /cbnrhs/nrhs 
      data pi/3.14159265358979323846264338328d0/
      EXTERNAL rhs,rkqs

      if (nloop.eq.0) then
         apiout = api0
         return
      endif

      nrhs=0

c..   integration bounds (note that log(mu^2) is the integration variable)
      l0 = 0.d0
      lf = 2.*dlog(mu/mu0)
      apif(1)=api0

c..   see documentation for odeint.for:
      eps=1.0d-8
      h1=dabs(lf-l0)/10.d0
      hmin=0.0d0
      kmax=100
      dxsav=dabs(lf-l0)/20.d0

c..   initialize beta-function (common block /bfunc/):
      call inibeta(nf,nloop)

c..   check if input values are reasonable
      dlam = mu0*dexp(-1.d0/(2.d0*beta0*api0))
      if (mu.le.dlam) then
         write(6,2001) dlam,mu,mu0,api0*pi
      endif

c..   integrate RG-equation:

      call odeint(apif,NVAR,l0,lf,eps,h1,hmin,nok,nbad,rhs,rkqs)

      if (verb.eq.1) then
         write(6,'(/1x,a,t30,i3)') 'Successful steps:',nok
         write(6,'(1x,a,t30,i3)') 'Bad steps:',nbad
         write(6,'(1x,a,t30,i3)') 'Function evaluations:',nrhs
         write(6,'(1x,a,t30,i3)') 'Stored intermediate values:',kount
      endif

c..   api(mu):
      apiout = apif(1)

 2001 format(' -> <subroutine runalpha>',/,
     &     ' - WARNING: mu-value too low.',/,
     &     ' -     Should be significantly larger than  ',1f8.3,'.',/,
     &     ' -             mu = ',1f8.3,' GeV',/,
     &     ' -            mu0 = ',1f8.3,' GeV',/,
     &     ' -        api0*pi = ',1f8.3,/,
     &     ' -     Integration might break down.',/,
     &     '<- <subroutine runalpha>'
     &     )

      END

C-}}}
C-{{{ subroutine rhs:

      subroutine rhs(logmumu0,api,ainteg)
C..
c..   RG-equation:   (d api)/(d log(mu^2)) = api*beta(api)
C..
      implicit real*8 (a-h,o-z)
      integer nrhs
      real*8 api(*),ainteg(*),logmumu0
      common /bfunc/ beta0,beta1,beta2,beta3
      COMMON nrhs
      nrhs=nrhs+1
      ainteg(1) = api(1)*(- beta0*api(1) - beta1*api(1)**2 - beta2
     &     *api(1)**3 - beta3*api(1)**4)
      end

C-}}}
C-{{{ subroutine inibeta:

      subroutine inibeta(nf,nloopin)
C..
c..   initialize beta function
C..
      implicit real*8 (a-h,o-z)
      real*8 nf
      data z3/1.2020569031595942853997/
      common /bfunc/ beta0,beta1,beta2,beta3

      beta0 = (33 - 2*nf)/12.d0
      beta1 = (102 - (38*nf)/3.d0)/16.d0
      beta2 = (2857/2.d0 - (5033*nf)/18.d0 + (325*nf**2)/54.d0)/64.d0
      beta3 = (149753/6.d0 + (1093*nf**3)/729.d0 + 3564*z3 + nf**2
     &     *(50065/162.d0 + (6472*z3)/81.d0) - nf*(1078361/162.d0 +
     &     (6508*z3)/27.d0))/256.d0

      
      nloop=nloopin

      if (nloop.gt.4) then
         write(6,*) '-> <subroutine inibeta>:'
         write(6,*)
     &        ' - 5-loop beta function unknown. Using 4-loop instead.'
         write(6,*) '<- <subroutine inibeta>'
         nloop=4
      endif
      if (nloop.lt.4) then
         beta3 = 0d0
         if (nloop.lt.3) then
            beta2 = 0d0
            if (nloop.lt.2) then
               beta1 = 0d0
               if (nloop.lt.1) then
                  beta0=0d0
               endif
            endif
         endif
      endif
      end

C-}}}

C-}}}
C-{{{ subroutine runmass:

      subroutine runmass(mass0,api0,apif,nf,nloop,massout)
c..
c..   evaluates the running of the MS-bar quark mass
c..   by expanding the equation
c..   
c..   m(mu) = m(mu0) * exp( \int_a0^af dx gammam(x)/x/beta(x) )
c..   
c..   in terms of alpha_s. The results agree with RunDec.m.
c..   
c..   
c..   Input:
c..   ------
c..   mass0  :  m(mu0)
c..   api0   :  alpha_s(mu0)/pi
c..   apif   :  alpha_s(muf)/pi
c..   nf     :  number of flavors
c..   nloop  :  order of calculation (nloop=1..4)
c..
c..   Output:
c..   -------
c..   massout:  m(muf)
c..   
      implicit real*8 (a-h,o-z)
      real*8 mass0,massout,massfun
      real*8 nf
      external massfun
      parameter(accmass=1.d-6)
      common /bfunc/ beta0,beta1,beta2,beta3
      common /gfunc/ gamma0,gamma1,gamma2,gamma3

      if (nloop.eq.0) then
         massout = mass0
         return
      endif

      call inigamma(nf,nloop)
      call inibeta(nf,nloop)

      bb1 = beta1/beta0
      bb2 = beta2/beta0
      bb3 = beta3/beta0

      cc0 = gamma0/beta0
      cc1 = gamma1/beta0
      cc2 = gamma2/beta0
      cc3 = gamma3/beta0

      cfunc1 = 1.d0
      cfunc2 = cc1 - bb1*cc0
      cfunc3 = 1/2.d0*((cc1-bb1*cc0)**2 + cc2 - bb1*cc1 + bb1**2*cc0 -
     &     bb2*cc0)
      cfunc4 = (1/6*(cc1 - bb1*cc0)**3 + 1/2*(cc1 - bb1*cc0)*(cc2 - bb1
     &     *cc1 + bb1**2*cc0 - bb2*cc0) + 1/3*(cc3 - bb1*cc2 + bb1**2
     &     *cc1 - bb2*cc1 - bb1**3*cc0 + 2*bb1*bb2*cc0 - bb3*cc0))

      if (nloop.lt.4) then
         cfunc4 = 0.d0
         if (nloop.lt.3) then
            cfunc3 = 0.d0
            if (nloop.lt.2) then
               cfunc2 = 0.d0
               if (nloop.lt.1) then
                  cfunc1 = 0.d0
               endif
            endif
         endif
      endif

      cfuncmu0 = cfunc1 + cfunc2*api0 + cfunc3*api0**2 + cfunc4*api0**3
      cfuncmuf = cfunc1 + cfunc2*apif + cfunc3*apif**2 + cfunc4*apif**3


      massout = mass0*(apif/api0)**cc0*cfuncmuf/cfuncmu0
      
      return
      end

C-}}}
C-{{{ subroutine inigamma:

      subroutine inigamma(nfin,nloopin)
C
C     initialize beta function
C
      implicit real*8 (a-h,o-z)
      real*8 nf,nfin
      data z3/1.2020569031595942853997/,
     &     z5/1.0369277551433699263/,
     &     pi/3.1415926535897932381/
      common /gfunc/ gamma0,gamma1,gamma2,gamma3

      nf = nfin

      gamma0 = 1.d0
      gamma1 = (67.33333333333333d0 - (20*nf)/9.d0)/16.d0
      gamma2 = (1249.d0 - (140*nf**2)/81.d0 + 2*nf*(-20.59259259259259d0
     &     - 48*z3) +(8*nf*(-46 + 48*z3))/9.d0)/64.d0
      gamma3 = (28413.91975308642d0 + (135680*z3)/27.d0 + nf**3*(-1
     &     .3662551440329218d0 + (64*z3)/27.d0) + nf**2*(21
     &     .57201646090535d0 - (16*Pi**4)/27.d0 + (800*z3)/9.d0) - 8800
     &     *z5 + nf*(-3397.1481481481483d0 + (88*Pi**4)/9.d0 - (34192
     &     *z3)/9.d0 + (18400*z5)/9.d0))/256.d0

      nloop=nloopin

      if (nloop.gt.4) then
         write(6,*) '-> <subroutine inigamma>:'
         write(6,*)
     &        ' - 5-loop gamma function unknown. Using 4-loop instead.'
         write(6,*) '<- <subroutine inigamma>'
         nloop=4
      endif
      if (nloop.lt.4) then
         gamma3 = 0d0
         if (nloop.lt.3) then
            gamma2 = 0d0
            if (nloop.lt.2) then
               gamma1 = 0d0
               if (nloop.lt.1) then
                  gamma0 = 0d0
               endif
            endif
         endif
      endif
      end


