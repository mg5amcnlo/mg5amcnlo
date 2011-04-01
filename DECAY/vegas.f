      SUBROUTINE vegas(region,ndim,fxn,init,ncall,itmx,nprn,tgral,sd,
     *chi2a)
      INTEGER init,itmx,ncall,ndim,nprn,NDMX,MXDIM
      REAL tgral,chi2a,sd,region(2*ndim),fxn,ALPH,TINY
      PARAMETER (ALPH=1.5,NDMX=50,MXDIM=10,TINY=1.e-30)
      EXTERNAL fxn
CU    USES fxn,xran1,rebin
      INTEGER i,idum,it,j,k,mds,nd,ndo,ng,npg,ia(MXDIM),kg(MXDIM)
      REAL calls,dv2g,dxg,f,f2,f2b,fb,rc,ti,tsi,wgt,xjac,xn,xnd,xo,
     *d(NDMX,MXDIM),di(NDMX,MXDIM),dt(MXDIM),dx(MXDIM),r(NDMX),x(MXDIM),
     *xi(NDMX,MXDIM),xin(NDMX),xran1,rndn
      EXTERNAL XRAN1
      DOUBLE PRECISION schi,si,swgt
      DATA idum/1/ ! DOES NOT AFFECT PREVIOUS SETTINGS
c
C     GLOBAL
C
      INTEGER NDIM1
      REAL    region1(2*MXDIM)
      COMMON /VEGAS_PAR1/NG,NDIM1
      COMMON /VEGAS_PAR2/region1,xi,xnd,dx
C
      SAVE ! make variables static
c
      if(init.le.0)then         !enter here on a cold start
        NDIM1=NDIM
        do j=1,2*ndim
           region1(j)=region(j)
        enddo
        mds=1
        ndo=1
        do 11 j=1,ndim
          xi(1,j)=1.
11      continue
      endif
c
      if (init.le.1)then !enter here to inherit the grid but not the answers
        si=0D0
        swgt=0D0
        schi=0D0
      endif 
c
      if (init.le.2)then !enter here to inherit the grid and the answers
        nd=NDMX
        ng=1
c
        if(mds.ne.0)then !setup for stratification
          ng=(ncall/2.+0.25)**(1./ndim) !number of calls per dimension
          mds=1
          if((2*ng-NDMX).ge.0)then
            mds=-1
            npg=ng/NDMX+1
            nd=ng/npg
            ng=npg*nd
          endif
        endif
        k=ng**ndim
        npg=max(ncall/k,2)
        calls=npg*k
cfax: returns the actual number of calls
        ncall=int(calls)
        dxg=1./ng
        dv2g=(calls*dxg**ndim)**2/npg/npg/(npg-1.)
        xnd=nd
        dxg=dxg*xnd
        xjac=1./calls
c        WRITE (*,*) 'NG,NDIM,K ',NG,NDIM,K
c        WRITE (*,*) 'NPG,CALLS,NCALL ',NPG,CALLS,NCALL
c        WRITE (*,*) 'ND,DXG     ',ND,DXG
        do 12 j=1,ndim
          dx(j)=region(j+ndim)-region(j)
          xjac=xjac*dx(j)
12      continue
c-- do the binning if necessary
        if(nd.ne.ndo)then
          do 13 i=1,nd
            r(i)=1.
13        continue
          do 14 j=1,ndim
            call rebin(ndo/xnd,nd,r,xin,xi(1,j))
14        continue
          ndo=nd
        endif
        if(nprn.ge.0) write(*,200) ndim,calls,it,itmx,nprn,ALPH,mds,nd
      endif 
c
c     MAIN ITERATION LOOP
c     if init>=3 it enters here
c
      do 28 it=1,itmx
        ti=0.
        tsi=0.
        do 16 j=1,ndim
          kg(j)=1
          do 15 i=1,nd
            d(i,j)=0.
            di(i,j)=0.
15        continue
16      continue
10      continue
          fb=0.
          f2b=0.
          do 19 k=1,npg
            wgt=xjac
            do 17 j=1,ndim
cfax: avoid random numbers exactly equal to 0. or 1.
 303           rndn=xran1(idum)
               if(rndn.eq.1e0.or.rndn.eq.0e0) goto 303
              xn=(kg(j)-rndn)*dxg+1.
c              write(*,*) k,kg(j),xn
              ia(j)=max(min(int(xn),NDMX),1)
              if(ia(j).gt.1)then
                xo=xi(ia(j),j)-xi(ia(j)-1,j)
                rc=xi(ia(j)-1,j)+(xn-ia(j))*xo
              else
                xo=xi(ia(j),j)
                rc=(xn-ia(j))*xo
              endif
              x(j)=region(j)+rc*dx(j)
              wgt=wgt*xo*xnd
17          continue
            f=wgt*fxn(x,wgt)
            f2=f*f
            fb=fb+f
            f2b=f2b+f2
            do 18 j=1,ndim
              di(ia(j),j)=di(ia(j),j)+f
              if(mds.ge.0) d(ia(j),j)=d(ia(j),j)+f2
18          continue
19        continue
          f2b=sqrt(f2b*npg)
          f2b=(f2b-fb)*(f2b+fb)
          if (f2b.le.0.) f2b=TINY
          ti=ti+fb
          tsi=tsi+f2b
          if(mds.lt.0)then
            do 21 j=1,ndim
              d(ia(j),j)=d(ia(j),j)+f2b
21          continue
          endif
c this is just a counter: total number of points thrown is ng**ndim
        do 22 k=ndim,1,-1
          kg(k)=mod(kg(k),ng)+1
          if(kg(k).ne.1) goto 10
22      continue
c
c     all points for this iteration have been thrown.
c     now it calculates the results for this iteration
c     and the cumulative result,chi^2
c
        tsi=tsi*dv2g  
        wgt=1./tsi
        si=si+dble(wgt)*dble(ti)
        schi=schi+dble(wgt)*dble(ti)**2
        swgt=swgt+dble(wgt)
        tgral=si/swgt
        chi2a=max((schi-si*tgral)/(it-.99d0),0.d0)
        sd=sqrt(1./swgt)
        tsi=sqrt(tsi)
        if(nprn.ge.0)then
           write(*,201) it,ti,tsi
          if(nprn.ne.0)then
            do 23 j=1,ndim
              write(*,202) j,(xi(i,j),di(i,j),i=1+nprn/2,nd,nprn)
23          continue
          endif
        endif
c
c     refines the grid after the iteration
c
        do 25 j=1,ndim
          xo=d(1,j)
          xn=d(2,j)
          d(1,j)=(xo+xn)/2.
          dt(j)=d(1,j)
          do 24 i=2,nd-1
            rc=xo+xn
            xo=xn
            xn=d(i+1,j)
            d(i,j)=(rc+xn)/3.
            dt(j)=dt(j)+d(i,j)
24        continue
          d(nd,j)=(xo+xn)/2.
          dt(j)=dt(j)+d(nd,j)
25      continue
c-- slow down the evolution
        do 27 j=1,ndim
          rc=0.
          do 26 i=1,nd
            if(d(i,j).lt.TINY) d(i,j)=TINY
            r(i)=((1.-d(i,j)/dt(j))/(log(dt(j))-log(d(i,j))))**ALPH
            rc=rc+r(i)
26        continue
          call rebin(rc/xnd,nd,r,xin,xi(1,j))
27      continue
c
28    continue !main iteration loop
c
      return
c
c     print out formats
c
200   FORMAT(/' input param. for vegas:  ndim=',i3,'  calls=',
     *f8.0/24x,'  it=',i5,'  itmx=',i5/24x,'  nprn=',i3,'  alph=',
     *f5.2/24x,'  mds=',i3,'   nd=',i4/)
c200   FORMAT(/' input parameters for vegas:  ndim=',i3,'  ncall=',
c     *f8.0/28x,'  it=',i5,'  itmx=',i5/28x,'  nprn=',i3,'  alph=',
c     *f5.2/28x,'  mds=',i3,'   nd=',i4/(30x,'xl(',i2,')= ',g11.4,' xu(',
c     *i2,')= ',g11.4))
201   FORMAT(' iter. no.',I3,': ','integral =',g14.7,' +/- ',g9.2)
c201   FORMAT(/' iteration no.',I3,': ','integral =',g14.7,'+/- ',g9.2/
c     *' all iterations:   integral =',g14.7,'+/- ',g9.2,' chi**2/it',
c     *'n =',g9.2)
202   FORMAT(/' data for axis ',I2/'    X       delta i       ',
     *'   x       delta i       ','    x       delta i       ',/(1x,
     *f7.5,1x,g11.4,5x,f7.5,1x,g11.4,5x,f7.5,1x,g11.4))
c
      END



      SUBROUTINE rebin(rc,nd,r,xin,xi)
      INTEGER nd
      PARAMETER (NDMX=50)
      REAL rc,r(NDMX),xi(NDMX),xin(NDMX)
      INTEGER i,k
      REAL dr,xn,xo
      k=0
      xn=0.
      dr=0.
      do 11 i=1,nd-1
1       if(rc.gt.dr)then
          k=k+1
          dr=dr+r(k)
          xo=xn
          xn=xi(k)
        goto 1
        endif
        dr=dr-rc
        xin(i)=xn-(xn-xo)*dr/r(k)
11    continue
      do 12 i=1,nd-1
        xi(i)=xin(i)
12    continue
      xi(nd)=1.
      return
      END
