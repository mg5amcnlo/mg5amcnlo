c Integrator Package for POWHEG
c      subroutine mint(fun,ndim,ncalls0,nitmax,imode,
c ndim=number of dimensions
c ncalls0=# of calls per iteration
c nitmax =# of iterations
c fun(xx,www,ifirst): returns the function to be integrated multiplied by www;
c                     xx(1:ndim) are the variables of integration
c                     ifirst=0: normal behaviour
c imode: integer flag
c
c imode=0:
c When called with imode=0 the routine integrates the absolute value of the function
c and sets up a grid xgrid(0:50,ndim) such that in each ndim-1 dimensional slice
c (i.e. xgrid(m-1,n)<xx(n)<xgrid(m,n)) the contribution of the integral is the same
c the array xgrid is setup at this stage; ans and err are the integral and its error 
c
c imode=1 (in fact #0)
c When called with imode=1, the routine performs the integral of the function fun
c using the grid xgrid. If some number in the array ifold, (say, ifold(n))
c is different from 1, it must be a divisor of 50, and the 50 intervals xgrid(0:50,n)
c are grouped into ifold(n) groups, each group containing 50/ifold(n) nearby
c intervals. For example, if ifold(1)=5, the 50 intervals for the first dimension
c are divided in 5 groups of 10. The integral is then performed by folding on top
c of each other these 5 groups. Suppose, for example, that we choose a random point
c in xx(1) = xgrid(2,1)+x*(xgrid(3,1)-xgrid(2,1)), in the group of the first 5 interval.
c we sum the contribution of this point to the contributions of points
c xgrid(2+m*10,1)+x*(xgrid(3+m*10,1)-xgrid(2+m*10,1)), with m=1,...,4.
c In the sequence of calls to the
c function fun, the call for the first point is performed with ifirst=0, and that for
c all subsequent points with ifirst=1, so that the function can avoid to compute
c quantities that only depend upon dimensions that have ifold=1, and do not change
c in each group of folded call. The values returned by fun in a sequence of folded
c calls with ifirst=0 and ifirst=1 are not used. The function itself must accumulate
c the values, and must return them when called with ifirst=2.
c 

      subroutine mint(fun,ndim,ncalls0,nitmax,imode,
     #     xgrid,xint,ymax,ans,err,old_grid,accuracy)
c imode=0: integrate and adapt the grid
c imode=1: frozen grid, compute the integral and the upper bounds
c others: same as 1 (for now)
      implicit none
      logical old_grid
      double precision accuracy
      integer nintervals,ndimmax
      parameter (nintervals=50,ndimmax=20)
      integer ncalls0,ndim,nitmax,imode
      real * 8 fun,xgrid(0:nintervals,ndim),xint,ymax(nintervals,ndim),
     #  ans,err
      real * 8 x(ndimmax),vol
      real * 8 xacc(0:nintervals,ndimmax)
      integer icell(ndimmax),ncell(ndimmax)
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      integer nhits(1:nintervals,ndimmax)
      real * 8 rand(ndimmax)
      real * 8 dx(ndimmax),f,vtot,etot,prod
      integer kdim,kint,kpoint,nit,ncalls,ibin,iret,nintcurr,ifirst
      real * 8 random_mint
      external random_mint,fun
      if(imode.eq.0) then
         do kdim=1,ndim
            ifold(kdim)=1
            if (.not. old_grid) then
              do kint=0,nintervals
                 xgrid(kint,kdim)=dble(kint)/nintervals
              enddo
            endif
         enddo
      elseif(imode.eq.1) then
         do kdim=1,ndim
            nintcurr=nintervals/ifold(kdim)
            if(nintcurr*ifold(kdim).ne.nintervals) then
               write(*,*)
     # 'mint: the values in the ifold array shoud be divisors of',
     #  nintervals
               stop
            endif
            do kint=1,nintcurr
               ymax(kint,kdim)=
     #              xint**(1d0/ndim)
            enddo
         enddo
      endif
      ncalls=ncalls0
      nit=0
      ans=0
      err=0
 10   continue
      nit=nit+1
      if (nit.ge.2.and.ans.eq.0d0) then
         xint=0d0
         return
      endif
      if(nit.gt.nitmax.or.(err/ans).lt.accuracy) then
         if(imode.eq.0) xint=ans
         return
      endif
      if(imode.eq.0) then
         do kdim=1,ndim
            do kint=0,nintervals
               xacc(kint,kdim)=0
               if(kint.gt.0) then
                  nhits(kint,kdim)=0
               endif
            enddo
         enddo
      endif
      vtot=0
      etot=0
      do kpoint=1,ncalls
c find random x, and its random cell
         do kdim=1,ndim
            kfold(kdim)=1
            ncell(kdim)=nintervals/ifold(kdim)*random_mint(25-kdim)+1
            rand(kdim)=random_mint(kdim)
         enddo
         f=0
         ifirst=0
 1       continue
         vol=1
         do kdim=1,ndim
            nintcurr=nintervals/ifold(kdim)
            icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
            ibin=icell(kdim)
            dx(kdim)=xgrid(icell(kdim),kdim)-xgrid(icell(kdim)-1,kdim)
            vol=vol*dx(kdim)*nintcurr
            x(kdim)=xgrid(icell(kdim)-1,kdim)+rand(kdim)*dx(kdim)
            if(imode.eq.0) nhits(ibin,kdim)=nhits(ibin,kdim)+1
         enddo
c contribution to integral
         if(imode.eq.0) then
            f=abs(fun(x,vol,ifirst))+f
         else
c this accumulated value will not be used
            f=fun(x,vol,ifirst)+f
            ifirst=1
            call nextlexi(ndim,ifold,kfold,iret)
            if(iret.eq.0) goto 1
c closing call: accumulated value with correct sign
            f=fun(x,vol,2)
         endif
c
         if(imode.eq.0) then
c accumulate the function in xacc(icell(kdim),kdim) to adjust the grid later
            do kdim=1,ndim
               xacc(icell(kdim),kdim)=xacc(icell(kdim),kdim)+f
            enddo
         else
c update the upper bounding envelope
            prod=1
            do kdim=1,ndim
               prod=prod*ymax(ncell(kdim),kdim)
            enddo
            prod=(f/prod)
            if(prod.gt.1) then
c This guarantees a 10% increase of the upper bound in this cell
               prod=1+0.1d0/ndim
               do kdim=1,ndim
                  ymax(ncell(kdim),kdim)=ymax(ncell(kdim),kdim)
     #          * prod
               enddo
            endif
         endif
         vtot=vtot+f/ncalls
         etot=etot+f**2/ncalls
      enddo
      if(imode.eq.0) then
c iteration is finished; now rearrange the grid
         do kdim=1,ndim
            call regrid(xacc(0,kdim),xgrid(0,kdim),
     #           nhits(1,kdim),nintervals,nit)
         enddo
      endif
c the abs is to avoid tiny negative values
      etot=sqrt(abs(etot-vtot**2)/ncalls)
      write(*,*) vtot,etot
      if(nit.eq.1) then
         ans=vtot
         err=etot
      else
c prevent annoying division by zero for nearly zero
c integrands
         if(etot.eq.0.and.err.eq.0) then
            if(ans.eq.vtot) then
               goto 10
            else
               err=abs(vtot-ans)
               etot=abs(vtot-ans)
            endif
         elseif(etot.eq.0) then
            etot=err
         elseif(err.eq.0) then
            err=etot
         endif
         ans=(ans/err+vtot/etot)/(1/err+1/etot)
         err=1/sqrt(1/err**2+1/etot**2)
      endif
      goto 10
      end

      subroutine regrid(xacc,xgrid,nhits,nint,nit)
      implicit none
      integer  nint,nhits(nint),nit
      real * 8 xacc(0:nint),xgrid(0:nint)
      real * 8 xn(100),r
      integer kint,jint
      do kint=1,nint
c xacc (xerr) already containe a factor equal to the interval size
c Thus the integral of rho is performed by summing up
         if(nhits(kint).ne.0) then
            xacc(kint)= xacc(kint-1)
     #           + abs(xacc(kint))/nhits(kint)
         else
            xacc(kint)=xacc(kint-1)
         endif
      enddo
      do kint=1,nint
         xacc(kint)=xacc(kint)/xacc(nint)
      enddo
c      write(11,*) 'set limits x 0 1 y 0 1'
c      write(11,*) 0, 0
c      do kint=1,nint
c         write(11,*) xgrid(kint),xacc(kint)
c      enddo
c      write(11,*) 'join 0'

      do kint=1,nint
         r=dble(kint)/nint

c         write(11,*) 0, r
c         write(11,*) 1, r
c         write(11,*) ' join'

         do jint=1,nint
            if(r.lt.xacc(jint)) then
               xn(kint)=xgrid(jint-1)+(r-xacc(jint-1))
     #        /(xacc(jint)-xacc(jint-1))*(xgrid(jint)-xgrid(jint-1))
               goto 11
            endif
         enddo
         if(jint.ne.nint+1.and.kint.ne.nint) then
            write(*,*) ' error',jint,nint
            stop
         endif
         xn(nint)=1
 11      continue
      enddo
      do kint=1,nint
         xgrid(kint)=xn(kint)
c         xgrid(kint)=(xn(kint)+2*xgrid(kint))/3
c         xgrid(kint)=(xn(kint)+xgrid(kint)*log(dble(nit)))
c     #        /(log(dble(nit))+1)
c         write(11,*) xgrid(kint), 0
c         write(11,*) xgrid(kint), 1
c         write(11,*) ' join'
      enddo
c      write(11,*) ' newplot'
      end

      subroutine nextlexi(ndim,iii,kkk,iret)
c kkk: array of integers 1 <= kkk(j) <= iii(j), j=1,ndim
c at each call iii is increased lexicographycally.
c for example, starting from ndim=3, kkk=(1,1,1), iii=(2,3,2)
c subsequent calls to nextlexi return
c         kkk(1)      kkk(2)      kkk(3)    iret
c 0 calls   1           1           1       0
c 1         1           1           2       0    
c 2         1           2           1       0
c 3         1           2           2       0
c 4         1           3           1       0
c 5         1           3           2       0
c 6         2           1           1       0
c 7         2           1           2       0
c 8         2           2           1       0
c 9         2           2           2       0
c 10        2           3           1       0
c 11        2           3           2       0
c 12        2           3           2       1
      implicit none
      integer ndim,iret,kkk(ndim),iii(ndim)
      integer k
      k=ndim
 1    continue
      if(kkk(k).lt.iii(k)) then
         kkk(k)=kkk(k)+1
         iret=0
         return
      else
         kkk(k)=1
         k=k-1
         if(k.eq.0) then
            iret=1
            return
         endif
         goto 1
      endif
      end


      subroutine gen(fun,ndim,xgrid,ymax,imode,x)
c imode=0 to initialize
c imode=1 to generate
c imode=3 store generation efficiency in x(1)
      implicit none
      integer ndim,imode
      integer nintervals,ndimmax
      parameter (nintervals=50,ndimmax=20)
      real * 8 fun,xgrid(0:nintervals,ndim),
     #         ymax(nintervals,ndim),x(ndim)
      real * 8 dx(ndimmax)
      integer icell(ndimmax),ncell(ndimmax)
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      real * 8 r,f,ubound,vol,random_mint,xmmm(nintervals,ndimmax)
      real * 8 rand(ndimmax)
      external fun,random_mint
      integer icalls,mcalls,kdim,kint,nintcurr,iret,ifirst
      save icalls,mcalls,xmmm
      if(imode.eq.0) then
         do kdim=1,ndim
            nintcurr=nintervals/ifold(kdim)
            xmmm(1,kdim)=ymax(1,kdim)
            do kint=2,nintcurr
               xmmm(kint,kdim)=xmmm(kint-1,kdim)+
     #              ymax(kint,kdim)
            enddo
            do kint=1,nintcurr
               xmmm(kint,kdim)=xmmm(kint,kdim)/xmmm(nintcurr,kdim)
            enddo
         enddo
         icalls=0
         mcalls=0
         return
      elseif(imode.eq.3) then
         if(icalls.gt.0) then
            x(1)=dble(mcalls)/icalls
         else
            x(1)=-1
         endif
         return
      endif
      mcalls=mcalls+1
 10   continue
      do kdim=1,ndim
         nintcurr=nintervals/ifold(kdim)
         r=random_mint(kdim)
         do kint=1,nintcurr
            if(r.lt.xmmm(kint,kdim)) then
               ncell(kdim)=kint
               goto 1
            endif
         enddo
 1       continue
         rand(kdim)=random_mint(kdim)
      enddo
      ubound=1
      do kdim=1,ndim
         ubound=ubound*ymax(ncell(kdim),kdim)
      enddo
      do kdim=1,ndim
         kfold(kdim)=1
      enddo
      f=0
      ifirst=0
 5    continue
      vol=1
      do kdim=1,ndim
         nintcurr=nintervals/ifold(kdim)
         icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
         dx(kdim)=xgrid(icell(kdim),kdim)-xgrid(icell(kdim)-1,kdim)
         vol=vol*dx(kdim)*nintervals/ifold(kdim)
         x(kdim)=xgrid(icell(kdim)-1,kdim)+rand(kdim)*dx(kdim)
      enddo
      f=f+fun(x,vol,ifirst)
      ifirst=1
      call nextlexi(ndim,ifold,kfold,iret)
      if(iret.eq.0) goto 5
c get final value (x and vol not used in this call)
      f=fun(x,vol,2)
      if(f.lt.0) then
         write(*,*) 'gen: non positive function'
         stop
      endif
      if(f.gt.ubound) then
c         call increasecnt
c     #       ('upper bound failure in inclusive cross section')
       write(*,*) "problem"
      endif
      ubound=ubound*random_mint(1)
      icalls=icalls+1
      if(ubound.gt.f) then
       write(*,*) "problem"
c         call increasecnt
c     #       ('vetoed calls in inclusive cross section')
         goto 10
      endif
      end

      double precision function random_mint(j)
      implicit none
      integer j
      double precision x

      call ntuple(x,0.d0,1.d0,j)
      random_mint=x

      return
      end
