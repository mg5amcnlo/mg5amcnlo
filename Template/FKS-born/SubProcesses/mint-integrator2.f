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
c imode=-1:
c same as imode=0 as far as this routine is concerned, except for the
c fact that a grid is read at the beginning (rather than initialized).
c The return value of imode will be zero.
c
c imode=0:
c When called with imode=0 the routine integrates the absolute value of
c the function and sets up a grid xgrid(0:50,ndim) such that in each
c ndim-1 dimensional slice (i.e. xgrid(m-1,n)<xx(n)<xgrid(m,n)) the
c contribution of the integral is the same the array xgrid is setup at
c this stage; ans and err are the integral and its error
c
c imode=1 (in fact #0)
c When called with imode=1, the routine performs the integral of the
c function fun using the grid xgrid. If some number in the array ifold,
c (say, ifold(n)) is different from 1, it must be a divisor of 50, and
c the 50 intervals xgrid(0:50,n) are grouped into ifold(n) groups, each
c group containing 50/ifold(n) nearby intervals. For example, if
c ifold(1)=5, the 50 intervals for the first dimension are divided in 5
c groups of 10. The integral is then performed by folding on top of each
c other these 5 groups. Suppose, for example, that we choose a random
c point in xx(1) = xgrid(2,1)+x*(xgrid(3,1)-xgrid(2,1)), in the group of
c the first 5 interval.  we sum the contribution of this point to the
c contributions of points
c xgrid(2+m*10,1)+x*(xgrid(3+m*10,1)-xgrid(2+m*10,1)), with m=1,...,4.
c In the sequence of calls to the function fun, the call for the first
c point is performed with ifirst=0, and that for all subsequent points
c with ifirst=1, so that the function can avoid to compute quantities
c that only depend upon dimensions that have ifold=1, and do not change
c in each group of folded call. The values returned by fun in a sequence
c of folded calls with ifirst=0 and ifirst=1 are not used. The function
c itself must accumulate the values, and must return them when called
c with ifirst=2.
c 

      subroutine mint(fun,ndim,ncalls0,nitmax,imode,
     #     xgrid,xint,ymax,ans_abs,err_abs,ans_sgn,err_sgn)
c imode=0: integrate and adapt the grid
c imode=1: frozen grid, compute the integral and the upper bounds
c others: same as 1 (for now)
      implicit none
      include "mint.inc"
      integer ncalls0,ndim,nitmax,imode
      real * 8 fun,xgrid(0:nintervals,ndim),xint,ymax(nintervals,ndim),
     #  ans_abs,err_abs,ans_sgn,err_sgn
      real * 8 x(ndimmax),vol
      real * 8 xacc(0:nintervals,ndimmax)
      integer icell(ndimmax),ncell(ndimmax)
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      integer nhits(1:nintervals,ndimmax)
      real * 8 rand(ndimmax)
      real * 8 dx(ndimmax),f_abs,f_sgn,vtot_abs,etot_abs,vtot_sgn
     &     ,etot_sgn,prod,f1
      integer kdim,kint,kpoint,nit,ncalls,ibin,iret,nintcurr,ifirst
      real * 8 ran3
      external ran3,fun
c Set to true to use more evenly distributed random numbers (imode=0
c only)
      logical even
c
      ncalls=ncalls0
      if(imode.eq.-1) then
         even=.true.
         imode=0
         do kdim=1,ndim
            ifold(kdim)=1
         enddo
      elseif(imode.eq.0) then
         even=.true.
         do kdim=1,ndim
            ifold(kdim)=1
            do kint=0,nintervals
               xgrid(kint,kdim)=dble(kint)/nintervals
            enddo
         enddo
      elseif(imode.eq.1) then
         even=.false.
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
      if (even) then
c Uses more evenly distributed random numbers. This overwrites the
c number of calls
         call initialize_even_random_numbers(ncalls0,ndim,ncalls)
         write (*,*) 'Update ncalls: ',ncalls0,' --> ',ncalls
      endif
      nit=0
      ans_abs=0
      err_abs=0
      ans_sgn=0
      err_sgn=0
 10   continue
      nit=nit+1
      if (nit.gt.nitmax/2+1) then
         even=.false.
      endif
      if(nit.gt.nitmax) then
         if(imode.eq.0) xint=ans_abs
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
      vtot_abs=0
      etot_abs=0
      vtot_sgn=0
      etot_sgn=0
      do kpoint=1,ncalls
c find random x, and its random cell
         do kdim=1,ndim
            kfold(kdim)=1
c if(even), we should compute the ncell and the rand from the ran3()
            if (even) then
               rand(kdim)=ran3(even)
               ncell(kdim)= min(int(rand(kdim)*nintervals)+1,
     &              nintervals)
               rand(kdim)=rand(kdim)*nintervals-(ncell(kdim)-1)
            else
cRF fix here if ran3() is exactly one
c$$$            ncell(kdim)=nintervals/ifold(kdim)*ran3(even))+1
               ncell(kdim)=min(int(nintervals/ifold(kdim)*ran3(even))+1,
     &              nintervals)
               rand(kdim)=ran3(even)
            endif
         enddo
         f_abs=0
         f_sgn=0
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
            f_sgn=f_sgn+fun(x,vol,ifirst,f1)
            f_abs=f_abs+f1
         else
c this accumulated value will not be used
            f_sgn=f_sgn+fun(x,vol,ifirst,f1)
            f_abs=f_abs+f1
            ifirst=1
            call nextlexi(ndim,ifold,kfold,iret)
            if(iret.eq.0) goto 1
c closing call: accumulated value with correct sign
            f_sgn=fun(x,vol,2,f_abs)
         endif
c
         if(imode.eq.0) then
c accumulate the function in xacc(icell(kdim),kdim) to adjust the grid later
            do kdim=1,ndim
               xacc(icell(kdim),kdim)=xacc(icell(kdim),kdim)+f_abs
            enddo
         else
c update the upper bounding envelope
            prod=1
            do kdim=1,ndim
               prod=prod*ymax(ncell(kdim),kdim)
            enddo
            prod=(f_abs/prod)
            if(prod.gt.1) then
c This guarantees a 10% increase of the upper bound in this cell
               prod=1+0.1d0/ndim
               do kdim=1,ndim
                  ymax(ncell(kdim),kdim)=ymax(ncell(kdim),kdim)
     #          * prod
               enddo
            endif
         endif
         vtot_abs=vtot_abs+f_abs/ncalls
         etot_abs=etot_abs+f_abs**2/ncalls
         vtot_sgn=vtot_sgn+f_sgn/ncalls
         etot_sgn=etot_abs
      enddo
      if(imode.eq.0) then
c iteration is finished; now rearrange the grid
         do kdim=1,ndim
            call regrid(xacc(0,kdim),xgrid(0,kdim),
     #           nhits(1,kdim),nintervals,nit)
         enddo
      endif
c the abs is to avoid tiny negative values
      etot_abs=sqrt(abs(etot_abs-vtot_abs**2)/ncalls)
      etot_sgn=sqrt(abs(etot_sgn-vtot_sgn**2)/ncalls)
      write(*,*) '|int|=',vtot_abs,etot_abs
      write(*,*) ' int =',vtot_sgn,etot_sgn
      if(nit.eq.1) then
         ans_abs=vtot_abs
         err_abs=etot_abs
         ans_sgn=vtot_sgn
         err_sgn=etot_sgn
      else
c prevent annoying division by zero for nearly zero
c integrands
         if(etot_abs.eq.0.and.err_abs.eq.0) then
            if(ans_abs.eq.vtot_abs) then
               goto 10
            else
               err_abs=abs(vtot_abs-ans_abs)
               etot_abs=abs(vtot_abs-ans_abs)
               err_sgn=err_abs
               etot_sgn=etot_abs
            endif
         elseif(etot_abs.eq.0) then
            etot_abs=err_abs
            etot_sgn=etot_abs
         elseif(err_abs.eq.0) then
            err_abs=etot_abs
            err_sgn=err_abs
         endif
         ans_abs=(ans_abs/err_abs+vtot_abs/etot_abs)/
     &        (1/err_abs+1/etot_abs)
         err_abs=1/sqrt(1/err_abs**2+1/etot_abs**2)
         ans_sgn=(ans_sgn/err_sgn+vtot_sgn/etot_sgn)/
     &        (1/err_sgn+1/etot_sgn)
         err_sgn=err_abs
      endif
c Also improve stats in plots
      call accum
c Do next iteration
      goto 10
      end

      subroutine regrid(xacc,xgrid,nhits,nint,nit)
      implicit none
      integer  nint,nhits(nint),nit
      real * 8 xacc(0:nint),xgrid(0:nint)
      real * 8 xn(nint),r,tiny
      parameter ( tiny=1d-8 )
      integer kint,jint
      logical plot_grid
      parameter (plot_grid=.false.)
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
cRF: Check that we have a reasonable result and update the accumulated
c results if need be
      do kint=1,nint
         if (xacc(kint).le.(xacc(kint-1)+tiny)) then
            write (*,*) 'Accumulated results need adaptation #1:'
            write (*,*) xacc(kint),xacc(kint-1),' become'
            xacc(kint)=xacc(kint-1)+tiny
            write (*,*) xacc(kint),xacc(kint-1)
         endif
      enddo
c it could happen that the change above yielded xacc() values greater
c than 1; should be fixed once more.
      xacc(nint)=1d0
      do kint=1,nint
         if (xacc(nint-kint).ge.(xacc(nint-kint+1)-tiny)) then
            write (*,*) 'Accumulated results need adaptation #2:'
            write (*,*) xacc(nint-kint),xacc(nint-kint+1),' become'
            xacc(nint-kint)=1d0-dble(kint)*tiny
            write (*,*) xacc(nint-kint),xacc(nint-kint+1)
         else
            exit
         endif
      enddo
cend RF

      if (plot_grid) then
         write(11,*) 'set limits x 0 1 y 0 1'
         write(11,*) 0, 0
         do kint=1,nint
            write(11,*) xgrid(kint),xacc(kint)
         enddo
         write(11,*) 'join 1'
      endif

      do kint=1,nint
         r=dble(kint)/dble(nint)

         if (plot_grid) then
            write(11,*) 0, r
            write(11,*) 1, r
            write(11,*) ' join'
         endif

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
         if (plot_grid) then
            write(11,*) xgrid(kint), 0
            write(11,*) xgrid(kint), 1
            write(11,*) ' join'
         endif
      enddo
      if (plot_grid) write(11,*) ' newplot'
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
      include "mint.inc"
      real * 8 fun,xgrid(0:nintervals,ndimmax),
     #         ymax(nintervals,ndimmax),x(ndimmax)
      real * 8 dx(ndimmax)
      integer icell(ndimmax),ncell(ndimmax)
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      real * 8 r,f_sgn,f_abs,f1,ubound,vol,ran3,xmmm(nintervals,ndimmax)
      real * 8 rand(ndimmax)
      external fun,ran3
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
         call increasecnt(' ',imode)
         return
      endif
      mcalls=mcalls+1
 10   continue
      do kdim=1,ndim
         nintcurr=nintervals/ifold(kdim)
         r=ran3(.false.)
         do kint=1,nintcurr
            if(r.lt.xmmm(kint,kdim)) then
               ncell(kdim)=kint
               goto 1
            endif
         enddo
 1       continue
         rand(kdim)=ran3(.false.)
      enddo
      ubound=1
      do kdim=1,ndim
         ubound=ubound*ymax(ncell(kdim),kdim)
      enddo
      do kdim=1,ndim
         kfold(kdim)=1
      enddo
      f_sgn=0
      f_abs=0
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
      f_sgn=f_sgn+fun(x,vol,ifirst,f1)
      f_abs=f_abs+f1
      ifirst=1
      call nextlexi(ndim,ifold,kfold,iret)
      if(iret.eq.0) goto 5
c get final value (x and vol not used in this call)
      f_sgn=fun(x,vol,2,f_abs)
      call increasecnt('another call to the function',imode)
      if (f_abs.eq.0d0) call increasecnt('failed generation cuts',imode)
      if(f_abs.lt.0) then
         write(*,*) 'gen: non positive function'
         stop
      endif
      if(f_abs.gt.ubound) then
         call increasecnt
     &        ('upper bound failure in inclusive cross section',imode)
      endif
      ubound=ubound*ran3(.false.)
      icalls=icalls+1
      if(ubound.gt.f_abs) then
         call increasecnt
     &        ('vetoed calls in inclusive cross section',imode)
         goto 10
      endif
      end


c Dummy subroutine (normally used with vegas when resuming plots)
      subroutine resume()
      end


      subroutine increasecnt(argument,imode)
c Be careful, argument should be at least 15 characters
c long for this subroutine to work properly
      implicit none
      character*(*) argument
      character*15 list(100)
      integer ilist(0:100),i,j,imode
      logical firsttime
      data firsttime/.true./
      save ilist,list

      if (firsttime) then
         ilist(0)=1
         do i=1,100
            ilist(i)=0
            list(i)='               '
         enddo
         firsttime=.false.
      endif

      if(imode.ne.3) then
         i=1
         do while (i.le.ilist(0))
            if(i.eq.ilist(0)) then
               list(i)=argument(1:15)
               ilist(i)=1
               ilist(0)=ilist(0)+1
               goto 14
            endif
            if (argument(1:15).eq.list(i)) then
               ilist(i)=ilist(i)+1
               goto 14
            endif
            i=i+1
            if (i.ge.100) then
               write (*,*) 'error #1 in increasecnt'
               do j=1,ilist(0)
                  write (*,*) list(j),ilist(j)
               enddo
               stop
            endif
         enddo
 14      continue
      else
         do i=1,ilist(0)-1
            write (*,*) list(i),ilist(i)
         enddo
      endif
      end

      double precision function ran3(even)
      implicit none
      double precision ran2,get_ran
      logical even
      external get_ran
      if (even) then
         ran3=get_ran()
      else
         ran3=ran2()
      endif
      return
      end

      subroutine initialize_even_random_numbers(ncalls0,ndim,ncalls)
c Recompute the number of calls. Uses the algorithm from VEGAS
      implicit none
      integer ncalls0,ndim,ncalls,i
      integer dim,ng,npg,k
      common /even_ran/dim,ng,npg,k
      dim=ndim
      ng=(ncalls0/2.)**(1./ndim)
      k=ng**ndim
      npg=ncalls0/k
      if(npg.lt.2)npg=2
      ncalls=npg*k
      write (*,*) ' dim:',dim,' ng:',ng,' npg:',npg,' k:',k
      return
      end


      double precision function get_ran()
      implicit none
      double precision ran2,dng
      external ran2
      logical firsttime
      data firsttime/.true./
      integer dim,ng,npg,k
      common /even_ran/dim,ng,npg,k
      integer maxdim
      parameter (maxdim=100)
      integer iii(maxdim),kkk(maxdim),i,iret
      integer current_dim
      save current_dim,dng,kkk,iii
      if (firsttime) then
         dng=1d0/dble(ng)
         current_dim=0
         do i=1,dim
           iii(i)=ng
           kkk(i)=1
        enddo
        firsttime=.false.
      endif
      current_dim=mod(current_dim,dim)+1
c This is the random number in the hypercube 'k' for current_dim
      get_ran=dng*(ran2()+dble(kkk(current_dim)-1))
c Got random numbers for all dimensions, update kkk() for the next call
      if (current_dim.eq.dim) then
         call nextlexi(dim,iii,kkk,iret)
         if (iret.eq.1) then
            call nextlexi(dim,iii,kkk,iret)
         endif
      endif
      return
      end


