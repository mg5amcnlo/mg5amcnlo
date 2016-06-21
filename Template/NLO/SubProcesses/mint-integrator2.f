ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c MINT Integrator Package
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Original version by Paolo Nason (for POWHEG (BOX))
c Modified by Rikkert Frederix (for aMC@NLO)
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
c Added the posibility to keep track of more than one integral:

c nintegrals=1 : the function that is used to update the grids. This is
c the ABS cross section. If imode.eq.1, this does not contain the
c virtual corrections because for them a separate maximum is kept using (5).
c nintegrals=2 : the actual cross section. This includes virtual corrections.
c nintegrals=3 : the cross section from the M_Virt/M_Born ratio alone:
c this defines the average virtual that is added to each phase-space
c point
c nintegrals=4 : the cross section of the actual virtual minus the
c average virtual. This is used to determine the fraction of phase-space
c points for which we include the virtual.
c nintegrals=5 : abs of 3
c nintegrals=6 : born*alpha_S/2Pi
c
      subroutine mint(fun,ndim,ncalls0,nitmax,imode,xgrid,ymax,ymax_virt
     $     ,ans,unc,chi2)
c imode= 0: integrate and adapt the grid
c imode= 1: frozen grid, compute the integral and the upper bounds
c imode=-1: same as imode=0, but use previously generated grids
c others: same as 1 (for now)
      implicit none
      include "mint.inc"
      include "FKSParams.inc"
      integer i,j,ncalls0,ndim,nitmax,imode
      real * 8 fun,xgrid(0:nintervals,ndimmax),xint,ymax(nintervals
     $     ,ndimmax),ans(nintegrals),unc(nintegrals),ans3(nintegrals,3)
     $     ,unc3(nintegrals,3),ans_l3(nintegrals),unc_l3(nintegrals)
     $     ,chi2_l3(nintegrals)
      real * 8 xint_virt,ymax_virt
      real * 8 x(ndimmax),vol
      real * 8 xacc(0:nintervals,ndimmax)
      integer icell(ndimmax),ncell(ndimmax),ncell_virt
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      integer nhits(nintervals,ndimmax)
      real * 8 rand(ndimmax)
      real * 8 dx(ndimmax),f(nintegrals),vtot(nintegrals)
     $     ,etot(nintegrals),prod,f1(nintegrals),chi2(nintegrals)
     $     ,efrac(nintegrals),dummy
      integer kdim,kint,kpoint,nit,ncalls,iret,nintcurr,nintcurr_virt
     $     ,ifirst,nit_included,kpoint_iter,non_zero_point(nintegrals)
     $     ,ntotcalls(nintegrals),nint_used,nint_used_virt,min_it
      real * 8 ran3
      external ran3,fun
      logical even,double_events,bad_iteration
      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction
      character*13 title(nintegrals)
      data title(1)/'ABS integral '/
      data title(2)/'Integral     '/
      data title(3)/'Virtual      '/
      data title(4)/'Virtual ratio'/
      data title(5)/'ABS virtual  '/
      data title(6)/'Born*ao2pi   '/
c APPLgrid switch
      integer iappl
      common /for_applgrid/ iappl
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
c if ncalls0 is greater than 0, use the default running, i.e. do not
c double the events after each iteration as well as use a fixed number
c of intervals in the grids.
      if (ncalls0.gt.0) then
         double_events=.false.
         nint_used=nintervals
         nint_used_virt=nintervals_virt
      else
c if ncalls0.le.0, reset it and double the events per iteration
         ncalls0=80*ndim
         double_events=.true.
         if (imode.eq.1 .or. imode.eq.-1) then
            nint_used=nintervals
            nint_used_virt=nintervals_virt
         else
            nint_used=min_inter
            nint_used_virt=min_inter
         endif
      endif
      bad_iteration=.false.
c
      ncalls=0  ! # PS points (updated below)
      if(imode.eq.-1) then
c Grids read from file
         even=.true.
         imode=0
         min_it=min_it0
         do kdim=1,ndim
            ifold(kdim)=1
         enddo
      elseif(imode.eq.0) then
c Initialize grids
         even=.true.
         min_it=min_it0
         do kdim=1,ndim
            ifold(kdim)=1
            do kint=0,nint_used
               xgrid(kint,kdim)=dble(kint)/nint_used
            enddo
         enddo
         call init_ave_virt(nint_used_virt,ndim)
      elseif(imode.eq.1) then
c Initialize upper bounding envelope
         xint=ans(1)
         xint_virt=ans(5)
         even=.false.
         min_it=min_it1
         do kdim=1,ndim
            nintcurr=nint_used/ifold(kdim)
            nintcurr_virt=nint_used_virt/ifold(kdim)
            if(nintcurr*ifold(kdim).ne.nint_used .or.
     &         nintcurr_virt*ifold(kdim).ne.nint_used_virt) then
               write(*,*) 'mint: the values in the ifold array'/
     $              /'shoud be divisors of',nint_used,'and'
     $              ,nint_used_virt
               stop
            endif
            do kint=1,nintcurr
               ymax(kint,kdim)=xint**(1d0/ndim)
            enddo
         enddo
         ymax_virt=xint_virt
      endif
      nit=0
      nit_included=0
      do i=1,nintegrals
         ans(i)=0d0
         unc(i)=0d0
         do j=1,3
            ans3(i,j)=0d0
            unc3(i,j)=0d0
         enddo
      enddo
c Main loop over the iterations
 10   continue
c This makes sure that current stdout is written to log.txt and cache is
c emptied.
      call flush(6)
      if(nit.ge.nitmax) then
c We did enough iterations, update arguments and return
         if(imode.eq.0) xint=ans(1)
         if(imode.eq.0) xint_virt=ans(5)
         if (nit_included.ge.2) then
            chi2(1)=chi2(1)/dble(nit_included-1)
         else
            chi2(1)=0d0
         endif
         write (*,*) '-------'
         ncalls0=ncalls*kpoint_iter ! return number of points used
         if (double_events) then
            nitmax=2
         else
            nitmax=nit_included
         endif
         return
      endif
      nit=nit+1
      write (*,*) '------- iteration',nit
      if (even .and. ncalls.ne.ncalls0) then
c Uses more evenly distributed random numbers. This overwrites the
c number of calls
         call initialize_even_random_numbers(ncalls0,ndim,ncalls)
         write (*,*) 'Update # PS points (even): ',ncalls0,' --> '
     &        ,ncalls
      elseif (ncalls0.ne.ncalls) then
         ncalls=ncalls0
         write (*,*) 'Update # PS points: ',ncalls0,' --> ',ncalls
      endif
c Reset the accumulated results for grid updating
      if(imode.eq.0) then
         do kdim=1,ndim
            do kint=0,nint_used
               xacc(kint,kdim)=0
               if(kint.gt.0) then
                  nhits(kint,kdim)=0
               endif
            enddo
         enddo
      endif
      do i=1,nintegrals
         vtot(i)=0
         etot(i)=0
      enddo
      kpoint_iter=0
      do i=1,nintegrals
         non_zero_point(i)=0
      enddo
c Loop over PS points
 2    kpoint_iter=kpoint_iter+1
      do kpoint=1,ncalls
c find random x, and its random cell
         do kdim=1,ndim
            kfold(kdim)=1
c if(even), we should compute the ncell and the rand from the ran3()
            if (even) then
               rand(kdim)=ran3(even)
               ncell(kdim)= min(int(rand(kdim)*nint_used)+1,
     &              nint_used)
               rand(kdim)=rand(kdim)*nint_used-(ncell(kdim)-1)
            else
               ncell(kdim)=min(int(nint_used/ifold(kdim)*ran3(even))+1,
     &              nint_used)
               rand(kdim)=ran3(even)
            endif
         enddo
         do i=1,nintegrals
            f(i)=0
         enddo
         ifirst=0
 1       continue
         vol=1
c compute jacobian ('vol') for the PS point
         do kdim=1,ndim
            nintcurr=nint_used/ifold(kdim)
            icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
            dx(kdim)=xgrid(icell(kdim),kdim)-xgrid(icell(kdim)-1,kdim)
            vol=vol*dx(kdim)*nintcurr
            x(kdim)=xgrid(icell(kdim)-1,kdim)+rand(kdim)*dx(kdim)
            if(imode.eq.0)
     &           nhits(icell(kdim),kdim)=nhits(icell(kdim),kdim)+1
         enddo
         call get_ave_virt(x,nint_used_virt,ndim,average_virtual)
c contribution to integral
         if(imode.eq.0) then
            dummy=fun(x,vol,ifirst,f1)
            do i=1,nintegrals
               f(i)=f(i)+f1(i)
            enddo
         else
c this accumulated value will not be used
            dummy=fun(x,vol,ifirst,f1)
            do i=1,nintegrals
               f(i)=f(i)+f1(i)
            enddo
            ifirst=1
            call nextlexi(ndim,ifold,kfold,iret)
            if(iret.eq.0) goto 1
c closing call: accumulated value with correct sign
            dummy=fun(x,vol,2,f1)
            do i=1,nintegrals
               f(i)=f1(i)
            enddo
         endif
c
         if(imode.eq.0) then
c accumulate the function in xacc(icell(kdim),kdim) to adjust the grid later
            do kdim=1,ndim
               xacc(icell(kdim),kdim)=xacc(icell(kdim),kdim)+f(1)
            enddo
c Set the Born contribution (to compute the average_virtual) to zero if
c the virtual was not computed for this phase-space point. Compensate by
c including the virtual_fraction.
            if (f(3).ne.0d0) then
               f(6)=f(6)/virtual_fraction
               call fill_ave_virt(x,nint_used_virt,ndim,f(3),f(6))
            else
               f(6)=0d0
            endif
         else
c update the upper bounding envelope total rate
            prod=1d0
            do kdim=1,ndim
               prod=prod*ymax(ncell(kdim),kdim)
            enddo
            prod=(f(1)/prod)
            if (prod.gt.1d0) then
c Weight for this PS point is larger than current upper bound. Increase
c the bound so that it is equal to the current max weight.  If the new
c point is more than twice as large as current upper bound, increase
c bound by factor 2 only to prevent a single unstable points to
c completely screw up the efficiency
               prod=min(2d0,prod)
               prod=prod**(1d0/dble(ndim))
               do kdim=1,ndim
                  ymax(ncell(kdim),kdim)=ymax(ncell(kdim),kdim)*prod
               enddo
            endif
c Update the upper bounding envelope virtual. Do not include the
c enhancement due to the virtual_fraction. (And again limit by factor 2
c at most).
            if (f(5)*virtual_fraction.gt.ymax_virt) ymax_virt=min(f(5)
     $           *virtual_fraction,ymax_virt*2d0)
c for consistent printing in the log files (in particular when doing LO
c runs), set also f(6) to zero when imode.eq.1 and the virtuals are not
c included.
            if (f(3).eq.0) f(6)=0d0
         endif
         do i=1,nintegrals
            if (f(i).ne.0d0) non_zero_point(i)=non_zero_point(i)+1
         enddo
c Add the PS point to the result of this iteration
         do i=1,nintegrals
            vtot(i)=vtot(i)+f(i)
            etot(i)=etot(i)+f(i)**2
         enddo
         if (f(1).ne.0d0) call HwU_add_points
      enddo
      do i=1,nintegrals
c Number of phase-space points used
         ntotcalls(i)=ncalls*kpoint_iter
c Special for the computation of the 'computed virtual'
         if (i.eq.4 .and. non_zero_point(i).ne.0 )
     &        ntotcalls(i) = non_zero_point(i)
      enddo
      if (ntotcalls(1).gt.max_points .and. non_zero_point(1).lt.25 .and.
     &     double_events) then
         write (*,*) 'ERROR: INTEGRAL APPEARS TO BE ZERO.'
         write (*,*) 'TRIED',ntotcalls(1),'PS POINTS AND ONLY '
     &        ,non_zero_point(1),' GAVE A NON-ZERO INTEGRAND.'
         stop 1
      endif
c Goto beginning of loop over PS points until enough points have found
c that pass cuts.
      if (non_zero_point(1).lt.ncalls .and. double_events) goto 2

c Iteration done. Update the accumulated results and print them to the
c screen
      do i=1,nintegrals
         vtot(i)=vtot(i)/dble(ntotcalls(i))
         etot(i)=etot(i)/dble(ntotcalls(i))
c the abs is to avoid tiny negative values
         etot(i)=sqrt(abs(etot(i)-vtot(i)**2)
     $        /dble(ntotcalls(i)))
         if (vtot(i).ne.0d0) then
            efrac(i)=abs(etot(i)/vtot(i))
         else
            efrac(i)=0d0
         endif
      enddo
      do i=1,nintegrals
         write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')
     $        title(i)//' =',vtot(i),' +/- ',etot(i),' (',efrac(i)*100d0
     $        ,'%)'
      enddo
C If there was a large fluctation in this iteration, be careful with
C including it in the accumalated results and plots.
      if (efrac(1).gt.0.3d0 .and. iappl.eq.0) then
c Do not include the results in the plots
         if (fixed_order) call accum(.false.)
         if (fixed_order) call HwU_accum_iter(.false.,ntotcalls(1))
      endif
      if (efrac(1).gt.0.3d0 .and. nit.gt.3 .and. iappl.eq.0) then
c Do not include the results in the updating of the grids.
         write (*,*) 'Large fluctuation ( >30 % ).'
     &        //'Not including iteration in results.'
c empty the accumulated results in the MC over integers
         call empty_MC_integer
c empty the accumalated results for the MINT grids
         if (imode.eq.0) then
c emptying accum. results is done above when the iteration starts
            continue
         elseif (imode.eq.1) then
c Cannot really skip the increase of the upper bounding envelope. So,
c simply continue here. Note that no matter how large the integrand for
c the PS point, the upper bounding envelope is at most increased by a
c factor 2, so this should be fine.
            continue
         endif
c double the number of points for the next iteration
         if (double_events) ncalls0=ncalls0*2
         if (bad_iteration .and. imode.eq.0 .and. double_events) then
c 2nd bad iteration is a row. Reset grids
            write (*,*)'2nd bad iteration in a row. '/
     &           /'Resetting grids and starting from scratch...'
            if (double_events) then
               if (imode.eq.0) nint_used=min_inter ! reset number of intervals
               ncalls0=ncalls0/8   ! Start with larger number
            endif
            nit=0
            nit_included=0
c Reset the MINT grids
            if (imode.eq.0) then
               do kdim=1,ndim
                  do kint=0,nint_used
                     xgrid(kint,kdim)=dble(kint)/nint_used
                  enddo
               enddo
               call init_ave_virt(nint_used_virt,ndim)
            elseif (imode.eq.1) then
               do kdim=1,ndim
                  nintcurr=nint_used/ifold(kdim)
                  do kint=1,nintcurr
                     ymax(kint,kdim)=xint**(1d0/ndim)
                  enddo
                  nintcurr_virt=nint_used_virt/ifold(kdim)
               enddo
               ymax_virt=xint_virt
            endif
            call reset_MC_grid  ! reset the grid for the integers
            if (fixed_order) call initplot  ! Also reset all the plots
            do i=1,nintegrals
               ans(i)=0d0
               unc(i)=0d0
               chi2(i)=0d0
               do j=1,3
                  ans3(i,j)=0d0
                  unc3(i,j)=0d0
               enddo
            enddo
            bad_iteration=.false.
         else
            bad_iteration=.true.
         endif
         goto 10
      else
         bad_iteration=.false.
      endif
      if(nit.eq.1) then
         do i=1,nintegrals
            ans(i)=vtot(i)
            unc(i)=etot(i)
         enddo
         write (*,'(a,1x,e10.4)') 'Chi^2 per d.o.f.',0d0
      else
c prevent annoying division by zero for nearly zero
c integrands
         do i=1,nintegrals
            if(etot(i).eq.0.and.unc(i).eq.0) then
               if(ans(i).eq.vtot(i) .and. i.eq.1) then
c double the number of points for the next iteration
                  if (double_events) ncalls0=ncalls0*2
                  goto 10
               else
                  unc(i)=abs(vtot(i)-ans(i))
                  etot(i)=abs(vtot(i)-ans(i))
               endif
            elseif(etot(i).eq.0) then
               etot(i)=unc(i)
            elseif(unc(i).eq.0) then ! 1st iteration; set to a large value
               unc(i)=etot(i)*1d99
            endif
            if (i.ne.1 .and. (etot(i).eq.0 .or. unc(i).eq.0)) then
               ans(i)=0d0
               unc(i)=0d0
               chi2(i)=0d0
            else
               ans(i)=(ans(i)/unc(i)+vtot(i)/etot(i))/
     &              (1/unc(i)+1/etot(i))
               unc(i)=1/sqrt(1/unc(i)**2+1/etot(i)**2)
               chi2(i)=chi2(i)+(vtot(i)-ans(i))**2/etot(i)**2
            endif
         enddo
         write (*,'(a,1x,e10.4)') 'Chi^2=',(vtot(1)-ans(1))**2
     $        /etot(1)**2
      endif

      nit_included=nit_included+1
      do i=1,nintegrals
         if (ans(i).ne.0d0) then
            efrac(i)=abs(unc(i)/ans(i))
         else
            efrac(i)=0d0
         endif
         write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')
     $        'accumulated results '//title(i)//' =',ans(i),' +/- '
     $        ,unc(i) ,' (',efrac(i)*100d0,'%)'
      enddo
      if (nit_included.le.1) then
         write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF ='
     $        ,0d0
      else
         write (*,'(a,1x,e10.4)') 'accumulated result Chi^2 per DoF =',
     &        chi2(1)/dble(nit_included-1)
      endif
      if (imode.eq.0) then
c     Update the average_virtual: a_new=(virt+a_old*born)/born
         if (vtot(6).ne.0d0) then
            if (average_virtual.eq.0d0) then ! i.e. first iteration
               average_virtual=vtot(3)/vtot(6)+average_virtual
            else  ! give some importance to the iterations already done
               average_virtual=(vtot(3)/vtot(6)+average_virtual*2d0)/2d0
            endif
         endif
c Update the fraction of the events for which we include the virtual corrections
c in the calculation
         virtual_fraction=max(min(virtual_fraction*max(min(2d0*etot(3)
     $        /etot(1),2d0),0.25d0),1d0),Min_virt_fraction)
         write (*,'(a,1x,f7.3,1x,f7.3)') 'update virtual fraction to:'
     $        ,virtual_fraction,average_virtual
      elseif (imode.eq.1) then
         write (*,'(a,1x,f7.3,1x,f7.3)') 'virtual fraction is:'
     $        ,virtual_fraction,average_virtual
      endif
c Update the results of the last tree iterations
      do i=1,nintegrals
         do j=1,2
            ans3(i,j)=ans3(i,j+1)
            unc3(i,j)=unc3(i,j+1)
         enddo
         ans3(i,3)=vtot(i)
         unc3(i,3)=etot(i)
      enddo
c Compute the results of the last three iterations
      if (nit_included.ge.4) then
         do i=1,nintegrals
            ans_l3(i)=0d0
            unc_l3(i)=ans3(i,1)*1d99
            chi2_l3(i)=0d0
            do j=1,3
               if (i.ne.1 .and. (unc_l3(i).eq.0d0 .or. unc3(i
     $              ,j).eq.0d0)) then
                  ans_l3(i)=0d0
                  unc_l3(i)=0d0
                  chi2_l3(i)=0d0
               else
                  ans_l3(i)=(ans_l3(i)/unc_l3(i)+ans3(i,j)/unc3(i,j))/
     &                 (1/unc_l3(i)+1/unc3(i,j))
                  unc_l3(i)=1/sqrt(1/unc_l3(i)**2+1/unc3(i,j)**2)
                  chi2_l3(i)=chi2_l3(i)+
     &                 (ans3(i,j)-ans_l3(i))**2/unc3(i,j)**2
               endif
            enddo
            chi2_l3(i)=chi2_l3(i)/2d0 ! three iterations, so 2 degrees of freedom
         enddo
         do i=1,2
            if (ans_l3(i).ne.0d0) then
               efrac(i)=abs(unc_l3(i)/ans_l3(i))
            else
               efrac(i)=0d0
            endif
            write(*,'(a,1x,e10.4,1x,a,1x,e10.4,1x,a,1x,f7.3,1x,a)')
     $           'accumulated results last 3 iterations '//title(i)/
     $           /' =' ,ans_l3(i),' +/- ',unc_l3(i) ,' (',efrac(i)*100d0
     $           ,'%)'
         enddo
         write(*,'(a,1x,e10.4)')
     $        'accumulated result last 3 iterrations Chi^2 per DoF ='
     $        ,chi2_l3(1)
      endif
      if(imode.eq.0) then
c Iteration is finished; now rearrange the grid
         do kdim=1,ndim
            call regrid(xacc(0,kdim),xgrid(0,kdim),nhits(1,kdim)
     $           ,nint_used)
         enddo
         call regrid_ave_virt(nint_used_virt,ndim)
c Regrid the MC over integers (used for the MC over FKS dirs)
         call regrid_MC_integer
      endif
c Quit if the desired accuracy has been reached
      if (nit_included.ge.min_it .and. accuracy.gt.0d0) then
         if (unc(1)/ans(1)*max(1d0,chi2(1)/dble(nit_included-1))
     $        .lt.accuracy) then
            write (*,*) 'Found desired accuracy'
            nit=nitmax
c Improve the stats in the plots
            if (fixed_order) call accum(.true.)
            if (fixed_order) call HwU_accum_iter(.true.,ntotcalls(1))
            goto 10
         elseif(unc_l3(1)/ans_l3(1)*max(1d0,chi2_l3(1)).lt.accuracy)
     $           then
            write (*,*)
     &           'Found desired accuracy in last 3 iterations'
            nit=nitmax
            do i=1,nintegrals
               ans(i)=ans_l3(i)
               unc(i)=unc_l3(i)
               chi2(i)=chi2_l3(i)*dble(nit_included-1)
            enddo
c Improve the stats in the plots
            if (fixed_order) call accum(.true.)
            if (fixed_order) call HwU_accum_iter(.true.,ntotcalls(1))
            goto 10
         endif
      endif
c Double the number of intervals in the grids if not yet reach the maximum
      if (2*nint_used.le.nintervals .and. double_events) then
         do kdim=1,ndim
            call double_grid(xgrid(0,kdim),nint_used)
         enddo
         nint_used=2*nint_used
      endif
      if (2*nint_used_virt.le.nintervals_virt .and. double_events) then
         call double_ave_virt(nint_used_virt,ndim)
         nint_used_virt=2*nint_used_virt
      endif

c double the number of points for the next iteration
      if (double_events) ncalls0=ncalls0*2
c Also improve stats in plots
      if (fixed_order) call accum(.true.)
      if (fixed_order) call HwU_accum_iter(.true.,ntotcalls(1))
c Do next iteration
      goto 10
      end

      subroutine double_grid(xgrid,ninter)
      implicit none
      include "mint.inc"
      integer  ninter
      real * 8 xgrid(0:nintervals)
      integer i
      do i=ninter,1,-1
         xgrid(i*2)=xgrid(i)
         xgrid(i*2-1)=(xgrid(i)+xgrid(i-1))/2d0
      enddo
      return
      end


      subroutine regrid(xacc,xgrid,nhits,ninter)
      implicit none
      include "mint.inc"
      integer  ninter,nhits(nintervals)
      real * 8 xacc(0:nintervals),xgrid(0:nintervals)
      real * 8 xn(nintervals),r,tiny,xl,xu,nl,nu
      parameter ( tiny=1d-8 )
      integer kint,jint
c Use the same smoothing as in VEGAS uses for the grids (i.e. use the
c average of the central and the two neighbouring grid points):
      xl=xacc(1)
      xu=xacc(2)
      xacc(1)=(xl+xu)/2d0
      nl=nhits(1)
      nu=nhits(2)
      nhits(1)=nint((nl+nu)/2d0)
      do kint=2,ninter-1
         xacc(kint)=xl+xu
         xl=xu
         xu=xacc(kint+1)
         xacc(kint)=(xacc(kint)+xu)/3d0
         nhits(kint)=nl+nu
         nl=nu
         nu=nhits(kint+1)
         nhits(kint)=nint((nhits(kint)+nu)/3d0)
      enddo
      xacc(ninter)=(xu+xl)/2d0
      nhits(ninter)=nint((nu+nl)/2d0)
c
      do kint=1,ninter
c xacc (xerr) already contains a factor equal to the interval size
c Thus the integral of rho is performed by summing up
         if(nhits(kint).ne.0) then
            xacc(kint)= xacc(kint-1)
     #           + abs(xacc(kint))/nhits(kint)
         else
            xacc(kint)=xacc(kint-1)
         endif
      enddo
      do kint=1,ninter
         xacc(kint)=xacc(kint)/xacc(ninter)
      enddo
c Check that we have a reasonable result and update the accumulated
c results if need be
      do kint=1,ninter
         if (xacc(kint).lt.(xacc(kint-1)+tiny)) then
            xacc(kint)=xacc(kint-1)+tiny
         endif
      enddo
c it could happen that the change above yielded xacc() values greater
c than 1: one more update needed
      xacc(ninter)=1d0
      do kint=1,ninter
         if (xacc(ninter-kint).gt.(xacc(ninter-kint+1)-tiny)) then
            xacc(ninter-kint)=1d0-dble(kint)*tiny
         else
            exit
         endif
      enddo

      do kint=1,ninter
         r=dble(kint)/dble(ninter)
         do jint=1,ninter
            if(r.lt.xacc(jint)) then
               xn(kint)=xgrid(jint-1)+(r-xacc(jint-1))
     #        /(xacc(jint)-xacc(jint-1))*(xgrid(jint)-xgrid(jint-1))
               goto 11
            endif
         enddo
         if(jint.ne.ninter+1.and.kint.ne.ninter) then
            write(*,*) ' error',jint,ninter
            stop
         endif
         xn(ninter)=1
 11      continue
      enddo
      do kint=1,ninter
         xgrid(kint)=xn(kint)
      enddo
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


      subroutine gen(fun,ndim,xgrid,ymax,ymax_virt,imode,x,vn)
c imode=0 to initialize
c imode=1 to generate
c imode=3 store generation efficiency in x(1)
      implicit none
      integer ndim,imode
      include "mint.inc"
      real * 8 fun,xgrid(0:nintervals,ndimmax),ymax(nintervals,ndimmax)
     $     ,ymax_virt,x(ndimmax)
      real * 8 dx(ndimmax),xx(ndimmax)
      integer icell(ndimmax),ncell(ndimmax),ncell_virt
      integer ifold(ndimmax),kfold(ndimmax)
      common/cifold/ifold
      real * 8 r,f(nintegrals),f1(nintegrals),ubound,vol,ran3
     $     ,xmmm(nintervals,ndimmax),dummy
      real * 8 rand(ndimmax)
      external fun,ran3
      integer icalls,mcalls,kdim,kint,nintcurr,nintcurr_virt,iret,ifirst
     $     ,i,vn,icalls_virt,mcalls_virt,icalls_nz,icalls_virt_nz
      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction
      save icalls,mcalls,icalls_virt,mcalls_virt,xmmm,icalls_nz
     $     ,icalls_virt_nz
      if(imode.eq.0) then
         do kdim=1,ndim
            nintcurr=nintervals/ifold(kdim)
            xmmm(1,kdim)=ymax(1,kdim)
            do kint=2,nintcurr
               xmmm(kint,kdim)=xmmm(kint-1,kdim)+ymax(kint,kdim)
            enddo
            do kint=1,nintcurr
               xmmm(kint,kdim)=xmmm(kint,kdim)/xmmm(nintcurr,kdim)
            enddo
         enddo
         icalls=0
         icalls_nz=0
         mcalls=0
         icalls_virt=0
         icalls_virt_nz=0
         mcalls_virt=0
         xx(2)=0d0
         xx(3)=0d0
         xx(5)=0d0
         xx(6)=0d0
         return
      elseif(imode.eq.3) then
         if(icalls.gt.0) then
            x(1)=dble(mcalls)/dble(icalls)
            x(2)=xx(2)/dble(icalls)
            x(3)=xx(3)/dble(icalls)
         else
            x(1)=-1d0
            x(2)=-1d0
            x(3)=-1d0
         endif
         if(icalls_virt.gt.0) then
            x(4)=dble(mcalls_virt)/dble(icalls_virt)
            x(5)=xx(5)/dble(icalls_virt)
            x(6)=xx(6)/dble(icalls_virt)
         else
            x(4)=-1d0
            x(5)=-1d0
            x(6)=-1d0
         endif
         call increasecnt(' ',imode)
         return
      endif
      if (vn.eq.1) then
         mcalls_virt=mcalls_virt+1
      elseif(vn.eq.2 .or. vn.eq.3) then
         mcalls=mcalls+1
      else
         write (*,*) 'vn not correct in mint-integrator2.f',vn,imode
         stop
      endif
 10   continue
      if (vn.eq.1) then
         icalls_virt=icalls_virt+1
      elseif(vn.eq.2 .or. vn.eq.3) then
         icalls=icalls+1
      endif
      if (vn.eq.1) then
c Choose cell flat
         do kdim=1,ndim
            ncell(kdim)=min(int(ran3(.false.)*nintcurr)+1,nintcurr)
            rand(kdim)=ran3(.false.)
         enddo
      elseif(vn.eq.2 .or. vn.eq.3) then
         do kdim=1,ndim
            nintcurr=nintervals/ifold(kdim)
            r=ran3(.false.)
            do kint=1,nintcurr
               if(r.lt.xmmm(kint,kdim)) then
                  ncell(kdim)=kint
                  exit
               endif
            enddo
            rand(kdim)=ran3(.false.)
         enddo
      endif
      if (vn.eq.2 .or. vn.eq.3) then
         ubound=1
         do kdim=1,ndim
            ubound=ubound*ymax(ncell(kdim),kdim)
         enddo
      endif
      do kdim=1,ndim
         kfold(kdim)=1
      enddo
      do i=1,nintegrals
         f(i)=0
      enddo
      ifirst=0
 5    continue
      vol=1
      do kdim=1,ndim
         nintcurr=nintervals/ifold(kdim)
         nintcurr_virt=nintervals_virt/ifold(kdim)
         icell(kdim)=ncell(kdim)+(kfold(kdim)-1)*nintcurr
         dx(kdim)=xgrid(icell(kdim),kdim)-xgrid(icell(kdim)-1,kdim)
         vol=vol*dx(kdim)*nintervals/ifold(kdim)
         x(kdim)=xgrid(icell(kdim)-1,kdim)+rand(kdim)*dx(kdim)
      enddo
      call get_ave_virt(x,nintcurr_virt,ndim,average_virtual)
      if (vn.eq.1) then
         ubound=ymax_virt
      endif
      dummy=fun(x,vol,ifirst,f1)
      do i=1,nintegrals
         f(i)=f(i)+f1(i)
      enddo
      ifirst=1
      call nextlexi(ndim,ifold,kfold,iret)
      if(iret.eq.0) goto 5
c get final value (x and vol not used in this call)
      dummy=fun(x,vol,2,f1)
      do i=1,nintegrals
         f(i)=f1(i)
      enddo
      if (vn.eq.2 .or. vn.eq.3) then
         xx(2)=xx(2)+f(2)
         xx(3)=xx(3)+f(1)
      else
         xx(5)=xx(5)+f(2)
         xx(6)=xx(6)+f(1)
      endif
      call increasecnt('another call to the function',imode)
      if (f(1).eq.0d0) then
         call increasecnt('failed generation cuts',imode)
      else
         if (vn.eq.1) then
            icalls_virt_nz=icalls_virt_nz+1
         elseif(vn.eq.2 .or.vn.eq.3) then
            icalls_nz=icalls_nz+1
         endif
      endif
      if(f(1).lt.0) then
         write(*,*) 'gen: non positive function'
         stop
      endif
      if(f(1).gt.ubound) then
         if (vn.eq.2) then
            call increasecnt('ubound fail novi',imode)
         elseif (vn.eq.1) then
            call increasecnt('ubound fail virt',imode)
         elseif (vn.eq.3) then
            call increasecnt('ubound fail born',imode)
         endif
      endif
      ubound=ubound*ran3(.false.)
      if(ubound.gt.f(1)) then
         call increasecnt
     &        ('vetoed calls in inclusive cross section',imode)
         goto 10
      endif
      if (vn.eq.2) then
         call increasecnt('events gen novi',imode)
      elseif (vn.eq.1) then
         call increasecnt('events gen virt',imode)
      elseif (vn.eq.3) then
         call increasecnt('events gen born',imode)
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
      logical firsttime
      common /even_ran/dim,ng,npg,k,firsttime
c Make sure that hypercubes are newly initialized
      firsttime=.true.
c Number of dimension of the integral
      dim=ndim
c Number of elements in which we can split one dimension
      ng=(ncalls0/2.)**(1./ndim)
c Total number of hypercubes
      k=ng**ndim
c Number of PS points in each hypercube (at least 2)
      npg=max(ncalls0/k,2)
c Number of PS points for this iteration
      ncalls=npg*k
      return
      end


      double precision function get_ran()
      implicit none
      double precision ran2,dng
      external ran2
      integer dim,ng,npg,k
      logical firsttime
      common /even_ran/dim,ng,npg,k,firsttime
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


      subroutine init_ave_virt(ninter,ndim)
      implicit none
      include "mint.inc"
      integer kdim,ndim,ninter,i
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt ,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
      do kdim=1,ndim
         do i=1,ninter
            nvirt(i,kdim)=0
            ave_virt(i,kdim)=0d0
            nvirt_acc(i,kdim)=0
            ave_virt_acc(i,kdim)=0d0
            ave_born_acc(i,kdim)=0d0
         enddo
      enddo
      return
      end

      subroutine get_ave_virt(x,ninter,ndim,average_virtual)
      implicit none
      include "mint.inc"
      integer kdim,ndim,ninter,ncell
      double precision x(ndimmax),average_virtual
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt ,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
      average_virtual=0d0
      do kdim=1,ndim
         ncell=min(int(x(kdim)*ninter)+1,ninter)
         average_virtual=average_virtual+ave_virt(ncell,kdim)
      enddo
      average_virtual=average_virtual/ndim
      return
      end

      subroutine fill_ave_virt(x,ninter,ndim,virtual,born)
      implicit none
      include "mint.inc"
      integer kdim,ndim,ninter,ncell
      double precision x(ndimmax),virtual,born
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt ,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
      do kdim=1,ndim
         ncell=min(int(x(kdim)*ninter)+1,ninter)
         nvirt_acc(ncell,kdim)=nvirt_acc(ncell,kdim)+1
         ave_virt_acc(ncell,kdim)=ave_virt_acc(ncell,kdim)+virtual
         ave_born_acc(ncell,kdim)=ave_born_acc(ncell,kdim)+born
      enddo
      return
      end

      subroutine regrid_ave_virt(ninter,ndim)
      implicit none
      include "mint.inc"
      integer ninter,ndim,kdim,i
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
c need to solve for k_new = (virt+k_old*born)/born
      do kdim=1,ndim
         do i=1,ninter
            if (ave_born_acc(i,kdim).eq.0d0) cycle
            if (ave_virt(i,kdim).eq.0d0) then ! i.e. first iteration
               ave_virt(i,kdim)= ave_virt_acc(i,kdim)/ave_born_acc(i
     $              ,kdim)+ave_virt(i,kdim)
            else  ! give some importance to the iterations already done
               ave_virt(i,kdim)=(ave_virt_acc(i,kdim)/ave_born_acc(i
     $              ,kdim)+ave_virt(i,kdim)*2d0)/2d0
            endif
         enddo
      enddo
c reset the acc values
      do kdim=1,ndim
         do i=1,ninter
            nvirt(i,kdim)=nvirt(i,kdim)+nvirt_acc(i,kdim)
            nvirt_acc(i,kdim)=0
            ave_born_acc(i,kdim)=0d0
            ave_virt_acc(i,kdim)=0d0
         enddo
      enddo
      return
      end


      subroutine double_ave_virt(ninter,ndim)
      implicit none
      include "mint.inc"
      integer kdim,ndim,i,ninter
      integer nvirt(nintervals_virt,ndimmax),nvirt_acc(nintervals_virt
     $     ,ndimmax)
      double precision ave_virt(nintervals_virt,ndimmax)
     $     ,ave_virt_acc(nintervals_virt,ndimmax)
     $     ,ave_born_acc(nintervals_virt ,ndimmax)
      common/c_ave_virt/ave_virt,ave_virt_acc,ave_born_acc,nvirt
     $     ,nvirt_acc
      do kdim=1,ndim
         do i=ninter,1,-1
            ave_virt(i*2,kdim)=ave_virt(i,kdim)
            if (nvirt(i,kdim).ne.0) then
               nvirt(i*2,kdim)=max(nvirt(i,kdim)/2,1)
            else
               nvirt(i*2,kdim)=0
            endif
            if (i.ne.1) then
               ave_virt(i*2-1,kdim)=(ave_virt(i,kdim)
     $              +ave_virt(i-1,kdim))/2d0
               if (nvirt(i,kdim)+nvirt(i-1,kdim).ne.0) then
                  nvirt(i*2-1,kdim)=
     &                 max((nvirt(i,kdim)+nvirt(i-1,kdim))/4,1)
               else
                  nvirt(i*2-1,kdim)=0
               endif
            endif
         enddo
      enddo
      return
      end

