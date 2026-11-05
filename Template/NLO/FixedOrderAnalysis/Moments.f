c Smooth fixed-order analysis with the HwU calling interface.
c Each call to HwU_add_points closes ONE correlated subtraction group.
c Native coefficients and full central-weight covariance are preserved.
      module HwU_wgts_info_len
      use iso_c_binding
      implicit none
      integer, parameter :: wgts_info_len=80
      contains
      integer(c_int) function get_wgts_info_len()
     &     bind(c,name="get_wgts_info_len")
      get_wgts_info_len=wgts_info_len
      end function
      end module

      module Moments_variables
      use HwU_wgts_info_len
      use, intrinsic :: ieee_arithmetic
      implicit none
      type moment_plot
         logical :: booked=.false.,smooth=.false.,touched=.false.
         integer :: nbins=0,ncoeff=0
         integer(kind=8) :: nactive=0
         character(len=256) :: title=''
         double precision :: xmin=0d0,xmax=0d0
         double precision, allocatable :: pending(:,:),mean(:,:)
         double precision, allocatable :: m2(:,:),result(:,:)
         double precision, allocatable :: covariance(:,:)
      end type
      type(moment_plot), allocatable :: plots(:)
      character(len=wgts_info_len), allocatable :: weight_names(:)
      double precision, allocatable :: scratch(:)
      integer :: nweights=0,error_estimation=3,niterations=0
      logical :: initialized=.false.
      save
      contains

      subroutine moments_fail(message)
      character(len=*), intent(in) :: message
      write(*,'(a)') 'Moments: '//trim(message)
      stop 1
      end subroutine

      subroutine check_label(label)
      integer, intent(in) :: label
      if (.not.initialized) call moments_fail('call HwU_inithist first')
      if (.not.allocated(plots)) then
         call moments_fail('histogram is not booked')
      endif
      if (label.lt.1.or.label.gt.size(plots)) then
         call moments_fail('histogram label is outside booked range')
      endif
      if (.not.plots(label)%booked) then
         call moments_fail('histogram label is not booked')
      endif
      end subroutine

c Tests are orthonormal under dx/(xmax-xmin), so a0 is the rate.
      subroutine legendre_tests(t,n,values)
      double precision, intent(in) :: t
      integer, intent(in) :: n
      double precision, intent(out) :: values(n)
      double precision p0,p1,p2
      integer k
      values(1)=1d0
      if (n.eq.1) return
      p0=1d0
      p1=t
      values(2)=sqrt(3d0)*t
      do k=2,n-1
         p2=((2*k-1)*t*p1-(k-1)*p0)/dble(k)
         values(k+1)=sqrt(dble(2*k+1))*p2
         p0=p1
         p1=p2
      enddo
      end subroutine

c Exact integrals of the density basis over a plotting bin.  The
c density is sum_k a_k sqrt(2k+1) Pk(t)/(xmax-xmin).
      subroutine bin_projection(plot,ibin,projection)
      type(moment_plot), intent(in) :: plot
      integer, intent(in) :: ibin
      double precision, intent(out) :: projection(plot%ncoeff)
      double precision tl,tr,pl(plot%ncoeff+1)
      double precision pr(plot%ncoeff+1)
      integer k
      projection=0d0
      if (.not.plot%smooth) then
         projection(ibin)=1d0
         return
      endif
      tl=-1d0+2d0*dble(ibin-1)/dble(plot%nbins)
      tr=-1d0+2d0*dble(ibin)/dble(plot%nbins)
      call legendre_tests(tl,plot%ncoeff+1,pl)
      call legendre_tests(tr,plot%ncoeff+1,pr)
      projection(1)=0.5d0*(tr-tl)
      do k=1,plot%ncoeff-1
         projection(k+1)=0.5d0/sqrt(dble(2*k+1))
     &        *((pr(k+2)-pl(k+2))/sqrt(dble(2*k+3))
     &        -(pr(k)-pl(k))/sqrt(dble(2*k-1)))
      enddo
      end subroutine
      end module

      subroutine HwU_inithist(nwgt,names)
      use Moments_variables
      implicit none
      integer nwgt,j
      character*(*) names(*)
      logical pineappl
      common /for_pineappl/ pineappl
      if (pineappl) then
         call moments_fail('PineAPPL is unsupported by this backend')
      endif
      if (nwgt.lt.1) call moments_fail('nweights must be positive')
      call HwU_deallocate_all
      nweights=nwgt
      allocate(weight_names(nweights))
      do j=1,nweights
         weight_names(j)=names(j)
      enddo
      error_estimation=3
      niterations=0
      initialized=.true.
      end

      subroutine set_error_estimation(mode)
      use Moments_variables
      implicit none
      integer mode
c Mode 1 uses an equal iteration average and propagated within-iteration
c covariance. Mode 3 uses exactly the scalar MINT iteration weights.
      if (mode.ne.1.and.mode.ne.3) then
         call moments_fail('only error_estimation 1 and 3 supported')
      endif
      if (niterations.ne.0) then
         call moments_fail('cannot change error mode after averaging')
      endif
      error_estimation=mode
      end

c Ordinary HwU_book retains indicator bins, including categorical rates.
      subroutine HwU_book(label,title,nbins,xmin,xmax)
      implicit none
      integer label,nbins
      double precision xmin,xmax
      character*(*) title
      call Moments_book_internal(label,title,nbins,xmin,xmax,
     &     nbins,.false.)
      end

c Explicit opt-in: nbins controls output resolution; nmodes controls the
c expansion. Every mode is retained, including in per-channel output.
      subroutine Moments_book(label,title,nbins,xmin,xmax,nmodes)
      implicit none
      integer label,nbins,nmodes
      double precision xmin,xmax
      character*(*) title
      call Moments_book_internal(label,title,nbins,xmin,xmax,
     &     nmodes,.true.)
      end

      subroutine Moments_book_internal(label,title,nbins,xmin,xmax,
     &     nmodes,smooth)
      use Moments_variables
      implicit none
      integer label,nbins,nmodes,oldsize
      double precision xmin,xmax
      logical smooth
      character*(*) title
      type(moment_plot), allocatable :: previous(:)
      if (.not.initialized) then
         call moments_fail('initialize before booking')
      endif
      if (label.lt.1.or.nbins.lt.1.or.nmodes.lt.1) then
         call moments_fail('labels, bins and modes must be positive')
      endif
      if (.not.ieee_is_finite(xmin).or.
     &     .not.ieee_is_finite(xmax)) then
         call moments_fail('nonfinite plotting limits')
      endif
      if (xmax.le.xmin) call moments_fail('xmax must exceed xmin')
      if (.not.ieee_is_finite(xmax-xmin)) then
         call moments_fail('plotting range width overflows')
      endif
      if (niterations.ne.0) then
         call moments_fail('cannot book after iteration accumulation')
      endif
      if (.not.allocated(plots)) then
         allocate(plots(label))
      elseif (size(plots).lt.label) then
         oldsize=size(plots)
         call move_alloc(plots,previous)
         allocate(plots(label))
         plots(1:oldsize)=previous
         deallocate(previous)
      endif
      if (plots(label)%booked) then
         call moments_fail('histogram label already booked')
      endif
      plots(label)%booked=.true.
      plots(label)%smooth=smooth
      plots(label)%nbins=nbins
      plots(label)%ncoeff=nmodes
      plots(label)%xmin=xmin
      plots(label)%xmax=xmax
      plots(label)%title=title
      if (allocated(scratch)) then
         if (size(scratch).lt.nmodes) deallocate(scratch)
      endif
      if (.not.allocated(scratch)) allocate(scratch(nmodes))
      allocate(plots(label)%pending(nweights,nmodes))
      allocate(plots(label)%mean(nweights,nmodes))
      allocate(plots(label)%m2(nmodes,nmodes))
      allocate(plots(label)%result(nweights,nmodes))
      allocate(plots(label)%covariance(nmodes,nmodes))
      plots(label)%pending=0d0
      plots(label)%mean=0d0
      plots(label)%m2=0d0
      plots(label)%result=0d0
      plots(label)%covariance=0d0
      end

      subroutine HwU_fill(label,x,weights)
      use Moments_variables
      implicit none
      integer label,j,k,ibin,n
      double precision x,weights(*),t
      call check_label(label)
      if (.not.ieee_is_finite(x)) then
         call moments_fail('nonfinite observable in HwU_fill')
      endif
      do j=1,nweights
         if (.not.ieee_is_finite(weights(j))) then
            call moments_fail('nonfinite weight in HwU_fill')
         endif
      enddo
c Include the left and right range endpoints; nothing outside is folded.
      if (x.lt.plots(label)%xmin.or.x.gt.plots(label)%xmax) return
      n=plots(label)%ncoeff
      if (plots(label)%smooth) then
         t=2d0*((x-plots(label)%xmin)
     &        /(plots(label)%xmax-plots(label)%xmin))-1d0
         call legendre_tests(max(-1d0,min(1d0,t)),n,scratch(1:n))
         do k=1,n
            do j=1,nweights
               plots(label)%pending(j,k)=plots(label)%pending(j,k)
     &              +weights(j)*scratch(k)
            enddo
         enddo
      else
         ibin=int((x-plots(label)%xmin)
     &        /(plots(label)%xmax-plots(label)%xmin)
     &        *dble(plots(label)%nbins))+1
         ibin=min(plots(label)%nbins,max(1,ibin))
         do j=1,nweights
            plots(label)%pending(j,ibin)=
     &           plots(label)%pending(j,ibin)+weights(j)
         enddo
      endif
      plots(label)%touched=.true.
      end

      subroutine HwU_add_points
      use Moments_variables
      implicit none
      integer label,k,l,n
      double precision dn,delta
      if (.not.initialized) then
         call moments_fail('initialize before filling')
      endif
      if (.not.allocated(plots)) return
      do label=1,size(plots)
         if (.not.plots(label)%booked) cycle
         if (.not.plots(label)%touched) cycle
         n=plots(label)%ncoeff
         scratch(1:n)=plots(label)%pending(1,:)-plots(label)%mean(1,:)
         plots(label)%nactive=plots(label)%nactive+1
         dn=dble(plots(label)%nactive)
c Welford covariance; event and counterevents have already summed.
         plots(label)%mean=plots(label)%mean+
     &        (plots(label)%pending-plots(label)%mean)/dn
         do l=1,n
            delta=plots(label)%pending(1,l)-plots(label)%mean(1,l)
            do k=1,n
               plots(label)%m2(k,l)=plots(label)%m2(k,l)
     &              +scratch(k)*delta
            enddo
         enddo
         plots(label)%pending=0d0
         plots(label)%touched=.false.
      enddo
      end

      subroutine HwU_accum_iter(include_iteration,npoints,values)
      use Moments_variables
      implicit none
      logical include_iteration
      integer npoints,label,k,l,n
      double precision values(2),alpha,dn,na,factor,ratio
      double precision, allocatable :: current(:,:),cov(:,:)
      if (.not.initialized) then
         call moments_fail('initialize before averaging')
      endif
      if (include_iteration.and.npoints.lt.2) then
         call moments_fail('covariance needs at least two samples')
      endif
      alpha=0d0
      if (include_iteration) then
         if (error_estimation.eq.3) then
            if (any(.not.ieee_is_finite(values)).or.
     &           any(values.lt.0d0)) then
               call moments_fail('invalid MINT iteration uncertainties')
            endif
         endif
         niterations=niterations+1
         if (niterations.eq.1) then
            alpha=1d0
         elseif (error_estimation.eq.1) then
            alpha=1d0/dble(niterations)
         else
            if (values(1).eq.0d0.and.values(2).eq.0d0) then
               alpha=1d0/dble(niterations)
            elseif (values(1).eq.0d0) then
               alpha=1d0
            elseif (values(2).eq.0d0) then
               alpha=0d0
            elseif (values(1).gt.values(2)) then
               ratio=values(2)/values(1)
               alpha=ratio/(1d0+ratio)
            else
               alpha=1d0/(1d0+values(1)/values(2))
            endif
         endif
      endif
      if (.not.allocated(plots)) return
      do label=1,size(plots)
         if (.not.plots(label)%booked) cycle
         if (plots(label)%touched) then
            call moments_fail('unclosed group at iteration boundary')
         endif
         if (include_iteration) then
            n=plots(label)%ncoeff
            allocate(current(nweights,n),cov(n,n))
            dn=dble(npoints)
            na=dble(plots(label)%nactive)
            if (na.gt.dn) then
               call moments_fail('more groups than MINT samples')
            endif
c Add the implicit zero groups (cuts, no fill, out of plotting range)
c as a second Welford population, avoiding sum(z*z)-sum(z)**2/N.
            current=plots(label)%mean*(na/dn)
            factor=na*(dn-na)/dn
            do l=1,n
               do k=1,n
                  cov(k,l)=(plots(label)%m2(k,l)+factor
     &                 *plots(label)%mean(1,k)*plots(label)%mean(1,l))
     &                 /(dn*(dn-1d0))
               enddo
            enddo
            cov=0.5d0*cov+0.5d0*transpose(cov)
            if (any(.not.ieee_is_finite(current)).or.
     &           any(.not.ieee_is_finite(cov))) then
               call moments_fail('nonfinite iteration covariance')
            endif
            plots(label)%result=(1d0-alpha)*plots(label)%result
     &           +alpha*current
            plots(label)%covariance=(1d0-alpha)**2
     &           *plots(label)%covariance+alpha**2*cov
            deallocate(current,cov)
         endif
         plots(label)%nactive=0
         plots(label)%mean=0d0
         plots(label)%m2=0d0
      enddo
      end

c This backend is for fixed-order integration; avoid silently using an
c unweighted-event normalisation/error convention here.
      subroutine finalize_histograms(npoints)
      use Moments_variables
      implicit none
      integer npoints
      call moments_fail('finalize_histograms is unsupported; use fNLO')
      end

      subroutine HwU_output(unit,xnorm)
      use Moments_variables
      implicit none
      integer unit,label,ibin,j,sideunit,ios,n,lenname
      logical named
      double precision xnorm,xl,xr,variance
      double precision, allocatable :: projection(:),weights(:)
      character(len=4096) filename
      if (.not.initialized) then
         call moments_fail('initialize before output')
      endif
      if (.not.ieee_is_finite(xnorm)) then
         call moments_fail('nonfinite output normalization')
      endif
      write(unit,'(a)',advance='no') '##& xmin & xmax & '
     &     //trim(weight_names(1))//' & dy'
      do j=2,nweights
         write(unit,'(a)',advance='no') ' & '//trim(weight_names(j))
      enddo
      write(unit,'(a)') ''
      write(unit,'(a)') ''
      if (allocated(plots)) then
         do label=1,size(plots)
            if (.not.plots(label)%booked) cycle
            n=plots(label)%ncoeff
            allocate(projection(n),weights(nweights))
            write(unit,'(a,i0,a)') '<histogram> ',
     &           plots(label)%nbins,
     &           ' " '//trim(plots(label)%title)//' "'
            do ibin=1,plots(label)%nbins
               call bin_projection(plots(label),ibin,projection)
               weights=matmul(plots(label)%result,projection)*xnorm
               variance=dot_product(projection,
     &              matmul(plots(label)%covariance,projection))
               variance=max(0d0,variance)*xnorm**2
               xl=plots(label)%xmin+(plots(label)%xmax
     &              -plots(label)%xmin)*(dble(ibin-1)
     &              /dble(plots(label)%nbins))
               xr=plots(label)%xmin+(plots(label)%xmax
     &              -plots(label)%xmin)*(dble(ibin)
     &              /dble(plots(label)%nbins))
               write(unit,'(*(1x,es24.16e3))') xl,xr,weights(1),
     &              sqrt(variance),(weights(j),j=2,nweights)
            enddo
            write(unit,'(a)') '<\histogram>'
            write(unit,'(a)') ''
            write(unit,'(a)') ''
            deallocate(projection,weights)
         enddo
      endif
c Save next to the corresponding HwU file, including custom names.
      filename=''
      named=.false.
      inquire(unit=unit,name=filename,named=named,iostat=ios)
      if (ios.ne.0.or..not.named.or.len_trim(filename).eq.0) then
         filename='MADatNLO.Moments'
      else
         lenname=len_trim(filename)
         if (lenname.ge.4) then
            if (filename(lenname-3:lenname).eq.'.HwU') then
               filename=filename(:lenname-4)//'.Moments'
            else
               filename=trim(filename)//'.Moments'
            endif
         else
            filename=trim(filename)//'.Moments'
         endif
      endif
      open(newunit=sideunit,file=trim(filename),status='replace',
     &     action='write',iostat=ios)
      if (ios.ne.0) call moments_fail('cannot open native output file')
      call Moments_output(sideunit,xnorm)
      close(sideunit)
      end

      subroutine Moments_output(unit,xnorm)
      use Moments_variables
      implicit none
      integer unit,label,k,l,j,nhist
      double precision xnorm
      character(len=9) kind
      if (.not.initialized) then
         call moments_fail('initialize before output')
      endif
      if (.not.ieee_is_finite(xnorm)) then
         call moments_fail('nonfinite native output normalization')
      endif
      nhist=0
      if (allocated(plots)) nhist=count(plots%booked)
      write(unit,'(a)') 'MOMENTS 1'
      write(unit,'(a,i0)') 'NWEIGHTS ',nweights
      do j=1,nweights
         write(unit,'(a)') 'WEIGHT '//trim(weight_names(j))
      enddo
      write(unit,'(a,i0)') 'NHIST ',nhist
      if (allocated(plots)) then
         do label=1,size(plots)
            if (.not.plots(label)%booked) cycle
            if (plots(label)%smooth) then
               kind='legendre'
            else
               kind='histogram'
            endif
            write(unit,'(a,i0,1x,a,2(1x,i0),2(1x,es24.16e3))')
     &           'HIST ',label,trim(kind),plots(label)%nbins,
     &           plots(label)%ncoeff,plots(label)%xmin,plots(label)%xmax
            write(unit,'(a)') 'TITLE '//trim(plots(label)%title)
            write(unit,'(a)') 'MEANS'
            do k=1,plots(label)%ncoeff
               write(unit,'(*(1x,es24.16e3))')
     &              (plots(label)%result(j,k)*xnorm,j=1,nweights)
            enddo
            write(unit,'(a)') 'COVARIANCE'
            do k=1,plots(label)%ncoeff
               write(unit,'(*(1x,es24.16e3))')
     &              (plots(label)%covariance(k,l)*xnorm**2,
     &              l=1,plots(label)%ncoeff)
            enddo
            write(unit,'(a)') 'ENDHIST'
         enddo
      endif
      write(unit,'(a)') 'END'
      end

      subroutine HwU_deallocate_all
      use Moments_variables
      implicit none
      if (allocated(plots)) deallocate(plots)
      if (allocated(weight_names)) deallocate(weight_names)
      if (allocated(scratch)) deallocate(scratch)
      nweights=0
      niterations=0
      initialized=.false.
      end

c Legacy topdrawer hooks called by the fixed-order drivers. Actual
c accumulation/output is performed by the HwU-compatible entry points.
      subroutine accum(idummy)
      implicit none
      logical idummy
      end

      subroutine addfil(filename)
      implicit none
      character*(*) filename
      end
