************************************************************************
* This is the file  avh_oni_cuba.f  of the package                     *
*                                                                      *
*                  Oneloop with Numerical Integration                  *
*                                                                      *
* for the evaluation of 1-loop scalar 1-, 2-, 3- and 4-point functions *
*                                                                      *
* author: Andreas van Hameren <hamerenREMOVETHIS@ifj.edu.pl>           *
*   date: 28-07-2010                                                   *
************************************************************************
*                                                                      *
* Have a look at the file  avh_oni_hello.f  for more information.      *
*                                                                      *
************************************************************************

      subroutine avh_oni_cuba( rslt ,func ,ndim ,label )
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex rslt
      external       func
      integer        ndim
      character(6)   label
      double precision sum0(2),sum1(2),sum2(2)
      integer avh_oni_unitcuba
*
* parameters cuba
      integer ncomp, mineval, maxeval, verbose, last ,avh_oni_maxeval
      double precision epsrel, epsabs
      parameter (ncomp = 2)
      parameter (epsrel = 1D-8)
      parameter (epsabs = 1D-8)
      parameter (verbose = 0)
      parameter (last = 4)
      parameter (mineval = 0)
!      parameter (maxeval = 10 000 000)
      integer key
      parameter (key = 0)
      integer nregions, neval, fail
      maxeval = avh_oni_maxeval()
*
      call cuhre(ndim, ncomp, func,
      include 'avh_oni_cuba.h'
     &    epsrel, epsabs, verbose + last, mineval, maxeval,
     &    key,
     &    nregions, neval, fail, sum1, sum2, sum0)
c        print *, "nregions =", nregions
c        print *, "neval    =", neval
c        print *, "fail     =", fail
c        print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
c     &    (integral(c), error(c), prob(c), c = 1, ncomp)
*
      if (avh_oni_unitcuba().gt.0) then
        write(avh_oni_unitcuba(),'(a27,a6,a9,e8.2,e9.2,a15,f3.1)')
     &  ' MESSAGE from avh_oni_cuba ',label
     &  ,': RelErr=',sum2(1)/dabs(sum1(1)),sum2(2)/dabs(sum1(2))
     &  ,', log10(Neval)=',dlog(1d0*neval)/dlog(10d0)
      endif
      rslt = dcmplx(sum1(1),sum1(2))
      end


      subroutine avh_oni_1dim( rslt ,func ,label )
*  ********************************************************************
*  ********************************************************************
      implicit none
      double complex rslt
      external       func
      character(6)   label
      integer npts2,limit,neval,ier,avh_oni_unitcuba
!      parameter( uunit=0 )
      parameter( npts2=2 )
      double precision epsabs,epsrel,points(npts2),abserr,x0,xn
      parameter( epsabs=0d0 ,epsrel=1d-8 )
!      parameter( epsabs=1d-3 ,epsrel=1d-3 )
      parameter( limit=100 )
      x0 = 0d0
      xn = 1d0
      points(npts2-1) = x0
      points(npts2  ) = xn
      call avh_oni_dqagpe( func ,x0,xn ,npts2,points
     &                    ,epsabs,epsrel ,limit ,rslt ,abserr,neval,ier)
      if (avh_oni_unitcuba().gt.0) then
        write(avh_oni_unitcuba(),'(a27,a6,a9,e8.2,a8,i6)')
     &  ' MESSAGE from avh_oni_1dim ',label
     &  ,': RelErr=',abserr/cdabs(rslt)
     &  ,', Neval=',neval
      endif
      end


      subroutine avh_oni_maxeval_set( maxeval_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer maxeval_in
      integer                  maxeval,nunit
      common/avh_oni_cuba_com/ maxeval,nunit
      integer avh_oni_maxeval,avh_oni_un_get
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        maxeval = avh_oni_maxeval()
      endif
      maxeval = maxeval_in
      if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &  'MESSAGE from avh_oni_set_maxeval: maxeval set to',maxeval
      end
*
      integer function avh_oni_maxeval()
      implicit none
      integer                  maxeval,nunit
      common/avh_oni_cuba_com/ maxeval,nunit
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        maxeval = 1000000
      endif
      avh_oni_maxeval = maxeval
      end


      subroutine avh_oni_unitcuba_set( unit_in )
*  ********************************************************************
*  ********************************************************************
      implicit none
      integer unit_in
      integer                  maxeval,nunit
      common/avh_oni_cuba_com/ maxeval,nunit
      integer avh_oni_unitcuba,avh_oni_un_get
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        nunit = avh_oni_unitcuba()
      endif
      nunit = unit_in
      if (avh_oni_un_get().gt.0) write(avh_oni_un_get(),*)
     &  'MESSAGE from avh_oni_set_unitcuba: unit set to',nunit
      end
*
      integer function avh_oni_unitcuba()
      implicit none
      integer                  maxeval,nunit
      common/avh_oni_cuba_com/ maxeval,nunit
      logical init
      data init/.true./
      save init
      if (init) then
        init = .false.
        nunit = 6
      endif
      avh_oni_unitcuba = nunit
      end
