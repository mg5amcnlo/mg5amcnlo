      subroutine BinothLHA(p,born_wgt,virt_wgt)
c
c Given the Born momenta, this is the Binoth-Les Houches interface file
c that calls the OLP and returns the virtual weights. For convenience
c also the born_wgt is passed to this subroutine.
c
C************************************************************************
c WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
C************************************************************************
c The Born in MadFKS -- and therefore also the virtual!-- should have a
c slightly adapted identical particle symmetry factor. The normal
c virtual weight as coming from the OLP should be divided by the number
c of gluons in the corresponding real-emission process (i.e.  the number
c of gluons in the Born plus one). This factor is passed to this
c subroutine in /numberofparticles/ common block, as "ngluons". So,
c divided virt_wgt by dble(ngluons) to get the correct virtual to be
c used in MadFKS. The born_wgt that is passed to this subroutine has
c already been divided by this factor.
C************************************************************************
c
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include 'born_nhel.inc'
      double precision pi, zero,mone
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0d0)
      double precision p(0:3,nexternal-1)
      double precision virt_wgt,born_wgt,double,single,virt_wgts(3)
      double precision mu,ao2pi,conversion,alpha_S
      save conversion
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      logical firsttime,firsttime_conversion
      data firsttime,firsttime_conversion /.true.,.true./
      double precision qes2
      common /coupl_es/ qes2
      integer nvtozero
      logical doVirtTest 
      common/cvirt2test/nvtozero,doVirtTest
      integer ivirtpoints,ivirtpointsExcept
      double precision  virtmax,virtmin,virtsum
      common/cvirt3test/virtmax,virtmin,virtsum,ivirtpoints,
     &     ivirtpointsExcept
      logical fksprefact
      parameter (fksprefact=.true.)
      integer ret_code
      double precision run_tolerance, madfks_single, madfks_double
      parameter (run_tolerance = 1d-4)
      double precision tolerance,acc_found,prec_found
      integer i,j
      integer nbad, nbadmax
c statistics for MadLoop
      double precision avgPoleRes(2),PoleDiff(2)
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1
      parameter (nbadmax = 5)
      double precision pmass(nexternal)
      integer goodhel(max_bhel),hel(0:max_bhel)
      save hel,goodhel
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      logical cpol
      include 'pmass.inc'
      data nbad / 0 /
c update the ren_scale for MadLoop and the couplings (should be the
c Ellis-Sexton scale)
      mu_r = sqrt(QES2)
      call update_as_param()
      alpha_S=g**2/(4d0*PI)
      ao2pi= alpha_S/(2d0*PI)
      virt_wgt= 0d0
      single  = 0d0
      double  = 0d0
      if (firsttime) then
         write(*,*) "alpha_s value used for the virtuals"/
     &        /" is (for the first PS point): ", alpha_S
         tolerance=1d-5
         call sloopmatrix_thres(p, virt_wgts, tolerance, acc_found,
     $        ret_code)
         virt_wgt= virt_wgts(1)/dble(ngluons)
         single  = virt_wgts(2)/dble(ngluons)
         double  = virt_wgts(3)/dble(ngluons)
      else
         tolerance=run_tolerance ! for the poles check below
c Just set the accuracy found to a positive value as it is not specified
c once the initial pole check is performed.
         acc_found=1.0d0
         if (mc_hel.eq.0) then
            mone=-1d0
            call sloopmatrix_thres(p,virt_wgts,mone,prec_found
     $           ,ret_code)
            virt_wgt= virt_wgts(1)/dble(ngluons)
            single  = virt_wgts(2)/dble(ngluons)
            double  = virt_wgts(3)/dble(ngluons)
         else
c Get random integer from importance sampling (the return value is
c filled in driver_mintMC.f; we cannot do it here, because we need to
c include the phase-space jacobians and all that)
            call get_MC_integer(2,hel(0),ihel,volh)
            fillh=.true.
            do i=ihel,ihel+(mc_hel-1) ! sum over i successive helicities
               mone=-1d0
               call sloopmatrixhel_thres(p,hel(i),virt_wgts,mone
     $              ,prec_found,ret_code)
               virt_wgt= virt_wgt +
     &              virt_wgts(1)*dble(goodhel(i))/(volh*dble(mc_hel))
               single  = single   +
     &              virt_wgts(2)*dble(goodhel(i))/(volh*dble(mc_hel))
               double  = double   +
     &              virt_wgts(3)*dble(goodhel(i))/(volh*dble(mc_hel))
            enddo
c Average over initial state helicities (and take the ngluon factor into
c account)
            if (nincoming.ne.2) then
               write (*,*)
     &              'Cannot do MC over helicities for 1->N processes'
               stop
            endif
            virt_wgt=virt_wgt/4d0/dble(ngluons)
            single  = single/4d0/dble(ngluons)
            double  = double/4d0/dble(ngluons)
         endif
      endif
c======================================================================
c If the Virtuals are in the Dimensional Reduction scheme, convert them
c to the CDR scheme with the following factor (not needed for MadLoop,
c because they are already in the CDR scheme format)
c      if (firsttime_conversion) then
c         call DRtoCDR(conversion)
c         firsttime_conversion=.false.
c      endif
c      virt_wgt=virt_wgt+conversion*born_wgt*ao2pi
c======================================================================
c check for poles cancellation for the first couple of PS points
c Check poles for the first PS points (but not for the initialization PS
c points)
      if (firsttime .and. mod(ret_code,100)/10.ne.3 .and.
     &     mod(ret_code,100)/10.ne.4) then 
         call getpoles(p,QES2,madfks_double,madfks_single,fksprefact)
         avgPoleRes(1)=(single+madfks_single)/2.0d0
         avgPoleRes(2)=(double+madfks_double)/2.0d0
         PoleDiff(1)=dabs(single - madfks_single)
         PoleDiff(2)=dabs(double - madfks_double)
         if ((dabs(avgPoleRes(1))+dabs(avgPoleRes(2))).ne.0d0) then
            cpol = .not. (((PoleDiff(1)+PoleDiff(2))/
     $          (dabs(avgPoleRes(1))+dabs(avgPoleRes(2)))).lt.tolerance)
         else
            cpol = .not.(PoleDiff(1)+PoleDiff(2).lt.tolerance)
         endif
         if (.not. cpol) then
            write(*,*) "---- POLES CANCELLED ----"
            firsttime = .false.
            if (mc_hel.ne.0) then
c Set-up the MC over helicities. This assumes that the 'HelFilter.dat'
c exists, which should be the case when firsttime is false.
               open (unit=67,file='HelFilter.dat',status='old',err=201)
               hel(0)=0
               j=0
               do i=1,max_bhel
                  read(67,*,err=201) goodhel(i)
                  if (goodhel(i).gt.-10000 .and. goodhel(i).ne.0) then
                     j=j+1
                     goodhel(j)=goodhel(i)
                     hel(0)=hel(0)+1
                     hel(j)=i
                  endif
               enddo
c Only do MC over helicities if there are 5 or more non-zero
c (independent) helicities
               if (hel(0).lt.5) then
                  write (*,'(a,i3,a)') 'Only ',hel(0)
     $                 ,' independent helicities:'/
     $                 /' switching to explicitly summing over them'
                  mc_hel=0
               endif
               close(67)
            endif
         else
            write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     &           tolerance
            write(*,*) " COEFFICIENT DOUBLE POLE:"
            write(*,*) "       MadFKS: ", madfks_double,
     &           "          OLP: ", double
            write(*,*) " COEFFICIENT SINGLE POLE:"
            write(*,*) "       MadFKS: ",madfks_single,
     &           "          OLP: ",single
            write(*,*) " FINITE:"
            write(*,*) "          OLP: ",virt_wgt
            write(*,*) 
            write(*,*) " MOMENTA (Exyzm): "
            do i = 1, nexternal-1
               write(*,*) i, p(0,i), p(1,i), p(2,i), p(3,i), pmass(i)
            enddo
            write(*,*) 
            write(*,*) " SCALE**2: ", QES2
            if (nbad .lt. nbadmax) then
               nbad = nbad + 1
               write(*,*) " Trying another PS point"
            else
               write(*,*) " TOO MANY FAILURES, QUITTING"
               stop
            endif
         endif
      endif
c Update the statistics using the ret_code:
      ntot = ntot+1             ! total number of PS
      if (ret_code/100.eq.1) then
         nsun = nsun+1          ! stability unknown
      elseif (ret_code/100.eq.2) then
         nsps = nsps+1          ! stable PS point
      elseif (ret_code/100.eq.3) then
         nups = nups+1          ! unstable PS point, but rescued
      elseif (ret_code/100.eq.4) then
         neps = neps+1          ! exceptional PS point: unstable, and not possible to rescue
      else
         n100=n100+1            ! no known ret_code (100)
      endif
      if (mod(ret_code,100)/10.eq.1 .or. mod(ret_code,100)/10.eq.3) then
         nddp = nddp+1          ! only double precision was used
         if (mod(ret_code,100)/10.eq.3) nini=nini+1 ! MadLoop initialization phase
      elseif (mod(ret_code,100)/10.eq.2 .or. mod(ret_code,100)/10.eq.4)
     $        then
         nqdp = nqdp+1          ! quadruple precision was used
         if (mod(ret_code,100)/10.eq.4) nini=nini+1 ! MadLoop initialization phase
      else
         n10=n10+1              ! no known ret_code (10)
      endif
      if (mod(ret_code,10).ne.0) then
         n1=n1+1                ! no known ret_code (1)
      endif

      if (.not. firsttime .and. ret_code/100.eq.4) then
         if (neps.lt.10) then
            if (neps.eq.1) then
               open(unit=78, file='UPS.log')
            else
               open(unit=78, file='UPS.log', access='append')
            endif
            write(78,*) '===== EPS #',neps,' ====='
            write(78,*) 'mu_r    =',mu_r           
            write(78,*) 'alpha_S =',alpha_S
            if (mc_hel.ne.0) then
               write (78,*)'helicity (MadLoop only)',hel(i),mc_hel
            endif
            write(78,*) '1/eps**2 expected from MadFKS=',madfks_double
            write(78,*) '1/eps**2 obtained in MadLoop =',double
            write(78,*) '1/eps    expected from MadFKS=',madfks_single
            write(78,*) '1/eps    obtained in MadLoop =',single
            write(78,*) 'finite   obtained in MadLoop =',virt_wgt
            do i = 1, nexternal-1
               write(78,'(i2,1x,5e25.15)') 
     &              i, p(0,i), p(1,i), p(2,i), p(3,i), pmass(i)
            enddo
            close(78)
         endif
      endif
      return
 201  write (*,*) 'Cannot do MC over hel:'/
     &     /' "HelFilter.dat" does not exist'/
     &     /' or does not have the correct format'
      stop
      end

      subroutine BinothLHAInit(filename)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      integer status,procnum
      double precision s,mu,sumdot
      external sumdot
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      character*13 filename
      common /LH_procnum /procnum

c Rocket:
c      call get_procnum(filename,procnum)
c      call Init(filename,status)
c      if (status.ne.1) then
c         write (*,*) 'Something wrong with Rocket Les Houches '//
c     &        'initialization',status
c$$$         stop
c      endif
c BlackHat:
c      call get_procnum(filename,procnum)
c      if(procnum.ne.1) then
c         write (*,*) 'Error in BinothLHAInit', procnum
c         stop
c       endif
c      call OLE_Init(filename//Char(0))
      return
      end


      subroutine DRtoCDR(conversion)
c This subroutine computes the sum in Eq. B.3 of the MadFKS paper
c for the conversion from dimensional reduction to conventional
c dimension regularization.
      implicit none
      double precision conversion
      double precision CA,CF
      parameter (CA=3d0,CF=4d0/3d0)
      integer i,triplet,octet
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "coupl.inc"

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      include "pmass.inc"

      triplet=0
      octet=0
      conversion = 0d0
      do i=1,nexternal
         if (i.ne.i_fks .and. i.ne.j_fks) then
            if (pmass(i).eq.0d0) then
               if (abs(particle_type(i)).eq.3) then
                  conversion=conversion-CF/2d0
                  triplet=triplet+1
               elseif (particle_type(i).eq.8) then
                  conversion=conversion-CA/6d0
                  octet=octet+1
               endif
            endif
         elseif(i.eq.min(i_fks,j_fks)) then
            if (pmass(j_fks).eq.0d0 .and. pmass(i_fks).eq.0d0) then
               if (m_type.eq.8) then
                  conversion=conversion-CA/6d0
                  octet=octet+1
               elseif (abs(m_type).eq.3)then
                  conversion=conversion-CF/2d0
                  triplet=triplet+1
               else
                  write (*,*)'Error in DRtoCDR, fks_mother must be'//
     &                 'triplet or octet',i,m_type
                  stop
               endif
            endif
         endif
      enddo
      write (*,*) 'From DR to CDR conversion: ',octet,' octets and ',
     &     triplet,' triplets in Born (both massless), sum =',conversion
      return
      
      end


      subroutine get_procnum(filename,procnum)
      implicit none
      integer procnum,lookhere,procsize
      character*13 filename
      character*176 buff
      logical done

      open (unit=68,file=filename,status='old')
      done=.false.
      do while (.not.done)
         read (68,'(a)',end=889)buff
         if (index(buff,'->').ne.0) then
c Rocket
c            lookhere=index(buff,'process')+7
c BlackHat
c            lookhere=index(buff,'|')
            if (lookhere.ne.0 .and. lookhere.lt.170) then
c Rocket
c               read (buff(lookhere+1:176),*) procnum
c BlackHat
c               read (buff(lookhere+1:176),*) procsize,procnum
c               if (procsize.ne.1) then
c                  write (*,*)
c     &                 'Can only deal with 1 procnum per (sub)process',
c     &                 procsize
c               else
                  write (*,*)'Read process number from contract file',
     &                 procnum
                  close(68)
                  return
c               endif
               done=.true.
            else
               write (*,*) 'syntax contract file not understandable',
     &              lookhere
               stop
            endif
         endif
      enddo
      stop

      close(68)

      return

 889  write (*,*) 'Error in contract file'
      stop
      end

