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
      double precision pi, zero
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
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
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
      double precision run_tolerance, madfks_single, madfks_double
      parameter (run_tolerance = 1d-4)
      double precision tolerance, acc_found
      integer i,j
      integer nbad, nbadmax
c statistics for MadLoop
      double precision avgPoleRes(2),PoleDiff(2)
      integer nunst, ntot
      common/ups_stats/nunst, ntot
      parameter (nbadmax = 5)
      double precision pmass(nexternal)
      logical unstable_point
      include 'pmass.inc'
      data nbad / 0 /
      if (isum_hel.ne.0) then
         write (*,*) 'Can only do explicit helicity sum'//
     &        ' for Virtual corrections',
     &        isum_hel
      endif
c update the ren_scale for MadLoop and the couplings (should be the
c Ellis-Sexton scale)
      mu_r = sqrt(QES2)
      call update_as_param()
      alpha_S=g**2/(4d0*PI)
      ao2pi= alpha_S/(2d0*PI)
      if (firsttime) then
          write(*,*) "alpha_s value used for the virtuals"/
     &     /" is (for the first PS point): ", alpha_S
          tolerance=1d-5
          call sloopmatrix_thres(p, virt_wgts, tolerance, acc_found)
      else
          tolerance=run_tolerance
c         Just set the accuracy found to a positive value as it is not
c         specified once the initial pole check is performed.
          acc_found=1.0d0
          call sloopmatrix(p, virt_wgts)
      endif
      virt_wgt= virt_wgts(1)/dble(ngluons)
      single  = virt_wgts(2)/dble(ngluons)
      double  = virt_wgts(3)/dble(ngluons)
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
c check for poles cancellation
c If MadLoop was still in initialization mode, then skip the check
c and it will use the next one for that purpose
      if (acc_found.lt.0.0d0) goto 111
      call getpoles(p,QES2,madfks_double,madfks_single,fksprefact)
      ntot = ntot+1
      avgPoleRes(1)=(single+madfks_single)/2.0d0
      avgPoleRes(2)=(double+madfks_double)/2.0d0
      PoleDiff(1)=dabs(single - madfks_single)
      PoleDiff(2)=dabs(double - madfks_double)
      if ((dabs(avgPoleRes(1))+dabs(avgPoleRes(2))).ne.0d0) then
           unstable_point = .not.
     1        (((PoleDiff(1)+PoleDiff(2))/
     1        (dabs(avgPoleRes(1))+dabs(avgPoleRes(2)))).lt.tolerance)
      else
           unstable_point = .not.
     1        ((PoleDiff(1)+PoleDiff(2)).lt.tolerance)
      endif
      if (unstable_point) then
          nunst = nunst+1
          if (nunst.lt.10) then
              if (nunst.eq.1) then
                open(unit=78, file='UPS.log')
              else
                open(unit=78, file='UPS.log', access='append')
              endif
              write(78,*) '===== UPS #',nunst,' ====='
              write(78,*) 'mu_r    =',mu_r           
              write(78,*) 'alpha_S =',alpha_S
              write(78,*) '1/eps**2 expected from MadFKS=',madfks_double
              write(78,*) '1/eps**2 obtained in MadLoop =',double
              write(78,*) '1/eps    expected from MadFKS=',madfks_single
              write(78,*) '1/eps    obtained in MadLoop =',single
              write(78,*) 'finite   obtained in MadLoop =',virt_wgt
              do i = 1, nexternal-1
                write(78,'(i2,1x,5e25.15)') 
     1 i, p(0,i), p(1,i), p(2,i), p(3,i), pmass(i)
              enddo
              close(78)
          endif
      endif


      if (firsttime) then
          if (.not. unstable_point) then
              write(*,*) "---- POLES CANCELLED ----"
              firsttime = .false.
          else
              write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     1         tolerance
              write(*,*) " COEFFICIENT DOUBLE POLE:"
              write(*,*) "       MadFKS: ", madfks_double,
     1                   "          OLP: ", double
              write(*,*) " COEFFICIENT SINGLE POLE:"
              write(*,*) "       MadFKS: ",madfks_single,
     1                   "          OLP: ",single
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
      if(doVirtTest.and.born_wgt.ne.0d0)then
         virtmax=max(virtmax,virt_wgt/born_wgt/ao2pi)
         virtmin=min(virtmin,virt_wgt/born_wgt/ao2pi)
         virtsum=virtsum+virt_wgt/born_wgt/ao2pi
      endif
111   continue
      return
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

