      subroutine LesHouches(p,born_wgt,virt_wgt)
c
c Given the Born momenta, this is the Binoth-Les Houches interface
c file that calls the OLP and returns the virtual weights. For
c convenience also the born_wgt is passed to this subroutine.
c
C********************************************************************
c WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
C********************************************************************
c The Born in MadFKS -- and therefore also the virtual!-- should have
c a slightly adapted identical particle symmetry factor. The normal
c virtual weight as coming from the OLP should be divided by the
c number of gluons in the corresponding real-emission process (i.e.
c the number of gluons in the Born plus one). This factor is passed
c to this subroutine in /numberofparticles/ common block, as
c "ngluons". So, divided virt_wgt by dble(ngluons) to get the correct
c virtual to be used in MadFKS.
C********************************************************************
c
      implicit none

      include "nexternal.inc"
      include "born_nhel.inc"
      include "coupl.inc"
      include "run.inc"

      double precision pi
      parameter (pi=3.1415926535897932385d0)
      integer procnum,i,j,dummyHel(nexternal-1)
      double precision p(0:3,nexternal-1)
      double precision virt_wgt,born_wgt,double,single
      double precision mu,alphaS,alphaEW,virt_wgts(3),UVnorm(2),s
      double precision virtcor,ao2pi,conversion,hel_fac,sumdot
      external sumdot
      logical firsttime
      data firsttime /.true./
      save conversion
      common /LH_procnum /procnum

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks

      character*79         hel_buff(2)
      common/to_helicity/  hel_buff


      INTEGER NHEL(NEXTERNAL-1,2)
      integer nhelall(nexternal-1,max_bhel)
      INTEGER GOODHEL(2),HEL_WGT
      common /c_nhelborn/ nhel,nhelall,goodhel,hel_wgt

      integer nhel_cts(nexternal-1),sum_hel

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      integer nvtozero
      logical doVirtTest
      common/cvirt2test/nvtozero,doVirtTest
      integer ivirtpoints,ivirtpointsExcept
      double precision  virtmax,virtmin,virtsum
      common/cvirt3test/virtmax,virtmin,virtsum,ivirtpoints,
     &     ivirtpointsExcept

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint
      double precision minmax_virt(2,2)
      logical exceptional
      save minmax_virt, exceptional

      double precision vegas_wgt
      double precision vegas_weight
      common/cvegas_weight/vegas_weight
      double precision epinv, epinv2
      common/to_poles/epinv, epinv2

      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      double precision qes2
      common /coupl_es/ qes2
      double precision conv
      logical fksprefact
      parameter (fksprefact=.true.)
      double precision tolerance, madfks_single, madfks_double
      parameter (tolerance = 1d-8)
      integer nbad, nbadmax
      parameter (nbadmax = 5)
      data nbad / 0 /
      include "pmass.inc"

      do i=1, nexternal-1
          dummyHel(i)=1
      enddo

c update the ren_scale for MadLoop and the couplings
      mu_r = scale
      call update_as_param()

      alphaS=g**2/(4d0*PI)
      if (firsttime) write(*,*) "alpha_s value used by MadLoop: ",
     1  alphas
      ao2pi=alphaS/(2d0*Pi)
      alphaEW=dble(gal(1))**2/(4d0*PI)
      call sloopmatrix(p, virt_wgts)
      virt_wgt= virt_wgts(1)/dble(ngluons)
      single  = virt_wgts(2)/dble(ngluons)
      double  = virt_wgts(3)/dble(ngluons)
      vegas_wgt=vegas_weight

c Ellis-Sexton scale squared
      mu = QES2
c     check for poles cancellation      
      if (firsttime) then
          call getpoles(p, mu, madfks_double, madfks_single, fksprefact)
          if (dabs(single - madfks_single).lt.tolerance .and.
     1        dabs(double - madfks_double).lt.tolerance) then
              write(*,*) "---- POLES CANCELLED ----"
              firsttime = .false.
          else
              write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     1         tolerance
              write(*,*) " DOUBLE:"
              write(*,*) "       MadFKS: ", madfks_double,
     1                   "          OLP: ", double
              write(*,*) " SINGLE:"
              write(*,*) "       MadFKS: ",madfks_single,
     1                   "          OLP: ",single
              if (nbad .lt. nbadmax) then
                  nbad = nbad + 1
                  write(*,*) " Trying another PS point"
              else
                  write(*,*) " TOO MANY FAILURES, QUITTING"
                  stop
              endif
          endif
      endif

      return
      end

      subroutine LesHouchesInit(filename)
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
c         write (*,*) 'Error in LesHouchesInit', procnum
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

c      write (*,*) 'From DR to CDR conversion: ',octet,' octets and ',
c     &     triplet,' triplets in Born (both massless), sum =',conversion
c
c         stop
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

