      subroutine BinothLHA(p,born_wgt,virt_wgt)
c
c Given the Born momenta, this is the Binoth-Les Houches interface file
c that calls the OLP and returns the virtual weights. For convenience
c also the born_wgt is passed to this subroutine.
c
C************************************************************************
c
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      double precision p(0:3,nexternal-1)
      double precision virt_wgt,born_wgt,double,single,virt_wgts(3)
      double precision mu,ao2pi,conversion,alpha_S
      save conversion
      logical firsttime,firsttime_conversion
      data firsttime,firsttime_conversion /.true.,.true./
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      double precision qes2
      common /coupl_es/ qes2
      logical fksprefact
      parameter (fksprefact=.true.)
      double precision tolerance, madfks_single, madfks_double
      parameter (tolerance = 1d-6)
      integer nbad, nbadmax
      parameter (nbadmax = 5)
      data nbad / 0 /
      if (isum_hel.ne.0) then
         write (*,*) 'Can only do explicit helicity sum'//
     &        ' for Virtual corrections',
     &        isum_hel
      endif
      virt_wgt=0d0
c update the ren_scale for MadLoop and the couplings (should be the
c Ellis-Sexton scale)
      mu_r = sqrt(QES2)
      call update_as_param()
      alpha_S=g**2/(4d0*PI)
      ao2pi= alpha_S/(2d0*PI)

c======================================================================
c Replace the following line with the call to the one-loop code you 
c wish to use. virt_wgts contains finite part, single and double pole
c      
c      call sloopmatrix(p, virt_wgts)
c      virt_wgt= virt_wgts(1)
c      single  = virt_wgts(2)
c      double  = virt_wgts(3)
c
c======================================================================

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

c======================================================================
c example for checking the cancelation of the poles
c      if (firsttime) then
c          call getpoles(p,QES2,madfks_double,madfks_single,fksprefact)
c          if (dabs(single - madfks_single).lt.tolerance .and.
c     &        dabs(double - madfks_double).lt.tolerance) then
c              write(*,*) "---- POLES CANCELLED ----"
c              firsttime = .false.
c          else
c              write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
c     &         tolerance
c              write(*,*) " COEFFICIENT DOUBLE POLE:"
c              write(*,*) "       MadFKS: ", madfks_double,
c     &                   "          OLP: ", double
c              write(*,*) " COEFFICIENT SINGLE POLE:"
c              write(*,*) "       MadFKS: ",madfks_single,
c     &                   "          OLP: ",single
c              write(*,*) " FINITE:"
c              write(*,*) "          OLP: ",virt_wgt
c              if (nbad .lt. nbadmax) then
c                  nbad = nbad + 1
c                  write(*,*) " Trying another PS point"
c              else
c                  write(*,*) " TOO MANY FAILURES, QUITTING"
c                  stop
c              endif
c          endif
c      endif
c======================================================================
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

