      subroutine BinothLHA(pin,born_wgt,virt_wgt)
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
      include "Binoth_proc.inc"
      include "FKSParams.inc"
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      double precision pin(0:3,nexternal-1),p(0:4,nexternal-1)
      double precision virt_wgt,born_wgt,double,single,born,virt_wgts(4)
      double precision mu,ao2pi,conversion,alpha_S
      save conversion
      logical firsttime_pole,firsttime_conversion,firsttime_init
      data firsttime_pole,firsttime_conversion,firsttime_init
     &     /.true.,.true.,.true./
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      double precision qes2
      common /coupl_es/ qes2
      logical fksprefact
      parameter (fksprefact=.true.)
      double precision tolerance, madfks_single, madfks_double
      integer nbad, nbadmax
      parameter (nbadmax = 5)
      data nbad / 0 /
      character*13 filename
      integer i,j
      double precision zero,pmass(nexternal)
      parameter (zero=0d0)
      include 'pmass.inc'
      if (isum_hel.ne.0) then
         write (*,*) 'Can only do explicit helicity sum'//
     &        ' for Virtual corrections',
     &        isum_hel
      endif
      virt_wgt=0d0
c update the ren_scale for the OLP and in the couplings (should be the
c Ellis-Sexton scale)
      mu_r = sqrt(QES2)
      call update_as_param()
      alpha_S=g**2/(4d0*PI)
      ao2pi= alpha_S/(2d0*PI)
c get the momenta in the BLHA format
      do i=1,nexternal-1
         do j=0,3
            p(j,i)=pin(j,i)
         enddo
         p(4,i)=pmass(i)
      enddo
c======================================================================
c Replace the following line with the call to the one-loop code you wish
c to use. virt_wgts contains finite part, single and double pole and the
c Born. 
c
      if (firsttime_init) then
         call BinothLHAInit()
         firsttime_init=.false.
      endif
      call OLP_EvalSubProcess(proc_label,p,mu_r,alpha_S,virt_wgts)
      double  = virt_wgts(1)
      single  = virt_wgts(2)
      virt_wgt= virt_wgts(3)
      born    = virt_wgts(4)
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
      if (firsttime_pole) then
         tolerance=IRPoleCheckThreshold  ! set in FKS_params.dat
         call getpoles(pin,QES2,madfks_double,madfks_single,fksprefact)
         if ( dabs(single - madfks_single).lt.tolerance .and.
     &        dabs(double - madfks_double).lt.tolerance) then
            write(*,*) "---- POLES CANCELLED ----"
            firsttime_pole = .false.
         else
            write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     &           tolerance
            write(*,*) " BORN:"
            write(*,*) "       MadFKS: ", born_wgt,
     &           "          OLP: ", born
            write(*,*) " COEFFICIENT DOUBLE POLE:"
            write(*,*) "       MadFKS: ", madfks_double,
     &           "          OLP: ", double
            write(*,*) " COEFFICIENT SINGLE POLE:"
            write(*,*) "       MadFKS: ",madfks_single,
     &           "          OLP: ",single
            write(*,*) " FINITE:"
            write(*,*) "          OLP: ",virt_wgt
            if (nbad .lt. nbadmax) then
               nbad = nbad + 1
               write(*,*) " Trying another PS point"
            else
               write(*,*) "ERROR: TOO MANY FAILURES, QUITTING"
               stop
            endif
         endif
      endif
c======================================================================
      return
      end

      subroutine BinothLHAInit()
      implicit none
      include 'coupl.inc'
      character*13 filename
      integer ierr
      filename='OLE_order.olc'
      ierr=0
      call OLP_Start(filename//Char(0),ierr)
      if (ierr.eq.0) then
         write (*,*) 'ERROR in the BinothLHAInit process initialization'
         stop
      endif
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
