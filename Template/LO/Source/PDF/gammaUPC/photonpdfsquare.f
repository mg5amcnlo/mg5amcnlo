      double precision function photonpdfsquare(x1,x2)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c     ih is now signed <0 for antiparticles
c     if ih<0 does not have a dedicated pdf, then the one for ih>0 will be called
c     and the sign of ipdg flipped accordingly.
c
c     ibeam is the beam identity 1/2
c      if set to -1/-2 it meand that ipdg should not be flipped even if ih<0
c      usefull for re-weighting
c***************************************************************************
      USE ElasticPhotonPhotonFlux
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x1,x2
C
C     Include
C
C      include '../pdf.inc'
C     Common block
      include '../pdf.inc'
c      character*7 pdlabel,epa_label
c      character*7 pdsublabel(2)
c      integer lhaid
c      common/to_pdf/lhaid,pdlabel,epa_label,pdsublabel

      double precision xx1,xx2

      integer nb_proton(2), nb_neutron(2) 
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      integer nb_hadron(2)
C      

      integer i,j
      double precision xlast(2,2),pdflast(2)
      character*7 pdlabellast(2)
      integer ireuse
      save xlast,pdflast,pdlabellast
      data xlast/4*-99d9/
      data pdflast/2*-99d9/
      data pdlabellast/2*'abcdefg'/

c     collider configuration
      integer lpp(2)
      double precision ebeamMG5(2),xbk(2),q2fact(2)
      common/to_collider/ebeamMG5,xbk,q2fact,lpp

      do i=1,2
         nb_hadron(i) = nb_proton(i)+nb_neutron(i)
      enddo
      xx1=x1*nb_hadron(1)
      xx2=x2*nb_hadron(2)
c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return photonpdfsquare=0
c     instead of stopping the code, as this might accidentally happen.
      if (xx1.eq.0d0.or.xx2.eq.0d0) then
         photonpdfsquare=0d0
         return
      elseif (xx1.lt.0d0 .or. xx1.gt.1d0) then
         if (nb_hadron(1).eq.1.or.x1.lt.0d0) then
            write (*,*) 'PDF#1 not supported for Bjorken x ', xx1
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF#1 not supported for Bjorken x ',xx1
            stop 1
         else
            photonpdfsquare = 0d0
            return
         endif
      elseif (xx2.lt.0d0 .or. xx2.gt.1d0) then
         if (nb_hadron(2).eq.1.or.x2.lt.0d0) then
            write (*,*) 'PDF#2 not supported for Bjorken x ', xx2
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF#2 not supported for Bjorken x ',xx2
            stop 1
         else
            photonpdfsquare = 0d0
            return
         endif
      endif

      ireuse = 0
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (xx1.eq.xlast(1,i) .and. xx2.eq.xlast(2,i) .and.
     $        pdlabel.eq.pdlabellast(i)) then
            ireuse = i
         endif
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0)then
         if (pdflast(ireuse).ne.-99d9) then
            photonpdfsquare = pdflast(ireuse)
            return 
         endif
      endif

c     Bjorken x and/or PDF set are not
c     identical to the saved values: this means a new event and we
c     should reset everything to compute new PDF values. Also, determine
c     if we should fill ireuse=1 or ireuse=2.
      if (ireuse.eq.0.and.xlast(1,1).ne.-99d9.and.xlast(2,1).ne.-99d9
     $     .and.xlast(1,2).ne.-99d9.and.xlast(2,2).ne.-99d9)then
         do i=1,2
            xlast(1:2,i)=-99d9
            pdflast(i)=-99d9
            pdlabellast(i)='abcdefg'
         enddo
c     everything has been reset. Now set ireuse=1 to fill the first
c     arrays of saved values below
         ireuse=1
      else if(ireuse.eq.0.and.xlast(1,1).ne.-99d9
     $        .and.xlast(2,1).ne.-99d9)then
c     This is first call after everything has been reset, so the first
c     arrays are already filled with the saved values (hence
c     xlast(1,1).ne.-99d9 and xlast(2,1).ne.-99d9). 
c     Fill the second arrays of saved values (done
c     below) by setting ireuse=2
         ireuse=2
      else if(ireuse.eq.0)then
c     Special: only used for the very first call to this function:
c     xlast(1,i), xlast(2,i) are initialized as data statements to be equal to -99d9
         ireuse=1
      endif

c     Give the current values to the arrays that should be
c     saved. 'pdflast' is filled below.
      xlast(1,ireuse)=xx1
      xlast(2,ireuse)=xx2
      pdlabellast(ireuse)=pdlabel

      if(pdlabel(1:4).eq.'edff') then
         USE_CHARGEFORMFACTOR4PHOTON=.FALSE.
      elseif(pdlabel(1:4).eq.'chff') then
         USE_CHARGEFORMFACTOR4PHOTON=.TRUE.
      else
         WRITE(*,*)"Error: do not know pdlabel = ",pdlabel
         STOP 2
      endif

c     write(*,*) 'running gamma-UPC'

      IF(nb_hadron(1).eq.1.and.nb_hadron(2).eq.1)THEN
         pdflast(ireuse)=PhotonPhotonFlux_pp(xx1,xx2)
      ELSEIF((nb_hadron(1).eq.1.and.nb_hadron(2).gt.1).or.
     $        (nb_hadron(2).eq.1.and.nb_hadron(1).gt.1))THEN
         pdflast(ireuse)=PhotonPhotonFlux_pA_WoodsSaxon(xx1,xx2)
      ELSEIF(nb_hadron(1).gt.1.and.nb_hadron(2).gt.1)THEN
         pdflast(ireuse)=PhotonPhotonFlux_AB_WoodsSaxon(xx1,xx2)
      ELSE
         WRITE(*,*)"Error: do not know nb_hadron(1:2) = ",nb_hadron(1:2)
         STOP 3
      ENDIF
      ! the particular normalisation for MG5 in heavy ion mode
      pdflast(ireuse)=pdflast(ireuse)*nb_hadron(1)*nb_hadron(2)
      photonpdfsquare=pdflast(ireuse)
      
      return
      end
     
