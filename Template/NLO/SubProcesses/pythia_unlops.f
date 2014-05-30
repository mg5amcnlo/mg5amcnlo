
      subroutine pythia_UNLOPS(p,passUNLOPScuts)
c Cut to apply are the following theta functions:
c    [ B + V + int R \theta(ptj - d(\phi_R)) ] \theta(d(\phi_B)-ptj)
c where ptj is the input merging scale and d(\phi) is the scale of the
c first cluster in the \phi phase-space. This means that for the
c real-emission momentum, we also have to apply a cut on the Born
c momentum at the same time. Hence, the Born momenta need to be passed
c here as well (via a common block)
      implicit none
      include "nexternal.inc"
      include "genps.inc"
      include "run.inc"
      include "coupl.inc"
      double precision zero
      parameter (zero=0d0)
c arguments
      double precision p(0:3,nexternal),eCM
      logical passUNLOPScuts
      INTEGER I, J
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
      double precision pin(5,nexternal)
      integer id(nexternal),ist(nexternal)
c Born momenta
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
c cut
      integer npart
      double precision pt_pythia,ptmin1,ptmin2,d1,d2
      include 'cuts.inc'
      double precision pmass(nexternal)
      include 'pmass.inc'

      pt_pythia=ptj
      d1=-1d0
      d2=-1d0
c Set passcut to true, if not it will be updated below.
      passUNLOPScuts=.true.

c convert momenta to pythia8 c++ format
      npart=0
      do i=1,nexternal
         if (p(0,i).eq.0d0) cycle
         npart=npart+1
         do j=1,4
            pin(j,npart)=p(mod(j,4),i)
         enddo
         pin(5,npart)=pmass(i)
         if (i.le.nincoming) then
            ist(npart)=-1
         else
            ist(npart)=1
         endif
      enddo
      if (npart.ne.nexternal-1 .and. npart.ne.nexternal) then
         write (*,*) 'ERROR #1 in pythia_unlops.f',npart,nexternal
         stop
      endif

 100  continue

      if (npart.eq.nexternal) then
         call get_ID_H(id)
      elseif (npart.eq.nexternal-1) then
         call get_ID_S(id)
      endif

      eCM=sqrt(4d0*ebeam(1)*ebeam(2))

      call pythia_unlops_cluster(eCM,pin,npart,id,ist,ptmin1,ptmin2)

      if (npart.eq.nexternal) then
         d1=ptmin1
      elseif(npart.eq.nexternal-1) then
c When cutting the n-body kinematics, increase the scale by 10% to try
c to take any MC masses into account (which will be assigned when
c writing out the events)
         d2=ptmin1*1.1d0
      endif

c In the case we just did the real emission, now, also compute the
c cluster scale for the underlying Born momenta. If we did Born, virtual
c or counter-event, we only need one scale.
      if (npart.eq.nexternal) then
         npart=0
         do i=1,nexternal
            if (i.eq.i_fks) cycle
            npart=npart+1
            do j=1,4
               pin(j,npart)=p_born(mod(j,4),npart)
            enddo
            pin(5,npart)=pmass(i)
            if (i.le.nincoming) then
               ist(npart)=-1
            else
               ist(npart)=1
            endif
         enddo
         if (npart.ne.nexternal-1) then
            write (*,*) 'ERROR #2 in pythia_unlops.f',npart,nexternal
            stop
         endif
         goto 100
      endif

c Here is the actual cut applied
      if (max(d1,d2) .lt. pt_pythia .or. min(d1,d2) .gt. pt_pythia) then
         passUNLOPScuts = .false.
         return
      endif
      return
      end
     

      subroutine pythia_UNLOPS_mass(p,ic,nparts,passUNLOPScuts)
      implicit none
      include "nexternal.inc"
      include "genps.inc"
      include "run.inc"
      double precision zero
      parameter (zero=0d0)
c arguments
      double precision p(0:4,2*nexternal-3),eCM
      integer ic(7,2*nexternal-3),nparts
      logical passUNLOPScuts
      INTEGER I,J
      double precision pin(5,nexternal-1)
      integer id(nexternal-1),ist(nexternal-1)
c cut
      integer npart
      double precision pt_pythia,ptmin1,ptmin2,d1,d2
      include 'cuts.inc'
      pt_pythia=ptj
      d1=-1d0
c Set passcut to true, if not it will be updated below.
      passUNLOPScuts=.true.

c convert momenta to pythia8 c++ format
      npart=0
      do i=1,nparts
         if (abs(ic(6,i)).ne.1) cycle
         npart=npart+1
         do j=1,5
            if (j.ne.4) then
               pin(j,npart)=p(j,i)
            else
               pin(j,npart)=p(0,i)
            endif
         enddo
         ist(npart)=ic(6,i)
         id(npart)=ic(1,i)
      enddo
      if (npart.ne.nexternal-1) then
         write (*,*) 'ERROR #1 in pythia_unlops_mass',npart,nexternal
         stop
      endif

      eCM=sqrt(4d0*ebeam(1)*ebeam(2))

      call pythia_unlops_cluster(eCM,pin,npart,id,ist,ptmin1,ptmin2)
      d2=ptmin1

c Here is the actual cut applied
      if (d2.lt.pt_pythia) then
         passUNLOPScuts = .false.
         return
      endif
      return
      end
