
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
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      double precision pin(5,nexternal)
      integer id(nexternal),ist(nexternal)
c Born momenta
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
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
      endif

 100  continue

      if (npart.eq.nexternal) then
         call get_ID_H(id)
         do i=1,nexternal
            if (id(i).eq.-21) id(i)=21
         enddo
      elseif (npart.eq.nexternal-1) then
         call get_ID_S(id)
      endif

      eCM=sqrt(4d0*ebeam(1)*ebeam(2))

      call pythia_unlops_cluster(eCM,pin,npart,id,ist,ptmin1,ptmin2)

      if (npart.eq.nexternal) then
         d1=ptmin1
      elseif(npart.eq.nexternal-1) then
         d2=ptmin1
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
               pin(j,npart)=p1_cnt(mod(j,4),i,0)
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
         endif
         goto 100
      endif

c Here is the actual cut applied
      if (max(d1,d2) .lt. pt_pythia .or. d1 .gt. pt_pythia) then
         passUNLOPScuts = .false.
         return
      endif
      return
      end
     
