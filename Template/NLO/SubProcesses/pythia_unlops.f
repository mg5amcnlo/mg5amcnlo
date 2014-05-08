
      subroutine pythia_UNLOPS(p,passUNLOPScuts)
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
c cut
      integer npart
      double precision pt_pythia,ptmin1,ptmin2
      include 'cuts.inc'
      double precision pmass(nexternal)
      include 'pmass.inc'

      pt_pythia=ptj
c Set passcut to true, if not it will be updated below.
      passUNLOPScuts=.true.

c convert momenta to pythia8 c++ format
      npart=0
      do i=1,nexternal
         if (p(0,i).gt.0d0) then
            npart=npart+1
         else
            cycle
         endif
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
      if (npart.eq.nexternal) then
         call get_ID_H(id)
      elseif (npart.eq.nexternal-1) then
         call get_ID_S(id)
      else
         write (*,*) 'Error: wrong number of particles in'/
     $        /' pythia_unlops.f',npart,nexternal
         stop
      endif

      eCM=sqrt(4d0*ebeam(1)*ebeam(2))

      call pythia_unlops_cluster(eCM,pin,npart,id,ist,ptmin1,ptmin2)

      if (npart.eq.nexternal-1) then
         ptmin2=ptmin1
         ptmin1=0d0
      elseif (npart.ne.nexternal) then
         write (*,*) 'ERROR in PYTHIA_UNLOPS:'/
     $        /' more than 1 zero-energy particle',npart,nexternal
         stop
      endif
      if (max(ptmin1,ptmin2) .lt. pt_pythia .or. min(ptmin1,ptmin2) .gt.
     $     pt_pythia)  THEN
         passUNLOPScuts = .FALSE.
         RETURN
      ENDIF
      return
      end
     
