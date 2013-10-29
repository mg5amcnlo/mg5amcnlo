c Wrapper routines for the fixed order analyses
      subroutine initplot
      implicit none
      include 'run.inc'
      include 'reweight0.inc'
      integer nwgt,max_weight
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*15 weights_info(max_weight)
      nwgt=1
      weights_info(nwgt)="central value  "
      if (do_rwgt_scale) then
         nwgt=nwgt+9
         if (numscales.ne.3) then
            write (*,*) 'ERROR #1 in initplot:',numscales
            stop 1
         endif
         write (weights_info(nwgt-8),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",1.0,"muF=",1d0
         write (weights_info(nwgt-7),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",1.0,"muF=",rw_Fscale_up
         write (weights_info(nwgt-6),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",1.0,"muF=",rw_Fscale_down
         write (weights_info(nwgt-5),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_up,"muF=",1d0
         write (weights_info(nwgt-4),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_up,"muF=",rw_Fscale_up
         write (weights_info(nwgt-3),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_up,"muF=",rw_Fscale_down
         write (weights_info(nwgt-2),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_down,"muF=",1d0
         write (weights_info(nwgt-1),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_down,"muF=",rw_Fscale_up
         write (weights_info(nwgt  ),'(a4,e3.1,x,a4,e3.1)')
     &        "muR=",rw_Rscale_down,"muF=",rw_Fscale_down
      endif
      if (do_rwgt_pdf) then
         npdfs=pdf_set_max-pdf_set_min+1
         if (nwgt+npdfs.gt.max_weight) then
            write (*,*) "ERROR in initplot: "/
     $           /"too many scales in reweighting"
            stop 1
         endif
         do i=nwgt+1,nwgt+npdfs
            write(weights_info(i),'(a4,i8,a3)')
     &           'PDF=',pdf_set_min-1+i,'   '
         enddo
         nwgt=nwgt+npdfs
      endif
      call analysis_begin(nwgt,weights_info)
      return
      end


      subroutine topout
      implicit none
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      real*8 xnorm
      if(usexinteg.and..not.mint) then
         xnorm=1.d0/float(itmax)
      elseif(mint) then
         xnorm=1.d0/float(ncall*itmax)
      endif
      call analysis_end(xnorm)
      return                
      end


      subroutine outfun(pp,ybst_til_tolab,www,itype)
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
C
C In MadFKS, the momenta PP given in input to this function are in the
C reduced parton c.m. frame. If need be, boost them to the lab frame.
C The rapidity of this boost is
C
C       YBST_TIL_TOLAB
C
C also given in input
C
C This is the rapidity that enters in the arguments of the sinh() and
C cosh() of the boost, in such a way that
C       ylab = ycm - ybst_til_tolab
C where ylab is the rapidity in the lab frame and ycm the rapidity
C in the center-of-momentum frame.
C
C *WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING**WARNING*
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'reweight0.inc'
      double precision pp(0:3,nexternal),ybst_til_tolab,www
      integer itype
      double precision p(0:4,nexternal),pplab(0:3,nexternal),chybst
     $     ,shybst,chybstmo
      integer i
      double precision xd(3)
      data (xd(i),i=1,3) /0d0,0d0,1d0/
      integer istatus(nexternal),iPDG(nexternal)
      double precision pmass(nexternal)
      common/to_mass/pmass
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
      integer nwgt,max_weight
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision wgts(max_weight)
c Born, n-body or (n+1)-body contribution:
      if(itype.eq.11) then
         ibody=2 ! (n+1)-body
      elseif(itype.eq.12)then
         ibody=1 ! n-body
      elseif(itype.eq.20)then
         ibody=0 ! Born
      else
         write(*,*)'Error in outfun: unknown itype',itype
         stop
      endif
c Boost the momenta to the lab frame:
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
        call boostwdir2(chybst,shybst,chybstmo,xd,
     #                  pp(0,i),pplab(0,i))
      enddo
c Fill the arrays (momenta, status and PDG):
      do i=1,nexternal
         if (i.le.nincoming) then
            istatus(i)=-1
         else
            istatus(i)=1
         endif
         do j=0,3
            p(j,i)=pplab(j,i)
         enddo
         p(4,i)=pmass(i)
         ipdg(i)=idup(i)
      enddo
c The weights comming from reweighting:
      nwgt=1
      wgts(1)=www
      if (do_rwgt_scale) then
         do i=1,numscales
            do j=1,numscales
               nwgt=nwgt+1
               wgts(nwgt)=wgtxsecmu(i,j)
            enddo
         enddo
      endif
      if (do_rwgt_pdf) then
         do i=1,2*numPDFpairs
            nwgt=nwgt+1
            wgts(nwgt)=wgtxsecPDF(i)
         enddo
      endif
      call analysis_fill(p,istatus,ipdg,wgts,ibody)
 999  return      
      end
