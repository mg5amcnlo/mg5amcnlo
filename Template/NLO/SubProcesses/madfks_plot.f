c Wrapper routines for the fixed order analyses
      subroutine initplot
      implicit none
      include 'run.inc'
      include "nexternal.inc"
      include 'reweight0.inc'
      integer nwgt,max_weight
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      character*50 weights_info(max_weight)
      character*13 temp
      integer i,npdfs,ii,jj,n,kk,nn
      double precision xsecScale_acc(maxscales,maxscales,maxdynscales)
     $     ,xsecPDFr_acc(0:maxPDFs,maxPDFsets)
      common /scale_pdf_print/xsecScale_acc,xsecPDFr_acc
      integer iappl
      common /for_applgrid/ iappl
      include "appl_common.inc"
      nwgt=1
      weights_info(nwgt)="central value               "
      if (do_rwgt_scale) then
         do kk=1,dyn_scale(0)
c set the weights_info string for scale variations
            if (lscalevar(kk)) then
               do ii=1,nint(scalevarF(0))
                  do jj=1,nint(scalevarR(0))
                     nwgt=nwgt+1
                     if (ickkw.ne.-1) then
                        write(weights_info(nwgt),
     &                                '(a4,i4,x,a4,f6.3,x,a4,f6.3)')
     $                       "dyn=",dyn_scale(kk),"muR=",scalevarR(jj)
     $                       ,"muF=",scalevarF(ii)
                     else
                        write(weights_info(nwgt),
     &                                '(a4,i4,x,a4,f6.3,x,a4,f6.3)')
     $                       "dyn=",dyn_scale(kk),"muS=",scalevarR(jj)
     $                       ,"muH=",scalevarF(ii)
                     endif
                  enddo
               enddo
            else
               nwgt=nwgt+1
               if (ickkw.ne.-1) then
                  write(weights_info(nwgt),'(a4,i4,x,a4,f6.3,x,a4,f6.3)')
     $                 "dyn=",dyn_scale(kk),"muR=",scalevarR(1)
     $                 ,"muF=",scalevarF(1)
               else
                  write(weights_info(nwgt),'(a4,i4,x,a4,f6.3,x,a4,f6.3)')
     $                 "dyn=",dyn_scale(kk),"muS=",scalevarR(1)
     $                 ,"muH=",scalevarF(1)
               endif
            endif
         enddo
      endif
      if (do_rwgt_pdf) then
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               write (*,*) "Including central PDF with "/
     $              /"uncertainties for "//lhaPDFsetname(nn)
            else
               write (*,*) "Including central PDF for "
     $              //lhaPDFsetname(nn)
            endif
c     Load all the PDF sets (the 1st one has already by loaded by the call
c     to "setrun")
            if (nn.gt.1) then
               call initpdfsetbynamem(nn,lhaPDFsetname(nn))
               if (lpdfvar(nn)) then
                  call numberPDFm(nn,nmemPDF(nn))
                  if (nmemPDF(nn).eq.1) then
                     nmemPDF(nn)=0
                     lpdfvar(nn)=.false.
                  endif
               else
                  nmemPDF(nn)=0
               endif
            endif
            if(nmemPDF(nn)+1.gt.maxPDFs)then
               write(*,*)'Too many PDFs: increase maxPDFs in '/
     $              /'reweight0.inc to ',numPDFs+1
               stop
            endif
c set the weights_info string for PDF variation
            if (lpdfvar(nn)) then
               do n=0,nmemPDF(nn)
                  nwgt=nwgt+1
                  write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)+n
                  write(weights_info(nwgt),'(a)') trim(adjustl(temp))/
     $                 /' '//trim(adjustl(lhaPDFsetname(nn)))
               enddo
            else
               nwgt=nwgt+1
               write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)
               write(weights_info(nwgt),'(a)') trim(adjustl(temp))/
     $              /' '//trim(adjustl(lhaPDFsetname(nn)))
            endif
         enddo
c start with central member of the first set
         call InitPDFm(1,0)
      endif
      if(iappl.ne.0)then
c Initialize grid parameters to negative values.
         appl_Q2min   = -1d0
         appl_Q2max   = -1d0
         appl_xmin    = -1d0
         appl_xmax    = -1d0
         appl_nQ2     = -1
         appl_Q2order = -1
         appl_nx      = -1
         appl_xorder  = -1
      endif
      call analysis_begin(nwgt,weights_info)
c To keep track of the accumulated results:
      do kk=1,dyn_scale(0)
         if (lscalevar(kk)) then
            do ii=1,nint(scalevarF(0))
               do jj=1,nint(scalevarR(0))
                  xsecScale_acc(jj,ii,kk)=0d0
               enddo
            enddo
         else
            xsecScale_acc(1,1,kk)=0d0
         endif
      enddo
      do nn=1,lhaPDFid(0)
         if (lpdfvar(nn)) then
            do n=0,nmemPDF(nn)
               xsecPDFr_acc(n,nn)=0d0
            enddo
         else
            xsecPDFr_acc(0,nn)=0d0
         endif
      enddo
      return
      end


      subroutine topout
      implicit none
      include "nexternal.inc"
      include 'reweight0.inc'
      include 'run.inc'
      integer ii,jj,n,kk,nn
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      logical useitmax
      common/cuseitmax/useitmax
      real*8 xnorm
      double precision xsecScale_acc(maxscales,maxscales,maxdynscales)
     $     ,xsecPDFr_acc(0:maxPDFs,maxPDFsets)
      common /scale_pdf_print/xsecScale_acc,xsecPDFr_acc
      integer iappl
      common /for_applgrid/ iappl
      include "appl_common.inc"
c
      if(usexinteg.and..not.mint) then
         xnorm=1.d0/float(itmax)
      elseif(mint) then
         xnorm=1.d0/float(ncall)
      else
         xnorm=1d0
      endif
      if(useitmax)xnorm=xnorm/float(itmax)
c Normalization factor for the APPLgrid grids
      if(iappl.ne.0) appl_norm_histo = 1d0 / dble(ncall*itmax)
      call analysis_end(xnorm)
c Write the accumulated results to a file
      open (unit=34,file='scale_pdf_dependence.dat',status='unknown')
      if (.not.useitmax) xnorm=xnorm/float(itmax)
      if (do_rwgt_scale) then
         write (34,*) "scale variations:"
         do kk=1,dyn_scale(0)
            if (lscalevar(kk)) then
               write (34,*) dyn_scale(kk),nint(scalevarR(0))
     $              ,nint(scalevarF(0))
               write (34,*) ((xsecScale_acc(jj,ii,kk)*xnorm,jj=1
     $              ,nint(scalevarR(0))),ii=1,nint(scalevarF(0)))
            else
               write (34,*) dyn_scale(kk),1,1
               write (34,*) xsecScale_acc(1,1,kk)*xnorm
            endif
         enddo
      endif
      if (do_rwgt_pdf) then
         write (34,*) "pdf variations:"
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               write (34,*) trim(adjustl(lhaPDFsetname(nn))),
     $              nmemPDF(nn)+1
               write (34,*) (xsecPDFr_acc(n,nn)*xnorm,n=0,nmemPDF(nn))
            else
               write(34,*) lhaPDFsetname(nn),nmemPDF(nn) + 1
               write (34,*) xsecPDFr_acc(0,nn)*xnorm
            endif
         enddo
      endif
      close(34)
      return                
      end


      subroutine outfun(pp,ybst_til_tolab,www,iPDG,itype)
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
      include 'genps.inc'
      include 'reweight.inc'
      include 'reweightNLO.inc'
      double precision pp(0:3,nexternal),ybst_til_tolab
      integer itype
      double precision p(0:4,nexternal),pplab(0:3,nexternal),chybst
     $     ,shybst,chybstmo
      integer i,j,ibody,i_wgt,ii,jj,kk,n,nn
      double precision xd(3)
      data (xd(i),i=1,3) /0d0,0d0,1d0/
      integer istatus(nexternal),iPDG(nexternal)
      double precision pmass(nexternal)
      common/to_mass/pmass
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      integer nwgt,max_weight
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision www(max_weight),wgtden,ratio
      double precision xsecScale_acc(maxscales,maxscales,maxdynscales)
     $     ,xsecPDFr_acc(0:maxPDFs,maxPDFsets)
      common /scale_pdf_print/xsecScale_acc,xsecPDFr_acc
      integer iappl
      common /for_applgrid/ iappl
      include "appl_common.inc"
c Born, n-body or (n+1)-body contribution:
      if(itype.eq.11) then
         ibody=1 ! (n+1)-body
      elseif(itype.eq.12)then
         ibody=2 ! n-body
      elseif(itype.eq.20)then
         ibody=3 ! Born
      else
         write(*,*)'Error in outfun: unknown itype',itype
         stop
      endif
c Boost the momenta to the lab frame:
      chybst=cosh(ybst_til_tolab)
      shybst=sinh(ybst_til_tolab)
      chybstmo=chybst-1.d0
      do i=3,nexternal
         call boostwdir2(chybst,shybst,chybstmo,xd,pp(0,i),pplab(0,i))
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
      enddo
      if(iappl.ne.0)then
         appl_itype     = ibody
         appl_www_histo = www(1)
      endif
      call analysis_fill(p,istatus,ipdg,www,ibody)
c Fill the accumulated results
      i_wgt=1
      if (do_rwgt_scale) then
         do kk=1,dyn_scale(0)
            if (lscalevar(kk)) then
               do ii=1,nint(scalevarF(0))
                  do jj=1,nint(scalevarR(0))
                     i_wgt=i_wgt+1
                     xsecScale_acc(jj,ii,kk)=xsecScale_acc(jj,ii,kk)
     $                    +www(i_wgt)
                  enddo
               enddo
            else
               i_wgt=i_wgt+1
               xsecScale_acc(1,1,kk)=xsecScale_acc(1,1,kk)
     $              +www(i_wgt)
            endif
         enddo
      endif
      if (do_rwgt_pdf) then
         do nn=1,lhaPDFid(0)
            if (lpdfvar(nn)) then
               do n=0,nmemPDF(nn)
                  i_wgt=i_wgt+1
                  xsecPDFr_acc(n,nn)=xsecPDFr_acc(n,nn)+www(i_wgt)
               enddo
            else
               i_wgt=i_wgt+1
               xsecPDFr_acc(0,nn)=xsecPDFr_acc(0,nn)+www(i_wgt)
            endif
         enddo
      endif
 999  return      
      end
