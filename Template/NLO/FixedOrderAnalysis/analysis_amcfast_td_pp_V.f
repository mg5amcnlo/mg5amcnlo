************************************************************************
*
*     Analysis routines for vector-boson production.
*
************************************************************************
      subroutine analysis_begin(nwgt,weights_info)
*
      implicit none
*
      include "dbook.inc"
*
      integer nwgt
      character*(*) weights_info(*)
      integer j,kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
      character*5 cc(2)
      data cc/"     "," Born"/
c*
c*     aMCfast common.
c*     Needed to redefine the grid parameters
c*
c      include "reweight_appl.inc"
c      include "appl_common.inc"
c*
c*     Grid parameters
c*
c      appl_Q2min   = 2500d0
c      appl_Q2max   = 40000d0
c      appl_xmin    = 2d-7
c      appl_xmax    = 1d0
c      appl_nQ2     = 10
c      appl_Q2order = 3
c      appl_nx      = 50
c      appl_xorder  = 3
*
*     Initialize histograms
*
      call inihist
*
*     Check whether the user is trying to fill to many histograms
*
      nwgt_analysis = nwgt
      if (nwgt_analysis*5.gt.nplots/2) then
         write(*,*) "error in analysis_begin: ",
     1              "too many histograms, increase NPLOTS to",
     2              nwgt_analysis * 5 * 2
         call exit(-10)
      endif
*
*     Define the observables
*
      do j=1,2
         do kk=1,nwgt_analysis
            l = ( kk - 1 ) * 6 + ( j - 1 ) * 3
            call bookup(l+1,"y_W  "//weights_info(kk)//cc(j),
     1                  1d0,-6d0,6d0)
            call bookup(l+2,"mT_W "//weights_info(kk)//cc(j),
     1                  5d0,70d0,170d0)
            call bookup(l+3,"pT_W "//weights_info(kk)//cc(j),
     1                  5d0,10d0,110d0)
         enddo
      enddo
*
      return
      end
*
************************************************************************
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
*
      implicit none
*
      include "nexternal.inc"
*
      integer i,kk,l
      integer ibody
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer nwgt_analysis
      common/c_analysis/nwgt_analysis
      double precision www
      double precision yW,mTW,pTW,mT2
      double precision getrapidity
      external getrapidity
*
*     All information can be found in analysis_td_template.f 
*
*     The ibody variable is:
*     ibody = 1 : (n+1)-body contribution
*     ibody = 2 : n-body contribution (excluding the Born)
*     ibody = 3 : Born contribution
*
      if(ibody.ne.1.and.ibody.ne.2.and.ibody.ne.3)then
         write(6,*) "Error: Invalid value for ibody =",ibody
         call exit(-10)
      endif
*
*     Check that the number of outcoming particles is correct
*
      if (nexternal.ne.4)then
         write(*,*) "error in analysis_fill: ",
     1              "only for Z/gamma or W production"
         call exit(-10)
      endif
*
*     Check that the third particle is either a Z/gamma or a W
*
      if (abs(ipdg(3)).ne.22.and.abs(ipdg(3)).ne.23.and.
     1    abs(ipdg(3)).ne.24) then
         write(*,*) "error in analysis_fill: ",
     1              "The outcoming particle must be a vector boson, ",
     2              "ID =",ipdg(3)
         call exit(-10)
      endif
*
*     Compute observables
*
*     Rapidity
      yW = getrapidity(p(0,3),p(3,3))
*     Transverse momentum
      pTW = dsqrt(p(1,3)**2d0 + p(2,3)**2d0)
*     Transverse mass
      mT2 = p(0,3)**2 - pTW**2
      if(mT2.gt.0d0)then
         mTW = dsqrt(mT2)
      else
         mTW = 0d0
      endif
*
*     Fill the histograms
*
      do i=1,2
         do kk=1,nwgt_analysis
            www = wgts(kk)
            l = ( kk - 1 ) * 6 + ( i - 1 ) * 3
*     Only fill Born histograms for ibody=3
            if (ibody.ne.3.and.i.eq.2) cycle
*
            call mfill(l+1,yW,www)
            call mfill(l+2,mTW,www)
            call mfill(l+3,pTW,www)
         enddo
      enddo
*
      return      
      end
*
************************************************************************
      subroutine analysis_end(xnorm)
*
      implicit none
*
      include "dbook.inc"
*
      character*14 ytit
      double precision xnorm
      integer i
      integer kk,l,nwgt_analysis
      common/c_analysis/nwgt_analysis
*
      call open_topdrawer_file
      call mclear
*
      do i=1,NPLOTS
         call mopera(i,"+",i,i,xnorm,0.d0)
         call mfinal(i)
      enddo
      ytit = "sigma per bin "
      do i=1,2
         do kk=1,nwgt_analysis
            l = ( kk - 1 ) * 6 + ( i - 1 ) * 3
            call multitop(l+1,3,2,"y_W  "," ","LIN")
            call multitop(l+2,3,2,"mT_W "," ","LIN")
            call multitop(l+3,3,2,"pT_W "," ","LIN")
         enddo
      enddo
*
      call close_topdrawer_file
*
      return                
      end
*
************************************************************************
*
*     Auxiliary functions for the kinematics
*
************************************************************************
      function getrapidity(en,pl)
*
      implicit none
*
      real*8 getrapidity,en,pl,tiny,xplus,xminus,y
      parameter (tiny=1d-8)
*
      xplus  = en + pl
      xminus = en - pl
*
      if(xplus.gt.tiny.and.xminus.gt.tiny)then
         if((xplus/xminus).gt.tiny.and.(xminus/xplus).gt.tiny)then
            y = 0.5d0 * log( xplus / xminus  )
         else
            y = sign(1d0,pl) * tiny
         endif
      else
         y = sign(1d0,pl) * tiny
      endif
*
      getrapidity = y
*
      return
      end
