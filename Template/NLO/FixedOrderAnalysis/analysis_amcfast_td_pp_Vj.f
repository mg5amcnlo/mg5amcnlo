************************************************************************
*
*     Analysis routines for Z/W+j production
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
     2               nwgt_analysis*5*2
         call exit(-10)
      endif
*
*     Define the observables
*
      do j=1,2
         do kk=1,nwgt_analysis
            l = ( kk - 1 ) * 4 + ( j - 1 ) * 2
            call bookup(l+1,"y_Z  "//weights_info(kk)//cc(j),
     1                  0.5d0,-6d0,6d0)
            call bookup(l+2,"pT_Z "//weights_info(kk)//cc(j),
     1                  5d0,30d0,130d0)
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
      double precision yZ,pTZ
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
      if (nexternal.ne.5)then
         write(*,*) "error in analysis_fill: "//
     1              "only for V+j production"
         call exit(-10)
      endif
*
*     Check that the third particle is either a Z/gamma or a W
*
      if (abs(ipdg(3)).ne.22.and.abs(ipdg(3)).ne.23.and.
     1    abs(ipdg(3)).ne.24) then
         write(*,*) "error in analysis_fill: "//
     1              "The first outcoming particle must be a vector",
     1              " boson, ID =",ipdg(3)
         call exit(-10)
      endif
*
*     Compute observables
*
*     Rapidity of the Z
      yZ  = getrapidity(p(0,3),p(3,3))
*     Transverse momentum of the Z
      pTZ = dsqrt(p(1,3)**2d0 + p(2,3)**2d0)
*
*     Fill the histograms
*
      do i=1,2
         do kk=1,nwgt_analysis
            www = wgts(kk)
            l = ( kk - 1 ) * 4 + ( i - 1 ) * 2
*     Only fill Born histograms for ibody=3
            if (ibody.ne.3.and.i.eq.2) cycle
*
            call mfill(l+1,yZ,www)
            call mfill(l+2,pTZ,www)
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
            l = ( kk - 1 ) * 4 + ( i - 1 ) * 2
            call multitop(l+1,3,2,"y_Z  "," ","LIN")
            call multitop(l+2,3,2,"pT_Z "," ","LIN")
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
